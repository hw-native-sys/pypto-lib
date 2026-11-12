# Copyright (c) PyPTO Contributors.
# This program is free software; you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# -----------------------------------------------------------------------------------------------------------
"""Compressed-attention EP2 layer body shared by C2A and C1A consumers.

The cache planes are explicit at this boundary.  The caller owns compressor
and index publication; this entry consumes the published BF16 SWA/compressed
rows and the prepared QLI indices, then runs the same mHC, MoE and residual
path as the complete SWA layer.
"""

from __future__ import annotations

from dataclasses import asdict

import pypto.language as pl
import pypto.language.distributed as pld

from .attention_block import make_attention_block
from .config import FLASH
from .decode_layer_l0 import (
    AUX_PAD,
    COUNT_PAD,
    D,
    EP_SIZE,
    LANE_ROWS,
    MOE_INTER,
    N_EXPERTS,
    N_LOCAL_EXPERTS,
    TOPK,
    decode_ep2_rank,
)
from .engram import COLS, K as ENGRAM_K, N as ENGRAM_N, make_gate
from .hc import make_hc_post_inline, make_hc_pre_inline
from .decode_indexer import (
    make_indexer_prepare,
    make_indexer_projection,
    make_indexer_query_quantize,
    make_indexer_select,
    make_l3_indexer_query,
)
from .layer2_c1c2 import make_c2_index_publish_inline
from .norm_rope import make_rope, make_rmsnorm
from .output_proj import make_output_projection_bf16
from .qkv_w8a8 import make_qkv_w8a8_inline
from .rmsnorm import NORM_T_TILE, rms_norm_tiled, rms_norm_tiled_from_fp32, rms_norm_tiled_with_fp32
from .sparse_attention import make_sparse_attention
from .swa_cache import make_swa_cache_publish


CONFIG = asdict(FLASH)
HC = FLASH.hc_mult
CHANNELS = FLASH.mix_hc
HIDDEN = FLASH.hidden_size
LATENT = FLASH.q_lora_rank
WIDTH = FLASH.num_attention_heads * FLASH.head_dim
ROPE_PAIRS = FLASH.qk_rope_head_dim // 2
O_RANK = FLASH.o_lora_rank
WINDOW = FLASH.sliding_window
HEADS = FLASH.num_attention_heads
HEAD_DIM = FLASH.head_dim
COMPRESSED_ROWS = 64
COMPRESSED_BLOCKS = 4
# The bounded EP2 replay owns the request's compressed page at local page 1.
# QLI indices are logical rows within that page; the physical cache tensor
# still carries the four captured pages so cache state comparisons stay exact.
COMPRESSED_PAGE_OFFSET = COMPRESSED_ROWS
INDEX_TOPK = FLASH.index_topk
VALUE_PADDING = 16


@pl.jit.inline
def _engram_linear(
    value: pl.Tensor[[16, ENGRAM_K], pl.BF16],
    weight: pl.Tensor[[ENGRAM_K, ENGRAM_N], pl.BF16],
    output: pl.InOut[pl.Tensor[[16, ENGRAM_N], pl.BF16]],
    active: pl.Scalar[pl.INT32],
):
    """Project captured Engram rows before the layer attention path."""

    blocks = ENGRAM_N // 64
    for block in pl.spmd(((active + 15) // 16) * blocks, name_hint="dsv41_c2a_engram_linear"):
        t0 = (block // blocks) * 16
        n0 = (block % blocks) * 64
        rows = pl.min(16, active - t0)
        acc = pl.create_tensor([16, 64], dtype=pl.FP32)
        for k0 in pl.pipeline(0, ENGRAM_K, 256, stage=2):
            lhs = pl.slice(value, [16, 256], [t0, k0], valid_shape=[rows, 256])
            lhs = pl.set_validshape(pl.fillpad(lhs, pad_value=pl.PadValue.zero), 16, 256)
            rhs = pl.slice(weight, [256, 64], [k0, n0])
            acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(k0 == 0))
        output[t0:t0 + 16, n0:n0 + 64] = pl.set_validshape(
            pl.cast(acc, pl.BF16, mode="rint"), rows, 64
        )


ENGRAM_GATE = pl.jit.inline(make_gate(16)._func)


QKV = make_qkv_w8a8_inline(16, CONFIG)
SWA_PUBLISH = make_swa_cache_publish(16, 2, HEAD_DIM)
C2_PUBLISH = make_c2_index_publish_inline(16, CONFIG, cache_blocks=2)
INDEXER_PROJECTION = pl.jit.inline(make_indexer_projection(16, CONFIG)._func)
INDEXER_QUERY_QUANTIZE = pl.jit.inline(make_indexer_query_quantize(16)._func)
INDEXER_SELECT = pl.jit.inline(make_indexer_select(16, 512, topk=INDEX_TOPK, compress_ratio=2)._func)
INDEXER_PREPARE = pl.jit.inline(make_indexer_prepare(16, topk=INDEX_TOPK, compress_ratio=2)._func)


@pl.jit.incore
def _pad_index_cache(
    source: pl.Tensor[[256, 128], pl.INT8],
    source_scale: pl.Tensor[[256], pl.FP16],
    output: pl.Tensor[[512, 128], pl.INT8],
    output_scale: pl.Tensor[[512], pl.FP16],
):
    row = pl.tile.get_block_idx()
    if row < 256:
        pl.store(pl.load(source, [row, 0], [1, 128]), [row, 0], output)
        pl.write(output_scale, [row], pl.read(source_scale, [row]))
    else:
        zero = pl.cast(pl.tile.full([1, 128], dtype=pl.INT32, value=0), pl.INT8, mode="trunc")
        pl.store(zero, [row, 0], output)
        pl.write(output_scale, [row], pl.cast(1.0, pl.FP16))


SPARSE = pl.jit.inline(
    make_sparse_attention(16, HEADS, HEAD_DIM, WINDOW, INDEX_TOPK, VALUE_PADDING)._func
)
OUTPUT = pl.jit.inline(
    make_output_projection_bf16(
        16, HEADS, HEAD_DIM, FLASH.qk_rope_head_dim, FLASH.o_groups, O_RANK, HIDDEN
    )._func
)
HC_PRE = make_hc_pre_inline(16, CONFIG)
HC_POST = make_hc_post_inline(16, CONFIG)
EP2_RANK = pl.jit.inline(decode_ep2_rank._func)


@pl.jit.inline(auto_scope=False)
def _gather_rows(
    cache: pl.Tensor[[COMPRESSED_BLOCKS, COMPRESSED_ROWS, 1, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[16, INDEX_TOPK], pl.INT32],
    selected: pl.Tensor[[16, INDEX_TOPK, HEAD_DIM + VALUE_PADDING], pl.BF16],
    active: pl.Scalar[pl.INT32],
):
    cache_flat = pl.reshape(cache, [COMPRESSED_BLOCKS * COMPRESSED_ROWS, HEAD_DIM])
    selected_flat = pl.reshape(selected, [16 * INDEX_TOPK, HEAD_DIM + VALUE_PADDING])
    with pl.spmd(active * INDEX_TOPK, name_hint="dsv41_c2a_compressed_gather"):
        block = pl.tile.get_block_idx()
        token = block // INDEX_TOPK
        column = block % INDEX_TOPK
        index = pl.read(indices, [token, column])
        value = pl.full([1, HEAD_DIM + VALUE_PADDING], dtype=pl.BF16, value=0.0)
        physical_index = index + COMPRESSED_PAGE_OFFSET
        if index >= 0 and physical_index < COMPRESSED_BLOCKS * COMPRESSED_ROWS:
            row = pl.cast(physical_index, pl.INDEX)
            value[0:1, VALUE_PADDING:] = cache_flat[row:row + 1, :]
        selected_flat[block:block + 1, :] = value


@pl.jit.inline(auto_scope=False)
def _gather_window(
    cache: pl.Tensor[[2, 128, 1, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[16, WINDOW], pl.INT32],
    selected: pl.Tensor[[16, WINDOW, HEAD_DIM + VALUE_PADDING], pl.BF16],
    active: pl.Scalar[pl.INT32],
):
    cache_flat = pl.reshape(cache, [2 * 128, HEAD_DIM])
    selected_flat = pl.reshape(selected, [16 * WINDOW, HEAD_DIM + VALUE_PADDING])
    with pl.spmd(active * WINDOW, name_hint="dsv41_c2a_window_gather"):
        block = pl.tile.get_block_idx()
        token = block // WINDOW
        column = block % WINDOW
        index = pl.read(indices, [token, column])
        value = pl.full([1, HEAD_DIM + VALUE_PADDING], dtype=pl.BF16, value=0.0)
        if index >= 0 and index < 2 * 128:
            row = pl.cast(index, pl.INDEX)
            value[0:1, VALUE_PADDING:] = cache_flat[row:row + 1, :]
        selected_flat[block:block + 1, :] = value


@pl.jit(auto_scope=False)
def decode_layer_ep2_sparse_rank(
    x_hc: pl.Tensor[[16, HC, HIDDEN], pl.BF16],
    engram_lookup: pl.Tensor[[16, ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[ENGRAM_K, ENGRAM_N], pl.BF16],
    engram_channel_weight: pl.Tensor[[HC, HIDDEN], pl.FP32],
    engram_mask: pl.Tensor[[16], pl.INT32],
    engram_kv: pl.InOut[pl.Tensor[[16, ENGRAM_N], pl.BF16]],
    engram_hidden: pl.InOut[pl.Tensor[[16, HC, HIDDEN], pl.BF16]],
    attn_hc_function: pl.Tensor[[CHANNELS, HC * HIDDEN], pl.FP32],
    attn_hc_scale: pl.Tensor[[3], pl.FP32],
    attn_hc_base: pl.Tensor[[CHANNELS], pl.FP32],
    attn_incoming: pl.Tensor[[16, HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[HIDDEN], pl.BF16],
    wq_a: pl.Tensor[[HIDDEN, LATENT], pl.INT8],
    wq_a_scale: pl.Tensor[[LATENT], pl.FP32],
    q_norm_weight: pl.Tensor[[LATENT], pl.BF16],
    wq_b: pl.Tensor[[LATENT, WIDTH], pl.INT8],
    wq_b_scale: pl.Tensor[[WIDTH], pl.FP32],
    wkv: pl.Tensor[[HIDDEN, HEAD_DIM], pl.INT8],
    wkv_scale: pl.Tensor[[HEAD_DIM], pl.FP32],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[8, WIDTH // 8, O_RANK], pl.BF16],
    wo_b: pl.Tensor[[8 * O_RANK, HIDDEN], pl.BF16],
    rope_cos: pl.Tensor[[16, ROPE_PAIRS], pl.FP32],
    rope_sin: pl.Tensor[[16, ROPE_PAIRS], pl.FP32],
    slots: pl.Tensor[[16, 2], pl.INT64],
    indices: pl.Tensor[[16, WINDOW], pl.INT32],
    prepared_indices: pl.Tensor[[16, INDEX_TOPK], pl.INT32],
    prepared_indices_source: pl.Tensor[[16, INDEX_TOPK], pl.INT32],
    sink: pl.Tensor[[HEADS], pl.FP32],
    positions: pl.InOut[pl.Tensor[[16], pl.INT64]],
    query_latent: pl.InOut[pl.Tensor[[16, LATENT], pl.BF16]],
    query: pl.InOut[pl.Tensor[[16, WIDTH], pl.BF16]],
    window_kv: pl.InOut[pl.Tensor[[16, HEAD_DIM], pl.BF16]],
    cache: pl.InOut[pl.Tensor[[2, 128, 1, HEAD_DIM], pl.BF16]],
    cache_lens: pl.InOut[pl.Tensor[[2], pl.INT32]],
    compressed_cache: pl.InOut[pl.Tensor[[COMPRESSED_BLOCKS, COMPRESSED_ROWS, 1, HEAD_DIM], pl.BF16]],
    index_cache: pl.InOut[pl.Tensor[[COMPRESSED_BLOCKS, COMPRESSED_ROWS, 1, 128], pl.INT8]],
    index_scale: pl.InOut[pl.Tensor[[COMPRESSED_BLOCKS, COMPRESSED_ROWS, 1, 1], pl.FP16]],
    compressor_state: pl.InOut[pl.Tensor[[4, 32, 2 * HEAD_DIM], pl.FP32]],
    positions32: pl.Tensor[[16], pl.INT32],
    compressed_slots: pl.Tensor[[16], pl.INT64],
    index_slots: pl.Tensor[[16], pl.INT64],
    query_start: pl.Tensor[[2], pl.INT32],
    requests: pl.Tensor[[16], pl.INT32],
    state_block_table: pl.Tensor[[1, 1], pl.INT32],
    compressor_wkv: pl.Tensor[[HIDDEN, HEAD_DIM], pl.BF16],
    compressor_wgate: pl.Tensor[[HIDDEN, HEAD_DIM], pl.BF16],
    compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[HEAD_DIM, 128], pl.BF16],
    index_norm: pl.Tensor[[128], pl.BF16],
    index_wq_b: pl.Tensor[[LATENT, 4096], pl.INT8],
    index_wq_b_scale: pl.Tensor[[4096], pl.FP32],
    index_weights_proj: pl.Tensor[[HIDDEN, 32], pl.BF16],
    attn_output: pl.InOut[pl.Tensor[[16, HIDDEN], pl.BF16]],
    attn_next_pre: pl.InOut[pl.Tensor[[16, HC], pl.FP32]],
    attn_post: pl.InOut[pl.Tensor[[16, HC], pl.FP32]],
    attn_comb: pl.InOut[pl.Tensor[[16, HC, HC], pl.FP32]],
    attn_collapsed: pl.InOut[pl.Tensor[[16, HIDDEN], pl.BF16]],
    attn_normed: pl.InOut[pl.Tensor[[16, HIDDEN], pl.BF16]],
    after_attention: pl.InOut[pl.Tensor[[16, HC, HIDDEN], pl.BF16]],
    ffn_hc_function: pl.Tensor[[CHANNELS, HC * HIDDEN], pl.FP32],
    ffn_hc_scale: pl.Tensor[[3], pl.FP32],
    ffn_hc_base: pl.Tensor[[CHANNELS], pl.FP32],
    ffn_norm_weight: pl.Tensor[[HIDDEN], pl.BF16],
    ffn_next_pre: pl.InOut[pl.Tensor[[16, HC], pl.FP32]],
    ffn_post: pl.InOut[pl.Tensor[[16, HC], pl.FP32]],
    ffn_comb: pl.InOut[pl.Tensor[[16, HC, HC], pl.FP32]],
    ffn_collapsed: pl.InOut[pl.Tensor[[16, HIDDEN], pl.BF16]],
    ffn_normed: pl.InOut[pl.Tensor[[16, HIDDEN], pl.BF16]],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    text_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    vision_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    input_ids: pl.Tensor[[16], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, 2560], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    shared_gate_up: pl.Tensor[[D, MOE_INTER * 2], pl.INT8],
    shared_gate_up_scale: pl.Tensor[[MOE_INTER * 2], pl.FP32],
    shared_down: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_down_scale: pl.Tensor[[D], pl.FP32],
    lane_x: pld.DistributedTensor[[LANE_ROWS, D], pl.INT8],
    lane_aux: pld.DistributedTensor[[LANE_ROWS, AUX_PAD], pl.FP32],
    lane_count: pld.DistributedTensor[[EP_SIZE, COUNT_PAD], pl.INT32],
    lane_ready: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    slot_out: pld.DistributedTensor[[16 * TOPK, D], pl.FP32],
    return_ready: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    logits: pl.InOut[pl.Tensor[[16, N_EXPERTS], pl.FP32]],
    route_ids: pl.InOut[pl.Tensor[[16, TOPK], pl.INT32]],
    route_weights: pl.InOut[pl.Tensor[[16, TOPK], pl.FP32]],
    moe_output: pl.Out[pl.Tensor[[16, D], pl.BF16]],
    output: pl.Out[pl.Tensor[[16, HC, HIDDEN], pl.BF16]],
    active: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    reuse: pl.Scalar[pl.INT32],
    has_engram: pl.Scalar[pl.INT32],
):
    attention_input = x_hc
    if has_engram == 1:
        _engram_linear(engram_lookup, engram_wkv, engram_kv, active)
        ENGRAM_GATE(
            x_hc,
            engram_kv,
            engram_channel_weight,
            engram_mask,
            engram_hidden,
            active,
        )
        attention_input = engram_hidden
    attn_collapsed_fp32 = pl.create_tensor([16, HIDDEN], dtype=pl.FP32)
    HC_PRE(
        attention_input, attn_hc_function, attn_hc_scale, attn_hc_base, attn_incoming,
        attn_next_pre, attn_post, attn_comb, attn_collapsed, attn_collapsed_fp32, active,
    )
    for block in pl.spmd((active + NORM_T_TILE - 1) // NORM_T_TILE, name_hint="dsv41_c2a_attn_norm"):
        rms_norm_tiled(attn_collapsed, attn_norm_weight, attn_normed, block, active)

    qkv_cache = pl.reshape(cache, [2, 128, HEAD_DIM])
    QKV(
        attn_normed, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight, rope_cos, rope_sin, positions,
        query_latent, query, window_kv, qkv_cache, cache_lens, active,
    )
    SWA_PUBLISH(window_kv, slots, cache, active)

    prepared_for_attention = prepared_indices
    if reuse == 1:
        # Reuse layers receive source candidates through a readonly operand.
        # The output InOut buffer is not a reliable source on every host ABI.
        prepared_for_attention = prepared_indices_source

    if reuse == 0:
        compressed_cache_pub = pl.reshape(compressed_cache, [2, 128, 1, HEAD_DIM])
        index_cache_pub = pl.reshape(index_cache, [2, 128, 1, 128])
        index_scale_pub = pl.reshape(index_scale, [2, 128, 1, 1])
        compressed = pl.create_tensor([16, HEAD_DIM], dtype=pl.BF16)
        index_key = pl.create_tensor([16, 128], dtype=pl.BF16)
        C2_PUBLISH(
            attn_normed, compressor_wkv, compressor_wgate, compressor_norm,
            index_wk, index_norm, rope_cos, rope_sin, positions32, compressed_slots,
            index_slots,
            query_start, requests, state_block_table, compressor_state,
            compressed_cache_pub, index_cache_pub, index_scale_pub,
            compressed, index_key, active,
        )

        index_query = pl.create_tensor([16, 32, 128], dtype=pl.BF16)
        index_payload = pl.create_tensor([16, 32, 128], dtype=pl.INT8)
        index_query_scale = pl.create_tensor([16, 32], dtype=pl.FP16)
        index_weights = pl.create_tensor([16, 32], dtype=pl.FP16)
        INDEXER_PROJECTION(
            attn_normed, query_latent, index_wq_b, index_wq_b_scale,
            index_weights_proj, rope_cos, rope_sin, index_query, index_weights, active,
        )
        INDEXER_QUERY_QUANTIZE(index_query, index_payload, index_query_scale, active)
        index_key_flat = pl.reshape(index_cache_pub, [256, 128])
        index_scale_flat = pl.reshape(index_scale_pub, [256])
        key_payload = pl.create_tensor([512, 128], dtype=pl.INT8)
        key_scale = pl.create_tensor([512], dtype=pl.FP16)
        _pad_index_cache(index_key_flat, index_scale_flat, key_payload, key_scale)
        qli_scores = pl.create_tensor([16, 512], dtype=pl.FP32)
        qli_indices = pl.create_tensor([16, INDEX_TOPK], dtype=pl.INT32)
        INDEXER_SELECT(
            index_payload, index_query_scale, key_payload, key_scale,
            index_weights, positions32, qli_scores, qli_indices, active,
        )
        INDEXER_PREPARE(qli_indices, positions32, prepared_indices, active)
    window_rows = pl.create_tensor([16, WINDOW, HEAD_DIM + VALUE_PADDING], dtype=pl.BF16)
    compressed_rows = pl.create_tensor([16, INDEX_TOPK, HEAD_DIM + VALUE_PADDING], dtype=pl.BF16)
    _gather_window(cache, indices, window_rows, active)
    _gather_rows(compressed_cache, prepared_for_attention, compressed_rows, active)
    sparse_padded = pl.create_tensor([16, HEADS, HEAD_DIM + VALUE_PADDING], dtype=pl.BF16)
    query_heads = pl.reshape(query, [16, HEADS, HEAD_DIM])
    SPARSE(
        query_heads, window_rows, indices,
        compressed_rows, prepared_for_attention, sink, sparse_padded, active,
    )
    if reuse == 1:
        for block in pl.spmd(active, name_hint="dsv41_c2a_reuse_candidates_out"):
            prepared_indices[block:block + 1, :] = prepared_for_attention[block:block + 1, :]
    attended = pl.create_tensor([16, WIDTH], dtype=pl.BF16)
    for block in pl.spmd(active * (HEADS // 16), name_hint="dsv41_c2a_sparse_unpad"):
        token = block // (HEADS // 16)
        head = (block % (HEADS // 16)) * 16
        attended[token:token + 1, head * HEAD_DIM:(head + 16) * HEAD_DIM] = pl.reshape(
            sparse_padded[token:token + 1, head:head + 16, VALUE_PADDING:],
            [1, 16 * HEAD_DIM],
        )
    rotated = pl.create_tensor([16, WIDTH], dtype=pl.BF16)
    grouped = pl.create_tensor([16, 8 * O_RANK], dtype=pl.BF16)
    OUTPUT(attended, wo_a, wo_b, rope_cos, rope_sin, rotated, grouped, attn_output, active)
    HC_POST(attn_output, attention_input, attn_post, attn_comb, after_attention, active)
    ffn_collapsed_fp32 = pl.create_tensor([16, HIDDEN], dtype=pl.FP32)
    HC_PRE(
        after_attention, ffn_hc_function, ffn_hc_scale, ffn_hc_base,
        attn_next_pre, ffn_next_pre, ffn_post, ffn_comb, ffn_collapsed, ffn_collapsed_fp32, active,
    )
    ffn_router_normed = pl.create_tensor([16, HIDDEN], dtype=pl.FP32)
    for block in pl.spmd((active + NORM_T_TILE - 1) // NORM_T_TILE, name_hint="dsv41_c2a_ffn_norm"):
        # The fused reference rounds the HC collapse to BF16 before RMSNorm;
        # expose that same BF16 result to both the experts and router.
        rms_norm_tiled_with_fp32(
            ffn_collapsed,
            ffn_norm_weight,
            ffn_normed,
            ffn_router_normed,
            block,
            active,
        )
    EP2_RANK(
        ffn_normed, ffn_router_normed, gate_weight, text_bias, vision_bias, input_ids,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale, routed_w2,
        routed_w2_scale, shared_gate_up, shared_gate_up_scale, shared_down,
        shared_down_scale, lane_x, lane_aux, lane_count, lane_ready, slot_out,
        return_ready, logits, route_ids, route_weights, moe_output, active,
        rank, epoch,
    )
    HC_POST(moe_output, after_attention, ffn_post, ffn_comb, output, active)
    return output


@pl.jit.host
def decode_layer_ep2_sparse(
    x_hc: pl.Tensor[[EP_SIZE, 16, HC, HIDDEN], pl.BF16],
    engram_lookup: pl.Tensor[[EP_SIZE, 16, ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[EP_SIZE, ENGRAM_K, ENGRAM_N], pl.BF16],
    engram_channel_weight: pl.Tensor[[EP_SIZE, HC, HIDDEN], pl.FP32],
    engram_mask: pl.Tensor[[EP_SIZE, 16], pl.INT32],
    engram_kv: pl.InOut[pl.Tensor[[EP_SIZE, 16, ENGRAM_N], pl.BF16]],
    engram_hidden: pl.InOut[pl.Tensor[[EP_SIZE, 16, HC, HIDDEN], pl.BF16]],
    attn_hc_function: pl.Tensor[[EP_SIZE, CHANNELS, HC * HIDDEN], pl.FP32],
    attn_hc_scale: pl.Tensor[[EP_SIZE, 3], pl.FP32],
    attn_hc_base: pl.Tensor[[EP_SIZE, CHANNELS], pl.FP32],
    attn_incoming: pl.Tensor[[EP_SIZE, 16, HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[EP_SIZE, HIDDEN], pl.BF16],
    wq_a: pl.Tensor[[EP_SIZE, HIDDEN, LATENT], pl.INT8],
    wq_a_scale: pl.Tensor[[EP_SIZE, LATENT], pl.FP32],
    q_norm_weight: pl.Tensor[[EP_SIZE, LATENT], pl.BF16],
    wq_b: pl.Tensor[[EP_SIZE, LATENT, WIDTH], pl.INT8],
    wq_b_scale: pl.Tensor[[EP_SIZE, WIDTH], pl.FP32],
    wkv: pl.Tensor[[EP_SIZE, HIDDEN, HEAD_DIM], pl.INT8],
    wkv_scale: pl.Tensor[[EP_SIZE, HEAD_DIM], pl.FP32],
    kv_norm_weight: pl.Tensor[[EP_SIZE, HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[EP_SIZE, 8, WIDTH // 8, O_RANK], pl.BF16],
    wo_b: pl.Tensor[[EP_SIZE, 8 * O_RANK, HIDDEN], pl.BF16],
    rope_cos: pl.Tensor[[EP_SIZE, 16, ROPE_PAIRS], pl.FP32],
    rope_sin: pl.Tensor[[EP_SIZE, 16, ROPE_PAIRS], pl.FP32],
    slots: pl.Tensor[[EP_SIZE, 16, 2], pl.INT64],
    indices: pl.Tensor[[EP_SIZE, 16, WINDOW], pl.INT32],
    prepared_indices: pl.InOut[pl.Tensor[[EP_SIZE, 16, INDEX_TOPK], pl.INT32]],
    prepared_indices_source: pl.Tensor[[EP_SIZE, 16, INDEX_TOPK], pl.INT32],
    sink: pl.Tensor[[EP_SIZE, HEADS], pl.FP32],
    positions: pl.InOut[pl.Tensor[[EP_SIZE, 16], pl.INT64]],
    query_latent: pl.InOut[pl.Tensor[[EP_SIZE, 16, LATENT], pl.BF16]],
    query: pl.InOut[pl.Tensor[[EP_SIZE, 16, WIDTH], pl.BF16]],
    window_kv: pl.InOut[pl.Tensor[[EP_SIZE, 16, HEAD_DIM], pl.BF16]],
    cache: pl.InOut[pl.Tensor[[EP_SIZE, 2, 128, 1, HEAD_DIM], pl.BF16]],
    cache_lens: pl.InOut[pl.Tensor[[EP_SIZE, 2], pl.INT32]],
    compressed_cache: pl.InOut[pl.Tensor[[EP_SIZE, COMPRESSED_BLOCKS, COMPRESSED_ROWS, 1, HEAD_DIM], pl.BF16]],
    index_cache: pl.InOut[pl.Tensor[[EP_SIZE, COMPRESSED_BLOCKS, COMPRESSED_ROWS, 1, 128], pl.INT8]],
    index_scale: pl.InOut[pl.Tensor[[EP_SIZE, COMPRESSED_BLOCKS, COMPRESSED_ROWS, 1, 1], pl.FP16]],
    compressor_state: pl.InOut[pl.Tensor[[EP_SIZE, 4, 32, 2 * HEAD_DIM], pl.FP32]],
    positions32: pl.Tensor[[EP_SIZE, 16], pl.INT32],
    compressed_slots: pl.Tensor[[EP_SIZE, 16], pl.INT64],
    index_slots: pl.Tensor[[EP_SIZE, 16], pl.INT64],
    query_start: pl.Tensor[[EP_SIZE, 2], pl.INT32],
    requests: pl.Tensor[[EP_SIZE, 16], pl.INT32],
    state_block_table: pl.Tensor[[EP_SIZE, 1, 1], pl.INT32],
    compressor_wkv: pl.Tensor[[EP_SIZE, HIDDEN, HEAD_DIM], pl.BF16],
    compressor_wgate: pl.Tensor[[EP_SIZE, HIDDEN, HEAD_DIM], pl.BF16],
    compressor_norm: pl.Tensor[[EP_SIZE, HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[EP_SIZE, HEAD_DIM, 128], pl.BF16],
    index_norm: pl.Tensor[[EP_SIZE, 128], pl.BF16],
    index_wq_b: pl.Tensor[[EP_SIZE, LATENT, 4096], pl.INT8],
    index_wq_b_scale: pl.Tensor[[EP_SIZE, 4096], pl.FP32],
    index_weights_proj: pl.Tensor[[EP_SIZE, HIDDEN, 32], pl.BF16],
    attn_output: pl.InOut[pl.Tensor[[EP_SIZE, 16, HIDDEN], pl.BF16]],
    attn_next_pre: pl.InOut[pl.Tensor[[EP_SIZE, 16, HC], pl.FP32]],
    attn_post: pl.InOut[pl.Tensor[[EP_SIZE, 16, HC], pl.FP32]],
    attn_comb: pl.InOut[pl.Tensor[[EP_SIZE, 16, HC, HC], pl.FP32]],
    attn_collapsed: pl.InOut[pl.Tensor[[EP_SIZE, 16, HIDDEN], pl.BF16]],
    attn_normed: pl.InOut[pl.Tensor[[EP_SIZE, 16, HIDDEN], pl.BF16]],
    after_attention: pl.InOut[pl.Tensor[[EP_SIZE, 16, HC, HIDDEN], pl.BF16]],
    ffn_hc_function: pl.Tensor[[EP_SIZE, CHANNELS, HC * HIDDEN], pl.FP32],
    ffn_hc_scale: pl.Tensor[[EP_SIZE, 3], pl.FP32],
    ffn_hc_base: pl.Tensor[[EP_SIZE, CHANNELS], pl.FP32],
    ffn_norm_weight: pl.Tensor[[EP_SIZE, HIDDEN], pl.BF16],
    ffn_next_pre: pl.InOut[pl.Tensor[[EP_SIZE, 16, HC], pl.FP32]],
    ffn_post: pl.InOut[pl.Tensor[[EP_SIZE, 16, HC], pl.FP32]],
    ffn_comb: pl.InOut[pl.Tensor[[EP_SIZE, 16, HC, HC], pl.FP32]],
    ffn_collapsed: pl.InOut[pl.Tensor[[EP_SIZE, 16, HIDDEN], pl.BF16]],
    ffn_normed: pl.InOut[pl.Tensor[[EP_SIZE, 16, HIDDEN], pl.BF16]],
    gate_weight: pl.Tensor[[EP_SIZE, N_EXPERTS, D], pl.FP32],
    text_bias: pl.Tensor[[EP_SIZE, N_EXPERTS], pl.FP32],
    vision_bias: pl.Tensor[[EP_SIZE, N_EXPERTS], pl.FP32],
    input_ids: pl.Tensor[[EP_SIZE, 16], pl.INT64],
    routed_w1: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, D, 2560], pl.INT8],
    routed_w2_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, D], pl.FP32],
    shared_gate_up: pl.Tensor[[EP_SIZE, D, MOE_INTER * 2], pl.INT8],
    shared_gate_up_scale: pl.Tensor[[EP_SIZE, MOE_INTER * 2], pl.FP32],
    shared_down: pl.Tensor[[EP_SIZE, MOE_INTER, D], pl.INT8],
    shared_down_scale: pl.Tensor[[EP_SIZE, D], pl.FP32],
    logits: pl.Out[pl.Tensor[[EP_SIZE, 16, N_EXPERTS], pl.FP32]],
    route_ids: pl.Out[pl.Tensor[[EP_SIZE, 16, TOPK], pl.INT32]],
    route_weights: pl.Out[pl.Tensor[[EP_SIZE, 16, TOPK], pl.FP32]],
    moe_output: pl.Out[pl.Tensor[[EP_SIZE, 16, D], pl.BF16]],
    output: pl.Out[pl.Tensor[[EP_SIZE, 16, HC, HIDDEN], pl.BF16]],
    active: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    reuse: pl.Scalar[pl.INT32],
    has_engram: pl.Scalar[pl.INT32],
):
    lane_x_buf = pld.alloc_window_buffer([LANE_ROWS, D], dtype=pl.INT8)
    lane_aux_buf = pld.alloc_window_buffer([LANE_ROWS, AUX_PAD], dtype=pl.FP32)
    lane_count_buf = pld.alloc_window_buffer([EP_SIZE, COUNT_PAD], dtype=pl.INT32)
    lane_ready_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
    slot_out_buf = pld.alloc_window_buffer([16 * TOPK, D], dtype=pl.FP32)
    return_ready_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(EP_SIZE):
        lane_x = pld.window(lane_x_buf, [LANE_ROWS, D], dtype=pl.INT8)
        lane_aux = pld.window(lane_aux_buf, [LANE_ROWS, AUX_PAD], dtype=pl.FP32)
        lane_count = pld.window(lane_count_buf, [EP_SIZE, COUNT_PAD], dtype=pl.INT32)
        lane_ready = pld.window(lane_ready_buf, [EP_SIZE, 1], dtype=pl.INT32)
        slot_out = pld.window(slot_out_buf, [16 * TOPK, D], dtype=pl.FP32)
        return_ready = pld.window(return_ready_buf, [EP_SIZE, 1], dtype=pl.INT32)
        decode_layer_ep2_sparse_rank(
            x_hc[rank], engram_lookup[rank], engram_wkv[rank], engram_channel_weight[rank],
            engram_mask[rank], engram_kv[rank], engram_hidden[rank],
            attn_hc_function[rank], attn_hc_scale[rank], attn_hc_base[rank],
            attn_incoming[rank], attn_norm_weight[rank], wq_a[rank], wq_a_scale[rank],
            q_norm_weight[rank], wq_b[rank], wq_b_scale[rank], wkv[rank], wkv_scale[rank],
            kv_norm_weight[rank], wo_a[rank], wo_b[rank], rope_cos[rank], rope_sin[rank],
            slots[rank], indices[rank], prepared_indices[rank], prepared_indices_source[rank],
            sink[rank], positions[rank],
            query_latent[rank], query[rank], window_kv[rank], cache[rank], cache_lens[rank],
            compressed_cache[rank], index_cache[rank], index_scale[rank], compressor_state[rank],
            positions32[rank], compressed_slots[rank], index_slots[rank], query_start[rank], requests[rank],
            state_block_table[rank], compressor_wkv[rank], compressor_wgate[rank], compressor_norm[rank],
            index_wk[rank], index_norm[rank], index_wq_b[rank], index_wq_b_scale[rank],
            index_weights_proj[rank], attn_output[rank], attn_next_pre[rank], attn_post[rank],
            attn_comb[rank], attn_collapsed[rank], attn_normed[rank], after_attention[rank],
            ffn_hc_function[rank], ffn_hc_scale[rank], ffn_hc_base[rank], ffn_norm_weight[rank],
            ffn_next_pre[rank], ffn_post[rank], ffn_comb[rank], ffn_collapsed[rank], ffn_normed[rank],
            gate_weight[rank], text_bias[rank], vision_bias[rank], input_ids[rank], routed_w1[rank],
            routed_w1_scale[rank], routed_w3[rank], routed_w3_scale[rank], routed_w2[rank],
            routed_w2_scale[rank], shared_gate_up[rank], shared_gate_up_scale[rank], shared_down[rank],
            shared_down_scale[rank], lane_x, lane_aux, lane_count, lane_ready,
            slot_out, return_ready,
            logits[rank], route_ids[rank], route_weights[rank], moe_output[rank], output[rank],
            active, rank, epoch, reuse, has_engram, device=rank,
        )


__all__ = ["decode_layer_ep2_sparse", "decode_layer_ep2_sparse_rank"]
