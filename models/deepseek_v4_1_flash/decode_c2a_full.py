# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""C2A Full decode Attention half-layer composition entry."""

import inspect
import math

import pypto.language as pl
import torch

from config import (
    FLASH as M, LOCAL_T_DYN, TP_SIZE, T_DYN, ORI_BLOCKS_DYN, CMP_BLOCKS_DYN,
    INDEX_BLOCKS_DYN, B_DYN, TABLE_DYN, Q_START_DYN, STATE_BLOCKS_DYN,
    COMPRESSED_CACHE_GROUP, INDEX_CACHE_GROUP, STATE_CAPACITY, WINDOW_CACHE_GROUP,
    MAX_BATCH_PER_DP, BLOCK_SIZE,
)
from compressor import publish_c2a_compressed, STATE_WIDTH, project_ratio2, compressor_ratio2, golden_compressor_ratio2_paged
from hc import hc_mixes, golden_mhc_mixes
from hc import hc_pre, golden_hc_pre
from hc import hc_post, golden_hc_post
from rmsnorm import rms_norm_tiled, golden_rms_norm
from qkv_proj_rope import q_proj_qr, q_proj_rope, kv_proj_rope, golden_rope_tail, golden_qkv_proj_rope
from sparse_attention import (
    PACKED_QUERY_TILE as QUERY_TILE, SELECTED_KEYS as SPARSE_WIDTH,
    gather_packed_chunk, attend_c2a_chunk, publish_window_row,
    golden_selected_attention,
)
from decode_indexer import decode_c2a_indexer, golden_c2a_index
from o_proj import o_proj, golden_o_proj
from golden import TensorSpec, ScalarSpec, ratio_allclose, mapped_pool_ratio_allclose, run
from utils import AttentionGoldenResult, paged_slots, window_metadata
from quant import (
    quantize_grouped_mxfp8_weight,
    dequantize_mxfp4_cache, golden_bf16_gemm, golden_mxfp8_quantize,
    pack_mx_b_scale, quantize_mxfp4_cache, quantize_mxfp8_cache,
)
from rope_tables import precompute_rope_tables

from compressor import decode_c2a_compressor_rope

# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
LOCAL_O_GROUPS = M.o_groups // TP_SIZE
LOCAL_O_WIDTH = M.o_groups // TP_SIZE * M.o_lora_rank
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
O_LORA = M.o_lora_rank
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
INDEX_H = M.index_n_heads
INDEX_DIM = M.index_head_dim
INDEX_TOPK = M.index_topk
MX_GROUP = M.weight_block_size[0]

IDX_GROUP = INDEX_CACHE_GROUP
CMP_GROUP = COMPRESSED_CACHE_GROUP
RATIO = 2
E8M0 = getattr(torch, "float8_e8m0fnu", torch.uint8)



@pl.jit.inline(auto_scope=False)
def decode_c2a_full(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    output: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    next_pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    hc_mixes(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, next_pre_mix, post_mix, residual_mix)
    x_attn = pl.create_tensor([tokens, D], dtype=pl.BF16)
    hc_pre(x_hc, pre_mix, x_attn)
    x_norm = pl.create_tensor([tokens, D], dtype=pl.BF16)
    with pl.spmd((tokens + 7) // 8, name_hint="c2a_input_norm") as cache_ready:
        worker = pl.tile.get_block_idx()
        rms_norm_tiled(x_attn, attn_norm_weight, x_norm, worker, tokens)
    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    query = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    window_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    q_proj_qr(x_norm, wq_a, wq_a_scale, q_norm_weight, qr, num_tokens)
    q_proj_rope(qr, wq_b, wq_b_scale, rope_cos, rope_sin, query, num_tokens)
    kv_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    kv_proj_rope(x_norm, wkv, wkv_scale, kv_norm_weight, rope_cos, rope_sin, kv_projection, window_kv, num_tokens)
    publish_window_row(window_kv, window_slots, window_cache, window_cache_scale, num_tokens)

    compressor_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.FP32)
    compressor_score = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.FP32)
    project_ratio2(x_norm, compressor_wkv, compressor_wgate, compressor_kv, compressor_score, num_tokens)
    latent = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    pool_tid, state_tid = compressor_ratio2(
        compressor_kv, compressor_score, query_start_loc, position_ids, token_to_req_indices,
        state_block_table, state_cache, compressor_norm_weight, latent, num_tokens, cache_ready,
    )

    select_tid = decode_c2a_indexer(
        x_norm, qr, latent, index_wk, index_norm_weight, index_wq_b, index_wq_b_scale,
        index_weights_proj, rope_cos, rope_sin, compressed_rope_cos, compressed_rope_sin,
        compressed_slots, index_cache, index_cache_scale, index_block_table, token_to_req_indices,
        compressed_lens, topk_indices, num_tokens, cache_ready, pool_tid,
    )
    rotated_latent = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    rotated_tid = decode_c2a_compressor_rope(
        latent, compressed_rope_cos, compressed_rope_sin, rotated_latent, num_tokens, pool_tid
    )
    compressed_publish_tid = publish_c2a_compressed(
        rotated_latent, compressed_slots, compressed_cache, compressed_cache_scale,
        num_tokens, cache_ready, rotated_tid,
    )

    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    selected = pl.create_tensor([QUERY_TILE, SPARSE_WIDTH, HEAD_DIM], dtype=pl.BF16)
    combined = pl.create_tensor([QUERY_TILE, SPARSE_WIDTH], dtype=pl.INT32)
    chunk_done = pl.system.task_dummy(deps=[select_tid, state_tid, compressed_publish_tid])
    for start in pl.range(0, num_tokens, QUERY_TILE):
        active = pl.min(QUERY_TILE, num_tokens - start)
        gather_tid = gather_packed_chunk(
            window_cache, window_cache_scale, window_indices, compressed_cache,
            compressed_cache_scale, topk_indices, selected, combined, start, active, chunk_done,
        )
        chunk_done = attend_c2a_chunk(query, selected, combined, attn_sink, attended, start, active, gather_tid)

    partial = pl.create_tensor([tokens, D], dtype=pl.FP32)
    o_proj(attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin, partial, num_tokens, None, group32=False)
    attn_out = pl.create_tensor([tokens, D], dtype=pl.BF16)
    with pl.spmd(tokens, name_hint="c2a_output_cast"):
        row = pl.tile.get_block_idx()
        for col in pl.range(0, D, 512):
            value = pl.tile.full([1, 512], dtype=pl.BF16, value=0.0)
            if row < num_tokens:
                partial_tile = pl.load(partial, [row, col], [1, 512])
                value = pl.cast(partial_tile, pl.BF16, mode="rint")
            pl.store(value, [row, col], attn_out)
    hc_post(attn_out, x_hc, post_mix, residual_mix, output)
    return output, next_pre_mix


@pl.jit
def l2_decode_c2a_full(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.InOut[pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.InOut[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    output: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(0, LOCAL_T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    query_start_loc.bind_dynamic(0, Q_START_DYN)
    state_cache.bind_dynamic(0, STATE_BLOCKS_DYN)
    return decode_c2a_full(
        x_hc,
        pre_mix,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_norm_weight,
        wq_a,
        wq_a_scale,
        q_norm_weight,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        kv_norm_weight,
        attn_sink,
        wo_a, wo_a_scale,
        wo_b,
        wo_b_scale,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache,
        window_cache_scale,
        compressed_cache,
        compressed_cache_scale,
        token_to_req_indices,
        compressed_lens,
        index_cache,
        index_cache_scale,
        index_block_table,
        position_ids,
        compressed_rope_cos,
        compressed_rope_sin,
        compressor_wkv,
        compressor_wgate,
        query_start_loc,
        state_block_table,
        state_cache,
        compressor_norm_weight,
        compressed_slots,
        index_wk,
        index_norm_weight,
        index_wq_b,
        index_wq_b_scale,
        index_weights_proj,
        topk_indices,
        output,
        next_pre_mix,
        num_tokens,
    )


@pl.jit.host
def l3_decode_c2a_full(
    x_hc: pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[TP_SIZE, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[TP_SIZE, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[TP_SIZE, MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[TP_SIZE, D], pl.BF16],
    wq_a: pl.Tensor[[TP_SIZE, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[TP_SIZE, D // MX_GROUP, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[TP_SIZE, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[TP_SIZE, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[TP_SIZE, Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[TP_SIZE, D // MX_GROUP, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[TP_SIZE, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0],
    rope_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[TP_SIZE, T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[pl.Tensor[[TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]],
    token_to_req_indices: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[TP_SIZE, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[TP_SIZE, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[TP_SIZE, B_DYN, TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.FP32],
    compressor_wgate: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[TP_SIZE, Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[TP_SIZE, B_DYN, 1], pl.INT32],
    state_cache: pl.InOut[pl.Tensor[[TP_SIZE, STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    index_wk: pl.Tensor[[TP_SIZE, HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[TP_SIZE, INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[TP_SIZE, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[TP_SIZE, Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[TP_SIZE, D, INDEX_H], pl.BF16],
    topk_indices: pl.InOut[pl.Tensor[[TP_SIZE, T_DYN, INDEX_TOPK], pl.INT32]],
    output: pl.Out[pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(1, LOCAL_T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(1, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(1, B_DYN)
    index_block_table.bind_dynamic(2, TABLE_DYN)
    query_start_loc.bind_dynamic(1, Q_START_DYN)
    state_cache.bind_dynamic(1, STATE_BLOCKS_DYN)
    wq_a_scale_mx: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[0]
    wq_b_scale_mx: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[0]
    wkv_scale_mx: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[0]
    wo_b_scale_mx: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[0]
    index_wq_b_scale_mx: pl.Tensor[[Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = index_wq_b_scale[0]
    l2_decode_c2a_full(
        x_hc[0],
        pre_mix[0],
        hc_attn_fn[0],
        hc_attn_scale[0],
        hc_attn_base[0],
        attn_norm_weight[0],
        wq_a[0],
        wq_a_scale_mx,
        q_norm_weight[0],
        wq_b[0],
        wq_b_scale_mx,
        wkv[0],
        wkv_scale_mx,
        kv_norm_weight[0],
        attn_sink[0],
        wo_a[0], wo_a_scale[0],
        wo_b[0],
        wo_b_scale_mx,
        rope_cos[0],
        rope_sin[0],
        window_slots[0],
        window_indices[0],
        window_cache[0],
        window_cache_scale[0],
        compressed_cache[0],
        compressed_cache_scale[0],
        token_to_req_indices[0],
        compressed_lens[0],
        index_cache[0],
        index_cache_scale[0],
        index_block_table[0],
        position_ids[0],
        compressed_rope_cos[0],
        compressed_rope_sin[0],
        compressor_wkv[0],
        compressor_wgate[0],
        query_start_loc[0],
        state_block_table[0],
        state_cache[0],
        compressor_norm_weight[0],
        compressed_slots[0],
        index_wk[0],
        index_norm_weight[0],
        index_wq_b[0],
        index_wq_b_scale_mx,
        index_weights_proj[0],
        topk_indices[0],
        output[0],
        next_pre_mix[0],
        num_tokens,
        device=0,
    )


def _write_rows(cache, rows, payload, width):
    """Rewrite only ``rows`` of a paged cache, leaving every other byte alone."""
    cache.view(torch.uint8).reshape(-1, width)[rows] = payload.contiguous().view(torch.uint8).reshape(-1, width)


def golden_reference_c2a(inputs: dict, *, fp32_output=False) -> dict:
    """Ratio-2 compressed sparse attention; publishes only the addressed cache rows."""
    t = inputs
    x = t["x"]
    tokens = x.shape[0]
    index_dim = t["index_norm_weight"].numel()
    head_dim = t["window_cache"].shape[-1]

    # a. Query/KV projection and rotary embedding.
    q, kv, qr = golden_qkv_proj_rope(
        x, t["wq_a"], t["wq_a_scale"], t["q_norm_weight"], t["wq_b"], t["wq_b_scale"],
        t["wkv"], t["wkv_scale"], t["kv_norm_weight"], t["rope_cos"], t["rope_sin"],
        sequential_query=True,
    )

    # b. Sliding-window cache: MXFP8, group 32, BF16 scales; addressed rows only.
    window_cache = t["window_cache"].clone()
    window_cache_scale = t["window_cache_scale"].clone()
    window_payload, window_scales = quantize_mxfp8_cache(kv)
    window_slots = t["window_slots"].long()
    window_valid = window_slots >= 0
    window_rows = window_slots[window_valid]
    # Boolean indexing of Float8 on CPU requires an exact FP32 roundtrip.
    visible_payload = window_payload.float()[window_valid].to(window_payload.dtype)
    _write_rows(window_cache, window_rows, visible_payload, head_dim)
    window_cache_scale.reshape(-1, head_dim // MX_GROUP)[window_rows] = window_scales[window_valid]

    # c. Ratio-2 compressor: per-channel two-way softmax over the pair, pooled row rounded to
    #    BF16 before RMSNorm, published exactly when position_ids % 2 == 1. Ring rows
    #    contain the KV projection followed by the gate score.
    state_cache = t["state_cache"].clone()
    latent, publish = golden_compressor_ratio2_paged(
        x, t["query_start_loc"], t["position_ids"], t["token_to_req_indices"],
        t["state_block_table"], state_cache, t["compressor_wkv"], t["compressor_wgate"],
        t["compressor_norm_weight"],
    )
    # d. Compressed KV: rotate the latent tail, then packed FP4 with group-16 BF16 scales.
    compressed_cache = t["compressed_cache"].clone()
    compressed_cache_scale = t["compressed_cache_scale"].clone()
    rotated = golden_rope_tail(latent, t["compressed_rope_cos"], t["compressed_rope_sin"])
    compressed_slots = t["compressed_slots"].long()
    compressed_valid = (compressed_slots >= 0) & publish
    compressed_rows = compressed_slots[compressed_valid]
    if compressed_rows.numel():
        payload, scale = quantize_mxfp4_cache(rotated[compressed_valid], CMP_GROUP, "bf16")
        _write_rows(compressed_cache, compressed_rows, payload, head_dim // 2)
        # BF16 scales are 2 bytes/element; _write_rows' width is in bytes, so the
        # group-16 scale row (head_dim // CMP_GROUP elements) is twice that in bytes.
        _write_rows(compressed_cache_scale, compressed_rows, scale, (head_dim // CMP_GROUP) * 2)

    # Publish RNE E2M1 index keys with group-32 E8M0 scales.
    index_cache = t["index_cache"].clone()
    index_cache_scale = t["index_cache_scale"].clone()
    index_key = golden_rms_norm(golden_bf16_gemm(latent, t["index_wk"]), t["index_norm_weight"])
    index_key = golden_rope_tail(index_key, t["compressed_rope_cos"], t["compressed_rope_sin"])
    if compressed_rows.numel():
        payload, scale = quantize_mxfp4_cache(index_key[compressed_valid], IDX_GROUP, "e8m0", round_to_even=True)
        _write_rows(index_cache, compressed_rows, payload, index_dim // 2)
        _write_rows(index_cache_scale, compressed_rows, scale, index_dim // IDX_GROUP)

    # Score FP4 queries in original coordinate order with the shared BF16 epilogue.
    topk_indices = golden_c2a_index(t, qr, index_cache, index_cache_scale)

    # g. Sparse attention: one online softmax over window rows plus selected compressed rows.
    window_values = (
        window_cache.float() * window_cache_scale.float().repeat_interleave(MX_GROUP, -1)
    ).reshape(-1, head_dim)
    compressed_values = dequantize_mxfp4_cache(compressed_cache, compressed_cache_scale, CMP_GROUP, "bf16").flatten(0, -2)
    attended = golden_selected_attention(
        q, window_values.reshape(-1, 1, 1, head_dim), t["window_indices"],
        compressed_values.reshape(-1, 1, 1, head_dim), topk_indices, t["attn_sink"],
    )

    # h. Inverse RoPE, grouped low-rank output projection, per-rank partial.
    output = golden_o_proj(
        attended, t["rope_cos"], t["rope_sin"], t["wo_a"], t["wo_b"], t["wo_b_scale"],
        wo_a_scale=t["wo_a_scale"], output_dtype=torch.float32 if fp32_output else None, sequential_projection=True,
    )

    return {
        "output": output,
        "window_cache": window_cache,
        "window_cache_scale": window_cache_scale,
        "compressed_cache": compressed_cache,
        "compressed_cache_scale": compressed_cache_scale,
        "index_cache": index_cache,
        "index_cache_scale": index_cache_scale,
        "state_cache": state_cache,
        "topk_indices": topk_indices,
    }


def golden_decode_attn_c2a_full(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    attn_sink: torch.Tensor,
    wo_a: torch.Tensor,
    wo_a_scale: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    window_slots: torch.Tensor,
    window_indices: torch.Tensor,
    window_cache: torch.Tensor,
    window_cache_scale: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_cache_scale: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    compressed_lens: torch.Tensor,
    index_cache: torch.Tensor,
    index_cache_scale: torch.Tensor,
    index_block_table: torch.Tensor,
    position_ids: torch.Tensor,
    compressed_rope_cos: torch.Tensor,
    compressed_rope_sin: torch.Tensor,
    compressor_wkv: torch.Tensor,
    compressor_wgate: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_block_table: torch.Tensor,
    state_cache: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    compressed_slots: torch.Tensor,
    index_wk: torch.Tensor,
    index_norm_weight: torch.Tensor,
    index_wq_b: torch.Tensor,
    index_wq_b_scale: torch.Tensor,
    index_weights_proj: torch.Tensor,
) -> AttentionGoldenResult:
    """Evaluate C2A decode with request-owned ring state and paged caches."""
    inputs = locals()
    return AttentionGoldenResult(**golden_reference_c2a(inputs))


_PREFIXES = {
    "mixed": (1, 63, 128, 255, 0, 511, 129, 256),
    "long": (1024, 1027, 1290, 63, 1535, 128, 2048, 255),
}


def _weight(gen, k, n):
    """MXFP8 input-major weight with Cube-ordered MX_B_NN scales."""
    payload, codes = golden_mxfp8_quantize(torch.randn(n, k, generator=gen) / math.sqrt(k))
    return payload.T.contiguous(), pack_mx_b_scale(codes.T.contiguous()).view(E8M0)


def _plan(tokens, requests, case):
    """Ragged query lengths and per-request KV prefixes."""
    lengths = [tokens // requests + (index < tokens % requests) for index in range(requests)]
    pattern = _PREFIXES[case if case == "long" else "mixed"]
    prefixes = [pattern[index % len(pattern)] for index in range(requests)]
    return lengths, prefixes


def make_c2a_inputs(tokens=24, requests=6, seed=17, case="mixed", start_pos=None):
    """Deterministic inputs for ``golden_decode_attn_c2a_full``.

    case: "mixed" keeps contexts short so every causal position is selected, "long" pushes
    part of the batch past index_topk so the scored Top-K path runs, "masked" clears every
    slot and index, "zero" zeroes activations and all three cache payloads.
    """
    if case not in ("mixed", "long", "masked", "zero"):
        raise ValueError(f"unsupported case {case!r}")
    if not 1 <= requests <= min(MAX_BATCH_PER_DP, tokens):
        raise ValueError(f"requests must be in [1, min({MAX_BATCH_PER_DP}, tokens)]")
    gen = torch.Generator().manual_seed(seed)

    lengths, prefixes = _plan(tokens, requests, case)
    if start_pos is not None:
        lengths, prefixes = [1] * requests, list(start_pos)
    token_to_req_indices = torch.repeat_interleave(torch.arange(requests), torch.tensor(lengths))
    positions = torch.cat(
        [torch.arange(prefix, prefix + length) for prefix, length in zip(prefixes, lengths)]
    )
    maximum = int(positions.max())
    window_columns = maximum // BLOCK_SIZE + 1
    compressed_columns = max(4, (maximum // RATIO) // BLOCK_SIZE + 1)

    # One spare page per cache is never mapped: a control row for untouched-byte checks.
    window_pages = requests * window_columns + 1
    compressed_pages = requests * compressed_columns + 1
    def table(pages, columns):
        rows = torch.randperm(pages, generator=gen)[: requests * columns]
        return rows.reshape(requests, columns).to(torch.int32)

    window_table = table(window_pages, window_columns)
    index_block_table = table(compressed_pages, compressed_columns)

    window_slots, window_indices, _ = window_metadata(positions, token_to_req_indices, window_table)
    compressed_slots = paged_slots(positions, token_to_req_indices, index_block_table, BLOCK_SIZE, RATIO, True)
    compressed_lens = torch.div(positions + 1, RATIO, rounding_mode="floor").to(torch.int32)

    # Fixture initialization only: production callers retain these full tables.
    table_rows = int(positions.max()) + 1
    freqs_cos, freqs_sin = precompute_rope_tables(table_rows, compressed_attention=True)
    compressed_freqs_cos, compressed_freqs_sin = freqs_cos, freqs_sin
    rope_cos, rope_sin = freqs_cos[positions], freqs_sin[positions]
    complete = (positions + 1).remainder(RATIO) == 0
    pair_start = torch.where(complete, positions + 1 - RATIO, torch.full_like(positions, -1))
    compressed_rope_cos = compressed_freqs_cos[pair_start.clamp_min(0)].masked_fill(~complete[:, None], 1.0)
    compressed_rope_sin = compressed_freqs_sin[pair_start.clamp_min(0)].masked_fill(~complete[:, None], 0.0)

    wq_a, wq_a_scale = _weight(gen, D, Q_LORA)
    wq_b, wq_b_scale = _weight(gen, Q_LORA, LOCAL_H * HEAD_DIM)
    wkv, wkv_scale = _weight(gen, D, HEAD_DIM)
    wo_b, wo_b_scale = _weight(gen, LOCAL_O_WIDTH, D)
    index_wq_b, index_wq_b_scale = _weight(gen, Q_LORA, INDEX_H * INDEX_DIM)

    compressor_wkv = torch.randn(D, HEAD_DIM, generator=gen) / math.sqrt(D)
    compressor_wgate = torch.randn(D, HEAD_DIM, generator=gen) / math.sqrt(D)
    wo_a_values = torch.randn(LOCAL_O_GROUPS, O_LORA, O_GROUP_IN, generator=gen)
    # Seed each request's predecessor projections in its physical ring block.
    pending = torch.randn(MAX_BATCH_PER_DP, D, generator=gen).bfloat16().float()
    state_cache = torch.zeros(MAX_BATCH_PER_DP + 1, STATE_CAPACITY, STATE_WIDTH)
    state_block_table = torch.arange(requests - 1, -1, -1, dtype=torch.int32).unsqueeze(1) + 1
    for request, prefix in enumerate(prefixes):
        if prefix:
            block = int(state_block_table[request, 0])
            slot = (prefix - 1) % STATE_CAPACITY
            state_cache[block, slot, :HEAD_DIM] = pending[request] @ compressor_wkv
            state_cache[block, slot, HEAD_DIM:] = pending[request] @ compressor_wgate
    query_start_loc = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)

    window_cache, window_cache_scales = quantize_mxfp8_cache(
        torch.randn(window_pages, BLOCK_SIZE, 1, HEAD_DIM, generator=gen).bfloat16()
    )
    compressed_cache, compressed_cache_scale = quantize_mxfp4_cache(
        torch.randn(compressed_pages, BLOCK_SIZE, 1, HEAD_DIM, generator=gen).bfloat16(),
        CMP_GROUP,
        "bf16",
    )
    index_cache, index_cache_codes = quantize_mxfp4_cache(
        torch.randn(compressed_pages, BLOCK_SIZE, 1, INDEX_DIM, generator=gen).bfloat16(),
        IDX_GROUP,
        "e8m0",
    )

    values = {
        "x": torch.randn(tokens, D, generator=gen).bfloat16(),
        "wq_a": wq_a,
        "wq_a_scale": wq_a_scale,
        "q_norm_weight": (torch.randn(Q_LORA, generator=gen) * 0.1 + 1).bfloat16(),
        "wq_b": wq_b,
        "wq_b_scale": wq_b_scale,
        "wkv": wkv,
        "wkv_scale": wkv_scale,
        "kv_norm_weight": (torch.randn(HEAD_DIM, generator=gen) * 0.1 + 1).bfloat16(),
        "attn_sink": torch.randn(LOCAL_H, generator=gen) * 2,
        "wo_a": (wo_a_values / math.sqrt(O_GROUP_IN)).bfloat16(),
        "wo_b": wo_b,
        "wo_b_scale": wo_b_scale,
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
        "window_slots": window_slots,
        "window_indices": window_indices,
        "window_cache": window_cache,
        "window_cache_scale": window_cache_scales,
        "compressed_cache": compressed_cache,
        "compressed_cache_scale": compressed_cache_scale,
        "token_to_req_indices": token_to_req_indices.to(torch.int32),
        "compressed_lens": compressed_lens,
        "index_cache": index_cache,
        "index_cache_scale": index_cache_codes.view(E8M0),
        "index_block_table": index_block_table,
        "position_ids": positions.to(torch.int32),
        "compressed_rope_cos": compressed_rope_cos,
        "compressed_rope_sin": compressed_rope_sin,
        "compressor_wkv": compressor_wkv,
        "compressor_wgate": compressor_wgate,
        "query_start_loc": query_start_loc,
        "state_block_table": state_block_table,
        "state_cache": state_cache,
        "compressor_norm_weight": (torch.randn(HEAD_DIM, generator=gen) * 0.1 + 1).bfloat16(),
        "compressed_slots": compressed_slots,
        "index_wk": (torch.randn(HEAD_DIM, INDEX_DIM, generator=gen) / math.sqrt(HEAD_DIM)).bfloat16(),
        "index_norm_weight": (torch.randn(INDEX_DIM, generator=gen) * 0.1 + 1).bfloat16(),
        "index_wq_b": index_wq_b,
        "index_wq_b_scale": index_wq_b_scale,
        "index_weights_proj": (torch.randn(D, INDEX_H, generator=gen) / math.sqrt(D)).bfloat16(),
    }

    values["wo_a"], values["wo_a_scale"] = quantize_grouped_mxfp8_weight(values["wo_a"])

    if case == "masked":
        values["window_slots"] = torch.full_like(window_slots, -1)
        values["window_indices"] = torch.full_like(window_indices, -1)
        values["compressed_slots"] = torch.full_like(compressed_slots, -1)
        values["compressed_lens"] = torch.zeros_like(compressed_lens)
        values["state_block_table"] = torch.full_like(state_block_table, -1)
    if case == "zero":
        values["x"] = torch.zeros_like(values["x"])
        values["window_cache"] = torch.zeros_like(window_cache.view(torch.uint8)).view(window_cache.dtype)
        values["compressed_cache"] = torch.zeros_like(compressed_cache)
        values["index_cache"] = torch.zeros_like(index_cache)
        values["state_cache"] = torch.zeros_like(state_cache)

    print(
        f"[FIXTURE] c2a tokens={tokens} requests={requests} case={case} "
        f"lengths={lengths} prefixes={prefixes} "
        f"max_compressed_len={int(values['compressed_lens'].max())} "
        f"window_pages={window_pages} compressed_pages={compressed_pages} "
        f"index_block_table={tuple(index_block_table.shape)}"
    )
    return values


MUTABLE_NAMES = (
    "window_cache",
    "window_cache_scale",
    "compressed_cache",
    "compressed_cache_scale",
    "index_cache",
    "index_cache_scale",
    "state_cache",
)


CACHE_SLOTS = {
    "window_cache": "window_slots",
    "window_cache_scale": "window_slots",
    "compressed_cache": "compressed_slots",
    "compressed_cache_scale": "compressed_slots",
    "index_cache": "compressed_slots",
    "index_cache_scale": "compressed_slots",
}


CACHE_PAYLOADS = {
    "window_cache": ("window_cache_scale", None, None),
    "compressed_cache": ("compressed_cache_scale", 16, "bf16"),
    "index_cache": ("index_cache_scale", 32, "e8m0"),
}


def cache_compare(world_size, tokens, names=None):
    """Mapped rows must match byte for byte, except a derived scale allows a one-code mismatch.

    An E8M0 scale is a bare exponent, coarse enough that device and CPU agree exactly, and so
    does every payload here. A BF16 scale (``compressed_cache_scale``) is a derived quantity
    (amax over a group), so the device and the CPU golden can differ by a ULP and flip one
    code at a quantization boundary the same way a payload can; every other mapped row stays
    unchanged.
    """
    options = dict(
        mapping_shape=(world_size, tokens),
        block_size=BLOCK_SIZE,
        leading_rank_axis=True,
        atol=0.0,
        rtol=0.0,
    )
    inexact_scales = {
        scale_name for scale_name, _, scale_format in CACHE_PAYLOADS.values()
        if scale_format not in (None, "e8m0")
    }
    return {
        name: mapped_pool_ratio_allclose(
            CACHE_SLOTS[name],
            pool_name=name,
            max_error_ratio=0.01 if name in inexact_scales else 0.0,
            **options,
        )
        for name in (CACHE_SLOTS if names is None else names)
    }


REUSE_INPUT_NAMES = (
    "x", "wq_a", "wq_a_scale", "q_norm_weight", "wq_b", "wq_b_scale", "wkv", "wkv_scale",
    "kv_norm_weight", "attn_sink", "wo_a", "wo_a_scale", "wo_b", "wo_b_scale", "rope_cos", "rope_sin",
    "window_slots", "window_indices", "window_cache", "window_cache_scale", "compressed_cache",
    "compressed_cache_scale", "compressed_indices",
)


def golden_reference_c2a_reuse(inputs: dict, *, fp32_output=False) -> dict:
    """Ratio-2 reuse attention; publishes only the addressed sliding-window rows.

    Same transcription as the full-mode reference, with the compressor, the index-key
    publication and the indexer removed: the selection arrives as an input and the
    compressed pool is read-only.
    """
    t = inputs
    x = t["x"]
    tokens = x.shape[0]
    head_dim = t["window_cache"].shape[-1]

    q, kv, qr = golden_qkv_proj_rope(
        x, t["wq_a"], t["wq_a_scale"], t["q_norm_weight"], t["wq_b"], t["wq_b_scale"],
        t["wkv"], t["wkv_scale"], t["kv_norm_weight"], t["rope_cos"], t["rope_sin"],
        sequential_query=True,
    )

    window_cache = t["window_cache"].clone()
    window_cache_scale = t["window_cache_scale"].clone()
    window_payload, window_scales = quantize_mxfp8_cache(kv)
    window_slots = t["window_slots"].long()
    window_valid = window_slots >= 0
    window_rows = window_slots[window_valid]
    # Boolean indexing of Float8 on CPU requires an exact FP32 roundtrip.
    visible_payload = window_payload.float()[window_valid].to(window_payload.dtype)
    _write_rows(window_cache, window_rows, visible_payload, head_dim)
    window_cache_scale.reshape(-1, head_dim // MX_GROUP)[window_rows] = window_scales[window_valid]

    window_values = (
        window_cache.float() * window_cache_scale.float().repeat_interleave(MX_GROUP, -1)
    ).reshape(-1, head_dim)
    compressed_values = dequantize_mxfp4_cache(
        t["compressed_cache"], t["compressed_cache_scale"], CMP_GROUP, "bf16"
    ).flatten(0, -2)
    attended = golden_selected_attention(
        q, window_values.reshape(-1, 1, 1, head_dim), t["window_indices"],
        compressed_values.reshape(-1, 1, 1, head_dim), t["compressed_indices"], t["attn_sink"],
    )

    output = golden_o_proj(
        attended, t["rope_cos"], t["rope_sin"], t["wo_a"], t["wo_b"], t["wo_b_scale"],
        wo_a_scale=t["wo_a_scale"], output_dtype=torch.float32 if fp32_output else None, sequential_projection=True,
    )

    return {
        "output": output,
        "window_cache": window_cache,
        "window_cache_scale": window_cache_scale,
    }


def make_c2a_reuse_inputs(tokens=32, requests=6, seed=17, case="mixed", start_pos=None):
    """Inputs for a reuse layer, taken from what a full layer actually published.

    The compressed pool and the Top-K row list come from running the full-mode
    reference on the same batch, so the fixture exercises the real Full-to-Reuse
    hand-off rather than a synthetic selection.
    """
    source = make_c2a_inputs(tokens=tokens, requests=requests, seed=seed, case=case, start_pos=start_pos)
    published = golden_reference_c2a(source)
    values = {name: source[name] for name in REUSE_INPUT_NAMES if name != "compressed_indices"}
    values["compressed_cache"] = published["compressed_cache"]
    values["compressed_cache_scale"] = published["compressed_cache_scale"]
    values["compressed_indices"] = published["topk_indices"]
    reachable = int((published["topk_indices"] >= 0).sum())
    print(
        f"[FIXTURE] c2a reuse tokens={tokens} requests={requests} case={case} "
        f"selected_rows={reachable} source_published_rows="
        f"{int((source['compressed_slots'] >= 0).sum())}"
    )
    return values


def build_hc_validation_values(mode, tokens=24, requests=6, seed=17, case="mixed", start_pos=None):
    """Build TP1 C2A weights, cache state, and mHC inputs."""
    values = (make_c2a_inputs if mode == "full" else make_c2a_reuse_inputs)(
        tokens=tokens, requests=requests, seed=seed, case=case, start_pos=start_pos,
    )
    del values["x"]
    values = {name: value.unsqueeze(0) for name, value in values.items()}
    generator = torch.Generator().manual_seed(seed + 1000)
    shapes = {
        "x_hc": [tokens, HC_MULT, D],
        "pre_mix": [tokens, HC_MULT],
        "hc_attn_fn": [MIX_HC, HC_DIM],
        "hc_attn_scale": [3],
        "hc_attn_base": [MIX_HC],
        "attn_norm_weight": [D],
    }
    for name, shape in shapes.items():
        value = torch.randn(shape, generator=generator)
        if name == "hc_attn_fn":
            value /= HC_DIM**0.5
        elif name == "attn_norm_weight":
            value = torch.ones(shape, dtype=torch.bfloat16)
        if name in ("x_hc", "pre_mix"):
            value = torch.randn([1, *shape], generator=generator)
            if name == "pre_mix":
                value = torch.softmax(value, dim=-1)
        else:
            value = value.unsqueeze(0)
        values[name] = value
    if mode == "full":
        values["topk_indices"] = torch.full((1, tokens, INDEX_TOPK), -1, dtype=torch.int32)
    values["output"] = torch.zeros(1, tokens, HC_MULT, D)
    values["next_pre_mix"] = torch.zeros(1, tokens, HC_MULT)
    return values


CACHE_STATE_NAMES = (*MUTABLE_NAMES, "topk_indices")


def golden_c2a_hc_case(tensors, attn_reference):
    """Compose TP1 mHC, C2A attention, cache publication, and mHC post."""
    x_hc = tensors["x_hc"][0]
    pre_mix, post_mix, residual_mix = golden_mhc_mixes(
        x_hc, tensors["hc_attn_fn"][0], tensors["hc_attn_scale"][0], tensors["hc_attn_base"][0],
    )
    tensors["next_pre_mix"][0].copy_(pre_mix)
    x_attn = golden_hc_pre(x_hc, tensors["pre_mix"][0]).to(torch.bfloat16)
    x_norm = golden_rms_norm(x_attn, tensors["attn_norm_weight"][0])
    names = inspect.signature(attn_reference).parameters
    inputs = {name: tensors[name][0] for name in names if name != "x"}
    result = attn_reference(x=x_norm, **inputs)
    for name in CACHE_STATE_NAMES:
        value = getattr(result, name, None)
        if value is not None and name in tensors:
            tensors[name][0].view(torch.uint8).copy_(value.contiguous().view(torch.uint8))
    sublayer = result.output.bfloat16()
    tensors["output"][0].copy_(golden_hc_post(sublayer, x_hc, post_mix, residual_mix).to(torch.float32))


def golden_decode_c2a_full_case(tensors):
    golden_c2a_hc_case(tensors, golden_decode_attn_c2a_full)


if __name__ == "__main__":
    from utils import parse_decode_startpos

    import argparse
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="TP1 A5 mHC C2A full validation")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    startpos_help = "one position per request; omitted: default validation workload"
    parser.add_argument("--startpos", type=parse_decode_startpos, help=startpos_help)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--case", choices=["mixed", "long", "masked", "zero"], default="mixed")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.set_defaults(tokens=24, requests=6)
    args = parser.parse_args()
    if args.startpos is not None:
        args.tokens = args.requests = len(args.startpos)
    values = build_hc_validation_values("full", args.tokens, args.requests, args.seed, args.case, start_pos=args.startpos)
    host = l3_decode_c2a_full
    specs = [
        TensorSpec(name, list(values[name].shape), values[name].dtype, init_value=lambda name=name: values[name].clone())
        for name in host.param_names if name != "num_tokens"
    ]
    specs.append(ScalarSpec("num_tokens", torch.int32, args.tokens))
    comparisons = {
        "output": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
        "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
    }
    comparisons.update(cache_compare(1, args.tokens))
    comparisons.update({
        "state_cache": ratio_allclose(atol=1e-3, rtol=1e-3),
        "topk_indices": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.02),
    })
    result = run(
        fn=host,
        specs=specs,
        golden_fn=golden_decode_c2a_full_case,
        compare_fn=comparisons,
        compile_only=args.compile_only,
        save_data=args.save_data,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        config=dict(
            platform=args.platform,
            enable_chip_swimlane=args.enable_chip_swimlane,
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0),
        ),
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
