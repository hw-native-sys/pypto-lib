# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode layer assembly on the paged dynamic-batch interface.

The layer dispatches inline on one attention mode at trace time: the mode chain
owns HC pre, RMSNorm, the paged projection/publication/selection chain, TP
output reduce and HC post; the layer owns Engram and the FFN prolog (HC pre
with the attention's delayed pre-mix, RMSNorm, the sqrtsoftplus gate).  Because
a per-token EP16 exchange cannot share a compilation unit with the dynamic
typed chain (see KNOWN_PYPTO_ISSUES.md), the exchange runs in a second, static
host program and the final HC post in a third, both staged over shared host
buffers.  Decode differs from prefill only in the bounded token count.

The layer file is validation-only; the forward assembles its own chain loop."""

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from config import (
    B_DYN, BLOCK_SIZE, CMP_BLOCKS_DYN, EP_SIZE, FLASH as M, INDEX_BLOCKS_DYN, LOCAL_HEADS,
    N_LOCAL_EXPERTS, ORI_BLOCKS_DYN, POOL_BLOCKS_DYN, DECODE_MAX_TOKENS, STATE_BLOCKS_DYN,
    STATE_CAPACITY, TABLE_DYN, T_DYN, TP_SIZE, WINDOW_KEYS,
)
from engram import engram_from_rows
from hc import hc_post, hc_pre
from layer_reference import (
    MODE_C1A_FULL, MODE_C1A_REINDEX, MODE_C1A_REUSE, MODE_C2A_FULL,
    MODE_C2A_REUSE, MODE_SWA, build_compare, build_layer_specs,
    make_layer_golden, run_staged_programs,
)
from moe_exchange import COUNT_PAD, moe_ep
from moe_gate import moe_gate
from decode_c1a_full import decode_c1a_full
from decode_c1a_reindex import decode_c1a_reindex
from decode_c2a_full import decode_c2a_full
from decode_c2a_reuse import decode_reuse
from decode_swa import decode_swa
from rmsnorm import rms_norm


# model config
D = M.hidden_size
HEAD_DIM = M.head_dim
Q_LORA = M.q_lora_rank
LOCAL_Q = LOCAL_HEADS * HEAD_DIM
O_GROUPS = M.o_groups // (M.num_attention_heads // LOCAL_HEADS)
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
O_LORA = M.o_lora_rank
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
MIX_PAD = 32
INDEX_TOPK = M.index_topk
IDX_DIM = M.index_head_dim
IDX_HEADS = M.index_n_heads
TOPK = M.num_experts_per_tok
N_EXPERTS = M.n_routed_experts
INTER = M.moe_intermediate_size
ENGRAM_K = (M.engram_max_ngram_size - 1) * M.engram_n_heads * M.engram_head_dim
ENGRAM_OUT = (HC_MULT + 1) * D
STATE_WIDTH = 2 * HEAD_DIM

# FFN/MoE tiling: one host-driven EP16 exchange per layer.  The bounded layer
# suite carries at most 16 rows per rank; a wider token range compiles the tile
# at its own bound.  The receive windows hold one token per (local expert,
# source) pair for the whole tile.
# Exchange tile = the compiled token bound (the bounded decode plan carries two
# tokens per rank); an exactly sized tile keeps pad rows out of the exchange.
MOE_TILE = 2
MOE_RECV_ROWS = N_LOCAL_EXPERTS * EP_SIZE * MOE_TILE
MOE_SLOT_ROWS = MOE_TILE * TOPK


@pl.jit.inline(auto_scope=False)
def decode_layer(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_PAD, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[Q_LORA], pl.FP32],
    q_norm_w: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_Q], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[LOCAL_Q], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[HEAD_DIM], pl.FP32],
    kv_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    sink: pl.Tensor[[LOCAL_HEADS], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, WINDOW_KEYS], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_DIM], pl.BF16],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    compressed_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    compressor_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    compressor_wgate: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8, pl.NZ],
    index_wq_b_scale: pl.Tensor[[IDX_HEADS * IDX_DIM], pl.FP32],
    index_weights_proj: pl.Tensor[[D, IDX_HEADS], pl.BF16],
    index_wk: pl.Tensor[[IDX_DIM, HEAD_DIM], pl.BF16],
    index_norm_w: pl.Tensor[[IDX_DIM], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    topk_values: pl.Tensor[[T_DYN, INDEX_TOPK], pl.FP32],
    pool_block_ids: pl.Tensor[[T_DYN, POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    engram_lookup: pl.Tensor[[T_DYN, ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[ENGRAM_OUT, ENGRAM_K], pl.BF16],
    engram_qk: pl.Tensor[[4, D], pl.FP32],
    engram_rotation: pl.Tensor[[32, 32], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_PAD, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    hidden: pl.Tensor[[T_DYN, D], pl.BF16],
    attn_input: pl.Tensor[[T_DYN, D], pl.BF16],
    attn_out: pl.Tensor[[T_DYN, D], pl.BF16],
    x_hc_mid: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    ffn_input: pl.Tensor[[T_DYN, D], pl.BF16],
    ffn_normed: pl.Tensor[[T_DYN, D], pl.BF16],
    route_indices: pl.Tensor[[T_DYN, TOPK], pl.INT32],
    route_weights: pl.Tensor[[T_DYN, TOPK], pl.FP32],
    ffn_post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    ffn_residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    next_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    mode: pl.constexpr,
    has_engram: pl.constexpr,
):
    """One decoder layer: inline per-mode attention chain, then the FFN prolog.

    The mode dispatch is resolved at trace time (``mode`` is a constexpr), so a
    compiled program carries exactly one attention chain.  The chain owns HC
    pre, RMSNorm, attention, TP reduce and HC post; this function owns Engram
    and the FFN prolog, and stages the rows for the separate EP16 program.
    """
    tokens = pl.tensor.dim(x_hc, 0)
    attn_pre_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    attention_x = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
    if has_engram == 1:
        _engram_ready = engram_from_rows(engram_lookup, engram_wkv, engram_qk, engram_rotation, x_hc, attention_x)
    else:
        source_hc = pl.reshape(x_hc, [tokens * HC_MULT, D])
        target_hc = pl.reshape(attention_x, [tokens * HC_MULT, D])
        for hc_row in pl.spmd(tokens * HC_MULT, name_hint="attention_input_copy"):
            row_value = source_hc[hc_row : hc_row + 1, 0:D]
            target_hc[hc_row : hc_row + 1, 0:D] = row_value
    if mode == MODE_SWA:
        decode_swa(
            attention_x, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
            wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
            sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache,
            output_window, output_arrived, hidden, attn_input, attn_out, x_hc_mid, attn_pre_mix,
            group_base, tp_rank, num_tokens, attention_epoch,
        )
    elif mode == MODE_C2A_FULL:
        decode_c2a_full(
            attention_x, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
            wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
            sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache,
            compressed_cache, request_ids, compressed_lens, index_cache, index_block_table,
            compressed_cos, compressed_sin, compressed_slots,
            position_ids, state_block_table, state_cache,
            compressor_wkv, compressor_wgate, compressor_norm,
            index_wq_b, index_wq_b_scale, index_weights_proj, index_wk, index_norm_w,
            topk_indices, topk_values, pool_block_ids, pool_block_counts,
            output_window, output_arrived, hidden, attn_input, attn_out, x_hc_mid, attn_pre_mix,
            group_base, tp_rank, num_tokens, attention_epoch,
        )
    elif mode == MODE_C2A_REUSE:
        decode_reuse(
            attention_x, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
            wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
            sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache,
            compressed_cache, topk_indices,
            output_window, output_arrived, hidden, attn_input, attn_out, x_hc_mid, attn_pre_mix,
            group_base, tp_rank, num_tokens, attention_epoch,
        )
    elif mode == MODE_C1A_FULL:
        decode_c1a_full(
            attention_x, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
            wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
            sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache,
            compressed_cache, request_ids, compressed_lens, index_cache, index_block_table,
            compressed_cos, compressed_sin, compressed_slots,
            compressor_wkv, compressor_norm,
            index_wq_b, index_wq_b_scale, index_weights_proj, index_wk, index_norm_w,
            topk_indices, topk_values, pool_block_ids, pool_block_counts,
            output_window, output_arrived, hidden, attn_input, attn_out, x_hc_mid, attn_pre_mix,
            group_base, tp_rank, num_tokens, attention_epoch,
        )
    elif mode == MODE_C1A_REUSE:
        decode_reuse(
            attention_x, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
            wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
            sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache,
            compressed_cache, topk_indices,
            output_window, output_arrived, hidden, attn_input, attn_out, x_hc_mid, attn_pre_mix,
            group_base, tp_rank, num_tokens, attention_epoch,
        )
    else:
        decode_c1a_reindex(
            attention_x, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
            wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
            sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache,
            compressed_cache, request_ids, compressed_lens, index_cache, index_block_table,
            index_wq_b, index_wq_b_scale, index_weights_proj,
            topk_indices, topk_values, pool_block_ids, pool_block_counts,
            output_window, output_arrived, hidden, attn_input, attn_out, x_hc_mid, attn_pre_mix,
            group_base, tp_rank, num_tokens, attention_epoch,
        )
    hc_pre(
        x_hc_mid, attn_pre_mix, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        ffn_input, next_pre_mix, ffn_post_mix, ffn_residual_mix,
    )
    ffn_normed = rms_norm(ffn_input, ffn_norm_w, ffn_normed, width=D, output_fp32=False, column_tile=256)
    route_weights = moe_gate(
        ffn_normed, gate_weight, gate_bias, route_indices, route_weights, topk=TOPK, topk_pad=8,
    )


def _build_layer_entries(mode, has_engram):
    """Build the phase-A rank and the host phase orchestrator for one mode.

    The host entry runs five device phases per layer: the mode chain plus FFN
    prolog (phase A), the padded MoE-tile staging, the EP16 exchange (phase B),
    the routed-row unpad and the final HC post (phase C).  Every phase launches
    on the same 16 rank devices, so per-device stream order carries the
    intermediate buffers between phases.
    """

    @pl.jit
    def decode_layer_rank(
        x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
        pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
        hc_attn_fn: pl.Tensor[[MIX_PAD, HC_DIM], pl.FP32],
        hc_attn_scale: pl.Tensor[[3], pl.FP32],
        hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
        attn_norm_w: pl.Tensor[[D], pl.BF16],
        wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ],
        wq_a_scale: pl.Tensor[[Q_LORA], pl.FP32],
        q_norm_w: pl.Tensor[[Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[Q_LORA, LOCAL_Q], pl.INT8, pl.NZ],
        wq_b_scale: pl.Tensor[[LOCAL_Q], pl.FP32],
        wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ],
        wkv_scale: pl.Tensor[[HEAD_DIM], pl.FP32],
        kv_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
        sink: pl.Tensor[[LOCAL_HEADS], pl.FP32],
        wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.BF16],
        rope_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
        window_slots: pl.Tensor[[T_DYN], pl.INT64],
        window_indices: pl.Tensor[[T_DYN, WINDOW_KEYS], pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
        compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
        request_ids: pl.Tensor[[T_DYN], pl.INT32],
        compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
        index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_DIM], pl.BF16]],
        index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
        compressed_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
        compressed_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
        compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
        position_ids: pl.Tensor[[T_DYN], pl.INT32],
        state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
        state_cache: pl.InOut[pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
        compressor_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
        compressor_wgate: pl.Tensor[[HEAD_DIM, D], pl.BF16],
        compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16],
        index_wq_b: pl.Tensor[[Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8, pl.NZ],
        index_wq_b_scale: pl.Tensor[[IDX_HEADS * IDX_DIM], pl.FP32],
        index_weights_proj: pl.Tensor[[D, IDX_HEADS], pl.BF16],
        index_wk: pl.Tensor[[IDX_DIM, HEAD_DIM], pl.BF16],
        index_norm_w: pl.Tensor[[IDX_DIM], pl.BF16],
        topk_indices: pl.InOut[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
        topk_values: pl.InOut[pl.Tensor[[T_DYN, INDEX_TOPK], pl.FP32]],
        pool_block_ids: pl.InOut[pl.Tensor[[T_DYN, POOL_BLOCKS_DYN], pl.INT32]],
        pool_block_counts: pl.InOut[pl.Tensor[[T_DYN], pl.INT32]],
        engram_lookup: pl.Tensor[[T_DYN, ENGRAM_K], pl.BF16],
        engram_wkv: pl.Tensor[[ENGRAM_OUT, ENGRAM_K], pl.BF16],
        engram_qk: pl.Tensor[[4, D], pl.FP32],
        engram_rotation: pl.Tensor[[32, 32], pl.FP32],
        hc_ffn_fn: pl.Tensor[[MIX_PAD, HC_DIM], pl.FP32],
        hc_ffn_scale: pl.Tensor[[3], pl.FP32],
        hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
        ffn_norm_w: pl.Tensor[[D], pl.BF16],
        gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
        gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
        hidden: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
        attn_input: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
        attn_out: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
        x_hc_mid: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
        ffn_input: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
        ffn_normed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
        route_indices: pl.Out[pl.Tensor[[T_DYN, TOPK], pl.INT32]],
        route_weights: pl.Out[pl.Tensor[[T_DYN, TOPK], pl.FP32]],
        ffn_post_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
        ffn_residual_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32]],
        next_pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
        moe_x: pl.Out[pl.Tensor[[MOE_TILE, D], pl.BF16]],
        moe_indices: pl.Out[pl.Tensor[[MOE_TILE, TOPK], pl.INT32]],
        moe_weights: pl.Out[pl.Tensor[[MOE_TILE, TOPK], pl.FP32]],
        output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.BF16],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
        group_base: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
        token_counts: pl.Tensor[[EP_SIZE], pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Run the mode chain and the FFN prolog for one rank (phase A)."""
        x_hc.bind_dynamic(0, T_DYN)
        pre_mix.bind_dynamic(0, T_DYN)
        rope_cos.bind_dynamic(0, T_DYN)
        rope_sin.bind_dynamic(0, T_DYN)
        window_slots.bind_dynamic(0, T_DYN)
        window_indices.bind_dynamic(0, T_DYN)
        window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
        compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
        request_ids.bind_dynamic(0, T_DYN)
        compressed_lens.bind_dynamic(0, T_DYN)
        index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
        index_block_table.bind_dynamic(0, B_DYN)
        index_block_table.bind_dynamic(1, TABLE_DYN)
        compressed_cos.bind_dynamic(0, T_DYN)
        compressed_sin.bind_dynamic(0, T_DYN)
        compressed_slots.bind_dynamic(0, T_DYN)
        position_ids.bind_dynamic(0, T_DYN)
        state_block_table.bind_dynamic(0, B_DYN)
        state_cache.bind_dynamic(0, STATE_BLOCKS_DYN)
        topk_indices.bind_dynamic(0, T_DYN)
        topk_values.bind_dynamic(0, T_DYN)
        pool_block_ids.bind_dynamic(0, T_DYN)
        pool_block_ids.bind_dynamic(1, POOL_BLOCKS_DYN)
        pool_block_counts.bind_dynamic(0, T_DYN)
        engram_lookup.bind_dynamic(0, T_DYN)
        hidden.bind_dynamic(0, T_DYN)
        attn_input.bind_dynamic(0, T_DYN)
        attn_out.bind_dynamic(0, T_DYN)
        x_hc_mid.bind_dynamic(0, T_DYN)
        ffn_input.bind_dynamic(0, T_DYN)
        ffn_normed.bind_dynamic(0, T_DYN)
        route_indices.bind_dynamic(0, T_DYN)
        route_weights.bind_dynamic(0, T_DYN)
        ffn_post_mix.bind_dynamic(0, T_DYN)
        ffn_residual_mix.bind_dynamic(0, T_DYN)
        next_pre_mix.bind_dynamic(0, T_DYN)
        num_tokens = pl.read(token_counts, [my_rank])
        decode_layer(
            x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
            wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv,
            wkv_scale, kv_norm_w, sink, wo_a, wo_b, rope_cos,
            rope_sin, window_slots, window_indices, window_cache, compressed_cache, request_ids,
            compressed_lens, index_cache, index_block_table, compressed_cos, compressed_sin, compressed_slots,
            position_ids, state_block_table, state_cache, compressor_wkv, compressor_wgate, compressor_norm,
            index_wq_b, index_wq_b_scale, index_weights_proj, index_wk, index_norm_w, topk_indices,
            topk_values, pool_block_ids, pool_block_counts, engram_lookup, engram_wkv, engram_qk, engram_rotation,
            hc_ffn_fn, hc_ffn_scale, hc_ffn_base, ffn_norm_w, gate_weight, gate_bias,
            hidden, attn_input, attn_out, x_hc_mid, ffn_input,
            ffn_normed, route_indices, route_weights, ffn_post_mix, ffn_residual_mix, next_pre_mix,
            output_window, output_arrived, group_base, tp_rank, num_tokens, attention_epoch,
            mode=mode, has_engram=has_engram)
        # Stage the FFN rows into the fixed MoE tile.  Pad rows route to expert 0
        # with zero weight, so they contribute nothing to the combine.
        for _pad_block in pl.spmd(1, name_hint="layer_pad_moe"):
            for row in pl.range(MOE_TILE):
                if row < num_tokens:
                    pl.store(pl.tile.load(ffn_normed, [row, 0], [1, D]), [row, 0], moe_x)
                    for slot in pl.range(TOPK):
                        pl.write(moe_indices, [row, slot], pl.read(route_indices, [row, slot]))
                        pl.write(moe_weights, [row, slot], pl.read(route_weights, [row, slot]))
                else:
                    pl.store(pl.tile.full([1, D], dtype=pl.BF16, value=0.0), [row, 0], moe_x)
                    for slot in pl.range(TOPK):
                        pl.write(moe_indices, [row, slot], pl.cast(0, target_type=pl.INT32))
                        pl.write(moe_weights, [row, slot], pl.cast(0.0, target_type=pl.FP32))

    @pl.jit
    def layer_pad_moe_rank(
        ffn_normed: pl.Tensor[[T_DYN, D], pl.BF16],
        route_indices: pl.Tensor[[T_DYN, TOPK], pl.INT32],
        route_weights: pl.Tensor[[T_DYN, TOPK], pl.FP32],
        moe_x: pl.Out[pl.Tensor[[MOE_TILE, D], pl.BF16]],
        moe_indices: pl.Out[pl.Tensor[[MOE_TILE, TOPK], pl.INT32]],
        moe_weights: pl.Out[pl.Tensor[[MOE_TILE, TOPK], pl.FP32]],
        token_counts: pl.Tensor[[EP_SIZE], pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ):
        """Copy the active rows into the padded MoE tile; pad rows route with weight zero."""
        ffn_normed.bind_dynamic(0, T_DYN)
        route_indices.bind_dynamic(0, T_DYN)
        route_weights.bind_dynamic(0, T_DYN)
        tokens = pl.read(token_counts, [my_rank])
        for _pad_block in pl.spmd(1, name_hint="layer_pad_moe"):
            for row in pl.range(MOE_TILE):
                if row < tokens:
                    pl.store(pl.tile.load(ffn_normed, [row, 0], [1, D]), [row, 0], moe_x)
                    for slot in pl.range(TOPK):
                        pl.write(moe_indices, [row, slot], pl.read(route_indices, [row, slot]))
                        pl.write(moe_weights, [row, slot], pl.read(route_weights, [row, slot]))
                else:
                    pl.store(pl.tile.full([1, D], dtype=pl.BF16, value=0.0), [row, 0], moe_x)
                    for slot in pl.range(TOPK):
                        pl.write(moe_indices, [row, slot], pl.cast(0, target_type=pl.INT32))
                        pl.write(moe_weights, [row, slot], pl.cast(0.0, target_type=pl.FP32))

    @pl.jit
    def layer_moe_tile_rank(
        x: pl.Tensor[[MOE_TILE, D], pl.BF16],
        indices: pl.Tensor[[MOE_TILE, TOPK], pl.INT32],
        weights: pl.Tensor[[MOE_TILE, TOPK], pl.FP32],
        w1: pl.Tensor[[N_LOCAL_EXPERTS, D, INTER], pl.INT8, pl.NZ],
        w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
        w3: pl.Tensor[[N_LOCAL_EXPERTS, D, INTER], pl.INT8, pl.NZ],
        w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
        w2: pl.Tensor[[N_LOCAL_EXPERTS, INTER, D], pl.INT8, pl.NZ],
        w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
        shared_w1: pl.Tensor[[D, INTER], pl.INT8, pl.NZ],
        shared_w1_scale: pl.Tensor[[INTER], pl.FP32],
        shared_w3: pl.Tensor[[D, INTER], pl.INT8, pl.NZ],
        shared_w3_scale: pl.Tensor[[INTER], pl.FP32],
        shared_w2: pl.Tensor[[INTER, D], pl.INT8, pl.NZ],
        shared_w2_scale: pl.Tensor[[D], pl.FP32],
        recv_x: pld.DistributedTensor[[MOE_RECV_ROWS, D], pl.BF16],
        recv_aux: pld.DistributedTensor[[MOE_RECV_ROWS, 8], pl.FP32],
        recv_meta: pld.DistributedTensor[[EP_SIZE, COUNT_PAD], pl.INT32],
        meta_ready: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
        data_ready: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
        return_slots: pld.DistributedTensor[[MOE_SLOT_ROWS, D], pl.BF16],
        return_ready: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
        consumed: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
        return_consumed: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
        routed: pl.Out[pl.Tensor[[MOE_TILE, D], pl.BF16]],
        my_rank: pl.Scalar[pl.INT32],
        epoch: pl.Scalar[pl.INT32],
    ):
        """Run one EP16 exchange on the padded tile (phase B)."""
        _moe_value, _moe_done = moe_ep(
            x, indices, weights,
            w1, w1_scale, w3, w3_scale, w2, w2_scale,
            shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, shared_w2, shared_w2_scale,
            recv_x, recv_aux, recv_meta, meta_ready, data_ready, return_slots, return_ready,
            consumed, return_consumed,
            routed, my_rank, epoch, capacity=MOE_TILE, rows=MOE_TILE,
        )

    @pl.jit
    def layer_unpad_moe_rank(
        routed: pl.Tensor[[MOE_TILE, D], pl.BF16],
        ffn_owned: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
        token_counts: pl.Tensor[[EP_SIZE], pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ):
        """Keep the active routed rows (drop the zero-weight padding)."""
        ffn_owned.bind_dynamic(0, T_DYN)
        tokens = pl.read(token_counts, [my_rank])
        for row in pl.spmd(tokens, name_hint="layer_unpad_moe"):
            pl.store(pl.tile.load(routed, [row, 0], [1, D]), [row, 0], ffn_owned)

    @pl.jit
    def layer_epilogue_rank(
        x_hc_mid: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
        ffn_owned: pl.Tensor[[T_DYN, D], pl.BF16],
        ffn_post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
        ffn_residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
        x_hc_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
        token_counts: pl.Tensor[[EP_SIZE], pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
    ):
        """Mix the routed FFN result into the residual streams (phase C)."""
        x_hc_mid.bind_dynamic(0, T_DYN)
        ffn_owned.bind_dynamic(0, T_DYN)
        ffn_post_mix.bind_dynamic(0, T_DYN)
        ffn_residual_mix.bind_dynamic(0, T_DYN)
        x_hc_out.bind_dynamic(0, T_DYN)
        hc_post(ffn_owned, x_hc_mid, ffn_post_mix, ffn_residual_mix, x_hc_out)

    @pl.jit.host
    def decode_layer_entry(
        x_hc: pl.Tensor[[EP_SIZE, T_DYN, HC_MULT, D], pl.FP32],
        pre_mix: pl.Tensor[[EP_SIZE, T_DYN, HC_MULT], pl.FP32],
        hc_attn_fn: pl.Tensor[[EP_SIZE, MIX_PAD, HC_DIM], pl.FP32],
        hc_attn_scale: pl.Tensor[[EP_SIZE, 3], pl.FP32],
        hc_attn_base: pl.Tensor[[EP_SIZE, MIX_HC], pl.FP32],
        attn_norm_w: pl.Tensor[[EP_SIZE, D], pl.BF16],
        wq_a: pl.Tensor[[EP_SIZE, D, Q_LORA], pl.INT8],
        wq_a_scale: pl.Tensor[[EP_SIZE, Q_LORA], pl.FP32],
        q_norm_w: pl.Tensor[[EP_SIZE, Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[EP_SIZE, Q_LORA, LOCAL_Q], pl.INT8],
        wq_b_scale: pl.Tensor[[EP_SIZE, LOCAL_Q], pl.FP32],
        wkv: pl.Tensor[[EP_SIZE, D, HEAD_DIM], pl.INT8],
        wkv_scale: pl.Tensor[[EP_SIZE, HEAD_DIM], pl.FP32],
        kv_norm_w: pl.Tensor[[EP_SIZE, HEAD_DIM], pl.BF16],
        sink: pl.Tensor[[EP_SIZE, LOCAL_HEADS], pl.FP32],
        wo_a: pl.Tensor[[EP_SIZE, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[EP_SIZE, D, O_GROUPS * O_LORA], pl.BF16],
        rope_cos: pl.Tensor[[EP_SIZE, T_DYN, 32], pl.FP32],
        rope_sin: pl.Tensor[[EP_SIZE, T_DYN, 32], pl.FP32],
        window_slots: pl.Tensor[[EP_SIZE, T_DYN], pl.INT64],
        window_indices: pl.Tensor[[EP_SIZE, T_DYN, WINDOW_KEYS], pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[EP_SIZE, ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
        compressed_cache: pl.InOut[pl.Tensor[[EP_SIZE, CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
        request_ids: pl.Tensor[[EP_SIZE, T_DYN], pl.INT32],
        compressed_lens: pl.Tensor[[EP_SIZE, T_DYN], pl.INT32],
        index_cache: pl.InOut[pl.Tensor[[EP_SIZE, INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_DIM], pl.BF16]],
        index_block_table: pl.Tensor[[EP_SIZE, B_DYN, TABLE_DYN], pl.INT32],
        compressed_cos: pl.Tensor[[EP_SIZE, T_DYN, 32], pl.FP32],
        compressed_sin: pl.Tensor[[EP_SIZE, T_DYN, 32], pl.FP32],
        compressed_slots: pl.Tensor[[EP_SIZE, T_DYN], pl.INT64],
        position_ids: pl.Tensor[[EP_SIZE, T_DYN], pl.INT32],
        state_block_table: pl.Tensor[[EP_SIZE, B_DYN, 1], pl.INT32],
        state_cache: pl.InOut[pl.Tensor[[EP_SIZE, STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
        compressor_wkv: pl.Tensor[[EP_SIZE, HEAD_DIM, D], pl.BF16],
        compressor_wgate: pl.Tensor[[EP_SIZE, HEAD_DIM, D], pl.BF16],
        compressor_norm: pl.Tensor[[EP_SIZE, HEAD_DIM], pl.BF16],
        index_wq_b: pl.Tensor[[EP_SIZE, Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8],
        index_wq_b_scale: pl.Tensor[[EP_SIZE, IDX_HEADS * IDX_DIM], pl.FP32],
        index_weights_proj: pl.Tensor[[EP_SIZE, D, IDX_HEADS], pl.BF16],
        index_wk: pl.Tensor[[EP_SIZE, IDX_DIM, HEAD_DIM], pl.BF16],
        index_norm_w: pl.Tensor[[EP_SIZE, IDX_DIM], pl.BF16],
        topk_indices: pl.InOut[pl.Tensor[[EP_SIZE, T_DYN, INDEX_TOPK], pl.INT32]],
        topk_values: pl.InOut[pl.Tensor[[EP_SIZE, T_DYN, INDEX_TOPK], pl.FP32]],
        pool_block_ids: pl.InOut[pl.Tensor[[EP_SIZE, T_DYN, POOL_BLOCKS_DYN], pl.INT32]],
        pool_block_counts: pl.InOut[pl.Tensor[[EP_SIZE, T_DYN], pl.INT32]],
        engram_lookup: pl.Tensor[[EP_SIZE, T_DYN, ENGRAM_K], pl.BF16],
        engram_wkv: pl.Tensor[[EP_SIZE, ENGRAM_OUT, ENGRAM_K], pl.BF16],
        engram_qk: pl.Tensor[[EP_SIZE, 4, D], pl.FP32],
        engram_rotation: pl.Tensor[[EP_SIZE, 32, 32], pl.FP32],
        hc_ffn_fn: pl.Tensor[[EP_SIZE, MIX_PAD, HC_DIM], pl.FP32],
        hc_ffn_scale: pl.Tensor[[EP_SIZE, 3], pl.FP32],
        hc_ffn_base: pl.Tensor[[EP_SIZE, MIX_HC], pl.FP32],
        ffn_norm_w: pl.Tensor[[EP_SIZE, D], pl.BF16],
        gate_weight: pl.Tensor[[EP_SIZE, N_EXPERTS, D], pl.FP32],
        gate_bias: pl.Tensor[[EP_SIZE, N_EXPERTS], pl.FP32],
        w1: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, D, INTER], pl.INT8],
        w1_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, INTER], pl.FP32],
        w3: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, D, INTER], pl.INT8],
        w3_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, INTER], pl.FP32],
        w2: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, INTER, D], pl.INT8],
        w2_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, D], pl.FP32],
        shared_w1: pl.Tensor[[EP_SIZE, D, INTER], pl.INT8],
        shared_w1_scale: pl.Tensor[[EP_SIZE, INTER], pl.FP32],
        shared_w3: pl.Tensor[[EP_SIZE, D, INTER], pl.INT8],
        shared_w3_scale: pl.Tensor[[EP_SIZE, INTER], pl.FP32],
        shared_w2: pl.Tensor[[EP_SIZE, INTER, D], pl.INT8],
        shared_w2_scale: pl.Tensor[[EP_SIZE, D], pl.FP32],
        hidden: pl.Out[pl.Tensor[[EP_SIZE, T_DYN, D], pl.BF16]],
        attn_input: pl.Out[pl.Tensor[[EP_SIZE, T_DYN, D], pl.BF16]],
        attn_out: pl.Out[pl.Tensor[[EP_SIZE, T_DYN, D], pl.BF16]],
        x_hc_mid: pl.Out[pl.Tensor[[EP_SIZE, T_DYN, HC_MULT, D], pl.FP32]],
        ffn_input: pl.Out[pl.Tensor[[EP_SIZE, T_DYN, D], pl.BF16]],
        ffn_normed: pl.Out[pl.Tensor[[EP_SIZE, T_DYN, D], pl.BF16]],
        route_indices: pl.Out[pl.Tensor[[EP_SIZE, T_DYN, TOPK], pl.INT32]],
        route_weights: pl.Out[pl.Tensor[[EP_SIZE, T_DYN, TOPK], pl.FP32]],
        next_pre_mix: pl.Out[pl.Tensor[[EP_SIZE, T_DYN, HC_MULT], pl.FP32]],
        ffn_post_mix: pl.Out[pl.Tensor[[EP_SIZE, T_DYN, HC_MULT], pl.FP32]],
        ffn_residual_mix: pl.Out[pl.Tensor[[EP_SIZE, T_DYN, HC_MULT, HC_MULT], pl.FP32]],
        moe_x: pl.Out[pl.Tensor[[EP_SIZE, MOE_TILE, D], pl.BF16]],
        moe_indices: pl.Out[pl.Tensor[[EP_SIZE, MOE_TILE, TOPK], pl.INT32]],
        moe_weights: pl.Out[pl.Tensor[[EP_SIZE, MOE_TILE, TOPK], pl.FP32]],
        token_counts: pl.Tensor[[EP_SIZE], pl.INT32],
        epoch: pl.Scalar[pl.INT32],
    ):
        """Run the mode chain, FFN prolog and MoE staging for every rank (phase A)."""
        x_hc.bind_dynamic(1, T_DYN)
        pre_mix.bind_dynamic(1, T_DYN)
        rope_cos.bind_dynamic(1, T_DYN)
        rope_sin.bind_dynamic(1, T_DYN)
        window_slots.bind_dynamic(1, T_DYN)
        window_indices.bind_dynamic(1, T_DYN)
        window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
        compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
        request_ids.bind_dynamic(1, T_DYN)
        compressed_lens.bind_dynamic(1, T_DYN)
        index_cache.bind_dynamic(1, INDEX_BLOCKS_DYN)
        index_block_table.bind_dynamic(1, B_DYN)
        index_block_table.bind_dynamic(2, TABLE_DYN)
        compressed_cos.bind_dynamic(1, T_DYN)
        compressed_sin.bind_dynamic(1, T_DYN)
        compressed_slots.bind_dynamic(1, T_DYN)
        position_ids.bind_dynamic(1, T_DYN)
        state_block_table.bind_dynamic(1, B_DYN)
        state_cache.bind_dynamic(1, STATE_BLOCKS_DYN)
        topk_indices.bind_dynamic(1, T_DYN)
        topk_values.bind_dynamic(1, T_DYN)
        pool_block_ids.bind_dynamic(1, T_DYN)
        pool_block_ids.bind_dynamic(2, POOL_BLOCKS_DYN)
        pool_block_counts.bind_dynamic(1, T_DYN)
        engram_lookup.bind_dynamic(1, T_DYN)
        hidden.bind_dynamic(1, T_DYN)
        attn_input.bind_dynamic(1, T_DYN)
        attn_out.bind_dynamic(1, T_DYN)
        x_hc_mid.bind_dynamic(1, T_DYN)
        ffn_input.bind_dynamic(1, T_DYN)
        ffn_normed.bind_dynamic(1, T_DYN)
        route_indices.bind_dynamic(1, T_DYN)
        route_weights.bind_dynamic(1, T_DYN)
        next_pre_mix.bind_dynamic(1, T_DYN)
        ffn_post_mix.bind_dynamic(1, T_DYN)
        ffn_residual_mix.bind_dynamic(1, T_DYN)
        output_buffer = pld.alloc_window_buffer([DECODE_MAX_TOKENS, D], dtype=pl.BF16)
        output_signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            output_window = pld.window(output_buffer, [DECODE_MAX_TOKENS, D], dtype=pl.BF16)
            output_arrived = pld.window(output_signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
            decode_layer_rank(
                x_hc[rank], pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], attn_norm_w[rank], wq_a[rank], wq_a_scale[rank], q_norm_w[rank], wq_b[rank], wq_b_scale[rank], wkv[rank], wkv_scale[rank], kv_norm_w[rank], sink[rank], wo_a[rank], wo_b[rank], rope_cos[rank], rope_sin[rank], window_slots[rank], window_indices[rank], window_cache[rank], compressed_cache[rank], request_ids[rank], compressed_lens[rank], index_cache[rank], index_block_table[rank], compressed_cos[rank], compressed_sin[rank], compressed_slots[rank], position_ids[rank], state_block_table[rank], state_cache[rank], compressor_wkv[rank], compressor_wgate[rank], compressor_norm[rank], index_wq_b[rank], index_wq_b_scale[rank], index_weights_proj[rank], index_wk[rank], index_norm_w[rank], topk_indices[rank], topk_values[rank], pool_block_ids[rank], pool_block_counts[rank], engram_lookup[rank], engram_wkv[rank], engram_qk[rank], engram_rotation[rank], hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank], ffn_norm_w[rank], gate_weight[rank], gate_bias[rank], hidden[rank], attn_input[rank], attn_out[rank], x_hc_mid[rank], ffn_input[rank], ffn_normed[rank], route_indices[rank], route_weights[rank], ffn_post_mix[rank], ffn_residual_mix[rank], next_pre_mix[rank], moe_x[rank], moe_indices[rank], moe_weights[rank],
                output_window, output_arrived,
                rank, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE, token_counts, epoch,
                device=rank)

    @pl.jit.host
    def decode_layer_moe_entry(
        moe_x: pl.Tensor[[EP_SIZE, MOE_TILE, D], pl.BF16],
        moe_indices: pl.Tensor[[EP_SIZE, MOE_TILE, TOPK], pl.INT32],
        moe_weights: pl.Tensor[[EP_SIZE, MOE_TILE, TOPK], pl.FP32],
        w1: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, D, INTER], pl.INT8],
        w1_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, INTER], pl.FP32],
        w3: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, D, INTER], pl.INT8],
        w3_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, INTER], pl.FP32],
        w2: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, INTER, D], pl.INT8],
        w2_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, D], pl.FP32],
        shared_w1: pl.Tensor[[EP_SIZE, D, INTER], pl.INT8],
        shared_w1_scale: pl.Tensor[[EP_SIZE, INTER], pl.FP32],
        shared_w3: pl.Tensor[[EP_SIZE, D, INTER], pl.INT8],
        shared_w3_scale: pl.Tensor[[EP_SIZE, INTER], pl.FP32],
        shared_w2: pl.Tensor[[EP_SIZE, INTER, D], pl.INT8],
        shared_w2_scale: pl.Tensor[[EP_SIZE, D], pl.FP32],
        routed_pad: pl.Out[pl.Tensor[[EP_SIZE, MOE_TILE, D], pl.BF16]],
        epoch: pl.Scalar[pl.INT32],
    ):
        """Run the EP16 exchange on the staged tile for every rank (phase B)."""
        recv_x_buffer = pld.alloc_window_buffer([MOE_RECV_ROWS, D], dtype=pl.BF16)
        recv_aux_buffer = pld.alloc_window_buffer([MOE_RECV_ROWS, 8], dtype=pl.FP32)
        recv_meta_buffer = pld.alloc_window_buffer([EP_SIZE, COUNT_PAD], dtype=pl.INT32)
        meta_ready_buffer = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
        data_ready_buffer = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
        return_buffer = pld.alloc_window_buffer([MOE_SLOT_ROWS, D], dtype=pl.BF16)
        return_ready_buffer = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
        consumed_buffer = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
        return_consumed_buffer = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            recv_x = pld.window(recv_x_buffer, [MOE_RECV_ROWS, D], dtype=pl.BF16)
            recv_aux = pld.window(recv_aux_buffer, [MOE_RECV_ROWS, 8], dtype=pl.FP32)
            recv_meta = pld.window(recv_meta_buffer, [EP_SIZE, COUNT_PAD], dtype=pl.INT32)
            meta_ready = pld.window(meta_ready_buffer, [EP_SIZE, 1], dtype=pl.INT32)
            data_ready = pld.window(data_ready_buffer, [EP_SIZE, 1], dtype=pl.INT32)
            return_slots = pld.window(return_buffer, [MOE_SLOT_ROWS, D], dtype=pl.BF16)
            return_ready = pld.window(return_ready_buffer, [EP_SIZE, 1], dtype=pl.INT32)
            consumed = pld.window(consumed_buffer, [EP_SIZE, 1], dtype=pl.INT32)
            return_consumed = pld.window(return_consumed_buffer, [EP_SIZE, 1], dtype=pl.INT32)
            layer_moe_tile_rank(
                moe_x[rank], moe_indices[rank], moe_weights[rank],
                w1[rank], w1_scale[rank], w3[rank], w3_scale[rank], w2[rank], w2_scale[rank],
                shared_w1[rank], shared_w1_scale[rank], shared_w3[rank], shared_w3_scale[rank],
                shared_w2[rank], shared_w2_scale[rank],
                recv_x, recv_aux, recv_meta, meta_ready, data_ready, return_slots,
                return_ready, consumed, return_consumed,
                routed_pad[rank], rank, epoch, device=rank)

    @pl.jit.host
    def decode_layer_epilogue_entry(
        x_hc_mid: pl.Tensor[[EP_SIZE, T_DYN, HC_MULT, D], pl.FP32],
        ffn_post_mix: pl.Tensor[[EP_SIZE, T_DYN, HC_MULT], pl.FP32],
        ffn_residual_mix: pl.Tensor[[EP_SIZE, T_DYN, HC_MULT, HC_MULT], pl.FP32],
        routed_pad: pl.Tensor[[EP_SIZE, MOE_TILE, D], pl.BF16],
        ffn_owned: pl.Out[pl.Tensor[[EP_SIZE, T_DYN, D], pl.BF16]],
        x_hc_out: pl.Out[pl.Tensor[[EP_SIZE, T_DYN, HC_MULT, D], pl.FP32]],
        token_counts: pl.Tensor[[EP_SIZE], pl.INT32],
    ):
        """Drop the tile padding and mix the routed result into the residuals (phase C)."""
        x_hc_mid.bind_dynamic(1, T_DYN)
        ffn_post_mix.bind_dynamic(1, T_DYN)
        ffn_residual_mix.bind_dynamic(1, T_DYN)
        ffn_owned.bind_dynamic(1, T_DYN)
        x_hc_out.bind_dynamic(1, T_DYN)
        for rank in pl.range(pld.world_size()):
            layer_unpad_moe_rank(routed_pad[rank], ffn_owned[rank], token_counts, rank, device=rank)
        for rank in pl.range(pld.world_size()):
            layer_epilogue_rank(
                x_hc_mid[rank], ffn_owned[rank], ffn_post_mix[rank], ffn_residual_mix[rank],
                x_hc_out[rank], token_counts, rank, device=rank)

    return {
        "prolog": decode_layer_entry,
        "moe": decode_layer_moe_entry,
        "epilogue": decode_layer_epilogue_entry,
    }


# Host program parameters, in each entry's own order.  The three programs
# cannot share one compilation unit: a dynamic token symbol in the same program
# as the per-token EP16 exchange is not provably stable (see
# KNOWN_PYPTO_ISSUES.md), so the exchange is compiled on its own static tile.
PROLOG_NAMES = (
    "x_hc", "pre_mix", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_a_scale", "q_norm_w", "wq_b", "wq_b_scale", "wkv", "wkv_scale",
    "kv_norm_w", "sink", "wo_a", "wo_b", "rope_cos", "rope_sin",
    "window_slots", "window_indices", "window_cache", "compressed_cache",
    "request_ids", "compressed_lens", "index_cache", "index_block_table",
    "compressed_cos", "compressed_sin", "compressed_slots", "position_ids",
    "state_block_table", "state_cache", "compressor_wkv", "compressor_wgate",
    "compressor_norm", "index_wq_b", "index_wq_b_scale", "index_weights_proj",
    "index_wk", "index_norm_w", "topk_indices", "topk_values", "pool_block_ids",
    "pool_block_counts", "engram_lookup", "engram_wkv", "engram_qk", "engram_rotation", "hc_ffn_fn",
    "hc_ffn_scale", "hc_ffn_base", "ffn_norm_w", "gate_weight", "gate_bias",
    "w1", "w1_scale", "w3", "w3_scale", "w2", "w2_scale",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
    "shared_w2", "shared_w2_scale", "hidden", "attn_input", "attn_out",
    "x_hc_mid", "ffn_input", "ffn_normed", "route_indices", "route_weights",
    "next_pre_mix", "ffn_post_mix", "ffn_residual_mix",
    "moe_x", "moe_indices", "moe_weights", "token_counts",
)
MOE_NAMES = (
    "moe_x", "moe_indices", "moe_weights",
    "w1", "w1_scale", "w3", "w3_scale", "w2", "w2_scale",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
    "shared_w2", "shared_w2_scale", "routed_pad",
)
EPILOGUE_NAMES = (
    "x_hc_mid", "ffn_post_mix", "ffn_residual_mix", "routed_pad",
    "ffn_owned", "x_hc_out", "token_counts",
)


def _run_staged_layer(entries, specs, mode, has_engram, golden_fn, compare, config, compile_only=False):
    """Run the three host programs through the shared staged runner."""
    return run_staged_programs(
        entries,
        (
            ("prolog", PROLOG_NAMES, True),
            ("moe", MOE_NAMES, True),
            ("epilogue", EPILOGUE_NAMES, False),
        ),
        specs, golden_fn, compare, config,
        extra_inputs={"mode": mode, "has_engram": 1 if has_engram else 0},
        compile_only=compile_only,
    )


CASES = (
    ("swa_layer0", MODE_SWA, False),
    ("swa_layer1_engram", MODE_SWA, True),
    ("c2a_full_layer14_engram", MODE_C2A_FULL, True),
    ("c2a_full_layer2", MODE_C2A_FULL, False),
    ("c2a_reuse_layer3", MODE_C2A_REUSE, False),
    ("c1a_full_layer20", MODE_C1A_FULL, False),
    ("c1a_reuse_layer21", MODE_C1A_REUSE, False),
    ("c1a_reindex_layer24", MODE_C1A_REINDEX, False),
)


if __name__ == "__main__":
    import argparse

    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser()
    from config import add_parallel_arguments
    add_parallel_arguments(parser)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(EP_SIZE)))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--case", default="all",
        help="run one bounded case by name (diagnostic); the default runs every case",
    )
    args = parser.parse_args()
    device_ids = [int(value) for value in args.device.split(",") if value != ""]
    if len(device_ids) != EP_SIZE:
        raise SystemExit(f"need exactly {EP_SIZE} device ids for EP{EP_SIZE}, got {device_ids}")
    torch.manual_seed(args.seed)
    compare = build_compare()
    cases = [(name, mode, has_engram) for name, mode, has_engram in CASES]
    if args.case != "all":
        cases = [case for case in cases if case[0] == args.case]
        if not cases:
            raise SystemExit(f"unknown case {args.case!r}")
    failures = []
    for name, mode, has_engram in cases:
        print(f"[LAYER] {name}", flush=True)
        ratio = 2 if mode in (MODE_C2A_FULL, MODE_C2A_REUSE) else 1
        entries = _build_layer_entries(mode, 1 if has_engram else 0)
        try:
            ok, error = _run_staged_layer(
                entries,
                build_layer_specs("decode", mode, has_engram, args.seed, moe_tile=MOE_TILE),
                mode, has_engram, make_layer_golden(mode, has_engram, ratio), compare,
                dict(
                    distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
                    backend_type=args.backend_type,
                    execution_mode=args.execution_mode,
                ),
                compile_only=args.compile_only,
            )
        except (AssertionError, ValueError, RuntimeError) as exc:
            ok, error = False, str(exc)
        if not ok:
            failures.append(f"{name}: {error or 'validation failed'}")
        else:
            print(f"[RUN] PASS {'(compile only)' if args.compile_only else ''}", flush=True)
    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)
