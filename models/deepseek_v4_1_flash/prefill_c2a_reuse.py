# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed-prefill C2A Reuse wired through mHC."""


import pypto.language as pl
import pypto.language.distributed as pld
import functools
import torch

from hc import hc_pre
from hc import hc_mixes
from config import (
    FLASH as M, CMP_BLOCKS_DYN, ORI_BLOCKS_DYN, T_DYN, TP_SIZE, COMPRESSED_CACHE_GROUP, PREFILL_MAX_TOKENS,
    WINDOW_CACHE_GROUP, SP_T_DYN,
)
from prefill_cp_allgather import prefill_sp_input_allgather
from o_proj import (
    prefill_sp_output_reduce_scatter, o_proj, prefill_tp_output_all_reduce, golden_o_proj,
)
from hc import prefill_sp_post, hc_post
from prefill_c2a_full import (
    CMP_PACKED, CMP_SCALES, REUSE_INPUT_NAMES, make_c2a_inputs, golden_reference_c2a,
    reference_attention, _flat_cache_rows, _write_rows, MX_GROUP, CMP_GROUP, ATTEND_TILE,
)
from qkv_proj_rope import q_proj_qr, q_proj_rope, kv_proj_rope, golden_rope_tail
from sparse_attention import (
    PACKED_QUERY_TILE as QUERY_TILE, SELECTED_KEYS as SPARSE_WIDTH, gather_packed_chunk,
    prefill_attend_sparse, prefill_publish_window,
)
from quant import golden_mxfp8_linear, quantize_mxfp8_cache
from rmsnorm import golden_rms_norm, rms_norm_tiled


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
INDEX_TOPK = M.index_topk


@pl.jit.inline(auto_scope=False)
def prefill_c2a_reuse_attention(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
    ],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    partial: pl.Tensor[[T_DYN, D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Return replicated output for standalone C2A attention callers."""
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="c2a_reuse_previous_epoch", allow_early_resolve=False
    ) as cache_ready:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(
                output_arrived, offsets=[peer, 0], expected=(attention_epoch - 1) * 2,
                cmp=pld.WaitCmp.Ge,
            )
    tokens = pl.tensor.dim(x, 0)

    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    query = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    window_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    kv_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    q_proj_qr(x, wq_a, wq_a_scale, q_norm_weight, qr, num_tokens)
    q_proj_rope(qr, wq_b, wq_b_scale, rope_cos, rope_sin, query, num_tokens)
    kv_proj_rope(x, wkv, wkv_scale, kv_norm_weight, rope_cos, rope_sin, kv_projection, window_kv, num_tokens)
    prefill_publish_window(window_kv, window_slots, window_cache, window_cache_scale, num_tokens, cache_ready)

    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    selected = pl.create_tensor([QUERY_TILE, SPARSE_WIDTH, HEAD_DIM], dtype=pl.BF16)
    combined = pl.create_tensor([QUERY_TILE, SPARSE_WIDTH], dtype=pl.INT32)
    chunk_done = cache_ready
    for start in pl.range(0, num_tokens, QUERY_TILE):
        active = pl.min(QUERY_TILE, num_tokens - start)
        gather_tid = gather_packed_chunk(
            window_cache, window_cache_scale, window_indices, compressed_cache,
            compressed_cache_scale, compressed_indices, selected, combined, start, active,
            chunk_done,
        )
        chunk_done = prefill_attend_sparse(
            query, selected, combined, attn_sink, attended, start, active, gather_tid
        )

    o_proj(attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin, partial, num_tokens, None, group32=False)
    return partial


@pl.jit.inline
def prefill_c2a_reuse_sp(
    x_hc: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
    ],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    input_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    attn_input: pl.Tensor[[T_DYN, D], pl.BF16],
    attn_output: pl.Tensor[[SP_T_DYN, D], pl.BF16],
    next_pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
    x_hc_out: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Run one C2A Reuse attention sublayer between mHC pre and post; see ``prefill_c2a_full``."""
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    local_hidden = pl.create_tensor([tokens, D], dtype=pl.BF16)
    hc_mixes(
        x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, next_pre_mix, post_mix, residual_mix,
    )
    hc_collapsed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    hc_pre(x_hc, pre_mix, hc_collapsed)
    hc_norm_tokens = pl.tensor.dim(hc_collapsed, 0)
    for hc_norm_block in pl.spmd((hc_norm_tokens + 7) // 8, name_hint="rms_norm"):
        rms_norm_tiled(hc_collapsed, attn_norm_weight, local_hidden, hc_norm_block, hc_norm_tokens)
    prefill_sp_input_allgather(
        local_hidden, input_window, input_arrived, attn_input,
        group_base, tp_rank, num_tokens, attention_epoch,
    )
    group_tokens = pl.tensor.dim(attn_input, 0)
    partial = pl.create_tensor([group_tokens, D], dtype=pl.FP32)
    if num_tokens > 0:
        prefill_c2a_reuse_attention(
            attn_input, wq_a, wq_a_scale, q_norm_weight, wq_b,
            wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
            wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
            window_slots, window_indices, window_cache, window_cache_scale, compressed_cache,
            compressed_cache_scale, compressed_indices, output_arrived, partial, num_tokens,
            attention_epoch,
        )
    prefill_sp_output_reduce_scatter(
        partial, output_window, output_arrived, attn_output,
        group_base, tp_rank, num_tokens, attention_epoch,
    )
    prefill_sp_post(
        attn_output, x_hc, post_mix, residual_mix, x_hc_out, next_pre_mix, tp_rank, num_tokens,
    )
    return x_hc_out


@pl.jit.inline
def prefill_c2a_reuse(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
    ],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    attn_input: pl.Tensor[[T_DYN, D], pl.BF16],
    attn_output: pl.Tensor[[T_DYN, D], pl.BF16],
    next_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    x_hc_out: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Run one C2A Reuse attention sublayer between mHC pre and post; see ``prefill_c2a_full``."""
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    hc_mixes(
        x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, next_pre_mix, post_mix, residual_mix,
    )
    hc_collapsed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    hc_pre(x_hc, pre_mix, hc_collapsed)
    hc_norm_tokens = pl.tensor.dim(hc_collapsed, 0)
    for hc_norm_block in pl.spmd((hc_norm_tokens + 7) // 8, name_hint="rms_norm"):
        rms_norm_tiled(hc_collapsed, attn_norm_weight, attn_input, hc_norm_block, hc_norm_tokens)
    attention_tokens = pl.tensor.dim(attn_input, 0)
    attention_partial = pl.create_tensor([attention_tokens, D], dtype=pl.FP32)
    prefill_c2a_reuse_attention(
        attn_input, wq_a, wq_a_scale, q_norm_weight, wq_b,
        wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        window_slots, window_indices, window_cache, window_cache_scale, compressed_cache,
        compressed_cache_scale, compressed_indices, output_arrived, attention_partial, num_tokens,
        attention_epoch,
    )
    prefill_tp_output_all_reduce(
        attention_partial, output_window, output_arrived, attn_output, group_base, tp_rank, num_tokens, attention_epoch,
    )
    hc_post(attn_output, x_hc, post_mix, residual_mix, x_hc_out)
    return x_hc_out


RANKS_DYN = pl.dynamic("RANKS_DYN")


@pl.jit
def l2_prefill_c2a_reuse(
    x_hc: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8],
    compressed_cache_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_SCALES], pl.BF16],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    attn_input: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    attn_output: pl.Out[pl.Tensor[[SP_T_DYN, D], pl.BF16]],
    next_pre_mix: pl.Out[pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32]],
    x_hc_out: pl.Out[pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32]],
    input_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Tensor[[1], pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Bind the runtime shapes and run the sublayer once on one rank."""
    x_hc.bind_dynamic(0, SP_T_DYN)
    attn_input.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    active = pl.read(num_tokens, [0])
    prefill_c2a_reuse_sp(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight,
        wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
        kv_norm_weight, attn_sink, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        window_slots, window_indices, window_cache, window_cache_scale,
        compressed_cache, compressed_cache_scale, compressed_indices, input_window, input_arrived, output_window,
        output_arrived, attn_input, attn_output, next_pre_mix, x_hc_out,
        rank // TP_SIZE * TP_SIZE, rank % TP_SIZE, active, attention_epoch,
    )
    return x_hc_out, next_pre_mix, attn_output, attn_input, window_cache, window_cache_scale


@pl.jit.host
def l3_prefill_c2a_reuse(
    x_hc: pl.Tensor[[RANKS_DYN, SP_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[RANKS_DYN, SP_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[RANKS_DYN, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[RANKS_DYN, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[RANKS_DYN, MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[RANKS_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[RANKS_DYN, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[RANKS_DYN, D // 32, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[RANKS_DYN, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[RANKS_DYN, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[RANKS_DYN, Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[RANKS_DYN, D // 32, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[RANKS_DYN, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[RANKS_DYN, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[RANKS_DYN, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[RANKS_DYN, LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[RANKS_DYN, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[RANKS_DYN, LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
    rope_cos: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[RANKS_DYN, T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache: pl.Tensor[[RANKS_DYN, CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8],
    compressed_cache_scale: pl.Tensor[[RANKS_DYN, CMP_BLOCKS_DYN, 128, 1, CMP_SCALES], pl.BF16],
    compressed_indices: pl.Tensor[[RANKS_DYN, T_DYN, INDEX_TOPK], pl.INT32],
    attn_input: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, D], pl.BF16]],
    attn_output: pl.Out[pl.Tensor[[RANKS_DYN, SP_T_DYN, D], pl.BF16]],
    next_pre_mix: pl.Out[pl.Tensor[[RANKS_DYN, SP_T_DYN, HC_MULT], pl.FP32]],
    x_hc_out: pl.Out[pl.Tensor[[RANKS_DYN, SP_T_DYN, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Tensor[[RANKS_DYN, 1], pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Allocate the TP communication windows and launch one rank entry per device."""
    x_hc.bind_dynamic(0, RANKS_DYN)
    x_hc.bind_dynamic(1, SP_T_DYN)
    attn_input.bind_dynamic(1, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
    input_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    input_signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    data_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.FP32)
    signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        input_data = pld.window(input_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        input_signal = pld.window(input_signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        data = pld.window(data_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.FP32)
        signal = pld.window(signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        # Each rank consumes packed MX_B_NN scale rows.
        wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
        wq_b_scale_r: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[rank]
        wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
        wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
        l2_prefill_c2a_reuse(
            x_hc[rank], pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_weight[rank], wq_a[rank], wq_a_scale_r, q_norm_weight[rank], wq_b[rank],
            wq_b_scale_r, wkv[rank], wkv_scale_r, kv_norm_weight[rank],
            attn_sink[rank], wo_a[rank], wo_a_scale[rank], wo_b[rank], wo_b_scale_r, rope_cos[rank],
            rope_sin[rank], window_slots[rank], window_indices[rank], window_cache[rank],
            window_cache_scale[rank], compressed_cache[rank], compressed_cache_scale[rank],
            compressed_indices[rank], attn_input[rank], attn_output[rank], next_pre_mix[rank],
            x_hc_out[rank], input_data, input_signal, data, signal, rank, num_tokens[rank], attention_epoch, device=rank,
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
    local_heads = t["attn_sink"].numel()

    qr = golden_rms_norm(golden_mxfp8_linear(x, t["wq_a"], t["wq_a_scale"]), t["q_norm_weight"])
    q = golden_mxfp8_linear(qr, t["wq_b"], t["wq_b_scale"]).unflatten(-1, (local_heads, head_dim))
    q = golden_rope_tail(q, t["rope_cos"], t["rope_sin"])
    kv = golden_rms_norm(golden_mxfp8_linear(x, t["wkv"], t["wkv_scale"]), t["kv_norm_weight"])
    kv = golden_rope_tail(kv, t["rope_cos"], t["rope_sin"])

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
    compressed_values = _flat_cache_rows(
        t["compressed_cache"], t["compressed_cache_scale"], CMP_GROUP, "bf16"
    )
    window_indices = t["window_indices"].long()
    compressed_indices = t["compressed_indices"].long()
    gathered = torch.cat(
        (
            window_values[window_indices.clamp_min(0).reshape(-1)].reshape(tokens, -1, head_dim),
            compressed_values[compressed_indices.clamp_min(0).reshape(-1)].reshape(
                tokens, -1, head_dim
            ),
        ),
        dim=1,
    ).to(torch.bfloat16)
    visible = torch.cat((window_indices >= 0, compressed_indices >= 0), dim=1)
    gathered = gathered.masked_fill(~visible[..., None], 0)

    maximum = torch.full((tokens, local_heads), -1e30)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros(tokens, local_heads, head_dim, dtype=torch.float32)
    for start in range(0, gathered.shape[1], ATTEND_TILE):
        values = gathered[:, start : start + ATTEND_TILE].float()
        logits = torch.einsum("thd,tkd->thk", q.float(), values) * head_dim**-0.5
        logits = logits.masked_fill(~visible[:, None, start : start + ATTEND_TILE], -torch.inf)
        new_maximum = torch.maximum(maximum, logits.amax(-1))
        correction = (maximum - new_maximum).exp()
        probabilities = (logits - new_maximum[..., None]).exp()
        denominator = denominator * correction + probabilities.sum(-1)
        numerator = numerator * correction[..., None] + torch.einsum(
            "thk,tkd->thd", probabilities.to(torch.bfloat16).float(), values
        )
        maximum = new_maximum
    final_maximum = torch.maximum(maximum, t["attn_sink"].float()[None])
    correction = (maximum - final_maximum).exp()
    denominator = denominator * correction + (t["attn_sink"].float()[None] - final_maximum).exp()
    attended = (numerator * (correction / denominator)[..., None]).to(torch.bfloat16)

    output = golden_o_proj(
        attended, t["rope_cos"], t["rope_sin"], t["wo_a"], t["wo_b"], t["wo_b_scale"],
        wo_a_scale=t["wo_a_scale"], output_dtype=torch.float32 if fp32_output else None,
        sequential_projection=True,
    )

    return {
        "output": output,
        "window_cache": window_cache,
        "window_cache_scale": window_cache_scale,
    }


def make_c2a_reuse_inputs(tokens=32, requests=6, seed=17, case="mixed", mode="decode", start_pos=None):
    """Inputs for a reuse layer, taken from what a full layer actually published.

    The compressed pool and the Top-K row list come from running the full-mode
    reference on the same batch, so the fixture exercises the real Full-to-Reuse
    hand-off rather than a synthetic selection.
    """
    source = make_c2a_inputs(tokens=tokens, requests=requests, seed=seed, case=case, mode=mode, start_pos=start_pos)
    published = golden_reference_c2a(source)
    values = {name: source[name] for name in REUSE_INPUT_NAMES if name != "compressed_indices"}
    values["compressed_cache"] = published["compressed_cache"]
    values["compressed_cache_scale"] = published["compressed_cache_scale"]
    values["compressed_indices"] = published["topk_indices"]
    reachable = int((published["topk_indices"] >= 0).sum())
    print(
        f"[FIXTURE] c2a reuse tokens={tokens} requests={requests} mode={mode} case={case} "
        f"selected_rows={reachable} source_published_rows="
        f"{int((source['compressed_slots'] >= 0).sum())}"
    )
    return values


if __name__ == "__main__":
    import argparse
    import os

    import torch

    from config import MAX_BATCH_PER_DP
    from golden import run
    from prefill_c2a_full import PREFILL_ATTN_RING_HEAP, build_specs, make_compare, make_golden
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 prefill C2A reuse with mHC: A5 precision")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim"])
    parser.add_argument("-d", "--device", default=None, help="comma-separated device IDs; default: 0 through TP*DP-1")
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=[1, 2, 4])
    parser.add_argument("--dp", type=int, default=1, choices=[1, 2])
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--dp-tokens", type=str, help="comma-separated active counts, one per DP group")
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--case", default="mixed", choices=["mixed", "long", "masked", "zero"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    args.bench = os.environ.get("PYPTO_BENCH", "0") == "1"
    devices = list(range(TP_SIZE * args.dp))
    if args.device:
        devices = [int(value) for value in args.device.split(",")]
    if len(devices) != TP_SIZE * args.dp or len(set(devices)) != len(devices) or min(devices) < 0:
        parser.error(f"--device must name {TP_SIZE * args.dp} distinct non-negative device IDs")
    if not 1 <= args.tokens <= PREFILL_MAX_TOKENS:
        parser.error(f"--tokens must be in [1, {PREFILL_MAX_TOKENS}]")
    if not 1 <= args.requests <= min(MAX_BATCH_PER_DP, args.tokens):
        parser.error(f"--requests must be in [1, {min(MAX_BATCH_PER_DP, args.tokens)}]")
    args.dp_tokens = [int(value) for value in args.dp_tokens.split(",")] if args.dp_tokens else [args.tokens] * args.dp
    if len(args.dp_tokens) != args.dp or any(count < 0 or count > args.tokens for count in args.dp_tokens):
        parser.error("--dp-tokens must provide one count in [0, tokens] per DP group")
    torch.set_num_threads(8)

    print(
        f"[C2A-HC] mode=reuse tokens={args.tokens} requests={args.requests} TP={TP_SIZE} "
        f"DP={args.dp} case={args.case} seed={args.seed} devices={devices}"
    )
    initial_state = {}
    result = run(
        fn=l3_prefill_c2a_reuse,
        specs=build_specs(args, "reuse", initial_state, make_inputs=make_c2a_reuse_inputs),
        golden_fn=make_golden("reuse", attention_reference=functools.partial(reference_attention, reference=golden_reference_c2a_reuse)),
        compile_only=args.compile_only,
        config=dict(
            platform=args.platform,
            distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
            ring_heap=PREFILL_ATTN_RING_HEAP,
        ),
        compare_fn=make_compare("reuse", len(devices), args.tokens),
    )
    if args.compile_only and result.passed:
        print("[C2A-HC] Compilation passed; device accuracy was NOT validated.")
    if not result.passed:
        raise SystemExit(result.error or 1)
