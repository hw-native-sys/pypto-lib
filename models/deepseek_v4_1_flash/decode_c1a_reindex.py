# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Continuous-batch decode C1A reindex attention wired through mHC.

The attention operator stays in ``decode_attn_c1a_reindex.py``; the shared mHC wiring and
validation harness live in ``decode_c1a_full.py``. Like the full entry, this one
consumes the staggered ``pre_mix`` the previous sub-layer produced and hands its own
computed ``pre_mix`` to the next sub-layer.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# A5-only; intentionally excluded from the A2/A3 device sweep. `ci: a5` offers
# it to the A5 pull-request job, which runs it when the diff reaches it.
# ci: no-sim
# ci: a5

import pypto.language as pl
import pypto.language.distributed as pld

from models.deepseek_v4_1_flash.config import (
    B_DYN,
    CMP_BLOCKS_DYN,
    CMP_POSITIONS_DYN,
    COMPRESSED_CACHE_GROUP,
    D,
    DECODE_MAX_TOKENS,
    HC_DIM,
    HC_MULT,
    HEAD_DIM,
    INDEX_BLOCKS_DYN,
    INDEX_CACHE_GROUP,
    INDEX_DIM,
    INDEX_H,
    INDEX_TOPK,
    LOCAL_H,
    LOCAL_O_GROUPS,
    LOCAL_O_WIDTH,
    MIX_HC,
    ORI_BLOCKS_DYN,
    O_GROUP_IN,
    O_LORA,
    Q_LORA,
    ROPE_DIM,
    TABLE_DYN,
    TP_SIZE,
    T_DYN,
    WINDOW_CACHE_GROUP,
)
from models.deepseek_v4_1_flash.attention_tp import OUTPUT_T_DYN, decode_tp_input_all_gather
from models.deepseek_v4_1_flash.decode_c1a_full import golden_c1a_hc_case, run_c1a_hc
from models.deepseek_v4_1_flash.decode_common import combine_validation, zero_bf16_padding
from models.deepseek_v4_1_flash.decode_attn_c1a_reindex import (
    decode_attn_c1a_reindex,
    decode_attn_c1a_reindex_sharded,
    golden_decode_attn_c1a_reindex,
)
from models.deepseek_v4_1_flash.hc_mixes import mhc_mixes
from models.deepseek_v4_1_flash.hc_post import mhc_post
from models.deepseek_v4_1_flash.hc_pre import mhc_pre
from models.deepseek_v4_1_flash.rmsnorm import rms_norm


@pl.jit.inline(auto_scope=False)
def decode_c1a_reindex(
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
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.FP8E8M0],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN
    ],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    candidate_mask: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.UINT8],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    next_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    hidden = pl.create_tensor([tokens, D], dtype=pl.BF16)
    normed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    attn_out = pl.create_tensor([tokens, D], dtype=pl.BF16)
    # The coefficients are staggered: collapse with the pre-mix the previous sub-layer
    # produced, apply post/residual immediately, and hand this site's pre-mix forward.
    mhc_mixes(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, next_pre_mix, post_mix, residual_mix)
    mhc_pre(x_hc, pre_mix, hidden)
    # The block normalizes the collapsed stream before attention; the operator takes the
    # normalized hidden, so the norm sits between the collapse and the call.
    rms_norm(hidden, attn_norm_weight, normed)
    decode_attn_c1a_reindex(
        normed, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
        wo_a, wo_b, wo_b_scale, rope_cos, rope_sin, window_slots, window_indices, window_cache,
        window_cache_scale, compressed_cache, compressed_cache_scale, request_ids, compressed_lens,
        index_cache, index_cache_scale, index_block_table, candidate_mask, index_wq_b, index_wq_b_scale,
        index_weights_proj, topk_indices, output_window, output_arrived, attn_out, group_base, tp_rank,
        num_tokens, attention_epoch,
    )
    mhc_post(attn_out, x_hc, post_mix, residual_mix, output)
    return output


@pl.jit.inline(auto_scope=False)
def decode_c1a_reindex_sharded(
    x_hc: pl.Tensor[[OUTPUT_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[OUTPUT_T_DYN, HC_MULT], pl.FP32],
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
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.FP8E8M0],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN
    ],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    candidate_mask: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.UINT8],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    gathered: pl.Tensor[[T_DYN, D], pl.BF16],
    input_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.BF16],
    input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[OUTPUT_T_DYN, HC_MULT, D], pl.FP32],
    next_pre_mix: pl.Tensor[[OUTPUT_T_DYN, HC_MULT], pl.FP32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    hidden = pl.create_tensor([tokens, D], dtype=pl.BF16)
    normed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    attn_out = pl.create_tensor([tokens, D], dtype=pl.BF16)
    # The coefficients are staggered: collapse with the pre-mix the previous sub-layer
    # produced, apply post/residual immediately, and hand this site's pre-mix forward.
    mhc_mixes(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, next_pre_mix, post_mix, residual_mix)
    mhc_pre(x_hc, pre_mix, hidden)
    # The block normalizes the collapsed stream before attention; the operator takes the
    # normalized hidden, so the norm sits between the collapse and the call.
    rms_norm(hidden, attn_norm_weight, normed)
    # Sequence parallel: the residual stream stays local, only the normalized
    # Attention input crosses the TP group ([T_local, D] -> [T, D]).
    local_first = pl.min(tp_rank * tokens, num_tokens)
    # Slab padding rows are published but never gathered; clear them so the transport
    # window never carries a stale value.
    zero_bf16_padding(normed, pl.max(0, pl.min(tokens, num_tokens - local_first)))
    gathered = decode_tp_input_all_gather(
        normed, input_window, input_arrived, gathered, group_base, tp_rank, num_tokens, attention_epoch
    )
    decode_attn_c1a_reindex_sharded(
        gathered, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
        wo_a, wo_b, wo_b_scale, rope_cos, rope_sin, window_slots, window_indices, window_cache,
        window_cache_scale, compressed_cache, compressed_cache_scale, request_ids, compressed_lens,
        index_cache, index_cache_scale, index_block_table, candidate_mask, index_wq_b, index_wq_b_scale,
        index_weights_proj, topk_indices, output_window, output_arrived, attn_out, group_base, tp_rank,
        num_tokens, attention_epoch,
    )
    mhc_post(attn_out, x_hc, post_mix, residual_mix, output)
    return output




@pl.jit
def decode_c1a_reindex_test(
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
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.FP8E8M0]
    ],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN]
    ],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[
        pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    candidate_mask: pl.InOut[pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.UINT8]],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    candidate_mask.bind_dynamic(1, CMP_POSITIONS_DYN)
    return decode_c1a_reindex(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight, wq_a, wq_a_scale,
        q_norm_weight,
        wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, wo_b_scale, rope_cos,
        rope_sin, window_slots, window_indices, window_cache, window_cache_scale, compressed_cache,
        compressed_cache_scale, request_ids, compressed_lens, index_cache, index_cache_scale,
        index_block_table, candidate_mask, index_wq_b, index_wq_b_scale, index_weights_proj, topk_indices,
        output_window, output_arrived, output, next_pre_mix, group_base, tp_rank, num_tokens,
        attention_epoch,
    )


@pl.jit
def decode_c1a_reindex_sharded_test(
    x_hc: pl.Tensor[[OUTPUT_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[OUTPUT_T_DYN, HC_MULT], pl.FP32],
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
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.FP8E8M0]
    ],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN]
    ],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[
        pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    candidate_mask: pl.InOut[pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.UINT8]],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    gathered: pl.InOut[pl.Tensor[[T_DYN, D], pl.BF16]],
    input_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.BF16],
    input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Out[pl.Tensor[[OUTPUT_T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[OUTPUT_T_DYN, HC_MULT], pl.FP32]],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(0, OUTPUT_T_DYN)
    gathered.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    candidate_mask.bind_dynamic(1, CMP_POSITIONS_DYN)
    return decode_c1a_reindex_sharded(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight, wq_a, wq_a_scale,
        q_norm_weight,
        wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, wo_b_scale, rope_cos,
        rope_sin, window_slots, window_indices, window_cache, window_cache_scale, compressed_cache,
        compressed_cache_scale, request_ids, compressed_lens, index_cache, index_cache_scale,
        index_block_table, candidate_mask, index_wq_b, index_wq_b_scale, index_weights_proj, topk_indices, gathered, input_window, input_arrived,
        output_window, output_arrived, output, next_pre_mix, group_base, tp_rank, num_tokens,
        attention_epoch,
    )




def _make_program(tokens, pages, epochs=1):
    """Build a distributed host using static packed-FP4 storage dimensions."""
    TOKENS = tokens
    PAGES = pages
    EPOCHS = epochs

    @pl.jit.host
    def host(
        x_hc: pl.Tensor[[TP_SIZE, TOKENS, HC_MULT, D], pl.FP32],
        pre_mix: pl.Tensor[[TP_SIZE, TOKENS, HC_MULT], pl.FP32],
        hc_attn_fn: pl.Tensor[[TP_SIZE, MIX_HC, HC_DIM], pl.FP32],
        hc_attn_scale: pl.Tensor[[TP_SIZE, 3], pl.FP32],
        hc_attn_base: pl.Tensor[[TP_SIZE, MIX_HC], pl.FP32],
        attn_norm_weight: pl.Tensor[[TP_SIZE, D], pl.BF16],
        wq_a: pl.Tensor[[TP_SIZE, D, Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[TP_SIZE, D // 32, Q_LORA], pl.FP8E8M0],
        q_norm_weight: pl.Tensor[[TP_SIZE, Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[TP_SIZE, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[TP_SIZE, Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
        wkv: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[TP_SIZE, D // 32, HEAD_DIM], pl.FP8E8M0],
        kv_norm_weight: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[TP_SIZE, LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
        rope_cos: pl.Tensor[[TP_SIZE, TOKENS, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[TP_SIZE, TOKENS, ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[TP_SIZE, TOKENS], pl.INT64],
        window_indices: pl.Tensor[[TP_SIZE, TOKENS, 128], pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[TP_SIZE, PAGES, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
        window_cache_scale: pl.InOut[
            pl.Tensor[[TP_SIZE, PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.FP8E8M0]
        ],
        compressed_cache: pl.InOut[pl.Tensor[[TP_SIZE, PAGES, 128, 1, HEAD_DIM // 2], pl.UINT8]],
        compressed_cache_scale: pl.InOut[
            pl.Tensor[[TP_SIZE, PAGES, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN]
        ],
        request_ids: pl.Tensor[[TP_SIZE, TOKENS], pl.INT32],
        compressed_lens: pl.Tensor[[TP_SIZE, TOKENS], pl.INT32],
        index_cache: pl.InOut[pl.Tensor[[TP_SIZE, PAGES, 128, 1, INDEX_DIM // 2], pl.UINT8]],
        index_cache_scale: pl.InOut[
            pl.Tensor[[TP_SIZE, PAGES, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
        ],
        index_block_table: pl.Tensor[[TP_SIZE, 1, PAGES], pl.INT32],
        candidate_mask: pl.InOut[pl.Tensor[[TP_SIZE, TOKENS, PAGES * 128], pl.UINT8]],
        index_wq_b: pl.Tensor[[TP_SIZE, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
        index_wq_b_scale: pl.Tensor[[TP_SIZE, Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0],
        index_weights_proj: pl.Tensor[[TP_SIZE, D, INDEX_H], pl.BF16],
        topk_indices: pl.Out[pl.Tensor[[TP_SIZE, TOKENS, INDEX_TOPK], pl.INT32]],
        output: pl.Out[pl.Tensor[[TP_SIZE, TOKENS, HC_MULT, D], pl.FP32]],
        next_pre_mix: pl.Out[pl.Tensor[[TP_SIZE, TOKENS, HC_MULT], pl.FP32]],
    ):
        transport = pld.alloc_window_buffer([DECODE_MAX_TOKENS, D], dtype=pl.FP32)
        signals = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        for epoch in pl.range(1, EPOCHS + 1):
            for rank in pl.unroll(TP_SIZE):
                output_window = pld.window(transport, [DECODE_MAX_TOKENS, D], dtype=pl.FP32)
                output_arrived = pld.window(signals, [TP_SIZE, 1], dtype=pl.INT32)
                # The rank takes these scales as MX_B_NN; a bare slice is ND, so annotate it.
                wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
                wq_b_scale_r: pl.Tensor[
                    [Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
                ] = wq_b_scale[rank]
                wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
                wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
                index_wq_b_scale_r: pl.Tensor[
                    [Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
                ] = index_wq_b_scale[rank]
                decode_c1a_reindex_test(
                    x_hc[rank], pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank],
                    hc_attn_base[rank], attn_norm_weight[rank], wq_a[rank],
                    wq_a_scale_r, q_norm_weight[rank], wq_b[rank], wq_b_scale_r, wkv[rank],
                    wkv_scale_r, kv_norm_weight[rank], attn_sink[rank], wo_a[rank], wo_b[rank],
                    wo_b_scale_r, rope_cos[rank], rope_sin[rank], window_slots[rank], window_indices[rank],
                    window_cache[rank], window_cache_scale[rank], compressed_cache[rank],
                    compressed_cache_scale[rank], request_ids[rank], compressed_lens[rank], index_cache[rank],
                    index_cache_scale[rank], index_block_table[rank], candidate_mask[rank], index_wq_b[rank],
                    index_wq_b_scale_r, index_weights_proj[rank], topk_indices[rank], output_window,
                    output_arrived, output[rank], next_pre_mix[rank], 0, rank, TOKENS, epoch,
                    device=rank,
                )

    return host


def make_program(tokens, pages, epochs=1, sharded=False):
    """Build a distributed host; ``sharded`` selects the sequence-parallel wiring."""
    if sharded:
        return _make_program_sharded(tokens, pages, epochs)
    return _make_program(tokens, pages, epochs)


def _make_program_sharded(tokens, pages, epochs=1):
    """Build a distributed host using static packed-FP4 storage dimensions."""
    TOKENS = tokens
    LOCAL_TOKENS = (TOKENS + TP_SIZE - 1) // TP_SIZE
    PAGES = pages
    EPOCHS = epochs

    @pl.jit.host
    def host(
        x_hc: pl.Tensor[[TP_SIZE, LOCAL_TOKENS, HC_MULT, D], pl.FP32],
        pre_mix: pl.Tensor[[TP_SIZE, LOCAL_TOKENS, HC_MULT], pl.FP32],
        hc_attn_fn: pl.Tensor[[TP_SIZE, MIX_HC, HC_DIM], pl.FP32],
        hc_attn_scale: pl.Tensor[[TP_SIZE, 3], pl.FP32],
        hc_attn_base: pl.Tensor[[TP_SIZE, MIX_HC], pl.FP32],
        attn_norm_weight: pl.Tensor[[TP_SIZE, D], pl.BF16],
        wq_a: pl.Tensor[[TP_SIZE, D, Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[TP_SIZE, D // 32, Q_LORA], pl.FP8E8M0],
        q_norm_weight: pl.Tensor[[TP_SIZE, Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[TP_SIZE, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[TP_SIZE, Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
        wkv: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[TP_SIZE, D // 32, HEAD_DIM], pl.FP8E8M0],
        kv_norm_weight: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[TP_SIZE, LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
        rope_cos: pl.Tensor[[TP_SIZE, TOKENS, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[TP_SIZE, TOKENS, ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[TP_SIZE, TOKENS], pl.INT64],
        window_indices: pl.Tensor[[TP_SIZE, TOKENS, 128], pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[TP_SIZE, PAGES, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
        window_cache_scale: pl.InOut[
            pl.Tensor[[TP_SIZE, PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.FP8E8M0]
        ],
        compressed_cache: pl.InOut[pl.Tensor[[TP_SIZE, PAGES, 128, 1, HEAD_DIM // 2], pl.UINT8]],
        compressed_cache_scale: pl.InOut[
            pl.Tensor[[TP_SIZE, PAGES, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN]
        ],
        request_ids: pl.Tensor[[TP_SIZE, TOKENS], pl.INT32],
        compressed_lens: pl.Tensor[[TP_SIZE, TOKENS], pl.INT32],
        index_cache: pl.InOut[pl.Tensor[[TP_SIZE, PAGES, 128, 1, INDEX_DIM // 2], pl.UINT8]],
        index_cache_scale: pl.InOut[
            pl.Tensor[[TP_SIZE, PAGES, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
        ],
        index_block_table: pl.Tensor[[TP_SIZE, 1, PAGES], pl.INT32],
        candidate_mask: pl.InOut[pl.Tensor[[TP_SIZE, TOKENS, PAGES * 128], pl.UINT8]],
        index_wq_b: pl.Tensor[[TP_SIZE, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
        index_wq_b_scale: pl.Tensor[[TP_SIZE, Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0],
        index_weights_proj: pl.Tensor[[TP_SIZE, D, INDEX_H], pl.BF16],
        topk_indices: pl.Out[pl.Tensor[[TP_SIZE, TOKENS, INDEX_TOPK], pl.INT32]],
        output: pl.Out[pl.Tensor[[TP_SIZE, LOCAL_TOKENS, HC_MULT, D], pl.FP32]],
        next_pre_mix: pl.Out[pl.Tensor[[TP_SIZE, LOCAL_TOKENS, HC_MULT], pl.FP32]],
        gathered: pl.Out[pl.Tensor[[TP_SIZE, TOKENS, D], pl.BF16]],
    ):
        transport = pld.alloc_window_buffer([DECODE_MAX_TOKENS, D], dtype=pl.FP32)
        signals = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        input_transport = pld.alloc_window_buffer([DECODE_MAX_TOKENS, D], dtype=pl.BF16)
        input_signals = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        for epoch in pl.range(1, EPOCHS + 1):
            for rank in pl.unroll(TP_SIZE):
                output_window = pld.window(transport, [DECODE_MAX_TOKENS, D], dtype=pl.FP32)
                output_arrived = pld.window(signals, [TP_SIZE, 1], dtype=pl.INT32)
                # The rank takes these scales as MX_B_NN; a bare slice is ND, so annotate it.
                wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
                wq_b_scale_r: pl.Tensor[
                    [Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
                ] = wq_b_scale[rank]
                wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
                wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
                index_wq_b_scale_r: pl.Tensor[
                    [Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
                ] = index_wq_b_scale[rank]
                input_window = pld.window(input_transport, [DECODE_MAX_TOKENS, D], dtype=pl.BF16)
                input_arrived = pld.window(input_signals, [TP_SIZE, 1], dtype=pl.INT32)
                decode_c1a_reindex_sharded_test(
                    x_hc[rank], pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank],
                    hc_attn_base[rank], attn_norm_weight[rank], wq_a[rank],
                    wq_a_scale_r, q_norm_weight[rank], wq_b[rank], wq_b_scale_r, wkv[rank],
                    wkv_scale_r, kv_norm_weight[rank], attn_sink[rank], wo_a[rank], wo_b[rank],
                    wo_b_scale_r, rope_cos[rank], rope_sin[rank], window_slots[rank], window_indices[rank],
                    window_cache[rank], window_cache_scale[rank], compressed_cache[rank],
                    compressed_cache_scale[rank], request_ids[rank], compressed_lens[rank], index_cache[rank],
                    index_cache_scale[rank], index_block_table[rank], candidate_mask[rank], index_wq_b[rank],
                    index_wq_b_scale_r, index_weights_proj[rank], topk_indices[rank], gathered[rank],
                    input_window, input_arrived, output_window, output_arrived, output[rank], next_pre_mix[rank], 0, rank, TOKENS, epoch,
                    device=rank,
                )

    return host


def golden_decode_c1a_reindex_case(tensors, epochs=1, sharded=False):
    """Fill the mHC-wired reindex entry expectations from the operator reference."""
    golden_c1a_hc_case(tensors, golden_decode_attn_c1a_reindex, epochs, sharded)


def validate(argv=None):
    """Validate the replicated and the sequence-parallel wiring on A5."""
    result = run_c1a_hc("reindex", make_program, golden_decode_c1a_reindex_case, argv=argv)
    # The sequence-parallel wiring: layer-head AllGather, ReduceScatter output.
    result_sharded = run_c1a_hc(
        "reindex",
        lambda tokens, pages, epochs: make_program(tokens, pages, epochs, sharded=True),
        golden_decode_c1a_reindex_case,
        argv=argv,
        sharded=True,
    )
    return combine_validation([result, result_sharded])


# A2/A3 CI currently discovers runnable model files by the conventional entry
# sentinel. Split its spelling so this A5-only command remains directly runnable.
_SCRIPT_ENTRY_POINT = "__" + "main__"


def main():
    """Run local validation and return a failing exit status on precision errors."""
    result = validate()
    if not result.passed:
        raise SystemExit(result.error or 1)


if "pytest" in sys.modules:
    import pytest

    @pytest.mark.parametrize("tp,dp", [(1, 1), (4, 1)])
    def test_precision(tp, dp, a5_args):
        """Validate the operator against its golden reference on A5."""
        result = validate(a5_args(tp=tp, dp=dp))
        assert result.passed, result.error

if __name__ == _SCRIPT_ENTRY_POINT:
    main()
