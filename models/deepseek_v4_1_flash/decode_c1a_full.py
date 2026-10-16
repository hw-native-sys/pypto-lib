# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Continuous-batch decode C1A full attention wired through mHC."""

import utils as _decode_utils

import inspect

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from golden import ScalarSpec, TensorSpec, ratio_allclose, run
from rope_tables import precompute_rope_tables
from hc_pre import mhc_pre_norm
from decode_cp_allgather import slab_owner
from config import LOCAL_T_DYN
from decode_cp_allgather import decode_tp_input_all_gather
from config import (
    COMPRESSED_CACHE_GROUP,
    DECODE_INPUT_MAX_TOKENS,
    DECODE_MAX_TOKENS,
    INDEX_CACHE_GROUP,
    MAX_BATCH_PER_DP,
    WINDOW_CACHE_GROUP,
    FLASH as M,
    B_DYN,
    CMP_BLOCKS_DYN,
    CMP_POSITIONS_DYN,
    INDEX_BLOCKS_DYN,
    ORI_BLOCKS_DYN,
    TABLE_DYN,
    T_DYN,
    TP_SIZE,
)


from decode_attn_c1a import (
    CACHE_STATE_NAMES,
    OUTPUT_ATOL,
    OUTPUT_MAX_ERROR_RATIO,
    OUTPUT_RTOL,
    build_validation_values,
    check_fp4_boundaries,
    decode_attn_c1a_full,
    decode_attn_c1a_full_sharded,
    cache_compare,
    golden_decode_attn_c1a_full,
)
from hc_mixes import golden_mhc_mixes, mhc_mixes
from hc_post import golden_mhc_post, mhc_post
from hc_pre import golden_mhc_pre
from rmsnorm import golden_rms_norm


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


@pl.jit.inline(auto_scope=False)
def decode_c1a_full(
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
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    candidate_mask: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.UINT8],
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
    # The collapse + norm block itself is shared with the spec-driven modes.
    mhc_mixes(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, next_pre_mix, post_mix, residual_mix)
    mhc_pre_norm(x_hc, pre_mix, attn_norm_weight, hidden, normed, num_tokens)
    decode_attn_c1a_full(
        normed, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin, window_slots, window_indices, window_cache,
        window_cache_scale, compressed_cache, compressed_cache_scale, request_ids, compressed_lens,
        index_cache, index_cache_scale, index_block_table, compressed_rope_cos, compressed_rope_sin,
        compressor_wkv, compressor_norm_weight, compressed_slots, index_wk, index_norm_weight, index_wq_b,
        index_wq_b_scale, index_weights_proj, topk_indices, candidate_mask, output_window, output_arrived,
        attn_out, group_base, tp_rank, num_tokens, attention_epoch,
    )
    mhc_post(attn_out, x_hc, post_mix, residual_mix, output)
    return output


@pl.jit.inline(auto_scope=False)
def decode_c1a_full_sharded(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
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
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    candidate_mask: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.UINT8],
    gathered: pl.Tensor[[T_DYN, D], pl.BF16],
    input_window: pld.DistributedTensor[[DECODE_INPUT_MAX_TOKENS, D], pl.BF16],
    input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    next_pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
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
    # Sequence parallel: the residual stream stays local, only the normalized
    # Attention input crosses the TP group ([T_local, D] -> [T, D]).  The shared
    # block takes this rank's own row count, so it zeroes the slab's padding rows
    # before they are published.
    _, local_count = slab_owner(tp_rank, tokens, num_tokens)
    mhc_mixes(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, next_pre_mix, post_mix, residual_mix)
    mhc_pre_norm(x_hc, pre_mix, attn_norm_weight, hidden, normed, local_count)
    gathered = decode_tp_input_all_gather(
        normed, input_window, input_arrived, gathered, group_base, tp_rank, num_tokens, attention_epoch
    )
    decode_attn_c1a_full_sharded(
        gathered, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin, window_slots, window_indices, window_cache,
        window_cache_scale, compressed_cache, compressed_cache_scale, request_ids, compressed_lens,
        index_cache, index_cache_scale, index_block_table, compressed_rope_cos, compressed_rope_sin,
        compressor_wkv, compressor_norm_weight, compressed_slots, index_wk, index_norm_weight, index_wq_b,
        index_wq_b_scale, index_weights_proj, topk_indices, candidate_mask, output_window, output_arrived,
        attn_out, group_base, tp_rank, num_tokens, attention_epoch,
    )
    mhc_post(attn_out, x_hc, post_mix, residual_mix, output)
    return output


@pl.jit
def l2_decode_c1a_full(
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
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]
    ],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[
        pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    candidate_mask: pl.Out[pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.UINT8]],
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
    return decode_c1a_full(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight, wq_a, wq_a_scale,
        q_norm_weight,
        wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin, window_slots,
        window_indices, window_cache, window_cache_scale, compressed_cache, compressed_cache_scale,
        request_ids, compressed_lens, index_cache, index_cache_scale, index_block_table, compressed_rope_cos,
        compressed_rope_sin, compressor_wkv, compressor_norm_weight, compressed_slots, index_wk,
        index_norm_weight, index_wq_b, index_wq_b_scale, index_weights_proj, topk_indices, candidate_mask,
        output_window, output_arrived, output, next_pre_mix, group_base, tp_rank, num_tokens,
        attention_epoch,
    )


@pl.jit
def l2_decode_c1a_full_sharded(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
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
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]
    ],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[
        pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    candidate_mask: pl.Out[pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.UINT8]],
    gathered: pl.InOut[pl.Tensor[[T_DYN, D], pl.BF16]],
    input_window: pld.DistributedTensor[[DECODE_INPUT_MAX_TOKENS, D], pl.BF16],
    input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32]],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(0, LOCAL_T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    candidate_mask.bind_dynamic(1, CMP_POSITIONS_DYN)
    gathered.bind_dynamic(0, T_DYN)
    return decode_c1a_full_sharded(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight, wq_a, wq_a_scale,
        q_norm_weight,
        wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin, window_slots,
        window_indices, window_cache, window_cache_scale, compressed_cache, compressed_cache_scale,
        request_ids, compressed_lens, index_cache, index_cache_scale, index_block_table, compressed_rope_cos,
        compressed_rope_sin, compressor_wkv, compressor_norm_weight, compressed_slots, index_wk,
        index_norm_weight, index_wq_b, index_wq_b_scale, index_weights_proj, topk_indices, candidate_mask,
        gathered, input_window, input_arrived, output_window, output_arrived, output, next_pre_mix,
        group_base, tp_rank, num_tokens,
        attention_epoch,
    )


@pl.jit.host
def l3_decode_c1a_full(
    x_hc: pl.Tensor[[TP_SIZE, T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[TP_SIZE, T_DYN, HC_MULT], pl.FP32],
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
    wo_a: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
    rope_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[TP_SIZE, T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache: pl.InOut[pl.Tensor[[TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[[TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]
    ],
    request_ids: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[TP_SIZE, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[
        pl.Tensor[[TP_SIZE, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    index_block_table: pl.Tensor[[TP_SIZE, B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    index_wk: pl.Tensor[[TP_SIZE, HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[TP_SIZE, INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[TP_SIZE, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[TP_SIZE, Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[TP_SIZE, D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, INDEX_TOPK], pl.INT32]],
    candidate_mask: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, CMP_POSITIONS_DYN], pl.UINT8]],
    output: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(1, T_DYN)
    rope_cos.bind_dynamic(1, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(1, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(1, B_DYN)
    index_block_table.bind_dynamic(2, TABLE_DYN)
    candidate_mask.bind_dynamic(2, CMP_POSITIONS_DYN)
    transport = pld.alloc_window_buffer([DECODE_MAX_TOKENS, D], dtype=pl.FP32)
    signals = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
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
        l2_decode_c1a_full(
            x_hc[rank], pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank],
            hc_attn_base[rank], attn_norm_weight[rank], wq_a[rank],
            wq_a_scale_r, q_norm_weight[rank], wq_b[rank], wq_b_scale_r, wkv[rank],
            wkv_scale_r, kv_norm_weight[rank], attn_sink[rank], wo_a[rank], wo_a_scale[rank], wo_b[rank],
            wo_b_scale_r, rope_cos[rank], rope_sin[rank], window_slots[rank], window_indices[rank],
            window_cache[rank], window_cache_scale[rank], compressed_cache[rank],
            compressed_cache_scale[rank], request_ids[rank], compressed_lens[rank], index_cache[rank],
            index_cache_scale[rank], index_block_table[rank], compressed_rope_cos[rank],
            compressed_rope_sin[rank], compressor_wkv[rank], compressor_norm_weight[rank],
            compressed_slots[rank], index_wk[rank], index_norm_weight[rank], index_wq_b[rank],
            index_wq_b_scale_r, index_weights_proj[rank], topk_indices[rank], candidate_mask[rank],
            output_window, output_arrived, output[rank], next_pre_mix[rank], 0, rank, num_tokens,
            attention_epoch, device=rank,
        )


@pl.jit.host
def l3_decode_c1a_full_sharded(
    x_hc: pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT], pl.FP32],
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
    wo_a: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
    rope_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[TP_SIZE, T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache: pl.InOut[pl.Tensor[[TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[[TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]
    ],
    request_ids: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[TP_SIZE, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[
        pl.Tensor[[TP_SIZE, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    index_block_table: pl.Tensor[[TP_SIZE, B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    index_wk: pl.Tensor[[TP_SIZE, HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[TP_SIZE, INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[TP_SIZE, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[TP_SIZE, Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[TP_SIZE, D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, INDEX_TOPK], pl.INT32]],
    candidate_mask: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, CMP_POSITIONS_DYN], pl.UINT8]],
    output: pl.Out[pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT], pl.FP32]],
    gathered: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(1, LOCAL_T_DYN)
    rope_cos.bind_dynamic(1, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(1, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(1, B_DYN)
    index_block_table.bind_dynamic(2, TABLE_DYN)
    candidate_mask.bind_dynamic(2, CMP_POSITIONS_DYN)
    transport = pld.alloc_window_buffer([DECODE_MAX_TOKENS, D], dtype=pl.FP32)
    signals = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    input_transport = pld.alloc_window_buffer([DECODE_INPUT_MAX_TOKENS, D], dtype=pl.BF16)
    input_signals = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
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
        input_window = pld.window(input_transport, [DECODE_INPUT_MAX_TOKENS, D], dtype=pl.BF16)
        input_arrived = pld.window(input_signals, [TP_SIZE, 1], dtype=pl.INT32)
        l2_decode_c1a_full_sharded(
            x_hc[rank], pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank],
            hc_attn_base[rank], attn_norm_weight[rank], wq_a[rank],
            wq_a_scale_r, q_norm_weight[rank], wq_b[rank], wq_b_scale_r, wkv[rank],
            wkv_scale_r, kv_norm_weight[rank], attn_sink[rank], wo_a[rank], wo_a_scale[rank], wo_b[rank],
            wo_b_scale_r, rope_cos[rank], rope_sin[rank], window_slots[rank], window_indices[rank],
            window_cache[rank], window_cache_scale[rank], compressed_cache[rank],
            compressed_cache_scale[rank], request_ids[rank], compressed_lens[rank], index_cache[rank],
            index_cache_scale[rank], index_block_table[rank], compressed_rope_cos[rank],
            compressed_rope_sin[rank], compressor_wkv[rank], compressor_norm_weight[rank],
            compressed_slots[rank], index_wk[rank], index_norm_weight[rank], index_wq_b[rank],
            index_wq_b_scale_r, index_weights_proj[rank], topk_indices[rank], candidate_mask[rank],
            gathered[rank], input_window, input_arrived, output_window, output_arrived,
            output[rank], next_pre_mix[rank], 0, rank, num_tokens, attention_epoch, device=rank,
        )


def configure_c1a_requests(values, mode, start_pos):
    """Configure independent single-token requests and their physical cache rows."""
    ranks, tokens = values["request_ids"].shape
    pages = values["index_block_table"].shape[-1]
    if start_pos is None:
        return
    lengths = torch.tensor(start_pos, dtype=torch.int32) + 1
    page_counts = (lengths + 127) // 128
    offsets = torch.cat((torch.zeros(1, dtype=torch.int32), page_counts.cumsum(0)[:-1]))
    table = torch.zeros(tokens, pages, dtype=torch.int32)
    for request, (offset, count) in enumerate(zip(offsets.tolist(), page_counts.tolist())):
        table[request, :count] = torch.arange(offset, offset + count, dtype=torch.int32)
    values["index_block_table"] = table.unsqueeze(0).expand(ranks, -1, -1).contiguous()
    values["request_ids"] = torch.arange(tokens, dtype=torch.int32).repeat(ranks, 1)
    values["compressed_lens"] = lengths.repeat(ranks, 1)
    positions = torch.tensor(start_pos, dtype=torch.int64)
    slots = offsets.long() * 128 + positions
    values["window_slots"] = slots.repeat(ranks, 1)
    values["window_indices"].fill_(-1)
    values["compressed_indices"].fill_(-1)
    values["candidate_mask"].zero_()
    for request, (offset, position) in enumerate(zip(offsets.tolist(), positions.tolist())):
        history = torch.arange(max(0, position - 127), position + 1, dtype=torch.int32)
        values["window_indices"][:, request, :history.numel()] = offset * 128 + history
        count = min(position + 1, INDEX_TOPK)
        values["compressed_indices"][:, request, :count] = offset * 128 + torch.arange(count)
        values["candidate_mask"][:, request, :position + 1] = 1
    freqs_cos, freqs_sin = precompute_rope_tables(int(positions.max()) + 1, compressed_attention=True)
    values["rope_cos"] = freqs_cos[positions].unsqueeze(0).repeat(ranks, 1, 1)
    values["rope_sin"] = freqs_sin[positions].unsqueeze(0).repeat(ranks, 1, 1)
    if mode == "full":
        values["compressed_slots"] = slots.repeat(ranks, 1)
        values["compressed_rope_cos"] = values["rope_cos"].clone()
        values["compressed_rope_sin"] = values["rope_sin"].clone()


def build_hc_validation_values(mode, tokens, pages, requests=1, seed=17, case="random", sharded=False, start_pos=None):
    """Attention fixture plus the HC stream and mixing parameters.

    With ``sharded`` each rank holds its own contiguous slab of the batch's token
    rows, while every per-token Attention operand stays full range (each rank keeps
    a replica, exactly like the kernel's parameter list).
    """
    import math

    values = build_validation_values(mode, tokens, pages, requests, seed, case)
    configure_c1a_requests(values, mode, start_pos)
    del values["x"]
    generator = torch.Generator().manual_seed(seed)
    local_tokens = (tokens + TP_SIZE - 1) // TP_SIZE if sharded else tokens
    values["x_hc"] = torch.randn(TP_SIZE, local_tokens, HC_MULT, D, generator=generator)
    # The coefficients are staggered: this site collapses with the pre-mix the previous
    # sub-layer produced (identity one-hot at the very first site). The fixture carries a
    # general sigmoid-distributed mix instead of the seed so every lane participates.
    values["pre_mix"] = torch.sigmoid(torch.randn(TP_SIZE, local_tokens, HC_MULT, generator=generator))
    values["hc_attn_fn"] = torch.randn(TP_SIZE, MIX_HC, HC_DIM, generator=generator) / math.sqrt(HC_DIM)
    values["hc_attn_scale"] = torch.randn(TP_SIZE, 3, generator=generator)
    values["hc_attn_base"] = torch.randn(TP_SIZE, MIX_HC, generator=generator)
    # The block normalizes the collapsed stream before attention; the checkpoint carries one
    # weight per layer, so the fixture does too. BF16: the harness takes each spec's dtype from
    # the fixture tensor, and the kernel declares the weight in the activation dtype.
    values["attn_norm_weight"] = (
        torch.randn(TP_SIZE, D, generator=generator) * 0.1 + 1.0
    ).to(torch.bfloat16)
    values["output"] = torch.zeros(TP_SIZE, local_tokens, HC_MULT, D, dtype=torch.float32)
    values["next_pre_mix"] = torch.zeros(TP_SIZE, local_tokens, HC_MULT, dtype=torch.float32)
    # Attention-input gather scratch: one full-token buffer per rank.
    values["gathered"] = torch.zeros(TP_SIZE, tokens, D, dtype=torch.bfloat16)
    return values


def golden_c1a_hc_case(tensors, attn_golden, sharded=False):
    """Reference for an mHC-wired entry: mixes -> attention -> expansion.

    One DP group collapses its stream with the staggered pre-mix the fixture carries, normalizes
    it, runs the attention golden per rank, hands this site's computed pre-mix
    to the next sub-layer, publishes the cache state the operator returned, reduces the
    attention partials, and expands the reduced result back onto the four streams. With
    ``sharded`` the ranks hold disjoint token rows: the normalized inputs are gathered in
    rank order for the attention pass and each rank only expands the rows it owns.
    """
    parameters = [name for name in inspect.signature(attn_golden).parameters if name != "x"]
    full_tokens = tensors["rope_cos"].shape[1]

    def publish(rank, result):
        for name in CACHE_STATE_NAMES:
            value = getattr(result, name, None)
            if value is not None and name in tensors:
                destination = tensors[name][rank]
                if name == "candidate_mask":
                    value = value.clone()
                    destination.zero_()
                    destination[:, : value.shape[1]].copy_(value.to(destination.dtype))
                else:
                    destination.view(torch.uint8).copy_(value.contiguous().view(torch.uint8))

    partials = []
    mixes = []
    normalized_local = []
    for rank in range(TP_SIZE):
        x_hc = tensors["x_hc"][rank]
        next_pre_mix, post_mix, residual_mix = golden_mhc_mixes(
            x_hc, tensors["hc_attn_fn"][rank], tensors["hc_attn_scale"][rank], tensors["hc_attn_base"][rank]
        )
        tensors["next_pre_mix"][rank].copy_(next_pre_mix)
        hidden = golden_mhc_pre(x_hc, tensors["pre_mix"][rank])
        normalized = golden_rms_norm(hidden, tensors["attn_norm_weight"][rank])
        mixes.append((x_hc, post_mix, residual_mix))
        if sharded:
            normalized_local.append(normalized)
            continue
        result = attn_golden(x=normalized, **{name: tensors[name][rank] for name in parameters})
        partials.append(result.output.float())
        publish(rank, result)
    if sharded:
        # The ranks contribute contiguous rows; padding past the token count falls away.
        gathered = torch.cat(normalized_local, dim=0)[:full_tokens]
        # Every rank runs the full-range pass with its own weight shard, so the
        # partials still have to be summed (the ReduceScatter the kernel performs).
        for rank in range(TP_SIZE):
            result = attn_golden(x=gathered, **{name: tensors[name][rank] for name in parameters})
            partials.append(result.output.float())
            publish(rank, result)
            # The harness compares the scratch too, so the reference fills it in.
            tensors["gathered"][rank].zero_()
            tensors["gathered"][rank][: gathered.shape[0]].copy_(gathered)
    reduced = torch.zeros_like(partials[0])
    for partial in partials:
        reduced += partial
    sublayer = reduced.to(torch.bfloat16)
    if sharded:
        width = (full_tokens + TP_SIZE - 1) // TP_SIZE
        for rank, (x_hc, post_mix, residual_mix) in enumerate(mixes):
            first = min(rank * width, full_tokens)
            count = max(0, min(width, full_tokens - first))
            tensors["output"][rank].zero_()
            if count:
                tensors["output"][rank][:count].copy_(
                    golden_mhc_post(sublayer[first : first + count], x_hc[:count], post_mix[:count], residual_mix[:count])
                )
        return
    for rank, (x_hc, post_mix, residual_mix) in enumerate(mixes):
        tensors["output"][rank].copy_(golden_mhc_post(sublayer, x_hc, post_mix, residual_mix))


def golden_decode_c1a_full_case(tensors, sharded=False):
    """Fill the mHC-wired full attention entry expectations from the operator reference."""
    golden_c1a_hc_case(tensors, golden_decode_attn_c1a_full, sharded)


if __name__ == "__main__":
    import argparse

    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="TP1/2/4 A5 mHC C1A full validation")
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=[1, 2, 4])
    parser.add_argument(
        "--dp", type=int, default=1, choices=[1],
        help="one data-parallel group; the shared A5 entry command passes 1",
    )
    parser.add_argument("-p", "--platform", type=str, default="a5", choices=["a5", "a5sim"])
    parser.add_argument(
        "-d", "--device", "--devices", type=str, default=None,
        help="comma-separated physical device IDs, one per rank; default: 0 through TP-1",
    )
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--pages", type=int, default=8)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--case", choices=["random", "ragged", "masked", "sink"], default="random")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--check-fp4", action="store_true", help="run the CPU E2M1 midpoint check and exit")
    parser.add_argument("--wiring", choices=("both", "replicated", "sharded"), default="both",
                        help="which wiring to run; replay directories use a separate subdirectory per wiring")
    parser.add_argument("--start-pos", type=str, default=None,
                        help="absolute decode start positions; a comma-separated list sets batch to its length")
    args = parser.parse_args()
    if args.check_fp4:
        raise SystemExit(check_fp4_boundaries())
    if args.start_pos is not None:
        try:
            positions = [int(value.strip()) for value in args.start_pos.split(",")]
        except ValueError:
            parser.error("--start-pos must contain comma-separated integers")
        if not 1 <= len(positions) <= min(MAX_BATCH_PER_DP, DECODE_MAX_TOKENS):
            parser.error("--start-pos batch exceeds the decode request capacity")
        if any(position < 0 or position >= M.max_position_embeddings for position in positions):
            parser.error(f"--start-pos values must be in [0, {M.max_position_embeddings})")
        if args.case not in ("random", "mixed"):
            parser.error("--start-pos requires the default random/mixed case")
        args.start_pos = positions
        args.tokens = args.requests = len(positions)
        args.pages = sum((position + 128) // 128 for position in positions)
    if args.wiring in ("both", "sharded"):
        if args.tokens < 1 or args.tp < 1:
            parser.error("sequence-parallel requires positive tokens and tp")
        args.local_tokens = (args.tokens + args.tp - 1) // args.tp
    if args.tp != TP_SIZE:
        parser.error(f"--tp {args.tp} does not match the TP_SIZE {TP_SIZE} the operator was built for")
    try:
        devices = list(range(TP_SIZE)) if args.device is None else [int(d) for d in args.device.split(",")]
    except ValueError:
        parser.error("device IDs must be comma-separated integers")
    if len(devices) != TP_SIZE or len(set(devices)) != len(devices) or min(devices) < 0:
        parser.error(f"device IDs must be {TP_SIZE} distinct nonnegative integers, one per rank")
    if not 1 <= args.tokens <= min(DECODE_MAX_TOKENS, args.pages * 128) or args.pages < 1:
        parser.error(f"require 1 <= tokens <= min({DECODE_MAX_TOKENS}, pages * 128), pages >= 1")

    if not 1 <= args.requests <= min(MAX_BATCH_PER_DP, args.tokens):
        parser.error(f"require 1 <= requests <= min({MAX_BATCH_PER_DP}, tokens {args.tokens})")

    results = []
    wirings = ("replicated", "sharded") if args.wiring == "both" else (args.wiring,)
    for wiring in wirings:
        sharded = wiring == "sharded"
        host = l3_decode_c1a_full_sharded if sharded else l3_decode_c1a_full
        values = build_hc_validation_values(
            "full", args.tokens, args.pages, args.requests, args.seed, args.case,
            sharded, args.start_pos,
        )
        # Independent initial storage for each harness allocation.
        specs = [
            TensorSpec(n, list(values[n].shape), values[n].dtype, init_value=lambda n=n: values[n].clone())
            for n in host.param_names if n not in ("num_tokens", "attention_epoch")
        ]

        specs.append(ScalarSpec("num_tokens", torch.int32, args.tokens))
        specs.append(ScalarSpec("attention_epoch", torch.int32, 1, compile_runtime=True))

        def golden_fn(tensors, sharded=sharded):
            golden_decode_c1a_full_case(tensors, sharded)

        output_budget = ratio_allclose(atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL, max_error_ratio=OUTPUT_MAX_ERROR_RATIO)
        comparisons = {
            "topk_indices": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.02),
            "candidate_mask": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
            "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            "output": output_budget,
            # BF16 rounding tolerance for gathered rows.
            "gathered": ratio_allclose(atol=2 ** -6, rtol=2 ** -6),
            **cache_compare(len(devices), args.tokens, "full"),
        }
        comparisons = {name: fn for name, fn in comparisons.items() if name in host.param_names}
        result = run(
            fn=host,
            specs=specs,
            golden_fn=golden_fn,
            compile_only=args.compile_only,
            save_data=args.save_data,
            golden_data=_decode_utils.wiring_replay_dir(args.golden_data, wiring),
            runtime_dir=_decode_utils.wiring_replay_dir(args.runtime_dir, wiring),
            config=dict(
                platform=args.platform,
                dump_passes=args.dump_passes,
                enable_chip_swimlane=args.enable_chip_swimlane,
                distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
            ),
            rtol=1e-3,
            atol=1e-3,
            compare_fn=comparisons,
        )
        results.append(result)
    for result in results:
        if not result.passed:
            raise SystemExit(result.error or 1)
