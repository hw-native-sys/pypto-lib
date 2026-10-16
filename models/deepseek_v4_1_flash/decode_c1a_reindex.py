# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Continuous-batch decode C1A reindex attention wired through mHC."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import (
    COMPRESSED_CACHE_GROUP,
    DECODE_INPUT_MAX_TOKENS,
    DECODE_MAX_TOKENS,
    MAX_BATCH_PER_DP,
    INDEX_CACHE_GROUP,
    WINDOW_CACHE_GROUP,
    FLASH as M,
    B_DYN,
    CMP_BLOCKS_DYN,
    INDEX_BLOCKS_DYN,
    INDEX_POOL_BLOCKS_DYN,
    ORI_BLOCKS_DYN,
    TABLE_DYN,
    T_DYN,
    TP_SIZE,
)


from config import LOCAL_T_DYN
from decode_cp_allgather import decode_tp_input_all_gather
from decode_c1a_full import golden_c1a_hc_case, build_hc_validation_values
from decode_attn_c1a import OUTPUT_ATOL, OUTPUT_RTOL, OUTPUT_MAX_ERROR_RATIO, check_fp4_boundaries
from decode_attn_c1a import cache_compare
from golden import ScalarSpec, TensorSpec, ratio_allclose, run
import utils as _decode_utils
from hc_pre import c1a_mhc_pre_norm as mhc_pre_norm
from decode_cp_allgather import slab_owner
from decode_attn_c1a import (
    c1a_finish,
    c1a_finish_sharded,
    c1a_index,
    c1a_prepare,
    c1a_previous_epoch,
    golden_decode_attn_c1a_reindex,
)
from decode_indexer import score_reindex, select_pool_index_topk
from hc_mixes import c1a_mhc_mixes as mhc_mixes
from hc_post import c1a_mhc_post as mhc_post


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


def make_decode_attn_c1a_reindex(output_reduce=None):
    """Build the decode_attn_c1a_reindex leaf around one output collective."""
    reducer = output_reduce or c1a_finish

    @pl.jit.inline(auto_scope=False)
    def decode_attn_c1a_reindex(
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
        wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
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
        pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
        pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
        index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
        index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
        index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
        topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
        output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        output: pl.Tensor[[LOCAL_T_DYN, D], pl.BF16],
        group_base: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        cache_ready = c1a_previous_epoch(output_arrived, attention_epoch)
        (qr, query, qr_tid, q_tid) = c1a_prepare(
            x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, rope_cos,
            rope_sin, window_slots, window_cache, window_cache_scale, num_tokens, cache_ready,
        )
        # Allocate eight compact score columns per stored pool block.
        score_width = pl.tensor.dim(pool_block_ids, 1) * 8
        (index_query, index_query_scale, index_weights, scores, query_tid, weights_tid) = c1a_index(
            x, qr, index_wq_b, index_wq_b_scale, index_weights_proj, rope_cos, rope_sin,
            num_tokens, score_width, cache_ready, qr_tid,
        )
        # The previous Attention epoch includes the Full pool publication.
        scores_tid = score_reindex(
            index_query, index_query_scale, index_weights, index_cache, index_cache_scale,
            index_block_table, request_ids, compressed_lens, pool_block_ids, pool_block_counts,
            scores, num_tokens, cache_ready, query_tid, weights_tid, cache_ready,
        )
        topk_tid = select_pool_index_topk(
            scores, pool_block_ids, pool_block_counts, index_block_table, request_ids, compressed_lens,
            topk_indices, num_tokens, scores_tid,
        )
        reducer(
            query, window_cache, window_cache_scale, compressed_cache, compressed_cache_scale, window_indices,
            topk_indices, attn_sink, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin, output_window, output_arrived,
            output, group_base, tp_rank, num_tokens, attention_epoch, q_tid, topk_tid,
        )
        return output, topk_indices

    return decode_attn_c1a_reindex


decode_attn_c1a_reindex = make_decode_attn_c1a_reindex()
decode_attn_c1a_reindex_sharded = make_decode_attn_c1a_reindex(c1a_finish_sharded)


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
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
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
    # The collapse + norm block itself is shared with the spec-driven modes.
    mhc_mixes(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, next_pre_mix, post_mix, residual_mix)
    mhc_pre_norm(x_hc, pre_mix, attn_norm_weight, hidden, normed, num_tokens)
    decode_attn_c1a_reindex(
        normed, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
        wo_a, wo_b, wo_b_scale, rope_cos, rope_sin, window_slots, window_indices, window_cache,
        window_cache_scale, compressed_cache, compressed_cache_scale, request_ids, compressed_lens,
        index_cache, index_cache_scale, index_block_table, pool_block_ids, pool_block_counts,
        index_wq_b, index_wq_b_scale, index_weights_proj, topk_indices, output_window, output_arrived,
        attn_out, group_base, tp_rank, num_tokens, attention_epoch,
    )
    mhc_post(attn_out, x_hc, post_mix, residual_mix, output)
    return output


@pl.jit.inline(auto_scope=False)
def decode_c1a_reindex_sharded(
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
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
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
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
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
    decode_attn_c1a_reindex_sharded(
        gathered, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
        wo_a, wo_b, wo_b_scale, rope_cos, rope_sin, window_slots, window_indices, window_cache,
        window_cache_scale, compressed_cache, compressed_cache_scale, request_ids, compressed_lens,
        index_cache, index_cache_scale, index_block_table, pool_block_ids, pool_block_counts,
        index_wq_b, index_wq_b_scale, index_weights_proj, topk_indices, output_window, output_arrived,
        attn_out, group_base, tp_rank, num_tokens, attention_epoch,
    )
    mhc_post(attn_out, x_hc, post_mix, residual_mix, output)
    return output


@pl.jit
def l2_decode_c1a_reindex(
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
        pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]
    ],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    pool_block_ids: pl.InOut[pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.InOut[pl.Tensor[[T_DYN], pl.INT32]],
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
    pool_block_ids.bind_dynamic(1, INDEX_POOL_BLOCKS_DYN)
    return decode_c1a_reindex(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight, wq_a, wq_a_scale,
        q_norm_weight,
        wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, wo_b_scale, rope_cos,
        rope_sin, window_slots, window_indices, window_cache, window_cache_scale, compressed_cache,
        compressed_cache_scale, request_ids, compressed_lens, index_cache, index_cache_scale,
        index_block_table, pool_block_ids, pool_block_counts, index_wq_b, index_wq_b_scale,
        index_weights_proj, topk_indices, output_window, output_arrived, output, next_pre_mix,
        group_base, tp_rank, num_tokens, attention_epoch,
    )


@pl.jit
def l2_decode_c1a_reindex_sharded(
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
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
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
    index_cache_scale: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    pool_block_ids: pl.InOut[pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.InOut[pl.Tensor[[T_DYN], pl.INT32]],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
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
    gathered.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    pool_block_ids.bind_dynamic(1, INDEX_POOL_BLOCKS_DYN)
    return decode_c1a_reindex_sharded(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight, wq_a, wq_a_scale,
        q_norm_weight,
        wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, wo_b_scale, rope_cos,
        rope_sin, window_slots, window_indices, window_cache, window_cache_scale, compressed_cache,
        compressed_cache_scale, request_ids, compressed_lens, index_cache, index_cache_scale,
        index_block_table, pool_block_ids, pool_block_counts, index_wq_b, index_wq_b_scale,
        index_weights_proj, topk_indices, gathered, input_window, input_arrived, output_window,
        output_arrived, output, next_pre_mix, group_base, tp_rank, num_tokens, attention_epoch,
    )


@pl.jit.host
def l3_decode_c1a_reindex(
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
    wo_a: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
    rope_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[TP_SIZE, T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[pl.Tensor[[TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]],
    request_ids: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[TP_SIZE, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[TP_SIZE, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[TP_SIZE, B_DYN, TABLE_DYN], pl.INT32],
    pool_block_ids: pl.InOut[pl.Tensor[[TP_SIZE, T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.InOut[pl.Tensor[[TP_SIZE, T_DYN], pl.INT32]],
    index_wq_b: pl.Tensor[[TP_SIZE, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[TP_SIZE, Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[TP_SIZE, D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, INDEX_TOPK], pl.INT32]],
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
    pool_block_ids.bind_dynamic(2, INDEX_POOL_BLOCKS_DYN)
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
        l2_decode_c1a_reindex(
            x_hc[rank], pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank],
            hc_attn_base[rank], attn_norm_weight[rank], wq_a[rank],
            wq_a_scale_r, q_norm_weight[rank], wq_b[rank], wq_b_scale_r, wkv[rank],
            wkv_scale_r, kv_norm_weight[rank], attn_sink[rank], wo_a[rank], wo_b[rank],
            wo_b_scale_r, rope_cos[rank], rope_sin[rank], window_slots[rank], window_indices[rank],
            window_cache[rank], window_cache_scale[rank], compressed_cache[rank],
            compressed_cache_scale[rank], request_ids[rank], compressed_lens[rank], index_cache[rank],
            index_cache_scale[rank], index_block_table[rank], pool_block_ids[rank], pool_block_counts[rank], index_wq_b[rank],
            index_wq_b_scale_r, index_weights_proj[rank], topk_indices[rank], output_window,
            output_arrived, output[rank], next_pre_mix[rank], 0, rank, num_tokens, attention_epoch,
            device=rank,
        )


@pl.jit.host
def l3_decode_c1a_reindex_sharded(
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
    wo_a: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
    rope_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[TP_SIZE, T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[pl.Tensor[[TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]],
    request_ids: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[TP_SIZE, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[TP_SIZE, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[TP_SIZE, B_DYN, TABLE_DYN], pl.INT32],
    pool_block_ids: pl.InOut[pl.Tensor[[TP_SIZE, T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.InOut[pl.Tensor[[TP_SIZE, T_DYN], pl.INT32]],
    index_wq_b: pl.Tensor[[TP_SIZE, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[TP_SIZE, Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[TP_SIZE, D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, INDEX_TOPK], pl.INT32]],
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
    pool_block_ids.bind_dynamic(2, INDEX_POOL_BLOCKS_DYN)
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
        l2_decode_c1a_reindex_sharded(
            x_hc[rank], pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank],
            hc_attn_base[rank], attn_norm_weight[rank], wq_a[rank],
            wq_a_scale_r, q_norm_weight[rank], wq_b[rank], wq_b_scale_r, wkv[rank],
            wkv_scale_r, kv_norm_weight[rank], attn_sink[rank], wo_a[rank], wo_b[rank],
            wo_b_scale_r, rope_cos[rank], rope_sin[rank], window_slots[rank], window_indices[rank],
            window_cache[rank], window_cache_scale[rank], compressed_cache[rank],
            compressed_cache_scale[rank], request_ids[rank], compressed_lens[rank], index_cache[rank],
            index_cache_scale[rank], index_block_table[rank], pool_block_ids[rank], pool_block_counts[rank], index_wq_b[rank],
            index_wq_b_scale_r, index_weights_proj[rank], topk_indices[rank], gathered[rank],
            input_window, input_arrived, output_window, output_arrived, output[rank], next_pre_mix[rank], 0, rank, num_tokens, attention_epoch,
            device=rank,
        )


def golden_decode_c1a_reindex_case(tensors, sharded=False):
    """Fill the mHC-wired reindex entry expectations from the operator reference."""
    golden_c1a_hc_case(tensors, golden_decode_attn_c1a_reindex, sharded)


if __name__ == "__main__":
    import argparse

    import torch

    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="TP1/2/4 A5 mHC C1A reindex validation")
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
        host = l3_decode_c1a_reindex_sharded if sharded else l3_decode_c1a_reindex
        values = build_hc_validation_values(
            "reindex", args.tokens, args.pages, args.requests, args.seed, args.case,
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
            golden_decode_c1a_reindex_case(tensors, sharded)

        output_budget = ratio_allclose(atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL, max_error_ratio=OUTPUT_MAX_ERROR_RATIO)
        comparisons = {
            "topk_indices": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.02),
            "pool_block_ids": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
            "pool_block_counts": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
            "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            "output": output_budget,
            # BF16 rounding tolerance for gathered rows.
            "gathered": ratio_allclose(atol=2 ** -6, rtol=2 ** -6),
            **cache_compare(len(devices), args.tokens, "reindex"),
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
