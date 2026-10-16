# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""C2A Full decode Attention half-layer composition entry."""

import hc_post as _decode_hc_post
import hc_pre as _decode_hc_pre
import utils as _decode_utils

import pypto.language as pl
import pypto.language.distributed as pld

from config import (
    FLASH as M,
    DECODE_INPUT_MAX_TOKENS,
    DECODE_MAX_TOKENS,
    TP_SIZE,
    T_DYN,
    ORI_BLOCKS_DYN,
    CMP_BLOCKS_DYN,
    INDEX_BLOCKS_DYN,
    B_DYN,
    TABLE_DYN,
    Q_START_DYN,
    STATE_BLOCKS_DYN,
    AttentionMode,
    COMPRESSED_CACHE_GROUP,
    INDEX_CACHE_GROUP,
    STATE_CAPACITY,
    WINDOW_CACHE_GROUP,
)

from compressor import STATE_WIDTH


import decode_attn_c2a as full
from golden import ratio_allclose

# PyPTO resolves inline calls and JIT-body constants only from direct names.
from config import LOCAL_T_DYN
from decode_cp_allgather import decode_tp_input_all_gather
from decode_attn_c2a import decode_attn_c2a_full, decode_attn_c2a_full_sharded
from hc_pre import attention_pre
from decode_cp_allgather import slab_owner
from hc_post import mhc_post


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

KIND = "C2A_FULL"
REPRESENTATIVE_LAYER_ID = 2
KERNEL_READY = True
LEAF_NAMES = (*tuple(name for name in full.INPUT_NAMES if name != "x"), "topk_indices")
ATTENTION_SPEC_NAMES = (
    *_decode_utils.BOUNDARY_PREFIX_NAMES,
    *LEAF_NAMES,
    *_decode_utils.BOUNDARY_OUTPUT_NAMES,
    *_decode_utils.SCALAR_NAMES,
)
SHARDED_ATTENTION_SPEC_NAMES = (
    *_decode_utils.BOUNDARY_PREFIX_NAMES,
    *LEAF_NAMES,
    *_decode_utils.BOUNDARY_OUTPUT_NAMES,
    "gathered",
    *_decode_utils.SCALAR_NAMES,
)


def skip_reason(layer_id=REPRESENTATIVE_LAYER_ID):
    layer = M.layer_config(layer_id)
    if layer.compression_ratio != 2 or layer.mode != AttentionMode.FULL:
        return f"layer {layer_id} does not resolve to {KIND}"
    return None if KERNEL_READY else "C2A Full attention half-layer composition is pending"


@pl.jit.inline(auto_scope=False)
def decode_c2a_full(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
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
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
    ],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0
    ],
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
    attention_output: pl.Tensor[[T_DYN, D], pl.BF16],
    attention_hidden: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    attention_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Run one C2A Full sublayer and publish its cache, state, and next pre-mix."""
    tokens = pl.tensor.dim(x_hc, 0)
    attention_input = pl.create_tensor([tokens, D], dtype=pl.BF16)
    normalized_attention = pl.create_tensor([tokens, D], dtype=pl.BF16)
    post_mix, residual_mix = attention_pre(
        x_hc,
        incoming_pre_mix,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_norm_weight,
        attention_input,
        normalized_attention,
        attention_pre_mix,
        num_tokens,
    )
    group_base = rank // TP_SIZE * TP_SIZE
    tp_rank = rank % TP_SIZE
    decode_attn_c2a_full(
        normalized_attention,
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
        output_window,
        output_arrived,
        attention_output,
        group_base,
        tp_rank,
        num_tokens,
        attention_epoch,
    )
    mhc_post(attention_output, x_hc, post_mix, residual_mix, attention_hidden)
    return attention_hidden, attention_pre_mix


@pl.jit.inline(auto_scope=False)
def decode_c2a_full_sharded(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
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
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
    ],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0
    ],
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
    attention_output: pl.Tensor[[LOCAL_T_DYN, D], pl.BF16],
    attention_hidden: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    attention_pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    gathered: pl.Tensor[[T_DYN, D], pl.BF16],
    input_window: pld.DistributedTensor[[DECODE_INPUT_MAX_TOKENS, D], pl.BF16],
    input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Run one C2A Full sublayer and publish its cache, state, and next pre-mix."""
    tokens = pl.tensor.dim(x_hc, 0)
    group_base = rank // TP_SIZE * TP_SIZE
    tp_rank = rank % TP_SIZE
    # Local rows follow the fixed physical slab (``ceil(tokens / TP)``), the same
    # mapping the Attention-input AllGather publishes.  The active count only masks
    # the slab suffix, so T < TP and empty owners keep their row indices.
    _, local_count = slab_owner(tp_rank, tokens, num_tokens)
    attention_input = pl.create_tensor([tokens, D], dtype=pl.BF16)
    normalized_attention = pl.create_tensor([tokens, D], dtype=pl.BF16)
    post_mix, residual_mix = attention_pre(
        x_hc,
        incoming_pre_mix,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_norm_weight,
        attention_input,
        normalized_attention,
        attention_pre_mix,
        local_count,
    )
    gathered = decode_tp_input_all_gather(
        normalized_attention, input_window, input_arrived, gathered, group_base, tp_rank,
        num_tokens, attention_epoch,
    )
    decode_attn_c2a_full_sharded(
        gathered,
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
        output_window,
        output_arrived,
        attention_output,
        group_base,
        tp_rank,
        num_tokens,
        attention_epoch,
    )
    mhc_post(attention_output, x_hc, post_mix, residual_mix, attention_hidden)
    return attention_hidden, attention_pre_mix


@pl.jit
def l2_decode_c2a_full(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
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
    window_cache_scale: pl.InOut[
        pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]
    ],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[
        pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
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
    attention_output: pl.InOut[pl.Tensor[[T_DYN, D], pl.BF16]],
    attention_hidden: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    attention_pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Dispatch one production C2A Full composition to a device-selected JIT entry."""
    return decode_c2a_full(
        x_hc,
        incoming_pre_mix,
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
        attention_output,
        attention_hidden,
        attention_pre_mix,
        output_window,
        output_arrived,
        rank,
        num_tokens,
        attention_epoch,
    )


@pl.jit
def l2_decode_c2a_full_sharded(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
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
    window_cache_scale: pl.InOut[
        pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]
    ],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[
        pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
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
    attention_output: pl.InOut[pl.Tensor[[LOCAL_T_DYN, D], pl.BF16]],
    attention_hidden: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    attention_pre_mix: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32]],
    gathered: pl.Tensor[[T_DYN, D], pl.BF16],
    input_window: pld.DistributedTensor[[DECODE_INPUT_MAX_TOKENS, D], pl.BF16],
    input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Dispatch one production C2A Full composition to a device-selected JIT entry."""
    return decode_c2a_full_sharded(
        x_hc,
        incoming_pre_mix,
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
        attention_output,
        attention_hidden,
        attention_pre_mix,
        gathered,
        input_window,
        input_arrived,
        output_window,
        output_arrived,
        rank,
        num_tokens,
        attention_epoch,
    )


@pl.jit.host
def l3_decode_c2a_full(
    x_hc: pl.Tensor[[TP_SIZE, T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[TP_SIZE, T_DYN, HC_MULT], pl.FP32],
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
    attention_output: pl.InOut[pl.Tensor[[TP_SIZE, T_DYN, D], pl.BF16]],
    attention_hidden: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, HC_MULT, D], pl.FP32]],
    attention_pre_mix: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, HC_MULT], pl.FP32]],
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
    query_start_loc.bind_dynamic(1, Q_START_DYN)
    state_cache.bind_dynamic(1, STATE_BLOCKS_DYN)
    data_buffer = pld.alloc_window_buffer([DECODE_MAX_TOKENS, D], dtype=pl.FP32)
    arrived_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(TP_SIZE):
        data = pld.window(data_buffer, [DECODE_MAX_TOKENS, D], dtype=pl.FP32)
        arrived = pld.window(arrived_buffer, [TP_SIZE, 1], dtype=pl.INT32)
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
        l2_decode_c2a_full(
            x_hc[rank],
            incoming_pre_mix[rank],
            hc_attn_fn[rank],
            hc_attn_scale[rank],
            hc_attn_base[rank],
            attn_norm_weight[rank],
            wq_a[rank],
            wq_a_scale_r,
            q_norm_weight[rank],
            wq_b[rank],
            wq_b_scale_r,
            wkv[rank],
            wkv_scale_r,
            kv_norm_weight[rank],
            attn_sink[rank],
            wo_a[rank], wo_a_scale[rank],
            wo_b[rank],
            wo_b_scale_r,
            rope_cos[rank],
            rope_sin[rank],
            window_slots[rank],
            window_indices[rank],
            window_cache[rank],
            window_cache_scale[rank],
            compressed_cache[rank],
            compressed_cache_scale[rank],
            token_to_req_indices[rank],
            compressed_lens[rank],
            index_cache[rank],
            index_cache_scale[rank],
            index_block_table[rank],
            position_ids[rank],
            compressed_rope_cos[rank],
            compressed_rope_sin[rank],
            compressor_wkv[rank],
            compressor_wgate[rank],
            query_start_loc[rank],
            state_block_table[rank],
            state_cache[rank],
            compressor_norm_weight[rank],
            compressed_slots[rank],
            index_wk[rank],
            index_norm_weight[rank],
            index_wq_b[rank],
            index_wq_b_scale_r,
            index_weights_proj[rank],
            topk_indices[rank],
            attention_output[rank],
            attention_hidden[rank],
            attention_pre_mix[rank],
            data,
            arrived,
            rank,
            num_tokens,
            attention_epoch,
            device=rank,
        )


@pl.jit.host
def l3_decode_c2a_full_sharded(
    x_hc: pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT], pl.FP32],
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
    attention_output: pl.InOut[pl.Tensor[[TP_SIZE, LOCAL_T_DYN, D], pl.BF16]],
    attention_hidden: pl.Out[pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    attention_pre_mix: pl.Out[pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT], pl.FP32]],
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
    query_start_loc.bind_dynamic(1, Q_START_DYN)
    state_cache.bind_dynamic(1, STATE_BLOCKS_DYN)
    data_buffer = pld.alloc_window_buffer([DECODE_MAX_TOKENS, D], dtype=pl.FP32)
    arrived_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    input_transport = pld.alloc_window_buffer([DECODE_INPUT_MAX_TOKENS, D], dtype=pl.BF16)
    input_signals = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(TP_SIZE):
        data = pld.window(data_buffer, [DECODE_MAX_TOKENS, D], dtype=pl.FP32)
        arrived = pld.window(arrived_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        input_window = pld.window(input_transport, [DECODE_INPUT_MAX_TOKENS, D], dtype=pl.BF16)
        input_arrived = pld.window(input_signals, [TP_SIZE, 1], dtype=pl.INT32)
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
        l2_decode_c2a_full_sharded(
            x_hc[rank],
            incoming_pre_mix[rank],
            hc_attn_fn[rank],
            hc_attn_scale[rank],
            hc_attn_base[rank],
            attn_norm_weight[rank],
            wq_a[rank],
            wq_a_scale_r,
            q_norm_weight[rank],
            wq_b[rank],
            wq_b_scale_r,
            wkv[rank],
            wkv_scale_r,
            kv_norm_weight[rank],
            attn_sink[rank],
            wo_a[rank], wo_a_scale[rank],
            wo_b[rank],
            wo_b_scale_r,
            rope_cos[rank],
            rope_sin[rank],
            window_slots[rank],
            window_indices[rank],
            window_cache[rank],
            window_cache_scale[rank],
            compressed_cache[rank],
            compressed_cache_scale[rank],
            token_to_req_indices[rank],
            compressed_lens[rank],
            index_cache[rank],
            index_cache_scale[rank],
            index_block_table[rank],
            position_ids[rank],
            compressed_rope_cos[rank],
            compressed_rope_sin[rank],
            compressor_wkv[rank],
            compressor_wgate[rank],
            query_start_loc[rank],
            state_block_table[rank],
            state_cache[rank],
            compressor_norm_weight[rank],
            compressed_slots[rank],
            index_wk[rank],
            index_norm_weight[rank],
            index_wq_b[rank],
            index_wq_b_scale_r,
            index_weights_proj[rank],
            topk_indices[rank],
            attention_output[rank],
            attention_hidden[rank],
            attention_pre_mix[rank],
            gathered[rank],
            input_window,
            input_arrived,
            data,
            arrived,
            rank,
            num_tokens,
            attention_epoch,
            device=rank,
        )


def build_specs(args, initial_state=None):
    del initial_state
    return _decode_utils.assemble_specs(args, full.build_specs(args, "decode", full_rope_tables=False), ATTENTION_SPEC_NAMES)


def build_specs_sharded(args):
    """Specs with per-rank token slabs plus the Attention-input gather scratch."""
    return _decode_utils.assemble_specs(
        args,
        full.build_specs(args, "decode", full_rope_tables=False),
        SHARDED_ATTENTION_SPEC_NAMES,
        sharded=True,
    )


def golden_decode_c2a_full(tensors):
    normalized, post, residual = _decode_hc_pre.golden_attention_pre(tensors)
    leaf = dict(tensors, x=normalized, output=tensors["attention_output"])
    full.golden_c2a(leaf, full_rope_tables=False)
    _decode_hc_post.golden_attention_post(tensors, post, residual)


def golden_decode_c2a_full_sharded(tensors):
    gathered, post, residual = _decode_hc_pre.golden_attention_pre_sharded(tensors)
    leaf = dict(tensors, x=gathered, output=tensors["attention_output"])
    full.golden_c2a(leaf, full_rope_tables=False, sharded=True)
    _decode_hc_post.golden_attention_post(tensors, post, residual)


def comparisons(args, devices):
    compare = _decode_utils.make_boundary_comparisons(ratio_allclose(atol=1e-2, rtol=1.0 / 128))
    compare.update(
        {
            "attention_output": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
            "state_cache": ratio_allclose(atol=1e-3, rtol=1e-3),
            "topk_indices": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.02),
        }
    )
    compare.update(full.cache_compare(len(devices), args.tokens))
    return compare


def comparisons_sharded(args, devices):
    compare = _decode_utils.make_boundary_comparisons(ratio_allclose(atol=1e-2, rtol=1.0 / 128))
    compare.update(
        {
            "attention_output": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
            "gathered": ratio_allclose(atol=2 ** -6, rtol=2 ** -6),
            "state_cache": ratio_allclose(atol=1e-3, rtol=1e-3),
            "topk_indices": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.02),
        }
    )
    compare.update(full.cache_compare(len(devices), args.tokens))
    return compare


if __name__ == "__main__":
    import argparse
    from pypto.ir import DistributedConfig
    from golden import run
    from config import MAX_BATCH_PER_DP

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 C2A Full decode Attention composition")
    parser.add_argument("--stage", choices=("attention", "block"), default="attention")
    parser.add_argument("--cpu-golden", action="store_true")
    parser.add_argument("-p", "--platform", default="a5")
    parser.add_argument("-d", "--device", default=None)
    parser.add_argument("--tp", type=int, choices=(1, 4), default=TP_SIZE)
    parser.add_argument("--layer-id", type=int, default=REPRESENTATIVE_LAYER_ID)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--active-tokens", type=int)
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--case", choices=('mixed', 'long', 'masked', 'zero'), default="mixed")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--wiring", choices=("both", "replicated", "sharded"), default="both",
                        help="which wiring to run")
    parser.add_argument("--start-pos", type=str, default=None,
                        help="absolute decode start positions; a comma-separated list sets batch to its length")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    args = parser.parse_args()
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
        if args.active_tokens is not None and args.active_tokens != args.tokens:
            parser.error("--start-pos requires every request to be active")
        args.active_tokens = args.tokens
    if args.cpu_golden:
        parser.error("Block CPU goldens are provided by decode_layer.py")
    if args.stage == "block":
        parser.error("Block composition is provided by decode_layer.py")
    if not 1 <= args.tokens <= DECODE_MAX_TOKENS:
        parser.error("tokens out of range")
    args.active_tokens = args.tokens if args.active_tokens is None else args.active_tokens
    if not 1 <= args.active_tokens <= args.tokens or not 1 <= args.requests <= args.active_tokens:
        parser.error("require 1 <= requests <= active tokens <= tokens")
    if args.requests > MAX_BATCH_PER_DP:
        parser.error(f"requests must not exceed {MAX_BATCH_PER_DP}")
    if args.active_tokens != args.tokens:
        parser.error("this Attention mode requires active tokens to equal token capacity")
    args.dp, args.bench = 1, False
    if args.tp != TP_SIZE:
        parser.error("--tp must match the import-time tensor parallel configuration")
    devices = (
        list(range(args.tp))
        if args.device is None or args.compile_only
        else [int(device) for device in args.device.split(",")]
    )
    if len(devices) != args.tp or len(set(devices)) != args.tp or min(devices) < 0:
        parser.error("--device must name exactly TP distinct nonnegative device IDs")
    reason = skip_reason(args.layer_id)
    if reason:
        parser.error(reason)
    results = []
    wirings = ("replicated", "sharded") if args.wiring == "both" else (args.wiring,)
    for wiring in wirings:
        sharded = wiring == "sharded"
        if sharded:
            if args.tokens < 1 or args.tp < 1:
                parser.error("sequence-parallel requires positive tokens and tp")
            if args.active_tokens > args.tokens:
                parser.error("active tokens cannot exceed physical token capacity")
            args.local_tokens = (args.tokens + args.tp - 1) // args.tp
        specs = build_specs_sharded(args) if sharded else build_specs(args)
        compare = comparisons_sharded(args, devices) if sharded else comparisons(args, devices)
        program = l3_decode_c2a_full_sharded if sharded else l3_decode_c2a_full
        golden_fn = golden_decode_c2a_full_sharded if sharded else golden_decode_c2a_full
        result = run(
            fn=program,
            specs=specs,
            golden_fn=golden_fn,
            compare_fn=compare,
            config={
                "platform": args.platform,
                "enable_chip_swimlane": args.enable_chip_swimlane,
                "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0),
            },
            compile_only=args.compile_only,
            save_data=args.save_data,
        )
        kind_name = f"{KIND} sequence-parallel" if sharded else KIND
        print(f"[HALF] layer={args.layer_id} kind={kind_name} work_dir={result.work_dir}")
        results.append(result)
    for result in results:
        if not result.passed:
            raise SystemExit(result.error or 1)
