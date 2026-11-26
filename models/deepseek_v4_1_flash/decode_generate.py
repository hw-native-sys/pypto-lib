# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""V4.1 generation composite retaining backbone and head in one device dispatch."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import MOE_TOKENS
from decode_fwd import (
    BLOCK_SIZE,
    B_DYN,
    C2A_SOURCE_COUNT,
    COMPRESSED_CACHE_BYTES,
    COMPRESSED_POOL_UNIT_ROWS,
    D,
    DISPATCH_PACKET_WIDTH,
    DSPARK_MAIN_IN,
    ENGRAM_K,
    EP_SIZE,
    FWD_C1A_TABLE_DYN,
    FWD_C2A_TABLE_DYN,
    FWD_CMP_BLOCKS_DYN,
    FWD_INDEX_BLOCKS_DYN,
    FWD_ORI_BLOCKS_DYN,
    FWD_STATE_BLOCKS_DYN,
    HC_DIM,
    HC_MULT,
    HEAD_DIM,
    INDEX_CACHE_GROUP,
    INDEX_DIM,
    INDEX_H,
    INDEX_SOURCE_COUNT,
    KV_OUT,
    KV_SOURCE_COUNT,
    LOCAL_H,
    LOCAL_O_GROUPS,
    LOCAL_O_WIDTH,
    MIX_HC,
    MOE_INTER,
    MX_GROUP,
    MX_PACKED_LANE_COLS,
    MX_W1_PACKED_ROWS,
    MX_W2_PACKED_ROWS,
    MX_W3_PACKED_ROWS,
    N_EXPERTS,
    N_LOCAL_EXPERTS,
    O_GROUP_IN,
    O_LORA,
    Q_LORA,
    Q_START_DYN,
    RECV_MAX,
    ROPE_DIM,
    ROPE_ROWS_DYN,
    ROUTE_T_DYN,
    ROUTE_WIDTH,
    SIGNAL_PAD,
    STATE_CAPACITY,
    STATE_WIDTH,
    TOPK,
    T_DYN,
    WEIGHT_BANK_SIZE,
    WINDOW_CACHE_BYTES,
    decode_fwd,
)
from expert_context import CACHE_ROWS, CACHE_COLS, EXPERT_LAYERS, EXPERT_SLOTS

ROUTED_BANK_ROWS = EXPERT_SLOTS
from offload_decode_moe import clear_moe_signals
from generation_head import generation_head
from lm_head import LM_HEAD_TP_SIZE, VOCAB


@pl.jit(auto_scope=False)
def l2_decode_generate(
    engram_lookup: pl.Tensor[[2, T_DYN, ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[2, ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    engram_wkv_scale: pl.Tensor[[2 * (ENGRAM_K // 32), KV_OUT], pl.FP8E8M0],
    engram_weight: pl.Tensor[[2, HC_MULT, D], pl.FP32],
    x_hc: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    pre_mix: pl.InOut[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    dspark_target_hidden: pl.Out[pl.Tensor[[T_DYN, DSPARK_MAIN_IN], pl.BF16]],
    hc_attn_fn: pl.Tensor[[WEIGHT_BANK_SIZE * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[WEIGHT_BANK_SIZE * MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[WEIGHT_BANK_SIZE * D], pl.BF16],
    wq_a: pl.Tensor[[WEIGHT_BANK_SIZE, D, Q_LORA], pl.FP8E4M3FN],
    q_norm_weight: pl.Tensor[[WEIGHT_BANK_SIZE * Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[WEIGHT_BANK_SIZE, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wkv: pl.Tensor[[WEIGHT_BANK_SIZE, D, HEAD_DIM], pl.FP8E4M3FN],
    kv_norm_weight: pl.Tensor[[WEIGHT_BANK_SIZE * HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[WEIGHT_BANK_SIZE * LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[WEIGHT_BANK_SIZE * LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[WEIGHT_BANK_SIZE * LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[WEIGHT_BANK_SIZE, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    freqs_cos: pl.Tensor[[2, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    freqs_sin: pl.Tensor[[2, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, BLOCK_SIZE], pl.INT32],
    window_cache_pool: pl.InOut[pl.Tensor[[FWD_ORI_BLOCKS_DYN, BLOCK_SIZE, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    compressed_cache_pool: pl.InOut[pl.Tensor[[FWD_CMP_BLOCKS_DYN, COMPRESSED_POOL_UNIT_ROWS, 1, COMPRESSED_CACHE_BYTES], pl.UINT8]],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    index_cache_pool: pl.InOut[pl.Tensor[[FWD_INDEX_BLOCKS_DYN, COMPRESSED_POOL_UNIT_ROWS, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale_pool: pl.InOut[
        pl.Tensor[[FWD_INDEX_BLOCKS_DYN, COMPRESSED_POOL_UNIT_ROWS, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    c2a_index_block_table: pl.Tensor[[B_DYN, FWD_C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[B_DYN, FWD_C1A_TABLE_DYN], pl.INT32],
    c2a_compressor_wkv: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16, pl.NZ],
    c2a_compressor_wgate: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16, pl.NZ],
    c1a_compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache_pool: pl.InOut[pl.Tensor[[FWD_STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[KV_SOURCE_COUNT * HEAD_DIM], pl.BF16],
    c2a_compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    c1a_compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[INDEX_SOURCE_COUNT, HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_SOURCE_COUNT * INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[INDEX_SOURCE_COUNT, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_weights_proj: pl.Tensor[[INDEX_SOURCE_COUNT, D, INDEX_H], pl.BF16, pl.NZ],
    hc_ffn_fn: pl.Tensor[[WEIGHT_BANK_SIZE * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[WEIGHT_BANK_SIZE * MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[WEIGHT_BANK_SIZE * D], pl.BF16],
    gate_weight: pl.Tensor[[WEIGHT_BANK_SIZE * N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[WEIGHT_BANK_SIZE * N_EXPERTS], pl.FP32],
    routed_w1: pl.InOut[pl.Tensor[[ROUTED_BANK_ROWS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8]],
    routed_w2: pl.InOut[pl.Tensor[[ROUTED_BANK_ROWS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8]],
    routed_w3: pl.InOut[pl.Tensor[[ROUTED_BANK_ROWS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8]],
    shared_w1: pl.Tensor[[WEIGHT_BANK_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w2: pl.Tensor[[WEIGHT_BANK_SIZE, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w3: pl.Tensor[[WEIGHT_BANK_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    recv_meta: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, DISPATCH_PACKET_WIDTH], pl.INT8],
    recv_routes: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32],
    data_arrived: pld.DistributedTensor[
        [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32
    ],
    routed_output: pld.DistributedTensor[[ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[
        [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32
    ],
    moe_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    attention_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    wq_a_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (D // MX_GROUP), Q_LORA], pl.FP8E8M0],
    wq_b_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (Q_LORA // MX_GROUP), LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (D // MX_GROUP), HEAD_DIM], pl.FP8E8M0],
    wo_b_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (LOCAL_O_WIDTH // MX_GROUP), D], pl.FP8E8M0],
    index_wq_b_scale: pl.Tensor[[INDEX_SOURCE_COUNT * (Q_LORA // MX_GROUP), INDEX_H * INDEX_DIM], pl.FP8E8M0],
    routed_w1_scale: pl.InOut[pl.Tensor[[ROUTED_BANK_ROWS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0]],
    routed_w2_scale: pl.InOut[pl.Tensor[[ROUTED_BANK_ROWS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0]],
    routed_w3_scale: pl.InOut[pl.Tensor[[ROUTED_BANK_ROWS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0]],
    shared_w1_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w2_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    shared_w3_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[D, VOCAB // LM_HEAD_TP_SIZE], pl.BF16, pl.NZ],
    logit_rows: pl.Tensor[[16], pl.INT32],
    sampling_temperatures: pl.Tensor[[16], pl.FP32],
    sampling_top_ks: pl.Tensor[[16], pl.INT32],
    sampling_top_ps: pl.Tensor[[16], pl.FP32],
    sampling_seeds: pl.Tensor[[16], pl.INT32],
    sampling_positions: pl.Tensor[[16], pl.INT32],
    logits: pl.Out[pl.Tensor[[16, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[16, 8], pl.INT32]],
    head_hidden: pld.DistributedTensor[[LM_HEAD_TP_SIZE * 16, D], pl.BF16],
    head_hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    head_logits: pld.DistributedTensor[[16, VOCAB], pl.BF16],
    head_logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    expert_state: pl.InOut[pl.Tensor[[CACHE_ROWS, CACHE_COLS], pl.INT32]],
    expert_descriptors: pl.Tensor[[EXPERT_LAYERS, 6, 4], pl.INT64],
):
    result, final_pre_mix = decode_fwd(
        engram_lookup, engram_wkv, engram_wkv_scale, engram_weight,
        x_hc, pre_mix, dspark_target_hidden,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_weight,
        wq_a, q_norm_weight, wq_b,
        wkv, kv_norm_weight,
        attn_sink,
        wo_a, wo_a_scale, wo_b,
        freqs_cos, freqs_sin,
        window_slots, window_indices,
        window_cache_pool,
        compressed_cache_pool,
        token_to_req_indices, position_ids,
        index_cache_pool, index_cache_scale_pool,
        c2a_index_block_table, c1a_index_block_table,
        c2a_compressor_wkv, c2a_compressor_wgate, c1a_compressor_wkv,
        query_start_loc, state_block_table, state_cache_pool,
        compressor_norm_weight,
        c2a_compressed_slots, c1a_compressed_slots,
        index_wk, index_norm_weight, index_wq_b, index_weights_proj,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        ffn_norm_weight,
        gate_weight, correction_bias,
        routed_w1, routed_w2, routed_w3,
        shared_w1, shared_w2, shared_w3,
        recv_meta, recv_x, recv_routes,
        data_arrived, routed_output, combine_arrived,
        moe_num_tokens, attention_num_tokens, rank,
        wq_a_scale, wq_b_scale, wkv_scale, wo_b_scale,
        index_wq_b_scale,
        routed_w1_scale, routed_w2_scale, routed_w3_scale,
        shared_w1_scale, shared_w2_scale, shared_w3_scale,
        expert_state=expert_state, expert_descriptors=expert_descriptors,
        expert_slots=EXPERT_SLOTS, offload=True,
    )
    clear_moe_signals(result, data_arrived, combine_arrived, anchor_indices=(0, 0, 0))
    generation_head(
        result, final_pre_mix, final_norm_weight, lm_head_weight, logit_rows,
        sampling_temperatures, sampling_top_ks, sampling_top_ps, sampling_seeds, sampling_positions,
        logits, sampled_ids, head_hidden, head_hidden_done, head_logits, head_logits_done, rank,
    )
    return logits, sampled_ids


@pl.jit.host
def l3_decode_generate(
    engram_lookup: pl.Tensor[[EP_SIZE, 2, T_DYN, ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[EP_SIZE, 2, ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    engram_wkv_scale: pl.Tensor[[EP_SIZE, 2 * (ENGRAM_K // 32), KV_OUT], pl.FP8E8M0],
    engram_weight: pl.Tensor[[EP_SIZE, 2, HC_MULT, D], pl.FP32],
    x_hc: pl.InOut[pl.Tensor[[EP_SIZE, T_DYN, HC_MULT, D], pl.FP32]],
    pre_mix: pl.InOut[pl.Tensor[[EP_SIZE, T_DYN, HC_MULT], pl.FP32]],
    dspark_target_hidden: pl.Out[pl.Tensor[[EP_SIZE, T_DYN, DSPARK_MAIN_IN], pl.BF16]],
    hc_attn_fn: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE * MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE * D], pl.BF16],
    wq_a: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE * (D // MX_GROUP), Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE * Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE * (Q_LORA // MX_GROUP), LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE * (D // MX_GROUP), HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE * HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE * LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE * LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE * LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE * (LOCAL_O_WIDTH // MX_GROUP), D], pl.FP8E8M0],
    freqs_cos: pl.Tensor[[EP_SIZE, 2, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    freqs_sin: pl.Tensor[[EP_SIZE, 2, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[EP_SIZE, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[EP_SIZE, T_DYN, BLOCK_SIZE], pl.INT32],
    window_cache_pool: pl.InOut[pl.Tensor[[EP_SIZE, FWD_ORI_BLOCKS_DYN, BLOCK_SIZE, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    compressed_cache_pool: pl.InOut[pl.Tensor[[EP_SIZE, FWD_CMP_BLOCKS_DYN, COMPRESSED_POOL_UNIT_ROWS, 1, COMPRESSED_CACHE_BYTES], pl.UINT8]],
    token_to_req_indices: pl.Tensor[[EP_SIZE, T_DYN], pl.INT32],
    position_ids: pl.Tensor[[EP_SIZE, T_DYN], pl.INT32],
    index_cache_pool: pl.InOut[
        pl.Tensor[[EP_SIZE, FWD_INDEX_BLOCKS_DYN, COMPRESSED_POOL_UNIT_ROWS, 1, INDEX_DIM // 2], pl.UINT8]
    ],
    index_cache_scale_pool: pl.InOut[
        pl.Tensor[[EP_SIZE, FWD_INDEX_BLOCKS_DYN, COMPRESSED_POOL_UNIT_ROWS, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    c2a_index_block_table: pl.Tensor[[EP_SIZE, B_DYN, FWD_C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[EP_SIZE, B_DYN, FWD_C1A_TABLE_DYN], pl.INT32],
    c2a_compressor_wkv: pl.Tensor[[EP_SIZE, C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16],
    c2a_compressor_wgate: pl.Tensor[[EP_SIZE, C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16],
    c1a_compressor_wkv: pl.Tensor[[EP_SIZE, D, HEAD_DIM], pl.BF16],
    query_start_loc: pl.Tensor[[EP_SIZE, Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[EP_SIZE, B_DYN, 1], pl.INT32],
    state_cache_pool: pl.InOut[
        pl.Tensor[[EP_SIZE, FWD_STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]
    ],
    compressor_norm_weight: pl.Tensor[[EP_SIZE, KV_SOURCE_COUNT * HEAD_DIM], pl.BF16],
    c2a_compressed_slots: pl.Tensor[[EP_SIZE, T_DYN], pl.INT64],
    c1a_compressed_slots: pl.Tensor[[EP_SIZE, T_DYN], pl.INT64],
    index_wk: pl.Tensor[[EP_SIZE, INDEX_SOURCE_COUNT, HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[EP_SIZE, INDEX_SOURCE_COUNT * INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[EP_SIZE, INDEX_SOURCE_COUNT, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[
        [EP_SIZE, INDEX_SOURCE_COUNT * (Q_LORA // MX_GROUP), INDEX_H * INDEX_DIM], pl.FP8E8M0
    ],
    index_weights_proj: pl.Tensor[[EP_SIZE, INDEX_SOURCE_COUNT, D, INDEX_H], pl.BF16],
    hc_ffn_fn: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE * MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE * D], pl.BF16],
    gate_weight: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE * N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE * N_EXPERTS], pl.FP32],
    routed_w1: pl.InOut[pl.Tensor[[EP_SIZE, ROUTED_BANK_ROWS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8]],
    routed_w1_scale: pl.InOut[pl.Tensor[[EP_SIZE, ROUTED_BANK_ROWS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0]],
    routed_w2: pl.InOut[pl.Tensor[[EP_SIZE, ROUTED_BANK_ROWS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8]],
    routed_w2_scale: pl.InOut[pl.Tensor[[EP_SIZE, ROUTED_BANK_ROWS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0]],
    routed_w3: pl.InOut[pl.Tensor[[EP_SIZE, ROUTED_BANK_ROWS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8]],
    routed_w3_scale: pl.InOut[pl.Tensor[[EP_SIZE, ROUTED_BANK_ROWS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0]],
    shared_w1: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[EP_SIZE, WEIGHT_BANK_SIZE * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    moe_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    attention_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    final_norm_weight: pl.Tensor[[EP_SIZE, D], pl.BF16],
    lm_head_weight: pl.Tensor[[EP_SIZE, D, VOCAB // LM_HEAD_TP_SIZE], pl.BF16],
    logit_rows: pl.Tensor[[EP_SIZE, 16], pl.INT32],
    sampling_temperatures: pl.Tensor[[EP_SIZE, 16], pl.FP32],
    sampling_top_ks: pl.Tensor[[EP_SIZE, 16], pl.INT32],
    sampling_top_ps: pl.Tensor[[EP_SIZE, 16], pl.FP32],
    sampling_seeds: pl.Tensor[[EP_SIZE, 16], pl.INT32],
    sampling_positions: pl.Tensor[[EP_SIZE, 16], pl.INT32],
    logits: pl.Out[pl.Tensor[[EP_SIZE, 16, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[EP_SIZE, 16, 8], pl.INT32]],
    expert_state: pl.InOut[pl.Tensor[[EP_SIZE, CACHE_ROWS, CACHE_COLS], pl.INT32]],
    expert_descriptors: pl.Tensor[[EP_SIZE, EXPERT_LAYERS, 6, 4], pl.INT64],
):
    """Dispatch one decode and generation program per rank using the mapped expert banks."""
    x_hc.bind_dynamic(1, T_DYN)
    pre_mix.bind_dynamic(1, T_DYN)
    dspark_target_hidden.bind_dynamic(1, T_DYN)
    window_cache_pool.bind_dynamic(1, FWD_ORI_BLOCKS_DYN)
    compressed_cache_pool.bind_dynamic(1, FWD_CMP_BLOCKS_DYN)
    index_cache_pool.bind_dynamic(1, FWD_INDEX_BLOCKS_DYN)
    index_cache_scale_pool.bind_dynamic(1, FWD_INDEX_BLOCKS_DYN)
    state_cache_pool.bind_dynamic(1, FWD_STATE_BLOCKS_DYN)
    c2a_compressed_slots.bind_dynamic(1, T_DYN)
    c1a_compressed_slots.bind_dynamic(1, T_DYN)
    c2a_index_block_table.bind_dynamic(1, B_DYN)
    c2a_index_block_table.bind_dynamic(2, FWD_C2A_TABLE_DYN)
    c1a_index_block_table.bind_dynamic(1, B_DYN)
    c1a_index_block_table.bind_dynamic(2, FWD_C1A_TABLE_DYN)
    position_ids.bind_dynamic(1, T_DYN)
    freqs_cos.bind_dynamic(2, ROPE_ROWS_DYN)
    freqs_sin.bind_dynamic(2, ROPE_ROWS_DYN)
    recv_meta_buffer = pld.alloc_window_buffer([EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
    recv_x_buffer = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, DISPATCH_PACKET_WIDTH], dtype=pl.INT8)
    recv_routes_buffer = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
    data_arrived_buffer = pld.alloc_window_buffer(
        [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], dtype=pl.INT32
    )
    routed_output_buffer = pld.alloc_window_buffer([MOE_TOKENS * TOPK, D], dtype=pl.BF16)
    combine_arrived_buffer = pld.alloc_window_buffer(
        [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], dtype=pl.INT32
    )
    hidden_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE * 16, D], dtype=pl.BF16)
    hidden_done_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    logits_buffer = pld.alloc_window_buffer([16, VOCAB], dtype=pl.BF16)
    logits_done_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buffer, [EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buffer, [N_LOCAL_EXPERTS * RECV_MAX, DISPATCH_PACKET_WIDTH], dtype=pl.INT8)
        recv_routes = pld.window(
            recv_routes_buffer, [N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32
        )
        data_arrived = pld.window(
            data_arrived_buffer,
            [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD],
            dtype=pl.INT32,
        )
        routed_output = pld.window(routed_output_buffer, [MOE_TOKENS * TOPK, D], dtype=pl.BF16)
        combine_arrived = pld.window(
            combine_arrived_buffer,
            [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD],
            dtype=pl.INT32,
        )
        head_hidden = pld.window(hidden_buffer, [LM_HEAD_TP_SIZE * 16, D], dtype=pl.BF16)
        head_hidden_done = pld.window(hidden_done_buffer, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        head_logits = pld.window(logits_buffer, [16, VOCAB], dtype=pl.BF16)
        head_logits_done = pld.window(logits_done_buffer, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        l2_decode_generate(
            engram_lookup[rank], engram_wkv[rank], engram_wkv_scale[rank], engram_weight[rank],
            x_hc[rank], pre_mix[rank], dspark_target_hidden[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_weight[rank],
            wq_a[rank], q_norm_weight[rank], wq_b[rank],
            wkv[rank], kv_norm_weight[rank],
            attn_sink[rank],
            wo_a[rank], wo_a_scale[rank], wo_b[rank],
            freqs_cos[rank], freqs_sin[rank],
            window_slots[rank], window_indices[rank],
            window_cache_pool[rank],
            compressed_cache_pool[rank],
            token_to_req_indices[rank], position_ids[rank],
            index_cache_pool[rank], index_cache_scale_pool[rank],
            c2a_index_block_table[rank], c1a_index_block_table[rank],
            c2a_compressor_wkv[rank], c2a_compressor_wgate[rank], c1a_compressor_wkv[rank],
            query_start_loc[rank], state_block_table[rank], state_cache_pool[rank],
            compressor_norm_weight[rank],
            c2a_compressed_slots[rank], c1a_compressed_slots[rank],
            index_wk[rank], index_norm_weight[rank], index_wq_b[rank], index_weights_proj[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            ffn_norm_weight[rank],
            gate_weight[rank], correction_bias[rank],
            routed_w1[rank], routed_w2[rank], routed_w3[rank],
            shared_w1[rank], shared_w2[rank], shared_w3[rank],
            recv_meta, recv_x, recv_routes,
            data_arrived, routed_output, combine_arrived,
            moe_num_tokens, attention_num_tokens, rank,
            wq_a_scale[rank], wq_b_scale[rank], wkv_scale[rank], wo_b_scale[rank],
            index_wq_b_scale[rank],
            routed_w1_scale[rank], routed_w2_scale[rank], routed_w3_scale[rank],
            shared_w1_scale[rank], shared_w2_scale[rank], shared_w3_scale[rank],
            final_norm_weight[rank], lm_head_weight[rank], logit_rows[rank],
            sampling_temperatures[rank], sampling_top_ks[rank], sampling_top_ps[rank],
            sampling_seeds[rank], sampling_positions[rank], logits[rank], sampled_ids[rank],
            head_hidden, head_hidden_done, head_logits, head_logits_done,
            expert_state=expert_state[rank],
            expert_descriptors=expert_descriptors[rank], device=rank,
        )
    return logits, sampled_ids
