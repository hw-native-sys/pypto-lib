# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
# ci: no-sim
"""DeepSeek-V4.1 decode attention and MoE block."""


import pypto.language as pl
import pypto.language.distributed as pld
from config import (
    FLASH as M, MOE_TOKENS, B_DYN, T_DYN, LOCAL_T_DYN, Q_START_DYN, ROUTE_T_DYN, TABLE_DYN, ORI_BLOCKS_DYN,
    CMP_BLOCKS_DYN, INDEX_BLOCKS_DYN, INDEX_POOL_BLOCKS_DYN, STATE_BLOCKS_DYN, COMPRESSED_CACHE_GROUP,
    INDEX_CACHE_GROUP, STATE_CAPACITY, WINDOW_CACHE_GROUP,
)
from compressor import STATE_WIDTH
from decode_c1a_full import decode_c1a_full
from decode_c1a_reindex import decode_c1a_reindex
from decode_c1a_reuse import decode_c1a_reuse
from decode_c2a_full import decode_c2a_full
from decode_c2a_reuse import decode_c2a_reuse
from decode_swa import decode_swa
from expert_routed import MX_PACKED_LANE_COLS, MX_W1_PACKED_ROWS, MX_W2_PACKED_ROWS, MX_W3_PACKED_ROWS
from decode_moe import (
    moe, AUX_WIDTH, EP_SIZE, N_LOCAL_EXPERTS, RECV_MAX, ROUTE_WIDTH, SIGNAL_PAD, clear_moe_signals,
)


# model config
LOCAL_H = M.num_attention_heads
LOCAL_O_GROUPS = M.o_groups
LOCAL_O_WIDTH = M.o_groups * M.o_lora_rank
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
N_EXPERTS = M.n_routed_experts
TOPK = M.num_experts_per_tok
MOE_INTER = M.moe_intermediate_size
MX_GROUP = M.weight_block_size[0]


@pl.jit
def l2_decode_layer(
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
    window_cache_scale: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]],
    host_token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    c1a_compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.InOut[pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    topk_indices: pl.InOut[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    pool_block_ids: pl.InOut[pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.InOut[pl.Tensor[[T_DYN], pl.INT32]],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    attention_output: pl.Out[pl.Tensor[[LOCAL_T_DYN, D], pl.BF16]],
    attention_hidden: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    attention_pre_mix: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32]],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
    shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    next_pre_mix: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32]],
    x_mixed: pl.Out[pl.Tensor[[LOCAL_T_DYN, D], pl.BF16]],
    x_next: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    recv_meta: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], pl.UINT8],
    recv_weights: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], pl.FP32],
    recv_routes: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32],
    arrived: pld.DistributedTensor[[EP_SIZE, SIGNAL_PAD], pl.INT32],
    data_arrived: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32],
    routed_output: pld.DistributedTensor[[ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32],
    moe_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    attention_num_tokens: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Run one scheduled Attention sublayer followed by the EP MoE sublayer."""
    x_hc.bind_dynamic(0, LOCAL_T_DYN)
    incoming_pre_mix.bind_dynamic(0, LOCAL_T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    pool_block_ids.bind_dynamic(1, INDEX_POOL_BLOCKS_DYN)
    next_pre_mix.bind_dynamic(0, LOCAL_T_DYN)
    x_mixed.bind_dynamic(0, LOCAL_T_DYN)
    x_next.bind_dynamic(0, LOCAL_T_DYN)

    if layer_id < 2:
        decode_swa(
            x_hc, incoming_pre_mix,
            hc_attn_fn, hc_attn_scale, hc_attn_base,
            attn_norm_weight,
            wq_a, wq_a_scale, q_norm_weight,
            wq_b, wq_b_scale,
            wkv, wkv_scale, kv_norm_weight,
            attn_sink,
            wo_a, wo_a_scale, wo_b, wo_b_scale,
            rope_cos, rope_sin,
            window_slots, window_indices,
            window_cache, window_cache_scale,
            attention_output, attention_hidden, attention_pre_mix,
            attention_num_tokens,
        )
    elif layer_id < 20:
        if (layer_id - 2) % 6 == 0:
            decode_c2a_full(
                x_hc, incoming_pre_mix,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                attn_norm_weight,
                wq_a, wq_a_scale, q_norm_weight,
                wq_b, wq_b_scale,
                wkv, wkv_scale, kv_norm_weight,
                attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale,
                rope_cos, rope_sin,
                window_slots, window_indices,
                window_cache, window_cache_scale, compressed_cache, compressed_cache_scale,
                token_to_req_indices, compressed_lens,
                index_cache, index_cache_scale, index_block_table,
                position_ids,
                compressed_rope_cos, compressed_rope_sin,
                compressor_wkv,
                compressor_wgate,
                query_start_loc, state_block_table, state_cache,
                compressor_norm_weight,
                compressed_slots,
                index_wk, index_norm_weight,
                index_wq_b, index_wq_b_scale, index_weights_proj,
                topk_indices,
                attention_hidden, attention_pre_mix,
                attention_num_tokens,
            )
        else:
            decode_c2a_reuse(
                x_hc, incoming_pre_mix,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                attn_norm_weight,
                wq_a, wq_a_scale, q_norm_weight,
                wq_b, wq_b_scale,
                wkv, wkv_scale, kv_norm_weight,
                attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale,
                rope_cos, rope_sin,
                window_slots, window_indices,
                window_cache, window_cache_scale, compressed_cache, compressed_cache_scale,
                topk_indices,
                attention_hidden, attention_pre_mix,
                attention_num_tokens,
            )
    elif layer_id == 20:
        decode_c1a_full(
            x_hc, incoming_pre_mix,
            hc_attn_fn, hc_attn_scale, hc_attn_base,
            attn_norm_weight,
            wq_a, wq_a_scale, q_norm_weight,
            wq_b, wq_b_scale,
            wkv, wkv_scale, kv_norm_weight,
            attn_sink,
            wo_a, wo_a_scale, wo_b, wo_b_scale,
            rope_cos, rope_sin,
            window_slots, window_indices,
            window_cache, window_cache_scale, compressed_cache, compressed_cache_scale,
            host_token_to_req_indices, token_to_req_indices, compressed_lens,
            index_cache, index_cache_scale, index_block_table,
            compressed_rope_cos, compressed_rope_sin,
            c1a_compressor_wkv,
            compressor_norm_weight,
            compressed_slots,
            index_wk, index_norm_weight,
            index_wq_b, index_wq_b_scale, index_weights_proj,
            topk_indices, pool_block_ids, pool_block_counts,
            attention_hidden, attention_pre_mix,
            attention_num_tokens,
        )
    elif layer_id % 4 == 0:
        decode_c1a_reindex(
            x_hc, incoming_pre_mix,
            hc_attn_fn, hc_attn_scale, hc_attn_base,
            attn_norm_weight,
            wq_a, wq_a_scale, q_norm_weight,
            wq_b, wq_b_scale,
            wkv, wkv_scale, kv_norm_weight,
            attn_sink,
            wo_a, wo_a_scale, wo_b, wo_b_scale,
            rope_cos, rope_sin,
            window_slots, window_indices,
            window_cache, window_cache_scale, compressed_cache, compressed_cache_scale,
            token_to_req_indices, compressed_lens,
            index_cache, index_cache_scale, index_block_table,
            pool_block_ids, pool_block_counts,
            index_wq_b, index_wq_b_scale, index_weights_proj,
            topk_indices,
            attention_hidden, attention_pre_mix,
            attention_num_tokens,
        )
    else:
        decode_c1a_reuse(
            x_hc, incoming_pre_mix,
            hc_attn_fn, hc_attn_scale, hc_attn_base,
            attn_norm_weight,
            wq_a, wq_a_scale, q_norm_weight,
            wq_b, wq_b_scale,
            wkv, wkv_scale, kv_norm_weight,
            attn_sink,
            wo_a, wo_a_scale, wo_b, wo_b_scale,
            rope_cos, rope_sin,
            window_slots, window_indices,
            window_cache, window_cache_scale, compressed_cache, compressed_cache_scale,
            compressed_indices,
            attention_hidden, attention_pre_mix,
            attention_num_tokens,
        )

    local_moe_tokens = pl.read(moe_num_tokens, [rank])
    force_topk = pl.create_tensor([MOE_TOKENS, TOPK], dtype=pl.INT32)
    moe(
        attention_hidden, attention_pre_mix,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        ffn_norm_weight,
        gate_weight,
        correction_bias,
        routed_w1, routed_w1_scale,
        routed_w2, routed_w2_scale,
        routed_w3, routed_w3_scale,
        mxfp4_pair_lut,
        shared_w1, shared_w1_scale,
        shared_w2, shared_w2_scale,
        shared_w3, shared_w3_scale,
        next_pre_mix, x_mixed, x_next,
        recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
        arrived, data_arrived, routed_output, combine_arrived,
        local_moe_tokens, rank, moe_epoch,
        force_topk,
    )
    clear_moe_signals(x_next, arrived, data_arrived, combine_arrived, anchor_indices=(0, 0, 0))


@pl.jit.host
def l3_decode_layer(
    x_hc: pl.Tensor[[EP_SIZE, LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[EP_SIZE, LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[EP_SIZE, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[EP_SIZE, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[EP_SIZE, MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[EP_SIZE, D], pl.BF16],
    wq_a: pl.Tensor[[EP_SIZE, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[EP_SIZE, D // MX_GROUP, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[EP_SIZE, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[EP_SIZE, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[EP_SIZE, Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[EP_SIZE, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[EP_SIZE, D // MX_GROUP, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[EP_SIZE, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[EP_SIZE, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[EP_SIZE, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[EP_SIZE, LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[EP_SIZE, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[EP_SIZE, LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0],
    rope_cos: pl.Tensor[[EP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[EP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[EP_SIZE, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[EP_SIZE, T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[EP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[EP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[EP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[pl.Tensor[[EP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]],
    host_token_to_req_indices: pl.Tensor[[EP_SIZE, T_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[EP_SIZE, T_DYN], pl.INT32],
    position_ids: pl.Tensor[[EP_SIZE, T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[EP_SIZE, T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[EP_SIZE, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[EP_SIZE, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[EP_SIZE, B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[EP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[EP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[EP_SIZE, D, HEAD_DIM], pl.BF16],
    compressor_wgate: pl.Tensor[[EP_SIZE, D, HEAD_DIM], pl.BF16],
    c1a_compressor_wkv: pl.Tensor[[EP_SIZE, D, HEAD_DIM], pl.BF16],
    query_start_loc: pl.Tensor[[EP_SIZE, Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[EP_SIZE, B_DYN, 1], pl.INT32],
    state_cache: pl.InOut[pl.Tensor[[EP_SIZE, STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[EP_SIZE, HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[EP_SIZE, T_DYN], pl.INT64],
    index_wk: pl.Tensor[[EP_SIZE, HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[EP_SIZE, INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[EP_SIZE, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[EP_SIZE, Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[EP_SIZE, D, INDEX_H], pl.BF16],
    topk_indices: pl.InOut[pl.Tensor[[EP_SIZE, T_DYN, INDEX_TOPK], pl.INT32]],
    pool_block_ids: pl.InOut[pl.Tensor[[EP_SIZE, T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.InOut[pl.Tensor[[EP_SIZE, T_DYN], pl.INT32]],
    compressed_indices: pl.Tensor[[EP_SIZE, T_DYN, INDEX_TOPK], pl.INT32],
    attention_output: pl.Out[pl.Tensor[[EP_SIZE, LOCAL_T_DYN, D], pl.BF16]],
    attention_hidden: pl.Out[pl.Tensor[[EP_SIZE, LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    attention_pre_mix: pl.Out[pl.Tensor[[EP_SIZE, LOCAL_T_DYN, HC_MULT], pl.FP32]],
    hc_ffn_fn: pl.Tensor[[EP_SIZE, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[EP_SIZE, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[EP_SIZE, MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[EP_SIZE, D], pl.BF16],
    gate_weight: pl.Tensor[[EP_SIZE, N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[EP_SIZE, N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    mxfp4_pair_lut: pl.Tensor[[EP_SIZE, 2, 256], pl.INT16],
    shared_w1: pl.Tensor[[EP_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[EP_SIZE, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[EP_SIZE, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[EP_SIZE, MOE_INTER // MX_GROUP, D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[EP_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[EP_SIZE, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
    next_pre_mix: pl.Out[pl.Tensor[[EP_SIZE, LOCAL_T_DYN, HC_MULT], pl.FP32]],
    x_mixed: pl.Out[pl.Tensor[[EP_SIZE, LOCAL_T_DYN, D], pl.BF16]],
    x_next: pl.Out[pl.Tensor[[EP_SIZE, LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    moe_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    attention_num_tokens: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Launch one complete TP1 decode Block on the EP world."""
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(1, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(1, B_DYN)
    index_block_table.bind_dynamic(2, TABLE_DYN)
    state_cache.bind_dynamic(1, STATE_BLOCKS_DYN)
    pool_block_ids.bind_dynamic(2, INDEX_POOL_BLOCKS_DYN)
    recv_meta_buffer = pld.alloc_window_buffer([EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
    recv_x_buffer = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)
    recv_scale_buffer = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], dtype=pl.UINT8)
    recv_weights_buffer = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
    recv_routes_buffer = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
    arrived_buffer = pld.alloc_window_buffer([EP_SIZE, SIGNAL_PAD], dtype=pl.INT32)
    data_arrived_buffer = pld.alloc_window_buffer([EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], dtype=pl.INT32 )
    routed_output_buffer = pld.alloc_window_buffer([MOE_TOKENS * TOPK, D], dtype=pl.BF16)
    combine_arrived_buffer = pld.alloc_window_buffer([EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], dtype=pl.INT32 )
    for rank in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buffer, [EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buffer, [N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)
        recv_scale = pld.window(recv_scale_buffer, [N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], dtype=pl.UINT8)
        recv_weights = pld.window(recv_weights_buffer, [N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
        recv_routes = pld.window(recv_routes_buffer, [N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
        arrived = pld.window(arrived_buffer, [EP_SIZE, SIGNAL_PAD], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buffer, [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], dtype=pl.INT32)
        routed_output = pld.window(routed_output_buffer, [MOE_TOKENS * TOPK, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buffer, [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], dtype=pl.INT32)
        wq_a_scale_rank: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
        wq_b_scale_rank: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[rank]
        wkv_scale_rank: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
        wo_b_scale_rank: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
        index_wq_b_scale_rank: pl.Tensor[[Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = index_wq_b_scale[rank]
        routed_w1_scale_rank: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = routed_w1_scale[rank]
        routed_w2_scale_rank: pl.Tensor[[N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN] = routed_w2_scale[rank]
        routed_w3_scale_rank: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = routed_w3_scale[rank]
        shared_w1_scale_rank: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w1_scale[rank]
        shared_w2_scale_rank: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = shared_w2_scale[rank]
        shared_w3_scale_rank: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w3_scale[rank]
        l2_decode_layer(
            x_hc[rank], incoming_pre_mix[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_weight[rank],
            wq_a[rank],
            wq_a_scale_rank,
            q_norm_weight[rank],
            wq_b[rank], wq_b_scale_rank,
            wkv[rank],
            wkv_scale_rank,
            kv_norm_weight[rank],
            attn_sink[rank],
            wo_a[rank], wo_a_scale[rank], wo_b[rank], wo_b_scale_rank,
            rope_cos[rank], rope_sin[rank],
            window_slots[rank], window_indices[rank],
            window_cache[rank], window_cache_scale[rank], compressed_cache[rank], compressed_cache_scale[rank],
            host_token_to_req_indices[rank], token_to_req_indices[rank],
            position_ids[rank], compressed_lens[rank],
            index_cache[rank], index_cache_scale[rank], index_block_table[rank],
            compressed_rope_cos[rank], compressed_rope_sin[rank],
            compressor_wkv[rank],
            compressor_wgate[rank],
            c1a_compressor_wkv[rank],
            query_start_loc[rank], state_block_table[rank], state_cache[rank],
            compressor_norm_weight[rank],
            compressed_slots[rank],
            index_wk[rank], index_norm_weight[rank],
            index_wq_b[rank], index_wq_b_scale_rank, index_weights_proj[rank],
            topk_indices[rank], pool_block_ids[rank], pool_block_counts[rank], compressed_indices[rank],
            attention_output[rank], attention_hidden[rank], attention_pre_mix[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            ffn_norm_weight[rank],
            gate_weight[rank],
            correction_bias[rank],
            routed_w1[rank], routed_w1_scale_rank,
            routed_w2[rank], routed_w2_scale_rank,
            routed_w3[rank], routed_w3_scale_rank,
            mxfp4_pair_lut[rank],
            shared_w1[rank], shared_w1_scale_rank,
            shared_w2[rank], shared_w2_scale_rank,
            shared_w3[rank], shared_w3_scale_rank,
            next_pre_mix[rank], x_mixed[rank], x_next[rank],
            recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
            arrived, data_arrived, routed_output, combine_arrived,
            moe_num_tokens, layer_id, rank, attention_num_tokens, moe_epoch,
            device=rank,
        )


import inspect
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping
from quant import (
    decode_e8m0, pack_mx_b_scale, unpack_mx_b_scale, quantize_mxfp4_cache, quantize_mxfp4_weight,
    quantize_mxfp8_cache, quantize_grouped_mxfp8_weight, dequantize_mxfp4_cache,
)
from config import DECODE_MAX_TOKENS
from decode_c1a_full import golden_c1a
from decode_c2a_full import golden_c2a
from decode_swa import golden_swa
from gate import golden_gate_reference as gate
from hc import golden_mhc_mixes, golden_hc_post, golden_hc_collapse
from rmsnorm import golden_rms_norm
from sparse_attention import golden_swa_attention


class DecodeLayerKind(IntEnum):
    """Attention branch used by one decode Block."""

    SWA = 0
    C2A_FULL = 1
    C2A_REUSE = 2
    C1A_FULL = 3
    C1A_REINDEX = 4
    C1A_REUSE = 5


REPRESENTATIVE_LAYER_IDS = {
    DecodeLayerKind.SWA: 0,
    DecodeLayerKind.C2A_FULL: 2,
    DecodeLayerKind.C2A_REUSE: 3,
    DecodeLayerKind.C1A_FULL: 20,
    DecodeLayerKind.C1A_REINDEX: 24,
    DecodeLayerKind.C1A_REUSE: 21,
}


def decode_layer_kind(layer_id: int) -> DecodeLayerKind:
    """Map the checkpoint-backed layer schedule to one decode branch."""
    layer = M.layer_config(layer_id)
    if layer.mode == 'swa':
        return DecodeLayerKind.SWA
    if layer.compression_ratio == 2 and layer.mode == 'full':
        return DecodeLayerKind.C2A_FULL
    if layer.compression_ratio == 2 and layer.mode == 'reuse':
        return DecodeLayerKind.C2A_REUSE
    if layer.compression_ratio == 1 and layer.mode == 'full':
        return DecodeLayerKind.C1A_FULL
    if layer.compression_ratio == 1 and layer.mode == 'reindex':
        return DecodeLayerKind.C1A_REINDEX
    if layer.compression_ratio == 1 and layer.mode == 'reuse':
        return DecodeLayerKind.C1A_REUSE
    raise ValueError(f"unsupported decode layer {layer_id}: ratio={layer.compression_ratio}, mode={layer.mode}" )


@dataclass(frozen=True)
class DecodeLayerGoldenResult:
    """Block outputs, intermediate boundaries, and attention state updates."""

    output: torch.Tensor
    next_pre_mix: torch.Tensor
    attention_input: torch.Tensor
    attention_output: torch.Tensor
    attention_hidden: torch.Tensor
    ffn_input: torch.Tensor
    ffn_output: torch.Tensor
    attention: dict


def _select_inputs(function, values: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    selected = {}
    for name, parameter in inspect.signature(function).parameters.items():
        if name in overrides:
            selected[name] = overrides[name]
        elif name in values:
            selected[name] = values[name]
        elif parameter.default is not inspect.Parameter.empty:
            continue
        else:
            raise KeyError(f"missing {function.__name__} input {name}")
    return selected


def _golden_expert(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    gate_value = F.linear(x.float(), w1.float()).clamp(max=M.swiglu_limit)
    up_value = F.linear(x.float(), w3.float()).clamp(-M.swiglu_limit, M.swiglu_limit)
    return F.linear(F.silu(gate_value) * up_value, w2.float())


def _dequantize_mxfp8(weight: torch.Tensor, logical_scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Decode an input-major MXFP8 matrix using logical ``[K/32, N]`` scales."""
    scales = decode_e8m0(logical_scale_e8m0).repeat_interleave(MX_GROUP, dim=-2)
    return weight.float() * scales


def golden_decode_moe(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    routed_w1: torch.Tensor,
    routed_w1_scale: torch.Tensor,
    routed_w2: torch.Tensor,
    routed_w2_scale: torch.Tensor,
    routed_w3: torch.Tensor,
    routed_w3_scale: torch.Tensor,
    shared_w1: torch.Tensor,
    shared_w1_scale: torch.Tensor,
    shared_w2: torch.Tensor,
    shared_w2_scale: torch.Tensor,
    shared_w3: torch.Tensor,
    shared_w3_scale: torch.Tensor,
    num_tokens: int | None = None,
) -> torch.Tensor:
    """Evaluate one rank's dense MoE math without the EP transport harness."""
    shape = x.shape
    normalized = golden_rms_norm(x.reshape(-1, shape[-1]), norm_weight)
    active_tokens = normalized.shape[0] if num_tokens is None else num_tokens
    if not 0 <= active_tokens <= normalized.shape[0]:
        raise ValueError("num_tokens must fit the local decode capacity")
    normalized = normalized[:active_tokens]
    routed_w1 = dequantize_mxfp4_cache(routed_w1, routed_w1_scale, 32, "e8m0")
    routed_w2 = dequantize_mxfp4_cache(routed_w2, routed_w2_scale, 32, "e8m0")
    routed_w3 = dequantize_mxfp4_cache(routed_w3, routed_w3_scale, 32, "e8m0")
    shared_w1 = _dequantize_mxfp8(shared_w1, unpack_mx_b_scale(shared_w1_scale)).transpose(-2, -1)
    shared_w2 = _dequantize_mxfp8(shared_w2, unpack_mx_b_scale(shared_w2_scale)).transpose(-2, -1)
    shared_w3 = _dequantize_mxfp8(shared_w3, unpack_mx_b_scale(shared_w3_scale)).transpose(-2, -1)
    route_weights, expert_indices = gate(normalized, gate_weight, correction_bias)
    output = _golden_expert(normalized, shared_w1, shared_w2, shared_w3)
    for expert_id in range(routed_w1.shape[0]):
        token_rows, route_columns = torch.where(expert_indices == expert_id)
        if token_rows.numel() == 0:
            continue
        routed = _golden_expert(
            normalized[token_rows],
            routed_w1[expert_id], routed_w2[expert_id], routed_w3[expert_id],
        )
        output[token_rows] += routed * route_weights[token_rows, route_columns].unsqueeze(-1)
    result = x.reshape(-1, shape[-1]).clone()
    result[:active_tokens] = output.to(x.dtype)
    return result.reshape(shape)


def golden_decode_layer(
    layer_id: int,
    x_hc: torch.Tensor,
    incoming_pre_mix: torch.Tensor,
    hc_attn_fn: torch.Tensor,
    hc_attn_scale: torch.Tensor,
    hc_attn_base: torch.Tensor,
    attn_norm_weight: torch.Tensor,
    hc_ffn_fn: torch.Tensor,
    hc_ffn_scale: torch.Tensor,
    hc_ffn_base: torch.Tensor,
    ffn_norm_weight: torch.Tensor,
    attention_inputs: Mapping[str, Any],
    moe_inputs: Mapping[str, Any],
    num_tokens: int | None = None,
) -> DecodeLayerGoldenResult:
    """Evaluate the mHC-Attention-mHC-MoE-mHC Block order."""
    if num_tokens is not None and num_tokens != x_hc.shape[0]:
        raise ValueError("Block golden currently requires all capacity rows to be active")
    layer = M.layer_config(layer_id)
    attention_tensors = {
        name: value.clone() for name, value in attention_inputs.items()
        if isinstance(value, torch.Tensor)
    }
    attention_tensors.update({
        "x_hc": x_hc, "pre_mix": incoming_pre_mix,
        "hc_attn_fn": hc_attn_fn, "hc_attn_scale": hc_attn_scale,
        "hc_attn_base": hc_attn_base, "attn_norm_weight": attn_norm_weight,
        "output": torch.empty_like(x_hc),
        "next_pre_mix": torch.empty_like(incoming_pre_mix),
    })
    if layer.mode == "swa":
        attention_tensors.update({
            "incoming_pre_mix": attention_tensors["pre_mix"],
            "attention_hidden": attention_tensors["output"],
            "attention_pre_mix": attention_tensors["next_pre_mix"],
            "attention_output": torch.empty_like(x_hc[:, 0], dtype=torch.bfloat16),
            "num_tokens": x_hc.shape[0],
        })
        golden_swa(attention_tensors, reference=golden_swa_attention)
        for name in (
            "compressed_cache", "compressed_cache_scale", "index_cache", "index_cache_scale", "state_cache",
            "topk_indices", "candidate_mask", "pool_block_ids", "pool_block_counts",
        ):
            attention_tensors[name] = None
    else:
        if layer.compression_ratio == 1 and layer.mode == "full":
            attention_tensors["device_token_to_req_indices"] = attention_tensors["token_to_req_indices"]
        golden = golden_c1a if layer.compression_ratio == 1 else golden_c2a
        golden(attention_tensors, mode=layer.mode)
    attention = {
        name: attention_tensors[name] if isinstance(attention_tensors[name], torch.Tensor) else None
        for name in (
            "window_cache", "window_cache_scale", "compressed_cache", "compressed_cache_scale",
            "index_cache", "index_cache_scale", "state_cache", "topk_indices", "candidate_mask",
            "pool_block_ids", "pool_block_counts",
        )
    }
    attention["output"] = attention_tensors["attention_output"]
    attention_hidden = attention_tensors["output"]
    attn_pre = attention_tensors["next_pre_mix"]
    attention_input = attention_tensors["attention_input"]
    next_pre_mix, ffn_post, ffn_residual = golden_mhc_mixes(attention_hidden, hc_ffn_fn, hc_ffn_scale, hc_ffn_base)
    ffn_input = golden_hc_collapse(attention_hidden, attn_pre).to(torch.bfloat16)
    moe_overrides = {"x": ffn_input, "norm_weight": ffn_norm_weight}
    if num_tokens is not None:
        moe_overrides["num_tokens"] = num_tokens
    moe_kwargs = _select_inputs(golden_decode_moe, moe_inputs, moe_overrides)
    ffn_output = golden_decode_moe(**moe_kwargs)
    output = golden_hc_post(ffn_output, attention_hidden, ffn_post, ffn_residual)
    return DecodeLayerGoldenResult(
        output=output, next_pre_mix=next_pre_mix,
        attention_input=attention_input, attention_output=attention["output"], attention_hidden=attention_hidden,
        ffn_input=ffn_input, ffn_output=ffn_output,
        attention=attention,
    )


def _torch_dtype(dtype):
    values = {
        pl.BF16: torch.bfloat16,
        pl.FP32: torch.float32,
        pl.FP8E4M3FN: torch.float8_e4m3fn,
        pl.FP8E8M0: torch.float8_e8m0fnu,
        pl.INT16: torch.int16,
        pl.INT32: torch.int32,
        pl.INT64: torch.int64,
        pl.UINT8: torch.uint8,
    }
    return values[dtype]


def _fixture_extent(dim, global_tokens):
    if isinstance(dim, int):
        return dim
    name = repr(dim)
    if "LOCAL_T_DYN" in name:
        return MOE_TOKENS
    if "T_DYN" in name:
        return global_tokens
    if "Q_START_DYN" in name:
        return 2
    if "INDEX_POOL_BLOCKS_DYN" in name:
        # Minimum aligned pool extent for the SWA fixture.
        return 32
    return 1


def build_tensor_specs(layer_id=0, seed=0):
    """Build an L3 SWA decode block fixture with mHC and EP MoE."""
    if decode_layer_kind(layer_id) != DecodeLayerKind.SWA:
        raise ValueError("the complete-layer fixture currently supports SWA layers 0 and 1")

    from dataclasses import replace
    from types import SimpleNamespace

    from golden import ScalarSpec, TensorSpec
    import decode_swa
    from decode_moe import build_moe_specs

    global_tokens = DECODE_MAX_TOKENS
    args = SimpleNamespace(
        tokens=global_tokens, active_tokens=global_tokens, local_tokens=MOE_TOKENS,
        requests=min(global_tokens, 8), dp=1,
        seed=seed, epochs=1, bench=False, pages=1, case="mixed",
    )
    rank_specs = [
        {spec.name: spec for spec in decode_swa.build_swa_specs(SimpleNamespace(**dict(vars(args), seed=seed + rank)))}
        for rank in range(EP_SIZE)
    ]
    by_name = {}
    attention_outputs = {"attention_output", "attention_hidden", "attention_pre_mix", "num_tokens"}
    boundary_inputs = {"x_hc", "incoming_pre_mix", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight"}
    for name, spec in rank_specs[0].items():
        if not isinstance(spec, TensorSpec) or name in attention_outputs:
            continue
        sources = [spec] * EP_SIZE if name in boundary_inputs else [rank[name] for rank in rank_specs]

        def grouped_init(sources=sources):
            values = [source.create_tensor().clone() for source in sources]
            return torch.stack([value.view(torch.uint8) for value in values]).view(values[0].dtype)

        by_name[name] = replace(spec, shape=[EP_SIZE, *spec.shape], init_value=grouped_init, resident="stacked")

    moe_aliases = {
        "norm_weight": "ffn_norm_weight",
        "num_tokens": "moe_num_tokens",
    }
    for spec in build_moe_specs(_fixture_moe_counts()):
        if not isinstance(spec, TensorSpec) or spec.name in {"x_hc", "pre_mix"}:
            continue
        name = moe_aliases.get(spec.name, spec.name)
        if name in l3_decode_layer.param_names:
            by_name[name] = spec if name == spec.name else replace(spec, name=name)

    mhc_generator = torch.Generator().manual_seed(seed + 73)
    hc_ffn_fn = torch.randn(MIX_HC, HC_DIM, generator=mhc_generator) * 0.0635
    hc_ffn_scale = torch.tensor([0.11334, 0.035901, 0.058183])
    hc_ffn_base = torch.tensor([
        2.4153, -2.0252, -2.0019, -2.1947,
        -1.5430, -3.0228, -6.8248, 0.5894,
        2.1916, -7.2132, -3.0938, -2.1119,
        -3.0161, 3.3293, -3.2224, -4.0226,
        -2.0428, -3.3478, 3.0893, -3.4166,
        -1.8144, -3.8147, -3.1307, 1.7862,
    ])
    calibrated_mhc = {
        "hc_ffn_fn": hc_ffn_fn,
        "hc_ffn_scale": hc_ffn_scale,
        "hc_ffn_base": hc_ffn_base,
    }
    for name, value in calibrated_mhc.items():
        stacked = value.unsqueeze(0).expand(EP_SIZE, *value.shape).contiguous()
        by_name[name] = replace(by_name[name], init_value=lambda stacked=stacked: stacked.clone())

    signature = inspect.signature(l3_decode_layer._func)
    outputs = {
        "attention_output", "attention_hidden", "attention_pre_mix",
        "next_pre_mix", "x_mixed", "x_next",
    }
    scalar_values = {
        "layer_id": layer_id,
        "attention_num_tokens": global_tokens,
        "moe_epoch": 1,
    }
    specs = []
    for name in l3_decode_layer.param_names:
        if name in scalar_values:
            specs.append(ScalarSpec(name, torch.int32, scalar_values[name]))
            continue
        if name in by_name:
            specs.append(by_name[name])
            continue
        annotation = signature.parameters[name].annotation
        shape = [_fixture_extent(dim, global_tokens) for dim in annotation.shape]
        dtype = _torch_dtype(annotation.dtype)
        if name in outputs:
            specs.append(TensorSpec(name, shape, dtype))
        else:
            specs.append(
                TensorSpec(
                    name, shape, dtype,
                    init_value=lambda shape=shape, dtype=dtype: torch.zeros(shape, dtype=dtype),
                )
            )
    device_requests = next(spec for spec in specs if spec.name == "token_to_req_indices")
    for index, spec in enumerate(specs):
        if spec.name == "host_token_to_req_indices":
            specs[index] = replace(device_requests, name="host_token_to_req_indices", resident=None)
    return specs


def _fixture_moe_counts():
    """Count valid decode rows in each rank's padded physical slab."""
    return torch.tensor(
        [max(0, min(MOE_TOKENS, DECODE_MAX_TOKENS)) for rank in range(EP_SIZE)],
        dtype=torch.int32,
    )


def golden_l3_decode_layer(tensors):
    """Run the local Attention golden, then the distributed MoE golden."""
    import decode_swa
    from decode_moe import golden_moe

    global_tokens = int(tensors["attention_num_tokens"])
    attention_hidden = tensors["attention_hidden"]
    attention_pre_mix = tensors["attention_pre_mix"]
    for group_base in range(EP_SIZE):
        group = {}
        for name, value in tensors.items():
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == EP_SIZE:
                group[name] = value[group_base]
            else:
                group[name] = value
        group["attention_output"] = tensors["attention_output"][group_base]
        group["attention_hidden"] = attention_hidden[group_base]
        group["attention_pre_mix"] = attention_pre_mix[group_base]
        group["num_tokens"] = global_tokens
        decode_swa.golden_swa(group)

    moe_tensors = dict(tensors)
    moe_tensors["x_hc"] = attention_hidden
    moe_tensors["pre_mix"] = attention_pre_mix
    moe_tensors["norm_weight"] = tensors["ffn_norm_weight"]
    moe_tensors["num_tokens"] = tensors["moe_num_tokens"]
    golden_moe(moe_tensors)


def _compare_active_rows_per_rank(compare, counts):
    """Apply a composed-layer precision budget independently to every EP rank."""
    def compare_active(actual, expected, **kwargs):
        for rank, active in enumerate(counts.tolist()):
            if active == 0:
                continue
            passed, detail = compare(actual[rank : rank + 1, :active], expected[rank : rank + 1, :active], **kwargs)
            if not passed:
                return False, f"rank {rank} active rows [:{active}] failed\n{detail}"
        return True, "every EP rank's active rows pass"

    return compare_active


def _compare_moe_from_actual_attention(counts):
    """Validate MoE against the actual, already-validated Attention boundary."""
    from decode_moe import _local_mhc_compare, golden_moe

    compare_moe = _local_mhc_compare(counts)

    def compare(actual, expected, *, actual_outputs, inputs, **kwargs):
        actual_f = actual.float()
        expected_f = expected.float()
        diff = actual_f - expected_f
        rel_l2 = diff.norm() / expected_f.norm().clamp_min(1e-12)
        print(f"[PRECISION] composed x_next rel_l2={rel_l2.item():.8g} " f"max_abs={diff.abs().max().item():.8g}" )

        conditioned = dict(inputs)
        conditioned["x_hc"] = actual_outputs["attention_hidden"]
        conditioned["pre_mix"] = actual_outputs["attention_pre_mix"]
        conditioned["norm_weight"] = inputs["ffn_norm_weight"]
        conditioned["num_tokens"] = inputs["moe_num_tokens"]
        conditioned["next_pre_mix"] = torch.empty_like(actual_outputs["next_pre_mix"])
        conditioned["x_mixed"] = torch.empty_like(actual_outputs["x_mixed"])
        conditioned["x_next"] = torch.empty_like(actual)
        golden_moe(conditioned)
        return compare_moe(
            actual,
            conditioned["x_next"],
            actual_outputs=actual_outputs,
            inputs={"num_tokens": inputs["moe_num_tokens"]},
            **kwargs,
        )

    return compare


def _compare_next_pre_mix_from_actual_attention(counts):
    """Check mHC coefficients independently of accepted Attention error."""
    from golden import ratio_allclose
    from hc import golden_mhc_mixes

    compare_mixes = _compare_active_rows_per_rank(ratio_allclose(atol=2.5e-5, rtol=5e-3, max_error_ratio=0.03), counts)

    def compare(actual, expected, *, actual_outputs, inputs, **kwargs):
        conditioned = torch.zeros_like(expected)
        for rank, active in enumerate(counts.tolist()):
            if active:
                pre, _, _ = golden_mhc_mixes(
                    actual_outputs["attention_hidden"][rank, :active],
                    inputs["hc_ffn_fn"][rank], inputs["hc_ffn_scale"][rank],
                    inputs["hc_ffn_base"][rank],
                )
                conditioned[rank, :active] = pre
        return compare_mixes(actual, conditioned, actual_outputs=actual_outputs, inputs=inputs, **kwargs)

    return compare


def _mxfp8_weight(input_dim: int, output_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.randn(input_dim, output_dim).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    scale = pack_mx_b_scale(torch.full((input_dim // 32, output_dim), 127, dtype=torch.uint8))
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is not None:
        scale = scale.view(e8m0_dtype)
    return weight, scale


def _attention_values() -> dict[str, torch.Tensor | None]:
    """Build small deterministic inputs for all attention golden modes."""
    from quant import pack_bf16_nz, pack_fp8_nz

    torch.manual_seed(7)
    wq_a, wq_a_scale = _mxfp8_weight(64, 64)
    wq_b, wq_b_scale = _mxfp8_weight(64, 128)
    wkv, wkv_scale = _mxfp8_weight(64, 64)
    wo_b, wo_b_scale = _mxfp8_weight(64, 64)
    wq_a = pack_fp8_nz(wq_a)
    wq_b = pack_fp8_nz(wq_b)
    wkv = pack_fp8_nz(wkv)
    wo_b = pack_fp8_nz(wo_b)
    index_wq_b, index_wq_b_scale = _mxfp8_weight(64, 16 * 32)
    index_wq_b = pack_fp8_nz(index_wq_b)
    window_cache, window_cache_scale = quantize_mxfp8_cache(torch.zeros(1, 128, 1, 64))
    compressed_cache, compressed_cache_scale = quantize_mxfp4_cache(
        torch.randn(1, 128, 1, 64), group_size=16, scale_format="bf16"
    )
    index_cache, index_cache_scale = quantize_mxfp4_cache(
        torch.randn(1, 128, 1, 32), group_size=32, scale_format="e8m0"
    )
    values = {
        "x": torch.randn(2, 64, dtype=torch.bfloat16),
        "wq_a": wq_a,
        "wq_a_scale": wq_a_scale,
        "q_norm_weight": torch.ones(64, dtype=torch.bfloat16),
        "wq_b": wq_b,
        "wq_b_scale": wq_b_scale,
        "wkv": wkv,
        "wkv_scale": wkv_scale,
        "kv_norm_weight": torch.ones(64, dtype=torch.bfloat16),
        "attn_sink": torch.randn(2),
        "wo_a": torch.randn(2, 32, 64, dtype=torch.bfloat16),
        "wo_b": wo_b,
        "wo_b_scale": wo_b_scale,
        "rope_cos": torch.ones(2, 1),
        "rope_sin": torch.zeros(2, 1),
        "window_slots": torch.tensor([0, 1]),
        "window_indices": torch.tensor([[0, -1], [0, 1]], dtype=torch.int32),
        "window_cache": window_cache,
        "window_cache_scale": window_cache_scale,
        "compressed_cache": compressed_cache,
        "compressed_cache_scale": compressed_cache_scale,
        "compressed_indices": torch.tensor([[-1, -1], [0, -1]], dtype=torch.int32),
        "host_token_to_req_indices": torch.tensor([0, 0], dtype=torch.int32),
        "token_to_req_indices": torch.tensor([0, 0], dtype=torch.int32),
        "position_ids": torch.tensor([0, 1], dtype=torch.int32),
        "compressed_lens": torch.tensor([0, 1], dtype=torch.int32),
        "compressed_rope_cos": torch.ones(2, 1),
        "compressed_rope_sin": torch.zeros(2, 1),
        "index_cache": index_cache,
        "index_cache_scale": index_cache_scale,
        "index_block_table": torch.tensor([[0]], dtype=torch.int32),
        "pool_block_ids": torch.tensor([[-1] * 32, [0] + [-1] * 31], dtype=torch.int32),
        "pool_block_counts": torch.tensor([0, 1], dtype=torch.int32),
        "compressor_wkv": torch.randn(64, 64),
        "compressor_wgate": torch.randn(64, 64),
        "query_start_loc": torch.tensor([0, 2], dtype=torch.int32),
        "state_block_table": torch.tensor([[1]], dtype=torch.int32),
        "state_cache": torch.zeros(2, 4, 128),
        "compressor_norm_weight": torch.ones(64, dtype=torch.bfloat16),
        "compressed_slots": torch.tensor([-1, 0]),
        "index_wk": pack_bf16_nz(torch.randn(64, 32, dtype=torch.bfloat16)),
        "index_norm_weight": torch.ones(32, dtype=torch.bfloat16),
        "index_wq_b": index_wq_b,
        "index_wq_b_scale": index_wq_b_scale,
        "index_weights_proj": pack_bf16_nz(torch.randn(64, 16, dtype=torch.bfloat16)),
    }
    values["wo_a"], values["wo_a_scale"] = quantize_grouped_mxfp8_weight(values["wo_a"])
    return values


def make_decode_layer_golden_inputs(layer_id: int) -> dict:
    """Build a deterministic small Block fixture for one representative layer."""
    torch.manual_seed(17 + layer_id)
    from quant import pack_bf16_nz

    attention = _attention_values()
    attention["compressor_wkv"] = pack_bf16_nz(attention["compressor_wkv"].to(torch.bfloat16))
    attention["compressor_wgate"] = pack_bf16_nz(attention["compressor_wgate"].to(torch.bfloat16))
    if layer_id >= 20:
        attention["compressed_lens"] = torch.tensor([1, 2], dtype=torch.int32)
        attention["compressed_slots"] = torch.tensor([0, 1], dtype=torch.int64)
    if layer_id in (3, 21):
        attention["compressed_indices"] = torch.tensor([[0, -1], [0, 1]], dtype=torch.int32)

    routed_w1, routed_w1_scale = quantize_mxfp4_weight(torch.randn(8, 64, 64))
    routed_w2, routed_w2_scale = quantize_mxfp4_weight(torch.randn(8, 64, 64))
    routed_w3, routed_w3_scale = quantize_mxfp4_weight(torch.randn(8, 64, 64))
    shared_scale = pack_mx_b_scale(torch.full((2, 64), 127, dtype=torch.uint8))
    moe = {
        "gate_weight": torch.randn(8, 64),
        "correction_bias": torch.randn(8),
        "routed_w1": routed_w1,
        "routed_w1_scale": routed_w1_scale,
        "routed_w2": routed_w2,
        "routed_w2_scale": routed_w2_scale,
        "routed_w3": routed_w3,
        "routed_w3_scale": routed_w3_scale,
        "shared_w1": torch.randn(64, 64).to(torch.float8_e4m3fn),
        "shared_w1_scale": shared_scale,
        "shared_w2": torch.randn(64, 64).to(torch.float8_e4m3fn),
        "shared_w2_scale": shared_scale,
        "shared_w3": torch.randn(64, 64).to(torch.float8_e4m3fn),
        "shared_w3_scale": shared_scale,
    }
    incoming_pre_mix = torch.zeros(2, 4)
    incoming_pre_mix[:, 0] = 1.0
    return {
        "layer_id": layer_id,
        "x_hc": torch.randn(2, 4, 64),
        "incoming_pre_mix": incoming_pre_mix,
        "hc_attn_fn": torch.randn(24, 256) / 16,
        "hc_attn_scale": torch.randn(3),
        "hc_attn_base": torch.randn(24),
        "attn_norm_weight": torch.ones(64, dtype=torch.bfloat16),
        "hc_ffn_fn": torch.randn(24, 256) / 16,
        "hc_ffn_scale": torch.randn(3),
        "hc_ffn_base": torch.randn(24),
        "ffn_norm_weight": torch.ones(64, dtype=torch.bfloat16),
        "attention_inputs": attention,
        "moe_inputs": moe,
    }


if __name__ == "__main__":
    import argparse
    from config import SUPPORTED_EP_SIZES

    from golden import mapped_pool_ratio_allclose, ratio_allclose, ratio_reldiff, run
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("--ep", type=int, default=EP_SIZE, choices=list(SUPPORTED_EP_SIZES))
    parser.add_argument("-d", "--device", default=",".join(str(rank) for rank in range(EP_SIZE)))
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--runtime-dir", default=None)
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data", default=None)
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--cpu-golden", action="store_true")
    args = parser.parse_args()
    if args.cpu_golden:
        for layer_id in REPRESENTATIVE_LAYER_IDS.values():
            result = golden_decode_layer(**make_decode_layer_golden_inputs(layer_id))
            if result.output.dtype is not torch.float32:
                raise RuntimeError(f"layer {layer_id} produced {result.output.dtype}, expected torch.float32")
            if not bool(torch.isfinite(result.output).all()):
                raise RuntimeError(f"layer {layer_id} produced non-finite output")
            print(
                f"[GOLDEN] PASS decode_layer layer={layer_id} "
                f"output={tuple(result.output.shape)} next_pre_mix={tuple(result.next_pre_mix.shape)}"
            )

        first_inputs = make_decode_layer_golden_inputs(0)
        first = golden_decode_layer(**first_inputs)
        tokens = first.output.shape[0]
        # Restore the next layer's residual stream and delayed pre-mix.
        second_inputs = make_decode_layer_golden_inputs(2)
        second_inputs["x_hc"] = first.output
        second_inputs["incoming_pre_mix"] = first.next_pre_mix
        second = golden_decode_layer(**second_inputs)

        if second.output.shape != first.output.shape or second.next_pre_mix.shape != first.next_pre_mix.shape:
            raise RuntimeError("consecutive Decode layers changed the residual boundary shape")
        if not bool(torch.isfinite(second.output).all()):
            raise RuntimeError("the second Decode layer produced non-finite hidden state")
        print(
            f"[GOLDEN] PASS decode_layer chain layers=(0,2) TP=1 "
            f"output={tuple(second.output.shape)} tokens={tokens}"
        )
        raise SystemExit(0)
    if args.ep != EP_SIZE:
        parser.error("--ep is import-time configuration; pass it on the Python command line")
    devices = [int(device) for device in args.device.split(",") if device]
    if len(devices) != EP_SIZE:
        parser.error(f"need exactly {EP_SIZE} devices, got {devices}")
    specs = build_tensor_specs(args.layer_id, args.seed)
    counts = _fixture_moe_counts()
    exact = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
    result = run(
        fn=l3_decode_layer,
        specs=specs,
        golden_fn=golden_l3_decode_layer,
        compile_only=args.compile_only,
        save_data=args.save_data,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        config=dict(
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            dump_passes=args.dump_passes,
            enable_chip_swimlane=args.enable_chip_swimlane,
            distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
            # Attention and the packed EP MoE keep intermediate tensors live together.
            ring_heap=1073741824,
        ),
        compare_fn={
            # Window-cache payload and scale tolerances on mapped attention rows.
            "window_cache": mapped_pool_ratio_allclose(
                "window_slots", pool_name="window cache", mapping_shape=(EP_SIZE, DECODE_MAX_TOKENS),
                block_size=128, leading_rank_axis=True,
                atol=2 ** -9, rtol=2 ** -3, max_error_ratio=0.01,
            ),
            "window_cache_scale": mapped_pool_ratio_allclose(
                "window_slots", pool_name="window cache scale", mapping_shape=(EP_SIZE, DECODE_MAX_TOKENS),
                block_size=128, leading_rank_axis=True,
                atol=0.0, rtol=0.02, max_error_ratio=0.01,
            ),
            "compressed_cache": exact,
            "compressed_cache_scale": exact,
            "index_cache": exact,
            "index_cache_scale": exact,
            "state_cache": exact,
            "topk_indices": exact,
            "pool_block_ids": exact,
            "pool_block_counts": exact,
            # Layer integration budget: at most 1.5% outliers per rank.
            "attention_output": _compare_active_rows_per_rank(
                ratio_allclose(atol=1e-2, rtol=1.0 / 128, max_error_ratio=0.015),
                counts,
            ),
            "attention_hidden": _compare_active_rows_per_rank(
                ratio_allclose(atol=1e-2, rtol=1.0 / 128, max_error_ratio=0.015),
                counts,
            ),
            "attention_pre_mix": ratio_allclose(atol=1e-4, rtol=1e-4),
            "next_pre_mix": _compare_next_pre_mix_from_actual_attention(counts),
            "x_mixed": _compare_active_rows_per_rank(
                ratio_reldiff(diff_thd=0.01, pct_thd=0.05),
                counts,
            ),
            "x_next": _compare_moe_from_actual_attention(counts),
        },
    )
    if result is not None and not result.passed:
        raise SystemExit(result.error or 1)
