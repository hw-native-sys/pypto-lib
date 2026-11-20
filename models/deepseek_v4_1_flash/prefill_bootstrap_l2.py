# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 terminal-prefill bootstrap in one L2 per rank."""

import pypto.language as pl
import pypto.language.distributed as pld
import config as C
import dspark_init as init
import dspark_drafter as draft
import markov
from dspark_init import bootstrap_inputs, prepare_prefill_drafter_geometry, publish_bootstrap
from dspark_drafter import dspark_drafter
from markov import markov_output

C_EP_SIZE = C.EP_SIZE
C_ORI_BLOCKS_DYN = C.ORI_BLOCKS_DYN
DRAFT_BLOCK_SIZE = draft.BLOCK_SIZE
DRAFT_B_DYN = draft.B_DYN
DRAFT_D = draft.D
DRAFT_DISPATCH_PACKET_WIDTH = draft.DISPATCH_PACKET_WIDTH
DRAFT_DSPARK_DRAFT_LAYERS = draft.DSPARK_DRAFT_LAYERS
DRAFT_EP_SIZE = draft.EP_SIZE
DRAFT_HC_DIM = draft.HC_DIM
DRAFT_HC_MULT = draft.HC_MULT
DRAFT_HEAD_DIM = draft.HEAD_DIM
DRAFT_LOCAL_H = draft.LOCAL_H
DRAFT_LOCAL_O_GROUPS = draft.LOCAL_O_GROUPS
DRAFT_LOCAL_O_WIDTH = draft.LOCAL_O_WIDTH
DRAFT_MAIN_IN = draft.MAIN_IN
DRAFT_MIX_HC = draft.MIX_HC
DRAFT_MOE_INTER = draft.MOE_INTER
DRAFT_MOE_TOKENS = draft.MOE_TOKENS
DRAFT_MX_GROUP = draft.MX_GROUP
DRAFT_MX_PACKED_LANE_COLS = draft.MX_PACKED_LANE_COLS
DRAFT_MX_W1_PACKED_ROWS = draft.MX_W1_PACKED_ROWS
DRAFT_MX_W2_PACKED_ROWS = draft.MX_W2_PACKED_ROWS
DRAFT_MX_W3_PACKED_ROWS = draft.MX_W3_PACKED_ROWS
DRAFT_N_EXPERTS = draft.N_EXPERTS
DRAFT_N_LOCAL_EXPERTS = draft.N_LOCAL_EXPERTS
DRAFT_O_GROUP_IN = draft.O_GROUP_IN
DRAFT_O_LORA = draft.O_LORA
DRAFT_Q_LORA = draft.Q_LORA
DRAFT_RECV_MAX = draft.RECV_MAX
DRAFT_ROUTE_WIDTH = draft.ROUTE_WIDTH
DRAFT_SIGNAL_PAD = draft.SIGNAL_PAD
DRAFT_TOPK = draft.TOPK
DRAFT_T_DYN = draft.T_DYN
DRAFT_VOCAB = draft.VOCAB
DRAFT_WINDOW_CACHE_BYTES = draft.WINDOW_CACHE_BYTES
INIT_DECODE_BATCH = init.DECODE_BATCH
INIT_DSPARK_DRAFT_LAYERS = init.DSPARK_DRAFT_LAYERS
INIT_DSPARK_QUERY_WIDTH = init.DSPARK_QUERY_WIDTH
INIT_EP_SIZE = init.EP_SIZE
INIT_PREFILL_SAMPLE_ROWS = init.PREFILL_SAMPLE_ROWS
INIT_ROPE_HALF = init.ROPE_HALF
INIT_ROPE_ROWS_DYN = init.ROPE_ROWS_DYN
INIT_SAMPLED_IDS_PAD = init.SAMPLED_IDS_PAD
INIT_STATE_META_WIDTH = init.STATE_META_WIDTH
INIT_STATE_TOKEN_WIDTH = init.STATE_TOKEN_WIDTH
INIT_TABLE_DYN = init.TABLE_DYN
INIT_T_DYN = init.T_DYN
MARKOV_B_DYN = markov.B_DYN
MARKOV_D = markov.D
MARKOV_DSPARK_QUERY_WIDTH = markov.DSPARK_QUERY_WIDTH
MARKOV_EP_SIZE = markov.EP_SIZE
MARKOV_GROUP_LOGIT_ROWS = markov.GROUP_LOGIT_ROWS
MARKOV_LM_HEAD_TP_SIZE = markov.LM_HEAD_TP_SIZE
MARKOV_LOGIT_ROWS = markov.LOGIT_ROWS
MARKOV_MARKOV_RANK = markov.MARKOV_RANK
MARKOV_VOCAB = markov.VOCAB
MARKOV_VOCAB_PER_TP = markov.VOCAB_PER_TP


@pl.jit(auto_scope=False)
def l2_prefill_bootstrap_dspark(
    slot_ids: pl.Tensor[[INIT_DECODE_BATCH], pl.INT32],
    generations: pl.Tensor[[INIT_DECODE_BATCH], pl.INT32],
    prompt_lengths: pl.Tensor[[INIT_DECODE_BATCH], pl.INT32],
    position_limits: pl.Tensor[[INIT_DECODE_BATCH], pl.INT32],
    sampled_rows: pl.Tensor[[INIT_DECODE_BATCH], pl.INT32],
    sampled_ids: pl.Tensor[[INIT_PREFILL_SAMPLE_ROWS, INIT_SAMPLED_IDS_PAD], pl.INT32],
    state_meta: pl.InOut[pl.Tensor[[INIT_DECODE_BATCH, INIT_STATE_META_WIDTH], pl.INT32]],
    next_prefill_tokens: pl.Out[pl.Tensor[[INIT_DECODE_BATCH], pl.INT64]],
    first_query_positions: pl.Out[pl.Tensor[[INIT_DECODE_BATCH], pl.INT32]],
    positions: pl.Tensor[[INIT_T_DYN], pl.INT32],
    request_ids: pl.Tensor[[INIT_T_DYN], pl.INT32],
    num_tokens: pl.Tensor[[1], pl.INT32],
    block_tables: pl.Tensor[[INIT_DSPARK_DRAFT_LAYERS, INIT_DECODE_BATCH, INIT_TABLE_DYN], pl.INT32],
    rope_cos_table: pl.Tensor[[INIT_ROPE_ROWS_DYN, INIT_ROPE_HALF], pl.FP32],
    rope_sin_table: pl.Tensor[[INIT_ROPE_ROWS_DYN, INIT_ROPE_HALF], pl.FP32],
    context_slots: pl.Out[pl.Tensor[[INIT_DSPARK_DRAFT_LAYERS, INIT_T_DYN], pl.INT64]],
    context_rope_cos: pl.Out[pl.Tensor[[INIT_T_DYN, INIT_ROPE_HALF], pl.FP32]],
    context_rope_sin: pl.Out[pl.Tensor[[INIT_T_DYN, INIT_ROPE_HALF], pl.FP32]],
    query_rope_cos: pl.Out[pl.Tensor[[INIT_DECODE_BATCH * INIT_DSPARK_QUERY_WIDTH, INIT_ROPE_HALF], pl.FP32]],
    query_rope_sin: pl.Out[pl.Tensor[[INIT_DECODE_BATCH * INIT_DSPARK_QUERY_WIDTH, INIT_ROPE_HALF], pl.FP32]],
    target_hidden: pl.Tensor[[DRAFT_T_DYN, DRAFT_MAIN_IN], pl.BF16],
    main_proj_weight: pl.Tensor[[DRAFT_MAIN_IN, DRAFT_D], pl.FP8E4M3FN],
    main_proj_scale: pl.Tensor[[DRAFT_MAIN_IN // DRAFT_MX_GROUP, DRAFT_D], pl.FP8E8M0],
    main_norm_weight: pl.Tensor[[DRAFT_D], pl.BF16],
    num_sampled: pl.Tensor[[DRAFT_B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[DRAFT_B_DYN], pl.INT64],
    embedding_weight: pl.Tensor[[DRAFT_VOCAB, DRAFT_D], pl.BF16],
    hc_attn_fn: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC, DRAFT_HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_D], pl.BF16],
    wq_a: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_D, DRAFT_Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_D // DRAFT_MX_GROUP, DRAFT_Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_Q_LORA, DRAFT_LOCAL_H * DRAFT_HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_Q_LORA // DRAFT_MX_GROUP, DRAFT_LOCAL_H * DRAFT_HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_D, DRAFT_HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_D // DRAFT_MX_GROUP, DRAFT_HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_GROUPS, DRAFT_O_GROUP_IN, DRAFT_O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_GROUPS, DRAFT_O_GROUP_IN // DRAFT_MX_GROUP, DRAFT_O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_WIDTH, DRAFT_D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_WIDTH // DRAFT_MX_GROUP, DRAFT_D], pl.FP8E8M0],
    window_cache: pl.InOut[pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, C_ORI_BLOCKS_DYN, DRAFT_BLOCK_SIZE, 1, DRAFT_WINDOW_CACHE_BYTES], pl.UINT8]],
    hc_ffn_fn: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC, DRAFT_HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_D], pl.BF16],
    gate_weight: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_N_EXPERTS, DRAFT_D], pl.BF16],
    correction_bias: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS, DRAFT_MX_W1_PACKED_ROWS, DRAFT_MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS * (DRAFT_D // DRAFT_MX_GROUP), DRAFT_MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS, DRAFT_MX_W2_PACKED_ROWS, DRAFT_MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS * (DRAFT_MOE_INTER // DRAFT_MX_GROUP), DRAFT_D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS, DRAFT_MX_W3_PACKED_ROWS, DRAFT_MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS * (DRAFT_D // DRAFT_MX_GROUP), DRAFT_MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_D, DRAFT_MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_D // DRAFT_MX_GROUP, DRAFT_MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_MOE_INTER, DRAFT_D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_MOE_INTER // DRAFT_MX_GROUP, DRAFT_D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_D, DRAFT_MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_D // DRAFT_MX_GROUP, DRAFT_MOE_INTER], pl.FP8E8M0],
    draft_x_hc: pl.Out[pl.Tensor[[DRAFT_MOE_TOKENS, DRAFT_HC_MULT, DRAFT_D], pl.FP32]],
    draft_pre_mix: pl.Out[pl.Tensor[[DRAFT_MOE_TOKENS, DRAFT_HC_MULT], pl.FP32]],
    recv_meta: pld.DistributedTensor[[DRAFT_EP_SIZE, DRAFT_N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, DRAFT_DISPATCH_PACKET_WIDTH], pl.INT8],
    recv_routes: pld.DistributedTensor[[DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, DRAFT_ROUTE_WIDTH], pl.INT32],
    data_arrived: pld.DistributedTensor[[DRAFT_EP_SIZE, DRAFT_N_LOCAL_EXPERTS, DRAFT_SIGNAL_PAD], pl.INT32],
    routed_output: pld.DistributedTensor[[DRAFT_MOE_TOKENS * DRAFT_TOPK, DRAFT_D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[DRAFT_EP_SIZE, DRAFT_N_LOCAL_EXPERTS, DRAFT_SIGNAL_PAD], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    head_hidden: pl.Out[pl.Tensor[[MARKOV_B_DYN, MARKOV_DSPARK_QUERY_WIDTH, MARKOV_D], pl.BF16]],
    final_norm_weight: pl.Tensor[[MARKOV_D], pl.BF16],
    lm_head_weight: pl.Tensor[[MARKOV_D, MARKOV_VOCAB_PER_TP], pl.BF16, pl.NZ],
    markov_w1: pl.Tensor[[MARKOV_VOCAB, MARKOV_MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[MARKOV_VOCAB, MARKOV_MARKOV_RANK], pl.BF16],
    confidence_weight: pl.Tensor[[1, MARKOV_D + MARKOV_MARKOV_RANK], pl.FP32],
    draft_ids: pl.Out[pl.Tensor[[MARKOV_B_DYN, MARKOV_DSPARK_QUERY_WIDTH], pl.INT32]],
    confidences: pl.Out[pl.Tensor[[MARKOV_B_DYN, MARKOV_DSPARK_QUERY_WIDTH], pl.FP32]],
    hidden_window: pld.DistributedTensor[[MARKOV_GROUP_LOGIT_ROWS, MARKOV_D], pl.BF16],
    hidden_done: pld.DistributedTensor[[MARKOV_LM_HEAD_TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MARKOV_LOGIT_ROWS, MARKOV_VOCAB], pl.BF16],
    logits_done: pld.DistributedTensor[[MARKOV_LM_HEAD_TP_SIZE, 1], pl.INT32],
    logits_epoch: pl.Scalar[pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[INIT_DECODE_BATCH, INIT_STATE_TOKEN_WIDTH], pl.INT64]],
):
    bootstrap_inputs(
        slot_ids, generations, prompt_lengths, position_limits,
        sampled_rows, sampled_ids, state_meta, next_prefill_tokens,
        first_query_positions,
    )
    prepare_prefill_drafter_geometry(
        positions, request_ids, num_tokens, first_query_positions,
        block_tables, rope_cos_table, rope_sin_table, context_slots,
        context_rope_cos, context_rope_sin, query_rope_cos, query_rope_sin,
    )
    dspark_drafter(
        target_hidden, main_proj_weight, main_proj_scale, main_norm_weight,
        num_sampled, last_sampled, next_prefill_tokens, next_prefill_tokens,
        first_query_positions, block_tables, context_slots, context_rope_cos,
        context_rope_sin, query_rope_cos, query_rope_sin, embedding_weight,
        hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight,
        wq_a, wq_a_scale, q_norm_weight, wq_b,
        wq_b_scale, wkv, wkv_scale, kv_norm_weight,
        attn_sink, wo_a, wo_a_scale, wo_b,
        wo_b_scale, window_cache, hc_ffn_fn, hc_ffn_scale,
        hc_ffn_base, ffn_norm_weight, gate_weight, correction_bias,
        routed_w1, routed_w1_scale, routed_w2, routed_w2_scale,
        routed_w3, routed_w3_scale, shared_w1, shared_w1_scale,
        shared_w2, shared_w2_scale, shared_w3, shared_w3_scale,
        draft_x_hc, draft_pre_mix, recv_meta, recv_x,
        recv_routes, data_arrived, routed_output, combine_arrived,
        rank, moe_epoch,
    )
    markov_output(
        draft_x_hc, draft_pre_mix, head_hidden, final_norm_weight,
        lm_head_weight, num_sampled, last_sampled, next_prefill_tokens,
        next_prefill_tokens, markov_w1, markov_w2, confidence_weight,
        draft_ids, confidences, hidden_window, hidden_done,
        logits_window, logits_done, rank, logits_epoch,
    )
    publish_bootstrap(
        slot_ids, generations, prompt_lengths, position_limits,
        sampled_rows, sampled_ids, draft_ids, state_tokens,
        state_meta,
    )


@pl.jit.host
def l3_prefill_bootstrap_dspark(
    slot_ids: pl.Tensor[[INIT_EP_SIZE, INIT_DECODE_BATCH], pl.INT32],
    generations: pl.Tensor[[INIT_EP_SIZE, INIT_DECODE_BATCH], pl.INT32],
    prompt_lengths: pl.Tensor[[INIT_EP_SIZE, INIT_DECODE_BATCH], pl.INT32],
    position_limits: pl.Tensor[[INIT_EP_SIZE, INIT_DECODE_BATCH], pl.INT32],
    sampled_rows: pl.Tensor[[INIT_EP_SIZE, INIT_DECODE_BATCH], pl.INT32],
    sampled_ids: pl.Tensor[[INIT_EP_SIZE, INIT_PREFILL_SAMPLE_ROWS, INIT_SAMPLED_IDS_PAD], pl.INT32],
    state_meta: pl.InOut[pl.Tensor[[INIT_EP_SIZE, INIT_DECODE_BATCH, INIT_STATE_META_WIDTH], pl.INT32]],
    next_prefill_tokens: pl.Out[pl.Tensor[[INIT_EP_SIZE, INIT_DECODE_BATCH], pl.INT64]],
    first_query_positions: pl.Out[pl.Tensor[[INIT_EP_SIZE, INIT_DECODE_BATCH], pl.INT32]],
    positions: pl.Tensor[[INIT_EP_SIZE, INIT_T_DYN], pl.INT32],
    request_ids: pl.Tensor[[INIT_EP_SIZE, INIT_T_DYN], pl.INT32],
    num_tokens: pl.Tensor[[INIT_EP_SIZE, 1], pl.INT32],
    block_tables: pl.Tensor[[INIT_EP_SIZE, INIT_DSPARK_DRAFT_LAYERS, INIT_DECODE_BATCH, INIT_TABLE_DYN], pl.INT32],
    rope_cos_table: pl.Tensor[[INIT_EP_SIZE, INIT_ROPE_ROWS_DYN, INIT_ROPE_HALF], pl.FP32],
    rope_sin_table: pl.Tensor[[INIT_EP_SIZE, INIT_ROPE_ROWS_DYN, INIT_ROPE_HALF], pl.FP32],
    context_slots: pl.Out[pl.Tensor[[INIT_EP_SIZE, INIT_DSPARK_DRAFT_LAYERS, INIT_T_DYN], pl.INT64]],
    context_rope_cos: pl.Out[pl.Tensor[[INIT_EP_SIZE, INIT_T_DYN, INIT_ROPE_HALF], pl.FP32]],
    context_rope_sin: pl.Out[pl.Tensor[[INIT_EP_SIZE, INIT_T_DYN, INIT_ROPE_HALF], pl.FP32]],
    query_rope_cos: pl.Out[pl.Tensor[[INIT_EP_SIZE, INIT_DECODE_BATCH * INIT_DSPARK_QUERY_WIDTH, INIT_ROPE_HALF], pl.FP32]],
    query_rope_sin: pl.Out[pl.Tensor[[INIT_EP_SIZE, INIT_DECODE_BATCH * INIT_DSPARK_QUERY_WIDTH, INIT_ROPE_HALF], pl.FP32]],
    target_hidden: pl.Tensor[[C_EP_SIZE, DRAFT_T_DYN, DRAFT_MAIN_IN], pl.BF16],
    main_proj_weight: pl.Tensor[[C_EP_SIZE, DRAFT_MAIN_IN, DRAFT_D], pl.FP8E4M3FN],
    main_proj_scale: pl.Tensor[[C_EP_SIZE, DRAFT_MAIN_IN // DRAFT_MX_GROUP, DRAFT_D], pl.FP8E8M0],
    main_norm_weight: pl.Tensor[[C_EP_SIZE, DRAFT_D], pl.BF16],
    num_sampled: pl.Tensor[[C_EP_SIZE, DRAFT_B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[C_EP_SIZE, DRAFT_B_DYN], pl.INT64],
    embedding_weight: pl.Tensor[[C_EP_SIZE, DRAFT_VOCAB, DRAFT_D], pl.BF16],
    hc_attn_fn: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC, DRAFT_HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_D], pl.BF16],
    wq_a: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_D, DRAFT_Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_D // DRAFT_MX_GROUP, DRAFT_Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_Q_LORA, DRAFT_LOCAL_H * DRAFT_HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_Q_LORA // DRAFT_MX_GROUP, DRAFT_LOCAL_H * DRAFT_HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_D, DRAFT_HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_D // DRAFT_MX_GROUP, DRAFT_HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_GROUPS, DRAFT_O_GROUP_IN, DRAFT_O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_GROUPS, DRAFT_O_GROUP_IN // DRAFT_MX_GROUP, DRAFT_O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_WIDTH, DRAFT_D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_WIDTH // DRAFT_MX_GROUP, DRAFT_D], pl.FP8E8M0],
    window_cache: pl.InOut[pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, C_ORI_BLOCKS_DYN, DRAFT_BLOCK_SIZE, 1, DRAFT_WINDOW_CACHE_BYTES], pl.UINT8]],
    hc_ffn_fn: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC, DRAFT_HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_D], pl.BF16],
    gate_weight: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_N_EXPERTS, DRAFT_D], pl.BF16],
    correction_bias: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS, DRAFT_MX_W1_PACKED_ROWS, DRAFT_MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS * (DRAFT_D // DRAFT_MX_GROUP), DRAFT_MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS, DRAFT_MX_W2_PACKED_ROWS, DRAFT_MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS * (DRAFT_MOE_INTER // DRAFT_MX_GROUP), DRAFT_D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS, DRAFT_MX_W3_PACKED_ROWS, DRAFT_MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS * (DRAFT_D // DRAFT_MX_GROUP), DRAFT_MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_D, DRAFT_MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_D // DRAFT_MX_GROUP, DRAFT_MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_MOE_INTER, DRAFT_D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_MOE_INTER // DRAFT_MX_GROUP, DRAFT_D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_D, DRAFT_MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[C_EP_SIZE, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_D // DRAFT_MX_GROUP, DRAFT_MOE_INTER], pl.FP8E8M0],
    draft_x_hc: pl.Out[pl.Tensor[[C_EP_SIZE, DRAFT_MOE_TOKENS, DRAFT_HC_MULT, DRAFT_D], pl.FP32]],
    draft_pre_mix: pl.Out[pl.Tensor[[C_EP_SIZE, DRAFT_MOE_TOKENS, DRAFT_HC_MULT], pl.FP32]],
    moe_epoch: pl.Scalar[pl.INT32],
    head_hidden: pl.Out[pl.Tensor[[MARKOV_EP_SIZE, MARKOV_B_DYN, MARKOV_DSPARK_QUERY_WIDTH, MARKOV_D], pl.BF16]],
    final_norm_weight: pl.Tensor[[MARKOV_EP_SIZE, MARKOV_D], pl.BF16],
    lm_head_weight: pl.Tensor[[MARKOV_EP_SIZE, MARKOV_D, MARKOV_VOCAB_PER_TP], pl.BF16],
    markov_w1: pl.Tensor[[MARKOV_EP_SIZE, MARKOV_VOCAB, MARKOV_MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[MARKOV_EP_SIZE, MARKOV_VOCAB, MARKOV_MARKOV_RANK], pl.BF16],
    confidence_weight: pl.Tensor[[MARKOV_EP_SIZE, 1, MARKOV_D + MARKOV_MARKOV_RANK], pl.FP32],
    draft_ids: pl.Out[pl.Tensor[[MARKOV_EP_SIZE, MARKOV_B_DYN, MARKOV_DSPARK_QUERY_WIDTH], pl.INT32]],
    confidences: pl.Out[pl.Tensor[[MARKOV_EP_SIZE, MARKOV_B_DYN, MARKOV_DSPARK_QUERY_WIDTH], pl.FP32]],
    logits_epoch: pl.Scalar[pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[INIT_EP_SIZE, INIT_DECODE_BATCH, INIT_STATE_TOKEN_WIDTH], pl.INT64]],
):
    draft_recv_meta_buf = pld.alloc_window_buffer([DRAFT_EP_SIZE, DRAFT_N_LOCAL_EXPERTS], dtype=pl.INT32)
    draft_recv_x_buf = pld.alloc_window_buffer([DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, DRAFT_DISPATCH_PACKET_WIDTH], dtype=pl.INT8)
    draft_recv_routes_buf = pld.alloc_window_buffer([DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, DRAFT_ROUTE_WIDTH], dtype=pl.INT32)
    draft_data_arrived_buf = pld.alloc_window_buffer([DRAFT_EP_SIZE, DRAFT_N_LOCAL_EXPERTS, DRAFT_SIGNAL_PAD], dtype=pl.INT32)
    draft_routed_output_buf = pld.alloc_window_buffer([DRAFT_MOE_TOKENS * DRAFT_TOPK, DRAFT_D], dtype=pl.BF16)
    draft_combine_arrived_buf = pld.alloc_window_buffer([DRAFT_EP_SIZE, DRAFT_N_LOCAL_EXPERTS, DRAFT_SIGNAL_PAD], dtype=pl.INT32)
    markov_hidden_window_buf = pld.alloc_window_buffer([MARKOV_GROUP_LOGIT_ROWS, MARKOV_D], dtype=pl.BF16)
    markov_hidden_done_buf = pld.alloc_window_buffer([MARKOV_LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    markov_logits_window_buf = pld.alloc_window_buffer([MARKOV_LOGIT_ROWS, MARKOV_VOCAB], dtype=pl.BF16)
    markov_logits_done_buf = pld.alloc_window_buffer([MARKOV_LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        recv_meta = pld.window(draft_recv_meta_buf, [DRAFT_EP_SIZE, DRAFT_N_LOCAL_EXPERTS], dtype=pl.INT32)
        recv_x = pld.window(draft_recv_x_buf, [DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, DRAFT_DISPATCH_PACKET_WIDTH], dtype=pl.INT8)
        recv_routes = pld.window(draft_recv_routes_buf, [DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, DRAFT_ROUTE_WIDTH], dtype=pl.INT32)
        data_arrived = pld.window(draft_data_arrived_buf, [DRAFT_EP_SIZE, DRAFT_N_LOCAL_EXPERTS, DRAFT_SIGNAL_PAD], dtype=pl.INT32)
        routed_output = pld.window(draft_routed_output_buf, [DRAFT_MOE_TOKENS * DRAFT_TOPK, DRAFT_D], dtype=pl.BF16)
        combine_arrived = pld.window(draft_combine_arrived_buf, [DRAFT_EP_SIZE, DRAFT_N_LOCAL_EXPERTS, DRAFT_SIGNAL_PAD], dtype=pl.INT32)
        hidden_window = pld.window(markov_hidden_window_buf, [MARKOV_GROUP_LOGIT_ROWS, MARKOV_D], dtype=pl.BF16)
        hidden_done = pld.window(markov_hidden_done_buf, [MARKOV_LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        logits_window = pld.window(markov_logits_window_buf, [MARKOV_LOGIT_ROWS, MARKOV_VOCAB], dtype=pl.BF16)
        logits_done = pld.window(markov_logits_done_buf, [MARKOV_LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        l2_prefill_bootstrap_dspark(
            slot_ids[rank], generations[rank], prompt_lengths[rank], position_limits[rank],
            sampled_rows[rank], sampled_ids[rank], state_meta[rank], next_prefill_tokens[rank],
            first_query_positions[rank], positions[rank], request_ids[rank], num_tokens[rank],
            block_tables[rank], rope_cos_table[rank], rope_sin_table[rank], context_slots[rank],
            context_rope_cos[rank], context_rope_sin[rank], query_rope_cos[rank], query_rope_sin[rank],
            target_hidden[rank], main_proj_weight[rank], main_proj_scale[rank], main_norm_weight[rank],
            num_sampled[rank], last_sampled[rank], embedding_weight[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], attn_norm_weight[rank],
            wq_a[rank], wq_a_scale[rank], q_norm_weight[rank], wq_b[rank],
            wq_b_scale[rank], wkv[rank], wkv_scale[rank], kv_norm_weight[rank],
            attn_sink[rank], wo_a[rank], wo_a_scale[rank], wo_b[rank],
            wo_b_scale[rank], window_cache[rank], hc_ffn_fn[rank], hc_ffn_scale[rank],
            hc_ffn_base[rank], ffn_norm_weight[rank], gate_weight[rank], correction_bias[rank],
            routed_w1[rank], routed_w1_scale[rank], routed_w2[rank], routed_w2_scale[rank],
            routed_w3[rank], routed_w3_scale[rank], shared_w1[rank], shared_w1_scale[rank],
            shared_w2[rank], shared_w2_scale[rank], shared_w3[rank], shared_w3_scale[rank],
            draft_x_hc[rank], draft_pre_mix[rank], recv_meta, recv_x,
            recv_routes, data_arrived, routed_output, combine_arrived,
            rank, moe_epoch, head_hidden[rank], final_norm_weight[rank],
            lm_head_weight[rank], markov_w1[rank], markov_w2[rank], confidence_weight[rank],
            draft_ids[rank], confidences[rank], hidden_window, hidden_done,
            logits_window, logits_done, logits_epoch, state_tokens[rank],
            device=rank,
        )
