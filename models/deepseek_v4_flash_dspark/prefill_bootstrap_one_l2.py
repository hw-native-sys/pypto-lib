# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One-L2 batched prefill bootstrap."""

import pypto.language as pl
import pypto.language.distributed as pld
import dspark_drafter
import dspark_markov
import prefill_bootstrap
import prefill_bootstrap_metadata
import prefill_state
from prefill_bootstrap import prepare_bootstrap_token_rank, publish_bootstrap_rank
from prefill_bootstrap_metadata import prepare_bootstrap_metadata_rank
from prefill_state import extract_prefill_tails_rank

DRAFTER_ATTENTION_WINDOW_ROWS = dspark_drafter.ATTENTION_WINDOW_ROWS
DRAFTER_AUX_PAD = dspark_drafter.AUX_PAD
DRAFTER_BLOCK_SIZE = dspark_drafter.BLOCK_SIZE
DRAFTER_B_DYN = dspark_drafter.B_DYN
DRAFTER_CP_CONTEXT_T_DYN = dspark_drafter.CP_CONTEXT_T_DYN
DRAFTER_D = dspark_drafter.D
DRAFTER_DSPARK_CP_SIZE = dspark_drafter.DSPARK_CP_SIZE
DRAFTER_DSPARK_DRAFT_LAYERS = dspark_drafter.DSPARK_DRAFT_LAYERS
DRAFTER_DSPARK_MAX_BATCH = dspark_drafter.DSPARK_MAX_BATCH
DRAFTER_DSPARK_QUERY_PAD = dspark_drafter.DSPARK_QUERY_PAD
DRAFTER_DSPARK_QUERY_WIDTH = dspark_drafter.DSPARK_QUERY_WIDTH
DRAFTER_H = dspark_drafter.H
DRAFTER_HC_DIM = dspark_drafter.HC_DIM
DRAFTER_HC_MULT = dspark_drafter.HC_MULT
DRAFTER_HEAD_DIM = dspark_drafter.HEAD_DIM
DRAFTER_IDX_PAD = dspark_drafter.IDX_PAD
DRAFTER_LOCAL_O_GROUPS = dspark_drafter.LOCAL_O_GROUPS
DRAFTER_MAIN_IN = dspark_drafter.MAIN_IN
DRAFTER_MIX_HC = dspark_drafter.MIX_HC
DRAFTER_MOE_INTER = dspark_drafter.MOE_INTER
DRAFTER_N_EXPERTS_GLOBAL = dspark_drafter.N_EXPERTS_GLOBAL
DRAFTER_N_LOCAL = dspark_drafter.N_LOCAL
DRAFTER_N_RANKS = dspark_drafter.N_RANKS
DRAFTER_N_ROUTES = dspark_drafter.N_ROUTES
DRAFTER_ORI_BLOCK_NUM = dspark_drafter.ORI_BLOCK_NUM
DRAFTER_ORI_MAX_BLOCKS = dspark_drafter.ORI_MAX_BLOCKS
DRAFTER_O_GROUP_IN = dspark_drafter.O_GROUP_IN
DRAFTER_O_LORA = dspark_drafter.O_LORA
DRAFTER_O_WINDOW_ROWS = dspark_drafter.O_WINDOW_ROWS
DRAFTER_PREFILL_GROUP_CAP = dspark_drafter.PREFILL_GROUP_CAP
DRAFTER_Q_LORA = dspark_drafter.Q_LORA
DRAFTER_RECV_MAX = dspark_drafter.RECV_MAX
DRAFTER_ROPE_DIM = dspark_drafter.ROPE_DIM
DRAFTER_TOPK = dspark_drafter.TOPK
DRAFTER_TP_SIZE = dspark_drafter.TP_SIZE
DRAFTER_T_MAIN_DYN = dspark_drafter.T_MAIN_DYN
DRAFTER_T_QUERY = dspark_drafter.T_QUERY
DRAFTER_VOCAB = dspark_drafter.VOCAB
EXTRACT_BOOTSTRAP_BATCH = prefill_state.BOOTSTRAP_BATCH
EXTRACT_EP = prefill_state.EP
EXTRACT_LEASES_DYN = prefill_state.LEASES_DYN
EXTRACT_LOCAL_ROWS_DYN = prefill_state.LOCAL_ROWS_DYN
EXTRACT_MAIN_DIM = prefill_state.MAIN_DIM
EXTRACT_META_WIDTH = prefill_state.META_WIDTH
EXTRACT_TAIL_ROWS = prefill_state.TAIL_ROWS
EXTRACT_TP = prefill_state.TP
MARKOV_B_DYN = dspark_markov.B_DYN
MARKOV_D = dspark_markov.D
MARKOV_DSPARK_MARKOV_RANK = dspark_markov.DSPARK_MARKOV_RANK
MARKOV_DSPARK_QUERY_WIDTH = dspark_markov.DSPARK_QUERY_WIDTH
MARKOV_GROUP_LOGIT_ROWS = dspark_markov.GROUP_LOGIT_ROWS
MARKOV_MAX_LOGIT_ROWS = dspark_markov.MAX_LOGIT_ROWS
MARKOV_TP_SIZE = dspark_markov.TP_SIZE
MARKOV_VOCAB = dspark_markov.VOCAB
MARKOV_VOCAB_PER_TP = dspark_markov.VOCAB_PER_TP
MARKOV_WORLD_SIZE = dspark_markov.WORLD_SIZE
METADATA_BATCH = prefill_bootstrap_metadata.BATCH
METADATA_CONTEXT = prefill_bootstrap_metadata.CONTEXT
METADATA_EP = prefill_bootstrap_metadata.EP
METADATA_GROUP_QUERY = prefill_bootstrap_metadata.GROUP_QUERY
METADATA_LEASES_DYN = prefill_bootstrap_metadata.LEASES_DYN
METADATA_POSITIONS_DYN = prefill_bootstrap_metadata.POSITIONS_DYN
METADATA_QUERY = prefill_bootstrap_metadata.QUERY
METADATA_ROPE_DIM = prefill_bootstrap_metadata.ROPE_DIM
METADATA_TABLE_DEPTH = prefill_bootstrap_metadata.TABLE_DEPTH
METADATA_TP = prefill_bootstrap_metadata.TP
PUBLISH_DESCRIPTOR_WIDTH = prefill_bootstrap.DESCRIPTOR_WIDTH
PUBLISH_EP = prefill_bootstrap.EP
PUBLISH_LEASES_DYN = prefill_bootstrap.LEASES_DYN
PUBLISH_MAX_REQUESTS = prefill_bootstrap.MAX_REQUESTS
PUBLISH_PAYLOAD_WIDTH = prefill_bootstrap.PAYLOAD_WIDTH
PUBLISH_REQUESTS_DYN = prefill_bootstrap.REQUESTS_DYN
PUBLISH_TP = prefill_bootstrap.TP
TOKENS_DESCRIPTOR_WIDTH = prefill_bootstrap.DESCRIPTOR_WIDTH
TOKENS_EP = prefill_bootstrap.EP
TOKENS_MAX_REQUESTS = prefill_bootstrap.MAX_REQUESTS
TOKENS_REQUESTS_DYN = prefill_bootstrap.REQUESTS_DYN
TOKENS_TP = prefill_bootstrap.TP


from decode_prepare import fence_drafter_head_hidden

token_inline = pl.jit.inline(auto_scope=False)(prepare_bootstrap_token_rank._func)
metadata_inline = pl.jit.inline(auto_scope=False)(prepare_bootstrap_metadata_rank._func)
extract_inline = pl.jit.inline(auto_scope=False)(extract_prefill_tails_rank._func)
publish_inline = pl.jit.inline(auto_scope=False)(publish_bootstrap_rank._func)
dspark_drafter_inline = dspark_drafter.dspark_drafter
markov_inline = dspark_markov.distributed_markov_sample

@pl.jit(auto_scope=False)
def l2_bootstrap_one_l2(
    descriptors: pl.Tensor[[TOKENS_REQUESTS_DYN, TOKENS_DESCRIPTOR_WIDTH], pl.INT32],
    sampled_ids: pl.Tensor[[TOKENS_MAX_REQUESTS, 8], pl.INT32],
    next_prefill_tokens: pl.InOut[pl.Tensor[[TOKENS_REQUESTS_DYN], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[METADATA_LEASES_DYN, 6], pl.INT32]],
    full_cos: pl.Tensor[[METADATA_POSITIONS_DYN, METADATA_ROPE_DIM], pl.BF16],
    full_sin: pl.Tensor[[METADATA_POSITIONS_DYN, METADATA_ROPE_DIM], pl.BF16],
    tail_selectors: pl.InOut[pl.Tensor[[METADATA_BATCH, 4], pl.INT32]],
    num_sampled: pl.InOut[pl.Tensor[[METADATA_BATCH], pl.INT32]],
    last_sampled: pl.InOut[pl.Tensor[[METADATA_BATCH], pl.INT64]],
    anchor_positions: pl.InOut[pl.Tensor[[METADATA_BATCH], pl.INT32]],
    block_tables: pl.InOut[pl.Tensor[[3, METADATA_BATCH, METADATA_TABLE_DEPTH], pl.INT32]],
    context_group_position_ids: pl.InOut[pl.Tensor[[METADATA_CONTEXT], pl.INT32]],
    context_group_slot_mapping: pl.InOut[pl.Tensor[[3, METADATA_CONTEXT], pl.INT64]],
    context_group_freqs_cos: pl.InOut[pl.Tensor[[METADATA_CONTEXT, METADATA_ROPE_DIM], pl.BF16]],
    context_group_freqs_sin: pl.InOut[pl.Tensor[[METADATA_CONTEXT, METADATA_ROPE_DIM], pl.BF16]],
    query_group_position_ids: pl.InOut[pl.Tensor[[METADATA_GROUP_QUERY], pl.INT32]],
    query_group_slot_mapping: pl.InOut[pl.Tensor[[3, METADATA_GROUP_QUERY], pl.INT64]],
    query_freqs_cos: pl.InOut[pl.Tensor[[METADATA_QUERY, METADATA_ROPE_DIM], pl.BF16]],
    query_freqs_sin: pl.InOut[pl.Tensor[[METADATA_QUERY, METADATA_ROPE_DIM], pl.BF16]],
    query_group_freqs_cos: pl.InOut[pl.Tensor[[METADATA_GROUP_QUERY, METADATA_ROPE_DIM], pl.BF16]],
    query_group_freqs_sin: pl.InOut[pl.Tensor[[METADATA_GROUP_QUERY, METADATA_ROPE_DIM], pl.BF16]],
    logit_row_indices: pl.InOut[pl.Tensor[[128], pl.INT32]],
    tail: pl.Tensor[[EXTRACT_LEASES_DYN, EXTRACT_TAIL_ROWS, EXTRACT_MAIN_DIM], pl.BF16],
    positions: pl.Tensor[[EXTRACT_LEASES_DYN, EXTRACT_TAIL_ROWS], pl.INT32],
    context_hidden: pl.InOut[pl.Tensor[[EXTRACT_LOCAL_ROWS_DYN, EXTRACT_MAIN_DIM], pl.BF16]],
    initial_hidden: pl.Out[pl.Tensor[[DRAFTER_DSPARK_MAX_BATCH * DRAFTER_DSPARK_QUERY_PAD, DRAFTER_HC_MULT, DRAFTER_D], pl.FP32]],
    intermediate_hidden: pl.Out[pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS, DRAFTER_DSPARK_MAX_BATCH * DRAFTER_DSPARK_QUERY_PAD, DRAFTER_HC_MULT, DRAFTER_D], pl.FP32]],
    main_proj_weight: pl.Tensor[[DRAFTER_D, DRAFTER_MAIN_IN], pl.BF16],
    main_norm_weight: pl.Tensor[[DRAFTER_D], pl.BF16],
    embedding_weight: pl.Tensor[[DRAFTER_VOCAB, DRAFTER_D], pl.BF16],
    hc_attn_fn: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_MIX_HC, DRAFTER_HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_D], pl.BF16],
    wq_a: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS, DRAFTER_D, DRAFTER_Q_LORA], pl.BF16, pl.NZ],
    wq_b: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS, DRAFTER_Q_LORA, DRAFTER_H * DRAFTER_HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_H * DRAFTER_HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_D, DRAFTER_HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_HEAD_DIM], pl.BF16],
    kv_caches: pl.InOut[pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS, DRAFTER_ORI_BLOCK_NUM, DRAFTER_BLOCK_SIZE, 1, DRAFTER_HEAD_DIM], pl.BF16]],
    attn_sink: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_H], pl.FP32],
    wo_a: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_LOCAL_O_GROUPS, DRAFTER_O_LORA, DRAFTER_O_GROUP_IN], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_LOCAL_O_GROUPS, DRAFTER_D, DRAFTER_O_LORA], pl.INT8, pl.NZ],
    wo_b_scale: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_MIX_HC, DRAFTER_HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_D], pl.BF16],
    gate_w: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_N_EXPERTS_GLOBAL, DRAFTER_D], pl.FP32],
    gate_bias: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_VOCAB, DRAFTER_TOPK], pl.INT32],
    routed_w1: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_N_LOCAL, DRAFTER_MOE_INTER, DRAFTER_D], pl.INT8, pl.NZ],
    routed_w1_scale: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_N_LOCAL, DRAFTER_MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_N_LOCAL, DRAFTER_MOE_INTER, DRAFTER_D], pl.INT8, pl.NZ],
    routed_w3_scale: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_N_LOCAL, DRAFTER_MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_N_LOCAL, DRAFTER_D, DRAFTER_MOE_INTER], pl.INT8, pl.NZ],
    routed_w2_scale: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_N_LOCAL, DRAFTER_D], pl.FP32],
    shared_w1: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS, DRAFTER_MOE_INTER, DRAFTER_D], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS, DRAFTER_MOE_INTER, DRAFTER_D], pl.INT8, pl.NZ],
    shared_w3_scale: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS, DRAFTER_D, DRAFTER_MOE_INTER], pl.INT8, pl.NZ],
    shared_w2_scale: pl.Tensor[[DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_D], pl.FP32],
    hc_head_fn: pl.Tensor[[DRAFTER_HC_MULT, DRAFTER_HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[DRAFTER_HC_MULT], pl.FP32],
    head_hidden: pl.InOut[pl.Tensor[[DRAFTER_B_DYN, DRAFTER_DSPARK_QUERY_WIDTH, DRAFTER_D], pl.BF16]],
    final_norm_weight: pl.Tensor[[MARKOV_D], pl.BF16],
    lm_head_weight: pl.Tensor[[MARKOV_VOCAB_PER_TP, MARKOV_D], pl.BF16, pl.NZ],
    markov_w1: pl.Tensor[[MARKOV_VOCAB, MARKOV_DSPARK_MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[MARKOV_VOCAB, MARKOV_DSPARK_MARKOV_RANK], pl.BF16],
    confidence_head_weight: pl.Tensor[[1, MARKOV_D + MARKOV_DSPARK_MARKOV_RANK], pl.FP32],
    draft_token_ids: pl.InOut[pl.Tensor[[MARKOV_B_DYN, MARKOV_DSPARK_QUERY_WIDTH], pl.INT32]],
    confidence_probs: pl.Out[pl.Tensor[[MARKOV_B_DYN, MARKOV_DSPARK_QUERY_WIDTH], pl.FP32]],
    state_tokens: pl.InOut[pl.Tensor[[PUBLISH_LEASES_DYN, 8], pl.INT64]],
    hidden_gather_window: pld.DistributedTensor[[DRAFTER_PREFILL_GROUP_CAP, DRAFTER_D], pl.BF16],
    hidden_gather_signal: pld.DistributedTensor[[DRAFTER_DSPARK_CP_SIZE, 1], pl.INT32],
    attention_window: pld.DistributedTensor[[DRAFTER_ATTENTION_WINDOW_ROWS, DRAFTER_O_GROUP_IN], pl.BF16],
    attention_signal: pld.DistributedTensor[[DRAFTER_TP_SIZE, 1], pl.INT32],
    o_window: pld.DistributedTensor[[DRAFTER_O_WINDOW_ROWS, DRAFTER_D], pl.BF16],
    o_signal: pld.DistributedTensor[[DRAFTER_TP_SIZE, 1], pl.INT32],
    recv_meta: pld.DistributedTensor[[DRAFTER_N_RANKS, DRAFTER_N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[DRAFTER_N_LOCAL * DRAFTER_RECV_MAX, DRAFTER_D], pl.INT8],
    recv_aux: pld.DistributedTensor[[DRAFTER_N_LOCAL * DRAFTER_RECV_MAX, DRAFTER_AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[DRAFTER_N_LOCAL * DRAFTER_RECV_MAX, DRAFTER_IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[DRAFTER_N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[DRAFTER_N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[DRAFTER_N_ROUTES, DRAFTER_D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[DRAFTER_N_RANKS, 1], pl.INT32],
    hidden_window: pld.DistributedTensor[[MARKOV_GROUP_LOGIT_ROWS, MARKOV_D], pl.BF16],
    logits_window: pld.DistributedTensor[[MARKOV_MAX_LOGIT_ROWS, MARKOV_VOCAB], pl.FP32],
    hidden_done: pld.DistributedTensor[[MARKOV_TP_SIZE, 1], pl.INT32],
    logits_done: pld.DistributedTensor[[MARKOV_TP_SIZE, 1], pl.INT32],
    window: pl.InOut[pld.DistributedTensor[[PUBLISH_TP * PUBLISH_MAX_REQUESTS, PUBLISH_PAYLOAD_WIDTH], pl.INT64]],
    signal: pl.InOut[pld.DistributedTensor[[PUBLISH_TP, 1], pl.INT32]],
    rank: pl.Scalar[pl.INT32],
):
    with pl.scope():
        token_inline(descriptors, sampled_ids, next_prefill_tokens, rank % TOKENS_TP)
    with pl.scope():
        metadata_inline(descriptors, state_meta, full_cos, full_sin, tail_selectors, num_sampled, last_sampled, anchor_positions, block_tables, context_group_position_ids, context_group_slot_mapping, context_group_freqs_cos, context_group_freqs_sin, query_group_position_ids, query_group_slot_mapping, query_freqs_cos, query_freqs_sin, query_group_freqs_cos, query_group_freqs_sin, logit_row_indices, rank % METADATA_TP)
    with pl.scope():
        extract_inline(tail_selectors, state_meta, tail, positions, context_hidden, rank % EXTRACT_TP)
    with pl.scope():
        rank_head, head_ready = dspark_drafter_inline(context_hidden, main_proj_weight, main_norm_weight, num_sampled, last_sampled, next_prefill_tokens, embedding_weight, context_group_position_ids, context_group_slot_mapping, anchor_positions, block_tables, query_group_position_ids, query_group_slot_mapping, context_group_freqs_cos, context_group_freqs_sin, query_freqs_cos, query_freqs_sin, query_group_freqs_cos, query_group_freqs_sin, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv, kv_caches, attn_sink, wo_a, wo_b, wo_b_scale, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, ffn_norm_w, gate_w, gate_bias, tid2eid, routed_w1, routed_w1_scale, routed_w3, routed_w3_scale, routed_w2, routed_w2_scale, shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, shared_w2, shared_w2_scale, hc_head_fn, hc_head_scale, hc_head_base, initial_hidden, intermediate_hidden, head_hidden, hidden_gather_window, hidden_gather_signal, attention_window, attention_signal, o_window, o_signal, recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived, routed_y_buf, combine_arrived, rank // DRAFTER_DSPARK_CP_SIZE * DRAFTER_DSPARK_CP_SIZE, rank % DRAFTER_DSPARK_CP_SIZE, rank)
        fence_drafter_head_hidden(rank_head, head_ready)
    with pl.scope():
        markov_inline(rank_head, final_norm_weight, lm_head_weight, logit_row_indices, num_sampled, last_sampled, next_prefill_tokens, markov_w1, markov_w2, confidence_head_weight, draft_token_ids, confidence_probs, hidden_window, hidden_done, logits_window, logits_done, rank // MARKOV_TP_SIZE * MARKOV_TP_SIZE, rank % MARKOV_TP_SIZE)
    with pl.scope():
        publish_inline(descriptors, sampled_ids, draft_token_ids, state_tokens, state_meta, window, signal, rank // PUBLISH_TP * PUBLISH_TP, rank % PUBLISH_TP)


@pl.jit.host
def l3_bootstrap_one_l2(
    descriptors: pl.Tensor[[TOKENS_EP, TOKENS_REQUESTS_DYN, TOKENS_DESCRIPTOR_WIDTH], pl.INT32],
    sampled_ids: pl.Tensor[[TOKENS_EP, TOKENS_MAX_REQUESTS, 8], pl.INT32],
    next_prefill_tokens: pl.InOut[pl.Tensor[[TOKENS_EP, TOKENS_REQUESTS_DYN], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[METADATA_EP, METADATA_LEASES_DYN, 6], pl.INT32]],
    full_cos: pl.Tensor[[METADATA_EP, METADATA_POSITIONS_DYN, METADATA_ROPE_DIM], pl.BF16],
    full_sin: pl.Tensor[[METADATA_EP, METADATA_POSITIONS_DYN, METADATA_ROPE_DIM], pl.BF16],
    tail_selectors: pl.InOut[pl.Tensor[[METADATA_EP, METADATA_BATCH, 4], pl.INT32]],
    num_sampled: pl.InOut[pl.Tensor[[METADATA_EP, METADATA_BATCH], pl.INT32]],
    last_sampled: pl.InOut[pl.Tensor[[METADATA_EP, METADATA_BATCH], pl.INT64]],
    anchor_positions: pl.InOut[pl.Tensor[[METADATA_EP, METADATA_BATCH], pl.INT32]],
    block_tables: pl.InOut[pl.Tensor[[METADATA_EP, 3, METADATA_BATCH, METADATA_TABLE_DEPTH], pl.INT32]],
    context_group_position_ids: pl.InOut[pl.Tensor[[METADATA_EP, METADATA_CONTEXT], pl.INT32]],
    context_group_slot_mapping: pl.InOut[pl.Tensor[[METADATA_EP, 3, METADATA_CONTEXT], pl.INT64]],
    context_group_freqs_cos: pl.InOut[pl.Tensor[[METADATA_EP, METADATA_CONTEXT, METADATA_ROPE_DIM], pl.BF16]],
    context_group_freqs_sin: pl.InOut[pl.Tensor[[METADATA_EP, METADATA_CONTEXT, METADATA_ROPE_DIM], pl.BF16]],
    query_group_position_ids: pl.InOut[pl.Tensor[[METADATA_EP, METADATA_GROUP_QUERY], pl.INT32]],
    query_group_slot_mapping: pl.InOut[pl.Tensor[[METADATA_EP, 3, METADATA_GROUP_QUERY], pl.INT64]],
    query_freqs_cos: pl.InOut[pl.Tensor[[METADATA_EP, METADATA_QUERY, METADATA_ROPE_DIM], pl.BF16]],
    query_freqs_sin: pl.InOut[pl.Tensor[[METADATA_EP, METADATA_QUERY, METADATA_ROPE_DIM], pl.BF16]],
    query_group_freqs_cos: pl.InOut[pl.Tensor[[METADATA_EP, METADATA_GROUP_QUERY, METADATA_ROPE_DIM], pl.BF16]],
    query_group_freqs_sin: pl.InOut[pl.Tensor[[METADATA_EP, METADATA_GROUP_QUERY, METADATA_ROPE_DIM], pl.BF16]],
    logit_row_indices: pl.InOut[pl.Tensor[[METADATA_EP, 128], pl.INT32]],
    tail: pl.Tensor[[EXTRACT_EP, EXTRACT_LEASES_DYN, EXTRACT_TAIL_ROWS, EXTRACT_MAIN_DIM], pl.BF16],
    positions: pl.Tensor[[EXTRACT_EP, EXTRACT_LEASES_DYN, EXTRACT_TAIL_ROWS], pl.INT32],
    context_hidden: pl.InOut[pl.Tensor[[EXTRACT_EP, EXTRACT_LOCAL_ROWS_DYN, EXTRACT_MAIN_DIM], pl.BF16]],
    initial_hidden: pl.Out[pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_MAX_BATCH * DRAFTER_DSPARK_QUERY_PAD, DRAFTER_HC_MULT, DRAFTER_D], pl.FP32]],
    intermediate_hidden: pl.Out[pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS, DRAFTER_DSPARK_MAX_BATCH * DRAFTER_DSPARK_QUERY_PAD, DRAFTER_HC_MULT, DRAFTER_D], pl.FP32]],
    main_proj_weight: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_D, DRAFTER_MAIN_IN], pl.BF16],
    main_norm_weight: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_D], pl.BF16],
    embedding_weight: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_VOCAB, DRAFTER_D], pl.BF16],
    hc_attn_fn: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_MIX_HC, DRAFTER_HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_D], pl.BF16],
    wq_a: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS, DRAFTER_D, DRAFTER_Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS, DRAFTER_Q_LORA, DRAFTER_H * DRAFTER_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_H * DRAFTER_HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_D, DRAFTER_HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_HEAD_DIM], pl.BF16],
    kv_caches: pl.InOut[pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS, DRAFTER_ORI_BLOCK_NUM, DRAFTER_BLOCK_SIZE, 1, DRAFTER_HEAD_DIM], pl.BF16]],
    attn_sink: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_H], pl.FP32],
    wo_a: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_LOCAL_O_GROUPS, DRAFTER_O_LORA, DRAFTER_O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_LOCAL_O_GROUPS, DRAFTER_D, DRAFTER_O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_MIX_HC, DRAFTER_HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_D], pl.BF16],
    gate_w: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_N_EXPERTS_GLOBAL, DRAFTER_D], pl.FP32],
    gate_bias: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_VOCAB, DRAFTER_TOPK], pl.INT32],
    routed_w1: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_N_LOCAL, DRAFTER_MOE_INTER, DRAFTER_D], pl.INT8],
    routed_w1_scale: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_N_LOCAL, DRAFTER_MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_N_LOCAL, DRAFTER_MOE_INTER, DRAFTER_D], pl.INT8],
    routed_w3_scale: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_N_LOCAL, DRAFTER_MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_N_LOCAL, DRAFTER_D, DRAFTER_MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_N_LOCAL, DRAFTER_D], pl.FP32],
    shared_w1: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS, DRAFTER_MOE_INTER, DRAFTER_D], pl.INT8],
    shared_w1_scale: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS, DRAFTER_MOE_INTER, DRAFTER_D], pl.INT8],
    shared_w3_scale: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS, DRAFTER_D, DRAFTER_MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_DSPARK_DRAFT_LAYERS * DRAFTER_D], pl.FP32],
    hc_head_fn: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_HC_MULT, DRAFTER_HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[DRAFTER_N_RANKS, 1], pl.FP32],
    hc_head_base: pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_HC_MULT], pl.FP32],
    head_hidden: pl.InOut[pl.Tensor[[DRAFTER_N_RANKS, DRAFTER_B_DYN, DRAFTER_DSPARK_QUERY_WIDTH, DRAFTER_D], pl.BF16]],
    final_norm_weight: pl.Tensor[[MARKOV_WORLD_SIZE, MARKOV_D], pl.BF16],
    lm_head_weight: pl.Tensor[[MARKOV_WORLD_SIZE, MARKOV_VOCAB_PER_TP, MARKOV_D], pl.BF16],
    markov_w1: pl.Tensor[[MARKOV_WORLD_SIZE, MARKOV_VOCAB, MARKOV_DSPARK_MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[MARKOV_WORLD_SIZE, MARKOV_VOCAB, MARKOV_DSPARK_MARKOV_RANK], pl.BF16],
    confidence_head_weight: pl.Tensor[[MARKOV_WORLD_SIZE, 1, MARKOV_D + MARKOV_DSPARK_MARKOV_RANK], pl.FP32],
    draft_token_ids: pl.InOut[pl.Tensor[[MARKOV_WORLD_SIZE, MARKOV_B_DYN, MARKOV_DSPARK_QUERY_WIDTH], pl.INT32]],
    confidence_probs: pl.Out[pl.Tensor[[MARKOV_WORLD_SIZE, MARKOV_B_DYN, MARKOV_DSPARK_QUERY_WIDTH], pl.FP32]],
    state_tokens: pl.InOut[pl.Tensor[[PUBLISH_EP, PUBLISH_LEASES_DYN, 8], pl.INT64]],
):
    descriptors.bind_dynamic(1, TOKENS_REQUESTS_DYN)
    next_prefill_tokens.bind_dynamic(1, TOKENS_REQUESTS_DYN)
    state_meta.bind_dynamic(1, METADATA_LEASES_DYN)
    full_cos.bind_dynamic(1, METADATA_POSITIONS_DYN)
    full_sin.bind_dynamic(1, METADATA_POSITIONS_DYN)
    state_meta.bind_dynamic(1, EXTRACT_LEASES_DYN)
    tail.bind_dynamic(1, EXTRACT_LEASES_DYN)
    positions.bind_dynamic(1, EXTRACT_LEASES_DYN)
    context_hidden.bind_dynamic(1, EXTRACT_LOCAL_ROWS_DYN)
    context_hidden.bind_dynamic(1, DRAFTER_T_MAIN_DYN)
    context_group_position_ids.bind_dynamic(1, DRAFTER_CP_CONTEXT_T_DYN)
    context_group_slot_mapping.bind_dynamic(2, DRAFTER_CP_CONTEXT_T_DYN)
    context_group_freqs_cos.bind_dynamic(1, DRAFTER_CP_CONTEXT_T_DYN)
    context_group_freqs_sin.bind_dynamic(1, DRAFTER_CP_CONTEXT_T_DYN)
    num_sampled.bind_dynamic(1, DRAFTER_B_DYN)
    last_sampled.bind_dynamic(1, DRAFTER_B_DYN)
    next_prefill_tokens.bind_dynamic(1, DRAFTER_B_DYN)
    anchor_positions.bind_dynamic(1, DRAFTER_B_DYN)
    block_tables.bind_dynamic(2, DRAFTER_B_DYN)
    head_hidden.bind_dynamic(1, DRAFTER_B_DYN)
    head_hidden.bind_dynamic(1, MARKOV_B_DYN)
    num_sampled.bind_dynamic(1, MARKOV_B_DYN)
    last_sampled.bind_dynamic(1, MARKOV_B_DYN)
    next_prefill_tokens.bind_dynamic(1, MARKOV_B_DYN)
    draft_token_ids.bind_dynamic(1, MARKOV_B_DYN)
    confidence_probs.bind_dynamic(1, MARKOV_B_DYN)
    descriptors.bind_dynamic(1, PUBLISH_REQUESTS_DYN)
    draft_token_ids.bind_dynamic(1, PUBLISH_REQUESTS_DYN)
    state_tokens.bind_dynamic(1, PUBLISH_LEASES_DYN)
    state_meta.bind_dynamic(1, PUBLISH_LEASES_DYN)
    hidden_gather_window_buf = pld.alloc_window_buffer([DRAFTER_PREFILL_GROUP_CAP, DRAFTER_D], dtype=pl.BF16)
    hidden_gather_signal_buf = pld.alloc_window_buffer([DRAFTER_DSPARK_CP_SIZE, 1], dtype=pl.INT32)
    attention_window_buf = pld.alloc_window_buffer([DRAFTER_ATTENTION_WINDOW_ROWS, DRAFTER_O_GROUP_IN], dtype=pl.BF16)
    attention_signal_buf = pld.alloc_window_buffer([DRAFTER_TP_SIZE, 1], dtype=pl.INT32)
    o_window_buf = pld.alloc_window_buffer([DRAFTER_O_WINDOW_ROWS, DRAFTER_D], dtype=pl.BF16)
    o_signal_buf = pld.alloc_window_buffer([DRAFTER_TP_SIZE, 1], dtype=pl.INT32)
    recv_meta_buf = pld.alloc_window_buffer([DRAFTER_N_RANKS, DRAFTER_N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([DRAFTER_N_LOCAL * DRAFTER_RECV_MAX, DRAFTER_D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer([DRAFTER_N_LOCAL * DRAFTER_RECV_MAX, DRAFTER_AUX_PAD], dtype=pl.FP32)
    recv_route_buf = pld.alloc_window_buffer([DRAFTER_N_LOCAL * DRAFTER_RECV_MAX, DRAFTER_IDX_PAD], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([DRAFTER_N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([DRAFTER_N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([DRAFTER_N_ROUTES, DRAFTER_D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([DRAFTER_N_RANKS, 1], dtype=pl.INT32)
    hidden_window_buf = pld.alloc_window_buffer(MARKOV_GROUP_LOGIT_ROWS * MARKOV_D * 2)
    logits_window_buf = pld.alloc_window_buffer(MARKOV_MAX_LOGIT_ROWS * MARKOV_VOCAB * 4)
    hidden_done_buf = pld.alloc_window_buffer(MARKOV_TP_SIZE * 4)
    logits_done_buf = pld.alloc_window_buffer(MARKOV_TP_SIZE * 4)
    window_buffer = pld.alloc_window_buffer([PUBLISH_TP * PUBLISH_MAX_REQUESTS, PUBLISH_PAYLOAD_WIDTH], dtype=pl.INT64)
    signal_buffer = pld.alloc_window_buffer([PUBLISH_TP, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        hidden_gather_window = pld.window(hidden_gather_window_buf, [DRAFTER_PREFILL_GROUP_CAP, DRAFTER_D], dtype=pl.BF16)
        hidden_gather_signal = pld.window(hidden_gather_signal_buf, [DRAFTER_DSPARK_CP_SIZE, 1], dtype=pl.INT32)
        attention_window = pld.window(attention_window_buf, [DRAFTER_ATTENTION_WINDOW_ROWS, DRAFTER_O_GROUP_IN], dtype=pl.BF16)
        attention_signal = pld.window(attention_signal_buf, [DRAFTER_TP_SIZE, 1], dtype=pl.INT32)
        o_window = pld.window(o_window_buf, [DRAFTER_O_WINDOW_ROWS, DRAFTER_D], dtype=pl.BF16)
        o_signal = pld.window(o_signal_buf, [DRAFTER_TP_SIZE, 1], dtype=pl.INT32)
        recv_meta = pld.window(recv_meta_buf, [DRAFTER_N_RANKS, DRAFTER_N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [DRAFTER_N_LOCAL * DRAFTER_RECV_MAX, DRAFTER_D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [DRAFTER_N_LOCAL * DRAFTER_RECV_MAX, DRAFTER_AUX_PAD], dtype=pl.FP32)
        recv_route = pld.window(recv_route_buf, [DRAFTER_N_LOCAL * DRAFTER_RECV_MAX, DRAFTER_IDX_PAD], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [DRAFTER_N_RANKS, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [DRAFTER_N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [DRAFTER_N_ROUTES, DRAFTER_D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [DRAFTER_N_RANKS, 1], dtype=pl.INT32)
        hidden_window = pld.window(hidden_window_buf, [MARKOV_GROUP_LOGIT_ROWS, MARKOV_D], dtype=pl.BF16)
        logits_window = pld.window(logits_window_buf, [MARKOV_MAX_LOGIT_ROWS, MARKOV_VOCAB], dtype=pl.FP32)
        hidden_done = pld.window(hidden_done_buf, [MARKOV_TP_SIZE, 1], dtype=pl.INT32)
        logits_done = pld.window(logits_done_buf, [MARKOV_TP_SIZE, 1], dtype=pl.INT32)
        window = pld.window(window_buffer, [PUBLISH_TP * PUBLISH_MAX_REQUESTS, PUBLISH_PAYLOAD_WIDTH], dtype=pl.INT64)
        signal = pld.window(signal_buffer, [PUBLISH_TP, 1], dtype=pl.INT32)
        l2_bootstrap_one_l2(
            descriptors[rank],
            sampled_ids[rank],
            next_prefill_tokens[rank],
            state_meta[rank],
            full_cos[rank],
            full_sin[rank],
            tail_selectors[rank],
            num_sampled[rank],
            last_sampled[rank],
            anchor_positions[rank],
            block_tables[rank],
            context_group_position_ids[rank],
            context_group_slot_mapping[rank],
            context_group_freqs_cos[rank],
            context_group_freqs_sin[rank],
            query_group_position_ids[rank],
            query_group_slot_mapping[rank],
            query_freqs_cos[rank],
            query_freqs_sin[rank],
            query_group_freqs_cos[rank],
            query_group_freqs_sin[rank],
            logit_row_indices[rank],
            tail[rank],
            positions[rank],
            context_hidden[rank],
            initial_hidden[rank],
            intermediate_hidden[rank],
            main_proj_weight[rank],
            main_norm_weight[rank],
            embedding_weight[rank],
            hc_attn_fn[rank],
            hc_attn_scale[rank],
            hc_attn_base[rank],
            attn_norm_w[rank],
            wq_a[rank],
            wq_b[rank],
            wq_b_scale[rank],
            wkv[rank],
            gamma_cq[rank],
            gamma_ckv[rank],
            kv_caches[rank],
            attn_sink[rank],
            wo_a[rank],
            wo_b[rank],
            wo_b_scale[rank],
            hc_ffn_fn[rank],
            hc_ffn_scale[rank],
            hc_ffn_base[rank],
            ffn_norm_w[rank],
            gate_w[rank],
            gate_bias[rank],
            tid2eid[rank],
            routed_w1[rank],
            routed_w1_scale[rank],
            routed_w3[rank],
            routed_w3_scale[rank],
            routed_w2[rank],
            routed_w2_scale[rank],
            shared_w1[rank],
            shared_w1_scale[rank],
            shared_w3[rank],
            shared_w3_scale[rank],
            shared_w2[rank],
            shared_w2_scale[rank],
            hc_head_fn[rank],
            hc_head_scale[rank],
            hc_head_base[rank],
            head_hidden[rank],
            final_norm_weight[rank],
            lm_head_weight[rank],
            markov_w1[rank],
            markov_w2[rank],
            confidence_head_weight[rank],
            draft_token_ids[rank],
            confidence_probs[rank],
            state_tokens[rank],
            hidden_gather_window,
            hidden_gather_signal,
            attention_window,
            attention_signal,
            o_window,
            o_signal,
            recv_meta,
            recv_x,
            recv_aux,
            recv_route,
            arrived,
            data_arrived,
            routed_y_buf,
            combine_arrived,
            hidden_window,
            logits_window,
            hidden_done,
            logits_done,
            window,
            signal,
            rank,
            device=rank,
        )
