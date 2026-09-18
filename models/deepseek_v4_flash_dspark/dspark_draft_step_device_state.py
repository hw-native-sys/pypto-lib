# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compose the unchanged DSpark drafter, Markov sampler, and state commit as one L2."""

import sys

# This fused entry point is the canonical TP4/EP16 DSpark program.  The drafter
# and LM head resolve their parallel sizes while they are imported, before the
# standalone argparse block below runs, so seed argv here when the caller did
# not provide explicit values.  Keeping TP in argv is important because the
# drafter removes only the temporary arguments that it adds itself.
if not any(arg == "--tp" or arg.startswith("--tp=") for arg in sys.argv):
    sys.argv.extend(("--tp", "4"))
if not any(arg == "--ep" or arg.startswith("--ep=") for arg in sys.argv):
    sys.argv.extend(("--ep", "16"))

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig

from dspark_device_state import (
    LOCAL_BATCH as DEVICE_STATE_LOCAL_BATCH,
    STATE_ANCHOR_POSITION,
    STATE_CAPACITY,
    STATE_DRAFT_COUNT,
    STATE_GENERATION,
    STATE_META_WIDTH,
    STATE_POSITION_LIMIT,
    STATE_TOKEN_WIDTH,
    STATE_VALID,
    commit_drafts_to_device_state,
)
from dspark_drafter import (
    ATTENTION_WINDOW_ROWS,
    AUX_PAD,
    BLOCK_SIZE,
    B_DYN,
    CP_CONTEXT_T_DYN,
    D,
    DSPARK_CP_SIZE,
    DSPARK_DRAFT_LAYERS,
    DSPARK_QUERY_WIDTH,
    H,
    HC_DIM,
    HC_MULT,
    HEAD_DIM,
    IDX_PAD,
    LOCAL_O_GROUPS,
    LOCAL_O_WIDTH,
    MAIN_IN,
    MIX_HC,
    MOE_INTER,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_RANKS,
    N_ROUTES,
    ORI_BLOCK_NUM,
    ORI_MAX_BLOCKS,
    O_GROUP_IN,
    O_LORA,
    O_WINDOW_ROWS,
    PREFILL_GROUP_CAP,
    Q_LORA,
    RECV_MAX,
    ROPE_DIM,
    T,
    TOPK,
    TP_SIZE,
    T_MAIN_DYN,
    T_QUERY,
    VOCAB,
    dspark_drafter_inline,
)
from dspark_markov import DSPARK_MARKOV_RANK, l2_distributed_markov_sample_inline
from lm_head import GROUP_LOGIT_ROWS, MAX_LOGIT_ROWS, VOCAB_PER_TP


def dspark_draft_step_device_state(
    target_hidden: pl.Tensor[[T_MAIN_DYN, MAIN_IN], pl.BF16],
    main_proj_weight: pl.Tensor[[D, MAIN_IN], pl.BF16],
    main_norm_weight: pl.Tensor[[D], pl.BF16],
    num_sampled: pl.Tensor[[B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[B_DYN], pl.INT64],
    embedding_weight: pl.Tensor[[VOCAB, D], pl.BF16],
    context_group_position_ids: pl.Tensor[[CP_CONTEXT_T_DYN], pl.INT32],
    context_group_slot_mapping: pl.Tensor[[DSPARK_DRAFT_LAYERS, CP_CONTEXT_T_DYN], pl.INT64],
    anchor_positions: pl.Tensor[[B_DYN], pl.INT32],
    block_tables: pl.Tensor[[DSPARK_DRAFT_LAYERS, B_DYN, ORI_MAX_BLOCKS], pl.INT32],
    query_group_position_ids: pl.Tensor[[DSPARK_CP_SIZE * T_QUERY], pl.INT32],
    query_group_slot_mapping: pl.Tensor[
        [DSPARK_DRAFT_LAYERS, DSPARK_CP_SIZE * T_QUERY], pl.INT64
    ],
    context_group_freqs_cos: pl.Tensor[[CP_CONTEXT_T_DYN, ROPE_DIM], pl.BF16],
    context_group_freqs_sin: pl.Tensor[[CP_CONTEXT_T_DYN, ROPE_DIM], pl.BF16],
    query_freqs_cos: pl.Tensor[[T_QUERY, ROPE_DIM], pl.BF16],
    query_freqs_sin: pl.Tensor[[T_QUERY, ROPE_DIM], pl.BF16],
    query_group_freqs_cos: pl.Tensor[[DSPARK_CP_SIZE * T_QUERY, ROPE_DIM], pl.BF16],
    query_group_freqs_sin: pl.Tensor[[DSPARK_CP_SIZE * T_QUERY, ROPE_DIM], pl.BF16],
    hc_attn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[DSPARK_DRAFT_LAYERS * D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[DSPARK_DRAFT_LAYERS * Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[DSPARK_DRAFT_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[DSPARK_DRAFT_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[DSPARK_DRAFT_LAYERS * HEAD_DIM], pl.BF16],
    kv_caches: pl.InOut[
        pl.Tensor[[DSPARK_DRAFT_LAYERS, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    attn_sink: pl.Tensor[[DSPARK_DRAFT_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[DSPARK_DRAFT_LAYERS * LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[DSPARK_DRAFT_LAYERS * D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[DSPARK_DRAFT_LAYERS * VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS * D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.FP32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    initial_hidden: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    intermediate_hidden: pl.Out[
        pl.Tensor[[DSPARK_DRAFT_LAYERS, T, HC_MULT, D], pl.FP32]
    ],
    head_hidden: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16]],
    hidden_gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    hidden_gather_signal: pld.DistributedTensor[[DSPARK_CP_SIZE, 1], pl.INT32],
    attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.BF16],
    o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    dsa_cp_rank: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    markov_w1: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    confidence_head_weight: pl.Tensor[[1, D + DSPARK_MARKOV_RANK], pl.FP32],
    draft_token_ids: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.INT32]],
    confidence_probs: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.FP32]],
    markov_hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    markov_hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    markov_logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    markov_logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    state_slot_ids: pl.Tensor[[DEVICE_STATE_LOCAL_BATCH], pl.INT32],
    state_generations: pl.Tensor[[DEVICE_STATE_LOCAL_BATCH], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[STATE_CAPACITY, STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[STATE_CAPACITY, STATE_META_WIDTH], pl.INT32]],
):
    """Run the three existing implementations in one rank-local L2 program."""
    with pl.scope():
        head_hidden, head_hidden_ready_tid = dspark_drafter_inline(
            target_hidden, main_proj_weight, main_norm_weight,
            num_sampled, last_sampled, next_prefill_tokens, embedding_weight,
            context_group_position_ids, context_group_slot_mapping,
            anchor_positions, block_tables,
            query_group_position_ids, query_group_slot_mapping,
            context_group_freqs_cos, context_group_freqs_sin,
            query_freqs_cos, query_freqs_sin,
            query_group_freqs_cos, query_group_freqs_sin,
            hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
            wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
            kv_caches, attn_sink, wo_a, wo_b, wo_b_scale,
            hc_ffn_fn, hc_ffn_scale, hc_ffn_base, ffn_norm_w,
            gate_w, gate_bias, tid2eid,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
            shared_w2, shared_w2_scale,
            hc_head_fn, hc_head_scale, hc_head_base,
            initial_hidden, intermediate_hidden, head_hidden,
            hidden_gather_window, hidden_gather_signal,
            attention_window, attention_signal, o_window, o_signal,
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            group_base, dsa_cp_rank, my_rank,
        )
        batch = pl.tensor.dim(head_hidden, 0)
        active_tokens = batch * DSPARK_QUERY_WIDTH
        head_hidden_flat = pl.reshape(head_hidden, [active_tokens, D])
        with pl.spmd(
            active_tokens,
            name_hint="dspark_drafter_markov_bridge",
            deps=[head_hidden_ready_tid],
        ):
            token = pl.tile.get_block_idx()
            head_hidden_flat[token : token + 1, :] = head_hidden_flat[
                token : token + 1, :
            ]
    with pl.scope():
        draft_token_ids, confidence_probs = l2_distributed_markov_sample_inline(
            head_hidden, final_norm_weight, lm_head_weight, logit_row_indices,
            num_sampled, last_sampled, next_prefill_tokens, markov_w1, markov_w2,
            confidence_head_weight, draft_token_ids, confidence_probs,
            markov_hidden_window, markov_hidden_done,
            markov_logits_window, markov_logits_done,
            group_base, dsa_cp_rank,
        )
    with pl.scope():
        state_tokens, state_meta = commit_drafts_to_device_state(
            state_slot_ids, state_generations, state_tokens, state_meta, draft_token_ids
        )
    return head_hidden, draft_token_ids, confidence_probs, state_tokens, state_meta


# Keep the validated standalone L3 while allowing the full decode program to
# inline this exact implementation after target acceptance.
_dspark_draft_step_device_state_impl = dspark_draft_step_device_state
dspark_draft_step_device_state_inline = pl.jit.inline(auto_scope=False)(
    _dspark_draft_step_device_state_impl
)
dspark_draft_step_device_state = pl.jit(auto_scope=False)(
    _dspark_draft_step_device_state_impl
)


@pl.jit.host
def l3_dspark_draft_step_device_state(
    target_hidden: pl.Tensor[[N_RANKS, T_MAIN_DYN, MAIN_IN], pl.BF16],
    initial_hidden: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32]],
    intermediate_hidden: pl.Out[
        pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS, T, HC_MULT, D], pl.FP32]
    ],
    main_proj_weight: pl.Tensor[[N_RANKS, D, MAIN_IN], pl.BF16],
    main_norm_weight: pl.Tensor[[N_RANKS, D], pl.BF16],
    num_sampled: pl.Tensor[[N_RANKS, B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[N_RANKS, B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[N_RANKS, B_DYN], pl.INT64],
    embedding_weight: pl.Tensor[[N_RANKS, VOCAB, D], pl.BF16],
    context_group_position_ids: pl.Tensor[[N_RANKS, CP_CONTEXT_T_DYN], pl.INT32],
    context_group_slot_mapping: pl.Tensor[
        [N_RANKS, DSPARK_DRAFT_LAYERS, CP_CONTEXT_T_DYN], pl.INT64
    ],
    anchor_positions: pl.Tensor[[N_RANKS, B_DYN], pl.INT32],
    block_tables: pl.Tensor[
        [N_RANKS, DSPARK_DRAFT_LAYERS, B_DYN, ORI_MAX_BLOCKS], pl.INT32
    ],
    query_group_position_ids: pl.Tensor[[N_RANKS, DSPARK_CP_SIZE * T_QUERY], pl.INT32],
    query_group_slot_mapping: pl.Tensor[
        [N_RANKS, DSPARK_DRAFT_LAYERS, DSPARK_CP_SIZE * T_QUERY], pl.INT64
    ],
    context_group_freqs_cos: pl.Tensor[[N_RANKS, CP_CONTEXT_T_DYN, ROPE_DIM], pl.BF16],
    context_group_freqs_sin: pl.Tensor[[N_RANKS, CP_CONTEXT_T_DYN, ROPE_DIM], pl.BF16],
    query_freqs_cos: pl.Tensor[[N_RANKS, T_QUERY, ROPE_DIM], pl.BF16],
    query_freqs_sin: pl.Tensor[[N_RANKS, T_QUERY, ROPE_DIM], pl.BF16],
    query_group_freqs_cos: pl.Tensor[
        [N_RANKS, DSPARK_CP_SIZE * T_QUERY, ROPE_DIM], pl.BF16
    ],
    query_group_freqs_sin: pl.Tensor[
        [N_RANKS, DSPARK_CP_SIZE * T_QUERY, ROPE_DIM], pl.BF16
    ],
    hc_attn_fn: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * HEAD_DIM], pl.BF16],
    kv_caches: pl.InOut[
        pl.Tensor[
            [N_RANKS, DSPARK_DRAFT_LAYERS, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    attn_sink: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[
        [N_RANKS, DSPARK_DRAFT_LAYERS * LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16
    ],
    wo_b: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[
        [N_RANKS, DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8
    ],
    routed_w1_scale: pl.Tensor[
        [N_RANKS, DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER], pl.FP32
    ],
    routed_w3: pl.Tensor[
        [N_RANKS, DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8
    ],
    routed_w3_scale: pl.Tensor[
        [N_RANKS, DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER], pl.FP32
    ],
    routed_w2: pl.Tensor[
        [N_RANKS, DSPARK_DRAFT_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8
    ],
    routed_w2_scale: pl.Tensor[
        [N_RANKS, DSPARK_DRAFT_LAYERS * N_LOCAL, D], pl.FP32
    ],
    shared_w1: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * D], pl.FP32],
    hc_head_fn: pl.Tensor[[N_RANKS, HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[N_RANKS, 1], pl.FP32],
    hc_head_base: pl.Tensor[[N_RANKS, HC_MULT], pl.FP32],
    head_hidden: pl.Out[pl.Tensor[[N_RANKS, B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16]],
    final_norm_weight: pl.Tensor[[N_RANKS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[N_RANKS, VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS], pl.INT32],
    markov_w1: pl.Tensor[[N_RANKS, VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[N_RANKS, VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    confidence_head_weight: pl.Tensor[
        [N_RANKS, 1, D + DSPARK_MARKOV_RANK], pl.FP32
    ],
    draft_token_ids: pl.Out[
        pl.Tensor[[N_RANKS, B_DYN, DSPARK_QUERY_WIDTH], pl.INT32]
    ],
    confidence_probs: pl.Out[
        pl.Tensor[[N_RANKS, B_DYN, DSPARK_QUERY_WIDTH], pl.FP32]
    ],
    state_slot_ids: pl.Tensor[[N_RANKS, DEVICE_STATE_LOCAL_BATCH], pl.INT32],
    state_generations: pl.Tensor[[N_RANKS, DEVICE_STATE_LOCAL_BATCH], pl.INT32],
    state_tokens: pl.InOut[
        pl.Tensor[[N_RANKS, STATE_CAPACITY, STATE_TOKEN_WIDTH], pl.INT64]
    ],
    state_meta: pl.InOut[
        pl.Tensor[[N_RANKS, STATE_CAPACITY, STATE_META_WIDTH], pl.INT32]
    ],
):
    """Launch the fused rank-local program over the canonical 16-rank topology."""
    target_hidden.bind_dynamic(1, T_MAIN_DYN)
    context_group_position_ids.bind_dynamic(1, CP_CONTEXT_T_DYN)
    context_group_slot_mapping.bind_dynamic(2, CP_CONTEXT_T_DYN)
    context_group_freqs_cos.bind_dynamic(1, CP_CONTEXT_T_DYN)
    context_group_freqs_sin.bind_dynamic(1, CP_CONTEXT_T_DYN)
    num_sampled.bind_dynamic(1, B_DYN)
    last_sampled.bind_dynamic(1, B_DYN)
    next_prefill_tokens.bind_dynamic(1, B_DYN)
    anchor_positions.bind_dynamic(1, B_DYN)
    block_tables.bind_dynamic(2, B_DYN)
    head_hidden.bind_dynamic(1, B_DYN)
    draft_token_ids.bind_dynamic(1, B_DYN)
    confidence_probs.bind_dynamic(1, B_DYN)

    hidden_gather_buf = pld.alloc_window_buffer([PREFILL_GROUP_CAP, D], dtype=pl.BF16)
    hidden_signal_buf = pld.alloc_window_buffer([DSPARK_CP_SIZE, 1], dtype=pl.INT32)
    attention_buf = pld.alloc_window_buffer([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
    attention_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_buf = pld.alloc_window_buffer([O_WINDOW_ROWS, D], dtype=pl.BF16)
    o_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    recv_route_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    markov_hidden_buf = pld.alloc_window_buffer(GROUP_LOGIT_ROWS * D * 2)
    markov_logits_buf = pld.alloc_window_buffer(MAX_LOGIT_ROWS * VOCAB * 4)
    markov_hidden_done_buf = pld.alloc_window_buffer(TP_SIZE * 4)
    markov_logits_done_buf = pld.alloc_window_buffer(TP_SIZE * 4)

    for rank in pl.range(pld.world_size()):
        hidden_gather_window = pld.window(
            hidden_gather_buf, [PREFILL_GROUP_CAP, D], dtype=pl.BF16
        )
        hidden_gather_signal = pld.window(
            hidden_signal_buf, [DSPARK_CP_SIZE, 1], dtype=pl.INT32
        )
        attention_window = pld.window(
            attention_buf, [ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16
        )
        attention_signal = pld.window(
            attention_signal_buf, [TP_SIZE, 1], dtype=pl.INT32
        )
        o_window = pld.window(o_buf, [O_WINDOW_ROWS, D], dtype=pl.BF16)
        o_signal = pld.window(o_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        recv_meta_window = pld.window(
            recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32
        )
        recv_x_window = pld.window(
            recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8
        )
        recv_aux_window = pld.window(
            recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32
        )
        recv_route_window = pld.window(
            recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32
        )
        arrived_window = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived_window = pld.window(
            data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32
        )
        routed_y_window = pld.window(routed_y_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_window = pld.window(combine_buf, [N_RANKS, 1], dtype=pl.INT32)
        markov_hidden_window = pld.window(
            markov_hidden_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16
        )
        markov_hidden_done = pld.window(
            markov_hidden_done_buf, [TP_SIZE, 1], dtype=pl.INT32
        )
        markov_logits_window = pld.window(
            markov_logits_buf, [MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32
        )
        markov_logits_done = pld.window(
            markov_logits_done_buf, [TP_SIZE, 1], dtype=pl.INT32
        )
        dspark_draft_step_device_state(
            target_hidden[rank], main_proj_weight[rank], main_norm_weight[rank],
            num_sampled[rank], last_sampled[rank], next_prefill_tokens[rank],
            embedding_weight[rank], context_group_position_ids[rank],
            context_group_slot_mapping[rank], anchor_positions[rank], block_tables[rank],
            query_group_position_ids[rank], query_group_slot_mapping[rank],
            context_group_freqs_cos[rank], context_group_freqs_sin[rank],
            query_freqs_cos[rank], query_freqs_sin[rank],
            query_group_freqs_cos[rank], query_group_freqs_sin[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], attn_norm_w[rank],
            wq_a[rank], wq_b[rank], wq_b_scale[rank], wkv[rank], gamma_cq[rank], gamma_ckv[rank],
            kv_caches[rank], attn_sink[rank], wo_a[rank], wo_b[rank], wo_b_scale[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank], ffn_norm_w[rank],
            gate_w[rank], gate_bias[rank], tid2eid[rank], routed_w1[rank],
            routed_w1_scale[rank], routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank], shared_w1[rank],
            shared_w1_scale[rank], shared_w3[rank], shared_w3_scale[rank],
            shared_w2[rank], shared_w2_scale[rank], hc_head_fn[rank],
            hc_head_scale[rank], hc_head_base[rank], initial_hidden[rank],
            intermediate_hidden[rank], head_hidden[rank],
            hidden_gather_window, hidden_gather_signal,
            attention_window, attention_signal, o_window, o_signal,
            recv_meta_window, recv_x_window, recv_aux_window, recv_route_window,
            arrived_window, data_arrived_window, routed_y_window, combine_window,
            rank // DSPARK_CP_SIZE * DSPARK_CP_SIZE, rank % DSPARK_CP_SIZE, rank,
            final_norm_weight[rank], lm_head_weight[rank], logit_row_indices[rank],
            markov_w1[rank], markov_w2[rank], confidence_head_weight[rank],
            draft_token_ids[rank], confidence_probs[rank],
            markov_hidden_window, markov_hidden_done,
            markov_logits_window, markov_logits_done,
            state_slot_ids[rank], state_generations[rank], state_tokens[rank],
            state_meta[rank], device=rank,
        )


import dspark_drafter as _drafter_module
import dspark_markov as _markov_module


def build_tensor_specs(batch: int):
    """Build one deterministic end-to-end drafter/Markov/state fixture."""
    import torch
    from golden import TensorSpec

    specs = _drafter_module.build_tensor_specs(batch, mode="decode")
    existing = {spec.name for spec in specs}
    markov_specs = _markov_module.build_tensor_specs(batch, distributed=True)
    for spec in markov_specs:
        markov_world = spec.shape[0]
        if markov_world != N_RANKS:
            spec.shape[0] = N_RANKS
            if callable(spec.init_value):
                init_value = spec.init_value

                def replicate_groups(init_value=init_value, markov_world=markov_world):
                    value = init_value()
                    repeats = [N_RANKS // markov_world] + [1] * (value.ndim - 1)
                    return value.repeat(*repeats)

                spec.init_value = replicate_groups
        if spec.name not in existing:
            specs.append(spec)
            existing.add(spec.name)

    def init_slots():
        slots = torch.full(
            (N_RANKS, DEVICE_STATE_LOCAL_BATCH), -1, dtype=torch.int32
        )
        slots[:, :batch] = torch.arange(batch, dtype=torch.int32)
        return slots

    def init_generations():
        generations = torch.full(
            (N_RANKS, DEVICE_STATE_LOCAL_BATCH), -1, dtype=torch.int32
        )
        generations[:, :batch] = 1
        return generations

    def init_state_meta():
        meta = torch.zeros(
            N_RANKS, STATE_CAPACITY, STATE_META_WIDTH, dtype=torch.int32
        )
        meta[:, :batch, STATE_VALID] = 1
        meta[:, :batch, STATE_GENERATION] = 1
        meta[:, :batch, STATE_POSITION_LIMIT] = _drafter_module.M.max_position_embeddings
        return meta

    specs.extend(
        [
            TensorSpec(
                "state_slot_ids",
                [N_RANKS, DEVICE_STATE_LOCAL_BATCH],
                torch.int32,
                init_value=init_slots,
            ),
            TensorSpec(
                "state_generations",
                [N_RANKS, DEVICE_STATE_LOCAL_BATCH],
                torch.int32,
                init_value=init_generations,
            ),
            TensorSpec(
                "state_tokens",
                [N_RANKS, STATE_CAPACITY, STATE_TOKEN_WIDTH],
                torch.int64,
                resident="stacked",
            ),
            TensorSpec(
                "state_meta",
                [N_RANKS, STATE_CAPACITY, STATE_META_WIDTH],
                torch.int32,
                init_value=init_state_meta,
                resident="stacked",
            ),
        ]
    )
    return specs


def golden_dspark_draft_step_device_state(tensors):
    """Apply the three existing golden stages in the fused program order."""
    import torch

    _drafter_module.golden_dspark_drafter(tensors)
    for rank in range(N_RANKS):
        group_base = rank // TP_SIZE * TP_SIZE
        lm_head_weight = torch.cat(
            [
                tensors["lm_head_weight"][group_base + tp_rank]
                for tp_rank in range(TP_SIZE)
            ],
            dim=0,
        )
        hidden_fp32 = tensors["head_hidden"][rank].float()
        inv_rms = torch.rsqrt(
            hidden_fp32.square().mean(dim=-1, keepdim=True)
            + _markov_module.EPS
        )
        normalized = (
            hidden_fp32
            * inv_rms
            * tensors["final_norm_weight"][rank].float()
        ).to(torch.bfloat16)
        support = 8
        base_logits = normalized.float().matmul(
            lm_head_weight[:support].float().t()
        )
        previous = torch.where(
            tensors["num_sampled"][rank] > 0,
            tensors["last_sampled"][rank],
            tensors["next_prefill_tokens"][rank],
        ).long()
        for step in range(DSPARK_QUERY_WIDTH):
            embedding = tensors["markov_w1"][rank].float().index_select(
                0, previous
            )
            confidence_input = torch.cat(
                [hidden_fp32[:, step], embedding], dim=-1
            )
            confidence_logits = confidence_input.matmul(
                tensors["confidence_head_weight"][rank].float().t()
            ).squeeze(-1)
            tensors["confidence_probs"][rank, :, step] = torch.sigmoid(
                confidence_logits
            )
            markov_bias = embedding.matmul(
                tensors["markov_w2"][rank, :support].float().t()
            )
            scores = base_logits[:, step] + markov_bias
            # Rows outside the deterministic support are zero. Include their
            # first row explicitly so argmax matches the full-vocabulary op.
            scores = torch.cat(
                [scores, torch.zeros(scores.shape[0], 1)], dim=-1
            )
            previous = torch.argmax(scores, dim=-1)
            tensors["draft_token_ids"][rank, :, step] = previous.to(
                torch.int32
            )
    for rank in range(N_RANKS):
        for request in range(tensors["draft_token_ids"].shape[1]):
            slot = int(tensors["state_slot_ids"][rank, request])
            generation = int(tensors["state_generations"][rank, request])
            if (
                slot >= 0
                and int(tensors["state_meta"][rank, slot, STATE_VALID]) == 1
                and int(tensors["state_meta"][rank, slot, STATE_GENERATION]) == generation
                and int(tensors["state_meta"][rank, slot, STATE_ANCHOR_POSITION])
                + DSPARK_QUERY_WIDTH
                < int(tensors["state_meta"][rank, slot, STATE_POSITION_LIMIT])
            ):
                tensors["state_tokens"][rank, slot, 1:] = tensors[
                    "draft_token_ids"
                ][rank, request].to(tensors["state_tokens"].dtype)
                tensors["state_meta"][rank, slot, STATE_DRAFT_COUNT] = DSPARK_QUERY_WIDTH


def main():
    import argparse
    from golden import run

    parser = argparse.ArgumentParser(
        description="Validate fused DSpark drafter, Markov, and device-state commit."
    )
    parser.add_argument("--batch", type=int, choices=(4, 8, 12, 16), default=4)
    parser.add_argument("--tp", type=int, choices=(4,), default=TP_SIZE)
    parser.add_argument("--ep", type=int, choices=(4, 16), default=N_RANKS)
    parser.add_argument("-p", "--platform", choices=("a2a3", "a2a3sim"), default="a2a3")
    parser.add_argument(
        "-d", "--device", default=",".join(str(rank) for rank in range(N_RANKS))
    )
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--dump-passes", action="store_true")
    args = parser.parse_args()

    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) < N_RANKS:
        parser.error(f"expected at least {N_RANKS} device ids")
    result = run(
        fn=l3_dspark_draft_step_device_state,
        specs=build_tensor_specs(args.batch),
        golden_fn=golden_dspark_draft_step_device_state,
        compile_only=args.compile_only,
        config=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:N_RANKS], num_sub_workers=0
            ),
            platform=args.platform,
            ring_heap=(4 * 1024 * 1024 * 1024,) * 4,
        ),
        rtol=2e-3,
        atol=2e-3,
        compare_fn={"kv_caches": _drafter_module._dspark_kv_cache_compare()},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
