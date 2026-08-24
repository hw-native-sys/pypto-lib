# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=8
"""Strict DeepSeek-V4 EPLB MTP-core interval."""

import argparse

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from eplb_fixture import (
    EPLB_EP_SIZE,
    EPLB_EXPERTS_PER_RANK,
    EPLB_START_POS,
    EPLB_TOKENS,
    EPLB_TP_SIZE,
    configure_eplb_argv,
    replace_eplb_routing_specs,
    validate_eplb_topology,
)

configure_eplb_argv()

import decode_mtp
from decode_swa import (
    B,
    BLOCK_SIZE,
    HEAD_DIM,
    H,
    MAX_SEQ_LEN,
    O_GROUP_IN,
    O_GROUPS,
    O_LORA,
    ORI_BLOCK_NUM,
    ORI_BLOCK_NUM_DYN,
    ORI_TABLE_MAX_BLOCKS,
    Q_LORA,
    ROPE_HEAD_DIM,
    T,
    WIN,
    attention_swa,
    golden_attention_swa,
)
from hc_head import golden_hc_head, hc_head
from lm_head import (
    GROUP_LOGIT_ROWS,
    MAX_LOGIT_ROWS,
    TP_SIZE as LM_HEAD_TP_SIZE,
    VOCAB as LM_HEAD_VOCAB,
    VOCAB_PER_TP,
    clear_lm_head_signals,
    golden_lm_head,
    lm_head_core,
)
from moe import (
    AUX_PAD,
    D,
    EXPERTS_PER_RANK,
    HC_DIM,
    HC_MULT,
    IDX_PAD,
    MIX_HC,
    MOE_INTER,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_RANKS,
    N_ROUTES,
    RECV_MAX,
    TOPK as MOE_TOPK,
    VOCAB as MOE_VOCAB,
    clear_moe_signals,
    golden_moe,
    moe,
)
from mtp_projection import golden_mtp_projection, mtp_projection
from rmsnorm import golden_rms_norm, rms_norm


# model config
EPLB_MTP_LAYER_ID = 0

# communication
MTP_MOE_EPOCH = 1
LM_HEAD_COMM_EPOCH = 1


validate_eplb_topology(
    ep_size=N_RANKS,
    tp_size=LM_HEAD_TP_SIZE,
    experts_per_rank=EXPERTS_PER_RANK,
    num_experts=N_EXPERTS_GLOBAL,
    tokens=T,
    topk=MOE_TOPK,
)


@pl.jit(auto_scope=False)
def eplb_mtp_core_logits(
    hidden_states: pl.Tensor[[T, D], pl.BF16],
    prev_pre_hc_hidden: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    swa_slot_mapping: pl.Tensor[[T], pl.INT64],
    swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    swa_lens: pl.Tensor[[T], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    enorm_w: pl.Tensor[[D], pl.FP32],
    hnorm_w: pl.Tensor[[D], pl.FP32],
    e_proj_w: pl.Tensor[[D, D], pl.INT8],
    e_proj_w_scale: pl.Tensor[[D], pl.FP32],
    e_proj_smooth: pl.Tensor[[D], pl.FP32],
    h_proj_w: pl.Tensor[[D, D], pl.INT8],
    h_proj_w_scale: pl.Tensor[[D], pl.FP32],
    h_proj_smooth: pl.Tensor[[D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[MOE_VOCAB, MOE_TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    mtp_hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    mtp_hc_head_scale: pl.Tensor[[1], pl.FP32],
    mtp_hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    mtp_norm_w: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    hidden_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    next_pre_hc_hidden: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    lm_head_hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    lm_head_hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    lm_head_logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32],
    lm_head_logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor[[MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]:
    # Main MTP now carries SWA and compressed RoPE profiles together.  The
    # EPLB interval still consumes the SWA profile through attention_swa's
    # two-dimensional contract.
    swa_cos_profile: pl.Tensor[[1, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.slice(
        freqs_cos, [1, MAX_SEQ_LEN, ROPE_HEAD_DIM], [0, 0, 0]
    )
    swa_sin_profile: pl.Tensor[[1, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.slice(
        freqs_sin, [1, MAX_SEQ_LEN, ROPE_HEAD_DIM], [0, 0, 0]
    )
    swa_freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.reshape(
        swa_cos_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM]
    )
    swa_freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.reshape(
        swa_sin_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM]
    )
    projected_hidden = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
    with pl.scope():
        mtp_projection(
            hidden_states, prev_pre_hc_hidden,
            enorm_w, hnorm_w,
            e_proj_w, e_proj_w_scale, e_proj_smooth,
            h_proj_w, h_proj_w_scale, h_proj_smooth,
            projected_hidden,
        )

    x_attn = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
    with pl.scope():
        attention_swa(
            projected_hidden,
            hc_attn_fn, hc_attn_scale, hc_attn_base,
            attn_norm_w,
            wq_a, wq_b, wq_b_scale,
            wkv, gamma_cq, gamma_ckv,
            swa_freqs_cos, swa_freqs_sin,
            kv_cache,
            swa_slot_mapping, swa_indices, swa_lens, position_ids,
            attn_sink, wo_a, wo_b, wo_b_scale,
            x_attn,
        )

    with pl.scope():
        moe(
            x_attn,
            hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
            norm_w, gate_w, gate_bias, tid2eid, input_ids,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
            shared_w2, shared_w2_scale,
            next_pre_hc_hidden,
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            pl.cast(EPLB_MTP_LAYER_ID, pl.INT32), num_tokens, my_rank,
            pl.cast(MTP_MOE_EPOCH, pl.INT32),
        )

    x_head = pl.create_tensor([T, D], dtype=pl.BF16)
    with pl.scope():
        hc_head(next_pre_hc_hidden, mtp_hc_head_fn, mtp_hc_head_scale, mtp_hc_head_base, x_head)
        rms_norm(x_head, mtp_norm_w, hidden_out)

    with pl.scope():
        lm_head_core(
            hidden_out, lm_head_weight, logit_row_indices, logits,
            lm_head_hidden_window, lm_head_hidden_done,
            lm_head_logits_window, lm_head_logits_done,
            my_rank // LM_HEAD_TP_SIZE * LM_HEAD_TP_SIZE,
            my_rank % LM_HEAD_TP_SIZE,
            LM_HEAD_COMM_EPOCH,
        )
    return logits


@pl.jit(auto_scope=False)
def eplb_mtp_core_cleanup(
    next_pre_hc_hidden: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    logits: pl.Tensor[[MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    lm_head_hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    lm_head_logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
) -> pl.Tensor[[MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]:
    clear_moe_signals(next_pre_hc_hidden, arrived, data_arrived, combine_arrived)
    clear_lm_head_signals(logits, lm_head_hidden_done, lm_head_logits_done)
    return logits


@pl.jit.host
def l3_eplb_mtp_core(
    hidden_states: pl.Tensor[[N_RANKS, T, D], pl.BF16],
    prev_pre_hc_hidden: pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32],
    swa_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    swa_indices: pl.Tensor[[N_RANKS, T, WIN], pl.INT32],
    swa_lens: pl.Tensor[[N_RANKS, T], pl.INT32],
    position_ids: pl.Tensor[[N_RANKS, T], pl.INT32],
    enorm_w: pl.Tensor[[N_RANKS, D], pl.FP32],
    hnorm_w: pl.Tensor[[N_RANKS, D], pl.FP32],
    e_proj_w: pl.Tensor[[N_RANKS, D, D], pl.INT8],
    e_proj_w_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    e_proj_smooth: pl.Tensor[[N_RANKS, D], pl.FP32],
    h_proj_w: pl.Tensor[[N_RANKS, D, D], pl.INT8],
    h_proj_w_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    h_proj_smooth: pl.Tensor[[N_RANKS, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[N_RANKS, 2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, 2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    attn_sink: pl.Tensor[[N_RANKS, H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, MOE_VOCAB, MOE_TOPK], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, T], pl.INT64],
    routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    mtp_hc_head_fn: pl.Tensor[[N_RANKS, HC_MULT, HC_DIM], pl.FP32],
    mtp_hc_head_scale: pl.Tensor[[N_RANKS, 1], pl.FP32],
    mtp_hc_head_base: pl.Tensor[[N_RANKS, HC_MULT], pl.FP32],
    mtp_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[N_RANKS, VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS], pl.INT32],
    hidden_out: pl.Out[pl.Tensor[[N_RANKS, T, D], pl.BF16]],
    next_pre_hc_hidden: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32]],
    logits: pl.Out[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    recv_route_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    lm_head_hidden_window_buf = pld.alloc_window_buffer([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    lm_head_logits_window_buf = pld.alloc_window_buffer([MAX_LOGIT_ROWS, LM_HEAD_VOCAB], dtype=pl.FP32)
    lm_head_hidden_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    lm_head_logits_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)

    for r in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        lm_head_hidden_window = pld.window(lm_head_hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        lm_head_hidden_done = pld.window(lm_head_hidden_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        lm_head_logits_window = pld.window(lm_head_logits_window_buf, [MAX_LOGIT_ROWS, LM_HEAD_VOCAB], dtype=pl.FP32)
        lm_head_logits_done = pld.window(lm_head_logits_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        eplb_mtp_core_logits(
            hidden_states[r], prev_pre_hc_hidden[r],
            swa_slot_mapping[r], swa_indices[r], swa_lens[r],
            position_ids[r],
            enorm_w[r], hnorm_w[r],
            e_proj_w[r], e_proj_w_scale[r], e_proj_smooth[r],
            h_proj_w[r], h_proj_w_scale[r], h_proj_smooth[r],
            hc_attn_fn[r], hc_attn_scale[r], hc_attn_base[r],
            attn_norm_w[r],
            wq_a[r], wq_b[r], wq_b_scale[r],
            wkv[r], gamma_cq[r], gamma_ckv[r],
            freqs_cos[r], freqs_sin[r],
            kv_cache[r],
            attn_sink[r], wo_a[r], wo_b[r], wo_b_scale[r],
            hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r],
            norm_w[r], gate_w[r], gate_bias[r], tid2eid[r], input_ids[r],
            routed_w1[r], routed_w1_scale[r], routed_w3[r], routed_w3_scale[r],
            routed_w2[r], routed_w2_scale[r],
            shared_w1[r], shared_w1_scale[r], shared_w3[r], shared_w3_scale[r],
            shared_w2[r], shared_w2_scale[r],
            mtp_hc_head_fn[r], mtp_hc_head_scale[r], mtp_hc_head_base[r], mtp_norm_w[r],
            lm_head_weight[r], logit_row_indices[r],
            hidden_out[r], next_pre_hc_hidden[r], logits[r],
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            lm_head_hidden_window, lm_head_hidden_done,
            lm_head_logits_window, lm_head_logits_done,
            r, num_tokens,
            device=r,
        )

    for r in pl.range(pld.world_size()):
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        lm_head_hidden_done = pld.window(lm_head_hidden_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        lm_head_logits_done = pld.window(lm_head_logits_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        eplb_mtp_core_cleanup(
            next_pre_hc_hidden[r], logits[r],
            arrived, data_arrived, combine_arrived,
            lm_head_hidden_done, lm_head_logits_done,
            device=r,
        )


def build_tensor_specs(
    start_pos=EPLB_START_POS,
    num_tokens=EPLB_TOKENS,
    ori_block_num=ORI_BLOCK_NUM,
):
    import torch
    from golden import TensorSpec
    from utils import block_table, paged_slot_mapping, swa_indices_and_lens

    validate_eplb_topology(
        ep_size=N_RANKS,
        tp_size=LM_HEAD_TP_SIZE,
        experts_per_rank=EXPERTS_PER_RANK,
        num_experts=N_EXPERTS_GLOBAL,
        tokens=num_tokens,
        topk=MOE_TOPK,
        start_pos=start_pos,
    )
    base_specs = decode_mtp.build_tensor_specs(start_pos=start_pos, num_tokens=num_tokens, ori_block_num=ori_block_num)
    base_specs = replace_eplb_routing_specs(base_specs, active_tokens=num_tokens)
    base_by_name = {spec.name: spec for spec in base_specs}
    position_spec = base_by_name["position_ids"]
    prepared_metadata = None

    def init_hidden_states():
        return torch.randn(N_RANKS, T, D).to(torch.bfloat16)

    def init_prev_pre_hc_hidden():
        return torch.randn(N_RANKS, T, HC_MULT, D)

    def init_prepared_metadata():
        nonlocal prepared_metadata
        if prepared_metadata is not None:
            return prepared_metadata

        positions = position_spec.create_tensor()
        table = block_table(batch=B, table_blocks=ORI_TABLE_MAX_BLOCKS, physical_blocks=ori_block_num)
        slot_mappings = []
        indices = []
        lens = []
        for rank in range(N_RANKS):
            rank_positions = positions[rank].reshape(B, T // B)
            rank_slots = paged_slot_mapping(rank_positions, table, block_size=BLOCK_SIZE).reshape(-1)
            rank_indices, rank_lens = swa_indices_and_lens(
                rank_positions, table, block_size=BLOCK_SIZE, window=WIN,
            )
            slot_mappings.append(rank_slots.contiguous())
            indices.append(rank_indices.contiguous())
            lens.append(rank_lens.contiguous())
        slot_mapping = torch.stack(slot_mappings, dim=0).to(torch.int64).contiguous()
        swa_index = torch.stack(indices, dim=0).to(torch.int32).contiguous()
        swa_len = torch.stack(lens, dim=0).to(torch.int32).contiguous()

        cache_rows = ori_block_num * BLOCK_SIZE
        valid_slots = slot_mapping[slot_mapping >= 0]
        valid_indices = swa_index[swa_index >= 0]
        if valid_slots.numel() and int(valid_slots.max()) >= cache_rows:
            raise ValueError("prepared SWA slot mapping exceeds the requested KV-cache capacity")
        if valid_indices.numel() and int(valid_indices.max()) >= cache_rows:
            raise ValueError("prepared SWA indices exceed the requested KV-cache capacity")
        if bool((swa_len < 0).any()) or bool((swa_len > WIN).any()):
            raise ValueError("prepared SWA lengths must be in the inclusive range [0, WIN]")

        prepared_metadata = slot_mapping, swa_index, swa_len
        return prepared_metadata

    def init_swa_slot_mapping():
        return init_prepared_metadata()[0]

    def init_swa_indices():
        return init_prepared_metadata()[1]

    def init_swa_lens():
        return init_prepared_metadata()[2]

    prepared_specs = {
        "hidden_states": TensorSpec(
            "hidden_states", [N_RANKS, T, D], torch.bfloat16,
            init_value=init_hidden_states, resident="stacked",
        ),
        "prev_pre_hc_hidden": TensorSpec(
            "prev_pre_hc_hidden", [N_RANKS, T, HC_MULT, D], torch.float32,
            init_value=init_prev_pre_hc_hidden, resident="stacked",
        ),
        "swa_slot_mapping": TensorSpec(
            "swa_slot_mapping", [N_RANKS, T], torch.int64,
            init_value=init_swa_slot_mapping, resident="stacked",
        ),
        "swa_indices": TensorSpec(
            "swa_indices", [N_RANKS, T, WIN], torch.int32,
            init_value=init_swa_indices, resident="stacked",
        ),
        "swa_lens": TensorSpec(
            "swa_lens", [N_RANKS, T], torch.int32,
            init_value=init_swa_lens, resident="stacked",
        ),
    }
    ordered_names = [
        "hidden_states", "prev_pre_hc_hidden",
        "swa_slot_mapping", "swa_indices", "swa_lens",
        "position_ids",
        "enorm_w", "hnorm_w",
        "e_proj_w", "e_proj_w_scale", "e_proj_smooth",
        "h_proj_w", "h_proj_w_scale", "h_proj_smooth",
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
        "attn_norm_w",
        "wq_a", "wq_b", "wq_b_scale",
        "wkv", "gamma_cq", "gamma_ckv",
        "freqs_cos", "freqs_sin",
        "kv_cache",
        "attn_sink", "wo_a", "wo_b", "wo_b_scale",
        "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base",
        "norm_w", "gate_w", "gate_bias", "tid2eid", "input_ids",
        "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
        "routed_w2", "routed_w2_scale",
        "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
        "shared_w2", "shared_w2_scale",
        "mtp_hc_head_fn", "mtp_hc_head_scale", "mtp_hc_head_base", "mtp_norm_w",
        "lm_head_weight", "logit_row_indices",
        "hidden_out", "next_pre_hc_hidden", "logits", "num_tokens",
    ]
    specs = []
    for name in ordered_names:
        if name in prepared_specs:
            specs.append(prepared_specs[name])
        elif name in base_by_name:
            specs.append(base_by_name[name])
        else:
            raise ValueError(f"missing base TensorSpec for EPLB MTP-core parameter {name!r}")
    return specs


def _golden_lm_head_groups(tensors):
    for group_base in range(0, N_RANKS, LM_HEAD_TP_SIZE):
        group_end = group_base + LM_HEAD_TP_SIZE
        golden_lm_head({
            "hidden_states": tensors["hidden_out"][group_base:group_end],
            "lm_head_weight": tensors["lm_head_weight"][group_base:group_end],
            "logit_row_indices": tensors["logit_row_indices"][group_base:group_end],
            "logits": tensors["logits"][group_base:group_end],
        })


def golden_eplb_mtp_core(tensors):
    import torch

    projected_hidden = torch.empty_like(tensors["prev_pre_hc_hidden"])
    for rank in range(N_RANKS):
        golden_mtp_projection({
            "hidden_states": tensors["hidden_states"][rank],
            "prev_hidden_states": tensors["prev_pre_hc_hidden"][rank],
            "enorm_w": tensors["enorm_w"][rank],
            "hnorm_w": tensors["hnorm_w"][rank],
            "e_proj_w": tensors["e_proj_w"][rank],
            "e_proj_w_scale": tensors["e_proj_w_scale"][rank],
            "e_proj_smooth": tensors["e_proj_smooth"][rank],
            "h_proj_w": tensors["h_proj_w"][rank],
            "h_proj_w_scale": tensors["h_proj_w_scale"][rank],
            "h_proj_smooth": tensors["h_proj_smooth"][rank],
            "hidden_states_out": projected_hidden[rank],
        })

    x_attn = torch.empty_like(projected_hidden)
    for rank in range(N_RANKS):
        golden_attention_swa({
            "x_hc": projected_hidden[rank],
            "hc_attn_fn": tensors["hc_attn_fn"][rank],
            "hc_attn_scale": tensors["hc_attn_scale"][rank],
            "hc_attn_base": tensors["hc_attn_base"][rank],
            "attn_norm_w": tensors["attn_norm_w"][rank],
            "wq_a": tensors["wq_a"][rank],
            "wq_b": tensors["wq_b"][rank],
            "wq_b_scale": tensors["wq_b_scale"][rank],
            "wkv": tensors["wkv"][rank],
            "gamma_cq": tensors["gamma_cq"][rank],
            "gamma_ckv": tensors["gamma_ckv"][rank],
            "freqs_cos": tensors["freqs_cos"][rank, 0],
            "freqs_sin": tensors["freqs_sin"][rank, 0],
            "kv_cache": tensors["kv_cache"][rank],
            "swa_slot_mapping": tensors["swa_slot_mapping"][rank],
            "swa_indices": tensors["swa_indices"][rank],
            "swa_lens": tensors["swa_lens"][rank],
            "position_ids": tensors["position_ids"][rank],
            "attn_sink": tensors["attn_sink"][rank],
            "wo_a": tensors["wo_a"][rank],
            "wo_b": tensors["wo_b"][rank],
            "wo_b_scale": tensors["wo_b_scale"][rank],
            "x_out": x_attn[rank],
        })

    moe_tensors = dict(tensors)
    moe_tensors["x_hc"] = x_attn
    moe_tensors["x_next"] = tensors["next_pre_hc_hidden"]
    moe_tensors["layer_id"] = EPLB_MTP_LAYER_ID
    moe_tensors["num_tokens"] = int(tensors["num_tokens"])
    golden_moe(moe_tensors)

    for rank in range(N_RANKS):
        x_head = torch.empty_like(tensors["hidden_out"][rank])
        golden_hc_head({
            "x_hc": tensors["next_pre_hc_hidden"][rank],
            "hc_head_fn": tensors["mtp_hc_head_fn"][rank],
            "hc_head_scale": tensors["mtp_hc_head_scale"][rank],
            "hc_head_base": tensors["mtp_hc_head_base"][rank],
            "y": x_head,
        })
        tensors["hidden_out"][rank] = golden_rms_norm(x_head, tensors["mtp_norm_w"][rank])

    _golden_lm_head_groups(tensors)


def golden_finite_smoke(_tensors):
    return None


def finite_tensor_compare(actual, _expected, **_context):
    import torch

    finite = torch.isfinite(actual)
    if bool(finite.all()):
        return True, ""
    invalid = int((~finite).sum().item())
    return False, f"{invalid}/{actual.numel()} values are non-finite"


def finite_output_compare_map(specs):
    return {
        spec.name: finite_tensor_compare
        for spec in specs
        if getattr(spec, "is_output", False)
    }


def main():
    from golden import ratio_reldiff, run_jit

    parser = argparse.ArgumentParser(description="DeepSeek-V4 strict EPLB MTP-core benchmark.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--ep", type=int, default=EPLB_EP_SIZE, choices=[EPLB_EP_SIZE])
    parser.add_argument("--tp", type=int, default=EPLB_TP_SIZE, choices=[EPLB_TP_SIZE])
    parser.add_argument(
        "--experts-per-rank", type=int, default=EPLB_EXPERTS_PER_RANK, choices=[EPLB_EXPERTS_PER_RANK]
    )
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(N_RANKS)))
    parser.add_argument("--start-pos", type=int, default=EPLB_START_POS, choices=[EPLB_START_POS])
    parser.add_argument("--num-tokens", type=int, default=EPLB_TOKENS, choices=[EPLB_TOKENS])
    parser.add_argument("--ori-block-num", type=int, default=ORI_BLOCK_NUM)
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    parser.add_argument("--finite-only", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(device_id) for device_id in args.device.split(",")]
    validate_eplb_topology(
        ep_size=args.ep,
        tp_size=args.tp,
        experts_per_rank=args.experts_per_rank,
        num_experts=N_EXPERTS_GLOBAL,
        tokens=args.num_tokens,
        topk=MOE_TOPK,
        start_pos=args.start_pos,
    )
    if len(device_ids) != EPLB_EP_SIZE:
        raise ValueError(f"EPLB benchmark needs exactly {EPLB_EP_SIZE} devices, got {device_ids}")

    specs = build_tensor_specs(
        start_pos=args.start_pos,
        num_tokens=args.num_tokens,
        ori_block_num=args.ori_block_num,
    )
    compare_fn = {
        "hidden_out": ratio_reldiff(diff_thd=0.02, pct_thd=0.10),
        "next_pre_hc_hidden": ratio_reldiff(diff_thd=0.02, pct_thd=0.05),
        "kv_cache": ratio_reldiff(diff_thd=0.01, pct_thd=0.05),
        "logits": ratio_reldiff(diff_thd=0.02, pct_thd=0.10),
    }
    golden_fn = golden_eplb_mtp_core
    if args.finite_only:
        compare_fn = finite_output_compare_map(specs)
        golden_fn = golden_finite_smoke

    result = run_jit(
        fn=l3_eplb_mtp_core,
        specs=specs,
        golden_fn=golden_fn,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=device_ids[:N_RANKS], num_sub_workers=0),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn=compare_fn,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
