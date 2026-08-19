# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""DeepSeek-V4 MTP decode layer.

This is the MTP draft-model analogue of ``decode_layer.py`` for the decode-time
SWA path:

    current hidden + previous pre-hc hidden
        -> MTP projection
        -> SWA attention
        -> MoE/FFN
        -> next pre-hc hidden
        -> MTP hc_head
        -> RMSNorm
        -> hidden output

The pre-hc hidden tensors keep the projection/prefill MTP layout
``[T, HC_MULT, D]`` so serving can feed the returned state into the next draft
step. The normalized hidden output is ready for logits computation.
"""

import argparse

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from config import ACTIVE as M, DECODE_START_POS
from decode_attention_csa import COMPRESS_RATIO as CSA_COMPRESS_RATIO
from decode_attention_swa import (
    BLOCK_SIZE,
    HEAD_DIM,
    H,
    MAX_SEQ_LEN,
    O_GROUP_IN,
    O_GROUPS,
    O_LORA,
    ORI_BLOCK_NUM,
    Q_LORA,
    ROPE_HEAD_DIM,
    T,
    WIN,
    attention_swa,
    build_tensor_specs as build_swa_tensor_specs,
    golden_attention_swa,
)
from hc_head import golden_hc_head, hc_head
from moe import (
    AUX_PAD,
    D,
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
    _token_partition_ratio_reldiff,
    build_tensor_specs as build_moe_tensor_specs,
    golden_moe,
    moe,
)
from mtp_projection import (
    golden_mtp_projection,
    mtp_projection,
)
from rmsnorm import golden_rms_norm, rms_norm


MTP_LAYER_ID = M.num_hidden_layers
MTP_MOE_EPOCH = 1


@pl.jit
def mtp_decode_layer(
    hidden_states: pl.Tensor[[T, D], pl.BF16],
    prev_pre_hc_hidden: pl.Tensor[[T, HC_MULT, D], pl.FP32],
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
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    swa_slot_mapping: pl.Tensor[[T], pl.INT64],
    swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    swa_lens: pl.Tensor[[T], pl.INT32],
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
    hidden_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    next_pre_hc_hidden: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pl.InOut[pld.DistributedTensor[[N_RANKS, 1], pl.INT32]],
    my_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, HC_MULT, D], pl.BF16]:
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
    mtp_projection(
        hidden_states,
        prev_pre_hc_hidden,
        enorm_w,
        hnorm_w,
        e_proj_w,
        e_proj_w_scale,
        e_proj_smooth,
        h_proj_w,
        h_proj_w_scale,
        h_proj_smooth,
        projected_hidden,
    )
    x_attn = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
    attention_swa(
        projected_hidden,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_norm_w,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        gamma_cq,
        gamma_ckv,
        swa_freqs_cos,
        swa_freqs_sin,
        kv_cache,
        swa_slot_mapping,
        swa_indices,
        swa_lens,
        position_ids,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        x_attn,
    )
    moe(
        x_attn,
        hc_ffn_fn,
        hc_ffn_scale,
        hc_ffn_base,
        norm_w,
        gate_w,
        gate_bias,
        tid2eid,
        input_ids,
        routed_w1,
        routed_w1_scale,
        routed_w3,
        routed_w3_scale,
        routed_w2,
        routed_w2_scale,
        shared_w1,
        shared_w1_scale,
        shared_w3,
        shared_w3_scale,
        shared_w2,
        shared_w2_scale,
        next_pre_hc_hidden,
        recv_meta,
        recv_x,
        recv_aux,
        recv_route,
        arrived,
        data_arrived,
        routed_y_buf,
        combine_arrived,
        pl.cast(MTP_LAYER_ID, pl.INT32),
        num_tokens,
        my_rank,
        pl.cast(MTP_MOE_EPOCH, pl.INT32),
    )

    # Retire this dispatch's persistent MoE signal credits.
    neg_epochs = pl.cast(0 - MTP_MOE_EPOCH, pl.INT32)
    neg_data = pl.cast(0 - MTP_MOE_EPOCH * N_LOCAL, pl.INT32)
    neg_combine = pl.cast(0 - MTP_MOE_EPOCH * (N_LOCAL + 1), pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_signal_retire"):
        _pre_hc_anchor = pl.read(next_pre_hc_hidden, [0, 0, 0])
        pld.system.notify(
            target=combine_arrived,
            peer=my_rank,
            offsets=[my_rank, 0],
            value=neg_epochs,
            op=pld.NotifyOp.AtomicAdd,
        )
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.notify(
                    target=arrived,
                    peer=my_rank,
                    offsets=[src, 0],
                    value=neg_epochs,
                    op=pld.NotifyOp.AtomicAdd,
                )
                pld.system.notify(
                    target=data_arrived,
                    peer=my_rank,
                    offsets=[src, 0],
                    value=neg_data,
                    op=pld.NotifyOp.AtomicAdd,
                )
                pld.system.notify(
                    target=combine_arrived,
                    peer=my_rank,
                    offsets=[src, 0],
                    value=neg_combine,
                    op=pld.NotifyOp.AtomicAdd,
                )
    x_head = pl.create_tensor([T, D], dtype=pl.BF16)
    hc_head(next_pre_hc_hidden, mtp_hc_head_fn, mtp_hc_head_scale, mtp_hc_head_base, x_head)
    rms_norm(x_head, mtp_norm_w, hidden_out)
    return hidden_out


@pl.jit.host
def l3_mtp_decode_layer(
    hidden_states: pl.Tensor[[N_RANKS, T, D], pl.BF16],
    prev_pre_hc_hidden: pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32],
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
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    swa_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    swa_indices: pl.Tensor[[N_RANKS, T, WIN], pl.INT32],
    swa_lens: pl.Tensor[[N_RANKS, T], pl.INT32],
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
    hidden_out: pl.Out[pl.Tensor[[N_RANKS, T, D], pl.BF16]],
    next_pre_hc_hidden: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32]],
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

    for r in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        mtp_decode_layer(
            hidden_states[r],
            prev_pre_hc_hidden[r],
            position_ids[r],
            enorm_w[r], hnorm_w[r],
            e_proj_w[r], e_proj_w_scale[r], e_proj_smooth[r],
            h_proj_w[r], h_proj_w_scale[r], h_proj_smooth[r],
            hc_attn_fn[r], hc_attn_scale[r], hc_attn_base[r], attn_norm_w[r],
            wq_a[r], wq_b[r], wq_b_scale[r], wkv[r], gamma_cq[r], gamma_ckv[r],
            freqs_cos[r], freqs_sin[r],
            kv_cache[r], swa_slot_mapping[r], swa_indices[r], swa_lens[r],
            attn_sink[r], wo_a[r], wo_b[r], wo_b_scale[r],
            hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r], norm_w[r],
            gate_w[r], gate_bias[r], tid2eid[r], input_ids[r],
            routed_w1[r], routed_w1_scale[r], routed_w3[r], routed_w3_scale[r],
            routed_w2[r], routed_w2_scale[r],
            shared_w1[r], shared_w1_scale[r], shared_w3[r], shared_w3_scale[r],
            shared_w2[r], shared_w2_scale[r],
            mtp_hc_head_fn[r], mtp_hc_head_scale[r], mtp_hc_head_base[r], mtp_norm_w[r],
            hidden_out[r],
            next_pre_hc_hidden[r],
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
            routed_y_buf, combine_arrived,
            r, num_tokens,
            device=r,
        )


def _ranked_init(single_spec, *, replicated=False):
    import torch

    def init():
        if replicated:
            value = single_spec.create_tensor()
            return value.unsqueeze(0).expand(N_RANKS, *value.shape).contiguous()
        return torch.stack([single_spec.create_tensor() for _ in range(N_RANKS)], dim=0)

    return init


def _ranked_spec(name, spec, *, replicated=False, is_output=False):
    from golden import TensorSpec

    return TensorSpec(
        name,
        [N_RANKS, *spec.shape],
        spec.dtype,
        init_value=_ranked_init(spec, replicated=replicated),
        is_output=is_output,
    )


def _make_rope_profile_spec(name, swa_spec, csa_spec):
    import torch
    from golden import TensorSpec

    if tuple(swa_spec.shape) != tuple(csa_spec.shape):
        raise ValueError(
            f"{name} leaf shapes must match, got SWA {swa_spec.shape} and CSA {csa_spec.shape}"
        )
    if swa_spec.dtype != csa_spec.dtype:
        raise ValueError(
            f"{name} leaf dtypes must match, got SWA {swa_spec.dtype} and CSA {csa_spec.dtype}"
        )

    def init_value():
        return torch.stack(
            (swa_spec.create_tensor(), csa_spec.create_tensor()), dim=0
        ).contiguous()

    return TensorSpec(
        name,
        [2, *swa_spec.shape],
        swa_spec.dtype,
        init_value=init_value,
    )


def _projection_specs():
    import torch
    from golden import TensorSpec
    from mtp_projection import _quantize_weight_per_out

    e_proj_cache = None
    h_proj_cache = None

    def init_proj_pair():
        weights = []
        scales = []
        for _ in range(N_RANKS):
            w = (torch.rand(D, D) / D ** 0.5).to(torch.bfloat16)
            w_i8, scale = _quantize_weight_per_out(w)
            weights.append(w_i8)
            scales.append(scale.float())
        return torch.stack(weights, dim=0).contiguous(), torch.stack(scales, dim=0).contiguous()

    def init_e_proj_w():
        nonlocal e_proj_cache
        e_proj_cache = init_proj_pair()
        return e_proj_cache[0]

    def init_e_proj_w_scale():
        nonlocal e_proj_cache
        if e_proj_cache is None:
            e_proj_cache = init_proj_pair()
        return e_proj_cache[1]

    def init_h_proj_w():
        nonlocal h_proj_cache
        h_proj_cache = init_proj_pair()
        return h_proj_cache[0]

    def init_h_proj_w_scale():
        nonlocal h_proj_cache
        if h_proj_cache is None:
            h_proj_cache = init_proj_pair()
        return h_proj_cache[1]

    return {
        "hidden_states": TensorSpec(
            "hidden_states",
            [N_RANKS, T, D],
            torch.bfloat16,
            init_value=lambda: torch.randn(N_RANKS, T, D).to(torch.bfloat16),
        ),
        "prev_pre_hc_hidden": TensorSpec(
            "prev_pre_hc_hidden",
            [N_RANKS, T, HC_MULT, D],
            torch.float32,
            init_value=lambda: torch.randn(N_RANKS, T, HC_MULT, D).to(torch.bfloat16),
        ),
        "enorm_w": TensorSpec("enorm_w", [N_RANKS, D], torch.float32, init_value=lambda: torch.ones(N_RANKS, D)),
        "hnorm_w": TensorSpec("hnorm_w", [N_RANKS, D], torch.float32, init_value=lambda: torch.ones(N_RANKS, D)),
        "e_proj_w": TensorSpec("e_proj_w", [N_RANKS, D, D], torch.int8, init_value=init_e_proj_w),
        "e_proj_w_scale": TensorSpec("e_proj_w_scale", [N_RANKS, D], torch.float32, init_value=init_e_proj_w_scale),
        "e_proj_smooth": TensorSpec("e_proj_smooth", [N_RANKS, D], torch.float32, init_value=lambda: torch.ones(N_RANKS, D)),
        "h_proj_w": TensorSpec("h_proj_w", [N_RANKS, D, D], torch.int8, init_value=init_h_proj_w),
        "h_proj_w_scale": TensorSpec("h_proj_w_scale", [N_RANKS, D], torch.float32, init_value=init_h_proj_w_scale),
        "h_proj_smooth": TensorSpec("h_proj_smooth", [N_RANKS, D], torch.float32, init_value=lambda: torch.ones(N_RANKS, D)),
    }


def _mtp_head_specs():
    import torch
    from golden import TensorSpec

    base = [5.9166, -3.6223, -2.9324, -3.3124]
    return {
        "mtp_hc_head_fn": TensorSpec(
            "mtp_hc_head_fn",
            [N_RANKS, HC_MULT, HC_DIM],
            torch.float32,
            init_value=lambda: torch.randn(N_RANKS, HC_MULT, HC_DIM) * 0.0519,
        ),
        "mtp_hc_head_scale": TensorSpec(
            "mtp_hc_head_scale",
            [N_RANKS, 1],
            torch.float32,
            init_value=lambda: torch.full((N_RANKS, 1), 0.076099, dtype=torch.float32),
        ),
        "mtp_hc_head_base": TensorSpec(
            "mtp_hc_head_base",
            [N_RANKS, HC_MULT],
            torch.float32,
            init_value=lambda: torch.tensor(base, dtype=torch.float32)
            .view(1, HC_MULT)
            .expand(N_RANKS, -1)
            .contiguous(),
        ),
        "mtp_norm_w": TensorSpec(
            "mtp_norm_w",
            [N_RANKS, D],
            torch.bfloat16,
            init_value=lambda: (torch.randn(N_RANKS, D) * 0.1 + 1.0).to(torch.bfloat16),
        ),
    }


def build_tensor_specs(start_pos=DECODE_START_POS, num_tokens=T):
    import torch
    from golden import ScalarSpec, TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables

    projection_specs = _projection_specs()
    mtp_head_specs = _mtp_head_specs()
    swa_specs = {
        spec.name: spec
        for spec in build_swa_tensor_specs(start_pos)
        if isinstance(spec, TensorSpec)
    }
    compressed_freqs_cos, compressed_freqs_sin = build_deepseek_v4_rope_tables(
        M, CSA_COMPRESS_RATIO, dtype=torch.bfloat16
    )
    compressed_rope_specs = {
        "freqs_cos": TensorSpec(
            "freqs_cos",
            list(compressed_freqs_cos.shape),
            compressed_freqs_cos.dtype,
            init_value=lambda: compressed_freqs_cos.clone(),
        ),
        "freqs_sin": TensorSpec(
            "freqs_sin",
            list(compressed_freqs_sin.shape),
            compressed_freqs_sin.dtype,
            init_value=lambda: compressed_freqs_sin.clone(),
        ),
    }
    rope_specs = {
        name: _make_rope_profile_spec(name, swa_specs[name], compressed_rope_specs[name])
        for name in ("freqs_cos", "freqs_sin")
    }
    moe_specs = {
        spec.name: spec
        for spec in build_moe_tensor_specs(layer_id=MTP_LAYER_ID, num_tokens=num_tokens)
        if isinstance(spec, TensorSpec)
    }

    replicated_attention = {
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "attn_norm_w",
        "wq_a",
        "wq_b",
        "wq_b_scale",
        "wkv",
        "gamma_cq",
        "gamma_ckv",
        "freqs_cos",
        "freqs_sin",
        "attn_sink",
        "wo_a",
        "wo_b",
        "wo_b_scale",
    }
    ordered_names = [
        "hidden_states", "prev_pre_hc_hidden", "position_ids",
        "enorm_w", "hnorm_w", "e_proj_w", "e_proj_w_scale", "e_proj_smooth",
        "h_proj_w", "h_proj_w_scale", "h_proj_smooth",
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
        "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
        "freqs_cos", "freqs_sin", "kv_cache",
        "swa_slot_mapping", "swa_indices", "swa_lens",
        "attn_sink", "wo_a", "wo_b", "wo_b_scale",
        "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
        "gate_w", "gate_bias", "tid2eid", "input_ids",
        "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
        "routed_w2", "routed_w2_scale",
        "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
        "shared_w2", "shared_w2_scale",
        "mtp_hc_head_fn", "mtp_hc_head_scale", "mtp_hc_head_base", "mtp_norm_w",
    ]

    specs = []
    for name in ordered_names:
        if name in projection_specs:
            specs.append(projection_specs[name])
        elif name in mtp_head_specs:
            specs.append(mtp_head_specs[name])
        elif name in moe_specs:
            specs.append(moe_specs[name])
        elif name in rope_specs:
            specs.append(_ranked_spec(name, rope_specs[name], replicated=True))
        else:
            specs.append(
                _ranked_spec(
                    name,
                    swa_specs[name],
                    replicated=name in replicated_attention,
                    is_output=swa_specs[name].is_output,
                )
            )

    resident_names = replicated_attention | {
        "enorm_w", "hnorm_w", "e_proj_w", "e_proj_w_scale", "e_proj_smooth",
        "h_proj_w", "h_proj_w_scale", "h_proj_smooth",
        "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
        "gate_w", "gate_bias", "tid2eid",
        "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
        "routed_w2", "routed_w2_scale",
        "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
        "shared_w2", "shared_w2_scale",
        "mtp_hc_head_fn", "mtp_hc_head_scale", "mtp_hc_head_base", "mtp_norm_w",
    }
    for spec in specs:
        if spec.name in resident_names:
            spec.resident = "stacked"

    specs.append(TensorSpec("hidden_out", [N_RANKS, T, D], torch.bfloat16, is_output=True))
    specs.append(TensorSpec("next_pre_hc_hidden", [N_RANKS, T, HC_MULT, D], torch.float32, is_output=True))
    specs.append(ScalarSpec("num_tokens", torch.int32, num_tokens))
    return specs


def golden_mtp_decode_layer(tensors):
    import torch

    num_tokens = int(tensors["num_tokens"])
    projected = torch.empty_like(tensors["prev_pre_hc_hidden"])
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
            "hidden_states_out": projected[rank],
        })

    x_attn = torch.empty_like(projected)
    for rank in range(N_RANKS):
        golden_attention_swa({
            "x_hc": projected[rank],
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
    moe_tensors["layer_id"] = MTP_LAYER_ID
    moe_tensors["num_tokens"] = num_tokens
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


def _mapped_pool_ratio_reldiff(
    mapping_name,
    *,
    diff_thd,
    pct_thd,
):
    """Apply a relative-diff budget only to allocator-mapped cache rows."""
    import torch

    from golden import ratio_reldiff

    mapped_compare = ratio_reldiff(diff_thd=diff_thd, pct_thd=pct_thd)

    def compare(actual, expected, **kwargs):
        if actual.shape != expected.shape:
            return False, (
                f"    pool shape mismatch: actual={tuple(actual.shape)} "
                f"expected={tuple(expected.shape)}"
            )
        if actual.ndim < 3:
            return False, f"    mapped pool must have rank >= 3, got {tuple(actual.shape)}"

        rank_count = actual.shape[0]
        actual_rows = actual.cpu().reshape(rank_count, -1, actual.shape[-1])
        expected_rows = expected.cpu().reshape(rank_count, -1, expected.shape[-1])
        row_count = actual_rows.shape[1]
        for label, rows in (("actual", actual_rows), ("expected", expected_rows)):
            if torch.is_floating_point(rows):
                nonfinite = ~torch.isfinite(rows)
                if nonfinite.any().item():
                    return False, (
                        f"    {label} pool contains "
                        f"{int(nonfinite.count_nonzero().item())} non-finite value(s)"
                    )

        mapping = kwargs.get("inputs", {}).get(mapping_name)
        if mapping is None:
            return False, f"    compare_fn misconfigured: missing input '{mapping_name}'"
        mapping = mapping.cpu().to(torch.int64)
        if mapping.ndim != 2 or mapping.shape[0] != rank_count:
            return False, (
                f"    '{mapping_name}' shape {tuple(mapping.shape)} does not match "
                f"ranked pool shape {tuple(actual.shape)}"
            )

        invalid_negative = mapping < -1
        if invalid_negative.any().item():
            first = invalid_negative.nonzero(as_tuple=False)[0]
            rank, token = (int(first[0].item()), int(first[1].item()))
            return False, (
                f"    '{mapping_name}'[{rank}, {token}]="
                f"{int(mapping[rank, token].item())} is invalid; "
                "only -1 is a negative sentinel"
            )
        valid = mapping >= 0
        out_of_range = valid & (mapping >= row_count)
        if out_of_range.any().item():
            first = out_of_range.nonzero(as_tuple=False)[0]
            rank, token = (int(first[0].item()), int(first[1].item()))
            return False, (
                f"    '{mapping_name}'[{rank}, {token}]="
                f"{int(mapping[rank, token].item())} is outside "
                f"physical row range [0, {row_count})"
            )

        written_rows = torch.zeros((rank_count, row_count), dtype=torch.bool)
        for rank in range(rank_count):
            rank_mapping = mapping[rank, valid[rank]]
            if rank_mapping.numel() != torch.unique(rank_mapping).numel():
                return False, f"    '{mapping_name}' contains duplicate rows on rank {rank}"
            written_rows[rank, rank_mapping] = True

        equal_rows = (actual_rows == expected_rows).all(dim=-1)
        stray_rows = ~written_rows & ~equal_rows
        if stray_rows.any().item():
            first = stray_rows.nonzero(as_tuple=False)[0]
            rank, row = (int(first[0].item()), int(first[1].item()))
            changed = int(
                (actual_rows[rank, row] != expected_rows[rank, row])
                .count_nonzero()
                .item()
            )
            return False, (
                f"    unmapped physical row changed: rank={rank} row={row} "
                f"changed_values={changed} mapping='{mapping_name}'"
            )

        if not written_rows.any().item():
            return True, ""
        ok, detail = mapped_compare(
            actual_rows[written_rows],
            expected_rows[written_rows],
            **kwargs,
        )
        if ok:
            return True, ""
        return False, f"    mapped rows from '{mapping_name}':\n{detail}"

    compare.__name__ = (
        f"mapped_pool_ratio_reldiff(mapping={mapping_name}, "
        f"diff_thd={diff_thd}, pct_thd={pct_thd})"
    )
    return compare


def main():
    import torch

    from golden import run_jit

    parser = argparse.ArgumentParser(description="DeepSeek-V4 MTP decode layer driver.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--ep", type=int, default=N_RANKS, choices=[2, 4, 8],
                        help="EP world size / rank count (parsed at import by moe).")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(N_RANKS)),
                        help=f"comma-separated device ids; need at least {N_RANKS}")
    parser.add_argument("--start-pos", type=int, default=DECODE_START_POS)
    parser.add_argument("--num-tokens", type=int, default=T)
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for reproducible inputs and golden")
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) >= N_RANKS, f"need at least {N_RANKS} devices, got {device_ids}"

    result = run_jit(
        fn=l3_mtp_decode_layer,
        specs=build_tensor_specs(start_pos=args.start_pos, num_tokens=args.num_tokens),
        golden_fn=golden_mtp_decode_layer,
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
        compare_fn={
            "hidden_out": _token_partition_ratio_reldiff(
                args.num_tokens,
                diff_thd=0.02,
                pct_thd=0.10,
                max_abs_diff=float("inf"),
            ),
            # This FP32 state follows the composed BF16 attention and MoE path.
            # Seeded EP2 measurements place 3.88-4.95% of points above 2%
            # relative difference, with rel-L2 near 1% and max absolute error
            # below 0.17. Keep the per-point threshold, add modest distribution
            # margin and a meaningful absolute guard for near-zero crossings.
            "next_pre_hc_hidden": _token_partition_ratio_reldiff(
                args.num_tokens,
                diff_thd=0.02,
                pct_thd=0.06,
                max_abs_diff=0.25,
            ),
            "kv_cache": _mapped_pool_ratio_reldiff(
                "swa_slot_mapping",
                diff_thd=0.01,
                pct_thd=0.05,
            ),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
