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
"""DeepSeek-V4 MTP packed-prefill forward.

This mirrors the official MTP block:
``e_proj(enorm(hidden_states)).unsqueeze(2) + h_proj(hnorm(prev_hidden_states))``
followed by one SWA block, MoE, MTP ``hc_head`` and MTP RMSNorm.  The input
``prev_hidden_states`` and ``pre_hc_hidden_out`` keep the official pre-hc layout
``[T, HC_MULT, D]``.  ``hidden_states`` is the shared embedding/current-token
hidden input in token-major ``[T, D]`` layout.
"""

import argparse

import pypto.language as pl
import pypto.language.distributed as pld
from golden import run_jit
from pypto.ir.distributed_compiled_program import DistributedConfig

import config

config.MOE_TOKENS = config.PREFILL_TOKENS

from config import FLASH as M, PREFILL_TOKENS
from hc_head import (
    EPS as HC_HEAD_RMS_EPS,
    HC_DIM_INV as HC_HEAD_DIM_INV,
    HC_EPS as HC_HEAD_EPS,
    LINEAR_K_TILE as HC_HEAD_LINEAR_K_TILE,
    RMS_K_TILE as HC_HEAD_RMS_K_TILE,
    hc_head,
)
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
    TOPK,
    VOCAB,
    build_tensor_specs as build_moe_tensor_specs,
    golden_moe,
    moe,
)
from mtp_projection import _quantize_weight_per_out, golden_mtp_projection, mtp_projection
from prefill_attention_swa import (
    BLOCK_NUM,
    BLOCK_SIZE,
    H,
    HEAD_DIM,
    MAX_SEQ_LEN,
    O_GROUP_IN,
    O_GROUPS,
    O_LORA,
    Q_LORA,
    ROPE_HEAD_DIM,
    golden_prefill_attention_swa,
    prefill_attention_swa,
)
from prefill_fwd import build_single_layer_tensor_specs
from rmsnorm import golden_rms_norm, rms_norm


T = PREFILL_TOKENS
MTP_LAYER_ID = M.num_hidden_layers
MTP_MOE_EPOCH = 1


@pl.jit
def mtp_prefill_fwd(
    hidden_states: pl.Tensor[[T, D], pl.BF16],
    prev_hidden_states: pl.Tensor[[T, HC_MULT, D], pl.FP32],
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
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[BLOCK_NUM], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
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
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
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
    pre_hc_hidden_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, D], pl.BF16]:
    nt: pl.Scalar[pl.INT32] = num_tokens
    projected = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
    x_attn = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)

    mtp_projection(
        hidden_states, prev_hidden_states,
        enorm_w, hnorm_w,
        e_proj_w, e_proj_w_scale, e_proj_smooth,
        h_proj_w, h_proj_w_scale, h_proj_smooth,
        projected,
    )

    prefill_attention_swa(
        projected,
        hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
        wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        kv_cache, ori_block_table, ori_slot_mapping,
        position_ids,
        attn_sink, wo_a, wo_b, wo_b_scale,
        x_attn, nt,
    )

    moe(
        x_attn,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias, tid2eid, input_ids,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        pre_hc_hidden_out,
        recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
        routed_y_buf, combine_arrived,
        pl.cast(MTP_LAYER_ID, pl.INT32), nt, my_rank, pl.cast(MTP_MOE_EPOCH, pl.INT32),
    )

    x_head = pl.create_tensor([T, D], dtype=pl.BF16)
    hc_head(pre_hc_hidden_out, mtp_hc_head_fn, mtp_hc_head_scale, mtp_hc_head_base, x_head)
    rms_norm(x_head, mtp_norm_w, hidden_out)
    return hidden_out


@pl.jit.host
def l3_mtp_prefill_fwd(
    hidden_states: pl.Tensor[[N_RANKS, T, D], pl.BF16],
    prev_hidden_states: pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32],
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
    freqs_cos: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[N_RANKS, BLOCK_NUM], pl.INT32],
    ori_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    position_ids: pl.Tensor[[N_RANKS, T], pl.INT32],
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
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
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
    pre_hc_hidden_out: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    recv_meta_buf = pld.alloc_window_buffer(N_RANKS * N_LOCAL * 4)
    recv_x_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * D)
    recv_aux_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * AUX_PAD * 4)
    recv_route_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * IDX_PAD * 4)
    arrived_buf = pld.alloc_window_buffer(N_RANKS * 4)
    data_arrived_buf = pld.alloc_window_buffer(N_RANKS * 4)
    routed_y_buf_buf = pld.alloc_window_buffer(N_ROUTES * D * 2)
    combine_arrived_buf = pld.alloc_window_buffer(N_RANKS * 4)

    for r in pl.range(pld.world_size()):
        recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32] = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8] = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32] = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32] = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32] = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32] = pld.window(data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16] = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32] = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        mtp_prefill_fwd(
            hidden_states[r],
            prev_hidden_states[r],
            enorm_w[r], hnorm_w[r],
            e_proj_w[r], e_proj_w_scale[r], e_proj_smooth[r],
            h_proj_w[r], h_proj_w_scale[r], h_proj_smooth[r],
            hc_attn_fn[r], hc_attn_scale[r], hc_attn_base[r], attn_norm_w[r],
            wq_a[r], wq_b[r], wq_b_scale[r], wkv[r], gamma_cq[r], gamma_ckv[r],
            freqs_cos[r], freqs_sin[r],
            kv_cache[r], ori_block_table[r], ori_slot_mapping[r],
            position_ids[r],
            attn_sink[r], wo_a[r], wo_b[r], wo_b_scale[r],
            hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r], norm_w[r],
            gate_w[r], gate_bias[r], tid2eid[r], input_ids[r],
            routed_w1[r], routed_w1_scale[r], routed_w3[r], routed_w3_scale[r],
            routed_w2[r], routed_w2_scale[r],
            shared_w1[r], shared_w1_scale[r], shared_w3[r], shared_w3_scale[r],
            shared_w2[r], shared_w2_scale[r],
            mtp_hc_head_fn[r], mtp_hc_head_scale[r], mtp_hc_head_base[r], mtp_norm_w[r],
            hidden_out[r], pre_hc_hidden_out[r],
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
            routed_y_buf, combine_arrived,
            r, num_tokens,
            device=r,
        )


def _ranked(spec, torch, is_output=False):
    from golden import TensorSpec

    return TensorSpec(
        spec.name,
        list(spec.shape),
        spec.dtype,
        init_value=spec.init_value,
        is_output=is_output,
    )


def _projection_specs():
    import torch
    from golden import TensorSpec

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

    return [
        TensorSpec("hidden_states", [N_RANKS, T, D], torch.bfloat16, init_value=lambda: torch.randn(N_RANKS, T, D).to(torch.bfloat16)),
        TensorSpec("prev_hidden_states", [N_RANKS, T, HC_MULT, D], torch.float32,
                   init_value=lambda: torch.randn(N_RANKS, T, HC_MULT, D).to(torch.bfloat16)),
        TensorSpec("enorm_w", [N_RANKS, D], torch.float32, init_value=lambda: torch.ones(N_RANKS, D)),
        TensorSpec("hnorm_w", [N_RANKS, D], torch.float32, init_value=lambda: torch.ones(N_RANKS, D)),
        TensorSpec("e_proj_w", [N_RANKS, D, D], torch.int8, init_value=init_e_proj_w),
        TensorSpec("e_proj_w_scale", [N_RANKS, D], torch.float32, init_value=init_e_proj_w_scale),
        TensorSpec("e_proj_smooth", [N_RANKS, D], torch.float32, init_value=lambda: torch.ones(N_RANKS, D)),
        TensorSpec("h_proj_w", [N_RANKS, D, D], torch.int8, init_value=init_h_proj_w),
        TensorSpec("h_proj_w_scale", [N_RANKS, D], torch.float32, init_value=init_h_proj_w_scale),
        TensorSpec("h_proj_smooth", [N_RANKS, D], torch.float32, init_value=lambda: torch.ones(N_RANKS, D)),
    ]


def _mtp_head_specs():
    import torch
    from golden import TensorSpec

    base = [5.9166, -3.6223, -2.9324, -3.3124]
    return [
        TensorSpec("mtp_hc_head_fn", [N_RANKS, HC_MULT, HC_DIM], torch.float32,
                   init_value=lambda: torch.randn(N_RANKS, HC_MULT, HC_DIM) * 0.0519),
        TensorSpec("mtp_hc_head_scale", [N_RANKS, 1], torch.float32,
                   init_value=lambda: torch.full((N_RANKS, 1), 0.076099, dtype=torch.float32)),
        TensorSpec("mtp_hc_head_base", [N_RANKS, HC_MULT], torch.float32,
                   init_value=lambda: torch.tensor(base, dtype=torch.float32).view(1, HC_MULT).expand(N_RANKS, -1).contiguous()),
        TensorSpec("mtp_norm_w", [N_RANKS, D], torch.bfloat16,
                   init_value=lambda: (torch.randn(N_RANKS, D) * 0.1 + 1.0).to(torch.bfloat16)),
    ]


def build_tensor_specs(start_pos=0, num_tokens=T):
    import torch
    from golden import ScalarSpec, TensorSpec

    base = {
        spec.name: spec
        for spec in build_single_layer_tensor_specs(start_pos=start_pos, num_tokens=num_tokens, layer_id=MTP_LAYER_ID)
        if isinstance(spec, TensorSpec)
    }
    projection = {spec.name: spec for spec in _projection_specs()}
    mtp_head = {spec.name: spec for spec in _mtp_head_specs()}
    moe_specs = {spec.name: spec for spec in build_moe_tensor_specs(layer_id=MTP_LAYER_ID, num_tokens=num_tokens) if isinstance(spec, TensorSpec)}

    ordered_names = [
        "hidden_states", "prev_hidden_states",
        "enorm_w", "hnorm_w", "e_proj_w", "e_proj_w_scale", "e_proj_smooth",
        "h_proj_w", "h_proj_w_scale", "h_proj_smooth",
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
        "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
        "freqs_cos", "freqs_sin", "kv_cache", "ori_block_table", "ori_slot_mapping",
        "position_ids",
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
        if name in projection:
            specs.append(projection[name])
        elif name in mtp_head:
            specs.append(mtp_head[name])
        elif name in moe_specs and name not in base:
            specs.append(_ranked(moe_specs[name], torch))
        else:
            specs.append(_ranked(base[name], torch))

    specs.append(TensorSpec("hidden_out", [N_RANKS, T, D], torch.bfloat16, is_output=True))
    specs.append(TensorSpec("pre_hc_hidden_out", [N_RANKS, T, HC_MULT, D], torch.float32, is_output=True))
    specs.append(ScalarSpec("num_tokens", torch.int32, num_tokens))
    return specs


def _golden_hc_head_prefill(x_hc, hc_head_fn, hc_head_scale, hc_head_base):
    import torch

    token_count = x_hc.shape[0]
    shape = x_hc.shape
    x_flat_2d = x_hc.reshape(token_count, HC_DIM).float()
    hc_head_fn = hc_head_fn.float()

    sq_sum = torch.zeros(token_count, 1, dtype=torch.float32)
    for k0 in range(0, HC_DIM, HC_HEAD_RMS_K_TILE):
        x_chunk = x_flat_2d[:, k0:k0 + HC_HEAD_RMS_K_TILE]
        sq_sum += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
    rsqrt = torch.rsqrt(sq_sum * HC_HEAD_DIM_INV + HC_HEAD_RMS_EPS)

    mix_cols = []
    for h in range(HC_MULT):
        mix_col = torch.zeros(token_count, 1, dtype=torch.float32)
        for k0 in range(0, HC_DIM, HC_HEAD_LINEAR_K_TILE):
            x_chunk = x_flat_2d[:, k0:k0 + HC_HEAD_LINEAR_K_TILE]
            w_chunk = hc_head_fn[h:h + 1, k0:k0 + HC_HEAD_LINEAR_K_TILE]
            mix_col += (x_chunk * w_chunk).sum(dim=1, keepdim=True)
        mix_cols.append(mix_col * rsqrt)
    mixes = torch.cat(mix_cols, dim=1).reshape(token_count, HC_MULT)

    pre = torch.sigmoid(mixes * hc_head_scale.float() + hc_head_base.float()) + HC_HEAD_EPS
    x_view = x_hc.float().view(shape)
    if HC_MULT == 4:
        y = (
            x_view[:, 0, :] * pre[:, 0:1]
            + x_view[:, 1, :] * pre[:, 1:2]
        ) + (
            x_view[:, 2, :] * pre[:, 2:3]
            + x_view[:, 3, :] * pre[:, 3:4]
        )
    else:
        y = torch.zeros(token_count, D, dtype=torch.float32)
        for h in range(HC_MULT):
            y += x_view[:, h, :] * pre[:, h:h + 1]

    rounded = (y.contiguous().view(torch.int32) + 0x8000) & -0x10000
    return rounded.view(torch.float32).to(torch.bfloat16)


def golden_mtp_prefill_fwd(tensors):
    import torch

    num_tokens = int(tensors["num_tokens"])

    projected = torch.zeros_like(tensors["prev_hidden_states"])
    for rank in range(N_RANKS):
        golden_mtp_projection({
            "hidden_states": tensors["hidden_states"][rank],
            "prev_hidden_states": tensors["prev_hidden_states"][rank],
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

    x_attn = torch.zeros_like(projected)
    for rank in range(N_RANKS):
        golden_prefill_attention_swa({
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
            "freqs_cos": tensors["freqs_cos"][rank],
            "freqs_sin": tensors["freqs_sin"][rank],
            "kv_cache": tensors["kv_cache"][rank],
            "block_table": tensors["ori_block_table"][rank],
            "ori_slot_mapping": tensors["ori_slot_mapping"][rank],
            "position_ids": tensors["position_ids"][rank],
            "attn_sink": tensors["attn_sink"][rank],
            "wo_a": tensors["wo_a"][rank],
            "wo_b": tensors["wo_b"][rank],
            "wo_b_scale": tensors["wo_b_scale"][rank],
            "x_out": x_attn[rank],
            "num_tokens": num_tokens,
        })

    moe_tensors = dict(tensors)
    moe_tensors["x_hc"] = x_attn
    moe_tensors["x_next"] = tensors["pre_hc_hidden_out"]
    moe_tensors["layer_id"] = MTP_LAYER_ID
    moe_tensors["num_tokens"] = num_tokens
    golden_moe(moe_tensors)

    for rank in range(N_RANKS):
        x_head = _golden_hc_head_prefill(
            tensors["pre_hc_hidden_out"][rank],
            tensors["mtp_hc_head_fn"][rank],
            tensors["mtp_hc_head_scale"][rank],
            tensors["mtp_hc_head_base"][rank],
        )
        tensors["hidden_out"][rank] = golden_rms_norm(x_head, tensors["mtp_norm_w"][rank])


def valid_ratio_reldiff(num_tokens, diff_thd, pct_thd):
    from golden import ratio_reldiff

    base_cmp = ratio_reldiff(diff_thd=diff_thd, pct_thd=pct_thd)

    def cmp(actual, expected, **kwargs):
        return base_cmp(actual[:, :num_tokens], expected[:, :num_tokens], **kwargs)

    cmp.__name__ = f"valid_ratio_reldiff(num_tokens={num_tokens})"
    return cmp


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V4 MTP packed-prefill forward driver.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument("--ep", type=int, default=N_RANKS, choices=[2, 4, 8],
                        help="EP world size / rank count (parsed at import by moe).")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(N_RANKS)),
                        help=f"comma-separated device ids; need at least {N_RANKS}")
    parser.add_argument("--start-pos", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=T)
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    parser.add_argument("--enable-scope-stats", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) >= N_RANKS, f"need at least {N_RANKS} devices, got {device_ids}"

    result = run_jit(
        fn=l3_mtp_prefill_fwd,
        specs=build_tensor_specs(start_pos=args.start_pos, num_tokens=args.num_tokens),
        golden_fn=golden_mtp_prefill_fwd,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        save_data=False,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=device_ids[:N_RANKS], num_sub_workers=0),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_l2_swimlane=args.enable_l2_swimlane,
            enable_scope_stats=args.enable_scope_stats,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "hidden_out": valid_ratio_reldiff(args.num_tokens, diff_thd=0.02, pct_thd=0.05),
            "pre_hc_hidden_out": valid_ratio_reldiff(args.num_tokens, diff_thd=0.02, pct_thd=0.05),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
