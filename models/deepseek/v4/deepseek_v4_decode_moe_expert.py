# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE local expert + shared expert compute (decode, EP single-card)."""


import pypto.language as pl


B = 16  # demo 4
S = 1
T = B * S

D = 4096            # flash:4096 pro:7168
MOE_INTER = 4096    # flash:2048 pro:3072
TOPK = 2            # flash:6 pro:6 (n_activated_experts)
SWIGLU_LIMIT = 0.0  # flash:10.0 pro:10.0 (pro drops it on shared experts)

EP_WORLD_SIZE = 1   # demo 1; flash/pro depend on deployment (e.g. pro 16)
EP_RANK = 0
N_EXPERTS = 8       # flash:256 pro:384
N_LOCAL_EXPERTS = N_EXPERTS // EP_WORLD_SIZE
EXPERTS_START_IDX = EP_RANK * N_LOCAL_EXPERTS  # global id offset; recv_expert_id carries global ids

RECV_TOTAL_MAX = 32  # demo 32; sized as T*TOPK/EP plus imbalance headroom

K_CHUNK = 512
INTER_K = 512
INTER_CHUNK = 256
D_OUT_CHUNK = 512
RECV_Y_INIT_CHUNK = 1024

INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4  # amax floor: keeps padding/all-zero rows from producing 127/0 = inf
QUANT_CHUNK = 256  # column chunk size for two-pass per-row INT8 quant (vec budget aware)


@pl.jit.inline
def moe_expert(
    recv_x: pl.Tensor[[RECV_TOTAL_MAX, D], pl.BF16],
    # Global expert id per row; padding rows must carry an id outside
    # [EXPERTS_START_IDX, EXPERTS_START_IDX + N_LOCAL_EXPERTS) (e.g. -1).
    recv_expert_id: pl.Tensor[[RECV_TOTAL_MAX, 1], pl.INT32],
    recv_weights: pl.Tensor[[RECV_TOTAL_MAX, 1], pl.FP32],
    x_local: pl.Tensor[[T, D], pl.BF16],
    expert_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    expert_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    expert_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    expert_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    expert_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    expert_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    recv_y: pl.Tensor[[RECV_TOTAL_MAX, D], pl.BF16],
    sh: pl.Tensor[[T, D], pl.BF16],
):
    # Stage 0: zero-init recv_y so cross-expert accumulation is exact.
    for d0 in pl.parallel(0, D, RECV_Y_INIT_CHUNK):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="recv_y_zero"):
            zero_chunk = pl.full([RECV_TOTAL_MAX, RECV_Y_INIT_CHUNK], dtype=pl.FP32, value=0.0)
            recv_y[:, d0 : d0 + RECV_Y_INIT_CHUNK] = pl.cast(zero_chunk, target_type=pl.BF16)

    # Stage 0b: per-token A8 quant of recv_x and x_local.
    recv_x_i8 = pl.create_tensor([RECV_TOTAL_MAX, D], dtype=pl.INT8)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="recv_x_q"):
        rx_amax = pl.full([1, RECV_TOTAL_MAX], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for k0 in pl.range(0, D, QUANT_CHUNK):
            rx_a_f32 = pl.cast(recv_x[:, k0 : k0 + QUANT_CHUNK], target_type=pl.FP32)
            rx_a_abs = pl.maximum(rx_a_f32, pl.neg(rx_a_f32))
            rx_a_max = pl.reshape(pl.row_max(rx_a_abs), [1, RECV_TOTAL_MAX])
            rx_amax = pl.maximum(rx_amax, rx_a_max)
        rx_sq_row = pl.div(pl.full([1, RECV_TOTAL_MAX], dtype=pl.FP32, value=INT8_SCALE_MAX), rx_amax)
        recv_x_scale_dq = pl.reshape(pl.recip(rx_sq_row), [RECV_TOTAL_MAX, 1])
        rx_sq_col = pl.reshape(rx_sq_row, [RECV_TOTAL_MAX, 1])
        for k1 in pl.range(0, D, QUANT_CHUNK):
            rx_q_f32 = pl.cast(recv_x[:, k1 : k1 + QUANT_CHUNK], target_type=pl.FP32)
            rx_q_scaled = pl.row_expand_mul(rx_q_f32, rx_sq_col)
            rx_q_i32 = pl.cast(rx_q_scaled, target_type=pl.INT32, mode="round")
            rx_q_half = pl.cast(rx_q_i32, target_type=pl.FP16, mode="round")
            recv_x_i8[:, k1 : k1 + QUANT_CHUNK] = pl.cast(rx_q_half, target_type=pl.INT8, mode="trunc")

    x_local_i8 = pl.create_tensor([T, D], dtype=pl.INT8)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="x_local_q"):
        xl_amax = pl.full([1, T], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for k0 in pl.range(0, D, QUANT_CHUNK):
            xl_a_f32 = pl.cast(x_local[:, k0 : k0 + QUANT_CHUNK], target_type=pl.FP32)
            xl_a_abs = pl.maximum(xl_a_f32, pl.neg(xl_a_f32))
            xl_a_max = pl.reshape(pl.row_max(xl_a_abs), [1, T])
            xl_amax = pl.maximum(xl_amax, xl_a_max)
        xl_sq_row = pl.div(pl.full([1, T], dtype=pl.FP32, value=INT8_SCALE_MAX), xl_amax)
        x_local_scale_dq = pl.reshape(pl.recip(xl_sq_row), [T, 1])
        xl_sq_col = pl.reshape(xl_sq_row, [T, 1])
        for k1 in pl.range(0, D, QUANT_CHUNK):
            xl_q_f32 = pl.cast(x_local[:, k1 : k1 + QUANT_CHUNK], target_type=pl.FP32)
            xl_q_scaled = pl.row_expand_mul(xl_q_f32, xl_sq_col)
            xl_q_i32 = pl.cast(xl_q_scaled, target_type=pl.INT32, mode="round")
            xl_q_half = pl.cast(xl_q_i32, target_type=pl.FP16, mode="round")
            x_local_i8[:, k1 : k1 + QUANT_CHUNK] = pl.cast(xl_q_half, target_type=pl.INT8, mode="trunc")

    # Stage 1: routed local experts. local_i iterates global ids (matching
    # recv_expert_id), expert_w* are indexed by local_offset. Each row matches
    # at most one local_i, so mask + `recv_y +=` realizes scatter-by-mask.
    # TODO: with proper col_gather/col_scatter, group rows by expert before
    # the matmul to raise per-expert batch and cube utilization.
    for local_i in pl.range(EXPERTS_START_IDX, EXPERTS_START_IDX + N_LOCAL_EXPERTS):
        local_offset = local_i - EXPERTS_START_IDX
        scale_col = pl.create_tensor([RECV_TOTAL_MAX, 1], dtype=pl.FP32)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_scale_col"):
            local_i_i32 = pl.cast(local_i, pl.INT32)
            expert_id_row = pl.reshape(recv_expert_id, [1, RECV_TOTAL_MAX])
            weights_row = pl.reshape(recv_weights, [1, RECV_TOTAL_MAX])
            mask = pl.tensor.cmp(expert_id_row, local_i_i32, cmp_type=0)
            mask_fp32 = pl.cast(mask, target_type=pl.FP32)
            scaled_row = pl.mul(weights_row, mask_fp32)
            scale_col[:, :] = pl.reshape(scaled_row, [RECV_TOTAL_MAX, 1])

        # Stage post-SwiGLU activation buffer (per-row amax taken over MOE_INTER).
        h_tile_fp32 = pl.create_tensor([RECV_TOTAL_MAX, MOE_INTER], dtype=pl.FP32)

        for n0 in pl.parallel(0, MOE_INTER, INTER_CHUNK):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_gate_up_matmul"):
                x_init = recv_x_i8[:, 0 : K_CHUNK]
                w1_init = expert_w1[local_offset, n0 : n0 + INTER_CHUNK, 0 : K_CHUNK]
                w3_init = expert_w3[local_offset, n0 : n0 + INTER_CHUNK, 0 : K_CHUNK]
                gate_acc = pl.matmul(x_init, w1_init, b_trans=True, out_dtype=pl.INT32)
                up_acc = pl.matmul(x_init, w3_init, b_trans=True, out_dtype=pl.INT32)
                for k0 in pl.range(K_CHUNK, D, K_CHUNK):
                    x_k = recv_x_i8[:, k0 : k0 + K_CHUNK]
                    w1_k = expert_w1[local_offset, n0 : n0 + INTER_CHUNK, k0 : k0 + K_CHUNK]
                    w3_k = expert_w3[local_offset, n0 : n0 + INTER_CHUNK, k0 : k0 + K_CHUNK]
                    gate_acc = pl.matmul_acc(gate_acc, x_k, w1_k, b_trans=True)
                    up_acc = pl.matmul_acc(up_acc, x_k, w3_k, b_trans=True)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_gate_up_dequant"):
                # Scalar-indexed expert_w* keeps a leading size-1 dim; matmul promotes
                # that to a 3D batched output. Drop it so downstream stays 2D.
                gate_2d_i32 = pl.reshape(gate_acc, [RECV_TOTAL_MAX, INTER_CHUNK])
                up_2d_i32 = pl.reshape(up_acc, [RECV_TOTAL_MAX, INTER_CHUNK])
                w1_scale_chunk = pl.reshape(expert_w1_scale[local_offset, n0 : n0 + INTER_CHUNK], [1, INTER_CHUNK])
                w3_scale_chunk = pl.reshape(expert_w3_scale[local_offset, n0 : n0 + INTER_CHUNK], [1, INTER_CHUNK])
                gate_2d = pl.cast(gate_2d_i32, target_type=pl.FP32, mode="none")
                up_2d = pl.cast(up_2d_i32, target_type=pl.FP32, mode="none")
                gate_2d = pl.col_expand_mul(pl.row_expand_mul(gate_2d, recv_x_scale_dq), w1_scale_chunk)
                up_2d = pl.col_expand_mul(pl.row_expand_mul(up_2d, recv_x_scale_dq), w3_scale_chunk)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_swiglu"):
                if SWIGLU_LIMIT > 0.0:
                    gate_2d = pl.minimum(gate_2d, SWIGLU_LIMIT)
                    up_2d = pl.maximum(pl.minimum(up_2d, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_2d)), 1.0))
                silu = pl.mul(gate_2d, sigmoid)
                gated = pl.mul(silu, up_2d)
                # Pre-multiply routing scale before A8 requant so the requant amax
                # reflects the same magnitude the BF16 path produced for w2 input.
                h_chunk = pl.row_expand_mul(gated, scale_col)
                h_tile_fp32[:, n0 : n0 + INTER_CHUNK] = h_chunk

        # Per-row A8 requant of h_tile (amax across full MOE_INTER row).
        h_tile_i8 = pl.create_tensor([RECV_TOTAL_MAX, MOE_INTER], dtype=pl.INT8)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_h_q"):
            eh_amax = pl.full([1, RECV_TOTAL_MAX], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for k0 in pl.range(0, MOE_INTER, QUANT_CHUNK):
                eh_a_f32 = h_tile_fp32[:, k0 : k0 + QUANT_CHUNK]
                eh_a_abs = pl.maximum(eh_a_f32, pl.neg(eh_a_f32))
                eh_a_max = pl.reshape(pl.row_max(eh_a_abs), [1, RECV_TOTAL_MAX])
                eh_amax = pl.maximum(eh_amax, eh_a_max)
            eh_sq_row = pl.div(pl.full([1, RECV_TOTAL_MAX], dtype=pl.FP32, value=INT8_SCALE_MAX), eh_amax)
            h_tile_scale_dq = pl.reshape(pl.recip(eh_sq_row), [RECV_TOTAL_MAX, 1])
            eh_sq_col = pl.reshape(eh_sq_row, [RECV_TOTAL_MAX, 1])
            for k1 in pl.range(0, MOE_INTER, QUANT_CHUNK):
                eh_q_f32 = h_tile_fp32[:, k1 : k1 + QUANT_CHUNK]
                eh_q_scaled = pl.row_expand_mul(eh_q_f32, eh_sq_col)
                eh_q_i32 = pl.cast(eh_q_scaled, target_type=pl.INT32, mode="round")
                eh_q_half = pl.cast(eh_q_i32, target_type=pl.FP16, mode="round")
                h_tile_i8[:, k1 : k1 + QUANT_CHUNK] = pl.cast(eh_q_half, target_type=pl.INT8, mode="trunc")

        for d0 in pl.parallel(0, D, D_OUT_CHUNK):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_w2_matmul"):
                h_init = h_tile_i8[:, 0 : INTER_K]
                w2_init = expert_w2[local_offset, d0 : d0 + D_OUT_CHUNK, 0 : INTER_K]
                y_acc = pl.matmul(h_init, w2_init, b_trans=True, out_dtype=pl.INT32)
                for k0 in pl.range(INTER_K, MOE_INTER, INTER_K):
                    h_k = h_tile_i8[:, k0 : k0 + INTER_K]
                    w2_k = expert_w2[local_offset, d0 : d0 + D_OUT_CHUNK, k0 : k0 + INTER_K]
                    y_acc = pl.matmul_acc(y_acc, h_k, w2_k, b_trans=True)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_w2_dequant"):
                y_2d_i32 = pl.reshape(y_acc, [RECV_TOTAL_MAX, D_OUT_CHUNK])
                w2_scale_chunk = pl.reshape(expert_w2_scale[local_offset, d0 : d0 + D_OUT_CHUNK], [1, D_OUT_CHUNK])
                y_2d = pl.cast(y_2d_i32, target_type=pl.FP32, mode="none")
                y_2d = pl.col_expand_mul(pl.row_expand_mul(y_2d, h_tile_scale_dq), w2_scale_chunk)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_recv_y_accum"):
                existing = pl.cast(recv_y[:, d0 : d0 + D_OUT_CHUNK], target_type=pl.FP32)
                summed = pl.add(existing, y_2d)
                recv_y[:, d0 : d0 + D_OUT_CHUNK] = pl.cast(summed, target_type=pl.BF16)

    # Stage 2: shared expert
    sh_tile_fp32 = pl.create_tensor([T, MOE_INTER], dtype=pl.FP32)

    for n0 in pl.parallel(0, MOE_INTER, INTER_CHUNK):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_gate_up_matmul"):
            xs_init = x_local_i8[:, 0 : K_CHUNK]
            sw1_init = shared_w1[n0 : n0 + INTER_CHUNK, 0 : K_CHUNK]
            sw3_init = shared_w3[n0 : n0 + INTER_CHUNK, 0 : K_CHUNK]
            sh_gate_acc = pl.matmul(xs_init, sw1_init, b_trans=True, out_dtype=pl.INT32)
            sh_up_acc = pl.matmul(xs_init, sw3_init, b_trans=True, out_dtype=pl.INT32)
            for k0 in pl.range(K_CHUNK, D, K_CHUNK):
                xs_k = x_local_i8[:, k0 : k0 + K_CHUNK]
                sw1_k = shared_w1[n0 : n0 + INTER_CHUNK, k0 : k0 + K_CHUNK]
                sw3_k = shared_w3[n0 : n0 + INTER_CHUNK, k0 : k0 + K_CHUNK]
                sh_gate_acc = pl.matmul_acc(sh_gate_acc, xs_k, sw1_k, b_trans=True)
                sh_up_acc = pl.matmul_acc(sh_up_acc, xs_k, sw3_k, b_trans=True)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_gate_up_dequant"):
            sw1_scale_chunk = pl.reshape(shared_w1_scale[n0 : n0 + INTER_CHUNK], [1, INTER_CHUNK])
            sw3_scale_chunk = pl.reshape(shared_w3_scale[n0 : n0 + INTER_CHUNK], [1, INTER_CHUNK])
            sh_gate = pl.cast(sh_gate_acc, target_type=pl.FP32, mode="none")
            sh_up = pl.cast(sh_up_acc, target_type=pl.FP32, mode="none")
            sh_gate = pl.col_expand_mul(pl.row_expand_mul(sh_gate, x_local_scale_dq), sw1_scale_chunk)
            sh_up = pl.col_expand_mul(pl.row_expand_mul(sh_up, x_local_scale_dq), sw3_scale_chunk)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_swiglu"):
            sh_sigmoid = pl.recip(pl.add(pl.exp(pl.neg(sh_gate)), 1.0))
            sh_silu = pl.mul(sh_gate, sh_sigmoid)
            sh_gated = pl.mul(sh_silu, sh_up)
            sh_tile_fp32[:, n0 : n0 + INTER_CHUNK] = sh_gated

    sh_tile_i8 = pl.create_tensor([T, MOE_INTER], dtype=pl.INT8)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_h_q"):
        shq_amax = pl.full([1, T], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for k0 in pl.range(0, MOE_INTER, QUANT_CHUNK):
            shq_a_f32 = sh_tile_fp32[:, k0 : k0 + QUANT_CHUNK]
            shq_a_abs = pl.maximum(shq_a_f32, pl.neg(shq_a_f32))
            shq_a_max = pl.reshape(pl.row_max(shq_a_abs), [1, T])
            shq_amax = pl.maximum(shq_amax, shq_a_max)
        shq_sq_row = pl.div(pl.full([1, T], dtype=pl.FP32, value=INT8_SCALE_MAX), shq_amax)
        sh_tile_scale_dq = pl.reshape(pl.recip(shq_sq_row), [T, 1])
        shq_sq_col = pl.reshape(shq_sq_row, [T, 1])
        for k1 in pl.range(0, MOE_INTER, QUANT_CHUNK):
            shq_q_f32 = sh_tile_fp32[:, k1 : k1 + QUANT_CHUNK]
            shq_q_scaled = pl.row_expand_mul(shq_q_f32, shq_sq_col)
            shq_q_i32 = pl.cast(shq_q_scaled, target_type=pl.INT32, mode="round")
            shq_q_half = pl.cast(shq_q_i32, target_type=pl.FP16, mode="round")
            sh_tile_i8[:, k1 : k1 + QUANT_CHUNK] = pl.cast(shq_q_half, target_type=pl.INT8, mode="trunc")

    for d0 in pl.parallel(0, D, D_OUT_CHUNK):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_w2_matmul"):
            hs_init = sh_tile_i8[:, 0 : INTER_K]
            sw2_init = shared_w2[d0 : d0 + D_OUT_CHUNK, 0 : INTER_K]
            sh_y_acc = pl.matmul(hs_init, sw2_init, b_trans=True, out_dtype=pl.INT32)
            for k0 in pl.range(INTER_K, MOE_INTER, INTER_K):
                hs_k = sh_tile_i8[:, k0 : k0 + INTER_K]
                sw2_k = shared_w2[d0 : d0 + D_OUT_CHUNK, k0 : k0 + INTER_K]
                sh_y_acc = pl.matmul_acc(sh_y_acc, hs_k, sw2_k, b_trans=True)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_w2_dequant"):
            sw2_scale_chunk = pl.reshape(shared_w2_scale[d0 : d0 + D_OUT_CHUNK], [1, D_OUT_CHUNK])
            sh_y = pl.cast(sh_y_acc, target_type=pl.FP32, mode="none")
            sh_y = pl.col_expand_mul(pl.row_expand_mul(sh_y, sh_tile_scale_dq), sw2_scale_chunk)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_write"):
            sh[:, d0 : d0 + D_OUT_CHUNK] = pl.cast(sh_y, target_type=pl.BF16)


@pl.jit
def moe_expert_test(
    recv_x: pl.Tensor[[RECV_TOTAL_MAX, D], pl.BF16],
    recv_expert_id: pl.Tensor[[RECV_TOTAL_MAX, 1], pl.INT32],
    recv_weights: pl.Tensor[[RECV_TOTAL_MAX, 1], pl.FP32],
    x_local: pl.Tensor[[T, D], pl.BF16],
    expert_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    expert_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    expert_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    expert_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    expert_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    expert_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    recv_y: pl.Out[pl.Tensor[[RECV_TOTAL_MAX, D], pl.BF16]],
    sh: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    moe_expert(
        recv_x, recv_expert_id, recv_weights, x_local,
        expert_w1, expert_w1_scale, expert_w3, expert_w3_scale,
        expert_w2, expert_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        recv_y, sh,
    )
    return recv_y, sh


def _round_half_away_from_zero(x):
    import torch
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


def _int8_quant_per_row(x):
    """Per-row (per-token) INT8 symmetric quant matching v3.2 scope2 Stage 2.6."""
    import torch
    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = _round_half_away_from_zero(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


def _quant_w_per_channel(w):
    """Per-output-channel INT8 quant on the last axis. Returns (i8_tensor, dequant_scale).

    For w shaped [..., N, K] (b_trans=True layout), the per-channel scale has shape [..., N].
    """
    import torch
    amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.unsqueeze(-1)
    w_i8 = _round_half_away_from_zero(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


def golden_moe_expert(tensors):
    """Torch reference (model.py 596-644). recv_y is the partial routed contribution
    only; AllToAllv combine and `+sh` happen in the host orchestrator.

    W8A8C16: all six weight matmuls (routed w1/w2/w3, shared sw1/sw2/sw3) use W8
    per-channel int8 weights with A8 per-token int8 activations; silu*up output
    is re-quantized to A8 before each w2 matmul. flash: same six matmuls run
    fp8 (default Linear dtype), with routed experts optionally fp4."""
    import torch
    import torch.nn.functional as F

    def dequant_w(w_i8, w_scale):
        return w_i8.to(torch.float32) * w_scale.unsqueeze(-1)

    recv_x = tensors["recv_x"].float()
    recv_expert_id = tensors["recv_expert_id"][:, 0]
    recv_weights = tensors["recv_weights"][:, 0].float()
    x_local = tensors["x_local"].float()
    w1 = dequant_w(tensors["expert_w1"], tensors["expert_w1_scale"].float())
    w3 = dequant_w(tensors["expert_w3"], tensors["expert_w3_scale"].float())
    w2 = dequant_w(tensors["expert_w2"], tensors["expert_w2_scale"].float())
    sw1 = dequant_w(tensors["shared_w1"], tensors["shared_w1_scale"].float())
    sw3 = dequant_w(tensors["shared_w3"], tensors["shared_w3_scale"].float())
    sw2 = dequant_w(tensors["shared_w2"], tensors["shared_w2_scale"].float())

    # Mirror activation A8 round-trip on recv_x / x_local so kernel and golden share
    # the same input quant noise.
    recv_x_i8, recv_x_sd = _int8_quant_per_row(recv_x)
    recv_x = recv_x_i8.float() * recv_x_sd
    x_local_i8, x_local_sd = _int8_quant_per_row(x_local)
    x_local = x_local_i8.float() * x_local_sd

    recv_y = torch.zeros(RECV_TOTAL_MAX, D)
    for local_i in range(EXPERTS_START_IDX, EXPERTS_START_IDX + N_LOCAL_EXPERTS):
        local_offset = local_i - EXPERTS_START_IDX
        mask = (recv_expert_id == local_i)
        if mask.sum() == 0:
            continue
        x_sub = recv_x[mask]
        w_sub = recv_weights[mask]
        gate = x_sub @ w1[local_offset].T
        up = x_sub @ w3[local_offset].T
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h = F.silu(gate) * up
        h = h * w_sub.unsqueeze(-1)
        # A8 requant before w2 matmul — matches kernel `exp_h_int8_quant` stage.
        h_i8, h_sd = _int8_quant_per_row(h)
        h = h_i8.float() * h_sd
        recv_y[mask] = h @ w2[local_offset].T

    sh_gate = x_local @ sw1.T
    sh_up = x_local @ sw3.T
    sh_h = F.silu(sh_gate) * sh_up
    sh_h_i8, sh_h_sd = _int8_quant_per_row(sh_h)
    sh_h = sh_h_i8.float() * sh_h_sd
    sh = sh_h @ sw2.T

    tensors["recv_y"][:] = recv_y.to(torch.bfloat16)
    tensors["sh"][:] = sh.to(torch.bfloat16)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_recv_x():
        return torch.randn(RECV_TOTAL_MAX, D) * 0.05

    def init_recv_expert_id():
        # Global expert ids in [EXPERTS_START_IDX, EXPERTS_START_IDX + N_LOCAL_EXPERTS).
        ids = torch.arange(RECV_TOTAL_MAX, dtype=torch.int32) % N_LOCAL_EXPERTS + EXPERTS_START_IDX
        return ids[torch.randperm(RECV_TOTAL_MAX)].reshape(RECV_TOTAL_MAX, 1)

    def init_recv_weights():
        w = torch.rand(RECV_TOTAL_MAX) + 0.1
        return (w / w.sum() * TOPK).float().reshape(RECV_TOTAL_MAX, 1)

    def init_x_local():
        return torch.randn(T, D) * 0.05

    # Pre-quantize all six weights once so the i8 / scale specs see consistent values.
    w1_bf16 = (torch.randn(N_LOCAL_EXPERTS, MOE_INTER, D) / D ** 0.5).to(torch.bfloat16)
    w3_bf16 = (torch.randn(N_LOCAL_EXPERTS, MOE_INTER, D) / D ** 0.5).to(torch.bfloat16)
    w2_bf16 = (torch.randn(N_LOCAL_EXPERTS, D, MOE_INTER) / MOE_INTER ** 0.5).to(torch.bfloat16)
    sw1_bf16 = (torch.randn(MOE_INTER, D) / D ** 0.5).to(torch.bfloat16)
    sw3_bf16 = (torch.randn(MOE_INTER, D) / D ** 0.5).to(torch.bfloat16)
    sw2_bf16 = (torch.randn(D, MOE_INTER) / MOE_INTER ** 0.5).to(torch.bfloat16)

    w1_i8, w1_s = _quant_w_per_channel(w1_bf16)
    w3_i8, w3_s = _quant_w_per_channel(w3_bf16)
    w2_i8, w2_s = _quant_w_per_channel(w2_bf16)
    sw1_i8, sw1_s = _quant_w_per_channel(sw1_bf16)
    sw3_i8, sw3_s = _quant_w_per_channel(sw3_bf16)
    sw2_i8, sw2_s = _quant_w_per_channel(sw2_bf16)

    return [
        TensorSpec("recv_x", [RECV_TOTAL_MAX, D], torch.bfloat16, init_value=init_recv_x),
        TensorSpec("recv_expert_id", [RECV_TOTAL_MAX, 1], torch.int32, init_value=init_recv_expert_id),
        TensorSpec("recv_weights", [RECV_TOTAL_MAX, 1], torch.float32, init_value=init_recv_weights),
        TensorSpec("x_local", [T, D], torch.bfloat16, init_value=init_x_local),
        TensorSpec("expert_w1", [N_LOCAL_EXPERTS, MOE_INTER, D], torch.int8, init_value=lambda: w1_i8),
        TensorSpec("expert_w1_scale", [N_LOCAL_EXPERTS, MOE_INTER], torch.float32, init_value=lambda: w1_s),
        TensorSpec("expert_w3", [N_LOCAL_EXPERTS, MOE_INTER, D], torch.int8, init_value=lambda: w3_i8),
        TensorSpec("expert_w3_scale", [N_LOCAL_EXPERTS, MOE_INTER], torch.float32, init_value=lambda: w3_s),
        TensorSpec("expert_w2", [N_LOCAL_EXPERTS, D, MOE_INTER], torch.int8, init_value=lambda: w2_i8),
        TensorSpec("expert_w2_scale", [N_LOCAL_EXPERTS, D], torch.float32, init_value=lambda: w2_s),
        TensorSpec("shared_w1", [MOE_INTER, D], torch.int8, init_value=lambda: sw1_i8),
        TensorSpec("shared_w1_scale", [MOE_INTER], torch.float32, init_value=lambda: sw1_s),
        TensorSpec("shared_w3", [MOE_INTER, D], torch.int8, init_value=lambda: sw3_i8),
        TensorSpec("shared_w3_scale", [MOE_INTER], torch.float32, init_value=lambda: sw3_s),
        TensorSpec("shared_w2", [D, MOE_INTER], torch.int8, init_value=lambda: sw2_i8),
        TensorSpec("shared_w2_scale", [D], torch.float32, init_value=lambda: sw2_s),
        TensorSpec("recv_y", [RECV_TOTAL_MAX, D], torch.bfloat16, is_output=True),
        TensorSpec("sh", [T, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import RunConfig, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=moe_expert_test,
        specs=build_tensor_specs(),
        golden_fn=golden_moe_expert,
        config=RunConfig(
            rtol=1e-3,
            atol=1e-3,
            compile=dict(dump_passes=True),
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
                runtime_profiling=args.runtime_profiling,
            ),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
