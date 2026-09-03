# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE routed local expert compute (decode, EP single-card).

Only the routed-expert path lives here. The shared expert was split out
into ``expert_shared.py``; both kernels are composed in ``moe.py``.
"""


import pypto.language as pl

from config import (ACTIVE as M, DECODE_BATCH, DECODE_SEQ, INT8_SCALE_MAX, INT8_AMAX_EPS,
                    EP_WORLD_SIZE, RECV_MAX)


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit

# EP layout / recv buffers (single-card view: kernel only sees the local shard)
N_LOCAL_EXPERTS = M.n_routed_experts // EP_WORLD_SIZE

# tiling
RECV_TILE = 16
INTER_K = 512
MM_INTER_TILE = 256
MM_GATE_INNER = 4
MX_GROUP = 32
K_SCALE = D // MX_GROUP
MX_K_TILE = 256
MX_MM_INTER_TILE = 128
MX_K_SCALE_GROUPS = MX_K_TILE // MX_GROUP
assert MOE_INTER % MX_MM_INTER_TILE == 0
assert D % MX_K_TILE == 0
ACT_INTER_TILE = 128
ACT_GATE_INNER = 4
D_OUT_TILE = 256
# h_tile_i8 store innermost = QUANT_TILE bytes (int8); 512 hits the a2a3 L2 cache
# line (perf_hint PH001 flagged the prior 256B store as sub-line).
QUANT_TILE = 512
D_OUT_TILE_ACT = 512
W2_INNER = 4
# One task covers the architecture's full hidden dimension.
W2_ACT_INNER = 8 if M.name == "flash" else 14
TILES_PER_EXPERT = RECV_MAX // RECV_TILE

assert RECV_MAX % RECV_TILE == 0, "RECV_MAX must be a whole number of RECV_TILE row-tiles"
# Every `<dim> // <tile>` used as a loop/task bound must divide exactly, or the
# bound silently truncates and part of the tensor is never written.
assert D % (W2_ACT_INNER * D_OUT_TILE_ACT) == 0, \
    "W2_ACT_INNER * D_OUT_TILE_ACT must divide D (otherwise the w2-dequant task count truncates)"
assert MOE_INTER % QUANT_TILE == 0 and D % D_OUT_TILE == 0 and D % D_OUT_TILE_ACT == 0


@pl.jit.inline(auto_scope=False)
def expert_routed(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.FP8E4M3FN],
    recv_mx_scale: pl.Tensor[[N_LOCAL_EXPERTS * RECV_MAX, K_SCALE], pl.FP8E8M0, pl.MX_A_ZZ],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.FP8E4M3FN],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.FP8E4M3FN],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    recv_y: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
):
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL_EXPERTS * RECV_MAX, D])
    recv_x_flat = pl.reshape(recv_x, [N_LOCAL_EXPERTS * RECV_MAX, D])
    routed_w1_flat = pl.reshape(routed_w1, [N_LOCAL_EXPERTS * D, MOE_INTER])
    routed_w3_flat = pl.reshape(routed_w3, [N_LOCAL_EXPERTS * D, MOE_INTER])
    with pl.scope():
        # Keep only the requantized SwiGLU result across the W1/W3 and W2 phases.
        # Full INT32 gate/up tensors scale with RECV_MAX and exhaust the task
        # heap in packed prefill. Retain only INT8 SwiGLU rows and their scales.
        h_i8 = pl.create_tensor([N_LOCAL_EXPERTS * RECV_MAX, MOE_INTER], dtype=pl.INT8)
        h_scale_dq = pl.create_tensor(
            [N_LOCAL_EXPERTS * RECV_MAX, 1], dtype=pl.FP32, manual_dep=True
        )
        # Produce one gate/up row tile at a time, immediately activate and quantize
        # it, then release the INT32/FP32 temporaries at the tile scope boundary.
        for local_i in pl.parallel(N_LOCAL_EXPERTS):
            flat_base = local_i * RECV_MAX

            n_rows = pl.read(recv_expert_count, [local_i, 0])
            n_tiles = (n_rows + RECV_TILE - 1) // RECV_TILE

            for t in pl.parallel(n_tiles):
                t0 = t * RECV_TILE
                flat_t0 = flat_base + t0
                valid_rows = pl.min(RECV_TILE, n_rows - t0)

                with pl.scope():
                    gate_tile_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32)
                    up_tile_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32)

                    with pl.spmd(MOE_INTER // MX_MM_INTER_TILE, name_hint="exp_gate_mx_mm"):
                        nb_idx = pl.tile.get_block_idx()
                        n0 = nb_idx * MX_MM_INTER_TILE
                        w1_row_base = local_i * D
                        w1_scale_row_base = local_i * K_SCALE

                        xs0 = pl.load(recv_x_flat, [flat_t0, 0], [RECV_TILE, MX_K_TILE], target_memory=pl.Mem.Mat)
                        xs_scale0 = pl.load(recv_mx_scale, [flat_t0, 0], [RECV_TILE, MX_K_SCALE_GROUPS], target_memory=pl.Mem.Mat)
                        w1_k0 = pl.load(routed_w1_flat, [w1_row_base, n0], [MX_K_TILE, MX_MM_INTER_TILE], target_memory=pl.Mem.Mat)
                        w1_scale0 = pl.load(routed_w1_scale, [w1_scale_row_base, n0], [MX_K_SCALE_GROUPS, MX_MM_INTER_TILE], target_memory=pl.Mem.Mat)

                        xs0 = pl.move(xs0, target_memory=pl.Mem.Left)
                        xs_scale0 = pl.move(xs_scale0, target_memory=pl.Mem.LeftScale)
                        w1_k0 = pl.move(w1_k0, target_memory=pl.Mem.Right)
                        w1_scale0 = pl.move(w1_scale0, target_memory=pl.Mem.RightScale)

                        gate_acc = pl.matmul_mx(xs0, xs_scale0, w1_k0, w1_scale0)
                        for k0 in pl.pipeline(MX_K_TILE, D, MX_K_TILE, stage=2):
                            ks = k0 // MX_GROUP
                            xs_k = pl.load(recv_x_flat, [flat_t0, k0], [RECV_TILE, MX_K_TILE], target_memory=pl.Mem.Mat)
                            xs_scale_k = pl.load(recv_mx_scale, [flat_t0, ks], [RECV_TILE, MX_K_SCALE_GROUPS], target_memory=pl.Mem.Mat)
                            w1_k = pl.load(routed_w1_flat, [w1_row_base + k0, n0], [MX_K_TILE, MX_MM_INTER_TILE], target_memory=pl.Mem.Mat)
                            w1_scale_k = pl.load(routed_w1_scale, [w1_scale_row_base + ks, n0], [MX_K_SCALE_GROUPS, MX_MM_INTER_TILE], target_memory=pl.Mem.Mat)

                            xs_k = pl.move(xs_k, target_memory=pl.Mem.Left)
                            xs_scale_k = pl.move(xs_scale_k, target_memory=pl.Mem.LeftScale)
                            w1_k = pl.move(w1_k, target_memory=pl.Mem.Right)
                            w1_scale_k = pl.move(w1_scale_k, target_memory=pl.Mem.RightScale)
                            gate_acc = pl.matmul_mx_acc(gate_acc, xs_k, xs_scale_k, w1_k, w1_scale_k)
                        gate_tile_fp32 = pl.store(gate_acc, [0, n0], gate_tile_fp32)

                    with pl.spmd(MOE_INTER // MX_MM_INTER_TILE, name_hint="exp_up_mx_mm"):
                        ub_idx = pl.tile.get_block_idx()
                        n0 = ub_idx * MX_MM_INTER_TILE
                        w3_row_base = local_i * D
                        w3_scale_row_base = local_i * K_SCALE
                        xs0 = pl.load(recv_x_flat, [flat_t0, 0], [RECV_TILE, MX_K_TILE], target_memory=pl.Mem.Mat)
                        xs_scale0 = pl.load(recv_mx_scale, [flat_t0, 0], [RECV_TILE, MX_K_SCALE_GROUPS], target_memory=pl.Mem.Mat)
                        w3_k0 = pl.load(routed_w3_flat, [w3_row_base, n0], [MX_K_TILE, MX_MM_INTER_TILE], target_memory=pl.Mem.Mat)
                        w3_scale0 = pl.load(routed_w3_scale, [w3_scale_row_base, n0], [MX_K_SCALE_GROUPS, MX_MM_INTER_TILE], target_memory=pl.Mem.Mat)

                        xs0 = pl.move(xs0, target_memory=pl.Mem.Left)
                        xs_scale0 = pl.move(xs_scale0, target_memory=pl.Mem.LeftScale)
                        w3_k0 = pl.move(w3_k0, target_memory=pl.Mem.Right)
                        w3_scale0 = pl.move(w3_scale0, target_memory=pl.Mem.RightScale)

                        up_acc = pl.matmul_mx(xs0, xs_scale0, w3_k0, w3_scale0)
                        for k0 in pl.pipeline(MX_K_TILE, D, MX_K_TILE, stage=2):
                            ks = k0 // MX_GROUP
                            xs_k = pl.load(recv_x_flat, [flat_t0, k0], [RECV_TILE, MX_K_TILE], target_memory=pl.Mem.Mat)
                            xs_scale_k = pl.load(recv_mx_scale, [flat_t0, ks], [RECV_TILE, MX_K_SCALE_GROUPS], target_memory=pl.Mem.Mat)
                            w3_k = pl.load(routed_w3_flat, [w3_row_base + k0, n0], [MX_K_TILE, MX_MM_INTER_TILE], target_memory=pl.Mem.Mat)
                            w3_scale_k = pl.load(routed_w3_scale, [w3_scale_row_base + ks, n0], [MX_K_SCALE_GROUPS, MX_MM_INTER_TILE], target_memory=pl.Mem.Mat)

                            xs_k = pl.move(xs_k, target_memory=pl.Mem.Left)
                            xs_scale_k = pl.move(xs_scale_k, target_memory=pl.Mem.LeftScale)
                            w3_k = pl.move(w3_k, target_memory=pl.Mem.Right)
                            w3_scale_k = pl.move(w3_scale_k, target_memory=pl.Mem.RightScale)
                            
                            up_acc = pl.matmul_mx_acc(up_acc, xs_k, xs_scale_k, w3_k, w3_scale_k)
                        up_tile_fp32 = pl.store(up_acc, [0, n0], up_tile_fp32)

                    h_tile_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32)
                    with pl.spmd(
                        MOE_INTER // (ACT_GATE_INNER * ACT_INTER_TILE),
                        name_hint="exp_gate_up_act",
                    ):
                        ab_idx = pl.tile.get_block_idx()
                        a_base = ab_idx * (ACT_GATE_INNER * ACT_INTER_TILE)
                        for ag in pl.pipeline(ACT_GATE_INNER, stage=2):
                            a0 = a_base + ag * ACT_INTER_TILE
                            gate_2d = gate_tile_fp32[:, a0 : a0 + ACT_INTER_TILE]
                            up_2d = up_tile_fp32[:, a0 : a0 + ACT_INTER_TILE]
                            if SWIGLU_LIMIT > 0.0:
                                gate_2d = pl.minimum(gate_2d, SWIGLU_LIMIT)
                                up_2d = pl.maximum(pl.minimum(up_2d, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_2d)), 1.0))
                            silu = pl.mul(gate_2d, sigmoid)
                            gated = pl.mul(silu, up_2d)
                            gated_valid = pl.set_validshape(gated, valid_rows, ACT_INTER_TILE)
                            h_tile_fp32[:, a0 : a0 + ACT_INTER_TILE] = pl.fillpad(
                                gated_valid, pad_value=pl.PadValue.zero
                            )

                    h_tile_i8 = h_i8[flat_t0 : flat_t0 + RECV_TILE]
                    h_tile_scale_dq = h_scale_dq[flat_t0 : flat_t0 + RECV_TILE]
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_h_q"):
                        eh_amax = pl.full([1, RECV_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                        for k0 in pl.pipeline(0, MOE_INTER, QUANT_TILE, stage=2):
                            eh_a_f32 = h_tile_fp32[:, k0 : k0 + QUANT_TILE]
                            eh_a_abs = pl.maximum(eh_a_f32, pl.neg(eh_a_f32))
                            eh_a_max = pl.reshape(pl.row_max(eh_a_abs), [1, RECV_TILE])
                            eh_amax = pl.maximum(eh_amax, eh_a_max)
                        eh_sq_row = pl.div(
                            pl.full([1, RECV_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX),
                            eh_amax,
                        )
                        h_tile_scale_dq[:, :] = pl.reshape(pl.recip(eh_sq_row), [RECV_TILE, 1])
                        eh_sq_col = pl.reshape(eh_sq_row, [RECV_TILE, 1])
                        for k1 in pl.pipeline(0, MOE_INTER, QUANT_TILE, stage=2):
                            eh_q_f32 = h_tile_fp32[:, k1 : k1 + QUANT_TILE]
                            eh_q_scaled = pl.row_expand_mul(eh_q_f32, eh_sq_col)
                            eh_q_i32 = pl.cast(eh_q_scaled, target_type=pl.INT32, mode="rint")
                            eh_q_half = pl.cast(eh_q_i32, target_type=pl.FP16, mode="round")
                            h_tile_i8[:, k1 : k1 + QUANT_TILE] = pl.cast(
                                eh_q_half, target_type=pl.INT8, mode="trunc"
                            )

        with pl.scope():
            for local_e in pl.parallel(N_LOCAL_EXPERTS):
                e_flat_base = local_e * RECV_MAX

                e_rows = pl.read(recv_expert_count, [local_e, 0])
                e_tiles = (e_rows + RECV_TILE - 1) // RECV_TILE

                for tt in pl.parallel(e_tiles):
                    tt0 = tt * RECV_TILE
                    flat_tt0 = e_flat_base + tt0
                    h_tile_i8 = h_i8[flat_tt0 : flat_tt0 + RECV_TILE]
                    h_tile_scale_dq = h_scale_dq[flat_tt0 : flat_tt0 + RECV_TILE]

                    y_i32 = pl.create_tensor([RECV_TILE, D], dtype=pl.INT32)
                    with pl.spmd(
                        D // (W2_INNER * D_OUT_TILE),
                        name_hint="exp_w2_mm",
                    ):
                        wb_idx = pl.tile.get_block_idx()
                        d_base = wb_idx * (W2_INNER * D_OUT_TILE)
                        for dg in pl.range(W2_INNER):
                            d0 = d_base + dg * D_OUT_TILE
                            y_acc = pl.create_tensor([1, RECV_TILE, D_OUT_TILE], dtype=pl.INT32)
                            for k0 in pl.pipeline(0, MOE_INTER, INTER_K, stage=2):
                                h_k = h_tile_i8[:, k0 : k0 + INTER_K]
                                w2_k = routed_w2[local_e : local_e + 1, d0 : d0 + D_OUT_TILE, k0 : k0 + INTER_K]
                                if k0 == 0:
                                    y_acc = pl.matmul(h_k, w2_k, b_trans=True, out_dtype=pl.INT32)
                                else:
                                    y_acc = pl.matmul_acc(y_acc, h_k, w2_k, b_trans=True)
                            y_i32[:, d0 : d0 + D_OUT_TILE] = pl.reshape(y_acc, [RECV_TILE, D_OUT_TILE])

                    recv_y_tile = pl.create_tensor([RECV_TILE, D], dtype=pl.BF16)
                    with pl.spmd(
                        D // (W2_ACT_INNER * D_OUT_TILE_ACT),
                        name_hint="exp_w2_act",
                    ):
                        db_idx = pl.tile.get_block_idx()
                        act_d_base = db_idx * (W2_ACT_INNER * D_OUT_TILE_ACT)
                        w_col_blk = pl.reshape(
                            recv_weights[local_e : local_e + 1, tt0 : tt0 + RECV_TILE],
                            [RECV_TILE, 1],
                        )
                        row_scale_blk = pl.mul(h_tile_scale_dq, w_col_blk)
                        for dg in pl.pipeline(W2_ACT_INNER, stage=2):
                            act_d0 = act_d_base + dg * D_OUT_TILE_ACT
                            y_2d_i32 = y_i32[:, act_d0 : act_d0 + D_OUT_TILE_ACT]
                            w2_scale_chunk = routed_w2_scale[local_e : local_e + 1, act_d0 : act_d0 + D_OUT_TILE_ACT]
                            y_2d = pl.cast(y_2d_i32, target_type=pl.FP32, mode="none")
                            y_2d = pl.col_expand_mul(pl.row_expand_mul(y_2d, row_scale_blk), w2_scale_chunk)
                            recv_y_tile[:, act_d0 : act_d0 + D_OUT_TILE_ACT] = pl.cast(
                                y_2d, target_type=pl.BF16, mode="rint"
                            )
                    recv_y_flat = pl.assemble(recv_y_flat, recv_y_tile, [flat_tt0, 0])

    return recv_y


@pl.jit
def expert_routed_test(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.FP8E4M3FN],
    recv_mx_scale: pl.Tensor[[N_LOCAL_EXPERTS * RECV_MAX, K_SCALE], pl.FP8E8M0, pl.MX_A_ZZ],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.FP8E4M3FN],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.FP8E4M3FN],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    recv_y: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16]],
):
    expert_routed(
        recv_x, recv_mx_scale, recv_weights, recv_expert_count,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        recv_y,
    )
    return recv_y


def _int8_quant_per_row(x):
    """Per-row (per-token) INT8 symmetric quant matching v3.2 scope2 Stage 2.6."""
    import torch
    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


def golden_expert_routed(tensors):
    """Torch reference: MX W1/W3 + INT8 W2."""
    import torch
    import torch.nn.functional as F

    from mx_utils import decode_e8m0_codes, matmul_mx_golden

    recv_x = tensors["recv_x"]
    recv_mx_scale = decode_e8m0_codes(tensors["recv_mx_scale"], side="a").reshape(
        N_LOCAL_EXPERTS, RECV_MAX, K_SCALE,
    )
    recv_weights = tensors["recv_weights"].float()
    recv_expert_count = tensors["recv_expert_count"]
    w1_fp8 = tensors["routed_w1"]
    w1_scale = decode_e8m0_codes(tensors["routed_w1_scale"], side="b").reshape(
        N_LOCAL_EXPERTS, K_SCALE, MOE_INTER,
    )
    w3_fp8 = tensors["routed_w3"]
    w3_scale = decode_e8m0_codes(tensors["routed_w3_scale"], side="b").reshape(
        N_LOCAL_EXPERTS, K_SCALE, MOE_INTER,
    )
    w2_i8 = tensors["routed_w2"]
    w2_scale = tensors["routed_w2_scale"].float()

    recv_y = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D)
    for e in range(N_LOCAL_EXPERTS):
        n_rows = int(recv_expert_count[e, 0].item())
        if n_rows == 0:
            continue
        x_sub = recv_x[e, :n_rows, :]
        x_scale = recv_mx_scale[e, :n_rows, :]
        w_per_row = recv_weights[e, :n_rows].reshape(-1, 1)

        gate = matmul_mx_golden(x_sub, x_scale, w1_fp8[e], w1_scale[e])
        up = matmul_mx_golden(x_sub, x_scale, w3_fp8[e], w3_scale[e])
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h = F.silu(gate) * up
        h_i8, h_sd = _int8_quant_per_row(h)
        y_int = h_i8.to(torch.int32) @ w2_i8[e].to(torch.int32).T
        recv_y[e, :n_rows, :] = y_int.to(torch.float32) * (h_sd * w_per_row) * w2_scale[e].unsqueeze(0)

    tensors["recv_y"][:] = recv_y.to(torch.bfloat16)


def gen_routed_weight_int8_w2(shape, dequant_std):
    """Synthesize INT8 W2 weights + FP32 scales (W2 stays INT8)."""
    import torch

    FP4_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    FP4_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    FP4_MAX, TINY = 6.0, 1e-20
    GROUP = 32
    CHUNK_ELEMS = 1 << 25

    *lead, out, inn = shape
    n_lead = 1
    for dim in lead:
        n_lead *= dim

    W = torch.randn(*shape).reshape(n_lead, out, inn)
    w_i8 = torch.empty(n_lead, out, inn, dtype=torch.int8)
    scale = torch.empty(n_lead, out, 1, dtype=torch.float32)

    step = max(1, CHUNK_ELEMS // (out * inn))
    for i0 in range(0, n_lead, step):
        w = W[i0:i0 + step]
        wg = w.reshape(-1, out, inn // GROUP, GROUP)
        absw = wg.abs()
        grp_scale = torch.exp2(torch.ceil(torch.log2((absw.amax(-1, keepdim=True) / FP4_MAX).clamp_min(TINY))))
        idx = torch.bucketize(absw.div_(grp_scale), FP4_MID).clamp_max_(7)
        wq = (torch.sign(wg) * FP4_MAG[idx]).mul_(grp_scale).reshape(w.shape)
        amax = wq.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
        chan_scale = amax / INT8_SCALE_MAX
        w_i8[i0:i0 + step] = torch.round(wq.div_(chan_scale)).clamp_(
            -INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
        scale[i0:i0 + step] = chan_scale
    del W

    scale = (scale * (dequant_std / (w_i8.float() * scale).std())).squeeze(-1).float()
    return w_i8.reshape(*shape), scale.reshape(*lead, out)


def gen_routed_mx_w13(n_experts, dequant_std, seed_base=0):
    """MXFP8 W1/W3 per local expert in Cube ``[D, MOE_INTER]`` layout."""
    import torch
    from mx_utils import gen_mxfp8_weight_kn_device

    w1_list, w1s_list, w3_list, w3s_list = [], [], [], []
    for e in range(n_experts):
        w1, w1s = gen_mxfp8_weight_kn_device(MOE_INTER, D, dequant_std, seed=seed_base + e * 2)
        w3, w3s = gen_mxfp8_weight_kn_device(MOE_INTER, D, dequant_std, seed=seed_base + e * 2 + 1)
        w1_list.append(w1)
        w1s_list.append(w1s)
        w3_list.append(w3)
        w3s_list.append(w3s)
    return (
        torch.stack(w1_list),
        torch.stack(w1s_list).reshape(n_experts * K_SCALE, MOE_INTER),
        torch.stack(w3_list),
        torch.stack(w3s_list).reshape(n_experts * K_SCALE, MOE_INTER),
    )


def build_tensor_specs():
    import torch
    from golden import TensorSpec
    from mx_utils import host_mxfp8_activation

    # Across-layer-mean dequant std (typical layer) of the real routed experts.
    ROUTED_DEQUANT_STD = {"w1": 2.47e-2, "w2": 2.44e-2, "w3": 2.46e-2}

    total = B * S * M.num_experts_per_tok
    counts = torch.bincount(
        torch.randint(0, N_LOCAL_EXPERTS, (total,)),
        minlength=N_LOCAL_EXPERTS,
    ).to(torch.int32)
    counts_2d = counts.reshape(N_LOCAL_EXPERTS, 1)

    x_bf16 = torch.randn(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.bfloat16)
    valid_mask_3d = (
        torch.arange(RECV_MAX).reshape(1, RECV_MAX, 1) < counts.reshape(N_LOCAL_EXPERTS, 1, 1)
    )
    recv_x_pre = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.float8_e4m3fn)
    recv_mx_scale_pre = torch.zeros(N_LOCAL_EXPERTS * RECV_MAX, K_SCALE, dtype=torch.float8_e8m0fnu)
    for e in range(N_LOCAL_EXPERTS):
        n = int(counts_2d[e, 0].item())
        if n > 0:
            xf, xs = host_mxfp8_activation(x_bf16[e, :n, :])
            recv_x_pre[e, :n, :] = xf
            recv_mx_scale_pre[e * RECV_MAX : e * RECV_MAX + n, :] = xs
    valid_mask_2d = valid_mask_3d.squeeze(-1)
    recv_x_pre = torch.where(
        valid_mask_3d,
        recv_x_pre,
        torch.zeros_like(recv_x_pre),
    )

    def init_recv_x():
        return recv_x_pre

    def init_recv_mx_scale():
        return recv_mx_scale_pre

    def init_recv_expert_count():
        return counts_2d

    recv_weights_pre = torch.rand(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.float32)
    recv_weights_pre = torch.where(
        valid_mask_2d, recv_weights_pre, torch.zeros_like(recv_weights_pre)
    )

    def init_recv_weights():
        return recv_weights_pre

    rw1_fp8, rw1_scale, rw3_fp8, rw3_scale = gen_routed_mx_w13(
        N_LOCAL_EXPERTS, ROUTED_DEQUANT_STD["w1"], seed_base=10
    )
    w2_i8, w2_s = gen_routed_weight_int8_w2(
        (N_LOCAL_EXPERTS, D, MOE_INTER), ROUTED_DEQUANT_STD["w2"]
    )

    fp8 = torch.float8_e4m3fn
    fp8_e8m0 = torch.float8_e8m0fnu

    return [
        TensorSpec("recv_x", [N_LOCAL_EXPERTS, RECV_MAX, D], fp8, init_value=init_recv_x),
        TensorSpec("recv_mx_scale", [N_LOCAL_EXPERTS * RECV_MAX, K_SCALE], fp8_e8m0, init_value=init_recv_mx_scale),
        TensorSpec("recv_weights", [N_LOCAL_EXPERTS, RECV_MAX], torch.float32, init_value=init_recv_weights),
        TensorSpec("recv_expert_count", [N_LOCAL_EXPERTS, 1], torch.int32, init_value=init_recv_expert_count),
        TensorSpec("routed_w1", [N_LOCAL_EXPERTS, D, MOE_INTER], fp8, init_value=lambda: rw1_fp8),
        TensorSpec("routed_w1_scale", [N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], fp8_e8m0, init_value=lambda: rw1_scale),
        TensorSpec("routed_w3", [N_LOCAL_EXPERTS, D, MOE_INTER], fp8, init_value=lambda: rw3_fp8),
        TensorSpec("routed_w3_scale", [N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], fp8_e8m0, init_value=lambda: rw3_scale),
        TensorSpec("routed_w2", [N_LOCAL_EXPERTS, D, MOE_INTER], torch.int8, init_value=lambda: w2_i8),
        TensorSpec("routed_w2_scale", [N_LOCAL_EXPERTS, D], torch.float32, init_value=lambda: w2_s),
        TensorSpec("recv_y", [N_LOCAL_EXPERTS, RECV_MAX, D], torch.bfloat16),
    ]


def active_recv_ratio_reldiff(*, diff_thd, pct_thd):
    """Validate compact expert rows without borrowing padding capacity.

    ``recv_y`` has a static ``RECV_MAX`` extent for every local expert, but
    only the prefix described by ``recv_expert_count`` is computed. Numerical
    tolerance applies to those active rows only; every inactive row must stay
    bitwise equal to golden so stale or out-of-range writes cannot be hidden by
    the much larger padded buffer.
    """
    import torch

    from golden import ratio_reldiff

    active_compare = ratio_reldiff(diff_thd=diff_thd, pct_thd=pct_thd)

    def compare(actual, expected, **kwargs):
        if actual.shape != expected.shape:
            return False, (
                f"    output shape mismatch: actual={tuple(actual.shape)} "
                f"expected={tuple(expected.shape)}"
            )
        if actual.ndim != 3:
            return False, f"    recv_y must have rank 3, got {tuple(actual.shape)}"

        counts = kwargs.get("inputs", {}).get("recv_expert_count")
        if counts is None:
            return False, "    compare_fn misconfigured: missing input 'recv_expert_count'"
        counts = counts.cpu().to(torch.int64).reshape(-1)
        expert_count, recv_max, _ = actual.shape
        if counts.numel() != expert_count:
            return False, (
                f"    recv_expert_count has {counts.numel()} values, "
                f"expected {expert_count}"
            )
        invalid = (counts < 0) | (counts > recv_max)
        if invalid.any().item():
            expert = int(invalid.nonzero(as_tuple=False)[0].item())
            return False, (
                f"    recv_expert_count[{expert}]={int(counts[expert].item())} "
                f"is outside [0, {recv_max}]"
            )

        rows = torch.arange(recv_max, dtype=torch.int64).reshape(1, -1)
        active = rows < counts.reshape(-1, 1)
        inactive = ~active

        actual_f = actual.float()
        expected_f = expected.float()
        for label, values in (("actual", actual_f), ("expected", expected_f)):
            invalid_values = ~torch.isfinite(values)
            if invalid_values.any().item():
                return False, (
                    f"    illegal values in {label}: "
                    f"count={int(invalid_values.sum().item())}"
                )

        if inactive.any().item() and not torch.equal(actual[inactive], expected[inactive]):
            changed = int((actual[inactive] != expected[inactive]).sum().item())
            return False, f"    inactive recv_y rows changed: changed_values={changed}"

        if not active.any().item():
            return True, ""
        ok, detail = active_compare(actual[active], expected[active], **kwargs)
        if ok:
            return True, ""
        return False, f"    active recv_y rows:\n{detail}"

    compare.__name__ = (
        f"active_recv_ratio_reldiff(diff_thd={diff_thd}, pct_thd={pct_thd})"
    )
    return compare


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a5",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        fn=expert_routed_test,
        specs=build_tensor_specs(),
        golden_fn=golden_expert_routed,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            # BF16 recv_y, ~1 ULP. Gen weights reproduce real(L21): 0.016% vs 0.015% of points > 1e-3.
            "recv_y": active_recv_ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
