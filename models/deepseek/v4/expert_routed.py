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

from config import (FLASH as M, DECODE_BATCH, DECODE_SEQ, INT8_SCALE_MAX, INT8_AMAX_EPS,
                    EP_WORLD_SIZE, EP_RANK, RECV_MAX)


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit

# EP layout / recv buffers (single-card view: kernel only sees the local shard)
N_LOCAL_EXPERTS = M.n_routed_experts // EP_WORLD_SIZE
EXPERTS_START_IDX = EP_RANK * N_LOCAL_EXPERTS

# tiling
# Sized for the large-batch / decode-batch target of ~24 rows/expert. RECV_TILE=32
# runs one row-tile per expert at that occupancy (no weight re-stream); the <=8-row
# pure-decode path can drop it to 16 (which re-streams the weight above 16 rows).
# RECV_TILE must divide RECV_MAX (per-expert row buffer) to avoid over-read.
RECV_TILE = 32
# K_TILE=512 (not 1024): each split gate/up cube task holds ONE weight L1 tile of
# MM_INTER_TILE*K_TILE, double-buffered, so Mat ~= MM_INTER_TILE*K_TILE*1*2(buf). At
# K_TILE=512 that lets MM_INTER_TILE reach 256 (256*512*2 = 256KB, fits the 512KB Mat);
# K_TILE=1024 would need 512KB and cap N at 128. Both N and K stay >= the 512B
# B-cache-line (B is K-contiguous under b_trans) so MTE2 stays efficient.
K_TILE = 512
INTER_K = 512
# gate/up N-tiling is split across the decoupled cube (AIC) and vector (AIV) scopes,
# meeting only through the gate_i32/up_i32 GM buffer, so each sizes its N-frag to its
# own bottleneck -- the cube frag to L1/Mat (see K_TILE above), the vector frag to UB.
MM_INTER_TILE = 256   # cube N frag; L1/Mat bound: MM_INTER_TILE*K_TILE*1(split weight)*2(buf) <= 512KB
MM_GATE_INNER = 4
ACT_INTER_TILE = 128   # vector N frag; UB bound, independent of the cube frag
ACT_GATE_INNER = 4
D_OUT_TILE = 256   # w2 N frag; UB bound. The decoupled vector scope frees the room the
                   # fused form lacked; 512 overflows AllocateMemoryAddr.
QUANT_TILE = 256
D_OUT_TILE_ACT = 512
W2_INNER = 4       # keeps w2 spmd N-block count D//(W2_INNER*D_OUT_TILE)=4 with D_OUT_TILE=256
W2_ACT_INNER = 8

assert MOE_INTER % (MM_GATE_INNER * MM_INTER_TILE) == 0, "gate/up cube tiling must cover MOE_INTER"
assert MOE_INTER % (ACT_GATE_INNER * ACT_INTER_TILE) == 0, "gate/up vector tiling must cover MOE_INTER"
assert D % K_TILE == 0, "gate/up K-loop must cover D"
assert D % (W2_INNER * D_OUT_TILE) == 0, "w2 cube tiling must cover D"
assert D % (W2_ACT_INNER * D_OUT_TILE_ACT) == 0, "w2 vector tiling must cover D"


@pl.jit.inline
def expert_routed(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    recv_y: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
):
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL_EXPERTS * RECV_MAX, D])

    # Iterate each local expert, then process its rows in tiles of RECV_TILE.
    for local_i in pl.parallel(N_LOCAL_EXPERTS):
        n_rows = pl.read(recv_expert_count, [local_i, 0])
        n_tiles = (n_rows + RECV_TILE - 1) // RECV_TILE
        flat_base = local_i * RECV_MAX  # row offset in recv_y_flat

        for t in pl.parallel(n_tiles):
            t0 = t * RECV_TILE
            flat_t0 = flat_base + t0

            valid_rows = pl.min(RECV_TILE, n_rows - t0)

            # Stage 1a: gate/up. Decoupled from the vector epilogue (cf. the un-mixed
            # qproj in qkv_proj_rope.py): a pure-cube matmul scope writes INT32
            # accumulators to GM, then a separate vector scope does dequant + SwiGLU +
            # routing-weight mul. The cube is split again into a gate (w1) task and an
            # up (w3) task so each holds only ONE weight L1 tile -- that halves Mat
            # (MM_INTER_TILE*K_TILE*1*2buf), letting MM_INTER_TILE reach 256 at the
            # cache-line-safe K_TILE=512; the fused form held w1+w3 and so capped N at
            # 128. Costs one extra recv_x L1 reload per task (x is tiny vs the weights).
            h_tile_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32)
            gate_i32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT32)
            up_i32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT32)

            for nb_idx in pl.spmd(MOE_INTER // (MM_GATE_INNER * MM_INTER_TILE), name_hint="exp_gate_mm"):
                n_base = nb_idx * (MM_GATE_INNER * MM_INTER_TILE)
                for ng in pl.range(MM_GATE_INNER):
                    n0 = n_base + ng * MM_INTER_TILE
                    # Seed the accumulator on the first K iter with a plain matmul
                    # (matmul_acc from a zero-init carry trips the TLOAD DN->NZ ISA
                    # assertion, pypto#1540). Keep that k0 == 0 conditional inside the
                    # pl.pipeline loop rather than peeling it out as a prologue, so the
                    # first chunk's weight MTE2 load overlaps the rest of the pipeline.
                    # This cube is MTE2-bound, so the overlap fills that gap; INT32
                    # accumulation order is unchanged.
                    gate_acc = pl.create_tensor([1, RECV_TILE, MM_INTER_TILE], dtype=pl.INT32)
                    for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                        x_k = recv_x[local_i : local_i + 1, t0 : t0 + RECV_TILE, k0 : k0 + K_TILE]
                        w1_k = routed_w1[local_i : local_i + 1, n0 : n0 + MM_INTER_TILE, k0 : k0 + K_TILE]
                        if k0 == 0:
                            gate_acc = pl.matmul(x_k, w1_k, b_trans=True, out_dtype=pl.INT32)
                        else:
                            gate_acc = pl.matmul_acc(gate_acc, x_k, w1_k, b_trans=True)
                    gate_i32[:, n0 : n0 + MM_INTER_TILE] = pl.reshape(gate_acc, [RECV_TILE, MM_INTER_TILE])

            for nb_idx in pl.spmd(MOE_INTER // (MM_GATE_INNER * MM_INTER_TILE), name_hint="exp_up_mm"):
                n_base = nb_idx * (MM_GATE_INNER * MM_INTER_TILE)
                for ng in pl.range(MM_GATE_INNER):
                    n0 = n_base + ng * MM_INTER_TILE
                    up_acc = pl.create_tensor([1, RECV_TILE, MM_INTER_TILE], dtype=pl.INT32)
                    for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                        x_k = recv_x[local_i : local_i + 1, t0 : t0 + RECV_TILE, k0 : k0 + K_TILE]
                        w3_k = routed_w3[local_i : local_i + 1, n0 : n0 + MM_INTER_TILE, k0 : k0 + K_TILE]
                        if k0 == 0:
                            up_acc = pl.matmul(x_k, w3_k, b_trans=True, out_dtype=pl.INT32)
                        else:
                            up_acc = pl.matmul_acc(up_acc, x_k, w3_k, b_trans=True)
                    up_i32[:, n0 : n0 + MM_INTER_TILE] = pl.reshape(up_acc, [RECV_TILE, MM_INTER_TILE])

            for nb_idx in pl.spmd(MOE_INTER // (ACT_GATE_INNER * ACT_INTER_TILE), name_hint="exp_gate_up_act"):
                n_base = nb_idx * (ACT_GATE_INNER * ACT_INTER_TILE)
                for ng in pl.pipeline(ACT_GATE_INNER, stage=2):
                    n0 = n_base + ng * ACT_INTER_TILE
                    gate_2d_i32 = gate_i32[:, n0 : n0 + ACT_INTER_TILE]
                    up_2d_i32 = up_i32[:, n0 : n0 + ACT_INTER_TILE]
                    recv_x_scale_dq = pl.reshape(recv_scale_dq[local_i : local_i + 1, t0 : t0 + RECV_TILE], [RECV_TILE, 1])
                    w1_scale_chunk = routed_w1_scale[local_i : local_i + 1, n0 : n0 + ACT_INTER_TILE]
                    w3_scale_chunk = routed_w3_scale[local_i : local_i + 1, n0 : n0 + ACT_INTER_TILE]
                    gate_2d = pl.cast(gate_2d_i32, target_type=pl.FP32, mode="none")
                    up_2d = pl.cast(up_2d_i32, target_type=pl.FP32, mode="none")
                    gate_2d = pl.col_expand_mul(pl.row_expand_mul(gate_2d, recv_x_scale_dq), w1_scale_chunk)
                    up_2d = pl.col_expand_mul(pl.row_expand_mul(up_2d, recv_x_scale_dq), w3_scale_chunk)
                    if SWIGLU_LIMIT > 0.0:
                        gate_2d = pl.minimum(gate_2d, SWIGLU_LIMIT)
                        up_2d = pl.maximum(pl.minimum(up_2d, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                    sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_2d)), 1.0))
                    silu = pl.mul(gate_2d, sigmoid)
                    gated = pl.mul(silu, up_2d)
                    # Zero rows >= valid_rows so dirty recv_x tail rows don't leak into recv_y.
                    gated_valid = pl.set_validshape(gated, valid_rows, ACT_INTER_TILE)
                    gated_masked = pl.fillpad(gated_valid, pad_value=pl.PadValue.zero)
                    h_tile_fp32[:, n0 : n0 + ACT_INTER_TILE] = gated_masked

            # Per-row A8 requant of h_tile (amax across full MOE_INTER row).
            h_tile_i8 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT8)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_h_q"):
                eh_amax = pl.full([1, RECV_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                for k0 in pl.pipeline(0, MOE_INTER, QUANT_TILE, stage=2):
                    eh_a_f32 = h_tile_fp32[:, k0 : k0 + QUANT_TILE]
                    eh_a_abs = pl.maximum(eh_a_f32, pl.neg(eh_a_f32))
                    eh_a_max = pl.reshape(pl.row_max(eh_a_abs), [1, RECV_TILE])
                    eh_amax = pl.maximum(eh_amax, eh_a_max)
                eh_sq_row = pl.div(
                    pl.full([1, RECV_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), eh_amax
                )
                h_tile_scale_dq = pl.reshape(pl.recip(eh_sq_row), [RECV_TILE, 1])
                eh_sq_col = pl.reshape(eh_sq_row, [RECV_TILE, 1])
                for k1 in pl.pipeline(0, MOE_INTER, QUANT_TILE, stage=2):
                    eh_q_f32 = h_tile_fp32[:, k1 : k1 + QUANT_TILE]
                    eh_q_scaled = pl.row_expand_mul(eh_q_f32, eh_sq_col)
                    eh_q_i32 = pl.cast(eh_q_scaled, target_type=pl.INT32, mode="rint")
                    eh_q_half = pl.cast(eh_q_i32, target_type=pl.FP16, mode="round")
                    h_tile_i8[:, k1 : k1 + QUANT_TILE] = pl.cast(eh_q_half, target_type=pl.INT8, mode="trunc")

            # Stage 1b: w2. Decoupled like gate/up -- pure-cube matmul scope
            # (INT32 -> GM) then a separate vector scope (dequant + routing-weight
            # mul + BF16 write). The routing weight is applied row-wise here (vs.
            # the legacy single-card path that applied it in combine's reduce);
            # multi-card combine then just sums the TOPK rows per token.
            y_i32 = pl.create_tensor([RECV_TILE, D], dtype=pl.INT32)

            for db_idx in pl.spmd(D // (W2_INNER * D_OUT_TILE), name_hint="exp_w2_mm"):
                d_base = db_idx * (W2_INNER * D_OUT_TILE)
                for dg in pl.range(W2_INNER):
                    d0 = d_base + dg * D_OUT_TILE
                    # Seed the accumulator on the first K iter with a plain matmul
                    # (pypto#1540, see exp_gate_mm), keeping the k0 == 0 conditional
                    # inside the pl.pipeline loop so each weight MTE2 load overlaps the
                    # previous tile's cube compute. INT32 accumulation order is unchanged.
                    y_acc = pl.create_tensor([1, RECV_TILE, D_OUT_TILE], dtype=pl.INT32)
                    for k0 in pl.pipeline(0, MOE_INTER, INTER_K, stage=2):
                        h_k = h_tile_i8[:, k0 : k0 + INTER_K]
                        w2_k = routed_w2[local_i : local_i + 1, d0 : d0 + D_OUT_TILE, k0 : k0 + INTER_K]
                        if k0 == 0:
                            y_acc = pl.matmul(h_k, w2_k, b_trans=True, out_dtype=pl.INT32)
                        else:
                            y_acc = pl.matmul_acc(y_acc, h_k, w2_k, b_trans=True)
                    y_i32[:, d0 : d0 + D_OUT_TILE] = pl.reshape(y_acc, [RECV_TILE, D_OUT_TILE])

            for db_idx in pl.spmd(D // (W2_ACT_INNER * D_OUT_TILE_ACT), name_hint="exp_w2_act"):
                d_base = db_idx * (W2_ACT_INNER * D_OUT_TILE_ACT)
                # Fold the per-row h-dequant scale and the per-row routing weight
                # into a single per-row scale, computed once per spmd block
                # (instead of an extra row_expand_mul per d-tile).
                w_col_blk = pl.reshape(
                    recv_weights[local_i : local_i + 1, t0 : t0 + RECV_TILE],
                    [RECV_TILE, 1],
                )
                row_scale_blk = pl.mul(h_tile_scale_dq, w_col_blk)
                for dg in pl.pipeline(W2_ACT_INNER, stage=2):
                    d0 = d_base + dg * D_OUT_TILE_ACT
                    y_2d_i32 = y_i32[:, d0 : d0 + D_OUT_TILE_ACT]
                    w2_scale_chunk = routed_w2_scale[local_i : local_i + 1, d0 : d0 + D_OUT_TILE_ACT]
                    y_2d = pl.cast(y_2d_i32, target_type=pl.FP32, mode="none")
                    y_2d = pl.col_expand_mul(pl.row_expand_mul(y_2d, row_scale_blk), w2_scale_chunk)
                    recv_y_flat[flat_t0 : flat_t0 + RECV_TILE, d0 : d0 + D_OUT_TILE_ACT] = pl.cast(
                        y_2d, target_type=pl.BF16, mode="rint"
                    )

    # The @pl.inline parser requires inline call expressions to have a return
    # value; recv_y is convenient because it's already pl.Out.
    return recv_y


@pl.jit
def expert_routed_test(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    recv_y: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16]],
):
    expert_routed(
        recv_x, recv_scale_dq, recv_weights, recv_expert_count,
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
    """Torch reference for the routed expert. recv_y is the per-row routing-
    weight-scaled SwiGLU output, ready for combine reduce to simply sum.

    Per-expert layout: recv_x[e, 0:cnt[e], :] is the valid INT8 receive
    payload; recv_y[e, cnt[e]:, :] stays at zero."""
    import torch
    import torch.nn.functional as F

    def dequant_w(w_i8, w_scale):
        return w_i8.to(torch.float32) * w_scale.unsqueeze(-1)

    recv_x_i8 = tensors["recv_x"]  # INT8, pre-quantized in dispatch
    recv_scale_dq = tensors["recv_scale_dq"].float()  # [E, RECV_MAX]
    recv_weights = tensors["recv_weights"].float()  # [E, RECV_MAX]
    recv_expert_count = tensors["recv_expert_count"]  # [E, 1] int32
    w1 = dequant_w(tensors["routed_w1"], tensors["routed_w1_scale"].float())
    w3 = dequant_w(tensors["routed_w3"], tensors["routed_w3_scale"].float())
    w2 = dequant_w(tensors["routed_w2"], tensors["routed_w2_scale"].float())

    recv_y = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D)
    for e in range(N_LOCAL_EXPERTS):
        n_rows = int(recv_expert_count[e, 0].item())
        if n_rows == 0:
            continue
        x_sub_i8 = recv_x_i8[e, :n_rows, :]
        x_sub_sd = recv_scale_dq[e, :n_rows].reshape(-1, 1)
        x_sub_q = x_sub_i8.float() * x_sub_sd
        w_per_row = recv_weights[e, :n_rows].reshape(-1, 1)

        gate = x_sub_q @ w1[e].T
        up = x_sub_q @ w3[e].T
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h = F.silu(gate) * up
        # A8 requant before w2 matmul.
        h_i8, h_sd = _int8_quant_per_row(h)
        h = h_i8.float() * (h_sd * w_per_row)
        recv_y[e, :n_rows, :] = h @ w2[e].T

    tensors["recv_y"][:] = recv_y.to(torch.bfloat16)


def gen_routed_weight(shape, dequant_std):
    """Synthesize a routed-expert per-channel-symmetric INT8 weight + FP32 scale by
    simulating the real DeepSeek-V4-Flash MXFP4 routed-expert quant grid (e2m1, per-32-group
    E8M0 scale), then re-quantizing per-output-channel. A plain ``randn`` INT8 is wrong:
    routed collapses onto ~37 discrete levels with an ~11.6% zero spike (the FP4 grid) and a
    per-channel scale CV ~0.09 (the fine group scale flattens it). Per-output-channel INT8 is
    scale-invariant, so the level structure / zero spike emerge from the grid alone and
    ``dequant_std`` only sets the absolute scale magnitude. (shared experts use a different
    grid -- see expert_shared.gen_shared_weight.)

    ``shape`` last dim = reduction (in) dim; leading dims map to the per-output-channel
    scale shape ([E, out, in] -> scale [E, out]).
    """
    import torch

    FP4_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    FP4_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])  # nearest-grid bounds
    FP4_MAX, TINY = 6.0, 1e-20

    def sim_fp4(W, group=32):    # e2m1 + per-32-group E8M0 (round-up) scale on the in dim
        *lead, out, inn = W.shape
        Wg = W.reshape(*lead, out, inn // group, group)
        scale = torch.exp2(torch.ceil(torch.log2((Wg.abs().amax(-1, keepdim=True) / FP4_MAX).clamp_min(TINY))))
        q = torch.sign(Wg) * FP4_MAG[torch.searchsorted(FP4_MID, (Wg / scale).abs()).clamp_max(7)]
        return (q * scale).reshape(*lead, out, inn)

    Wq = sim_fp4(torch.randn(*shape))
    amax = Wq.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale = amax / INT8_SCALE_MAX
    w_i8 = torch.round(Wq / scale).clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
    scale = (scale * (dequant_std / (w_i8.float() * scale).std())).squeeze(-1).float()
    return w_i8, scale


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    # Across-layer-mean dequant std (typical layer) of the real DeepSeek-V4-Flash MXFP4
    # routed experts; gen_routed_weight simulates the FP4 grid (see its docstring).
    ROUTED_DEQUANT_STD = {"w1": 2.47e-2, "w2": 2.44e-2, "w3": 2.46e-2}

    # Distribute B*S*TOPK token-expert pairs uniformly across local experts.
    total = B * S * M.num_experts_per_tok
    counts = torch.bincount(
        torch.randint(0, N_LOCAL_EXPERTS, (total,)),
        minlength=N_LOCAL_EXPERTS,
    ).to(torch.int32)
    counts_2d = counts.reshape(N_LOCAL_EXPERTS, 1)

    # Build a consistent INT8 recv_x + per-row dequant scale (dispatch is
    # responsible for per-token quantization). Invalid tail rows go to INT8 0
    # with scale 0 so dequant produces 0.
    x_bf16 = torch.randn(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.bfloat16)
    valid_mask_3d = (
        torch.arange(RECV_MAX).reshape(1, RECV_MAX, 1) < counts.reshape(N_LOCAL_EXPERTS, 1, 1)
    )
    recv_x_i8_pre, recv_scale_dq_pre = _int8_quant_per_row(x_bf16)
    recv_x_i8_pre = torch.where(valid_mask_3d, recv_x_i8_pre, torch.zeros_like(recv_x_i8_pre))
    valid_mask_2d = valid_mask_3d.squeeze(-1)
    recv_scale_dq_pre = torch.where(
        valid_mask_2d,
        recv_scale_dq_pre.squeeze(-1),
        torch.zeros_like(recv_scale_dq_pre.squeeze(-1)),
    )

    def init_recv_x():
        return recv_x_i8_pre

    def init_recv_scale_dq():
        return recv_scale_dq_pre.float()

    def init_recv_expert_count():
        return counts_2d

    # Per-row routing weight in [0, 1); tail rows (slot >= count) stay 0 so
    # they don't perturb the BF16 round-trip in expert_routed.
    recv_weights_pre = torch.rand(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.float32)
    recv_weights_pre = torch.where(
        valid_mask_2d, recv_weights_pre, torch.zeros_like(recv_weights_pre)
    )

    def init_recv_weights():
        return recv_weights_pre

    # Synthesize (int8, per-channel scale) by simulating the real MXFP4 routed-expert
    # quant grid (see gen_routed_weight). The kernel + golden both consume int8+scale.
    w1_i8, w1_s = gen_routed_weight((N_LOCAL_EXPERTS, MOE_INTER, D), ROUTED_DEQUANT_STD["w1"])
    w3_i8, w3_s = gen_routed_weight((N_LOCAL_EXPERTS, MOE_INTER, D), ROUTED_DEQUANT_STD["w3"])
    w2_i8, w2_s = gen_routed_weight((N_LOCAL_EXPERTS, D, MOE_INTER), ROUTED_DEQUANT_STD["w2"])

    return [
        TensorSpec("recv_x", [N_LOCAL_EXPERTS, RECV_MAX, D], torch.int8, init_value=init_recv_x),
        TensorSpec("recv_scale_dq", [N_LOCAL_EXPERTS, RECV_MAX], torch.float32, init_value=init_recv_scale_dq),
        TensorSpec("recv_weights", [N_LOCAL_EXPERTS, RECV_MAX], torch.float32, init_value=init_recv_weights),
        TensorSpec("recv_expert_count", [N_LOCAL_EXPERTS, 1], torch.int32, init_value=init_recv_expert_count),
        TensorSpec("routed_w1", [N_LOCAL_EXPERTS, MOE_INTER, D], torch.int8, init_value=lambda: w1_i8),
        TensorSpec("routed_w1_scale", [N_LOCAL_EXPERTS, MOE_INTER], torch.float32, init_value=lambda: w1_s),
        TensorSpec("routed_w3", [N_LOCAL_EXPERTS, MOE_INTER, D], torch.int8, init_value=lambda: w3_i8),
        TensorSpec("routed_w3_scale", [N_LOCAL_EXPERTS, MOE_INTER], torch.float32, init_value=lambda: w3_s),
        TensorSpec("routed_w2", [N_LOCAL_EXPERTS, D, MOE_INTER], torch.int8, init_value=lambda: w2_i8),
        TensorSpec("routed_w2_scale", [N_LOCAL_EXPERTS, D], torch.float32, init_value=lambda: w2_s),
        TensorSpec("recv_y", [N_LOCAL_EXPERTS, RECV_MAX, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_reldiff, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=expert_routed_test,
        specs=build_tensor_specs(),
        golden_fn=golden_expert_routed,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            # BF16 recv_y, ~1 ULP. Gen weights reproduce real(L21): 0.016% vs 0.015% of points > 1e-3.
            "recv_y": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
