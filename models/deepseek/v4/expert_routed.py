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
RECV_TILE = 32
K_TILE = 512
INTER_K = 512
INTER_TILE = 64
D_OUT_TILE = 64
QUANT_TILE = 256


@pl.jit.inline
def expert_routed(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
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

            # Stage 1a: gate/up matmul + dequant + SwiGLU + routing-weight mul.
            h_tile_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32)

            for nb_idx in pl.spmd(MOE_INTER // (4 * INTER_TILE), name_hint="exp_gate_up"):
                n_base = nb_idx * (4 * INTER_TILE)
                for ng in pl.range(4):
                    n0 = n_base + ng * INTER_TILE
                    # Peel-first-iter K loop: 3D-rhs b_trans=True matmul under
                    # pl.pipeline + create_tensor + if-kb==0 carry triggers a
                    # TLOAD DN->NZ ISA assertion (pypto#1540).
                    x_init = recv_x[local_i : local_i + 1, t0 : t0 + RECV_TILE, 0 : K_TILE]
                    w1_init = routed_w1[local_i : local_i + 1, n0 : n0 + INTER_TILE, 0 : K_TILE]
                    w3_init = routed_w3[local_i : local_i + 1, n0 : n0 + INTER_TILE, 0 : K_TILE]
                    gate_acc = pl.matmul(x_init, w1_init, b_trans=True, out_dtype=pl.INT32)
                    up_acc = pl.matmul(x_init, w3_init, b_trans=True, out_dtype=pl.INT32)
                    for k0 in pl.range(K_TILE, D, K_TILE):
                        x_k = recv_x[local_i : local_i + 1, t0 : t0 + RECV_TILE, k0 : k0 + K_TILE]
                        w1_k = routed_w1[local_i : local_i + 1, n0 : n0 + INTER_TILE, k0 : k0 + K_TILE]
                        w3_k = routed_w3[local_i : local_i + 1, n0 : n0 + INTER_TILE, k0 : k0 + K_TILE]
                        gate_acc = pl.matmul_acc(gate_acc, x_k, w1_k, b_trans=True)
                        up_acc = pl.matmul_acc(up_acc, x_k, w3_k, b_trans=True)

                    gate_2d_i32 = pl.reshape(gate_acc, [RECV_TILE, INTER_TILE])
                    up_2d_i32 = pl.reshape(up_acc, [RECV_TILE, INTER_TILE])
                    recv_x_scale_dq = pl.reshape(recv_scale_dq[local_i : local_i + 1, t0 : t0 + RECV_TILE], [RECV_TILE, 1])
                    w1_scale_chunk = routed_w1_scale[local_i : local_i + 1, n0 : n0 + INTER_TILE]
                    w3_scale_chunk = routed_w3_scale[local_i : local_i + 1, n0 : n0 + INTER_TILE]
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
                    gated_valid = pl.tensor.set_validshape(gated, valid_rows, INTER_TILE)
                    gated_masked = pl.fillpad(gated_valid, pad_value=pl.PadValue.zero)
                    h_tile_fp32[:, n0 : n0 + INTER_TILE] = gated_masked

            # Per-row A8 requant of h_tile (amax across full MOE_INTER row).
            h_tile_i8 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT8)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_h_q"):
                eh_amax = pl.full([1, RECV_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                for k0 in pl.range(0, MOE_INTER, QUANT_TILE):
                    eh_a_f32 = h_tile_fp32[:, k0 : k0 + QUANT_TILE]
                    eh_a_abs = pl.maximum(eh_a_f32, pl.neg(eh_a_f32))
                    eh_a_max = pl.reshape(pl.row_max(eh_a_abs), [1, RECV_TILE])
                    eh_amax = pl.maximum(eh_amax, eh_a_max)
                eh_sq_row = pl.div(
                    pl.full([1, RECV_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), eh_amax
                )
                h_tile_scale_dq = pl.reshape(pl.recip(eh_sq_row), [RECV_TILE, 1])
                eh_sq_col = pl.reshape(eh_sq_row, [RECV_TILE, 1])
                for k1 in pl.range(0, MOE_INTER, QUANT_TILE):
                    eh_q_f32 = h_tile_fp32[:, k1 : k1 + QUANT_TILE]
                    eh_q_scaled = pl.row_expand_mul(eh_q_f32, eh_sq_col)
                    eh_q_i32 = pl.cast(eh_q_scaled, target_type=pl.INT32, mode="rint")
                    eh_q_half = pl.cast(eh_q_i32, target_type=pl.FP16, mode="round")
                    h_tile_i8[:, k1 : k1 + QUANT_TILE] = pl.cast(eh_q_half, target_type=pl.INT8, mode="trunc")

            # Stage 1b: w2 matmul + dequant + write recv_y.
            for db_idx in pl.spmd(D // (16 * D_OUT_TILE), name_hint="exp_w2"):
                d_base = db_idx * (16 * D_OUT_TILE)
                for dg in pl.range(16):
                    d0 = d_base + dg * D_OUT_TILE
                    # Peel-first-iter K loop: see pypto#1540 note in exp_gate_up.
                    h_init = h_tile_i8[:, 0 : INTER_K]
                    w2_init = routed_w2[local_i : local_i + 1, d0 : d0 + D_OUT_TILE, 0 : INTER_K]
                    y_acc = pl.matmul(h_init, w2_init, b_trans=True, out_dtype=pl.INT32)
                    for k0 in pl.range(INTER_K, MOE_INTER, INTER_K):
                        h_k = h_tile_i8[:, k0 : k0 + INTER_K]
                        w2_k = routed_w2[local_i : local_i + 1, d0 : d0 + D_OUT_TILE, k0 : k0 + INTER_K]
                        y_acc = pl.matmul_acc(y_acc, h_k, w2_k, b_trans=True)

                    y_2d_i32 = pl.reshape(y_acc, [RECV_TILE, D_OUT_TILE])
                    w2_scale_chunk = routed_w2_scale[local_i : local_i + 1, d0 : d0 + D_OUT_TILE]
                    y_2d = pl.cast(y_2d_i32, target_type=pl.FP32, mode="none")
                    y_2d = pl.col_expand_mul(pl.row_expand_mul(y_2d, h_tile_scale_dq), w2_scale_chunk)
                    recv_y_flat[flat_t0 : flat_t0 + RECV_TILE, d0 : d0 + D_OUT_TILE] = pl.cast(
                        y_2d, target_type=pl.BF16, mode="rint"
                    )


@pl.jit
def expert_routed_test(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
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
        recv_x, recv_scale_dq, recv_expert_count,
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


def _quant_w_per_channel(w):
    """Per-output-channel INT8 quant on the last axis. Returns (i8_tensor, dequant_scale).

    For w shaped [..., N, K] (b_trans=True layout), the per-channel scale has shape [..., N].
    """
    import torch
    amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.unsqueeze(-1)
    w_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


def golden_expert_routed(tensors):
    """Torch reference for the routed expert. recv_y is the partial routed
    contribution only (without routing-weight scaling — that is applied in
    combine); AllToAllv combine and the shared-expert add happen in the
    host orchestrator.

    Per-expert layout: recv_x[e, 0:cnt[e], :] is the valid INT8 receive
    payload; recv_y[e, cnt[e]:, :] stays at zero."""
    import torch
    import torch.nn.functional as F

    def dequant_w(w_i8, w_scale):
        return w_i8.to(torch.float32) * w_scale.unsqueeze(-1)

    recv_x_i8 = tensors["recv_x"]  # INT8, pre-quantized in dispatch
    recv_scale_dq = tensors["recv_scale_dq"].float()  # [E, RECV_MAX]
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

        gate = x_sub_q @ w1[e].T
        up = x_sub_q @ w3[e].T
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h = F.silu(gate) * up
        # A8 requant before w2 matmul.
        h_i8, h_sd = _int8_quant_per_row(h)
        h = h_i8.float() * h_sd
        recv_y[e, :n_rows, :] = h @ w2[e].T

    tensors["recv_y"][:] = recv_y.to(torch.bfloat16)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

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

    # Pre-quantize all three weights once so the i8 / scale specs see consistent values.
    w1_bf16 = (torch.randn(N_LOCAL_EXPERTS, MOE_INTER, D) / D ** 0.5).to(torch.bfloat16)
    w3_bf16 = (torch.randn(N_LOCAL_EXPERTS, MOE_INTER, D) / D ** 0.5).to(torch.bfloat16)
    w2_bf16 = (torch.randn(N_LOCAL_EXPERTS, D, MOE_INTER) / MOE_INTER ** 0.5).to(torch.bfloat16)

    w1_i8, w1_s = _quant_w_per_channel(w1_bf16)
    w3_i8, w3_s = _quant_w_per_channel(w3_bf16)
    w2_i8, w2_s = _quant_w_per_channel(w2_bf16)

    return [
        TensorSpec("recv_x", [N_LOCAL_EXPERTS, RECV_MAX, D], torch.int8, init_value=init_recv_x),
        TensorSpec("recv_scale_dq", [N_LOCAL_EXPERTS, RECV_MAX], torch.float32, init_value=init_recv_scale_dq),
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
    args = parser.parse_args()

    result = run_jit(
        fn=expert_routed_test,
        specs=build_tensor_specs(),
        golden_fn=golden_expert_routed,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "recv_y": ratio_reldiff(diff_thd=0.01, pct_thd=0.05),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
