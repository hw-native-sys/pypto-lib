# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE routed local expert — MXFP4 weights + dyn MX act.

AscendC Hybrid is W4A8 (MXFP4 weight × MXFP8 act). PTOAS 0.48/0.50
``tmatmul.mx`` only accepts same-dtype pairs (fp8/fp8 or fp4/fp4), so the
device path temporarily uses dynamic MXFP4 activations to match FP4 weights.
Switch back to ``mxfp8_e4m3`` act when PTOAS gains mixed-pair support.

  BF16 recv_x → dynamic MXFP4 → MX GEMM (FP4 w1/w3) → SwiGLU
  → dynamic MXFP4 → MX GEMM (FP4 w2) → × routing weight → BF16 recv_y

Weights are Right matrices for ``matmul_mx`` (``pl.FP4`` / ``float4_e2m1fn_x2``):
  w1/w3: ``[E, D, MOE_INTER]`` + scale ``[E, D/32, MOE_INTER]`` (MX_B_NN packed)
  w2:    ``[E, MOE_INTER, D]`` + scale ``[E, MOE_INTER/32, D]``
"""

import pypto.language as pl

from config import (
    FLASH as M,
    DECODE_BATCH,
    DECODE_SEQ,
    EP_WORLD_SIZE,
    RECV_MAX,
    MX_BLOCK_K,
)
from mx_quant_common import ATOL_RTOL, gen_mxfp4_weight_kn, routed_expert_mx_golden


B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit
N_LOCAL_EXPERTS = M.n_routed_experts // EP_WORLD_SIZE
D_SCALE = D // MX_BLOCK_K
INTER_SCALE = MOE_INTER // MX_BLOCK_K

RECV_TILE = 16
K_TILE = 128
MM_INTER_TILE = 128
ACT_INTER_TILE = 256
D_OUT_TILE = 128

assert RECV_MAX % RECV_TILE == 0
assert D % MX_BLOCK_K == 0 and MOE_INTER % MX_BLOCK_K == 0
assert SWIGLU_LIMIT > 0.0

_GATE_SPMD = MOE_INTER // MM_INTER_TILE
_GATE_K_CHUNKS = D // K_TILE
_DOWN_SPMD = D // D_OUT_TILE
_DOWN_K_CHUNKS = MOE_INTER // K_TILE
# Concurrent: all local experts × recv tiles × SPMD × K-chunks
_MX_PHASE_SLOTS = max(_GATE_SPMD * _GATE_K_CHUNKS, _DOWN_SPMD * _DOWN_K_CHUNKS)
_MX_WS_SLOTS = N_LOCAL_EXPERTS * (RECV_MAX // RECV_TILE) * _MX_PHASE_SLOTS


@pl.jit.inline
def expert_routed(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.FP4],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, D_SCALE, MOE_INTER], pl.FP8E8M0],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.FP4],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, D_SCALE, MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.FP4],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER_SCALE, D], pl.FP8E8M0],
    recv_y: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
):
    mx_scale_ws = pl.create_tensor(
        [_MX_WS_SLOTS * RECV_TILE, K_TILE // MX_BLOCK_K], dtype=pl.FP8E8M0
    )
    for local_e in pl.parallel(N_LOCAL_EXPERTS):
        e_rows = pl.read(recv_expert_count, [local_e, 0])
        e_tiles = (e_rows + RECV_TILE - 1) // RECV_TILE
        # 2D GM views for MX scale loads (mx_layout requires rank-2 tensors).
        w1s_e = pl.reshape(
            pl.slice(routed_w1_scale, [1, D_SCALE, MOE_INTER], [local_e, 0, 0]),
            [D_SCALE, MOE_INTER],
        )
        w3s_e = pl.reshape(
            pl.slice(routed_w3_scale, [1, D_SCALE, MOE_INTER], [local_e, 0, 0]),
            [D_SCALE, MOE_INTER],
        )
        w2s_e = pl.reshape(
            pl.slice(routed_w2_scale, [1, INTER_SCALE, D], [local_e, 0, 0]),
            [INTER_SCALE, D],
        )

        for tt in pl.parallel(e_tiles):
            tt0 = tt * RECV_TILE
            # Static slot base (worst-case RECV_MAX tiles) for concurrent experts × tiles.
            tile_base = (local_e * (RECV_MAX // RECV_TILE) + tt) * _MX_PHASE_SLOTS

            gate_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32)
            up_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32)

            for nb_idx in pl.spmd(MOE_INTER // MM_INTER_TILE, name_hint="exp_gate_mm"):
                n0 = nb_idx * MM_INTER_TILE
                gate_acc = pl.create_tile(
                    [RECV_TILE, MM_INTER_TILE], dtype=pl.FP32, target_memory=pl.Mem.Acc
                )
                for k0 in pl.range(0, D, K_TILE):
                    x_tile = pl.load(
                        recv_x,
                        [local_e, tt0, k0],
                        [1, RECV_TILE, K_TILE],
                        target_memory=pl.Mem.Vec,
                    )
                    x_2d = pl.reshape(x_tile, [RECV_TILE, K_TILE])
                    # PTOAS: MXFP4 tquant needs bf16/f16 src; mx matmul needs fp4×fp4.
                    x_q, x_s = pl.mx_quant(x_2d, mode="mxfp4")
                    w_tile = pl.load(
                        routed_w1,
                        [local_e, k0, n0],
                        [1, K_TILE, MM_INTER_TILE],
                        target_memory=pl.Mem.Mat,
                    )
                    w_2d = pl.reshape(w_tile, [K_TILE, MM_INTER_TILE])
                    ws_tile = pl.load(
                        w1s_e,
                        [k0 // MX_BLOCK_K, n0],
                        [K_TILE // MX_BLOCK_K, MM_INTER_TILE],
                        target_memory=pl.Mem.Mat,
                        mx_layout="mx_b_nn",
                    )
                    srow = (
                        tile_base + nb_idx * _GATE_K_CHUNKS + k0 // K_TILE
                    ) * RECV_TILE
                    # mx_quant(mode="mxfp4") already returns packed FP4 (!pto.f4E2M1x2).
                    la = pl.move(
                        pl.move(x_q, target_memory=pl.Mem.Mat),
                        target_memory=pl.Mem.Left,
                    )
                    la = pl.set_validshape(la, RECV_TILE, K_TILE)
                    pl.store(pl.tile.reinterpret_view(x_s, pl.FP8E8M0), [srow, 0], mx_scale_ws)
                    las = pl.move(
                        pl.load(
                            mx_scale_ws,
                            [srow, 0],
                            [RECV_TILE, K_TILE // MX_BLOCK_K],
                            target_memory=pl.Mem.Mat,
                            mx_layout="mx_a_zz",
                        ),
                        target_memory=pl.Mem.LeftScale,
                    )
                    las = pl.tget_scale_addr(las, la)
                    las = pl.set_validshape(las, RECV_TILE, K_TILE // MX_BLOCK_K)
                    rb = pl.move(w_2d, target_memory=pl.Mem.Right)
                    rbs = pl.move(ws_tile, target_memory=pl.Mem.RightScale)
                    rbs = pl.tget_scale_addr(rbs, rb)
                    gate_acc = pl.matmul_mx_acc(gate_acc, la, las, rb, rbs)
                pl.store(gate_acc, [0, n0], gate_fp32)

            for nb_idx in pl.spmd(MOE_INTER // MM_INTER_TILE, name_hint="exp_up_mm"):
                n0 = nb_idx * MM_INTER_TILE
                up_acc = pl.create_tile(
                    [RECV_TILE, MM_INTER_TILE], dtype=pl.FP32, target_memory=pl.Mem.Acc
                )
                for k0 in pl.range(0, D, K_TILE):
                    x_tile = pl.load(
                        recv_x,
                        [local_e, tt0, k0],
                        [1, RECV_TILE, K_TILE],
                        target_memory=pl.Mem.Vec,
                    )
                    x_2d = pl.reshape(x_tile, [RECV_TILE, K_TILE])
                    x_q, x_s = pl.mx_quant(x_2d, mode="mxfp4")
                    w_tile = pl.load(
                        routed_w3,
                        [local_e, k0, n0],
                        [1, K_TILE, MM_INTER_TILE],
                        target_memory=pl.Mem.Mat,
                    )
                    w_2d = pl.reshape(w_tile, [K_TILE, MM_INTER_TILE])
                    ws_tile = pl.load(
                        w3s_e,
                        [k0 // MX_BLOCK_K, n0],
                        [K_TILE // MX_BLOCK_K, MM_INTER_TILE],
                        target_memory=pl.Mem.Mat,
                        mx_layout="mx_b_nn",
                    )
                    srow = (
                        tile_base + nb_idx * _GATE_K_CHUNKS + k0 // K_TILE
                    ) * RECV_TILE
                    # mx_quant(mode="mxfp4") already returns packed FP4 (!pto.f4E2M1x2).
                    la = pl.move(
                        pl.move(x_q, target_memory=pl.Mem.Mat),
                        target_memory=pl.Mem.Left,
                    )
                    la = pl.set_validshape(la, RECV_TILE, K_TILE)
                    pl.store(pl.tile.reinterpret_view(x_s, pl.FP8E8M0), [srow, 0], mx_scale_ws)
                    las = pl.move(
                        pl.load(
                            mx_scale_ws,
                            [srow, 0],
                            [RECV_TILE, K_TILE // MX_BLOCK_K],
                            target_memory=pl.Mem.Mat,
                            mx_layout="mx_a_zz",
                        ),
                        target_memory=pl.Mem.LeftScale,
                    )
                    las = pl.tget_scale_addr(las, la)
                    las = pl.set_validshape(las, RECV_TILE, K_TILE // MX_BLOCK_K)
                    rb = pl.move(w_2d, target_memory=pl.Mem.Right)
                    rbs = pl.move(ws_tile, target_memory=pl.Mem.RightScale)
                    rbs = pl.tget_scale_addr(rbs, rb)
                    up_acc = pl.matmul_mx_acc(up_acc, la, las, rb, rbs)
                pl.store(up_acc, [0, n0], up_fp32)

            h_tile_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32)
            for part in pl.spmd(MOE_INTER // ACT_INTER_TILE, name_hint="exp_swiglu"):
                n0 = part * ACT_INTER_TILE
                gate_rows = gate_fp32[:, n0 : n0 + ACT_INTER_TILE]
                up_rows = up_fp32[:, n0 : n0 + ACT_INTER_TILE]
                gate_clamped = pl.minimum(gate_rows, SWIGLU_LIMIT)
                up_clamped = pl.maximum(pl.minimum(up_rows, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_clamped)), 1.0))
                gated = pl.mul(pl.mul(gate_clamped, sigmoid), up_clamped)
                h_tile_fp32[:, n0 : n0 + ACT_INTER_TILE] = gated

            for db_idx in pl.spmd(D // D_OUT_TILE, name_hint="exp_w2_mm"):
                d0 = db_idx * D_OUT_TILE
                y_acc = pl.create_tile(
                    [RECV_TILE, D_OUT_TILE], dtype=pl.FP32, target_memory=pl.Mem.Acc
                )
                for k0 in pl.range(0, MOE_INTER, K_TILE):
                    h_tile = pl.load(
                        h_tile_fp32, [0, k0], [RECV_TILE, K_TILE], target_memory=pl.Mem.Vec
                    )
                    h_bf16 = pl.cast(h_tile, target_type=pl.BF16, mode="rint")
                    h_q, h_s = pl.mx_quant(h_bf16, mode="mxfp4")
                    w_tile = pl.load(
                        routed_w2,
                        [local_e, k0, d0],
                        [1, K_TILE, D_OUT_TILE],
                        target_memory=pl.Mem.Mat,
                    )
                    w_2d = pl.reshape(w_tile, [K_TILE, D_OUT_TILE])
                    ws_tile = pl.load(
                        w2s_e,
                        [k0 // MX_BLOCK_K, d0],
                        [K_TILE // MX_BLOCK_K, D_OUT_TILE],
                        target_memory=pl.Mem.Mat,
                        mx_layout="mx_b_nn",
                    )
                    srow = (
                        tile_base + db_idx * _DOWN_K_CHUNKS + k0 // K_TILE
                    ) * RECV_TILE
                    # mx_quant(mode="mxfp4") already returns packed FP4 (!pto.f4E2M1x2).
                    la = pl.move(
                        pl.move(h_q, target_memory=pl.Mem.Mat),
                        target_memory=pl.Mem.Left,
                    )
                    la = pl.set_validshape(la, RECV_TILE, K_TILE)
                    pl.store(pl.tile.reinterpret_view(h_s, pl.FP8E8M0), [srow, 0], mx_scale_ws)
                    las = pl.move(
                        pl.load(
                            mx_scale_ws,
                            [srow, 0],
                            [RECV_TILE, K_TILE // MX_BLOCK_K],
                            target_memory=pl.Mem.Mat,
                            mx_layout="mx_a_zz",
                        ),
                        target_memory=pl.Mem.LeftScale,
                    )
                    las = pl.tget_scale_addr(las, la)
                    las = pl.set_validshape(las, RECV_TILE, K_TILE // MX_BLOCK_K)
                    rb = pl.move(w_2d, target_memory=pl.Mem.Right)
                    rbs = pl.move(ws_tile, target_memory=pl.Mem.RightScale)
                    rbs = pl.tget_scale_addr(rbs, rb)
                    y_acc = pl.matmul_mx_acc(y_acc, la, las, rb, rbs)
                # Apply routing weights then store BF16.
                w_col = pl.load(
                    recv_weights,
                    [local_e, tt0],
                    [1, RECV_TILE],
                    target_memory=pl.Mem.Vec,
                )
                w_col = pl.reshape(w_col, [RECV_TILE, 1])
                # trowexpandmul requires row-major dst/src (Acc is col_major).
                y_fp32 = pl.move(
                    y_acc, target_memory=pl.Mem.Vec, blayout=pl.TileLayout.row_major
                )
                y_scaled = pl.row_expand_mul(y_fp32, w_col)
                y_bf16 = pl.cast(y_scaled, target_type=pl.BF16, mode="rint")
                # tstore needs ND (row_major + none_box); Acc→Vec path leaves boxed slayout.
                y_bf16 = pl.move(
                    y_bf16,
                    target_memory=pl.Mem.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                pl.store(y_bf16, [local_e, tt0, d0], recv_y)

    return recv_y


@pl.jit
def expert_routed_test(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.FP4],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, D_SCALE, MOE_INTER], pl.FP8E8M0],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.FP4],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, D_SCALE, MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.FP4],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER_SCALE, D], pl.FP8E8M0],
    recv_y: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16]],
):
    expert_routed(
        recv_x, recv_weights, recv_expert_count,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        recv_y,
    )
    return recv_y


def gen_routed_weight(shape, dequant_std):
    """Synthesize routed-expert MXFP4 weight + E8M0 scale (AscendC block=32).

    ``shape`` is historical ``[E, out, in]``. Returns Right-matrix storage
    ``(w_kn [E, in, out] float4_e2m1fn_x2, scale [E, in/32, out] float8_e8m0fnu)``.
    """
    import torch

    *lead, out, inn = shape
    n_lead = 1
    for dim in lead:
        n_lead *= dim
    ws = []
    ss = []
    for _ in range(n_lead):
        w, s = gen_mxfp4_weight_kn((inn, out), dequant_std=dequant_std, pack_nn=True)
        ws.append(w)
        ss.append(s)
    w_out = torch.stack(ws, dim=0).reshape(*lead, inn, out)
    s_out = torch.stack(ss, dim=0).reshape(*lead, inn // MX_BLOCK_K, out)
    return w_out, s_out


def golden_expert_routed(tensors):
    """Torch reference for routed expert (+ routing weights).

    TODO(L1): currently still AscendC W4A8 (MXFP8 act × MXFP4 weight). Device is
    temporary W4A4 (MXFP4×MXFP4) until PTOAS supports mixed matmul — align golden
    to device before numerical verification (see MXFP8_MXFP4_REWRITE.md §3 L1).
    """
    import torch

    y = routed_expert_mx_golden(
        recv_x_bf16=tensors["recv_x"],
        recv_weights=tensors["recv_weights"],
        recv_expert_count=tensors["recv_expert_count"],
        w1_x2=tensors["routed_w1"],
        w1_scale=tensors["routed_w1_scale"],
        w3_x2=tensors["routed_w3"],
        w3_scale=tensors["routed_w3_scale"],
        w2_x2=tensors["routed_w2"],
        w2_scale=tensors["routed_w2_scale"],
        swiglu_limit=SWIGLU_LIMIT,
        scales_packed_nn=True,
    )
    tensors["recv_y"][:] = y.to(torch.bfloat16)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    ROUTED_DEQUANT_STD = {"w1": 2.47e-2, "w2": 2.44e-2, "w3": 2.46e-2}

    total = B * S * M.num_experts_per_tok
    counts = torch.bincount(
        torch.randint(0, N_LOCAL_EXPERTS, (total,)),
        minlength=N_LOCAL_EXPERTS,
    ).clamp(max=RECV_MAX)
    recv_expert_count = counts.to(torch.int32).reshape(N_LOCAL_EXPERTS, 1)

    recv_x = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.bfloat16)
    recv_weights = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.float32)
    for e in range(N_LOCAL_EXPERTS):
        n = int(counts[e].item())
        if n == 0:
            continue
        recv_x[e, :n] = torch.randn(n, D, dtype=torch.bfloat16)
        recv_weights[e, :n] = torch.rand(n) * 0.5 + 0.5

    w1, w1_s = gen_routed_weight((N_LOCAL_EXPERTS, MOE_INTER, D), ROUTED_DEQUANT_STD["w1"])
    w3, w3_s = gen_routed_weight((N_LOCAL_EXPERTS, MOE_INTER, D), ROUTED_DEQUANT_STD["w3"])
    w2, w2_s = gen_routed_weight((N_LOCAL_EXPERTS, D, MOE_INTER), ROUTED_DEQUANT_STD["w2"])

    return [
        TensorSpec("recv_x", [N_LOCAL_EXPERTS, RECV_MAX, D], torch.bfloat16, init_value=lambda: recv_x),
        TensorSpec(
            "recv_weights", [N_LOCAL_EXPERTS, RECV_MAX], torch.float32, init_value=lambda: recv_weights
        ),
        TensorSpec(
            "recv_expert_count",
            [N_LOCAL_EXPERTS, 1],
            torch.int32,
            init_value=lambda: recv_expert_count,
        ),
        TensorSpec(
            "routed_w1",
            [N_LOCAL_EXPERTS, D, MOE_INTER],
            torch.float4_e2m1fn_x2,
            init_value=lambda: w1,
        ),
        TensorSpec(
            "routed_w1_scale",
            [N_LOCAL_EXPERTS, D_SCALE, MOE_INTER],
            torch.float8_e8m0fnu,
            init_value=lambda: w1_s,
        ),
        TensorSpec(
            "routed_w3",
            [N_LOCAL_EXPERTS, D, MOE_INTER],
            torch.float4_e2m1fn_x2,
            init_value=lambda: w3,
        ),
        TensorSpec(
            "routed_w3_scale",
            [N_LOCAL_EXPERTS, D_SCALE, MOE_INTER],
            torch.float8_e8m0fnu,
            init_value=lambda: w3_s,
        ),
        TensorSpec(
            "routed_w2",
            [N_LOCAL_EXPERTS, MOE_INTER, D],
            torch.float4_e2m1fn_x2,
            init_value=lambda: w2,
        ),
        TensorSpec(
            "routed_w2_scale",
            [N_LOCAL_EXPERTS, INTER_SCALE, D],
            torch.float8_e8m0fnu,
            init_value=lambda: w2_s,
        ),
        TensorSpec("recv_y", [N_LOCAL_EXPERTS, RECV_MAX, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_reldiff, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a5",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    moe_tol = ATOL_RTOL["moe_mx"]
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
        rtol=moe_tol["rtol"],
        atol=moe_tol["atol"],
        compare_fn={
            "recv_y": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
