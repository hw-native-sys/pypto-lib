# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE shared expert compute (decode, EP single-card) — Hybrid MXFP8.

Aligned with AscendC ``MxFp8MoEGMMMethod`` / ``MxFp8LinearMethod`` for the shared
expert FFN:

  BF16 x → dynamic MXFP8 (e4m3 + e8m0, block=32) → MX GEMM (gate/up)
  → SwiGLU → dynamic MXFP8 → MX GEMM (down) → BF16

Weights are stored as Right matrices for ``matmul_mx``:
  w1/w3: ``[D, MOE_INTER]`` FP8E4M3FN + scale ``[D/32, MOE_INTER]`` FP8E8M0 (MX_B_NN packed)
  w2:    ``[MOE_INTER, D]`` FP8E4M3FN + scale ``[MOE_INTER/32, D]`` FP8E8M0 (MX_B_NN packed)

Note: ``moe.py`` still expects the legacy INT8 API until Step 3 of the rewrite plan.
"""

import pypto.language as pl

from config import FLASH as M, MOE_TOKENS, MX_BLOCK_K
from mx_quant_common import (
    ATOL_RTOL,
    gen_mxfp8_weight_kn,
    shared_expert_mxfp8_golden,
)


# model config
T = MOE_TOKENS
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit
D_SCALE = D // MX_BLOCK_K
INTER_SCALE = MOE_INTER // MX_BLOCK_K

# tiling (K tiles must be divisible by MX_BLOCK_K=32)
SH_M_TILE = 16
SH_ROW_PAD = 8
SH_ROWS_PER_BLOCK = 2
T_PAD = ((T + SH_M_TILE - 1) // SH_M_TILE) * SH_M_TILE
assert T <= SH_M_TILE or T % SH_M_TILE == 0, \
    "expert_shared needs T <= SH_M_TILE (decode) or T a multiple of SH_M_TILE (prefill)"
SH_VALID_M = T if T < SH_M_TILE else SH_M_TILE
N_MTILES = T_PAD // SH_M_TILE
assert SH_VALID_M % SH_ROWS_PER_BLOCK == 0
assert D % MX_BLOCK_K == 0 and MOE_INTER % MX_BLOCK_K == 0

K_TILE = 128          # along D / MOE_INTER reduction; % 32 == 0
MM_INTER_TILE = 128   # along MOE_INTER (N) for gate/up
ACT_INTER_TILE = 256  # AIV SwiGLU chunk
D_OUT_TILE = 128      # along D for down proj


@pl.jit.inline
def expert_shared(
    x_local: pl.Tensor[[T, D], pl.BF16],
    shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[D_SCALE, MOE_INTER], pl.FP8E8M0],
    shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[D_SCALE, MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[INTER_SCALE, D], pl.FP8E8M0],
    sh: pl.Tensor[[T, D], pl.BF16],
):
    for mt in pl.parallel(N_MTILES):
        ts0 = mt * SH_M_TILE

        gate_fp32 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.FP32)
        up_fp32 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.FP32)

        # gate (w1): dyn MX quant(x) @ w1  → FP32
        for nb_idx in pl.spmd(MOE_INTER // MM_INTER_TILE, name_hint="sh_gate_mm"):
            n0 = nb_idx * MM_INTER_TILE
            gate_acc = pl.create_tile(
                [SH_M_TILE, MM_INTER_TILE], dtype=pl.FP32, target_memory=pl.Mem.Acc
            )
            for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                x_tile = pl.load(
                    x_local,
                    [ts0, k0],
                    [SH_M_TILE, K_TILE],
                    valid_shapes=[SH_VALID_M, K_TILE],
                    target_memory=pl.Mem.Vec,
                )
                x_f = pl.cast(x_tile, target_type=pl.FP32, mode="none")
                x_q, x_s = pl.mx_quant(x_f, mode="mxfp8_e4m3")
                w_tile = pl.load(
                    shared_w1, [k0, n0], [K_TILE, MM_INTER_TILE], target_memory=pl.Mem.Mat
                )
                ws_tile = pl.load(
                    shared_w1_scale,
                    [k0 // MX_BLOCK_K, n0],
                    [K_TILE // MX_BLOCK_K, MM_INTER_TILE],
                    target_memory=pl.Mem.Mat,
                    mx_layout="mx_b_nn",
                )
                la = pl.move(
                    pl.move(x_q, target_memory=pl.Mem.Mat), target_memory=pl.Mem.Left
                )
                las = pl.move(
                    pl.move(x_s, target_memory=pl.Mem.Mat), target_memory=pl.Mem.LeftScale
                )
                rb = pl.move(w_tile, target_memory=pl.Mem.Right)
                rbs = pl.move(ws_tile, target_memory=pl.Mem.RightScale)
                las = pl.tget_scale_addr(las, la)
                rbs = pl.tget_scale_addr(rbs, rb)
                gate_acc = pl.matmul_mx_acc(gate_acc, la, las, rb, rbs)
            pl.store(gate_acc, [0, n0], gate_fp32)

        # up (w3)
        for nb_idx in pl.spmd(MOE_INTER // MM_INTER_TILE, name_hint="sh_up_mm"):
            n0 = nb_idx * MM_INTER_TILE
            up_acc = pl.create_tile(
                [SH_M_TILE, MM_INTER_TILE], dtype=pl.FP32, target_memory=pl.Mem.Acc
            )
            for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                x_tile = pl.load(
                    x_local,
                    [ts0, k0],
                    [SH_M_TILE, K_TILE],
                    valid_shapes=[SH_VALID_M, K_TILE],
                    target_memory=pl.Mem.Vec,
                )
                x_f = pl.cast(x_tile, target_type=pl.FP32, mode="none")
                x_q, x_s = pl.mx_quant(x_f, mode="mxfp8_e4m3")
                w_tile = pl.load(
                    shared_w3, [k0, n0], [K_TILE, MM_INTER_TILE], target_memory=pl.Mem.Mat
                )
                ws_tile = pl.load(
                    shared_w3_scale,
                    [k0 // MX_BLOCK_K, n0],
                    [K_TILE // MX_BLOCK_K, MM_INTER_TILE],
                    target_memory=pl.Mem.Mat,
                    mx_layout="mx_b_nn",
                )
                la = pl.move(
                    pl.move(x_q, target_memory=pl.Mem.Mat), target_memory=pl.Mem.Left
                )
                las = pl.move(
                    pl.move(x_s, target_memory=pl.Mem.Mat), target_memory=pl.Mem.LeftScale
                )
                rb = pl.move(w_tile, target_memory=pl.Mem.Right)
                rbs = pl.move(ws_tile, target_memory=pl.Mem.RightScale)
                las = pl.tget_scale_addr(las, la)
                rbs = pl.tget_scale_addr(rbs, rb)
                up_acc = pl.matmul_mx_acc(up_acc, la, las, rb, rbs)
            pl.store(up_acc, [0, n0], up_fp32)

        # SwiGLU → h_fp32 (full intermediate)
        h_tile_fp32 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.FP32)
        for row_block in pl.spmd(
            SH_VALID_M // SH_ROWS_PER_BLOCK, name_hint="sh_swiglu"
        ):
            row0 = row_block * SH_ROWS_PER_BLOCK
            for part in pl.pipeline(0, MOE_INTER // ACT_INTER_TILE, stage=1):
                n0 = part * ACT_INTER_TILE
                gate_rows = pl.slice(
                    gate_fp32,
                    [SH_ROW_PAD, ACT_INTER_TILE],
                    [row0, n0],
                    valid_shape=[SH_ROWS_PER_BLOCK, ACT_INTER_TILE],
                )
                up_rows = pl.slice(
                    up_fp32,
                    [SH_ROW_PAD, ACT_INTER_TILE],
                    [row0, n0],
                    valid_shape=[SH_ROWS_PER_BLOCK, ACT_INTER_TILE],
                )
                gate_clamped = pl.minimum(gate_rows, SWIGLU_LIMIT)
                up_clamped = pl.maximum(
                    pl.minimum(up_rows, SWIGLU_LIMIT), -SWIGLU_LIMIT
                )
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_clamped)), 1.0))
                gated = pl.mul(pl.mul(gate_clamped, sigmoid), up_clamped)
                h_tile_fp32[
                    row0 : row0 + SH_ROWS_PER_BLOCK,
                    n0 : n0 + ACT_INTER_TILE,
                ] = gated[0:SH_ROWS_PER_BLOCK, :]

        # down (w2): dyn MX quant(h) @ w2 → BF16
        for db_idx in pl.spmd(D // D_OUT_TILE, name_hint="sh_w2_mm"):
            d0 = db_idx * D_OUT_TILE
            y_acc = pl.create_tile(
                [SH_M_TILE, D_OUT_TILE], dtype=pl.FP32, target_memory=pl.Mem.Acc
            )
            for k0 in pl.pipeline(0, MOE_INTER, K_TILE, stage=2):
                h_tile = pl.load(
                    h_tile_fp32, [0, k0], [SH_M_TILE, K_TILE], target_memory=pl.Mem.Vec
                )
                h_q, h_s = pl.mx_quant(h_tile, mode="mxfp8_e4m3")
                w_tile = pl.load(
                    shared_w2, [k0, d0], [K_TILE, D_OUT_TILE], target_memory=pl.Mem.Mat
                )
                ws_tile = pl.load(
                    shared_w2_scale,
                    [k0 // MX_BLOCK_K, d0],
                    [K_TILE // MX_BLOCK_K, D_OUT_TILE],
                    target_memory=pl.Mem.Mat,
                    mx_layout="mx_b_nn",
                )
                la = pl.move(
                    pl.move(h_q, target_memory=pl.Mem.Mat), target_memory=pl.Mem.Left
                )
                las = pl.move(
                    pl.move(h_s, target_memory=pl.Mem.Mat), target_memory=pl.Mem.LeftScale
                )
                rb = pl.move(w_tile, target_memory=pl.Mem.Right)
                rbs = pl.move(ws_tile, target_memory=pl.Mem.RightScale)
                las = pl.tget_scale_addr(las, la)
                rbs = pl.tget_scale_addr(rbs, rb)
                y_acc = pl.matmul_mx_acc(y_acc, la, las, rb, rbs)
            y_bf16 = pl.cast(y_acc, target_type=pl.BF16, mode="rint")
            pl.store(y_bf16, [ts0, d0], sh)

    return sh


@pl.jit
def expert_shared_test(
    x_local: pl.Tensor[[T, D], pl.BF16],
    shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[D_SCALE, MOE_INTER], pl.FP8E8M0],
    shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[D_SCALE, MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[INTER_SCALE, D], pl.FP8E8M0],
    sh: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    expert_shared(
        x_local,
        shared_w1, shared_w1_scale,
        shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        sh,
    )
    return sh


def gen_shared_weight(shape, dequant_std, chan_cv):
    """Synthesize shared-expert MXFP8 weight + E8M0 scale (AscendC block=32).

    ``shape`` is the historical ``[out, in]`` layout. Returns Right-matrix storage
    ``(w_kn [in, out] float8_e4m3fn, scale_kn [in/32, out] float8_e8m0fnu)`` with
    MX_B_NN packing applied to the scale buffer for device TLoad.
    """
    out, inn = shape
    return gen_mxfp8_weight_kn(
        (inn, out), dequant_std=dequant_std, chan_cv=chan_cv, pack_nn=True
    )


def golden_expert_shared(tensors):
    """Torch reference matching AscendC shared MXFP8 FFN (dyn MX → GEMM → SwiGLU → …)."""
    import torch

    sh = shared_expert_mxfp8_golden(
        x_bf16=tensors["x_local"],
        w1_fp8=tensors["shared_w1"],
        w1_scale=tensors["shared_w1_scale"],
        w3_fp8=tensors["shared_w3"],
        w3_scale=tensors["shared_w3_scale"],
        w2_fp8=tensors["shared_w2"],
        w2_scale=tensors["shared_w2_scale"],
        swiglu_limit=SWIGLU_LIMIT,
        scales_packed_nn=True,
    )
    tensors["sh"][:] = sh.to(torch.bfloat16)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    x_local_bf16 = torch.randn(T, D, dtype=torch.bfloat16)

    # Real MXFP8 grid (block=32); chan_cv reproduces per-output-channel magnitude spread.
    SHARED_DEQUANT_STD = {"w1": 1.71e-2, "w2": 1.68e-2, "w3": 1.70e-2}
    sw1, sw1_s = gen_shared_weight((MOE_INTER, D), SHARED_DEQUANT_STD["w1"], chan_cv=0.50)
    sw3, sw3_s = gen_shared_weight((MOE_INTER, D), SHARED_DEQUANT_STD["w3"], chan_cv=0.50)
    sw2, sw2_s = gen_shared_weight((D, MOE_INTER), SHARED_DEQUANT_STD["w2"], chan_cv=0.33)

    return [
        TensorSpec("x_local", [T, D], torch.bfloat16, init_value=lambda: x_local_bf16),
        TensorSpec("shared_w1", [D, MOE_INTER], torch.float8_e4m3fn, init_value=lambda: sw1),
        TensorSpec(
            "shared_w1_scale", [D_SCALE, MOE_INTER], torch.float8_e8m0fnu, init_value=lambda: sw1_s
        ),
        TensorSpec("shared_w3", [D, MOE_INTER], torch.float8_e4m3fn, init_value=lambda: sw3),
        TensorSpec(
            "shared_w3_scale", [D_SCALE, MOE_INTER], torch.float8_e8m0fnu, init_value=lambda: sw3_s
        ),
        TensorSpec("shared_w2", [MOE_INTER, D], torch.float8_e4m3fn, init_value=lambda: sw2),
        TensorSpec(
            "shared_w2_scale", [INTER_SCALE, D], torch.float8_e8m0fnu, init_value=lambda: sw2_s
        ),
        TensorSpec("sh", [T, D], torch.bfloat16, is_output=True),
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
        fn=expert_shared_test,
        specs=build_tensor_specs(),
        golden_fn=golden_expert_shared,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=moe_tol["rtol"],
        atol=moe_tol["atol"],
        compare_fn={
            "sh": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
