# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""DeepSeek-V4.1 shared MXFP8 W13, SwiGLU requantization, and W2."""

import sys
from math import gcd
from pathlib import Path

import pypto.language as pl

from pypto.runtime import pto_isa_include_dir

import config


def _parse_int_option(flag, default):
    for index, argument in enumerate(sys.argv):
        if argument == flag and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if argument.startswith(flag + "="):
            return int(argument.split("=", 1)[1])
    return default


if __name__ == "__main__":
    config.set_moe_tokens(_parse_int_option("--tokens", 80))


from config import FLASH as M

from config import MOE_TOKENS


# model config
T = MOE_TOKENS
T_PAD = ((T + 15) // 16) * 16
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit
MX_GROUP = M.weight_block_size[0]

# tiling
GATE_UP_N = 2 * MOE_INTER
K_SCALE = D // MX_GROUP
H_SCALE = MOE_INTER // MX_GROUP
W13_M_TILE = 80
W13_N_TILE = 256
W13_BLOCKS = GATE_UP_N // W13_N_TILE
W13_GROUPS = (T_PAD + W13_M_TILE - 1) // W13_M_TILE
W13_ROWS = W13_GROUPS * W13_M_TILE
W13_PAD_ROWS = 16  # staging granularity when the token block is not a whole tile
SWIGLU_WORKERS = 60
W2_M_TILE = gcd(T_PAD, W13_M_TILE)
W2_N_TILE = 256
W2_K_TILE = 256
W2_K_STAGE = 2
W2_SYNC_START = True

assert T_PAD % W13_PAD_ROWS == 0
assert T_PAD % 16 == 0
assert GATE_UP_N % W13_N_TILE == 0
assert MOE_INTER % W2_K_TILE == 0
assert D % W2_N_TILE == 0
assert W2_M_TILE % 16 == 0


@pl.jit.extern(
    core_type="aic",
    source=Path(__file__).parent / "kernels" / "expert_shared_w13.cpp",
    include_dirs=(pto_isa_include_dir(),),
)
def _shared_w13_mm(
    x_local: pl.Tensor,
    x_local_scale: pl.Tensor,
    shared_w1: pl.Tensor,
    shared_w1_scale: pl.Tensor,
    shared_w3: pl.Tensor,
    shared_w3_scale: pl.Tensor,
    gate_up: pl.Out[pl.Tensor],
    rows: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    """Project ND MXFP8 activations with ND weights and resident MX scales."""
    ...


@pl.jit.extern(
    core_type="aiv",
    source=Path(__file__).parent / "kernels" / "expert_shared_swiglu.cpp",
    include_dirs=(pto_isa_include_dir(),),
)
def _shared_swiglu_vf(
    gate_up: pl.Tensor,
    payload: pl.Out[pl.Tensor],
    scales: pl.Out[pl.Tensor],
):
    """Fuse the SwiGLU epilogue with upward MXFP8 requantization in one VF."""
    ...


@pl.jit.inline
def _w13_padded_inputs(
    x_local: pl.Tensor,
    x_local_scale: pl.Tensor,
):
    """Pad payload and packed scales to whole W13 row tiles."""
    x_padded = pl.create_tensor([W13_ROWS, D], dtype=pl.FP8E4M3FN)
    scale_padded = pl.create_tensor([1, W13_ROWS * K_SCALE], dtype=pl.FP8E8M0)
    scale_col = W13_PAD_ROWS * K_SCALE
    with pl.spmd(T_PAD // W13_PAD_ROWS, name_hint="sh_w13_pad"):
        row0 = pl.tile.get_block_idx() * W13_PAD_ROWS
        x_rows = pl.load(x_local, [row0, 0], [W13_PAD_ROWS, D])
        x_padded = pl.store(x_rows, [row0, 0], x_padded)
        scale_rows = pl.load(x_local_scale, [0, row0 * K_SCALE], [1, scale_col])
        scale_padded = pl.store(scale_rows, [0, row0 * K_SCALE], scale_padded)
    return x_padded, scale_padded


@pl.jit.inline
def _w13_direct_inputs(
    x_local: pl.Tensor,
    x_local_scale: pl.Tensor,
):
    """The physical token block already spans whole W13 row tiles."""
    return x_local, x_local_scale


# Selected at import time so the fused program traces exactly one input path.
_W13_INPUTS = _w13_direct_inputs if W13_ROWS == T_PAD else _w13_padded_inputs


@pl.jit.inline
def _shared_swiglu(
    gate_up: pl.Tensor,
    payload: pl.Out[pl.Tensor],
    scales: pl.Out[pl.Tensor],
):
    """SwiGLU epilogue plus grouped MXFP8 requantization over the token block."""
    with pl.spmd(SWIGLU_WORKERS, name_hint="sh_swiglu_vf"):
        _shared_swiglu_vf(gate_up, payload, scales)
    return payload, scales


import torch.nn.functional as F


@pl.jit.inline
def expert_shared(
    x_local: pl.Tensor[[T_PAD, D], pl.FP8E4M3FN],
    x_local_scale: pl.Tensor[[1, T_PAD * (D // MX_GROUP)], pl.FP8E8M0],
    shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    sh: pl.Tensor[[T, D], pl.BF16],
):
    """Shared expert with device MXFP8 activation quantization before both projections."""
    # Merged W1/W3 projection into BF16 gate/up columns.
    gate_up = pl.create_tensor([T_PAD, GATE_UP_N], dtype=pl.BF16)
    x_w13, x_w13_scale = _W13_INPUTS(x_local, x_local_scale)
    for group in pl.parallel(W13_GROUPS):
        row0 = group * W13_M_TILE
        group_rows = pl.min(W13_M_TILE, T_PAD - row0)
        x_group = pl.tensor.slice(x_w13, [W13_M_TILE, D], [row0, 0])
        # Contiguous packed MX_A_ZZ row tiles.
        x_scale_group = pl.tensor.slice(
            x_w13_scale,
            [1, W13_M_TILE * K_SCALE],
            [0, row0 * K_SCALE],
        )
        gate_up_group = pl.tensor.slice(gate_up, [W13_M_TILE, GATE_UP_N], [row0, 0])
        with pl.spmd(W13_BLOCKS, name_hint="sh_w13_mx_mm"):
            gate_up_group = _shared_w13_mm(
                x_group, x_scale_group,
                shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
                gate_up_group, group_rows,
            )

    # BF16-rounded SwiGLU, FP8 payload, and packed MX_A_ZZ scales.
    h_tile_mx = pl.create_tensor([T_PAD, MOE_INTER], dtype=pl.FP8E4M3FN)
    h_scale_mx = pl.create_tensor([T_PAD, H_SCALE], dtype=pl.FP8E8M0)
    h_tile_mx, h_scale_mx = _shared_swiglu(gate_up, h_tile_mx, h_scale_mx)

    # Complete row tiles keep MX_A_ZZ scale loads inside the packed backing.
    h_tile_scale_zz = pl.tensor.view(
        h_scale_mx, [T_PAD, H_SCALE], layout=pl.MX_A_ZZ,
    )
    rows = pl.tensor.dim(sh, 0)
    for wm in pl.parallel((T_PAD + W2_M_TILE - 1) // W2_M_TILE):
        row0 = wm * W2_M_TILE
        valid_m = pl.min(W2_M_TILE, T_PAD - row0)
        valid_out = pl.min(W2_M_TILE, rows - row0)
        with pl.spmd(D // W2_N_TILE, name_hint="sh_w2_mm", sync_start=W2_SYNC_START):
            d0 = pl.tile.get_block_idx() * W2_N_TILE
            hs0 = pl.load(h_tile_mx, [row0, 0], [W2_M_TILE, W2_K_TILE], valid_shape=[valid_m, W2_K_TILE])
            hs_scale0 = pl.load(h_tile_scale_zz, [row0, 0], [W2_M_TILE, W2_K_TILE // MX_GROUP], valid_shape=[valid_m, W2_K_TILE // MX_GROUP])
            sw2_0 = pl.load(shared_w2, [0, d0], [W2_K_TILE, W2_N_TILE])
            sw2_scale0 = pl.load(
                shared_w2_scale, [0, d0], [W2_K_TILE // MX_GROUP, W2_N_TILE],
            )
            y_acc = pl.matmul_mx(hs0, hs_scale0, sw2_0, sw2_scale0)
            for k0 in pl.pipeline(W2_K_TILE, MOE_INTER, W2_K_TILE, stage=W2_K_STAGE):
                hs_part = pl.load(h_tile_mx, [row0, k0], [W2_M_TILE, W2_K_TILE], valid_shape=[valid_m, W2_K_TILE])
                hs_scale_part = pl.load(
                    h_tile_scale_zz,
                    [row0, k0 // MX_GROUP],
                    [W2_M_TILE, W2_K_TILE // MX_GROUP], valid_shape=[valid_m, W2_K_TILE // MX_GROUP],
                )
                sw2_part = pl.load(shared_w2, [k0, d0], [W2_K_TILE, W2_N_TILE])
                sw2_scale_part = pl.load(
                    shared_w2_scale, [k0 // MX_GROUP, d0], [W2_K_TILE // MX_GROUP, W2_N_TILE],
                )
                y_acc = pl.matmul_mx_acc(
                    y_acc, hs_part, hs_scale_part, sw2_part, sw2_scale_part,
                )
            # Fixpipe rounds the accumulator directly to BF16.
            sh = pl.store(pl.set_validshape(y_acc, valid_out, W2_N_TILE), [row0, d0], sh)

    return sh


@pl.jit
def l2_expert_shared(
    x_local: pl.Tensor[[T_PAD, D], pl.FP8E4M3FN],
    x_local_scale: pl.Tensor[[1, T_PAD * (D // MX_GROUP)], pl.FP8E8M0],
    shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    sh: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    expert_shared(
        x_local, x_local_scale, shared_w1, shared_w1_scale,
        shared_w3, shared_w3_scale, shared_w2, shared_w2_scale, sh,
    )
    return sh


def golden_expert_shared(tensors):
    """Reference for MXFP8 W1/W3, SwiGLU requantization, and MXFP8 W2."""
    import torch

    from quant import unpack_mx_scale_codes as decode_e8m0_codes, quantize_mxfp8_upward_e8m0 as host_quant_mxfp8
    from gemm import golden_gemm_mxfp8_to_fp32_dequantized as matmul_mx_golden

    x_fp8 = tensors["x_local"][:T]
    x_scale = decode_e8m0_codes(
        tensors["x_local_scale"].reshape(T_PAD, D // MX_GROUP), side="a",
    )[:T]
    w1_fp8 = tensors["shared_w1"]
    w1_scale = decode_e8m0_codes(tensors["shared_w1_scale"], side="b")
    w3_fp8 = tensors["shared_w3"]
    w3_scale = decode_e8m0_codes(tensors["shared_w3_scale"], side="b")
    w2_fp8 = tensors["shared_w2"]
    w2_scale = decode_e8m0_codes(tensors["shared_w2_scale"], side="b")

    sh_gate = matmul_mx_golden(x_fp8, x_scale, w1_fp8, w1_scale).bfloat16().float()
    sh_up = matmul_mx_golden(x_fp8, x_scale, w3_fp8, w3_scale).bfloat16().float()
    if SWIGLU_LIMIT > 0:
        sh_gate = sh_gate.clamp(max=SWIGLU_LIMIT)
        sh_up = sh_up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    sh_h = (F.silu(sh_gate) * sh_up).bfloat16().float()
    sh_h_fp8, sh_h_scale = host_quant_mxfp8(sh_h)
    sh = matmul_mx_golden(sh_h_fp8, sh_h_scale, w2_fp8, w2_scale)

    tensors["sh"][:] = sh.to(torch.bfloat16)


def build_tensor_specs():
    import torch
    from golden.spec import TensorSpec
    from quant import generate_mxfp8_weight_kn as gen_mxfp8_weight_kn_device, quantize_mxfp8_upward_e8m0 as host_mxfp8_activation

    x_local_bf16 = torch.randn(T_PAD, D, dtype=torch.bfloat16)
    if T < T_PAD:
        x_local_bf16[T:, :] = 0
    x_local_fp8, x_local_scale_codes = host_mxfp8_activation(x_local_bf16)
    # matmul_mx expects the physical MX_A_ZZ scale order.
    from quant import pack_mx_a_scale
    x_local_scale_codes = pack_mx_a_scale(x_local_scale_codes)
    fp8_e8m0 = getattr(torch, "float8_e8m0fnu", torch.uint8)
    x_local_scale = x_local_scale_codes.contiguous().view(fp8_e8m0)

    SHARED_DEQUANT_STD = {"w1": 1.71e-2, "w2": 1.68e-2, "w3": 1.70e-2}
    sw1_fp8, sw1_scale = gen_mxfp8_weight_kn_device(MOE_INTER, D, SHARED_DEQUANT_STD["w1"], chan_cv=0.50, seed=1)
    sw3_fp8, sw3_scale = gen_mxfp8_weight_kn_device(MOE_INTER, D, SHARED_DEQUANT_STD["w3"], chan_cv=0.50, seed=2)
    sw2_fp8, sw2_scale = gen_mxfp8_weight_kn_device(
        D, MOE_INTER, SHARED_DEQUANT_STD["w2"], chan_cv=0.33, seed=3
    )
    sw1_scale = sw1_scale.contiguous().view(fp8_e8m0)
    sw3_scale = sw3_scale.contiguous().view(fp8_e8m0)
    sw2_scale = sw2_scale.contiguous().view(fp8_e8m0)

    fp8 = torch.float8_e4m3fn

    return [
        TensorSpec("x_local", [T_PAD, D], fp8, init_value=lambda: x_local_fp8),
        TensorSpec("x_local_scale", [1, T_PAD * (D // MX_GROUP)], fp8_e8m0, init_value=lambda: x_local_scale.reshape(1, -1)),
        TensorSpec("shared_w1", [D, MOE_INTER], fp8, init_value=lambda: sw1_fp8),
        TensorSpec("shared_w1_scale", [D // MX_GROUP, MOE_INTER], fp8_e8m0, init_value=lambda: sw1_scale),
        TensorSpec("shared_w3", [D, MOE_INTER], fp8, init_value=lambda: sw3_fp8),
        TensorSpec("shared_w3_scale", [D // MX_GROUP, MOE_INTER], fp8_e8m0, init_value=lambda: sw3_scale),
        TensorSpec("shared_w2", [MOE_INTER, D], fp8, init_value=lambda: sw2_fp8),
        TensorSpec("shared_w2_scale", [MOE_INTER // MX_GROUP, D], fp8_e8m0, init_value=lambda: sw2_scale),
        TensorSpec("sh", [T, D], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse
    from golden.validation import ratio_reldiff
    from golden.runner import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    if args.execution_mode:
        parser.error("this entry requires hardware; --sim is unsupported")
    if args.tokens is None:
        if args.golden_data or args.runtime_dir:
            parser.error("--golden-data and --runtime-dir require an explicit --tokens value")
        import subprocess

        for tokens in (16, 80, 96):
            print(f"Shared expert validation: tokens={tokens}", flush=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--tokens", str(tokens), *sys.argv[1:]]
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                raise SystemExit(completed.returncode)
        raise SystemExit(0)

    result = run(
        fn=l2_expert_shared,
        specs=build_tensor_specs(),
        golden_fn=golden_expert_shared,
        golden_data=args.golden_data,
        save_data=args.save_data,
        runtime_dir=args.runtime_dir,
        compile_only=args.compile_only,
        config=dict(
            dump_passes=args.dump_passes,
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "sh": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
