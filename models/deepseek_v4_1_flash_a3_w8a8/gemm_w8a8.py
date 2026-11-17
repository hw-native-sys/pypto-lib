# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""W8A8 INT8 GEMM: per-token activations x per-output-channel weights -> BF16."""

import pypto.language as pl
from config import T_DYN
import torch

from utils import pack_nz, quant_w_per_channel, unpack_nz


# Dynamic shape variable.

# tiling
M_TILE = 16
N_TILE = 128
K_TILE = 256


@pl.jit.inline(auto_scope=False)
def w8a8_gemm(
    x_i8: pl.Tensor,
    x_scale: pl.Tensor,
    weight: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.Tensor,
    width: pl.constexpr,
    output_width: pl.constexpr,
    output_fp32: pl.constexpr = False,
):
    """Accumulate INT8 products in INT32 and dequantize both scales into BF16 or FP32."""
    t_dim = pl.tensor.dim(x_i8, 0)
    # A5 layout: token tiles are the parallel loop and n-tiles are the SPMD
    # block index.  The NZ column offset is then the (provable) block index
    # times N_TILE; neither a modulo-derived offset nor a sequential dynamic
    # range survives the combined-program front end at inline depth.
    token_tiles = (t_dim + M_TILE - 1) // M_TILE
    n_tiles = output_width // N_TILE
    for t_block in pl.parallel(token_tiles):
        t0 = t_block * M_TILE
        rows = pl.min(M_TILE, t_dim - t0)
        for n_block in pl.spmd(n_tiles, name_hint="w8a8_gemm"):
            n0 = n_block * N_TILE
            acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, width, K_TILE, stage=2):
                a = pl.slice(x_i8, [M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                a = pl.set_validshape(pl.fillpad(a, pad_value=pl.PadValue.zero), M_TILE, K_TILE)
                b = weight[k0 : k0 + K_TILE, n0 : n0 + N_TILE]
                acc = pl.matmul_acc(acc, a, b, init_cond=(k0 == 0))
            acc_fp32 = pl.cast(acc, target_type=pl.FP32)
            row_scale = pl.reshape(x_scale[t0 : t0 + M_TILE, 0:1], [M_TILE, 1])
            col_scale = pl.reshape(weight_scale[n0 : n0 + N_TILE], [1, N_TILE])
            scaled = pl.row_expand_mul(acc_fp32, row_scale)
            scaled = pl.col_expand_mul(scaled, col_scale)
            if output_fp32:
                output[t0 : t0 + M_TILE, n0 : n0 + N_TILE] = pl.set_validshape(scaled, rows, N_TILE)
            else:
                out_tile = pl.cast(scaled, target_type=pl.BF16, mode="rint")
                output[t0 : t0 + M_TILE, n0 : n0 + N_TILE] = pl.set_validshape(out_tile, rows, N_TILE)
    return output


@pl.jit
def w8a8_gemm_test(
    x_i8: pl.Tensor[[T_DYN, 5120], pl.INT8],
    x_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    weight: pl.Tensor[[5120, 2304], pl.INT8, pl.NZ],
    weight_scale: pl.Tensor[[2304], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, 2304], pl.BF16]],
):
    x_i8.bind_dynamic(0, T_DYN)
    x_scale.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    w8a8_gemm(x_i8, x_scale, weight, weight_scale, output, width=5120, output_width=2304, output_fp32=False)
    return output


def golden_w8a8_gemm(tensors):
    """Torch reference: INT32 accumulation with per-token and per-channel dequant."""
    x_i8 = tensors["x_i8"]
    x_scale = tensors["x_scale"]
    weight = unpack_nz(tensors["weight"])
    weight_scale = tensors["weight_scale"]
    acc = torch.matmul(x_i8.to(torch.int32), weight.to(torch.int32))
    value = acc.float() * x_scale.float() * weight_scale.float().reshape(1, -1)
    tensors["output"][:] = value.to(torch.bfloat16)


def build_tensor_specs(tokens: int = 64):
    from golden import TensorSpec

    def make_weight():
        logical, scale = quant_w_per_channel(torch.randn(2304, 5120, dtype=torch.float32) * 0.05)
        return pack_nz(logical.transpose(0, 1).contiguous()), scale

    weight_holder = {}

    def weight_arg():
        if "value" not in weight_holder:
            weight_holder["value"] = make_weight()
        return weight_holder["value"][0]

    def scale_arg():
        weight_arg()
        return weight_holder["value"][1]

    return [
        TensorSpec("x_i8", [tokens, 5120], torch.int8, init_value=lambda: torch.randint(-127, 128, (tokens, 5120))),
        TensorSpec("x_scale", [tokens, 1], torch.float32, init_value=lambda: torch.rand(tokens, 1) * 0.02 + 1e-4),
        TensorSpec("weight", [5120, 2304], torch.int8, init_value=weight_arg),
        TensorSpec("weight_scale", [2304], torch.float32, init_value=scale_arg),
        TensorSpec("output", [tokens, 2304], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    from config import add_parallel_arguments
    add_parallel_arguments(parser)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    result = run(
        fn=w8a8_gemm_test,
        specs=build_tensor_specs(args.tokens),
        golden_fn=golden_w8a8_gemm,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=1e-2,
        atol=1e-2,
        compare_fn={"output": ratio_allclose(atol=1e-2, rtol=1.0 / 128, max_error_ratio=5e-3)},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
