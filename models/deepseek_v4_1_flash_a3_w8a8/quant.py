# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""W8A8 dynamic per-token INT8 activation quantization."""

import pypto.language as pl
import torch

from config import INT8_AMAX_EPS, INT8_SCALE_MAX, T_DYN
from utils import int8_quant_per_token


# Dynamic shape variable.

# tiling
T_TILE = 8
D_TILE = 256


@pl.jit.inline(auto_scope=False)
def quantize_int8_per_token(
    x: pl.Tensor,
    x_i8: pl.Tensor,
    x_scale: pl.Tensor,
    width: pl.constexpr,
):
    """Quantize each token row to INT8 with one FP32 dequant scale per row."""
    t_dim = pl.tensor.dim(x, 0)
    blocks = (t_dim + T_TILE - 1) // T_TILE
    for block in pl.spmd(blocks, name_hint="int8_quant_per_token"):
        t0 = block * T_TILE
        valid_rows = pl.min(T_TILE, t_dim - t0)
        amax = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for d0 in pl.pipeline(0, width, D_TILE, stage=2):
            source = pl.slice(x, [T_TILE, D_TILE], [t0, d0], valid_shape=[valid_rows, D_TILE])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), T_TILE, D_TILE)
            value = pl.cast(source, target_type=pl.FP32)
            row_amax = pl.reshape(pl.row_max(pl.abs(value)), [1, T_TILE])
            amax = pl.maximum(amax, row_amax)
        amax = pl.maximum(amax, INT8_AMAX_EPS)
        scale_quant = pl.div(pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), amax)
        scale_dequant = pl.recip(scale_quant)
        scale_dequant_t = pl.reshape(scale_dequant, [T_TILE, 1])
        valid_scale = pl.set_validshape(scale_dequant_t, valid_rows, 1)
        x_scale[t0 : t0 + T_TILE, 0:1] = valid_scale
        scale_quant_t = pl.reshape(scale_quant, [T_TILE, 1])
        for d0 in pl.pipeline(0, width, D_TILE, stage=2):
            source = pl.slice(x, [T_TILE, D_TILE], [t0, d0], valid_shape=[valid_rows, D_TILE])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), T_TILE, D_TILE)
            value = pl.cast(source, target_type=pl.FP32)
            scaled = pl.row_expand_mul(value, scale_quant_t)
            rounded_i32 = pl.cast(scaled, target_type=pl.INT32, mode="rint")
            rounded_f16 = pl.cast(rounded_i32, target_type=pl.FP16, mode="round")
            quantized = pl.cast(rounded_f16, target_type=pl.INT8, mode="trunc")
            valid_quantized = pl.set_validshape(quantized, valid_rows, D_TILE)
            x_i8[t0 : t0 + T_TILE, d0 : d0 + D_TILE] = valid_quantized
    return x_i8


@pl.jit
def int8_quant_test(
    x: pl.Tensor[[T_DYN, 5120], pl.BF16],
    x_i8: pl.Out[pl.Tensor[[T_DYN, 5120], pl.INT8]],
    x_scale: pl.Out[pl.Tensor[[T_DYN, 1], pl.FP32]],
):
    x.bind_dynamic(0, T_DYN)
    x_i8.bind_dynamic(0, T_DYN)
    x_scale.bind_dynamic(0, T_DYN)
    quantize_int8_per_token(x, x_i8, x_scale, width=5120)
    return x_i8, x_scale


def golden_int8_quant_per_token(tensors):
    """Torch reference for dynamic per-token INT8 quantization."""
    x = tensors["x"]
    x_i8, x_scale = int8_quant_per_token(x)
    tensors["x_i8"][:] = x_i8
    tensors["x_scale"][:] = x_scale


def build_tensor_specs(tokens: int = 16):
    from golden import TensorSpec

    return [
        TensorSpec("x", [tokens, 5120], torch.bfloat16, init_value=torch.randn),
        TensorSpec("x_i8", [tokens, 5120], torch.int8),
        TensorSpec("x_scale", [tokens, 1], torch.float32),
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
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    result = run(
        fn=int8_quant_test,
        specs=build_tensor_specs(args.tokens),
        golden_fn=golden_int8_quant_per_token,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=1e-5,
        atol=1e-5,
        compare_fn={
            "x_i8": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=1e-3),
            "x_scale": ratio_allclose(atol=0.0, rtol=1.0 / 128, max_error_ratio=1e-3),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
