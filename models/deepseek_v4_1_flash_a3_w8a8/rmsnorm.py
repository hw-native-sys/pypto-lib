# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""BF16 and FP32 RMSNorm with FP32 accumulation over 512-column chunks."""

import pypto.language as pl
import torch

from config import FLASH as M, T_DYN


# Dynamic shape variable.

# model config
D = M.hidden_size
EPS = M.rms_norm_eps

# tiling
T_TILE = 8
D_TILE = 256


@pl.jit.inline(auto_scope=False)
def rms_norm(
    x: pl.Tensor,
    norm_w: pl.Tensor,
    x_normed: pl.Tensor,
    width: pl.constexpr,
    output_fp32: pl.constexpr,
    column_tile: pl.constexpr = D_TILE,
):
    """Normalize BF16 token rows and apply the BF16 gamma weight."""
    t_dim = pl.tensor.dim(x, 0)
    blocks = (t_dim + T_TILE - 1) // T_TILE
    for block in pl.spmd(blocks, name_hint="rms_norm"):
        t0 = block * T_TILE
        valid_rows = pl.min(T_TILE, t_dim - t0)
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for d0 in pl.pipeline(0, width, column_tile, stage=2):
            source = pl.slice(x, [T_TILE, column_tile], [t0, d0], valid_shape=[valid_rows, column_tile])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), T_TILE, column_tile)
            value = pl.cast(source, target_type=pl.FP32)
            row_sum = pl.reshape(pl.row_sum(pl.mul(value, value)), [1, T_TILE])
            sq_sum = pl.add(sq_sum, row_sum)
        inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, 1.0 / width), EPS), high_precision=True)
        inv_rms_t = pl.reshape(inv_rms, [T_TILE, 1])
        for d0 in pl.pipeline(0, width, column_tile, stage=2):
            source = pl.slice(x, [T_TILE, column_tile], [t0, d0], valid_shape=[valid_rows, column_tile])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), T_TILE, column_tile)
            value = pl.cast(source, target_type=pl.FP32)
            gamma = pl.reshape(pl.cast(norm_w[d0 : d0 + column_tile], target_type=pl.FP32), [1, column_tile])
            normalized = pl.col_expand_mul(pl.row_expand_mul(value, inv_rms_t), gamma)
            if output_fp32:
                x_normed[t0 : t0 + T_TILE, d0 : d0 + column_tile] = pl.set_validshape(normalized, valid_rows, column_tile)
            else:
                rounded = pl.cast(normalized, target_type=pl.BF16, mode="rint")
                x_normed[t0 : t0 + T_TILE, d0 : d0 + column_tile] = pl.set_validshape(rounded, valid_rows, column_tile)
    return x_normed


@pl.jit.inline(auto_scope=False)
def rms_norm_fp32(
    x: pl.Tensor,
    norm_w: pl.Tensor,
    x_normed: pl.Tensor,
    width: pl.constexpr,
    output_fp32: pl.constexpr,
):
    """Normalize FP32 token rows and apply the BF16 gamma weight."""
    t_dim = pl.tensor.dim(x, 0)
    blocks = (t_dim + T_TILE - 1) // T_TILE
    for block in pl.spmd(blocks, name_hint="rms_norm_fp32"):
        t0 = block * T_TILE
        valid_rows = pl.min(T_TILE, t_dim - t0)
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for d0 in pl.pipeline(0, width, D_TILE, stage=2):
            source = pl.slice(x, [T_TILE, D_TILE], [t0, d0], valid_shape=[valid_rows, D_TILE])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), T_TILE, D_TILE)
            row_sum = pl.reshape(pl.row_sum(pl.mul(source, source)), [1, T_TILE])
            sq_sum = pl.add(sq_sum, row_sum)
        inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, 1.0 / width), EPS), high_precision=True)
        inv_rms_t = pl.reshape(inv_rms, [T_TILE, 1])
        for d0 in pl.pipeline(0, width, D_TILE, stage=2):
            source = pl.slice(x, [T_TILE, D_TILE], [t0, d0], valid_shape=[valid_rows, D_TILE])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), T_TILE, D_TILE)
            gamma = pl.reshape(pl.cast(norm_w[d0 : d0 + D_TILE], target_type=pl.FP32), [1, D_TILE])
            normalized = pl.col_expand_mul(pl.row_expand_mul(source, inv_rms_t), gamma)
            if output_fp32:
                x_normed[t0 : t0 + T_TILE, d0 : d0 + D_TILE] = pl.set_validshape(normalized, valid_rows, D_TILE)
            else:
                rounded = pl.cast(normalized, target_type=pl.BF16, mode="rint")
                x_normed[t0 : t0 + T_TILE, d0 : d0 + D_TILE] = pl.set_validshape(rounded, valid_rows, D_TILE)
    return x_normed


@pl.jit.inline(auto_scope=False)
def rms_norm_fp32_unit(
    x: pl.Tensor,
    x_normed: pl.Tensor,
    width: pl.constexpr,
    output_fp32: pl.constexpr,
):
    """Normalize FP32 token rows without a gamma weight."""
    t_dim = pl.tensor.dim(x, 0)
    blocks = (t_dim + T_TILE - 1) // T_TILE
    for block in pl.spmd(blocks, name_hint="rms_norm_fp32_unit"):
        t0 = block * T_TILE
        valid_rows = pl.min(T_TILE, t_dim - t0)
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for d0 in pl.pipeline(0, width, D_TILE, stage=2):
            source = pl.slice(x, [T_TILE, D_TILE], [t0, d0], valid_shape=[valid_rows, D_TILE])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), T_TILE, D_TILE)
            row_sum = pl.reshape(pl.row_sum(pl.mul(source, source)), [1, T_TILE])
            sq_sum = pl.add(sq_sum, row_sum)
        inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, 1.0 / width), EPS), high_precision=True)
        inv_rms_t = pl.reshape(inv_rms, [T_TILE, 1])
        for d0 in pl.pipeline(0, width, D_TILE, stage=2):
            source = pl.slice(x, [T_TILE, D_TILE], [t0, d0], valid_shape=[valid_rows, D_TILE])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), T_TILE, D_TILE)
            normalized = pl.row_expand_mul(source, inv_rms_t)
            if output_fp32:
                x_normed[t0 : t0 + T_TILE, d0 : d0 + D_TILE] = pl.set_validshape(normalized, valid_rows, D_TILE)
            else:
                rounded = pl.cast(normalized, target_type=pl.BF16, mode="rint")
                x_normed[t0 : t0 + T_TILE, d0 : d0 + D_TILE] = pl.set_validshape(rounded, valid_rows, D_TILE)
    return x_normed


@pl.jit
def rms_norm_test(
    x: pl.Tensor[[T_DYN, 5120], pl.BF16],
    norm_w: pl.Tensor[[5120], pl.BF16],
    x_normed: pl.Out[pl.Tensor[[T_DYN, 5120], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    x_normed.bind_dynamic(0, T_DYN)
    rms_norm(x, norm_w, x_normed, width=5120, output_fp32=False, column_tile=256)
    return x_normed


def golden_rms_norm(tensors):
    """Torch reference: FP32 mean-square normalization and BF16 rounding."""
    value = tensors["x"].float()
    inv_rms = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + EPS)
    tensors["x_normed"][:] = (value * inv_rms * tensors["norm_w"].float()).to(torch.bfloat16)


def build_tensor_specs(tokens: int = 64):
    from golden import TensorSpec

    return [
        TensorSpec("x", [tokens, 5120], torch.bfloat16, init_value=torch.randn),
        TensorSpec("norm_w", [5120], torch.bfloat16, init_value=lambda: torch.randn(5120) * 0.1 + 1.0),
        TensorSpec("x_normed", [tokens, 5120], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run

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
        fn=rms_norm_test,
        specs=build_tensor_specs(args.tokens),
        golden_fn=golden_rms_norm,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=1e-2,
        atol=1e-2,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
