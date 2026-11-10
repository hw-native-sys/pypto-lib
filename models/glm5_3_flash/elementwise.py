# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared FP32 activations and independent CPU reduction references."""

import pypto.language as pl
import torch

FP32_NEG_INF = -3.4028234663852886e38  # Finite mask floor for sort and softmax.
M = pl.dynamic("ELEMENTWISE_M")
N = pl.dynamic("ELEMENTWISE_N")


@pl.jit.inline
def sigmoid_fp32(x: pl.Tensor[[M, N], pl.FP32], ROWS: pl.constexpr, COLS: pl.constexpr):
    argument = pl.create_tensor([ROWS, COLS], dtype=pl.FP32)
    argument[:, :] = pl.neg(x)
    exponential = pl.exp(argument)
    denominator = pl.create_tensor([ROWS, COLS], dtype=pl.FP32)
    denominator[:, :] = pl.add(exponential, 1.0)
    return pl.recip(denominator)


@pl.jit.inline
def silu(x: pl.Tensor[[M, N], pl.FP32], ROWS: pl.constexpr, COLS: pl.constexpr):
    """Evaluate FP32 SiLU for BF16 convolution outputs."""
    value = pl.create_tensor([ROWS, COLS], dtype=pl.FP32)
    value[:, :] = x
    argument = pl.create_tensor([ROWS, COLS], dtype=pl.FP32)
    argument[:, :] = pl.neg(value)
    exponential = pl.exp(argument)
    denominator = pl.create_tensor([ROWS, COLS], dtype=pl.FP32)
    denominator[:, :] = pl.add(exponential, 1.0)
    return pl.div(value, denominator)


def _sum_fp32_reference(
    value: torch.Tensor,
    dim: int = -1,
    keepdim: bool = True,
    tile_size: int | None = None,
) -> torch.Tensor:
    """Round an independent wide reduction to FP32, preserving requested tile boundaries."""
    if tile_size is None:
        return value.double().sum(dim=dim, keepdim=keepdim).float()
    total = None
    for tile in value.split(tile_size, dim=dim):
        partial = tile.float().sum(dim=dim, keepdim=True)
        total = partial if total is None else total + partial
    return total if keepdim else total.squeeze(dim)
