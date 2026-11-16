# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""FP32 activations, corrected RMSNorm primitives and CPU reduction references."""

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


@pl.jit.inline
def accurate_rsqrt(x: pl.Tensor[[M, N], pl.FP32]):
    """Refine a high-precision inverse root with both FP32 product residuals."""
    approximate = pl.rsqrt(x, high_precision=True)
    squared = pl.mul(approximate, approximate)
    ah = pl.mul(approximate, 4097.0)
    ah = pl.sub(ah, pl.sub(ah, approximate))
    al = pl.sub(approximate, ah)
    bh = pl.mul(approximate, 4097.0)
    bh = pl.sub(bh, pl.sub(bh, approximate))
    bl = pl.sub(approximate, bh)
    pe = pl.add(pl.add(pl.sub(pl.mul(ah, bh), squared), pl.mul(ah, bl)), pl.mul(al, bh))
    pe = pl.add(pe, pl.mul(al, bl))
    square_tail = pe
    product = pl.mul(squared, x)
    ah = pl.mul(squared, 4097.0)
    ah = pl.sub(ah, pl.sub(ah, squared))
    al = pl.sub(squared, ah)
    bh = pl.mul(x, 4097.0)
    bh = pl.sub(bh, pl.sub(bh, x))
    bl = pl.sub(x, bh)
    pe = pl.add(pl.add(pl.sub(pl.mul(ah, bh), product), pl.mul(ah, bl)), pl.mul(al, bh))
    pe = pl.add(pe, pl.mul(al, bl))
    product_tail = pl.add(pe, pl.mul(square_tail, x))
    residual = pl.sub(pl.add(pl.neg(product), 1.0), product_tail)
    correction = pl.mul(pl.mul(approximate, 0.5), residual)
    return pl.add(approximate, correction)


@pl.jit.inline
def accurate_row_sum_parts(x: pl.Tensor[[M, N], pl.FP32], N_BOUND: pl.constexpr):
    """Return high/low row sums within a power-of-two column bound of at least 128."""
    row_magnitude = pl.row_max(pl.abs(x))
    bounded_magnitude = pl.mul(row_magnitude, N_BOUND / 128)
    magnitude = pl.maximum(bounded_magnitude, 1e-30)
    magnitude_bits = pl.reinterpret_view(magnitude, pl.INT32)
    exponent = pl.shr(magnitude_bits, 23)
    # N_BOUND rounded lanes sum to at most 2**30 in signed INT32.
    scale_exponent = pl.add(pl.neg(exponent), 276)
    scale_bits = pl.shl(scale_exponent, 23)
    scale = pl.reinterpret_view(scale_bits, pl.FP32)
    scaled = pl.row_expand_mul(x, scale)
    integral = pl.cast(scaled, pl.INT32, mode="rint")
    integral_sum = pl.row_sum(integral)
    high = pl.cast(integral_sum, pl.FP32, mode="rint")
    high_integral = pl.cast(high, pl.INT32, mode="rint")
    lost = pl.cast(pl.sub(integral_sum, high_integral), pl.FP32, mode="rint")
    remainder = pl.sub(scaled, pl.cast(integral, pl.FP32, mode="rint"))
    low = pl.add(pl.row_sum(remainder), lost)
    inverse_bits = pl.shl(pl.sub(exponent, 22), 23)
    inverse_scale = pl.reinterpret_view(inverse_bits, pl.FP32)
    return pl.mul(high, inverse_scale), pl.mul(low, inverse_scale)


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
