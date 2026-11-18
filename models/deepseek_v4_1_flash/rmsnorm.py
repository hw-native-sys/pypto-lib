# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4.1 Flash RMSNorm kernels and shared Torch references."""

from __future__ import annotations

import pypto.language as pl
import torch

from config import FLASH as M


# model config
EPS = M.rms_norm_eps

# tiling
NORM_T_TILE = 8


# Tile dimensions bind to the constexpr width during specialization.
@pl.jit.inline(auto_scope=False)
def _rms_norm_rows(
    source: pl.Tile[[rows, width], pl.BF16, pl.Mem.Vec],  # noqa: F821
    gamma: pl.Tile[[width], pl.BF16, pl.Mem.Vec],  # noqa: F821
    width: pl.constexpr,
    rows: pl.constexpr,
    fold: pl.constexpr,
):
    """Normalize rows with folded squares and Newton inverse-root correction."""
    source_fp32 = pl.cast(source, pl.FP32)
    if width > 512:
        squares = pl.mul(source_fp32, source_fp32)
        folded = pl.tile.slice(squares, [rows, fold], [0, 0])
        leading = pl.tile.slice(squares, [rows, width - fold], [0, 0])
        trailing = pl.tile.slice(squares, [rows, width - fold], [0, fold])
        combined = pl.add(leading, trailing)
        folded = pl.tile.assemble(folded, combined, [0, 0])
        folded = pl.reshape(folded, [rows * fold])
    source_fp32 = pl.reshape(source_fp32, [rows * width])
    normalized = pl.create_tile([rows * width], dtype=pl.BF16, target_memory=pl.Mem.Vec)
    with pl.reg_scope():
        full = pl.reg.full_mask(lanes=64)
        lane = pl.reg.index_gen(0, lanes=64, dtype=pl.UINT32)
        for row in pl.hardware_loop(rows):
            partials = pl.reg.broadcast(0.0, lanes=64, dtype=pl.FP32)
            for chunk in pl.hardware_loop(fold // 64):
                if width <= 512:
                    left = pl.reg.load(source_fp32, row * width + chunk * 64, lanes=64)
                    right = pl.reg.load(source_fp32, row * width + fold + chunk * 64, lanes=64)
                    left_square = pl.reg.mul(left, left, full)
                    right_square = pl.reg.mul(right, right, full)
                    square = pl.reg.add(left_square, right_square, full)
                else:
                    square = pl.reg.load(folded, row * fold + chunk * 64, lanes=64)
                partial = pl.reg.reduce_sum(square, full, reassoc=True)
                partial = pl.reg.broadcast(partial, lanes=64, dtype=pl.FP32)
                slot_index = pl.cast(chunk, pl.UINT32)
                slot = pl.reg.compares(lane, slot_index, full, predicate="eq")
                partials = pl.reg.select(slot, partial, partials)
            partial_mask = pl.reg.active_mask(fold // 64, lanes=64)
            total = pl.reg.reduce_sum(partials, partial_mask, reassoc=True)
            total = pl.reg.broadcast(total, lanes=64, dtype=pl.FP32)
            mean_square = pl.reg.muls(total, 1.0 / width, full)
            variance = pl.reg.adds(mean_square, EPS, full)
            one = pl.reg.broadcast(1.0, lanes=64, dtype=pl.FP32)
            reciprocal = pl.reg.div(one, variance, full)
            root = pl.reg.sqrt(reciprocal, full)
            negative_half = pl.reg.muls(variance, -0.5, full)
            product = pl.reg.mul(negative_half, root, full)
            three_halves = pl.reg.broadcast(1.5, lanes=64, dtype=pl.FP32)
            factor = pl.reg.mula(three_halves, product, root, full)
            inverse = pl.reg.mul(root, factor, full)
            negative_variance = pl.reg.muls(variance, -1.0, full)
            correction = pl.reg.mula(one, negative_variance, reciprocal, full)
            negative_inverse = pl.reg.muls(inverse, -1.0, full)
            error = pl.reg.mula(reciprocal, negative_inverse, inverse, full)
            correction = pl.reg.mula(correction, variance, error, full)
            correction = pl.reg.mul(correction, inverse, full)
            half = pl.reg.broadcast(0.5, lanes=64, dtype=pl.FP32)
            inverse = pl.reg.mula(inverse, correction, half, full)
            variance_bits = pl.reg.interpret_cast(variance, dtype=pl.UINT32)
            is_infinite = pl.reg.compares(variance_bits, 2139095040, full, predicate="eq")
            zero = pl.reg.broadcast(0.0, lanes=64, dtype=pl.FP32)
            inverse = pl.reg.select(is_infinite, zero, inverse)
            for chunk in pl.hardware_loop(width // 64):
                column = chunk * 64
                value = pl.reg.load(source_fp32, row * width + column, lanes=64)
                gamma_raw = pl.reg.load(gamma, column, lanes=64)
                gain = pl.reg.convert(gamma_raw, dtype=pl.FP32)
                scaled = pl.reg.mul(value, inverse, full)
                result = pl.reg.mul(scaled, gain, full)
                rounded = pl.reg.convert(result, dtype=pl.BF16, rounding="R", saturate=False)
                _written = pl.reg.store(normalized, row * width + column, rounded)
    return normalized


@pl.jit.inline(auto_scope=False)
def rms_norm(
    x: pl.Tensor,
    weight: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.constexpr,
    zero_tail: pl.constexpr,
):
    """Normalize BF16 rows with folded FP32 reduction and inverse-root correction."""
    capacity = pl.tensor.dim(output, 0)
    requested = pl.cast(num_tokens, pl.INDEX)
    active = pl.max(pl.min(requested, capacity), 0)
    count = capacity
    if not zero_tail:
        count = active
    blocks = (count + NORM_T_TILE - 1) // NORM_T_TILE
    workers = pl.max(1, pl.min(blocks, pl.system.available_cluster_count() * 2))
    if width <= 512:
        workers = pl.min(workers, 8)
    with pl.spmd(workers, name_hint="rms_norm") as ready:
        worker = pl.tile.get_block_idx()
        gamma = pl.load(weight, [0], [width], target_memory=pl.Mem.Vec)
        if width <= 512:
            for start in pl.range(worker * NORM_T_TILE, count, workers * NORM_T_TILE):
                rows = pl.min(NORM_T_TILE, count - start)
                active_rows = pl.max(pl.min(NORM_T_TILE, active - start), 0)
                if active_rows > 0:
                    source = pl.load(x, [start, 0], [NORM_T_TILE, width], valid_shape=[active_rows, width])
                    source = pl.fillpad(source, pad_value=pl.PadValue.zero)
                    source = pl.set_validshape(source, NORM_T_TILE, width)
                    normalized = _rms_norm_rows(source, gamma, width, NORM_T_TILE, width // 2)
                    output_rows = pl.reshape(normalized, [NORM_T_TILE, width])
                    pl.store(pl.set_validshape(output_rows, rows, width), [start, 0], output)
                else:
                    zero_rows = pl.tile.full([NORM_T_TILE, width], dtype=pl.BF16, value=0.0)
                    pl.store(pl.set_validshape(zero_rows, rows, width), [start, 0], output)
        else:
            for row in pl.range(worker, count, workers):
                if row < active:
                    source = pl.load(x, [row, 0], [1, width], target_memory=pl.Mem.Vec)
                    normalized = _rms_norm_rows(source, gamma, width, 1, width * 4 // 5)
                    normalized_row = pl.reshape(normalized, [1, width])
                    pl.store(normalized_row, [row, 0], output)
                else:
                    zero_row = pl.tile.full([1, width], dtype=pl.BF16, value=0.0)
                    pl.store(zero_row, [row, 0], output)
    return ready


def golden_rms_norm_tiled(x: torch.Tensor, norm_w: torch.Tensor) -> torch.Tensor:
    """Apply the shared RMSNorm reference to hidden-width rows."""
    return golden_rms_norm(x, norm_w)


def golden_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Model folded FP32 square reduction and fused Newton correction."""
    value = x.float()
    width = value.shape[-1]
    square = value.square()
    fold = 1 << max((width - 1).bit_length() - 1, 0)
    combined = square[..., :fold].clone()
    combined[..., :width - fold] += square[..., fold:]
    if fold >= 64:
        partials = combined.reshape(*value.shape[:-1], -1, 64)
        while partials.shape[-1] > 1:
            partials = partials[..., 0::2] + partials[..., 1::2]
        total = partials.squeeze(-1)
    else:
        total = combined
    while total.shape[-1] > 1:
        total = total[..., 0::2] + total[..., 1::2]
    variance = total * (1.0 / width) + eps

    def fma(acc, lhs, rhs):
        return (acc.double() + lhs.double() * rhs.double()).float()

    one = torch.ones_like(variance)
    reciprocal = one / variance
    root = torch.sqrt(reciprocal)
    product = (variance * -0.5) * root
    inverse = root * fma(one * 1.5, product, root)
    correction = fma(one, -variance, reciprocal)
    error = fma(reciprocal, -inverse, inverse)
    correction = fma(correction, variance, error) * inverse
    inverse = fma(inverse, correction, one * 0.5)
    inverse = torch.where(torch.isposinf(variance), torch.zeros_like(inverse), inverse)
    inverse = torch.where(variance == 0, torch.full_like(inverse, float("inf")), inverse)
    return (value * inverse * weight.float()).to(x.dtype)
