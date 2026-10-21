# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4.1 Flash RMSNorm kernels and shared Torch references."""

import pypto.language as pl
import torch

from config import FLASH as M, T_DYN


# model config
D = M.hidden_size


EPS = M.rms_norm_eps
NORM_T_TILE = 8
NORM_D_TILE = 512


@pl.jit.inline
def _rms_norm_tiled(
    x: pl.Tensor,
    norm_w: pl.Tensor[[D], pl.BF16],
    x_normed: pl.Tensor,
    block: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    compensated: pl.constexpr,
    output_fp32: pl.constexpr,
):
    """Normalize eight rows with optional compensated 512-column accumulation."""
    t0 = block * NORM_T_TILE
    remaining_rows = pl.max(num_tokens - t0, 0)
    valid_rows = pl.min(NORM_T_TILE, remaining_rows)
    sq_sum = pl.full([1, NORM_T_TILE], dtype=pl.FP32, value=0.0)
    compensation = pl.full([1, NORM_T_TILE], dtype=pl.FP32, value=0.0)
    for kb in pl.pipeline(D // NORM_D_TILE, stage=2):
        k0 = kb * NORM_D_TILE
        source = pl.slice(x, [NORM_T_TILE, NORM_D_TILE], [t0, k0], valid_shape=[valid_rows, NORM_D_TILE])
        source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), NORM_T_TILE, NORM_D_TILE)
        value = pl.cast(source, target_type=pl.FP32)
        partial = pl.reshape(pl.row_sum(pl.mul(value, value)), [1, NORM_T_TILE])
        if compensated:
            delta = pl.sub(partial, compensation)
            next_sum = pl.add(sq_sum, delta)
            compensation = pl.sub(pl.sub(next_sum, sq_sum), delta)
            sq_sum = next_sum
        else:
            sq_sum = pl.add(sq_sum, partial)
    inv_rms = pl.reshape(
        pl.rsqrt(pl.add(pl.mul(sq_sum, 1.0 / D), EPS), high_precision=True), [NORM_T_TILE, 1]
    )
    for kb in pl.pipeline(D // NORM_D_TILE, stage=2):
        k0 = kb * NORM_D_TILE
        source = pl.slice(x, [NORM_T_TILE, NORM_D_TILE], [t0, k0], valid_shape=[valid_rows, NORM_D_TILE])
        source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), NORM_T_TILE, NORM_D_TILE)
        value = pl.cast(source, target_type=pl.FP32)
        gamma = pl.reshape(pl.cast(norm_w[k0 : k0 + NORM_D_TILE], target_type=pl.FP32), [1, NORM_D_TILE])
        normalized = pl.col_expand_mul(pl.row_expand_mul(value, inv_rms), gamma)
        rounded = pl.cast(normalized, target_type=pl.BF16, mode="rint")
        if output_fp32:
            rounded_fp32 = pl.cast(rounded, pl.FP32)
            x_normed[t0 : t0 + NORM_T_TILE, k0 : k0 + NORM_D_TILE] = rounded_fp32
        else:
            valid_output = pl.set_validshape(rounded, valid_rows, NORM_D_TILE)
            x_normed[t0 : t0 + NORM_T_TILE, k0 : k0 + NORM_D_TILE] = valid_output
    return x_normed


@pl.jit.inline(auto_scope=False)
def rms_norm_tiled(
    x: pl.Tensor,
    norm_w: pl.Tensor[[D], pl.BF16],
    x_normed: pl.Tensor,
    block: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    output_fp32: pl.constexpr,
):
    """Normalize eight rows with BF16 rounding and optional padded FP32 output."""
    return _rms_norm_tiled(x, norm_w, x_normed, block, num_tokens, compensated=False, output_fp32=output_fp32)


@pl.jit.inline(auto_scope=False)
def rms_norm_tiled_compensated(
    x: pl.Tensor,
    norm_w: pl.Tensor[[D], pl.BF16],
    x_normed: pl.Tensor,
    block: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Normalize final projection inputs with compensated FP32 chunk sums."""
    return _rms_norm_tiled(x, norm_w, x_normed, block, num_tokens, compensated=True, output_fp32=False)


@pl.jit.inline
def rms_norm(
    x: pl.Tensor,
    weight: pl.Tensor,
    output: pl.Tensor,
    block: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.constexpr,
):
    """Normalize one eight-row block using whole-row reduction."""
    t = block * 8
    rows = pl.min(8, num_tokens - t)
    source = pl.slice(x, [8, width], [t, 0], valid_shape=[rows, width])
    source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 8, width)
    value = pl.cast(source, pl.FP32)
    row_inverse = pl.rsqrt(
        pl.add(pl.mul(pl.row_sum(pl.mul(value, value)), 1.0 / width), EPS),
        high_precision=True,
    )
    gamma = pl.reshape(pl.cast(weight[:], pl.FP32), [1, width])
    normalized = pl.col_expand_mul(pl.row_expand_mul(value, row_inverse), gamma)
    output[t : t + 8, :] = pl.set_validshape(
        pl.cast(normalized, pl.BF16, mode="rint"), rows, width
    )
    return output


def golden_rms_norm_tiled(x: torch.Tensor, norm_w: torch.Tensor) -> torch.Tensor:
    """Apply RMSNorm with FP32 reference accumulation over 512-column chunks."""
    value = x.float()
    sq_sum = torch.zeros(value.shape[0], 1, dtype=torch.float32)
    for d0 in range(0, D, NORM_D_TILE):
        chunk = value[:, d0 : d0 + NORM_D_TILE]
        sq_sum += (chunk * chunk).sum(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(sq_sum * (1.0 / D) + EPS)
    return (value * inv_rms * norm_w.float()).to(torch.bfloat16)


def golden_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = M.rms_norm_eps) -> torch.Tensor:
    """Apply RMSNorm in FP32 and restore the activation dtype."""
    dtype = x.dtype
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    return (value * weight.float()).to(dtype)


def make_norm_with_deps(width, *, max_workers=None, name_hint="attention_rmsnorm"):
    """Specialize whole-row RMSNorm with an explicit producer dependency."""
    worker_limit = 2**31 - 1 if max_workers is None else max_workers

    @pl.jit.inline
    def normalize(
        x: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width], pl.BF16],
        output: pl.Tensor[[T_DYN, width], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
        ready: pl.Scalar[pl.TASK_ID],
    ):
        blocks = (num_tokens + 7) // 8
        workers = pl.min(blocks, worker_limit)
        with pl.spmd(workers, name_hint=name_hint, deps=[ready]) as norm_tid:
            worker = pl.tile.get_block_idx()
            for block in pl.range(worker, blocks, workers):
                rms_norm(x, weight, output, block, num_tokens, width=width)
        return norm_tid

    return normalize
