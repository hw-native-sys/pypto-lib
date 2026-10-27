# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GLM-5.3-Flash BF16 hyper-connection mixing and stream collapse."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataclasses import dataclass

import pypto.language as pl
import torch

from models.glm5_3_flash.config import FLASH, HC_DIM, HC_MULT, MIX_HC, D, Glm53FlashConfig
from models.glm5_3_flash.elementwise import (
    _sum_fp32_reference, accurate_divide, accurate_exp, accurate_row_sum, accurate_row_sum_parts,
    accurate_rsqrt, accurate_sigmoid,
)
from models.glm5_3_flash.linear import bf16_linear
from models.glm5_3_flash.rmsnorm import unweighted_rms_norm
from models.glm5_3_flash.utils import _assert_close, _check, _randn, _report

T_DYN = pl.dynamic('GLM53_T_DYN')

HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = FLASH.hc_sinkhorn_iters
HC_EPS = FLASH.hc_eps
NORM_EPS = FLASH.rms_norm_eps
MIX_D_TILE = 256
MIX_PAD = 32  # MIX_HC = 24 padded to the cube's N granularity
HC_PAD = 8  # HC_MULT = 4 padded so an FP32 row is 32 bytes
RMS_T_TILE = 8  # token tile of the sum-of-squares scope
LINEAR_T_TILE = 16  # cube M tile; below 16 the matmul is not L0 split
COMB_T_TILE = 8  # token tile of the softmax / Sinkhorn scope
POST_COEFF_TILE = 16  # minimum 32-byte BF16 vector row
RMS_K_TILE = 128  # HC_DIM reduction chunk of the sum of squares


@pl.jit.inline
def sinkhorn_normalize(
    values: pl.Tensor[[HC_PAD, HC_PAD], pl.FP32],
    COLUMNS: pl.constexpr,
    EPS: pl.constexpr,
):
    rows = pl.create_tensor([HC_PAD, HC_PAD], dtype=pl.FP32)
    if COLUMNS:
        rows[:, :] = pl.transpose(values, 0, 1)
    else:
        rows[:, :] = values
    totals = accurate_row_sum(rows)
    ones = pl.full([HC_PAD, HC_PAD], dtype=pl.FP32, value=1.0)
    denominator = pl.create_tensor([HC_PAD, HC_PAD], dtype=pl.FP32)
    denominator[:, :] = pl.row_expand_mul(ones, pl.add(pl.mul(totals, 0.5), EPS))
    normalized = accurate_divide(rows, denominator)
    result = pl.create_tensor([HC_PAD, HC_PAD], dtype=pl.FP32)
    if COLUMNS:
        result[:, :] = pl.transpose(normalized, 0, 1)
    else:
        result[:, :] = normalized
    return result


@pl.jit.inline
def mhc_mixes(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    function: pl.Tensor[[MIX_HC, HC_DIM], pl.BF16],
    scale: pl.Tensor[[3], pl.FP32],
    base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
):
    t_dim = pl.tensor.dim(x_hc, 0)
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE

    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    for block in pl.spmd((t_dim + RMS_T_TILE - 1) // RMS_T_TILE, name_hint="mhc_rms"):
        t0 = block * RMS_T_TILE
        valid_rows = pl.min(RMS_T_TILE, t_dim - t0)
        sq_sum = pl.full([1, RMS_T_TILE], dtype=pl.FP32, value=0.0)
        sq_error = pl.full([1, RMS_T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HC_DIM // RMS_K_TILE, stage=4):
            k0 = kb * RMS_K_TILE
            rms_x_tile = pl.slice(x_flat, [RMS_T_TILE, RMS_K_TILE], [t0, k0], valid_shape=[valid_rows, RMS_K_TILE])
            rms_value = pl.cast(rms_x_tile, pl.FP32)
            rms_square = pl.create_tensor([RMS_T_TILE, RMS_K_TILE], dtype=pl.FP32)
            rms_square[:, :] = pl.mul(rms_value, rms_value)
            part_high, part_low = accurate_row_sum_parts(rms_square)
            sq_part = pl.reshape(part_high, [1, RMS_T_TILE])
            part_error = pl.reshape(part_low, [1, RMS_T_TILE])
            sq_updated = pl.add(sq_sum, sq_part)
            recovered = pl.sub(sq_updated, sq_sum)
            error = pl.add(pl.sub(sq_sum, pl.sub(sq_updated, recovered)), pl.sub(sq_part, recovered))
            sq_error = pl.add(sq_error, pl.add(error, part_error))
            sq_sum = sq_updated
        rms_arg = pl.create_tensor([1, RMS_T_TILE], dtype=pl.FP32)
        rms_arg[:, :] = pl.add(pl.mul(pl.add(sq_sum, sq_error), HC_DIM_INV), NORM_EPS)
        inverse = accurate_rsqrt(rms_arg)
        inv_rms[t0 : t0 + RMS_T_TILE, 0:1] = pl.reshape(inverse, [RMS_T_TILE, 1])

    mixes_raw = pl.create_tensor([t_linear, MIX_PAD], dtype=pl.FP32)
    for block in pl.spmd(t_linear // LINEAR_T_TILE, name_hint="mhc_linear"):
        t0 = block * LINEAR_T_TILE
        valid_rows = pl.min(LINEAR_T_TILE, t_dim - t0)
        zero = pl.cast(0, pl.INDEX)
        columns = pl.cast(MIX_HC, pl.INDEX)
        raw = bf16_linear(x_flat, function, t0, zero, valid_rows, columns, LINEAR_T_TILE, MIX_PAD, HC_DIM, True)
        mixes_raw[t0 : t0 + LINEAR_T_TILE, 0:MIX_PAD] = raw

    residual_flat = pl.reshape(residual_mix, [t_dim * HC_MULT, HC_MULT])
    for t in pl.spmd(t_dim, name_hint="mhc_coefficients"):
        inv = pl.read(inv_rms, [t, 0])
        s0 = pl.read(scale, [0])
        s1 = pl.read(scale, [1])
        s2 = pl.read(scale, [2])
        pre_raw = pl.create_tensor([1, HC_PAD], dtype=pl.FP32)
        post_raw = pl.create_tensor([1, HC_PAD], dtype=pl.FP32)
        pre_base = pl.create_tensor([1, HC_PAD], dtype=pl.FP32)
        post_base = pl.create_tensor([1, HC_PAD], dtype=pl.FP32)
        for j in pl.unroll(HC_PAD):
            h = j % HC_MULT
            pl.write(pre_raw, [0, j], pl.read(mixes_raw, [t, h]))
            pl.write(post_raw, [0, j], pl.read(mixes_raw, [t, HC_MULT + h]))
            pl.write(pre_base, [0, j], pl.read(base, [h]))
            pl.write(post_base, [0, j], pl.read(base, [HC_MULT + h]))
        pre_logits = pl.create_tensor([1, HC_PAD], dtype=pl.FP32)
        post_logits = pl.create_tensor([1, HC_PAD], dtype=pl.FP32)
        pre_logits[:, :] = pl.add(pl.mul(pl.mul(pre_raw, inv), s0), pre_base)
        post_logits[:, :] = pl.add(pl.mul(pl.mul(post_raw, inv), s1), post_base)
        pre_sigmoid = accurate_sigmoid(pre_logits, 1, HC_PAD)
        post_sigmoid = accurate_sigmoid(post_logits, 1, HC_PAD)
        pre = pl.add(pre_sigmoid, HC_EPS)
        post = pl.mul(post_sigmoid, 2.0)
        pre_mix[t : t + 1, :] = pl.set_validshape(pre, 1, HC_MULT)
        post_mix[t : t + 1, :] = pl.set_validshape(post, 1, HC_MULT)

        # Four identical quadrants; each reduction counts its logical axis twice.
        comb_raw = pl.create_tensor([HC_PAD, HC_PAD], dtype=pl.FP32)
        biases = pl.create_tensor([HC_PAD, HC_PAD], dtype=pl.FP32)
        for i in pl.unroll(HC_PAD):
            for j in pl.unroll(HC_PAD):
                lane = 2 * HC_MULT + (i % HC_MULT) * HC_MULT + j % HC_MULT
                pl.write(comb_raw, [i, j], pl.read(mixes_raw, [t, lane]))
                pl.write(biases, [i, j], pl.read(base, [lane]))
        logits = pl.create_tensor([HC_PAD, HC_PAD], dtype=pl.FP32)
        logits[:, :] = pl.add(pl.mul(pl.mul(comb_raw, inv), s2), biases)
        exp_argument = pl.create_tensor([HC_PAD, HC_PAD], dtype=pl.FP32)
        exp_argument[:, :] = pl.row_expand_sub(logits, pl.row_max(logits))
        exps_value = accurate_exp(exp_argument)
        values = pl.create_tensor([HC_PAD, HC_PAD], dtype=pl.FP32)
        values[:, :] = exps_value
        normalized = sinkhorn_normalize(values, False, 0.0)
        values[:, :] = pl.add(normalized, HC_EPS)
        normalized = sinkhorn_normalize(values, True, HC_EPS)
        values[:, :] = normalized
        for iteration in pl.range(HC_SINKHORN_ITER - 1):
            normalized = sinkhorn_normalize(values, False, HC_EPS)
            values[:, :] = normalized
            normalized = sinkhorn_normalize(values, True, HC_EPS)
            values[:, :] = normalized
        residual_flat[t * HC_MULT : t * HC_MULT + HC_MULT, :] = pl.set_validshape(values, HC_MULT, HC_MULT)
    return pre_mix, post_mix, residual_mix


@pl.jit.inline
def mhc_pre(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
):
    """Collapse the four streams into one sublayer input, one block per token.

    ``sum_m pre_mix[t, m] * x_hc[t, m, :]`` accumulated in FP32 and rounded once, which
    is what the reference does. The mix is a per-token scalar, so it is read rather
    than broadcast from a tile: the four streams of a token are four ``D``-wide rows of
    the flattened stream, and the D loop pipelines over them.

    The stream is BF16 here where DeepSeek-V4 keeps it FP32, so every stream tile is
    widened before the multiply and only the result is rounded back.

    Returns the collapsed rows and their task ID for downstream ordering.
    """
    t_dim = pl.tensor.dim(x_hc, 0)
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    with pl.spmd(t_dim, name_hint="mhc_pre") as pre_tid:
        t = pl.tile.get_block_idx()
        for d0 in pl.pipeline(0, D, MIX_D_TILE, stage=2):
            value = pl.mul(pl.cast(x_flat[t:t + 1, d0:d0 + MIX_D_TILE], target_type=pl.FP32), pl.read(pre_mix, [t, 0]))
            correction = pl.full([1, MIX_D_TILE], dtype=pl.FP32, value=0.0)
            for stream in pl.unroll(HC_MULT - 1):
                head = stream + 1
                tile = x_flat[t : t + 1, head * D + d0 : head * D + d0 + MIX_D_TILE]
                partial = pl.mul(pl.cast(tile, target_type=pl.FP32), pl.read(pre_mix, [t, head]))
                updated = pl.add(value, partial)
                recovered = pl.sub(updated, value)
                error = pl.add(pl.sub(value, pl.sub(updated, recovered)), pl.sub(partial, recovered))
                correction = pl.add(correction, error)
                value = updated
            output[t:t + 1, d0:d0 + MIX_D_TILE] = pl.cast(pl.add(value, correction), target_type=pl.BF16, mode="rint")
    return output, pre_tid


@pl.jit.inline
def mhc_post(
    sublayer: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
):
    """Expand the sublayer result into the streams and mix them, one block per stream.

    ``output[t, j, :] = post_mix[t, j] * sublayer[t, :] + sum_i residual_mix[t, i, j] *
    residual[t, i, :]``. Both mixes are per-token scalars, so a block owns one output
    stream of one token and reads its four coefficients; the D loop pipelines and the
    stream loop unrolls, so the four residual rows are read once per D tile.

    Donor: ``models/deepseek_v4_1_flash/hc_post.py``. What changes is the dtype — the
    GLM stream is BF16. Coefficients, the sublayer product, the residual matrix
    product and the final addition each retain the official BF16 boundaries.

    Returns the mixed streams so the caller can chain them.
    """
    t_dim = pl.tensor.dim(sublayer, 0)
    residual_rounded = pl.create_tensor([t_dim, HC_MULT * HC_MULT], dtype=pl.FP32)
    post_rounded = pl.create_tensor([t_dim, HC_MULT], dtype=pl.FP32)
    residual_mix_flat = pl.reshape(residual_mix, [t_dim, HC_MULT * HC_MULT])
    for t in pl.spmd(t_dim, name_hint="mhc_post_coefficients"):
        post_tile = pl.slice(post_mix, [1, POST_COEFF_TILE], [t, 0], valid_shape=[1, HC_MULT])
        post_bf16 = pl.cast(post_tile, pl.BF16, mode="rint")
        post_rounded[t : t + 1, :] = pl.set_validshape(pl.cast(post_bf16, pl.FP32), 1, HC_MULT)
        comb_tile = residual_mix_flat[t : t + 1, :]
        comb_bf16 = pl.cast(comb_tile, pl.BF16, mode="rint")
        residual_rounded[t : t + 1, :] = pl.cast(comb_bf16, pl.FP32)
    output_flat = pl.reshape(output, [t_dim, HC_DIM])
    residual_flat = pl.reshape(residual, [t_dim, HC_DIM])
    for block in pl.spmd(t_dim * HC_MULT, name_hint="mhc_post"):
        t = block // HC_MULT
        out_h = block % HC_MULT
        for d0 in pl.pipeline(0, D, MIX_D_TILE, stage=2):
            x_tile = pl.cast(sublayer[t : t + 1, d0 : d0 + MIX_D_TILE], target_type=pl.FP32)
            update = pl.cast(pl.mul(x_tile, pl.read(post_rounded, [t, out_h])), pl.BF16, mode="rint")
            value = pl.full([1, MIX_D_TILE], dtype=pl.FP32, value=0.0)
            correction = pl.full([1, MIX_D_TILE], dtype=pl.FP32, value=0.0)
            for in_h in pl.unroll(HC_MULT):
                residual_tile = residual_flat[t : t + 1, in_h * D + d0 : in_h * D + d0 + MIX_D_TILE]
                partial_cast = pl.cast(residual_tile, target_type=pl.FP32)
                partial_read = pl.read(residual_rounded, [t, in_h * HC_MULT + out_h])
                partial = pl.mul(partial_cast, partial_read)
                updated = pl.add(value, partial)
                recovered = pl.sub(updated, value)
                error = pl.add(pl.sub(value, pl.sub(updated, recovered)), pl.sub(partial, recovered))
                correction = pl.add(correction, error)
                value = updated
            residual_bf16 = pl.cast(pl.add(value, correction), pl.BF16, mode="rint")
            combined = pl.add(pl.cast(update, pl.FP32), pl.cast(residual_bf16, pl.FP32))
            output_flat[t:t + 1, out_h * D + d0:out_h * D + d0 + MIX_D_TILE] = pl.cast(combined, target_type=pl.BF16, mode="rint")
    return output


@pl.jit.inline
def mhc_head(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
):
    """Collapse the final streams by an unweighted mean, one block per token."""
    t_dim = pl.tensor.dim(x_hc, 0)
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    for t in pl.spmd(t_dim, name_hint="mhc_head"):
        for d0 in pl.pipeline(0, D, MIX_D_TILE, stage=2):
            value = pl.cast(x_flat[t : t + 1, d0 : d0 + MIX_D_TILE], target_type=pl.FP32)
            correction = pl.full([1, MIX_D_TILE], dtype=pl.FP32, value=0.0)
            for stream in pl.unroll(HC_MULT - 1):
                head = stream + 1
                tile = x_flat[t : t + 1, head * D + d0 : head * D + d0 + MIX_D_TILE]
                partial = pl.cast(tile, target_type=pl.FP32)
                updated = pl.add(value, partial)
                recovered = pl.sub(updated, value)
                error = pl.add(pl.sub(value, pl.sub(updated, recovered)), pl.sub(partial, recovered))
                correction = pl.add(correction, error)
                value = updated
            output_value_add = pl.add(value, correction)
            output_value_mul = pl.mul(output_value_add, 1.0 / HC_MULT)
            output[t:t + 1, d0:d0 + MIX_D_TILE] = pl.cast(output_value_mul, target_type=pl.BF16, mode="rint")
    return output


@pl.jit.inline
def plain_residual_pre(
    streams: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    pre: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
):
    """Read a replicated plain residual stream and clear unused mHC diagnostics."""
    rows = pl.tensor.dim(streams, 0)
    coefficient_elements = rows * HC_MULT
    residual_flat = pl.reshape(residual, [rows, HC_MULT * HC_MULT])
    post_flat = pl.reshape(post, [1, coefficient_elements])
    pre_flat = pl.reshape(pre, [1, coefficient_elements])
    for block in pl.spmd(((rows + 7) // 8) * (D // 128), name_hint="plain_residual_pre"):
        row0 = block // (D // 128) * 8
        col0 = block % (D // 128) * 128
        valid = pl.min(8, rows - row0)
        source = pl.slice(streams, [8, 1, 128], [row0, 0, col0], valid_shape=[valid, 1, 128])
        output[row0 : row0 + 8, col0 : col0 + 128] = pl.reshape(source, [8, 128])
        if col0 == 0:
            zeros = pl.full([1, 8 * HC_MULT], dtype=pl.FP32, value=0.0)
            pre_flat[:, row0 * HC_MULT : (row0 + 8) * HC_MULT] = pl.set_validshape(zeros, 1, valid * HC_MULT)
            post_flat[:, row0 * HC_MULT : (row0 + 8) * HC_MULT] = pl.set_validshape(zeros, 1, valid * HC_MULT)
            residual_zero = pl.full([8, HC_MULT * HC_MULT], dtype=pl.FP32, value=0.0)
            residual_flat[row0 : row0 + 8, :] = pl.set_validshape(residual_zero, valid, HC_MULT * HC_MULT)
    return output


@pl.jit.inline
def plain_residual_post(
    update: pl.Tensor[[T_DYN, D], pl.BF16],
    streams: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    output: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
):
    """Add one BF16 sublayer result to the plain residual and replicate its lanes."""
    rows = pl.tensor.dim(streams, 0)
    for block in pl.spmd(rows * (D // 128), name_hint="plain_residual_post"):
        row = block // (D // 128)
        col0 = block % (D // 128) * 128
        residual = pl.cast(streams[row : row + 1, 0:1, col0 : col0 + 128], pl.FP32)
        residual_row = pl.reshape(residual, [1, 128])
        value = pl.cast(update[row : row + 1, col0 : col0 + 128], pl.FP32)
        combined = pl.cast(pl.add(residual_row, value), pl.BF16, mode="rint")
        expanded = pl.reshape(combined, [1, 1, 128])
        for lane in pl.unroll(HC_MULT):
            output[row : row + 1, lane : lane + 1, col0 : col0 + 128] = expanded
    return output


@pl.jit
def mhc_pre_test(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    """Run one stream collapse for golden.run validation."""
    x_hc.bind_dynamic(0, T_DYN)
    pre_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    mhc_pre(x_hc, pre_mix, output)
    return output


@pl.jit
def mhc_post_test(
    sublayer: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]],
):
    """Run one residual expansion for golden.run validation."""
    sublayer.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    mhc_post(sublayer, residual, post_mix, residual_mix, output)
    return output


@pl.jit
def mhc_head_test(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    """Run one final collapse for golden.run validation."""
    x_hc.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    mhc_head(x_hc, output)
    return output


@pl.jit
def mhc_mixes_test(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    function: pl.Tensor[[MIX_HC, HC_DIM], pl.BF16],
    scale: pl.Tensor[[3], pl.FP32],
    base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    post_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    residual_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32]],
):
    """Run mHC coefficient generation for standalone validation."""
    x_hc.bind_dynamic(0, T_DYN)
    pre_mix.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    mhc_mixes(x_hc, function, scale, base, pre_mix, post_mix, residual_mix)
    return residual_mix


@pl.jit.inline
def seed_hc_streams(
    embedding_rows: pl.Tensor[[T_DYN, D], pl.BF16],
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
):
    """Broadcast each BF16 embedding row into the hyper-connection streams."""
    rows = pl.tensor.dim(embedding_rows, 0)
    x_flat = pl.reshape(x_hc, [rows, HC_DIM])
    with pl.spmd((rows + 15) // 16, name_hint="seed_hc_streams"):
        t0 = pl.tile.get_block_idx() * 16
        for row in pl.unroll(16):
            t = t0 + row
            embedded = pl.tile.load(embedding_rows, [t, 0], [1, D])
            for stream in pl.unroll(HC_MULT):
                pl.tile.store(embedded, [t, stream * D], x_flat)


def hc_pre(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Collapse the four hyper-connection streams into one sublayer input."""
    return (x.float() * pre_mix.float().unsqueeze(-1)).double().sum(dim=-2).float().to(x.dtype)


def hc_seed(inputs_embeds: torch.Tensor, hc_mult: int = FLASH.hc_mult) -> torch.Tensor:
    """Seed the hyper-connection stream by replicating the embedding.

    ``Glm5NextTextModel.forward`` does
    ``inputs_embeds.unsqueeze(2).expand(-1, -1, hc_mult, -1).contiguous()``, so the
    stream inherits the embedding's dtype: **BF16, not FP32**. DeepSeek-V4 carries
    its HC stream in FP32, so the ABIs are not interchangeable.
    """
    return inputs_embeds.unsqueeze(-2).expand(*inputs_embeds.shape[:-1], hc_mult, -1).contiguous()


def hc_mixes(
    x: torch.Tensor,
    function: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    iterations: int = FLASH.hc_sinkhorn_iters,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute mHC mixes with FP32 projections, unaries and reductions."""
    flat = x.flatten(-2).float()
    square_sum = flat.double().square().sum(dim=-1, keepdim=True).float()
    inv = torch.rsqrt((square_sum / flat.shape[-1] + FLASH.rms_norm_eps).double()).float()
    mixes = (flat.double() @ function.double().t()).float() * inv
    hc = FLASH.hc_mult
    (p, q, r) = mixes.split((hc, hc, hc * hc), -1)
    pre_logits = p * scale[0] + base[:hc]
    post_logits = q * scale[1] + base[hc : 2 * hc]
    pre = (1.0 / (1.0 + torch.exp(-pre_logits.double()).float()).double()).float() + FLASH.hc_eps
    post = 2 * (1.0 / (1.0 + torch.exp(-post_logits.double()).float()).double()).float()
    logits = r.unflatten(-1, (hc, hc)) * scale[2] + base[2 * hc :].unflatten(-1, (hc, hc))
    exps = torch.exp((logits - logits.amax(-1, keepdim=True)).double()).float()
    residual = (exps.double() / _sum_fp32_reference(exps).double()).float() + FLASH.hc_eps
    residual = (residual.double() / (_sum_fp32_reference(residual, -2) + FLASH.hc_eps).double()).float()
    for _ in range(iterations - 1):
        residual = (residual.double() / (_sum_fp32_reference(residual) + FLASH.hc_eps).double()).float()
        residual = (residual.double() / (_sum_fp32_reference(residual, -2) + FLASH.hc_eps).double()).float()
    return (pre, post, residual)


def hc_post(
    sublayer: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    residual_mix: torch.Tensor,
) -> torch.Tensor:
    """Expand a sublayer result and mix the four residual streams.

    Preserve the official decoder's stream-dtype coefficient, multiplication,
    matmul-output and addition boundaries.
    """
    dtype = residual.dtype
    post = post_mix.to(dtype).float().unsqueeze(-1)
    update = (post * sublayer.to(dtype).float().unsqueeze(-2)).to(dtype)
    coefficients = residual_mix.to(dtype).double().transpose(-1, -2)
    mixed_residual = (coefficients @ residual.double()).float().to(dtype)
    return (update.float() + mixed_residual.float()).to(dtype)


def hc_head(x: torch.Tensor) -> torch.Tensor:
    """Collapse the final HC streams.

    Unlike DeepSeek-V4, which applies the last layer's delayed pre-mix, GLM-5.3-Flash
    uses an unweighted mean (``Glm5NextTextHyperHead.forward`` is
    ``hidden_streams.mean(dim=2)``).
    """
    return x.double().mean(dim=-2).float().to(torch.bfloat16)


def golden_mhc_pre(x_hc: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    return hc_pre(x_hc, pre_mix).to(torch.bfloat16)


def golden_mhc_post(
    sublayer: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    residual_mix: torch.Tensor,
) -> torch.Tensor:
    return hc_post(sublayer, residual, post_mix, residual_mix).to(torch.bfloat16)


def build_mhc_pre_tensor_specs(tokens: int = 20):
    """Build one deterministic stream collapse at the real hidden size."""
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(97)

    def init_x_hc():
        return torch.randn(tokens, HC_MULT, D, generator=generator).bfloat16()

    def init_pre_mix():
        # The reference mixes are sigmoid outputs, so they are positive and O(1).
        return torch.sigmoid(torch.randn(tokens, HC_MULT, generator=generator)).float()

    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.bfloat16, init_value=init_x_hc),
        TensorSpec("pre_mix", [tokens, HC_MULT], torch.float32, init_value=init_pre_mix),
        TensorSpec("output", [tokens, D], torch.bfloat16),
    ]


def build_mhc_post_tensor_specs(tokens: int = 20):
    """Build one deterministic residual expansion at the real hidden size.

    ``residual_mix`` is drawn doubly stochastic, which is what the 20 Sinkhorn
    iterations in :func:`hc_mixes` produce, so the fixture exercises the
    magnitudes the kernel actually sees.
    """
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(101)

    def init_sublayer():
        return torch.randn(tokens, D, generator=generator).bfloat16()

    def init_residual():
        return torch.randn(tokens, HC_MULT, D, generator=generator).bfloat16()

    def init_post_mix():
        return (2 * torch.sigmoid(torch.randn(tokens, HC_MULT, generator=generator))).float()

    def init_residual_mix():
        mix = torch.softmax(torch.randn(tokens, HC_MULT, HC_MULT, generator=generator), dim=-1)
        for _ in range(8):
            mix = mix / mix.sum(dim=-1, keepdim=True)
            mix = mix / mix.sum(dim=-2, keepdim=True)
        return mix.float()

    return [
        TensorSpec("sublayer", [tokens, D], torch.bfloat16, init_value=init_sublayer),
        TensorSpec("residual", [tokens, HC_MULT, D], torch.bfloat16, init_value=init_residual),
        TensorSpec("post_mix", [tokens, HC_MULT], torch.float32, init_value=init_post_mix),
        TensorSpec("residual_mix", [tokens, HC_MULT, HC_MULT], torch.float32, init_value=init_residual_mix),
        TensorSpec("output", [tokens, HC_MULT, D], torch.bfloat16),
    ]


def build_mhc_head_tensor_specs(tokens: int = 20):
    """Build one deterministic final collapse at the real hidden size."""
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(103)

    def init_x_hc():
        return torch.randn(tokens, HC_MULT, D, generator=generator).bfloat16()

    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.bfloat16, init_value=init_x_hc),
        TensorSpec("output", [tokens, D], torch.bfloat16),
    ]


def golden_mhc_pre_case(tensors):
    """Fill the expected collapse for :func:`build_mhc_pre_tensor_specs`."""
    tensors["output"][:] = golden_mhc_pre(tensors["x_hc"], tensors["pre_mix"])


def golden_mhc_post_case(tensors):
    """Fill the expected streams for :func:`build_mhc_post_tensor_specs`."""
    tensors["output"][:] = golden_mhc_post(
        tensors["sublayer"], tensors["residual"], tensors["post_mix"], tensors["residual_mix"],
    )


def golden_mhc_head_case(tensors):
    """Fill the expected collapse for :func:`build_mhc_head_tensor_specs`."""
    tensors["output"][:] = hc_head(tensors["x_hc"])


def build_mhc_mixes_tensor_specs(tokens: int = 20):
    """Build deterministic inputs and outputs for coefficient validation."""
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(3)

    def init_x_hc():
        return torch.randn(tokens, HC_MULT, D, generator=generator).bfloat16()

    def init_function():
        return (torch.randn(MIX_HC, HC_DIM, generator=generator) / HC_DIM**0.5).bfloat16()

    def init_scale():
        return torch.randn(3, generator=generator)

    def init_base():
        return torch.randn(MIX_HC, generator=generator)

    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.bfloat16, init_value=init_x_hc),
        TensorSpec("function", [MIX_HC, HC_DIM], torch.bfloat16, init_value=init_function),
        TensorSpec("scale", [3], torch.float32, init_value=init_scale),
        TensorSpec("base", [MIX_HC], torch.float32, init_value=init_base),
        TensorSpec("pre_mix", [tokens, HC_MULT], torch.float32),
        TensorSpec("post_mix", [tokens, HC_MULT], torch.float32),
        TensorSpec("residual_mix", [tokens, HC_MULT, HC_MULT], torch.float32),
    ]


def golden_mhc_mixes_case(tensors):
    """Fill the expected coefficient outputs."""
    (pre_mix, post_mix, residual_mix) = hc_mixes(
        tensors["x_hc"], tensors["function"], tensors["scale"], tensors["base"],
    )
    tensors["pre_mix"][:] = pre_mix
    tensors["post_mix"][:] = post_mix
    tensors["residual_mix"][:] = residual_mix


def _check_mhc_replicated_quadrants_equal_four_by_four_sinkhorn():
    g = torch.Generator().manual_seed(41)
    logits = torch.randn(3, 4, 4, generator=g) * 10
    original = logits.softmax(-1) + FLASH.hc_eps
    original = original / (original.sum(-2, keepdim=True) + FLASH.hc_eps)
    wide = logits.repeat(1, 2, 2)
    exp = (wide - wide.amax(-1, keepdim=True)).exp()
    wide = exp / (exp.sum(-1, keepdim=True) * 0.5) + FLASH.hc_eps
    wide = wide / (wide.sum(-2, keepdim=True) * 0.5 + FLASH.hc_eps)
    for _ in range(FLASH.hc_sinkhorn_iters - 1):
        original = original / (original.sum(-1, keepdim=True) + FLASH.hc_eps)
        original = original / (original.sum(-2, keepdim=True) + FLASH.hc_eps)
        wide = wide / (wide.sum(-1, keepdim=True) * 0.5 + FLASH.hc_eps)
        wide = wide / (wide.sum(-2, keepdim=True) * 0.5 + FLASH.hc_eps)
    torch.testing.assert_close(original, wide[:, :4, :4], atol=1e-06, rtol=1e-05)


def _check_mhc_zero_input_coefficients():
    (pre, post, residual) = hc_mixes(
        torch.zeros(2, 4, 4096).bfloat16(), torch.zeros(24, 16384).bfloat16(), torch.ones(3), torch.zeros(24),
    )
    torch.testing.assert_close(pre, torch.full_like(pre, 0.5 + FLASH.hc_eps))
    torch.testing.assert_close(post, torch.ones_like(post))
    torch.testing.assert_close(residual, torch.full_like(residual, 0.25), atol=1e-06, rtol=1e-05)


def run_mhc_boundary_goldens():
    """Check reference equations and boundary cases on CPU."""
    _check_mhc_replicated_quadrants_equal_four_by_four_sinkhorn()
    _check_mhc_zero_input_coefficients()
    print("[GOLDEN] PASS mhc boundary checks")


def run_mhc_goldens(golden_mixes, golden_pre, golden_post, golden_head) -> None:
    """Exercise the four mHC references on one reduced hyper-connection site."""
    torch.manual_seed(11)
    tokens, width = 6, FLASH.hc_mult
    hidden = 64
    hc_dim = width * hidden
    mix_hc = FLASH.mix_hc

    x_hc = torch.randn(tokens, width, hidden, dtype=torch.bfloat16)
    function = torch.randn(mix_hc, hc_dim) * 0.02
    scale = torch.randn(3)
    base = torch.randn(mix_hc) * 0.1

    pre_mix, post_mix, residual_mix = golden_mixes(x_hc, function, scale, base)
    _check(pre_mix.shape == (tokens, width), f"pre_mix shape {tuple(pre_mix.shape)}")
    _check(post_mix.shape == (tokens, width), f"post_mix shape {tuple(post_mix.shape)}")
    _check(residual_mix.shape == (tokens, width, width), f"residual_mix shape {tuple(residual_mix.shape)}")
    # Sinkhorn must leave the residual mix doubly stochastic.
    for axis in (-1, -2):
        deviation = (residual_mix.sum(dim=axis) - 1.0).abs().max()
        _check(deviation < 1e-3, f"residual_mix is not doubly stochastic on axis {axis}: {deviation}")
    _check(bool((pre_mix > 0).all()), "pre_mix must be positive")

    sublayer_input = golden_pre(x_hc, pre_mix)
    _check(sublayer_input.shape == (tokens, hidden), f"mhc_pre shape {tuple(sublayer_input.shape)}")
    _check(sublayer_input.dtype is torch.bfloat16, f"mhc_pre dtype {sublayer_input.dtype}")

    streams = golden_post(sublayer_input, x_hc, post_mix, residual_mix)
    _check(streams.shape == x_hc.shape, f"mhc_post shape {tuple(streams.shape)}")
    _check(streams.dtype is torch.bfloat16, f"mhc_post dtype {streams.dtype}")

    collapsed = golden_head(x_hc)
    _check(collapsed.shape == (tokens, hidden), f"mhc_head shape {tuple(collapsed.shape)}")
    _check(collapsed.dtype is torch.bfloat16, f"mhc_head dtype {collapsed.dtype}")
    # GLM-5.3-Flash collapses with an unweighted mean, not a learned head.
    expected = x_hc.float().mean(dim=-2).to(torch.bfloat16)
    _check(torch.equal(collapsed, expected), "mhc_head must be the unweighted stream mean")

    _report("mhc")


@dataclass
class MhcTensors:
    """One hyper-connection site's mapping parameters (``attn_hc`` or ``ffn_hc``)."""

    fn: torch.Tensor  # [mix_hc, hc_mult * D]
    base: torch.Tensor  # [mix_hc] FP32
    scale: torch.Tensor  # [3] FP32


def mhc_mapping(
    streams: torch.Tensor, weights: MhcTensors, model: Glm53FlashConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """mHC mapping: ``pre``, ``post``, Sinkhorn-``comb``, and the collapsed stream.

    Port of ``Glm5NextTextHyperConnection.forward``. ``comb`` gets one initial
    column normalization then ``iters - 1`` row/column passes; ``pre`` collapses
    the ``hc_mult`` streams into the sub-layer input.
    """
    hc = model.hc_mult
    flat = unweighted_rms_norm(streams.flatten(start_dim=2), model.rms_norm_eps)
    logits = flat @ weights.fn.t()
    pre_w, post_w, comb_w = logits.split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = weights.base.split([hc, hc, hc * hc])
    pre_scale, post_scale, comb_scale = weights.scale.unbind(0)

    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + model.hc_eps
    post = 2 * torch.sigmoid(post_w * post_scale + post_b)
    comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
    comb = torch.softmax(comb_logits, dim=-1) + model.hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + model.hc_eps)
    for _ in range(model.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + model.hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + model.hc_eps)
    collapsed = (pre.unsqueeze(-1) * streams).sum(dim=2)
    return pre, post, comb, collapsed


def combine_streams(
    post: torch.Tensor, comb: torch.Tensor, sublayer_out: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    """Decoder-layer combine: ``post * sublayer_out + comb^T @ residual`` (HF eq.)."""
    return post.unsqueeze(-1) * sublayer_out.unsqueeze(-2) + comb.transpose(-1, -2) @ residual


def synthetic_mhc(model: Glm53FlashConfig, generator: torch.Generator) -> MhcTensors:
    """mHC mapping parameters at the HF init distribution (fn~N(0,0.02), base 0, scale 1)."""
    return MhcTensors(
        fn=_randn((model.mix_hc, model.hc_dim), generator), base=torch.zeros(model.mix_hc),
        scale=torch.ones(3),
    )


def _test_mhc(model: Glm53FlashConfig) -> None:
    torch.manual_seed(11)
    hc, hidden = model.hc_mult, model.hidden_size
    streams = torch.randn(2, 5, hc, hidden) * 0.5
    weights = synthetic_mhc(model, torch.Generator().manual_seed(7))
    pre, post, comb, collapsed = mhc_mapping(streams, weights, model)

    assert pre.shape == (2, 5, hc) and post.shape == (2, 5, hc)
    assert comb.shape == (2, 5, hc, hc) and collapsed.shape == (2, 5, hidden)
    assert float(pre.min()) >= model.hc_eps and float(pre.max()) <= 1.0 + model.hc_eps + 1e-5
    assert 0.0 <= float(post.min()) and float(post.max()) <= 2.0 + 1e-5
    # Sinkhorn: columns are exact by construction (final op normalizes them);
    # rows only converge linearly, and the shipped 20 iterations on a 4x4 leave
    # O(1e-2) row deviation — verified against the verbatim HF snippet.
    row_error = (comb.sum(-1) - 1).abs().max()
    col_error = (comb.sum(-2) - 1).abs().max()
    assert float(row_error) < 0.05 and float(col_error) < 1e-3, (float(row_error), float(col_error))
    _assert_close(collapsed, torch.einsum("bsh,bshd->bsd", pre, streams), "mHC collapse")
    # Combine against a per-stream loop: out_h = post_h * sublayer + sum_k comb[k,h] * res_k.
    sublayer = torch.randn(2, 5, hidden)
    manual = torch.stack(
        [post[..., h, None] * sublayer + sum((comb[..., k, h, None] * streams[..., k, :] for k in range(hc))) for h in range(hc)],
        dim=2,
    )
    _assert_close(combine_streams(post, comb, sublayer, streams), manual, "mHC combine formula")
    print(f"  mHC mapping: shapes, ranges, sinkhorn rows {float(row_error):.2e}")


def main():
    """Prove the goldens on CPU, then validate the three stream scopes on device.

    Each output is one BF16 rounding of an FP32 sum over ``HC_MULT`` = 4 terms, so the
    budget is the rounding plus the accumulation order. Coefficients use an FP32 gate.
    """
    import argparse

    from golden import ratio_allclose, run

    run_mhc_goldens(hc_mixes, golden_mhc_pre, golden_mhc_post, hc_head)

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--case", choices=["all", "mixes", "pre", "post", "head"], default="all")
    parser.add_argument("--golden-only", action="store_true")
    args = parser.parse_args()
    run_mhc_boundary_goldens()
    if args.golden_only:
        return

    compare = ratio_allclose(atol=1e-4, rtol=1.0 / 128)
    cases = (
        ("mixes", mhc_mixes_test, build_mhc_mixes_tensor_specs(args.tokens), golden_mhc_mixes_case),
        ("pre", mhc_pre_test, build_mhc_pre_tensor_specs(args.tokens), golden_mhc_pre_case),
        ("post", mhc_post_test, build_mhc_post_tensor_specs(args.tokens), golden_mhc_post_case),
        ("head", mhc_head_test, build_mhc_head_tensor_specs(args.tokens), golden_mhc_head_case),
    )
    for name, fn, specs, golden_fn in cases:
        if args.case not in ("all", name):
            continue
        result = run(
            fn=fn,
            specs=specs,
            golden_fn=golden_fn,
            config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
            rtol=1.0 / 128,
            atol=0.0001,
            compare_fn={key: ratio_allclose(atol=1e-05, rtol=0.001) for key in ("pre_mix", "post_mix", "residual_mix")} if name == "mixes" else {"output": compare},
            compile_only=args.compile_only,
        )
        print(f"{name}: {result}")
        if not result.passed:
            raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()
