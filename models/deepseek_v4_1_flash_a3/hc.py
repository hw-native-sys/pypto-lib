# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 HC coefficient generation, stream collapse, residual expansion, and final collapse."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

import math

import pypto.language as pl
import torch
import torch.nn.functional as F

from .activation_quant import (
    DECODE_QUANT_REDUCE_TILE, _decode_checkpoint_codes, _quantize_projection_activation_into,
)
from .config import FLASH as M
from .dsl_types import SP_T_DYN, T_DYN


# model config
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
NORM_EPS = M.rms_norm_eps

# tiling
MIX_PAD = 32
HC_PAD = 8
T_TILE = 8
LINEAR_T_TILE = 16
SINKHORN_LOAD_T_TILE = 8
SINKHORN_T_TILE = 2
SINKHORN_WORKER_TILE = 24
RMS_K_TILE = 1024
LINEAR_K_TILE = 256
LINEAR_OK = 4
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK
D_TILE = 2560
PRE_T_TILE = 2
HEAD_D_TILE = 5120
PRE_NORM_T_TILE = 8
PRE_NORM_D_TILE = 256
POST_WORKER_TILE = 64
# Eight FP32 row sums fill one 32-byte reduction output.
PREFILL_RMS_D_TILE = D // 2
PREFILL_COLLAPSE_ROWS = D // PREFILL_RMS_D_TILE
PREFILL_RMS_ROWS = HC_DIM // PREFILL_RMS_D_TILE
PREFILL_RMS_T_TILE = 8
PREFILL_PROJ_T_TILE = 32
PREFILL_PROJ_N_TILE = 32
PREFILL_PROJ_K_TILE = 1024
PREFILL_SINKHORN_T_TILE = 16
PREFILL_SINKHORN_WIDE_T_TILE = 64
PREFILL_SINKHORN_LANE_BITS = (PREFILL_SINKHORN_T_TILE * HC_MULT - 1).bit_length()
PREFILL_SINKHORN_WIDE_LANE_BITS = (PREFILL_SINKHORN_WIDE_T_TILE * HC_MULT - 1).bit_length()
PREFILL_SINKHORN_STREAM_BITS = (HC_MULT - 1).bit_length()
PREFILL_WORKER_TILE = 48
PREFILL_MERGE_T_TILE = 16
PREFILL_POST_D_TILE = 2560
PREFILL_POST_WORKERS = 48
DECODE_RMS_K_TILE = 2048
DECODE_RMS_ROW_WIDTH = 2560
DECODE_RMS_ROWS = HC_DIM // DECODE_RMS_ROW_WIDTH
DECODE_LINEAR_OK = 8
DECODE_LINEAR_K_PER_SPLIT = HC_DIM // DECODE_LINEAR_OK
DECODE_NORM_T_TILE = 8
DECODE_NORM_D_TILE = 512
DECODE_POST_STREAM_TILE = 4
POST_VECTOR_COLUMNS = pl.dynamic("A3_HC_POST_COLUMNS")
DECODE_POST_D_TILE = 1280
DECODE_POST_D_TILE_WIDE = 2560


@pl.jit.inline
def hc_sinkhorn(
    inv_rms: pl.Tensor,
    partials: pl.Tensor,
    scale: pl.Tensor,
    base: pl.Tensor,
    pre_mix: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    one_token: pl.constexpr = False,
):
    """Normalize independent HC matrices in two-row tiles."""
    tokens = pl.tensor.dim(pre_mix, 0)
    linear_tokens = pl.tensor.dim(inv_rms, 0)
    base_row = pl.reshape(base, [1, MIX_HC])
    residual_flat = pl.reshape(residual_mix, [tokens, HC_MULT * HC_MULT])
    pre_flat = pl.reshape(pre_mix, [1, tokens * HC_MULT])
    post_flat = pl.reshape(post_mix, [1, tokens * HC_MULT])
    scale0 = pl.read(scale, [0])
    scale1 = pl.read(scale, [1])
    scale2 = pl.read(scale, [2])
    blocks = (tokens + SINKHORN_T_TILE - 1) // SINKHORN_T_TILE
    if one_token:
        blocks = 1
    workers = pl.min(pl.min(blocks, SINKHORN_WORKER_TILE), pl.system.available_aiv_count())
    with pl.spmd(workers, name_hint="mhc_sinkhorn") as ready:
        worker = pl.tile.get_block_idx()
        lanes = pl.tile.arange(0, [1, HC_PAD * HC_PAD], dtype=pl.INT32)
        lane_values = pl.cast(lanes, pl.FP32)
        token = pl.cast(pl.shrs(lanes, 5), pl.FP32)
        wide_row = pl.cast(pl.shrs(lanes, 3), pl.FP32)
        row = pl.sub(wide_row, pl.mul(token, 4.0))
        groups = pl.cast(pl.shrs(lanes, 2), pl.FP32)
        column = pl.sub(lane_values, pl.mul(groups, 4.0))
        token_offset = pl.mul(token, 32.0)
        row_offset = pl.mul(row, 4.0)
        indices = pl.add(token_offset, row_offset)
        indices = pl.add(indices, column)
        indices = pl.add(indices, 8.0)
        gather_indices = pl.cast(indices, pl.INT32, mode="rint")
        gather_tmp = pl.create_tile([1, HC_PAD * HC_PAD], dtype=pl.INT32)
        column_token = pl.sub(groups, pl.mul(wide_row, 2.0))
        different_token = pl.abs(pl.sub(token, column_token))
        diagonal = pl.add(pl.mul(different_token, -1.0), 1.0)
        diagonal = pl.reshape(diagonal, [HC_PAD, HC_PAD])
        output_lanes = pl.tile.arange(0, [1, 2 * HC_MULT * HC_MULT], dtype=pl.INT32)
        output_lane_values = pl.cast(output_lanes, pl.FP32)
        output_token = pl.cast(pl.shrs(output_lanes, 4), pl.FP32)
        output_groups = pl.cast(pl.shrs(output_lanes, 2), pl.FP32)
        output_row = pl.sub(output_groups, pl.mul(output_token, 4.0))
        output_column = pl.sub(output_lane_values, pl.mul(output_groups, 4.0))
        output_token_offset = pl.mul(output_token, 36.0)
        output_row_offset = pl.mul(output_row, 8.0)
        output_indices = pl.add(output_token_offset, output_row_offset)
        output_indices = pl.add(output_indices, output_column)
        output_gather_indices = pl.cast(output_indices, pl.INT32, mode="rint")
        output_tmp = pl.create_tile([1, 2 * HC_MULT * HC_MULT], dtype=pl.INT32)
        coefficient_lanes = pl.tile.arange(0, [1, SINKHORN_T_TILE * HC_MULT], dtype=pl.INT32)
        coefficient_token = pl.shrs(coefficient_lanes, 2)
        coefficient_token = pl.cast(coefficient_token, pl.FP32)
        coefficient_values = pl.cast(coefficient_lanes, pl.FP32)
        coefficient_offsets = pl.mul(coefficient_token, 1.0 * (MIX_PAD - HC_MULT))
        pre_indices = pl.add(coefficient_values, coefficient_offsets)
        post_indices = pl.add(pre_indices, 1.0 * HC_MULT)
        pre_indices = pl.cast(pre_indices, pl.INT32, mode="rint")
        post_indices = pl.cast(post_indices, pl.INT32, mode="rint")
        coefficient_tmp = pl.create_tile([1, SINKHORN_T_TILE * HC_MULT], dtype=pl.INT32)
        for block in pl.range(worker, blocks, workers):
            t0 = block * SINKHORN_T_TILE
            valid_rows = pl.min(SINKHORN_T_TILE, tokens - t0)
            if one_token:
                valid_rows = 1
            load_t0 = t0 // SINKHORN_LOAD_T_TILE * SINKHORN_LOAD_T_TILE
            local_t0 = t0 - load_t0
            total = pl.tile.full([SINKHORN_T_TILE, MIX_PAD], dtype=pl.FP32, value=0.0)
            for split in pl.unroll(LINEAR_OK):
                offset = split * linear_tokens + t0
                part = pl.load(partials, [offset, 0], [SINKHORN_T_TILE, MIX_PAD])
                total = pl.add(total, part)
            inverse_group = pl.load(inv_rms, [load_t0, 0], [SINKHORN_LOAD_T_TILE, 1])
            inverse0 = pl.tile.read(inverse_group, [local_t0, 0])
            inverse1 = pl.tile.read(inverse_group, [local_t0 + 1, 0])
            row0 = pl.tile.slice(total, [1, MIX_PAD], [0, 0])
            row1 = pl.tile.slice(total, [1, MIX_PAD], [1, 0])
            normalized0 = pl.mul(row0, inverse0)
            normalized1 = pl.mul(row1, inverse1)
            normalized = pl.create_tile([SINKHORN_T_TILE, MIX_PAD], dtype=pl.FP32)
            normalized = pl.tile.assemble(normalized, normalized0, [0, 0])
            normalized = pl.tile.assemble(normalized, normalized1, [1, 0])
            bias = pl.load(base_row, [0, 0], [1, MIX_PAD], valid_shape=[1, MIX_HC])
            bias = pl.fillpad(bias, pad_value=pl.PadValue.zero)
            pre_scaled = pl.mul(normalized, scale0)
            pre_logits = pl.col_expand_add(pre_scaled, bias)
            pre_logits_flat = pl.reshape(pre_logits, [1, SINKHORN_T_TILE * MIX_PAD])
            pre_logits = pl.tile.gather(pre_logits_flat, pre_indices, coefficient_tmp)
            pre_negated = pl.neg(pre_logits)
            pre_exp = pl.exp(pre_negated)
            pre_denominator = pl.add(pre_exp, 1.0)
            pre = pl.recip(pre_denominator)
            pre = pl.add(pre, HC_EPS)
            pre = pl.set_validshape(pre, 1, valid_rows * HC_MULT)
            pl.store(pre, [0, t0 * HC_MULT], pre_flat)
            post_scaled = pl.mul(normalized, scale1)
            post_logits = pl.col_expand_add(post_scaled, bias)
            post_logits_flat = pl.reshape(post_logits, [1, SINKHORN_T_TILE * MIX_PAD])
            post_logits = pl.tile.gather(post_logits_flat, post_indices, coefficient_tmp)
            post_negated = pl.neg(post_logits)
            post_exp = pl.exp(post_negated)
            post_denominator = pl.add(post_exp, 1.0)
            post = pl.recip(post_denominator)
            post = pl.mul(post, 2.0)
            post = pl.set_validshape(post, 1, valid_rows * HC_MULT)
            pl.store(post, [0, t0 * HC_MULT], post_flat)
            comb_scaled = pl.mul(normalized, scale2)
            comb_logits = pl.col_expand_add(comb_scaled, bias)
            comb_logits = pl.slice(comb_logits, [SINKHORN_T_TILE, MIX_PAD], [0, 0])
            comb_logits = pl.mul(comb_logits, 1.0)
            comb_logits = pl.set_validshape(comb_logits, valid_rows, MIX_HC)
            comb_logits = pl.fillpad(comb_logits, pad_value=pl.PadValue.zero)
            comb_logits = pl.set_validshape(comb_logits, SINKHORN_T_TILE, MIX_PAD)
            flat_logits = pl.reshape(comb_logits, [1, SINKHORN_T_TILE * MIX_PAD])
            gathered = pl.tile.gather(flat_logits, gather_indices, gather_tmp)
            rows = pl.reshape(gathered, [HC_PAD, HC_PAD])
            row_tmp = pl.create_tile([HC_PAD, HC_PAD], dtype=pl.FP32)
            maximum = pl.row_max(rows, row_tmp)
            rows = pl.row_expand_sub(rows, maximum)
            rows = pl.exp(rows)
            rows = pl.mul(rows, diagonal)
            row_sum = pl.row_sum(rows, row_tmp)
            rows = pl.row_expand_div(rows, row_sum)
            rows = pl.add(rows, HC_EPS)
            rows = pl.mul(rows, diagonal)
            column_sum = pl.col_sum(rows)
            column_sum = pl.add(column_sum, HC_EPS)
            rows = pl.col_expand_div(rows, column_sum)
            for _ in pl.unroll(HC_SINKHORN_ITER - 1):
                row_sum = pl.row_sum(rows, row_tmp)
                row_sum = pl.add(row_sum, HC_EPS)
                rows = pl.row_expand_div(rows, row_sum)
                column_sum = pl.col_sum(rows)
                column_sum = pl.add(column_sum, HC_EPS)
                rows = pl.col_expand_div(rows, column_sum)
            flat_rows = pl.reshape(rows, [1, HC_PAD * HC_PAD])
            packed = pl.tile.gather(flat_rows, output_gather_indices, output_tmp)
            output = pl.reshape(packed, [SINKHORN_T_TILE, HC_MULT * HC_MULT])
            output = pl.set_validshape(output, valid_rows, HC_MULT * HC_MULT)
            pl.store(output, [t0, 0], residual_flat)
    return ready


@pl.jit.inline
def hc_sinkhorn_decode(
    inv_rms: pl.Tensor,
    partials: pl.Tensor,
    scale: pl.Tensor,
    base: pl.Tensor,
    pre_mix: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    one_token: pl.constexpr = False,
):
    """Normalize independent HC matrices in two-row tiles."""
    tokens = pl.tensor.dim(pre_mix, 0)
    linear_tokens = pl.tensor.dim(inv_rms, 0)
    base_row = pl.reshape(base, [1, MIX_HC])
    residual_flat = pl.reshape(residual_mix, [tokens, HC_MULT * HC_MULT])
    pre_flat = pl.reshape(pre_mix, [1, tokens * HC_MULT])
    post_flat = pl.reshape(post_mix, [1, tokens * HC_MULT])
    scale0 = pl.read(scale, [0])
    scale1 = pl.read(scale, [1])
    scale2 = pl.read(scale, [2])
    blocks = (tokens + SINKHORN_T_TILE - 1) // SINKHORN_T_TILE
    if one_token:
        blocks = 1
    workers = pl.min(pl.min(blocks, SINKHORN_WORKER_TILE), pl.system.available_aiv_count())
    with pl.spmd(workers, name_hint="mhc_sinkhorn") as ready:
        worker = pl.tile.get_block_idx()
        lanes = pl.tile.arange(0, [1, HC_PAD * HC_PAD], dtype=pl.INT32)
        lane_values = pl.cast(lanes, pl.FP32)
        token = pl.cast(pl.shrs(lanes, 5), pl.FP32)
        wide_row = pl.cast(pl.shrs(lanes, 3), pl.FP32)
        row = pl.sub(wide_row, pl.mul(token, 4.0))
        groups = pl.cast(pl.shrs(lanes, 2), pl.FP32)
        column = pl.sub(lane_values, pl.mul(groups, 4.0))
        token_offset = pl.mul(token, 32.0)
        row_offset = pl.mul(row, 4.0)
        indices = pl.add(token_offset, row_offset)
        indices = pl.add(indices, column)
        indices = pl.add(indices, 8.0)
        gather_indices = pl.cast(indices, pl.INT32, mode="rint")
        gather_tmp = pl.create_tile([1, HC_PAD * HC_PAD], dtype=pl.INT32)
        column_token = pl.sub(groups, pl.mul(wide_row, 2.0))
        different_token = pl.abs(pl.sub(token, column_token))
        diagonal = pl.add(pl.mul(different_token, -1.0), 1.0)
        diagonal = pl.reshape(diagonal, [HC_PAD, HC_PAD])
        output_lanes = pl.tile.arange(0, [1, 2 * HC_MULT * HC_MULT], dtype=pl.INT32)
        output_lane_values = pl.cast(output_lanes, pl.FP32)
        output_token = pl.cast(pl.shrs(output_lanes, 4), pl.FP32)
        output_groups = pl.cast(pl.shrs(output_lanes, 2), pl.FP32)
        output_row = pl.sub(output_groups, pl.mul(output_token, 4.0))
        output_column = pl.sub(output_lane_values, pl.mul(output_groups, 4.0))
        output_token_offset = pl.mul(output_token, 36.0)
        output_row_offset = pl.mul(output_row, 8.0)
        output_indices = pl.add(output_token_offset, output_row_offset)
        output_indices = pl.add(output_indices, output_column)
        output_gather_indices = pl.cast(output_indices, pl.INT32, mode="rint")
        output_tmp = pl.create_tile([1, 2 * HC_MULT * HC_MULT], dtype=pl.INT32)
        coefficient_lanes = pl.tile.arange(0, [1, SINKHORN_T_TILE * HC_MULT], dtype=pl.INT32)
        coefficient_token = pl.shrs(coefficient_lanes, 2)
        coefficient_token = pl.cast(coefficient_token, pl.FP32)
        coefficient_values = pl.cast(coefficient_lanes, pl.FP32)
        coefficient_offsets = pl.mul(coefficient_token, 1.0 * (MIX_PAD - HC_MULT))
        pre_indices = pl.add(coefficient_values, coefficient_offsets)
        post_indices = pl.add(pre_indices, 1.0 * HC_MULT)
        pre_indices = pl.cast(pre_indices, pl.INT32, mode="rint")
        post_indices = pl.cast(post_indices, pl.INT32, mode="rint")
        coefficient_tmp = pl.create_tile([1, SINKHORN_T_TILE * HC_MULT], dtype=pl.INT32)
        for block in pl.range(worker, blocks, workers):
            t0 = block * SINKHORN_T_TILE
            valid_rows = pl.min(SINKHORN_T_TILE, tokens - t0)
            if one_token:
                valid_rows = 1
            load_t0 = t0 // SINKHORN_LOAD_T_TILE * SINKHORN_LOAD_T_TILE
            local_t0 = t0 - load_t0
            total = pl.tile.full([SINKHORN_T_TILE, MIX_PAD], dtype=pl.FP32, value=0.0)
            for split in pl.range(DECODE_LINEAR_OK):
                offset = split * linear_tokens + t0
                part = pl.load(partials, [offset, 0], [SINKHORN_T_TILE, MIX_PAD])
                total = pl.add(total, part)
            inverse_group = pl.load(inv_rms, [load_t0, 0], [SINKHORN_LOAD_T_TILE, 1])
            inverse0 = pl.tile.read(inverse_group, [local_t0, 0])
            inverse1 = pl.tile.read(inverse_group, [local_t0 + 1, 0])
            row0 = pl.tile.slice(total, [1, MIX_PAD], [0, 0])
            row1 = pl.tile.slice(total, [1, MIX_PAD], [1, 0])
            normalized0 = pl.mul(row0, inverse0)
            normalized1 = pl.mul(row1, inverse1)
            normalized = pl.create_tile([SINKHORN_T_TILE, MIX_PAD], dtype=pl.FP32)
            normalized = pl.tile.assemble(normalized, normalized0, [0, 0])
            normalized = pl.tile.assemble(normalized, normalized1, [1, 0])
            bias = pl.load(base_row, [0, 0], [1, MIX_PAD], valid_shape=[1, MIX_HC])
            bias = pl.fillpad(bias, pad_value=pl.PadValue.zero)
            pre_scaled = pl.mul(normalized, scale0)
            pre_logits = pl.col_expand_add(pre_scaled, bias)
            pre_logits_flat = pl.reshape(pre_logits, [1, SINKHORN_T_TILE * MIX_PAD])
            pre_logits = pl.tile.gather(pre_logits_flat, pre_indices, coefficient_tmp)
            pre_negated = pl.neg(pre_logits)
            pre_exp = pl.exp(pre_negated)
            pre_denominator = pl.add(pre_exp, 1.0)
            pre = pl.recip(pre_denominator)
            pre = pl.add(pre, HC_EPS)
            pre = pl.set_validshape(pre, 1, valid_rows * HC_MULT)
            pl.store(pre, [0, t0 * HC_MULT], pre_flat)
            post_scaled = pl.mul(normalized, scale1)
            post_logits = pl.col_expand_add(post_scaled, bias)
            post_logits_flat = pl.reshape(post_logits, [1, SINKHORN_T_TILE * MIX_PAD])
            post_logits = pl.tile.gather(post_logits_flat, post_indices, coefficient_tmp)
            post_negated = pl.neg(post_logits)
            post_exp = pl.exp(post_negated)
            post_denominator = pl.add(post_exp, 1.0)
            post = pl.recip(post_denominator)
            post = pl.mul(post, 2.0)
            post = pl.set_validshape(post, 1, valid_rows * HC_MULT)
            pl.store(post, [0, t0 * HC_MULT], post_flat)
            comb_scaled = pl.mul(normalized, scale2)
            comb_logits = pl.col_expand_add(comb_scaled, bias)
            comb_logits = pl.slice(comb_logits, [SINKHORN_T_TILE, MIX_PAD], [0, 0])
            comb_logits = pl.mul(comb_logits, 1.0)
            comb_logits = pl.set_validshape(comb_logits, valid_rows, MIX_HC)
            comb_logits = pl.fillpad(comb_logits, pad_value=pl.PadValue.zero)
            comb_logits = pl.set_validshape(comb_logits, SINKHORN_T_TILE, MIX_PAD)
            flat_logits = pl.reshape(comb_logits, [1, SINKHORN_T_TILE * MIX_PAD])
            gathered = pl.tile.gather(flat_logits, gather_indices, gather_tmp)
            rows = pl.reshape(gathered, [HC_PAD, HC_PAD])
            row_tmp = pl.create_tile([HC_PAD, HC_PAD], dtype=pl.FP32)
            maximum = pl.row_max(rows, row_tmp)
            rows = pl.row_expand_sub(rows, maximum)
            rows = pl.exp(rows)
            rows = pl.mul(rows, diagonal)
            row_sum = pl.row_sum(rows, row_tmp)
            rows = pl.row_expand_div(rows, row_sum)
            rows = pl.add(rows, HC_EPS)
            rows = pl.mul(rows, diagonal)
            column_sum = pl.col_sum(rows)
            column_sum = pl.add(column_sum, HC_EPS)
            rows = pl.col_expand_div(rows, column_sum)
            for _ in pl.unroll(HC_SINKHORN_ITER - 1):
                row_sum = pl.row_sum(rows, row_tmp)
                row_sum = pl.add(row_sum, HC_EPS)
                rows = pl.row_expand_div(rows, row_sum)
                column_sum = pl.col_sum(rows)
                column_sum = pl.add(column_sum, HC_EPS)
                rows = pl.col_expand_div(rows, column_sum)
            flat_rows = pl.reshape(rows, [1, HC_PAD * HC_PAD])
            packed = pl.tile.gather(flat_rows, output_gather_indices, output_tmp)
            output = pl.reshape(packed, [SINKHORN_T_TILE, HC_MULT * HC_MULT])
            output = pl.set_validshape(output, valid_rows, HC_MULT * HC_MULT)
            pl.store(output, [t0, 0], residual_flat)
    return ready


@pl.jit.inline
def hc_sinkhorn_decode_token(
    inv_rms: pl.Tensor,
    partials: pl.Tensor,
    scale: pl.Tensor,
    base: pl.Tensor,
    pre_mix: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
):
    """Normalize a single Decode token's residual mixing matrix."""
    tokens = pl.tensor.dim(pre_mix, 0)
    linear_tokens = pl.tensor.dim(inv_rms, 0)
    bias_row = pl.reshape(base, [1, MIX_HC])
    residual_flat = pl.reshape(residual_mix, [tokens, HC_MULT * HC_MULT])
    with pl.spmd(1, name_hint="mhc_sinkhorn") as ready:
        t = pl.tile.get_block_idx()
        total = pl.tile.full([1, MIX_PAD], dtype=pl.FP32, value=0.0)
        for split in pl.unroll(DECODE_LINEAR_OK):
            part = pl.load(partials, [split * linear_tokens + t, 0], [1, MIX_PAD])
            total = pl.add(total, part)
        inv_group = pl.load(inv_rms, [0, 0], [SINKHORN_LOAD_T_TILE, 1])
        normalized = pl.mul(total, pl.tile.read(inv_group, [0, 0]))
        bias = pl.load(bias_row, [0, 0], [1, MIX_PAD], valid_shape=[1, MIX_HC])
        bias = pl.fillpad(bias, pad_value=pl.PadValue.zero)
        coeff_lanes = pl.tile.arange(0, [1, HC_PAD], dtype=pl.INT32)
        coeff_tmp = pl.create_tile([1, HC_PAD], dtype=pl.INT32)
        pre_logits = pl.add(pl.mul(normalized, pl.read(scale, [0])), bias)
        pre_logits = pl.tile.gather(pre_logits, coeff_lanes, coeff_tmp)
        post_logits = pl.add(pl.mul(normalized, pl.read(scale, [1])), bias)
        post_logits = pl.tile.gather(post_logits, pl.add(coeff_lanes, HC_MULT), coeff_tmp)
        pre = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0)), HC_EPS)
        post = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0)), 2.0)
        pl.store(pl.set_validshape(pre, 1, HC_MULT), [t, 0], pre_mix)
        pl.store(pl.set_validshape(post, 1, HC_MULT), [t, 0], post_mix)

        # Four rows are replicated across the eight aligned FP32 lanes.
        lanes = pl.tile.arange(0, [1, HC_MULT * HC_PAD], dtype=pl.INT32)
        axis = pl.shrs(lanes, 3)
        group = pl.shrs(lanes, 2)
        stream = pl.sub(lanes, pl.shls(group, 2))
        pack_index = pl.add(pl.add(pl.shls(stream, 2), axis), 2 * HC_MULT)
        transpose_index = pl.add(pl.shls(stream, 3), axis)
        gather_tmp = pl.create_tile([1, HC_MULT * HC_PAD], dtype=pl.INT32)
        logits = pl.add(pl.mul(normalized, pl.read(scale, [2])), bias)
        packed = pl.tile.gather(logits, pack_index, gather_tmp)
        packed = pl.reshape(packed, [HC_MULT, HC_PAD])
        maximum = pl.col_max(packed)
        values = pl.exp(pl.col_expand_sub(packed, maximum))
        denom = pl.col_sum(values)
        probability = pl.add(pl.col_expand_div(values, denom), HC_EPS)
        for _ in pl.unroll(2 * HC_SINKHORN_ITER - 1):
            flat = pl.reshape(probability, [1, HC_MULT * HC_PAD])
            swapped = pl.tile.gather(flat, transpose_index, gather_tmp)
            swapped = pl.reshape(swapped, [HC_MULT, HC_PAD])
            denom = pl.add(pl.col_sum(swapped), HC_EPS)
            probability = pl.col_expand_div(swapped, denom)
        final_values = pl.reshape(probability, [1, HC_MULT * HC_PAD])
        out_lanes = pl.tile.arange(0, [1, HC_MULT * HC_MULT], dtype=pl.INT32)
        out_index = pl.add(out_lanes, pl.shls(pl.shrs(out_lanes, 2), 2))
        out_tmp = pl.create_tile([1, HC_MULT * HC_MULT], dtype=pl.INT32)
        packed_output = pl.tile.gather(final_values, out_index, out_tmp)
        pl.store(packed_output, [t, 0], residual_flat)
    return ready


@pl.jit.inline
def hc_decode_token_rms(
    x_hc: pl.Tensor,
    t_dim: pl.Scalar[pl.INDEX],
    inv_rms: pl.Tensor,
):
    """Compute the FP32 residual RMS for the short Decode paths."""
    # Partition each token into vector-length rows and keep its reduction local.
    rms_view = pl.reshape(x_hc, [t_dim * DECODE_RMS_ROWS, DECODE_RMS_ROW_WIDTH])
    rms_tiles = t_dim
    rms_workers = pl.max(1, pl.min(rms_tiles, pl.system.available_aiv_count()))
    for rms_worker in pl.spmd(rms_workers, name_hint="mhc_rms", allow_early_resolve=True):
        for token in pl.range(rms_worker, rms_tiles, rms_workers):
            sums = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
            scratch = pl.create_tile([DECODE_RMS_ROWS, DECODE_RMS_ROW_WIDTH], dtype=pl.FP32)
            values = pl.load(
                rms_view, [token * DECODE_RMS_ROWS, 0], [DECODE_RMS_ROWS, DECODE_RMS_ROW_WIDTH],
            )
            squares = pl.mul(values, values)
            rms_part = pl.reshape(pl.row_sum(squares, tmp_tile=scratch), [1, 8])
            sums = pl.add(sums, rms_part)
            total = pl.tile.read(sums, [0, 0])
            for lane in pl.unroll(1, 8):
                total = total + pl.tile.read(sums, [0, lane])
            padded = pl.add(pl.tile.full([1, 8], dtype=pl.FP32, value=0.0), total)
            inverse_tmp = pl.create_tile([1, 8], dtype=pl.FP32)
            inverse = pl.tile.rsqrt(pl.add(pl.mul(padded, HC_DIM_INV), NORM_EPS), inverse_tmp, high_precision=True)
            inverse = pl.reshape(inverse, [8, 1])
            pl.store(pl.set_validshape(inverse, 1, 1), [token, 0], inv_rms)


@pl.jit.inline
def hc_short_split_linear(x_hc: pl.Tensor, function: pl.Tensor, first: pl.constexpr, valid_columns: pl.constexpr, splits: pl.constexpr, k_tile: pl.constexpr):
    t_dim = pl.tensor.dim(x_hc, 0)
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    projected = pl.create_tensor([splits, 16], dtype=pl.FP32)
    linear_tiles = splits
    linear_workers = pl.max(1, pl.min(linear_tiles, pl.system.available_cluster_count()))
    with pl.spmd(linear_workers, name_hint="mhc_linear_part", allow_early_resolve=True):
        linear_worker = pl.tile.get_block_idx()
        for worker in pl.range(linear_worker, linear_tiles, linear_workers):
            k0 = worker * (HC_DIM // splits)
            acc = pl.create_tile([LINEAR_T_TILE, 16], dtype=pl.FP32)
            for chunk in pl.pipeline(HC_DIM // splits // k_tile, stage=2):
                col = k0 + chunk * k_tile
                x = pl.load(x_flat, [0, col], [LINEAR_T_TILE, k_tile], valid_shape=[1, k_tile], target_memory=pl.Mem.Mat)
                w = pl.load(function, [first, col], [16, k_tile], valid_shape=[valid_columns, k_tile], target_memory=pl.Mem.Mat)
                acc = pl.matmul_acc(acc, x, pl.tile.transpose_view(w), init_cond=(chunk == 0))
            pl.store(pl.set_validshape(acc, 1, 16), [worker, 0], projected)
    return projected


@pl.jit.inline
def hc_short_split_coefficients(inv_rms: pl.Tensor, partials: pl.Tensor, scale: pl.Tensor, base: pl.Tensor,
                               pre_mix: pl.Tensor, post_mix: pl.Tensor, residual_mix: pl.Tensor,
                               residual_path: pl.constexpr, first: pl.constexpr, valid_columns: pl.constexpr, splits: pl.constexpr):
    tokens = pl.tensor.dim(pre_mix, 0)
    bias_row = pl.reshape(base, [1, MIX_HC])
    residual_flat = pl.reshape(residual_mix, [tokens, HC_MULT * HC_MULT])
    with pl.spmd(1, name_hint="mhc_mix_part"):
        t = pl.tile.get_block_idx()
        block = pl.load(partials, [0, 0], [splits, 16])
        total = pl.col_sum(block)
        inv_group = pl.load(inv_rms, [0, 0], [SINKHORN_LOAD_T_TILE, 1])
        normalized = pl.mul(total, pl.tile.read(inv_group, [0, 0]))
        bias = pl.load(bias_row, [0, first], [1, 16], valid_shape=[1, valid_columns])
        bias = pl.fillpad(bias, pad_value=pl.PadValue.zero)
        if residual_path:
            # Four rows are replicated across the eight aligned FP32 lanes.
            lanes = pl.tile.arange(0, [1, HC_MULT * HC_PAD], dtype=pl.INT32)
            axis = pl.shrs(lanes, 3)
            group = pl.shrs(lanes, 2)
            stream = pl.sub(lanes, pl.shls(group, 2))
            pack_index = pl.add(pl.shls(stream, 2), axis)
            transpose_index = pl.add(pl.shls(stream, 3), axis)
            gather_tmp = pl.create_tile([1, HC_MULT * HC_PAD], dtype=pl.INT32)
            logits = pl.add(pl.mul(normalized, pl.read(scale, [2])), bias)
            packed = pl.tile.gather(logits, pack_index, gather_tmp)
            packed = pl.reshape(packed, [HC_MULT, HC_PAD])
            maximum = pl.col_max(packed)
            values = pl.exp(pl.col_expand_sub(packed, maximum))
            denom = pl.col_sum(values)
            probability = pl.add(pl.col_expand_div(values, denom), HC_EPS)
            for _ in pl.unroll(2 * HC_SINKHORN_ITER - 1):
                flat = pl.reshape(probability, [1, HC_MULT * HC_PAD])
                swapped = pl.tile.gather(flat, transpose_index, gather_tmp)
                swapped = pl.reshape(swapped, [HC_MULT, HC_PAD])
                denom = pl.add(pl.col_sum(swapped), HC_EPS)
                probability = pl.col_expand_div(swapped, denom)
            final_values = pl.reshape(probability, [1, HC_MULT * HC_PAD])
            out_lanes = pl.tile.arange(0, [1, HC_MULT * HC_MULT], dtype=pl.INT32)
            out_index = pl.add(out_lanes, pl.shls(pl.shrs(out_lanes, 2), 2))
            out_tmp = pl.create_tile([1, HC_MULT * HC_MULT], dtype=pl.INT32)
            packed_output = pl.tile.gather(final_values, out_index, out_tmp)
            pl.store(packed_output, [t, 0], residual_flat)
        else:
            coeff_lanes = pl.tile.arange(0, [1, HC_PAD], dtype=pl.INT32)
            coeff_tmp = pl.create_tile([1, HC_PAD], dtype=pl.INT32)
            pre_logits = pl.add(pl.mul(normalized, pl.read(scale, [0])), bias)
            pre_logits = pl.tile.gather(pre_logits, coeff_lanes, coeff_tmp)
            post_logits = pl.add(pl.mul(normalized, pl.read(scale, [1])), bias)
            post_logits = pl.tile.gather(post_logits, pl.add(coeff_lanes, HC_MULT), coeff_tmp)
            pre = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0)), HC_EPS)
            post = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0)), 2.0)
            pl.store(pl.set_validshape(pre, 1, HC_MULT), [t, 0], pre_mix)
            pl.store(pl.set_validshape(post, 1, HC_MULT), [t, 0], post_mix)


@pl.jit.inline
def hc_decode_mixes_separate_token(x_hc: pl.Tensor, function: pl.Tensor, scale: pl.Tensor, base: pl.Tensor,
                                   pre_mix: pl.Tensor, post_mix: pl.Tensor, residual_mix: pl.Tensor):
    # Residual normalization takes longer than the two sigmoids. Start it
    # as soon as its own projection completes.
    tokens = pl.tensor.dim(x_hc, 0)
    inv = pl.create_tensor([LINEAR_T_TILE, 1], dtype=pl.FP32)
    hc_decode_token_rms(x_hc, tokens, inv)
    residual_projection = hc_short_split_linear(x_hc, function, 8, 16, 10, 512)
    hc_short_split_coefficients(
        inv, residual_projection, scale, base, pre_mix, post_mix, residual_mix, True, 8, 16, 10,
    )
    sigmoid_projection = hc_short_split_linear(x_hc, function, 0, 8, 5, 512)
    hc_short_split_coefficients(
        inv, sigmoid_projection, scale, base, pre_mix, post_mix, residual_mix, False, 0, 8, 5,
    )


@pl.jit.inline
def hc_decode_mixes(
    x_hc: pl.Tensor,
    function: pl.Tensor,
    scale: pl.Tensor,
    base: pl.Tensor,
    pre_mix: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    """Select the short-slab Decode kernel or the shared multi-token path."""
    if num_tokens == 1:
        # Keep the split-K and reduction order of the physical-slab path.
        inv_rms, partials = hc_project(x_hc, function, one_token=True)
        hc_sinkhorn(inv_rms, partials, scale, base, pre_mix, post_mix, residual_mix, one_token=True)
    else:
        inv_rms_common, partials_common = hc_project(x_hc, function, one_token=False)
        hc_sinkhorn(
            inv_rms_common, partials_common, scale, base, pre_mix, post_mix, residual_mix, one_token=False,
        )


@pl.jit.inline
def hc_project(x_hc: pl.Tensor, function: pl.Tensor, one_token: pl.constexpr = False):
    """Compute the inverse RMS and split-K coefficient projection."""
    t_dim = pl.tensor.dim(x_hc, 0)
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])

    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    rms_blocks = (t_dim + T_TILE - 1) // T_TILE
    if one_token:
        rms_blocks = 1
    rms_tiles = rms_blocks
    rms_workers = pl.max(1, pl.min(rms_tiles, pl.system.available_aiv_count()))
    for rms_worker in pl.spmd(rms_workers, name_hint="mhc_rms"):
        for block in pl.range(rms_worker, rms_tiles, rms_workers):
            t0 = block * T_TILE
            valid_rows = pl.min(T_TILE, t_dim - t0)
            if one_token:
                valid_rows = 1
            sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(HC_DIM // RMS_K_TILE, stage=4):
                k0 = kb * RMS_K_TILE
                rms_x_tile = pl.slice(
                    x_flat,
                    [T_TILE, RMS_K_TILE],
                    [t0, k0],
                    valid_shape=[valid_rows, RMS_K_TILE],
                )
                sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(rms_x_tile, rms_x_tile)), [1, T_TILE]))
            rms_arg = pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS)
            inv_rms[t0 : t0 + T_TILE, 0:1] = pl.reshape(
                pl.rsqrt(rms_arg, high_precision=True), [T_TILE, 1]
            )

    mixes_partials = pl.create_tensor([LINEAR_OK * t_linear, MIX_PAD], dtype=pl.FP32)
    linear_blocks = t_linear // LINEAR_T_TILE
    if one_token:
        linear_blocks = 1
    linear_tiles = linear_blocks * LINEAR_OK
    linear_workers = pl.max(1, pl.min(linear_tiles, pl.system.available_cluster_count()))
    for linear_worker in pl.spmd(linear_workers, name_hint="mhc_linear"):
        for task in pl.range(linear_worker, linear_tiles, linear_workers):
            t0 = (task // LINEAR_OK) * LINEAR_T_TILE
            split = task % LINEAR_OK
            k_base = split * LINEAR_K_PER_SPLIT
            valid_rows = pl.min(LINEAR_T_TILE, t_dim - t0)
            if one_token:
                valid_rows = 1
            acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
            for kb in pl.pipeline(LINEAR_K_PER_SPLIT // LINEAR_K_TILE, stage=2):
                k0 = k_base + kb * LINEAR_K_TILE
                linear_x_tile = pl.slice(
                    x_flat,
                    [LINEAR_T_TILE, LINEAR_K_TILE],
                    [t0, k0],
                    valid_shape=[valid_rows, LINEAR_K_TILE],
                )
                w_tile = pl.slice(
                    function,
                    [MIX_PAD, LINEAR_K_TILE],
                    [0, k0],
                    valid_shape=[MIX_HC, LINEAR_K_TILE],
                )
                acc = pl.matmul_acc(acc, linear_x_tile, w_tile, b_trans=True, init_cond=(kb == 0))
            partial = split * t_linear + t0
            mixes_partials[partial : partial + LINEAR_T_TILE, 0:MIX_PAD] = acc

    return inv_rms, mixes_partials


@pl.jit.inline
def hc_project_decode(
    x_hc: pl.Tensor,
    function: pl.Tensor,
    inv_rms: pl.Tensor,
    mixes_partials: pl.Tensor,
    short_attention: pl.constexpr,
    one_token: pl.constexpr = False,
):
    """Compute the inverse RMS and split-K coefficient projection."""
    t_dim = pl.tensor.dim(x_hc, 0)
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])

    if short_attention:
        hc_decode_token_rms(x_hc, t_dim, inv_rms)
    else:
        rms_blocks = (t_dim + T_TILE - 1) // T_TILE
        if one_token:
            rms_blocks = 1
        rms_tiles = rms_blocks
        rms_workers = pl.max(1, pl.min(rms_tiles, pl.system.available_aiv_count()))
        for rms_worker in pl.spmd(rms_workers, name_hint="mhc_rms"):
            for block in pl.range(rms_worker, rms_tiles, rms_workers):
                t0 = block * T_TILE
                valid_rows = pl.min(T_TILE, t_dim - t0)
                if one_token:
                    valid_rows = 1
                sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
                for kb in pl.pipeline(HC_DIM // DECODE_RMS_K_TILE, stage=1):
                    k0 = kb * DECODE_RMS_K_TILE
                    rms_x_tile = pl.slice(
                        x_flat,
                        [T_TILE, DECODE_RMS_K_TILE],
                        [t0, k0],
                        valid_shape=[valid_rows, DECODE_RMS_K_TILE],
                    )
                    sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(rms_x_tile, rms_x_tile)), [1, T_TILE]))
                rms_arg = pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS)
                inv_rms[t0 : t0 + T_TILE, 0:1] = pl.reshape(
                    pl.rsqrt(rms_arg, high_precision=True), [T_TILE, 1]
                )

    linear_blocks = t_linear // LINEAR_T_TILE
    if one_token:
        linear_blocks = 1
    linear_tiles = linear_blocks * DECODE_LINEAR_OK
    linear_workers = pl.max(1, pl.min(linear_tiles, pl.system.available_cluster_count()))
    for linear_worker in pl.spmd(
        linear_workers,
        name_hint="mhc_linear", allow_early_resolve=short_attention,
    ):
        for task in pl.range(linear_worker, linear_tiles, linear_workers):
            t0 = (task // DECODE_LINEAR_OK) * LINEAR_T_TILE
            split = task % DECODE_LINEAR_OK
            k_base = split * DECODE_LINEAR_K_PER_SPLIT
            valid_rows = pl.min(LINEAR_T_TILE, t_dim - t0)
            if one_token:
                valid_rows = 1
            acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
            for kb in pl.pipeline(DECODE_LINEAR_K_PER_SPLIT // LINEAR_K_TILE, stage=2):
                k0 = k_base + kb * LINEAR_K_TILE
                linear_x_tile = pl.slice(
                    x_flat,
                    [LINEAR_T_TILE, LINEAR_K_TILE],
                    [t0, k0],
                    valid_shape=[valid_rows, LINEAR_K_TILE],
                )
                w_tile = pl.slice(
                    function,
                    [MIX_PAD, LINEAR_K_TILE],
                    [0, k0],
                    valid_shape=[MIX_HC, LINEAR_K_TILE],
                )
                acc = pl.matmul_acc(acc, linear_x_tile, w_tile, b_trans=True, init_cond=(kb == 0))
            partial = split * t_linear + t0
            mixes_partials[partial : partial + LINEAR_T_TILE, 0:MIX_PAD] = acc



@pl.jit.inline
def hc_project_prefill(
    x_hc: pl.Tensor,
    function: pl.Tensor,
    incoming_pre_mix: pl.Tensor,
    output: pl.Tensor,
    scratch_deps: pl.Array[1, pl.TASK_ID],
):
    """Batch Prefill RMS, projection and incoming-stream collapse."""
    tokens = pl.tensor.dim(x_hc, 0)
    flat = pl.reshape(x_hc, [tokens, HC_DIM])
    streams = pl.reshape(x_hc, [tokens * PREFILL_RMS_ROWS, PREFILL_RMS_D_TILE])
    collapsed_chunks = pl.reshape(output, [tokens * PREFILL_COLLAPSE_ROWS, PREFILL_RMS_D_TILE])
    inverse = pl.create_tensor([tokens, PREFILL_RMS_T_TILE], dtype=pl.FP32)
    workers = pl.min(pl.min(tokens, PREFILL_WORKER_TILE), pl.system.available_aiv_count())
    with pl.spmd(workers, name_hint="mhc_rms", deps=[scratch_deps]) as _rms_ready:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, tokens, workers):
            values = pl.load(streams, [token * PREFILL_RMS_ROWS, 0], [PREFILL_RMS_ROWS, PREFILL_RMS_D_TILE])
            stream0 = pl.slice(values, [PREFILL_COLLAPSE_ROWS, PREFILL_RMS_D_TILE], [0, 0])
            stream1 = pl.slice(values, [PREFILL_COLLAPSE_ROWS, PREFILL_RMS_D_TILE], [PREFILL_COLLAPSE_ROWS, 0])
            stream2 = pl.slice(values, [PREFILL_COLLAPSE_ROWS, PREFILL_RMS_D_TILE], [2 * PREFILL_COLLAPSE_ROWS, 0])
            stream3 = pl.slice(values, [PREFILL_COLLAPSE_ROWS, PREFILL_RMS_D_TILE], [3 * PREFILL_COLLAPSE_ROWS, 0])
            weight0 = pl.read(incoming_pre_mix, [token, 0])
            weight1 = pl.read(incoming_pre_mix, [token, 1])
            weight2 = pl.read(incoming_pre_mix, [token, 2])
            weight3 = pl.read(incoming_pre_mix, [token, 3])
            first_pair = pl.add(pl.mul(stream0, weight0), pl.mul(stream1, weight1))
            second_pair = pl.add(pl.mul(stream2, weight2), pl.mul(stream3, weight3))
            collapsed = pl.add(first_pair, second_pair)
            collapsed = pl.cast(collapsed, pl.BF16, mode="rint")
            pl.store(collapsed, [token * PREFILL_COLLAPSE_ROWS, 0], collapsed_chunks)

            square = pl.mul(values, values)
            sum_tmp = pl.create_tile([PREFILL_RMS_ROWS, PREFILL_RMS_D_TILE], dtype=pl.FP32)
            partial_sums = pl.row_sum(square, sum_tmp)
            square_sum = pl.tile.read(partial_sums, [0, 0])
            for chunk in pl.unroll(1, PREFILL_RMS_ROWS):
                square_sum = square_sum + pl.tile.read(partial_sums, [chunk, 0])
            zero = pl.tile.full(
                [1, PREFILL_RMS_T_TILE], dtype=pl.FP32, value=0.0
            )
            square_sum_tile = pl.add(zero, square_sum)
            rms_arg = pl.add(pl.mul(square_sum_tile, HC_DIM_INV), NORM_EPS)
            rsqrt_tmp = pl.create_tile([1, PREFILL_RMS_T_TILE], dtype=pl.FP32)
            inverse_tile = pl.tile.rsqrt(rms_arg, rsqrt_tmp, high_precision=True)
            pl.store(inverse_tile, [token, 0], inverse)

    accumulated = pl.create_tensor([tokens, MIX_PAD], dtype=pl.FP32)
    channel_blocks = (MIX_HC + PREFILL_PROJ_N_TILE - 1) // PREFILL_PROJ_N_TILE
    projection_blocks = (tokens + PREFILL_PROJ_T_TILE - 1) // PREFILL_PROJ_T_TILE
    projection_tiles = projection_blocks * channel_blocks
    projection_workers = pl.max(1, pl.min(projection_tiles, pl.system.available_cluster_count()))
    for projection_worker in pl.spmd(projection_workers, name_hint="mhc_linear"):
        for block in pl.range(projection_worker, projection_tiles, projection_workers):
            token = (block // channel_blocks) * PREFILL_PROJ_T_TILE
            channel = (block % channel_blocks) * PREFILL_PROJ_N_TILE
            rows = pl.min(PREFILL_PROJ_T_TILE, tokens - token)
            columns = pl.min(PREFILL_PROJ_N_TILE, MIX_HC - channel)
            acc = pl.create_tile(
                [PREFILL_PROJ_T_TILE, PREFILL_PROJ_N_TILE],
                dtype=pl.FP32,
                target_memory=pl.Mem.Acc,
                compact=True,
            )
            for inner in pl.pipeline(0, HC_DIM, PREFILL_PROJ_K_TILE, stage=2):
                source = pl.load(
                    flat,
                    [token, inner],
                    [PREFILL_PROJ_T_TILE, PREFILL_PROJ_K_TILE],
                    valid_shape=[rows, PREFILL_PROJ_K_TILE],
                    target_memory=pl.Mem.Mat,
                )
                weight = pl.load(
                    function,
                    [channel, inner],
                    [PREFILL_PROJ_N_TILE, PREFILL_PROJ_K_TILE],
                    valid_shape=[columns, PREFILL_PROJ_K_TILE],
                    target_memory=pl.Mem.Mat,
                )
                acc = pl.matmul_acc(
                    acc,
                    source,
                    pl.tile.transpose_view(weight),
                    init_cond=(inner == 0),
                )
            projection = pl.set_validshape(acc, rows, columns)
            pl.store(projection, [token, channel], accumulated)
    return inverse, accumulated


@pl.jit.inline
def hc_sinkhorn_prefill(
    inv_rms: pl.Tensor,
    projected: pl.Tensor,
    scale: pl.Tensor,
    base: pl.Tensor,
    pre_mix: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    token_tile: pl.constexpr,
    lane_bits: pl.constexpr,
    lanes_per_stream: pl.constexpr,
    matrix_elements: pl.constexpr,
):
    """Batch Prefill sigmoid and Sinkhorn coefficient generation."""
    tokens = pl.tensor.dim(pre_mix, 0)
    residual_flat = pl.reshape(residual_mix, [tokens, HC_MULT * HC_MULT])
    bias = pl.reshape(base, [1, MIX_HC])
    scale0 = pl.read(scale, [0])
    scale1 = pl.read(scale, [1])
    scale2 = pl.read(scale, [2])
    blocks = (tokens + token_tile - 1) // token_tile
    workers = pl.min(pl.min(blocks, PREFILL_WORKER_TILE), pl.system.available_aiv_count())
    with pl.spmd(workers, name_hint="mhc_sinkhorn") as ready:
        worker = pl.tile.get_block_idx()
        # Pack [token, row, column] as [column, token * HC_MULT + row].
        lanes = pl.tile.arange(0, [1, matrix_elements], dtype=pl.INT32)
        axis = pl.shrs(lanes, lane_bits)
        lane = pl.sub(lanes, pl.shls(axis, lane_bits))
        pack_index = pl.add(pl.shls(lane, PREFILL_SINKHORN_STREAM_BITS), axis)
        # Swap the two matrix axes independently for every token.
        lane_token = pl.shrs(lane, PREFILL_SINKHORN_STREAM_BITS)
        token_base = pl.shls(lane_token, PREFILL_SINKHORN_STREAM_BITS)
        stream = pl.sub(lane, token_base)
        transpose_index = pl.add(pl.shls(stream, lane_bits), token_base)
        transpose_index = pl.add(transpose_index, axis)
        # The final column normalization leaves [row, token * HC_MULT + column].
        output_row = pl.shrs(lanes, PREFILL_SINKHORN_STREAM_BITS)
        output_column = pl.sub(lanes, pl.shls(output_row, PREFILL_SINKHORN_STREAM_BITS))
        output_token = pl.shrs(output_row, PREFILL_SINKHORN_STREAM_BITS)
        output_token_base = pl.shls(output_token, PREFILL_SINKHORN_STREAM_BITS)
        output_stream = pl.sub(output_row, output_token_base)
        unpack_index = pl.add(pl.shls(output_stream, lane_bits), output_token_base)
        unpack_index = pl.add(unpack_index, output_column)
        gather_tmp = pl.create_tile([1, matrix_elements], dtype=pl.INT32)
        for block in pl.range(worker, blocks, workers):
            token = block * token_tile
            rows = pl.min(token_tile, tokens - token)
            inverse_group = pl.load(
                inv_rms,
                [token, 0],
                [token_tile, PREFILL_RMS_T_TILE],
                valid_shape=[rows, PREFILL_RMS_T_TILE],
            )
            inverse_tmp = pl.create_tile(
                [token_tile, PREFILL_RMS_T_TILE], dtype=pl.FP32
            )
            inverse = pl.row_min(inverse_group, tmp_tile=inverse_tmp)

            raw_pre = pl.load(
                projected,
                [token, 0],
                [token_tile, MIX_PAD],
                valid_shape=[rows, HC_MULT],
            )
            raw_post = pl.load(
                projected,
                [token, HC_MULT],
                [token_tile, MIX_PAD],
                valid_shape=[rows, HC_MULT],
            )
            pre_bias = pl.load(bias, [0, 0], [1, MIX_PAD], valid_shape=[1, HC_MULT])
            post_bias = pl.load(bias, [0, HC_MULT], [1, MIX_PAD], valid_shape=[1, HC_MULT])
            pre_normalized = pl.row_expand_mul(raw_pre, inverse)
            post_normalized = pl.row_expand_mul(raw_post, inverse)
            pre_logits = pl.col_expand_add(pl.mul(pre_normalized, scale0), pre_bias)
            post_logits = pl.col_expand_add(pl.mul(post_normalized, scale1), post_bias)
            pre = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0)), HC_EPS)
            post = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0)), 2.0)
            pre = pl.set_validshape(pre, rows, HC_MULT)
            post = pl.set_validshape(post, rows, HC_MULT)
            pl.store(pre, [token, 0], pre_mix)
            pl.store(post, [token, 0], post_mix)

            logits = pl.load(
                projected, [token, 2 * HC_MULT], [token_tile, HC_MULT * HC_MULT],
                valid_shape=[rows, HC_MULT * HC_MULT],
            )
            residual_bias = pl.load(bias, [0, 2 * HC_MULT], [1, HC_MULT * HC_MULT])
            logits = pl.row_expand_mul(logits, inverse)
            logits = pl.col_expand_add(pl.mul(logits, scale2), residual_bias)
            logits = pl.fillpad(logits, pad_value=pl.PadValue.zero)
            logits = pl.set_validshape(logits, token_tile, HC_MULT * HC_MULT)
            logits_flat = pl.reshape(logits, [1, matrix_elements])
            packed = pl.tile.gather(logits_flat, pack_index, gather_tmp)
            packed = pl.reshape(packed, [HC_MULT, lanes_per_stream])
            maximum = pl.col_max(packed)
            exp_values = pl.exp(pl.col_expand_sub(packed, maximum))
            denominator = pl.col_sum(exp_values)
            probability = pl.add(pl.col_expand_div(exp_values, denominator), HC_EPS)

            # Alternate the axes by transposing each token's four-by-four matrix.
            for _ in pl.unroll(2 * HC_SINKHORN_ITER - 1):
                flat = pl.reshape(probability, [1, matrix_elements])
                rearranged = pl.tile.gather(flat, transpose_index, gather_tmp)
                transposed = pl.reshape(rearranged, [HC_MULT, lanes_per_stream])
                denominator = pl.add(pl.col_sum(transposed), HC_EPS)
                probability = pl.col_expand_div(transposed, denominator)
            flat = pl.reshape(probability, [1, matrix_elements])
            compact = pl.tile.gather(flat, unpack_index, gather_tmp)
            compact = pl.reshape(compact, [token_tile, HC_MULT * HC_MULT])
            compact = pl.set_validshape(compact, rows, HC_MULT * HC_MULT)
            pl.store(compact, [token, 0], residual_flat)
    return ready


@pl.jit.inline
def hc_head(
    x_hc: pl.Tensor,
    pre_mix: pl.Tensor,
    output: pl.Tensor,
):
    """Collapse HC streams with the delayed pre-mix and BF16 rounding."""
    t_dim = pl.tensor.dim(x_hc, 0)
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    collapse_tiles = (t_dim + PRE_T_TILE - 1) // PRE_T_TILE
    collapse_workers = pl.max(1, pl.min(collapse_tiles, pl.system.available_aiv_count()))
    for collapse_worker in pl.spmd(collapse_workers, name_hint="hc_collapse", allow_early_resolve=True):
        for block in pl.range(collapse_worker, collapse_tiles, collapse_workers):
            for t in pl.range(block * PRE_T_TILE, pl.min((block + 1) * PRE_T_TILE, t_dim)):
                pre_tile = pl.load(pre_mix, [t, 0], [1, HC_PAD], valid_shape=[1, HC_MULT])
                pre0 = pl.tile.read(pre_tile, [0, 0])
                pre1 = pl.tile.read(pre_tile, [0, 1])
                pre2 = pl.tile.read(pre_tile, [0, 2])
                pre3 = pl.tile.read(pre_tile, [0, 3])
                for d0 in pl.pipeline(0, D, HEAD_D_TILE, stage=2):
                    x0 = pl.load(x_flat, [t, d0], [1, HEAD_D_TILE])
                    x1 = pl.load(x_flat, [t, D + d0], [1, HEAD_D_TILE])
                    x2 = pl.load(x_flat, [t, 2 * D + d0], [1, HEAD_D_TILE])
                    x3 = pl.load(x_flat, [t, 3 * D + d0], [1, HEAD_D_TILE])
                    y0 = pl.mul(x0, pre0)
                    y1 = pl.mul(x1, pre1)
                    y2 = pl.mul(x2, pre2)
                    y3 = pl.mul(x3, pre3)
                    y01 = pl.add(y0, y1)
                    y23 = pl.add(y2, y3)
                    y_tile = pl.add(y01, y23)
                    y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
                    pl.store(y_bf16, [t, d0], output)
    return output


@pl.jit.inline
def hc_head_norm(
    x_hc: pl.Tensor,
    pre_mix: pl.Tensor,
    norm_weight: pl.Tensor[[D], pl.BF16],
    output: pl.Tensor,
    one_token: pl.constexpr = False,
):
    """Collapse HC rows and normalize their rounded BF16 values."""
    tokens = pl.tensor.dim(x_hc, 0)
    x_flat = pl.reshape(x_hc, [tokens, HC_DIM])
    blocks = (tokens + PRE_NORM_T_TILE - 1) // PRE_NORM_T_TILE
    if one_token:
        blocks = 1
    collapse_norm_tiles = blocks
    collapse_norm_workers = pl.max(1, pl.min(collapse_norm_tiles, pl.system.available_aiv_count()))
    with pl.spmd(collapse_norm_workers, name_hint="hc_collapse_norm") as ready:
        collapse_norm_worker = pl.tile.get_block_idx()
        for block in pl.range(collapse_norm_worker, collapse_norm_tiles, collapse_norm_workers):
            t0 = block * PRE_NORM_T_TILE
            valid_rows = pl.min(PRE_NORM_T_TILE, tokens - t0)
            if one_token:
                valid_rows = 1
            pre_tile = pl.slice(pre_mix, [PRE_NORM_T_TILE, HC_PAD], [t0, 0], valid_shape=[valid_rows, HC_MULT])
            pre_tile = pl.set_validshape(pre_tile, PRE_NORM_T_TILE, HC_PAD)
            pre_rows = pl.transpose(pre_tile, 0, 1)
            pre0 = pl.reshape(pre_rows[0:1, 0:PRE_NORM_T_TILE], [PRE_NORM_T_TILE, 1])
            pre1 = pl.reshape(pre_rows[1:2, 0:PRE_NORM_T_TILE], [PRE_NORM_T_TILE, 1])
            pre2 = pl.reshape(pre_rows[2:3, 0:PRE_NORM_T_TILE], [PRE_NORM_T_TILE, 1])
            pre3 = pl.reshape(pre_rows[3:4, 0:PRE_NORM_T_TILE], [PRE_NORM_T_TILE, 1])
            pre0 = pl.set_validshape(pre0, valid_rows, 1)
            pre1 = pl.set_validshape(pre1, valid_rows, 1)
            pre2 = pl.set_validshape(pre2, valid_rows, 1)
            pre3 = pl.set_validshape(pre3, valid_rows, 1)
            collapsed = pl.create_tensor([D // PRE_NORM_D_TILE * PRE_NORM_T_TILE, PRE_NORM_D_TILE], dtype=pl.BF16)
            sq_sum = pl.full([1, PRE_NORM_T_TILE], dtype=pl.FP32, value=0.0)
            for d0 in pl.pipeline(0, D, PRE_NORM_D_TILE, stage=2):
                x0 = pl.slice(x_flat, [PRE_NORM_T_TILE, PRE_NORM_D_TILE], [t0, d0], valid_shape=[valid_rows, PRE_NORM_D_TILE])
                x1 = pl.slice(x_flat, [PRE_NORM_T_TILE, PRE_NORM_D_TILE], [t0, D + d0], valid_shape=[valid_rows, PRE_NORM_D_TILE])
                x2 = pl.slice(x_flat, [PRE_NORM_T_TILE, PRE_NORM_D_TILE], [t0, 2 * D + d0], valid_shape=[valid_rows, PRE_NORM_D_TILE])
                x3 = pl.slice(x_flat, [PRE_NORM_T_TILE, PRE_NORM_D_TILE], [t0, 3 * D + d0], valid_shape=[valid_rows, PRE_NORM_D_TILE])
                y0 = pl.row_expand_mul(x0, pre0)
                y1 = pl.row_expand_mul(x1, pre1)
                y2 = pl.row_expand_mul(x2, pre2)
                y3 = pl.row_expand_mul(x3, pre3)
                y01 = pl.add(y0, y1)
                y23 = pl.add(y2, y3)
                mixed = pl.add(y01, y23)
                rounded = pl.cast(mixed, target_type=pl.BF16, mode="rint")
                cache_row = d0 // PRE_NORM_D_TILE * PRE_NORM_T_TILE
                collapsed[cache_row : cache_row + PRE_NORM_T_TILE, 0:PRE_NORM_D_TILE] = rounded
                padded = pl.fillpad(rounded, pad_value=pl.PadValue.zero)
                padded = pl.set_validshape(padded, PRE_NORM_T_TILE, PRE_NORM_D_TILE)
                value = pl.cast(padded, target_type=pl.FP32)
                squared = pl.mul(value, value)
                partial = pl.reshape(pl.row_sum(squared), [1, PRE_NORM_T_TILE])
                sq_sum[0:1, 0:PRE_NORM_T_TILE] = pl.add(sq_sum, partial)
            rms_arg = pl.add(pl.mul(sq_sum, 1.0 / D), NORM_EPS)
            inverse = pl.rsqrt(rms_arg, high_precision=True)
            inverse = pl.reshape(inverse, [PRE_NORM_T_TILE, 1])
            for d0 in pl.pipeline(0, D, PRE_NORM_D_TILE, stage=2):
                cache_row = d0 // PRE_NORM_D_TILE * PRE_NORM_T_TILE
                value = pl.cast(collapsed[cache_row : cache_row + PRE_NORM_T_TILE, 0:PRE_NORM_D_TILE], target_type=pl.FP32)
                gamma = pl.cast(norm_weight[d0 : d0 + PRE_NORM_D_TILE], target_type=pl.FP32)
                gamma = pl.reshape(gamma, [1, PRE_NORM_D_TILE])
                scaled = pl.row_expand_mul(value, inverse)
                normalized = pl.col_expand_mul(scaled, gamma)
                result = pl.cast(normalized, target_type=pl.BF16, mode="rint")
                result = pl.set_validshape(result, valid_rows, PRE_NORM_D_TILE)
                output[t0 : t0 + PRE_NORM_T_TILE, d0 : d0 + PRE_NORM_D_TILE] = result
    return ready


@pl.jit.inline
def hc_head_norm_decode(
    x_hc: pl.Tensor,
    pre_mix: pl.Tensor,
    norm_weight: pl.Tensor[[D], pl.BF16],
    output: pl.Tensor,
):
    """Collapse and normalize one Decode row in vector memory."""
    tokens = pl.tensor.dim(x_hc, 0)
    x_flat = pl.reshape(x_hc, [tokens, HC_DIM])
    gamma_flat = pl.reshape(norm_weight, [1, D])
    decode_norm_tiles = tokens
    decode_norm_workers = pl.max(1, pl.min(decode_norm_tiles, pl.system.available_aiv_count()))
    with pl.spmd(decode_norm_workers, name_hint="hc_decode_collapse_norm") as ready:
        decode_norm_worker = pl.tile.get_block_idx()
        for t in pl.range(decode_norm_worker, decode_norm_tiles, decode_norm_workers):
            weights = pl.load(pre_mix, [t, 0], [1, HC_PAD], valid_shape=[1, HC_MULT])
            p0 = pl.mul(pl.load(x_flat, [t, 0], [1, D]), pl.tile.read(weights, [0, 0]))
            p1 = pl.mul(pl.load(x_flat, [t, D], [1, D]), pl.tile.read(weights, [0, 1]))
            p01 = pl.add(p0, p1)
            p2 = pl.mul(pl.load(x_flat, [t, 2 * D], [1, D]), pl.tile.read(weights, [0, 2]))
            p3 = pl.mul(pl.load(x_flat, [t, 3 * D], [1, D]), pl.tile.read(weights, [0, 3]))
            p23 = pl.add(p2, p3)
            rounded = pl.cast(pl.add(p01, p23), pl.BF16, mode="rint")
            values = pl.cast(rounded, pl.FP32)
            lanes = pl.reshape(values, [8, D // 8])
            squares = pl.mul(lanes, lanes)
            scratch = pl.create_tile([8, D // 8], dtype=pl.FP32)
            partials = pl.reshape(pl.row_sum(squares, tmp_tile=scratch), [1, 8])
            square_sum = pl.tile.read(partials, [0, 0])
            for lane in pl.unroll(1, 8):
                square_sum = square_sum + pl.tile.read(partials, [0, lane])
            rms_arg = pl.mul(pl.tile.full([1, 8], dtype=pl.FP32, value=1.0 / D), square_sum)
            rsqrt_tmp = pl.create_tile([1, 8], dtype=pl.FP32)
            inverse = pl.tile.rsqrt(pl.add(rms_arg, NORM_EPS), rsqrt_tmp, high_precision=True)
            scaled = pl.mul(values, pl.tile.read(inverse, [0, 0]))
            gamma = pl.cast(pl.load(gamma_flat, [0, 0], [1, D]), pl.FP32)
            result = pl.cast(pl.mul(scaled, gamma), pl.BF16, mode="rint")
            pl.store(result, [t, 0], output)
    return ready


@pl.jit.inline
def hc_head_norm_adaptive(
    x_hc: pl.Tensor,
    pre_mix: pl.Tensor,
    norm_weight: pl.Tensor[[D], pl.BF16],
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    """Use the short Decode norm tile only for a single-token slab."""
    if num_tokens == 1:
        ready = hc_head_norm(x_hc, pre_mix, norm_weight, output, one_token=True)
    else:
        ready = hc_head_norm(x_hc, pre_mix, norm_weight, output, one_token=False)
    return ready


@pl.jit.inline(auto_scope=False)
def hc_head_norm_quantize_decode(
    x_hc: pl.Tensor,
    pre_mix: pl.Tensor,
    norm_weight: pl.Tensor[[D], pl.BF16],
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    zero_padding: pl.constexpr,
):
    """Prepare the shared BF16 and W8A8 attention inputs for a Decode slab."""
    padded_rows = (pl.tensor.dim(output, 0) + 15) // 16 * 16
    payload = pl.create_tensor([padded_rows, D], dtype=pl.INT8)
    scale = pl.create_tensor([padded_rows, 1], dtype=pl.FP32)
    if num_tokens == 1:
        tokens = pl.tensor.dim(x_hc, 0)
        x_flat = pl.reshape(x_hc, [tokens, HC_DIM])
        gamma_flat = pl.reshape(norm_weight, [1, D])
        decode_quant_tiles = 1 + D // DECODE_QUANT_REDUCE_TILE
        decode_quant_workers = pl.max(1, pl.min(decode_quant_tiles, pl.system.available_aiv_count()))
        with pl.spmd(decode_quant_workers, name_hint="hc_decode_collapse_norm_quant") as ready:
            decode_quant_worker = pl.tile.get_block_idx()
            for worker in pl.range(decode_quant_worker, decode_quant_tiles, decode_quant_workers):
                if worker == 0:
                    weights = pl.load(pre_mix, [0, 0], [1, HC_PAD], valid_shape=[1, HC_MULT])
                    p0 = pl.mul(pl.load(x_flat, [0, 0], [1, D]), pl.tile.read(weights, [0, 0]))
                    p1 = pl.mul(pl.load(x_flat, [0, D], [1, D]), pl.tile.read(weights, [0, 1]))
                    p01 = pl.add(p0, p1)
                    p2 = pl.mul(pl.load(x_flat, [0, 2 * D], [1, D]), pl.tile.read(weights, [0, 2]))
                    p3 = pl.mul(pl.load(x_flat, [0, 3 * D], [1, D]), pl.tile.read(weights, [0, 3]))
                    p23 = pl.add(p2, p3)
                    rounded = pl.cast(pl.add(p01, p23), pl.BF16, mode="rint")
                    values = pl.cast(rounded, pl.FP32)
                    # Match hc_head_norm: one hardware row sum per 256 channels, then left-to-right addition.
                    lanes = pl.reshape(values, [D // PRE_NORM_D_TILE, PRE_NORM_D_TILE])
                    squares = pl.mul(lanes, lanes)
                    scratch = pl.create_tile([8, PRE_NORM_D_TILE], dtype=pl.FP32)
                    first = pl.row_sum(pl.tile.slice(squares, [8, PRE_NORM_D_TILE], [0, 0]), tmp_tile=scratch)
                    middle = pl.row_sum(pl.tile.slice(squares, [8, PRE_NORM_D_TILE], [8, 0]), tmp_tile=scratch)
                    last = pl.row_sum(pl.tile.slice(squares, [8, PRE_NORM_D_TILE], [12, 0]), tmp_tile=scratch)
                    square_sum = pl.tile.read(first, [0, 0])
                    for lane in pl.unroll(1, 8):
                        square_sum = square_sum + pl.tile.read(first, [lane, 0])
                    for lane in pl.unroll(0, 8):
                        square_sum = square_sum + pl.tile.read(middle, [lane, 0])
                    for lane in pl.unroll(4, 8):
                        square_sum = square_sum + pl.tile.read(last, [lane, 0])
                    rms_arg = pl.mul(pl.tile.full([1, 8], dtype=pl.FP32, value=1.0 / D), square_sum)
                    rsqrt_tmp = pl.create_tile([1, 8], dtype=pl.FP32)
                    inverse = pl.tile.rsqrt(pl.add(rms_arg, NORM_EPS), rsqrt_tmp, high_precision=True)
                    scaled = pl.mul(values, pl.tile.read(inverse, [0, 0]))
                    gamma = pl.cast(pl.load(gamma_flat, [0, 0], [1, D]), pl.FP32)
                    result = pl.cast(pl.mul(scaled, gamma), pl.BF16, mode="rint")

                    grouped = pl.reshape(result, [8, D // 8])
                    fp32 = pl.cast(grouped, pl.FP32)
                    peak_scratch = pl.create_tile([8, D // 8], dtype=pl.FP32)
                    peaks = pl.reshape(pl.row_max(pl.abs(fp32), tmp_tile=peak_scratch), [1, 8])
                    peak_zero = pl.tile.full([8, 8], dtype=pl.FP32, value=0.0)
                    maximum = pl.row_max(pl.col_expand_add(peak_zero, peaks), tmp_tile=peak_zero)
                    pl.store(result, [0, 0], output)
                    live_scale = pl.maximum(pl.mul(maximum, 1.0 / 127.0), 1e-30)
                    inactive = pl.tile.full([1, 8], dtype=pl.FP32, value=1e-30)
                    scale_lanes = pl.cast(pl.tile.arange(0, [1, 8], dtype=pl.INT32), pl.FP32)
                    active_lane = pl.tile.cmps(scale_lanes, 0.0, cmp_type=0)
                    scales = pl.tile.select(active_lane, pl.reshape(live_scale, [1, 8]), inactive)
                    pl.store(pl.reshape(scales, [8, 1]), [0, 0], scale)
                    pl.store(pl.reshape(inactive, [8, 1]), [8, 0], scale)
                    denominator = pl.mul(pl.maximum(maximum, 127e-30), 1.0 / 128.0)
                    reciprocal = pl.recip(denominator)
                    codes = _decode_checkpoint_codes(fp32, denominator, reciprocal)
                    pl.store(pl.reshape(codes, [1, D]), [0, 0], payload)
                else:
                    col = (worker - 1) * DECODE_QUANT_REDUCE_TILE
                    pad = pl.cast(pl.tile.full([8, DECODE_QUANT_REDUCE_TILE], dtype=pl.FP32, value=0.0), pl.INT8, mode="rint")
                    pl.store(pl.set_validshape(pad, 7, DECODE_QUANT_REDUCE_TILE), [1, col], payload)
                    pl.store(pad, [8, col], payload)
    else:
        ready = hc_head_norm_adaptive(x_hc, pre_mix, norm_weight, output, num_tokens)
        if zero_padding:
            zero_bf16_padding(output, pl.min(num_tokens, pl.tensor.dim(output, 0)))
        _quantize_projection_activation_into(output, payload, scale, num_tokens, padded_rows, D, True, False)
    return payload, scale, ready


@pl.jit.inline
def hc_ffn_decode_mixes(
    x_hc: pl.Tensor,
    function: pl.Tensor,
    scale: pl.Tensor,
    base: pl.Tensor,
    pre_mix: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    one_token: pl.constexpr = False,
):
    """Compute FFN coefficients after Decode attention has collapsed the streams."""
    t_dim = pl.tensor.dim(x_hc, 0)
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    partials = pl.create_tensor([DECODE_LINEAR_OK * t_linear, MIX_PAD], dtype=pl.FP32)
    hc_project_decode(x_hc, function, inv_rms, partials, short_attention=False, one_token=one_token)
    return hc_sinkhorn_decode(inv_rms, partials, scale, base, pre_mix, post_mix, residual_mix, one_token=one_token)


@pl.jit.inline
def hc_pre_prefill(
    x_hc: pl.Tensor,
    incoming_pre_mix: pl.Tensor,
    function: pl.Tensor,
    scale: pl.Tensor,
    base: pl.Tensor,
    output: pl.Tensor,
    next_pre_mix: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    scratch_deps: pl.Array[1, pl.TASK_ID],
):
    """Collapse the Prefill streams and generate the delayed HC coefficients."""
    prefill_inv_rms, prefill_projected = hc_project_prefill(x_hc, function, incoming_pre_mix, output, scratch_deps)
    tokens = pl.tensor.dim(x_hc, 0)
    if tokens >= PREFILL_SINKHORN_WIDE_T_TILE * PREFILL_WORKER_TILE:
        mixes_ready = hc_sinkhorn_prefill(
            prefill_inv_rms, prefill_projected, scale, base, next_pre_mix, post_mix, residual_mix,
            PREFILL_SINKHORN_WIDE_T_TILE, PREFILL_SINKHORN_WIDE_LANE_BITS,
            PREFILL_SINKHORN_WIDE_T_TILE * HC_MULT, PREFILL_SINKHORN_WIDE_T_TILE * HC_MULT * HC_MULT,
        )
    else:
        mixes_ready = hc_sinkhorn_prefill(
            prefill_inv_rms, prefill_projected, scale, base, next_pre_mix, post_mix, residual_mix,
            PREFILL_SINKHORN_T_TILE, PREFILL_SINKHORN_LANE_BITS,
            PREFILL_SINKHORN_T_TILE * HC_MULT, PREFILL_SINKHORN_T_TILE * HC_MULT * HC_MULT,
        )
    return mixes_ready


@pl.jit.inline
def hc_pre(
    x_hc: pl.Tensor,
    incoming_pre_mix: pl.Tensor,
    function: pl.Tensor,
    scale: pl.Tensor,
    base: pl.Tensor,
    output: pl.Tensor,
    next_pre_mix: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    prefill: pl.constexpr = False,
):
    """Generate delayed HC coefficients and collapse the incoming streams."""
    if prefill:
        scratch_deps = pl.array.create(1, pl.TASK_ID)
        mixes_ready = hc_pre_prefill(
            x_hc, incoming_pre_mix, function, scale, base,
            output, next_pre_mix, post_mix, residual_mix, scratch_deps,
        )
    else:
        mixes_ready = hc_ffn_decode_mixes(
            x_hc, function, scale, base, next_pre_mix, post_mix, residual_mix, one_token=False,
        )
        hc_head(x_hc, incoming_pre_mix, output)
    return mixes_ready


@pl.jit.inline
def zero_bf16_padding(x: pl.Tensor, valid_rows: pl.Scalar[pl.INT32]):
    """Clear physical rows that belong to an empty or partial TP slab.

    A TP rank with no logical token still publishes one fixed-capacity slab so
    every rank executes the same distributed window protocol.  Clearing the
    unused rows makes that padding deterministic and prevents an empty owner
    from publishing uninitialized data.
    """
    tokens = pl.tensor.dim(x, 0)
    if tokens > 0 and valid_rows < tokens:
        blocks = (tokens - valid_rows + 3) // 4
        workers = pl.min(blocks, pl.system.available_aiv_count())
        for worker in pl.spmd(workers, name_hint="decode_sp_zero_padding"):
            for block in pl.range(worker, blocks, workers):
                first = valid_rows + block * 4
                rows = pl.min(4, tokens - first)
                empty = pl.tile.full([4, D], dtype=pl.BF16, value=0.0)
                pl.store(pl.set_validshape(empty, rows, D), [first, 0], x)


@pl.jit.inline(auto_scope=False)
def hc_post_decode_block(
    sublayer: pl.Tensor,
    residual_flat: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix_flat: pl.Tensor,
    output_flat: pl.Tensor,
    block: pl.Scalar[pl.INDEX],
    stream_tile: pl.constexpr,
    d_tile: pl.constexpr,
):
    """Process one Decode hidden-width tile while sharing residual loads."""
    pair_block = block // (D // d_tile)
    t = pair_block // (HC_MULT // stream_tile)
    out_base = (pair_block % (HC_MULT // stream_tile)) * stream_tile
    d0 = (block % (D // d_tile)) * d_tile
    x_tile = pl.cast(sublayer[t : t + 1, d0 : d0 + d_tile], pl.FP32)
    residual0 = residual_flat[t : t + 1, d0 : d0 + d_tile]
    residual1 = residual_flat[t : t + 1, D + d0 : D + d0 + d_tile]
    residual2 = residual_flat[t : t + 1, 2 * D + d0 : 2 * D + d0 + d_tile]
    residual3 = residual_flat[t : t + 1, 3 * D + d0 : 3 * D + d0 + d_tile]
    for out_offset in pl.unroll(stream_tile):
        out_h = out_base + out_offset
        update = pl.mul(x_tile, pl.read(post_mix, [t, out_h]))
        term0 = pl.mul(residual0, pl.read(residual_mix_flat, [t, out_h]))
        term1 = pl.mul(residual1, pl.read(residual_mix_flat, [t, HC_MULT + out_h]))
        value = pl.add(term0, term1)
        term2 = pl.mul(residual2, pl.read(residual_mix_flat, [t, 2 * HC_MULT + out_h]))
        value = pl.add(value, term2)
        term3 = pl.mul(residual3, pl.read(residual_mix_flat, [t, 3 * HC_MULT + out_h]))
        value = pl.add(value, term3)
        value = pl.add(value, update)
        output_flat[t : t + 1, out_h * D + d0 : out_h * D + d0 + d_tile] = value
    return output_flat


@pl.jit.inline(auto_scope=False)
def hc_post_token_tile(
    x_tile: pl.Tile[[1, POST_VECTOR_COLUMNS], pl.FP32],
    residual_flat: pl.Tensor,
    post_mix: pl.Tensor,
    mixes_flat: pl.Tensor,
    output_flat: pl.Tensor,
    col: pl.Scalar[pl.INDEX],
    columns: pl.constexpr,
):
    """Mix one FFN Decode token from a BF16-rounded TP tile."""
    update_tile = pl.reshape(x_tile, [1, columns])
    r0 = pl.load(residual_flat, [0, col], [1, columns])
    r1 = pl.load(residual_flat, [0, 1 * D + col], [1, columns])
    r2 = pl.load(residual_flat, [0, 2 * D + col], [1, columns])
    r3 = pl.load(residual_flat, [0, 3 * D + col], [1, columns])
    for stream in pl.unroll(HC_MULT):
        update = pl.mul(update_tile, pl.read(post_mix, [0, stream]))
        value = pl.add(
            pl.mul(r0, pl.read(mixes_flat, [0, stream])),
            pl.mul(r1, pl.read(mixes_flat, [0, 1 * HC_MULT + stream])),
        )
        value = pl.add(value, pl.mul(r2, pl.read(mixes_flat, [0, 2 * HC_MULT + stream])))
        value = pl.add(value, pl.mul(r3, pl.read(mixes_flat, [0, 3 * HC_MULT + stream])))
        result = pl.add(value, update)
        pl.store(result, [0, stream * D + col], output_flat)


@pl.jit.inline(auto_scope=False)
def hc_post_collapse_tile(
    x_tile: pl.Tile[[1, POST_VECTOR_COLUMNS], pl.FP32],
    residual_flat: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix_flat: pl.Tensor,
    output_flat: pl.Tensor,
    t: pl.Scalar[pl.INDEX],
    d0: pl.Scalar[pl.INDEX],
    incoming: pl.Tensor,
    collapsed: pl.Tensor,
    d_tile: pl.constexpr,
):
    """Mix one Decode tile and collapse its four FP32 streams."""
    x_tile = pl.reshape(x_tile, [1, d_tile])
    residual0 = pl.load(residual_flat, [t, d0], [1, d_tile])
    residual1 = pl.load(residual_flat, [t, D + d0], [1, d_tile])
    residual2 = pl.load(residual_flat, [t, 2 * D + d0], [1, d_tile])
    residual3 = pl.load(residual_flat, [t, 3 * D + d0], [1, d_tile])
    weights = pl.load(incoming, [t, 0], [1, HC_PAD], valid_shape=[1, HC_MULT])
    first_pair = pl.tile.full([1, d_tile], dtype=pl.FP32, value=0.0)
    second_pair = pl.tile.full([1, d_tile], dtype=pl.FP32, value=0.0)
    for out_h in pl.unroll(HC_MULT):
        update = pl.mul(x_tile, pl.read(post_mix, [t, out_h]))
        term0 = pl.mul(residual0, pl.read(residual_mix_flat, [t, out_h]))
        term1 = pl.mul(residual1, pl.read(residual_mix_flat, [t, HC_MULT + out_h]))
        value = pl.add(term0, term1)
        term2 = pl.mul(residual2, pl.read(residual_mix_flat, [t, 2 * HC_MULT + out_h]))
        value = pl.add(value, term2)
        term3 = pl.mul(residual3, pl.read(residual_mix_flat, [t, 3 * HC_MULT + out_h]))
        value = pl.add(value, term3)
        value = pl.add(value, update)
        pl.store(value, [t, out_h * D + d0], output_flat)
        weighted = pl.mul(value, pl.tile.read(weights, [0, out_h]))
        if out_h == 0:
            first_pair = weighted
        elif out_h == 1:
            first_pair = pl.add(first_pair, weighted)
        elif out_h == 2:
            second_pair = weighted
        else:
            second_pair = pl.add(second_pair, weighted)
    reduced = pl.cast(pl.add(first_pair, second_pair), pl.BF16, mode="rint")
    pl.store(reduced, [t, d0], collapsed)
    return collapsed


@pl.jit.inline(auto_scope=False)
def hc_post_collapse_block(
    sublayer: pl.Tensor,
    residual_flat: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix_flat: pl.Tensor,
    output_flat: pl.Tensor,
    block: pl.Scalar[pl.INDEX],
    incoming: pl.Tensor,
    collapsed: pl.Tensor,
    d_tile: pl.constexpr,
):
    """Mix one Decode tile and collapse its four FP32 streams."""
    t = block // (D // d_tile)
    d0 = (block % (D // d_tile)) * d_tile
    x_tile = pl.cast(pl.load(sublayer, [t, d0], [1, d_tile]), pl.FP32)
    return hc_post_collapse_tile(
        x_tile, residual_flat, post_mix, residual_mix_flat, output_flat,
        t, d0, incoming, collapsed, d_tile,
    )


@pl.jit.inline
def hc_post_decode_run(
    sublayer: pl.Tensor,
    residual: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    output: pl.Tensor,
    deps: pl.Array[2, pl.TASK_ID],
    d_tile: pl.constexpr,
):
    """Run one Decode mHC-post tile shape after its producer dependencies."""
    t_dim = pl.tensor.dim(sublayer, 0)
    residual_flat = pl.reshape(residual, [t_dim, HC_DIM])
    residual_mix_flat = pl.reshape(residual_mix, [t_dim, HC_MULT * HC_MULT])
    output_flat = pl.reshape(output, [t_dim, HC_DIM])
    total_work = t_dim * (D // d_tile) * (HC_MULT // DECODE_POST_STREAM_TILE)
    workers = pl.min(pl.min(total_work, POST_WORKER_TILE), pl.system.available_aiv_count())
    with pl.spmd(workers, name_hint="hc_post_decode", deps=[deps]):
        worker = pl.tile.get_block_idx()
        for block in pl.pipeline(worker, total_work, workers, stage=1):
            hc_post_decode_block(
                sublayer, residual_flat, post_mix, residual_mix_flat, output_flat,
                block, DECODE_POST_STREAM_TILE, d_tile,
            )
    return output


@pl.jit.inline
def hc_post_decode(
    sublayer: pl.Tensor,
    residual: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    output: pl.Tensor,
    deps: pl.Array[2, pl.TASK_ID],
):
    """Route short and multi-token Decode slabs to their measured tile shapes."""
    t_dim = pl.tensor.dim(sublayer, 0)
    if t_dim <= 1:
        hc_post_decode_run(
            sublayer, residual, post_mix, residual_mix, output, deps,
            DECODE_POST_D_TILE,
        )
    else:
        hc_post_decode_run(
            sublayer, residual, post_mix, residual_mix, output, deps,
            DECODE_POST_D_TILE_WIDE,
        )
    return output


@pl.jit.inline
def hc_post_collapse_run(
    sublayer: pl.Tensor,
    residual: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    incoming: pl.Tensor,
    output: pl.Tensor,
    collapsed: pl.Tensor,
    d_tile: pl.constexpr,
):
    """Write residual streams and the BF16 FFN input in one producer."""
    count = pl.tensor.dim(sublayer, 0)
    residual_flat = pl.reshape(residual, [count, HC_DIM])
    residual_mix_flat = pl.reshape(residual_mix, [count, HC_MULT * HC_MULT])
    output_flat = pl.reshape(output, [count, HC_DIM])
    work = count * (D // d_tile)
    workers = pl.max(1, pl.min(pl.min(work, POST_WORKER_TILE), pl.system.available_aiv_count()))
    for worker in pl.spmd(workers, name_hint="hc_post_collapse"):
        for block in pl.pipeline(worker, work, workers, stage=1):
            hc_post_collapse_block(
                sublayer, residual_flat, post_mix, residual_mix_flat, output_flat,
                block, incoming, collapsed, d_tile,
            )
    return output


@pl.jit.inline
def hc_post_collapse(
    sublayer: pl.Tensor,
    residual: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    incoming: pl.Tensor,
    output: pl.Tensor,
    collapsed: pl.Tensor,
):
    """Mix the Decode attention result and prepare the FFN input."""
    count = pl.tensor.dim(sublayer, 0)
    if count <= 1:
        hc_post_collapse_run(
            sublayer, residual, post_mix, residual_mix, incoming,
            output, collapsed, DECODE_POST_D_TILE,
        )
    else:
        hc_post_collapse_run(
            sublayer, residual, post_mix, residual_mix, incoming,
            output, collapsed, DECODE_POST_D_TILE_WIDE,
        )
    return output


@pl.jit.inline(auto_scope=False)
def hc_post_block(
    sublayer: pl.Tensor,
    residual_flat: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix_flat: pl.Tensor,
    output_flat: pl.Tensor,
    block: pl.Scalar[pl.INDEX],
    stream_tile: pl.constexpr,
):
    """Mix one hidden-width tile for one or four output streams."""
    pair_block = block // (D // D_TILE)
    t = pair_block // (HC_MULT // stream_tile)
    out_base = (pair_block % (HC_MULT // stream_tile)) * stream_tile
    d0 = (block % (D // D_TILE)) * D_TILE
    x_tile = pl.cast(sublayer[t : t + 1, d0 : d0 + D_TILE], pl.FP32)
    residual0 = residual_flat[t : t + 1, d0 : d0 + D_TILE]
    residual1 = residual_flat[t : t + 1, D + d0 : D + d0 + D_TILE]
    residual2 = residual_flat[t : t + 1, 2 * D + d0 : 2 * D + d0 + D_TILE]
    residual3 = residual_flat[t : t + 1, 3 * D + d0 : 3 * D + d0 + D_TILE]
    for out_offset in pl.unroll(stream_tile):
        out_h = out_base + out_offset
        update = pl.mul(x_tile, pl.read(post_mix, [t, out_h]))
        term0 = pl.mul(residual0, pl.read(residual_mix_flat, [t, out_h]))
        term1 = pl.mul(residual1, pl.read(residual_mix_flat, [t, HC_MULT + out_h]))
        value = pl.add(term0, term1)
        term2 = pl.mul(residual2, pl.read(residual_mix_flat, [t, 2 * HC_MULT + out_h]))
        value = pl.add(value, term2)
        term3 = pl.mul(residual3, pl.read(residual_mix_flat, [t, 3 * HC_MULT + out_h]))
        value = pl.add(value, term3)
        value = pl.add(value, update)
        output_flat[t : t + 1, out_h * D + d0 : out_h * D + d0 + D_TILE] = value
    return output_flat


@pl.jit.inline
def hc_post(
    sublayer: pl.Tensor,
    residual: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    output: pl.Tensor,
    deps: pl.Array[2, pl.TASK_ID],
    prefill: pl.constexpr = False,
):
    """Mix all four residual streams after the supplied producer dependencies."""
    t_dim = pl.tensor.dim(sublayer, 0)
    residual_flat = pl.reshape(residual, [t_dim, HC_DIM])
    residual_mix_flat = pl.reshape(residual_mix, [t_dim, HC_MULT * HC_MULT])
    output_flat = pl.reshape(output, [t_dim, HC_DIM])
    if prefill:
        # Each hidden tile supplies all four output streams.
        tiles = t_dim * (D // PREFILL_POST_D_TILE)
        workers = pl.min(pl.min(tiles, PREFILL_POST_WORKERS), pl.system.available_aiv_count())
        with pl.spmd(workers, name_hint="hc_post_prefill", deps=[deps]):
            worker = pl.tile.get_block_idx()
            begin = worker * tiles // workers
            end = (worker + 1) * tiles // workers
            for block in pl.pipeline(begin, end, stage=2):
                token = block // (D // PREFILL_POST_D_TILE)
                d0 = (block % (D // PREFILL_POST_D_TILE)) * PREFILL_POST_D_TILE
                hidden = pl.load(sublayer, [token, d0], [1, PREFILL_POST_D_TILE])
                update = pl.cast(hidden, pl.FP32)
                residual0 = pl.load(residual_flat, [token, d0], [1, PREFILL_POST_D_TILE])
                residual1 = pl.load(residual_flat, [token, D + d0], [1, PREFILL_POST_D_TILE])
                residual2 = pl.load(residual_flat, [token, 2 * D + d0], [1, PREFILL_POST_D_TILE])
                residual3 = pl.load(residual_flat, [token, 3 * D + d0], [1, PREFILL_POST_D_TILE])
                for out_h in pl.unroll(HC_MULT):
                    post_weight = pl.read(post_mix, [token, out_h])
                    weight0 = pl.read(residual_mix_flat, [token, out_h])
                    weight1 = pl.read(residual_mix_flat, [token, HC_MULT + out_h])
                    weight2 = pl.read(residual_mix_flat, [token, 2 * HC_MULT + out_h])
                    weight3 = pl.read(residual_mix_flat, [token, 3 * HC_MULT + out_h])
                    value = pl.mul(update, post_weight)
                    value = pl.add(value, pl.mul(residual0, weight0))
                    value = pl.add(value, pl.mul(residual1, weight1))
                    value = pl.add(value, pl.mul(residual2, weight2))
                    value = pl.add(value, pl.mul(residual3, weight3))
                    pl.store(value, [token, out_h * D + d0], output_flat)
    else:
        hc_post_decode(
            sublayer, residual, post_mix, residual_mix, output, deps=deps,
        )
    return output


@pl.jit.inline
def prefill_sp_post(
    sublayer: pl.Tensor[[SP_T_DYN, D], pl.BF16],
    residual: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    post_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[SP_T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    next_pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Run local mHC-post, writing active values and zero padding in one producer.

    A later pure-output clear is not an ordering edge for the post task. Each
    residual element is instead written exactly once, with hc_post's FP32
    accumulation order and FP32 output.
    """
    tokens = pl.tensor.dim(residual, 0)
    residual_flat = pl.reshape(residual, [tokens, HC_MULT * D])
    mix_flat = pl.reshape(residual_mix, [tokens, HC_MULT * HC_MULT])
    output_flat = pl.reshape(output, [tokens, HC_MULT * D])
    tiles = tokens * HC_MULT
    workers = pl.min(pl.min(tiles, PREFILL_POST_WORKERS), pl.system.available_aiv_count())
    for worker in pl.spmd(workers, name_hint="sp_mhc_post"):
        for block in pl.pipeline(worker, tiles, workers, stage=2):
            row = block // HC_MULT
            out_h = block % HC_MULT
            for col in pl.pipeline(0, D, PREFILL_POST_D_TILE, stage=2):
                value = pl.full([1, PREFILL_POST_D_TILE], dtype=pl.FP32, value=0.0)
                if tp_rank * tokens + row < num_tokens:
                    hidden = pl.cast(
                        sublayer[row:row + 1, col:col + PREFILL_POST_D_TILE], pl.FP32
                    )
                    update = pl.mul(hidden, pl.read(post_mix, [row, out_h]))
                    lane0 = residual_flat[row:row + 1, col:col + PREFILL_POST_D_TILE]
                    value = pl.mul(lane0, pl.read(mix_flat, [row, out_h]))
                    for index in pl.unroll(HC_MULT - 1):
                        in_h = index + 1
                        lane = residual_flat[
                            row:row + 1,
                            in_h * D + col:in_h * D + col + PREFILL_POST_D_TILE,
                        ]
                        value = pl.add(
                            value,
                            pl.mul(lane, pl.read(mix_flat, [row, in_h * HC_MULT + out_h])),
                        )
                    value = pl.add(value, update)
                output_flat[
                    row:row + 1,
                    out_h * D + col:out_h * D + col + PREFILL_POST_D_TILE,
                ] = value
    padding_tiles = tokens
    padding_workers = pl.max(1, pl.min(padding_tiles, pl.system.available_aiv_count()))
    for padding_worker in pl.spmd(padding_workers, name_hint="sp_mix_zero_padding"):
        for row in pl.range(padding_worker, padding_tiles, padding_workers):
            if tp_rank * tokens + row >= num_tokens:
                zero_mix = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                next_pre_mix[row:row + 1, :] = pl.set_validshape(zero_mix, 1, HC_MULT)
    return output, next_pre_mix


def golden_mhc_mixes(
    x_hc: torch.Tensor,
    function: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    iterations: int = M.hc_sinkhorn_iters,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project one HC stream into pre, post, and doubly-stochastic residual mixes."""
    flat = x_hc.flatten(-2).float()
    inverse_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + M.rms_norm_eps)
    mixes = F.linear(flat, function.float()) * inverse_rms
    hc = M.hc_mult
    pre_raw, post_raw, residual_raw = mixes.split((hc, hc, hc * hc), dim=-1)
    pre = torch.sigmoid(pre_raw * scale[0] + base[:hc]) + M.hc_eps
    post = 2 * torch.sigmoid(post_raw * scale[1] + base[hc : 2 * hc])
    residual_base = base[2 * hc :].unflatten(-1, (hc, hc))
    residual = (residual_raw.unflatten(-1, (hc, hc)) * scale[2] + residual_base).softmax(dim=-1)
    residual = residual + M.hc_eps
    residual = residual / (residual.sum(dim=-2, keepdim=True) + M.hc_eps)
    for _ in range(iterations - 1):
        residual = residual / (residual.sum(dim=-1, keepdim=True) + M.hc_eps)
        residual = residual / (residual.sum(dim=-2, keepdim=True) + M.hc_eps)
    return pre, post, residual


def golden_hc_collapse(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Collapse four HC streams with the kernel's pairwise FP32 reduction."""
    products = x.float() * pre_mix.float().unsqueeze(-1)
    first_pair = products[..., 0, :] + products[..., 1, :]
    second_pair = products[..., 2, :] + products[..., 3, :]
    return (first_pair + second_pair).to(x.dtype)


def golden_hc_post(
    sublayer: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    residual_mix: torch.Tensor,
) -> torch.Tensor:
    """Expand a sublayer result and mix the four residual streams in FP32."""
    update = post_mix.unsqueeze(-1) * sublayer.unsqueeze(-2)
    skip = (residual_mix.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=-3)
    return update.float() + skip.float()


def golden_l2_hc_pre(tensors):
    """Compute the four outputs with the incoming mix independent of the next mix."""
    pre_mix, post_mix, residual_mix = golden_mhc_mixes(
        tensors["x_hc"], tensors["function"], tensors["scale"], tensors["base"],
    )
    tensors["pre_mix"][:] = pre_mix
    tensors["post_mix"][:] = post_mix
    tensors["residual_mix"][:] = residual_mix
    collapsed = golden_hc_collapse(tensors["x_hc"], tensors["incoming_pre_mix"])
    tensors["output"][:] = collapsed.to(torch.bfloat16)


def golden_l2_hc_post(tensors):
    """Fill the expanded residual stream with FP32 accumulation and storage."""
    tensors["output"][:] = golden_hc_post(
        tensors["sublayer"], tensors["residual"], tensors["post_mix"], tensors["residual_mix"],
    )


def golden_l2_hc_head(tensors):
    """Fill the expected collapsed stream output."""
    tensors["output"][:] = golden_hc_collapse(tensors["x_hc"], tensors["pre_mix"]).to(torch.bfloat16)


HC_INPUT_NAMES = ("x_hc", "pre_mix", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight")


def make_hc_inputs(tokens, seed, case):
    """mHC inputs with distinct per-token magnitudes so both RMS statistics stay observable."""
    gen = torch.Generator().manual_seed(seed)
    magnitudes = (0.5 + torch.arange(tokens) % 4).reshape(-1, 1, 1)
    x_hc = (torch.randn(tokens, HC_MULT, D, generator=gen) * magnitudes).bfloat16().float()
    values = {
        "x_hc": x_hc,
        "pre_mix": torch.sigmoid(torch.randn(tokens, HC_MULT, generator=gen)) + M.hc_eps,
        "hc_attn_fn": torch.randn(MIX_HC, HC_DIM, generator=gen) / math.sqrt(HC_DIM),
        "hc_attn_scale": torch.randn(3, generator=gen),
        "hc_attn_base": torch.randn(MIX_HC, generator=gen),
        "attn_norm_weight": (torch.randn(D, generator=gen) * 0.1 + 1).bfloat16(),
    }
    if case == "zero":
        values["x_hc"] = torch.zeros_like(x_hc)
    return values


@pl.jit
def l2_hc_pre(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    function: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    scale: pl.Tensor[[3], pl.FP32],
    base: pl.Tensor[[MIX_HC], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    post_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    residual_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32]],
):
    """Validate the shared HcPre computation."""
    x_hc.bind_dynamic(0, T_DYN)
    incoming_pre_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    pre_mix.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    hc_pre(
        x_hc, incoming_pre_mix, function, scale, base, output,
        pre_mix, post_mix, residual_mix, prefill=True,
    )
    return output


def build_l2_hc_pre_specs(batch: int = 33, sequence: int = 1):
    """Build coefficient and independent incoming-mix fixtures."""
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(3)
    incoming_generator = torch.Generator().manual_seed(5)

    def init_x_hc():
        return torch.randn(tokens, HC_MULT, D, generator=generator)

    def init_incoming_pre_mix():
        return torch.sigmoid(torch.randn(tokens, HC_MULT, generator=incoming_generator)) + HC_EPS

    def init_function():
        return torch.randn(MIX_HC, HC_DIM, generator=generator) / math.sqrt(HC_DIM)

    def init_scale():
        return torch.randn(3, generator=generator)

    def init_base():
        return torch.randn(MIX_HC, generator=generator)

    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.float32, init_value=init_x_hc),
        TensorSpec("incoming_pre_mix", [tokens, HC_MULT], torch.float32, init_value=init_incoming_pre_mix),
        TensorSpec("function", [MIX_HC, HC_DIM], torch.float32, init_value=init_function),
        TensorSpec("scale", [3], torch.float32, init_value=init_scale),
        TensorSpec("base", [MIX_HC], torch.float32, init_value=init_base),
        TensorSpec("output", [tokens, D], torch.bfloat16),
        TensorSpec("pre_mix", [tokens, HC_MULT], torch.float32),
        TensorSpec("post_mix", [tokens, HC_MULT], torch.float32),
        TensorSpec("residual_mix", [tokens, HC_MULT, HC_MULT], torch.float32),
    ]


@pl.jit
def l2_hc_post(
    sublayer: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    """Validate HC post mapping and residual mixing."""
    sublayer.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    hc_post_deps = pl.array.create(2, pl.TASK_ID)
    hc_post(sublayer, residual, post_mix, residual_mix, output, deps=hc_post_deps, prefill=True)
    return output



@pl.jit
def l2_hc_post_decode(
    sublayer: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    """Validate the short-slab Decode mHC-post implementation."""
    sublayer.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    hc_post_deps = pl.array.create(2, pl.TASK_ID)
    hc_post_decode(sublayer, residual, post_mix, residual_mix, output, deps=hc_post_deps)
    return output


@pl.jit
def l2_hc_decode_pre(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    function: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    scale: pl.Tensor[[3], pl.FP32],
    base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    post_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    residual_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32]],
):
    """Validate the Decode mHC-pre path, including collapse and RMSNorm."""
    x_hc.bind_dynamic(0, T_DYN)
    incoming_pre_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    pre_mix.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    num_tokens = pl.cast(pl.tensor.dim(x_hc, 0), pl.INT32)
    hc_decode_mixes(x_hc, function, scale, base, pre_mix, post_mix, residual_mix, num_tokens)
    hc_head_norm_adaptive(x_hc, incoming_pre_mix, attn_norm_weight, output, num_tokens)
    return output


@pl.jit
def l2_hc_decode_pre_common(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    function: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    scale: pl.Tensor[[3], pl.FP32],
    base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    post_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    residual_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32]],
):
    """Reference Decode composition using the shared norm tile."""
    x_hc.bind_dynamic(0, T_DYN)
    incoming_pre_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    pre_mix.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    inv_rms, partials = hc_project(x_hc, function, one_token=False)
    hc_sinkhorn(inv_rms, partials, scale, base, pre_mix, post_mix, residual_mix, one_token=False)
    hc_head_norm(x_hc, incoming_pre_mix, attn_norm_weight, output, one_token=False)
    return output


def build_l2_hc_decode_pre_specs(batch: int = 1, sequence: int = 1):
    """Build deterministic short-slab Decode inputs for mHC pre validation."""
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(11)

    def init_x_hc():
        return torch.randn(tokens, HC_MULT, D, generator=generator)

    def init_incoming_pre_mix():
        return torch.sigmoid(torch.randn(tokens, HC_MULT, generator=generator)) + HC_EPS

    def init_function():
        return torch.randn(MIX_HC, HC_DIM, generator=generator) / math.sqrt(HC_DIM)

    def init_scale():
        return torch.randn(3, generator=generator)

    def init_base():
        return torch.randn(MIX_HC, generator=generator)

    def init_attn_norm_weight():
        return (torch.randn(D, generator=generator) * 0.1 + 1).bfloat16()

    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.float32, init_value=init_x_hc),
        TensorSpec("incoming_pre_mix", [tokens, HC_MULT], torch.float32, init_value=init_incoming_pre_mix),
        TensorSpec("function", [MIX_HC, HC_DIM], torch.float32, init_value=init_function),
        TensorSpec("scale", [3], torch.float32, init_value=init_scale),
        TensorSpec("base", [MIX_HC], torch.float32, init_value=init_base),
        TensorSpec("attn_norm_weight", [D], torch.bfloat16, init_value=init_attn_norm_weight),
        TensorSpec("output", [tokens, D], torch.bfloat16),
        TensorSpec("pre_mix", [tokens, HC_MULT], torch.float32),
        TensorSpec("post_mix", [tokens, HC_MULT], torch.float32),
        TensorSpec("residual_mix", [tokens, HC_MULT, HC_MULT], torch.float32),
    ]


def golden_l2_hc_decode_pre(tensors):
    """Compute Decode mHC coefficients and BF16 collapse plus RMSNorm."""
    pre_mix, post_mix, residual_mix = golden_mhc_mixes(
        tensors["x_hc"], tensors["function"], tensors["scale"], tensors["base"],
    )
    tensors["pre_mix"][:] = pre_mix
    tensors["post_mix"][:] = post_mix
    tensors["residual_mix"][:] = residual_mix
    collapsed = golden_hc_collapse(tensors["x_hc"], tensors["incoming_pre_mix"]).bfloat16()
    inverse = torch.rsqrt(collapsed.float().square().mean(dim=-1, keepdim=True) + NORM_EPS)
    tensors["output"][:] = (collapsed.float() * inverse * tensors["attn_norm_weight"].float()).bfloat16()


def build_l2_hc_post_specs(batch: int = 33, sequence: int = 1):
    """Build BF16-valued residual streams and nonsymmetric mixing coefficients."""
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(7)

    def init_sublayer():
        return torch.randn(tokens, D, generator=generator).bfloat16()

    def init_residual():
        return torch.randn(tokens, HC_MULT, D, generator=generator)

    def init_post_mix():
        return 2 * torch.sigmoid(torch.randn(tokens, HC_MULT, generator=generator))

    def init_residual_mix():
        mix = torch.rand(tokens, HC_MULT, HC_MULT, generator=generator) + HC_EPS
        for _ in range(HC_SINKHORN_ITER):
            mix = mix / (mix.sum(dim=-1, keepdim=True) + HC_EPS)
            mix = mix / (mix.sum(dim=-2, keepdim=True) + HC_EPS)
        return mix

    return [
        TensorSpec("sublayer", [tokens, D], torch.bfloat16, init_value=init_sublayer),
        TensorSpec("residual", [tokens, HC_MULT, D], torch.float32, init_value=init_residual),
        TensorSpec("post_mix", [tokens, HC_MULT], torch.float32, init_value=init_post_mix),
        TensorSpec("residual_mix", [tokens, HC_MULT, HC_MULT], torch.float32, init_value=init_residual_mix),
        TensorSpec("output", [tokens, HC_MULT, D], torch.float32),
    ]


@pl.jit
def l2_hc_head(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    """Validate the final HC stream collapse."""
    x_hc.bind_dynamic(0, T_DYN)
    pre_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    hc_head(x_hc, pre_mix, output)
    return output


def build_l2_hc_head_specs(batch: int = 33, sequence: int = 1):
    """Build deterministic inputs and output for stream-collapse validation."""
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(3)

    def init_x_hc():
        return torch.randn(tokens, HC_MULT, D, generator=generator)

    def init_pre_mix():
        return torch.rand(tokens, HC_MULT, generator=generator)

    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.float32, init_value=init_x_hc),
        TensorSpec("pre_mix", [tokens, HC_MULT], torch.float32, init_value=init_pre_mix),
        TensorSpec("output", [tokens, D], torch.bfloat16),
    ]


@pl.jit
def l2_hc_post_collapse(
    sublayer: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    incoming: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    collapsed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    """Validate both Decode outputs of the fused attention-to-FFN boundary."""
    sublayer.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    incoming.bind_dynamic(0, T_DYN)
    collapsed.bind_dynamic(0, T_DYN)
    hc_post_collapse(sublayer, residual, post_mix, residual_mix, incoming, output, collapsed)
    return output


def build_l2_hc_post_collapse_specs(batch: int = 1, sequence: int = 1):
    """Build Post inputs and the coefficients for the following collapse."""
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(11)
    return build_l2_hc_post_specs(batch, sequence) + [
        TensorSpec(
            "incoming", [tokens, HC_MULT], torch.float32,
            init_value=lambda: torch.rand(tokens, HC_MULT, generator=generator),
        ),
        TensorSpec("collapsed", [tokens, D], torch.bfloat16),
    ]


def golden_l2_hc_post_collapse(tensors):
    """Expand the FP32 residual streams before collapsing and casting to BF16."""
    golden_l2_hc_post(tensors)
    tensors["collapsed"][:] = golden_hc_collapse(tensors["output"], tensors["incoming"]).bfloat16()


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 HC pre, post, and head validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=33)
    parser.add_argument("--sequence", type=int, default=1)
    parser.add_argument(
        "--op",
        choices=["pre", "post", "head", "decode_pre", "decode_pre_common", "decode_post", "decode_post_collapse", "all"],
        default="all",
    )
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data")
    parser.add_argument("--runtime-dir")
    parser.add_argument("--enable-chip-swimlane", type=int, choices=[1, 2, 3, 4], default=0)
    args = parser.parse_args()
    if args.op == "all" and (args.golden_data or args.runtime_dir):
        parser.error("replay requires selecting one --op")
    cases = [
        ("pre", l2_hc_pre, build_l2_hc_pre_specs, golden_l2_hc_pre),
        ("post", l2_hc_post, build_l2_hc_post_specs, golden_l2_hc_post),
        ("head", l2_hc_head, build_l2_hc_head_specs, golden_l2_hc_head),
        ("decode_pre", l2_hc_decode_pre, build_l2_hc_decode_pre_specs, golden_l2_hc_decode_pre),
        ("decode_pre_common", l2_hc_decode_pre_common, build_l2_hc_decode_pre_specs, golden_l2_hc_decode_pre),
        ("decode_post", l2_hc_post_decode, build_l2_hc_post_specs, golden_l2_hc_post),
        ("decode_post_collapse", l2_hc_post_collapse, build_l2_hc_post_collapse_specs, golden_l2_hc_post_collapse),
    ]
    failures = []
    for name, fn, build_specs, golden_fn in cases:
        if args.op != "all" and args.op != name:
            continue
        print(f"[HC] {name}", flush=True)
        if name in ("pre", "decode_pre", "decode_pre_common"):
            compare_fn = {
                "pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
                "post_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
                "residual_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            }
            compare_fn["output"] = ratio_allclose(atol=1e-4, rtol=1.0 / 128)
        elif name in ("post", "decode_post"):
            compare_fn = {"output": ratio_allclose(atol=1e-6, rtol=1e-5)}
        elif name == "decode_post_collapse":
            compare_fn = {
                "output": ratio_allclose(atol=1e-6, rtol=1e-5),
                "collapsed": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            }
        else:
            compare_fn = {"output": ratio_allclose(atol=1e-4, rtol=1.0 / 128)}
        config = {"backend_type": args.backend_type, "execution_mode": args.execution_mode}
        config.update(device_id=args.device, enable_chip_swimlane=args.enable_chip_swimlane)
        result = run(
            fn=fn,
            specs=build_specs(args.batch, args.sequence),
            golden_fn=golden_fn,
            config=config,
            rtol=1e-3,
            atol=1e-3,
            compare_fn=compare_fn,
            compile_only=args.compile_only,
            save_data=args.save_data,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
        )
        if not result.passed:
            failures.append(f"{name}: {result.error or 'validation failed'}")
    if failures:
        raise SystemExit("; ".join(failures))
