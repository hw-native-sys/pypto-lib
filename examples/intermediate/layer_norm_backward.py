# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""LayerNorm backward — dx, dgamma, dbeta from dy, x, gamma; fp32. build() shape-dispatches among resident/streamed/striped variants (r005 fallback + r008 refresh); all are kept."""
import pypto.language as pl

DEFAULT_SHAPE = {"M": 512, "N": 1024, "dtype": "fp32"}


MICRO_ROWS = 8
GROUP_ROWS = 32
ROW_TILE = 8
COL_TILE = 128
REDUCE_COLS = 128
REDUCE_ROWS = 32
RESIDENT_TILE_BUDGET = 8192
MODE = "resident"
PHYSICAL_HIDDEN = 8
CONSUMER_COLS = 512
WIDE_DISPATCH = 1


@pl.jit
def r005_resident_backward(
    dy: pl.Tensor,
    x: pl.Tensor,
    gamma: pl.Tensor,
    dgamma: pl.Out[pl.Tensor],
    dbeta: pl.Out[pl.Tensor],
    dx: pl.Out[pl.Tensor],
):
    rows, hidden = x.shape
    groups = (rows + GROUP_ROWS - 1) // GROUP_ROWS
    partial_gamma = pl.create_tensor([groups, hidden], dtype=pl.FP32)
    partial_beta = pl.create_tensor([groups, hidden], dtype=pl.FP32)
    for group_start in pl.parallel(0, rows, GROUP_ROWS):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="r005_resident_backward"):
            gamma_acc = pl.full([1, PHYSICAL_HIDDEN], dtype=pl.FP32, value=0.0)
            beta_acc = pl.full([1, PHYSICAL_HIDDEN], dtype=pl.FP32, value=0.0)
            weight = pl.cast(pl.slice(gamma, [1, PHYSICAL_HIDDEN], [0, 0], valid_shape=[1, hidden]), pl.FP32)
            for row_offset in pl.range(0, pl.min(GROUP_ROWS, rows - group_start), MICRO_ROWS):
                row_start = group_start + row_offset
                valid_rows = pl.min(MICRO_ROWS, rows - row_start)
                tile_x = pl.slice(x, [MICRO_ROWS, PHYSICAL_HIDDEN], [row_start, 0], valid_shape=[valid_rows, hidden])
                tile_x = pl.cast(tile_x, pl.FP32)
                tile_dy = pl.slice(dy, [MICRO_ROWS, PHYSICAL_HIDDEN], [row_start, 0], valid_shape=[valid_rows, hidden])
                tile_dy = pl.cast(tile_dy, pl.FP32)
                mean = pl.reshape(pl.row_sum(tile_x), [1, MICRO_ROWS])
                mean = pl.reshape(pl.mul(mean, 1.0 / hidden), [MICRO_ROWS, 1])
                centered = pl.fillpad(pl.row_expand_sub(tile_x, mean), pad_value=pl.PadValue.zero)
                variance = pl.reshape(pl.row_sum(pl.mul(centered, centered)), [1, MICRO_ROWS])
                variance = pl.mul(variance, 1.0 / hidden)
                inv_std = pl.rsqrt(pl.add(variance, 1e-5), high_precision=True)
                inv_std = pl.reshape(inv_std, [MICRO_ROWS, 1])
                normalized = pl.row_expand_mul(centered, inv_std)
                weighted_dy = pl.col_expand_mul(tile_dy, weight)
                mean_dy = pl.reshape(pl.row_sum(weighted_dy), [1, MICRO_ROWS])
                mean_dy = pl.reshape(pl.mul(mean_dy, 1.0 / hidden), [MICRO_ROWS, 1])
                mean_projected = pl.reshape(pl.row_sum(pl.mul(weighted_dy, normalized)), [1, MICRO_ROWS])
                mean_projected = pl.reshape(pl.mul(mean_projected, 1.0 / hidden), [MICRO_ROWS, 1])
                correction = pl.row_expand_mul(normalized, mean_projected)
                residual = pl.sub(pl.row_expand_sub(weighted_dy, mean_dy), correction)
                output = pl.set_validshape(pl.row_expand_mul(residual, inv_std), valid_rows, hidden)
                dx = pl.assemble(dx, pl.cast(output, dx.dtype), [row_start, 0])
                gamma_acc = pl.add(gamma_acc, pl.col_sum(pl.mul(tile_dy, normalized)))
                beta_acc = pl.add(beta_acc, pl.col_sum(tile_dy))
            group_id = group_start // GROUP_ROWS
            partial_gamma = pl.assemble(partial_gamma, pl.set_validshape(gamma_acc, 1, hidden), [group_id, 0])
            partial_beta = pl.assemble(partial_beta, pl.set_validshape(beta_acc, 1, hidden), [group_id, 0])
    for col_start in pl.parallel(0, hidden, REDUCE_COLS):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="affine_reduce"):
            valid_cols = pl.min(REDUCE_COLS, hidden - col_start)
            gamma_acc = pl.full([1, REDUCE_COLS], dtype=pl.FP32, value=0.0)
            beta_acc = pl.full([1, REDUCE_COLS], dtype=pl.FP32, value=0.0)
            for group_start in pl.range(0, groups, REDUCE_ROWS):
                valid_groups = pl.min(REDUCE_ROWS, groups - group_start)
                gamma_chunk = pl.slice(partial_gamma, [REDUCE_ROWS, REDUCE_COLS], [group_start, col_start], valid_shape=[valid_groups, valid_cols])
                beta_chunk = pl.slice(partial_beta, [REDUCE_ROWS, REDUCE_COLS], [group_start, col_start], valid_shape=[valid_groups, valid_cols])
                gamma_acc = pl.add(gamma_acc, pl.col_sum(gamma_chunk))
                beta_acc = pl.add(beta_acc, pl.col_sum(beta_chunk))
            dgamma = pl.assemble(dgamma, pl.set_validshape(gamma_acc, 1, valid_cols), [0, col_start])
            dbeta = pl.assemble(dbeta, pl.set_validshape(beta_acc, 1, valid_cols), [0, col_start])
    return dx


@pl.jit
def r005_streamed_backward(
    dy: pl.Tensor,
    x: pl.Tensor,
    gamma: pl.Tensor,
    dgamma: pl.Out[pl.Tensor],
    dbeta: pl.Out[pl.Tensor],
    dx: pl.Out[pl.Tensor],
):
    rows, hidden = x.shape
    groups = (rows + ROW_TILE - 1) // ROW_TILE
    partial_gamma = pl.create_tensor([groups, hidden], dtype=pl.FP32)
    partial_beta = pl.create_tensor([groups, hidden], dtype=pl.FP32)
    for row_start in pl.parallel(0, rows, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="r005_streamed_backward"):
            valid_rows = pl.min(ROW_TILE, rows - row_start)
            mean_acc = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
            for col_start in pl.range(0, hidden, COL_TILE):
                valid_cols = pl.min(COL_TILE, hidden - col_start)
                tile_x = pl.slice(x, [ROW_TILE, COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
                tile_x = pl.cast(tile_x, pl.FP32)
                mean_acc = pl.add(mean_acc, pl.reshape(pl.row_sum(pl.mul(tile_x, 1.0 / hidden)), [1, ROW_TILE]))
            mean = pl.reshape(mean_acc, [ROW_TILE, 1])
            variance_acc = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
            weighted_acc = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
            projected_acc = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
            for col_start in pl.range(0, hidden, COL_TILE):
                valid_cols = pl.min(COL_TILE, hidden - col_start)
                tile_x = pl.slice(x, [ROW_TILE, COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
                tile_x = pl.cast(tile_x, pl.FP32)
                tile_dy = pl.slice(dy, [ROW_TILE, COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
                tile_dy = pl.cast(tile_dy, pl.FP32)
                weight = pl.slice(gamma, [1, COL_TILE], [0, col_start], valid_shape=[1, valid_cols])
                weight = pl.cast(weight, pl.FP32)
                centered = pl.fillpad(pl.row_expand_sub(tile_x, mean), pad_value=pl.PadValue.zero)
                weighted = pl.col_expand_mul(tile_dy, weight)
                variance_acc = pl.add(variance_acc, pl.reshape(pl.row_sum(pl.mul(centered, centered)), [1, ROW_TILE]))
                weighted_acc = pl.add(weighted_acc, pl.reshape(pl.row_sum(weighted), [1, ROW_TILE]))
                projected_acc = pl.add(projected_acc, pl.reshape(pl.row_sum(pl.mul(weighted, centered)), [1, ROW_TILE]))
            inv_std = pl.rsqrt(pl.add(pl.mul(variance_acc, 1.0 / hidden), 1e-5), high_precision=True)
            mean_weighted = pl.reshape(pl.mul(weighted_acc, 1.0 / hidden), [ROW_TILE, 1])
            mean_projected = pl.reshape(pl.mul(pl.mul(projected_acc, 1.0 / hidden), inv_std), [ROW_TILE, 1])
            inv_std = pl.reshape(inv_std, [ROW_TILE, 1])
            for col_start in pl.range(0, hidden, COL_TILE):
                valid_cols = pl.min(COL_TILE, hidden - col_start)
                tile_x = pl.slice(x, [ROW_TILE, COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
                tile_x = pl.cast(tile_x, pl.FP32)
                tile_dy = pl.slice(dy, [ROW_TILE, COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
                tile_dy = pl.cast(tile_dy, pl.FP32)
                weight = pl.slice(gamma, [1, COL_TILE], [0, col_start], valid_shape=[1, valid_cols])
                weight = pl.cast(weight, pl.FP32)
                normalized = pl.fillpad(pl.row_expand_mul(pl.row_expand_sub(tile_x, mean), inv_std), pad_value=pl.PadValue.zero)
                weighted = pl.col_expand_mul(tile_dy, weight)
                correction = pl.row_expand_mul(normalized, mean_projected)
                residual = pl.sub(pl.row_expand_sub(weighted, mean_weighted), correction)
                output = pl.set_validshape(pl.row_expand_mul(residual, inv_std), valid_rows, valid_cols)
                dx = pl.assemble(dx, pl.cast(output, dx.dtype), [row_start, col_start])
                gamma_part = pl.set_validshape(pl.col_sum(pl.mul(tile_dy, normalized)), 1, valid_cols)
                beta_part = pl.set_validshape(pl.col_sum(tile_dy), 1, valid_cols)
                partial_gamma = pl.assemble(partial_gamma, gamma_part, [row_start // ROW_TILE, col_start])
                partial_beta = pl.assemble(partial_beta, beta_part, [row_start // ROW_TILE, col_start])
    for col_start in pl.parallel(0, hidden, REDUCE_COLS):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="affine_reduce"):
            valid_cols = pl.min(REDUCE_COLS, hidden - col_start)
            gamma_acc = pl.full([1, REDUCE_COLS], dtype=pl.FP32, value=0.0)
            beta_acc = pl.full([1, REDUCE_COLS], dtype=pl.FP32, value=0.0)
            for group_start in pl.range(0, groups, REDUCE_ROWS):
                valid_groups = pl.min(REDUCE_ROWS, groups - group_start)
                gamma_chunk = pl.slice(partial_gamma, [REDUCE_ROWS, REDUCE_COLS], [group_start, col_start], valid_shape=[valid_groups, valid_cols])
                beta_chunk = pl.slice(partial_beta, [REDUCE_ROWS, REDUCE_COLS], [group_start, col_start], valid_shape=[valid_groups, valid_cols])
                gamma_acc = pl.add(gamma_acc, pl.col_sum(gamma_chunk))
                beta_acc = pl.add(beta_acc, pl.col_sum(beta_chunk))
            dgamma = pl.assemble(dgamma, pl.set_validshape(gamma_acc, 1, valid_cols), [0, col_start])
            dbeta = pl.assemble(dbeta, pl.set_validshape(beta_acc, 1, valid_cols), [0, col_start])
    return dx


@pl.jit
def r005_single_column_backward(
    dy: pl.Tensor,
    x: pl.Tensor,
    gamma: pl.Tensor,
    dgamma: pl.Out[pl.Tensor],
    dbeta: pl.Out[pl.Tensor],
    dx: pl.Out[pl.Tensor],
):
    rows, hidden = x.shape
    flat_dy = pl.reshape(dy, [1, rows])
    flat_dx = pl.reshape(dx, [1, rows])
    for task in pl.parallel(1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="r005_single_column_backward"):
            beta_acc = pl.full([8, 32], dtype=pl.FP32, value=0.0)
            for row_start in pl.range(0, rows, 32):
                valid_cols = pl.min(32, rows - row_start)
                upstream = pl.cast(pl.slice(flat_dy, [8, 32], [0, row_start], valid_shape=[1, valid_cols]), pl.FP32)
                beta_acc = pl.add(beta_acc, pl.fillpad(upstream, pad_value=pl.PadValue.zero))
                zeros = pl.full([1, 32], dtype=pl.FP32, value=0.0)
                flat_dx = pl.assemble(flat_dx, pl.set_validshape(pl.cast(zeros, pl.BF16), 1, valid_cols), [0, row_start])
            beta_sum = pl.reshape(pl.row_sum(beta_acc), [1, 8])
            dgamma = pl.assemble(dgamma, pl.set_validshape(pl.full([1, 8], dtype=pl.FP32, value=0.0), 1, 1), [0, 0])
            dbeta = pl.assemble(dbeta, pl.set_validshape(beta_sum, 1, 1), [0, 0])
    dx = pl.reshape(flat_dx, [rows, 1])
    return dx


@pl.jit
def r005_resident_backward_fp32(
    dy: pl.Tensor,
    x: pl.Tensor,
    gamma: pl.Tensor,
    dgamma: pl.Out[pl.Tensor],
    dbeta: pl.Out[pl.Tensor],
    dx: pl.Out[pl.Tensor],
):
    rows, hidden = x.shape
    groups = (rows + GROUP_ROWS - 1) // GROUP_ROWS
    partial_gamma = pl.create_tensor([groups, hidden], dtype=pl.FP32)
    partial_beta = pl.create_tensor([groups, hidden], dtype=pl.FP32)
    for group_start in pl.parallel(0, rows, GROUP_ROWS):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="r005_resident_backward"):
            gamma_acc = pl.full([1, PHYSICAL_HIDDEN], dtype=pl.FP32, value=0.0)
            beta_acc = pl.full([1, PHYSICAL_HIDDEN], dtype=pl.FP32, value=0.0)
            weight = pl.slice(gamma, [1, PHYSICAL_HIDDEN], [0, 0], valid_shape=[1, hidden])
            for row_offset in pl.range(0, pl.min(GROUP_ROWS, rows - group_start), MICRO_ROWS):
                row_start = group_start + row_offset
                valid_rows = pl.min(MICRO_ROWS, rows - row_start)
                tile_x = pl.slice(x, [MICRO_ROWS, PHYSICAL_HIDDEN], [row_start, 0], valid_shape=[valid_rows, hidden])
                tile_x = tile_x
                tile_dy = pl.slice(dy, [MICRO_ROWS, PHYSICAL_HIDDEN], [row_start, 0], valid_shape=[valid_rows, hidden])
                tile_dy = tile_dy
                mean = pl.reshape(pl.row_sum(tile_x), [1, MICRO_ROWS])
                mean = pl.reshape(pl.mul(mean, 1.0 / hidden), [MICRO_ROWS, 1])
                centered = pl.fillpad(pl.row_expand_sub(tile_x, mean), pad_value=pl.PadValue.zero)
                variance = pl.reshape(pl.row_sum(pl.mul(centered, centered)), [1, MICRO_ROWS])
                variance = pl.mul(variance, 1.0 / hidden)
                inv_std = pl.rsqrt(pl.add(variance, 1e-5), high_precision=True)
                inv_std = pl.reshape(inv_std, [MICRO_ROWS, 1])
                normalized = pl.row_expand_mul(centered, inv_std)
                weighted_dy = pl.col_expand_mul(tile_dy, weight)
                mean_dy = pl.reshape(pl.row_sum(weighted_dy), [1, MICRO_ROWS])
                mean_dy = pl.reshape(pl.mul(mean_dy, 1.0 / hidden), [MICRO_ROWS, 1])
                mean_projected = pl.reshape(pl.row_sum(pl.mul(weighted_dy, normalized)), [1, MICRO_ROWS])
                mean_projected = pl.reshape(pl.mul(mean_projected, 1.0 / hidden), [MICRO_ROWS, 1])
                correction = pl.row_expand_mul(normalized, mean_projected)
                residual = pl.sub(pl.row_expand_sub(weighted_dy, mean_dy), correction)
                output = pl.set_validshape(pl.row_expand_mul(residual, inv_std), valid_rows, hidden)
                dx = pl.assemble(dx, output, [row_start, 0])
                gamma_acc = pl.add(gamma_acc, pl.col_sum(pl.mul(tile_dy, normalized)))
                beta_acc = pl.add(beta_acc, pl.col_sum(tile_dy))
            group_id = group_start // GROUP_ROWS
            partial_gamma = pl.assemble(partial_gamma, pl.set_validshape(gamma_acc, 1, hidden), [group_id, 0])
            partial_beta = pl.assemble(partial_beta, pl.set_validshape(beta_acc, 1, hidden), [group_id, 0])
    for col_start in pl.parallel(0, hidden, REDUCE_COLS):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="affine_reduce"):
            valid_cols = pl.min(REDUCE_COLS, hidden - col_start)
            gamma_acc = pl.full([1, REDUCE_COLS], dtype=pl.FP32, value=0.0)
            beta_acc = pl.full([1, REDUCE_COLS], dtype=pl.FP32, value=0.0)
            for group_start in pl.range(0, groups, REDUCE_ROWS):
                valid_groups = pl.min(REDUCE_ROWS, groups - group_start)
                gamma_chunk = pl.slice(partial_gamma, [REDUCE_ROWS, REDUCE_COLS], [group_start, col_start], valid_shape=[valid_groups, valid_cols])
                beta_chunk = pl.slice(partial_beta, [REDUCE_ROWS, REDUCE_COLS], [group_start, col_start], valid_shape=[valid_groups, valid_cols])
                gamma_acc = pl.add(gamma_acc, pl.col_sum(gamma_chunk))
                beta_acc = pl.add(beta_acc, pl.col_sum(beta_chunk))
            dgamma = pl.assemble(dgamma, pl.set_validshape(gamma_acc, 1, valid_cols), [0, col_start])
            dbeta = pl.assemble(dbeta, pl.set_validshape(beta_acc, 1, valid_cols), [0, col_start])
    return dx


@pl.jit
def r005_streamed_backward_fp32(
    dy: pl.Tensor,
    x: pl.Tensor,
    gamma: pl.Tensor,
    dgamma: pl.Out[pl.Tensor],
    dbeta: pl.Out[pl.Tensor],
    dx: pl.Out[pl.Tensor],
):
    rows, hidden = x.shape
    groups = (rows + ROW_TILE - 1) // ROW_TILE
    partial_gamma = pl.create_tensor([groups, hidden], dtype=pl.FP32)
    partial_beta = pl.create_tensor([groups, hidden], dtype=pl.FP32)
    for row_start in pl.parallel(0, rows, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="r005_streamed_backward"):
            valid_rows = pl.min(ROW_TILE, rows - row_start)
            mean_acc = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
            for col_start in pl.range(0, hidden, COL_TILE):
                valid_cols = pl.min(COL_TILE, hidden - col_start)
                tile_x = pl.slice(x, [ROW_TILE, COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
                tile_x = tile_x
                mean_acc = pl.add(mean_acc, pl.reshape(pl.row_sum(pl.mul(tile_x, 1.0 / hidden)), [1, ROW_TILE]))
            mean = pl.reshape(mean_acc, [ROW_TILE, 1])
            variance_acc = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
            weighted_acc = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
            projected_acc = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
            for col_start in pl.range(0, hidden, COL_TILE):
                valid_cols = pl.min(COL_TILE, hidden - col_start)
                tile_x = pl.slice(x, [ROW_TILE, COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
                tile_x = tile_x
                tile_dy = pl.slice(dy, [ROW_TILE, COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
                tile_dy = tile_dy
                weight = pl.slice(gamma, [1, COL_TILE], [0, col_start], valid_shape=[1, valid_cols])
                weight = weight
                centered = pl.fillpad(pl.row_expand_sub(tile_x, mean), pad_value=pl.PadValue.zero)
                weighted = pl.col_expand_mul(tile_dy, weight)
                variance_acc = pl.add(variance_acc, pl.reshape(pl.row_sum(pl.mul(centered, centered)), [1, ROW_TILE]))
                weighted_acc = pl.add(weighted_acc, pl.reshape(pl.row_sum(weighted), [1, ROW_TILE]))
                projected_acc = pl.add(projected_acc, pl.reshape(pl.row_sum(pl.mul(weighted, centered)), [1, ROW_TILE]))
            inv_std = pl.rsqrt(pl.add(pl.mul(variance_acc, 1.0 / hidden), 1e-5), high_precision=True)
            mean_weighted = pl.reshape(pl.mul(weighted_acc, 1.0 / hidden), [ROW_TILE, 1])
            mean_projected = pl.reshape(pl.mul(pl.mul(projected_acc, 1.0 / hidden), inv_std), [ROW_TILE, 1])
            inv_std = pl.reshape(inv_std, [ROW_TILE, 1])
            for col_start in pl.range(0, hidden, COL_TILE):
                valid_cols = pl.min(COL_TILE, hidden - col_start)
                tile_x = pl.slice(x, [ROW_TILE, COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
                tile_x = tile_x
                tile_dy = pl.slice(dy, [ROW_TILE, COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
                tile_dy = tile_dy
                weight = pl.slice(gamma, [1, COL_TILE], [0, col_start], valid_shape=[1, valid_cols])
                weight = weight
                normalized = pl.fillpad(pl.row_expand_mul(pl.row_expand_sub(tile_x, mean), inv_std), pad_value=pl.PadValue.zero)
                weighted = pl.col_expand_mul(tile_dy, weight)
                correction = pl.row_expand_mul(normalized, mean_projected)
                residual = pl.sub(pl.row_expand_sub(weighted, mean_weighted), correction)
                output = pl.set_validshape(pl.row_expand_mul(residual, inv_std), valid_rows, valid_cols)
                dx = pl.assemble(dx, output, [row_start, col_start])
                gamma_part = pl.set_validshape(pl.col_sum(pl.mul(tile_dy, normalized)), 1, valid_cols)
                beta_part = pl.set_validshape(pl.col_sum(tile_dy), 1, valid_cols)
                partial_gamma = pl.assemble(partial_gamma, gamma_part, [row_start // ROW_TILE, col_start])
                partial_beta = pl.assemble(partial_beta, beta_part, [row_start // ROW_TILE, col_start])
    for col_start in pl.parallel(0, hidden, REDUCE_COLS):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="affine_reduce"):
            valid_cols = pl.min(REDUCE_COLS, hidden - col_start)
            gamma_acc = pl.full([1, REDUCE_COLS], dtype=pl.FP32, value=0.0)
            beta_acc = pl.full([1, REDUCE_COLS], dtype=pl.FP32, value=0.0)
            for group_start in pl.range(0, groups, REDUCE_ROWS):
                valid_groups = pl.min(REDUCE_ROWS, groups - group_start)
                gamma_chunk = pl.slice(partial_gamma, [REDUCE_ROWS, REDUCE_COLS], [group_start, col_start], valid_shape=[valid_groups, valid_cols])
                beta_chunk = pl.slice(partial_beta, [REDUCE_ROWS, REDUCE_COLS], [group_start, col_start], valid_shape=[valid_groups, valid_cols])
                gamma_acc = pl.add(gamma_acc, pl.col_sum(gamma_chunk))
                beta_acc = pl.add(beta_acc, pl.col_sum(beta_chunk))
            dgamma = pl.assemble(dgamma, pl.set_validshape(gamma_acc, 1, valid_cols), [0, col_start])
            dbeta = pl.assemble(dbeta, pl.set_validshape(beta_acc, 1, valid_cols), [0, col_start])
    return dx


@pl.jit
def r005_single_column_backward_fp32(
    dy: pl.Tensor,
    x: pl.Tensor,
    gamma: pl.Tensor,
    dgamma: pl.Out[pl.Tensor],
    dbeta: pl.Out[pl.Tensor],
    dx: pl.Out[pl.Tensor],
):
    rows, hidden = x.shape
    flat_dy = pl.reshape(dy, [1, rows])
    flat_dx = pl.reshape(dx, [1, rows])
    for task in pl.parallel(1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="r005_single_column_backward"):
            beta_acc = pl.full([8, 32], dtype=pl.FP32, value=0.0)
            for row_start in pl.range(0, rows, 32):
                valid_cols = pl.min(32, rows - row_start)
                upstream = pl.slice(flat_dy, [8, 32], [0, row_start], valid_shape=[1, valid_cols])
                beta_acc = pl.add(beta_acc, pl.fillpad(upstream, pad_value=pl.PadValue.zero))
                zeros = pl.full([1, 32], dtype=pl.FP32, value=0.0)
                flat_dx = pl.assemble(flat_dx, pl.set_validshape(zeros, 1, valid_cols), [0, row_start])
            beta_sum = pl.reshape(pl.row_sum(beta_acc), [1, 8])
            dgamma = pl.assemble(dgamma, pl.set_validshape(pl.full([1, 8], dtype=pl.FP32, value=0.0), 1, 1), [0, 0])
            dbeta = pl.assemble(dbeta, pl.set_validshape(beta_sum, 1, 1), [0, 0])
    dx = pl.reshape(flat_dx, [rows, 1])
    return dx


@pl.jit
def r005_wide_striped_backward(
    dy: pl.Tensor,
    x: pl.Tensor,
    gamma: pl.Tensor,
    dgamma: pl.Out[pl.Tensor],
    dbeta: pl.Out[pl.Tensor],
    dx: pl.Out[pl.Tensor],
):
    rows, hidden = x.shape
    groups = (rows + GROUP_ROWS - 1) // GROUP_ROWS
    partial_gamma = pl.create_tensor([groups, hidden], dtype=pl.FP32)
    partial_beta = pl.create_tensor([groups, hidden], dtype=pl.FP32)
    for group_start in pl.parallel(0, rows, GROUP_ROWS):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="r005_wide_striped_backward"):
            gamma_acc = pl.full([1, PHYSICAL_HIDDEN], dtype=pl.FP32, value=0.0)
            beta_acc = pl.full([1, PHYSICAL_HIDDEN], dtype=pl.FP32, value=0.0)
            weight = pl.cast(pl.slice(gamma, [1, PHYSICAL_HIDDEN], [0, 0], valid_shape=[1, hidden]), pl.FP32)
            for row_offset in pl.range(0, pl.min(GROUP_ROWS, rows - group_start), MICRO_ROWS):
                row_start = group_start + row_offset
                valid_rows = pl.min(MICRO_ROWS, rows - row_start)
                tile_x = pl.slice(x, [MICRO_ROWS, PHYSICAL_HIDDEN], [row_start, 0], valid_shape=[valid_rows, hidden])
                tile_x = pl.cast(tile_x, pl.FP32)
                mean = pl.row_sum(pl.mul(tile_x, 1.0 / hidden))
                centered = pl.fillpad(pl.row_expand_sub(tile_x, mean), pad_value=pl.PadValue.zero)
                variance = pl.row_sum(pl.mul(pl.mul(centered, centered), 1.0 / hidden))
                variance = pl.reshape(variance, [1, MICRO_ROWS])
                inv_std = pl.rsqrt(pl.add(variance, 1e-5), high_precision=True)
                inv_std = pl.reshape(inv_std, [MICRO_ROWS, 1])
                normalized = pl.row_expand_mul(centered, inv_std)
                stored_dy = pl.slice(dy, [MICRO_ROWS, PHYSICAL_HIDDEN], [row_start, 0], valid_shape=[valid_rows, hidden])
                weighted_acc = pl.full([1, MICRO_ROWS], dtype=pl.FP32, value=0.0)
                projected_acc = pl.full([1, MICRO_ROWS], dtype=pl.FP32, value=0.0)
                for col_start in pl.unroll(0, hidden, CONSUMER_COLS):
                    valid_cols = pl.min(CONSUMER_COLS, hidden - col_start)
                    normalized_stripe = pl.slice(normalized, [MICRO_ROWS, CONSUMER_COLS], [0, col_start], valid_shape=[valid_rows, valid_cols])
                    dy_stripe = pl.cast(pl.slice(stored_dy, [MICRO_ROWS, CONSUMER_COLS], [0, col_start], valid_shape=[valid_rows, valid_cols]), pl.FP32)
                    weight_stripe = pl.slice(weight, [1, CONSUMER_COLS], [0, col_start], valid_shape=[1, valid_cols])
                    beta_stripe = pl.slice(beta_acc, [1, CONSUMER_COLS], [0, col_start])
                    gamma_stripe = pl.slice(gamma_acc, [1, CONSUMER_COLS], [0, col_start])
                    beta_stripe = pl.add(beta_stripe, pl.col_sum(dy_stripe))
                    gamma_stripe = pl.add(gamma_stripe, pl.col_sum(pl.mul(dy_stripe, normalized_stripe)))
                    beta_acc = pl.assemble(beta_acc, beta_stripe, [0, col_start])
                    gamma_acc = pl.assemble(gamma_acc, gamma_stripe, [0, col_start])
                    weighted = pl.col_expand_mul(dy_stripe, weight_stripe)
                    weighted_acc = pl.add(weighted_acc, pl.reshape(pl.row_sum(weighted), [1, MICRO_ROWS]))
                    projected_acc = pl.add(projected_acc, pl.reshape(pl.row_sum(pl.mul(weighted, normalized_stripe)), [1, MICRO_ROWS]))
                mean_dy = pl.reshape(pl.mul(weighted_acc, 1.0 / hidden), [MICRO_ROWS, 1])
                mean_projected = pl.reshape(pl.mul(projected_acc, 1.0 / hidden), [MICRO_ROWS, 1])
                for col_start in pl.unroll(0, hidden, CONSUMER_COLS):
                    valid_cols = pl.min(CONSUMER_COLS, hidden - col_start)
                    normalized_stripe = pl.slice(normalized, [MICRO_ROWS, CONSUMER_COLS], [0, col_start], valid_shape=[valid_rows, valid_cols])
                    dy_stripe = pl.cast(pl.slice(stored_dy, [MICRO_ROWS, CONSUMER_COLS], [0, col_start], valid_shape=[valid_rows, valid_cols]), pl.FP32)
                    weight_stripe = pl.slice(weight, [1, CONSUMER_COLS], [0, col_start], valid_shape=[1, valid_cols])
                    weighted = pl.col_expand_mul(dy_stripe, weight_stripe)
                    correction = pl.row_expand_mul(normalized_stripe, mean_projected)
                    residual = pl.sub(pl.row_expand_sub(weighted, mean_dy), correction)
                    output = pl.set_validshape(pl.row_expand_mul(residual, inv_std), valid_rows, valid_cols)
                    dx = pl.assemble(dx, pl.cast(output, dx.dtype), [row_start, col_start])
            group_id = group_start // GROUP_ROWS
            partial_gamma = pl.assemble(partial_gamma, pl.set_validshape(gamma_acc, 1, hidden), [group_id, 0])
            partial_beta = pl.assemble(partial_beta, pl.set_validshape(beta_acc, 1, hidden), [group_id, 0])
    for col_start in pl.parallel(0, hidden, REDUCE_COLS):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="affine_reduce"):
            valid_cols = pl.min(REDUCE_COLS, hidden - col_start)
            gamma_acc = pl.full([1, REDUCE_COLS], dtype=pl.FP32, value=0.0)
            beta_acc = pl.full([1, REDUCE_COLS], dtype=pl.FP32, value=0.0)
            for group_start in pl.range(0, groups, REDUCE_ROWS):
                valid_groups = pl.min(REDUCE_ROWS, groups - group_start)
                gamma_chunk = pl.slice(partial_gamma, [REDUCE_ROWS, REDUCE_COLS], [group_start, col_start], valid_shape=[valid_groups, valid_cols])
                beta_chunk = pl.slice(partial_beta, [REDUCE_ROWS, REDUCE_COLS], [group_start, col_start], valid_shape=[valid_groups, valid_cols])
                gamma_acc = pl.add(gamma_acc, pl.col_sum(gamma_chunk))
                beta_acc = pl.add(beta_acc, pl.col_sum(beta_chunk))
            dgamma = pl.assemble(dgamma, pl.set_validshape(gamma_acc, 1, valid_cols), [0, col_start])
            dbeta = pl.assemble(dbeta, pl.set_validshape(beta_acc, 1, valid_cols), [0, col_start])
    return dx


def golden_backward(tensors):
    import torch

    inputs = tensors["x"].float()
    upstream = tensors["dy"].float()
    weights = tensors["gamma"].float()
    mean = inputs.mean(dim=-1, keepdim=True)
    variance = inputs.var(dim=-1, keepdim=True, unbiased=False)
    inv_std = torch.rsqrt(variance + 1e-5)
    normalized = (inputs - mean) * inv_std
    weighted = upstream * weights
    tensors["dx"][:] = inv_std * (
        weighted - weighted.mean(dim=-1, keepdim=True)
        - normalized * (weighted * normalized).mean(dim=-1, keepdim=True)
    )
    tensors["dgamma"][:] = (upstream * normalized).sum(dim=0, keepdim=True)
    tensors["dbeta"][:] = upstream.sum(dim=0, keepdim=True)


def build_fallback(shape):
    import torch
    from golden import TensorSpec

    global PHYSICAL_HIDDEN
    rows, hidden = int(shape["M"]), int(shape["N"])
    if rows < 1 or hidden < 1 or shape["dtype"] not in ("fp32", "bf16"):
        raise ValueError("This campaign requires positive dimensions and fp32 inputs")
    PHYSICAL_HIDDEN = ((hidden + 15) // 16) * 16
    input_dtype = torch.bfloat16 if shape["dtype"] == "bf16" else torch.float32
    specs = [
        TensorSpec("dy", [rows, hidden], input_dtype, init_value=torch.randn),
        TensorSpec("x", [rows, hidden], input_dtype, init_value=torch.randn),
        TensorSpec("gamma", [1, hidden], input_dtype, init_value=torch.randn),
        TensorSpec("dgamma", [1, hidden], torch.float32),
        TensorSpec("dbeta", [1, hidden], torch.float32),
        TensorSpec("dx", [rows, hidden], input_dtype),
    ]
    fn = r005_single_column_backward if hidden == 1 else (r005_resident_backward if PHYSICAL_HIDDEN <= 1024 else r005_streamed_backward)
    if shape["dtype"] == "fp32":
        fn = r005_single_column_backward_fp32 if hidden == 1 else (r005_resident_backward_fp32 if PHYSICAL_HIDDEN <= 1024 else r005_streamed_backward_fp32)
    if WIDE_DISPATCH and shape["dtype"] == "bf16" and 1024 < hidden <= 1536:
        PHYSICAL_HIDDEN = ((hidden + CONSUMER_COLS - 1) // CONSUMER_COLS) * CONSUMER_COLS
        fn = r005_wide_striped_backward
    tolerance = 2e-2 if shape["dtype"] == "bf16" else 1e-4
    return fn, specs, golden_backward, tolerance, tolerance


INIT_COLS = 128


@pl.jit
def r008_resident_backward(
    dy: pl.Tensor,
    x: pl.Tensor,
    gamma: pl.Tensor,
    dgamma: pl.Out[pl.Tensor],
    dbeta: pl.Out[pl.Tensor],
    dx: pl.Out[pl.Tensor],
):
    rows, hidden = x.shape
    groups = (rows + GROUP_ROWS - 1) // GROUP_ROWS
    initializers = (hidden + INIT_COLS - 1) // INIT_COLS
    with pl.spmd(initializers, name_hint="affine_zero") as zero_tid:
        init_start = pl.tile.get_block_idx() * INIT_COLS
        init_valid = pl.min(INIT_COLS, hidden - init_start)
        zeros = pl.set_validshape(pl.full([1, INIT_COLS], dtype=pl.FP32, value=0.0), 1, init_valid)
        dgamma = pl.assemble(dgamma, zeros, [0, init_start])
        dbeta = pl.assemble(dbeta, zeros, [0, init_start])
    with pl.spmd(groups, name_hint="fused_centered_atomic", deps=[zero_tid]) as owner_tid:
        group_start = pl.tile.get_block_idx() * GROUP_ROWS
        gamma_acc = pl.full([1, PHYSICAL_HIDDEN], dtype=pl.FP32, value=0.0)
        beta_acc = pl.full([1, PHYSICAL_HIDDEN], dtype=pl.FP32, value=0.0)
        weight = pl.cast(pl.slice(gamma, [1, PHYSICAL_HIDDEN], [0, 0], valid_shape=[1, hidden]), pl.FP32)
        for row_offset in pl.range(0, pl.min(GROUP_ROWS, rows - group_start), MICRO_ROWS):
            row_start = group_start + row_offset
            valid_rows = pl.min(MICRO_ROWS, rows - row_start)
            tile_x = pl.slice(x, [MICRO_ROWS, PHYSICAL_HIDDEN], [row_start, 0], valid_shape=[valid_rows, hidden])
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_dy = pl.slice(dy, [MICRO_ROWS, PHYSICAL_HIDDEN], [row_start, 0], valid_shape=[valid_rows, hidden])
            tile_dy = pl.cast(tile_dy, pl.FP32)
            mean = pl.reshape(pl.row_sum(tile_x), [1, MICRO_ROWS])
            mean = pl.reshape(pl.mul(mean, 1.0 / hidden), [MICRO_ROWS, 1])
            centered = pl.fillpad(pl.row_expand_sub(tile_x, mean), pad_value=pl.PadValue.zero)
            variance = pl.reshape(pl.row_sum(pl.mul(centered, centered)), [1, MICRO_ROWS])
            variance = pl.mul(variance, 1.0 / hidden)
            inv_std = pl.rsqrt(pl.add(variance, 1e-5), high_precision=True)
            inv_std = pl.reshape(inv_std, [MICRO_ROWS, 1])
            normalized = pl.row_expand_mul(centered, inv_std)
            weighted_dy = pl.col_expand_mul(tile_dy, weight)
            mean_dy = pl.reshape(pl.row_sum(weighted_dy), [1, MICRO_ROWS])
            mean_dy = pl.reshape(pl.mul(mean_dy, 1.0 / hidden), [MICRO_ROWS, 1])
            mean_projected = pl.reshape(pl.row_sum(pl.mul(weighted_dy, normalized)), [1, MICRO_ROWS])
            mean_projected = pl.reshape(pl.mul(mean_projected, 1.0 / hidden), [MICRO_ROWS, 1])
            correction = pl.row_expand_mul(normalized, mean_projected)
            residual = pl.sub(pl.row_expand_sub(weighted_dy, mean_dy), correction)
            output = pl.set_validshape(pl.row_expand_mul(residual, inv_std), valid_rows, hidden)
            dx = pl.assemble(dx, pl.cast(output, dx.dtype), [row_start, 0])
            gamma_acc = pl.add(gamma_acc, pl.col_sum(pl.mul(tile_dy, normalized)))
            beta_acc = pl.add(beta_acc, pl.col_sum(tile_dy))
        dgamma = pl.assemble(dgamma, pl.set_validshape(gamma_acc, 1, hidden), [0, 0], atomic=pl.AtomicType.Add)
        dbeta = pl.assemble(dbeta, pl.set_validshape(beta_acc, 1, hidden), [0, 0], atomic=pl.AtomicType.Add)
    return dx

@pl.jit
def r008_wide_striped_backward(
    dy: pl.Tensor,
    x: pl.Tensor,
    gamma: pl.Tensor,
    dgamma: pl.Out[pl.Tensor],
    dbeta: pl.Out[pl.Tensor],
    dx: pl.Out[pl.Tensor],
):
    rows, hidden = x.shape
    groups = (rows + GROUP_ROWS - 1) // GROUP_ROWS
    initializers = (hidden + INIT_COLS - 1) // INIT_COLS
    with pl.spmd(initializers, name_hint="affine_zero") as zero_tid:
        init_start = pl.tile.get_block_idx() * INIT_COLS
        init_valid = pl.min(INIT_COLS, hidden - init_start)
        zeros = pl.set_validshape(pl.full([1, INIT_COLS], dtype=pl.FP32, value=0.0), 1, init_valid)
        dgamma = pl.assemble(dgamma, zeros, [0, init_start])
        dbeta = pl.assemble(dbeta, zeros, [0, init_start])
    with pl.spmd(groups, name_hint="fused_centered_atomic", deps=[zero_tid]) as owner_tid:
        group_start = pl.tile.get_block_idx() * GROUP_ROWS
        gamma_acc = pl.full([1, PHYSICAL_HIDDEN], dtype=pl.FP32, value=0.0)
        beta_acc = pl.full([1, PHYSICAL_HIDDEN], dtype=pl.FP32, value=0.0)
        weight = pl.cast(pl.slice(gamma, [1, PHYSICAL_HIDDEN], [0, 0], valid_shape=[1, hidden]), pl.FP32)
        for row_offset in pl.range(0, pl.min(GROUP_ROWS, rows - group_start), MICRO_ROWS):
            row_start = group_start + row_offset
            valid_rows = pl.min(MICRO_ROWS, rows - row_start)
            tile_x = pl.slice(x, [MICRO_ROWS, PHYSICAL_HIDDEN], [row_start, 0], valid_shape=[valid_rows, hidden])
            tile_x = pl.cast(tile_x, pl.FP32)
            mean = pl.row_sum(pl.mul(tile_x, 1.0 / hidden))
            centered = pl.fillpad(pl.row_expand_sub(tile_x, mean), pad_value=pl.PadValue.zero)
            variance = pl.row_sum(pl.mul(pl.mul(centered, centered), 1.0 / hidden))
            variance = pl.reshape(variance, [1, MICRO_ROWS])
            inv_std = pl.rsqrt(pl.add(variance, 1e-5), high_precision=True)
            inv_std = pl.reshape(inv_std, [MICRO_ROWS, 1])
            normalized = pl.row_expand_mul(centered, inv_std)
            stored_dy = pl.slice(dy, [MICRO_ROWS, PHYSICAL_HIDDEN], [row_start, 0], valid_shape=[valid_rows, hidden])
            weighted_acc = pl.full([1, MICRO_ROWS], dtype=pl.FP32, value=0.0)
            projected_acc = pl.full([1, MICRO_ROWS], dtype=pl.FP32, value=0.0)
            for col_start in pl.unroll(0, hidden, CONSUMER_COLS):
                valid_cols = pl.min(CONSUMER_COLS, hidden - col_start)
                normalized_stripe = pl.slice(normalized, [MICRO_ROWS, CONSUMER_COLS], [0, col_start], valid_shape=[valid_rows, valid_cols])
                dy_stripe = pl.cast(pl.slice(stored_dy, [MICRO_ROWS, CONSUMER_COLS], [0, col_start], valid_shape=[valid_rows, valid_cols]), pl.FP32)
                weight_stripe = pl.slice(weight, [1, CONSUMER_COLS], [0, col_start], valid_shape=[1, valid_cols])
                beta_stripe = pl.slice(beta_acc, [1, CONSUMER_COLS], [0, col_start])
                gamma_stripe = pl.slice(gamma_acc, [1, CONSUMER_COLS], [0, col_start])
                beta_stripe = pl.add(beta_stripe, pl.col_sum(dy_stripe))
                gamma_stripe = pl.add(gamma_stripe, pl.col_sum(pl.mul(dy_stripe, normalized_stripe)))
                beta_acc = pl.assemble(beta_acc, beta_stripe, [0, col_start])
                gamma_acc = pl.assemble(gamma_acc, gamma_stripe, [0, col_start])
                weighted = pl.col_expand_mul(dy_stripe, weight_stripe)
                weighted_acc = pl.add(weighted_acc, pl.reshape(pl.row_sum(weighted), [1, MICRO_ROWS]))
                projected_acc = pl.add(projected_acc, pl.reshape(pl.row_sum(pl.mul(weighted, normalized_stripe)), [1, MICRO_ROWS]))
            mean_dy = pl.reshape(pl.mul(weighted_acc, 1.0 / hidden), [MICRO_ROWS, 1])
            mean_projected = pl.reshape(pl.mul(projected_acc, 1.0 / hidden), [MICRO_ROWS, 1])
            for col_start in pl.unroll(0, hidden, CONSUMER_COLS):
                valid_cols = pl.min(CONSUMER_COLS, hidden - col_start)
                normalized_stripe = pl.slice(normalized, [MICRO_ROWS, CONSUMER_COLS], [0, col_start], valid_shape=[valid_rows, valid_cols])
                dy_stripe = pl.cast(pl.slice(stored_dy, [MICRO_ROWS, CONSUMER_COLS], [0, col_start], valid_shape=[valid_rows, valid_cols]), pl.FP32)
                weight_stripe = pl.slice(weight, [1, CONSUMER_COLS], [0, col_start], valid_shape=[1, valid_cols])
                weighted = pl.col_expand_mul(dy_stripe, weight_stripe)
                correction = pl.row_expand_mul(normalized_stripe, mean_projected)
                residual = pl.sub(pl.row_expand_sub(weighted, mean_dy), correction)
                output = pl.set_validshape(pl.row_expand_mul(residual, inv_std), valid_rows, valid_cols)
                dx = pl.assemble(dx, pl.cast(output, dx.dtype), [row_start, col_start])
        dgamma = pl.assemble(dgamma, pl.set_validshape(gamma_acc, 1, hidden), [0, 0], atomic=pl.AtomicType.Add)
        dbeta = pl.assemble(dbeta, pl.set_validshape(beta_acc, 1, hidden), [0, 0], atomic=pl.AtomicType.Add)
    return dx

@pl.jit
def r008_streamed_backward(
    dy: pl.Tensor,
    x: pl.Tensor,
    gamma: pl.Tensor,
    dgamma: pl.Out[pl.Tensor],
    dbeta: pl.Out[pl.Tensor],
    dx: pl.Out[pl.Tensor],
):
    rows, hidden = x.shape
    groups = (rows + ROW_TILE - 1) // ROW_TILE
    initializers = (hidden + INIT_COLS - 1) // INIT_COLS
    with pl.spmd(initializers, name_hint="affine_zero") as zero_tid:
        init_start = pl.tile.get_block_idx() * INIT_COLS
        init_valid = pl.min(INIT_COLS, hidden - init_start)
        zeros = pl.set_validshape(pl.full([1, INIT_COLS], dtype=pl.FP32, value=0.0), 1, init_valid)
        dgamma = pl.assemble(dgamma, zeros, [0, init_start])
        dbeta = pl.assemble(dbeta, zeros, [0, init_start])
    with pl.spmd(groups, name_hint="fused_centered_atomic", deps=[zero_tid]) as owner_tid:
        row_start = pl.tile.get_block_idx() * ROW_TILE
        valid_rows = pl.min(ROW_TILE, rows - row_start)
        mean_acc = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
        for col_start in pl.range(0, hidden, COL_TILE):
            valid_cols = pl.min(COL_TILE, hidden - col_start)
            tile_x = pl.slice(x, [ROW_TILE, COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
            tile_x = pl.cast(tile_x, pl.FP32)
            mean_acc = pl.add(mean_acc, pl.reshape(pl.row_sum(pl.mul(tile_x, 1.0 / hidden)), [1, ROW_TILE]))
        mean = pl.reshape(mean_acc, [ROW_TILE, 1])
        variance_acc = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
        weighted_acc = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
        projected_acc = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
        for col_start in pl.range(0, hidden, COL_TILE):
            valid_cols = pl.min(COL_TILE, hidden - col_start)
            tile_x = pl.slice(x, [ROW_TILE, COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_dy = pl.slice(dy, [ROW_TILE, COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
            tile_dy = pl.cast(tile_dy, pl.FP32)
            weight = pl.slice(gamma, [1, COL_TILE], [0, col_start], valid_shape=[1, valid_cols])
            weight = pl.cast(weight, pl.FP32)
            centered = pl.fillpad(pl.row_expand_sub(tile_x, mean), pad_value=pl.PadValue.zero)
            weighted = pl.col_expand_mul(tile_dy, weight)
            variance_acc = pl.add(variance_acc, pl.reshape(pl.row_sum(pl.mul(centered, centered)), [1, ROW_TILE]))
            weighted_acc = pl.add(weighted_acc, pl.reshape(pl.row_sum(weighted), [1, ROW_TILE]))
            projected_acc = pl.add(projected_acc, pl.reshape(pl.row_sum(pl.mul(weighted, centered)), [1, ROW_TILE]))
        inv_std = pl.rsqrt(pl.add(pl.mul(variance_acc, 1.0 / hidden), 1e-5), high_precision=True)
        mean_weighted = pl.reshape(pl.mul(weighted_acc, 1.0 / hidden), [ROW_TILE, 1])
        mean_projected = pl.reshape(pl.mul(pl.mul(projected_acc, 1.0 / hidden), inv_std), [ROW_TILE, 1])
        inv_std = pl.reshape(inv_std, [ROW_TILE, 1])
        for col_start in pl.range(0, hidden, COL_TILE):
            valid_cols = pl.min(COL_TILE, hidden - col_start)
            tile_x = pl.slice(x, [ROW_TILE, COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
            tile_x = pl.cast(tile_x, pl.FP32)
            tile_dy = pl.slice(dy, [ROW_TILE, COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
            tile_dy = pl.cast(tile_dy, pl.FP32)
            weight = pl.slice(gamma, [1, COL_TILE], [0, col_start], valid_shape=[1, valid_cols])
            weight = pl.cast(weight, pl.FP32)
            normalized = pl.fillpad(pl.row_expand_mul(pl.row_expand_sub(tile_x, mean), inv_std), pad_value=pl.PadValue.zero)
            weighted = pl.col_expand_mul(tile_dy, weight)
            correction = pl.row_expand_mul(normalized, mean_projected)
            residual = pl.sub(pl.row_expand_sub(weighted, mean_weighted), correction)
            output = pl.set_validshape(pl.row_expand_mul(residual, inv_std), valid_rows, valid_cols)
            dx = pl.assemble(dx, pl.cast(output, dx.dtype), [row_start, col_start])
            gamma_part = pl.set_validshape(pl.col_sum(pl.mul(tile_dy, normalized)), 1, valid_cols)
            beta_part = pl.set_validshape(pl.col_sum(tile_dy), 1, valid_cols)
            dgamma = pl.assemble(dgamma, gamma_part, [0, col_start], atomic=pl.AtomicType.Add)
            dbeta = pl.assemble(dbeta, beta_part, [0, col_start], atomic=pl.AtomicType.Add)
    return dx


def build(shape):
    fn, specs, golden_fn, rtol, atol = build_fallback(shape)
    hidden = int(shape["N"])
    if shape["dtype"] == "bf16" and hidden > 1:
        if PHYSICAL_HIDDEN <= 1024:
            fn = r008_resident_backward
        elif 1024 < hidden <= 1536:
            fn = r008_wide_striped_backward
        else:
            fn = r008_streamed_backward
    return fn, specs, golden_fn, rtol, atol

if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    args = parser.parse_args()

    fn, specs, golden_fn, rtol, atol = build(DEFAULT_SHAPE)
    result = run(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn,
        config=dict(
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=rtol,
        atol=atol,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
