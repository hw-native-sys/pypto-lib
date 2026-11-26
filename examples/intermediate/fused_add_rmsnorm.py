# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fused residual-add + RMSNorm — h = x + res; y = h * rsqrt(mean(h^2) + eps) * gamma; bf16 storage, fp32 accumulation. build() shape-dispatches among resident/replay variants; all are kept."""
import pypto.language as pl

DEFAULT_SHAPE = {"M": 512, "N": 1024, "dtype": "bf16"}


ROW_TILE = 2
GRID_NUMERATOR = 2
EPS = 1e-6
VEC_LIMIT_BYTES = 188416
CERTIFIED_MAX_HIDDEN = 7168
GAMMA_POLICY = 1
FULL_GAMMA_MAX_HIDDEN = 6656


@pl.jit
def resident_full_gamma(x: pl.Tensor, res: pl.Tensor, gamma: pl.Tensor,
                        h: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    tile_count = (rows + ROW_TILE - 1) // ROW_TILE
    block_count = pl.min(tile_count, pl.max(1, pl.system.available_aiv_count() * GRID_NUMERATOR // 2))
    trip_count = (tile_count + block_count - 1) // block_count
    for block_id in pl.spmd(block_count, name_hint="resident_gamma_grid_stride"):
        gamma_cached = pl.slice(gamma, [1, hidden], [0, 0])
        for iteration in pl.range(0, trip_count, 1):
            tile_id = block_id + iteration * block_count
            if tile_id < tile_count:
                row_start = tile_id * ROW_TILE
                valid_rows = pl.min(ROW_TILE, rows - row_start)
                x_tile = pl.slice(x, [ROW_TILE, hidden], [row_start, 0])
                res_tile = pl.slice(res, [ROW_TILE, hidden], [row_start, 0])
                residual = pl.add(pl.cast(x_tile, target_type=pl.FP32), pl.cast(res_tile, target_type=pl.FP32))
                residual_matrix = pl.reshape(residual, [ROW_TILE * 8, hidden // 8])
                partials = pl.row_sum(pl.mul(residual_matrix, residual_matrix))
                partials = pl.reshape(partials, [ROW_TILE, 8])
                padded_partials = pl.full([8, 8], dtype=pl.FP32, value=0.0)
                padded_partials = pl.assemble(padded_partials, partials, [0, 0])
                square_sum = pl.row_sum(padded_partials)
                mean_square = pl.add(pl.mul(square_sum, 1.0 / hidden), EPS)
                inverse_rms = pl.rsqrt(mean_square)
                inverse_rms = pl.reshape(inverse_rms, [8, 1])
                inverse_repeated = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.0), inverse_rms)
                inverse_segments = pl.reshape(pl.slice(inverse_repeated, [ROW_TILE, 8], [0, 0]), [ROW_TILE * 8, 1])
                h = pl.assemble(h, pl.cast(residual, target_type=pl.BF16, mode="rint"), [row_start, 0])
                normalized = pl.row_expand_mul(residual_matrix, inverse_segments)
                normalized = pl.reshape(normalized, [ROW_TILE, hidden])
                normalized = pl.col_expand_mul(normalized, gamma_cached)
                y = pl.assemble(y, pl.cast(normalized, target_type=pl.BF16, mode="rint"), [row_start, 0])
    return h, y


@pl.jit
def resident_spmd_stage1(x: pl.Tensor, res: pl.Tensor, gamma: pl.Tensor,
                         h: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    tile_count = (rows + ROW_TILE - 1) // ROW_TILE
    block_count = pl.min(tile_count, pl.max(1, pl.system.available_aiv_count() * GRID_NUMERATOR // 2))
    trip_count = (tile_count + block_count - 1) // block_count
    for block_id in pl.spmd(block_count, name_hint="resident_grid_stride"):
        for iteration in pl.range(0, trip_count, 1):
            tile_id = block_id + iteration * block_count
            if tile_id < tile_count:
                row_start = tile_id * ROW_TILE
                valid_rows = pl.min(ROW_TILE, rows - row_start)
                x_tile = pl.slice(x, [ROW_TILE, hidden], [row_start, 0])
                res_tile = pl.slice(res, [ROW_TILE, hidden], [row_start, 0])
                residual = pl.add(pl.cast(x_tile, target_type=pl.FP32), pl.cast(res_tile, target_type=pl.FP32))
                residual_matrix = pl.reshape(residual, [ROW_TILE * 8, hidden // 8])
                partials = pl.row_sum(pl.mul(residual_matrix, residual_matrix))
                partials = pl.reshape(partials, [ROW_TILE, 8])
                padded_partials = pl.full([8, 8], dtype=pl.FP32, value=0.0)
                padded_partials = pl.assemble(padded_partials, partials, [0, 0])
                square_sum = pl.row_sum(padded_partials)
                mean_square = pl.add(pl.mul(square_sum, 1.0 / hidden), EPS)
                inverse_rms = pl.rsqrt(mean_square)
                inverse_rms = pl.reshape(inverse_rms, [8, 1])
                h = pl.assemble(h, pl.cast(residual, target_type=pl.BF16, mode="rint"), [row_start, 0])
                inverse_repeated = pl.row_expand_mul(pl.full([8, 8], dtype=pl.FP32, value=1.0), inverse_rms)
                inverse_segments = pl.reshape(pl.slice(inverse_repeated, [ROW_TILE, 8], [0, 0]), [ROW_TILE * 8, 1])
                normalized = pl.row_expand_mul(residual_matrix, inverse_segments)
                normalized = pl.reshape(normalized, [ROW_TILE, hidden])
                gamma_tile = pl.slice(gamma, [1, hidden], [0, 0])
                normalized = pl.col_expand_mul(normalized, gamma_tile)
                y = pl.assemble(y, pl.cast(normalized, target_type=pl.BF16, mode="rint"), [row_start, 0])

    return h, y

import pypto.language as pl

REPLAY_ROW_TILE = 8
REPLAY_COL_TILE = 128
EPS = 1e-6


@pl.jit
def replay_add_rmsnorm(x: pl.Tensor, res: pl.Tensor, gamma: pl.Tensor,
                      h: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    for row_start in pl.parallel(0, rows, REPLAY_ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="replayed_residual"):
            valid_rows = pl.min(REPLAY_ROW_TILE, rows - row_start)
            square_sum = pl.full([1, REPLAY_ROW_TILE], dtype=pl.FP32, value=0.0)
            for col_start in pl.range(0, hidden, REPLAY_COL_TILE):
                valid_cols = pl.min(REPLAY_COL_TILE, hidden - col_start)
                x_tile = pl.slice(x, [REPLAY_ROW_TILE, REPLAY_COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
                res_tile = pl.slice(res, [REPLAY_ROW_TILE, REPLAY_COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
                residual_tile = pl.add(x_tile, res_tile)
                chunk_sum = pl.row_sum(pl.mul(residual_tile, residual_tile))
                chunk_sum = pl.reshape(chunk_sum, [1, REPLAY_ROW_TILE])
                square_sum = pl.add(square_sum, chunk_sum)
            mean_square = pl.add(pl.mul(square_sum, 1.0 / hidden), EPS)
            inverse_rms = pl.rsqrt(mean_square, high_precision=True)
            inverse_rms = pl.reshape(inverse_rms, [REPLAY_ROW_TILE, 1])
            for col_start in pl.range(0, hidden, REPLAY_COL_TILE):
                valid_cols = pl.min(REPLAY_COL_TILE, hidden - col_start)
                x_tile = pl.slice(x, [REPLAY_ROW_TILE, REPLAY_COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
                res_tile = pl.slice(res, [REPLAY_ROW_TILE, REPLAY_COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
                residual_tile = pl.add(x_tile, res_tile)
                gamma_tile = pl.slice(gamma, [1, REPLAY_COL_TILE], [0, col_start], valid_shape=[1, valid_cols])
                normalized = pl.row_expand_mul(residual_tile, inverse_rms)
                normalized = pl.col_expand_mul(normalized, gamma_tile)
                h[row_start:row_start + REPLAY_ROW_TILE, col_start:col_start + REPLAY_COL_TILE] = residual_tile
                y[row_start:row_start + REPLAY_ROW_TILE, col_start:col_start + REPLAY_COL_TILE] = normalized
    return h, y


import pypto.language as pl

REPLAY_ROW_TILE = 8
REPLAY_COL_TILE = 128
EPS = 1e-6


@pl.jit
def replay_add_rmsnorm_bf16(x: pl.Tensor, res: pl.Tensor, gamma: pl.Tensor,
                      h: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    for row_start in pl.parallel(0, rows, REPLAY_ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="replayed_residual"):
            valid_rows = pl.min(REPLAY_ROW_TILE, rows - row_start)
            square_sum = pl.full([1, REPLAY_ROW_TILE], dtype=pl.FP32, value=0.0)
            for col_start in pl.range(0, hidden, REPLAY_COL_TILE):
                valid_cols = pl.min(REPLAY_COL_TILE, hidden - col_start)
                x_tile = pl.slice(x, [REPLAY_ROW_TILE, REPLAY_COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
                res_tile = pl.slice(res, [REPLAY_ROW_TILE, REPLAY_COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
                x_tile = pl.cast(x_tile, target_type=pl.FP32)
                res_tile = pl.cast(res_tile, target_type=pl.FP32)
                residual_tile = pl.add(x_tile, res_tile)
                chunk_sum = pl.row_sum(pl.mul(residual_tile, residual_tile))
                chunk_sum = pl.reshape(chunk_sum, [1, REPLAY_ROW_TILE])
                square_sum = pl.add(square_sum, chunk_sum)
            mean_square = pl.add(pl.mul(square_sum, 1.0 / hidden), EPS)
            inverse_rms = pl.rsqrt(mean_square, high_precision=True)
            inverse_rms = pl.reshape(inverse_rms, [REPLAY_ROW_TILE, 1])
            for col_start in pl.range(0, hidden, REPLAY_COL_TILE):
                valid_cols = pl.min(REPLAY_COL_TILE, hidden - col_start)
                x_tile = pl.slice(x, [REPLAY_ROW_TILE, REPLAY_COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
                res_tile = pl.slice(res, [REPLAY_ROW_TILE, REPLAY_COL_TILE], [row_start, col_start], valid_shape=[valid_rows, valid_cols])
                x_tile = pl.cast(x_tile, target_type=pl.FP32)
                res_tile = pl.cast(res_tile, target_type=pl.FP32)
                residual_tile = pl.add(x_tile, res_tile)
                gamma_tile = pl.slice(gamma, [1, REPLAY_COL_TILE], [0, col_start], valid_shape=[1, valid_cols])
                normalized = pl.row_expand_mul(residual_tile, inverse_rms)
                normalized = pl.col_expand_mul(normalized, gamma_tile)
                residual_tile = pl.cast(residual_tile, target_type=pl.BF16, mode="rint")
                normalized = pl.cast(normalized, target_type=pl.BF16, mode="rint")
                h[row_start:row_start + REPLAY_ROW_TILE, col_start:col_start + REPLAY_COL_TILE] = residual_tile
                y[row_start:row_start + REPLAY_ROW_TILE, col_start:col_start + REPLAY_COL_TILE] = normalized
    return h, y




@pl.jit
def resident_compact_wide(x: pl.Tensor, res: pl.Tensor, gamma: pl.Tensor,
                         h: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    tile_count = rows // 2
    block_count = pl.min(tile_count, pl.max(1, pl.system.available_aiv_count() * GRID_NUMERATOR // 2))
    for block_id in pl.spmd(block_count, name_hint="resident_compact_wide_phase_carry"):
        gamma_cached = pl.load(gamma, [0, 0], [1, hidden])
        local_count = (tile_count - 1 - block_id) // block_count + 1
        prime_x = pl.load(x, [block_id * 2, 0], [2, hidden])
        prime_res = pl.load(res, [block_id * 2, 0], [2, hidden])
        for iteration, (x_tile, res_tile, h_iter, y_iter) in pl.range(0, local_count - 1, 1, init_values=(prime_x, prime_res, h, y)):
            row_start = (block_id + iteration * block_count) * 2
            next_row_start = row_start + block_count * 2
            h = h_iter
            y = y_iter
            residual = pl.create_tile([2, hidden], dtype=pl.FP32)
            for promotion_row in pl.unroll(0, 2, 1):
                for promotion_part in pl.unroll(0, 2, 1):
                    input_row = pl.slice(x_tile, [1, hidden // 2], [promotion_row, promotion_part * (hidden // 2)])
                    residual_input_row = pl.slice(res_tile, [1, hidden // 2], [promotion_row, promotion_part * (hidden // 2)])
                    promoted_row = pl.add(pl.cast(input_row, target_type=pl.FP32), pl.cast(residual_input_row, target_type=pl.FP32))
                    residual = pl.assemble(residual, promoted_row, [promotion_row, promotion_part * (hidden // 2)])
            residual_matrix = pl.reshape(residual, [2 * 16, hidden // 16])
            next_x = pl.load(x, [next_row_start, 0], [2, hidden])
            next_res = pl.load(res, [next_row_start, 0], [2, hidden])
            partials = pl.tile.full([1, 32], dtype=pl.FP32, value=0.0)
            for segment_start in pl.unroll(0, 32, 8):
                segment_batch = pl.slice(residual_matrix, [8, hidden // 16], [segment_start, 0])
                segment_square = pl.mul(segment_batch, segment_batch)
                segment_scratch = pl.create_tile([8, hidden // 16], dtype=pl.FP32)
                segment_sum = pl.row_sum(segment_square, tmp_tile=segment_scratch)
                segment_sum_row = pl.reshape(segment_sum, [1, 8])
                partials = pl.assemble(partials, segment_sum_row, [0, segment_start])
            partial_rows = pl.reshape(partials, [2, 16])
            padded_partials = pl.tile.full([8, 16], dtype=pl.FP32, value=0.0)
            padded_partials = pl.assemble(padded_partials, partial_rows, [0, 0])
            row_scratch = pl.create_tile([8, 16], dtype=pl.FP32)
            square_sum = pl.row_sum(padded_partials, tmp_tile=row_scratch)
            mean_square = pl.add(pl.mul(square_sum, 1.0 / hidden), EPS)
            inverse_rms = pl.rsqrt(mean_square)
            inverse_column = pl.reshape(inverse_rms, [8, 1])
            inverse_repeated = pl.row_expand_mul(pl.tile.full([8, 16], dtype=pl.FP32, value=1.0), inverse_column)
            inverse_segments = pl.reshape(pl.slice(inverse_repeated, [2, 16], [0, 0]), [2 * 16, 1])
            h = pl.store(pl.cast(residual, target_type=pl.BF16, mode="rint"), [row_start, 0], h)
            normalized = pl.row_expand_mul(residual_matrix, inverse_segments)
            normalized_rows = pl.reshape(normalized, [2, hidden])
            normalized_weighted = pl.col_expand_mul(normalized_rows, gamma_cached)
            y = pl.store(pl.cast(normalized_weighted, target_type=pl.BF16, mode="rint"), [row_start, 0], y)
            last_x, last_res, h_last, y_last = pl.yield_(next_x, next_res, h, y)
        x_tile = last_x
        res_tile = last_res
        h = h_last
        y = y_last
        row_start = (block_id + (local_count - 1) * block_count) * 2
        residual = pl.create_tile([2, hidden], dtype=pl.FP32)
        for promotion_row in pl.unroll(0, 2, 1):
            for promotion_part in pl.unroll(0, 2, 1):
                input_row = pl.slice(x_tile, [1, hidden // 2], [promotion_row, promotion_part * (hidden // 2)])
                residual_input_row = pl.slice(res_tile, [1, hidden // 2], [promotion_row, promotion_part * (hidden // 2)])
                promoted_row = pl.add(pl.cast(input_row, target_type=pl.FP32), pl.cast(residual_input_row, target_type=pl.FP32))
                residual = pl.assemble(residual, promoted_row, [promotion_row, promotion_part * (hidden // 2)])
        residual_matrix = pl.reshape(residual, [2 * 16, hidden // 16])
        partials = pl.tile.full([1, 32], dtype=pl.FP32, value=0.0)
        for segment_start in pl.unroll(0, 32, 8):
            segment_batch = pl.slice(residual_matrix, [8, hidden // 16], [segment_start, 0])
            segment_square = pl.mul(segment_batch, segment_batch)
            segment_scratch = pl.create_tile([8, hidden // 16], dtype=pl.FP32)
            segment_sum = pl.row_sum(segment_square, tmp_tile=segment_scratch)
            segment_sum_row = pl.reshape(segment_sum, [1, 8])
            partials = pl.assemble(partials, segment_sum_row, [0, segment_start])
        partial_rows = pl.reshape(partials, [2, 16])
        padded_partials = pl.tile.full([8, 16], dtype=pl.FP32, value=0.0)
        padded_partials = pl.assemble(padded_partials, partial_rows, [0, 0])
        row_scratch = pl.create_tile([8, 16], dtype=pl.FP32)
        square_sum = pl.row_sum(padded_partials, tmp_tile=row_scratch)
        mean_square = pl.add(pl.mul(square_sum, 1.0 / hidden), EPS)
        inverse_rms = pl.rsqrt(mean_square)
        inverse_column = pl.reshape(inverse_rms, [8, 1])
        inverse_repeated = pl.row_expand_mul(pl.tile.full([8, 16], dtype=pl.FP32, value=1.0), inverse_column)
        inverse_segments = pl.reshape(pl.slice(inverse_repeated, [2, 16], [0, 0]), [2 * 16, 1])
        h = pl.store(pl.cast(residual, target_type=pl.BF16, mode="rint"), [row_start, 0], h)
        normalized = pl.row_expand_mul(residual_matrix, inverse_segments)
        normalized_rows = pl.reshape(normalized, [2, hidden])
        normalized_weighted = pl.col_expand_mul(normalized_rows, gamma_cached)
        y = pl.store(pl.cast(normalized_weighted, target_type=pl.BF16, mode="rint"), [row_start, 0], y)
    return h, y


@pl.jit
def resident_compact_narrow(x: pl.Tensor, res: pl.Tensor, gamma: pl.Tensor,
                         h: pl.Out[pl.Tensor], y: pl.Out[pl.Tensor]):
    rows, hidden = x.shape
    tile_count = rows // 4
    block_count = pl.min(tile_count, pl.max(1, pl.system.available_aiv_count() * GRID_NUMERATOR // 2))
    for block_id in pl.spmd(block_count, name_hint="resident_compact_narrow_phase_carry"):
        gamma_cached = pl.load(gamma, [0, 0], [1, hidden])
        local_count = (tile_count - 1 - block_id) // block_count + 1
        prime_x = pl.load(x, [block_id * 4, 0], [4, hidden])
        prime_res = pl.load(res, [block_id * 4, 0], [4, hidden])
        for iteration, (x_tile, res_tile, h_iter, y_iter) in pl.range(0, local_count - 1, 1, init_values=(prime_x, prime_res, h, y)):
            row_start = (block_id + iteration * block_count) * 4
            next_row_start = row_start + block_count * 4
            h = h_iter
            y = y_iter
            residual = pl.create_tile([4, hidden], dtype=pl.FP32)
            for promotion_row in pl.unroll(0, 4, 1):
                input_row = pl.slice(x_tile, [1, hidden], [promotion_row, 0])
                residual_input_row = pl.slice(res_tile, [1, hidden], [promotion_row, 0])
                promoted_row = pl.add(pl.cast(input_row, target_type=pl.FP32), pl.cast(residual_input_row, target_type=pl.FP32))
                residual = pl.assemble(residual, promoted_row, [promotion_row, 0])
            residual_matrix = pl.reshape(residual, [4 * 8, hidden // 8])
            next_x = pl.load(x, [next_row_start, 0], [4, hidden])
            next_res = pl.load(res, [next_row_start, 0], [4, hidden])
            partials = pl.tile.full([1, 32], dtype=pl.FP32, value=0.0)
            for segment_start in pl.unroll(0, 32, 8):
                segment_batch = pl.slice(residual_matrix, [8, hidden // 8], [segment_start, 0])
                segment_square = pl.mul(segment_batch, segment_batch)
                segment_scratch = pl.create_tile([8, hidden // 8], dtype=pl.FP32)
                segment_sum = pl.row_sum(segment_square, tmp_tile=segment_scratch)
                segment_sum_row = pl.reshape(segment_sum, [1, 8])
                partials = pl.assemble(partials, segment_sum_row, [0, segment_start])
            partial_rows = pl.reshape(partials, [4, 8])
            padded_partials = pl.tile.full([8, 8], dtype=pl.FP32, value=0.0)
            padded_partials = pl.assemble(padded_partials, partial_rows, [0, 0])
            row_scratch = pl.create_tile([8, 8], dtype=pl.FP32)
            square_sum = pl.row_sum(padded_partials, tmp_tile=row_scratch)
            mean_square = pl.add(pl.mul(square_sum, 1.0 / hidden), EPS)
            inverse_rms = pl.rsqrt(mean_square)
            inverse_column = pl.reshape(inverse_rms, [8, 1])
            inverse_repeated = pl.row_expand_mul(pl.tile.full([8, 8], dtype=pl.FP32, value=1.0), inverse_column)
            inverse_segments = pl.reshape(pl.slice(inverse_repeated, [4, 8], [0, 0]), [4 * 8, 1])
            h = pl.store(pl.cast(residual, target_type=pl.BF16, mode="rint"), [row_start, 0], h)
            normalized = pl.row_expand_mul(residual_matrix, inverse_segments)
            normalized_rows = pl.reshape(normalized, [4, hidden])
            normalized_weighted = pl.col_expand_mul(normalized_rows, gamma_cached)
            y = pl.store(pl.cast(normalized_weighted, target_type=pl.BF16, mode="rint"), [row_start, 0], y)
            last_x, last_res, h_last, y_last = pl.yield_(next_x, next_res, h, y)
        x_tile = last_x
        res_tile = last_res
        h = h_last
        y = y_last
        row_start = (block_id + (local_count - 1) * block_count) * 4
        residual = pl.create_tile([4, hidden], dtype=pl.FP32)
        for promotion_row in pl.unroll(0, 4, 1):
            input_row = pl.slice(x_tile, [1, hidden], [promotion_row, 0])
            residual_input_row = pl.slice(res_tile, [1, hidden], [promotion_row, 0])
            promoted_row = pl.add(pl.cast(input_row, target_type=pl.FP32), pl.cast(residual_input_row, target_type=pl.FP32))
            residual = pl.assemble(residual, promoted_row, [promotion_row, 0])
        residual_matrix = pl.reshape(residual, [4 * 8, hidden // 8])
        partials = pl.tile.full([1, 32], dtype=pl.FP32, value=0.0)
        for segment_start in pl.unroll(0, 32, 8):
            segment_batch = pl.slice(residual_matrix, [8, hidden // 8], [segment_start, 0])
            segment_square = pl.mul(segment_batch, segment_batch)
            segment_scratch = pl.create_tile([8, hidden // 8], dtype=pl.FP32)
            segment_sum = pl.row_sum(segment_square, tmp_tile=segment_scratch)
            segment_sum_row = pl.reshape(segment_sum, [1, 8])
            partials = pl.assemble(partials, segment_sum_row, [0, segment_start])
        partial_rows = pl.reshape(partials, [4, 8])
        padded_partials = pl.tile.full([8, 8], dtype=pl.FP32, value=0.0)
        padded_partials = pl.assemble(padded_partials, partial_rows, [0, 0])
        row_scratch = pl.create_tile([8, 8], dtype=pl.FP32)
        square_sum = pl.row_sum(padded_partials, tmp_tile=row_scratch)
        mean_square = pl.add(pl.mul(square_sum, 1.0 / hidden), EPS)
        inverse_rms = pl.rsqrt(mean_square)
        inverse_column = pl.reshape(inverse_rms, [8, 1])
        inverse_repeated = pl.row_expand_mul(pl.tile.full([8, 8], dtype=pl.FP32, value=1.0), inverse_column)
        inverse_segments = pl.reshape(pl.slice(inverse_repeated, [4, 8], [0, 0]), [4 * 8, 1])
        h = pl.store(pl.cast(residual, target_type=pl.BF16, mode="rint"), [row_start, 0], h)
        normalized = pl.row_expand_mul(residual_matrix, inverse_segments)
        normalized_rows = pl.reshape(normalized, [4, hidden])
        normalized_weighted = pl.col_expand_mul(normalized_rows, gamma_cached)
        y = pl.store(pl.cast(normalized_weighted, target_type=pl.BF16, mode="rint"), [row_start, 0], y)
    return h, y


def golden_fn(tensors):
    import torch
    residual = tensors["x"].float() + tensors["res"].float()
    inverse_rms = torch.rsqrt(residual.square().mean(dim=-1, keepdim=True) + EPS)
    tensors["h"][:] = residual
    tensors["y"][:] = residual * inverse_rms * tensors["gamma"].float()


def build(shape):
    import torch
    from golden import TensorSpec
    rows, hidden = int(shape["M"]), int(shape["N"])
    if rows <= 0 or hidden <= 0 or shape["dtype"] not in ("fp32", "bf16"):
        raise ValueError("Expected positive M/N and fp32 or bf16 dtype")
    resident_bytes_bound = 24 * hidden + 64
    resident_ok = (shape["dtype"] == "bf16" and hidden % 128 == 0
                   and rows % ROW_TILE == 0 and hidden <= CERTIFIED_MAX_HIDDEN
                   and resident_bytes_bound <= VEC_LIMIT_BYTES)
    if resident_ok:
        full_gamma_ok = (GAMMA_POLICY == 1 and hidden <= FULL_GAMMA_MAX_HIDDEN
                         and 28 * hidden + 64 <= VEC_LIMIT_BYTES)
        fn = resident_full_gamma if full_gamma_ok else resident_spmd_stage1
    else:
        fn = replay_add_rmsnorm_bf16 if shape["dtype"] == "bf16" else replay_add_rmsnorm
    if shape["dtype"] == "bf16":
        if rows % 4 == 0 and hidden in (2048, 3072, 4096):
            fn = resident_compact_narrow
        elif rows % 2 == 0 and hidden == 7168:
            fn = resident_compact_wide
    dtype = torch.float32 if shape["dtype"] == "fp32" else torch.bfloat16
    specs = [
        TensorSpec("x", [rows, hidden], dtype, init_value=torch.randn),
        TensorSpec("res", [rows, hidden], dtype, init_value=torch.randn),
        TensorSpec("gamma", [1, hidden], torch.float32, init_value=torch.randn),
        TensorSpec("h", [rows, hidden], dtype),
        TensorSpec("y", [rows, hidden], dtype),
    ]
    tolerance = 1e-4 if shape["dtype"] == "fp32" else 2e-2
    return fn, specs, golden_fn, tolerance, tolerance

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
