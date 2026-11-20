# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Greedy and temperature/top-k/top-p/Gumbel sampling for DeepSeek-V4.1 logits."""

import sys

import pypto.language as pl

from config import DECODE_MAX_TOKENS, DECODE_SEQ
from config import FLASH as M


def _parse_int_argv(name, default=None):
    for i, tok in enumerate(sys.argv):
        if tok == name and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if tok.startswith(f"{name}="):
            return int(tok.split("=", 1)[1])
    return default


# model config
VOCAB = M.vocab_size
SAMPLED_IDS_PAD = 8

# Standalone validation shape.
if __name__ == "__main__":
    SAMPLE_ROWS = _parse_int_argv("--rows", DECODE_MAX_TOKENS)
    if SAMPLE_ROWS <= 0:
        raise SystemExit(f"--rows must be positive, got {SAMPLE_ROWS}")
else:
    SAMPLE_ROWS = DECODE_MAX_TOKENS

# tiling
SAMPLE_WORKERS_TILE = 64
TEMPERATURE_WIDTH_TILE = 808
SAMPLE_BLOCK_ROWS_TILE = 8
ARGMAX_WIDTH_TILE = 808
ARGMAX_ROWS_TILE = 8
FILTER_WIDTH_TILE = 1280
TOPK_ROW_WIDTH_TILE = 640
PROBE_WIDTH_TILE = SAMPLE_BLOCK_ROWS_TILE * TOPK_ROW_WIDTH_TILE
FILTER_LANE_TILE = FILTER_WIDTH_TILE // SAMPLE_BLOCK_ROWS_TILE
ARGMAX_BLOCK_TILE = ARGMAX_ROWS_TILE * ARGMAX_WIDTH_TILE

# Boundary searches over FP32 bit patterns.
TOPK_SEARCH_STEPS = 32
TOPP_SEARCH_STEPS = 30
TOPP_ABOVE_ONE_BITS = 0x3F800001

# sampling constants
FP32_NEG_INF = -3.4028234663852886e38
SAMPLING_EPS = 1e-5
RANDOM_KEY_MODULUS = 1073741824
HASH_MULTIPLIER = 0x045D9F3B
KEY_HASH_MULTIPLIER = 0x27220A95
TOKEN_HASH_OFFSET = 0x5BD1E995
KEY_HASH_OFFSET = 0x632BE593
POSITION_MULTIPLIER = 65537
UINT23_SCALE = 1.0 / 8388608.0
GREEDY_INDEX_SENTINEL = 1073741824


@pl.jit.inline
def _greedy_sample_logits(
    logits_grid: pl.Tensor,
    row: pl.Scalar[pl.INDEX],
):
    """Select the first maximum token id from one full-vocabulary logits row."""
    row_base = row * (VOCAB // ARGMAX_WIDTH_TILE)
    running_max = pl.full([ARGMAX_ROWS_TILE, ARGMAX_WIDTH_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
    running_base = pl.full([ARGMAX_ROWS_TILE, ARGMAX_WIDTH_TILE], dtype=pl.INT32, value=0)
    for block in pl.range(VOCAB // ARGMAX_WIDTH_TILE // ARGMAX_ROWS_TILE):
        block_row = row_base + block * ARGMAX_ROWS_TILE
        scores = logits_grid[block_row : block_row + ARGMAX_ROWS_TILE, 0:ARGMAX_WIDTH_TILE]
        is_newer = pl.cmp(scores, running_max, cmp_type=4)
        newer = pl.cast(is_newer, target_type=pl.INT32)
        running_max = pl.maximum(running_max, scores)
        block_base = pl.cast(block * ARGMAX_ROWS_TILE * ARGMAX_WIDTH_TILE, pl.INT32)
        to_new = pl.add(pl.neg(running_base), block_base)
        running_base = pl.add(running_base, pl.mul(newer, to_new))

    lane_maxima = pl.row_max(running_max)
    lane_zeros = pl.full([ARGMAX_ROWS_TILE, ARGMAX_WIDTH_TILE], dtype=pl.FP32, value=0.0)
    lane_broadcast = pl.row_expand_add(lane_zeros, lane_maxima)
    best_value = pl.read(pl.col_max(lane_broadcast), [0, 0])

    ramp_zeros = pl.full([ARGMAX_ROWS_TILE, ARGMAX_WIDTH_TILE], dtype=pl.INT32, value=0)
    column_ids = pl.arange(0, [1, ARGMAX_WIDTH_TILE], dtype=pl.INT32)
    column_ramp = pl.col_expand(ramp_zeros, column_ids)
    flat_index = pl.add(running_base, column_ramp)
    is_max = pl.cmp(running_max, best_value, cmp_type=0)
    hit = pl.cast(is_max, target_type=pl.INT32)
    index_sentinel = pl.cast(GREEDY_INDEX_SENTINEL, pl.INT32)
    negative_index_sentinel = pl.cast(-GREEDY_INDEX_SENTINEL, pl.INT32)
    offset_index = pl.add(flat_index, negative_index_sentinel)
    candidates = pl.add(pl.mul(hit, offset_index), index_sentinel)
    lane_indices = pl.row_min(candidates)
    best_index = pl.read(lane_indices, [0, 0])
    for lane in pl.range(1, ARGMAX_ROWS_TILE):
        lane_term = pl.cast(lane * ARGMAX_WIDTH_TILE, pl.INT32)
        lane_best = pl.read(lane_indices, [lane, 0]) + lane_term
        best_index = pl.min(best_index, lane_best)
    return best_index


@pl.jit.inline
def _gumbel_argmax(
    logits: pl.Tensor,
    row: pl.Scalar[pl.INDEX],
    random_key: pl.Scalar[pl.INT32],
    row_max: pl.Scalar[pl.FP32],
    temperature: pl.Scalar[pl.FP32],
):
    """Add counter-based Gumbel noise and reduce the full vocabulary."""
    best_score = pl.cast(FP32_NEG_INF, pl.FP32)
    best_index = pl.cast(0, pl.INT32)
    for block in pl.range(VOCAB // ARGMAX_BLOCK_TILE):
        start = block * ARGMAX_BLOCK_TILE
        raw_tile = pl.load(logits, [row, start], [1, ARGMAX_BLOCK_TILE], target_memory=pl.Mem.Vec)
        raw = pl.reshape(raw_tile, [ARGMAX_BLOCK_TILE])
        noisy = pl.create_tile([ARGMAX_ROWS_TILE * ARGMAX_WIDTH_TILE], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        with pl.reg_scope():
            full = pl.reg.full_mask(lanes=64)
            mask = pl.reg.broadcast(0x7FFFFFFF, lanes=64, dtype=pl.INT32)
            key = pl.reg.broadcast(random_key, lanes=64, dtype=pl.INT32)
            key = pl.reg.adds(key, KEY_HASH_OFFSET, full)
            shifted = pl.reg.shift_rights(key, 16, full)
            key = pl.reg.bitwise_xor(key, shifted, full)
            key = pl.reg.muls(key, KEY_HASH_MULTIPLIER, full)
            key = pl.reg.bitwise_and(key, mask, full)
            shifted = pl.reg.shift_rights(key, 16, full)
            key = pl.reg.bitwise_xor(key, shifted, full)
            key = pl.reg.muls(key, KEY_HASH_MULTIPLIER, full)
            key = pl.reg.bitwise_and(key, mask, full)
            shifted = pl.reg.shift_rights(key, 16, full)
            key = pl.reg.bitwise_xor(key, shifted, full)
            maximum = pl.reg.broadcast(row_max, lanes=64, dtype=pl.FP32)
            divisor = pl.reg.broadcast(temperature, lanes=64, dtype=pl.FP32)
            for segment in pl.hardware_loop(ARGMAX_BLOCK_TILE // 64):
                offset = segment * 64
                token = pl.reg.index_gen(pl.cast(start + offset, pl.INT32), lanes=64, dtype=pl.INT32)
                token = pl.reg.adds(token, TOKEN_HASH_OFFSET, full)
                shifted = pl.reg.shift_rights(token, 16, full)
                token = pl.reg.bitwise_xor(token, shifted, full)
                token = pl.reg.muls(token, HASH_MULTIPLIER, full)
                token = pl.reg.bitwise_and(token, mask, full)
                shifted = pl.reg.shift_rights(token, 16, full)
                token = pl.reg.bitwise_xor(token, shifted, full)
                token = pl.reg.muls(token, HASH_MULTIPLIER, full)
                token = pl.reg.bitwise_and(token, mask, full)
                shifted = pl.reg.shift_rights(token, 16, full)
                token = pl.reg.bitwise_xor(token, shifted, full)
                bits = pl.reg.bitwise_xor(token, key, full)
                bits = pl.reg.shift_rights(bits, 8, full)
                uniform = pl.reg.convert(bits, dtype=pl.FP32)
                uniform = pl.reg.adds(uniform, 0.5, full)
                uniform = pl.reg.muls(uniform, UINT23_SCALE, full)
                log_uniform = pl.reg.log(uniform, full)
                exp_noise = pl.reg.neg(log_uniform, full)
                log_noise = pl.reg.log(exp_noise, full)
                raw_score = pl.reg.load(raw, offset, lanes=64)
                centered = pl.reg.sub(raw_score, maximum, full)
                scaled = pl.reg.div(centered, divisor, full)
                score = pl.reg.sub(scaled, log_noise, full)
                _written = pl.reg.store(noisy, offset, score)
        scores = pl.reshape(noisy, [ARGMAX_ROWS_TILE, ARGMAX_WIDTH_TILE])
        tmp = pl.create_tile([ARGMAX_ROWS_TILE, ARGMAX_WIDTH_TILE], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        winners = pl.tile.row_argmax(scores, tmp)
        for lane in pl.range(ARGMAX_ROWS_TILE):
            index = pl.tile.read(winners, [lane, 0])
            value = pl.tile.read(scores, [lane, pl.cast(index, pl.INDEX)])
            if value > best_score:
                best_score = value
                best_index = pl.cast(start + lane * ARGMAX_WIDTH_TILE, pl.INT32) + index
    return best_index


@pl.jit.inline
def _sample_worker_row(
    slot: pl.Scalar[pl.INDEX],
    sample_rows: pl.Scalar[pl.INDEX],
):
    """Reverse alternate row groups across the sampling workers."""
    group_start = slot // SAMPLE_WORKERS_TILE * SAMPLE_WORKERS_TILE
    row = slot
    if slot // SAMPLE_WORKERS_TILE % 2 == 1:
        group_end = pl.min(group_start + SAMPLE_WORKERS_TILE, sample_rows)
        row = group_start + group_end - 1 - slot
    return row


@pl.jit.inline
def _row_max(
    logits: pl.Tensor,
    row: pl.Scalar[pl.INDEX],
):
    maxima = pl.full([SAMPLE_BLOCK_ROWS_TILE, TEMPERATURE_WIDTH_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
    block_width = SAMPLE_BLOCK_ROWS_TILE * TEMPERATURE_WIDTH_TILE
    for vocab_tile in pl.range(VOCAB // block_width):
        start = vocab_tile * block_width
        flat_scores = logits[row : row + 1, start : start + block_width]
        scores = pl.reshape(flat_scores, [SAMPLE_BLOCK_ROWS_TILE, TEMPERATURE_WIDTH_TILE])
        maxima = pl.maximum(maxima, scores)
    lane_maxima = pl.row_max(maxima)
    row_max = pl.cast(FP32_NEG_INF, pl.FP32)
    for lane in pl.range(SAMPLE_BLOCK_ROWS_TILE):
        row_max = pl.max(row_max, pl.read(lane_maxima, [lane, 0]))
    return row_max


@pl.jit.inline
def _scale_logits(
    logits: pl.Tensor,
    row: pl.Scalar[pl.INDEX],
    scaled_logits: pl.Tensor,
    scratch_row: pl.Scalar[pl.INDEX],
    row_max: pl.Scalar[pl.FP32],
    temperature: pl.Scalar[pl.FP32],
):
    block_width = SAMPLE_BLOCK_ROWS_TILE * TEMPERATURE_WIDTH_TILE
    for vocab_tile in pl.range(VOCAB // block_width):
        start = vocab_tile * block_width
        flat_scores = logits[row : row + 1, start : start + block_width]
        centered_scores = pl.sub(flat_scores, row_max)
        scaled_scores = pl.div(centered_scores, temperature)
        scaled_logits[scratch_row : scratch_row + 1, start : start + block_width] = scaled_scores
    return scaled_logits


@pl.jit.inline
def _apply_top_k_row(
    scaled_logits: pl.Tensor,
    top_k: pl.Scalar[pl.INT32],
    row: pl.Scalar[pl.INDEX],
):
    # Signed IEEE magnitude keys preserve finite FP32 order, including zero ties.
    search_lower = pl.cast(-2139095039, pl.INT64)
    search_upper = pl.cast(2139095039, pl.INT64)
    tail_rows = VOCAB // TOPK_ROW_WIDTH_TILE % SAMPLE_BLOCK_ROWS_TILE
    tail_start = VOCAB // PROBE_WIDTH_TILE * PROBE_WIDTH_TILE
    tail_flat = scaled_logits[row : row + 1, tail_start : VOCAB]
    tail_scores = pl.reshape(tail_flat, [tail_rows, TOPK_ROW_WIDTH_TILE])
    tail_padded = pl.full([SAMPLE_BLOCK_ROWS_TILE, TOPK_ROW_WIDTH_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
    tail_padded[:tail_rows, :] = tail_scores
    for search_step in pl.range(TOPK_SEARCH_STEPS):
        midpoint = search_lower + (search_upper - search_lower + pl.cast(1, pl.INT64)) // pl.cast(2, pl.INT64)
        pivot_bits_scalar = pl.cast(midpoint, pl.INT32)
        if midpoint < 0:
            pivot_bits_scalar = pl.cast(-midpoint - pl.cast(2147483648, pl.INT64), pl.INT32)
        pivot_bits = pl.full([8, 8], dtype=pl.INT32, value=0)
        pl.write(pivot_bits, [0, 0], pivot_bits_scalar)
        pivot_values = pl.reinterpret_view(pivot_bits, pl.FP32)
        pivot = pl.read(pivot_values, [0, 0])
        tail_mask = pl.cmp(tail_padded, pivot, cmp_type=5)
        tail_counts = pl.row_sum(tail_mask)
        counts = pl.mul(tail_counts, 0.0)
        for count_block in pl.range(VOCAB // PROBE_WIDTH_TILE):
            count_start = count_block * PROBE_WIDTH_TILE
            count_flat = scaled_logits[row : row + 1, count_start : count_start + PROBE_WIDTH_TILE]
            count_scores = pl.reshape(count_flat, [SAMPLE_BLOCK_ROWS_TILE, TOPK_ROW_WIDTH_TILE])
            count_mask = pl.cmp(count_scores, pivot, cmp_type=5)
            count_rows = pl.row_sum(count_mask)
            counts = pl.add(counts, count_rows)
        count = pl.cast(0, pl.INT32)
        for lane in pl.range(SAMPLE_BLOCK_ROWS_TILE):
            count = count + pl.cast(pl.read(counts, [lane, 0]), pl.INT32)
        for lane in pl.range(tail_rows):
            count = count + pl.cast(pl.read(tail_counts, [lane, 0]), pl.INT32)
        if count >= top_k:
            search_lower = midpoint
        else:
            search_upper = midpoint - pl.cast(1, pl.INT64)

    boundary_bits_scalar = pl.cast(search_lower, pl.INT32)
    if search_lower < 0:
        boundary_bits_scalar = pl.cast(-search_lower - pl.cast(2147483648, pl.INT64), pl.INT32)
    boundary_bits = pl.full([8, 8], dtype=pl.INT32, value=0)
    pl.write(boundary_bits, [0, 0], boundary_bits_scalar)
    boundary_values = pl.reinterpret_view(boundary_bits, pl.FP32)
    boundary = pl.read(boundary_values, [0, 0])

    greater_boundary_count = pl.cast(0, pl.INT32)
    for greater_block in pl.range(VOCAB // PROBE_WIDTH_TILE):
        greater_start = greater_block * PROBE_WIDTH_TILE
        greater_flat = scaled_logits[row : row + 1, greater_start : greater_start + PROBE_WIDTH_TILE]
        greater_scores = pl.reshape(greater_flat, [SAMPLE_BLOCK_ROWS_TILE, TOPK_ROW_WIDTH_TILE])
        greater_boundary_mask = pl.cmp(greater_scores, boundary, cmp_type=4)
        greater_boundary_rows = pl.row_sum(greater_boundary_mask)
        for greater_lane in pl.range(SAMPLE_BLOCK_ROWS_TILE):
            greater_lane_fp32 = pl.read(greater_boundary_rows, [greater_lane, 0])
            greater_lane_count = pl.cast(greater_lane_fp32, pl.INT32)
            greater_boundary_count = greater_boundary_count + greater_lane_count

    greater_tail_mask = pl.cmp(tail_padded, boundary, cmp_type=4)
    greater_tail_rows = pl.row_sum(greater_tail_mask)
    for greater_tail_lane in pl.range(tail_rows):
        greater_tail_fp32 = pl.read(greater_tail_rows, [greater_tail_lane, 0])
        greater_tail_count = pl.cast(greater_tail_fp32, pl.INT32)
        greater_boundary_count = greater_boundary_count + greater_tail_count

    boundary_keep_count = top_k - greater_boundary_count
    boundary_kept = pl.cast(0, pl.INT32)
    for mask_tile in pl.range(VOCAB // FILTER_WIDTH_TILE):
        mask_start = mask_tile * FILTER_WIDTH_TILE
        mask_scores_flat = scaled_logits[row : row + 1, mask_start : mask_start + FILTER_WIDTH_TILE]
        mask_scores = pl.reshape(mask_scores_flat, [SAMPLE_BLOCK_ROWS_TILE, FILTER_LANE_TILE])
        boundary_mask = pl.cmp(mask_scores, boundary, cmp_type=0)
        boundary_rows = pl.row_sum(boundary_mask)
        tile_boundary_count = pl.cast(0, pl.INT32)
        for boundary_lane in pl.range(SAMPLE_BLOCK_ROWS_TILE):
            lane_count_fp32 = pl.read(boundary_rows, [boundary_lane, 0])
            tile_boundary_count = tile_boundary_count + pl.cast(lane_count_fp32, pl.INT32)

        if boundary_kept + tile_boundary_count <= boundary_keep_count:
            keep_mask = pl.cmp(mask_scores, boundary, cmp_type=5)
            reject_mask = pl.cmp(mask_scores, boundary, cmp_type=2)
            kept_scores = pl.mul(mask_scores, keep_mask)
            rejected_scores = pl.mul(reject_mask, FP32_NEG_INF)
            filtered_scores = pl.add(kept_scores, rejected_scores)
            filtered_flat = pl.reshape(filtered_scores, [1, FILTER_WIDTH_TILE])
            scaled_logits[row : row + 1, mask_start : mask_start + FILTER_WIDTH_TILE] = filtered_flat
            boundary_kept = boundary_kept + tile_boundary_count
        elif boundary_kept >= boundary_keep_count:
            keep_mask = pl.cmp(mask_scores, boundary, cmp_type=4)
            reject_mask = pl.cmp(mask_scores, boundary, cmp_type=3)
            kept_scores = pl.mul(mask_scores, keep_mask)
            rejected_scores = pl.mul(reject_mask, FP32_NEG_INF)
            filtered_scores = pl.add(kept_scores, rejected_scores)
            filtered_flat = pl.reshape(filtered_scores, [1, FILTER_WIDTH_TILE])
            scaled_logits[row : row + 1, mask_start : mask_start + FILTER_WIDTH_TILE] = filtered_flat
        else:
            for mask_lane in pl.range(FILTER_WIDTH_TILE):
                mask_row = mask_lane // FILTER_LANE_TILE
                mask_column = mask_lane % FILTER_LANE_TILE
                token_score = pl.read(mask_scores, [mask_row, mask_column])
                if token_score < boundary:
                    pl.write(mask_scores, [mask_row, mask_column], FP32_NEG_INF)
                elif token_score == boundary:
                    if boundary_kept < boundary_keep_count:
                        boundary_kept = boundary_kept + pl.cast(1, pl.INT32)
                    else:
                        pl.write(mask_scores, [mask_row, mask_column], FP32_NEG_INF)
            filtered_flat = pl.reshape(mask_scores, [1, FILTER_WIDTH_TILE])
            scaled_logits[row : row + 1, mask_start : mask_start + FILTER_WIDTH_TILE] = filtered_flat
    return scaled_logits


@pl.jit.inline
def _top_p_mass(
    probabilities: pl.Tensor,
    row: pl.Scalar[pl.INDEX],
    pivot: pl.Scalar[pl.FP32],
):
    """Sum one row's probabilities greater than or equal to the pivot."""
    tail_start = VOCAB // PROBE_WIDTH_TILE * PROBE_WIDTH_TILE
    tail_rows = VOCAB // TOPK_ROW_WIDTH_TILE % SAMPLE_BLOCK_ROWS_TILE
    tail_flat = probabilities[row : row + 1, tail_start : VOCAB]
    tail_scores = pl.reshape(tail_flat, [tail_rows, TOPK_ROW_WIDTH_TILE])
    tail_pad = pl.full([SAMPLE_BLOCK_ROWS_TILE, TOPK_ROW_WIDTH_TILE], dtype=pl.FP32, value=0.0)
    tail_pad[0:tail_rows, :] = tail_scores
    tail_mass_mask = pl.cmp(tail_pad, pivot, cmp_type=5)
    masked_tail = pl.mul(tail_pad, tail_mass_mask)
    masses = pl.row_sum(masked_tail)
    for mass_block in pl.range(VOCAB // PROBE_WIDTH_TILE):
        mass_start = mass_block * PROBE_WIDTH_TILE
        mass_flat = probabilities[row : row + 1, mass_start : mass_start + PROBE_WIDTH_TILE]
        mass_scores = pl.reshape(mass_flat, [SAMPLE_BLOCK_ROWS_TILE, TOPK_ROW_WIDTH_TILE])
        mass_mask = pl.cmp(mass_scores, pivot, cmp_type=5)
        masked_scores = pl.mul(mass_scores, mass_mask)
        mass_rows = pl.row_sum(masked_scores)
        masses = pl.add(masses, mass_rows)
    mass = pl.cast(0.0, pl.FP32)
    for lane in pl.range(SAMPLE_BLOCK_ROWS_TILE):
        mass = mass + pl.read(masses, [lane, 0])
    return mass


@pl.jit.inline
def _apply_top_p_row(
    scaled_logits: pl.Tensor,
    top_p: pl.Scalar[pl.FP32],
    probabilities: pl.Tensor,
    row: pl.Scalar[pl.INDEX],
):
    row_max = pl.cast(FP32_NEG_INF, pl.FP32)
    for max_block in pl.range(VOCAB // PROBE_WIDTH_TILE):
        max_start = max_block * PROBE_WIDTH_TILE
        max_flat = scaled_logits[row : row + 1, max_start : max_start + PROBE_WIDTH_TILE]
        max_scores = pl.reshape(max_flat, [SAMPLE_BLOCK_ROWS_TILE, TOPK_ROW_WIDTH_TILE])
        block_maxima = pl.row_max(max_scores)
        for max_lane in pl.range(SAMPLE_BLOCK_ROWS_TILE):
            lane_maximum = pl.read(block_maxima, [max_lane, 0])
            if lane_maximum > row_max:
                row_max = lane_maximum
    tail_start = VOCAB // PROBE_WIDTH_TILE * PROBE_WIDTH_TILE
    tail_rows = VOCAB // TOPK_ROW_WIDTH_TILE % SAMPLE_BLOCK_ROWS_TILE
    tail_flat = scaled_logits[row : row + 1, tail_start : VOCAB]
    tail_scores = pl.reshape(tail_flat, [tail_rows, TOPK_ROW_WIDTH_TILE])
    tail_max_pad = pl.full([SAMPLE_BLOCK_ROWS_TILE, TOPK_ROW_WIDTH_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
    tail_max_pad[0:tail_rows, :] = tail_scores
    tail_maxima = pl.row_max(tail_max_pad)
    for tail_max_lane in pl.range(tail_rows):
        tail_maximum = pl.read(tail_maxima, [tail_max_lane, 0])
        if tail_maximum > row_max:
            row_max = tail_maximum

    normalizer = pl.cast(0.0, pl.FP32)
    for exp_block in pl.range(VOCAB // PROBE_WIDTH_TILE):
        exp_start = exp_block * PROBE_WIDTH_TILE
        exp_flat = scaled_logits[row : row + 1, exp_start : exp_start + PROBE_WIDTH_TILE]
        exp_scores = pl.reshape(exp_flat, [SAMPLE_BLOCK_ROWS_TILE, TOPK_ROW_WIDTH_TILE])
        shifted_scores = pl.sub(exp_scores, row_max)
        exp_scores = pl.exp(shifted_scores)
        exp_written = pl.reshape(exp_scores, [1, PROBE_WIDTH_TILE])
        probabilities[row : row + 1, exp_start : exp_start + PROBE_WIDTH_TILE] = exp_written
        exp_rows = pl.row_sum(exp_scores)
        for exp_lane in pl.range(SAMPLE_BLOCK_ROWS_TILE):
            lane_exponential = pl.read(exp_rows, [exp_lane, 0])
            normalizer = normalizer + lane_exponential
    tail_shifted = pl.sub(tail_scores, row_max)
    tail_exp = pl.exp(tail_shifted)
    tail_exp_written = pl.reshape(tail_exp, [1, tail_rows * TOPK_ROW_WIDTH_TILE])
    probabilities[row : row + 1, tail_start : VOCAB] = tail_exp_written
    tail_exp_pad = pl.full([SAMPLE_BLOCK_ROWS_TILE, TOPK_ROW_WIDTH_TILE], dtype=pl.FP32, value=0.0)
    tail_exp_pad[0:tail_rows, :] = tail_exp
    tail_exp_rows = pl.row_sum(tail_exp_pad)
    for tail_exp_lane in pl.range(tail_rows):
        tail_lane_exponential = pl.read(tail_exp_rows, [tail_exp_lane, 0])
        normalizer = normalizer + tail_lane_exponential

    for div_block in pl.range(VOCAB // PROBE_WIDTH_TILE):
        div_start = div_block * PROBE_WIDTH_TILE
        div_flat = probabilities[row : row + 1, div_start : div_start + PROBE_WIDTH_TILE]
        div_scores = pl.reshape(div_flat, [SAMPLE_BLOCK_ROWS_TILE, TOPK_ROW_WIDTH_TILE])
        div_scores = pl.div(div_scores, normalizer)
        div_written = pl.reshape(div_scores, [1, PROBE_WIDTH_TILE])
        probabilities[row : row + 1, div_start : div_start + PROBE_WIDTH_TILE] = div_written
    tail_prob = pl.div(tail_exp, normalizer)
    tail_prob_written = pl.reshape(tail_prob, [1, tail_rows * TOPK_ROW_WIDTH_TILE])
    probabilities[row : row + 1, tail_start : VOCAB] = tail_prob_written

    zero_pivot = pl.cast(0.0, pl.FP32)
    total_mass = _top_p_mass(probabilities, row, zero_pivot)
    if total_mass >= top_p:
        # Nucleus boundary: largest probability value whose tail mass still reaches top_p.
        search_lower = pl.cast(0, pl.INT64)
        search_upper = pl.cast(TOPP_ABOVE_ONE_BITS, pl.INT64)
        for search_step in pl.range(TOPP_SEARCH_STEPS):
            midpoint = search_lower + (search_upper - search_lower + pl.cast(1, pl.INT64)) // pl.cast(2, pl.INT64)
            pivot_bits = pl.full([8, 8], dtype=pl.INT32, value=0)
            pl.write(pivot_bits, [0, 0], pl.cast(midpoint, pl.INT32))
            pivot_values = pl.reinterpret_view(pivot_bits, pl.FP32)
            pivot = pl.read(pivot_values, [0, 0])
            probe_mass = _top_p_mass(probabilities, row, pivot)
            if probe_mass >= top_p:
                search_lower = midpoint
            else:
                search_upper = midpoint - pl.cast(1, pl.INT64)

        if search_lower > 0:
            boundary_bits = pl.full([8, 8], dtype=pl.INT32, value=0)
            pl.write(boundary_bits, [0, 0], pl.cast(search_lower, pl.INT32))
            boundary_values = pl.reinterpret_view(boundary_bits, pl.FP32)
            boundary = pl.read(boundary_values, [0, 0])

            # p > boundary is p >= nextafter(boundary): positive-FP32 bit patterns are value-ordered.
            above_bits = pl.full([8, 8], dtype=pl.INT32, value=0)
            pl.write(above_bits, [0, 0], pl.cast(search_lower + pl.cast(1, pl.INT64), pl.INT32))
            above_values = pl.reinterpret_view(above_bits, pl.FP32)
            above_pivot = pl.read(above_values, [0, 0])
            above_mass = _top_p_mass(probabilities, row, above_pivot)
            cumulative_mass = above_mass
            for mask_tile in pl.range(VOCAB // FILTER_WIDTH_TILE):
                mask_start = mask_tile * FILTER_WIDTH_TILE
                prob_flat = probabilities[row : row + 1, mask_start : mask_start + FILTER_WIDTH_TILE]
                prob_scores = pl.reshape(prob_flat, [SAMPLE_BLOCK_ROWS_TILE, FILTER_LANE_TILE])
                logit_flat = scaled_logits[row : row + 1, mask_start : mask_start + FILTER_WIDTH_TILE]
                logit_scores = pl.reshape(logit_flat, [SAMPLE_BLOCK_ROWS_TILE, FILTER_LANE_TILE])
                tie_mask = pl.cmp(prob_scores, boundary, cmp_type=0)
                tie_mass_rows = pl.row_sum(pl.mul(prob_scores, tie_mask))
                tile_tie_mass = pl.cast(0.0, pl.FP32)
                for tie_lane in pl.range(SAMPLE_BLOCK_ROWS_TILE):
                    lane_tie_mass = pl.read(tie_mass_rows, [tie_lane, 0])
                    tile_tie_mass = tile_tie_mass + lane_tie_mass

                if cumulative_mass + tile_tie_mass <= top_p:
                    keep_mask = pl.cmp(prob_scores, boundary, cmp_type=5)
                    reject_mask = pl.cmp(prob_scores, boundary, cmp_type=2)
                    kept_scores = pl.mul(logit_scores, keep_mask)
                    rejected_scores = pl.mul(reject_mask, FP32_NEG_INF)
                    filtered_scores = pl.add(kept_scores, rejected_scores)
                    filtered_flat = pl.reshape(filtered_scores, [1, FILTER_WIDTH_TILE])
                    scaled_logits[row : row + 1, mask_start : mask_start + FILTER_WIDTH_TILE] = filtered_flat
                    cumulative_mass = cumulative_mass + tile_tie_mass
                elif cumulative_mass >= top_p:
                    keep_mask = pl.cmp(prob_scores, boundary, cmp_type=4)
                    reject_mask = pl.cmp(prob_scores, boundary, cmp_type=3)
                    kept_scores = pl.mul(logit_scores, keep_mask)
                    rejected_scores = pl.mul(reject_mask, FP32_NEG_INF)
                    filtered_scores = pl.add(kept_scores, rejected_scores)
                    filtered_flat = pl.reshape(filtered_scores, [1, FILTER_WIDTH_TILE])
                    scaled_logits[row : row + 1, mask_start : mask_start + FILTER_WIDTH_TILE] = filtered_flat
                else:
                    for mask_lane in pl.range(FILTER_WIDTH_TILE):
                        mask_row = mask_lane // FILTER_LANE_TILE
                        mask_column = mask_lane % FILTER_LANE_TILE
                        token_prob = pl.read(prob_scores, [mask_row, mask_column])
                        if token_prob < boundary:
                            pl.write(logit_scores, [mask_row, mask_column], FP32_NEG_INF)
                        elif token_prob == boundary:
                            if cumulative_mass < top_p:
                                cumulative_mass = cumulative_mass + token_prob
                            else:
                                pl.write(logit_scores, [mask_row, mask_column], FP32_NEG_INF)
                    filtered_flat = pl.reshape(logit_scores, [1, FILTER_WIDTH_TILE])
                    scaled_logits[row : row + 1, mask_start : mask_start + FILTER_WIDTH_TILE] = filtered_flat
    return scaled_logits


@pl.jit.inline
def _sample_row(
    logits: pl.Tensor,
    logits_grid: pl.Tensor,
    filtered_logits: pl.Tensor,
    probabilities: pl.Tensor,
    temperatures: pl.Tensor,
    top_ks: pl.Tensor,
    top_ps: pl.Tensor,
    seeds: pl.Tensor,
    positions: pl.Tensor,
    sampled_ids: pl.Tensor,
    row: pl.Scalar[pl.INDEX],
    scratch_row: pl.Scalar[pl.INDEX],
):
    temperature = pl.read(temperatures, [row])
    if temperature >= SAMPLING_EPS:
        seed = pl.read(seeds, [row])
        position = pl.read(positions, [row])
        seed_index = pl.cast(seed, pl.INDEX)
        position_index = pl.cast(position, pl.INDEX)
        random_key_index = seed_index + position_index * POSITION_MULTIPLIER
        random_key = pl.cast(random_key_index % RANDOM_KEY_MODULUS, pl.INT32)
        row_max = _row_max(logits, row)
        top_k = pl.read(top_ks, [row])
        top_p = pl.read(top_ps, [row])
        use_top_k = top_k > 0 and top_k < VOCAB
        use_top_p = top_p > 0.0 and top_p < 1.0
        if use_top_k or use_top_p:
            _scale_logits(logits, row, filtered_logits, scratch_row, row_max, temperature)
            if use_top_k:
                _apply_top_k_row(filtered_logits, top_k, scratch_row)
            if use_top_p:
                _apply_top_p_row(filtered_logits, top_p, probabilities, scratch_row)
            zero = pl.cast(0.0, pl.FP32)
            one = pl.cast(1.0, pl.FP32)
            best_index = _gumbel_argmax(filtered_logits, scratch_row, random_key, zero, one)
        else:
            best_index = _gumbel_argmax(logits, row, random_key, row_max, temperature)
    else:
        best_index = _greedy_sample_logits(logits_grid, row)
    sampled_row = pl.create_tensor([1, SAMPLED_IDS_PAD], dtype=pl.INT32)
    sampled_row[:, :] = pl.full([1, SAMPLED_IDS_PAD], dtype=pl.INT32, value=0)
    pl.write(sampled_row, [0, 0], best_index)
    sampled_ids[row : row + 1, :] = sampled_row


@pl.jit.inline
def sample(
    logits: pl.Tensor,
    sampling_temperatures: pl.Tensor,
    sampling_top_ks: pl.Tensor,
    sampling_top_ps: pl.Tensor,
    sampling_seeds: pl.Tensor,
    sampling_positions: pl.Tensor,
    sampled_ids: pl.Tensor,
):
    """Sample each logits row in one AIV worker."""
    sample_rows = pl.tensor.dim(logits, 0)
    workers = pl.min(sample_rows, SAMPLE_WORKERS_TILE)
    logits_grid = pl.reshape(logits, [sample_rows * (VOCAB // ARGMAX_WIDTH_TILE), ARGMAX_WIDTH_TILE])
    # Each worker reuses its two scratch rows across all assigned logits rows.
    processed_logits = pl.create_tensor([workers, VOCAB], dtype=pl.FP32)
    probabilities = pl.create_tensor([workers, VOCAB], dtype=pl.FP32)
    for worker in pl.spmd(workers, name_hint="sample_rows"):
        for slot in pl.range(worker, sample_rows, workers):
            row = _sample_worker_row(slot, sample_rows)
            _sample_row(
                logits, logits_grid, processed_logits, probabilities, sampling_temperatures,
                sampling_top_ks, sampling_top_ps, sampling_seeds, sampling_positions, sampled_ids, row, worker,
            )
    return sampled_ids


@pl.jit
def sample_test(
    logits: pl.Tensor[[SAMPLE_ROWS, VOCAB], pl.FP32],
    temperatures: pl.Tensor[[SAMPLE_ROWS], pl.FP32],
    top_ks: pl.Tensor[[SAMPLE_ROWS], pl.INT32],
    top_ps: pl.Tensor[[SAMPLE_ROWS], pl.FP32],
    seeds: pl.Tensor[[SAMPLE_ROWS], pl.INT32],
    positions: pl.Tensor[[SAMPLE_ROWS], pl.INT32],
    sampled_ids: pl.Out[pl.Tensor[[SAMPLE_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
):
    return sample(logits, temperatures, top_ks, top_ps, seeds, positions, sampled_ids)


def _gumbel_noise(seed, position):
    import numpy as np

    counters = np.arange(VOCAB, dtype=np.uint32)
    random_key = np.uint32((seed + position * POSITION_MULTIPLIER) % RANDOM_KEY_MODULUS)

    def hash_bits(values, multiplier, offset):
        values = np.asarray(values, dtype=np.uint32)
        with np.errstate(over="ignore"):
            values = values + np.uint32(offset)
            values = values & np.uint32(0x7FFFFFFF)
            values = values ^ (values >> np.uint32(16))
            values = values * np.uint32(multiplier)
            values = values & np.uint32(0x7FFFFFFF)
            values = values ^ (values >> np.uint32(16))
            values = values * np.uint32(multiplier)
            values = values & np.uint32(0x7FFFFFFF)
            values = values ^ (values >> np.uint32(16))
        return values

    random_bits = hash_bits(counters, HASH_MULTIPLIER, TOKEN_HASH_OFFSET) ^ hash_bits(
        random_key, KEY_HASH_MULTIPLIER, KEY_HASH_OFFSET
    )
    uniform_bits = (random_bits >> np.uint32(8)).astype(np.float32)
    uniform_centered = uniform_bits + np.float32(0.5)
    uniform = uniform_centered * np.float32(UINT23_SCALE)
    return -np.log(-np.log(uniform))


def build_tensor_specs(temperature=None, top_k=None, top_p=None, *, dspark_fixture=False):
    import torch

    from golden import TensorSpec

    def init_logits():
        generator = torch.Generator().manual_seed(20260821)
        logits = torch.randn(SAMPLE_ROWS, VOCAB, generator=generator, dtype=torch.float32)
        logits[0, 7] = 20.0
        if SAMPLE_ROWS > 2:
            logits[2].zero_()
        if SAMPLE_ROWS > 3:
            logits[3].fill_(-1.0)
            logits[3, [17, 60000, VOCAB - 1]] = 1.0
        if SAMPLE_ROWS > 4:
            logits[4, 42] = 20.0
        if SAMPLE_ROWS > 5:
            logits[5, 0:1200] = 10.0
        if SAMPLE_ROWS > 6:
            logits[6] = torch.linspace(-1e-10, 1e-10, VOCAB)
            logits[6, 0] = 1.0
        return logits

    def repeat_rows(values, dtype):
        repeats = (SAMPLE_ROWS + len(values) - 1) // len(values)
        return torch.tensor(values, dtype=dtype).repeat(repeats)[:SAMPLE_ROWS]

    temperatures = temperature
    top_ks = top_k
    top_ps = top_p
    seeds = repeat_rows([1, 7, 19, 1234, 42, 99, 2026, 65537], torch.int32)
    positions = repeat_rows([0, 1, 17, 1024, 4, 55, 4096, 32767], torch.int32)
    if temperatures is None:
        temperatures = repeat_rows([0.0, 0.3, 0.7, 1.0, 0.0, 0.5, 1.3, 2.0], torch.float32)
    if top_ks is None:
        top_ks = repeat_rows([VOCAB, 1, 8, 32, 128, 1000, 4096, VOCAB], torch.int32)
    if top_ps is None:
        top_ps = repeat_rows([1.0, 0.9, 0.95, 0.5, 1.0, 0.99, 0.8, 0.999], torch.float32)
    if dspark_fixture:
        batch = SAMPLE_ROWS // DECODE_SEQ
        seeds = torch.arange(batch, dtype=torch.int32).repeat_interleave(DECODE_SEQ)
        temperatures = torch.where(seeds % 2 == 0, 0.7, 0.0)
        positions = torch.arange(DECODE_SEQ, dtype=torch.int32).repeat(batch)
        top_ks = 0
        top_ps = 0.0

    return [
        TensorSpec("logits", [SAMPLE_ROWS, VOCAB], torch.float32, init_value=init_logits),
        TensorSpec("temperatures", [SAMPLE_ROWS], torch.float32, init_value=temperatures),
        TensorSpec("top_ks", [SAMPLE_ROWS], torch.int32, init_value=top_ks),
        TensorSpec("top_ps", [SAMPLE_ROWS], torch.float32, init_value=top_ps),
        TensorSpec("seeds", [SAMPLE_ROWS], torch.int32, init_value=seeds),
        TensorSpec("positions", [SAMPLE_ROWS], torch.int32, init_value=positions),
        TensorSpec("sampled_ids", [SAMPLE_ROWS, SAMPLED_IDS_PAD], torch.int32),
    ]


def golden_sample(tensors):
    import torch

    tensors["sampled_ids"].zero_()
    for row in range(SAMPLE_ROWS):
        logits = tensors["logits"][row].float()
        temperature = float(tensors["temperatures"][row])
        top_k = int(tensors["top_ks"][row])
        top_p = float(tensors["top_ps"][row])
        if temperature < SAMPLING_EPS:
            selected = torch.argmax(logits)
        else:
            seed = int(tensors["seeds"][row])
            position = int(tensors["positions"][row])
            noise = torch.from_numpy(_gumbel_noise(seed, position))
            scaled_logits = (logits - logits.max()) / temperature
            if 0 < top_k < VOCAB:
                boundary = torch.topk(scaled_logits, top_k).values[-1]
                greater_mask = scaled_logits > boundary
                boundary_indices = torch.nonzero(scaled_logits == boundary).flatten()
                boundary_keep = top_k - int(torch.sum(greater_mask))
                keep_mask = greater_mask.clone()
                keep_mask[boundary_indices[:boundary_keep]] = True
                filtered_logits = torch.full_like(scaled_logits, -torch.inf)
                filtered_logits[keep_mask] = scaled_logits[keep_mask]
                scaled_logits = filtered_logits
            if 0.0 < top_p < 1.0:
                # FP32 nucleus cutoff.
                probs = torch.softmax(scaled_logits, dim=-1)
                total_mass = probs.sum()
                if bool(total_mass >= top_p):
                    sorted_probs, _ = torch.sort(probs, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=0)
                    target = torch.tensor(top_p, dtype=torch.float32)
                    boundary_index = int(torch.searchsorted(cumulative, target, right=False))
                    boundary = sorted_probs[boundary_index]
                    keep_mask = probs > boundary
                    cumulative_mass = probs[keep_mask].sum()
                    tie_indices = torch.nonzero(probs == boundary).flatten()
                    for tie_index in tie_indices.tolist():
                        if bool(cumulative_mass < target):
                            keep_mask[tie_index] = True
                            cumulative_mass = cumulative_mass + probs[tie_index]
                    filtered_logits = torch.full_like(scaled_logits, -torch.inf)
                    filtered_logits[keep_mask] = scaled_logits[keep_mask]
                    scaled_logits = filtered_logits
            selected = torch.argmax(scaled_logits + noise)
        tensors["sampled_ids"][row, 0] = selected.to(torch.int32)


if __name__ == "__main__":
    import argparse
    import math

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", choices=("a5",), default="a5")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=SAMPLE_ROWS)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--enable-chip-swimlane", nargs="?", const=1, type=int, choices=(1, 2, 3, 4), default=0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--dspark-fixture", action="store_true", help="Use the DSpark decode smoke sampling parameters")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    args = parser.parse_args()

    if args.rows != SAMPLE_ROWS:
        parser.error(f"--rows has conflicting values; the entry compiled {SAMPLE_ROWS} rows")
    overrides = (args.temperature, args.top_k, args.top_p)
    if args.dspark_fixture and (SAMPLE_ROWS % DECODE_SEQ != 0 or any(value is not None for value in overrides)):
        parser.error("--dspark-fixture requires complete decode requests and default sampling parameters")
    if args.temperature is not None and not math.isfinite(args.temperature):
        parser.error(f"--temperature must be finite, got {args.temperature}")
    if args.temperature is not None and args.temperature < 0.0:
        parser.error(f"--temperature must be non-negative, got {args.temperature}")
    if args.top_p is not None and not 0.0 < args.top_p <= 1.0:
        parser.error(f"--top-p must be in (0, 1], got {args.top_p}; 1.0 disables nucleus filtering")
    result = run(
        fn=sample_test,
        specs=build_tensor_specs(args.temperature, args.top_k, args.top_p, dspark_fixture=args.dspark_fixture),
        golden_fn=golden_sample,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        config=dict(
            backend_type=args.platform,
            execution_mode=False,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=0,
        atol=0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
