# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Greedy and temperature/top-k/Gumbel sampling for DeepSeek-V4 logits."""

import pypto.language as pl

from config import DECODE_TOKENS, FLASH as M, FP32_NEG_INF


# model config
VOCAB = M.vocab_size
SAMPLE_ROWS = DECODE_TOKENS
SAMPLED_IDS_PAD = 8

# tiling
SAMPLE_ROW_WIDTH_TILE = 256
SAMPLE_BLOCK_ROWS_TILE = 8
SAMPLE_CANDIDATES_TILE = 512
GREEDY_ROW_WIDTH_TILE = 808
GREEDY_BLOCK_ROWS_TILE = 8
TOPK_ROW_WIDTH_TILE = 640
TOPK_SEARCH_STEPS = 18

# sampling constants
SAMPLING_EPS = 1e-5
RANDOM_KEY_MODULUS = 1073741824
HASH_MULTIPLIER = 0x045D9F3B
POSITION_MULTIPLIER = 65537
UINT23_SCALE = 1.0 / 8388608.0
FP32_POS_INF = 3.4028234663852886e38
GREEDY_INDEX_SENTINEL = 1073741824


@pl.jit.inline
def _counter_gumbel(
    block: pl.Scalar[pl.INDEX],
    random_key: pl.Scalar[pl.INT32],
):
    """Generate one Gumbel-noise tile from a key and vocabulary counters."""
    counter_zeros = pl.full([SAMPLE_BLOCK_ROWS_TILE, SAMPLE_ROW_WIDTH_TILE], dtype=pl.INT32, value=0)
    column_ids = pl.arange(0, [1, SAMPLE_ROW_WIDTH_TILE], dtype=pl.INT32)
    column_counters = pl.col_expand(counter_zeros, column_ids)
    row_counters_pad = pl.full([SAMPLE_BLOCK_ROWS_TILE, SAMPLE_BLOCK_ROWS_TILE], dtype=pl.INT32, value=0)
    for row in pl.range(SAMPLE_BLOCK_ROWS_TILE):
        row_counter = pl.cast(row * SAMPLE_ROW_WIDTH_TILE, pl.INT32)
        pl.write(row_counters_pad, [row, 0], row_counter)
    row_counters = pl.row_max(row_counters_pad)
    counters = pl.row_expand_add(column_counters, row_counters)
    block_base = pl.cast(block * SAMPLE_BLOCK_ROWS_TILE * SAMPLE_ROW_WIDTH_TILE, pl.INT32)
    counters = pl.add(counters, block_base)

    key_zeros = pl.mul(column_ids, pl.cast(0, pl.INT32))
    key_row = pl.add(key_zeros, random_key)
    random_key_tile = pl.col_expand(counter_zeros, key_row)
    positive_mask = pl.full([SAMPLE_BLOCK_ROWS_TILE, SAMPLE_ROW_WIDTH_TILE], dtype=pl.INT32, value=0x7FFFFFFF)
    random_bits = pl.xor(counters, random_key_tile)
    shifted = pl.shrs(random_bits, 16)
    random_bits = pl.xor(random_bits, shifted)
    random_bits = pl.mul(random_bits, pl.cast(HASH_MULTIPLIER, pl.INT32))
    random_bits = pl.and_(random_bits, positive_mask)
    shifted = pl.shrs(random_bits, 16)
    random_bits = pl.xor(random_bits, shifted)
    random_bits = pl.mul(random_bits, pl.cast(HASH_MULTIPLIER, pl.INT32))
    random_bits = pl.and_(random_bits, positive_mask)
    shifted = pl.shrs(random_bits, 16)
    random_bits = pl.xor(random_bits, shifted)

    uniform_bits = pl.shrs(random_bits, 8)
    uniform_fp32 = pl.cast(uniform_bits, pl.FP32)
    uniform_centered = pl.add(uniform_fp32, 0.5)
    uniform = pl.mul(uniform_centered, UINT23_SCALE)
    log_uniform = pl.log(uniform)
    negative_log_uniform = pl.neg(log_uniform)
    log_negative_log_uniform = pl.log(negative_log_uniform)
    return pl.neg(log_negative_log_uniform)


@pl.jit.inline
def _sample_filtered_logits(
    logits: pl.Tensor,
    row: pl.Scalar[pl.INDEX],
    random_flag: pl.Scalar[pl.INT32],
    random_key: pl.Scalar[pl.INT32],
):
    """Sample one preprocessed logits row with greedy decoding or Gumbel-max."""
    candidate_maxima = pl.full(
        [SAMPLE_BLOCK_ROWS_TILE, SAMPLE_CANDIDATES_TILE], dtype=pl.FP32, value=FP32_NEG_INF
    )
    candidate_token_ids = pl.full([1, SAMPLE_CANDIDATES_TILE], dtype=pl.INT32, value=0)
    for block in pl.range(VOCAB // SAMPLE_ROW_WIDTH_TILE // SAMPLE_BLOCK_ROWS_TILE):
        token_start = block * SAMPLE_BLOCK_ROWS_TILE * SAMPLE_ROW_WIDTH_TILE
        scores_flat = pl.slice(
            logits, [1, SAMPLE_BLOCK_ROWS_TILE * SAMPLE_ROW_WIDTH_TILE], [row, token_start]
        )
        scores = pl.reshape(scores_flat, [SAMPLE_BLOCK_ROWS_TILE, SAMPLE_ROW_WIDTH_TILE])
        if random_flag > 0:
            gumbel_noise = _counter_gumbel(block, random_key)
            scores = pl.add(scores, gumbel_noise)
        local_winners = pl.row_argmax(scores)
        for lane in pl.range(SAMPLE_BLOCK_ROWS_TILE):
            local_token = pl.read(local_winners, [lane, 0])
            local_score = pl.read(scores, [lane, pl.cast(local_token, pl.INDEX)])
            candidate = block * SAMPLE_BLOCK_ROWS_TILE + lane
            token_base = token_start + lane * SAMPLE_ROW_WIDTH_TILE
            token_id = pl.cast(token_base, pl.INT32) + local_token
            pl.write(candidate_maxima, [0, candidate], local_score)
            pl.write(candidate_token_ids, [0, candidate], token_id)

    tail_start = (
        VOCAB
        // SAMPLE_ROW_WIDTH_TILE
        // SAMPLE_BLOCK_ROWS_TILE
        * SAMPLE_BLOCK_ROWS_TILE
        * SAMPLE_ROW_WIDTH_TILE
    )
    tail_flat = pl.slice(
        logits,
        [1, VOCAB // SAMPLE_ROW_WIDTH_TILE % SAMPLE_BLOCK_ROWS_TILE * SAMPLE_ROW_WIDTH_TILE],
        [row, tail_start],
    )
    tail_scores = pl.reshape(
        tail_flat,
        [VOCAB // SAMPLE_ROW_WIDTH_TILE % SAMPLE_BLOCK_ROWS_TILE, SAMPLE_ROW_WIDTH_TILE],
    )
    tail_scores_pad = pl.full(
        [SAMPLE_BLOCK_ROWS_TILE, SAMPLE_ROW_WIDTH_TILE], dtype=pl.FP32, value=FP32_NEG_INF
    )
    tail_scores_pad[0 : VOCAB // SAMPLE_ROW_WIDTH_TILE % SAMPLE_BLOCK_ROWS_TILE, 0:SAMPLE_ROW_WIDTH_TILE] = (
        tail_scores
    )
    tail_block_scores = pl.mul(tail_scores_pad, 1.0)
    if random_flag > 0:
        tail_gumbel_noise = _counter_gumbel(
            VOCAB // SAMPLE_ROW_WIDTH_TILE // SAMPLE_BLOCK_ROWS_TILE,
            random_key,
        )
        tail_block_scores = pl.add(tail_scores_pad, tail_gumbel_noise)
    local_winners = pl.row_argmax(tail_block_scores)
    for lane in pl.range(VOCAB // SAMPLE_ROW_WIDTH_TILE % SAMPLE_BLOCK_ROWS_TILE):
        local_token = pl.read(local_winners, [lane, 0])
        local_score = pl.read(tail_block_scores, [lane, pl.cast(local_token, pl.INDEX)])
        candidate = VOCAB // SAMPLE_ROW_WIDTH_TILE // SAMPLE_BLOCK_ROWS_TILE * SAMPLE_BLOCK_ROWS_TILE + lane
        token_base = tail_start + lane * SAMPLE_ROW_WIDTH_TILE
        token_id = pl.cast(token_base, pl.INT32) + local_token
        pl.write(candidate_maxima, [0, candidate], local_score)
        pl.write(candidate_token_ids, [0, candidate], token_id)

    winning_candidate = pl.read(pl.row_argmax(candidate_maxima), [0, 0])
    return pl.read(candidate_token_ids, [0, pl.cast(winning_candidate, pl.INDEX)])


@pl.jit.inline
def _greedy_sample_logits(
    logits_grid: pl.Tensor,
    row: pl.Scalar[pl.INDEX],
):
    """Select the first maximum token id from one full-vocabulary logits row."""
    row_base = row * (VOCAB // GREEDY_ROW_WIDTH_TILE)
    running_max = pl.full(
        [GREEDY_BLOCK_ROWS_TILE, GREEDY_ROW_WIDTH_TILE], dtype=pl.FP32, value=FP32_NEG_INF
    )
    running_base = pl.full([GREEDY_BLOCK_ROWS_TILE, GREEDY_ROW_WIDTH_TILE], dtype=pl.INT32, value=0)
    for block in pl.range(VOCAB // GREEDY_ROW_WIDTH_TILE // GREEDY_BLOCK_ROWS_TILE):
        block_row = row_base + block * GREEDY_BLOCK_ROWS_TILE
        scores = logits_grid[block_row : block_row + GREEDY_BLOCK_ROWS_TILE, 0:GREEDY_ROW_WIDTH_TILE]
        is_newer = pl.cmp(scores, running_max, cmp_type=4)
        newer = pl.cast(is_newer, target_type=pl.INT32)
        running_max = pl.maximum(running_max, scores)
        block_base = pl.cast(block * GREEDY_BLOCK_ROWS_TILE * GREEDY_ROW_WIDTH_TILE, pl.INT32)
        to_new = pl.add(pl.neg(running_base), block_base)
        running_base = pl.add(running_base, pl.mul(newer, to_new))

    lane_maxima = pl.row_max(running_max)
    lane_zeros = pl.full([GREEDY_BLOCK_ROWS_TILE, GREEDY_ROW_WIDTH_TILE], dtype=pl.FP32, value=0.0)
    lane_broadcast = pl.row_expand_add(lane_zeros, lane_maxima)
    best_value = pl.read(pl.col_max(lane_broadcast), [0, 0])

    ramp_zeros = pl.full([GREEDY_BLOCK_ROWS_TILE, GREEDY_ROW_WIDTH_TILE], dtype=pl.INT32, value=0)
    column_ids = pl.arange(0, [1, GREEDY_ROW_WIDTH_TILE], dtype=pl.INT32)
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
    for lane in pl.range(1, GREEDY_BLOCK_ROWS_TILE):
        lane_term = pl.cast(lane * GREEDY_ROW_WIDTH_TILE, pl.INT32)
        lane_best = pl.read(lane_indices, [lane, 0]) + lane_term
        best_index = pl.min(best_index, lane_best)
    return best_index


@pl.jit.inline
def apply_temperature(
    logits: pl.Tensor,
    temperatures: pl.Tensor,
    scaled_logits: pl.Tensor,
):
    """Apply per-row temperature in an independent vocab-tiled stage."""
    sample_rows = pl.tensor.dim(logits, 0)
    for row in pl.spmd(sample_rows, name_hint="sample_apply_temperature"):
        temperature = pl.read(temperatures, [row])
        if temperature >= SAMPLING_EPS:
            for vocab_tile in pl.range(VOCAB // SAMPLE_ROW_WIDTH_TILE):
                vocab_start = vocab_tile * SAMPLE_ROW_WIDTH_TILE
                scores = pl.slice(logits, [1, SAMPLE_ROW_WIDTH_TILE], [row, vocab_start])
                scaled_scores = pl.div(scores, temperature)
                scaled_logits[row : row + 1, vocab_start : vocab_start + SAMPLE_ROW_WIDTH_TILE] = scaled_scores
    return scaled_logits


@pl.jit.inline
def apply_top_k(
    scaled_logits: pl.Tensor,
    temperatures: pl.Tensor,
    top_ks: pl.Tensor,
):
    """Mask logits below each row's top-k boundary in an independent stage."""
    sample_rows = pl.tensor.dim(scaled_logits, 0)
    for row in pl.spmd(sample_rows, name_hint="sample_apply_top_k"):
        temperature = pl.read(temperatures, [row])
        top_k = pl.read(top_ks, [row])
        if temperature >= SAMPLING_EPS and top_k > 0 and top_k < VOCAB:
            search_lower = pl.cast(FP32_POS_INF, pl.FP32)
            search_upper = pl.cast(FP32_NEG_INF, pl.FP32)
            for extrema_block in pl.range(VOCAB // TOPK_ROW_WIDTH_TILE // SAMPLE_BLOCK_ROWS_TILE):
                extrema_start = extrema_block * SAMPLE_BLOCK_ROWS_TILE * TOPK_ROW_WIDTH_TILE
                extrema_flat = pl.slice(
                    scaled_logits,
                    [1, SAMPLE_BLOCK_ROWS_TILE * TOPK_ROW_WIDTH_TILE],
                    [row, extrema_start],
                )
                extrema_scores = pl.reshape(extrema_flat, [SAMPLE_BLOCK_ROWS_TILE, TOPK_ROW_WIDTH_TILE])
                block_minima = pl.row_min(extrema_scores)
                block_maxima = pl.row_max(extrema_scores)
                for extrema_lane in pl.range(SAMPLE_BLOCK_ROWS_TILE):
                    lane_minimum = pl.read(block_minima, [extrema_lane, 0])
                    lane_maximum = pl.read(block_maxima, [extrema_lane, 0])
                    if lane_minimum < search_lower:
                        search_lower = lane_minimum
                    if lane_maximum > search_upper:
                        search_upper = lane_maximum

            tail_start = (
                VOCAB
                // TOPK_ROW_WIDTH_TILE
                // SAMPLE_BLOCK_ROWS_TILE
                * SAMPLE_BLOCK_ROWS_TILE
                * TOPK_ROW_WIDTH_TILE
            )
            tail_flat = pl.slice(
                scaled_logits,
                [1, VOCAB // TOPK_ROW_WIDTH_TILE % SAMPLE_BLOCK_ROWS_TILE * TOPK_ROW_WIDTH_TILE],
                [row, tail_start],
            )
            tail_scores = pl.reshape(
                tail_flat,
                [VOCAB // TOPK_ROW_WIDTH_TILE % SAMPLE_BLOCK_ROWS_TILE, TOPK_ROW_WIDTH_TILE],
            )
            tail_min_scores = pl.full(
                [SAMPLE_BLOCK_ROWS_TILE, TOPK_ROW_WIDTH_TILE],
                dtype=pl.FP32,
                value=FP32_POS_INF,
            )
            tail_min_scores[0 : VOCAB // TOPK_ROW_WIDTH_TILE % SAMPLE_BLOCK_ROWS_TILE, :] = tail_scores
            tail_max_scores = pl.full(
                [SAMPLE_BLOCK_ROWS_TILE, TOPK_ROW_WIDTH_TILE],
                dtype=pl.FP32,
                value=FP32_NEG_INF,
            )
            tail_max_scores[0 : VOCAB // TOPK_ROW_WIDTH_TILE % SAMPLE_BLOCK_ROWS_TILE, :] = tail_scores
            tail_minima = pl.row_min(tail_min_scores)
            tail_maxima = pl.row_max(tail_max_scores)
            for tail_lane in pl.range(VOCAB // TOPK_ROW_WIDTH_TILE % SAMPLE_BLOCK_ROWS_TILE):
                tail_minimum = pl.read(tail_minima, [tail_lane, 0])
                tail_maximum = pl.read(tail_maxima, [tail_lane, 0])
                if tail_minimum < search_lower:
                    search_lower = tail_minimum
                if tail_maximum > search_upper:
                    search_upper = tail_maximum

            for search_step in pl.range(TOPK_SEARCH_STEPS):
                lower_tile = pl.full(
                    [SAMPLE_BLOCK_ROWS_TILE, SAMPLE_BLOCK_ROWS_TILE],
                    dtype=pl.FP32,
                    value=0.0,
                )
                upper_tile = pl.full(
                    [SAMPLE_BLOCK_ROWS_TILE, SAMPLE_BLOCK_ROWS_TILE],
                    dtype=pl.FP32,
                    value=0.0,
                )
                pl.write(lower_tile, [0, 0], search_lower)
                pl.write(upper_tile, [0, 0], search_upper)
                search_range_tile = pl.sub(upper_tile, lower_tile)
                lower_pivot_tile = pl.add(lower_tile, pl.mul(search_range_tile, 1.0 / 3.0))
                upper_pivot_tile = pl.add(lower_tile, pl.mul(search_range_tile, 2.0 / 3.0))
                lower_pivot = pl.read(lower_pivot_tile, [0, 0])
                upper_pivot = pl.read(upper_pivot_tile, [0, 0])
                lower_count = pl.cast(0, pl.INT32)
                upper_count = pl.cast(0, pl.INT32)
                for count_block in pl.range(VOCAB // TOPK_ROW_WIDTH_TILE // SAMPLE_BLOCK_ROWS_TILE):
                    count_start = count_block * SAMPLE_BLOCK_ROWS_TILE * TOPK_ROW_WIDTH_TILE
                    count_flat = pl.slice(
                        scaled_logits,
                        [1, SAMPLE_BLOCK_ROWS_TILE * TOPK_ROW_WIDTH_TILE],
                        [row, count_start],
                    )
                    count_scores = pl.reshape(count_flat, [SAMPLE_BLOCK_ROWS_TILE, TOPK_ROW_WIDTH_TILE])
                    lower_mask = pl.cmp(count_scores, lower_pivot, cmp_type=4)
                    upper_mask = pl.cmp(count_scores, upper_pivot, cmp_type=4)
                    lower_rows = pl.row_sum(lower_mask)
                    upper_rows = pl.row_sum(upper_mask)
                    for count_lane in pl.range(SAMPLE_BLOCK_ROWS_TILE):
                        lower_lane_fp32 = pl.read(lower_rows, [count_lane, 0])
                        upper_lane_fp32 = pl.read(upper_rows, [count_lane, 0])
                        lower_count = lower_count + pl.cast(lower_lane_fp32, pl.INT32)
                        upper_count = upper_count + pl.cast(upper_lane_fp32, pl.INT32)

                tail_lower_mask = pl.cmp(tail_max_scores, lower_pivot, cmp_type=4)
                tail_upper_mask = pl.cmp(tail_max_scores, upper_pivot, cmp_type=4)
                tail_lower_rows = pl.row_sum(tail_lower_mask)
                tail_upper_rows = pl.row_sum(tail_upper_mask)
                for count_tail_lane in pl.range(VOCAB // TOPK_ROW_WIDTH_TILE % SAMPLE_BLOCK_ROWS_TILE):
                    lower_tail_fp32 = pl.read(tail_lower_rows, [count_tail_lane, 0])
                    upper_tail_fp32 = pl.read(tail_upper_rows, [count_tail_lane, 0])
                    lower_count = lower_count + pl.cast(lower_tail_fp32, pl.INT32)
                    upper_count = upper_count + pl.cast(upper_tail_fp32, pl.INT32)

                if upper_count >= top_k:
                    search_lower = upper_pivot
                elif lower_count >= top_k:
                    search_lower = lower_pivot
                    search_upper = upper_pivot
                else:
                    search_upper = lower_pivot

            boundary = pl.cast(FP32_POS_INF, pl.FP32)
            for boundary_block in pl.range(VOCAB // TOPK_ROW_WIDTH_TILE // SAMPLE_BLOCK_ROWS_TILE):
                boundary_start = boundary_block * SAMPLE_BLOCK_ROWS_TILE * TOPK_ROW_WIDTH_TILE
                boundary_flat = pl.slice(
                    scaled_logits,
                    [1, SAMPLE_BLOCK_ROWS_TILE * TOPK_ROW_WIDTH_TILE],
                    [row, boundary_start],
                )
                boundary_scores = pl.reshape(
                    boundary_flat,
                    [SAMPLE_BLOCK_ROWS_TILE, TOPK_ROW_WIDTH_TILE],
                )
                reject_boundary = pl.cmp(boundary_scores, search_lower, cmp_type=3)
                rejected_offset = pl.mul(reject_boundary, FP32_POS_INF)
                boundary_candidates = pl.add(boundary_scores, rejected_offset)
                boundary_minima = pl.row_min(boundary_candidates)
                for boundary_lane in pl.range(SAMPLE_BLOCK_ROWS_TILE):
                    lane_boundary = pl.read(boundary_minima, [boundary_lane, 0])
                    if lane_boundary < boundary:
                        boundary = lane_boundary

            boundary_tail_scores = pl.full(
                [SAMPLE_BLOCK_ROWS_TILE, TOPK_ROW_WIDTH_TILE],
                dtype=pl.FP32,
                value=FP32_POS_INF,
            )
            boundary_tail_scores[0 : VOCAB // TOPK_ROW_WIDTH_TILE % SAMPLE_BLOCK_ROWS_TILE, :] = tail_scores
            reject_tail_boundary = pl.cmp(boundary_tail_scores, search_lower, cmp_type=3)
            rejected_tail_offset = pl.mul(reject_tail_boundary, FP32_POS_INF)
            tail_boundary_candidates = pl.add(boundary_tail_scores, rejected_tail_offset)
            tail_boundary_minima = pl.row_min(tail_boundary_candidates)
            for boundary_tail_lane in pl.range(VOCAB // TOPK_ROW_WIDTH_TILE % SAMPLE_BLOCK_ROWS_TILE):
                tail_boundary = pl.read(tail_boundary_minima, [boundary_tail_lane, 0])
                if tail_boundary < boundary:
                    boundary = tail_boundary

            greater_boundary_count = pl.cast(0, pl.INT32)
            for greater_block in pl.range(VOCAB // TOPK_ROW_WIDTH_TILE // SAMPLE_BLOCK_ROWS_TILE):
                greater_start = greater_block * SAMPLE_BLOCK_ROWS_TILE * TOPK_ROW_WIDTH_TILE
                greater_flat = pl.slice(
                    scaled_logits,
                    [1, SAMPLE_BLOCK_ROWS_TILE * TOPK_ROW_WIDTH_TILE],
                    [row, greater_start],
                )
                greater_scores = pl.reshape(
                    greater_flat,
                    [SAMPLE_BLOCK_ROWS_TILE, TOPK_ROW_WIDTH_TILE],
                )
                greater_boundary_mask = pl.cmp(greater_scores, boundary, cmp_type=4)
                greater_boundary_rows = pl.row_sum(greater_boundary_mask)
                for greater_lane in pl.range(SAMPLE_BLOCK_ROWS_TILE):
                    greater_lane_fp32 = pl.read(greater_boundary_rows, [greater_lane, 0])
                    greater_lane_count = pl.cast(greater_lane_fp32, pl.INT32)
                    greater_boundary_count = greater_boundary_count + greater_lane_count

            greater_tail_scores_pad = pl.full(
                [SAMPLE_BLOCK_ROWS_TILE, TOPK_ROW_WIDTH_TILE],
                dtype=pl.FP32,
                value=FP32_NEG_INF,
            )
            greater_tail_scores_pad[0 : VOCAB // TOPK_ROW_WIDTH_TILE % SAMPLE_BLOCK_ROWS_TILE, :] = (
                tail_scores
            )
            greater_tail_mask = pl.cmp(greater_tail_scores_pad, boundary, cmp_type=4)
            greater_tail_rows = pl.row_sum(greater_tail_mask)
            for greater_tail_lane in pl.range(VOCAB // TOPK_ROW_WIDTH_TILE % SAMPLE_BLOCK_ROWS_TILE):
                greater_tail_fp32 = pl.read(greater_tail_rows, [greater_tail_lane, 0])
                greater_tail_count = pl.cast(greater_tail_fp32, pl.INT32)
                greater_boundary_count = greater_boundary_count + greater_tail_count

            boundary_keep_count = top_k - greater_boundary_count
            boundary_kept = pl.cast(0, pl.INT32)
            for mask_tile in pl.range(VOCAB // SAMPLE_ROW_WIDTH_TILE):
                mask_start = mask_tile * SAMPLE_ROW_WIDTH_TILE
                mask_scores_flat = pl.slice(scaled_logits, [1, SAMPLE_ROW_WIDTH_TILE], [row, mask_start])
                mask_scores = pl.reshape(mask_scores_flat, [SAMPLE_BLOCK_ROWS_TILE, 32])
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
                    filtered_flat = pl.reshape(filtered_scores, [1, SAMPLE_ROW_WIDTH_TILE])
                    scaled_logits[row : row + 1, mask_start : mask_start + SAMPLE_ROW_WIDTH_TILE] = (
                        filtered_flat
                    )
                    boundary_kept = boundary_kept + tile_boundary_count
                elif boundary_kept >= boundary_keep_count:
                    keep_mask = pl.cmp(mask_scores, boundary, cmp_type=4)
                    reject_mask = pl.cmp(mask_scores, boundary, cmp_type=3)
                    kept_scores = pl.mul(mask_scores, keep_mask)
                    rejected_scores = pl.mul(reject_mask, FP32_NEG_INF)
                    filtered_scores = pl.add(kept_scores, rejected_scores)
                    filtered_flat = pl.reshape(filtered_scores, [1, SAMPLE_ROW_WIDTH_TILE])
                    scaled_logits[row : row + 1, mask_start : mask_start + SAMPLE_ROW_WIDTH_TILE] = (
                        filtered_flat
                    )
                else:
                    for mask_lane in pl.range(SAMPLE_ROW_WIDTH_TILE):
                        mask_row = mask_lane // 32
                        mask_column = mask_lane % 32
                        token_score = pl.read(mask_scores, [mask_row, mask_column])
                        if token_score < boundary:
                            pl.write(mask_scores, [mask_row, mask_column], FP32_NEG_INF)
                        elif token_score == boundary:
                            if boundary_kept < boundary_keep_count:
                                boundary_kept = boundary_kept + pl.cast(1, pl.INT32)
                            else:
                                pl.write(mask_scores, [mask_row, mask_column], FP32_NEG_INF)
                    filtered_flat = pl.reshape(mask_scores, [1, SAMPLE_ROW_WIDTH_TILE])
                    scaled_logits[row : row + 1, mask_start : mask_start + SAMPLE_ROW_WIDTH_TILE] = (
                        filtered_flat
                    )
    return scaled_logits


@pl.jit.inline
def gumbel_sample(
    logits: pl.Tensor,
    filtered_logits: pl.Tensor,
    temperatures: pl.Tensor,
    seeds: pl.Tensor,
    positions: pl.Tensor,
    sampled_ids: pl.Tensor,
):
    """Run direct-logits greedy or filtered-logits Gumbel-max sampling."""
    sample_rows = pl.tensor.dim(filtered_logits, 0)
    logits_grid = pl.reshape(logits, [SAMPLE_ROWS * (VOCAB // GREEDY_ROW_WIDTH_TILE), GREEDY_ROW_WIDTH_TILE])
    for row in pl.spmd(sample_rows, name_hint="sample_gumbel_argmax"):
        temperature = pl.read(temperatures, [row])
        if temperature < SAMPLING_EPS:
            best_index = _greedy_sample_logits(logits_grid, row)
        else:
            random_flag = pl.cast(1, pl.INT32)
            seed = pl.read(seeds, [row])
            position = pl.read(positions, [row])
            seed_index = pl.cast(seed, pl.INDEX)
            position_index = pl.cast(position, pl.INDEX)
            random_key_index = seed_index + position_index * POSITION_MULTIPLIER
            random_key = pl.cast(random_key_index % RANDOM_KEY_MODULUS, pl.INT32)
            best_index = _sample_filtered_logits(filtered_logits, row, random_flag, random_key)
        sampled_row = pl.create_tensor([1, SAMPLED_IDS_PAD], dtype=pl.INT32)
        sampled_row[:, :] = pl.full([1, SAMPLED_IDS_PAD], dtype=pl.INT32, value=0)
        pl.write(sampled_row, [0, 0], best_index)
        sampled_ids[row : row + 1, :] = sampled_row
    return sampled_ids


@pl.jit.inline
def sample(
    logits: pl.Tensor,
    sampling_temperatures: pl.Tensor,
    sampling_top_ks: pl.Tensor,
    sampling_seeds: pl.Tensor,
    sampling_positions: pl.Tensor,
    sampled_ids: pl.Tensor,
):
    """Orchestrate independent temperature, top-k, and Gumbel stages."""
    processed_logits = pl.create_tensor([SAMPLE_ROWS, VOCAB], dtype=pl.FP32)
    apply_temperature(logits, sampling_temperatures, processed_logits)
    apply_top_k(processed_logits, sampling_temperatures, sampling_top_ks)
    return gumbel_sample(
        logits,
        processed_logits,
        sampling_temperatures,
        sampling_seeds,
        sampling_positions,
        sampled_ids,
    )


@pl.jit
def sample_test(
    logits: pl.Tensor[[SAMPLE_ROWS, VOCAB], pl.FP32],
    temperatures: pl.Tensor[[SAMPLE_ROWS], pl.FP32],
    top_ks: pl.Tensor[[SAMPLE_ROWS], pl.INT32],
    seeds: pl.Tensor[[SAMPLE_ROWS], pl.INT32],
    positions: pl.Tensor[[SAMPLE_ROWS], pl.INT32],
    sampled_ids: pl.Out[pl.Tensor[[SAMPLE_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
):
    return sample(logits, temperatures, top_ks, seeds, positions, sampled_ids)


def _gumbel_noise(seed, position):
    import numpy as np

    counters = np.arange(VOCAB, dtype=np.uint32)
    random_key = np.uint32((seed + position * POSITION_MULTIPLIER) % RANDOM_KEY_MODULUS)
    random_bits = counters ^ random_key
    random_bits ^= random_bits >> np.uint32(16)
    random_bits *= np.uint32(HASH_MULTIPLIER)
    random_bits &= np.uint32(0x7FFFFFFF)
    random_bits ^= random_bits >> np.uint32(16)
    random_bits *= np.uint32(HASH_MULTIPLIER)
    random_bits &= np.uint32(0x7FFFFFFF)
    random_bits ^= random_bits >> np.uint32(16)
    uniform_bits = (random_bits >> np.uint32(8)).astype(np.float32)
    uniform_centered = uniform_bits + np.float32(0.5)
    uniform = uniform_centered * np.float32(UINT23_SCALE)
    return -np.log(-np.log(uniform))


def build_tensor_specs(temperature=None, top_k=None):
    import torch

    from golden import TensorSpec

    def init_logits():
        generator = torch.Generator().manual_seed(20260821)
        logits = torch.randn(SAMPLE_ROWS, VOCAB, generator=generator, dtype=torch.float32)
        logits[0, 7] = 20.0
        if SAMPLE_ROWS > 4:
            logits[4, 42] = 20.0
        if SAMPLE_ROWS > 5:
            logits[5, 0:1200] = 10.0
        return logits

    def repeat_rows(values, dtype):
        repeats = (SAMPLE_ROWS + len(values) - 1) // len(values)
        return torch.tensor(values, dtype=dtype).repeat(repeats)[:SAMPLE_ROWS]

    return [
        TensorSpec("logits", [SAMPLE_ROWS, VOCAB], torch.float32, init_value=init_logits),
        TensorSpec(
            "temperatures",
            [SAMPLE_ROWS],
            torch.float32,
            init_value=(
                (lambda: torch.full([SAMPLE_ROWS], temperature, dtype=torch.float32))
                if temperature is not None
                else (lambda: repeat_rows([0.0, 0.3, 0.7, 1.0, 0.0, 0.5, 1.3, 2.0], torch.float32))
            ),
        ),
        TensorSpec(
            "top_ks",
            [SAMPLE_ROWS],
            torch.int32,
            init_value=(
                (lambda: torch.full([SAMPLE_ROWS], top_k, dtype=torch.int32))
                if top_k is not None
                else (lambda: repeat_rows([VOCAB, 1, 8, 32, 128, 1000, 4096, VOCAB], torch.int32))
            ),
        ),
        TensorSpec(
            "seeds",
            [SAMPLE_ROWS],
            torch.int32,
            init_value=lambda: repeat_rows([1, 7, 19, 1234, 42, 99, 2026, 65537], torch.int32),
        ),
        TensorSpec(
            "positions",
            [SAMPLE_ROWS],
            torch.int32,
            init_value=lambda: repeat_rows([0, 1, 17, 1024, 4, 55, 4096, 32767], torch.int32),
        ),
        TensorSpec("sampled_ids", [SAMPLE_ROWS, SAMPLED_IDS_PAD], torch.int32, is_output=True),
    ]


def golden_sample(tensors):
    import torch

    tensors["sampled_ids"].zero_()
    for row in range(SAMPLE_ROWS):
        logits = tensors["logits"][row].float()
        temperature = float(tensors["temperatures"][row])
        top_k = int(tensors["top_ks"][row])
        if temperature < SAMPLING_EPS:
            selected = torch.argmax(logits)
        else:
            seed = int(tensors["seeds"][row])
            position = int(tensors["positions"][row])
            noise = torch.from_numpy(_gumbel_noise(seed, position))
            scaled_logits = logits / temperature
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
            selected = torch.argmax(scaled_logits + noise)
        tensors["sampled_ids"][row, 0] = selected.to(torch.int32)


if __name__ == "__main__":
    import argparse

    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3sim", choices=["a2a3", "a2a3sim", "a5", "a5sim"]
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    args = parser.parse_args()

    if args.temperature is not None and args.temperature < 0.0:
        parser.error(f"--temperature must be non-negative, got {args.temperature}")
    result = run_jit(
        fn=sample_test,
        specs=build_tensor_specs(args.temperature, args.top_k),
        golden_fn=golden_sample,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=int(args.enable_l2_swimlane),
        ),
        rtol=0,
        atol=0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
