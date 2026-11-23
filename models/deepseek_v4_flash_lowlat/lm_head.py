# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Vocabulary projection and sampling for target decode."""

import pypto.language as pl
import sys
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig


if __package__:
    from .config import FLASH as M, FP32_NEG_INF
else:
    from config import FLASH as M, FP32_NEG_INF
if __package__:
    from .decode_shapes import DECODE_TOKENS
else:
    from decode_shapes import DECODE_TOKENS


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
TOPK_SEARCH_STEPS = 32

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
def _apply_temperature_row(
    logits: pl.Tensor, temperatures: pl.Tensor, scaled_logits: pl.Tensor,
    row: pl.Scalar[pl.INDEX],
):
    temperature = pl.read(temperatures, [row])
    if temperature >= SAMPLING_EPS:
        for vocab_tile in pl.range(VOCAB // SAMPLE_ROW_WIDTH_TILE):
            vocab_start = vocab_tile * SAMPLE_ROW_WIDTH_TILE
            scores = pl.slice(logits, [1, SAMPLE_ROW_WIDTH_TILE], [row, vocab_start])
            scaled_scores = pl.div(scores, temperature)
            scaled_logits[row : row + 1, vocab_start : vocab_start + SAMPLE_ROW_WIDTH_TILE] = scaled_scores
    return scaled_logits


@pl.jit.inline
def apply_temperature(
    logits: pl.Tensor,
    temperatures: pl.Tensor,
    scaled_logits: pl.Tensor,
):
    """Apply per-row temperature in an independent vocab-tiled stage."""
    sample_rows = pl.tensor.dim(logits, 0)
    for row in pl.spmd(sample_rows, name_hint="sample_apply_temperature", allow_early_resolve=True):
        _apply_temperature_row(logits, temperatures, scaled_logits, row)
    return scaled_logits


@pl.jit.inline
def _apply_top_k_row(
    scaled_logits: pl.Tensor, temperatures: pl.Tensor, top_ks: pl.Tensor,
    row: pl.Scalar[pl.INDEX],
):
    temperature = pl.read(temperatures, [row])
    top_k = pl.read(top_ks, [row])
    if temperature >= SAMPLING_EPS and top_k > 0 and top_k < VOCAB:
        # Signed IEEE magnitude keys preserve finite FP32 order, including zero ties.
        search_lower = pl.cast(-2139095039, pl.INT64)
        search_upper = pl.cast(2139095039, pl.INT64)
        tail_start = VOCAB // TOPK_ROW_WIDTH_TILE // SAMPLE_BLOCK_ROWS_TILE * SAMPLE_BLOCK_ROWS_TILE * TOPK_ROW_WIDTH_TILE
        tail_flat = pl.slice(
            scaled_logits,
            [1, VOCAB // TOPK_ROW_WIDTH_TILE % SAMPLE_BLOCK_ROWS_TILE * TOPK_ROW_WIDTH_TILE],
            [row, tail_start],
        )
        tail_scores = pl.reshape(
            tail_flat,
            [VOCAB // TOPK_ROW_WIDTH_TILE % SAMPLE_BLOCK_ROWS_TILE, TOPK_ROW_WIDTH_TILE],
        )
        tail_padded = pl.full(
            [SAMPLE_BLOCK_ROWS_TILE, TOPK_ROW_WIDTH_TILE], dtype=pl.FP32, value=FP32_NEG_INF,
        )
        tail_padded[0 : VOCAB // TOPK_ROW_WIDTH_TILE % SAMPLE_BLOCK_ROWS_TILE, :] = tail_scores
        for search_step in pl.range(TOPK_SEARCH_STEPS):
            midpoint = search_lower + (search_upper - search_lower + pl.cast(1, pl.INT64)) // pl.cast(2, pl.INT64)
            pivot_bits_scalar = pl.cast(midpoint, pl.INT32)
            if midpoint < 0:
                pivot_bits_scalar = pl.cast(-midpoint - pl.cast(2147483648, pl.INT64), pl.INT32)
            pivot_bits = pl.full([8, 8], dtype=pl.INT32, value=0)
            pl.write(pivot_bits, [0, 0], pivot_bits_scalar)
            pivot_values = pl.reinterpret_view(pivot_bits, pl.FP32)
            pivot = pl.read(pivot_values, [0, 0])
            count = pl.cast(0, pl.INT32)
            for count_block in pl.range(VOCAB // TOPK_ROW_WIDTH_TILE // SAMPLE_BLOCK_ROWS_TILE):
                count_start = count_block * SAMPLE_BLOCK_ROWS_TILE * TOPK_ROW_WIDTH_TILE
                count_flat = pl.slice(
                    scaled_logits, [1, SAMPLE_BLOCK_ROWS_TILE * TOPK_ROW_WIDTH_TILE], [row, count_start],
                )
                count_scores = pl.reshape(count_flat, [SAMPLE_BLOCK_ROWS_TILE, TOPK_ROW_WIDTH_TILE])
                count_mask = pl.cmp(count_scores, pivot, cmp_type=5)
                count_rows = pl.row_sum(count_mask)
                for count_lane in pl.range(SAMPLE_BLOCK_ROWS_TILE):
                    lane_count = pl.read(count_rows, [count_lane, 0])
                    count = count + pl.cast(lane_count, pl.INT32)
            tail_mask = pl.cmp(tail_padded, pivot, cmp_type=5)
            tail_rows = pl.row_sum(tail_mask)
            for tail_lane in pl.range(VOCAB // TOPK_ROW_WIDTH_TILE % SAMPLE_BLOCK_ROWS_TILE):
                tail_count = pl.read(tail_rows, [tail_lane, 0])
                count = count + pl.cast(tail_count, pl.INT32)
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
def apply_top_k(
    scaled_logits: pl.Tensor,
    temperatures: pl.Tensor,
    top_ks: pl.Tensor,
):
    """Mask logits below each row's top-k boundary in an independent stage."""
    sample_rows = pl.tensor.dim(scaled_logits, 0)
    for row in pl.spmd(sample_rows, name_hint="sample_apply_top_k", allow_early_resolve=True):
        _apply_top_k_row(scaled_logits, temperatures, top_ks, row)
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
def sample_from_greedy_partials(
    logits: pl.Tensor,
    sampling_temperatures: pl.Tensor,
    sampling_top_ks: pl.Tensor,
    sampling_seeds: pl.Tensor,
    sampling_positions: pl.Tensor,
    sampled_ids: pl.Tensor,
    greedy_scores: pl.Tensor,
    greedy_indices: pl.Tensor,
):
    """Sample each row with its temperature and top-k work in the same task."""
    processed_logits = pl.create_tensor([SAMPLE_ROWS, VOCAB], dtype=pl.FP32)
    greedy_parts = pl.tensor.dim(greedy_scores, 0) // SAMPLE_ROWS
    for row in pl.spmd(SAMPLE_ROWS, name_hint="sample_draw"):
        temperature = pl.read(sampling_temperatures, [row])
        if temperature < SAMPLING_EPS:
            best_score = pl.cast(FP32_NEG_INF, pl.FP32)
            best_index = pl.cast(GREEDY_INDEX_SENTINEL, pl.INT32)
            for part in pl.range(greedy_parts):
                part_row = part * SAMPLE_ROWS + row
                part_score = pl.read(greedy_scores, [part_row, 0])
                part_index = pl.read(greedy_indices, [part_row, 0])
                if part_score > best_score:
                    best_score = part_score
                    best_index = part_index
                elif part_score == best_score:
                    best_index = pl.min(best_index, part_index)
        else:
            _apply_temperature_row(logits, sampling_temperatures, processed_logits, row)
            _apply_top_k_row(processed_logits, sampling_temperatures, sampling_top_ks, row)
            random_flag = pl.cast(1, pl.INT32)
            seed = pl.read(sampling_seeds, [row])
            position = pl.read(sampling_positions, [row])
            seed_index = pl.cast(seed, pl.INDEX)
            position_index = pl.cast(position, pl.INDEX)
            random_key_index = seed_index + position_index * POSITION_MULTIPLIER
            random_key = pl.cast(random_key_index % RANDOM_KEY_MODULUS, pl.INT32)
            best_index = _sample_filtered_logits(processed_logits, row, random_flag, random_key)
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
    """Sample each row with its temperature and top-k work in the same task."""
    processed_logits = pl.create_tensor([SAMPLE_ROWS, VOCAB], dtype=pl.FP32)
    logits_grid = pl.reshape(logits, [SAMPLE_ROWS * (VOCAB // GREEDY_ROW_WIDTH_TILE), GREEDY_ROW_WIDTH_TILE])
    for row in pl.spmd(SAMPLE_ROWS, name_hint="sample_draw"):
        temperature = pl.read(sampling_temperatures, [row])
        if temperature < SAMPLING_EPS:
            best_index = _greedy_sample_logits(logits_grid, row)
        else:
            _apply_temperature_row(logits, sampling_temperatures, processed_logits, row)
            _apply_top_k_row(processed_logits, sampling_temperatures, sampling_top_ks, row)
            random_flag = pl.cast(1, pl.INT32)
            seed = pl.read(sampling_seeds, [row])
            position = pl.read(sampling_positions, [row])
            seed_index = pl.cast(seed, pl.INDEX)
            position_index = pl.cast(position, pl.INDEX)
            random_key_index = seed_index + position_index * POSITION_MULTIPLIER
            random_key = pl.cast(random_key_index % RANDOM_KEY_MODULUS, pl.INT32)
            best_index = _sample_filtered_logits(processed_logits, row, random_flag, random_key)
        sampled_row = pl.create_tensor([1, SAMPLED_IDS_PAD], dtype=pl.INT32)
        sampled_row[:, :] = pl.full([1, SAMPLED_IDS_PAD], dtype=pl.INT32, value=0)
        pl.write(sampled_row, [0, 0], best_index)
        sampled_ids[row : row + 1, :] = sampled_row
    return sampled_ids


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


# ci: devices=2



if __package__:
    from .config import FLASH as M
else:
    from config import FLASH as M
if __package__:
    from .decode_shapes import DECODE_TOKENS, TP as CONFIG_TP
else:
    from decode_shapes import DECODE_TOKENS, TP as CONFIG_TP


T_DYN = pl.dynamic('LOWLAT_TARGET_LM_HEAD_T_DYN')

# model config
D = M.hidden_size

# Parallelism, parsed off argv: the frontend needs both worlds static.
_TP_CHOICES = (2, 4, 8, 16)  # AIV_LANES = 2 owners per lane, so TP = 1 has no valid split
_DP_CHOICES = (1, 2, 4, 8, 16)
_TP_DEFAULT = CONFIG_TP


def _parse_int_argv(name, default=None):
    for i, tok in enumerate(sys.argv):
        if tok == name and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if tok.startswith(f"{name}="):
            return int(tok.split("=", 1)[1])
    return default


TP_SIZE: int = _parse_int_argv("--tp") or _TP_DEFAULT
# DP groups built by the l3_lm_head fixture; the kernel carries no DP extent.
DP_SIZE: int = _parse_int_argv("--dp") or 1
WORLD_SIZE = TP_SIZE * DP_SIZE
VOCAB_PER_TP = VOCAB // TP_SIZE
# AIV lanes per AICore; owners shard across them to keep every push and notify single.
AIV_LANES = 2
OWNER_PAIRS = TP_SIZE // AIV_LANES
OWNERS_PER_LANE = TP_SIZE // AIV_LANES
FUSED_LM_HEAD_CORES = 24
DONE_VALUE = 1

# Rows. Decode specializations override DECODE_TOKENS before importing this
# module. logit_row_indices picks the sources; unused rows stay zero.
MAX_LOGIT_ROWS = DECODE_TOKENS
GROUP_LOGIT_ROWS = TP_SIZE * MAX_LOGIT_ROWS
TEST_TOKENS = 16  # standalone fixture: hidden rows per card, > MAX_LOGIT_ROWS

# tiling -- both matmul tiles clear the 512 B contiguous-transfer floor
FUSED_K_TILE = 256
FUSED_VOCAB_TILE = 256
HIDDEN_GATHER_TILE = 512
LOGITS_COMM_TILE = 2048

# Derived layout. The vocab shard rarely divides the matmul tile, so the last tile
# is ragged: a plain ceil tile count, the weight load carries a valid_shape, and
# logits_shards pads to whole tiles so its store stays in bounds.
VOCAB_LAST_TILE = VOCAB_PER_TP % FUSED_VOCAB_TILE
VOCAB_TILES = VOCAB_PER_TP // FUSED_VOCAB_TILE + (1 if VOCAB_LAST_TILE != 0 else 0)
SHARDS_VOCAB = VOCAB_TILES * FUSED_VOCAB_TILE
# The shard lands whole in a GM scratch; per-owner pushes slice that scratch.
COMBINE_ROW_TILE = (MAX_LOGIT_ROWS + 7) // 8 * 8
MATMUL_GROUP_ROWS = (GROUP_LOGIT_ROWS + 15) // 16 * 16
SHARD_ROWS = MATMUL_GROUP_ROWS // AIV_LANES
# Combine blocks: one per vocab comm tile, capped at the core count; the tail tile
# rides the block the strided loop hands it next.
LOGITS_COMM_TAIL = VOCAB_PER_TP % LOGITS_COMM_TILE
N_LOGITS_COMM_TILES = VOCAB_PER_TP // LOGITS_COMM_TILE
LOGITS_COMM_BLOCKS = min(FUSED_LM_HEAD_CORES, N_LOGITS_COMM_TILES + (1 if LOGITS_COMM_TAIL != 0 else 0))
LOGITS_TAIL_BLOCK = N_LOGITS_COMM_TILES % LOGITS_COMM_BLOCKS

# Keeps the GM staging race-free: each UP_DOWN lane writes exactly the row band it
# later reads, and the two lanes' bands are disjoint.
# Cube rows include physical padding; consumer masks keep logical rows.
assert SHARD_ROWS * AIV_LANES == MATMUL_GROUP_ROWS
assert MATMUL_GROUP_ROWS >= GROUP_LOGIT_ROWS
assert MATMUL_GROUP_ROWS % 16 == 0, "physical matmul M extent must be a multiple of 16"
assert TP_SIZE in _TP_CHOICES, f"--tp must be one of {_TP_CHOICES} (got {TP_SIZE})"
assert DP_SIZE in _DP_CHOICES, f"--dp must be one of {_DP_CHOICES} (got {DP_SIZE})"


@pl.jit.inline(auto_scope=False)
def _lm_head_with_greedy_partials(
    hidden_states: pl.Tensor,
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
    greedy_scores: pl.Tensor[[LOGITS_COMM_BLOCKS * MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.FP32],
    greedy_indices: pl.Tensor[[LOGITS_COMM_BLOCKS * MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
):
    # Scratch is allocated just outside the scope that first writes it: a
    # create_tensor inside a pl.at yields a tile, not a GM tensor view.
    selected_hidden = pl.create_tensor([MAX_LOGIT_ROWS, D], dtype=pl.BF16)
    owner_hiddens = pl.create_tensor([MATMUL_GROUP_ROWS, D], dtype=pl.BF16)

    # Publish this card's logit rows into every group member's window slot: the
    # window holds one slot per group member and each card writes only its own,
    # `tp_rank * MAX_LOGIT_ROWS`. One block per logit row, one [1, D] put per peer.
    for row in pl.spmd(MAX_LOGIT_ROWS, name_hint="lm_head_dispatch_push", allow_early_resolve=True):
        hidden_rows = pl.tensor.dim(hidden_states, 0)
        source_row_raw = pl.read(logit_row_indices, [row])
        # Clamp so the load address is always inside hidden_states even if a
        # caller hands over a stale index; the -1 guard below decides whether
        # the row is actually used.
        safe_raw = pl.max(pl.min(source_row_raw, hidden_rows - 1), 0)
        # Full-width [1, D] tile: the block owns the whole row.
        selected_hidden[row : row + 1, :] = pl.full([1, D], dtype=pl.BF16, value=0.0)
        if source_row_raw >= 0:
            source_row = pl.cast(safe_raw, target_type=pl.INDEX)
            selected_hidden[row : row + 1, :] = hidden_states[source_row : source_row + 1, :]

        # Self-target rides the same put; put drains before the notify issues.
        for peer_tp in pl.range(TP_SIZE):
            pld.tensor.put(
                dst=hidden_window,
                peer=group_base + peer_tp,
                src=selected_hidden,
                dst_offsets=[tp_rank * MAX_LOGIT_ROWS + row, 0],
                src_offsets=[row, 0],
                shape=[1, D],
            )

        # Notify folded into the push: MAX_LOGIT_ROWS notifies per source per epoch.
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=hidden_done,
                    peer=group_base + peer_tp,
                    offsets=[tp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

    # Anchor the blocking wait to the local hidden-state producer.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="lm_head_dispatch_wait", allow_early_resolve=True
    ) as _dwait_tid:
        _hidden_anchor = pl.read(hidden_states, [0, 0])
        for owner_tp in pl.range(TP_SIZE):
            if owner_tp != tp_rank:
                pld.system.wait(
                    signal=hidden_done,
                    offsets=[owner_tp, 0],
                    expected=pl.cast(done_epoch * MAX_LOGIT_ROWS, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    # Window -> matmul operand: a local copy split over k-tiles. Keeps the matmul's
    # auto-dep on owner_hiddens.
    with pl.spmd(
        D // HIDDEN_GATHER_TILE,
        name_hint="lm_head_dispatch_gather",
        allow_early_resolve=True,
        deps=[_dwait_tid],
    ) as _dgather_tid:
        gkb = pl.tile.get_block_idx()
        gk0 = gkb * HIDDEN_GATHER_TILE
        if MATMUL_GROUP_ROWS == GROUP_LOGIT_ROWS:
            owner_hiddens[0:GROUP_LOGIT_ROWS, gk0 : gk0 + HIDDEN_GATHER_TILE] = hidden_window[
                :, gk0 : gk0 + HIDDEN_GATHER_TILE
            ]
        else:
            owner_hiddens[:, gk0 : gk0 + HIDDEN_GATHER_TILE] = pl.full(
                [MATMUL_GROUP_ROWS, HIDDEN_GATHER_TILE], dtype=pl.BF16, value=0.0
            )
            for owner in pl.range(TP_SIZE):
                src_row = owner * MAX_LOGIT_ROWS
                dst_lane = owner // OWNERS_PER_LANE
                dst_slot = owner % OWNERS_PER_LANE
                dst_row = dst_lane * SHARD_ROWS + dst_slot * MAX_LOGIT_ROWS
                owner_tile = hidden_window[src_row : src_row + MAX_LOGIT_ROWS, gk0 : gk0 + HIDDEN_GATHER_TILE]
                owner_hiddens[dst_row : dst_row + MAX_LOGIT_ROWS, gk0 : gk0 + HIDDEN_GATHER_TILE] = owner_tile

    # Mixed cube + comm kernel: the cube projects a vocab tile, pl.aiv_shard carries
    # that accumulator across the C->V edge, and pld.tensor.remote_store pushes each
    # owner's rows to that peer's window, so a tile goes on the wire while the cube
    # projects the next. The ragged last tile rides the same loop as a padded tile.
    logits_shards = pl.create_tensor([MATMUL_GROUP_ROWS, SHARDS_VOCAB], dtype=pl.FP32)
    with pl.spmd(
        FUSED_LM_HEAD_CORES,
        name_hint="lm_head_matmul_push",
        allow_early_resolve=True,
        optimizations=[pl.cross_core_slot(slot_num=2)],
    ) as _push_tid:
        lm_core = pl.tile.get_block_idx()
        vocab_base = tp_rank * VOCAB_PER_TP
        for mm_ob in pl.range(lm_core, VOCAB_TILES, FUSED_LM_HEAD_CORES):
            mm_o0 = mm_ob * FUSED_VOCAB_TILE
            # Real width of this tile: the full tile everywhere except the ragged
            # last one, where the weight rows run out early. The box stays static
            # so the accumulator keeps one shape for every iteration.
            mm_valid_n = pl.min(VOCAB_PER_TP - mm_o0, FUSED_VOCAB_TILE)
            # M is the full group extent -- one matmul per vocab tile.
            mm_acc = pl.create_tensor([MATMUL_GROUP_ROWS, FUSED_VOCAB_TILE], dtype=pl.FP32)
            for mm_kb in pl.pipeline(0, D // FUSED_K_TILE, stage=2):
                mm_k0 = mm_kb * FUSED_K_TILE
                mm_hidden_tile = owner_hiddens[:, mm_k0 : mm_k0 + FUSED_K_TILE]
                mm_weight_tile = pl.slice(
                    lm_head_weight,
                    [FUSED_VOCAB_TILE, FUSED_K_TILE],
                    [mm_o0, mm_k0],
                    valid_shape=[mm_valid_n, FUSED_K_TILE],
                )
                mm_acc = pl.matmul_acc(
                    mm_acc, mm_hidden_tile, mm_weight_tile, b_trans=True, init_cond=(mm_kb == 0)
                )

            # Columns must be fully valid across the C->V crossing: the boundary
            # transports the full box and rejects a runtime-valued column extent.
            # The ragged tile's padding is cut off by the push below.
            mm_acc = pl.set_validshape(mm_acc, MATMUL_GROUP_ROWS, FUSED_VOCAB_TILE)

            # WORKAROUND -- revert once a subview of a pl.aiv_shard result is legal
            # (filed against ptoas). The shard cannot be sliced, so it is stored whole
            # to GM and the per-owner pushes read slices back out.
            # Race-free by construction: UP_DOWN gives each lane a disjoint row band
            # that it both writes and reads, guarded by the SHARD_ROWS assert above.
            for aiv_id in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                mm_shard = pl.aiv_shard(mm_acc)
                lane_r0 = aiv_id * SHARD_ROWS
                logits_shards[lane_r0 : lane_r0 + SHARD_ROWS, mm_o0 : mm_o0 + FUSED_VOCAB_TILE] = mm_shard
                for owner_slot in pl.range(OWNERS_PER_LANE):
                    owner_tp = aiv_id * OWNERS_PER_LANE + owner_slot
                    src_r0 = aiv_id * SHARD_ROWS + owner_slot * MAX_LOGIT_ROWS
                    # valid_shape rather than a narrower box: the box has to stay
                    # static for the tile type, and a full-width push of the ragged
                    # tile would run past this rank's slice of the shared window
                    # into the next rank's. pl.min is inlined instead of reusing
                    # mm_valid_n because a named cube-side scalar cannot be
                    # materialized inside the AIV function.
                    pld.tensor.remote_store(
                        pl.slice(
                            logits_shards,
                            [MAX_LOGIT_ROWS, FUSED_VOCAB_TILE],
                            [src_r0, mm_o0],
                            valid_shape=[
                                MAX_LOGIT_ROWS,
                                pl.min(VOCAB_PER_TP - mm_o0, FUSED_VOCAB_TILE),
                            ],
                        ),
                        logits_window,
                        group_base + owner_tp,
                        [0, vocab_base + mm_o0],
                    )

        # Notify folded into the push: each block signals every peer after its own
        # stores, so a peer sees FUSED_LM_HEAD_CORES notifies per source per epoch.
        for aiv_id in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.NONE):
            for notify_pair in pl.range(OWNER_PAIRS):
                owner_tp = notify_pair * AIV_LANES + aiv_id
                if owner_tp != tp_rank:
                    pld.system.notify(
                        target=logits_done,
                        peer=group_base + owner_tp,
                        offsets=[tp_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )

    # Keep the blocking wait behind the local logits push.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="lm_head_combine_wait",
        allow_early_resolve=True,
        deps=[_push_tid],
    ) as _cwait_tid:
        for src_tp in pl.range(TP_SIZE):
            if src_tp != tp_rank:
                pld.system.wait(
                    signal=logits_done,
                    offsets=[src_tp, 0],
                    expected=pl.cast(done_epoch * FUSED_LM_HEAD_CORES, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    # Assemble full-vocabulary logits, same vocab-tile split. deps on _cwait_tid for
    # the peers' stores; our own tiles ride the local RAW edge on logits_window.
    with pl.spmd(
        LOGITS_COMM_BLOCKS, name_hint="lm_head_combine_gather", allow_early_resolve=True, deps=[_cwait_tid]
    ) as _gather_tid:
        gblk = pl.tile.get_block_idx()
        running_scores = pl.tile.full(
            [COMBINE_ROW_TILE, SAMPLED_IDS_PAD], dtype=pl.FP32, value=-3.4028234663852886e38
        )
        running_indices = pl.tile.full([COMBINE_ROW_TILE, SAMPLED_IDS_PAD], dtype=pl.INT32, value=1073741824)
        for src_tp in pl.range(TP_SIZE):
            src_vocab_base = src_tp * VOCAB_PER_TP
            for ob in pl.range(gblk, N_LOGITS_COMM_TILES, LOGITS_COMM_BLOCKS):
                o0 = ob * LOGITS_COMM_TILE
                lo = src_vocab_base + o0
                scores = pl.load(
                    logits_window,
                    [0, lo],
                    [COMBINE_ROW_TILE, LOGITS_COMM_TILE],
                    valid_shape=[MAX_LOGIT_ROWS, LOGITS_COMM_TILE],
                )
                pl.store(scores, [0, lo], logits)
                scores = pl.fillpad(scores, pad_value=pl.PadValue.zero)
                scores = pl.set_validshape(scores, COMBINE_ROW_TILE, LOGITS_COMM_TILE)
                reduce_tmp = pl.create_tile(
                    [COMBINE_ROW_TILE, LOGITS_COMM_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                tile_maxima = pl.row_max(scores, reduce_tmp)
                tile_indices = pl.row_argmax(scores, reduce_tmp)
                score_zeros = pl.tile.full([COMBINE_ROW_TILE, SAMPLED_IDS_PAD], dtype=pl.FP32, value=0.0)
                index_zeros = pl.tile.full([COMBINE_ROW_TILE, SAMPLED_IDS_PAD], dtype=pl.INT32, value=0)
                candidate_scores = pl.row_expand_add(score_zeros, tile_maxima)
                candidate_indices = pl.add(
                    pl.row_expand_add(index_zeros, tile_indices), pl.cast(lo, pl.INT32)
                )
                newer = pl.cmp(candidate_scores, running_scores, cmp_type=4)
                select_tmp = pl.create_tile(
                    [COMBINE_ROW_TILE, SAMPLED_IDS_PAD], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec
                )
                running_indices = pl.sel(newer, candidate_indices, running_indices, select_tmp)
                running_scores = pl.maximum(running_scores, candidate_scores)

            if LOGITS_COMM_TAIL != 0:
                if gblk == LOGITS_TAIL_BLOCK:
                    tail_o0 = N_LOGITS_COMM_TILES * LOGITS_COMM_TILE
                    tl = src_vocab_base + tail_o0
                    tail_scores = pl.load(
                        logits_window,
                        [0, tl],
                        [COMBINE_ROW_TILE, LOGITS_COMM_TAIL],
                        valid_shape=[MAX_LOGIT_ROWS, LOGITS_COMM_TAIL],
                    )
                    pl.store(tail_scores, [0, tl], logits)
                    tail_scores = pl.fillpad(tail_scores, pad_value=pl.PadValue.zero)
                    tail_scores = pl.set_validshape(tail_scores, COMBINE_ROW_TILE, LOGITS_COMM_TAIL)
                    tail_reduce_tmp = pl.create_tile(
                        [COMBINE_ROW_TILE, LOGITS_COMM_TAIL], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                    )
                    tail_tile_maxima = pl.row_max(tail_scores, tail_reduce_tmp)
                    tail_tile_indices = pl.row_argmax(tail_scores, tail_reduce_tmp)
                    tail_score_zeros = pl.tile.full(
                        [COMBINE_ROW_TILE, SAMPLED_IDS_PAD], dtype=pl.FP32, value=0.0
                    )
                    tail_index_zeros = pl.tile.full(
                        [COMBINE_ROW_TILE, SAMPLED_IDS_PAD], dtype=pl.INT32, value=0
                    )
                    tail_candidate_scores = pl.row_expand_add(tail_score_zeros, tail_tile_maxima)
                    tail_candidate_indices = pl.add(
                        pl.row_expand_add(tail_index_zeros, tail_tile_indices), pl.cast(tl, pl.INT32)
                    )
                    tail_newer = pl.cmp(tail_candidate_scores, running_scores, cmp_type=4)
                    tail_select_tmp = pl.create_tile(
                        [COMBINE_ROW_TILE, SAMPLED_IDS_PAD], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec
                    )
                    running_indices = pl.sel(
                        tail_newer, tail_candidate_indices, running_indices, tail_select_tmp
                    )
                    running_scores = pl.maximum(running_scores, tail_candidate_scores)

        partial_row = gblk * MAX_LOGIT_ROWS
        running_scores = pl.set_validshape(running_scores, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD)
        pl.store(running_scores, [partial_row, 0], greedy_scores)
        running_indices = pl.set_validshape(running_indices, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD)
        pl.store(running_indices, [partial_row, 0], greedy_indices)

    # Every local wait has observed all current-round peer notifies before the
    # logits gather can complete. Clear only this rank's counters so a retained
    # CommDomain can safely reuse the fixed done_epoch on the next forward.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_signal_clear") as s6_lm_cleanup_tid:
        _completion_anchor = pl.read(logits, [0, 0])
        zero = pl.cast(0, pl.INT32)
        for src_tp in pl.range(TP_SIZE):
            pl.write(hidden_done, [src_tp, 0], zero)
            pl.write(logits_done, [src_tp, 0], zero)
    return logits, s6_lm_cleanup_tid



@pl.jit.inline(auto_scope=False)
def lm_head(
    hidden_states: pl.Tensor,
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]:
    # Scratch is allocated just outside the scope that first writes it: a
    # create_tensor inside a pl.at yields a tile, not a GM tensor view.
    greedy_scores = pl.create_tensor([LOGITS_COMM_BLOCKS * MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], dtype=pl.FP32)
    greedy_indices = pl.create_tensor([LOGITS_COMM_BLOCKS * MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], dtype=pl.INT32)
    _lm_head_with_greedy_partials(
        hidden_states, lm_head_weight, logit_row_indices, logits,
        hidden_window, hidden_done, logits_window, logits_done,
        group_base, tp_rank, done_epoch, greedy_scores, greedy_indices,
    )
    return logits


@pl.jit
def lm_head_test(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]:
    lm_head(
        hidden_states, lm_head_weight, logit_row_indices, logits,
        hidden_window, hidden_done, logits_window, logits_done,
        group_base, tp_rank, done_epoch,
    )
    return logits


@pl.jit.inline(auto_scope=False)
def lm_head_with_sampling(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_temperatures: pl.Tensor[[MAX_LOGIT_ROWS], pl.FP32],
    sampling_top_ks: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_seeds: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_positions: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
):
    """Project logits and sample tokens in one opaque L2 entry."""
    greedy_scores = pl.create_tensor([LOGITS_COMM_BLOCKS * MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], dtype=pl.FP32)
    greedy_indices = pl.create_tensor([LOGITS_COMM_BLOCKS * MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], dtype=pl.INT32)
    _s6_logits, s6_lm_cleanup_tid = _lm_head_with_greedy_partials(
        hidden_states,
        lm_head_weight,
        logit_row_indices,
        logits,
        hidden_window,
        hidden_done,
        logits_window,
        logits_done,
        group_base,
        tp_rank,
        done_epoch,
        greedy_scores,
        greedy_indices,
    )
    sampled_ids = sample_from_greedy_partials(
        logits,
        sampling_temperatures,
        sampling_top_ks,
        sampling_seeds,
        sampling_positions,
        sampled_ids,
        greedy_scores,
        greedy_indices,
    )
    return logits, sampled_ids, s6_lm_cleanup_tid


@pl.jit
def lm_head_with_sampling_test(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_temperatures: pl.Tensor[[MAX_LOGIT_ROWS], pl.FP32],
    sampling_top_ks: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_seeds: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_positions: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
):
    """Standalone opaque entry for projection plus sampling tests."""
    return lm_head_with_sampling(
        hidden_states,
        lm_head_weight,
        logit_row_indices,
        sampling_temperatures,
        sampling_top_ks,
        sampling_seeds,
        sampling_positions,
        logits,
        sampled_ids,
        hidden_window,
        hidden_done,
        logits_window,
        logits_done,
        group_base,
        tp_rank,
        done_epoch,
    )


@pl.jit.host
def l3_lm_head(
    hidden_states: pl.Tensor[[WORLD_SIZE, TEST_TOKENS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[WORLD_SIZE, VOCAB_PER_TP, D], pl.BF16],
    sampling_temperatures: pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS], pl.FP32],
    sampling_top_ks: pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS], pl.INT32],
    sampling_seeds: pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS], pl.INT32],
    sampling_positions: pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Out[pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[
        pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]
    ],
    logit_row_indices: pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS], pl.INT32],
):
    # Windows are group-local: hidden_window holds one row slot per group member,
    # and every card receives only its own full-vocabulary logits.
    hidden_window_buf = pld.alloc_window_buffer(GROUP_LOGIT_ROWS * D * 2)
    logits_window_buf = pld.alloc_window_buffer(MAX_LOGIT_ROWS * VOCAB * 4)
    hidden_done_buf = pld.alloc_window_buffer(TP_SIZE * 4)
    logits_done_buf = pld.alloc_window_buffer(TP_SIZE * 4)

    for r in pl.range(pld.world_size()):
        hidden_window = pld.window(hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        hidden_done = pld.window(hidden_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        logits_window = pld.window(logits_window_buf, [MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
        logits_done = pld.window(logits_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        lm_head_with_sampling_test(
            hidden_states[r], lm_head_weight[r], logit_row_indices[r],
            sampling_temperatures[r], sampling_top_ks[r],
            sampling_seeds[r], sampling_positions[r],
            logits[r], sampled_ids[r],
            hidden_window, hidden_done, logits_window, logits_done,
            r // TP_SIZE * TP_SIZE, r % TP_SIZE, DONE_VALUE, device=r,
        )


def build_tensor_specs_sampling(temperature=None, top_k=None):
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
        TensorSpec("sampled_ids", [SAMPLE_ROWS, SAMPLED_IDS_PAD], torch.int32),
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


def golden_lm_head(tensors):
    import torch

    if __package__:
        from .utils import bf16_matmul_fp32
    else:
        from utils import bf16_matmul_fp32

    hidden = tensors["hidden_states"].float()
    # Card r holds shard r % TP_SIZE, so concatenating shards in index order
    # reproduces the global vocabulary order every owner assembles.
    weight = tensors["lm_head_weight"].float()
    full_weight = torch.cat([weight[tp] for tp in range(TP_SIZE)], dim=0)
    world_size = hidden.shape[0]
    # Project at most two owners per reference batch.
    for owner_start in range(0, world_size, 2):
        owner_stop = min(owner_start + 2, world_size)
        selected = torch.zeros((owner_stop - owner_start, MAX_LOGIT_ROWS, D), dtype=torch.float32)
        for owner_rank in range(owner_start, owner_stop):
            for row in range(MAX_LOGIT_ROWS):
                source_row = int(tensors["logit_row_indices"][owner_rank, row])
                if source_row >= 0:
                    source_row = min(source_row, hidden.shape[1] - 1)
                    selected[owner_rank - owner_start, row].copy_(hidden[owner_rank, source_row])
        tensors["logits"][owner_start:owner_stop] = bf16_matmul_fp32(selected, full_weight.t())
    if "sampled_ids" in tensors:
        for rank in range(world_size):
            golden_sample(
                {
                    "logits": tensors["logits"][rank],
                    "temperatures": tensors["sampling_temperatures"][rank],
                    "top_ks": tensors["sampling_top_ks"][rank],
                    "seeds": tensors["sampling_seeds"][rank],
                    "positions": tensors["sampling_positions"][rank],
                    "sampled_ids": tensors["sampled_ids"][rank],
                }
            )


def build_tensor_specs_lm(num_tokens=TEST_TOKENS):
    import torch
    from golden import TensorSpec

    active = max(min(num_tokens, MAX_LOGIT_ROWS), 0)

    def init_hidden_states():
        return (torch.randn(WORLD_SIZE, TEST_TOKENS, D) * 0.1).to(torch.bfloat16)

    def init_lm_head_weight():
        shards = (torch.randn(TP_SIZE, VOCAB_PER_TP, D) / D ** 0.5).to(torch.bfloat16)
        return torch.stack([shards[r % TP_SIZE] for r in range(WORLD_SIZE)], dim=0)

    def init_logit_row_indices():
        indices = torch.full((WORLD_SIZE, MAX_LOGIT_ROWS), -1, dtype=torch.int32)
        indices[:, :active] = torch.arange(active, dtype=torch.int32)
        return indices

    def init_sampling_temperatures():
        temperatures = torch.zeros((WORLD_SIZE, MAX_LOGIT_ROWS), dtype=torch.float32)
        temperatures[:, :active:2] = 0.7
        return temperatures

    def init_sampling_top_ks():
        top_ks = torch.full((WORLD_SIZE, MAX_LOGIT_ROWS), VOCAB, dtype=torch.int32)
        top_ks[:, :active:2] = 16
        return top_ks

    return [
        TensorSpec(
            "hidden_states",
            [WORLD_SIZE, TEST_TOKENS, D],
            torch.bfloat16,
            init_value=init_hidden_states,
        ),
        # One vocab shard per DP rank: card r carries a copy of shard
        # r % TP_SIZE, matching how resident args are handed out per rank. Keep
        # each rank-local shard on its consuming card across dispatches.
        TensorSpec(
            "lm_head_weight",
            [WORLD_SIZE, VOCAB_PER_TP, D],
            torch.bfloat16,
            init_value=init_lm_head_weight,
            resident="stacked",
        ),
        TensorSpec(
            "sampling_temperatures",
            [WORLD_SIZE, MAX_LOGIT_ROWS],
            torch.float32,
            init_value=init_sampling_temperatures,
        ),
        TensorSpec(
            "sampling_top_ks",
            [WORLD_SIZE, MAX_LOGIT_ROWS],
            torch.int32,
            init_value=init_sampling_top_ks,
        ),
        TensorSpec(
            "sampling_seeds",
            [WORLD_SIZE, MAX_LOGIT_ROWS],
            torch.int32,
            init_value=lambda: torch.zeros(WORLD_SIZE, MAX_LOGIT_ROWS, dtype=torch.int32),
        ),
        TensorSpec(
            "sampling_positions",
            [WORLD_SIZE, MAX_LOGIT_ROWS],
            torch.int32,
            init_value=lambda: torch.zeros(WORLD_SIZE, MAX_LOGIT_ROWS, dtype=torch.int32),
        ),
        TensorSpec(
            "logits",
            [WORLD_SIZE, MAX_LOGIT_ROWS, VOCAB],
            torch.float32,
        ),
        TensorSpec(
            "sampled_ids",
            [WORLD_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD],
            torch.int32,
        ),
        TensorSpec(
            "logit_row_indices",
            [WORLD_SIZE, MAX_LOGIT_ROWS],
            torch.int32,
            init_value=init_logit_row_indices,
        ),
    ]


def compare_logits(actual, expected, **_):
    import torch

    close = torch.isclose(actual, expected, rtol=1e-3, atol=1e-3)
    if bool(close.all()):
        return True, ""
    lines = []
    for owner in range(actual.shape[0]):
        for shard in range(TP_SIZE):
            start = shard * VOCAB_PER_TP
            end = start + VOCAB_PER_TP
            shard_actual = actual[owner, :, start:end]
            shard_close = close[owner, :, start:end]
            lines.append(
                f"    owner={owner} shard={shard}: "
                f"bad={int((~shard_close).sum())}/{MAX_LOGIT_ROWS * VOCAB_PER_TP} "
                f"zeros={int((shard_actual == 0).sum())}"
            )
    return False, "\n".join(lines)


def compare_sampled_ids(actual, _expected, *, actual_outputs, inputs=None, **_):
    import torch

    expected = torch.zeros_like(actual)
    if inputs is None or bool(torch.all(inputs["sampling_temperatures"] < SAMPLING_EPS)):
        expected[:, :, 0] = torch.argmax(actual_outputs["logits"].cpu(), dim=-1).to(torch.int32)
    else:
        for rank in range(actual.shape[0]):
            golden_sample({
                "logits": actual_outputs["logits"][rank].cpu(),
                "temperatures": inputs["sampling_temperatures"][rank],
                "top_ks": inputs["sampling_top_ks"][rank],
                "seeds": inputs["sampling_seeds"][rank],
                "positions": inputs["sampling_positions"][rank],
                "sampled_ids": expected[rank],
            })
    if torch.equal(actual, expected):
        return True, ""
    mismatch = actual != expected
    return False, (
        f"sampled_ids mismatch: bad={int(mismatch.sum())}/{actual.numel()} "
        f"actual={actual.tolist()} expected={expected.tolist()}"
    )


def validate_sampling():
    import argparse
    from math import isfinite

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3sim", choices=["a2a3", "a2a3sim", "a5", "a5sim"]
    )
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    args = parser.parse_args()

    if args.temperature is not None and (not isfinite(args.temperature) or args.temperature < 0.0):
        parser.error(f"--temperature must be finite and non-negative, got {args.temperature}")
    result = run(
        fn=sample_test,
        specs=build_tensor_specs_sampling(args.temperature, args.top_k),
        golden_fn=golden_sample,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        config=dict(
            backend_type=args.platform.removesuffix("sim"),
            execution_mode=args.sim or args.platform.endswith("sim"),
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



def validate_lm():
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=list(_TP_CHOICES),
                        help="LM-head tensor-parallel world size")
    parser.add_argument("--dp", type=int, default=DP_SIZE, choices=list(_DP_CHOICES),
                        help="DP groups (world size = tp * dp)")
    parser.add_argument("--num-tokens", type=int, default=TEST_TOKENS,
                        help="Active hidden rows each owner projects")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(WORLD_SIZE)),
                        help=f"comma-separated device ids; need at least {WORLD_SIZE}")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0,
                        choices=(0, 1, 2, 4))
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    required_devices = WORLD_SIZE
    assert len(device_ids) >= required_devices, (
        f"need at least {required_devices} devices, got {device_ids}"
    )
    assert args.tp == TP_SIZE and args.dp == DP_SIZE
    assert 1 <= args.num_tokens <= TEST_TOKENS

    fn = l3_lm_head
    specs = build_tensor_specs_lm(args.num_tokens)
    golden_fn = golden_lm_head
    compare_fn = {
        "logits": compare_logits,
        "sampled_ids": compare_sampled_ids,
    }

    result = run(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn,
        compare_fn=compare_fn,
        save_data=args.save_data,
        golden_data=args.golden_data,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        config=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=device_ids[:required_devices], num_sub_workers=0),
            backend_type=args.platform.removesuffix("sim"),
            execution_mode=args.sim or args.platform.endswith("sim"),
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    from models.deepseek_v4_flash_lowlat.utils import run_operator_validations

    run_operator_validations(
        {
            'sampling': validate_sampling,
            'lm': validate_lm,
        },
        single_device=('sampling',),
        automatic=('lm', 'sampling'),
        standalone_options={
            "lm": {"--tp", "--dp", "--num-tokens", "--runtime-dir", "--dump-passes", "--enable-chip-swimlane"},
            "sampling": {"--temperature", "--top-k", "--enable-l2-swimlane"},
        },
    )
