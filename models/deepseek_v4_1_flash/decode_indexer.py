# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode indexer query preparation, key publication, scoring, and TopK."""

from pathlib import Path

from pypto.runtime import pto_isa_include_dir

from gemm import gemm_bf16_to_bf16
import pypto.language as pl
from config import (
    B_DYN, FLASH, INDEX_BLOCKS_DYN, INDEX_CACHE_GROUP, INDEX_POOL_BLOCKS_DYN,
    INDEX_POOL_POSITIONS_DYN, TABLE_DYN, T_DYN,
)


# model config
D = FLASH.hidden_size
HEAD_DIM = FLASH.head_dim
INDEX_H = FLASH.index_n_heads
INDEX_DIM = FLASH.index_head_dim
INDEX_TOPK = FLASH.index_topk
Q_LORA = FLASH.q_lora_rank
ROPE_DIM = FLASH.qk_rope_head_dim
CANDIDATE_BLOCK_SIZE = FLASH.candidate_block_size
CANDIDATE_TOPK_BLOCKS = FLASH.candidate_topk_blocks
CANDIDATE_PAIR_WIDTH = 2 * CANDIDATE_TOPK_BLOCKS

_SCALE_COUNT = INDEX_DIM // INDEX_CACHE_GROUP

# tiling
INDEX_PROJ_N_TILE = 128
INDEX_WEIGHT_K_SPLIT_TILE = 4
INDEX_WEIGHT_K_TILE = 256
INDEX_PROJ_K_TILE = 256


CANDIDATE_WORKER_TILE = 12
CANDIDATE_SCORE_TILE = 1024
CANDIDATE_LEAF_TILE = 8192
CANDIDATE_SHORT_LEAF_TILE = 2048
CANDIDATE_MAX_LEAVES = (FLASH.max_position_embeddings // CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_LEAF_TILE + 1
INDEX_SCORE_POSITIONS_DYN = pl.dynamic("V41_INDEX_SCORE_POSITIONS_DYN")

@pl.jit.extern(core_type="mixed", aic_source=Path(__file__).parent / "kernels" / "idx_q.cpp",
               aiv_source=Path(__file__).parent / "kernels" / "idx_q.cpp", dual_aiv_dispatch=True,
               include_dirs=(pto_isa_include_dir(),))
def _idx_q(
    payload: pl.Tensor,
    weight: pl.Tensor,
    activation_scale: pl.Tensor,
    weight_scale: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Out[pl.Tensor],
    output_scale: pl.Out[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    """Project and quantize query heads with two C-to-V transfers per cube worker."""
    ...


@pl.jit.extern(core_type="mixed", aic_source=Path(__file__).parent / "kernels" / "index_score_topk.cpp",
               aiv_source=Path(__file__).parent / "kernels" / "index_score_topk.cpp", dual_aiv_dispatch=True,
               include_dirs=(pto_isa_include_dir(),))
def _idx_score_topk(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    index_cache: pl.Tensor,
    index_cache_scale: pl.Tensor,
    block_table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    output: pl.Out[pl.Tensor],
    runs: pl.Tensor,
    candidate_keys: pl.Out[pl.Tensor],
    candidate_indices: pl.Out[pl.Tensor],
    valid: pl.Tensor,
    physical: pl.Tensor,
    scores: pl.Out[pl.Tensor],
    width: pl.Scalar[pl.INDEX],
    run_stride: pl.Scalar[pl.INDEX],
    table_width: pl.Scalar[pl.INDEX],
    num_tokens: pl.Scalar[pl.INT32],
    gathered_rows: pl.Scalar[pl.INDEX],
    emit_scores: pl.Scalar[pl.INT32],
    shards: pl.Scalar[pl.INDEX],
    direct_short: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    """Score history shards and merge local TopK candidates within one MIX."""
    ...


@pl.jit.extern(core_type="mixed", aic_source=Path(__file__).parent / "kernels" / "index_score_topk.cpp",
               aiv_source=Path(__file__).parent / "kernels" / "index_score_topk.cpp", dual_aiv_dispatch=True,
               include_dirs=(pto_isa_include_dir(),))
def _idx_pool_score_topk(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    index_cache: pl.Tensor,
    index_cache_scale: pl.Tensor,
    block_table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    output: pl.Out[pl.Tensor],
    runs: pl.Tensor,
    candidate_keys: pl.Out[pl.Tensor],
    candidate_indices: pl.Out[pl.Tensor],
    valid: pl.Tensor,
    physical: pl.Tensor,
    scores: pl.Out[pl.Tensor],
    width: pl.Scalar[pl.INDEX],
    run_stride: pl.Scalar[pl.INDEX],
    table_width: pl.Scalar[pl.INDEX],
    num_tokens: pl.Scalar[pl.INT32],
    gathered_rows: pl.Scalar[pl.INDEX],
    emit_scores: pl.Scalar[pl.INT32],
    shards: pl.Scalar[pl.INDEX],
    direct_short: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    """Score gathered pool shards and merge local TopK candidates within one MIX."""
    ...


@pl.jit.inline
def _merge_candidate_pairs(
    arena: pl.Tensor,
    left: pl.Scalar[pl.INDEX],
    right: pl.Scalar[pl.INDEX],
    output: pl.Scalar[pl.INDEX],
):
    left_pairs = pl.load(arena, [left, 0], [1, CANDIDATE_PAIR_WIDTH])
    right_pairs = pl.load(arena, [right, 0], [1, CANDIDATE_PAIR_WIDTH])
    temporary = pl.tile.create([1, 2 * CANDIDATE_PAIR_WIDTH], dtype=pl.FP32)
    merged_all = pl.tile.mrgsort(left_pairs, right_pairs, tmp=temporary)
    merged = pl.tile.slice(merged_all, [1, CANDIDATE_PAIR_WIDTH], [0, 0])
    pl.store(merged, [output, 0], arena)


@pl.jit.inline
def _merge_candidate_level(
    arena: pl.Tensor,
    base: pl.Scalar[pl.INDEX],
    count: pl.Scalar[pl.INDEX],
):
    output_count = (count + 1) // 2
    for output in pl.range(output_count):
        left = base + 2 * output
        right = left + 1
        if right < base + count:
            _merge_candidate_pairs(arena, left, right, base + output)
        else:
            forwarded = pl.load(arena, [left, 0], [1, CANDIDATE_PAIR_WIDTH])
            pl.store(forwarded, [base + output, 0], arena)
    return output_count


@pl.jit.inline
def _sort_candidate_leaf(
    scores: pl.Tensor,
    arena: pl.Tensor,
    token: pl.Scalar[pl.INDEX],
    block_begin: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
):
    raw = pl.load(scores, [token, block_begin], [1, CANDIDATE_LEAF_TILE], valid_shape=[1, valid_count])
    padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
    floor = pl.tile.full([1, CANDIDATE_LEAF_TILE], dtype=pl.FP32, value=-1e30)
    values = pl.maximum(padded, floor)
    block_begin_i32 = pl.cast(block_begin, pl.INT32)
    indices = pl.add(pl.tile.arange(0, [1, CANDIDATE_LEAF_TILE], dtype=pl.INT32), block_begin_i32)
    pairs = pl.tile.sort32(values, pl.reinterpret_view(indices, pl.UINT32))
    pairs = pl.tile.mrgsort(pairs, block_len=64)
    pairs = pl.tile.mrgsort(pairs, block_len=256)
    pairs = pl.tile.mrgsort(pairs, block_len=1024)
    pairs = pl.tile.mrgsort(pairs, block_len=4096)
    top_pairs = pl.tile.slice(pairs, [1, CANDIDATE_PAIR_WIDTH], [0, 0])
    pl.store(top_pairs, [output_slot, 0], arena)


@pl.jit.inline
def _sort_candidate_short_leaf(
    scores: pl.Tensor,
    arena: pl.Tensor,
    token: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
):
    raw = pl.load(scores, [token, 0], [1, CANDIDATE_SHORT_LEAF_TILE], valid_shape=[1, valid_count])
    values = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
    values = pl.maximum(values, -1e30)
    indices = pl.tile.arange(0, [1, CANDIDATE_SHORT_LEAF_TILE], dtype=pl.INT32)
    pairs = pl.tile.sort32(values, pl.reinterpret_view(indices, pl.UINT32))
    pairs = pl.tile.mrgsort(pairs, block_len=64)
    pairs = pl.tile.mrgsort(pairs, block_len=256)
    pairs = pl.tile.mrgsort(pairs, block_len=1024)
    top_pairs = pl.tile.slice(pairs, [1, CANDIDATE_PAIR_WIDTH], [0, 0])
    pl.store(top_pairs, [output_slot, 0], arena)


@pl.jit.inline(auto_scope=False)
def _publish_candidate_counts(
    compressed_lens: pl.Tensor,
    pool_block_counts: pl.Tensor,
    active_tokens: pl.Scalar[pl.INDEX],
    positions: pl.Scalar[pl.INDEX],
    capacity: pl.Scalar[pl.INDEX],
):
    """Publish the complete count vector from one worker."""
    tokens = pl.tensor.dim(pool_block_counts, 0)
    for publish_token in pl.range(tokens):
        published_count = pl.cast(0, pl.INT32)
        if publish_token < active_tokens:
            publish_visible = pl.min(pl.max(pl.read(compressed_lens, [publish_token]), 0), positions)
            publish_blocks = (publish_visible + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
            published_count = pl.cast(pl.min(pl.min(publish_blocks, CANDIDATE_TOPK_BLOCKS), capacity), pl.INT32)
        pl.write(pool_block_counts, [publish_token], published_count)
    return pool_block_counts


@pl.jit.inline(auto_scope=False)
def select_candidate_block_pool(
    index_scores: pl.Tensor[[T_DYN, INDEX_SCORE_POSITIONS_DYN], pl.FP32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    scores_ready: pl.Scalar[pl.TASK_ID],
):
    """Publish sorted logical candidate block IDs and per-query counts."""
    tokens = pl.tensor.dim(pool_block_ids, 0)
    positions = pl.tensor.dim(index_scores, 1)
    capacity = pl.tensor.dim(pool_block_ids, 1)
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > tokens:
        active_tokens = pl.cast(tokens, pl.INDEX)
    block_count = (positions + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
    if block_count <= pl.min(capacity, CANDIDATE_TOPK_BLOCKS):
        all_workers = pl.max(1, pl.min(tokens, CANDIDATE_WORKER_TILE))
        with pl.spmd(all_workers, name_hint="pool_all") as all_tid:
            worker = pl.tile.get_block_idx()
            lane_tmp = pl.create_tile([1, 256], dtype=pl.FP32)
            lanes = pl.tile.arange(0, [1, CANDIDATE_TOPK_BLOCKS], dtype=pl.INT32, tmp=lane_tmp)
            for token in pl.range(worker, tokens, all_workers):
                ids = pl.tile.full([1, CANDIDATE_TOPK_BLOCKS], dtype=pl.INT32, value=-1)
                if token < active_tokens:
                    visible = pl.min(pl.max(pl.read(compressed_lens, [token]), 0), positions)
                    visible_blocks = (visible + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
                    valid = pl.tile.cmps(lanes, pl.cast(visible_blocks, pl.INT32), cmp_type=2)
                    ids = pl.tile.select(valid, lanes, ids)
                pl.store(pl.tile.set_validshape(ids, 1, capacity), [token, 0], pool_block_ids)
            if worker == 0:
                _publish_candidate_counts(compressed_lens, pool_block_counts, active_tokens, positions, capacity)
        pool_tid = pl.yield_(all_tid)
    else:
        padded_block_count = (block_count + CANDIDATE_LEAF_TILE - 1) // CANDIDATE_LEAF_TILE * CANDIDATE_LEAF_TILE
        score_tiles = (block_count + CANDIDATE_SCORE_TILE - 1) // CANDIDATE_SCORE_TILE
        block_scores = pl.create_tensor([tokens, padded_block_count], dtype=pl.FP32)
        # Dispatch one guarded work item for an empty batch.
        score_task_tokens = pl.max(active_tokens, pl.cast(1, pl.INDEX))
        score_work_items = score_task_tokens * score_tiles
        score_workers = pl.min(score_work_items, 32)

        with pl.spmd(score_workers, name_hint="pool_score", deps=[scores_ready]) as score_tid:
            worker = pl.tile.get_block_idx()
            for work in pl.range(worker, score_work_items, score_workers):
                token = work // score_tiles
                score_tile = work % score_tiles
                if token < active_tokens:
                    block_begin = score_tile * CANDIDATE_SCORE_TILE
                    position_begin = block_begin * CANDIDATE_BLOCK_SIZE
                    valid_blocks = pl.min(CANDIDATE_SCORE_TILE, block_count - block_begin)
                    valid_positions = pl.min(CANDIDATE_SCORE_TILE * CANDIDATE_BLOCK_SIZE, positions - position_begin)
                    raw = pl.load(
                        index_scores, [token, position_begin],
                        [1, CANDIDATE_SCORE_TILE * CANDIDATE_BLOCK_SIZE], valid_shape=[1, valid_positions],
                    )
                    padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
                    grouped = pl.reshape(padded, [CANDIDATE_SCORE_TILE, CANDIDATE_BLOCK_SIZE])
                    reduce_tmp = pl.create_tile([CANDIDATE_SCORE_TILE, CANDIDATE_BLOCK_SIZE], dtype=pl.FP32)
                    maxima = pl.reshape(pl.row_max(grouped, tmp_tile=reduce_tmp), [1, CANDIDATE_SCORE_TILE])
                    visible = pl.min(pl.max(pl.read(compressed_lens, [token]), 0), positions)
                    if visible > 0:
                        last_block = (visible - 1) // CANDIDATE_BLOCK_SIZE
                        if last_block >= block_begin and last_block < block_begin + valid_blocks:
                            pl.tile.write(maxima, [0, last_block - block_begin], 1e30)
                    maxima = pl.tile.set_validshape(maxima, 1, valid_blocks)
                    pl.store(maxima, [token, block_begin], block_scores)

        arena = pl.create_tensor([tokens * CANDIDATE_MAX_LEAVES, CANDIDATE_PAIR_WIDTH], dtype=pl.FP32)
        with pl.spmd(tokens, name_hint="pool_select", deps=[score_tid]) as selected_tid:
            token = pl.tile.get_block_idx()
            sorted_ids = pl.tile.full([1, CANDIDATE_TOPK_BLOCKS], dtype=pl.INT32, value=-1)
            if token < active_tokens:
                visible = pl.min(pl.max(pl.read(compressed_lens, [token]), 0), positions)
                visible_blocks = (visible + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
                arena_base = token * CANDIDATE_MAX_LEAVES
                count = 0
                if visible_blocks > 0 and visible_blocks <= CANDIDATE_SHORT_LEAF_TILE:
                    _sort_candidate_short_leaf(block_scores, arena, token, visible_blocks, arena_base)
                    count = 1
                elif visible_blocks > 0:
                    leaf_count = (visible_blocks + CANDIDATE_LEAF_TILE - 1) // CANDIDATE_LEAF_TILE
                    for leaf in pl.range(leaf_count):
                        block_begin = leaf * CANDIDATE_LEAF_TILE
                        valid_count = pl.min(CANDIDATE_LEAF_TILE, visible_blocks - block_begin)
                        _sort_candidate_leaf(block_scores, arena, token, block_begin, valid_count, arena_base + leaf)
                    count = leaf_count
                if count > 1:
                    count = _merge_candidate_level(arena, arena_base, count)
                if count > 1:
                    count = _merge_candidate_level(arena, arena_base, count)
                if count > 1:
                    count = _merge_candidate_level(arena, arena_base, count)
                if count > 1:
                    count = _merge_candidate_level(arena, arena_base, count)

                selected_count = pl.min(pl.min(visible_blocks, CANDIDATE_TOPK_BLOCKS), capacity)
                ordered_scores = pl.tile.full([1, CANDIDATE_TOPK_BLOCKS], dtype=pl.FP32, value=-1e30)
                ordered_indices = pl.tile.full([1, CANDIDATE_TOPK_BLOCKS], dtype=pl.INT32, value=-1)
                if selected_count > 0:
                    root = pl.load(arena, [arena_base, 0], [1, CANDIDATE_PAIR_WIDTH])
                    selected = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                    for lane in pl.range(selected_count):
                        block_id = pl.tile.read(selected, [0, lane])
                        pl.tile.write(ordered_scores, [0, lane], -pl.cast(block_id, pl.FP32))
                        pl.tile.write(ordered_indices, [0, lane], block_id)
                ordered = pl.tile.sort32(ordered_scores, pl.reinterpret_view(ordered_indices, pl.UINT32))
                ordered = pl.tile.mrgsort(ordered, block_len=64)
                ordered = pl.tile.mrgsort(ordered, block_len=256)
                ordered = pl.tile.mrgsort(ordered, block_len=1024)
                sorted_ids = pl.tile.gather_mask(ordered, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            # Publish the sorted prefix and -1 padding as one aligned pool row.
            pl.store(pl.tile.set_validshape(sorted_ids, 1, capacity), [token, 0], pool_block_ids)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="pool_count", deps=[selected_tid]) as counts_tid:
            _publish_candidate_counts(compressed_lens, pool_block_counts, active_tokens, positions, capacity)
        pool_tid = pl.yield_(counts_tid)
    return pool_tid


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels" / "idx_k_norm_rope_publish.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _idx_k_norm_rope_publish(
    source: pl.Tensor,
    weight: pl.Tensor,
    cosine: pl.Tensor,
    sine: pl.Tensor,
    slots: pl.Tensor,
    cache: pl.InOut[pl.Tensor],
    scales: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    """Normalize, rotate, and publish group-32 MXFP4 index keys with E8M0 scales."""
    ...


@pl.jit.inline(auto_scope=False)
def publish_index_rows(
    source: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    cosine: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sine: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    scales: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    kv_ready: pl.Scalar[pl.TASK_ID],
):
    """Fuse index RMSNorm, BF16 RoPE, and cache publication with one writer per page."""
    workers = pl.max(1, pl.min(num_tokens, 4))
    with pl.spmd(workers, name_hint="idx_k_norm_rope_publish", deps=[cache_ready, kv_ready]) as published:
        cache = _idx_k_norm_rope_publish(source, weight, cosine, sine, slots, cache, scales, num_tokens)
    return published


@pl.jit.inline(auto_scope=False)
def _plan_index_runs(
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    runs: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.Scalar[pl.INDEX],
    skip_short: pl.constexpr,
):
    """Plan contiguous request groups of at most six query rows."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="idx_plan_req") as plan_tid:
        run_count = pl.cast(0, pl.INT32)
        run_start = pl.cast(0, pl.INT32)
        previous_request = pl.cast(-1, pl.INT32)
        maximum_visible = pl.cast(0, pl.INT32)
        for token in pl.range(num_tokens):
            request = pl.read(request_ids, [token])
            if token > 0:
                if request != previous_request or token - run_start == 6:
                    pl.write(runs, [0, run_count], run_start)
                    pl.write(runs, [1, run_count], pl.cast(token, pl.INT32))
                    pl.write(runs, [2, run_count], maximum_visible)
                    run_count = run_count + pl.cast(1, pl.INT32)
                    run_start = pl.cast(token, pl.INT32)
                    maximum_visible = pl.cast(0, pl.INT32)
            visible = pl.cast(pl.max(0, pl.min(pl.read(lengths, [token]), width)), pl.INT32)
            if skip_short:
                if visible <= INDEX_TOPK:
                    visible = pl.cast(0, pl.INT32)
            maximum_visible = pl.max(maximum_visible, visible)
            pl.write(runs, [3, token + 1], run_count)
            previous_request = request
        if num_tokens > 0:
            pl.write(runs, [0, run_count], run_start)
            pl.write(runs, [1, run_count], num_tokens)
            pl.write(runs, [2, run_count], maximum_visible)
            run_count = run_count + pl.cast(1, pl.INT32)
        pl.write(runs, [3, 0], run_count)
    return plan_tid


@pl.jit.inline(auto_scope=False)
def score_select_reindex(
    query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM // 2], pl.UINT8],
    query_scale: pl.Tensor[[T_DYN, INDEX_H * _SCALE_COUNT], pl.FP8E8M0],
    weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
    index_cache: pl.Tensor,
    index_cache_scale: pl.Tensor,
    block_table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    scores: pl.Tensor[[T_DYN, INDEX_POOL_POSITIONS_DYN], pl.FP32],
    output: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    query_ready: pl.Scalar[pl.TASK_ID],
    weights_ready: pl.Scalar[pl.TASK_ID],
    pool_ready: pl.Scalar[pl.TASK_ID],
):
    """Route identity candidate pools to paged MIX and sparse pools to gathered MIX."""
    capacity = pl.tensor.dim(pool_block_ids, 1)
    width = pl.tensor.dim(scores, 1)
    table_width = pl.tensor.dim(block_table, 1)
    request_count = pl.tensor.dim(block_table, 0)
    cache_pages = pl.tensor.dim(index_cache, 0)
    flags = pl.create_tensor([32, 32], dtype=pl.INT32)
    with pl.spmd(32, name_hint="idx_plan_pool", deps=[cache_ready, pool_ready]) as plan_tid:
        worker = pl.tile.get_block_idx()
        dense = pl.cast(1, pl.INT32)
        mismatch = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0)
        for token in pl.range(worker, num_tokens, 32):
            count = pl.max(0, pl.min(pl.read(pool_block_counts, [token]), capacity))
            visible = pl.max(0, pl.read(lengths, [token]))
            request = pl.read(request_ids, [token])
            if count != (visible + 7) // 8 or visible > width or visible > table_width * 128:
                dense = pl.cast(0, pl.INT32)
            if request < 0 or request >= request_count:
                dense = pl.cast(0, pl.INT32)
            if dense == 1:
                for start in pl.range(0, count, 1024):
                    active = pl.min(1024, count - start)
                    raw = pl.load(pool_block_ids, [token, start], [1, 1024], valid_shape=[1, active])
                    ids = pl.tile.fillpad(raw, pad_value=pl.PadValue.zero)
                    lanes = pl.tile.arange(0, [1, 1024], dtype=pl.INT32)
                    expected = pl.add(lanes, pl.cast(start, pl.INT32))
                    error = pl.abs(pl.cast(pl.sub(ids, expected), pl.FP32))
                    valid = pl.tile.cmps(lanes, pl.cast(active, pl.INT32), cmp_type=2)
                    error = pl.tile.select(valid, error, 0.0)
                    mismatch = pl.add(mismatch, pl.col_sum(pl.reshape(error, [16, 64])))
                page_count = (visible + 127) // 128
                for start in pl.range(0, page_count, 128):
                    active = pl.min(128, page_count - start)
                    raw_pages = pl.load(block_table, [request, start], [1, 128], valid_shape=[1, active])
                    pages = pl.tile.fillpad(raw_pages, pad_value=pl.PadValue.zero)
                    bad = pl.tile.cmps(pages, pl.cast(cache_pages, pl.INT32), cmp_type=5)
                    page_errors = pl.tile.select(bad, pl.tile.full([1, 128], dtype=pl.FP32, value=1.0), 0.0)
                    mismatch = pl.add(mismatch, pl.col_sum(pl.reshape(page_errors, [2, 64])))
        sums = pl.col_sum(pl.reshape(mismatch, [8, 8]))
        for lane in pl.unroll(8):
            if pl.tile.read(sums, [0, lane]) != 0.0:
                dense = pl.cast(0, pl.INT32)
        pl.write(flags, [worker, 0], dense)
    dense = pl.cast(1, pl.INT32)
    for worker in pl.range(32):
        dense = dense * pl.read(flags, [worker, 0])
    if dense != 0:
        tokens = pl.tensor.dim(output, 0)
        dense_width = pl.min(width, table_width * 128)
        paged_scores = pl.create_tensor([tokens, dense_width], dtype=pl.FP32)
        selected, scores_tid = score_select_paged(
            query, query_scale, weights, index_cache, index_cache_scale,
            block_table, request_ids, lengths, paged_scores, output,
            num_tokens, plan_tid, query_ready, weights_ready, capacity, False,
        )
        result = selected
    else:
        result = _score_select_pool(
            query, query_scale, weights, index_cache, index_cache_scale,
            block_table, request_ids, lengths, pool_block_ids, pool_block_counts, scores, output,
            num_tokens, cache_ready, query_ready, weights_ready, plan_tid,
        )
    return result


@pl.jit.inline(auto_scope=False)
def _score_select_pool(
    query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM // 2], pl.UINT8],
    query_scale: pl.Tensor[[T_DYN, INDEX_H * _SCALE_COUNT], pl.FP8E8M0],
    weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
    index_cache: pl.Tensor,
    index_cache_scale: pl.Tensor,
    block_table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    scores: pl.Tensor[[T_DYN, INDEX_POOL_POSITIONS_DYN], pl.FP32],
    output: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    query_ready: pl.Scalar[pl.TASK_ID],
    weights_ready: pl.Scalar[pl.TASK_ID],
    pool_ready: pl.Scalar[pl.TASK_ID],
):
    """Gather candidate blocks and select compact scores with physical-row mappings."""
    capacity = pl.tensor.dim(pool_block_ids, 1)
    tokens = pl.tensor.dim(output, 0)
    width = capacity * CANDIDATE_BLOCK_SIZE
    panels = (width + 255) // 256
    gathered_rows = panels * 256
    total_rows = tokens * gathered_rows
    total_scales = total_rows * _SCALE_COUNT
    cache_pages = pl.tensor.dim(index_cache, 0)
    cache_rows = cache_pages * 128
    cache_flat = pl.reshape(index_cache, [cache_rows, INDEX_DIM // 2])
    scale_flat = pl.reshape(index_cache_scale, [cache_rows // 8, 32])
    payloads = pl.create_tensor([total_rows, INDEX_DIM // 2], dtype=pl.UINT8)
    scales = pl.create_tensor([1, total_scales], dtype=pl.UINT8)
    valid = pl.create_tensor([1, total_rows], dtype=pl.INT32)
    physical = pl.create_tensor([1, total_rows], dtype=pl.INT32)
    runs = pl.create_tensor([4, tokens + 1], dtype=pl.INT32)
    table_width = pl.tensor.dim(block_table, 1)
    request_count = pl.tensor.dim(block_table, 0)
    with pl.spmd(32, name_hint="idx_gather_pool", deps=[cache_ready, pool_ready]) as gather_tid:
        worker = pl.tile.get_block_idx()
        for work in pl.range(worker, num_tokens * panels, 32):
            token = work // panels
            panel = work % panels
            slot_begin = panel * 32
            count = pl.max(0, pl.min(pl.read(pool_block_counts, [token]), capacity))
            request = pl.read(request_ids, [token])
            visible = pl.max(0, pl.read(lengths, [token]))
            packed = pl.tile.full([256, INDEX_DIM // 2], dtype=pl.INT8, value=0)
            raw_scale = pl.tile.full([1, 256 * _SCALE_COUNT], dtype=pl.INT8, value=127)
            valid_lanes = pl.tile.full([1, 256], dtype=pl.INT32, value=0)
            physical_lanes = pl.tile.full([1, 256], dtype=pl.INT32, value=-1)
            for block in pl.range(32):
                slot = slot_begin + block
                if slot < count:
                    logical_block = pl.read(pool_block_ids, [token, slot])
                    if logical_block >= 0 and logical_block < (visible + 7) // 8:
                        logical_page = logical_block // 16
                        if request >= 0 and request < request_count and logical_page < table_width:
                            page = pl.read(block_table, [pl.cast(request, pl.INDEX), logical_page])
                            if page >= 0 and page < cache_pages:
                                physical_row = pl.cast(page, pl.INDEX) * 128 + (logical_block % 16) * 8
                                payload_raw = pl.load(cache_flat, [physical_row, 0], [8, INDEX_DIM // 2], target_memory=pl.MemorySpace.Vec)
                                payload = pl.reinterpret_view(payload_raw, pl.INT8)
                                packed[block * 8:block * 8 + 8, :] = payload
                                scales_raw = pl.load(scale_flat, [physical_row // 8, 0], [1, 32], target_memory=pl.MemorySpace.Vec)
                                block_scale_bytes = pl.reinterpret_view(scales_raw, pl.UINT8)
                                block_scales = pl.reinterpret_view(block_scale_bytes, pl.INT8)
                                raw_scale[:, block * 32:block * 32 + 32] = block_scales
                                lanes = pl.tile.arange(0, [1, 8], dtype=pl.INT32)
                                remaining = pl.cast(pl.min(8, visible - logical_block * 8), pl.INT32)
                                lane_valid = pl.tile.cmps(lanes, remaining, cmp_type=2)
                                ones = pl.tile.full([1, 8], dtype=pl.INT32, value=1)
                                block_valid = pl.tile.select(lane_valid, ones, 0)
                                block_physical = pl.add(lanes, pl.cast(physical_row, pl.INT32))
                                block_physical = pl.tile.select(lane_valid, block_physical, -1)
                                valid_lanes[:, block * 8:block * 8 + 8] = block_valid
                                physical_lanes[:, block * 8:block * 8 + 8] = block_physical
            row_offset = token * gathered_rows + panel * 256
            packed_bytes = pl.reinterpret_view(packed, pl.UINT8)
            pl.store(packed_bytes, [row_offset, 0], payloads)
            scale_bytes = pl.reinterpret_view(raw_scale, pl.UINT8)
            pl.store(scale_bytes, [0, row_offset * _SCALE_COUNT], scales)
            pl.store(valid_lanes, [0, row_offset], valid)
            pl.store(physical_lanes, [0, row_offset], physical)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="idx_plan_pool_runs", deps=[pool_ready]) as runs_tid:
        for token in pl.range(num_tokens):
            count = pl.max(0, pl.min(pl.read(pool_block_counts, [token]), capacity))
            pl.write(runs, [0, token], pl.cast(token, pl.INT32))
            pl.write(runs, [1, token], pl.cast(token + 1, pl.INT32))
            pl.write(runs, [2, token], pl.cast(count * 8, pl.INT32))
            pl.write(runs, [3, token + 1], pl.cast(token, pl.INT32))
        pl.write(runs, [3, 0], num_tokens)
    workers = pl.max(1, pl.min(32, pl.system.available_cluster_count()))
    candidate_keys = pl.create_tensor([tokens, workers * INDEX_TOPK], dtype=pl.UINT16)
    candidate_indices = pl.create_tensor([tokens, workers * INDEX_TOPK], dtype=pl.INT32)
    run_stride = pl.tensor.dim(runs, 1)
    query_rows = pl.slice(query, [tokens, INDEX_H * INDEX_DIM // 2], [0, 0])
    scale_rows = pl.slice(query_scale, [tokens, INDEX_H * _SCALE_COUNT], [0, 0])
    weight_rows = pl.slice(weights, [tokens, INDEX_H], [0, 0])
    request_rows = pl.reshape(request_ids, [tokens])
    length_rows = pl.reshape(lengths, [tokens])
    output_rows = pl.reshape(output, [tokens, INDEX_TOPK])
    score_rows = pl.slice(scores, [tokens, width], [0, 0])
    with pl.spmd(workers, name_hint="idx_score_topk", sync_start=True, deps=[gather_tid, runs_tid, query_ready, weights_ready]) as mix_tid:
        _idx_pool_score_topk(
            query_rows, scale_rows, weight_rows, payloads, scales,
            block_table, request_rows, length_rows, output_rows, runs, candidate_keys, candidate_indices,
            valid, physical, score_rows, width, run_stride, table_width, num_tokens, gathered_rows,
            pl.cast(0, pl.INT32), workers, pl.cast(0, pl.INT32),
        )
    return mix_tid


@pl.jit.inline(auto_scope=False)
def score_select_paged(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    index_cache: pl.Tensor,
    index_cache_scale: pl.Tensor,
    block_table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    scores: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    query_ready: pl.Scalar[pl.TASK_ID],
    weights_ready: pl.Scalar[pl.TASK_ID],
    pool_capacity: pl.Scalar[pl.INDEX],
    c2a: pl.constexpr,
):
    """Score history shards and merge their TopK within one MIX."""
    width = pl.tensor.dim(scores, 1)
    capacity = pl.tensor.dim(output, 0)
    runs = pl.create_tensor([4, capacity + 1], dtype=pl.INT32)
    plan_tid = _plan_index_runs(request_ids, lengths, runs, num_tokens, width, c2a)
    run_count = pl.read(runs, [3, 0])
    workers = pl.max(1, pl.min(32, pl.system.available_cluster_count()))
    panels = (width + 255) // 256
    shards = workers
    if panels > 100:
        shards = pl.max(workers, (run_count * panels + 99) // 100)
    block_count = (width + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
    emit_scores = pl.cast(0, pl.INT32)
    if not c2a:
        if block_count > pl.min(pool_capacity, CANDIDATE_TOPK_BLOCKS):
            emit_scores = pl.cast(1, pl.INT32)
    run_stride = pl.tensor.dim(runs, 1)
    table_width = pl.tensor.dim(block_table, 1)
    active_tokens = pl.cast(num_tokens, pl.INT32)
    direct_short = pl.cast(0, pl.INT32)
    if c2a:
        direct_short = pl.cast(1, pl.INT32)
    query_rows = pl.slice(query, [capacity, INDEX_H * INDEX_DIM // 2], [0, 0])
    scale_rows = pl.slice(query_scale, [capacity, INDEX_H * (INDEX_DIM // INDEX_CACHE_GROUP)], [0, 0])
    weight_rows = pl.slice(weights, [capacity, INDEX_H], [0, 0])
    request_rows = pl.reshape(request_ids, [capacity])
    length_rows = pl.reshape(lengths, [capacity])
    mixed_output = pl.reshape(output, [capacity, INDEX_TOPK])
    score_rows = pl.slice(scores, [capacity, width], [0, 0])
    candidate_keys = pl.create_tensor([capacity, shards * INDEX_TOPK], dtype=pl.UINT16)
    candidate_indices = pl.create_tensor([capacity, shards * INDEX_TOPK], dtype=pl.INT32)
    with pl.spmd(workers, name_hint="idx_score_topk", sync_start=True, deps=[cache_ready, query_ready, weights_ready, plan_tid]) as mix_tid:
        mixed_output = _idx_score_topk(
            query_rows, scale_rows, weight_rows, index_cache, index_cache_scale,
            block_table, request_rows, length_rows, mixed_output, runs, candidate_keys, candidate_indices,
            block_table, block_table, score_rows, width, run_stride, table_width, active_tokens, 0,
            emit_scores, shards, direct_short,
        )
    return mix_tid, mix_tid


@pl.jit.inline(auto_scope=False)
def project_index_weights(
    source: pl.Tensor,
    weight: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    tokens = pl.tensor.dim(source, 0)
    partials = pl.create_tensor([INDEX_WEIGHT_K_SPLIT_TILE * tokens, INDEX_H], dtype=pl.FP32)
    blocks = (num_tokens + 79) // 80
    with pl.spmd(INDEX_WEIGHT_K_SPLIT_TILE, name_hint="proj_idx_w", deps=[ready]) as cube_tid:
        split = pl.tile.get_block_idx()
        begin = split * (D // INDEX_WEIGHT_K_SPLIT_TILE)
        for block in pl.range(blocks):
            row = block * 80
            rows = pl.min(80, num_tokens - row)
            a = pl.load(source, [row, begin], [80, INDEX_WEIGHT_K_TILE], valid_shape=[rows, INDEX_WEIGHT_K_TILE], target_memory=pl.MemorySpace.Mat)
            w = pl.load(weight, [begin, 0], [INDEX_WEIGHT_K_TILE, INDEX_H], target_memory=pl.MemorySpace.Mat)
            full_a = pl.set_validshape(a, 80, INDEX_WEIGHT_K_TILE)
            acc = pl.matmul(full_a, w)
            k_begin = begin + INDEX_WEIGHT_K_TILE
            k_end = begin + D // INDEX_WEIGHT_K_SPLIT_TILE
            for k in pl.pipeline(k_begin, k_end, INDEX_WEIGHT_K_TILE, stage=2):
                a = pl.load(source, [row, k], [80, INDEX_WEIGHT_K_TILE], valid_shape=[rows, INDEX_WEIGHT_K_TILE], target_memory=pl.MemorySpace.Mat)
                w = pl.load(weight, [k, 0], [INDEX_WEIGHT_K_TILE, INDEX_H], target_memory=pl.MemorySpace.Mat)
                full_a = pl.set_validshape(a, 80, INDEX_WEIGHT_K_TILE)
                acc = pl.matmul_acc(acc, full_a, w)
            value = pl.set_validshape(acc, rows, INDEX_H)
            partials = pl.store(value, [split * tokens + row, 0], partials)
    with pl.spmd(pl.max(pl.min((num_tokens + 15) // 16, 8), 1), name_hint="reduce_idx_w", deps=[cube_tid]) as completed:
        worker = pl.tile.get_block_idx()
        workers = pl.tile.get_block_num()
        for row in pl.range(worker * 16, num_tokens, workers * 16):
            rows = pl.min(16, num_tokens - row)
            reduced = pl.load(partials, [row, 0], [16, INDEX_H], valid_shape=[rows, INDEX_H])
            for split in pl.range(1, INDEX_WEIGHT_K_SPLIT_TILE):
                partial_value = pl.load(partials, [split * tokens + row, 0], [16, INDEX_H], valid_shape=[rows, INDEX_H])
                reduced = pl.add(reduced, partial_value)
            rounded = pl.cast(reduced, pl.BF16, mode="rint")
            output = pl.store(rounded, [row, 0], output)
    return completed


@pl.jit.inline(auto_scope=False)
def c1a_index(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[
        [Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
    ],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    score_width: pl.Scalar[pl.INDEX],
    entry_ready: pl.Scalar[pl.TASK_ID],
    projection_payload: pl.Tensor,
    projection_scale_backing: pl.Tensor,
):
    tokens = pl.tensor.dim(x, 0)
    query = pl.create_tensor([tokens, INDEX_H * INDEX_DIM // 2], dtype=pl.UINT8)
    query_scale = pl.create_tensor([tokens, INDEX_H * (INDEX_DIM // INDEX_CACHE_GROUP)], dtype=pl.FP8E8M0)
    with pl.spmd(16, name_hint="idx_q", deps=[entry_ready]) as query_tid:
        query = _idx_q(
            projection_payload, index_wq_b, projection_scale_backing, index_wq_b_scale,
            cos, sin, query, query_scale, num_tokens,
        )
    weights = pl.create_tensor([tokens, INDEX_H], dtype=pl.BF16)
    weights_tid = project_index_weights(
        x, index_weights_proj, weights, num_tokens, entry_ready
    )
    scores = pl.create_tensor([tokens, score_width], dtype=pl.FP32)
    return query, query_scale, weights, scores, query_tid, weights_tid


@pl.jit.inline(auto_scope=False)
def project_index_key(
    source: pl.Tensor,
    weight: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
    max_workers: pl.constexpr,
):
    result, completed = gemm_bf16_to_bf16(
        source, weight, output, num_tokens, ready,
        width=HEAD_DIM, output_width=INDEX_DIM, n_tile=INDEX_PROJ_N_TILE, k_tile=INDEX_PROJ_K_TILE,
        workers=max_workers, name_hint='proj_idx_k',
    )
    return completed


@pl.jit.inline(auto_scope=False)
def decode_c2a_indexer(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    projection_payload: pl.Tensor,
    projection_scales: pl.Tensor,
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, (INDEX_DIM // 2)], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, (INDEX_DIM // INDEX_CACHE_GROUP)], pl.FP8E8M0],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    latent_ready: pl.Scalar[pl.TASK_ID],
):
    """Publish C2A keys and select long histories through the shared MX scorer."""
    tokens = pl.tensor.dim(latent, 0)
    projected_key = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    projected_tid = project_index_key(latent, index_wk, projected_key, num_tokens, latent_ready, max_workers=32)
    index_publish_tid = publish_index_rows(
        projected_key, index_norm_weight, compressed_rope_cos, compressed_rope_sin, compressed_slots,
        index_cache, index_cache_scale, num_tokens, cache_ready, projected_tid,
    )
    # The dense arena follows logical page-table capacity, never allocated cache pages.
    score_width = pl.tensor.dim(index_block_table, 1) * 128
    query, query_scale, raw_weights, scores, query_tid, weights_tid = c1a_index(
        x,
        index_wq_b, index_wq_b_scale, index_weights_proj,
        rope_cos, rope_sin,
        num_tokens, score_width,
        cache_ready,
        projection_payload, projection_scales,
    )
    selected_tid, scores_tid = score_select_paged(
        query, query_scale, raw_weights, index_cache, index_cache_scale,
        index_block_table, token_to_req_indices, compressed_lens, scores, topk_indices,
        num_tokens, index_publish_tid, query_tid, weights_tid, score_width // CANDIDATE_BLOCK_SIZE, True,
    )
    return selected_tid


import torch
import torch.nn.functional as F
from qkv_proj_rope import golden_rope_tail, golden_rope_interleave
from quant import dequantize_mxfp4_cache, quantize_mxfp4_cache
from gemm import golden_gemm_bf16_via_mxfp8_to_bf16_sequential
from rmsnorm import golden_rms_norm


def mapped_fp4_cache_compare(
    mapping_name: str,
    *,
    mapping_shape: tuple[int, ...],
    block_size: int,
    pool_name: str,
    leading_rank_axis: bool = False,
    max_error_ratio: float = 0.005,
):
    """Bound sparse E2M1 changes after floating-point model computation.

    The existing mapped comparator keeps every unwritten byte exact and bounds
    differing written bytes by the declared ratio. A second, unconditional gate
    rejects sign changes (including signed zero) or a jump over more than one
    magnitude level in either nibble. This is an end-to-end numerical policy,
    not an exact quantizer test or a proof of a floating-point error bound.
    """
    from golden import mapped_pool_ratio_allclose

    mapped = mapped_pool_ratio_allclose(
        mapping_name, mapping_shape=mapping_shape, block_size=block_size,
        leading_rank_axis=leading_rank_axis, pool_name=pool_name,
        atol=0.0, rtol=0.0, max_error_ratio=max_error_ratio,
    )

    def compare(actual, expected, **kwargs):
        if actual.dtype != torch.uint8 or expected.dtype != torch.uint8:
            return False, f"{pool_name}: packed E2M1 payload must have UINT8 storage"
        ok, detail = mapped(actual, expected, **kwargs)
        if not ok:
            return False, detail
        a = actual.detach().cpu().to(torch.int16)
        e = expected.detach().cpu().to(torch.int16)
        sign_changed = torch.bitwise_and(torch.bitwise_xor(a, e), 0x88) != 0
        if bool(sign_changed.any()):
            return False, f"{pool_name}: E2M1 sign changed in {int(sign_changed.sum())} byte(s)"
        low_distance = ((a & 7) - (e & 7)).abs()
        high_distance = (((a >> 4) & 7) - ((e >> 4) & 7)).abs()
        invalid = (low_distance > 1) | (high_distance > 1)
        if bool(invalid.any()):
            return False, f"{pool_name}: E2M1 magnitude skips a level in {int(invalid.sum())} byte(s)"
        return True, ""

    return compare


def golden_index_query_fp4(value: torch.Tensor) -> torch.Tensor:
    """Round group-32 index queries to E2M1 with upward E8M0 scales."""
    payload, scales = quantize_mxfp4_cache(value, 32, "e8m0", round_to_even=True)
    return dequantize_mxfp4_cache(payload, scales, 32, "e8m0").to(value.dtype)


def golden_index_key(
    latent: torch.Tensor,
    wk: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Project an unrotated compressor latent into a rotated index key."""
    from quant import unpack_bf16_nz

    projected = torch.matmul(latent.float(), unpack_bf16_nz(wk).float()).to(latent.dtype)
    key = golden_rms_norm(projected, norm_weight)
    rd = cos.shape[-1] * 2
    return torch.cat((key[..., :-rd], golden_rope_interleave(key[..., -rd:], cos, sin)), dim=-1)


def golden_select_candidate_block_pool(
    logits: torch.Tensor,
    compressed_lens: torch.Tensor | int,
    topk_blocks: int = FLASH.candidate_topk_blocks,
    block_size: int = FLASH.candidate_block_size,
    *,
    num_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sorted logical block IDs and counts with a causal tail and fixed selected prefix."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be rank-2, got shape {tuple(logits.shape)}")
    if block_size != CANDIDATE_BLOCK_SIZE:
        raise ValueError(f"block_size must be {CANDIDATE_BLOCK_SIZE}, got {block_size}")
    if topk_blocks != CANDIDATE_TOPK_BLOCKS:
        raise ValueError(f"topk_blocks must be {CANDIDATE_TOPK_BLOCKS}, got {topk_blocks}")

    tokens, width = logits.shape
    # Inactive tensor rows publish canonical empty pool state.
    active_tokens = tokens if num_tokens is None else max(0, min(tokens, int(num_tokens)))
    capacity = index_pool_block_capacity(width)
    ids = torch.full((tokens, capacity), -1, dtype=torch.int32, device=logits.device)
    counts = torch.zeros((tokens,), dtype=torch.int32, device=logits.device)
    if width == 0:
        return ids, counts

    lengths = torch.as_tensor(compressed_lens, device=logits.device, dtype=torch.int64)
    if lengths.ndim == 0:
        lengths = lengths.expand(tokens)
    if lengths.ndim != 1 or lengths.numel() != tokens:
        raise ValueError("compressed_lens must provide one causal length per score row")

    padded = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    block_scores = padded.unflatten(-1, (-1, block_size)).amax(dim=-1)
    for token in range(active_tokens):
        visible = max(0, min(int(lengths[token]), width))
        visible_blocks = (visible + block_size - 1) // block_size
        if visible_blocks == 0:
            continue
        selected_count = min(visible_blocks, topk_blocks, capacity)
        # Rank invalid block maxima at the device sort floor.
        scores = block_scores[token, :visible_blocks].clone().clamp_min(-1e30)
        # Include the newest causal block independently of its score.
        scores[visible_blocks - 1] = 1e30
        # Resolve equal block maxima by stable logical order.
        selected = torch.argsort(scores, descending=True, stable=True)[:selected_count].to(torch.int32)
        ids[token, :selected_count] = selected.sort().values
        counts[token] = selected_count
    return ids, counts


def golden_index_weighted_sum(dots, weights):
    """Accumulate heads in order with one BF16 rounding per fused multiply-add."""
    values = torch.zeros(dots.shape[-1], dtype=torch.bfloat16, device=dots.device)
    positive = dots.relu().double()
    head_weights = weights.double()
    for head in range(dots.shape[0]):
        values = (positive[head] * head_weights[head] + values.double()).bfloat16()
    return values.float()


def golden_c1a_index_scores(a, qr, index_payload, index_scale):
    """Rebuild the per-token index scores and the physical row behind every logical column."""
    from qkv_proj_rope import golden_rope_interleave as rope_interleave
    from quant import dequantize_mxfp4_cache, unpack_bf16_nz, unpack_fp8_nz

    x = a["x"]
    lengths = a["compressed_lens"]
    # Logical page-table score columns.
    width = a["index_block_table"].shape[-1] * 128
    heads = a["index_weights_proj"].shape[-1]
    iq = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(qr, unpack_fp8_nz(a["index_wq_b"]), a["index_wq_b_scale"])
    if iq.shape[-1] % heads:
        raise RuntimeError(f"index query width {iq.shape[-1]} is not divisible by {heads} heads")
    dim = iq.shape[-1] // heads
    rd = a["rope_cos"].shape[-1] * 2
    iq = iq.unflatten(-1, (heads, dim))
    iq = torch.cat((iq[..., :-rd], rope_interleave(iq[..., -rd:], a["rope_cos"], a["rope_sin"])), -1)
    iq = golden_index_query_fp4(iq)
    keys = dequantize_mxfp4_cache(index_payload, index_scale, 32, "e8m0").to(x.dtype).flatten(0, 1)[:, 0]
    # The device scales the projected head weight in FP32 and rounds it once to BF16.
    weight = ((x).float() @ unpack_bf16_nz(a["index_weights_proj"]).float()).to(torch.bfloat16).float() * (dim * heads) ** -0.5
    weight = weight.to(torch.bfloat16).float()
    scores = torch.full((x.shape[0], width), -torch.inf)
    physical = torch.full((x.shape[0], width), -1, dtype=torch.int64)
    for t in range(x.shape[0]):
        count = max(0, min(int(lengths[t]), width))
        pos = torch.arange(count)
        pages = a["index_block_table"][a["request_ids"][t].long(), pos // 128]
        valid = (pages >= 0) & (pages < index_payload.shape[0])
        valid_pos = pos[valid]
        rows = pages[valid] * 128 + valid_pos % 128
        physical[t, valid_pos] = rows
        if not valid_pos.numel():
            continue
        # BF16 dot products, ReLU, and sequential BF16 head FMAs.
        dots = ((iq[t]).float() @ (keys[rows.long()].T).float()).to(torch.bfloat16)
        values = golden_index_weighted_sum(dots, weight[t])
        scores[t, valid_pos] = values
    return scores, physical


def golden_pool_index_scores(scores, physical, pool_block_ids, pool_block_counts):
    """Pack dense logical scores into the Reindex block-pool score layout."""
    tokens, dense_width = scores.shape
    capacity = pool_block_ids.shape[-1]
    compact_width = capacity * FLASH.candidate_block_size
    compact_scores = torch.full((tokens, compact_width), -torch.inf, dtype=scores.dtype)
    compact_physical = torch.full((tokens, compact_width), -1, dtype=physical.dtype)
    for token in range(tokens):
        count = max(0, min(int(pool_block_counts[token]), capacity))
        for slot in range(count):
            block = int(pool_block_ids[token, slot])
            logical_begin = block * FLASH.candidate_block_size
            if block < 0 or logical_begin >= dense_width:
                continue
            logical_end = min(logical_begin + FLASH.candidate_block_size, dense_width)
            compact_begin = slot * FLASH.candidate_block_size
            compact_end = compact_begin + (logical_end - logical_begin)
            compact_scores[token, compact_begin:compact_end] = scores[token, logical_begin:logical_end]
            compact_physical[token, compact_begin:compact_end] = physical[token, logical_begin:logical_end]
    return compact_scores, compact_physical


def golden_c1a_index_topk(scores, physical, lengths, pool_block_counts=None):
    """Select finite TopK rows with stable ties in logical-position order."""
    topk = torch.full((scores.shape[0], INDEX_TOPK), -1, dtype=torch.int32)
    for token in range(scores.shape[0]):
        if pool_block_counts is None:
            count = min(int(lengths[token]), scores.shape[-1])
        else:
            count = min(int(pool_block_counts[token]) * FLASH.candidate_block_size, scores.shape[-1])
        chosen = torch.argsort(scores[token, :count], descending=True, stable=True)[:INDEX_TOPK]
        chosen = chosen[torch.isfinite(scores[token, chosen])].sort().values
        topk[token, :chosen.numel()] = physical[token, chosen].int()
    return topk


def _select_shared_c2a_topk(scores, physical_rows, lengths):
    """Select C2A rows in logical order, including every valid short-history row."""
    tokens, width = scores.shape
    output = torch.full((tokens, INDEX_TOPK), -1, dtype=torch.int32)
    for token in range(tokens):
        count = max(0, min(int(lengths[token]), width))
        if count <= INDEX_TOPK:
            output[token, :count] = physical_rows[token, :count].to(torch.int32)
            continue
        # Stable descending order makes the common logical-earliest cutoff explicit.
        logical = torch.argsort(scores[token, :count], descending=True, stable=True)[:INDEX_TOPK]
        logical = logical[torch.isfinite(scores[token, logical])].sort().values
        rows = physical_rows[token, logical]
        rows = rows[rows >= 0]
        output[token, :rows.numel()] = rows.to(torch.int32)
    return output


def golden_c2a_index(t, qr, index_cache, index_cache_scale):
    """Score and select C2A paged history with BF16 head reduction."""
    from quant import unpack_bf16_nz, unpack_fp8_nz

    x = t["x"]
    tokens = x.shape[0]
    index_dim = t["index_norm_weight"].numel()
    index_heads = t["index_weights_proj"].shape[-1]
    index_values = dequantize_mxfp4_cache(index_cache, index_cache_scale, INDEX_CACHE_GROUP, "e8m0").flatten(0, -2)
    # Accumulate the index-query projection in sequential group-32 partials.
    index_q = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(qr, unpack_fp8_nz(t["index_wq_b"]), t["index_wq_b_scale"]).unflatten(-1, (index_heads, index_dim))
    index_q = golden_rope_tail(index_q, t["rope_cos"], t["rope_sin"])
    index_q = golden_index_query_fp4(index_q)
    head_weights = ((x).float() @ unpack_bf16_nz(t["index_weights_proj"]).float()).to(torch.bfloat16).float()
    head_weights = (head_weights * (index_dim * index_heads) ** -0.5).to(torch.bfloat16).float()
    lens = t["compressed_lens"].long()
    score_width = t["index_block_table"].shape[1] * 128
    scores = torch.full((tokens, score_width), -torch.inf)
    physical_rows = torch.full((tokens, score_width), -1, dtype=torch.int64)
    for token in range(tokens):
        count = max(0, min(int(lens[token]), score_width))
        request = int(t["token_to_req_indices"][token])
        if not count or not 0 <= request < t["index_block_table"].shape[0]:
            continue
        logical = torch.arange(count)
        pages = t["index_block_table"][request, logical // 128].long()
        rows = pages * 128 + logical.remainder(128)
        valid = (pages >= 0) & (rows >= 0) & (rows < index_values.shape[0])
        valid_logical = logical[valid]
        valid_rows = rows[valid]
        physical_rows[token, valid_logical] = valid_rows
        if count <= INDEX_TOPK or not valid_rows.numel():
            continue
        # Round each fused head multiply-add to BF16.
        dots = ((index_q[token]).float() @ (index_values[valid_rows].to(torch.bfloat16).T).float()).to(torch.bfloat16)
        total = golden_index_weighted_sum(dots, head_weights[token])
        scores[token, valid_logical] = total
    return _select_shared_c2a_topk(scores, physical_rows, lens)


__all__ = [
    "c1a_index",
    "decode_c2a_indexer",
    "golden_c2a_index",
    "golden_c1a_index_scores",
    "golden_pool_index_scores",
    "golden_c1a_index_topk",
    "score_select_reindex",
]


def index_pool_block_capacity(score_width: int) -> int:
    """Return the aligned compact-pool capacity for one score row."""
    if score_width < 0:
        raise ValueError(f"score_width must be nonnegative, got {score_width}")
    block_size = FLASH.candidate_block_size
    blocks = max(1, (score_width + block_size - 1) // block_size)
    aligned = (blocks + 31) // 32 * 32
    return min(FLASH.candidate_topk_blocks, aligned)
