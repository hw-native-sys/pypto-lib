# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode indexer query preparation, key publication, scoring, and TopK."""

from gemm import (
    gemm_bf16_to_bf16_staged,
    gemm_bf16_via_mxfp8_to_bf16_staged,
)
import pypto.language as pl
from rmsnorm import make_norm_with_deps
from config import (
    B_DYN, CMP_POSITIONS_DYN, FLASH, INDEX_BLOCKS_DYN, INDEX_CACHE_GROUP, INDEX_POOL_BLOCKS_DYN,
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
INDEX_QUERY_WIDTH = FLASH.index_n_heads * INDEX_DIM
INDEX_QUERY_SCALE_WIDTH = INDEX_QUERY_WIDTH // 32
CANDIDATE_BLOCK_SIZE = FLASH.candidate_block_size
CANDIDATE_TOPK_BLOCKS = FLASH.candidate_topk_blocks
CANDIDATE_PAIR_WIDTH = 2 * CANDIDATE_TOPK_BLOCKS
TOPK_CACHED_PAGES = (FLASH.max_position_embeddings + 127) // 128

_PAGE_SIZE = 128
_SCALE_COUNT = INDEX_DIM // INDEX_CACHE_GROUP
_SCORE_SCALE = (INDEX_DIM * INDEX_H) ** -0.5

# tiling
INDEX_PROJ_N_TILE = 128
INDEX_WEIGHT_N_TILE = min(128, INDEX_H)
INDEX_PROJ_K_TILE = 256


CANDIDATE_SCORE_TILE = 1024
CANDIDATE_LEAF_TILE = 8192
CANDIDATE_SHORT_LEAF_TILE = 2048
CANDIDATE_MAX_LEAVES = (FLASH.max_position_embeddings // CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_LEAF_TILE + 1
INDEX_SCORE_POSITIONS_DYN = pl.dynamic("V41_INDEX_SCORE_POSITIONS_DYN")

# Request-group score capacities for the 8192-start-position workloads.
C1A_REQUEST_GROUP_SCORE_WIDTH = 8320
C2A_REQUEST_GROUP_SCORE_WIDTH = 4224






@pl.jit.inline(auto_scope=False)
def quantize_index_query(
    source: pl.Tensor[[T_DYN, INDEX_QUERY_WIDTH], pl.BF16],
    cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    output: pl.Tensor,
    output_scale: pl.Tensor[[T_DYN, INDEX_QUERY_SCALE_WIDTH], pl.FP8E8M0],
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Rotate query heads and quantize group-32 E2M1 carriers with E8M0 scales."""
    # Materialize the producer dependency before entering the scoped task.
    entry_deps = pl.array.create(1, pl.TASK_ID)
    entry_deps[0] = ready
    with pl.spmd(32, name_hint="idx_q_fp4", deps=[entry_deps[0]]) as quant_tid:
        worker = pl.tile.get_block_idx()
        for t in pl.range(worker, num_tokens, 32):
            raw = pl.load(source, [t, 0], [1, INDEX_QUERY_WIDTH])
            heads = pl.reshape(raw, [INDEX_H, INDEX_DIM])
            tail = pl.cast(heads[:, INDEX_DIM - ROPE_DIM:INDEX_DIM], pl.FP32)
            even = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            cosine = pl.load(cos, [t, 0], [1, ROPE_DIM // 2])
            sine = pl.mul(pl.load(sin, [t, 0], [1, ROPE_DIM // 2]), 1.0)
            even_cos = pl.col_expand_mul(even, cosine)
            odd_sin = pl.col_expand_mul(odd, sine)
            real = pl.sub(even_cos, odd_sin)
            even_sin = pl.col_expand_mul(even, sine)
            odd_cos = pl.col_expand_mul(odd, cosine)
            imag = pl.add(even_sin, odd_cos)
            real_lanes = pl.tile.full([INDEX_H, ROPE_DIM], dtype=pl.FP32, value=0.0)
            real_lanes = pl.tile.scatter_mask(real_lanes, real, mask_pattern=pl.tile.MaskPattern.P0101)
            imag_lanes = pl.tile.full([INDEX_H, ROPE_DIM], dtype=pl.FP32, value=0.0)
            imag_lanes = pl.tile.scatter_mask(imag_lanes, imag, mask_pattern=pl.tile.MaskPattern.P1010)
            real_bits = pl.reinterpret_view(real_lanes, pl.UINT32)
            imag_bits = pl.reinterpret_view(imag_lanes, pl.UINT32)
            rotated_bits = pl.or_(real_bits, imag_bits)
            rotated = pl.reinterpret_view(rotated_bits, pl.FP32)
            rounded_tail = pl.cast(rotated, pl.BF16, mode="rint")
            rotated_heads = pl.tile.assemble(heads, rounded_tail, [0, INDEX_DIM - ROPE_DIM])
            values = pl.reshape(pl.cast(rotated_heads, pl.FP32), [INDEX_QUERY_SCALE_WIDTH, 32])
            reduce_tmp = pl.tile.create([INDEX_QUERY_SCALE_WIDTH, 32], dtype=pl.FP32)
            raw_maximum = pl.row_max(pl.abs(values), tmp_tile=reduce_tmp)
            maximum = pl.maximum(raw_maximum, 7.052966104933725e-38)
            bits = pl.reinterpret_view(pl.mul(maximum, 1.0 / 6.0), pl.INT32)
            exponent = pl.shrs(pl.add(bits, 8388607), 23)
            scale = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
            normalized = pl.minimum(pl.maximum(pl.row_expand_div(values, scale), -6.0), 6.0)
            magnitude = pl.abs(normalized)
            rounded = pl.tile.full([INDEX_QUERY_SCALE_WIDTH, 32], dtype=pl.FP32, value=0.0)
            select_tmp = pl.tile.create([1, 16], dtype=pl.UINT32)
            predicate = pl.tile.cmps(magnitude, 0.25, cmp_type=4)
            level = pl.tile.full([INDEX_QUERY_SCALE_WIDTH, 32], dtype=pl.FP32, value=0.5)
            rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
            predicate = pl.tile.cmps(magnitude, 0.75, cmp_type=5)
            level = pl.tile.full([INDEX_QUERY_SCALE_WIDTH, 32], dtype=pl.FP32, value=1.0)
            rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
            predicate = pl.tile.cmps(magnitude, 1.25, cmp_type=4)
            level = pl.tile.full([INDEX_QUERY_SCALE_WIDTH, 32], dtype=pl.FP32, value=1.5)
            rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
            predicate = pl.tile.cmps(magnitude, 1.75, cmp_type=5)
            level = pl.tile.full([INDEX_QUERY_SCALE_WIDTH, 32], dtype=pl.FP32, value=2.0)
            rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
            predicate = pl.tile.cmps(magnitude, 2.5, cmp_type=4)
            level = pl.tile.full([INDEX_QUERY_SCALE_WIDTH, 32], dtype=pl.FP32, value=3.0)
            rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
            predicate = pl.tile.cmps(magnitude, 3.5, cmp_type=5)
            level = pl.tile.full([INDEX_QUERY_SCALE_WIDTH, 32], dtype=pl.FP32, value=4.0)
            rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
            predicate = pl.tile.cmps(magnitude, 5.0, cmp_type=4)
            level = pl.tile.full([INDEX_QUERY_SCALE_WIDTH, 32], dtype=pl.FP32, value=6.0)
            rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
            sign = pl.ands(pl.reinterpret_view(normalized, pl.INT32), -2147483648)
            signed_bits = pl.or_(pl.reinterpret_view(rounded, pl.INT32), sign)
            exact = pl.reinterpret_view(signed_bits, pl.FP32)
            packed_input = pl.reshape(exact, [1, INDEX_QUERY_WIDTH])
            exact_bf16 = pl.cast(packed_input, pl.BF16, mode="rint")
            packed_fp4 = pl.cast(exact_bf16, pl.FP4E2M1X2, mode="rint")
            payload = pl.reinterpret_view(packed_fp4, pl.UINT8)
            pl.store(payload, [t, 0], output)
            codes = pl.cast(pl.reshape(exponent, [1, INDEX_QUERY_SCALE_WIDTH]), pl.UINT8)
            pl.store(pl.reinterpret_view(codes, pl.FP8E8M0), [t, 0], output_scale)
    return quant_tid


def _make_materialize_index_topk():
    """Keep inferred row capacities independent across selector instances."""
    @pl.jit.inline
    def materialize(
        scores: pl.Tensor,
        block_table: pl.Tensor,
        request_ids: pl.Tensor,
        output: pl.Tensor,
        token: pl.Scalar[pl.INDEX],
        width: pl.Scalar[pl.INDEX],
        cutoff: pl.Scalar[pl.FP32],
        tie_quota: pl.Scalar[pl.INDEX],
    ):
        """Compact winners in logical order and gather physical pages from UB."""
        request = pl.cast(pl.read(request_ids, [token]), pl.INDEX)
        table_width = pl.tensor.dim(block_table, 1)
        valid_pages = pl.min(table_width, TOPK_CACHED_PAGES)
        raw_pages = pl.load(block_table, [request, 0], [1, TOPK_CACHED_PAGES], valid_shape=[1, valid_pages])
        pages = pl.tile.fillpad(raw_pages, pad_value=pl.PadValue.zero)
        pl.store(pl.tile.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1), [token, 0], output)
        ones = pl.tile.full([1, 256], dtype=pl.FP32, value=1.0)
        index_tmp = pl.create_tile([1, 192], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        lanes = pl.tile.arange(0, [1, 256], dtype=pl.INT32, tmp=index_tmp)
        lane_origin = pl.tile.full([1, 256], dtype=pl.INT32, value=256)
        # Unique integer lane keys: selected=256-lane, discarded=0.
        priority_fp32 = pl.cast(pl.sub(lane_origin, lanes), pl.FP32)
        tmp = pl.tile.create([1, 256], dtype=pl.UINT8)
        count = pl.cast(0, pl.INDEX)
        remaining = tie_quota
        for start in pl.range(0, width, 256):
            if count < INDEX_TOPK:
                valid = pl.min(256, width - start)
                raw = pl.load(scores, [token, start], [1, 256], valid_shape=[1, valid])
                values = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
                strict = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=4), ones, 0.0)
                ties = pl.tile.full([1, 256], dtype=pl.FP32, value=0.0)
                if remaining > 0:
                    eq = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=0), ones, 0.0)
                    eq_valid = pl.tile.cmps(lanes, pl.cast(valid, pl.INT32), cmp_type=2)
                    eq = pl.tile.select(eq_valid, eq, 0.0)
                    eq_sums64 = pl.col_sum(pl.reshape(eq, [4, 64]))
                    eq_sums8 = pl.col_sum(pl.reshape(eq_sums64, [8, 8]))
                    available = pl.cast(0, pl.INDEX)
                    for eq_lane in pl.unroll(8):
                        eq_partial = pl.cast(pl.tile.read(eq_sums8, [0, eq_lane]), pl.INT32)
                        available = available + pl.cast(eq_partial, pl.INDEX)
                    take = pl.min(available, remaining)
                    if take > 0:
                        if take == available:
                            ties = eq
                        else:
                            # Only the chunk crossing the tie quota needs this boundary.
                            tie_keys = pl.mul(eq, priority_fp32)
                            tie_pairs = pl.tile.sort32(tie_keys, pl.reinterpret_view(lanes, pl.UINT32))
                            tie_pairs = pl.tile.mrgsort(tie_pairs, block_len=64)
                            tie_left = pl.tile.slice(tie_pairs, [1, 256], [0, 0])
                            tie_right = pl.tile.slice(tie_pairs, [1, 256], [0, 256])
                            tie_tmp = pl.tile.create([1, 512], dtype=pl.FP32)
                            tie_merged = pl.tile.mrgsort(tie_left, tie_right, tmp=tie_tmp)
                            equal_keys = pl.tile.gather_mask(tie_merged, mask_pattern=pl.tile.MaskPattern.P0101)
                            last_key = pl.cast(pl.tile.read(equal_keys, [0, take - 1]), pl.INT32)
                            boundary = pl.cast(256, pl.INT32) - last_key
                            ties = pl.tile.select(pl.tile.cmps(lanes, boundary, cmp_type=3), eq, 0.0)
                        remaining = remaining - take
                keep = pl.maximum(strict, ties)
                keep_valid = pl.tile.set_validshape(keep, 1, valid)
                keep_padded = pl.tile.fillpad(keep_valid, pad_value=pl.PadValue.zero)
                keep_sums64 = pl.col_sum(pl.reshape(keep_padded, [4, 64]))
                keep_sums8 = pl.col_sum(pl.reshape(keep_sums64, [8, 8]))
                keep_count = pl.cast(0, pl.INDEX)
                for keep_lane in pl.unroll(8):
                    keep_partial = pl.cast(pl.tile.read(keep_sums8, [0, keep_lane]), pl.INT32)
                    keep_count = keep_count + pl.cast(keep_partial, pl.INDEX)
                selected_count = pl.min(keep_count, INDEX_TOPK - count)
                if selected_count > 0:
                    order_keys = pl.mul(keep_padded, priority_fp32)
                    order_pairs = pl.tile.sort32(order_keys, pl.reinterpret_view(lanes, pl.UINT32))
                    order_pairs = pl.tile.mrgsort(order_pairs, block_len=64)
                    order_left = pl.tile.slice(order_pairs, [1, 256], [0, 0])
                    order_right = pl.tile.slice(order_pairs, [1, 256], [0, 256])
                    order_tmp = pl.tile.create([1, 512], dtype=pl.FP32)
                    order_merged = pl.tile.mrgsort(order_left, order_right, tmp=order_tmp)
                    order_keys = pl.tile.gather_mask(order_merged, mask_pattern=pl.tile.MaskPattern.P0101)
                    offsets_i32 = pl.sub(lane_origin, pl.cast(order_keys, pl.INT32))
                    compact_valid = pl.tile.set_validshape(offsets_i32, 1, selected_count)
                    compact_safe = pl.tile.fillpad(compact_valid, pad_value=pl.PadValue.zero)
                    logical = pl.add(compact_safe, pl.cast(start, pl.INT32))
                    page_indices = pl.tile.shrs(logical, 7)
                    page_ids = pl.tile.gather(pages, page_indices, tmp)
                    offsets = pl.sub(logical, pl.mul(page_indices, 128))
                    physical = pl.add(pl.mul(page_ids, 128), offsets)
                    pl.store(pl.tile.set_validshape(physical, 1, selected_count), [token, count], output)
                    count = count + selected_count
        return output

    return materialize


_materialize_index_topk = _make_materialize_index_topk()

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
def select_candidate_block_pool(
    index_scores: pl.Tensor[[T_DYN, INDEX_SCORE_POSITIONS_DYN], pl.FP32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    scores_ready: pl.Scalar[pl.TASK_ID],
):
    """Publish sorted logical candidate block IDs and per-query counts."""
    tokens = pl.tensor.dim(index_scores, 0)
    positions = pl.tensor.dim(index_scores, 1)
    capacity = pl.tensor.dim(pool_block_ids, 1)
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > tokens:
        active_tokens = pl.cast(tokens, pl.INDEX)
    block_count = (positions + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
    if block_count <= pl.min(capacity, CANDIDATE_TOPK_BLOCKS):
        with pl.spmd(tokens, name_hint="pool_all") as all_tid:
            token = pl.tile.get_block_idx()
            lanes = pl.tile.arange(0, [1, CANDIDATE_TOPK_BLOCKS], dtype=pl.INT32)
            ids = pl.tile.full([1, CANDIDATE_TOPK_BLOCKS], dtype=pl.INT32, value=-1)
            if token < active_tokens:
                visible = pl.min(pl.max(pl.read(compressed_lens, [token]), 0), positions)
                visible_blocks = (visible + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
                valid = pl.tile.cmps(lanes, pl.cast(visible_blocks, pl.INT32), cmp_type=2)
                ids = pl.tile.select(valid, lanes, ids)
            pl.store(pl.tile.set_validshape(ids, 1, capacity), [token, 0], pool_block_ids)
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
        pool_tid = pl.yield_(selected_tid)
    # Publish cache-line-sharing counts from one worker after the ID rows.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="pool_count", deps=[pool_tid]) as counts_tid:
        for publish_token in pl.range(tokens):
            published_count = pl.cast(0, pl.INT32)
            if publish_token < active_tokens:
                publish_visible = pl.min(pl.max(pl.read(compressed_lens, [publish_token]), 0), positions)
                publish_blocks = (publish_visible + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
                published_count = pl.cast(pl.min(pl.min(publish_blocks, CANDIDATE_TOPK_BLOCKS), capacity), pl.INT32)
            pl.write(pool_block_counts, [publish_token], published_count)
    return counts_tid


@pl.jit.inline
def _index_key_rope_block(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    block: pl.Scalar[pl.INT32],
):
    t = block
    base = 0
    output[t : t + 1, base : base + (INDEX_DIM - ROPE_DIM)] = x[t : t + 1, base : base + (INDEX_DIM - ROPE_DIM)]
    tail = pl.cast(x[t : t + 1, base + (INDEX_DIM - ROPE_DIM) : base + INDEX_DIM], pl.FP32)
    even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
    odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
    c = cos[t : t + 1, :]
    s = pl.mul(sin[t : t + 1, :], 1.0)
    re = pl.sub(pl.mul(even, c), pl.mul(odd, s))
    im = pl.add(pl.mul(even, s), pl.mul(odd, c))
    rotated = pl.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
    rotated = pl.tensor.scatter(re, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
    rotated = pl.tensor.scatter(im, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
    output[t : t + 1, base + (INDEX_DIM - ROPE_DIM) : base + INDEX_DIM] = pl.cast(
        rotated, pl.BF16, mode="rint"
    )
    return output


def make_fp4_publish(width, group, scale_dtype, cache_dim, name_hint="pub_idx"):
    """Build an FP4 cache-publish kernel for one scale dtype.

    The BF16 (compressed) and E8M0 (index) encodings store different scale
    dtypes, so the two variants must be separate traced functions: PyPTO traces
    both arms of a Python ``if`` on a closure value and rejects the resulting
    dtype-divergent assignment.  This factory only dispatches between them.
    """
    if scale_dtype == pl.FP8E8M0:
        return _make_fp4_publish_e8m0(width, group, cache_dim, name_hint)
    return _make_fp4_publish_bf16(width, group, cache_dim, name_hint)


def _make_fp4_publish_bf16(width, group, cache_dim, name_hint):
    padded_width = max(width, 32 * group)

    @pl.jit.inline(auto_scope=False)
    def publish(
        source: pl.Tensor[[T_DYN, width], pl.BF16],
        slots: pl.Tensor[[T_DYN], pl.INT64],
        cache: pl.Tensor[[cache_dim, 128, 1, width // 2], pl.UINT8],
        scales: pl.Tensor[[cache_dim, 128, 1, width // group], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
        cache_ready: pl.Scalar[pl.TASK_ID],
        kv_ready: pl.Scalar[pl.TASK_ID],
    ):
        cache_rows = pl.tensor.dim(cache, 0) * 128
        flat = pl.reshape(cache, [cache_rows, width // 2])
        flat_scale = pl.reshape(scales, [cache_rows, width // group])
        # Each physical cache page has one publishing worker.
        workers = pl.min(num_tokens, 4)
        with pl.spmd(workers, name_hint=name_hint, deps=[cache_ready, kv_ready]) as published:
            worker = pl.tile.get_block_idx()
            for t in pl.range(num_tokens):
                slot = pl.read(slots, [t])
                if slot >= 0 and (slot // 128) % workers == worker:
                    physical = pl.cast(slot, pl.INDEX)
                    raw = pl.load(source, [t, 0], [1, padded_width], valid_shape=[1, width])
                    raw = pl.set_validshape(pl.fillpad(raw, pad_value=pl.PadValue.zero), 1, padded_width)
                    values = pl.reshape(pl.cast(raw, pl.FP32), [padded_width // group, group])
                    temporary = pl.tile.create([padded_width // group, group], dtype=pl.FP32)
                    amax = pl.row_max(pl.abs(values), tmp_tile=temporary)
                    factors_raw = pl.maximum(pl.mul(amax, 1.0 / 6.0), 0.001953125)
                    stored_scale = pl.cast(factors_raw, pl.BF16, mode="rint")
                    factors = pl.cast(stored_scale, pl.FP32)
                    normalized = pl.minimum(pl.maximum(pl.row_expand_div(values, factors), -6.0), 6.0)
                    magnitude = pl.abs(normalized)
                    rounded = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=0.0)
                    select_tmp = pl.tile.create([1, 16], dtype=pl.UINT32)
                    predicate = pl.tile.cmps(magnitude, 0.25, cmp_type=4)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=0.5)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    predicate = pl.tile.cmps(magnitude, 0.75, cmp_type=5)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=1.0)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    predicate = pl.tile.cmps(magnitude, 1.25, cmp_type=4)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=1.5)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    predicate = pl.tile.cmps(magnitude, 1.75, cmp_type=5)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=2.0)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    predicate = pl.tile.cmps(magnitude, 2.5, cmp_type=4)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=3.0)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    predicate = pl.tile.cmps(magnitude, 3.5, cmp_type=5)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=4.0)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    predicate = pl.tile.cmps(magnitude, 5.0, cmp_type=4)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=6.0)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    # Convert the exact RNE levels to E2M1 codes, then pack adjacent lanes.
                    codes = pl.add(pl.minimum(pl.mul(rounded, 2.0), 4.0),
                                   pl.minimum(pl.maximum(pl.sub(rounded, 2.0), 0.0), 3.0))
                    sign = pl.ands(pl.shrs(pl.reinterpret_view(normalized, pl.INT32), 28), 8)
                    codes = pl.reshape(pl.or_(pl.cast(codes, pl.INT32), sign), [1, padded_width])
                    low = pl.tile.gather_mask(
                        codes, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.INT32
                    )
                    high = pl.tile.gather_mask(
                        codes, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                    )
                    packed = pl.add(low, pl.mul(high, 16))
                    signed_bytes = pl.sub(packed, pl.mul(pl.shrs(packed, 7), 256))
                    payload_bytes = pl.reinterpret_view(pl.cast(signed_bytes, pl.INT8), pl.UINT8)
                    payload = pl.set_validshape(payload_bytes, 1, width // 2)
                    encoded = pl.set_validshape(pl.reshape(stored_scale, [1, padded_width // group]),
                                                1, width // group)
                    pl.store(payload, [physical, 0], flat)
                    pl.store(encoded, [physical, 0], flat_scale)
        return published

    return publish


def _make_fp4_publish_e8m0(width, group, cache_dim, name_hint):
    token_tile = 8
    padded_width = token_tile * width

    @pl.jit.inline(auto_scope=False)
    def publish(
        source: pl.Tensor[[T_DYN, width], pl.BF16],
        slots: pl.Tensor[[T_DYN], pl.INT64],
        cache: pl.Tensor[[cache_dim, 128, 1, width // 2], pl.UINT8],
        scales: pl.Tensor[[cache_dim, 128, 1, width // group], pl.FP8E8M0],
        num_tokens: pl.Scalar[pl.INT32],
        cache_ready: pl.Scalar[pl.TASK_ID],
        kv_ready: pl.Scalar[pl.TASK_ID],
    ):
        cache_rows = pl.tensor.dim(cache, 0) * 128
        flat = pl.reshape(cache, [cache_rows, width // 2])
        flat_scale = pl.reshape(scales, [cache_rows, width // group])
        # Scale rows may share a cache line, so one worker publishes them in order.
        with pl.at(
            level=pl.Level.CORE_GROUP, name_hint=name_hint, deps=[cache_ready, kv_ready]
        ) as published:
            for t in pl.range(0, num_tokens, token_tile):
                rows = pl.min(token_tile, num_tokens - t)
                raw = pl.load(source, [t, 0], [token_tile, width], valid_shape=[rows, width])
                raw = pl.set_validshape(pl.fillpad(raw, pad_value=pl.PadValue.zero), token_tile, width)
                values = pl.reshape(pl.cast(raw, pl.FP32), [padded_width // group, group])
                temporary = pl.tile.create([padded_width // group, group], dtype=pl.FP32)
                amax = pl.row_max(pl.abs(values), tmp_tile=temporary)
                bounded = pl.maximum(amax, 7.052966104933725e-38)
                bits = pl.reinterpret_view(pl.mul(bounded, 1.0 / 6.0), pl.INT32)
                exponent = pl.shrs(pl.add(bits, 8388607), 23)
                factors = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
                # Raw codes span 0..255; folding to a signed byte would saturate.
                stored_codes = exponent
                normalized = pl.minimum(pl.maximum(pl.row_expand_div(values, factors), -6.0), 6.0)
                magnitude = pl.abs(normalized)
                rounded = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=0.0)
                select_tmp = pl.tile.create([1, 16], dtype=pl.UINT32)
                predicate = pl.tile.cmps(magnitude, 0.25, cmp_type=4)
                level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=0.5)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                predicate = pl.tile.cmps(magnitude, 0.75, cmp_type=5)
                level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=1.0)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                predicate = pl.tile.cmps(magnitude, 1.25, cmp_type=4)
                level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=1.5)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                predicate = pl.tile.cmps(magnitude, 1.75, cmp_type=5)
                level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=2.0)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                predicate = pl.tile.cmps(magnitude, 2.5, cmp_type=4)
                level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=3.0)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                predicate = pl.tile.cmps(magnitude, 3.5, cmp_type=5)
                level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=4.0)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                predicate = pl.tile.cmps(magnitude, 5.0, cmp_type=4)
                level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=6.0)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                # Pack adjacent E2M1 codes.
                codes_base = pl.minimum(pl.mul(rounded, 2.0), 4.0)
                codes_extra = pl.minimum(pl.maximum(pl.sub(rounded, 2.0), 0.0), 3.0)
                codes = pl.add(codes_base, codes_extra)
                sign = pl.ands(pl.shrs(pl.reinterpret_view(normalized, pl.INT32), 28), 8)
                codes = pl.reshape(pl.or_(pl.cast(codes, pl.INT32), sign), [1, padded_width])
                low = pl.tile.gather_mask(codes, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.INT32)
                high = pl.tile.gather_mask(codes, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                packed = pl.add(low, pl.mul(high, 16))
                signed_bytes = pl.sub(packed, pl.mul(pl.shrs(packed, 7), 256))
                payload_bytes = pl.reinterpret_view(pl.cast(signed_bytes, pl.INT8), pl.UINT8)
                payload_rows = pl.reshape(payload_bytes, [token_tile, width // 2])
                scale_codes = pl.reshape(stored_codes, [1, padded_width // group])
                scale_lanes = pl.tile.arange(0, [1, 32], dtype=pl.INT32)
                scale_tmp = pl.tile.create([1, 128], dtype=pl.UINT8)
                for row in pl.range(rows):
                    slot = pl.read(slots, [t + row])
                    if slot >= 0:
                        physical = pl.cast(slot, pl.INDEX)
                        payload = pl.tile.slice(payload_rows, [1, width // 2], [row, 0])
                        scale_offset = pl.cast(row * (width // group), pl.INT32)
                        indices = pl.minimum(pl.add(scale_lanes, scale_offset), padded_width // group - 1)
                        selected_codes = pl.tile.gather(scale_codes, indices, scale_tmp)
                        encoded_bytes = pl.cast(selected_codes, pl.UINT8)
                        encoded = pl.reinterpret_view(encoded_bytes, pl.FP8E8M0)
                        encoded = pl.set_validshape(encoded, 1, width // group)
                        pl.store(payload, [physical, 0], flat)
                        pl.store(encoded, [physical, 0], flat_scale)
        return published

    return publish


def make_index_scores(skip_short=False, request_workers=32):
    """Reuse native MXFP4 queries and weights across mapped key pages."""
    interval_pages = 4
    workers = request_workers

    @pl.jit.inline
    def score(
        query: pl.Tensor,
        query_scale: pl.Tensor[[T_DYN, INDEX_H * _SCALE_COUNT], pl.FP8E8M0],
        weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
        index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
        index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, _SCALE_COUNT], pl.FP8E8M0],
        block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
        request_ids: pl.Tensor[[T_DYN], pl.INT32],
        lengths: pl.Tensor[[T_DYN], pl.INT32],
        output: pl.Tensor,
        num_tokens: pl.Scalar[pl.INT32],
        cache_ready: pl.Scalar[pl.TASK_ID],
        query_ready: pl.Scalar[pl.TASK_ID],
        weights_ready: pl.Scalar[pl.TASK_ID],
    ):
        width = pl.tensor.dim(output, 1)
        cache_rows = pl.tensor.dim(index_cache, 0) * _PAGE_SIZE
        cache_flat = pl.reshape(index_cache, [cache_rows, INDEX_DIM // 2])
        scale_flat = pl.reshape(index_cache_scale, [1, cache_rows * _SCALE_COUNT])
        query_rows = pl.tensor.dim(query, 0) * INDEX_H
        query_flat = pl.reshape(query, [query_rows, INDEX_DIM // 2])
        query_fp4 = pl.reinterpret_view(query_flat, pl.FP4E2M1X2)
        cache_fp4 = pl.reinterpret_view(cache_flat, pl.FP4E2M1X2)
        panels = (width + _PAGE_SIZE - 1) // _PAGE_SIZE
        intervals = (panels + interval_pages - 1) // interval_pages
        work_items = num_tokens * intervals

        with pl.spmd(
            workers, name_hint="idx_score_fp4",
            deps=[cache_ready, query_ready, weights_ready],
        ) as score_tid:
            worker = pl.tile.get_block_idx()
            for work in pl.range(worker, work_items, workers):
                token = work // intervals
                panel_begin = (work % intervals) * interval_pages
                panel_end = pl.min(panels, panel_begin + interval_pages)
                request = pl.read(request_ids, [token])
                visible = pl.min(pl.read(lengths, [token]), width)
                if skip_short:
                    if visible <= INDEX_TOPK:
                        visible = 0
                query_mat = pl.load(query_fp4, [token * INDEX_H, 0], [INDEX_H, INDEX_DIM // 2],
                                    target_memory=pl.MemorySpace.Mat)
                query_left = pl.tile.move(query_mat, pl.MemorySpace.Left)
                raw_query_scale = pl.load(query_scale, [token, 0], [1, INDEX_H * _SCALE_COUNT])
                raw_query_scale = pl.reinterpret_view(raw_query_scale, pl.UINT8)
                query_scale_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                query_scale_zz = pl.tmov_x2zz(raw_query_scale, query_scale_tmp, group_axis=1,
                                             dst_rows=INDEX_H, dst_cols=_SCALE_COUNT)
                query_scale_zz = pl.reinterpret_view(query_scale_zz, pl.FP8E8M0)
                query_scale_mat = pl.tile.move(query_scale_zz, pl.MemorySpace.Mat,
                                               blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major)
                query_scale_left = pl.tile.move(query_scale_mat, pl.MemorySpace.LeftScale)
                head_weight = pl.load(weights, [token, 0], [1, INDEX_H])
                head_weight_fp32 = pl.cast(head_weight, pl.FP32)
                head_weight = pl.mul(head_weight_fp32, _SCORE_SCALE)
                rounded_weight_bf16 = pl.cast(head_weight, pl.BF16, mode="rint")
                rounded_weight = pl.cast(rounded_weight_bf16, pl.FP32)
                rounded_weight = pl.reshape(rounded_weight, [INDEX_H, 1])
                for panel in pl.range(panel_begin, panel_end):
                    logical_begin = panel * _PAGE_SIZE
                    output_count = pl.min(_PAGE_SIZE, width - logical_begin)
                    negative_bits = pl.tile.full([1, _PAGE_SIZE], dtype=pl.INT32, value=-8388608)
                    masked = pl.reinterpret_view(negative_bits, pl.FP32)
                    if logical_begin < visible:
                        physical_page_i32 = pl.read(block_table, [pl.cast(request, pl.INDEX), panel])
                        if physical_page_i32 >= 0:
                            physical_row = pl.cast(physical_page_i32, pl.INDEX) * _PAGE_SIZE
                            keys_nk = pl.load(cache_fp4, [physical_row, 0], [_PAGE_SIZE, INDEX_DIM // 2],
                                              target_memory=pl.MemorySpace.Mat)
                            keys_mat = pl.tile.transpose_view(keys_nk)
                            keys_right = pl.tile.move(keys_mat, pl.MemorySpace.Right)
                            raw_scale = pl.load(scale_flat, [0, physical_row * _SCALE_COUNT],
                                                [1, _PAGE_SIZE * _SCALE_COUNT])
                            raw_scale = pl.reinterpret_view(raw_scale, pl.UINT8)
                            scale_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                            key_scale_zz = pl.tmov_x2zz(raw_scale, scale_tmp, group_axis=1,
                                                       dst_rows=_PAGE_SIZE, dst_cols=_SCALE_COUNT)
                            key_scale_values = pl.reinterpret_view(key_scale_zz, pl.FP8E8M0)
                            key_scale_nn = pl.tile.transpose_view(key_scale_values)
                            key_scale_mat = pl.tile.move(key_scale_nn, pl.MemorySpace.Mat,
                                                        blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major)
                            key_scale_right = pl.tile.move(key_scale_mat, pl.MemorySpace.RightScale)
                            dot_acc = pl.matmul_mx(query_left, query_scale_left, keys_right, key_scale_right)
                            dot = pl.cast(dot_acc, pl.BF16, mode="rint")
                            dot_fp32 = pl.cast(dot, pl.FP32)
                            relu = pl.maximum(dot_fp32, 0.0)
                            weighted = pl.row_expand_mul(relu, rounded_weight)
                            weighted_bf16 = pl.cast(weighted, pl.BF16, mode="rint")
                            weighted = pl.cast(weighted_bf16, pl.FP32)
                            head_sum = pl.col_sum(weighted)
                            total_bf16 = pl.cast(head_sum, pl.BF16, mode="rint")
                            total = pl.cast(total_bf16, pl.FP32)
                            valid_count = pl.min(_PAGE_SIZE, visible - logical_begin)
                            lanes = pl.tile.arange(0, [1, _PAGE_SIZE], dtype=pl.INT32)
                            in_range = pl.tile.cmps(lanes, pl.cast(valid_count, pl.INT32), cmp_type=2)
                            masked = pl.tile.select(in_range, total, masked)
                    masked_valid = pl.set_validshape(masked, 1, output_count)
                    pl.store(masked_valid, [token, logical_begin], output)
        return score_tid

    return score


score_full = make_index_scores()
score_c2a_full = make_index_scores(skip_short=True)
_score_c1a_request_group = make_index_scores(request_workers=16)
_score_c2a_request_group = make_index_scores(skip_short=True, request_workers=16)


@pl.jit.inline
def _use_request_groups(
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.Scalar[pl.INDEX],
    expected_width: pl.constexpr,
    skip_short: pl.constexpr,
):
    grouped = pl.cast(0, pl.INT32)
    middle = pl.cast(num_tokens // 2, pl.INDEX)
    if width == expected_width and middle > 0:
        if pl.read(request_ids, [middle - 1]) != pl.read(request_ids, [middle]):
            grouped = pl.cast(1, pl.INT32)
        if skip_short:
            for token in pl.range(num_tokens):
                if pl.read(lengths, [token]) <= INDEX_TOPK:
                    grouped = pl.cast(0, pl.INT32)
    return grouped


def _make_pool_index_scores(tile_n, panels_per_task, workers):
    """Gather packed candidate blocks and score them with native MXFP4."""
    blocks_per_panel = tile_n // 8
    blocks_per_task = blocks_per_panel * panels_per_task
    prefix_row = 0 if tile_n == 128 else panels_per_task

    @pl.jit.inline
    def score(
        query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM // 2], pl.UINT8],
        query_scale: pl.Tensor[[T_DYN, INDEX_H * _SCALE_COUNT], pl.FP8E8M0],
        weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
        index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
        index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, _SCALE_COUNT], pl.FP8E8M0],
        block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
        request_ids: pl.Tensor[[T_DYN], pl.INT32],
        lengths: pl.Tensor[[T_DYN], pl.INT32],
        pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
        pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
        prefix: pl.Tensor,
        output: pl.Tensor,
        num_tokens: pl.Scalar[pl.INT32],
        ready: pl.Scalar[pl.TASK_ID],
    ):
        capacity = pl.tensor.dim(pool_block_ids, 1)
        cache_pages = pl.tensor.dim(index_cache, 0)
        cache_rows = cache_pages * 128
        cache_flat = pl.reshape(index_cache, [cache_rows, INDEX_DIM // 2])
        scale_flat = pl.reshape(index_cache_scale, [cache_rows // 8, 32])
        query_flat = pl.reshape(query, [pl.tensor.dim(query, 0) * INDEX_H, INDEX_DIM // 2])
        query_fp4 = pl.reinterpret_view(query_flat, pl.FP4E2M1X2)
        tokens = pl.tensor.dim(query, 0)
        gathered_rows = ((capacity + blocks_per_panel - 1) // blocks_per_panel) * tile_n
        payload = pl.create_tensor([tokens * gathered_rows, INDEX_DIM // 2], dtype=pl.UINT8)
        gathered_scale = pl.create_tensor([1, tokens * gathered_rows * _SCALE_COUNT], dtype=pl.UINT8)
        gathered_valid = pl.create_tensor([tokens, gathered_rows], dtype=pl.INT32)
        gathered_fp4 = pl.reinterpret_view(payload, pl.FP4E2M1X2)
        table_width = pl.tensor.dim(block_table, 1)
        request_count = pl.tensor.dim(block_table, 0)
        with pl.spmd(workers, name_hint="idx_gather_fp4", deps=[ready]) as gather_tid:
            worker = pl.tile.get_block_idx()
            work_items = pl.read(prefix, [prefix_row, num_tokens])
            for work in pl.range(worker, work_items, workers):
                lower = pl.cast(0, pl.INDEX)
                upper = pl.cast(num_tokens, pl.INDEX)
                # Bound the INT32 token-prefix search to 31 steps.
                for _ in pl.range(32):
                    if lower + 1 < upper:
                        middle = (lower + upper) // 2
                        if pl.read(prefix, [prefix_row, middle]) <= work:
                            lower = middle
                        else:
                            upper = middle
                token = lower
                macro = work - pl.read(prefix, [prefix_row, token])
                first_slot = macro * blocks_per_task
                count = pl.max(0, pl.min(pl.read(pool_block_counts, [token]), capacity))
                request = pl.read(request_ids, [token])
                visible = pl.max(0, pl.read(lengths, [token]))
                for panel in pl.unroll(panels_per_task):
                    slot_begin = first_slot + panel * blocks_per_panel
                    if slot_begin < count:
                        packed = pl.tile.full([tile_n, INDEX_DIM // 2], dtype=pl.INT8, value=0)
                        raw_scale = pl.tile.full([1, tile_n * _SCALE_COUNT], dtype=pl.INT8, value=127)
                        valid_lanes = pl.tile.full([1, tile_n], dtype=pl.INT32, value=0)
                        for block in pl.range(blocks_per_panel):
                            slot = slot_begin + block
                            if slot < count:
                                logical_block = pl.read(pool_block_ids, [token, slot])
                                if logical_block >= 0 and logical_block < (visible + 7) // 8:
                                    logical_page = logical_block // 16
                                    if request >= 0 and request < request_count and logical_page < table_width:
                                        page = pl.read(block_table, [pl.cast(request, pl.INDEX), logical_page])
                                        if page >= 0 and page < cache_pages:
                                            physical_row = pl.cast(page, pl.INDEX) * 128 + (logical_block % 16) * 8
                                            block_raw = pl.load(cache_flat, [physical_row, 0], [8, INDEX_DIM // 2])
                                            block_payload = pl.reinterpret_view(block_raw, pl.INT8)
                                            packed = pl.tile.assemble(packed, block_payload, [block * 8, 0])
                                            scale_raw = pl.load(scale_flat, [physical_row // 8, 0], [1, 32])
                                            block_scale_bytes = pl.reinterpret_view(scale_raw, pl.UINT8)
                                            scales = pl.reinterpret_view(block_scale_bytes, pl.INT8)
                                            raw_scale = pl.tile.assemble(raw_scale, scales, [0, block * 32])
                                            lanes = pl.tile.arange(0, [1, 8], dtype=pl.INT32)
                                            remaining = pl.cast(pl.min(8, visible - logical_block * 8), pl.INT32)
                                            ones = pl.tile.full([1, 8], dtype=pl.INT32, value=1)
                                            valid = pl.tile.select(pl.tile.cmps(lanes, remaining, cmp_type=2), ones, 0)
                                            valid_lanes = pl.tile.assemble(valid_lanes, valid, [0, block * 8])
                        row_offset = token * gathered_rows + slot_begin * 8
                        packed_u8 = pl.reinterpret_view(packed, pl.UINT8)
                        pl.store(packed_u8, [row_offset, 0], payload)
                        scale_u8 = pl.reinterpret_view(raw_scale, pl.UINT8)
                        pl.store(scale_u8, [0, row_offset * _SCALE_COUNT], gathered_scale)
                        pl.store(valid_lanes, [token, slot_begin * 8], gathered_valid)
        with pl.spmd(workers, name_hint="idx_score_pool_fp4", deps=[gather_tid],
                     optimizations=[pl.cross_core_slot(slot_num=2)]) as score_tid:
            worker = pl.tile.get_block_idx()
            work_items = pl.read(prefix, [prefix_row, num_tokens])
            for work in pl.range(worker, work_items, workers):
                lower = pl.cast(0, pl.INDEX)
                upper = pl.cast(num_tokens, pl.INDEX)
                # Bound the INT32 token-prefix search to 31 steps.
                for _ in pl.range(32):
                    if lower + 1 < upper:
                        middle = (lower + upper) // 2
                        if pl.read(prefix, [prefix_row, middle]) <= work:
                            lower = middle
                        else:
                            upper = middle
                token = lower
                macro = work - pl.read(prefix, [prefix_row, token])
                first_slot = macro * blocks_per_task
                count = pl.max(0, pl.min(pl.read(pool_block_counts, [token]), capacity))
                request = pl.read(request_ids, [token])
                visible = pl.max(0, pl.read(lengths, [token]))
                query_mat = pl.load(query_fp4, [token * INDEX_H, 0], [INDEX_H, INDEX_DIM // 2],
                                    target_memory=pl.MemorySpace.Mat)
                query_left = pl.tile.move(query_mat, target_memory=pl.MemorySpace.Left)
                query_scale_loaded = pl.load(query_scale, [token, 0], [1, INDEX_H * _SCALE_COUNT])
                query_scale_raw = pl.reinterpret_view(query_scale_loaded, pl.UINT8)
                query_scale_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                query_scale_bytes = pl.tmov_x2zz(query_scale_raw, query_scale_tmp, group_axis=1,
                                                dst_rows=INDEX_H, dst_cols=_SCALE_COUNT)
                query_scale_zz = pl.reinterpret_view(query_scale_bytes, pl.FP8E8M0)
                query_scale_mat = pl.tile.move(
                    query_scale_zz, target_memory=pl.MemorySpace.Mat,
                    blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major,
                )
                query_scale_left = pl.tile.move(query_scale_mat, target_memory=pl.MemorySpace.LeftScale)
                for panel in pl.unroll(panels_per_task):
                    slot_begin = first_slot + panel * blocks_per_panel
                    if slot_begin < count:
                        row_offset = token * gathered_rows + slot_begin * 8
                        keys_nk = pl.load(gathered_fp4, [row_offset, 0], [tile_n, INDEX_DIM // 2],
                                          target_memory=pl.MemorySpace.Mat)
                        keys_kn = pl.tile.transpose_view(keys_nk)
                        keys_right = pl.tile.move(keys_kn, target_memory=pl.MemorySpace.Right)
                        raw_scale = pl.load(gathered_scale, [0, row_offset * _SCALE_COUNT], [1, tile_n * _SCALE_COUNT])
                        scale_tmp = pl.create_tile([1, 128], dtype=pl.UINT8)
                        scale_bytes = pl.tmov_x2zz(raw_scale, scale_tmp, group_axis=1,
                                                  dst_rows=tile_n, dst_cols=_SCALE_COUNT)
                        scale_zz = pl.reinterpret_view(scale_bytes, pl.FP8E8M0)
                        scale_nn = pl.tile.transpose_view(scale_zz)
                        dot_acc = pl.matmul_mx(query_left, query_scale_left, keys_right, scale_nn)
                        dot = pl.cast(dot_acc, pl.BF16, mode="rint")
                        valid_lanes = pl.load(gathered_valid, [token, slot_begin * 8], [1, tile_n])
                        dot_fp32 = pl.cast(dot, pl.FP32)
                        relu = pl.maximum(dot_fp32, 0.0)
                        weight_raw = pl.load(weights, [token, 0], [1, INDEX_H])
                        weight_fp32 = pl.cast(weight_raw, pl.FP32)
                        head_weight = pl.mul(weight_fp32, _SCORE_SCALE)
                        rounded_weight_bf16 = pl.cast(head_weight, pl.BF16, mode="rint")
                        rounded_weight = pl.cast(rounded_weight_bf16, pl.FP32)
                        weight_column = pl.reshape(rounded_weight, [INDEX_H, 1])
                        weighted = pl.row_expand_mul(relu, weight_column)
                        weighted_bf16 = pl.cast(weighted, pl.BF16, mode="rint")
                        weighted = pl.cast(weighted_bf16, pl.FP32)
                        head_sum = pl.col_sum(weighted)
                        total_bf16 = pl.cast(head_sum, pl.BF16, mode="rint")
                        total = pl.cast(total_bf16, pl.FP32)
                        invalid_bits = pl.tile.full([1, tile_n], dtype=pl.INT32, value=-8388608)
                        invalid = pl.reinterpret_view(invalid_bits, pl.FP32)
                        valid_mask = pl.tile.cmps(valid_lanes, 1, cmp_type=0)
                        masked = pl.tile.select(valid_mask, total, invalid)
                        valid_output = pl.min(tile_n, (capacity - slot_begin) * 8)
                        masked_valid = pl.set_validshape(masked, 1, valid_output)
                        pl.store(masked_valid, [token, slot_begin * 8], output)
        return score_tid

    return score



_pool_page = _make_pool_index_scores(128, 1, 32)
_pool_page_small = _make_pool_index_scores(128, 1, 16)
_pool_block = _make_pool_index_scores(256, 1, 32)
_pool_macro = _make_pool_index_scores(256, 2, 32)


@pl.jit.inline(auto_scope=False)
def score_reindex(
    query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM // 2], pl.UINT8],
    query_scale: pl.Tensor[[T_DYN, INDEX_H * _SCALE_COUNT], pl.FP8E8M0],
    weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, _SCALE_COUNT], pl.FP8E8M0],
    block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    lengths: pl.Tensor[[T_DYN], pl.INT32],
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    query_ready: pl.Scalar[pl.TASK_ID],
    weights_ready: pl.Scalar[pl.TASK_ID],
    pool_ready: pl.Scalar[pl.TASK_ID],
):
    """Route complete identity pools to dense scoring and sparse pools to packed FP4 scoring."""
    tokens = pl.tensor.dim(query, 0)
    capacity = pl.tensor.dim(pool_block_ids, 1)
    width = pl.tensor.dim(output, 1)
    prefix = pl.create_tensor([3, tokens + 1], dtype=pl.INT32)
    route = pl.create_tensor([1, 8], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="idx_plan_pool", deps=[pool_ready]) as plan_tid:
        small = pl.cast(0, pl.INT32)
        block = pl.cast(0, pl.INT32)
        macro = pl.cast(0, pl.INT32)
        dense = pl.cast(1, pl.INT32)
        mismatch_sums = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0)
        for kind in pl.unroll(3):
            pl.write(prefix, [kind, 0], pl.cast(0, pl.INT32))
        for token in pl.range(num_tokens):
            count = pl.max(0, pl.min(pl.read(pool_block_counts, [token]), capacity))
            visible = pl.max(0, pl.read(lengths, [token]))
            request = pl.read(request_ids, [token])
            if request < 0 or request >= pl.tensor.dim(block_table, 0):
                dense = pl.cast(0, pl.INT32)
            if count != (visible + 7) // 8 or visible > pl.tensor.dim(block_table, 1) * 128:
                dense = pl.cast(0, pl.INT32)
            if dense == 1 and count > 0:
                pool_raw = pl.load(pool_block_ids, [token, 0], [1, 2048], valid_shape=[1, capacity])
                pool = pl.tile.fillpad(pool_raw, pad_value=pl.PadValue.zero)
                lanes = pl.tile.arange(0, [1, 2048], dtype=pl.INT32)
                difference = pl.cast(pl.sub(pool, lanes), pl.FP32)
                active = pl.tile.cmps(lanes, pl.cast(count, pl.INT32), cmp_type=2)
                mismatch = pl.tile.select(active, pl.abs(difference), 0.0)
                sums64 = pl.col_sum(pl.reshape(mismatch, [32, 64]))
                mismatch_sums = pl.add(mismatch_sums, sums64)
                page_count = (visible + 127) // 128
                table_columns = pl.min(128, pl.tensor.dim(block_table, 1))
                pages_raw = pl.load(block_table, [request, 0], [1, 128], valid_shape=[1, table_columns])
                pages = pl.tile.fillpad(pages_raw, pad_value=pl.PadValue.zero)
                page_lanes = pl.tile.arange(0, [1, 128], dtype=pl.INT32)
                bad_page = pl.tile.cmps(pages, pl.cast(pl.tensor.dim(index_cache, 0), pl.INT32), cmp_type=5)
                page_errors = pl.tile.select(bad_page, pl.tile.full([1, 128], dtype=pl.FP32, value=1.0), 0.0)
                page_active = pl.tile.cmps(page_lanes, pl.cast(page_count, pl.INT32), cmp_type=2)
                page_errors = pl.tile.select(page_active, page_errors, 0.0)
                page_sums = pl.col_sum(pl.reshape(page_errors, [2, 64]))
                mismatch_sums = pl.add(mismatch_sums, page_sums)
            small = pl.cast(small + (count + 15) // 16, pl.INT32)
            block = pl.cast(block + (count + 31) // 32, pl.INT32)
            macro = pl.cast(macro + (count + 63) // 64, pl.INT32)
            pl.write(prefix, [0, token + 1], pl.cast(small, pl.INT32))
            pl.write(prefix, [1, token + 1], pl.cast(block, pl.INT32))
            pl.write(prefix, [2, token + 1], pl.cast(macro, pl.INT32))
        sums8 = pl.col_sum(pl.reshape(mismatch_sums, [8, 8]))
        for lane in pl.unroll(8):
            if pl.tile.read(sums8, [0, lane]) > 0.0:
                dense = pl.cast(0, pl.INT32)
        values = pl.tile.full([1, 8], dtype=pl.INT32, value=0)
        pl.tile.write(values, [0, 0], pl.cast(small, pl.INT32))
        pl.tile.write(values, [0, 1], pl.cast(block, pl.INT32))
        pl.tile.write(values, [0, 2], pl.cast(macro, pl.INT32))
        pl.tile.write(values, [0, 3], dense)
        pl.store(values, [0, 0], route)
    output_panels = (width + 127) // 128
    with pl.spmd(32, name_hint="idx_clear_pool", deps=[pool_ready]) as clear_tid:
        worker = pl.tile.get_block_idx()
        for work in pl.range(worker, num_tokens * output_panels, 32):
            token = work // output_panels
            begin = (work % output_panels) * 128
            invalid = pl.reinterpret_view(pl.tile.full([1, 128], dtype=pl.INT32, value=-8388608), pl.FP32)
            pl.store(pl.set_validshape(invalid, 1, pl.min(128, width - begin)), [token, begin], output)
    ready = pl.system.task_dummy(deps=[cache_ready, query_ready, weights_ready, plan_tid, clear_tid])
    small = pl.read(route, [0, 0])
    block = pl.read(route, [0, 1])
    macro = pl.read(route, [0, 2])
    wide = 0
    if 2 * block == small:
        if macro >= 32:
            wide = 2
        elif block >= 32:
            wide = 1
    dense = pl.read(route, [0, 3])
    if small == 0:
        result = ready
    elif dense == 1:
        result = score_full(
            query, query_scale, weights, index_cache, index_cache_scale,
            block_table, request_ids, lengths, output, num_tokens, ready, ready, ready,
        )
    elif wide == 2:
        result = _pool_macro(query, query_scale, weights, index_cache, index_cache_scale, block_table,
                             request_ids, lengths, pool_block_ids, pool_block_counts, prefix, output, num_tokens, ready)
    elif wide == 1:
        result = _pool_block(query, query_scale, weights, index_cache, index_cache_scale, block_table,
                             request_ids, lengths, pool_block_ids, pool_block_counts, prefix, output, num_tokens, ready)
    elif small <= 16:
        result = _pool_page_small(query, query_scale, weights, index_cache, index_cache_scale, block_table,
                                  request_ids, lengths, pool_block_ids, pool_block_counts, prefix, output, num_tokens, ready)
    else:
        result = _pool_page(query, query_scale, weights, index_cache, index_cache_scale, block_table,
                            request_ids, lengths, pool_block_ids, pool_block_counts, prefix, output, num_tokens, ready)
    return result


_TOPK_PAGE = 128
_TOPK_PAIR_WIDTH = 2 * INDEX_TOPK
_TOPK_LEAF_TILE = 2048
_TOPK_SORT_FLOOR = -3.4028234663852886e38
_TOPK_INVALID_SCORE_BITS = -8388608


def _make_index_topk_pairs(query_rows=False):
    """Keep one sort/merge algorithm for flat or query-major pair storage."""
    token_stride = 1 if query_rows else 0
    slot_row_stride = 0 if query_rows else 1
    slot_column_stride = _TOPK_PAIR_WIDTH if query_rows else 0

    @pl.jit.inline
    def sort_leaf(
        scores: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.FP32],
        pair_arena: pl.Tensor,
        token: pl.Scalar[pl.INDEX],
        logical_begin: pl.Scalar[pl.INDEX],
        valid_count: pl.Scalar[pl.INDEX],
        output_slot: pl.Scalar[pl.INDEX],
    ):
        raw = pl.load(scores, [token, logical_begin], [1, _TOPK_LEAF_TILE], valid_shape=[1, valid_count])
        padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
        values = pl.maximum(padded, pl.tile.full([1, _TOPK_LEAF_TILE], dtype=pl.FP32, value=_TOPK_SORT_FLOOR))
        index_tmp = pl.create_tile([1, 192], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        lane_indices = pl.tile.arange(0, [1, _TOPK_LEAF_TILE], dtype=pl.INT32, tmp=index_tmp)
        indices = pl.add(lane_indices, pl.cast(logical_begin, pl.INT32))
        pairs = pl.tile.sort32(values, pl.reinterpret_view(indices, pl.UINT32))
        pairs = pl.tile.mrgsort(pairs, block_len=64)
        pairs = pl.tile.mrgsort(pairs, block_len=256)
        pairs = pl.tile.mrgsort(pairs, block_len=1024)
        pl.store(
            pl.tile.slice(pairs, [1, _TOPK_PAIR_WIDTH], [0, 0]),
            [token * token_stride + output_slot * slot_row_stride, output_slot * slot_column_stride], pair_arena,
        )

    @pl.jit.inline
    def merge_rows(
        pair_arena: pl.Tensor,
        left_slot: pl.Scalar[pl.INDEX],
        right_slot: pl.Scalar[pl.INDEX],
        output_slot: pl.Scalar[pl.INDEX],
        token: pl.Scalar[pl.INDEX],
    ):
        row = token * token_stride
        left = pl.load(
            pair_arena, [row + left_slot * slot_row_stride, left_slot * slot_column_stride], [1, _TOPK_PAIR_WIDTH],
        )
        right = pl.load(
            pair_arena, [row + right_slot * slot_row_stride, right_slot * slot_column_stride], [1, _TOPK_PAIR_WIDTH],
        )
        merge_tmp = pl.tile.create([1, 2 * _TOPK_PAIR_WIDTH], dtype=pl.FP32)
        merged = pl.tile.mrgsort(left, right, tmp=merge_tmp)
        pl.store(
            pl.tile.slice(merged, [1, _TOPK_PAIR_WIDTH], [0, 0]),
            [row + output_slot * slot_row_stride, output_slot * slot_column_stride], pair_arena,
        )

    return sort_leaf, merge_rows


_sort_index_topk_leaf, _merge_index_topk_rows = _make_index_topk_pairs()
_sort_c2a_topk_leaf, _merge_c2a_topk_rows = _make_index_topk_pairs(query_rows=True)


def _make_select_index_topk(leaf_workers=32, row_workers=32):
    materialize = _make_materialize_index_topk()

    @pl.jit.inline(auto_scope=False)
    def select_index_topk(
        index_scores: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.FP32],
        block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
        request_ids: pl.Tensor[[T_DYN], pl.INT32],
        output: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        scores_ready: pl.Scalar[pl.TASK_ID],
    ):
        """Select exact Top-512 indices from dynamic C1A score rows."""
        width = pl.tensor.dim(index_scores, 1)
        tokens = pl.tensor.dim(index_scores, 0)
        if width <= INDEX_TOPK:
            with pl.spmd(row_workers, name_hint="idx_topk_short", deps=[scores_ready]) as short_tid:
                worker = pl.tile.get_block_idx()
                for token in pl.range(worker, num_tokens, row_workers):
                    invalid_bits = pl.tile.full([1, 8], dtype=pl.INT32, value=_TOPK_INVALID_SCORE_BITS)
                    invalid_score = pl.tile.read(pl.reinterpret_view(invalid_bits, pl.FP32), [0, 0])
                    materialize(
                        index_scores, block_table, request_ids, output,
                        token, width, invalid_score, pl.cast(0, pl.INDEX),
                    )
            topk_tid = pl.yield_(short_tid)
        else:
            leaf_count = (width + _TOPK_LEAF_TILE - 1) // _TOPK_LEAF_TILE
            arena_rows = tokens * leaf_count
            pair_arena = pl.create_tensor([arena_rows, _TOPK_PAIR_WIDTH], dtype=pl.FP32)
            total_leaves = pl.cast(num_tokens, pl.INDEX) * leaf_count
            with pl.spmd(leaf_workers, name_hint="idx_topk_leaf", deps=[scores_ready]) as leaf_tid:
                worker = pl.tile.get_block_idx()
                for work in pl.range(worker, total_leaves, leaf_workers):
                    token = work // leaf_count
                    leaf = work - token * leaf_count
                    logical_begin = leaf * _TOPK_LEAF_TILE
                    valid_count = pl.min(_TOPK_LEAF_TILE, width - logical_begin)
                    output_slot = token * leaf_count + leaf
                    _sort_index_topk_leaf(index_scores, pair_arena, token, logical_begin, valid_count, output_slot)
            with pl.spmd(row_workers, name_hint="idx_topk_merge", deps=[leaf_tid]) as merge_tid:
                worker = pl.tile.get_block_idx()
                for token in pl.range(worker, num_tokens, row_workers):
                    base = token * leaf_count
                    count = leaf_count
                    # INT32 logical positions require at most 21 binary merge levels.
                    for _ in pl.range(21):
                        if count > 1:
                            output_count = (count + 1) // 2
                            for output_slot in pl.range(output_count):
                                left = base + output_slot * 2
                                right = left + 1
                                if right < base + count:
                                    _merge_index_topk_rows(pair_arena, left, right, base + output_slot, token)
                                else:
                                    forwarded = pl.load(pair_arena, [left, 0], [1, _TOPK_PAIR_WIDTH])
                                    pl.store(forwarded, [base + output_slot, 0], pair_arena)
                            count = output_count
                    root = pl.load(pair_arena, [base, 0], [1, _TOPK_PAIR_WIDTH])
                    root_scores = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P0101)
                    cutoff = pl.tile.read(root_scores, [0, INDEX_TOPK - 1])
                    root_ones = pl.tile.full([1, INDEX_TOPK], dtype=pl.FP32, value=1.0)
                    above_flags = pl.tile.select(pl.tile.cmps(root_scores, cutoff, cmp_type=4), root_ones, 0.0)
                    sums64 = pl.col_sum(pl.reshape(above_flags, [8, 64]))
                    sums8 = pl.col_sum(pl.reshape(sums64, [8, 8]))
                    above = pl.cast(0, pl.INDEX)
                    for lane in pl.unroll(8):
                        partial = pl.cast(pl.tile.read(sums8, [0, lane]), pl.INT32)
                        above = above + pl.cast(partial, pl.INDEX)
                    materialize(
                        index_scores, block_table, request_ids, output,
                        token, width, cutoff, INDEX_TOPK - above,
                    )
            topk_tid = pl.yield_(merge_tid)
        return topk_tid


    return select_index_topk


select_index_topk = _make_select_index_topk()
_select_c1a_request_group_topk = _make_select_index_topk(leaf_workers=16, row_workers=16)


@pl.jit.inline(auto_scope=False)
def score_select_c1a_groups(
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
):
    """Score and select two request-aligned query groups or the full score arena."""
    width = pl.tensor.dim(scores, 1)
    grouped = _use_request_groups(request_ids, lengths, num_tokens, width, C1A_REQUEST_GROUP_SCORE_WIDTH, False)
    block_count = (width + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
    # Grouped scoring leaves the caller's scores arena unwritten. Keep grouping
    # disabled when the pool selector needs those scores to rank excess blocks.
    if block_count > pl.min(pool_capacity, CANDIDATE_TOPK_BLOCKS):
        grouped = pl.cast(0, pl.INT32)
    completion = pl.array.create(2, pl.TASK_ID)
    if grouped != 0:
        group_done = pl.array.create(2, pl.TASK_ID)
        split_rows = pl.cast(num_tokens, pl.INDEX) // 2
        for group in pl.unroll(2):
            begin = group * split_rows
            group_rows = split_rows + group * (pl.cast(num_tokens, pl.INDEX) % 2)
            group_query = pl.slice(query, [group_rows, INDEX_H * INDEX_DIM // 2], [begin, 0])
            group_scale = pl.slice(query_scale, [group_rows, INDEX_H * _SCALE_COUNT], [begin, 0])
            group_weights = pl.slice(weights, [group_rows, INDEX_H], [begin, 0])
            group_requests = pl.slice(request_ids, [group_rows], [begin])
            group_lengths = pl.slice(lengths, [group_rows], [begin])
            group_scores = pl.create_tensor([group_rows, width], dtype=pl.FP32)
            score_tid = _score_c1a_request_group(
                group_query, group_scale, group_weights, index_cache, index_cache_scale,
                block_table, group_requests, group_lengths, group_scores,
                pl.cast(group_rows, pl.INT32), cache_ready, query_ready, weights_ready,
            )
            group_output = pl.slice(output, [group_rows, INDEX_TOPK], [begin, 0])
            selected = _select_c1a_request_group_topk(
                group_scores, block_table, group_requests, group_output, pl.cast(group_rows, pl.INT32), score_tid,
            )
            group_done[group] = selected
        joined = pl.system.task_dummy(deps=[group_done[0], group_done[1]])
        completion[0] = joined
        completion[1] = joined
    else:
        scores_tid = score_full(
            query, query_scale, weights, index_cache, index_cache_scale,
            block_table, request_ids, lengths, scores,
            num_tokens, cache_ready, query_ready, weights_ready,
        )
        selected = select_index_topk(scores, block_table, request_ids, output, num_tokens, scores_tid)
        completion[0] = selected
        completion[1] = scores_tid
    return completion[0], completion[1]


@pl.jit.inline
def _materialize_pool_index_topk(
    scores: pl.Tensor[[T_DYN, INDEX_POOL_POSITIONS_DYN], pl.FP32],
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    output: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    token: pl.Scalar[pl.INDEX],
    width: pl.Scalar[pl.INDEX],
    cutoff: pl.Scalar[pl.FP32],
    tie_quota: pl.Scalar[pl.INDEX],
):
    """Map compact winners through candidate block IDs before looking up pages."""
    request = pl.cast(pl.read(request_ids, [token]), pl.INDEX)
    pl.store(pl.tile.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1), [token, 0], output)
    if request >= 0 and request < pl.tensor.dim(block_table, 0) and pl.tensor.dim(block_table, 1) > 0:
        capacity = pl.tensor.dim(pool_block_ids, 1)
        raw_pool = pl.load(pool_block_ids, [token, 0], [1, 2048], valid_shape=[1, capacity])
        pool = pl.tile.fillpad(raw_pool, pad_value=pl.PadValue.zero)
        table_width = pl.tensor.dim(block_table, 1)
        valid_pages = pl.min(table_width, TOPK_CACHED_PAGES)
        raw_pages = pl.load(block_table, [request, 0], [1, TOPK_CACHED_PAGES], valid_shape=[1, valid_pages])
        pages = pl.tile.fillpad(raw_pages, pad_value=pl.PadValue.zero)
        ones = pl.tile.full([1, 256], dtype=pl.FP32, value=1.0)
        lanes = pl.tile.arange(0, [1, 256], dtype=pl.INT32)
        lane_origin = pl.tile.full([1, 256], dtype=pl.INT32, value=256)
        # Unique integer lane keys: selected=256-lane, discarded=0.
        priority_fp32 = pl.cast(pl.sub(lane_origin, lanes), pl.FP32)
        tmp = pl.tile.create([1, 256], dtype=pl.UINT8)
        count = pl.cast(0, pl.INDEX)
        remaining = tie_quota
        for start in pl.range(0, width, 256):
            if count < INDEX_TOPK:
                valid = pl.min(256, width - start)
                raw = pl.load(scores, [token, start], [1, 256], valid_shape=[1, valid])
                values = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
                strict = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=4), ones, 0.0)
                ties = pl.tile.full([1, 256], dtype=pl.FP32, value=0.0)
                if remaining > 0:
                    eq = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=0), ones, 0.0)
                    eq_valid = pl.tile.cmps(lanes, pl.cast(valid, pl.INT32), cmp_type=2)
                    eq = pl.tile.select(eq_valid, eq, 0.0)
                    eq_sums64 = pl.col_sum(pl.reshape(eq, [4, 64]))
                    eq_sums8 = pl.col_sum(pl.reshape(eq_sums64, [8, 8]))
                    available = pl.cast(0, pl.INDEX)
                    for eq_lane in pl.unroll(8):
                        eq_partial = pl.cast(pl.tile.read(eq_sums8, [0, eq_lane]), pl.INT32)
                        available = available + pl.cast(eq_partial, pl.INDEX)
                    take = pl.min(available, remaining)
                    if take > 0:
                        if take == available:
                            ties = eq
                        else:
                            # Only the chunk crossing the tie quota needs this boundary.
                            tie_keys = pl.mul(eq, priority_fp32)
                            tie_pairs = pl.tile.sort32(tie_keys, pl.reinterpret_view(lanes, pl.UINT32))
                            tie_pairs = pl.tile.mrgsort(tie_pairs, block_len=64)
                            tie_left = pl.tile.slice(tie_pairs, [1, 256], [0, 0])
                            tie_right = pl.tile.slice(tie_pairs, [1, 256], [0, 256])
                            tie_tmp = pl.tile.create([1, 512], dtype=pl.FP32)
                            tie_merged = pl.tile.mrgsort(tie_left, tie_right, tmp=tie_tmp)
                            equal_keys = pl.tile.gather_mask(tie_merged, mask_pattern=pl.tile.MaskPattern.P0101)
                            last_key = pl.cast(pl.tile.read(equal_keys, [0, take - 1]), pl.INT32)
                            boundary = pl.cast(256, pl.INT32) - last_key
                            ties = pl.tile.select(pl.tile.cmps(lanes, boundary, cmp_type=3), eq, 0.0)
                        remaining = remaining - take
                keep = pl.maximum(strict, ties)
                keep_valid = pl.tile.set_validshape(keep, 1, valid)
                keep_padded = pl.tile.fillpad(keep_valid, pad_value=pl.PadValue.zero)
                keep_sums64 = pl.col_sum(pl.reshape(keep_padded, [4, 64]))
                keep_sums8 = pl.col_sum(pl.reshape(keep_sums64, [8, 8]))
                keep_count = pl.cast(0, pl.INDEX)
                for keep_lane in pl.unroll(8):
                    keep_partial = pl.cast(pl.tile.read(keep_sums8, [0, keep_lane]), pl.INT32)
                    keep_count = keep_count + pl.cast(keep_partial, pl.INDEX)
                selected_count = pl.min(keep_count, INDEX_TOPK - count)
                if selected_count > 0:
                    order_keys = pl.mul(keep_padded, priority_fp32)
                    order_pairs = pl.tile.sort32(order_keys, pl.reinterpret_view(lanes, pl.UINT32))
                    order_pairs = pl.tile.mrgsort(order_pairs, block_len=64)
                    order_left = pl.tile.slice(order_pairs, [1, 256], [0, 0])
                    order_right = pl.tile.slice(order_pairs, [1, 256], [0, 256])
                    order_tmp = pl.tile.create([1, 512], dtype=pl.FP32)
                    order_merged = pl.tile.mrgsort(order_left, order_right, tmp=order_tmp)
                    order_keys = pl.tile.gather_mask(order_merged, mask_pattern=pl.tile.MaskPattern.P0101)
                    offsets_i32 = pl.sub(lane_origin, pl.cast(order_keys, pl.INT32))
                    compact_valid = pl.tile.set_validshape(offsets_i32, 1, selected_count)
                    compact_safe = pl.tile.fillpad(compact_valid, pad_value=pl.PadValue.zero)
                    compact = pl.add(compact_safe, pl.cast(start, pl.INT32))
                    slots = pl.tile.shrs(compact, 3)
                    blocks = pl.tile.gather(pool, slots, tmp)
                    offsets_in_block = pl.sub(compact, pl.mul(slots, 8))
                    logical = pl.add(pl.mul(blocks, 8), offsets_in_block)
                    page_indices = pl.tile.shrs(logical, 7)
                    safe_indices = pl.maximum(page_indices, 0)
                    safe_indices = pl.minimum(safe_indices, pl.cast(valid_pages - 1, pl.INT32))
                    page_ids = pl.tile.gather(pages, safe_indices, tmp)
                    offsets = pl.sub(logical, pl.mul(page_indices, _TOPK_PAGE))
                    physical = pl.add(pl.mul(page_ids, _TOPK_PAGE), offsets)
                    pl.store(pl.tile.set_validshape(physical, 1, selected_count), [token, count], output)
                    count = count + selected_count
    return output


@pl.jit.inline(auto_scope=False)
def select_pool_index_topk(
    index_scores: pl.Tensor[[T_DYN, INDEX_POOL_POSITIONS_DYN], pl.FP32],
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    lengths: pl.Tensor[[T_DYN], pl.INT32],
    output: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    scores_ready: pl.Scalar[pl.TASK_ID],
):
    """Select candidate scores and restore the original physical cache-row ABI."""
    width = pl.tensor.dim(index_scores, 1)
    tokens = pl.tensor.dim(index_scores, 0)
    leaf_count = (width + _TOPK_LEAF_TILE - 1) // _TOPK_LEAF_TILE
    arena_rows = tokens * leaf_count
    pair_arena = pl.create_tensor([arena_rows, _TOPK_PAIR_WIDTH], dtype=pl.FP32)
    with pl.spmd(32, name_hint="pool_topk_short", deps=[scores_ready]) as short_tid:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, num_tokens, 32):
            if width <= INDEX_TOPK:
                invalid_bits = pl.tile.full([1, 8], dtype=pl.INT32, value=_TOPK_INVALID_SCORE_BITS)
                invalid_score = pl.tile.read(pl.reinterpret_view(invalid_bits, pl.FP32), [0, 0])
                _materialize_pool_index_topk(
                    index_scores, pool_block_ids, block_table, request_ids, output,
                    token, width, invalid_score, pl.cast(0, pl.INDEX),
                )
    total_leaves = pl.cast(num_tokens, pl.INDEX) * leaf_count
    with pl.spmd(32, name_hint="pool_topk_leaf", deps=[scores_ready]) as leaf_tid:
        worker = pl.tile.get_block_idx()
        for work in pl.range(worker, total_leaves, 32):
            if width > INDEX_TOPK:
                token = work // leaf_count
                leaf = work - token * leaf_count
                logical_begin = leaf * _TOPK_LEAF_TILE
                valid_count = pl.min(_TOPK_LEAF_TILE, width - logical_begin)
                output_slot = token * leaf_count + leaf
                _sort_index_topk_leaf(index_scores, pair_arena, token, logical_begin, valid_count, output_slot)
    with pl.spmd(32, name_hint="pool_topk_merge", deps=[short_tid, leaf_tid]) as merge_tid:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, num_tokens, 32):
            if width > INDEX_TOPK:
                base = token * leaf_count
                count = leaf_count
                # INT32 logical positions require at most 21 binary merge levels.
                for _ in pl.range(21):
                    if count > 1:
                        output_count = (count + 1) // 2
                        for output_slot in pl.range(output_count):
                            left = base + output_slot * 2
                            right = left + 1
                            if right < base + count:
                                _merge_index_topk_rows(pair_arena, left, right, base + output_slot, token)
                            else:
                                forwarded = pl.load(pair_arena, [left, 0], [1, _TOPK_PAIR_WIDTH])
                                pl.store(forwarded, [base + output_slot, 0], pair_arena)
                        count = output_count
                root = pl.load(pair_arena, [base, 0], [1, _TOPK_PAIR_WIDTH])
                root_scores = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P0101)
                cutoff = pl.tile.read(root_scores, [0, INDEX_TOPK - 1])
                root_ones = pl.tile.full([1, INDEX_TOPK], dtype=pl.FP32, value=1.0)
                above_flags = pl.tile.select(pl.tile.cmps(root_scores, cutoff, cmp_type=4), root_ones, 0.0)
                sums64 = pl.col_sum(pl.reshape(above_flags, [8, 64]))
                sums8 = pl.col_sum(pl.reshape(sums64, [8, 8]))
                above = pl.cast(0, pl.INDEX)
                for lane in pl.unroll(8):
                    partial = pl.cast(pl.tile.read(sums8, [0, lane]), pl.INT32)
                    above = above + pl.cast(partial, pl.INDEX)
                _materialize_pool_index_topk(
                    index_scores, pool_block_ids, block_table, request_ids, output,
                    token, width, cutoff, INDEX_TOPK - above,
                )
    return merge_tid


@pl.jit.inline(auto_scope=False)
def project_index_weights(
    source: pl.Tensor,
    weight: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    result, completed = gemm_bf16_to_bf16_staged(
        source, weight, output, num_tokens, ready,
        width=D, output_width=INDEX_H, n_tile=INDEX_WEIGHT_N_TILE, k_tile=INDEX_PROJ_K_TILE,
        workers=8, name_hint='proj_idx_w', cast_name='cast_idx_w',
    )
    return completed


@pl.jit.inline(auto_scope=False)
def c1a_index(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[
        [Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
    ],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    score_width: pl.Scalar[pl.INDEX],
    entry_ready: pl.Scalar[pl.TASK_ID],
    projection_payload: pl.Tensor,
    projection_scale_backing: pl.Tensor,
    reuse_projection_input: pl.constexpr,
    query_max_workers: pl.constexpr = 2147483647,
):
    tokens = pl.tensor.dim(x, 0)
    projected = pl.create_tensor([tokens, INDEX_H * INDEX_DIM], dtype=pl.BF16)
    if reuse_projection_input:
        scales_per_token = Q_LORA // 32
        for group_start in pl.parallel(0, num_tokens, 80):
            projection_payload_m80 = pl.tensor.slice(projection_payload, [80, Q_LORA], [group_start, 0])
            scale_start = group_start * scales_per_token
            backing_size = pl.tensor.dim(projection_scale_backing, 0) * pl.tensor.dim(projection_scale_backing, 1)
            activation_scales_nd = pl.tensor.view(projection_scale_backing, [1, backing_size], layout=pl.ND)
            scale_end = scale_start + 80 * scales_per_token
            activation_scales = activation_scales_nd[0:1, scale_start:scale_end]
            activation_scale = pl.tensor.view(activation_scales, [80, scales_per_token], layout=pl.MX_A_ZZ)
            scale_group_size = (Q_LORA // 32) * INDEX_QUERY_WIDTH
            index_scale_flat = pl.tensor.view(
                index_wq_b_scale, [1, scale_group_size], layout=pl.ND,
            )
            index_scale_group = pl.tensor.view(
                index_scale_flat[0:1, 0:scale_group_size], [Q_LORA // 32, INDEX_QUERY_WIDTH], layout=pl.MX_B_NN,
            )
            available_clusters = pl.system.available_cluster_count()
            workers = pl.min(
                pl.min(INDEX_QUERY_WIDTH // INDEX_PROJ_N_TILE, query_max_workers), available_clusters,
            )
            for worker in pl.spmd(workers, name_hint="proj_idx_q", sync_start=True):
                full_a = pl.load(
                    projection_payload_m80, [0, 0], [80, Q_LORA], target_memory=pl.MemorySpace.Mat,
                )
                block_num = pl.tile.get_block_num()
                for wave in pl.range(
                    (INDEX_QUERY_WIDTH // INDEX_PROJ_N_TILE + block_num - 1) // block_num,
                ):
                    block = wave * block_num + worker
                    if block < INDEX_QUERY_WIDTH // INDEX_PROJ_N_TILE:
                        n0 = block * INDEX_PROJ_N_TILE
                        rows = pl.min(80, num_tokens - group_start)
                        a0 = full_a[:, :INDEX_PROJ_K_TILE]
                        sa0 = pl.load(
                            activation_scale, [0, 0], [80, INDEX_PROJ_K_TILE // 32],
                        )
                        b0 = pl.load(
                            index_wq_b, [0, n0], [INDEX_PROJ_K_TILE, INDEX_PROJ_N_TILE],
                        )
                        sb0 = pl.load(
                            index_scale_group, [0, n0], [INDEX_PROJ_K_TILE // 32, INDEX_PROJ_N_TILE],
                        )
                        acc = pl.matmul_mx(a0, sa0, b0, sb0)
                        for k0 in pl.pipeline(INDEX_PROJ_K_TILE, Q_LORA, INDEX_PROJ_K_TILE, stage=2):
                            a = pl.tile.slice(full_a, [80, INDEX_PROJ_K_TILE], [0, k0])
                            sa = pl.load(
                                activation_scale, [0, k0 // 32], [80, INDEX_PROJ_K_TILE // 32],
                            )
                            b = pl.load(
                                index_wq_b, [k0, n0], [INDEX_PROJ_K_TILE, INDEX_PROJ_N_TILE],
                            )
                            sb = pl.load(
                                index_scale_group, [k0 // 32, n0],
                                [INDEX_PROJ_K_TILE // 32, INDEX_PROJ_N_TILE],
                            )
                            acc = pl.matmul_mx_acc(acc, a, sa, b, sb)
                        valid_acc = pl.set_validshape(acc, rows, INDEX_PROJ_N_TILE)
                        projected = pl.store(valid_acc, [group_start, n0], projected)
    else:
        gemm_bf16_via_mxfp8_to_bf16_staged(
            qr, index_wq_b, index_wq_b_scale, projected, num_tokens,
            Q_LORA, INDEX_QUERY_WIDTH, "proj_idx_q", "quant_idx_q",
            INDEX_PROJ_N_TILE, INDEX_PROJ_K_TILE, 1, query_max_workers,
        )
    query = pl.create_tensor([tokens, INDEX_H * INDEX_DIM // 2], dtype=pl.UINT8)
    query_scale = pl.create_tensor(
        [tokens, INDEX_H * (INDEX_DIM // INDEX_CACHE_GROUP)], dtype=pl.FP8E8M0
    )
    query_tid = quantize_index_query(
        projected, cos, sin, query, query_scale, num_tokens, entry_ready,
    )
    weights = pl.create_tensor([tokens, INDEX_H], dtype=pl.BF16)
    weights_tid = project_index_weights(
        x, index_weights_proj, weights, num_tokens, entry_ready
    )
    scores = pl.create_tensor([tokens, score_width], dtype=pl.FP32)
    return query, query_scale, weights, scores, query_tid, weights_tid


# C2A index-key projection, RMSNorm, and trailing RoPE.


@pl.jit.inline(auto_scope=False)
def c2a_project_index_key(
    source: pl.Tensor,
    weight: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    result, completed = gemm_bf16_to_bf16_staged(
        source, weight, output, num_tokens, ready,
        width=HEAD_DIM, output_width=INDEX_DIM, n_tile=INDEX_PROJ_N_TILE, k_tile=INDEX_PROJ_K_TILE,
        workers=32, name_hint='proj_idx_k', cast_name='cast_idx_k',
    )
    return completed

c2a_normalize_index_key = make_norm_with_deps(INDEX_DIM, name_hint="norm_idx_k")


@pl.jit.inline(auto_scope=False)
def c2a_rotate_index_key(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    blocks = num_tokens
    workers = pl.min(blocks, 2147483647)
    with pl.spmd(workers, name_hint='rope_idx_k', deps=[ready]) as rope_tid:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, blocks, workers):
            _index_key_rope_block(x, cos, sin, output, block)
    return rope_tid


@pl.jit.inline(auto_scope=False)
def prepare_c2a_index_key(
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    latent_ready: pl.Scalar[pl.TASK_ID],
):
    """Return rotated C2A index keys and the task that publishes their value."""
    tokens = pl.tensor.dim(latent, 0)
    projected = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    projected_tid = c2a_project_index_key(latent, index_wk, projected, num_tokens, latent_ready)
    normalized = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    normalized_tid = c2a_normalize_index_key(projected, index_norm_weight, normalized, num_tokens, projected_tid)
    rotated = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    rotated_tid = c2a_rotate_index_key(
        normalized,
        compressed_rope_cos, compressed_rope_sin,
        rotated,
        num_tokens, normalized_tid,
    )
    return rotated, rotated_tid


@pl.jit.inline
def _write_c2a_short_topk(
    block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    output: pl.Tensor,
    token: pl.Scalar[pl.INDEX],
):
    """Map an unscored C2A history of at most TopK positions to physical rows."""
    length = pl.read(lengths, [token])
    request = pl.cast(pl.read(request_ids, [token]), pl.INDEX)
    lanes = pl.tile.arange(0, [1, _TOPK_PAGE], dtype=pl.INT32)
    invalid = pl.tile.full([1, _TOPK_PAGE], dtype=pl.INT32, value=-1)
    for start in pl.range(0, INDEX_TOPK, _TOPK_PAGE):
        rows = invalid
        if start < length:
            valid_count = pl.min(_TOPK_PAGE, length - start)
            page = pl.read(block_table, [request, start // _TOPK_PAGE])
            if page >= 0:
                physical = pl.add(lanes, pl.cast(page * _TOPK_PAGE + start % _TOPK_PAGE, pl.INT32))
                valid = pl.tile.cmps(lanes, pl.cast(valid_count, pl.INT32), cmp_type=2)
                rows = pl.tile.select(valid, physical, invalid)
        pl.store(rows, [token, start], output)
    return output


@pl.jit.inline
def _sort_c2a_long_topk_physical(
    output: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    token: pl.Scalar[pl.INDEX],
):
    """Keep C2A long-history output in ascending physical-row order."""
    rows = pl.load(output, [token, 0], [1, INDEX_TOPK], target_memory=pl.MemorySpace.Vec)
    keys = pl.tile.select(pl.tile.cmps(rows, -1, cmp_type=4), pl.mul(pl.cast(rows, pl.FP32), -1.0), _TOPK_SORT_FLOOR)
    ordered = pl.tile.sort32(keys, pl.reinterpret_view(rows, pl.UINT32))
    ordered = pl.tile.mrgsort(ordered, block_len=64)
    ordered = pl.tile.mrgsort(ordered, block_len=256)
    final = pl.tile.gather_mask(
        pl.tile.slice(ordered, [1, _TOPK_PAIR_WIDTH], [0, 0]),
        mask_pattern=pl.tile.MaskPattern.P1010,
        output_dtype=pl.INT32,
    )
    pl.store(final, [token, 0], output)
    return output


@pl.jit.inline(auto_scope=False)
def _c2a_topk_all_long_ready(
    block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    scores_ready: pl.Scalar[pl.TASK_ID],
):
    return scores_ready


def _make_c2a_short_topk(row_workers):
    @pl.jit.inline(auto_scope=False)
    def short_stage(
        block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
        request_ids: pl.Tensor,
        lengths: pl.Tensor,
        output: pl.Tensor,
        num_tokens: pl.Scalar[pl.INT32],
        scores_ready: pl.Scalar[pl.TASK_ID],
    ):
        with pl.spmd(row_workers, name_hint="idx_topk_short", deps=[scores_ready]) as short_tid:
            worker = pl.tile.get_block_idx()
            for token in pl.range(worker, num_tokens, row_workers):
                if pl.read(lengths, [token]) <= INDEX_TOPK:
                    _write_c2a_short_topk(block_table, request_ids, lengths, output, token)
        return short_tid

    return short_stage


def _make_select_c2a_index_topk(leaf_workers=32, row_workers=32, all_long=False):
    materialize = _make_materialize_index_topk()
    short_stage = _c2a_topk_all_long_ready if all_long else _make_c2a_short_topk(row_workers)

    @pl.jit.inline(auto_scope=False)
    def select_c2a_index_topk(
        index_scores: pl.Tensor,
        block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
        request_ids: pl.Tensor,
        lengths: pl.Tensor,
        output: pl.Tensor,
        num_tokens: pl.Scalar[pl.INT32],
        scores_ready: pl.Scalar[pl.TASK_ID],
    ):
        """Map short histories directly and return long-history TopK in physical-row order."""
        width = pl.tensor.dim(index_scores, 1)
        tokens = pl.tensor.dim(index_scores, 0)
        leaf_count = (width + _TOPK_LEAF_TILE - 1) // _TOPK_LEAF_TILE
        # Preserve a directly visible query dimension in each generated wrapper.
        pair_storage = pl.create_tensor([tokens, leaf_count * _TOPK_PAIR_WIDTH], dtype=pl.FP32)

        short_tid = short_stage(block_table, request_ids, lengths, output, num_tokens, scores_ready)

        total_leaves = pl.cast(num_tokens, pl.INDEX) * leaf_count
        with pl.spmd(leaf_workers, name_hint="idx_topk_leaf", deps=[scores_ready]) as leaf_tid:
            worker = pl.tile.get_block_idx()
            for work in pl.range(worker, total_leaves, leaf_workers):
                token = work // leaf_count
                if pl.read(lengths, [token]) > INDEX_TOPK:
                    leaf = work - token * leaf_count
                    logical_begin = leaf * _TOPK_LEAF_TILE
                    valid_count = pl.min(_TOPK_LEAF_TILE, width - logical_begin)
                    _sort_c2a_topk_leaf(index_scores, pair_storage, token, logical_begin, valid_count, leaf)

        with pl.spmd(row_workers, name_hint="idx_topk_merge", deps=[short_tid, leaf_tid]) as merge_tid:
            worker = pl.tile.get_block_idx()
            for token in pl.range(worker, num_tokens, row_workers):
                if pl.read(lengths, [token]) > INDEX_TOPK:
                    count = leaf_count
                    for _ in pl.range(21):
                        if count > 1:
                            output_count = (count + 1) // 2
                            for output_slot in pl.range(output_count):
                                left = output_slot * 2
                                right = left + 1
                                if right < count:
                                    _merge_c2a_topk_rows(pair_storage, left, right, output_slot, token)
                                else:
                                    forwarded = pl.load(
                                        pair_storage, [token, left * _TOPK_PAIR_WIDTH], [1, _TOPK_PAIR_WIDTH]
                                    )
                                    pl.store(forwarded, [token, output_slot * _TOPK_PAIR_WIDTH], pair_storage)
                            count = output_count
                    root = pl.load(pair_storage, [token, 0], [1, _TOPK_PAIR_WIDTH])
                    root_scores = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P0101)
                    cutoff = pl.tile.read(root_scores, [0, INDEX_TOPK - 1])
                    root_ones = pl.tile.full([1, INDEX_TOPK], dtype=pl.FP32, value=1.0)
                    above_flags = pl.tile.select(pl.tile.cmps(root_scores, cutoff, cmp_type=4), root_ones, 0.0)
                    sums64 = pl.col_sum(pl.reshape(above_flags, [8, 64]))
                    sums8 = pl.col_sum(pl.reshape(sums64, [8, 8]))
                    above = pl.cast(0, pl.INDEX)
                    for lane in pl.unroll(8):
                        above = above + pl.cast(pl.tile.read(sums8, [0, lane]), pl.INT32)
                    materialize(
                        index_scores, block_table, request_ids, output,
                        token, width, cutoff, INDEX_TOPK - above,
                    )

        with pl.spmd(row_workers, name_hint="idx_topk_order", deps=[merge_tid]) as order_tid:
            worker = pl.tile.get_block_idx()
            for token in pl.range(worker, num_tokens, row_workers):
                if pl.read(lengths, [token]) > INDEX_TOPK:
                    _sort_c2a_long_topk_physical(output, token)
        return order_tid

    return select_c2a_index_topk


select_c2a_index_topk = _make_select_c2a_index_topk()
_select_c2a_request_group_topk = _make_select_c2a_index_topk(leaf_workers=16, row_workers=16, all_long=True)


@pl.jit.inline(auto_scope=False)
def _score_select_c2a_groups(
    query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM // 2], pl.UINT8],
    query_scale: pl.Tensor[[T_DYN, INDEX_H * _SCALE_COUNT], pl.FP8E8M0],
    weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, _SCALE_COUNT], pl.FP8E8M0],
    block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    lengths: pl.Tensor[[T_DYN], pl.INT32],
    scores: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.FP32],
    output: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    query_ready: pl.Scalar[pl.TASK_ID],
    weights_ready: pl.Scalar[pl.TASK_ID],
):
    """Select C2A rows with independent score and TopK groups."""
    width = pl.tensor.dim(scores, 1)
    request_grouped = _use_request_groups(request_ids, lengths, num_tokens, width, C2A_REQUEST_GROUP_SCORE_WIDTH, True)
    if request_grouped != 0:
        group_done = pl.array.create(2, pl.TASK_ID)
        split_rows = pl.cast(num_tokens, pl.INDEX) // 2
        for group in pl.unroll(2):
            begin = group * split_rows
            group_rows = split_rows + group * (pl.cast(num_tokens, pl.INDEX) % 2)
            group_query = pl.slice(query, [group_rows, INDEX_H * INDEX_DIM // 2], [begin, 0])
            group_scale = pl.slice(query_scale, [group_rows, INDEX_H * _SCALE_COUNT], [begin, 0])
            group_weights = pl.slice(weights, [group_rows, INDEX_H], [begin, 0])
            group_requests = pl.slice(request_ids, [group_rows], [begin])
            group_lengths = pl.slice(lengths, [group_rows], [begin])
            group_scores = pl.create_tensor([group_rows, width], dtype=pl.FP32)
            score_tid = _score_c2a_request_group(
                group_query, group_scale, group_weights, index_cache, index_cache_scale,
                block_table, group_requests, group_lengths, group_scores,
                pl.cast(group_rows, pl.INT32), cache_ready, query_ready, weights_ready,
            )
            group_output = pl.slice(output, [group_rows, INDEX_TOPK], [begin, 0])
            selected = _select_c2a_request_group_topk(
                group_scores, block_table, group_requests, group_lengths, group_output, pl.cast(group_rows, pl.INT32), score_tid,
            )
            group_done[group] = selected
        result = pl.system.task_dummy(deps=[group_done[0], group_done[1]])
    else:
        scores_tid = score_c2a_full(
            query, query_scale, weights, index_cache, index_cache_scale,
            block_table, request_ids, lengths, scores,
            num_tokens, cache_ready, query_ready, weights_ready,
        )
        result = select_c2a_index_topk(
            scores, block_table, request_ids, lengths, output, num_tokens, scores_tid,
        )
    return result


publish_c2a_index = make_fp4_publish(INDEX_DIM, INDEX_CACHE_GROUP, pl.FP8E8M0, INDEX_BLOCKS_DYN)


@pl.jit.inline(auto_scope=False)
def decode_c2a_indexer(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
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
    index_key, index_key_tid = prepare_c2a_index_key(
        latent, index_wk, index_norm_weight,
        compressed_rope_cos, compressed_rope_sin,
        num_tokens, latent_ready,
    )
    index_publish_tid = publish_c2a_index(
        index_key, compressed_slots,
        index_cache, index_cache_scale,
        num_tokens, cache_ready, index_key_tid,
    )
    # The dense arena follows logical page-table capacity, never allocated cache pages.
    score_width = pl.tensor.dim(index_block_table, 1) * 128
    query, query_scale, raw_weights, scores, query_tid, weights_tid = c1a_index(
        x, qr,
        index_wq_b, index_wq_b_scale, index_weights_proj,
        rope_cos, rope_sin,
        num_tokens, score_width,
        cache_ready,
        index_wq_b, index_wq_b_scale, reuse_projection_input=False,
        query_max_workers=8,
    )
    return _score_select_c2a_groups(
        query, query_scale, raw_weights, index_cache, index_cache_scale,
        index_block_table, token_to_req_indices, compressed_lens, scores, topk_indices,
        num_tokens, index_publish_tid, query_tid, weights_tid,
    )


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
    projected = torch.matmul(latent.float(), wk.float()).to(latent.dtype)
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


def golden_c1a_index_scores(a, qr, index_payload, index_scale):
    """Rebuild the per-token index scores and the physical row behind every logical column."""
    from qkv_proj_rope import golden_rope_interleave as rope_interleave
    from quant import dequantize_mxfp4_cache

    x = a["x"]
    lengths = a["compressed_lens"]
    # Logical page-table score columns.
    width = a["index_block_table"].shape[-1] * 128
    heads = a["index_weights_proj"].shape[-1]
    iq = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(qr, a["index_wq_b"], a["index_wq_b_scale"])
    if iq.shape[-1] % heads:
        raise RuntimeError(f"index query width {iq.shape[-1]} is not divisible by {heads} heads")
    dim = iq.shape[-1] // heads
    rd = a["rope_cos"].shape[-1] * 2
    iq = iq.unflatten(-1, (heads, dim))
    iq = torch.cat((iq[..., :-rd], rope_interleave(iq[..., -rd:], a["rope_cos"], a["rope_sin"])), -1)
    iq = golden_index_query_fp4(iq)
    keys = dequantize_mxfp4_cache(index_payload, index_scale, 32, "e8m0").to(x.dtype).flatten(0, 1)[:, 0]
    # The device scales the projected head weight in FP32 and rounds it once to BF16.
    weight = ((x).float() @ (a["index_weights_proj"]).float()).to(torch.bfloat16).float() * (dim * heads) ** -0.5
    weight = weight.to(torch.bfloat16).float()
    scores = torch.full((x.shape[0], width), -torch.inf)
    physical = torch.full((x.shape[0], width), -1, dtype=torch.int64)
    for t in range(x.shape[0]):
        count = max(0, min(int(lengths[t]), width))
        pos = torch.arange(count)
        pages = a["index_block_table"][a["request_ids"][t].long(), pos // 128]
        valid = pages >= 0
        valid_pos = pos[valid]
        rows = pages[valid] * 128 + valid_pos % 128
        physical[t, valid_pos] = rows
        if not valid_pos.numel():
            continue
        # Device order: BF16-cast the dot, ReLU in FP32, multiply by the rounded weight,
        # BF16-cast the product, sum in FP32, BF16-cast the total.
        dots = ((iq[t]).float() @ (keys[rows.long()].T).float()).to(torch.bfloat16)
        product = dots.float().relu() * weight[t, :, None]
        values = product.to(torch.bfloat16).float().sum(0).to(torch.bfloat16).float()
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
    """Select finite TopK rows with stable ties and ascending physical order."""
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
    """Apply C2A's direct-short and common logical-earliest selection contract."""
    tokens, width = scores.shape
    output = torch.full((tokens, INDEX_TOPK), -1, dtype=torch.int32)
    for token in range(tokens):
        count = max(0, min(int(lengths[token]), width))
        if count <= INDEX_TOPK:
            output[token, :count] = physical_rows[token, :count].to(torch.int32)
            continue
        # Stable descending order makes the common logical-earliest cutoff explicit.
        logical = torch.argsort(scores[token, :count], descending=True, stable=True)[:INDEX_TOPK]
        logical = logical[torch.isfinite(scores[token, logical])]
        rows = physical_rows[token, logical]
        rows = rows[rows >= 0].sort().values
        output[token, :rows.numel()] = rows.to(torch.int32)
    return output


def golden_c2a_index(t, qr, index_cache, index_cache_scale):
    """Score and select C2A paged history with BF16 head reduction."""
    x = t["x"]
    tokens = x.shape[0]
    index_dim = t["index_norm_weight"].numel()
    index_heads = t["index_weights_proj"].shape[-1]
    index_values = dequantize_mxfp4_cache(index_cache, index_cache_scale, INDEX_CACHE_GROUP, "e8m0").flatten(0, -2)
    # Accumulate the index-query projection in sequential group-32 partials.
    index_q = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(qr, t["index_wq_b"], t["index_wq_b_scale"]).unflatten(-1, (index_heads, index_dim))
    index_q = golden_rope_tail(index_q, t["rope_cos"], t["rope_sin"])
    index_q = golden_index_query_fp4(index_q)
    head_weights = ((x).float() @ (t["index_weights_proj"]).float()).to(torch.bfloat16).float()
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
        # Round dot, weighted values, and the head sum to BF16.
        dots = ((index_q[token]).float() @ (index_values[valid_rows].to(torch.bfloat16).T).float()).to(torch.bfloat16)
        weighted = dots.float().relu() * head_weights[token, :, None]
        weighted = weighted.to(torch.bfloat16).float()
        total = weighted.sum(dim=0).to(torch.bfloat16).float()
        scores[token, valid_logical] = total
    return _select_shared_c2a_topk(scores, physical_rows, lens)


__all__ = [
    "c1a_index",
    "decode_c2a_indexer",
    "golden_c2a_index",
    "golden_c1a_index_scores",
    "golden_pool_index_scores",
    "golden_c1a_index_topk",
    "make_index_scores",
    "score_full",
    "score_reindex",
    "score_c2a_full",
    "prepare_c2a_index_key",
    "select_c2a_index_topk",
    "select_pool_index_topk",
    "select_index_topk",
]


@pl.jit.inline(auto_scope=False)
def c1a_project_index_key(
    source: pl.Tensor,
    weight: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    result, completed = gemm_bf16_to_bf16_staged(
        source, weight, output, num_tokens, ready,
        width=HEAD_DIM, output_width=INDEX_DIM, n_tile=INDEX_PROJ_N_TILE, k_tile=INDEX_PROJ_K_TILE,
        workers=8, name_hint='proj_idx_k', cast_name='cast_idx_k',
    )
    return completed


@pl.jit.inline(auto_scope=False)
def c1a_rotate_index_key(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    blocks = num_tokens
    workers = pl.min(blocks, 32)
    with pl.spmd(workers, name_hint='rope_idx_k', deps=[ready]) as rope_tid:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, blocks, workers):
            _index_key_rope_block(x, cos, sin, output, block)
    return rope_tid


def index_pool_block_capacity(score_width: int) -> int:
    """Return the aligned compact-pool capacity for one score row."""
    if score_width < 0:
        raise ValueError(f"score_width must be nonnegative, got {score_width}")
    block_size = FLASH.candidate_block_size
    blocks = max(1, (score_width + block_size - 1) // block_size)
    aligned = (blocks + 31) // 32 * 32
    return min(FLASH.candidate_topk_blocks, aligned)


def make_l2_index_scores(sparse=False):
    @pl.jit(auto_scope=False)
    def entry(
        query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM // 2], pl.UINT8],
        query_scale: pl.Tensor[[T_DYN, INDEX_H * _SCALE_COUNT], pl.FP8E8M0],
        weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
        cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
        cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, _SCALE_COUNT], pl.FP8E8M0],
        block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
        request_ids: pl.Tensor[[T_DYN], pl.INT32],
        lengths: pl.Tensor[[T_DYN], pl.INT32],
        pool_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
        pool_counts: pl.Tensor[[T_DYN], pl.INT32],
        scores: pl.Out[pl.Tensor[[T_DYN, INDEX_SCORE_POSITIONS_DYN], pl.FP32]],
    ):
        tokens = pl.cast(pl.tensor.dim(query, 0), pl.INT32)
        ready = pl.system.task_dummy(deps=[])
        if sparse:
            score_reindex(
                query, query_scale, weights, cache, cache_scale, block_table, request_ids, lengths,
                pool_ids, pool_counts, scores, tokens, ready, ready, ready, ready,
            )
        else:
            score_full(
                query, query_scale, weights, cache, cache_scale, block_table, request_ids, lengths,
                scores, tokens, ready, ready, ready,
            )
        return scores

    return entry


def build_index_score_specs(tokens, sparse, seed=71):
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(seed)
    query = torch.randint(0, 256, (tokens, INDEX_H * INDEX_DIM // 2), generator=generator, dtype=torch.uint8)
    cache = torch.randint(0, 256, (9, 128, 1, INDEX_DIM // 2), generator=generator, dtype=torch.uint8)
    qs = torch.randint(125, 129, (tokens, INDEX_H * _SCALE_COUNT), generator=generator, dtype=torch.uint8)
    ks = torch.randint(125, 129, (9, 128, 1, _SCALE_COUNT), generator=generator, dtype=torch.uint8)
    weights = torch.randint(-8, 9, (tokens, INDEX_H), generator=generator).bfloat16() / 4
    table = torch.tensor([[2, -1, 0], [5, 3, 4], [8, 7, 6]], dtype=torch.int32)
    lengths = torch.tensor([0, 1, 127, 129, 273], dtype=torch.int32)[torch.arange(tokens) % 5]
    pool_ids = (torch.arange(32).expand(tokens, -1) * 7 % 35).to(torch.int32).contiguous()
    pool_ids[:, 3] = -1
    pool_counts = (torch.arange(tokens, dtype=torch.int32) % 3 * 15 + 2).clamp(max=32)
    return [
        TensorSpec("query", query.shape, torch.uint8, init_value=query),
        TensorSpec("query_scale", qs.shape, torch.float8_e8m0fnu, init_value=qs.view(torch.float8_e8m0fnu)),
        TensorSpec("weights", weights.shape, torch.bfloat16, init_value=weights),
        TensorSpec("cache", cache.shape, torch.uint8, init_value=cache),
        TensorSpec("cache_scale", ks.shape, torch.float8_e8m0fnu, init_value=ks.view(torch.float8_e8m0fnu)),
        TensorSpec("block_table", table.shape, torch.int32, init_value=table),
        TensorSpec("request_ids", (tokens,), torch.int32, init_value=torch.arange(tokens, dtype=torch.int32) % 3),
        TensorSpec("lengths", lengths.shape, torch.int32, init_value=lengths),
        TensorSpec("pool_ids", pool_ids.shape, torch.int32, init_value=pool_ids),
        TensorSpec("pool_counts", pool_counts.shape, torch.int32, init_value=pool_counts),
        TensorSpec("scores", (tokens, 256 if sparse else 273), torch.float32),
    ]


def golden_index_scores(tensors, sparse=False):
    def decode(payload, scale):
        raw = payload.view(torch.uint8)
        codes = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2).long()
        levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
        exponents = scale.view(torch.uint8).int() - 127
        return levels[codes] * torch.exp2(exponents.float()).repeat_interleave(32, dim=-1)

    tokens = tensors["query"].shape[0]
    query = decode(tensors["query"].reshape(tokens, INDEX_H, 64),
                   tensors["query_scale"].reshape(tokens, INDEX_H, 4))
    keys = decode(tensors["cache"], tensors["cache_scale"]).reshape(-1, INDEX_DIM)
    scores = tensors["scores"]
    scores.fill_(-float("inf"))
    for token in range(tokens):
        request = int(tensors["request_ids"][token])
        for column in range(scores.shape[1]):
            logical = column
            if sparse:
                if column // 8 >= int(tensors["pool_counts"][token]):
                    continue
                block = int(tensors["pool_ids"][token, column // 8])
                if block < 0:
                    continue
                logical = block * 8 + column % 8
            if logical >= int(tensors["lengths"][token]):
                continue
            page = int(tensors["block_table"][request, logical // 128])
            if page < 0:
                continue
            dot = (query[token].double() @ keys[page * 128 + logical % 128].double()).bfloat16().float()
            weight = (tensors["weights"][token].float() / 64).bfloat16().float()
            weighted = (dot.relu() * weight).bfloat16().float()
            scores[token, column] = weighted.sum().bfloat16().float()


if __name__ == "__main__":
    import argparse
    from functools import partial
    from golden import run

    parser = argparse.ArgumentParser(description="Native MXFP4 Indexer score validation")
    parser.add_argument("-p", "--platform", choices=["a5"], default="a5")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=17)
    parser.add_argument("--mode", choices=["full", "sparse"], default="full")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--runtime-dir")
    parser.add_argument("--golden-data")
    args = parser.parse_args()
    sparse = args.mode == "sparse"
    result = run(
        fn=make_l2_index_scores(sparse),
        specs=build_index_score_specs(args.num_tokens, sparse),
        golden_fn=partial(golden_index_scores, sparse=sparse),
        compile_only=args.compile_only,
        save_data=args.save_data,
        runtime_dir=args.runtime_dir,
        golden_data=args.golden_data,
        config=dict(backend_type=args.platform, execution_mode=args.sim, device_id=args.device),
        rtol=0.0,
        atol=0.0,
    )
    if not result.passed:
        raise SystemExit(1)
