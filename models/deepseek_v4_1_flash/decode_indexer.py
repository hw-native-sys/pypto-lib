# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode indexer query preparation, key publication, scoring, and TopK."""

from gemm import gemm_bf16_to_bf16_staged, gemm_bf16_via_mxfp8_to_bf16_staged
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
_SCORE_N_TILE = 256
_SCORE_MACRO_N_TILE = 512
_KEY_DECODE_TILE = 64
_PAIR_M_TILE = INDEX_H * 2


INDEX_PROJ_N_TILE = 128
INDEX_WEIGHT_N_TILE = min(128, INDEX_H)
INDEX_PROJ_K_TILE = 256


CANDIDATE_SCORE_TILE = 1024
CANDIDATE_LEAF_TILE = 8192
CANDIDATE_SHORT_LEAF_TILE = 2048
CANDIDATE_MAX_LEAVES = (FLASH.max_position_embeddings // CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_LEAF_TILE + 1
INDEX_SCORE_POSITIONS_DYN = pl.dynamic("V41_INDEX_SCORE_POSITIONS_DYN")


@pl.jit.inline
def _index_key_decode_lut():
    """Map packed FP4 nibbles to exact FP8 carrier bits."""
    codes = pl.tile.arange(0, [1, 16], dtype=pl.INT32)
    magnitude = pl.ands(codes, 7)
    values = pl.add(pl.shls(magnitude, 2), 48)
    values = pl.tile.select(pl.tile.cmps(magnitude, 0, cmp_type=0), 0, values)
    values = pl.tile.select(pl.tile.cmps(magnitude, 1, cmp_type=0), 48, values)
    values = pl.or_(values, pl.shls(pl.ands(codes, 8), 4))
    return pl.cast(values, pl.INT16)


@pl.jit.inline(auto_scope=False)
def quantize_index_query(
    source: pl.Tensor[[T_DYN, INDEX_QUERY_WIDTH], pl.BF16],
    output: pl.Tensor,
    output_scale: pl.Tensor[[T_DYN, INDEX_QUERY_SCALE_WIDTH], pl.FP8E8M0],
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
    private_query: pl.Tensor,
    pretransposed: pl.constexpr,
):
    """Quantize group-32 queries to exact E2M1 carriers in E4M3 and E8M0 scales."""
    # A metadata-only view stays in scope for the optional store below.
    private_tokens = pl.tensor.dim(private_query, 0)
    private_flat = pl.reshape(private_query, [private_tokens * INDEX_DIM, INDEX_H])
    # Materialize the producer dependency before entering the scoped task.
    entry_deps = pl.array.create(1, pl.TASK_ID)
    entry_deps[0] = ready
    with pl.spmd(32, name_hint="idx_q_fp4_as_fp8", deps=[entry_deps[0]]) as quant_tid:
        worker = pl.tile.get_block_idx()
        for t in pl.range(worker, num_tokens, 32):
            raw = pl.load(source, [t, 0], [1, INDEX_QUERY_WIDTH])
            values = pl.reshape(pl.cast(raw, pl.FP32), [INDEX_QUERY_SCALE_WIDTH, 32])
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
            # E2M1 values use exact E4M3 carriers.
            fp8_result = pl.cast(packed_input, pl.FP8E4M3FN, mode="rint")
            pl.store(fp8_result, [t, 0], output)
            if pretransposed:
                private_heads = pl.reshape(fp8_result, [INDEX_H, INDEX_DIM])
                private_words = pl.reinterpret_view(private_heads, pl.INT8)
                private_transposed = pl.tile.transpose(private_words, 0, 1)
                private_exact = pl.reinterpret_view(private_transposed, pl.FP8E4M3FN)
                pl.store(private_exact, [t * INDEX_DIM, 0], private_flat)
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
def _index_query_rope_block(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    block: pl.Scalar[pl.INT32],
):
    t = block // INDEX_H
    h = block % INDEX_H
    base = h * INDEX_DIM
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


def _make_query_interval_index_scores(interval_pages, skip_short=False, workers=32, pretransposed=False):
    """Retain reversed-MX query operands and weights across mapped page intervals."""

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
        query_flat = pl.reshape(query, [query_rows, INDEX_DIM])
        private_rows = pl.tensor.dim(query, 0) * INDEX_DIM
        private_query_flat = pl.reshape(query, [private_rows, INDEX_H])
        panels = (width + _PAGE_SIZE - 1) // _PAGE_SIZE
        intervals = (panels + interval_pages - 1) // interval_pages
        work_items = num_tokens * intervals

        with pl.spmd(
            workers, name_hint="c1a_index_score_query_interval_mxfp8",
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
                if pretransposed:
                    query_kn = pl.load(private_query_flat, [token * INDEX_DIM, 0], [INDEX_DIM, INDEX_H])
                else:
                    query_panel = pl.load(query_flat, [token * INDEX_H, 0], [INDEX_H, INDEX_DIM])
                    query_bytes = pl.reinterpret_view(query_panel, pl.INT8)
                    query_transpose = pl.tile.transpose(query_bytes, 0, 1)
                    query_kn = pl.reinterpret_view(query_transpose, pl.FP8E4M3FN)
                query_mat = pl.tile.move(query_kn, pl.MemorySpace.Mat,
                                         blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)
                query_right = pl.tile.move(query_mat, pl.MemorySpace.Right)
                raw_query_scale = pl.load(query_scale, [token, 0], [1, INDEX_H * _SCALE_COUNT])
                raw_query_scale = pl.reinterpret_view(raw_query_scale, pl.UINT8)
                query_scale_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                query_scale_zz = pl.tmov_x2zz(raw_query_scale, query_scale_tmp, group_axis=1,
                                             dst_rows=INDEX_H, dst_cols=_SCALE_COUNT)
                query_scale_zz = pl.reinterpret_view(query_scale_zz, pl.FP8E8M0)
                query_scale_nn = pl.tile.transpose_view(query_scale_zz)
                query_scale_mat = pl.tile.move(query_scale_nn, pl.MemorySpace.Mat,
                                               blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major)
                query_scale_right = pl.tile.move(query_scale_mat, pl.MemorySpace.RightScale)
                head_weight = pl.load(weights, [token, 0], [1, INDEX_H])
                head_weight = pl.mul(pl.cast(head_weight, pl.FP32), _SCORE_SCALE)
                rounded_weight = pl.cast(pl.cast(head_weight, pl.BF16, mode="rint"), pl.FP32)
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
                            packed = pl.load(cache_flat, [physical_row, 0], [_PAGE_SIZE, INDEX_DIM // 2])
                            packed_fp4 = pl.reinterpret_view(packed, pl.FP4E2M1X2)
                            exact_values = pl.cast(packed_fp4, pl.BF16)
                            keys = pl.cast(exact_values, pl.FP8E4M3FN, mode="rint")
                            raw_scale = pl.load(scale_flat, [0, physical_row * _SCALE_COUNT],
                                                [1, _PAGE_SIZE * _SCALE_COUNT])
                            raw_scale = pl.reinterpret_view(raw_scale, pl.UINT8)
                            scale_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                            key_scale_zz = pl.tmov_x2zz(raw_scale, scale_tmp, group_axis=1,
                                                       dst_rows=_PAGE_SIZE, dst_cols=_SCALE_COUNT)
                            key_scale_values = pl.reinterpret_view(key_scale_zz, pl.FP8E8M0)
                            keys_mat = pl.tile.move(keys, pl.MemorySpace.Mat,
                                                   blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major)
                            keys_left = pl.tile.move(keys_mat, pl.MemorySpace.Left)
                            key_scale_mat = pl.tile.move(key_scale_values, pl.MemorySpace.Mat,
                                                        blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major)
                            key_scale_left = pl.tile.move(key_scale_mat, pl.MemorySpace.LeftScale)
                            reverse_dot = pl.matmul_mx(keys_left, key_scale_left, query_right, query_scale_right)
                            reverse_vec = pl.tile.move(reverse_dot, pl.MemorySpace.Vec,
                                                       blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)
                            reverse_bf16 = pl.cast(reverse_vec, pl.BF16, mode="rint")
                            reverse_words = pl.reinterpret_view(reverse_bf16, pl.INT16)
                            dot_words = pl.tile.transpose(reverse_words, 0, 1)
                            dot = pl.reinterpret_view(dot_words, pl.BF16)
                            relu = pl.maximum(pl.cast(dot, pl.FP32), 0.0)
                            weighted = pl.row_expand_mul(relu, rounded_weight)
                            weighted = pl.cast(pl.cast(weighted, pl.BF16, mode="rint"), pl.FP32)
                            total = pl.cast(pl.cast(pl.col_sum(weighted), pl.BF16, mode="rint"), pl.FP32)
                            valid_count = pl.min(_PAGE_SIZE, visible - logical_begin)
                            lanes = pl.tile.arange(0, [1, _PAGE_SIZE], dtype=pl.INT32)
                            in_range = pl.tile.cmps(lanes, pl.cast(valid_count, pl.INT32), cmp_type=2)
                            masked = pl.tile.select(in_range, total, masked)
                    pl.store(pl.set_validshape(masked, 1, output_count), [token, logical_begin], output)
        return score_tid

    return score


def _make_page_index_scores(workers, skip_short=False):
    """Share decoded K/scale locally; use M64 for two queries of one request."""
    tile_n = _PAGE_SIZE
    packed_width = INDEX_DIM // 2
    scale_count = INDEX_DIM // INDEX_CACHE_GROUP

    @pl.jit.inline
    def score(
        query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
        query_scale: pl.Tensor[[T_DYN, INDEX_H * 4], pl.FP8E8M0],
        weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
        index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
        index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0],
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
        cache_rows = pl.tensor.dim(index_cache, 0) * 128
        cache_flat = pl.reshape(index_cache, [cache_rows, INDEX_DIM // 2])
        scale_flat = pl.reshape(index_cache_scale, [1, cache_rows * scale_count])
        query_rows = pl.tensor.dim(query, 0) * INDEX_H
        query_flat = pl.reshape(query, [query_rows, INDEX_DIM])
        tiles = (width + tile_n - 1) // tile_n
        work_items = ((num_tokens + 1) // 2) * tiles

        with pl.spmd(
            workers,
            name_hint="idx_score_page",
            deps=[cache_ready, query_ready, weights_ready],
        ) as score_tid:
            worker = pl.tile.get_block_idx()
            for work in pl.range(worker, work_items, workers):
                first_token = (work // tiles) * 2
                panel = work % tiles
                logical_begin = panel * tile_n
                output_count = pl.min(tile_n, width - logical_begin)
                first_request = pl.read(request_ids, [first_token])
                second_token = pl.min(first_token + 1, num_tokens - 1)
                second_request = pl.read(request_ids, [second_token])
                for anchor_offset in pl.range(2):
                    anchor = first_token + anchor_offset
                    if anchor < num_tokens:
                        if anchor_offset == 0 or first_request != second_request:
                            anchor_request = pl.read(request_ids, [anchor])
                            maximum_visible = pl.min(pl.read(lengths, [anchor]), width)
                            if skip_short:
                                if maximum_visible <= INDEX_TOPK:
                                    maximum_visible = 0
                            if first_request == second_request:
                                second_visible = pl.min(pl.read(lengths, [second_token]), width)
                                if skip_short:
                                    if second_visible <= INDEX_TOPK:
                                        second_visible = 0
                                maximum_visible = pl.max(maximum_visible, second_visible)
                            if logical_begin < maximum_visible:
                                physical_page_i32 = pl.read(block_table,
                                    [pl.cast(anchor_request, pl.INDEX), logical_begin // 128])
                                if physical_page_i32 >= 0:
                                    physical_page = pl.cast(physical_page_i32, pl.INDEX)
                                    physical_row = physical_page * 128 + logical_begin % 128
                                    packed = pl.load(cache_flat, [physical_row, 0], [tile_n, packed_width])
                                    packed_fp4 = pl.reinterpret_view(packed, pl.FP4E2M1X2)
                                    exact_values = pl.cast(packed_fp4, pl.BF16)
                                    keys = pl.cast(exact_values, pl.FP8E4M3FN, mode="rint")
                                    raw_scale = pl.reinterpret_view(
                                        pl.load(scale_flat, [0, physical_row * scale_count],
                                                [1, tile_n * scale_count]), pl.UINT8
                                    )
                                    scale_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                                    key_scale_zz = pl.tmov_x2zz(
                                        raw_scale, scale_tmp, group_axis=1,
                                        dst_rows=tile_n, dst_cols=scale_count,
                                    )
                                    key_scale_nn = pl.tile.transpose_view(pl.reinterpret_view(key_scale_zz, pl.FP8E8M0))
                                    key_flat = pl.reshape(keys, [1, tile_n * INDEX_DIM])
                                    key_bytes = pl.reinterpret_view(key_flat, pl.INT8, shape=[tile_n, INDEX_DIM])
                                    keys_kn = pl.reinterpret_view(
                                        pl.tile.transpose(key_bytes, 0, 1), pl.FP8E4M3FN,
                                    )
                                    keys_mat = pl.tile.move(keys_kn, target_memory=pl.MemorySpace.Mat,
                                                            blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)
                                    keys_right = pl.tile.move(keys_mat, target_memory=pl.MemorySpace.Right)
                                    key_scale_mat = pl.tile.move(key_scale_nn, target_memory=pl.MemorySpace.Mat,
                                                                 blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major)
                                    key_scale_right = pl.tile.move(key_scale_mat, target_memory=pl.MemorySpace.RightScale)
                                    share_pair = pl.cast(0, pl.INT32)
                                    if second_token > first_token and first_request == second_request:
                                        share_pair = pl.cast(1, pl.INT32)
                                    if skip_short:
                                        if pl.read(lengths, [first_token]) <= INDEX_TOPK or pl.read(lengths, [second_token]) <= INDEX_TOPK:
                                            share_pair = pl.cast(0, pl.INT32)
                                    if share_pair != 0:
                                        query_pair = pl.load(query_flat, [first_token * INDEX_H, 0],
                                                             [2 * INDEX_H, INDEX_DIM], target_memory=pl.MemorySpace.Vec)
                                        pair_scale_raw = pl.reinterpret_view(
                                            pl.load(query_scale, [first_token, 0], [2, INDEX_H * scale_count]), pl.UINT8)
                                        pair_scale_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                                        pair_scale_zz = pl.reinterpret_view(pl.tmov_x2zz(
                                            pl.reshape(pair_scale_raw, [1, 2 * INDEX_H * scale_count]), pair_scale_tmp,
                                            group_axis=1, dst_rows=64, dst_cols=scale_count), pl.FP8E8M0)
                                        pair_dot = pl.matmul_mx(query_pair, pair_scale_zz, keys_right, key_scale_right)
                                        pair_dot = pl.cast(pair_dot, pl.BF16, mode="rint")
                                        for pair_query_offset in pl.unroll(2):
                                            pair_token = first_token + pair_query_offset
                                            pair_negative_bits = pl.tile.full([1, tile_n], dtype=pl.INT32, value=-8388608)
                                            pair_masked = pl.reinterpret_view(pair_negative_bits, pl.FP32)
                                            pair_visible = pl.min(pl.read(lengths, [pair_token]), width)
                                            if logical_begin < pair_visible:
                                                pair_valid_count = pl.min(tile_n, pair_visible - logical_begin)
                                                pair_head_dot = pl.tile.slice(pair_dot, [INDEX_H, tile_n], [pair_query_offset * INDEX_H, 0])
                                                pair_relu = pl.maximum(pl.cast(pair_head_dot, pl.FP32), 0.0)
                                                pair_head_weight = pl.load(weights, [pair_token, 0], [1, INDEX_H])
                                                pair_head_weight = pl.mul(pl.cast(pair_head_weight, pl.FP32), ((INDEX_DIM * INDEX_H) ** -0.5))
                                                pair_rounded_weight = pl.cast(
                                                    pl.cast(pair_head_weight, pl.BF16, mode="rint"), pl.FP32
                                                )
                                                pair_weighted = pl.row_expand_mul(
                                                    pair_relu, pl.reshape(pair_rounded_weight, [INDEX_H, 1])
                                                )
                                                pair_weighted = pl.cast(
                                                    pl.cast(pair_weighted, pl.BF16, mode="rint"), pl.FP32
                                                )
                                                pair_total = pl.cast(
                                                    pl.cast(pl.col_sum(pair_weighted), pl.BF16, mode="rint"), pl.FP32
                                                )
                                                pair_valid_count_i32 = pl.cast(pair_valid_count, pl.INT32)
                                                pair_lanes_i32 = pl.tile.arange(0, [1, tile_n], dtype=pl.INT32)
                                                pair_predicate = pl.tile.cmps(
                                                    pair_lanes_i32, pair_valid_count_i32, cmp_type=2
                                                )
                                                pair_masked = pl.tile.select(pair_predicate, pair_total, pair_masked)
                                            pl.store(
                                                pl.set_validshape(pair_masked, 1, output_count),
                                                [pair_token, logical_begin],
                                                output,
                                            )
                                    else:
                                        for query_offset in pl.range(2):
                                            token = first_token + query_offset
                                            if token < num_tokens:
                                                query_request = pl.read(request_ids, [token])
                                                if query_request == anchor_request:
                                                    negative_bits = pl.tile.full([1, tile_n], dtype=pl.INT32, value=-8388608)
                                                    masked = pl.reinterpret_view(negative_bits, pl.FP32)
                                                    visible = pl.min(pl.read(lengths, [token]), width)
                                                    if skip_short:
                                                        if visible <= INDEX_TOPK:
                                                            visible = 0
                                                    if logical_begin < visible:
                                                        valid_count = pl.min(tile_n, visible - logical_begin)
                                                        query_panel = pl.load(
                                                            query_flat, [token * INDEX_H, 0], [INDEX_H, INDEX_DIM],
                                                            target_memory=pl.MemorySpace.Vec,
                                                        )
                                                        raw_query_scale = pl.reinterpret_view(
                                                            pl.load(query_scale, [token, 0], [1, INDEX_H * scale_count]),
                                                            pl.UINT8,
                                                        )
                                                        query_scale_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                                                        query_scale_zz = pl.reinterpret_view(
                                                            pl.tmov_x2zz(raw_query_scale, query_scale_tmp, group_axis=1,
                                                                         dst_rows=INDEX_H, dst_cols=scale_count),
                                                            pl.FP8E8M0,
                                                        )
                                                        dot = pl.matmul_mx(query_panel, query_scale_zz,
                                                                               keys_right, key_scale_right)
                                                        dot = pl.cast(dot, pl.BF16, mode="rint")
                                                        relu = pl.maximum(pl.cast(dot, pl.FP32), 0.0)
                                                        head_weight = pl.load(weights, [token, 0], [1, INDEX_H])
                                                        head_weight = pl.mul(pl.cast(head_weight, pl.FP32), ((INDEX_DIM * INDEX_H) ** -0.5))
                                                        rounded_weight = pl.cast(
                                                            pl.cast(head_weight, pl.BF16, mode="rint"), pl.FP32
                                                        )
                                                        weighted = pl.row_expand_mul(
                                                            relu, pl.reshape(rounded_weight, [INDEX_H, 1])
                                                        )
                                                        weighted = pl.cast(
                                                            pl.cast(weighted, pl.BF16, mode="rint"), pl.FP32
                                                        )
                                                        total = pl.cast(
                                                            pl.cast(pl.col_sum(weighted), pl.BF16, mode="rint"), pl.FP32
                                                        )
                                                        valid_count_i32 = pl.cast(valid_count, pl.INT32)
                                                        lanes_i32 = pl.tile.arange(0, [1, tile_n], dtype=pl.INT32)
                                                        predicate = pl.tile.cmps(lanes_i32, valid_count_i32, cmp_type=2)
                                                        masked = pl.tile.select(predicate, total, masked)
                                                    pl.store(
                                                        pl.set_validshape(masked, 1, output_count),
                                                        [token, logical_begin],
                                                        output,
                                                    )
                                else:
                                    for query_offset in pl.range(2):
                                        token = first_token + query_offset
                                        if token < num_tokens:
                                            query_request = pl.read(request_ids, [token])
                                            if query_request == anchor_request:
                                                negative_bits = pl.tile.full([1, tile_n], dtype=pl.INT32, value=-8388608)
                                                masked = pl.reinterpret_view(negative_bits, pl.FP32)
                                                pl.store(
                                                    pl.set_validshape(masked, 1, output_count),
                                                    [token, logical_begin],
                                                    output,
                                                )
                            else:
                                for query_offset in pl.range(2):
                                    token = first_token + query_offset
                                    if token < num_tokens:
                                        query_request = pl.read(request_ids, [token])
                                        if query_request == anchor_request:
                                            negative_bits = pl.tile.full([1, tile_n], dtype=pl.INT32, value=-8388608)
                                            masked = pl.reinterpret_view(negative_bits, pl.FP32)
                                            pl.store(
                                                pl.set_validshape(masked, 1, output_count),
                                                [token, logical_begin],
                                                output,
                                            )
        return score_tid

    return score


def _make_request_block_index_scores(skip_short=False):
    """Reuse local K across each contiguous request run and independent query dots."""
    packed_width = INDEX_DIM // 2

    @pl.jit.inline
    def score(
        query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
        query_scale: pl.Tensor[[T_DYN, INDEX_H * _SCALE_COUNT], pl.FP8E8M0],
        weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
        index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, _PAGE_SIZE, 1, INDEX_DIM // 2], pl.UINT8],
        index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, _PAGE_SIZE, 1, _SCALE_COUNT], pl.FP8E8M0],
        block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
        request_ids: pl.Tensor[[T_DYN], pl.INT32],
        lengths: pl.Tensor[[T_DYN], pl.INT32],
        output: pl.Tensor,
        runs: pl.Tensor,
        cache_ready: pl.Scalar[pl.TASK_ID],
        query_ready: pl.Scalar[pl.TASK_ID],
        weights_ready: pl.Scalar[pl.TASK_ID],
        plan_ready: pl.Scalar[pl.TASK_ID],
    ):
        width = pl.tensor.dim(output, 1)
        cache_rows = pl.tensor.dim(index_cache, 0) * _PAGE_SIZE
        cache_flat = pl.reshape(index_cache, [cache_rows, packed_width])
        scale_flat = pl.reshape(index_cache_scale, [1, cache_rows * _SCALE_COUNT])
        query_rows = pl.tensor.dim(query, 0) * INDEX_H
        query_flat = pl.reshape(query, [query_rows, INDEX_DIM])
        macro_tiles = (width + _SCORE_MACRO_N_TILE - 1) // _SCORE_MACRO_N_TILE

        with pl.spmd(
            32,
            name_hint="idx_score_req",
            deps=[cache_ready, query_ready, weights_ready, plan_ready],
            optimizations=[pl.cross_core_slot(slot_num=2)],
        ) as score_tid:
            worker = pl.tile.get_block_idx()
            work_items = pl.read(runs, [3, 0]) * macro_tiles

            for work in pl.range(worker, work_items, 32):
                run = work // macro_tiles
                run_begin = pl.read(runs, [0, run])
                run_end = pl.read(runs, [1, run])
                query_pairs = (run_end - run_begin + 1) // 2
                macro_index = work % macro_tiles
                logical_begin = macro_index * _SCORE_MACRO_N_TILE
                first_request = pl.read(request_ids, [run_begin])
                maximum_visible = pl.read(runs, [2, run])

                for half_index in pl.unroll(2):
                    half_begin = logical_begin + half_index * _SCORE_N_TILE
                    if half_begin < maximum_visible:
                        packed = pl.tile.full([_SCORE_N_TILE, packed_width], dtype=pl.INT8, value=0)
                        raw_scale = pl.tile.full([1, _SCORE_N_TILE * _SCALE_COUNT], dtype=pl.INT8, value=127)
                        for page_offset in pl.unroll(2):
                            page_begin = half_begin + page_offset * _PAGE_SIZE
                            if page_begin < maximum_visible:
                                page_id = pl.read(
                                    block_table,
                                    [pl.cast(first_request, pl.INDEX), page_begin // _PAGE_SIZE],
                                )
                                if page_id >= 0:
                                    page_row = pl.cast(page_id, pl.INDEX) * _PAGE_SIZE
                                    page_packed = pl.reinterpret_view(
                                        pl.load(
                                            cache_flat,
                                            [page_row, 0],
                                            [_PAGE_SIZE, packed_width],
                                            target_memory=pl.MemorySpace.Vec,
                                        ),
                                        pl.INT8,
                                    )
                                    packed = pl.tile.assemble(packed, page_packed, [page_offset * _PAGE_SIZE, 0])
                                    page_scale = pl.reinterpret_view(
                                        pl.reinterpret_view(
                                            pl.load(
                                                scale_flat,
                                                [0, page_row * _SCALE_COUNT],
                                                [1, _PAGE_SIZE * _SCALE_COUNT],
                                                target_memory=pl.MemorySpace.Vec,
                                            ),
                                            pl.UINT8,
                                        ),
                                        pl.INT8,
                                    )
                                    raw_scale = pl.tile.assemble(
                                        raw_scale,
                                        page_scale,
                                        [0, page_offset * _PAGE_SIZE * _SCALE_COUNT],
                                    )

                        key_bytes = pl.tile.full([_SCORE_N_TILE, INDEX_DIM], dtype=pl.INT8, value=0)
                        # Static UB slice offsets for packed FP4 aliases.
                        for decode_part in pl.unroll(_SCORE_N_TILE // _KEY_DECODE_TILE):
                            packed_part = pl.tile.slice(
                                packed, [_KEY_DECODE_TILE, packed_width],
                                [decode_part * _KEY_DECODE_TILE, 0],
                            )
                            packed_u8 = pl.reinterpret_view(packed_part, pl.UINT8)
                            packed_fp4 = pl.reinterpret_view(packed_u8, pl.FP4E2M1X2)
                            exact_values = pl.cast(packed_fp4, pl.BF16)
                            keys = pl.cast(exact_values, pl.FP8E4M3FN, mode="rint")
                            key_flat = pl.reshape(keys, [1, _KEY_DECODE_TILE * INDEX_DIM])
                            decoded_bytes = pl.reinterpret_view(key_flat, pl.INT8, shape=[_KEY_DECODE_TILE, INDEX_DIM])
                            key_bytes = pl.tile.assemble(key_bytes, decoded_bytes, [decode_part * _KEY_DECODE_TILE, 0])
                        key_scale_tmp = pl.create_tile([1, 128], dtype=pl.UINT8)
                        key_scale_zz = pl.reinterpret_view(
                            pl.tmov_x2zz(
                                pl.reinterpret_view(raw_scale, pl.UINT8),
                                key_scale_tmp,
                                group_axis=1,
                                dst_rows=_SCORE_N_TILE,
                                dst_cols=_SCALE_COUNT,
                            ),
                            pl.FP8E8M0,
                        )
                        key_scale_nn = pl.tile.transpose_view(key_scale_zz)
                        keys_kn = pl.reinterpret_view(pl.tile.transpose(key_bytes, 0, 1), pl.FP8E4M3FN)
                        key_mat = pl.tile.move(
                            keys_kn, target_memory=pl.MemorySpace.Mat,
                            blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major,
                        )
                        key_right = pl.tile.move(key_mat, target_memory=pl.MemorySpace.Right)
                        key_scale_mat = pl.tile.move(
                            key_scale_nn, target_memory=pl.MemorySpace.Mat,
                            blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major,
                        )
                        key_scale_right = pl.tile.move(key_scale_mat, target_memory=pl.MemorySpace.RightScale)
                        for shared_pair in pl.range(query_pairs):
                            first_token = run_begin + shared_pair * 2
                            pair_rows = pl.min(2, run_end - first_token)
                            query_pair = pl.load(
                                query_flat, [first_token * INDEX_H, 0],
                                [_PAIR_M_TILE, INDEX_DIM],
                                valid_shape=[pair_rows * INDEX_H, INDEX_DIM],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            query_pair = pl.fillpad(query_pair, pad_value=pl.PadValue.zero)
                            query_pair = pl.set_validshape(query_pair, _PAIR_M_TILE, INDEX_DIM)
                            pair_scale_raw = pl.reinterpret_view(
                                pl.load(query_scale, [first_token, 0], [2, INDEX_H * _SCALE_COUNT],
                                        valid_shape=[pair_rows, INDEX_H * _SCALE_COUNT],
                                        target_memory=pl.MemorySpace.Vec),
                                pl.UINT8,
                            )
                            pair_scale_raw = pl.fillpad(pair_scale_raw, pad_value=pl.PadValue.zero)
                            pair_scale_raw = pl.set_validshape(pair_scale_raw, 2, INDEX_H * _SCALE_COUNT)
                            pair_scale_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                            pair_scale_zz = pl.reinterpret_view(
                                pl.tmov_x2zz(
                                    pl.reshape(pair_scale_raw, [1, _PAIR_M_TILE * _SCALE_COUNT]),
                                    pair_scale_tmp,
                                    group_axis=1,
                                    dst_rows=_PAIR_M_TILE,
                                    dst_cols=_SCALE_COUNT,
                                ),
                                pl.FP8E8M0,
                            )
                            query_mat = pl.tile.move(
                                query_pair,
                                target_memory=pl.MemorySpace.Mat,
                                blayout=pl.TileLayout.row_major,
                                slayout=pl.TileLayout.row_major,
                            )
                            query_left = pl.tile.move(query_mat, target_memory=pl.MemorySpace.Left)
                            pair_scale_mat = pl.tile.move(
                                pair_scale_zz,
                                target_memory=pl.MemorySpace.Mat,
                                blayout=pl.TileLayout.row_major,
                                slayout=pl.TileLayout.row_major,
                            )
                            pair_scale_left = pl.tile.move(pair_scale_mat, target_memory=pl.MemorySpace.LeftScale)

                            dot = pl.matmul_mx(query_left, pair_scale_left, key_right, key_scale_right)
                            dot_bf16 = pl.cast(dot, pl.BF16, mode="rint")
                            for panel_index in pl.unroll(2):
                                output_begin = half_begin + panel_index * _PAGE_SIZE
                                if output_begin < width:
                                    output_count = pl.min(_PAGE_SIZE, width - output_begin)
                                    for pair_offset in pl.unroll(2):
                                        token = first_token + pair_offset
                                        if token < run_end:
                                            visible = pl.min(pl.read(lengths, [token]), width)
                                            if skip_short:
                                                if visible <= INDEX_TOPK:
                                                    visible = 0
                                            negative_bits = pl.tile.full([1, _PAGE_SIZE], dtype=pl.INT32, value=-8388608)
                                            masked = pl.reinterpret_view(negative_bits, pl.FP32)
                                            if output_begin < visible:
                                                valid_count = pl.min(_PAGE_SIZE, visible - output_begin)
                                                page_id = pl.read(
                                                    block_table,
                                                    [
                                                        pl.cast(first_request, pl.INDEX),
                                                        output_begin // _PAGE_SIZE,
                                                    ],
                                                )
                                                if page_id >= 0:
                                                    head_dot = pl.tile.slice(
                                                        dot_bf16,
                                                        [INDEX_H, _PAGE_SIZE],
                                                        [pair_offset * INDEX_H, panel_index * _PAGE_SIZE],
                                                    )
                                                    relu = pl.maximum(pl.cast(head_dot, pl.FP32), 0.0)
                                                    head_weight = pl.load(
                                                        weights,
                                                        [token, 0],
                                                        [1, INDEX_H],
                                                        target_memory=pl.MemorySpace.Vec,
                                                    )
                                                    head_weight = pl.mul(pl.cast(head_weight, pl.FP32), _SCORE_SCALE)
                                                    rounded_weight = pl.cast(
                                                        pl.cast(head_weight, pl.BF16, mode="rint"), pl.FP32
                                                    )
                                                    weight_column = pl.reshape(rounded_weight, [INDEX_H, 1])
                                                    weighted = pl.row_expand_mul(relu, weight_column)
                                                    weighted = pl.cast(pl.cast(weighted, pl.BF16, mode="rint"), pl.FP32)
                                                    total = pl.cast(
                                                        pl.cast(pl.col_sum(weighted), pl.BF16, mode="rint"),
                                                        pl.FP32,
                                                    )
                                                    lanes = pl.tile.arange(0, [1, _PAGE_SIZE], dtype=pl.INT32)
                                                    in_range = pl.tile.cmps(lanes, pl.cast(valid_count, pl.INT32), cmp_type=2)
                                                    masked = pl.tile.select(in_range, total, masked)
                                            masked_valid = pl.set_validshape(masked, 1, output_count)
                                            pl.store(masked_valid, [token, output_begin], output)
                    else:
                        for shared_pair in pl.range(query_pairs):
                            first_token = run_begin + shared_pair * 2
                            for panel_index in pl.unroll(2):
                                output_begin = half_begin + panel_index * _PAGE_SIZE
                                if output_begin < width:
                                    output_count = pl.min(_PAGE_SIZE, width - output_begin)
                                    for pair_offset in pl.unroll(2):
                                        token = first_token + pair_offset
                                        if token < run_end:
                                            negative_bits = pl.tile.full([1, _PAGE_SIZE], dtype=pl.INT32, value=-8388608)
                                            negative = pl.reinterpret_view(negative_bits, pl.FP32)
                                            negative_valid = pl.set_validshape(negative, 1, output_count)
                                            pl.store(negative_valid, [token, output_begin], output)
        return score_tid

    return score


@pl.jit.inline
def use_pretransposed_index_query(
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    lengths: pl.Tensor[[T_DYN], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.Scalar[pl.INDEX],
    skip_short: pl.constexpr,
):
    eligible = pl.cast(0, pl.INT32)
    if num_tokens > 0 and num_tokens <= 16 and num_tokens % 2 == 0 and width >= 4096:
        eligible = pl.cast(1, pl.INT32)
        for token in pl.range(1, num_tokens):
            if pl.read(request_ids, [token - 1]) == pl.read(request_ids, [token]):
                eligible = pl.cast(0, pl.INT32)
        if skip_short:
            long_rows = pl.cast(0, pl.INT32)
            for token in pl.range(num_tokens):
                if pl.read(lengths, [token]) > INDEX_TOPK:
                    long_rows = long_rows + pl.cast(1, pl.INT32)
            if long_rows == 0:
                eligible = pl.cast(0, pl.INT32)
    return eligible


def make_index_scores(skip_short=False, pretransposed=False):
    """Route long request runs to shared K panels and short inputs to N128 pages."""
    general = _make_page_index_scores(workers=32, skip_short=skip_short)
    general_small = _make_page_index_scores(workers=16, skip_short=skip_short)
    request_blocks = _make_request_block_index_scores(skip_short=skip_short)
    query_intervals4 = _make_query_interval_index_scores(4, skip_short=skip_short, pretransposed=pretransposed)
    query_intervals8 = _make_query_interval_index_scores(8, skip_short=skip_short, pretransposed=pretransposed)

    @pl.jit.inline
    def score(
        query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
        query_scale: pl.Tensor[[T_DYN, INDEX_H * 4], pl.FP8E8M0],
        weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
        index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
        index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0],
        block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
        request_ids: pl.Tensor[[T_DYN], pl.INT32],
        lengths: pl.Tensor[[T_DYN], pl.INT32],
        output: pl.Tensor,
        num_tokens: pl.Scalar[pl.INT32],
        cache_ready: pl.Scalar[pl.TASK_ID],
        query_ready: pl.Scalar[pl.TASK_ID],
        weights_ready: pl.Scalar[pl.TASK_ID],
        private_query: pl.Tensor,
        private_route: pl.Scalar[pl.INT32],
    ):
        width = pl.tensor.dim(output, 1)
        panels = (width + _PAGE_SIZE - 1) // _PAGE_SIZE
        paired_panels = ((num_tokens + 1) // 2) * panels
        if pretransposed:
            query_interval_route = private_route
        else:
            query_interval_route = use_pretransposed_index_query(
                request_ids, lengths, num_tokens, width, skip_short=False,
            )
        long_rows = pl.cast(num_tokens, pl.INT32)
        if skip_short:
            long_rows = pl.cast(0, pl.INT32)
            for token in pl.range(num_tokens):
                if pl.read(lengths, [token]) > INDEX_TOPK:
                    long_rows = long_rows + pl.cast(1, pl.INT32)
        if long_rows == 0:
            result = pl.system.task_dummy(deps=[cache_ready, query_ready, weights_ready])
        elif query_interval_route != 0:
            intervals4 = (panels + 3) // 4
            intervals8 = (panels + 7) // 8
            # Upper bounds on pages per worker for 32-worker cyclic scheduling.
            max_pages4 = ((num_tokens * intervals4 + 31) // 32) * 4
            max_pages8 = ((num_tokens * intervals8 + 31) // 32) * 8
            if max_pages8 <= max_pages4:
                if pretransposed:
                    result = query_intervals8(
                        private_query, query_scale, weights, index_cache, index_cache_scale,
                        block_table, request_ids, lengths, output,
                        num_tokens, cache_ready, query_ready, weights_ready,
                    )
                else:
                    result = query_intervals8(
                        query, query_scale, weights, index_cache, index_cache_scale,
                        block_table, request_ids, lengths, output,
                        num_tokens, cache_ready, query_ready, weights_ready,
                    )
            else:
                if pretransposed:
                    result = query_intervals4(
                        private_query, query_scale, weights, index_cache, index_cache_scale,
                        block_table, request_ids, lengths, output,
                        num_tokens, cache_ready, query_ready, weights_ready,
                    )
                else:
                    result = query_intervals4(
                        query, query_scale, weights, index_cache, index_cache_scale,
                        block_table, request_ids, lengths, output,
                        num_tokens, cache_ready, query_ready, weights_ready,
                    )
        elif width >= 4096:
            capacity = pl.tensor.dim(query, 0)
            runs = pl.create_tensor([4, capacity + 1], dtype=pl.INT32)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="idx_plan_req") as plan_tid:
                run_count = pl.cast(0, pl.INT32)
                run_start = pl.cast(0, pl.INT32)
                previous_request = pl.cast(-1, pl.INT32)
                maximum_visible = pl.cast(0, pl.INT32)
                maximum_queries = pl.cast(0, pl.INT32)
                for token in pl.range(num_tokens):
                    request = pl.read(request_ids, [token])
                    if token > 0:
                        if request != previous_request:
                            pl.write(runs, [0, run_count], run_start)
                            pl.write(runs, [1, run_count], pl.cast(token, pl.INT32))
                            pl.write(runs, [2, run_count], maximum_visible)
                            maximum_queries = pl.max(maximum_queries, pl.cast(token - run_start, pl.INT32))
                            run_count = run_count + pl.cast(1, pl.INT32)
                            run_start = pl.cast(token, pl.INT32)
                            maximum_visible = pl.cast(0, pl.INT32)
                    visible = pl.cast(pl.max(0, pl.min(pl.read(lengths, [token]), width)), pl.INT32)
                    if skip_short:
                        if visible <= INDEX_TOPK:
                            visible = pl.cast(0, pl.INT32)
                    maximum_visible = pl.max(maximum_visible, visible)
                    previous_request = request
                pl.write(runs, [0, run_count], run_start)
                pl.write(runs, [1, run_count], pl.cast(num_tokens, pl.INT32))
                pl.write(runs, [2, run_count], maximum_visible)
                maximum_queries = pl.max(maximum_queries, num_tokens - run_start)
                run_count = run_count + pl.cast(1, pl.INT32)
                use_request = pl.cast(1, pl.INT32)
                # Mixed single queries and pairs have no cross-pair K reuse to
                # justify padded M64 dots. Complete pairs still use the wider tile.
                if maximum_queries <= 2:
                    if num_tokens < 2 * run_count:
                        use_request = pl.cast(0, pl.INT32)
                pl.write(runs, [3, 0], run_count)
                pl.write(runs, [3, 1], use_request)
            if pl.read(runs, [3, 1]) == 0:
                ready = pl.system.task_dummy(deps=[cache_ready, query_ready, weights_ready, plan_tid])
                result = general(
                    query, query_scale, weights, index_cache, index_cache_scale,
                    block_table, request_ids, lengths, output,
                    num_tokens, ready, ready, ready,
                )
            else:
                result = request_blocks(
                    query, query_scale, weights, index_cache, index_cache_scale,
                    block_table, request_ids, lengths, output,
                    runs, cache_ready, query_ready, weights_ready, plan_tid,
                )
        elif paired_panels <= 16:
            result = general_small(
                query, query_scale, weights, index_cache, index_cache_scale,
                block_table, request_ids, lengths, output,
                num_tokens, cache_ready, query_ready, weights_ready,
            )
        else:
            result = general(
                query, query_scale, weights, index_cache, index_cache_scale,
                block_table, request_ids, lengths, output,
                num_tokens, cache_ready, query_ready, weights_ready,
            )
        return result

    return score


score_full = make_index_scores()
score_full_pretransposed = make_index_scores(pretransposed=True)
score_c2a_full = make_index_scores(skip_short=True)


def _make_pool_index_scores(tile_n, panels_per_task, workers):
    """Score candidate blocks locally with the same FP8/MX arithmetic as Full."""
    blocks_per_panel = tile_n // 8
    blocks_per_task = blocks_per_panel * panels_per_task
    prefix_row = 0 if tile_n == 128 else panels_per_task

    @pl.jit.inline
    def score(
        query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
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
        query_flat = pl.reshape(query, [pl.tensor.dim(query, 0) * INDEX_H, INDEX_DIM])
        table_width = pl.tensor.dim(block_table, 1)
        request_count = pl.tensor.dim(block_table, 0)
        with pl.spmd(
            workers, name_hint="idx_score_pool", deps=[ready],
            optimizations=[pl.cross_core_slot(slot_num=2)],
        ) as score_tid:
            worker = pl.tile.get_block_idx()
            lut = _index_key_decode_lut()
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
                query_panel = pl.load(query_flat, [token * INDEX_H, 0], [INDEX_H, INDEX_DIM])
                query_scale_raw = pl.reinterpret_view(
                    pl.load(query_scale, [token, 0], [1, INDEX_H * _SCALE_COUNT]), pl.UINT8
                )
                query_scale_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                query_scale_zz = pl.reinterpret_view(
                    pl.tmov_x2zz(query_scale_raw, query_scale_tmp, group_axis=1,
                                 dst_rows=INDEX_H, dst_cols=_SCALE_COUNT), pl.FP8E8M0
                )
                query_mat = pl.tile.move(
                    query_panel, target_memory=pl.MemorySpace.Mat,
                    blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major,
                )
                query_left = pl.tile.move(query_mat, target_memory=pl.MemorySpace.Left)
                query_scale_mat = pl.tile.move(
                    query_scale_zz, target_memory=pl.MemorySpace.Mat,
                    blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major,
                )
                query_scale_left = pl.tile.move(query_scale_mat, target_memory=pl.MemorySpace.LeftScale)
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
                                            payload = pl.reinterpret_view(
                                                pl.load(cache_flat, [physical_row, 0], [8, INDEX_DIM // 2]), pl.INT8
                                            )
                                            packed = pl.tile.assemble(packed, payload, [block * 8, 0])
                                            scales = pl.reinterpret_view(
                                                pl.reinterpret_view(
                                                    pl.load(scale_flat, [physical_row // 8, 0], [1, 32]), pl.UINT8
                                                ), pl.INT8
                                            )
                                            raw_scale = pl.tile.assemble(raw_scale, scales, [0, block * 32])
                                            lanes = pl.tile.arange(0, [1, 8], dtype=pl.INT32)
                                            remaining = pl.cast(pl.min(8, visible - logical_block * 8), pl.INT32)
                                            ones = pl.tile.full([1, 8], dtype=pl.INT32, value=1)
                                            valid = pl.tile.select(pl.tile.cmps(lanes, remaining, cmp_type=2), ones, 0)
                                            valid_lanes = pl.tile.assemble(valid_lanes, valid, [0, block * 8])
                        # Missing-page lanes remain zero-filled and are masked after a real tile transfer.
                        byte_values = pl.ands(pl.cast(packed, pl.INT16), 255)
                        low = pl.ands(byte_values, 15)
                        high = pl.ands(pl.shrs(byte_values, 4), 15)
                        lut_tmp = pl.create_tile([1, 32], dtype=pl.UINT8)
                        low_bytes = pl.tile.gather(lut, low, lut_tmp)
                        high_bytes = pl.tile.gather(lut, high, lut_tmp)
                        words = pl.or_(low_bytes, pl.shls(high_bytes, 8))
                        keys = pl.reinterpret_view(
                            pl.reinterpret_view(words, pl.INT8, shape=[tile_n, INDEX_DIM]), pl.FP8E4M3FN
                        )
                        scale_tmp = pl.create_tile([1, 128], dtype=pl.UINT8)
                        scale_zz = pl.reinterpret_view(
                            pl.tmov_x2zz(pl.reinterpret_view(raw_scale, pl.UINT8), scale_tmp,
                                         group_axis=1, dst_rows=tile_n, dst_cols=_SCALE_COUNT), pl.FP8E8M0
                        )
                        scale_nn = pl.tile.transpose_view(scale_zz)
                        keys_kn = pl.reinterpret_view(
                            pl.tile.transpose(pl.reinterpret_view(keys, pl.INT8), 0, 1), pl.FP8E4M3FN
                        )
                        # Initialize an independent dot for each history interval.
                        dot = pl.cast(pl.matmul_mx(query_left, query_scale_left, keys_kn, scale_nn),
                                      pl.BF16, mode="rint")
                        relu = pl.maximum(pl.cast(dot, pl.FP32), 0.0)
                        head_weight = pl.mul(pl.cast(pl.load(weights, [token, 0], [1, INDEX_H]), pl.FP32), _SCORE_SCALE)
                        rounded_weight = pl.cast(pl.cast(head_weight, pl.BF16, mode="rint"), pl.FP32)
                        weighted = pl.row_expand_mul(relu, pl.reshape(rounded_weight, [INDEX_H, 1]))
                        weighted = pl.cast(pl.cast(weighted, pl.BF16, mode="rint"), pl.FP32)
                        total = pl.cast(pl.cast(pl.col_sum(weighted), pl.BF16, mode="rint"), pl.FP32)
                        invalid = pl.reinterpret_view(
                            pl.tile.full([1, tile_n], dtype=pl.INT32, value=-8388608), pl.FP32
                        )
                        masked = pl.tile.select(pl.tile.cmps(valid_lanes, 1, cmp_type=0), total, invalid)
                        valid_output = pl.min(tile_n, (capacity - slot_begin) * 8)
                        pl.store(pl.set_validshape(masked, 1, valid_output), [token, slot_begin * 8], output)
        return score_tid

    return score


_pool_page = _make_pool_index_scores(128, 1, 32)
_pool_page_small = _make_pool_index_scores(128, 1, 16)
_pool_block = _make_pool_index_scores(256, 1, 32)
_pool_macro = _make_pool_index_scores(256, 2, 32)


@pl.jit.inline(auto_scope=False)
def score_reindex(
    query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
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
    """Route complete identity pools to dense scoring and sparse pools to compact FP8/MX leaves."""
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
            block_table, request_ids, lengths, output, num_tokens, ready, ready, ready, query, 0,
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
        indices = pl.add(pl.tile.arange(0, [1, _TOPK_LEAF_TILE], dtype=pl.INT32), pl.cast(logical_begin, pl.INT32))
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
        with pl.spmd(32, name_hint="idx_topk_short", deps=[scores_ready]) as short_tid:
            worker = pl.tile.get_block_idx()
            for token in pl.range(worker, num_tokens, 32):
                invalid_bits = pl.tile.full([1, 8], dtype=pl.INT32, value=_TOPK_INVALID_SCORE_BITS)
                invalid_score = pl.tile.read(pl.reinterpret_view(invalid_bits, pl.FP32), [0, 0])
                _materialize_index_topk(
                    index_scores, block_table, request_ids, output,
                    token, width, invalid_score, pl.cast(0, pl.INDEX),
                )
        topk_tid = pl.yield_(short_tid)
    else:
        leaf_count = (width + _TOPK_LEAF_TILE - 1) // _TOPK_LEAF_TILE
        arena_rows = tokens * leaf_count
        pair_arena = pl.create_tensor([arena_rows, _TOPK_PAIR_WIDTH], dtype=pl.FP32)
        total_leaves = pl.cast(num_tokens, pl.INDEX) * leaf_count
        with pl.spmd(32, name_hint="idx_topk_leaf", deps=[scores_ready]) as leaf_tid:
            worker = pl.tile.get_block_idx()
            for work in pl.range(worker, total_leaves, 32):
                token = work // leaf_count
                leaf = work - token * leaf_count
                logical_begin = leaf * _TOPK_LEAF_TILE
                valid_count = pl.min(_TOPK_LEAF_TILE, width - logical_begin)
                output_slot = token * leaf_count + leaf
                _sort_index_topk_leaf(index_scores, pair_arena, token, logical_begin, valid_count, output_slot)
        with pl.spmd(32, name_hint="idx_topk_merge", deps=[leaf_tid]) as merge_tid:
            worker = pl.tile.get_block_idx()
            for token in pl.range(worker, num_tokens, 32):
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
                _materialize_index_topk(
                    index_scores, block_table, request_ids, output,
                    token, width, cutoff, INDEX_TOPK - above,
                )
        topk_tid = pl.yield_(merge_tid)
    return topk_tid


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
def rotate_index_query(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    blocks = num_tokens * INDEX_H
    workers = pl.min(blocks, 32)
    with pl.spmd(workers, name_hint='rope_idx_q', deps=[ready]) as rope_tid:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, blocks, workers):
            _index_query_rope_block(x, cos, sin, output, block)
    return rope_tid


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
    private_query: pl.Tensor,
    private_route: pl.Scalar[pl.INT32],
    pretransposed: pl.constexpr,
    query_max_workers: pl.constexpr = 2147483647,
):
    tokens = pl.tensor.dim(x, 0)
    projected = pl.create_tensor([tokens, INDEX_H * INDEX_DIM], dtype=pl.BF16)
    gemm_bf16_via_mxfp8_to_bf16_staged(
        qr, index_wq_b, index_wq_b_scale, projected, num_tokens,
        Q_LORA, INDEX_QUERY_WIDTH, "proj_idx_q", "quant_idx_q",
        INDEX_PROJ_N_TILE, INDEX_PROJ_K_TILE, 1, query_max_workers,
    )
    rotated = pl.create_tensor([tokens, INDEX_H * INDEX_DIM], dtype=pl.BF16)
    rotated_tid = rotate_index_query(projected, cos, sin, rotated, num_tokens, entry_ready)
    query = pl.create_tensor([tokens, INDEX_H * INDEX_DIM], dtype=pl.FP8E4M3FN)
    query_scale = pl.create_tensor(
        [tokens, INDEX_H * (INDEX_DIM // INDEX_CACHE_GROUP)], dtype=pl.FP8E8M0
    )
    if pretransposed:
        if private_route != 0:
            query_tid = quantize_index_query(
                rotated, query, query_scale, num_tokens, rotated_tid, private_query, pretransposed=True,
            )
        else:
            query_tid = quantize_index_query(
                rotated, query, query_scale, num_tokens, rotated_tid, query, pretransposed=False,
            )
    else:
        query_tid = quantize_index_query(
            rotated, query, query_scale, num_tokens, rotated_tid, query, pretransposed=False,
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
_select_c2a_group6_topk = _make_select_c2a_index_topk(leaf_workers=18, row_workers=6, all_long=True)

_score_c2a_group6 = _make_query_interval_index_scores(8, skip_short=True, workers=30)


@pl.jit.inline(auto_scope=False)
def _score_select_c2a_groups(
    query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
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
    """Select C2A rows with two six-query score and TopK groups."""
    width = pl.tensor.dim(scores, 1)
    grouped = 0
    if num_tokens == 12 and width == 4224:
        grouped = 1
        for token in pl.range(12):
            if pl.read(lengths, [token]) <= INDEX_TOPK:
                grouped = 0
        for token in pl.range(1, 12):
            if pl.read(request_ids, [token - 1]) == pl.read(request_ids, [token]):
                grouped = 0
    if grouped != 0:
        query_first = query[0:6, :]
        scale_first = query_scale[0:6, :]
        weights_first = weights[0:6, :]
        requests_first = request_ids[0:6]
        lengths_first = lengths[0:6]
        scores_first = pl.create_tensor([6, width], dtype=pl.FP32)
        score_first_tid = _score_c2a_group6(
            query_first, scale_first, weights_first, index_cache, index_cache_scale,
            block_table, requests_first, lengths_first, scores_first,
            6, cache_ready, query_ready, weights_ready,
        )
        topk_first = output[0:6, :]
        order_first_tid = _select_c2a_group6_topk(
            scores_first, block_table, requests_first, lengths_first, topk_first, 6, score_first_tid,
        )
        query_second = query[6:12, :]
        scale_second = query_scale[6:12, :]
        weights_second = weights[6:12, :]
        requests_second = request_ids[6:12]
        lengths_second = lengths[6:12]
        scores_second = pl.create_tensor([6, width], dtype=pl.FP32)
        score_second_tid = _score_c2a_group6(
            query_second, scale_second, weights_second, index_cache, index_cache_scale,
            block_table, requests_second, lengths_second, scores_second,
            6, cache_ready, query_ready, weights_ready,
        )
        topk_second = output[6:12, :]
        order_second_tid = _select_c2a_group6_topk(
            scores_second, block_table, requests_second, lengths_second, topk_second, 6, score_second_tid,
        )
        result = pl.system.task_dummy(deps=[order_first_tid, order_second_tid])
    else:
        scores_tid = score_c2a_full(
            query, query_scale, weights, index_cache, index_cache_scale,
            block_table, request_ids, lengths, scores,
            num_tokens, cache_ready, query_ready, weights_ready, query, 0,
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
        cache_ready, index_wq_b, 0, pretransposed=False, query_max_workers=2147483647,
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
    "score_full_pretransposed",
    "use_pretransposed_index_query",
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
