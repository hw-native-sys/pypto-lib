# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""First-level candidate-block selection for the decoder hierarchical sparse indexer."""

import pypto.language as pl
import torch

from models.deepseek_v4_1_flash.config import FLASH
from models.deepseek_v4_1_flash.config import CMP_POSITIONS_DYN, T_DYN
from models.deepseek_v4_1_flash.golden import select_candidate_blocks


CANDIDATE_BLOCK_SIZE = FLASH.candidate_block_size
CANDIDATE_TOPK_BLOCKS = FLASH.candidate_topk_blocks
CANDIDATE_PAIR_WIDTH = 2 * CANDIDATE_TOPK_BLOCKS
CANDIDATE_SCORE_TILE = 1024
CANDIDATE_LEAF = 8192
CANDIDATE_SHORT_LEAF = 2048
CANDIDATE_MAX_BLOCKS = FLASH.max_position_embeddings // CANDIDATE_BLOCK_SIZE
CANDIDATE_MAX_LEAVES = (CANDIDATE_MAX_BLOCKS + CANDIDATE_LEAF - 1) // CANDIDATE_LEAF


def golden_hierarchical_sparse_indexer(
    index_scores: torch.Tensor,
    compressed_lens: torch.Tensor,
) -> torch.Tensor:
    return select_candidate_blocks(
        index_scores, compressed_lens, FLASH.candidate_topk_blocks, FLASH.candidate_block_size
    )


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
    raw = pl.load(
        scores,
        [token, block_begin],
        [1, CANDIDATE_LEAF],
        valid_shape=[1, valid_count],
    )
    padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
    floor = pl.tile.full([1, CANDIDATE_LEAF], dtype=pl.FP32, value=-1e30)
    values = pl.maximum(padded, floor)
    block_begin_i32 = pl.cast(block_begin, pl.INT32)
    indices = pl.add(
        pl.tile.arange(0, [1, CANDIDATE_LEAF], dtype=pl.INT32),
        block_begin_i32,
    )
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
    raw = pl.load(
        scores,
        [token, 0],
        [1, CANDIDATE_SHORT_LEAF],
        valid_shape=[1, valid_count],
    )
    values = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
    values = pl.maximum(values, -1e30)
    indices = pl.tile.arange(0, [1, CANDIDATE_SHORT_LEAF], dtype=pl.INT32)
    pairs = pl.tile.sort32(values, pl.reinterpret_view(indices, pl.UINT32))
    pairs = pl.tile.mrgsort(pairs, block_len=64)
    pairs = pl.tile.mrgsort(pairs, block_len=256)
    pairs = pl.tile.mrgsort(pairs, block_len=1024)
    top_pairs = pl.tile.slice(pairs, [1, CANDIDATE_PAIR_WIDTH], [0, 0])
    pl.store(top_pairs, [output_slot, 0], arena)


@pl.jit.inline(auto_scope=False)
def hierarchical_sparse_indexer(
    index_scores: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.FP32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    candidate_mask: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.UINT8],
):
    """Select top-scoring blocks and expand them into a position mask."""
    tokens = pl.tensor.dim(index_scores, 0)
    positions = pl.tensor.dim(candidate_mask, 1)
    block_count = (positions + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
    padded_block_count = (block_count + CANDIDATE_LEAF - 1) // CANDIDATE_LEAF * CANDIDATE_LEAF
    mask_tiles = (positions + 127) // 128
    padded_positions = mask_tiles * 128
    block_scores = pl.create_tensor([tokens, padded_block_count], dtype=pl.FP32)
    aligned_mask = pl.create_tensor([tokens, padded_positions], dtype=pl.UINT8)
    score_tiles = (block_count + CANDIDATE_SCORE_TILE - 1) // CANDIDATE_SCORE_TILE
    with pl.spmd(tokens * score_tiles, name_hint="c1a_candidate_block_scores") as score_tid:
        work = pl.tile.get_block_idx()
        token = work // score_tiles
        score_tile = work % score_tiles
        block_begin = score_tile * CANDIDATE_SCORE_TILE
        position_begin = block_begin * CANDIDATE_BLOCK_SIZE
        valid_blocks = pl.min(CANDIDATE_SCORE_TILE, block_count - block_begin)
        valid_positions = pl.min(
            CANDIDATE_SCORE_TILE * CANDIDATE_BLOCK_SIZE,
            positions - position_begin,
        )
        raw = pl.load(
            index_scores,
            [token, position_begin],
            [1, CANDIDATE_SCORE_TILE * CANDIDATE_BLOCK_SIZE],
            valid_shape=[1, valid_positions],
        )
        padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
        grouped = pl.reshape(padded, [CANDIDATE_SCORE_TILE, CANDIDATE_BLOCK_SIZE])
        reduce_tmp = pl.create_tile(
            [CANDIDATE_SCORE_TILE, CANDIDATE_BLOCK_SIZE],
            dtype=pl.FP32,
        )
        maxima = pl.reshape(pl.row_max(grouped, tmp_tile=reduce_tmp), [1, CANDIDATE_SCORE_TILE])
        visible = pl.read(compressed_lens, [token])
        if visible > 0:
            last_block = (visible - 1) // CANDIDATE_BLOCK_SIZE
            if last_block >= block_begin and last_block < block_begin + valid_blocks:
                pl.tile.write(maxima, [0, last_block - block_begin], 1e30)
        maxima = pl.tile.set_validshape(maxima, 1, valid_blocks)
        pl.store(maxima, [token, block_begin], block_scores)

    with pl.spmd(tokens * mask_tiles, name_hint="c1a_candidate_clear") as clear_tid:
        work = pl.tile.get_block_idx()
        token = work // mask_tiles
        tile = work % mask_tiles
        position = tile * 128
        empty_i32 = pl.tile.full([1, 128], dtype=pl.INT32, value=0)
        empty = pl.cast(empty_i32, pl.UINT8)
        pl.store(empty, [token, position], aligned_mask)

    arena = pl.create_tensor(
        [tokens * CANDIDATE_MAX_LEAVES, CANDIDATE_PAIR_WIDTH],
        dtype=pl.FP32,
    )
    with pl.spmd(tokens, name_hint="c1a_candidate_topk", deps=[score_tid, clear_tid]) as topk_tid:
        token = pl.tile.get_block_idx()
        visible = pl.read(compressed_lens, [token])
        visible_blocks = (visible + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
        arena_base = token * CANDIDATE_MAX_LEAVES
        count = 0
        if visible_blocks > 0 and visible_blocks <= CANDIDATE_SHORT_LEAF:
            _sort_candidate_short_leaf(block_scores, arena, token, visible_blocks, arena_base)
            count = 1
        elif visible_blocks > 0:
            leaf_count = (visible_blocks + CANDIDATE_LEAF - 1) // CANDIDATE_LEAF
            for leaf in pl.range(leaf_count):
                block_begin = leaf * CANDIDATE_LEAF
                valid_count = pl.min(CANDIDATE_LEAF, visible_blocks - block_begin)
                _sort_candidate_leaf(
                    block_scores,
                    arena,
                    token,
                    block_begin,
                    valid_count,
                    arena_base + leaf,
                )
            count = leaf_count
        if count > 1:
            count = _merge_candidate_level(arena, arena_base, count)
        if count > 1:
            count = _merge_candidate_level(arena, arena_base, count)
        if count > 1:
            count = _merge_candidate_level(arena, arena_base, count)
        if count > 1:
            count = _merge_candidate_level(arena, arena_base, count)

        if visible_blocks > 0:
            root = pl.load(arena, [arena_base, 0], [1, CANDIDATE_PAIR_WIDTH])
            selected = pl.tile.gather_mask(
                root,
                mask_pattern=pl.tile.MaskPattern.P1010,
                output_dtype=pl.INT32,
            )
            selected_count = pl.min(visible_blocks, CANDIDATE_TOPK_BLOCKS)
            for selected_lane in pl.range(selected_count):
                selected_block = pl.tile.read(selected, [0, selected_lane])
                position_begin = selected_block * CANDIDATE_BLOCK_SIZE
                for offset in pl.unroll(CANDIDATE_BLOCK_SIZE):
                    position = position_begin + offset
                    if position < positions:
                        # Each token owns a distinct 128-byte-aligned row.
                        pl.write(aligned_mask, [token, position], pl.cast(1, pl.UINT8))

    with pl.spmd(tokens * mask_tiles, name_hint="c1a_candidate_publish", deps=[topk_tid]):
        work = pl.tile.get_block_idx()
        token = work // mask_tiles
        tile = work % mask_tiles
        position = tile * 128
        valid = pl.min(128, positions - position)
        mask = pl.load(aligned_mask, [token, position], [1, 128])
        mask = pl.tile.set_validshape(mask, 1, valid)
        pl.store(mask, [token, position], candidate_mask)
    return candidate_mask


__all__ = ["golden_hierarchical_sparse_indexer", "hierarchical_sparse_indexer"]


if __name__ == "__main__":
    from models.deepseek_v4_1_flash._golden_smoke import run_hierarchical_indexer_golden

    run_hierarchical_indexer_golden(golden_hierarchical_sparse_indexer)
