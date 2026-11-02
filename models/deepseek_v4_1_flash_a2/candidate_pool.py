# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2/A3 C1A candidate-block publication.

The standalone A2 QLI leaf exposes a fixed 512-row score bucket.  This leaf
reduces those scores in groups of eight rows and publishes the same sorted
logical block IDs and counts that vLLM-Ascend passes to C1A reindex.
"""

from __future__ import annotations

import pypto.language as pl


CANDIDATE_BLOCK_SIZE = 8
CANDIDATE_TOPK_BLOCKS = 2048
SCORE_WIDTH = 512
POOL_CAPACITY = SCORE_WIDTH // CANDIDATE_BLOCK_SIZE


def make_candidate_block_pool(
    capacity: int,
    *,
    score_width: int = SCORE_WIDTH,
    topk_blocks: int = CANDIDATE_TOPK_BLOCKS,
    block_size: int = CANDIDATE_BLOCK_SIZE,
):
    """Build the fixed-bucket candidate publisher used by the decoder stages."""

    if capacity <= 0 or capacity % 16:
        raise ValueError("candidate pool capacity must be a positive multiple of 16")
    if score_width != SCORE_WIDTH or topk_blocks != CANDIDATE_TOPK_BLOCKS:
        raise ValueError("A2/A3 candidate pool uses a fixed 512-row score bucket")
    if block_size != CANDIDATE_BLOCK_SIZE:
        raise ValueError("A2/A3 candidate blocks contain eight compressed rows")

    @pl.jit
    def kernel(
        scores: pl.Tensor[[capacity, score_width], pl.FP32],
        compressed_lengths: pl.Tensor[[capacity], pl.INT32],
        pool_block_ids: pl.Out[pl.Tensor[[capacity, POOL_CAPACITY], pl.INT32]],
        pool_block_counts: pl.Out[pl.Tensor[[capacity], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(capacity, name_hint="dsv41_a2a3_candidate_pool"):
            token = pl.tile.get_block_idx()
            if token < active:
                raw = pl.load(scores, [token, 0], [1, score_width])
                grouped = pl.reshape(raw, [POOL_CAPACITY, CANDIDATE_BLOCK_SIZE])
                reduce_tmp = pl.tile.create(
                    [POOL_CAPACITY, CANDIDATE_BLOCK_SIZE], dtype=pl.FP32
                )
                maxima = pl.reshape(
                    pl.row_max(grouped, tmp_tile=reduce_tmp), [1, POOL_CAPACITY]
                )
                length = pl.max(pl.read(compressed_lengths, [token]), 0)
                visible_blocks = pl.min(
                    (length + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE,
                    POOL_CAPACITY,
                )
                lanes = pl.tile.arange(0, [1, POOL_CAPACITY], dtype=pl.INT32)
                for lane in pl.range(POOL_CAPACITY):
                    if lane >= visible_blocks:
                        pl.tile.write(maxima, [0, lane], -1.0e30)
                if visible_blocks > 0:
                    # The newest causal block must remain a candidate even if
                    # its score is below every historical block.
                    pl.tile.write(maxima, [0, visible_blocks - 1], 1.0e30)
                pairs = pl.tile.sort32(maxima, pl.reinterpret_view(lanes, pl.UINT32))
                pairs = pl.tile.mrgsort(pairs, block_len=64)
                sorted_ids = pl.tile.gather_mask(
                    pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                )

                # Sort the selected prefix by logical ID before publication.
                # The vLLM pool uses this order so a consumer can expand blocks
                # without carrying score order through the cache gather.
                ordered_scores = pl.tile.full([1, POOL_CAPACITY], dtype=pl.FP32, value=-1.0e30)
                ordered_ids = pl.tile.full([1, POOL_CAPACITY], dtype=pl.INT32, value=-1)
                selected_count = pl.min(visible_blocks, POOL_CAPACITY)
                for lane in pl.range(selected_count):
                    block_id = pl.tile.read(sorted_ids, [0, lane])
                    pl.tile.write(ordered_scores, [0, lane], -pl.cast(block_id, pl.FP32))
                    pl.tile.write(ordered_ids, [0, lane], block_id)
                ordered = pl.tile.sort32(
                    ordered_scores, pl.reinterpret_view(ordered_ids, pl.UINT32)
                )
                ordered = pl.tile.mrgsort(ordered, block_len=64)
                ordered_ids = pl.tile.gather_mask(
                    ordered, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                )
                pl.store(ordered_ids, [token, 0], pool_block_ids)
            else:
                empty = pl.tile.full([1, POOL_CAPACITY], dtype=pl.INT32, value=-1)
                pl.store(empty, [token, 0], pool_block_ids)
        # Scalar INT32 stores from one worker per token can share a cache line
        # on A2/A3.  Publish the compact count vector from a single worker.
        with pl.spmd(1, name_hint="dsv41_a2a3_candidate_counts"):
            worker = pl.tile.get_block_idx()
            if worker == 0:
                for token in pl.range(capacity):
                    length = pl.max(pl.read(compressed_lengths, [token]), 0)
                    visible_blocks = pl.min(
                        (length + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE,
                        POOL_CAPACITY,
                    )
                    count = pl.min(visible_blocks, POOL_CAPACITY)
                    if token >= active:
                        count = 0
                    pl.write(pool_block_counts, [token], pl.cast(count, pl.INT32))
        return pool_block_ids, pool_block_counts

    return kernel


__all__ = [
    "CANDIDATE_BLOCK_SIZE",
    "CANDIDATE_TOPK_BLOCKS",
    "POOL_CAPACITY",
    "SCORE_WIDTH",
    "make_candidate_block_pool",
]
