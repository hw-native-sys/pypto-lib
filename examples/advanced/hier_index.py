# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exact hierarchical sparse indexer for one decode query (two-level pooling).

    s_i  = sum_h w_h * dot(q_h, k_i)      # FP32 score for every KV position
    bm_j = max of scores in block j       # S // BLOCK block maxima
    pool = positions in the top-TOPB blocks
    (top_val, top_idx) = descending top-TOPK inside the pool

This is exact selection, not approximate sparse attention: every position is
scored, block maxima are exact, and the final top-TOPK indices are ordered
exactly as torch.topk returns them (BF16 inputs are promoted to FP32 before
any score arithmetic).

Pipeline: an SPMD scan scores BLOCK-sized packets and keeps per-worker maxima
in a compact scratch; a single core-group rebuilds the block maxima, sorts
each LEAF-sized run with sort32 + mrgsort, and merges runs into the global
top-TOPB blocks; an SPMD stage then gathers only the selected blocks' scores,
sorts them per leaf, and a final core-group merge emits the ordered top-TOPK.
"""

import pypto.language as pl

H = 8
DI = 128
BLOCK = 64
TOPB = 256
TOPK = 512
BIG = 1e30
LEAF = 2048
SCORE_BLOCKS = 1
FUSE_BLOCKMAX = False
NATIVE_BLOCKS = True
COMPACT = True
WORKERS = 48
RANKS_PER_OWNER = 4
GATHER_OWNERS = 48
PACKET_BLOCKS = 16
BALANCED_RANGES = 1
SCORE_ROWS = 64
SCORE_BLOCKS_PER_TILE = 1


# default KV positions: one 128K-token context (multiple of BLOCK, >= 32768)
SEQ_LEN = 131072


def _make_hier_index(seq_len, input_dtype):
    """Build the exact hierarchical indexer kernel for one KV length / input dtype."""
    S = int(seq_len)
    if S < 32768 or S % BLOCK:
        raise ValueError("S must be a multiple of 64 and at least 32768")
    dtype_name = "bf16" if input_dtype == pl.BF16 else "fp32"
    NB = S // BLOCK
    MAXIMUM_LINES = (NB + 15) // 16
    SEARCH_STEPS = TOPB.bit_length()
    USE_DIRECTORY = MAXIMUM_LINES <= 2**24
    PACKETS = (NB + PACKET_BLOCKS - 1) // PACKET_BLOCKS
    BALANCED = bool(BALANCED_RANGES) and NB <= 1048576
    ACTIVE_WORKERS = WORKERS if BALANCED else min(WORKERS, PACKETS)
    RANGE_BASE = NB // ACTIVE_WORKERS
    RANGE_EXTRA = NB % ACTIVE_WORKERS
    RANGE_CUTOFF = RANGE_EXTRA * (RANGE_BASE + 1)
    RANGE_DIVISOR = max(1, RANGE_BASE)
    EARLY_RECIPROCAL = 1.0 / (RANGE_BASE + 1)
    LATE_RECIPROCAL = 1.0 / RANGE_DIVISOR
    MAXIMA_STRIDE = max(16, ((NB + ACTIVE_WORKERS - 1) // ACTIVE_WORKERS + 15) // 16 * 16)
    NBL = (NB + LEAF - 1) // LEAF
    FAST_INVERSE = BALANCED and 512 <= NB <= 4096 and RANGE_BASE > 0
    STREAM = TOPB * BLOCK if COMPACT else S
    NSL = (STREAM + LEAF - 1) // LEAF
    LATE_JOIN = dtype_name == "bf16" and (131072 <= S <= 180224)
    PREJOIN_END = min(NSL, 7 + 1)
    BPAD = (NB + 511) // 512 * 512
    if NBL <= 1:
        if dtype_name == "bf16":

            @pl.jit
            def hierindex(
                q: pl.Tensor[[H, DI], input_dtype],
                kc: pl.Tensor[[S, DI], input_dtype],
                w: pl.Tensor[[H], input_dtype],
                ws_scr: pl.InOut[pl.Tensor[[1, S], pl.FP32]],
                ws_pool: pl.InOut[pl.Tensor[[1, S], pl.FP32]],
                ws_bm: pl.InOut[pl.Tensor[[1, NB], pl.FP32]],
                top_val: pl.Out[pl.Tensor[[1, TOPK], pl.FP32]],
                top_idx: pl.Out[pl.Tensor[[1, TOPK], pl.INT32]],
            ):
                scan_maxima = pl.create_tensor([ACTIVE_WORKERS, MAXIMA_STRIDE], dtype=pl.FP32)
                selected_blocks = pl.create_tensor([1, TOPB], dtype=pl.INT32)
                block_pairs_gm = pl.create_tensor([NBL, 2 * TOPB], dtype=pl.FP32)
                score_pairs_gm = pl.create_tensor([NSL, 2 * TOPK], dtype=pl.FP32)
                block_remaining = pl.create_tensor([1, BPAD], dtype=pl.FP32)
                with pl.spmd(
                    ACTIVE_WORKERS, name_hint="scan_score_packets", allow_early_resolve=True
                ) as scan_done:
                    worker = pl.tile.get_block_idx()
                    weights_fp32 = pl.cast(
                        pl.reshape(pl.load(w, [0], [16], valid_shape=[H]), [1, 16]), pl.FP32
                    )
                    weighted_q_0 = pl.mul(
                        pl.cast(pl.load(q, [0, 0], [1, DI]), pl.FP32), pl.read(weights_fp32, [0, 0])
                    )
                    weighted_q_1 = pl.mul(
                        pl.cast(pl.load(q, [1, 0], [1, DI]), pl.FP32), pl.read(weights_fp32, [0, 1])
                    )
                    weighted_q_2 = pl.mul(
                        pl.cast(pl.load(q, [2, 0], [1, DI]), pl.FP32), pl.read(weights_fp32, [0, 2])
                    )
                    weighted_q_3 = pl.mul(
                        pl.cast(pl.load(q, [3, 0], [1, DI]), pl.FP32), pl.read(weights_fp32, [0, 3])
                    )
                    weighted_q_4 = pl.mul(
                        pl.cast(pl.load(q, [4, 0], [1, DI]), pl.FP32), pl.read(weights_fp32, [0, 4])
                    )
                    weighted_q_5 = pl.mul(
                        pl.cast(pl.load(q, [5, 0], [1, DI]), pl.FP32), pl.read(weights_fp32, [0, 5])
                    )
                    weighted_q_6 = pl.mul(
                        pl.cast(pl.load(q, [6, 0], [1, DI]), pl.FP32), pl.read(weights_fp32, [0, 6])
                    )
                    weighted_q_7 = pl.mul(
                        pl.cast(pl.load(q, [7, 0], [1, DI]), pl.FP32), pl.read(weights_fp32, [0, 7])
                    )
                    q_pair_0 = pl.add(weighted_q_0, weighted_q_1)
                    q_pair_1 = pl.add(weighted_q_2, weighted_q_3)
                    q_pair_2 = pl.add(weighted_q_4, weighted_q_5)
                    q_pair_3 = pl.add(weighted_q_6, weighted_q_7)
                    q_eff = pl.add(pl.add(q_pair_0, q_pair_1), pl.add(q_pair_2, q_pair_3))
                    if BALANCED:
                        begin_block = worker * RANGE_BASE + pl.min(worker, RANGE_EXTRA)
                        owned_blocks = RANGE_BASE + pl.min(1, pl.max(0, RANGE_EXTRA - worker))
                        private_zero = pl.tile.full(
                            [1, MAXIMA_STRIDE], dtype=pl.FP32, value=-3.4028234663852886e38
                        )
                        for local_packet, (private_state,) in pl.range(
                            (owned_blocks + PACKET_BLOCKS - 1) // PACKET_BLOCKS, init_values=(private_zero,)
                        ):
                            packet_begin = begin_block + local_packet * PACKET_BLOCKS
                            valid_blocks = pl.min(PACKET_BLOCKS, owned_blocks - local_packet * PACKET_BLOCKS)
                            packet_zero = pl.tile.full(
                                [PACKET_BLOCKS, BLOCK], dtype=pl.FP32, value=-3.4028234663852886e38
                            )
                            for score_tile, (packet_state,) in pl.pipeline(
                                valid_blocks // SCORE_BLOCKS_PER_TILE, stage=2, init_values=(packet_zero,)
                            ):
                                subblock = score_tile * SCORE_BLOCKS_PER_TILE
                                sb = packet_begin + subblock
                                keys_full = pl.cast(pl.load(kc, [sb * BLOCK, 0], [SCORE_ROWS, DI]), pl.FP32)
                                acc = pl.row_sum(
                                    pl.col_expand_mul(keys_full, q_eff),
                                    pl.tile.create([SCORE_ROWS, DI], dtype=pl.FP32),
                                )
                                score_blocks = pl.reshape(acc, [SCORE_BLOCKS_PER_TILE, BLOCK])
                                packet_next = pl.tile.assemble(packet_state, score_blocks, [subblock, 0])
                                full_packet = pl.yield_(packet_next)
                            for tail_tile, (tail_state,) in pl.range(
                                valid_blocks % SCORE_BLOCKS_PER_TILE, init_values=(full_packet,)
                            ):
                                tail_subblock = (
                                    valid_blocks // SCORE_BLOCKS_PER_TILE
                                ) * SCORE_BLOCKS_PER_TILE + tail_tile
                                tail_block = packet_begin + tail_subblock
                                tail_keys = pl.cast(
                                    pl.load(kc, [tail_block * BLOCK, 0], [BLOCK, DI]), pl.FP32
                                )
                                tail_acc = pl.row_sum(
                                    pl.col_expand_mul(tail_keys, q_eff),
                                    pl.tile.create([BLOCK, DI], dtype=pl.FP32),
                                )
                                tail_scores = pl.reshape(tail_acc, [1, BLOCK])
                                tail_next = pl.tile.assemble(tail_state, tail_scores, [tail_subblock, 0])
                                packet_scores = pl.yield_(tail_next)
                            packet_flat = pl.set_validshape(
                                pl.reshape(packet_scores, [1, PACKET_BLOCKS * BLOCK]), 1, valid_blocks * BLOCK
                            )
                            pl.store(packet_flat, [0, packet_begin * BLOCK], ws_scr)
                            packet_maxima = pl.row_max(
                                packet_scores, pl.tile.create([PACKET_BLOCKS, BLOCK], dtype=pl.FP32)
                            )
                            maxima_flat = pl.set_validshape(
                                pl.reshape(packet_maxima, [1, PACKET_BLOCKS]), 1, valid_blocks
                            )
                            pool_zero_packet = pl.tile.full(
                                [1, PACKET_BLOCKS * BLOCK], dtype=pl.FP32, value=0.0
                            )
                            pl.store(
                                pl.set_validshape(pool_zero_packet, 1, valid_blocks * BLOCK),
                                [0, packet_begin * BLOCK],
                                ws_pool,
                            )
                            private_next = pl.tile.assemble(
                                private_state, maxima_flat, [0, local_packet * PACKET_BLOCKS]
                            )
                            private_maxima = pl.yield_(private_next)
                        pl.store(private_maxima, [worker, 0], scan_maxima)
                    else:
                        for packet in pl.range(worker, PACKETS, ACTIVE_WORKERS):
                            packet_begin = packet * PACKET_BLOCKS
                            valid_blocks = pl.min(PACKET_BLOCKS, NB - packet_begin)
                            packet_zero = pl.tile.full(
                                [PACKET_BLOCKS, BLOCK], dtype=pl.FP32, value=-3.4028234663852886e38
                            )
                            for score_tile, (packet_state,) in pl.pipeline(
                                valid_blocks // SCORE_BLOCKS_PER_TILE, stage=2, init_values=(packet_zero,)
                            ):
                                subblock = score_tile * SCORE_BLOCKS_PER_TILE
                                sb = packet_begin + subblock
                                keys_full = pl.cast(pl.load(kc, [sb * BLOCK, 0], [SCORE_ROWS, DI]), pl.FP32)
                                acc = pl.row_sum(
                                    pl.col_expand_mul(keys_full, q_eff),
                                    pl.tile.create([SCORE_ROWS, DI], dtype=pl.FP32),
                                )
                                score_blocks = pl.reshape(acc, [SCORE_BLOCKS_PER_TILE, BLOCK])
                                packet_next = pl.tile.assemble(packet_state, score_blocks, [subblock, 0])
                                full_packet = pl.yield_(packet_next)
                            for tail_tile, (tail_state,) in pl.range(
                                valid_blocks % SCORE_BLOCKS_PER_TILE, init_values=(full_packet,)
                            ):
                                tail_subblock = (
                                    valid_blocks // SCORE_BLOCKS_PER_TILE
                                ) * SCORE_BLOCKS_PER_TILE + tail_tile
                                tail_block = packet_begin + tail_subblock
                                tail_keys = pl.cast(
                                    pl.load(kc, [tail_block * BLOCK, 0], [BLOCK, DI]), pl.FP32
                                )
                                tail_acc = pl.row_sum(
                                    pl.col_expand_mul(tail_keys, q_eff),
                                    pl.tile.create([BLOCK, DI], dtype=pl.FP32),
                                )
                                tail_scores = pl.reshape(tail_acc, [1, BLOCK])
                                tail_next = pl.tile.assemble(tail_state, tail_scores, [tail_subblock, 0])
                                packet_scores = pl.yield_(tail_next)
                            packet_flat = pl.set_validshape(
                                pl.reshape(packet_scores, [1, PACKET_BLOCKS * BLOCK]), 1, valid_blocks * BLOCK
                            )
                            pl.store(packet_flat, [0, packet_begin * BLOCK], ws_scr)
                            packet_maxima = pl.row_max(
                                packet_scores, pl.tile.create([PACKET_BLOCKS, BLOCK], dtype=pl.FP32)
                            )
                            maxima_flat = pl.set_validshape(
                                pl.reshape(packet_maxima, [1, PACKET_BLOCKS]), 1, valid_blocks
                            )
                            pool_zero_packet = pl.tile.full(
                                [1, PACKET_BLOCKS * BLOCK], dtype=pl.FP32, value=0.0
                            )
                            pl.store(
                                pl.set_validshape(pool_zero_packet, 1, valid_blocks * BLOCK),
                                [0, packet_begin * BLOCK],
                                ws_pool,
                            )
                            pl.store(maxima_flat, [0, packet_begin], ws_bm)
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="block_chain_fused",
                    allow_early_resolve=True,
                    deps=[scan_done],
                ) as selection_done:
                    for leaf in pl.range(NBL):
                        valid_blocks = pl.min(LEAF, NB - leaf * LEAF)
                        if BALANCED:
                            logical_ids = pl.add(
                                pl.tile.arange(0, [1, LEAF], dtype=pl.INT32), pl.cast(leaf * LEAF, pl.INT32)
                            )
                            safe_ids = pl.tile.minimums(logical_ids, NB - 1)
                            late_ids = pl.tile.maximums(pl.sub(safe_ids, RANGE_CUTOFF), 0)
                            if FAST_INVERSE:
                                early_owner = pl.cast(
                                    pl.mul(pl.add(pl.cast(safe_ids, pl.FP32), 0.5), EARLY_RECIPROCAL),
                                    pl.INT32,
                                    mode="floor",
                                )
                                late_owner_relative = pl.cast(
                                    pl.mul(pl.add(pl.cast(late_ids, pl.FP32), 0.5), LATE_RECIPROCAL),
                                    pl.INT32,
                                    mode="floor",
                                )
                            else:
                                early_owner_estimate = pl.cast(
                                    pl.mul(pl.cast(safe_ids, pl.FP32), EARLY_RECIPROCAL),
                                    pl.INT32,
                                    mode="round",
                                )
                                early_owner_remainder = pl.sub(
                                    safe_ids, pl.mul(early_owner_estimate, RANGE_BASE + 1)
                                )
                                early_owner_too_high = pl.tile.cmps(
                                    pl.cast(early_owner_remainder, pl.FP32), 0.0, cmp_type=2
                                )
                                early_owner_lowered = pl.tile.select(
                                    early_owner_too_high,
                                    pl.sub(early_owner_estimate, 1),
                                    early_owner_estimate,
                                )
                                early_owner_next_product = pl.mul(
                                    pl.add(early_owner_lowered, 1), RANGE_BASE + 1
                                )
                                early_owner_too_low = pl.tile.cmp(
                                    pl.cast(early_owner_next_product, pl.FP32),
                                    pl.cast(safe_ids, pl.FP32),
                                    cmp_type=3,
                                )
                                early_owner = pl.tile.select(
                                    early_owner_too_low, pl.add(early_owner_lowered, 1), early_owner_lowered
                                )
                                late_owner_relative_estimate = pl.cast(
                                    pl.mul(pl.cast(late_ids, pl.FP32), LATE_RECIPROCAL),
                                    pl.INT32,
                                    mode="round",
                                )
                                late_owner_relative_remainder = pl.sub(
                                    late_ids, pl.mul(late_owner_relative_estimate, RANGE_DIVISOR)
                                )
                                late_owner_relative_too_high = pl.tile.cmps(
                                    pl.cast(late_owner_relative_remainder, pl.FP32), 0.0, cmp_type=2
                                )
                                late_owner_relative_lowered = pl.tile.select(
                                    late_owner_relative_too_high,
                                    pl.sub(late_owner_relative_estimate, 1),
                                    late_owner_relative_estimate,
                                )
                                late_owner_relative_next_product = pl.mul(
                                    pl.add(late_owner_relative_lowered, 1), RANGE_DIVISOR
                                )
                                late_owner_relative_too_low = pl.tile.cmp(
                                    pl.cast(late_owner_relative_next_product, pl.FP32),
                                    pl.cast(late_ids, pl.FP32),
                                    cmp_type=3,
                                )
                                late_owner_relative = pl.tile.select(
                                    late_owner_relative_too_low,
                                    pl.add(late_owner_relative_lowered, 1),
                                    late_owner_relative_lowered,
                                )
                            early_mask = pl.tile.cmps(pl.cast(safe_ids, pl.FP32), RANGE_CUTOFF, cmp_type=2)
                            late_owner = pl.add(late_owner_relative, RANGE_EXTRA)
                            owners = pl.tile.select(early_mask, early_owner, late_owner)
                            early_begin = pl.mul(early_owner, RANGE_BASE + 1)
                            late_begin = pl.add(pl.mul(late_owner, RANGE_BASE), RANGE_EXTRA)
                            owner_begin = pl.tile.select(early_mask, early_begin, late_begin)
                            private_offsets = pl.add(
                                pl.mul(owners, MAXIMA_STRIDE), pl.sub(safe_ids, owner_begin)
                            )
                            private_values = pl.reshape(
                                pl.load(scan_maxima, [0, 0], [ACTIVE_WORKERS, MAXIMA_STRIDE]),
                                [1, ACTIVE_WORKERS * MAXIMA_STRIDE],
                            )
                            gathered_maxima = pl.tile.gather(
                                private_values, private_offsets, pl.tile.create([1, LEAF], dtype=pl.INT32)
                            )
                            reconstructed_values = pl.set_validshape(gathered_maxima, 1, valid_blocks)
                            pl.store(reconstructed_values, [0, leaf * LEAF], ws_bm)
                            block_values = pl.yield_(reconstructed_values)
                        else:
                            original_values = pl.load(
                                ws_bm, [0, leaf * LEAF], [1, LEAF], valid_shape=[1, valid_blocks]
                            )
                            block_values = pl.yield_(original_values)
                        padded_block_values = pl.fillpad(block_values, pad_value=pl.PadValue.min)
                        block_ids = pl.add(
                            pl.tile.arange(0, [1, LEAF], dtype=pl.INT32), pl.cast(leaf * LEAF, pl.INT32)
                        )
                        block_pairs = pl.tile.sort32(
                            padded_block_values, pl.reinterpret_view(block_ids, pl.UINT32)
                        )
                        block_pairs = pl.tile.mrgsort(block_pairs, block_len=64)
                        block_pairs = pl.tile.mrgsort(block_pairs, block_len=256)
                        block_pairs = pl.tile.mrgsort(block_pairs, block_len=1024)
                        pl.store(pl.tile.slice(block_pairs, [1, 2 * TOPB], [0, 0]), [leaf, 0], block_pairs_gm)
                    block_pairs_gm_merge_tmp = pl.tile.create([1, 1024], dtype=pl.FP32)
                    for block_pairs_gm_merge_start in pl.range(1, NBL, 1):
                        block_pairs_gm_left = pl.load(block_pairs_gm, [0, 0], [1, 512])
                        block_pairs_gm_right_0 = pl.load(
                            block_pairs_gm, [block_pairs_gm_merge_start + 0, 0], [1, 512]
                        )
                        block_pairs_gm_merged = pl.tile.mrgsort(
                            block_pairs_gm_left, block_pairs_gm_right_0, tmp=block_pairs_gm_merge_tmp
                        )
                        pl.store(
                            pl.tile.slice(block_pairs_gm_merged, [1, 512], [0, 0]), [0, 0], block_pairs_gm
                        )
                    final_blocks = pl.load(block_pairs_gm, [0, 0], [1, 2 * TOPB])
                    chosen = pl.tile.gather_mask(
                        final_blocks, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                    )
                    pl.store(chosen, [0, 0], selected_blocks)
                with pl.spmd(
                    GATHER_OWNERS,
                    name_hint="selected_dma_gather",
                    allow_early_resolve=True,
                    deps=[selection_done],
                ) as gather_done:
                    gather_owner = pl.tile.get_block_idx()
                    chosen_ids = pl.load(selected_blocks, [0, 0], [1, TOPB])
                    pool_ones = pl.tile.full([1, BLOCK], dtype=pl.FP32, value=1.0)
                    chosen_lines = pl.tile.shrs(chosen_ids, 4)
                    predicate_zero = pl.tile.full([1, TOPB], dtype=pl.FP32, value=0.0)
                    predicate_ones = pl.tile.full([1, TOPB], dtype=pl.FP32, value=1.0)
                    for owner_line, (predicate_state,) in pl.range(
                        gather_owner, MAXIMUM_LINES, GATHER_OWNERS, init_values=(predicate_zero,)
                    ):
                        matches = pl.tile.cmps(chosen_lines, pl.cast(owner_line, pl.INT32), cmp_type=0)
                        match_values = pl.tile.select(matches, predicate_ones, 0.0)
                        owner_predicate = pl.yield_(pl.add(predicate_state, match_values))
                    count_lanes = pl.col_sum(pl.reshape(owner_predicate, [TOPB // 8, 8]))
                    owner_count_i32 = pl.cast(
                        pl.read(count_lanes, [0, 0])
                        + pl.read(count_lanes, [0, 1])
                        + pl.read(count_lanes, [0, 2])
                        + pl.read(count_lanes, [0, 3])
                        + pl.read(count_lanes, [0, 4])
                        + pl.read(count_lanes, [0, 5])
                        + pl.read(count_lanes, [0, 6])
                        + pl.read(count_lanes, [0, 7]),
                        pl.INT32,
                    )
                    owner_count = pl.cast(owner_count_i32, pl.INDEX)
                    original_ranks = pl.tile.arange(0, [1, TOPB], dtype=pl.INT32)
                    local_pairs = pl.tile.sort32(
                        owner_predicate, pl.reinterpret_view(original_ranks, pl.UINT32)
                    )
                    local_pairs = pl.tile.mrgsort(local_pairs, block_len=64)
                    local_pairs_left = pl.tile.slice(local_pairs, [1, TOPB], [0, 0])
                    local_pairs_right = pl.tile.slice(local_pairs, [1, TOPB], [0, TOPB])
                    local_pairs_tmp = pl.tile.create([1, 2 * TOPB], dtype=pl.FP32)
                    local_pairs = pl.tile.mrgsort(local_pairs_left, local_pairs_right, tmp=local_pairs_tmp)
                    local_ranks = pl.tile.gather_mask(
                        local_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                    )
                    for work_item in pl.range(owner_count):
                        selected = pl.cast(pl.read(local_ranks, [0, work_item]), pl.INDEX)
                        block_id = pl.read(chosen_ids, [0, selected])
                        block_offset = pl.cast(block_id, pl.INDEX)
                        offset = block_offset * BLOCK
                        original_maximum = pl.read(ws_bm, [0, block_offset])
                        pl.write(ws_bm, [0, block_offset], original_maximum - BIG)
                        pl.store(pool_ones, [0, offset], ws_pool)
                with pl.spmd(
                    NSL, name_hint="buffered_score_leaves", allow_early_resolve=True, deps=[selection_done]
                ) as score_leaves_done:
                    leaf = pl.tile.get_block_idx()
                    chosen_score_ids = pl.load(selected_blocks, [0, 0], [1, TOPB])
                    leaf_zero_values = pl.tile.full([LEAF // BLOCK, BLOCK], dtype=pl.FP32, value=0.0)
                    leaf_zero_ids = pl.tile.full([LEAF // BLOCK, BLOCK], dtype=pl.INT32, value=0)
                    for score_bundle, (value_state, id_state) in pl.range(
                        (LEAF // BLOCK) // 4, init_values=(leaf_zero_values, leaf_zero_ids)
                    ):
                        score_rank_0 = leaf * (LEAF // BLOCK) + score_bundle * 4 + 0
                        score_block_0 = pl.read(chosen_score_ids, [0, score_rank_0])
                        score_offset_0 = pl.cast(score_block_0, pl.INDEX) * BLOCK
                        score_values_0 = pl.load(ws_scr, [0, score_offset_0], [1, BLOCK])
                        score_rank_1 = leaf * (LEAF // BLOCK) + score_bundle * 4 + 1
                        score_block_1 = pl.read(chosen_score_ids, [0, score_rank_1])
                        score_offset_1 = pl.cast(score_block_1, pl.INDEX) * BLOCK
                        score_values_1 = pl.load(ws_scr, [0, score_offset_1], [1, BLOCK])
                        score_rank_2 = leaf * (LEAF // BLOCK) + score_bundle * 4 + 2
                        score_block_2 = pl.read(chosen_score_ids, [0, score_rank_2])
                        score_offset_2 = pl.cast(score_block_2, pl.INDEX) * BLOCK
                        score_values_2 = pl.load(ws_scr, [0, score_offset_2], [1, BLOCK])
                        score_rank_3 = leaf * (LEAF // BLOCK) + score_bundle * 4 + 3
                        score_block_3 = pl.read(chosen_score_ids, [0, score_rank_3])
                        score_offset_3 = pl.cast(score_block_3, pl.INDEX) * BLOCK
                        score_values_3 = pl.load(ws_scr, [0, score_offset_3], [1, BLOCK])
                        score_ids_0 = pl.add(
                            pl.tile.arange(0, [1, BLOCK], dtype=pl.INT32),
                            pl.cast(score_block_0 * BLOCK, pl.INT32),
                        )
                        assembled_values_0 = pl.tile.assemble(
                            value_state, score_values_0, [score_bundle * 4 + 0, 0]
                        )
                        assembled_ids_0 = pl.tile.assemble(id_state, score_ids_0, [score_bundle * 4 + 0, 0])
                        score_ids_1 = pl.add(
                            pl.tile.arange(0, [1, BLOCK], dtype=pl.INT32),
                            pl.cast(score_block_1 * BLOCK, pl.INT32),
                        )
                        assembled_values_1 = pl.tile.assemble(
                            assembled_values_0, score_values_1, [score_bundle * 4 + 1, 0]
                        )
                        assembled_ids_1 = pl.tile.assemble(
                            assembled_ids_0, score_ids_1, [score_bundle * 4 + 1, 0]
                        )
                        score_ids_2 = pl.add(
                            pl.tile.arange(0, [1, BLOCK], dtype=pl.INT32),
                            pl.cast(score_block_2 * BLOCK, pl.INT32),
                        )
                        assembled_values_2 = pl.tile.assemble(
                            assembled_values_1, score_values_2, [score_bundle * 4 + 2, 0]
                        )
                        assembled_ids_2 = pl.tile.assemble(
                            assembled_ids_1, score_ids_2, [score_bundle * 4 + 2, 0]
                        )
                        score_ids_3 = pl.add(
                            pl.tile.arange(0, [1, BLOCK], dtype=pl.INT32),
                            pl.cast(score_block_3 * BLOCK, pl.INT32),
                        )
                        assembled_values_3 = pl.tile.assemble(
                            assembled_values_2, score_values_3, [score_bundle * 4 + 3, 0]
                        )
                        assembled_ids_3 = pl.tile.assemble(
                            assembled_ids_2, score_ids_3, [score_bundle * 4 + 3, 0]
                        )
                        vals, ids = pl.yield_(assembled_values_3, assembled_ids_3)
                    pairs = pl.tile.sort32(
                        pl.reshape(vals, [1, LEAF]),
                        pl.reinterpret_view(pl.reshape(ids, [1, LEAF]), pl.UINT32),
                    )
                    pairs = pl.tile.mrgsort(pairs, block_len=64)
                    pairs = pl.tile.mrgsort(pairs, block_len=256)
                    pairs = pl.tile.mrgsort(pairs, block_len=1024)
                    pl.store(pl.tile.slice(pairs, [1, 2 * TOPK], [0, 0]), [leaf, 0], score_pairs_gm)
                if LATE_JOIN:
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="score_prefix_merge",
                        allow_early_resolve=False,
                        deps=[score_leaves_done],
                    ) as score_merge_done:
                        score_pairs_gm_merge_tmp = pl.tile.create([1, 2048], dtype=pl.FP32)
                        for score_pairs_gm_merge_start in pl.range(1, NSL, 1):
                            score_pairs_gm_left = pl.load(score_pairs_gm, [0, 0], [1, 1024])
                            score_pairs_gm_right_0 = pl.load(
                                score_pairs_gm, [score_pairs_gm_merge_start + 0, 0], [1, 1024]
                            )
                            score_pairs_gm_merged = pl.tile.mrgsort(
                                score_pairs_gm_left, score_pairs_gm_right_0, tmp=score_pairs_gm_merge_tmp
                            )
                            pl.store(
                                pl.tile.slice(score_pairs_gm_merged, [1, 1024], [0, 0]),
                                [0, 0],
                                score_pairs_gm,
                            )
                        final = pl.load(score_pairs_gm, [0, 0], [1, 2 * TOPK])
                        values = pl.tile.gather_mask(final, mask_pattern=pl.tile.MaskPattern.P0101)
                        indices = pl.tile.gather_mask(
                            final, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                        )
                        pl.store(values, [0, 0], top_val)
                        pl.store(indices, [0, 0], top_idx)
                    invocation_done = pl.system.task_dummy(deps=[score_merge_done, gather_done])
                else:
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="score_prefix_merge",
                        allow_early_resolve=False,
                        deps=[score_leaves_done, gather_done],
                    ):
                        score_pairs_gm_merge_tmp = pl.tile.create([1, 2048], dtype=pl.FP32)
                        for score_pairs_gm_merge_start in pl.range(1, NSL, 1):
                            score_pairs_gm_left = pl.load(score_pairs_gm, [0, 0], [1, 1024])
                            score_pairs_gm_right_0 = pl.load(
                                score_pairs_gm, [score_pairs_gm_merge_start + 0, 0], [1, 1024]
                            )
                            score_pairs_gm_merged = pl.tile.mrgsort(
                                score_pairs_gm_left, score_pairs_gm_right_0, tmp=score_pairs_gm_merge_tmp
                            )
                            pl.store(
                                pl.tile.slice(score_pairs_gm_merged, [1, 1024], [0, 0]),
                                [0, 0],
                                score_pairs_gm,
                            )
                        final = pl.load(score_pairs_gm, [0, 0], [1, 2 * TOPK])
                        values = pl.tile.gather_mask(final, mask_pattern=pl.tile.MaskPattern.P0101)
                        indices = pl.tile.gather_mask(
                            final, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                        )
                        pl.store(values, [0, 0], top_val)
                        pl.store(indices, [0, 0], top_idx)
                return top_val
        else:

            @pl.jit
            def hierindex(
                q: pl.Tensor[[H, DI], input_dtype],
                kc: pl.Tensor[[S, DI], input_dtype],
                w: pl.Tensor[[H], input_dtype],
                ws_scr: pl.InOut[pl.Tensor[[1, S], pl.FP32]],
                ws_pool: pl.InOut[pl.Tensor[[1, S], pl.FP32]],
                ws_bm: pl.InOut[pl.Tensor[[1, NB], pl.FP32]],
                top_val: pl.Out[pl.Tensor[[1, TOPK], pl.FP32]],
                top_idx: pl.Out[pl.Tensor[[1, TOPK], pl.INT32]],
            ):
                scan_maxima = pl.create_tensor([ACTIVE_WORKERS, MAXIMA_STRIDE], dtype=pl.FP32)
                selected_blocks = pl.create_tensor([1, TOPB], dtype=pl.INT32)
                block_pairs_gm = pl.create_tensor([NBL, 2 * TOPB], dtype=pl.FP32)
                score_pairs_gm = pl.create_tensor([NSL, 2 * TOPK], dtype=pl.FP32)
                block_remaining = pl.create_tensor([1, BPAD], dtype=pl.FP32)
                with pl.spmd(
                    ACTIVE_WORKERS, name_hint="scan_score_packets", allow_early_resolve=True
                ) as scan_done:
                    worker = pl.tile.get_block_idx()
                    weights_fp32 = pl.reshape(pl.load(w, [0], [16], valid_shape=[H]), [1, 16])
                    weighted_q_0 = pl.mul(pl.load(q, [0, 0], [1, DI]), pl.read(weights_fp32, [0, 0]))
                    weighted_q_1 = pl.mul(pl.load(q, [1, 0], [1, DI]), pl.read(weights_fp32, [0, 1]))
                    weighted_q_2 = pl.mul(pl.load(q, [2, 0], [1, DI]), pl.read(weights_fp32, [0, 2]))
                    weighted_q_3 = pl.mul(pl.load(q, [3, 0], [1, DI]), pl.read(weights_fp32, [0, 3]))
                    weighted_q_4 = pl.mul(pl.load(q, [4, 0], [1, DI]), pl.read(weights_fp32, [0, 4]))
                    weighted_q_5 = pl.mul(pl.load(q, [5, 0], [1, DI]), pl.read(weights_fp32, [0, 5]))
                    weighted_q_6 = pl.mul(pl.load(q, [6, 0], [1, DI]), pl.read(weights_fp32, [0, 6]))
                    weighted_q_7 = pl.mul(pl.load(q, [7, 0], [1, DI]), pl.read(weights_fp32, [0, 7]))
                    q_pair_0 = pl.add(weighted_q_0, weighted_q_1)
                    q_pair_1 = pl.add(weighted_q_2, weighted_q_3)
                    q_pair_2 = pl.add(weighted_q_4, weighted_q_5)
                    q_pair_3 = pl.add(weighted_q_6, weighted_q_7)
                    q_eff = pl.add(pl.add(q_pair_0, q_pair_1), pl.add(q_pair_2, q_pair_3))
                    if BALANCED:
                        begin_block = worker * RANGE_BASE + pl.min(worker, RANGE_EXTRA)
                        owned_blocks = RANGE_BASE + pl.min(1, pl.max(0, RANGE_EXTRA - worker))
                        private_zero = pl.tile.full(
                            [1, MAXIMA_STRIDE], dtype=pl.FP32, value=-3.4028234663852886e38
                        )
                        for local_packet, (private_state,) in pl.range(
                            (owned_blocks + PACKET_BLOCKS - 1) // PACKET_BLOCKS, init_values=(private_zero,)
                        ):
                            packet_begin = begin_block + local_packet * PACKET_BLOCKS
                            valid_blocks = pl.min(PACKET_BLOCKS, owned_blocks - local_packet * PACKET_BLOCKS)
                            packet_zero = pl.tile.full(
                                [PACKET_BLOCKS, BLOCK], dtype=pl.FP32, value=-3.4028234663852886e38
                            )
                            for score_tile, (packet_state,) in pl.range(
                                (valid_blocks + SCORE_BLOCKS_PER_TILE - 1) // SCORE_BLOCKS_PER_TILE,
                                init_values=(packet_zero,),
                            ):
                                subblock = score_tile * SCORE_BLOCKS_PER_TILE
                                sb = packet_begin + subblock
                                valid_rows = pl.min(SCORE_ROWS, (valid_blocks - subblock) * BLOCK)
                                kblk = pl.load(
                                    kc, [sb * BLOCK, 0], [SCORE_ROWS, DI], valid_shape=[valid_rows, DI]
                                )
                                keys_neutral = pl.fillpad(kblk, pad_value=pl.PadValue.zero)
                                keys_full = pl.set_validshape(keys_neutral, SCORE_ROWS, DI)
                                acc = pl.row_sum(
                                    pl.col_expand_mul(keys_full, q_eff),
                                    pl.tile.create([SCORE_ROWS, DI], dtype=pl.FP32),
                                )
                                score_blocks = pl.set_validshape(
                                    pl.reshape(acc, [SCORE_BLOCKS_PER_TILE, BLOCK]),
                                    valid_rows // BLOCK,
                                    BLOCK,
                                )
                                packet_next = pl.tile.assemble(packet_state, score_blocks, [subblock, 0])
                                packet_scores = pl.yield_(packet_next)
                            packet_flat = pl.set_validshape(
                                pl.reshape(packet_scores, [1, PACKET_BLOCKS * BLOCK]), 1, valid_blocks * BLOCK
                            )
                            pl.store(packet_flat, [0, packet_begin * BLOCK], ws_scr)
                            packet_maxima = pl.row_max(
                                packet_scores, pl.tile.create([PACKET_BLOCKS, BLOCK], dtype=pl.FP32)
                            )
                            maxima_flat = pl.set_validshape(
                                pl.reshape(packet_maxima, [1, PACKET_BLOCKS]), 1, valid_blocks
                            )
                            pool_zero_packet = pl.tile.full(
                                [1, PACKET_BLOCKS * BLOCK], dtype=pl.FP32, value=0.0
                            )
                            pl.store(
                                pl.set_validshape(pool_zero_packet, 1, valid_blocks * BLOCK),
                                [0, packet_begin * BLOCK],
                                ws_pool,
                            )
                            private_next = pl.tile.assemble(
                                private_state, maxima_flat, [0, local_packet * PACKET_BLOCKS]
                            )
                            private_maxima = pl.yield_(private_next)
                        pl.store(private_maxima, [worker, 0], scan_maxima)
                    else:
                        for packet in pl.range(worker, PACKETS, ACTIVE_WORKERS):
                            packet_begin = packet * PACKET_BLOCKS
                            valid_blocks = pl.min(PACKET_BLOCKS, NB - packet_begin)
                            packet_zero = pl.tile.full(
                                [PACKET_BLOCKS, BLOCK], dtype=pl.FP32, value=-3.4028234663852886e38
                            )
                            for score_tile, (packet_state,) in pl.range(
                                (valid_blocks + SCORE_BLOCKS_PER_TILE - 1) // SCORE_BLOCKS_PER_TILE,
                                init_values=(packet_zero,),
                            ):
                                subblock = score_tile * SCORE_BLOCKS_PER_TILE
                                sb = packet_begin + subblock
                                valid_rows = pl.min(SCORE_ROWS, (valid_blocks - subblock) * BLOCK)
                                kblk = pl.load(
                                    kc, [sb * BLOCK, 0], [SCORE_ROWS, DI], valid_shape=[valid_rows, DI]
                                )
                                keys_neutral = pl.fillpad(kblk, pad_value=pl.PadValue.zero)
                                keys_full = pl.set_validshape(keys_neutral, SCORE_ROWS, DI)
                                acc = pl.row_sum(
                                    pl.col_expand_mul(keys_full, q_eff),
                                    pl.tile.create([SCORE_ROWS, DI], dtype=pl.FP32),
                                )
                                score_blocks = pl.set_validshape(
                                    pl.reshape(acc, [SCORE_BLOCKS_PER_TILE, BLOCK]),
                                    valid_rows // BLOCK,
                                    BLOCK,
                                )
                                packet_next = pl.tile.assemble(packet_state, score_blocks, [subblock, 0])
                                packet_scores = pl.yield_(packet_next)
                            packet_flat = pl.set_validshape(
                                pl.reshape(packet_scores, [1, PACKET_BLOCKS * BLOCK]), 1, valid_blocks * BLOCK
                            )
                            pl.store(packet_flat, [0, packet_begin * BLOCK], ws_scr)
                            packet_maxima = pl.row_max(
                                packet_scores, pl.tile.create([PACKET_BLOCKS, BLOCK], dtype=pl.FP32)
                            )
                            maxima_flat = pl.set_validshape(
                                pl.reshape(packet_maxima, [1, PACKET_BLOCKS]), 1, valid_blocks
                            )
                            pool_zero_packet = pl.tile.full(
                                [1, PACKET_BLOCKS * BLOCK], dtype=pl.FP32, value=0.0
                            )
                            pl.store(
                                pl.set_validshape(pool_zero_packet, 1, valid_blocks * BLOCK),
                                [0, packet_begin * BLOCK],
                                ws_pool,
                            )
                            pl.store(maxima_flat, [0, packet_begin], ws_bm)
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="block_chain_fused",
                    allow_early_resolve=True,
                    deps=[scan_done],
                ) as selection_done:
                    for leaf in pl.range(NBL):
                        valid_blocks = pl.min(LEAF, NB - leaf * LEAF)
                        if BALANCED:
                            logical_ids = pl.add(
                                pl.tile.arange(0, [1, LEAF], dtype=pl.INT32), pl.cast(leaf * LEAF, pl.INT32)
                            )
                            safe_ids = pl.tile.minimums(logical_ids, NB - 1)
                            late_ids = pl.tile.maximums(pl.sub(safe_ids, RANGE_CUTOFF), 0)
                            if FAST_INVERSE:
                                early_owner = pl.cast(
                                    pl.mul(pl.add(pl.cast(safe_ids, pl.FP32), 0.5), EARLY_RECIPROCAL),
                                    pl.INT32,
                                    mode="floor",
                                )
                                late_owner_relative = pl.cast(
                                    pl.mul(pl.add(pl.cast(late_ids, pl.FP32), 0.5), LATE_RECIPROCAL),
                                    pl.INT32,
                                    mode="floor",
                                )
                            else:
                                early_owner_estimate = pl.cast(
                                    pl.mul(pl.cast(safe_ids, pl.FP32), EARLY_RECIPROCAL),
                                    pl.INT32,
                                    mode="round",
                                )
                                early_owner_remainder = pl.sub(
                                    safe_ids, pl.mul(early_owner_estimate, RANGE_BASE + 1)
                                )
                                early_owner_too_high = pl.tile.cmps(
                                    pl.cast(early_owner_remainder, pl.FP32), 0.0, cmp_type=2
                                )
                                early_owner_lowered = pl.tile.select(
                                    early_owner_too_high,
                                    pl.sub(early_owner_estimate, 1),
                                    early_owner_estimate,
                                )
                                early_owner_next_product = pl.mul(
                                    pl.add(early_owner_lowered, 1), RANGE_BASE + 1
                                )
                                early_owner_too_low = pl.tile.cmp(
                                    pl.cast(early_owner_next_product, pl.FP32),
                                    pl.cast(safe_ids, pl.FP32),
                                    cmp_type=3,
                                )
                                early_owner = pl.tile.select(
                                    early_owner_too_low, pl.add(early_owner_lowered, 1), early_owner_lowered
                                )
                                late_owner_relative_estimate = pl.cast(
                                    pl.mul(pl.cast(late_ids, pl.FP32), LATE_RECIPROCAL),
                                    pl.INT32,
                                    mode="round",
                                )
                                late_owner_relative_remainder = pl.sub(
                                    late_ids, pl.mul(late_owner_relative_estimate, RANGE_DIVISOR)
                                )
                                late_owner_relative_too_high = pl.tile.cmps(
                                    pl.cast(late_owner_relative_remainder, pl.FP32), 0.0, cmp_type=2
                                )
                                late_owner_relative_lowered = pl.tile.select(
                                    late_owner_relative_too_high,
                                    pl.sub(late_owner_relative_estimate, 1),
                                    late_owner_relative_estimate,
                                )
                                late_owner_relative_next_product = pl.mul(
                                    pl.add(late_owner_relative_lowered, 1), RANGE_DIVISOR
                                )
                                late_owner_relative_too_low = pl.tile.cmp(
                                    pl.cast(late_owner_relative_next_product, pl.FP32),
                                    pl.cast(late_ids, pl.FP32),
                                    cmp_type=3,
                                )
                                late_owner_relative = pl.tile.select(
                                    late_owner_relative_too_low,
                                    pl.add(late_owner_relative_lowered, 1),
                                    late_owner_relative_lowered,
                                )
                            early_mask = pl.tile.cmps(pl.cast(safe_ids, pl.FP32), RANGE_CUTOFF, cmp_type=2)
                            late_owner = pl.add(late_owner_relative, RANGE_EXTRA)
                            owners = pl.tile.select(early_mask, early_owner, late_owner)
                            early_begin = pl.mul(early_owner, RANGE_BASE + 1)
                            late_begin = pl.add(pl.mul(late_owner, RANGE_BASE), RANGE_EXTRA)
                            owner_begin = pl.tile.select(early_mask, early_begin, late_begin)
                            private_offsets = pl.add(
                                pl.mul(owners, MAXIMA_STRIDE), pl.sub(safe_ids, owner_begin)
                            )
                            private_values = pl.reshape(
                                pl.load(scan_maxima, [0, 0], [ACTIVE_WORKERS, MAXIMA_STRIDE]),
                                [1, ACTIVE_WORKERS * MAXIMA_STRIDE],
                            )
                            gathered_maxima = pl.tile.gather(
                                private_values, private_offsets, pl.tile.create([1, LEAF], dtype=pl.INT32)
                            )
                            reconstructed_values = pl.set_validshape(gathered_maxima, 1, valid_blocks)
                            pl.store(reconstructed_values, [0, leaf * LEAF], ws_bm)
                            block_values = pl.yield_(reconstructed_values)
                        else:
                            original_values = pl.load(
                                ws_bm, [0, leaf * LEAF], [1, LEAF], valid_shape=[1, valid_blocks]
                            )
                            block_values = pl.yield_(original_values)
                        padded_block_values = pl.fillpad(block_values, pad_value=pl.PadValue.min)
                        block_ids = pl.add(
                            pl.tile.arange(0, [1, LEAF], dtype=pl.INT32), pl.cast(leaf * LEAF, pl.INT32)
                        )
                        block_pairs = pl.tile.sort32(
                            padded_block_values, pl.reinterpret_view(block_ids, pl.UINT32)
                        )
                        block_pairs = pl.tile.mrgsort(block_pairs, block_len=64)
                        block_pairs = pl.tile.mrgsort(block_pairs, block_len=256)
                        block_pairs = pl.tile.mrgsort(block_pairs, block_len=1024)
                        pl.store(pl.tile.slice(block_pairs, [1, 2 * TOPB], [0, 0]), [leaf, 0], block_pairs_gm)
                    block_pairs_gm_merge_tmp = pl.tile.create([1, 1024], dtype=pl.FP32)
                    for block_pairs_gm_merge_start in pl.range(1, NBL, 1):
                        block_pairs_gm_left = pl.load(block_pairs_gm, [0, 0], [1, 512])
                        block_pairs_gm_right_0 = pl.load(
                            block_pairs_gm, [block_pairs_gm_merge_start + 0, 0], [1, 512]
                        )
                        block_pairs_gm_merged = pl.tile.mrgsort(
                            block_pairs_gm_left, block_pairs_gm_right_0, tmp=block_pairs_gm_merge_tmp
                        )
                        pl.store(
                            pl.tile.slice(block_pairs_gm_merged, [1, 512], [0, 0]), [0, 0], block_pairs_gm
                        )
                    final_blocks = pl.load(block_pairs_gm, [0, 0], [1, 2 * TOPB])
                    chosen = pl.tile.gather_mask(
                        final_blocks, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                    )
                    pl.store(chosen, [0, 0], selected_blocks)
                with pl.spmd(
                    GATHER_OWNERS,
                    name_hint="selected_dma_gather",
                    allow_early_resolve=True,
                    deps=[selection_done],
                ) as gather_done:
                    gather_owner = pl.tile.get_block_idx()
                    chosen_ids = pl.load(selected_blocks, [0, 0], [1, TOPB])
                    pool_ones = pl.tile.full([1, BLOCK], dtype=pl.FP32, value=1.0)
                    chosen_lines = pl.tile.shrs(chosen_ids, 4)
                    predicate_zero = pl.tile.full([1, TOPB], dtype=pl.FP32, value=0.0)
                    predicate_ones = pl.tile.full([1, TOPB], dtype=pl.FP32, value=1.0)
                    for owner_line, (predicate_state,) in pl.range(
                        gather_owner, MAXIMUM_LINES, GATHER_OWNERS, init_values=(predicate_zero,)
                    ):
                        matches = pl.tile.cmps(chosen_lines, pl.cast(owner_line, pl.INT32), cmp_type=0)
                        match_values = pl.tile.select(matches, predicate_ones, 0.0)
                        owner_predicate = pl.yield_(pl.add(predicate_state, match_values))
                    count_lanes = pl.col_sum(pl.reshape(owner_predicate, [TOPB // 8, 8]))
                    owner_count_i32 = pl.cast(
                        pl.read(count_lanes, [0, 0])
                        + pl.read(count_lanes, [0, 1])
                        + pl.read(count_lanes, [0, 2])
                        + pl.read(count_lanes, [0, 3])
                        + pl.read(count_lanes, [0, 4])
                        + pl.read(count_lanes, [0, 5])
                        + pl.read(count_lanes, [0, 6])
                        + pl.read(count_lanes, [0, 7]),
                        pl.INT32,
                    )
                    owner_count = pl.cast(owner_count_i32, pl.INDEX)
                    original_ranks = pl.tile.arange(0, [1, TOPB], dtype=pl.INT32)
                    local_pairs = pl.tile.sort32(
                        owner_predicate, pl.reinterpret_view(original_ranks, pl.UINT32)
                    )
                    local_pairs = pl.tile.mrgsort(local_pairs, block_len=64)
                    local_pairs_left = pl.tile.slice(local_pairs, [1, TOPB], [0, 0])
                    local_pairs_right = pl.tile.slice(local_pairs, [1, TOPB], [0, TOPB])
                    local_pairs_tmp = pl.tile.create([1, 2 * TOPB], dtype=pl.FP32)
                    local_pairs = pl.tile.mrgsort(local_pairs_left, local_pairs_right, tmp=local_pairs_tmp)
                    local_ranks = pl.tile.gather_mask(
                        local_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                    )
                    for work_item in pl.range(owner_count):
                        selected = pl.cast(pl.read(local_ranks, [0, work_item]), pl.INDEX)
                        block_id = pl.read(chosen_ids, [0, selected])
                        block_offset = pl.cast(block_id, pl.INDEX)
                        offset = block_offset * BLOCK
                        original_maximum = pl.read(ws_bm, [0, block_offset])
                        pl.write(ws_bm, [0, block_offset], original_maximum - BIG)
                        pl.store(pool_ones, [0, offset], ws_pool)
                with pl.spmd(
                    NSL, name_hint="buffered_score_leaves", allow_early_resolve=True, deps=[selection_done]
                ) as score_leaves_done:
                    leaf = pl.tile.get_block_idx()
                    chosen_score_ids = pl.load(selected_blocks, [0, 0], [1, TOPB])
                    leaf_zero_values = pl.tile.full([LEAF // BLOCK, BLOCK], dtype=pl.FP32, value=0.0)
                    leaf_zero_ids = pl.tile.full([LEAF // BLOCK, BLOCK], dtype=pl.INT32, value=0)
                    for score_bundle, (value_state, id_state) in pl.range(
                        (LEAF // BLOCK) // 4, init_values=(leaf_zero_values, leaf_zero_ids)
                    ):
                        score_rank_0 = leaf * (LEAF // BLOCK) + score_bundle * 4 + 0
                        score_block_0 = pl.read(chosen_score_ids, [0, score_rank_0])
                        score_offset_0 = pl.cast(score_block_0, pl.INDEX) * BLOCK
                        score_values_0 = pl.load(ws_scr, [0, score_offset_0], [1, BLOCK])
                        score_rank_1 = leaf * (LEAF // BLOCK) + score_bundle * 4 + 1
                        score_block_1 = pl.read(chosen_score_ids, [0, score_rank_1])
                        score_offset_1 = pl.cast(score_block_1, pl.INDEX) * BLOCK
                        score_values_1 = pl.load(ws_scr, [0, score_offset_1], [1, BLOCK])
                        score_rank_2 = leaf * (LEAF // BLOCK) + score_bundle * 4 + 2
                        score_block_2 = pl.read(chosen_score_ids, [0, score_rank_2])
                        score_offset_2 = pl.cast(score_block_2, pl.INDEX) * BLOCK
                        score_values_2 = pl.load(ws_scr, [0, score_offset_2], [1, BLOCK])
                        score_rank_3 = leaf * (LEAF // BLOCK) + score_bundle * 4 + 3
                        score_block_3 = pl.read(chosen_score_ids, [0, score_rank_3])
                        score_offset_3 = pl.cast(score_block_3, pl.INDEX) * BLOCK
                        score_values_3 = pl.load(ws_scr, [0, score_offset_3], [1, BLOCK])
                        score_ids_0 = pl.add(
                            pl.tile.arange(0, [1, BLOCK], dtype=pl.INT32),
                            pl.cast(score_block_0 * BLOCK, pl.INT32),
                        )
                        assembled_values_0 = pl.tile.assemble(
                            value_state, score_values_0, [score_bundle * 4 + 0, 0]
                        )
                        assembled_ids_0 = pl.tile.assemble(id_state, score_ids_0, [score_bundle * 4 + 0, 0])
                        score_ids_1 = pl.add(
                            pl.tile.arange(0, [1, BLOCK], dtype=pl.INT32),
                            pl.cast(score_block_1 * BLOCK, pl.INT32),
                        )
                        assembled_values_1 = pl.tile.assemble(
                            assembled_values_0, score_values_1, [score_bundle * 4 + 1, 0]
                        )
                        assembled_ids_1 = pl.tile.assemble(
                            assembled_ids_0, score_ids_1, [score_bundle * 4 + 1, 0]
                        )
                        score_ids_2 = pl.add(
                            pl.tile.arange(0, [1, BLOCK], dtype=pl.INT32),
                            pl.cast(score_block_2 * BLOCK, pl.INT32),
                        )
                        assembled_values_2 = pl.tile.assemble(
                            assembled_values_1, score_values_2, [score_bundle * 4 + 2, 0]
                        )
                        assembled_ids_2 = pl.tile.assemble(
                            assembled_ids_1, score_ids_2, [score_bundle * 4 + 2, 0]
                        )
                        score_ids_3 = pl.add(
                            pl.tile.arange(0, [1, BLOCK], dtype=pl.INT32),
                            pl.cast(score_block_3 * BLOCK, pl.INT32),
                        )
                        assembled_values_3 = pl.tile.assemble(
                            assembled_values_2, score_values_3, [score_bundle * 4 + 3, 0]
                        )
                        assembled_ids_3 = pl.tile.assemble(
                            assembled_ids_2, score_ids_3, [score_bundle * 4 + 3, 0]
                        )
                        vals, ids = pl.yield_(assembled_values_3, assembled_ids_3)
                    pairs = pl.tile.sort32(
                        pl.reshape(vals, [1, LEAF]),
                        pl.reinterpret_view(pl.reshape(ids, [1, LEAF]), pl.UINT32),
                    )
                    pairs = pl.tile.mrgsort(pairs, block_len=64)
                    pairs = pl.tile.mrgsort(pairs, block_len=256)
                    pairs = pl.tile.mrgsort(pairs, block_len=1024)
                    pl.store(pl.tile.slice(pairs, [1, 2 * TOPK], [0, 0]), [leaf, 0], score_pairs_gm)
                if LATE_JOIN:
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="score_prefix_merge",
                        allow_early_resolve=False,
                        deps=[score_leaves_done],
                    ) as score_merge_done:
                        score_pairs_gm_merge_tmp = pl.tile.create([1, 2048], dtype=pl.FP32)
                        for score_pairs_gm_merge_start in pl.range(1, NSL, 1):
                            score_pairs_gm_left = pl.load(score_pairs_gm, [0, 0], [1, 1024])
                            score_pairs_gm_right_0 = pl.load(
                                score_pairs_gm, [score_pairs_gm_merge_start + 0, 0], [1, 1024]
                            )
                            score_pairs_gm_merged = pl.tile.mrgsort(
                                score_pairs_gm_left, score_pairs_gm_right_0, tmp=score_pairs_gm_merge_tmp
                            )
                            pl.store(
                                pl.tile.slice(score_pairs_gm_merged, [1, 1024], [0, 0]),
                                [0, 0],
                                score_pairs_gm,
                            )
                        final = pl.load(score_pairs_gm, [0, 0], [1, 2 * TOPK])
                        values = pl.tile.gather_mask(final, mask_pattern=pl.tile.MaskPattern.P0101)
                        indices = pl.tile.gather_mask(
                            final, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                        )
                        pl.store(values, [0, 0], top_val)
                        pl.store(indices, [0, 0], top_idx)
                    invocation_done = pl.system.task_dummy(deps=[score_merge_done, gather_done])
                else:
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="score_prefix_merge",
                        allow_early_resolve=False,
                        deps=[score_leaves_done, gather_done],
                    ):
                        score_pairs_gm_merge_tmp = pl.tile.create([1, 2048], dtype=pl.FP32)
                        for score_pairs_gm_merge_start in pl.range(1, NSL, 1):
                            score_pairs_gm_left = pl.load(score_pairs_gm, [0, 0], [1, 1024])
                            score_pairs_gm_right_0 = pl.load(
                                score_pairs_gm, [score_pairs_gm_merge_start + 0, 0], [1, 1024]
                            )
                            score_pairs_gm_merged = pl.tile.mrgsort(
                                score_pairs_gm_left, score_pairs_gm_right_0, tmp=score_pairs_gm_merge_tmp
                            )
                            pl.store(
                                pl.tile.slice(score_pairs_gm_merged, [1, 1024], [0, 0]),
                                [0, 0],
                                score_pairs_gm,
                            )
                        final = pl.load(score_pairs_gm, [0, 0], [1, 2 * TOPK])
                        values = pl.tile.gather_mask(final, mask_pattern=pl.tile.MaskPattern.P0101)
                        indices = pl.tile.gather_mask(
                            final, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                        )
                        pl.store(values, [0, 0], top_val)
                        pl.store(indices, [0, 0], top_idx)
                return top_val
    else:
        if dtype_name == "bf16":

            @pl.jit
            def hierindex(
                q: pl.Tensor[[H, DI], input_dtype],
                kc: pl.Tensor[[S, DI], input_dtype],
                w: pl.Tensor[[H], input_dtype],
                ws_scr: pl.InOut[pl.Tensor[[1, S], pl.FP32]],
                ws_pool: pl.InOut[pl.Tensor[[1, S], pl.FP32]],
                ws_bm: pl.InOut[pl.Tensor[[1, NB], pl.FP32]],
                top_val: pl.Out[pl.Tensor[[1, TOPK], pl.FP32]],
                top_idx: pl.Out[pl.Tensor[[1, TOPK], pl.INT32]],
            ):
                scan_maxima = pl.create_tensor([ACTIVE_WORKERS, MAXIMA_STRIDE], dtype=pl.FP32)
                selected_blocks = pl.create_tensor([1, TOPB], dtype=pl.INT32)
                block_pairs_gm = pl.create_tensor([NBL, 2 * TOPB], dtype=pl.FP32)
                score_pairs_gm = pl.create_tensor([NSL, 2 * TOPK], dtype=pl.FP32)
                block_remaining = pl.create_tensor([1, BPAD], dtype=pl.FP32)
                with pl.spmd(
                    ACTIVE_WORKERS, name_hint="scan_score_packets", allow_early_resolve=True
                ) as scan_done:
                    worker = pl.tile.get_block_idx()
                    weights_fp32 = pl.cast(
                        pl.reshape(pl.load(w, [0], [16], valid_shape=[H]), [1, 16]), pl.FP32
                    )
                    weighted_q_0 = pl.mul(
                        pl.cast(pl.load(q, [0, 0], [1, DI]), pl.FP32), pl.read(weights_fp32, [0, 0])
                    )
                    weighted_q_1 = pl.mul(
                        pl.cast(pl.load(q, [1, 0], [1, DI]), pl.FP32), pl.read(weights_fp32, [0, 1])
                    )
                    weighted_q_2 = pl.mul(
                        pl.cast(pl.load(q, [2, 0], [1, DI]), pl.FP32), pl.read(weights_fp32, [0, 2])
                    )
                    weighted_q_3 = pl.mul(
                        pl.cast(pl.load(q, [3, 0], [1, DI]), pl.FP32), pl.read(weights_fp32, [0, 3])
                    )
                    weighted_q_4 = pl.mul(
                        pl.cast(pl.load(q, [4, 0], [1, DI]), pl.FP32), pl.read(weights_fp32, [0, 4])
                    )
                    weighted_q_5 = pl.mul(
                        pl.cast(pl.load(q, [5, 0], [1, DI]), pl.FP32), pl.read(weights_fp32, [0, 5])
                    )
                    weighted_q_6 = pl.mul(
                        pl.cast(pl.load(q, [6, 0], [1, DI]), pl.FP32), pl.read(weights_fp32, [0, 6])
                    )
                    weighted_q_7 = pl.mul(
                        pl.cast(pl.load(q, [7, 0], [1, DI]), pl.FP32), pl.read(weights_fp32, [0, 7])
                    )
                    q_pair_0 = pl.add(weighted_q_0, weighted_q_1)
                    q_pair_1 = pl.add(weighted_q_2, weighted_q_3)
                    q_pair_2 = pl.add(weighted_q_4, weighted_q_5)
                    q_pair_3 = pl.add(weighted_q_6, weighted_q_7)
                    q_eff = pl.add(pl.add(q_pair_0, q_pair_1), pl.add(q_pair_2, q_pair_3))
                    if BALANCED:
                        begin_block = worker * RANGE_BASE + pl.min(worker, RANGE_EXTRA)
                        owned_blocks = RANGE_BASE + pl.min(1, pl.max(0, RANGE_EXTRA - worker))
                        private_zero = pl.tile.full(
                            [1, MAXIMA_STRIDE], dtype=pl.FP32, value=-3.4028234663852886e38
                        )
                        for local_packet, (private_state,) in pl.range(
                            (owned_blocks + PACKET_BLOCKS - 1) // PACKET_BLOCKS, init_values=(private_zero,)
                        ):
                            packet_begin = begin_block + local_packet * PACKET_BLOCKS
                            valid_blocks = pl.min(PACKET_BLOCKS, owned_blocks - local_packet * PACKET_BLOCKS)
                            packet_zero = pl.tile.full(
                                [PACKET_BLOCKS, BLOCK], dtype=pl.FP32, value=-3.4028234663852886e38
                            )
                            for score_tile, (packet_state,) in pl.pipeline(
                                valid_blocks // SCORE_BLOCKS_PER_TILE, stage=2, init_values=(packet_zero,)
                            ):
                                subblock = score_tile * SCORE_BLOCKS_PER_TILE
                                sb = packet_begin + subblock
                                keys_full = pl.cast(pl.load(kc, [sb * BLOCK, 0], [SCORE_ROWS, DI]), pl.FP32)
                                acc = pl.row_sum(
                                    pl.col_expand_mul(keys_full, q_eff),
                                    pl.tile.create([SCORE_ROWS, DI], dtype=pl.FP32),
                                )
                                score_blocks = pl.reshape(acc, [SCORE_BLOCKS_PER_TILE, BLOCK])
                                packet_next = pl.tile.assemble(packet_state, score_blocks, [subblock, 0])
                                full_packet = pl.yield_(packet_next)
                            for tail_tile, (tail_state,) in pl.range(
                                valid_blocks % SCORE_BLOCKS_PER_TILE, init_values=(full_packet,)
                            ):
                                tail_subblock = (
                                    valid_blocks // SCORE_BLOCKS_PER_TILE
                                ) * SCORE_BLOCKS_PER_TILE + tail_tile
                                tail_block = packet_begin + tail_subblock
                                tail_keys = pl.cast(
                                    pl.load(kc, [tail_block * BLOCK, 0], [BLOCK, DI]), pl.FP32
                                )
                                tail_acc = pl.row_sum(
                                    pl.col_expand_mul(tail_keys, q_eff),
                                    pl.tile.create([BLOCK, DI], dtype=pl.FP32),
                                )
                                tail_scores = pl.reshape(tail_acc, [1, BLOCK])
                                tail_next = pl.tile.assemble(tail_state, tail_scores, [tail_subblock, 0])
                                packet_scores = pl.yield_(tail_next)
                            packet_flat = pl.set_validshape(
                                pl.reshape(packet_scores, [1, PACKET_BLOCKS * BLOCK]), 1, valid_blocks * BLOCK
                            )
                            pl.store(packet_flat, [0, packet_begin * BLOCK], ws_scr)
                            packet_maxima = pl.row_max(
                                packet_scores, pl.tile.create([PACKET_BLOCKS, BLOCK], dtype=pl.FP32)
                            )
                            maxima_flat = pl.set_validshape(
                                pl.reshape(packet_maxima, [1, PACKET_BLOCKS]), 1, valid_blocks
                            )
                            pool_zero_packet = pl.tile.full(
                                [1, PACKET_BLOCKS * BLOCK], dtype=pl.FP32, value=0.0
                            )
                            pl.store(
                                pl.set_validshape(pool_zero_packet, 1, valid_blocks * BLOCK),
                                [0, packet_begin * BLOCK],
                                ws_pool,
                            )
                            private_next = pl.tile.assemble(
                                private_state, maxima_flat, [0, local_packet * PACKET_BLOCKS]
                            )
                            private_maxima = pl.yield_(private_next)
                        pl.store(private_maxima, [worker, 0], scan_maxima)
                    else:
                        for packet in pl.range(worker, PACKETS, ACTIVE_WORKERS):
                            packet_begin = packet * PACKET_BLOCKS
                            valid_blocks = pl.min(PACKET_BLOCKS, NB - packet_begin)
                            packet_zero = pl.tile.full(
                                [PACKET_BLOCKS, BLOCK], dtype=pl.FP32, value=-3.4028234663852886e38
                            )
                            for score_tile, (packet_state,) in pl.pipeline(
                                valid_blocks // SCORE_BLOCKS_PER_TILE, stage=2, init_values=(packet_zero,)
                            ):
                                subblock = score_tile * SCORE_BLOCKS_PER_TILE
                                sb = packet_begin + subblock
                                keys_full = pl.cast(pl.load(kc, [sb * BLOCK, 0], [SCORE_ROWS, DI]), pl.FP32)
                                acc = pl.row_sum(
                                    pl.col_expand_mul(keys_full, q_eff),
                                    pl.tile.create([SCORE_ROWS, DI], dtype=pl.FP32),
                                )
                                score_blocks = pl.reshape(acc, [SCORE_BLOCKS_PER_TILE, BLOCK])
                                packet_next = pl.tile.assemble(packet_state, score_blocks, [subblock, 0])
                                full_packet = pl.yield_(packet_next)
                            for tail_tile, (tail_state,) in pl.range(
                                valid_blocks % SCORE_BLOCKS_PER_TILE, init_values=(full_packet,)
                            ):
                                tail_subblock = (
                                    valid_blocks // SCORE_BLOCKS_PER_TILE
                                ) * SCORE_BLOCKS_PER_TILE + tail_tile
                                tail_block = packet_begin + tail_subblock
                                tail_keys = pl.cast(
                                    pl.load(kc, [tail_block * BLOCK, 0], [BLOCK, DI]), pl.FP32
                                )
                                tail_acc = pl.row_sum(
                                    pl.col_expand_mul(tail_keys, q_eff),
                                    pl.tile.create([BLOCK, DI], dtype=pl.FP32),
                                )
                                tail_scores = pl.reshape(tail_acc, [1, BLOCK])
                                tail_next = pl.tile.assemble(tail_state, tail_scores, [tail_subblock, 0])
                                packet_scores = pl.yield_(tail_next)
                            packet_flat = pl.set_validshape(
                                pl.reshape(packet_scores, [1, PACKET_BLOCKS * BLOCK]), 1, valid_blocks * BLOCK
                            )
                            pl.store(packet_flat, [0, packet_begin * BLOCK], ws_scr)
                            packet_maxima = pl.row_max(
                                packet_scores, pl.tile.create([PACKET_BLOCKS, BLOCK], dtype=pl.FP32)
                            )
                            maxima_flat = pl.set_validshape(
                                pl.reshape(packet_maxima, [1, PACKET_BLOCKS]), 1, valid_blocks
                            )
                            pool_zero_packet = pl.tile.full(
                                [1, PACKET_BLOCKS * BLOCK], dtype=pl.FP32, value=0.0
                            )
                            pl.store(
                                pl.set_validshape(pool_zero_packet, 1, valid_blocks * BLOCK),
                                [0, packet_begin * BLOCK],
                                ws_pool,
                            )
                            pl.store(maxima_flat, [0, packet_begin], ws_bm)
                with pl.spmd(
                    NBL, name_hint="block_leaf_sort", allow_early_resolve=True, deps=[scan_done]
                ) as block_leaves_done:
                    leaf = pl.tile.get_block_idx()
                    valid_blocks = pl.min(LEAF, NB - leaf * LEAF)
                    if BALANCED:
                        logical_ids = pl.add(
                            pl.tile.arange(0, [1, LEAF], dtype=pl.INT32), pl.cast(leaf * LEAF, pl.INT32)
                        )
                        safe_ids = pl.tile.minimums(logical_ids, NB - 1)
                        late_ids = pl.tile.maximums(pl.sub(safe_ids, RANGE_CUTOFF), 0)
                        if FAST_INVERSE:
                            early_owner = pl.cast(
                                pl.mul(pl.add(pl.cast(safe_ids, pl.FP32), 0.5), EARLY_RECIPROCAL),
                                pl.INT32,
                                mode="floor",
                            )
                            late_owner_relative = pl.cast(
                                pl.mul(pl.add(pl.cast(late_ids, pl.FP32), 0.5), LATE_RECIPROCAL),
                                pl.INT32,
                                mode="floor",
                            )
                        else:
                            early_owner_estimate = pl.cast(
                                pl.mul(pl.cast(safe_ids, pl.FP32), EARLY_RECIPROCAL), pl.INT32, mode="round"
                            )
                            early_owner_remainder = pl.sub(
                                safe_ids, pl.mul(early_owner_estimate, RANGE_BASE + 1)
                            )
                            early_owner_too_high = pl.tile.cmps(
                                pl.cast(early_owner_remainder, pl.FP32), 0.0, cmp_type=2
                            )
                            early_owner_lowered = pl.tile.select(
                                early_owner_too_high, pl.sub(early_owner_estimate, 1), early_owner_estimate
                            )
                            early_owner_next_product = pl.mul(pl.add(early_owner_lowered, 1), RANGE_BASE + 1)
                            early_owner_too_low = pl.tile.cmp(
                                pl.cast(early_owner_next_product, pl.FP32),
                                pl.cast(safe_ids, pl.FP32),
                                cmp_type=3,
                            )
                            early_owner = pl.tile.select(
                                early_owner_too_low, pl.add(early_owner_lowered, 1), early_owner_lowered
                            )
                            late_owner_relative_estimate = pl.cast(
                                pl.mul(pl.cast(late_ids, pl.FP32), LATE_RECIPROCAL), pl.INT32, mode="round"
                            )
                            late_owner_relative_remainder = pl.sub(
                                late_ids, pl.mul(late_owner_relative_estimate, RANGE_DIVISOR)
                            )
                            late_owner_relative_too_high = pl.tile.cmps(
                                pl.cast(late_owner_relative_remainder, pl.FP32), 0.0, cmp_type=2
                            )
                            late_owner_relative_lowered = pl.tile.select(
                                late_owner_relative_too_high,
                                pl.sub(late_owner_relative_estimate, 1),
                                late_owner_relative_estimate,
                            )
                            late_owner_relative_next_product = pl.mul(
                                pl.add(late_owner_relative_lowered, 1), RANGE_DIVISOR
                            )
                            late_owner_relative_too_low = pl.tile.cmp(
                                pl.cast(late_owner_relative_next_product, pl.FP32),
                                pl.cast(late_ids, pl.FP32),
                                cmp_type=3,
                            )
                            late_owner_relative = pl.tile.select(
                                late_owner_relative_too_low,
                                pl.add(late_owner_relative_lowered, 1),
                                late_owner_relative_lowered,
                            )
                        early_mask = pl.tile.cmps(pl.cast(safe_ids, pl.FP32), RANGE_CUTOFF, cmp_type=2)
                        late_owner = pl.add(late_owner_relative, RANGE_EXTRA)
                        owners = pl.tile.select(early_mask, early_owner, late_owner)
                        early_begin = pl.mul(early_owner, RANGE_BASE + 1)
                        late_begin = pl.add(pl.mul(late_owner, RANGE_BASE), RANGE_EXTRA)
                        owner_begin = pl.tile.select(early_mask, early_begin, late_begin)
                        private_offsets = pl.add(pl.mul(owners, MAXIMA_STRIDE), pl.sub(safe_ids, owner_begin))
                        private_values = pl.reshape(
                            pl.load(scan_maxima, [0, 0], [ACTIVE_WORKERS, MAXIMA_STRIDE]),
                            [1, ACTIVE_WORKERS * MAXIMA_STRIDE],
                        )
                        gathered_maxima = pl.tile.gather(
                            private_values, private_offsets, pl.tile.create([1, LEAF], dtype=pl.INT32)
                        )
                        reconstructed_values = pl.set_validshape(gathered_maxima, 1, valid_blocks)
                        pl.store(reconstructed_values, [0, leaf * LEAF], ws_bm)
                        block_values = pl.yield_(reconstructed_values)
                    else:
                        original_values = pl.load(
                            ws_bm, [0, leaf * LEAF], [1, LEAF], valid_shape=[1, valid_blocks]
                        )
                        block_values = pl.yield_(original_values)
                    padded_block_values = pl.fillpad(block_values, pad_value=pl.PadValue.min)
                    block_ids = pl.add(
                        pl.tile.arange(0, [1, LEAF], dtype=pl.INT32), pl.cast(leaf * LEAF, pl.INT32)
                    )
                    block_pairs = pl.tile.sort32(
                        padded_block_values, pl.reinterpret_view(block_ids, pl.UINT32)
                    )
                    block_pairs = pl.tile.mrgsort(block_pairs, block_len=64)
                    block_pairs = pl.tile.mrgsort(block_pairs, block_len=256)
                    block_pairs = pl.tile.mrgsort(block_pairs, block_len=1024)
                    pl.store(pl.tile.slice(block_pairs, [1, 2 * TOPB], [0, 0]), [leaf, 0], block_pairs_gm)
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="block_prefix_merge",
                    allow_early_resolve=True,
                    deps=[block_leaves_done],
                ) as selection_done:
                    block_pairs_gm_merge_tmp = pl.tile.create([1, 1024], dtype=pl.FP32)
                    for block_pairs_gm_merge_start in pl.range(1, NBL, 1):
                        block_pairs_gm_left = pl.load(block_pairs_gm, [0, 0], [1, 512])
                        block_pairs_gm_right_0 = pl.load(
                            block_pairs_gm, [block_pairs_gm_merge_start + 0, 0], [1, 512]
                        )
                        block_pairs_gm_merged = pl.tile.mrgsort(
                            block_pairs_gm_left, block_pairs_gm_right_0, tmp=block_pairs_gm_merge_tmp
                        )
                        pl.store(
                            pl.tile.slice(block_pairs_gm_merged, [1, 512], [0, 0]), [0, 0], block_pairs_gm
                        )
                    final_blocks = pl.load(block_pairs_gm, [0, 0], [1, 2 * TOPB])
                    chosen = pl.tile.gather_mask(
                        final_blocks, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                    )
                    pl.store(chosen, [0, 0], selected_blocks)
                with pl.spmd(
                    GATHER_OWNERS,
                    name_hint="selected_dma_gather",
                    allow_early_resolve=True,
                    deps=[selection_done],
                ) as gather_done:
                    gather_owner = pl.tile.get_block_idx()
                    chosen_ids = pl.load(selected_blocks, [0, 0], [1, TOPB])
                    pool_ones = pl.tile.full([1, BLOCK], dtype=pl.FP32, value=1.0)
                    chosen_lines = pl.tile.shrs(chosen_ids, 4)
                    predicate_zero = pl.tile.full([1, TOPB], dtype=pl.FP32, value=0.0)
                    predicate_ones = pl.tile.full([1, TOPB], dtype=pl.FP32, value=1.0)
                    for owner_line, (predicate_state,) in pl.range(
                        gather_owner, MAXIMUM_LINES, GATHER_OWNERS, init_values=(predicate_zero,)
                    ):
                        matches = pl.tile.cmps(chosen_lines, pl.cast(owner_line, pl.INT32), cmp_type=0)
                        match_values = pl.tile.select(matches, predicate_ones, 0.0)
                        owner_predicate = pl.yield_(pl.add(predicate_state, match_values))
                    count_lanes = pl.col_sum(pl.reshape(owner_predicate, [TOPB // 8, 8]))
                    owner_count_i32 = pl.cast(
                        pl.read(count_lanes, [0, 0])
                        + pl.read(count_lanes, [0, 1])
                        + pl.read(count_lanes, [0, 2])
                        + pl.read(count_lanes, [0, 3])
                        + pl.read(count_lanes, [0, 4])
                        + pl.read(count_lanes, [0, 5])
                        + pl.read(count_lanes, [0, 6])
                        + pl.read(count_lanes, [0, 7]),
                        pl.INT32,
                    )
                    owner_count = pl.cast(owner_count_i32, pl.INDEX)
                    original_ranks = pl.tile.arange(0, [1, TOPB], dtype=pl.INT32)
                    local_pairs = pl.tile.sort32(
                        owner_predicate, pl.reinterpret_view(original_ranks, pl.UINT32)
                    )
                    local_pairs = pl.tile.mrgsort(local_pairs, block_len=64)
                    local_pairs_left = pl.tile.slice(local_pairs, [1, TOPB], [0, 0])
                    local_pairs_right = pl.tile.slice(local_pairs, [1, TOPB], [0, TOPB])
                    local_pairs_tmp = pl.tile.create([1, 2 * TOPB], dtype=pl.FP32)
                    local_pairs = pl.tile.mrgsort(local_pairs_left, local_pairs_right, tmp=local_pairs_tmp)
                    local_ranks = pl.tile.gather_mask(
                        local_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                    )
                    for work_item in pl.range(owner_count):
                        selected = pl.cast(pl.read(local_ranks, [0, work_item]), pl.INDEX)
                        block_id = pl.read(chosen_ids, [0, selected])
                        block_offset = pl.cast(block_id, pl.INDEX)
                        offset = block_offset * BLOCK
                        original_maximum = pl.read(ws_bm, [0, block_offset])
                        pl.write(ws_bm, [0, block_offset], original_maximum - BIG)
                        pl.store(pool_ones, [0, offset], ws_pool)
                with pl.spmd(
                    NSL, name_hint="buffered_score_leaves", allow_early_resolve=True, deps=[selection_done]
                ) as score_leaves_done:
                    leaf = pl.tile.get_block_idx()
                    chosen_score_ids = pl.load(selected_blocks, [0, 0], [1, TOPB])
                    leaf_zero_values = pl.tile.full([LEAF // BLOCK, BLOCK], dtype=pl.FP32, value=0.0)
                    leaf_zero_ids = pl.tile.full([LEAF // BLOCK, BLOCK], dtype=pl.INT32, value=0)
                    for score_bundle, (value_state, id_state) in pl.range(
                        (LEAF // BLOCK) // 4, init_values=(leaf_zero_values, leaf_zero_ids)
                    ):
                        score_rank_0 = leaf * (LEAF // BLOCK) + score_bundle * 4 + 0
                        score_block_0 = pl.read(chosen_score_ids, [0, score_rank_0])
                        score_offset_0 = pl.cast(score_block_0, pl.INDEX) * BLOCK
                        score_values_0 = pl.load(ws_scr, [0, score_offset_0], [1, BLOCK])
                        score_rank_1 = leaf * (LEAF // BLOCK) + score_bundle * 4 + 1
                        score_block_1 = pl.read(chosen_score_ids, [0, score_rank_1])
                        score_offset_1 = pl.cast(score_block_1, pl.INDEX) * BLOCK
                        score_values_1 = pl.load(ws_scr, [0, score_offset_1], [1, BLOCK])
                        score_rank_2 = leaf * (LEAF // BLOCK) + score_bundle * 4 + 2
                        score_block_2 = pl.read(chosen_score_ids, [0, score_rank_2])
                        score_offset_2 = pl.cast(score_block_2, pl.INDEX) * BLOCK
                        score_values_2 = pl.load(ws_scr, [0, score_offset_2], [1, BLOCK])
                        score_rank_3 = leaf * (LEAF // BLOCK) + score_bundle * 4 + 3
                        score_block_3 = pl.read(chosen_score_ids, [0, score_rank_3])
                        score_offset_3 = pl.cast(score_block_3, pl.INDEX) * BLOCK
                        score_values_3 = pl.load(ws_scr, [0, score_offset_3], [1, BLOCK])
                        score_ids_0 = pl.add(
                            pl.tile.arange(0, [1, BLOCK], dtype=pl.INT32),
                            pl.cast(score_block_0 * BLOCK, pl.INT32),
                        )
                        assembled_values_0 = pl.tile.assemble(
                            value_state, score_values_0, [score_bundle * 4 + 0, 0]
                        )
                        assembled_ids_0 = pl.tile.assemble(id_state, score_ids_0, [score_bundle * 4 + 0, 0])
                        score_ids_1 = pl.add(
                            pl.tile.arange(0, [1, BLOCK], dtype=pl.INT32),
                            pl.cast(score_block_1 * BLOCK, pl.INT32),
                        )
                        assembled_values_1 = pl.tile.assemble(
                            assembled_values_0, score_values_1, [score_bundle * 4 + 1, 0]
                        )
                        assembled_ids_1 = pl.tile.assemble(
                            assembled_ids_0, score_ids_1, [score_bundle * 4 + 1, 0]
                        )
                        score_ids_2 = pl.add(
                            pl.tile.arange(0, [1, BLOCK], dtype=pl.INT32),
                            pl.cast(score_block_2 * BLOCK, pl.INT32),
                        )
                        assembled_values_2 = pl.tile.assemble(
                            assembled_values_1, score_values_2, [score_bundle * 4 + 2, 0]
                        )
                        assembled_ids_2 = pl.tile.assemble(
                            assembled_ids_1, score_ids_2, [score_bundle * 4 + 2, 0]
                        )
                        score_ids_3 = pl.add(
                            pl.tile.arange(0, [1, BLOCK], dtype=pl.INT32),
                            pl.cast(score_block_3 * BLOCK, pl.INT32),
                        )
                        assembled_values_3 = pl.tile.assemble(
                            assembled_values_2, score_values_3, [score_bundle * 4 + 3, 0]
                        )
                        assembled_ids_3 = pl.tile.assemble(
                            assembled_ids_2, score_ids_3, [score_bundle * 4 + 3, 0]
                        )
                        vals, ids = pl.yield_(assembled_values_3, assembled_ids_3)
                    pairs = pl.tile.sort32(
                        pl.reshape(vals, [1, LEAF]),
                        pl.reinterpret_view(pl.reshape(ids, [1, LEAF]), pl.UINT32),
                    )
                    pairs = pl.tile.mrgsort(pairs, block_len=64)
                    pairs = pl.tile.mrgsort(pairs, block_len=256)
                    pairs = pl.tile.mrgsort(pairs, block_len=1024)
                    pl.store(pl.tile.slice(pairs, [1, 2 * TOPK], [0, 0]), [leaf, 0], score_pairs_gm)
                if LATE_JOIN:
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="score_prefix_merge",
                        allow_early_resolve=False,
                        deps=[score_leaves_done],
                    ) as score_merge_done:
                        score_pairs_gm_merge_tmp = pl.tile.create([1, 2048], dtype=pl.FP32)
                        for score_pairs_gm_merge_start in pl.range(1, NSL, 1):
                            score_pairs_gm_left = pl.load(score_pairs_gm, [0, 0], [1, 1024])
                            score_pairs_gm_right_0 = pl.load(
                                score_pairs_gm, [score_pairs_gm_merge_start + 0, 0], [1, 1024]
                            )
                            score_pairs_gm_merged = pl.tile.mrgsort(
                                score_pairs_gm_left, score_pairs_gm_right_0, tmp=score_pairs_gm_merge_tmp
                            )
                            pl.store(
                                pl.tile.slice(score_pairs_gm_merged, [1, 1024], [0, 0]),
                                [0, 0],
                                score_pairs_gm,
                            )
                        final = pl.load(score_pairs_gm, [0, 0], [1, 2 * TOPK])
                        values = pl.tile.gather_mask(final, mask_pattern=pl.tile.MaskPattern.P0101)
                        indices = pl.tile.gather_mask(
                            final, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                        )
                        pl.store(values, [0, 0], top_val)
                        pl.store(indices, [0, 0], top_idx)
                    invocation_done = pl.system.task_dummy(deps=[score_merge_done, gather_done])
                else:
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="score_prefix_merge",
                        allow_early_resolve=False,
                        deps=[score_leaves_done, gather_done],
                    ):
                        score_pairs_gm_merge_tmp = pl.tile.create([1, 2048], dtype=pl.FP32)
                        for score_pairs_gm_merge_start in pl.range(1, NSL, 1):
                            score_pairs_gm_left = pl.load(score_pairs_gm, [0, 0], [1, 1024])
                            score_pairs_gm_right_0 = pl.load(
                                score_pairs_gm, [score_pairs_gm_merge_start + 0, 0], [1, 1024]
                            )
                            score_pairs_gm_merged = pl.tile.mrgsort(
                                score_pairs_gm_left, score_pairs_gm_right_0, tmp=score_pairs_gm_merge_tmp
                            )
                            pl.store(
                                pl.tile.slice(score_pairs_gm_merged, [1, 1024], [0, 0]),
                                [0, 0],
                                score_pairs_gm,
                            )
                        final = pl.load(score_pairs_gm, [0, 0], [1, 2 * TOPK])
                        values = pl.tile.gather_mask(final, mask_pattern=pl.tile.MaskPattern.P0101)
                        indices = pl.tile.gather_mask(
                            final, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                        )
                        pl.store(values, [0, 0], top_val)
                        pl.store(indices, [0, 0], top_idx)
                return top_val
        else:

            @pl.jit
            def hierindex(
                q: pl.Tensor[[H, DI], input_dtype],
                kc: pl.Tensor[[S, DI], input_dtype],
                w: pl.Tensor[[H], input_dtype],
                ws_scr: pl.InOut[pl.Tensor[[1, S], pl.FP32]],
                ws_pool: pl.InOut[pl.Tensor[[1, S], pl.FP32]],
                ws_bm: pl.InOut[pl.Tensor[[1, NB], pl.FP32]],
                top_val: pl.Out[pl.Tensor[[1, TOPK], pl.FP32]],
                top_idx: pl.Out[pl.Tensor[[1, TOPK], pl.INT32]],
            ):
                scan_maxima = pl.create_tensor([ACTIVE_WORKERS, MAXIMA_STRIDE], dtype=pl.FP32)
                selected_blocks = pl.create_tensor([1, TOPB], dtype=pl.INT32)
                block_pairs_gm = pl.create_tensor([NBL, 2 * TOPB], dtype=pl.FP32)
                score_pairs_gm = pl.create_tensor([NSL, 2 * TOPK], dtype=pl.FP32)
                block_remaining = pl.create_tensor([1, BPAD], dtype=pl.FP32)
                with pl.spmd(
                    ACTIVE_WORKERS, name_hint="scan_score_packets", allow_early_resolve=True
                ) as scan_done:
                    worker = pl.tile.get_block_idx()
                    weights_fp32 = pl.reshape(pl.load(w, [0], [16], valid_shape=[H]), [1, 16])
                    weighted_q_0 = pl.mul(pl.load(q, [0, 0], [1, DI]), pl.read(weights_fp32, [0, 0]))
                    weighted_q_1 = pl.mul(pl.load(q, [1, 0], [1, DI]), pl.read(weights_fp32, [0, 1]))
                    weighted_q_2 = pl.mul(pl.load(q, [2, 0], [1, DI]), pl.read(weights_fp32, [0, 2]))
                    weighted_q_3 = pl.mul(pl.load(q, [3, 0], [1, DI]), pl.read(weights_fp32, [0, 3]))
                    weighted_q_4 = pl.mul(pl.load(q, [4, 0], [1, DI]), pl.read(weights_fp32, [0, 4]))
                    weighted_q_5 = pl.mul(pl.load(q, [5, 0], [1, DI]), pl.read(weights_fp32, [0, 5]))
                    weighted_q_6 = pl.mul(pl.load(q, [6, 0], [1, DI]), pl.read(weights_fp32, [0, 6]))
                    weighted_q_7 = pl.mul(pl.load(q, [7, 0], [1, DI]), pl.read(weights_fp32, [0, 7]))
                    q_pair_0 = pl.add(weighted_q_0, weighted_q_1)
                    q_pair_1 = pl.add(weighted_q_2, weighted_q_3)
                    q_pair_2 = pl.add(weighted_q_4, weighted_q_5)
                    q_pair_3 = pl.add(weighted_q_6, weighted_q_7)
                    q_eff = pl.add(pl.add(q_pair_0, q_pair_1), pl.add(q_pair_2, q_pair_3))
                    if BALANCED:
                        begin_block = worker * RANGE_BASE + pl.min(worker, RANGE_EXTRA)
                        owned_blocks = RANGE_BASE + pl.min(1, pl.max(0, RANGE_EXTRA - worker))
                        private_zero = pl.tile.full(
                            [1, MAXIMA_STRIDE], dtype=pl.FP32, value=-3.4028234663852886e38
                        )
                        for local_packet, (private_state,) in pl.range(
                            (owned_blocks + PACKET_BLOCKS - 1) // PACKET_BLOCKS, init_values=(private_zero,)
                        ):
                            packet_begin = begin_block + local_packet * PACKET_BLOCKS
                            valid_blocks = pl.min(PACKET_BLOCKS, owned_blocks - local_packet * PACKET_BLOCKS)
                            packet_zero = pl.tile.full(
                                [PACKET_BLOCKS, BLOCK], dtype=pl.FP32, value=-3.4028234663852886e38
                            )
                            for score_tile, (packet_state,) in pl.range(
                                (valid_blocks + SCORE_BLOCKS_PER_TILE - 1) // SCORE_BLOCKS_PER_TILE,
                                init_values=(packet_zero,),
                            ):
                                subblock = score_tile * SCORE_BLOCKS_PER_TILE
                                sb = packet_begin + subblock
                                valid_rows = pl.min(SCORE_ROWS, (valid_blocks - subblock) * BLOCK)
                                kblk = pl.load(
                                    kc, [sb * BLOCK, 0], [SCORE_ROWS, DI], valid_shape=[valid_rows, DI]
                                )
                                keys_neutral = pl.fillpad(kblk, pad_value=pl.PadValue.zero)
                                keys_full = pl.set_validshape(keys_neutral, SCORE_ROWS, DI)
                                acc = pl.row_sum(
                                    pl.col_expand_mul(keys_full, q_eff),
                                    pl.tile.create([SCORE_ROWS, DI], dtype=pl.FP32),
                                )
                                score_blocks = pl.set_validshape(
                                    pl.reshape(acc, [SCORE_BLOCKS_PER_TILE, BLOCK]),
                                    valid_rows // BLOCK,
                                    BLOCK,
                                )
                                packet_next = pl.tile.assemble(packet_state, score_blocks, [subblock, 0])
                                packet_scores = pl.yield_(packet_next)
                            packet_flat = pl.set_validshape(
                                pl.reshape(packet_scores, [1, PACKET_BLOCKS * BLOCK]), 1, valid_blocks * BLOCK
                            )
                            pl.store(packet_flat, [0, packet_begin * BLOCK], ws_scr)
                            packet_maxima = pl.row_max(
                                packet_scores, pl.tile.create([PACKET_BLOCKS, BLOCK], dtype=pl.FP32)
                            )
                            maxima_flat = pl.set_validshape(
                                pl.reshape(packet_maxima, [1, PACKET_BLOCKS]), 1, valid_blocks
                            )
                            pool_zero_packet = pl.tile.full(
                                [1, PACKET_BLOCKS * BLOCK], dtype=pl.FP32, value=0.0
                            )
                            pl.store(
                                pl.set_validshape(pool_zero_packet, 1, valid_blocks * BLOCK),
                                [0, packet_begin * BLOCK],
                                ws_pool,
                            )
                            private_next = pl.tile.assemble(
                                private_state, maxima_flat, [0, local_packet * PACKET_BLOCKS]
                            )
                            private_maxima = pl.yield_(private_next)
                        pl.store(private_maxima, [worker, 0], scan_maxima)
                    else:
                        for packet in pl.range(worker, PACKETS, ACTIVE_WORKERS):
                            packet_begin = packet * PACKET_BLOCKS
                            valid_blocks = pl.min(PACKET_BLOCKS, NB - packet_begin)
                            packet_zero = pl.tile.full(
                                [PACKET_BLOCKS, BLOCK], dtype=pl.FP32, value=-3.4028234663852886e38
                            )
                            for score_tile, (packet_state,) in pl.range(
                                (valid_blocks + SCORE_BLOCKS_PER_TILE - 1) // SCORE_BLOCKS_PER_TILE,
                                init_values=(packet_zero,),
                            ):
                                subblock = score_tile * SCORE_BLOCKS_PER_TILE
                                sb = packet_begin + subblock
                                valid_rows = pl.min(SCORE_ROWS, (valid_blocks - subblock) * BLOCK)
                                kblk = pl.load(
                                    kc, [sb * BLOCK, 0], [SCORE_ROWS, DI], valid_shape=[valid_rows, DI]
                                )
                                keys_neutral = pl.fillpad(kblk, pad_value=pl.PadValue.zero)
                                keys_full = pl.set_validshape(keys_neutral, SCORE_ROWS, DI)
                                acc = pl.row_sum(
                                    pl.col_expand_mul(keys_full, q_eff),
                                    pl.tile.create([SCORE_ROWS, DI], dtype=pl.FP32),
                                )
                                score_blocks = pl.set_validshape(
                                    pl.reshape(acc, [SCORE_BLOCKS_PER_TILE, BLOCK]),
                                    valid_rows // BLOCK,
                                    BLOCK,
                                )
                                packet_next = pl.tile.assemble(packet_state, score_blocks, [subblock, 0])
                                packet_scores = pl.yield_(packet_next)
                            packet_flat = pl.set_validshape(
                                pl.reshape(packet_scores, [1, PACKET_BLOCKS * BLOCK]), 1, valid_blocks * BLOCK
                            )
                            pl.store(packet_flat, [0, packet_begin * BLOCK], ws_scr)
                            packet_maxima = pl.row_max(
                                packet_scores, pl.tile.create([PACKET_BLOCKS, BLOCK], dtype=pl.FP32)
                            )
                            maxima_flat = pl.set_validshape(
                                pl.reshape(packet_maxima, [1, PACKET_BLOCKS]), 1, valid_blocks
                            )
                            pool_zero_packet = pl.tile.full(
                                [1, PACKET_BLOCKS * BLOCK], dtype=pl.FP32, value=0.0
                            )
                            pl.store(
                                pl.set_validshape(pool_zero_packet, 1, valid_blocks * BLOCK),
                                [0, packet_begin * BLOCK],
                                ws_pool,
                            )
                            pl.store(maxima_flat, [0, packet_begin], ws_bm)
                with pl.spmd(
                    NBL, name_hint="block_leaf_sort", allow_early_resolve=True, deps=[scan_done]
                ) as block_leaves_done:
                    leaf = pl.tile.get_block_idx()
                    valid_blocks = pl.min(LEAF, NB - leaf * LEAF)
                    if BALANCED:
                        logical_ids = pl.add(
                            pl.tile.arange(0, [1, LEAF], dtype=pl.INT32), pl.cast(leaf * LEAF, pl.INT32)
                        )
                        safe_ids = pl.tile.minimums(logical_ids, NB - 1)
                        late_ids = pl.tile.maximums(pl.sub(safe_ids, RANGE_CUTOFF), 0)
                        if FAST_INVERSE:
                            early_owner = pl.cast(
                                pl.mul(pl.add(pl.cast(safe_ids, pl.FP32), 0.5), EARLY_RECIPROCAL),
                                pl.INT32,
                                mode="floor",
                            )
                            late_owner_relative = pl.cast(
                                pl.mul(pl.add(pl.cast(late_ids, pl.FP32), 0.5), LATE_RECIPROCAL),
                                pl.INT32,
                                mode="floor",
                            )
                        else:
                            early_owner_estimate = pl.cast(
                                pl.mul(pl.cast(safe_ids, pl.FP32), EARLY_RECIPROCAL), pl.INT32, mode="round"
                            )
                            early_owner_remainder = pl.sub(
                                safe_ids, pl.mul(early_owner_estimate, RANGE_BASE + 1)
                            )
                            early_owner_too_high = pl.tile.cmps(
                                pl.cast(early_owner_remainder, pl.FP32), 0.0, cmp_type=2
                            )
                            early_owner_lowered = pl.tile.select(
                                early_owner_too_high, pl.sub(early_owner_estimate, 1), early_owner_estimate
                            )
                            early_owner_next_product = pl.mul(pl.add(early_owner_lowered, 1), RANGE_BASE + 1)
                            early_owner_too_low = pl.tile.cmp(
                                pl.cast(early_owner_next_product, pl.FP32),
                                pl.cast(safe_ids, pl.FP32),
                                cmp_type=3,
                            )
                            early_owner = pl.tile.select(
                                early_owner_too_low, pl.add(early_owner_lowered, 1), early_owner_lowered
                            )
                            late_owner_relative_estimate = pl.cast(
                                pl.mul(pl.cast(late_ids, pl.FP32), LATE_RECIPROCAL), pl.INT32, mode="round"
                            )
                            late_owner_relative_remainder = pl.sub(
                                late_ids, pl.mul(late_owner_relative_estimate, RANGE_DIVISOR)
                            )
                            late_owner_relative_too_high = pl.tile.cmps(
                                pl.cast(late_owner_relative_remainder, pl.FP32), 0.0, cmp_type=2
                            )
                            late_owner_relative_lowered = pl.tile.select(
                                late_owner_relative_too_high,
                                pl.sub(late_owner_relative_estimate, 1),
                                late_owner_relative_estimate,
                            )
                            late_owner_relative_next_product = pl.mul(
                                pl.add(late_owner_relative_lowered, 1), RANGE_DIVISOR
                            )
                            late_owner_relative_too_low = pl.tile.cmp(
                                pl.cast(late_owner_relative_next_product, pl.FP32),
                                pl.cast(late_ids, pl.FP32),
                                cmp_type=3,
                            )
                            late_owner_relative = pl.tile.select(
                                late_owner_relative_too_low,
                                pl.add(late_owner_relative_lowered, 1),
                                late_owner_relative_lowered,
                            )
                        early_mask = pl.tile.cmps(pl.cast(safe_ids, pl.FP32), RANGE_CUTOFF, cmp_type=2)
                        late_owner = pl.add(late_owner_relative, RANGE_EXTRA)
                        owners = pl.tile.select(early_mask, early_owner, late_owner)
                        early_begin = pl.mul(early_owner, RANGE_BASE + 1)
                        late_begin = pl.add(pl.mul(late_owner, RANGE_BASE), RANGE_EXTRA)
                        owner_begin = pl.tile.select(early_mask, early_begin, late_begin)
                        private_offsets = pl.add(pl.mul(owners, MAXIMA_STRIDE), pl.sub(safe_ids, owner_begin))
                        private_values = pl.reshape(
                            pl.load(scan_maxima, [0, 0], [ACTIVE_WORKERS, MAXIMA_STRIDE]),
                            [1, ACTIVE_WORKERS * MAXIMA_STRIDE],
                        )
                        gathered_maxima = pl.tile.gather(
                            private_values, private_offsets, pl.tile.create([1, LEAF], dtype=pl.INT32)
                        )
                        reconstructed_values = pl.set_validshape(gathered_maxima, 1, valid_blocks)
                        pl.store(reconstructed_values, [0, leaf * LEAF], ws_bm)
                        block_values = pl.yield_(reconstructed_values)
                    else:
                        original_values = pl.load(
                            ws_bm, [0, leaf * LEAF], [1, LEAF], valid_shape=[1, valid_blocks]
                        )
                        block_values = pl.yield_(original_values)
                    padded_block_values = pl.fillpad(block_values, pad_value=pl.PadValue.min)
                    block_ids = pl.add(
                        pl.tile.arange(0, [1, LEAF], dtype=pl.INT32), pl.cast(leaf * LEAF, pl.INT32)
                    )
                    block_pairs = pl.tile.sort32(
                        padded_block_values, pl.reinterpret_view(block_ids, pl.UINT32)
                    )
                    block_pairs = pl.tile.mrgsort(block_pairs, block_len=64)
                    block_pairs = pl.tile.mrgsort(block_pairs, block_len=256)
                    block_pairs = pl.tile.mrgsort(block_pairs, block_len=1024)
                    pl.store(pl.tile.slice(block_pairs, [1, 2 * TOPB], [0, 0]), [leaf, 0], block_pairs_gm)
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="block_prefix_merge",
                    allow_early_resolve=True,
                    deps=[block_leaves_done],
                ) as selection_done:
                    block_pairs_gm_merge_tmp = pl.tile.create([1, 1024], dtype=pl.FP32)
                    for block_pairs_gm_merge_start in pl.range(1, NBL, 1):
                        block_pairs_gm_left = pl.load(block_pairs_gm, [0, 0], [1, 512])
                        block_pairs_gm_right_0 = pl.load(
                            block_pairs_gm, [block_pairs_gm_merge_start + 0, 0], [1, 512]
                        )
                        block_pairs_gm_merged = pl.tile.mrgsort(
                            block_pairs_gm_left, block_pairs_gm_right_0, tmp=block_pairs_gm_merge_tmp
                        )
                        pl.store(
                            pl.tile.slice(block_pairs_gm_merged, [1, 512], [0, 0]), [0, 0], block_pairs_gm
                        )
                    final_blocks = pl.load(block_pairs_gm, [0, 0], [1, 2 * TOPB])
                    chosen = pl.tile.gather_mask(
                        final_blocks, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                    )
                    pl.store(chosen, [0, 0], selected_blocks)
                with pl.spmd(
                    GATHER_OWNERS,
                    name_hint="selected_dma_gather",
                    allow_early_resolve=True,
                    deps=[selection_done],
                ) as gather_done:
                    gather_owner = pl.tile.get_block_idx()
                    chosen_ids = pl.load(selected_blocks, [0, 0], [1, TOPB])
                    pool_ones = pl.tile.full([1, BLOCK], dtype=pl.FP32, value=1.0)
                    chosen_lines = pl.tile.shrs(chosen_ids, 4)
                    predicate_zero = pl.tile.full([1, TOPB], dtype=pl.FP32, value=0.0)
                    predicate_ones = pl.tile.full([1, TOPB], dtype=pl.FP32, value=1.0)
                    for owner_line, (predicate_state,) in pl.range(
                        gather_owner, MAXIMUM_LINES, GATHER_OWNERS, init_values=(predicate_zero,)
                    ):
                        matches = pl.tile.cmps(chosen_lines, pl.cast(owner_line, pl.INT32), cmp_type=0)
                        match_values = pl.tile.select(matches, predicate_ones, 0.0)
                        owner_predicate = pl.yield_(pl.add(predicate_state, match_values))
                    count_lanes = pl.col_sum(pl.reshape(owner_predicate, [TOPB // 8, 8]))
                    owner_count_i32 = pl.cast(
                        pl.read(count_lanes, [0, 0])
                        + pl.read(count_lanes, [0, 1])
                        + pl.read(count_lanes, [0, 2])
                        + pl.read(count_lanes, [0, 3])
                        + pl.read(count_lanes, [0, 4])
                        + pl.read(count_lanes, [0, 5])
                        + pl.read(count_lanes, [0, 6])
                        + pl.read(count_lanes, [0, 7]),
                        pl.INT32,
                    )
                    owner_count = pl.cast(owner_count_i32, pl.INDEX)
                    original_ranks = pl.tile.arange(0, [1, TOPB], dtype=pl.INT32)
                    local_pairs = pl.tile.sort32(
                        owner_predicate, pl.reinterpret_view(original_ranks, pl.UINT32)
                    )
                    local_pairs = pl.tile.mrgsort(local_pairs, block_len=64)
                    local_pairs_left = pl.tile.slice(local_pairs, [1, TOPB], [0, 0])
                    local_pairs_right = pl.tile.slice(local_pairs, [1, TOPB], [0, TOPB])
                    local_pairs_tmp = pl.tile.create([1, 2 * TOPB], dtype=pl.FP32)
                    local_pairs = pl.tile.mrgsort(local_pairs_left, local_pairs_right, tmp=local_pairs_tmp)
                    local_ranks = pl.tile.gather_mask(
                        local_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                    )
                    for work_item in pl.range(owner_count):
                        selected = pl.cast(pl.read(local_ranks, [0, work_item]), pl.INDEX)
                        block_id = pl.read(chosen_ids, [0, selected])
                        block_offset = pl.cast(block_id, pl.INDEX)
                        offset = block_offset * BLOCK
                        original_maximum = pl.read(ws_bm, [0, block_offset])
                        pl.write(ws_bm, [0, block_offset], original_maximum - BIG)
                        pl.store(pool_ones, [0, offset], ws_pool)
                with pl.spmd(
                    NSL, name_hint="buffered_score_leaves", allow_early_resolve=True, deps=[selection_done]
                ) as score_leaves_done:
                    leaf = pl.tile.get_block_idx()
                    chosen_score_ids = pl.load(selected_blocks, [0, 0], [1, TOPB])
                    leaf_zero_values = pl.tile.full([LEAF // BLOCK, BLOCK], dtype=pl.FP32, value=0.0)
                    leaf_zero_ids = pl.tile.full([LEAF // BLOCK, BLOCK], dtype=pl.INT32, value=0)
                    for score_bundle, (value_state, id_state) in pl.range(
                        (LEAF // BLOCK) // 4, init_values=(leaf_zero_values, leaf_zero_ids)
                    ):
                        score_rank_0 = leaf * (LEAF // BLOCK) + score_bundle * 4 + 0
                        score_block_0 = pl.read(chosen_score_ids, [0, score_rank_0])
                        score_offset_0 = pl.cast(score_block_0, pl.INDEX) * BLOCK
                        score_values_0 = pl.load(ws_scr, [0, score_offset_0], [1, BLOCK])
                        score_rank_1 = leaf * (LEAF // BLOCK) + score_bundle * 4 + 1
                        score_block_1 = pl.read(chosen_score_ids, [0, score_rank_1])
                        score_offset_1 = pl.cast(score_block_1, pl.INDEX) * BLOCK
                        score_values_1 = pl.load(ws_scr, [0, score_offset_1], [1, BLOCK])
                        score_rank_2 = leaf * (LEAF // BLOCK) + score_bundle * 4 + 2
                        score_block_2 = pl.read(chosen_score_ids, [0, score_rank_2])
                        score_offset_2 = pl.cast(score_block_2, pl.INDEX) * BLOCK
                        score_values_2 = pl.load(ws_scr, [0, score_offset_2], [1, BLOCK])
                        score_rank_3 = leaf * (LEAF // BLOCK) + score_bundle * 4 + 3
                        score_block_3 = pl.read(chosen_score_ids, [0, score_rank_3])
                        score_offset_3 = pl.cast(score_block_3, pl.INDEX) * BLOCK
                        score_values_3 = pl.load(ws_scr, [0, score_offset_3], [1, BLOCK])
                        score_ids_0 = pl.add(
                            pl.tile.arange(0, [1, BLOCK], dtype=pl.INT32),
                            pl.cast(score_block_0 * BLOCK, pl.INT32),
                        )
                        assembled_values_0 = pl.tile.assemble(
                            value_state, score_values_0, [score_bundle * 4 + 0, 0]
                        )
                        assembled_ids_0 = pl.tile.assemble(id_state, score_ids_0, [score_bundle * 4 + 0, 0])
                        score_ids_1 = pl.add(
                            pl.tile.arange(0, [1, BLOCK], dtype=pl.INT32),
                            pl.cast(score_block_1 * BLOCK, pl.INT32),
                        )
                        assembled_values_1 = pl.tile.assemble(
                            assembled_values_0, score_values_1, [score_bundle * 4 + 1, 0]
                        )
                        assembled_ids_1 = pl.tile.assemble(
                            assembled_ids_0, score_ids_1, [score_bundle * 4 + 1, 0]
                        )
                        score_ids_2 = pl.add(
                            pl.tile.arange(0, [1, BLOCK], dtype=pl.INT32),
                            pl.cast(score_block_2 * BLOCK, pl.INT32),
                        )
                        assembled_values_2 = pl.tile.assemble(
                            assembled_values_1, score_values_2, [score_bundle * 4 + 2, 0]
                        )
                        assembled_ids_2 = pl.tile.assemble(
                            assembled_ids_1, score_ids_2, [score_bundle * 4 + 2, 0]
                        )
                        score_ids_3 = pl.add(
                            pl.tile.arange(0, [1, BLOCK], dtype=pl.INT32),
                            pl.cast(score_block_3 * BLOCK, pl.INT32),
                        )
                        assembled_values_3 = pl.tile.assemble(
                            assembled_values_2, score_values_3, [score_bundle * 4 + 3, 0]
                        )
                        assembled_ids_3 = pl.tile.assemble(
                            assembled_ids_2, score_ids_3, [score_bundle * 4 + 3, 0]
                        )
                        vals, ids = pl.yield_(assembled_values_3, assembled_ids_3)
                    pairs = pl.tile.sort32(
                        pl.reshape(vals, [1, LEAF]),
                        pl.reinterpret_view(pl.reshape(ids, [1, LEAF]), pl.UINT32),
                    )
                    pairs = pl.tile.mrgsort(pairs, block_len=64)
                    pairs = pl.tile.mrgsort(pairs, block_len=256)
                    pairs = pl.tile.mrgsort(pairs, block_len=1024)
                    pl.store(pl.tile.slice(pairs, [1, 2 * TOPK], [0, 0]), [leaf, 0], score_pairs_gm)
                if LATE_JOIN:
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="score_prefix_merge",
                        allow_early_resolve=False,
                        deps=[score_leaves_done],
                    ) as score_merge_done:
                        score_pairs_gm_merge_tmp = pl.tile.create([1, 2048], dtype=pl.FP32)
                        for score_pairs_gm_merge_start in pl.range(1, NSL, 1):
                            score_pairs_gm_left = pl.load(score_pairs_gm, [0, 0], [1, 1024])
                            score_pairs_gm_right_0 = pl.load(
                                score_pairs_gm, [score_pairs_gm_merge_start + 0, 0], [1, 1024]
                            )
                            score_pairs_gm_merged = pl.tile.mrgsort(
                                score_pairs_gm_left, score_pairs_gm_right_0, tmp=score_pairs_gm_merge_tmp
                            )
                            pl.store(
                                pl.tile.slice(score_pairs_gm_merged, [1, 1024], [0, 0]),
                                [0, 0],
                                score_pairs_gm,
                            )
                        final = pl.load(score_pairs_gm, [0, 0], [1, 2 * TOPK])
                        values = pl.tile.gather_mask(final, mask_pattern=pl.tile.MaskPattern.P0101)
                        indices = pl.tile.gather_mask(
                            final, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                        )
                        pl.store(values, [0, 0], top_val)
                        pl.store(indices, [0, 0], top_idx)
                    invocation_done = pl.system.task_dummy(deps=[score_merge_done, gather_done])
                else:
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="score_prefix_merge",
                        allow_early_resolve=False,
                        deps=[score_leaves_done, gather_done],
                    ):
                        score_pairs_gm_merge_tmp = pl.tile.create([1, 2048], dtype=pl.FP32)
                        for score_pairs_gm_merge_start in pl.range(1, NSL, 1):
                            score_pairs_gm_left = pl.load(score_pairs_gm, [0, 0], [1, 1024])
                            score_pairs_gm_right_0 = pl.load(
                                score_pairs_gm, [score_pairs_gm_merge_start + 0, 0], [1, 1024]
                            )
                            score_pairs_gm_merged = pl.tile.mrgsort(
                                score_pairs_gm_left, score_pairs_gm_right_0, tmp=score_pairs_gm_merge_tmp
                            )
                            pl.store(
                                pl.tile.slice(score_pairs_gm_merged, [1, 1024], [0, 0]),
                                [0, 0],
                                score_pairs_gm,
                            )
                        final = pl.load(score_pairs_gm, [0, 0], [1, 2 * TOPK])
                        values = pl.tile.gather_mask(final, mask_pattern=pl.tile.MaskPattern.P0101)
                        indices = pl.tile.gather_mask(
                            final, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                        )
                        pl.store(values, [0, 0], top_val)
                        pl.store(indices, [0, 0], top_idx)
                return top_val

    return hierindex


hier_index = _make_hier_index(SEQ_LEN, pl.BF16)


def build_tensor_specs(seq_len: int = SEQ_LEN, torch_dtype=None):
    import torch
    from golden import TensorSpec

    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    s = int(seq_len)
    nb = s // BLOCK
    return [
        TensorSpec("q", [H, DI], torch_dtype, init_value=lambda: torch.randn(H, DI) * 0.5),
        TensorSpec("kc", [s, DI], torch_dtype, init_value=lambda: torch.randn(s, DI) * 0.5),
        TensorSpec("w", [H], torch_dtype, init_value=lambda: torch.rand(H) + 0.5),
        TensorSpec("ws_scr", [1, s], torch.float32),
        TensorSpec("ws_pool", [1, s], torch.float32),
        TensorSpec("ws_bm", [1, nb], torch.float32),
        TensorSpec("top_val", [1, TOPK], torch.float32),
        TensorSpec("top_idx", [1, TOPK], torch.int32),
    ]


def golden_hier_index(tensors):
    import torch

    s = int(tensors["kc"].shape[0])
    nb = s // BLOCK
    dots = tensors["kc"].float() @ tensors["q"].float().t()
    head_score_0 = dots[:, 0] * tensors["w"].float()[0]
    head_score_1 = dots[:, 1] * tensors["w"].float()[1]
    head_score_2 = dots[:, 2] * tensors["w"].float()[2]
    head_score_3 = dots[:, 3] * tensors["w"].float()[3]
    head_score_4 = dots[:, 4] * tensors["w"].float()[4]
    head_score_5 = dots[:, 5] * tensors["w"].float()[5]
    head_score_6 = dots[:, 6] * tensors["w"].float()[6]
    head_score_7 = dots[:, 7] * tensors["w"].float()[7]
    scores = head_score_0
    scores = scores + head_score_1
    scores = scores + head_score_2
    scores = scores + head_score_3
    scores = scores + head_score_4
    scores = scores + head_score_5
    scores = scores + head_score_6
    scores = scores + head_score_7
    tensors["ws_scr"][:] = scores.view(1, s)
    block_maxima = scores.view(nb, BLOCK).max(-1).values
    selected = torch.topk(block_maxima, TOPB).indices
    remaining = block_maxima.clone()
    remaining[selected] -= BIG
    tensors["ws_bm"][:] = remaining.view(1, nb)
    pool = torch.zeros(s)
    pool[selected.unsqueeze(1) * BLOCK + torch.arange(BLOCK)] = 1.0
    tensors["ws_pool"][:] = pool.view(1, s)
    masked = scores.masked_fill(pool == 0, -BIG)
    values, indices = torch.topk(masked, TOPK)
    tensors["top_val"][:] = values.view(1, TOPK)
    tensors["top_idx"][:] = indices.view(1, TOPK).to(torch.int32)


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    args = parser.parse_args()

    result = run(
        fn=hier_index,
        specs=build_tensor_specs(),
        golden_fn=golden_hier_index,
        config=dict(
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=0.0,
        atol=1e-4,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
