# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exact fixed-width primitives for the decode CSA indexer Top-K forest."""

import pypto.language as pl

from config import (
    BLOCK_SIZE,
    CSA_MAX_CANDIDATES,
    CSA_CANDIDATES_PER_LEAF,
    CSA_MAX_LEAVES_PER_QUERY,
    CSA_MAX_NODES_PER_QUERY,
    CSA_MAX_QUERIES,
    CSA_PAIR_WIDTH,
    CSA_TOPK,
    FLASH as M,
    FP32_NEG_INF,
)
from decode_metadata import (
    PHASE_D_LEAF_BEGIN,
    PHASE_D_LEAF_FIELDS,
    PHASE_D_LEAF_QUERY,
    PHASE_D_LEAF_VALID,
    PHASE_D_PAIR_FIELDS,
    PHASE_D_PAIR_LEFT_LEAF,
    PHASE_D_PAIR_LEFT_SLOT,
    PHASE_D_PAIR_OUTPUT_SLOT,
    PHASE_D_PAIR_RIGHT_LEAF,
    PHASE_D_PAIR_RIGHT_SLOT,
    PHASE_D_ROOT_FIELDS,
    PHASE_D_ROOT_SLOT,
    PHASE_D_SINGLETON_FIELDS,
    PHASE_D_SINGLETON_LEAF,
    PHASE_D_SINGLETON_SLOT,
    PHASE_D_UPPER_FIELDS,
    PHASE_D_UPPER_LEFT_SLOT,
    PHASE_D_UPPER_OUTPUT_SLOT,
    PHASE_D_UPPER_RIGHT_SLOT,
)


LEAF_DYN = pl.dynamic("LEAF_DYN")
QUERY_DYN = pl.dynamic("QUERY_DYN")
PAIR_GROUP_DYN = pl.dynamic("PAIR_GROUP_DYN")
SINGLETON_DYN = pl.dynamic("SINGLETON_DYN")
UPPER_MERGE_DYN = pl.dynamic("UPPER_MERGE_DYN")
REQUEST_DYN = pl.dynamic("REQUEST_DYN")
REQUEST_OFFSET_DYN = pl.dynamic("REQUEST_OFFSET_DYN")
PAGE_DYN = pl.dynamic("PAGE_DYN")
IDX_ROW_DYN = pl.dynamic("IDX_ROW_DYN")

IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
CSA_PAGE_ROWS = BLOCK_SIZE
# Keep the leaf score fragment below the platform L1/Vec admission limit.  A
# 2K leaf is still reduced with the same Top-K ordering; only the score-loading
# tile is smaller so the generated Vec footprint remains within 188416 bytes.
CSA_SCORE_TILE = 32
CSA_LEAF_SCORE_FRAGMENTS = CSA_CANDIDATES_PER_LEAF // CSA_SCORE_TILE
CSA_MAX_CANDIDATES_FP32 = 262144.0
CSA_TOPK_PAIR_GRID_W = 8
CSA_TOPK_MAX_PAIR_GROUPS = CSA_MAX_QUERIES * CSA_MAX_LEAVES_PER_QUERY // 2
CSA_TOPK_MAX_PAIR_WAVES = (
    CSA_TOPK_MAX_PAIR_GROUPS + CSA_TOPK_PAIR_GRID_W - 1
) // CSA_TOPK_PAIR_GRID_W
assert CSA_MAX_CANDIDATES_FP32 == CSA_MAX_CANDIDATES

TOPK_ARENA_ROWS = CSA_MAX_QUERIES * CSA_MAX_NODES_PER_QUERY
assert TOPK_ARENA_ROWS == 4080

@pl.jit.inline(auto_scope=False)
def score_select_2k_top512(
    query_vectors: pl.Tensor[[QUERY_DYN, IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8],
    query_scales: pl.Tensor[[QUERY_DYN, IDX_N_HEADS], pl.FP32],
    query_weights: pl.Tensor[[QUERY_DYN, IDX_N_HEADS], pl.FP32],
    idx_kv_cache_flat: pl.Tensor[[IDX_ROW_DYN, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale_flat: pl.Tensor[[IDX_ROW_DYN, 1], pl.FP32],
    query_request_ids: pl.Tensor[[QUERY_DYN], pl.INT32],
    idx_pages: pl.Tensor[[PAGE_DYN, 2], pl.INT32],
    idx_page_offsets: pl.Tensor[[REQUEST_OFFSET_DYN], pl.INT32],
    idx_windows: pl.Tensor[[REQUEST_DYN, 3], pl.INT32],
    request_epochs: pl.Tensor[[REQUEST_DYN], pl.INT32],
    leaf_descriptors: pl.Tensor[[LEAF_DYN, PHASE_D_LEAF_FIELDS], pl.INT32],
    pair_arena: pl.Tensor[[TOPK_ARENA_ROWS, CSA_PAIR_WIDTH], pl.FP32],
    leaf_id: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
) -> pl.Tensor[[TOPK_ARENA_ROWS, CSA_PAIR_WIDTH], pl.FP32]:
    """Score one 2K leaf with a fixed rolling Top-512 pair buffer."""
    query = pl.cast(
        pl.read(leaf_descriptors, [leaf_id, PHASE_D_LEAF_QUERY]),
        pl.INDEX,
    )
    request = pl.cast(pl.read(query_request_ids, [query]), pl.INDEX)
    logical_begin = pl.read(leaf_descriptors, [leaf_id, PHASE_D_LEAF_BEGIN])
    valid_candidates = pl.cast(
        pl.read(leaf_descriptors, [leaf_id, PHASE_D_LEAF_VALID]),
        pl.INDEX,
    )
    request_epoch = pl.read(request_epochs, [request])
    page_begin = pl.read(idx_page_offsets, [request])
    page_end = pl.read(idx_page_offsets, [request + 1])
    page_total = page_end - page_begin
    valid_begin = pl.read(idx_windows, [request, 0])
    valid_end = pl.read(idx_windows, [request, 1])
    head = pl.read(idx_windows, [request, 2])
    logical_page_base = valid_begin // CSA_PAGE_ROWS
    cache_rows = pl.tensor.dim(idx_kv_cache_flat, 0)
    block_count = cache_rows // CSA_PAGE_ROWS
    page_count = pl.tensor.dim(idx_pages, 0)
    query_vector = pl.reshape(query_vectors[query : query + 1, :, :], [IDX_N_HEADS, IDX_HEAD_DIM])
    query_scale = query_scales[query : query + 1, :]
    query_weight = query_weights[query : query + 1, :]
    fragment_count = (
        valid_candidates + CSA_SCORE_TILE - 1
    ) // CSA_SCORE_TILE
    running_pairs = pl.create_tensor([1, CSA_PAIR_WIDTH], dtype=pl.FP32)
    for fragment in pl.range(CSA_LEAF_SCORE_FRAGMENTS):
        fragment_begin = fragment * CSA_SCORE_TILE
        fragment_valid = pl.min(
            CSA_SCORE_TILE,
            valid_candidates - fragment_begin,
        )
        if fragment >= fragment_count:
            fragment_valid = 0
        kv_tile = pl.create_l1([CSA_SCORE_TILE, IDX_HEAD_DIM], pl.INT8)
        scale_tile = pl.create_tensor([CSA_SCORE_TILE, 1], dtype=pl.FP32)
        for lane in pl.range(CSA_SCORE_TILE):
            local_candidate = fragment_begin + lane
            logical_candidate = logical_begin + local_candidate
            source_valid = pl.cast(0, pl.INT32)
            source_row = pl.cast(0, pl.INDEX)
            if lane < fragment_valid:
                if logical_candidate >= valid_begin:
                    if logical_candidate < valid_end:
                        relative_page = (
                            logical_candidate // CSA_PAGE_ROWS - logical_page_base
                        )
                        if relative_page >= 0:
                            if relative_page < page_total:
                                page_index = (head + relative_page) % page_total
                                page_entry = page_begin + page_index
                                if page_entry >= 0:
                                    if page_entry < page_count:
                                        physical_page = pl.read(
                                            idx_pages, [page_entry, 0]
                                        )
                                        page_epoch = pl.read(
                                            idx_pages, [page_entry, 1]
                                        )
                                        if physical_page >= 0:
                                            if page_epoch == request_epoch:
                                                if physical_page < block_count:
                                                    source_valid = pl.cast(
                                                        1, pl.INT32
                                                    )
                                                    source_row = pl.cast(
                                                        physical_page * CSA_PAGE_ROWS
                                                        + logical_candidate % CSA_PAGE_ROWS,
                                                        pl.INDEX,
                                                    )
            if source_valid == 1:
                kv_tile = pl.gather_row(
                    kv_tile,
                    idx_kv_cache_flat,
                    [lane, 0],
                    [source_row, 0],
                    [1, IDX_HEAD_DIM],
                )
            else:
                kv_tile = pl.gather_row(
                    kv_tile,
                    idx_kv_cache_flat,
                    [lane, 0],
                    [0, 0],
                    [1, IDX_HEAD_DIM],
                )
            scale_value = pl.read(idx_kv_scale_flat, [source_row, 0])
            pl.write(scale_tile, [lane, 0], scale_value)
        score_i32 = pl.matmul(
            kv_tile,
            query_vector,
            b_trans=True,
            out_dtype=pl.INT32,
        )
        score_fp32 = pl.cast(score_i32, target_type=pl.FP32, mode="none")
        score_fp32 = pl.col_expand_mul(score_fp32, query_scale)
        score_fp32 = pl.maximum(score_fp32, 0.0)
        score_fp32 = pl.col_expand_mul(score_fp32, query_weight)
        score_fragment = pl.reshape(
            pl.mul(pl.row_sum(score_fp32), scale_tile),
            [1, CSA_SCORE_TILE],
        )
        # Materialize the padded sort inputs in ordinary tensors.  Using
        # set_validshape/fillpad here gives the mrgsort result a non-null pad
        # mode; copying its pair prefix into the plain arena then fails PTOAS'
        # subview pad check.  Explicit sentinels preserve the same ordering
        # while keeping both source and destination unpadded.
        score_padded = pl.full(
            [1, CSA_TOPK], dtype=pl.FP32, value=FP32_NEG_INF
        )
        idx_init = pl.full(
            [1, CSA_TOPK], dtype=pl.INT32, value=CSA_MAX_CANDIDATES
        )
        for lane in pl.range(CSA_SCORE_TILE):
            if lane < fragment_valid:
                pl.write(
                    score_padded,
                    [0, lane],
                    pl.read(score_fragment, [0, lane]),
                )
                pl.write(
                    idx_init,
                    [0, lane],
                    pl.cast(logical_begin + fragment_begin + lane, pl.INT32),
                )
        # The A2/A3 cast legalizer has no INT32->UINT32 path.  This is a
        # bit-preserving index reinterpretation, not a numeric conversion.
        idx_init = pl.reinterpret_view(idx_init, pl.UINT32)
        fragment_pairs = pl.sort32(score_padded, idx_init)
        fragment_pairs = pl.mrgsort(fragment_pairs, block_len=64)
        fragment_pairs = pl.mrgsort(fragment_pairs, block_len=256)
        if fragment == 0:
            pair_prefix_indices = pl.arange(
                0, [1, CSA_PAIR_WIDTH], dtype=pl.INT32
            )
            running_pairs[:, :] = pl.gather(
                fragment_pairs, dim=1, index=pair_prefix_indices
            )
        else:
            combined_pairs = pl.concat(running_pairs, fragment_pairs)
            merged_pairs = pl.mrgsort(combined_pairs, block_len=512)
            pair_prefix_indices = pl.arange(
                0, [1, CSA_PAIR_WIDTH], dtype=pl.INT32
            )
            running_pairs[:, :] = pl.gather(
                merged_pairs, dim=1, index=pair_prefix_indices
            )
    pair_arena[output_slot : output_slot + 1, :] = running_pairs
    return pair_arena




@pl.jit.incore
def init_topk_roots(
    topk_scores: pl.Tensor[[QUERY_DYN, CSA_TOPK], pl.FP32],
    topk_indices: pl.Tensor[[QUERY_DYN, CSA_TOPK], pl.INT32],
) -> None:
    query_count = pl.tensor.dim(topk_scores, 0)
    for query in pl.range(query_count):
        topk_scores[query : query + 1, :] = pl.full(
            [1, CSA_TOPK], dtype=pl.FP32, value=FP32_NEG_INF
        )
        topk_indices[query : query + 1, :] = pl.full(
            [1, CSA_TOPK], dtype=pl.INT32, value=-1
        )
    # Writes are in-place through the passed tensors; no return value.


@pl.jit.incore
def materialize_topk_root(
    pair_arena: pl.Tensor[[TOPK_ARENA_ROWS, CSA_PAIR_WIDTH], pl.FP32],
    topk_scores: pl.Tensor[[QUERY_DYN, CSA_TOPK], pl.FP32],
    topk_indices: pl.Tensor[[QUERY_DYN, CSA_TOPK], pl.INT32],
    query: pl.Scalar[pl.INDEX],
    root_slot_raw: pl.Scalar[pl.INT32],
) -> None:
    if root_slot_raw >= 0:
        root_slot = pl.cast(root_slot_raw, pl.INDEX)
        root_pairs = pair_arena[root_slot : root_slot + 1, :]
        topk_scores[query : query + 1, :] = pl.gather(
            root_pairs,
            mask_pattern=pl.tile.MaskPattern.P0101,
            output_dtype=pl.FP32,
        )
        root_indices = pl.gather(
            root_pairs,
            mask_pattern=pl.tile.MaskPattern.P1010,
            output_dtype=pl.INT32,
        )
        index_fp32 = pl.cast(root_indices, target_type=pl.FP32, mode="none")
        invalid_flag = pl.cast(
            pl.div(
                pl.minimum(index_fp32, CSA_MAX_CANDIDATES_FP32),
                CSA_MAX_CANDIDATES_FP32,
            ),
            target_type=pl.INT32,
            mode="trunc",
        )
        root_indices = pl.sub(
            pl.sub(root_indices, pl.mul(root_indices, invalid_flag)),
            invalid_flag,
        )
        topk_indices[query : query + 1, :] = root_indices
    # Writes are in-place through the passed tensors; no return value.


@pl.jit.inline
def active_score_topk_forest(
    query_vectors: pl.Tensor[[QUERY_DYN, IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8],
    query_scales: pl.Tensor[[QUERY_DYN, IDX_N_HEADS], pl.FP32],
    query_weights: pl.Tensor[[QUERY_DYN, IDX_N_HEADS], pl.FP32],
    idx_kv_cache_flat: pl.Tensor[[IDX_ROW_DYN, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale_flat: pl.Tensor[[IDX_ROW_DYN, 1], pl.FP32],
    query_request_ids: pl.Tensor[[QUERY_DYN], pl.INT32],
    idx_pages: pl.Tensor[[PAGE_DYN, 2], pl.INT32],
    idx_page_offsets: pl.Tensor[[REQUEST_OFFSET_DYN], pl.INT32],
    idx_windows: pl.Tensor[[REQUEST_DYN, 3], pl.INT32],
    request_epochs: pl.Tensor[[REQUEST_DYN], pl.INT32],
    leaf_descriptors: pl.Tensor[[LEAF_DYN, PHASE_D_LEAF_FIELDS], pl.INT32],
    pair_descriptors: pl.Tensor[[PAIR_GROUP_DYN, PHASE_D_PAIR_FIELDS], pl.INT32],
    singleton_descriptors: pl.Tensor[[SINGLETON_DYN, PHASE_D_SINGLETON_FIELDS], pl.INT32],
    upper_descriptors: pl.Tensor[[UPPER_MERGE_DYN, PHASE_D_UPPER_FIELDS], pl.INT32],
    root_descriptors: pl.Tensor[[QUERY_DYN, PHASE_D_ROOT_FIELDS], pl.INT32],
    pair_arena: pl.Tensor[[TOPK_ARENA_ROWS, CSA_PAIR_WIDTH], pl.FP32],
    topk_scores: pl.Tensor[[QUERY_DYN, CSA_TOPK], pl.FP32],
    topk_indices: pl.Tensor[[QUERY_DYN, CSA_TOPK], pl.INT32],
    pair_group_actual_count: pl.Scalar[pl.INT32],
    singleton_actual_count: pl.Scalar[pl.INT32],
    upper_merge_actual_count: pl.Scalar[pl.INT32],
    index_commit_dep: pl.Scalar[pl.TASK_ID],
    completion: pl.Array[1, pl.TASK_ID],
) -> None:
    """Submit only active score/select leaves and their exact merge forest.

    The ``*_actual_count`` scalars carry the real per-chunk descriptor counts,
    which may be smaller than the padded leading extents. Iterating only the
    valid prefixes prevents sentinel slot indices from reaching the arena.
    """
    pair_group_count = pair_group_actual_count
    singleton_count = singleton_actual_count
    upper_merge_count = upper_merge_actual_count
    query_count = pl.tensor.dim(root_descriptors, 0)
    root_tids = pl.array.create(CSA_MAX_QUERIES, pl.TASK_ID)
    upper_grid_count = pl.max(query_count, 1)

    with pl.manual_scope():
        with pl.spmd(1, deps=[index_commit_dep]) as init_tid:
            init_topk_roots(
                topk_scores,
                topk_indices,
            )

        pair_wave_tids = pl.array.create(CSA_TOPK_MAX_PAIR_WAVES, pl.TASK_ID)
        for pair_wave, (pair_wave_tids_iter,) in pl.range(
            (pair_group_count + CSA_TOPK_PAIR_GRID_W - 1)
            // CSA_TOPK_PAIR_GRID_W,
            init_values=(pair_wave_tids,),
        ):
            pair_begin = pair_wave * CSA_TOPK_PAIR_GRID_W
            pair_grid_count = pl.min(
                CSA_TOPK_PAIR_GRID_W,
                pair_group_count - pair_begin,
            )
            wave_ready = pl.system.task_dummy(
                deps=[index_commit_dep, pair_wave_tids_iter]
            )
            with pl.spmd(
                pair_grid_count,
                deps=[wave_ready],
            ) as left_wave_tid:
                group = pair_begin + pl.tile.get_block_idx()
                left_leaf_id = pl.cast(
                    pl.read(pair_descriptors, [group, PHASE_D_PAIR_LEFT_LEAF]),
                    pl.INDEX,
                )
                left_slot = pl.cast(
                    pl.read(pair_descriptors, [group, PHASE_D_PAIR_LEFT_SLOT]),
                    pl.INDEX,
                )
                _left_pair_arena = score_select_2k_top512(
                    query_vectors,
                    query_scales,
                    query_weights,
                    idx_kv_cache_flat,
                    idx_kv_scale_flat,
                    query_request_ids,
                    idx_pages,
                    idx_page_offsets,
                    idx_windows,
                    request_epochs,
                    leaf_descriptors,
                    pair_arena,
                    left_leaf_id,
                    left_slot,
                )
            with pl.spmd(
                pair_grid_count,
                deps=[left_wave_tid],
            ) as right_wave_tid:
                group = pair_begin + pl.tile.get_block_idx()
                right_leaf_id = pl.cast(
                    pl.read(pair_descriptors, [group, PHASE_D_PAIR_RIGHT_LEAF]),
                    pl.INDEX,
                )
                right_slot = pl.cast(
                    pl.read(pair_descriptors, [group, PHASE_D_PAIR_RIGHT_SLOT]),
                    pl.INDEX,
                )
                _right_pair_arena = score_select_2k_top512(
                    query_vectors,
                    query_scales,
                    query_weights,
                    idx_kv_cache_flat,
                    idx_kv_scale_flat,
                    query_request_ids,
                    idx_pages,
                    idx_page_offsets,
                    idx_windows,
                    request_epochs,
                    leaf_descriptors,
                    pair_arena,
                    right_leaf_id,
                    right_slot,
                )
            with pl.spmd(
                pair_grid_count,
                deps=[left_wave_tid, right_wave_tid],
            ) as pair_wave_tid:
                group = pair_begin + pl.tile.get_block_idx()
                left_slot = pl.cast(
                    pl.read(pair_descriptors, [group, PHASE_D_PAIR_LEFT_SLOT]),
                    pl.INDEX,
                )
                right_slot = pl.cast(
                    pl.read(pair_descriptors, [group, PHASE_D_PAIR_RIGHT_SLOT]),
                    pl.INDEX,
                )
                output_slot = pl.cast(
                    pl.read(pair_descriptors, [group, PHASE_D_PAIR_OUTPUT_SLOT]),
                    pl.INDEX,
                )
                merged_pairs = pl.mrgsort(
                    pair_arena[left_slot : left_slot + 1, :],
                    pair_arena[right_slot : right_slot + 1, :],
                )
                pair_arena[output_slot : output_slot + 1, :] = (
                    merged_pairs[:, 0:CSA_PAIR_WIDTH]
                )
            pair_wave_tids_iter[pair_wave] = pair_wave_tid
            _pair_wave_tids_after = pl.yield_(pair_wave_tids_iter)
        pair_tid = pl.system.task_dummy(
            deps=[pair_wave_tids, index_commit_dep]
        )

        singleton_grid_count = pl.max(singleton_count, 1)
        with pl.spmd(
            singleton_grid_count,
            name_hint="phase_d_singleton_score",
            # The pair and singleton leaves write disjoint logical rows, but
            # pair_arena is one coarse-grained InOut tensor in the runtime.
            # Serialize the singleton wave after all pair waves so the arena
            # publication cannot be reordered across those views.
            deps=[pair_tid],
        ) as singleton_score_tid:
            singleton = pl.tile.get_block_idx()
            if singleton < singleton_count:
                leaf_id = pl.cast(
                    pl.read(
                        singleton_descriptors,
                        [singleton, PHASE_D_SINGLETON_LEAF],
                    ),
                    pl.INDEX,
                )
                singleton_slot = pl.cast(
                    pl.read(
                        singleton_descriptors,
                        [singleton, PHASE_D_SINGLETON_SLOT],
                    ),
                    pl.INDEX,
                )
                _singleton_pair_arena = score_select_2k_top512(
                    query_vectors,
                    query_scales,
                    query_weights,
                    idx_kv_cache_flat,
                    idx_kv_scale_flat,
                    query_request_ids,
                    idx_pages,
                    idx_page_offsets,
                    idx_windows,
                    request_epochs,
                    leaf_descriptors,
                    pair_arena,
                    leaf_id,
                    singleton_slot,
                )
        singleton_tid = singleton_score_tid

        # Each query owns a disjoint root/merge slot range.  Keep the upper
        # forest as one baseline-shaped grid; unlike pair waves it does not
        # have multiple writers contending for the same coarse arena view.
        upper_grid_count = pl.max(query_count, 1)
        with pl.spmd(
            upper_grid_count,
            name_hint="phase_d_upper_merge",
            deps=[pair_tid, singleton_tid],
        ) as upper_tid:
            query = pl.tile.get_block_idx()
            if query < query_count:
                root_slot_raw = pl.read(
                    root_descriptors,
                    [query, PHASE_D_ROOT_SLOT],
                )
                previous_root_slot = pl.cast(-1, pl.INT32)
                for previous_query in pl.range(query):
                    previous_root_slot = pl.max(
                        previous_root_slot,
                        pl.read(
                            root_descriptors,
                            [previous_query, PHASE_D_ROOT_SLOT],
                        ),
                    )
                for merge in pl.range(upper_merge_count):
                    output_slot_raw = pl.read(
                        upper_descriptors,
                        [merge, PHASE_D_UPPER_OUTPUT_SLOT],
                    )
                    if output_slot_raw > previous_root_slot:
                        if output_slot_raw <= root_slot_raw:
                            left_slot = pl.cast(
                                pl.read(
                                    upper_descriptors,
                                    [merge, PHASE_D_UPPER_LEFT_SLOT],
                                ),
                                pl.INDEX,
                            )
                            right_slot = pl.cast(
                                pl.read(
                                    upper_descriptors,
                                    [merge, PHASE_D_UPPER_RIGHT_SLOT],
                                ),
                                pl.INDEX,
                            )
                            output_slot = pl.cast(output_slot_raw, pl.INDEX)
                            merged_pairs = pl.mrgsort(
                                pair_arena[left_slot : left_slot + 1, :],
                                pair_arena[right_slot : right_slot + 1, :],
                            )
                            pair_arena[output_slot : output_slot + 1, :] = (
                                merged_pairs[:, 0:CSA_PAIR_WIDTH]
                            )

        for query, (root_tids_iter,) in pl.range(
            query_count,
            init_values=(root_tids,),
        ):
            root_slot_raw = pl.read(
                root_descriptors,
                [query, PHASE_D_ROOT_SLOT],
            )
            with pl.spmd(
                1,
                deps=[init_tid, upper_tid],
            ) as root_tid:
                materialize_topk_root(
                    pair_arena,
                    topk_scores,
                    topk_indices,
                    query,
                    root_slot_raw,
                )
            root_tids_iter[query] = root_tid
            root_tids_after_roots = pl.yield_(root_tids_iter)

    completion[0] = pl.system.task_dummy(
        deps=[root_tids_after_roots[query] for query in range(CSA_MAX_QUERIES)]
    )
    # Writes are in-place through the passed tensors; no return value.
