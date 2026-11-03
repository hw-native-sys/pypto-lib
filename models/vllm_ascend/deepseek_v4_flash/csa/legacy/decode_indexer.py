# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CSA integration dependency adapted from pypto-lib 205255b4/decode_indexer.py."""

import pypto.language as pl

from ..common.config import (
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    DECODE_BATCH,
    DECODE_SEQ,
    FP32_NEG_INF,
    INDEXER_NATIVE_CUBE_MIN_ROWS,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    TP,
)
from ..common.config import (
    FLASH as M,
)
from .indexer_scale_layout import SCALE_LAYOUT_CAPACITY
from ..common.layout import (
    INDEXER_KEY_BYTES,
    INDEXER_PAGE_BYTES_DYN,
    INDEXER_TABLE_COLUMNS_DYN,
)
from .topk_threshold_stages import threshold_merge_top512

B_DYN = pl.dynamic("B_DYN")

T_DYN = pl.dynamic("T_DYN")  # T = B * S

B = DECODE_BATCH // TP

S = DECODE_SEQ

T = B * S

D = M.hidden_size

Q_LORA = M.q_lora_rank

ROPE_HEAD_DIM = M.qk_rope_head_dim

IDX_N_HEADS = M.index_n_heads

IDX_HEAD_DIM = M.index_head_dim

HADAMARD_SCALE = IDX_HEAD_DIM**-0.5


IDX_NOPE_HEAD_DIM = M.index_nope_head_dim

WEIGHTS_SCALE = M.index_weights_scale

MAX_SEQ_LEN = M.max_position_embeddings

COMPRESS_RATIO = 4  # the indexer only runs on ratio-4 layers

IDX_TOPK = M.index_topk

INNER_OVERLAP = COMPRESS_RATIO == 4

INNER_COFF = 1 + int(INNER_OVERLAP)

INNER_HEAD_DIM = IDX_HEAD_DIM

INNER_OUT_DIM = INNER_COFF * INNER_HEAD_DIM

INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE

INNER_STATE_LEN = INNER_COFF * COMPRESS_RATIO

INNER_STATE_BLOCK_NUM_DYN = pl.dynamic("INNER_STATE_BLOCK_NUM_DYN")

INNER_STATE_DIM = 2 * INNER_OUT_DIM

IDX_MAX_ROWS = MAX_SEQ_LEN // COMPRESS_RATIO

IDX_MAX_BLOCKS = (IDX_MAX_ROWS + BLOCK_SIZE - 1) // BLOCK_SIZE

IDX_NATIVE_CACHE_BLOCK_NUM_DYN = pl.dynamic("IDX_NATIVE_CACHE_BLOCK_NUM_DYN")

CACHE_TILE = min(64, BLOCK_SIZE)

Q_TILE = 256

Q_OUT_TILE = 1024  # Query-projection output tile

T_PAD = ((T + 16 - 1) // 16) * 16
# query 投影专用的分块：上游 e68e091 "Reuse full-K indexer query weights across row
# tiles"。原来按（行块 × 输出块）循环、每个行块都把整条 K 的权重重载一遍，decode 下
# 行数少而 K 大，权重重载就是主要开销。改成按输出列循环、整条 K 的权重只加载一次，
# 再在其上遍历行块。weights 投影仍用 MM_ROW_TILE，两者分开，故另起名字。
QR_MM_ROW_TILE = 64
QR_MM_N_TILE = 256
QR_MM_T_PAD = ((T + QR_MM_ROW_TILE - 1) // QR_MM_ROW_TILE) * QR_MM_ROW_TILE

MM_ROW_TILE = 16

MM_N_TILE = min(512, (128 * 1024) // (MM_ROW_TILE * 4))

DEQUANT_T_TILE = min(T, 8)

HEAD_DIM_TILE = 32

# 上游口径：按 K 切 4 份并行、N 一次吃满 IDX_N_HEADS。
WEIGHTS_OK = 4  # Weights-projection K tile count
WEIGHTS_K_TILE = D // WEIGHTS_OK
D_TILE = 512

QH_QUANT_TILE = 64

QH_QUANT_WORKERS = 48  # Query Hadamard quantization workers

QR_PROJ_WORKERS = 24  # Query projection workers

QH_MM_TILE = 64  # Query Hadamard cube tile

QH_WORKERS = 24  # Query Hadamard matmul workers

WEIGHTS_WORKERS = 24  # Weights-projection workers

TP1_WEIGHTS_WORKERS = 8  # TP1 weights-projection workers

QH_HEAD_DIM_TILE = 64

DQ_ROPE_H_TILE = 4  # heads per fused query dequant + RoPE unit

DQ_ROPE_WORKERS = 48  # fused query dequant + RoPE workers

TOPK_PAIR_WIDTH = 2 * IDX_TOPK

TOPK_CANDIDATES_PER_LEAF = 8192

TOPK_MAX_CANDIDATES = IDX_MAX_ROWS

TOPK_MAX_LEAVES = (TOPK_MAX_CANDIDATES + TOPK_CANDIDATES_PER_LEAF - 1) // TOPK_CANDIDATES_PER_LEAF

TOPK_ROWS_PER_QUERY = TOPK_MAX_LEAVES * 2

TOPK_QUERY_WORKERS = 48  # Top-K query-merge workers

TOPK_MERGE_FAN_IN = 4  # QLI V2: accumulated Top-512 plus three incoming roots.

TOPK_ARENA_ROWS = T_PAD * TOPK_ROWS_PER_QUERY

TOPK_SCORE_WORKERS = 24  # Top-K score workers
LONG_S6_MIN_QUERY_ROWS = 4 * S  # Four requests can fill 24 workers with six balanced leaves.

SCORE_TILE = 384

BUFFERED_SCORE_TILE = 768
BUFFERED_LONG_SCORE_TILE = 1024
BUFFERED_SCORE_LANE_TILE = BUFFERED_SCORE_TILE // 2
BUFFERED_SCORE_SCALE = 1024.0
NATIVE_QLI_WEIGHT_ROWS = 16
# 13259f92：G6按head行拆为两半，WS仍累计完整六query后单次写回。
SCORE_SPLIT_MIN_QUERY_ROWS = 32
NATIVE_QLI_QK_COLS = 128
SCORE_READY_EVENT = 0
SCORE_CONSUMED_EVENT = 1

SCORE_LANE_ROWS = SCORE_TILE // 2

SCORE_MAX_QUERY_GROUP = S
SCORE_ARENA_ROWS = max(T_PAD, TOPK_SCORE_WORKERS * 2 * SCORE_MAX_QUERY_GROUP)
assert S % SCORE_MAX_QUERY_GROUP == 0 and T_PAD % SCORE_MAX_QUERY_GROUP == 0


# 单SPMD已验证的半叶直接归并资源边界。
SHORT_PREFETCH_HALF_SORT_LIMIT = 2048


@pl.jit.inline
def merge2_top512_pairs(
    pair_arena: pl.Tensor,
    left_slot: pl.Scalar[pl.INDEX],
    right_slot: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
) -> None:
    """Merge two arena rows and store their exact Top-512 pair row."""
    left = pl.load(pair_arena, [left_slot, 0], [1, TOPK_PAIR_WIDTH])
    right = pl.load(pair_arena, [right_slot, 0], [1, TOPK_PAIR_WIDTH])
    merge_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
    # Native QLI MergeSort places the incoming chunk before the accumulated
    # list, which determines the ordering when scores are exactly equal.
    merged_all = pl.tile.mrgsort(right, left, tmp=merge_tmp)
    merged = pl.tile.slice(merged_all, [1, TOPK_PAIR_WIDTH], [0, 0])
    pl.store(merged, [output_slot, 0], pair_arena)


@pl.jit.inline
def merge_top512_roots(
    pair_arena: pl.Tensor,
    arena_base: pl.Scalar[pl.INDEX],
    half_count: pl.Scalar[pl.INDEX],
) -> pl.Tile:
    """Keep the exact Top-512 prefix in UB between four-way merge rounds.

    QLI V2's ProcessLD retains the accumulated root until final publication.
    Explicit extract gives the loop-carried value a 1024-float allocation;
    carrying a slice of the full merge output would also carry its storage.
    """
    root = pl.load(pair_arena, [arena_base, 0], [1, TOPK_PAIR_WIDTH])
    for child in pl.range(1, half_count, TOPK_MERGE_FAN_IN - 1):
        incoming1 = pl.load(pair_arena, [arena_base + child, 0], [1, TOPK_PAIR_WIDTH])
        if child + 2 < half_count:
            incoming2 = pl.load(pair_arena, [arena_base + child + 1, 0], [1, TOPK_PAIR_WIDTH])
            incoming3 = pl.load(pair_arena, [arena_base + child + 2, 0], [1, TOPK_PAIR_WIDTH])
            merge_tmp4 = pl.tile.create([1, TOPK_MERGE_FAN_IN * TOPK_PAIR_WIDTH], dtype=pl.FP32)
            merged4 = pl.tile.mrgsort(incoming3, incoming2, incoming1, root, tmp=merge_tmp4, exhausted=True)
            root = pl.tile.extract(merged4, 0, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec)
        elif child + 1 < half_count:
            incoming2 = pl.load(pair_arena, [arena_base + child + 1, 0], [1, TOPK_PAIR_WIDTH])
            merge_tmp3 = pl.tile.create([1, 3 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
            merged3 = pl.tile.mrgsort(incoming2, incoming1, root, tmp=merge_tmp3, exhausted=True)
            root = pl.tile.extract(merged3, 0, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec)
        else:
            merge_tmp2 = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
            merged2 = pl.tile.mrgsort(incoming1, root, tmp=merge_tmp2)
            root = pl.tile.extract(merged2, 0, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec)
    return root


@pl.jit.inline
def indexer_topk_half_leaf(
    score_arena: pl.Tensor[[SCORE_ARENA_ROWS, TOPK_CANDIDATES_PER_LEAF], pl.FP32],
    pair_arena: pl.Tensor[[TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], pl.FP32],
    score_row: pl.Scalar[pl.INDEX],
    logical_begin: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
) -> None:
    """Sort a contiguous half-leaf and store its Top-512 pairs."""
    logical_begin_i32 = pl.cast(logical_begin, pl.INT32)
    if valid_count <= 512:
        short_indices = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), logical_begin_i32)
        short_raw = pl.load(score_arena, [score_row, 0], [1, 512], valid_shape=[1, valid_count])
        short_scores = pl.tile.fillpad(short_raw, pad_value=pl.PadValue.min)
        short_scores = pl.maximum(short_scores, FP32_NEG_INF)
        short_pairs = pl.sort32(short_scores, pl.reinterpret_view(short_indices, pl.UINT32))
        short_pairs = pl.mrgsort(short_pairs, block_len=64)
        short_pairs = pl.mrgsort(short_pairs, block_len=256)
        pl.store(short_pairs, [output_slot, 0], pair_arena)
    elif valid_count <= 1024:
        small_indices = pl.add(pl.tile.arange(0, [1, 1024], dtype=pl.INT32), logical_begin_i32)
        small_raw = pl.load(score_arena, [score_row, 0], [1, 1024], valid_shape=[1, valid_count])
        small_scores = pl.tile.fillpad(small_raw, pad_value=pl.PadValue.min)
        small_scores = pl.maximum(small_scores, FP32_NEG_INF)
        small_pairs = pl.sort32(small_scores, pl.reinterpret_view(small_indices, pl.UINT32))
        small_pairs = pl.mrgsort(small_pairs, block_len=64)
        small_pairs = pl.mrgsort(small_pairs, block_len=256)
        small_left = pl.tile.slice(small_pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
        small_right = pl.tile.slice(small_pairs, [1, TOPK_PAIR_WIDTH], [0, 1024])
        small_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
        small_merged = pl.tile.mrgsort(small_left, small_right, tmp=small_tmp)
        small_top = pl.tile.slice(small_merged, [1, TOPK_PAIR_WIDTH], [0, 0])
        pl.store(small_top, [output_slot, 0], pair_arena)
    elif valid_count <= 2048:
        leaf_indices = pl.add(pl.tile.arange(0, [1, 2048], dtype=pl.INT32), logical_begin_i32)
        leaf_scores_raw = pl.load(score_arena, [score_row, 0], [1, 2048], valid_shape=[1, valid_count])
        leaf_scores = pl.tile.fillpad(leaf_scores_raw, pad_value=pl.PadValue.min)
        leaf_scores = pl.maximum(leaf_scores, FP32_NEG_INF)
        pairs = pl.sort32(leaf_scores, pl.reinterpret_view(leaf_indices, pl.UINT32))
        pairs = pl.mrgsort(pairs, block_len=64)
        pairs = pl.mrgsort(pairs, block_len=256)
        pairs = pl.mrgsort(pairs, block_len=1024)
        pl.store(pl.tile.slice(pairs, [1, TOPK_PAIR_WIDTH], [0, 0]), [output_slot, 0], pair_arena)
    elif valid_count <= 2560:
        # QLI V2 aligns only the live sorting groups. Keep the existing
        # later-2048-chunk priority while avoiding padding five groups to eight.
        partial_indices = pl.add(pl.tile.arange(0, [1, 2560], dtype=pl.INT32), logical_begin_i32)
        partial_raw = pl.load(score_arena, [score_row, 0], [1, 2560], valid_shape=[1, valid_count])
        partial_scores = pl.tile.fillpad(partial_raw, pad_value=pl.PadValue.min)
        partial_scores = pl.maximum(partial_scores, FP32_NEG_INF)
        partial_pairs = pl.sort32(partial_scores, pl.reinterpret_view(partial_indices, pl.UINT32))
        partial_pairs = pl.mrgsort(partial_pairs, block_len=64)
        partial_pairs = pl.mrgsort(partial_pairs, block_len=256)
        partial_prefix = pl.set_validshape(partial_pairs, 1, 4096)
        partial_prefix = pl.mrgsort(partial_prefix, block_len=1024)
        partial_left = pl.tile.slice(partial_prefix, [1, TOPK_PAIR_WIDTH], [0, 0])
        partial_right = pl.tile.slice(partial_pairs, [1, TOPK_PAIR_WIDTH], [0, 4096])
        partial_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
        partial_merged = pl.tile.mrgsort(partial_right, partial_left, tmp=partial_tmp)
        pl.store(pl.tile.slice(partial_merged, [1, TOPK_PAIR_WIDTH], [0, 0]), [output_slot, 0], pair_arena)
    elif valid_count <= 3072:
        triple_indices = pl.add(pl.tile.arange(0, [1, 3072], dtype=pl.INT32), logical_begin_i32)
        triple_raw = pl.load(score_arena, [score_row, 0], [1, 3072], valid_shape=[1, valid_count])
        triple_scores = pl.tile.fillpad(triple_raw, pad_value=pl.PadValue.min)
        triple_scores = pl.maximum(triple_scores, FP32_NEG_INF)
        triple_pairs = pl.sort32(triple_scores, pl.reinterpret_view(triple_indices, pl.UINT32))
        triple_pairs = pl.mrgsort(triple_pairs, block_len=64)
        triple_pairs = pl.mrgsort(triple_pairs, block_len=256)
        triple_prefix = pl.set_validshape(triple_pairs, 1, 4096)
        triple_prefix = pl.mrgsort(triple_prefix, block_len=1024)
        triple_left = pl.tile.slice(triple_prefix, [1, TOPK_PAIR_WIDTH], [0, 0])
        triple_right0 = pl.tile.slice(triple_pairs, [1, TOPK_PAIR_WIDTH], [0, 4096])
        triple_right1 = pl.tile.slice(triple_pairs, [1, TOPK_PAIR_WIDTH], [0, 5120])
        triple_tmp = pl.tile.create([1, 3 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
        # The two right runs retain their original order; both precede the
        # first 2048 candidates, exactly as in the padded two-root merge.
        triple_merged = pl.tile.mrgsort(triple_right0, triple_right1, triple_left, tmp=triple_tmp)
        pl.store(pl.tile.slice(triple_merged, [1, TOPK_PAIR_WIDTH], [0, 0]), [output_slot, 0], pair_arena)
    elif valid_count <= 4096:
        medium_indices = pl.add(pl.tile.arange(0, [1, 4096], dtype=pl.INT32), logical_begin_i32)
        medium_scores_raw = pl.load(score_arena, [score_row, 0], [1, 4096], valid_shape=[1, valid_count])
        medium_scores = pl.tile.fillpad(medium_scores_raw, pad_value=pl.PadValue.min)
        medium_scores = pl.maximum(medium_scores, FP32_NEG_INF)
        medium_pairs = pl.sort32(medium_scores, pl.reinterpret_view(medium_indices, pl.UINT32))
        medium_pairs = pl.mrgsort(medium_pairs, block_len=64)
        medium_pairs = pl.mrgsort(medium_pairs, block_len=256)
        medium_pairs = pl.mrgsort(medium_pairs, block_len=1024)
        medium_left = pl.tile.slice(medium_pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
        medium_right = pl.tile.slice(medium_pairs, [1, TOPK_PAIR_WIDTH], [0, 4096])
        medium_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
        # Native sorts 2048 candidates at a time, then merges the new chunk first.
        medium_merged = pl.tile.mrgsort(medium_right, medium_left, tmp=medium_tmp)
        pl.store(pl.tile.slice(medium_merged, [1, TOPK_PAIR_WIDTH], [0, 0]), [output_slot, 0], pair_arena)


@pl.jit.inline
def indexer_topk_segment_pairs(
    score_arena: pl.Tensor,
    score_row: pl.Scalar[pl.INDEX],
    logical_begin: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    score_offset: pl.constexpr,
) -> pl.Tile:
    """Sort the at-most-2048 candidates with the original tie rule."""
    logical_i32 = pl.cast(logical_begin, pl.INT32)
    if valid_count <= 512:
        tiny_indices = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), logical_i32)
        tiny_raw = pl.load(score_arena, [score_row, score_offset], [1, 512], valid_shape=[1, valid_count])
        tiny_scores = pl.maximum(pl.tile.fillpad(tiny_raw, pad_value=pl.PadValue.min), FP32_NEG_INF)
        tiny_pairs = pl.sort32(tiny_scores, pl.reinterpret_view(tiny_indices, pl.UINT32))
        tiny_pairs = pl.mrgsort(tiny_pairs, block_len=64)
        tiny_pairs = pl.mrgsort(tiny_pairs, block_len=256)
        root = pl.tile.extract(tiny_pairs, 0, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec)
    elif valid_count <= 1024:
        small_indices = pl.add(pl.tile.arange(0, [1, 1024], dtype=pl.INT32), logical_i32)
        small_raw = pl.load(score_arena, [score_row, score_offset], [1, 1024], valid_shape=[1, valid_count])
        small_scores = pl.maximum(pl.tile.fillpad(small_raw, pad_value=pl.PadValue.min), FP32_NEG_INF)
        small_pairs = pl.sort32(small_scores, pl.reinterpret_view(small_indices, pl.UINT32))
        small_pairs = pl.mrgsort(small_pairs, block_len=64)
        small_pairs = pl.mrgsort(small_pairs, block_len=256)
        small_left = pl.tile.slice(small_pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
        small_right = pl.tile.slice(small_pairs, [1, TOPK_PAIR_WIDTH], [0, 1024])
        small_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
        small_merged = pl.tile.mrgsort(small_left, small_right, tmp=small_tmp)
        root = pl.tile.extract(small_merged, 0, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec)
    else:
        large_indices = pl.add(pl.tile.arange(0, [1, 2048], dtype=pl.INT32), logical_i32)
        large_raw = pl.load(score_arena, [score_row, score_offset], [1, 2048], valid_shape=[1, valid_count])
        large_scores = pl.maximum(pl.tile.fillpad(large_raw, pad_value=pl.PadValue.min), FP32_NEG_INF)
        large_pairs = pl.sort32(large_scores, pl.reinterpret_view(large_indices, pl.UINT32))
        large_pairs = pl.mrgsort(large_pairs, block_len=64)
        large_pairs = pl.mrgsort(large_pairs, block_len=256)
        large_pairs = pl.mrgsort(large_pairs, block_len=1024)
        root = pl.tile.extract(large_pairs, 0, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec)
    return root


@pl.jit.inline
def indexer_use_long_s6(query_count: pl.Scalar[pl.INDEX]):
    use_group = pl.cast(0, pl.INT32)
    if query_count >= LONG_S6_MIN_QUERY_ROWS:
        use_group = pl.cast(1, pl.INT32)
    if query_count == S:
        use_group = pl.cast(1, pl.INT32)
    return use_group


@pl.jit.inline
def indexer_long_leaf_plan(
    max_cache_count: pl.Scalar[pl.INDEX],
    query_groups: pl.Scalar[pl.INDEX],
):
    """Balance the supported request groups over the 24 Score workers.

    S6 reuses each Key panel for all six queries. Round four groups to six
    leaves, eight/sixteen groups to three-leaf multiples, within the existing
    pair arena. B1 targets 24 leaves, capped by the actual tile count;
    other request counts preserve the original leaf partition.
    """
    tile_count = (max_cache_count + BUFFERED_LONG_SCORE_TILE - 1) // BUFFERED_LONG_SCORE_TILE
    leaf_count = pl.max((max_cache_count + TOPK_CANDIDATES_PER_LEAF - 1) // TOPK_CANDIDATES_PER_LEAF, 1)
    leaf_tiles = TOPK_CANDIDATES_PER_LEAF // BUFFERED_LONG_SCORE_TILE
    extra_leaves = 0
    leaf_multiple = 1
    # 用户允许B1专门策略：整请求复用Key，增加独立叶覆盖24个Cube worker。
    if query_groups == 1:
        leaf_multiple = TOPK_SCORE_WORKERS
    elif 3 * query_groups == 2 * TOPK_SCORE_WORKERS:  # noqa: SIM114 - PyPTO Scalar branches
        leaf_multiple = 3
    elif 3 * query_groups == TOPK_SCORE_WORKERS:
        leaf_multiple = 3
    elif 6 * query_groups == TOPK_SCORE_WORKERS:
        leaf_multiple = 6
    if leaf_multiple > 1:
        balanced_count = (leaf_count + leaf_multiple - 1) // leaf_multiple * leaf_multiple
        if query_groups == 1:
            # 至少一叶避免空历史除零，也不创建超过实际1024列段数的叶。
            balanced_count = pl.min(balanced_count, pl.max(tile_count, 1))
        if balanced_count <= TOPK_MAX_LEAVES:
            leaf_count = balanced_count
            leaf_tiles = pl.max(tile_count // balanced_count, 1)
            extra_leaves = tile_count % balanced_count
    return leaf_count, leaf_tiles, extra_leaves


@pl.jit.inline
def indexer_topk_query_merge_one(
    query: pl.Scalar[pl.INDEX],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    pair_arena: pl.Tensor[[TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], pl.FP32],
    topk_scores: pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32],
    topk_indices: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32],
    multiway: pl.constexpr,
    leaf_tiles: pl.Scalar[pl.INDEX],
    extra_leaves: pl.Scalar[pl.INDEX],
):
    """Merge the selected leaf roots and publish one query's Top-512."""
    batch_idx = query // S
    position = pl.read(position_ids, [query])
    cache_len = pl.read(kv_seq_lens, [batch_idx]) // COMPRESS_RATIO
    cache_bound = pl.min(cache_len, (position + 1) // COMPRESS_RATIO)
    visible_count = pl.min(cache_bound, TOPK_MAX_CANDIDATES)
    if visible_count > 0:
        leaf_count = (visible_count + TOPK_CANDIDATES_PER_LEAF - 1) // TOPK_CANDIDATES_PER_LEAF
        if multiway:
            visible_tiles = (visible_count + BUFFERED_LONG_SCORE_TILE - 1) // BUFFERED_LONG_SCORE_TILE
            larger_leaf_tiles = extra_leaves * (leaf_tiles + 1)
            larger_leaf_count = pl.min(extra_leaves, (visible_tiles + leaf_tiles) // (leaf_tiles + 1))
            remaining_tiles = pl.max(visible_tiles - larger_leaf_tiles, 0)
            leaf_count = larger_leaf_count + (remaining_tiles + leaf_tiles - 1) // leaf_tiles
        half_count = leaf_count * 2
        if multiway:
            long_s6_group = indexer_use_long_s6(pl.tensor.dim(position_ids, 0))
            if long_s6_group == 1:
                half_count = leaf_count
        arena_base = query * TOPK_ROWS_PER_QUERY
        if multiway:
            root_pairs = merge_top512_roots(pair_arena, arena_base, half_count)
        else:
            for child in pl.range(1, half_count):
                merge2_top512_pairs(pair_arena, arena_base, arena_base + child, arena_base)
            root_slot = arena_base
            root_pairs = pl.load(pair_arena, [root_slot, 0], [1, TOPK_PAIR_WIDTH])
        root_scores = pl.tile.gather_mask(root_pairs, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.FP32)
        pl.store(root_scores, [query, 0], topk_scores)
        root_indices = pl.tile.gather_mask(root_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        if visible_count >= IDX_TOPK:
            pl.store(root_indices, [query, 0], topk_indices)
        else:
            output_indices = pl.tile.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
            for lane in pl.range(visible_count):
                pl.tile.write(output_indices, [0, lane], pl.tile.read(root_indices, [0, lane]))
            pl.store(output_indices, [query, 0], topk_indices)
    else:
        pl.store(pl.tile.full([1, IDX_TOPK], dtype=pl.FP32, value=FP32_NEG_INF), [query, 0], topk_scores)
        pl.store(pl.tile.full([1, IDX_TOPK], dtype=pl.INT32, value=-1), [query, 0], topk_indices)


@pl.jit.incore
def indexer_topk_query_merge(
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    pair_arena: pl.Tensor[[TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], pl.FP32],
    topk_scores: pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32],
    topk_indices: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32],
    multiway: pl.constexpr,
):
    """Merge query roots on one persistent worker per physical AIV."""
    worker = pl.tile.get_block_idx()
    query_count = pl.tensor.dim(position_ids, 0)
    merge_leaf_tiles = TOPK_CANDIDATES_PER_LEAF // BUFFERED_LONG_SCORE_TILE
    merge_extra_leaves = 0
    if multiway:
        max_cache_count = 0
        for batch in pl.range(query_count // S):
            max_cache_count = pl.max(max_cache_count, pl.read(kv_seq_lens, [batch]) // COMPRESS_RATIO)
        query_groups = query_count // 2
        merge_s6_group = indexer_use_long_s6(query_count)
        if merge_s6_group == 1:
            query_groups = query_count // S
        _merge_leaf_count, merge_leaf_tiles, merge_extra_leaves = indexer_long_leaf_plan(
            pl.min(max_cache_count, TOPK_MAX_CANDIDATES), query_groups
        )
    for query in pl.range(worker, query_count, TOPK_QUERY_WORKERS):
        indexer_topk_query_merge_one(
            query,
            position_ids,
            kv_seq_lens,
            pair_arena,
            topk_scores,
            topk_indices,
            multiway,
            merge_leaf_tiles,
            merge_extra_leaves,
        )


@pl.jit.inline
def indexer_topk_leaf_publish(
    score_arena: pl.Tensor[[SCORE_ARENA_ROWS, TOPK_CANDIDATES_PER_LEAF], pl.FP32],
    query: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    topk_scores: pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32],
    topk_indices: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32],
) -> None:
    """Sort one populated leaf and directly publish its Top-512 pairs."""
    if valid_count <= 2048:
        short_indices = pl.tile.arange(0, [1, 2048], dtype=pl.INT32)
        short_raw = pl.load(score_arena, [query, 0], [1, 2048], valid_shape=[1, valid_count])
        short_scores = pl.tile.fillpad(short_raw, pad_value=pl.PadValue.min)
        short_scores = pl.maximum(short_scores, FP32_NEG_INF)
        short_pairs = pl.sort32(short_scores, pl.reinterpret_view(short_indices, pl.UINT32))
        short_pairs = pl.mrgsort(short_pairs, block_len=64)
        short_pairs = pl.mrgsort(short_pairs, block_len=256)
        short_pairs = pl.mrgsort(short_pairs, block_len=1024)
        short_top = pl.tile.slice(short_pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
        short_values = pl.tile.gather_mask(short_top, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.FP32)
        short_selected = pl.tile.gather_mask(short_top, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        pl.store(short_values, [query, 0], topk_scores)
        if valid_count >= IDX_TOPK:
            pl.store(short_selected, [query, 0], topk_indices)
        else:
            short_output = pl.tile.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
            for lane in pl.range(valid_count):
                pl.tile.write(short_output, [0, lane], pl.tile.read(short_selected, [0, lane]))
            pl.store(short_output, [query, 0], topk_indices)
    elif valid_count <= 4096:
        medium_indices = pl.tile.arange(0, [1, 4096], dtype=pl.INT32)
        medium_raw = pl.load(score_arena, [query, 0], [1, 4096], valid_shape=[1, valid_count])
        medium_scores = pl.tile.fillpad(medium_raw, pad_value=pl.PadValue.min)
        medium_scores = pl.maximum(medium_scores, FP32_NEG_INF)
        medium_pairs = pl.sort32(medium_scores, pl.reinterpret_view(medium_indices, pl.UINT32))
        medium_pairs = pl.mrgsort(medium_pairs, block_len=64)
        medium_pairs = pl.mrgsort(medium_pairs, block_len=256)
        medium_pairs = pl.mrgsort(medium_pairs, block_len=1024)
        medium_left = pl.tile.slice(medium_pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
        medium_right = pl.tile.slice(medium_pairs, [1, TOPK_PAIR_WIDTH], [0, 4096])
        medium_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
        medium_merged = pl.tile.mrgsort(medium_right, medium_left, tmp=medium_tmp)
        medium_top = pl.tile.slice(medium_merged, [1, TOPK_PAIR_WIDTH], [0, 0])
        medium_values = pl.tile.gather_mask(medium_top, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.FP32)
        medium_selected = pl.tile.gather_mask(medium_top, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        pl.store(medium_values, [query, 0], topk_scores)
        pl.store(medium_selected, [query, 0], topk_indices)
    else:
        full_indices = pl.tile.arange(0, [1, TOPK_CANDIDATES_PER_LEAF], dtype=pl.INT32)
        full_raw = pl.load(score_arena, [query, 0], [1, TOPK_CANDIDATES_PER_LEAF], valid_shape=[1, valid_count])
        full_scores = pl.tile.fillpad(full_raw, pad_value=pl.PadValue.min)
        full_min = pl.tile.full([1, TOPK_CANDIDATES_PER_LEAF], dtype=pl.FP32, value=FP32_NEG_INF)
        full_scores = pl.maximum(full_scores, full_min)
        full_pairs = pl.sort32(full_scores, pl.reinterpret_view(full_indices, pl.UINT32))
        full_pairs = pl.mrgsort(full_pairs, block_len=64)
        full_pairs = pl.mrgsort(full_pairs, block_len=256)
        full_pairs = pl.mrgsort(full_pairs, block_len=1024)
        # Preserve the same later-chunk tie priority in the single-leaf path.
        full_first = pl.tile.slice(full_pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
        full_second = pl.tile.slice(full_pairs, [1, TOPK_PAIR_WIDTH], [0, 4096])
        full_third = pl.tile.slice(full_pairs, [1, TOPK_PAIR_WIDTH], [0, 8192])
        full_fourth = pl.tile.slice(full_pairs, [1, TOPK_PAIR_WIDTH], [0, 12288])
        full_tmp = pl.tile.create([1, 4 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
        full_merged = pl.tile.mrgsort(full_fourth, full_third, full_second, full_first, tmp=full_tmp)
        full_top = pl.tile.slice(full_merged, [1, TOPK_PAIR_WIDTH], [0, 0])
        full_values = pl.tile.gather_mask(full_top, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.FP32)
        full_selected = pl.tile.gather_mask(full_top, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        pl.store(full_values, [query, 0], topk_scores)
        pl.store(full_selected, [query, 0], topk_indices)


@pl.jit.incore
def indexer_topk_single_leaf_publish(
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    score_arena: pl.Tensor[[SCORE_ARENA_ROWS, TOPK_CANDIDATES_PER_LEAF], pl.FP32],
    topk_scores: pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32],
    topk_indices: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32],
):
    """Sort a single leaf and publish each query's Top-512 directly."""
    worker = pl.tile.get_block_idx()
    query_count = pl.tensor.dim(position_ids, 0)
    for query in pl.range(worker, query_count, TOPK_QUERY_WORKERS):
        position = pl.read(position_ids, [query])
        cache_len = pl.read(kv_seq_lens, [query // S]) // COMPRESS_RATIO
        visible_count = pl.max(pl.min(cache_len, (position + 1) // COMPRESS_RATIO), 0)
        if visible_count > 0:
            indexer_topk_leaf_publish(score_arena, query, visible_count, topk_scores, topk_indices)
        else:
            pl.store(pl.tile.full([1, IDX_TOPK], dtype=pl.FP32, value=FP32_NEG_INF), [query, 0], topk_scores)
            pl.store(pl.tile.full([1, IDX_TOPK], dtype=pl.INT32, value=-1), [query, 0], topk_indices)


@pl.jit.inline(auto_scope=False)
def indexer_head_coefficients(
    qr_hadamard_scale_dq: pl.Tensor[[T_PAD * IDX_N_HEADS, 1], pl.FP32],
    weights: pl.Tensor[[T_PAD, IDX_N_HEADS], pl.FP32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    qh_quant_tid: pl.Scalar[pl.TASK_ID],
    weights_tid: pl.Scalar[pl.TASK_ID],
    query_group_size: pl.constexpr,
):
    """把同组query的FP16系数放在对角块，一次Cube完成各自的head规约。"""
    coefficients = pl.create_tensor(
        [T_PAD // query_group_size * NATIVE_QLI_WEIGHT_ROWS, query_group_size * IDX_N_HEADS], dtype=pl.FP16
    )
    coefficient_scales = pl.reshape(qr_hadamard_scale_dq, [T_PAD, IDX_N_HEADS])
    # Trim only workers whose original stride-48 loop has no iteration.
    # Keep every nonempty worker's query groups and arithmetic unchanged.
    coefficient_workers = pl.min(
        TOPK_QUERY_WORKERS, pl.tensor.dim(position_ids, 0) // query_group_size
    )
    with pl.spmd(
        coefficient_workers,
        name_hint="indexer_head_coefficients",
        deps=[qh_quant_tid, weights_tid],
        allow_early_resolve=True,
    ) as coefficients_tid:
        coefficient_worker = pl.tile.get_block_idx()
        coefficient_count = pl.tensor.dim(position_ids, 0)
        for coefficient_pair in pl.range(coefficient_worker, coefficient_count // query_group_size, TOPK_QUERY_WORKERS):
            # Equal-width UB rows avoid A3's partial-column TMOV restriction.
            # Block row lane*(group+1) is the original diagonal [lane, lane*64].
            coefficient_rows = pl.tile.full(
                [NATIVE_QLI_WEIGHT_ROWS * query_group_size, IDX_N_HEADS], dtype=pl.FP16, value=0.0
            )
            # Native ProcessVec0 loads and multiplies the complete S1 group.
            # Keep the original FP16 rounding order and diagonal publication.
            coefficient_begin = coefficient_pair * query_group_size
            query_scales = pl.load(
                coefficient_scales, [coefficient_begin, 0], [query_group_size, IDX_N_HEADS]
            )
            query_weights = pl.load(weights, [coefficient_begin, 0], [query_group_size, IDX_N_HEADS])
            query_scale_half = pl.cast(query_scales, pl.FP16, mode="rint")
            query_weight_half = pl.cast(query_weights, pl.FP16, mode="rint")
            head_coefficients = pl.mul(query_scale_half, query_weight_half)
            for coefficient_lane in pl.unroll(query_group_size):
                head_coefficient = pl.tile.extract(
                    head_coefficients, coefficient_lane, 0, [1, IDX_N_HEADS], target_memory=pl.MemorySpace.Vec
                )
                coefficient_rows = pl.tile.assemble(
                    coefficient_rows, head_coefficient, [coefficient_lane * (query_group_size + 1), 0]
                )
            # Keep the complete zero-padded diagonal block in UB, then publish it
            # once, like Native ProcessVec0's UB layout followed by one CopyOut.
            coefficient_matrix = pl.tile.reshape(
                coefficient_rows, [NATIVE_QLI_WEIGHT_ROWS, query_group_size * IDX_N_HEADS]
            )
            pl.store(coefficient_matrix, [coefficient_pair * NATIVE_QLI_WEIGHT_ROWS, 0], coefficients)
    return coefficients, coefficients_tid


@pl.jit.inline(auto_scope=False)
def indexer_score_topk_native_cube(
    qr_hadamard_i8: pl.Tensor[[T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8],
    qr_hadamard_scale_dq: pl.Tensor[[T_PAD * IDX_N_HEADS, 1], pl.FP32],
    weights: pl.Tensor[[T_PAD, IDX_N_HEADS], pl.FP32],
    idx_native_kv_cache: pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8],
    idx_block_table: pl.Tensor[[B_DYN, INDEXER_TABLE_COLUMNS_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    score_arena: pl.Tensor[[SCORE_ARENA_ROWS, TOPK_CANDIDATES_PER_LEAF], pl.FP32],
    pair_arena: pl.Tensor[[TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], pl.FP32],
    qh_quant_tid: pl.Scalar[pl.TASK_ID],
    weights_tid: pl.Scalar[pl.TASK_ID],
    cache_write_tid: pl.Scalar[pl.TASK_ID],
    query_group_size: pl.constexpr,
    qk_panel_cols: pl.constexpr,
    score_tile: pl.constexpr,
    score_sync_start: pl.constexpr,
    score_early_resolve: pl.constexpr,
    key_prefetch_panels: pl.constexpr,
    key_prefetch_to_l0: pl.constexpr,
    balance_leaves: pl.constexpr,
    split_score_rows: pl.constexpr,
    scale_layout: pl.Tensor[[SCALE_LAYOUT_CAPACITY], pl.INT32],
    layout_tid: pl.Scalar[pl.TASK_ID],
):
    """同组query共用Key；长G6可用双M192早FIX，保持单MIX双槽READY/ACK。"""
    native_page_bytes = pl.tensor.dim(idx_native_kv_cache, 1)
    # Zero-copy GM descriptors inside orchestration, as validation log §116.
    # One writable root allocation avoids partial-overlap Torch ABI arguments.
    cache_bytes = pl.tensor.dim(idx_native_kv_cache, 0) * native_page_bytes
    cache_flat = pl.reshape(idx_native_kv_cache, [cache_bytes])
    key_rows = cache_bytes // IDX_HEAD_DIM
    shifted_rows = (cache_bytes - 64) // IDX_HEAD_DIM
    idx_kv_cache = pl.reshape(cache_flat[0 : key_rows * IDX_HEAD_DIM], [key_rows, IDX_HEAD_DIM])
    idx_kv_cache_shift64 = pl.reshape(cache_flat[64 : 64 + shifted_rows * IDX_HEAD_DIM], [shifted_rows, IDX_HEAD_DIM])
    coefficients, coefficients_tid = indexer_head_coefficients(
        qr_hadamard_scale_dq,
        weights,
        position_ids,
        qh_quant_tid,
        weights_tid,
        query_group_size,
    )
    # 显式值通道，避免多层inline后仅attrs引用的TaskId失去绑定。
    proof_dependencies = pl.array.create(1, pl.TASK_ID)
    proof_dependencies[0] = layout_tid
    # 每个worker两个通信槽，每槽容纳整个query组的分数。
    buf_score_transfer = pl.create_tensor([TOPK_SCORE_WORKERS * 2 * query_group_size, score_tile], dtype=pl.FP32)
    buf_score_ffts = pl.create_tensor([256], dtype=pl.INT64)
    with pl.spmd(
        TOPK_SCORE_WORKERS,
        name_hint="indexer_score_topk_native_pair",
        deps=[coefficients_tid, cache_write_tid, proof_dependencies[0]],
        sync_start=score_sync_start,
        allow_early_resolve=score_early_resolve,
    ) as buffered_leaf_tid:
        buf_worker = pl.tile.get_block_idx()
        buf_query_count = pl.tensor.dim(position_ids, 0)
        buf_max_cache_len = 0
        for buf_batch in pl.range(buf_query_count // S):
            buf_max_cache_len = pl.max(buf_max_cache_len, pl.read(kv_seq_lens, [buf_batch]) // COMPRESS_RATIO)
        buf_capped_history = pl.min(buf_max_cache_len, TOPK_MAX_CANDIDATES)
        buf_max_leaves = pl.max((buf_capped_history + TOPK_CANDIDATES_PER_LEAF - 1) // TOPK_CANDIDATES_PER_LEAF, 1)
        buf_leaf_tiles = TOPK_CANDIDATES_PER_LEAF // BUFFERED_LONG_SCORE_TILE
        buf_extra_leaves = 0
        if balance_leaves:
            buf_max_leaves, buf_leaf_tiles, buf_extra_leaves = indexer_long_leaf_plan(
                buf_capped_history, buf_query_count // query_group_size
            )
        pl.system.set_ffts(buf_score_ffts)
        # S=6可作为一整组或三组2，均不跨请求；共享Key用组内最后query可见范围。
        if split_score_rows:
            for buf_item in pl.range(
                buf_worker, buf_query_count // query_group_size * buf_max_leaves, TOPK_SCORE_WORKERS
            ):
                if buf_query_count < query_group_size * TOPK_SCORE_WORKERS:
                    buf_query = buf_item % (buf_query_count // query_group_size) * query_group_size
                    buf_leaf = buf_item // (buf_query_count // query_group_size)
                else:
                    buf_query = buf_item // buf_max_leaves * query_group_size
                    buf_leaf = buf_item % buf_max_leaves
                buf_batch_idx = buf_query // S
                buf_last_position = pl.read(position_ids, [buf_query + query_group_size - 1])
                buf_cache_len = pl.read(kv_seq_lens, [buf_batch_idx]) // COMPRESS_RATIO
                buf_visible_count = pl.max(
                    pl.min(pl.min(buf_cache_len, (buf_last_position + 1) // COMPRESS_RATIO), TOPK_MAX_CANDIDATES), 0
                )
                buf_logical_begin = buf_leaf * TOPK_CANDIDATES_PER_LEAF
                buf_leaf_capacity = TOPK_CANDIDATES_PER_LEAF
                if balance_leaves:
                    buf_logical_begin = (
                        buf_leaf * buf_leaf_tiles + pl.min(buf_leaf, buf_extra_leaves)
                    ) * BUFFERED_LONG_SCORE_TILE
                    buf_leaf_capacity = (
                        buf_leaf_tiles + pl.cast(buf_leaf < buf_extra_leaves, pl.INDEX)
                    ) * BUFFERED_LONG_SCORE_TILE
                if buf_logical_begin < buf_visible_count:
                    buf_valid_count = pl.min(buf_leaf_capacity, buf_visible_count - buf_logical_begin)
                    buf_tile_count = (buf_valid_count + score_tile - 1) // score_tile
                    buf_lane_span = pl.min(buf_tile_count * (score_tile // 2), TOPK_CANDIDATES_PER_LEAF // 2)
                    buf_score_iters = (buf_lane_span + score_tile // 2 - 1) // (score_tile // 2)
                    buf_query_vector = pl.load(
                        qr_hadamard_i8,
                        [buf_query * IDX_N_HEADS, 0],
                        [query_group_size * IDX_N_HEADS, IDX_HEAD_DIM],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    buf_coefficients_l1 = pl.load(
                        coefficients,
                        [buf_query // query_group_size * NATIVE_QLI_WEIGHT_ROWS, 0],
                        [NATIVE_QLI_WEIGHT_ROWS, query_group_size * IDX_N_HEADS],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    split_query0 = pl.tile.extract(
                        buf_query_vector,
                        0 * (query_group_size // 2 * IDX_N_HEADS),
                        0,
                        [query_group_size // 2 * IDX_N_HEADS, IDX_HEAD_DIM],
                        target_memory=pl.MemorySpace.Left,
                    )
                    split_coefficient0 = pl.tile.extract(
                        buf_coefficients_l1,
                        0,
                        0 * (query_group_size // 2 * IDX_N_HEADS),
                        [NATIVE_QLI_WEIGHT_ROWS, query_group_size // 2 * IDX_N_HEADS],
                        target_memory=pl.MemorySpace.Left,
                    )
                    split_qk_acc0 = pl.tile.create(
                        [query_group_size // 2 * IDX_N_HEADS, qk_panel_cols],
                        dtype=pl.INT32,
                        target_memory=pl.MemorySpace.Acc,
                    )
                    split_query1 = pl.tile.extract(
                        buf_query_vector,
                        1 * (query_group_size // 2 * IDX_N_HEADS),
                        0,
                        [query_group_size // 2 * IDX_N_HEADS, IDX_HEAD_DIM],
                        target_memory=pl.MemorySpace.Left,
                    )
                    split_coefficient1 = pl.tile.extract(
                        buf_coefficients_l1,
                        0,
                        1 * (query_group_size // 2 * IDX_N_HEADS),
                        [NATIVE_QLI_WEIGHT_ROWS, query_group_size // 2 * IDX_N_HEADS],
                        target_memory=pl.MemorySpace.Left,
                    )
                    split_qk_acc1 = pl.tile.create(
                        [query_group_size // 2 * IDX_N_HEADS, qk_panel_cols],
                        dtype=pl.INT32,
                        target_memory=pl.MemorySpace.Acc,
                    )
                    split_ws_acc = pl.tile.create(
                        [NATIVE_QLI_WEIGHT_ROWS, qk_panel_cols], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc
                    )
                    for buf_score_step in pl.range(buf_score_iters):
                        if buf_score_step >= 2:
                            pl.system.sync_wait(
                                SCORE_CONSUMED_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC
                            )
                        buf_score_begin = buf_score_step * (score_tile // 2)
                        if balance_leaves:
                            buf_score_begin = buf_score_step * score_tile
                        buf_transfer_row = buf_worker * 2 * query_group_size + buf_score_step % 2 * query_group_size
                        split_previous0 = pl.tile.create(
                            [query_group_size // 2 * IDX_N_HEADS, qk_panel_cols],
                            dtype=pl.FP16,
                            target_memory=pl.MemorySpace.Mat,
                        )
                        split_previous1 = pl.tile.create(
                            [query_group_size // 2 * IDX_N_HEADS, qk_panel_cols],
                            dtype=pl.FP16,
                            target_memory=pl.MemorySpace.Mat,
                        )
                        buf_current_key = pl.tile.create(
                            [IDX_HEAD_DIM, qk_panel_cols], dtype=pl.INT8, target_memory=pl.MemorySpace.Right
                        )
                        buf_prefetched_key = pl.tile.create(
                            [IDX_HEAD_DIM, qk_panel_cols], dtype=pl.INT8, target_memory=pl.MemorySpace.Right
                        )
                        buf_key_l1_pool = pl.tile.create(
                            [2 * qk_panel_cols, IDX_HEAD_DIM], dtype=pl.INT8, target_memory=pl.MemorySpace.Mat
                        )
                        if key_prefetch_panels > 0:
                            buf_prime_col = 0
                            for buf_prime_key_page in pl.unroll(qk_panel_cols // BLOCK_SIZE):
                                buf_prime_key_row = (
                                    buf_logical_begin
                                    + buf_score_begin
                                    + buf_prime_col // (score_tile // 2) * buf_lane_span
                                    + buf_prime_col % (score_tile // 2)
                                    + buf_prime_key_page * BLOCK_SIZE
                                )
                                if balance_leaves:
                                    buf_prime_key_row = (
                                        buf_logical_begin
                                        + buf_score_begin
                                        + buf_prime_col
                                        + buf_prime_key_page * BLOCK_SIZE
                                    )
                                buf_prime_safe_page = pl.min(
                                    buf_prime_key_row // BLOCK_SIZE, pl.max((buf_cache_len - 1) // BLOCK_SIZE, 0)
                                )
                                buf_prime_physical_page = pl.max(
                                    pl.cast(pl.read(idx_block_table, [buf_batch_idx, buf_prime_safe_page]), pl.INDEX), 0
                                )
                                buf_prime_native_byte = buf_prime_physical_page * native_page_bytes
                                if buf_prime_native_byte % IDX_HEAD_DIM == 0:
                                    buf_key_l1_pool = pl.gather_row(
                                        buf_key_l1_pool,
                                        idx_kv_cache,
                                        [buf_prime_key_page * BLOCK_SIZE, 0],
                                        [buf_prime_native_byte // IDX_HEAD_DIM, 0],
                                        [BLOCK_SIZE, IDX_HEAD_DIM],
                                    )
                                else:
                                    buf_key_l1_pool = pl.gather_row(
                                        buf_key_l1_pool,
                                        idx_kv_cache_shift64,
                                        [buf_prime_key_page * BLOCK_SIZE, 0],
                                        [buf_prime_native_byte // IDX_HEAD_DIM, 0],
                                        [BLOCK_SIZE, IDX_HEAD_DIM],
                                    )
                            if key_prefetch_to_l0:
                                buf_current_key = pl.tile.extract(
                                    pl.tile.transpose_view(buf_key_l1_pool),
                                    0,
                                    0,
                                    [IDX_HEAD_DIM, qk_panel_cols],
                                    target_memory=pl.MemorySpace.Right,
                                )
                        for buf_qk_panel in pl.unroll(score_tile // qk_panel_cols):
                            buf_panel_col = buf_qk_panel * qk_panel_cols
                            buf_load_col = (buf_qk_panel + key_prefetch_panels) * qk_panel_cols
                            buf_load_slot = (buf_qk_panel + key_prefetch_panels) % 2 * qk_panel_cols
                            if key_prefetch_panels > 0:
                                if not key_prefetch_to_l0:
                                    buf_current_key = pl.tile.extract(
                                        pl.tile.transpose_view(buf_key_l1_pool),
                                        0,
                                        buf_qk_panel % 2 * qk_panel_cols,
                                        [IDX_HEAD_DIM, qk_panel_cols],
                                        target_memory=pl.MemorySpace.Right,
                                    )
                            if buf_qk_panel + key_prefetch_panels < score_tile // qk_panel_cols:
                                buf_panel_keys = pl.tile.create(
                                    [qk_panel_cols, IDX_HEAD_DIM], dtype=pl.INT8, target_memory=pl.MemorySpace.Mat
                                )
                                for buf_key_page in pl.unroll(qk_panel_cols // BLOCK_SIZE):
                                    buf_key_row = (
                                        buf_logical_begin
                                        + buf_score_begin
                                        + buf_load_col // (score_tile // 2) * buf_lane_span
                                        + buf_load_col % (score_tile // 2)
                                        + buf_key_page * BLOCK_SIZE
                                    )
                                    if balance_leaves:
                                        buf_key_row = (
                                            buf_logical_begin
                                            + buf_score_begin
                                            + buf_load_col
                                            + buf_key_page * BLOCK_SIZE
                                        )
                                    buf_safe_page = pl.min(
                                        buf_key_row // BLOCK_SIZE, pl.max((buf_cache_len - 1) // BLOCK_SIZE, 0)
                                    )
                                    buf_physical_page = pl.max(
                                        pl.cast(pl.read(idx_block_table, [buf_batch_idx, buf_safe_page]), pl.INDEX), 0
                                    )
                                    buf_native_byte = buf_physical_page * native_page_bytes
                                    if buf_native_byte % IDX_HEAD_DIM == 0:
                                        if key_prefetch_panels > 0:
                                            buf_key_l1_pool = pl.gather_row(
                                                buf_key_l1_pool,
                                                idx_kv_cache,
                                                [buf_load_slot + buf_key_page * BLOCK_SIZE, 0],
                                                [buf_native_byte // IDX_HEAD_DIM, 0],
                                                [BLOCK_SIZE, IDX_HEAD_DIM],
                                            )
                                        else:
                                            buf_panel_keys = pl.gather_row(
                                                buf_panel_keys,
                                                idx_kv_cache,
                                                [buf_key_page * BLOCK_SIZE, 0],
                                                [buf_native_byte // IDX_HEAD_DIM, 0],
                                                [BLOCK_SIZE, IDX_HEAD_DIM],
                                            )
                                    elif key_prefetch_panels > 0:
                                        buf_key_l1_pool = pl.gather_row(
                                            buf_key_l1_pool,
                                            idx_kv_cache_shift64,
                                            [buf_load_slot + buf_key_page * BLOCK_SIZE, 0],
                                            [buf_native_byte // IDX_HEAD_DIM, 0],
                                            [BLOCK_SIZE, IDX_HEAD_DIM],
                                        )
                                    else:
                                        buf_panel_keys = pl.gather_row(
                                            buf_panel_keys,
                                            idx_kv_cache_shift64,
                                            [buf_key_page * BLOCK_SIZE, 0],
                                            [buf_native_byte // IDX_HEAD_DIM, 0],
                                            [BLOCK_SIZE, IDX_HEAD_DIM],
                                        )
                                if key_prefetch_panels > 0:
                                    if key_prefetch_to_l0:
                                        buf_prefetched_key = pl.tile.extract(
                                            pl.tile.transpose_view(buf_key_l1_pool),
                                            0,
                                            buf_load_slot,
                                            [IDX_HEAD_DIM, qk_panel_cols],
                                            target_memory=pl.MemorySpace.Right,
                                        )
                                else:
                                    buf_prefetched_key = pl.tile.move(
                                        pl.tile.transpose_view(buf_panel_keys), target_memory=pl.MemorySpace.Right
                                    )
                            if key_prefetch_panels > 0:
                                buf_key_panel = buf_current_key
                            else:
                                buf_key_panel = buf_prefetched_key
                            split_current0 = pl.tile.create(
                                [query_group_size // 2 * IDX_N_HEADS, qk_panel_cols],
                                dtype=pl.FP16,
                                target_memory=pl.MemorySpace.Mat,
                            )
                            split_current1 = pl.tile.create(
                                [query_group_size // 2 * IDX_N_HEADS, qk_panel_cols],
                                dtype=pl.FP16,
                                target_memory=pl.MemorySpace.Mat,
                            )
                            if buf_qk_panel > 0:
                                split_right0 = pl.tile.extract(
                                    split_previous0,
                                    0,
                                    0,
                                    [query_group_size // 2 * IDX_N_HEADS, qk_panel_cols],
                                    target_memory=pl.MemorySpace.Right,
                                )
                                split_right1 = pl.tile.extract(
                                    split_previous1,
                                    0,
                                    0,
                                    [query_group_size // 2 * IDX_N_HEADS, qk_panel_cols],
                                    target_memory=pl.MemorySpace.Right,
                                )
                                split_qk0 = pl.tile.matmul_acc(
                                    split_qk_acc0, split_query0, buf_key_panel, init_cond=True
                                )
                                split_current0 = pl.tile.assemble(
                                    split_current0,
                                    split_qk0,
                                    [0, 0],
                                    pre_quant=1.0 / BUFFERED_SCORE_SCALE,
                                    pre_relu=True,
                                )
                                split_qk1 = pl.tile.matmul_acc(
                                    split_qk_acc1, split_query1, buf_key_panel, init_cond=True
                                )
                                split_current1 = pl.tile.assemble(
                                    split_current1,
                                    split_qk1,
                                    [0, 0],
                                    pre_quant=1.0 / BUFFERED_SCORE_SCALE,
                                    pre_relu=True,
                                )
                                split_ws_first = pl.tile.matmul_acc(
                                    split_ws_acc, split_coefficient0, split_right0, init_cond=True
                                )
                                split_ws_done = pl.tile.matmul_acc(split_ws_first, split_coefficient1, split_right1)
                                pl.store(
                                    pl.set_validshape(split_ws_done, query_group_size, qk_panel_cols),
                                    [buf_transfer_row, buf_panel_col - qk_panel_cols],
                                    buf_score_transfer,
                                )
                            else:
                                split_qk0 = pl.tile.matmul_acc(
                                    split_qk_acc0, split_query0, buf_key_panel, init_cond=True
                                )
                                split_current0 = pl.tile.assemble(
                                    split_current0,
                                    split_qk0,
                                    [0, 0],
                                    pre_quant=1.0 / BUFFERED_SCORE_SCALE,
                                    pre_relu=True,
                                )
                                split_qk1 = pl.tile.matmul_acc(
                                    split_qk_acc1, split_query1, buf_key_panel, init_cond=True
                                )
                                split_current1 = pl.tile.assemble(
                                    split_current1,
                                    split_qk1,
                                    [0, 0],
                                    pre_quant=1.0 / BUFFERED_SCORE_SCALE,
                                    pre_relu=True,
                                )
                            split_previous0 = split_current0
                            split_previous1 = split_current1
                            if key_prefetch_panels > 0:
                                if key_prefetch_to_l0:
                                    if buf_qk_panel + 1 < score_tile // qk_panel_cols:
                                        buf_current_key = buf_prefetched_key
                        split_right0_last = pl.tile.extract(
                            split_previous0,
                            0,
                            0,
                            [query_group_size // 2 * IDX_N_HEADS, qk_panel_cols],
                            target_memory=pl.MemorySpace.Right,
                        )
                        split_right1_last = pl.tile.extract(
                            split_previous1,
                            0,
                            0,
                            [query_group_size // 2 * IDX_N_HEADS, qk_panel_cols],
                            target_memory=pl.MemorySpace.Right,
                        )
                        split_ws_first_last = pl.tile.matmul_acc(
                            split_ws_acc, split_coefficient0, split_right0_last, init_cond=True
                        )
                        split_ws_done_last = pl.tile.matmul_acc(
                            split_ws_first_last, split_coefficient1, split_right1_last
                        )
                        pl.store(
                            pl.set_validshape(split_ws_done_last, query_group_size, qk_panel_cols),
                            [buf_transfer_row, score_tile - qk_panel_cols],
                            buf_score_transfer,
                        )
                        pl.system.sync_set(
                            SCORE_READY_EVENT, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC
                        )
                    for buf_score_drain in pl.range(pl.min(buf_score_iters, 2)):
                        pl.system.sync_wait(SCORE_CONSUMED_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
        else:
            for buf_item in pl.range(
                buf_worker, buf_query_count // query_group_size * buf_max_leaves, TOPK_SCORE_WORKERS
            ):
                if buf_query_count < query_group_size * TOPK_SCORE_WORKERS:
                    # 小批次先分完整leaf，避免极短尾leaf占掉完整leaf的轮转名额。
                    buf_query = buf_item % (buf_query_count // query_group_size) * query_group_size
                    buf_leaf = buf_item // (buf_query_count // query_group_size)
                else:
                    buf_query = buf_item // buf_max_leaves * query_group_size
                    buf_leaf = buf_item % buf_max_leaves
                buf_batch_idx = buf_query // S
                buf_last_position = pl.read(position_ids, [buf_query + query_group_size - 1])
                buf_cache_len = pl.read(kv_seq_lens, [buf_batch_idx]) // COMPRESS_RATIO
                buf_visible_count = pl.max(
                    pl.min(pl.min(buf_cache_len, (buf_last_position + 1) // COMPRESS_RATIO), TOPK_MAX_CANDIDATES), 0
                )
                buf_logical_begin = buf_leaf * TOPK_CANDIDATES_PER_LEAF
                buf_leaf_capacity = TOPK_CANDIDATES_PER_LEAF
                if balance_leaves:
                    buf_logical_begin = (
                        buf_leaf * buf_leaf_tiles + pl.min(buf_leaf, buf_extra_leaves)
                    ) * BUFFERED_LONG_SCORE_TILE
                    buf_leaf_capacity = (
                        buf_leaf_tiles + pl.cast(buf_leaf < buf_extra_leaves, pl.INDEX)
                    ) * BUFFERED_LONG_SCORE_TILE
                if buf_logical_begin < buf_visible_count:
                    buf_valid_count = pl.min(buf_leaf_capacity, buf_visible_count - buf_logical_begin)
                    buf_tile_count = (buf_valid_count + score_tile - 1) // score_tile
                    buf_lane_span = pl.min(buf_tile_count * (score_tile // 2), TOPK_CANDIDATES_PER_LEAF // 2)
                    buf_score_iters = (buf_lane_span + (score_tile // 2) - 1) // (score_tile // 2)
                    buf_query_vector = pl.load(
                        qr_hadamard_i8,
                        [buf_query * IDX_N_HEADS, 0],
                        [query_group_size * IDX_N_HEADS, IDX_HEAD_DIM],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    buf_coefficients_l1 = pl.load(
                        coefficients,
                        [buf_query // query_group_size * NATIVE_QLI_WEIGHT_ROWS, 0],
                        [NATIVE_QLI_WEIGHT_ROWS, query_group_size * IDX_N_HEADS],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    buf_query_left = pl.tile.move(buf_query_vector, target_memory=pl.MemorySpace.Left)
                    buf_coefficient_pair = pl.tile.move(buf_coefficients_l1, target_memory=pl.MemorySpace.Left)
                    for buf_score_step in pl.range(buf_score_iters):
                        if buf_score_step >= 2:
                            pl.system.sync_wait(
                                SCORE_CONSUMED_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC
                            )
                        buf_score_begin = buf_score_step * (score_tile // 2)
                        if balance_leaves:
                            buf_score_begin = buf_score_step * score_tile
                        buf_transfer_row = buf_worker * 2 * query_group_size + buf_score_step % 2 * query_group_size
                        buf_previous_l1 = pl.tile.create(
                            [query_group_size * IDX_N_HEADS, qk_panel_cols],
                            dtype=pl.FP16,
                            target_memory=pl.MemorySpace.Mat,
                        )
                        buf_current_key = pl.tile.create(
                            [IDX_HEAD_DIM, qk_panel_cols], dtype=pl.INT8, target_memory=pl.MemorySpace.Right
                        )
                        buf_prefetched_key = pl.tile.create(
                            [IDX_HEAD_DIM, qk_panel_cols], dtype=pl.INT8, target_memory=pl.MemorySpace.Right
                        )
                        # Native QLI keeps Key L1 separate from FIXPIPE's Score L1.
                        # A live two-panel pool prevents the allocator from recycling
                        # each next-Key staging tile as the current Score destination.
                        buf_key_l1_pool = pl.tile.create(
                            [2 * qk_panel_cols, IDX_HEAD_DIM], dtype=pl.INT8, target_memory=pl.MemorySpace.Mat
                        )
                        # Keep QK/WS outside a conditional prologue so their Mat
                        # intermediates retain the original lifetimes.
                        if key_prefetch_panels > 0:
                            buf_prime_col = 0
                            # Native QLI streams N128 key panels, overlapping the next
                            # MTE2 load with QK/WS instead of loading the entire N768/1024.
                            for buf_prime_key_page in pl.unroll(qk_panel_cols // BLOCK_SIZE):
                                buf_prime_key_row = (
                                    buf_logical_begin
                                    + buf_score_begin
                                    + (buf_prime_col // (score_tile // 2)) * buf_lane_span
                                    + buf_prime_col % (score_tile // 2)
                                    + buf_prime_key_page * BLOCK_SIZE
                                )
                                if balance_leaves:
                                    buf_prime_key_row = (
                                        buf_logical_begin
                                        + buf_score_begin
                                        + buf_prime_col
                                        + buf_prime_key_page * BLOCK_SIZE
                                    )
                                buf_prime_safe_page = pl.min(
                                    buf_prime_key_row // BLOCK_SIZE, pl.max((buf_cache_len - 1) // BLOCK_SIZE, 0)
                                )
                                buf_prime_physical_page = pl.max(
                                    pl.cast(pl.read(idx_block_table, [buf_batch_idx, buf_prime_safe_page]), pl.INDEX), 0
                                )
                                buf_prime_native_byte = buf_prime_physical_page * native_page_bytes
                                if buf_prime_native_byte % IDX_HEAD_DIM == 0:
                                    buf_key_l1_pool = pl.gather_row(
                                        buf_key_l1_pool,
                                        idx_kv_cache,
                                        [buf_prime_key_page * BLOCK_SIZE, 0],
                                        [buf_prime_native_byte // IDX_HEAD_DIM, 0],
                                        [BLOCK_SIZE, IDX_HEAD_DIM],
                                    )
                                else:
                                    buf_key_l1_pool = pl.gather_row(
                                        buf_key_l1_pool,
                                        idx_kv_cache_shift64,
                                        [buf_prime_key_page * BLOCK_SIZE, 0],
                                        [buf_prime_native_byte // IDX_HEAD_DIM, 0],
                                        [BLOCK_SIZE, IDX_HEAD_DIM],
                                    )
                            if key_prefetch_to_l0:
                                buf_current_key = pl.tile.extract(
                                    pl.tile.transpose_view(buf_key_l1_pool),
                                    0,
                                    0,
                                    [IDX_HEAD_DIM, qk_panel_cols],
                                    target_memory=pl.MemorySpace.Right,
                                )
                        for buf_qk_panel in pl.unroll(score_tile // qk_panel_cols):
                            buf_panel_col = buf_qk_panel * qk_panel_cols
                            buf_load_col = (buf_qk_panel + key_prefetch_panels) * qk_panel_cols
                            buf_load_slot = (buf_qk_panel + key_prefetch_panels) % 2 * qk_panel_cols
                            if key_prefetch_panels > 0:
                                if not key_prefetch_to_l0:
                                    # M192/N128 leaves only one 16KiB Key Right tile.
                                    # Extract the current Key before staging the next
                                    # L1 panel so its move does not depend on that DMA.
                                    buf_current_key = pl.tile.extract(
                                        pl.tile.transpose_view(buf_key_l1_pool),
                                        0,
                                        buf_qk_panel % 2 * qk_panel_cols,
                                        [IDX_HEAD_DIM, qk_panel_cols],
                                        target_memory=pl.MemorySpace.Right,
                                    )
                            if buf_qk_panel + key_prefetch_panels < score_tile // qk_panel_cols:
                                # Native QLI streams N128 key panels, overlapping the next
                                # MTE2 load with QK/WS instead of loading the entire N768/1024.
                                buf_panel_keys = pl.tile.create(
                                    [qk_panel_cols, IDX_HEAD_DIM],
                                    dtype=pl.INT8,
                                    target_memory=pl.MemorySpace.Mat,
                                )
                                for buf_key_page in pl.unroll(qk_panel_cols // BLOCK_SIZE):
                                    buf_key_row = (
                                        buf_logical_begin
                                        + buf_score_begin
                                        + (buf_load_col // (score_tile // 2)) * buf_lane_span
                                        + buf_load_col % (score_tile // 2)
                                        + buf_key_page * BLOCK_SIZE
                                    )
                                    if balance_leaves:
                                        buf_key_row = (
                                            buf_logical_begin
                                            + buf_score_begin
                                            + buf_load_col
                                            + buf_key_page * BLOCK_SIZE
                                        )
                                    buf_safe_page = pl.min(
                                        buf_key_row // BLOCK_SIZE, pl.max((buf_cache_len - 1) // BLOCK_SIZE, 0)
                                    )
                                    buf_physical_page = pl.max(
                                        pl.cast(pl.read(idx_block_table, [buf_batch_idx, buf_safe_page]), pl.INDEX), 0
                                    )
                                    buf_native_byte = buf_physical_page * native_page_bytes
                                    if buf_native_byte % IDX_HEAD_DIM == 0:
                                        if key_prefetch_panels > 0:
                                            buf_key_l1_pool = pl.gather_row(
                                                buf_key_l1_pool,
                                                idx_kv_cache,
                                                [buf_load_slot + buf_key_page * BLOCK_SIZE, 0],
                                                [buf_native_byte // IDX_HEAD_DIM, 0],
                                                [BLOCK_SIZE, IDX_HEAD_DIM],
                                            )
                                        else:
                                            buf_panel_keys = pl.gather_row(
                                                buf_panel_keys,
                                                idx_kv_cache,
                                                [buf_key_page * BLOCK_SIZE, 0],
                                                [buf_native_byte // IDX_HEAD_DIM, 0],
                                                [BLOCK_SIZE, IDX_HEAD_DIM],
                                            )
                                    else:
                                        if key_prefetch_panels > 0:
                                            buf_key_l1_pool = pl.gather_row(
                                                buf_key_l1_pool,
                                                idx_kv_cache_shift64,
                                                [buf_load_slot + buf_key_page * BLOCK_SIZE, 0],
                                                [buf_native_byte // IDX_HEAD_DIM, 0],
                                                [BLOCK_SIZE, IDX_HEAD_DIM],
                                            )
                                        else:
                                            buf_panel_keys = pl.gather_row(
                                                buf_panel_keys,
                                                idx_kv_cache_shift64,
                                                [buf_key_page * BLOCK_SIZE, 0],
                                                [buf_native_byte // IDX_HEAD_DIM, 0],
                                                [BLOCK_SIZE, IDX_HEAD_DIM],
                                            )
                                if key_prefetch_panels > 0:
                                    if key_prefetch_to_l0:
                                        buf_prefetched_key = pl.tile.extract(
                                            pl.tile.transpose_view(buf_key_l1_pool),
                                            0,
                                            buf_load_slot,
                                            [IDX_HEAD_DIM, qk_panel_cols],
                                            target_memory=pl.MemorySpace.Right,
                                        )
                                else:
                                    buf_prefetched_key = pl.tile.move(
                                        pl.tile.transpose_view(buf_panel_keys),
                                        target_memory=pl.MemorySpace.Right,
                                    )
                            if key_prefetch_panels > 0:
                                buf_key_panel = buf_current_key
                            else:
                                buf_key_panel = buf_prefetched_key
                            buf_scores_l1 = pl.tile.create(
                                [query_group_size * IDX_N_HEADS, qk_panel_cols],
                                dtype=pl.FP16,
                                target_memory=pl.MemorySpace.Mat,
                            )
                            if buf_qk_panel > 0:
                                buf_previous_pair = pl.tile.extract(
                                    buf_previous_l1,
                                    0,
                                    0,
                                    [query_group_size * IDX_N_HEADS, qk_panel_cols],
                                    target_memory=pl.MemorySpace.Right,
                                )
                                # QK与上一panel的head规约同时存活；M192的Acc占96KiB，WS另占8KiB。
                                buf_scores = pl.tile.matmul(buf_query_left, buf_key_panel)
                                buf_reduced_pair = pl.tile.matmul(buf_coefficient_pair, buf_previous_pair)
                                buf_scores_l1 = pl.tile.assemble(
                                    buf_scores_l1,
                                    buf_scores,
                                    [0, 0],
                                    pre_quant=1.0 / BUFFERED_SCORE_SCALE,
                                    pre_relu=True,
                                )
                                pl.store(
                                    pl.set_validshape(buf_reduced_pair, query_group_size, qk_panel_cols),
                                    [buf_transfer_row, buf_panel_col - qk_panel_cols],
                                    buf_score_transfer,
                                )
                            else:
                                buf_scores = pl.tile.matmul(buf_query_left, buf_key_panel)
                                buf_scores_l1 = pl.tile.assemble(
                                    buf_scores_l1,
                                    buf_scores,
                                    [0, 0],
                                    pre_quant=1.0 / BUFFERED_SCORE_SCALE,
                                    pre_relu=True,
                                )
                            buf_previous_l1 = buf_scores_l1
                            if key_prefetch_panels > 0:
                                if key_prefetch_to_l0:
                                    if buf_qk_panel + 1 < score_tile // qk_panel_cols:
                                        buf_current_key = buf_prefetched_key
                        buf_last_pair = pl.tile.extract(
                            buf_previous_l1,
                            0,
                            0,
                            [query_group_size * IDX_N_HEADS, qk_panel_cols],
                            target_memory=pl.MemorySpace.Right,
                        )
                        buf_last_scores = pl.tile.matmul(buf_coefficient_pair, buf_last_pair)
                        pl.store(
                            pl.set_validshape(buf_last_scores, query_group_size, qk_panel_cols),
                            [buf_transfer_row, score_tile - qk_panel_cols],
                            buf_score_transfer,
                        )
                        pl.system.sync_set(
                            SCORE_READY_EVENT, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC
                        )
                    for buf_score_drain in pl.range(pl.min(buf_score_iters, 2)):
                        pl.system.sync_wait(SCORE_CONSUMED_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)

        for buf_score_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            if balance_leaves:
                for single_item in pl.range(
                    buf_worker, buf_query_count // query_group_size * buf_max_leaves, TOPK_SCORE_WORKERS
                ):
                    if buf_query_count < query_group_size * TOPK_SCORE_WORKERS:
                        single_query = single_item % (buf_query_count // query_group_size) * query_group_size
                        single_leaf = single_item // (buf_query_count // query_group_size)
                    else:
                        single_query = single_item // buf_max_leaves * query_group_size
                        single_leaf = single_item % buf_max_leaves
                    single_batch = single_query // S
                    single_cache_len = pl.read(kv_seq_lens, [single_batch]) // COMPRESS_RATIO
                    single_last_position = pl.read(position_ids, [single_query + query_group_size - 1])
                    single_visible = pl.max(
                        pl.min(
                            pl.min(single_cache_len, (single_last_position + 1) // COMPRESS_RATIO), TOPK_MAX_CANDIDATES
                        ),
                        0,
                    )
                    single_begin = (single_leaf * buf_leaf_tiles + pl.min(single_leaf, buf_extra_leaves)) * score_tile
                    single_capacity = (buf_leaf_tiles + pl.cast(single_leaf < buf_extra_leaves, pl.INDEX)) * score_tile
                    if single_begin < single_visible:
                        single_valid = pl.min(single_capacity, single_visible - single_begin)
                        single_steps = (single_valid + score_tile - 1) // score_tile
                        # 同一调用的完整页证明；后续每个leaf/两个AIV复用，不重复扫描。
                        single_page_begin = single_begin // BLOCK_SIZE
                        single_page_count = (single_valid + BLOCK_SIZE - 1) // BLOCK_SIZE
                        layout_row = 2 * single_batch
                        single_contiguous = pl.read(scale_layout, [layout_row + 1])
                        single_first_page = pl.cast(pl.read(scale_layout, [layout_row]), pl.INDEX)
                        single_first_page = single_first_page + pl.cast(single_contiguous, pl.INDEX) * single_page_begin
                        # Native QLI ProcessVec1 assigns complete query rows to each AIV.
                        # Keep one root per query and a two-step contiguous 2048-score segment.
                        single_roots = pl.tile.full(
                            [query_group_size // 2, TOPK_PAIR_WIDTH], dtype=pl.FP32, value=FP32_NEG_INF
                        )
                        single_segments = pl.tile.full(
                            [query_group_size, BUFFERED_LONG_SCORE_TILE], dtype=pl.FP32, value=FP32_NEG_INF
                        )
                        for single_step in pl.range(single_steps):
                            single_score_begin = single_begin + single_step * score_tile
                            single_transfer_row = (
                                buf_worker * 2 * query_group_size
                                + single_step % 2 * query_group_size
                                + buf_score_lane * (query_group_size // 2)
                            )
                            # Scales are shared across the three queries on one AIV. Reading
                            # the complete candidate tile duplicates scale traffic across AIVs.
                            single_scale_bytes = pl.tile.reinterpret_view(
                                pl.fillpad(
                                    pl.tile.full([1, score_tile], dtype=pl.FP16, value=0.0), pad_value=pl.PadValue.zero
                                ),
                                pl.INT8,
                            )
                            if single_contiguous != 0:
                                # Native每页的key后仍是原64B scale；源行距由Tensor描述符
                                # 的page_bytes提供。一次二维MTE2搬入多页，不碰key区域。
                                run_pages = pl.min(
                                    score_tile // BLOCK_SIZE,
                                    single_page_count - single_step * (score_tile // BLOCK_SIZE),
                                )
                                run_first = single_first_page + single_step * (score_tile // BLOCK_SIZE) * pl.cast(
                                    single_contiguous, pl.INDEX
                                )
                                run_lowest = run_first
                                if single_contiguous < 0:
                                    run_lowest = run_first - run_pages + 1
                                run_bytes = pl.load(
                                    idx_native_kv_cache,
                                    [run_lowest, INDEXER_KEY_BYTES],
                                    [score_tile // BLOCK_SIZE, BLOCK_SIZE * 2],
                                    valid_shape=[run_pages, BLOCK_SIZE * 2],
                                )
                                # 尾块只加载真实页。按FP16填零未加载的行，再展平，避免
                                # INT8常量填充的不支持及将UB未初始化数据带入后续数学。
                                run_half = pl.tile.reinterpret_view(run_bytes, pl.FP16)
                                run_clean = pl.fillpad(run_half, pad_value=pl.PadValue.zero)
                                run_full = pl.set_validshape(run_clean, score_tile // BLOCK_SIZE, BLOCK_SIZE)
                                run_flat = pl.reshape(run_full, [1, score_tile])
                                if single_contiguous < 0:
                                    # MTE2始终按物理地址递增读。逆序页在UB按64B页倒序，
                                    # 每页内部32个FP16保持原顺序。TGATHERB每索引搬32B，
                                    # 两个索引组成一页，仅需64个UINT32字节偏移。
                                    run_blocks = pl.tile.ci(0, [1, 2 * (score_tile // BLOCK_SIZE)], dtype=pl.INT32)
                                    run_page_ids = pl.tile.shrs(run_blocks, 1)
                                    run_halves = pl.sub(run_blocks, pl.mul(run_page_ids, 2))
                                    run_reverse_pages = pl.maximum(
                                        pl.add(pl.neg(run_page_ids), pl.cast(run_pages - 1, pl.INT32)), 0
                                    )
                                    run_offsets = pl.add(
                                        pl.mul(run_reverse_pages, BLOCK_SIZE * 2), pl.mul(run_halves, 32)
                                    )
                                    run_reversed = pl.tile.gatherb(
                                        run_flat, pl.tile.reinterpret_view(run_offsets, pl.UINT32)
                                    )
                                    # 尾页后的索引钳制到已加载第0页，不访问未加载或负地址；
                                    # 随后按真实页数补零，保证无效区不会带入陈旧UB数据。
                                    run_used = pl.set_validshape(run_reversed, 1, run_pages * BLOCK_SIZE)
                                    run_flat = pl.set_validshape(
                                        pl.fillpad(run_used, pad_value=pl.PadValue.zero), 1, score_tile
                                    )
                                single_scale_bytes = pl.tile.reinterpret_view(run_flat, pl.INT8)
                            else:
                                # 两AIV读取同一候选区间的scale；页表与slab容量检查先于gather。
                                for single_scale_page in pl.range(score_tile // BLOCK_SIZE):
                                    single_scale_logical = pl.min(
                                        single_score_begin // BLOCK_SIZE + single_scale_page,
                                        pl.max((single_cache_len - 1) // BLOCK_SIZE, 0),
                                    )
                                    if single_scale_logical >= 0 and single_scale_logical < pl.tensor.dim(
                                        idx_block_table, 1
                                    ):
                                        single_scale_physical = pl.cast(
                                            pl.read(idx_block_table, [single_batch, single_scale_logical]), pl.INDEX
                                        )
                                        if single_scale_physical >= 0 and single_scale_physical < pl.tensor.dim(
                                            idx_native_kv_cache, 0
                                        ):
                                            single_scale_bytes = pl.gather_row(
                                                single_scale_bytes,
                                                idx_native_kv_cache,
                                                [0, single_scale_page * BLOCK_SIZE * 2],
                                                [single_scale_physical, INDEXER_KEY_BYTES],
                                                [1, BLOCK_SIZE * 2],
                                            )
                            single_scale_half = pl.tile.reinterpret_view(single_scale_bytes, pl.FP16)
                            single_kv_scale = pl.cast(single_scale_half, pl.FP32)
                            pl.system.sync_wait(SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                            single_scores = pl.load(
                                buf_score_transfer, [single_transfer_row, 0], [query_group_size // 2, score_tile]
                            )
                            pl.system.sync_set(
                                SCORE_CONSUMED_EVENT, pipe=pl.PipeType.MTE2, ffts_mode=2, core_type=pl.KernelType.AIV
                            )
                            for single_local in pl.unroll(query_group_size // 2):
                                single_score_row = pl.tile.extract(
                                    single_scores, single_local, 0, [1, score_tile], target_memory=pl.MemorySpace.Vec
                                )
                                single_scaled = pl.mul(single_score_row, single_kv_scale)
                                # Full-width row assemble avoids a partial-column UB move.
                                single_segments = pl.tile.assemble(
                                    single_segments, single_scaled, [single_local * 2 + single_step % 2, 0]
                                )
                            if single_step % 2 == 1 or single_step + 1 == single_steps:
                                single_segment_begin = single_begin + (single_step // 2) * (2 * score_tile)
                                single_segment_span = (single_step % 2 + 1) * score_tile
                                single_segment_rows = pl.tile.reshape(
                                    single_segments, [query_group_size // 2, 2 * BUFFERED_LONG_SCORE_TILE]
                                )
                                for single_local in pl.unroll(query_group_size // 2):
                                    single_query_lane = buf_score_lane * (query_group_size // 2) + single_local
                                    single_position = pl.read(position_ids, [single_query + single_query_lane])
                                    single_query_visible = pl.max(
                                        pl.min(
                                            pl.min(single_cache_len, (single_position + 1) // COMPRESS_RATIO),
                                            TOPK_MAX_CANDIDATES,
                                        ),
                                        0,
                                    )
                                    single_segment_valid = pl.max(
                                        pl.min(single_query_visible - single_segment_begin, single_segment_span), 0
                                    )
                                    if single_segment_valid > 0:
                                        single_segment_data = pl.tile.extract(
                                            single_segment_rows,
                                            single_local,
                                            0,
                                            [1, 2 * BUFFERED_LONG_SCORE_TILE],
                                            target_memory=pl.MemorySpace.Vec,
                                        )
                                        if single_step >= 2 and single_segment_valid > 512:
                                            threshold_old_root = pl.tile.extract(
                                                single_roots,
                                                single_local,
                                                0,
                                                [1, TOPK_PAIR_WIDTH],
                                                target_memory=pl.MemorySpace.Vec,
                                            )
                                            threshold_next = threshold_merge_top512(
                                                single_segment_data,
                                                threshold_old_root,
                                                pl.cast(single_segment_begin, pl.INT32),
                                                pl.cast(single_segment_valid, pl.INT32),
                                            )
                                            single_roots = pl.tile.assemble(
                                                single_roots,
                                                threshold_next,
                                                [single_local, 0],
                                            )
                                        else:
                                            logical_i32 = pl.cast(single_segment_begin, pl.INT32)
                                            if single_segment_valid <= 512:
                                                tiny_indices = pl.add(
                                                    pl.tile.arange(0, [1, 512], dtype=pl.INT32),
                                                    logical_i32,
                                                )
                                                tiny_tile = pl.tile.extract(
                                                    single_segment_data,
                                                    0,
                                                    0,
                                                    [1, 512],
                                                    target_memory=pl.MemorySpace.Vec,
                                                )
                                                tiny_raw = pl.set_validshape(tiny_tile, 1, single_segment_valid)
                                                tiny_scores = pl.maximum(
                                                    pl.tile.fillpad(tiny_raw, pad_value=pl.PadValue.min),
                                                    FP32_NEG_INF,
                                                )
                                                tiny_pairs = pl.sort32(
                                                    tiny_scores,
                                                    pl.reinterpret_view(tiny_indices, pl.UINT32),
                                                )
                                                tiny_pairs = pl.mrgsort(tiny_pairs, block_len=64)
                                                tiny_pairs = pl.mrgsort(tiny_pairs, block_len=256)
                                                single_new_root = pl.tile.extract(
                                                    tiny_pairs,
                                                    0,
                                                    0,
                                                    [1, TOPK_PAIR_WIDTH],
                                                    target_memory=pl.MemorySpace.Vec,
                                                )
                                            elif single_segment_valid <= 1024:
                                                small_indices = pl.add(
                                                    pl.tile.arange(0, [1, 1024], dtype=pl.INT32),
                                                    logical_i32,
                                                )
                                                small_tile = pl.tile.extract(
                                                    single_segment_data,
                                                    0,
                                                    0,
                                                    [1, 1024],
                                                    target_memory=pl.MemorySpace.Vec,
                                                )
                                                small_raw = pl.set_validshape(small_tile, 1, single_segment_valid)
                                                small_scores = pl.maximum(
                                                    pl.tile.fillpad(small_raw, pad_value=pl.PadValue.min),
                                                    FP32_NEG_INF,
                                                )
                                                small_pairs = pl.sort32(
                                                    small_scores,
                                                    pl.reinterpret_view(small_indices, pl.UINT32),
                                                )
                                                small_pairs = pl.mrgsort(small_pairs, block_len=64)
                                                small_pairs = pl.mrgsort(small_pairs, block_len=256)
                                                small_left = pl.tile.slice(small_pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
                                                small_right = pl.tile.slice(
                                                    small_pairs,
                                                    [1, TOPK_PAIR_WIDTH],
                                                    [0, 1024],
                                                )
                                                small_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
                                                small_merged = pl.tile.mrgsort(small_left, small_right, tmp=small_tmp)
                                                single_new_root = pl.tile.extract(
                                                    small_merged,
                                                    0,
                                                    0,
                                                    [1, TOPK_PAIR_WIDTH],
                                                    target_memory=pl.MemorySpace.Vec,
                                                )
                                            else:
                                                large_indices = pl.add(
                                                    pl.tile.arange(0, [1, 2048], dtype=pl.INT32),
                                                    logical_i32,
                                                )
                                                large_tile = pl.tile.extract(
                                                    single_segment_data,
                                                    0,
                                                    0,
                                                    [1, 2048],
                                                    target_memory=pl.MemorySpace.Vec,
                                                )
                                                large_raw = pl.set_validshape(large_tile, 1, single_segment_valid)
                                                large_scores = pl.maximum(
                                                    pl.tile.fillpad(large_raw, pad_value=pl.PadValue.min),
                                                    FP32_NEG_INF,
                                                )
                                                large_pairs = pl.sort32(
                                                    large_scores,
                                                    pl.reinterpret_view(large_indices, pl.UINT32),
                                                )
                                                large_pairs = pl.mrgsort(large_pairs, block_len=64)
                                                large_pairs = pl.mrgsort(large_pairs, block_len=256)
                                                large_pairs = pl.mrgsort(large_pairs, block_len=1024)
                                                single_new_root = pl.tile.extract(
                                                    large_pairs,
                                                    0,
                                                    0,
                                                    [1, TOPK_PAIR_WIDTH],
                                                    target_memory=pl.MemorySpace.Vec,
                                                )
                                            if single_step < 2:
                                                single_roots = pl.tile.assemble(
                                                    single_roots, single_new_root, [single_local, 0]
                                                )
                                            else:
                                                single_old_root = pl.tile.extract(
                                                    single_roots,
                                                    single_local,
                                                    0,
                                                    [1, TOPK_PAIR_WIDTH],
                                                    target_memory=pl.MemorySpace.Vec,
                                                )
                                                single_merge_tmp = pl.tile.create(
                                                    [1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32
                                                )
                                                # Native merges each new contiguous 2048 segment
                                                # before the accumulated root when scores tie.
                                                single_merged = pl.tile.mrgsort(
                                                    single_new_root, single_old_root, tmp=single_merge_tmp
                                                )
                                                single_next_root = pl.tile.extract(
                                                    single_merged,
                                                    0,
                                                    0,
                                                    [1, TOPK_PAIR_WIDTH],
                                                    target_memory=pl.MemorySpace.Vec,
                                                )
                                                single_roots = pl.tile.assemble(
                                                    single_roots, single_next_root, [single_local, 0]
                                                )
                        for single_local in pl.unroll(query_group_size // 2):
                            single_query_lane = buf_score_lane * (query_group_size // 2) + single_local
                            single_root = pl.tile.extract(
                                single_roots, single_local, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec
                            )
                            pl.store(
                                single_root,
                                [(single_query + single_query_lane) * TOPK_ROWS_PER_QUERY + single_leaf, 0],
                                pair_arena,
                            )
            else:
                for buf_item in pl.range(
                    buf_worker, buf_query_count // query_group_size * buf_max_leaves, TOPK_SCORE_WORKERS
                ):
                    if buf_query_count < query_group_size * TOPK_SCORE_WORKERS:
                        buf_query = buf_item % (buf_query_count // query_group_size) * query_group_size
                        buf_leaf = buf_item // (buf_query_count // query_group_size)
                    else:
                        buf_query = buf_item // buf_max_leaves * query_group_size
                        buf_leaf = buf_item % buf_max_leaves
                    buf_batch_idx = buf_query // S
                    buf_last_position = pl.read(position_ids, [buf_query + query_group_size - 1])
                    buf_cache_len = pl.read(kv_seq_lens, [buf_batch_idx]) // COMPRESS_RATIO
                    buf_visible_count = pl.max(
                        pl.min(pl.min(buf_cache_len, (buf_last_position + 1) // COMPRESS_RATIO), TOPK_MAX_CANDIDATES), 0
                    )
                    buf_logical_begin = buf_leaf * TOPK_CANDIDATES_PER_LEAF
                    buf_leaf_capacity = TOPK_CANDIDATES_PER_LEAF
                    if balance_leaves:
                        buf_logical_begin = (
                            buf_leaf * buf_leaf_tiles + pl.min(buf_leaf, buf_extra_leaves)
                        ) * BUFFERED_LONG_SCORE_TILE
                        buf_leaf_capacity = (
                            buf_leaf_tiles + pl.cast(buf_leaf < buf_extra_leaves, pl.INDEX)
                        ) * BUFFERED_LONG_SCORE_TILE
                    if buf_logical_begin < buf_visible_count:
                        buf_valid_count = pl.min(buf_leaf_capacity, buf_visible_count - buf_logical_begin)
                        buf_tile_count = (buf_valid_count + score_tile - 1) // score_tile
                        buf_lane_span = pl.min(buf_tile_count * (score_tile // 2), TOPK_CANDIDATES_PER_LEAF // 2)
                        buf_score_iters = (buf_lane_span + (score_tile // 2) - 1) // (score_tile // 2)
                        buf_lane_begin = buf_score_lane * buf_lane_span
                        # QLI V2 retains the accumulated root in UB until publication.
                        prefix_roots = pl.tile.full(
                            [query_group_size, TOPK_PAIR_WIDTH], dtype=pl.FP32, value=FP32_NEG_INF
                        )
                        for buf_score_step in pl.range(buf_score_iters):
                            buf_score_begin = buf_score_step * (score_tile // 2)
                            buf_transfer_row = buf_worker * 2 * query_group_size + buf_score_step % 2 * query_group_size
                            # Native页内scale只依赖cache_write，不依赖Cube结果。
                            # 在等待Score前加载scale，让独立分页读取与Cube执行交叠。
                            buf_scale_bytes = pl.tile.create([1, score_tile], dtype=pl.INT8)
                            for buf_scale_page in pl.range((score_tile // 2) // BLOCK_SIZE):
                                buf_scale_row = (
                                    buf_logical_begin + buf_score_begin + buf_lane_begin + buf_scale_page * BLOCK_SIZE
                                )
                                buf_scale_safe_page = pl.min(
                                    buf_scale_row // BLOCK_SIZE, pl.max((buf_cache_len - 1) // BLOCK_SIZE, 0)
                                )
                                buf_scale_physical_page = pl.max(
                                    pl.cast(pl.read(idx_block_table, [buf_batch_idx, buf_scale_safe_page]), pl.INDEX), 0
                                )
                                buf_scale_bytes = pl.gather_row(
                                    buf_scale_bytes,
                                    idx_native_kv_cache,
                                    [0, buf_scale_page * BLOCK_SIZE * 2],
                                    [buf_scale_physical_page, INDEXER_KEY_BYTES],
                                    [1, BLOCK_SIZE * 2],
                                )
                            buf_scale_half = pl.tile.reinterpret_view(buf_scale_bytes, pl.FP16)
                            buf_kv_scale = pl.cast(buf_scale_half, target_type=pl.FP32)
                            pl.system.sync_wait(SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                            buf_score_sums = pl.load(
                                buf_score_transfer,
                                [buf_transfer_row, buf_score_lane * (score_tile // 2)],
                                [query_group_size, (score_tile // 2)],
                            )
                            pl.system.sync_set(
                                SCORE_CONSUMED_EVENT, pipe=pl.PipeType.MTE2, ffts_mode=2, core_type=pl.KernelType.AIV
                            )
                            for buf_query_lane in pl.unroll(query_group_size):
                                buf_position = pl.read(position_ids, [buf_query + buf_query_lane])
                                buf_query_visible = pl.max(
                                    pl.min(
                                        pl.min(buf_cache_len, (buf_position + 1) // COMPRESS_RATIO), TOPK_MAX_CANDIDATES
                                    ),
                                    0,
                                )
                                buf_lane_valid_rows = pl.max(
                                    pl.min(
                                        buf_query_visible - buf_logical_begin - buf_score_begin - buf_lane_begin,
                                        (score_tile // 2),
                                    ),
                                    0,
                                )
                                buf_score_sum = pl.tile.slice(buf_score_sums, [1, score_tile // 2], [buf_query_lane, 0])
                                buf_score_row = pl.mul(buf_score_sum, buf_kv_scale)
                                if buf_lane_valid_rows > 0:
                                    pl.store(
                                        pl.set_validshape(buf_score_row, 1, buf_lane_valid_rows),
                                        [
                                            buf_worker * 2 * query_group_size + buf_query_lane * 2 + buf_score_lane,
                                            buf_score_begin,
                                        ],
                                        score_arena,
                                    )
                            if balance_leaves:
                                if buf_score_step == 3 and buf_score_iters > 4:
                                    for stream_lane in pl.unroll(query_group_size):
                                        stream_position = pl.read(position_ids, [buf_query + stream_lane])
                                        stream_visible = pl.max(
                                            pl.min(
                                                pl.min(buf_cache_len, (stream_position + 1) // COMPRESS_RATIO),
                                                TOPK_MAX_CANDIDATES,
                                            ),
                                            0,
                                        )
                                        stream_begin = buf_logical_begin + buf_lane_begin
                                        stream_valid = pl.max(pl.min(stream_visible - stream_begin, 2048), 0)
                                        if stream_valid > 0:
                                            stream_root = indexer_topk_segment_pairs(
                                                score_arena,
                                                buf_worker * 2 * query_group_size + stream_lane * 2 + buf_score_lane,
                                                stream_begin,
                                                stream_valid,
                                                0,
                                            )
                                            prefix_roots = pl.tile.assemble(prefix_roots, stream_root, [stream_lane, 0])
                        for buf_query_lane in pl.unroll(query_group_size):
                            buf_position = pl.read(position_ids, [buf_query + buf_query_lane])
                            buf_query_visible = pl.max(
                                pl.min(
                                    pl.min(buf_cache_len, (buf_position + 1) // COMPRESS_RATIO), TOPK_MAX_CANDIDATES
                                ),
                                0,
                            )
                            buf_half_begin = buf_logical_begin + buf_lane_begin
                            buf_half_valid = pl.max(pl.min(buf_query_visible - buf_half_begin, buf_lane_span), 0)
                            buf_half_slot = (
                                (buf_query + buf_query_lane) * TOPK_ROWS_PER_QUERY + buf_leaf * 2 + buf_score_lane
                            )
                            if balance_leaves and buf_score_iters > 4:
                                if buf_half_valid > 2048:
                                    tail = indexer_topk_segment_pairs(
                                        score_arena,
                                        buf_worker * 2 * query_group_size + buf_query_lane * 2 + buf_score_lane,
                                        buf_half_begin + 2048,
                                        buf_half_valid - 2048,
                                        2048,
                                    )
                                    prefix = pl.tile.extract(
                                        prefix_roots,
                                        buf_query_lane,
                                        0,
                                        [1, TOPK_PAIR_WIDTH],
                                        target_memory=pl.MemorySpace.Vec,
                                    )
                                    merge_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
                                    # Preserve the later-2048-chunk priority of the
                                    # existing 2560/3072/4096 half-leaf paths.
                                    merged = pl.tile.mrgsort(tail, prefix, tmp=merge_tmp)
                                    pl.store(
                                        pl.tile.slice(merged, [1, TOPK_PAIR_WIDTH], [0, 0]),
                                        [buf_half_slot, 0],
                                        pair_arena,
                                    )
                                else:
                                    prefix = pl.tile.extract(
                                        prefix_roots,
                                        buf_query_lane,
                                        0,
                                        [1, TOPK_PAIR_WIDTH],
                                        target_memory=pl.MemorySpace.Vec,
                                    )
                                    pl.store(prefix, [buf_half_slot, 0], pair_arena)
                            else:
                                if buf_half_valid > 0:
                                    indexer_topk_half_leaf(
                                        score_arena,
                                        pair_arena,
                                        buf_worker * 2 * query_group_size + buf_query_lane * 2 + buf_score_lane,
                                        buf_half_begin,
                                        buf_half_valid,
                                        buf_half_slot,
                                    )
                                else:
                                    pl.store(
                                        pl.tile.full([1, TOPK_PAIR_WIDTH], dtype=pl.FP32, value=FP32_NEG_INF),
                                        [buf_half_slot, 0],
                                        pair_arena,
                                    )
    return buffered_leaf_tid


@pl.jit.inline(auto_scope=False)
def indexer_score_topk_forest(
    qr_hadamard_i8: pl.Tensor[[T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8],
    qr_hadamard_scale_dq: pl.Tensor[[T_PAD * IDX_N_HEADS, 1], pl.FP32],
    weights: pl.Tensor[[T_PAD, IDX_N_HEADS], pl.FP32],
    idx_native_kv_cache: pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8],
    idx_block_table: pl.Tensor[[B_DYN, INDEXER_TABLE_COLUMNS_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    topk_scores: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32]],
    topk_idxs: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32]],
    qh_quant_tid: pl.Scalar[pl.TASK_ID],
    weights_tid: pl.Scalar[pl.TASK_ID],
    cache_write_tid: pl.Scalar[pl.TASK_ID],
    host_max_seq_len: pl.Scalar[pl.INT32],
    scale_layout: pl.Tensor[[SCALE_LAYOUT_CAPACITY], pl.INT32],
    layout_tid: pl.Scalar[pl.TASK_ID],
):
    """Score leaves and merge Top-K roots; long S6 publishes one root per leaf."""
    native_page_bytes = pl.tensor.dim(idx_native_kv_cache, 1)
    # Zero-copy GM descriptors inside orchestration, as validation log §116.
    # One writable root allocation avoids partial-overlap Torch ABI arguments.
    cache_bytes = pl.tensor.dim(idx_native_kv_cache, 0) * native_page_bytes
    cache_flat = pl.reshape(idx_native_kv_cache, [cache_bytes])
    key_rows = cache_bytes // IDX_HEAD_DIM
    shifted_rows = (cache_bytes - 64) // IDX_HEAD_DIM
    idx_kv_cache = pl.reshape(cache_flat[0 : key_rows * IDX_HEAD_DIM], [key_rows, IDX_HEAD_DIM])
    idx_kv_cache_shift64 = pl.reshape(cache_flat[64 : 64 + shifted_rows * IDX_HEAD_DIM], [shifted_rows, IDX_HEAD_DIM])
    pair_arena = pl.create_tensor([TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], dtype=pl.FP32)
    # The whole batch uses query rows for one leaf, or private lane rows for multiple leaves.
    score_arena = pl.create_tensor([SCORE_ARENA_ROWS, TOPK_CANDIDATES_PER_LEAF], dtype=pl.FP32)
    # 8K及长上下文统一尝试片上FP16/Cube规约；更短历史保留Vector路径。
    max_topk_cache_len = host_max_seq_len // COMPRESS_RATIO
    if max_topk_cache_len >= INDEXER_NATIVE_CUBE_MIN_ROWS:
        # 长历史的完整leaf每半区4096候选：512一轮，11轮降至8轮。
        # 短历史保留384半区，避免扩大最后一轮的padding计算。
        if max_topk_cache_len > TOPK_CANDIDATES_PER_LEAF:
            # 长档整组准入，并禁止Score提前释放下游，抑制重复图重放的长尾。
            # 短档沿用原调度；sync_start不代表各核硬件同时起跑。
            # 完整S6共用Key；N64令M384的QK累加仍为96 KiB。
            # B4/B8通过leaf分配补齐并行度，复用整请求Key；B<4仍用双query。
            long_s6_group = indexer_use_long_s6(pl.tensor.dim(position_ids, 0))
            if long_s6_group == 1:
                if pl.tensor.dim(position_ids, 0) >= SCORE_SPLIT_MIN_QUERY_ROWS:
                    score_tid = indexer_score_topk_native_cube(
                        qr_hadamard_i8,
                        qr_hadamard_scale_dq,
                        weights,
                        idx_native_kv_cache,
                        idx_block_table,
                        position_ids,
                        kv_seq_lens,
                        score_arena,
                        pair_arena,
                        qh_quant_tid,
                        weights_tid,
                        cache_write_tid,
                        S,
                        64,
                        BUFFERED_LONG_SCORE_TILE,
                        True,
                        False,
                        1,
                        True,
                        True,
                        True,
                        scale_layout,
                        layout_tid,
                    )
                else:
                    score_tid = indexer_score_topk_native_cube(
                        qr_hadamard_i8,
                        qr_hadamard_scale_dq,
                        weights,
                        idx_native_kv_cache,
                        idx_block_table,
                        position_ids,
                        kv_seq_lens,
                        score_arena,
                        pair_arena,
                        qh_quant_tid,
                        weights_tid,
                        cache_write_tid,
                        S,
                        64,
                        BUFFERED_LONG_SCORE_TILE,
                        True,
                        False,
                        1,
                        True,
                        True,
                        False,
                        scale_layout,
                        layout_tid,
                    )
            else:
                score_tid = indexer_score_topk_native_cube(
                    qr_hadamard_i8,
                    qr_hadamard_scale_dq,
                    weights,
                    idx_native_kv_cache,
                    idx_block_table,
                    position_ids,
                    kv_seq_lens,
                    score_arena,
                    pair_arena,
                    qh_quant_tid,
                    weights_tid,
                    cache_write_tid,
                    2,
                    NATIVE_QLI_QK_COLS,
                    BUFFERED_LONG_SCORE_TILE,
                    True,
                    False,
                    # M128/N128 fits two 16KiB Key slots plus 32KiB WS in L0B.
                    1,
                    True,
                    False,
                    False,
                    scale_layout,
                    layout_tid,
                )
        else:
            short_queries = pl.tensor.dim(position_ids, 0)
            short_pair_waves = (short_queries // 2 + TOPK_SCORE_WORKERS - 1) // TOPK_SCORE_WORKERS
            short_request_waves = (short_queries // S + TOPK_SCORE_WORKERS - 1) // TOPK_SCORE_WORKERS
            # 先保证整请求分组覆盖所有worker；最忙核query工作量增幅上限25%。
            # S6的B24/B40满足，B16/B32保留双query；不按脚本切换源码。
            short_use_request = pl.cast(0, pl.INT32)
            if short_queries >= S * TOPK_SCORE_WORKERS:
                if 4 * S * short_request_waves <= 5 * 2 * short_pair_waves:
                    short_use_request = pl.cast(1, pl.INT32)
            if short_use_request == 1:
                score_tid = indexer_score_topk_native_cube(
                    qr_hadamard_i8,
                    qr_hadamard_scale_dq,
                    weights,
                    idx_native_kv_cache,
                    idx_block_table,
                    position_ids,
                    kv_seq_lens,
                    score_arena,
                    pair_arena,
                    qh_quant_tid,
                    weights_tid,
                    cache_write_tid,
                    S,
                    64,
                    BUFFERED_SCORE_TILE,
                    False,
                    True,
                    0,
                    True,
                    False,
                    False,
                    scale_layout,
                    layout_tid,
                )
            else:
                # ef608731：半叶排序不超过2048项时预取一个Key面板。
                short_steps_upper = (max_topk_cache_len + BUFFERED_SCORE_TILE - 1) // BUFFERED_SCORE_TILE
                short_half_span_upper = short_steps_upper * (BUFFERED_SCORE_TILE // 2)
                if short_half_span_upper <= SHORT_PREFETCH_HALF_SORT_LIMIT:
                    score_tid = indexer_score_topk_native_cube(
                        qr_hadamard_i8,
                        qr_hadamard_scale_dq,
                        weights,
                        idx_native_kv_cache,
                        idx_block_table,
                        position_ids,
                        kv_seq_lens,
                        score_arena,
                        pair_arena,
                        qh_quant_tid,
                        weights_tid,
                        cache_write_tid,
                        2,
                        NATIVE_QLI_QK_COLS,
                        BUFFERED_SCORE_TILE,
                        False,
                        True,
                        1,
                        True,
                        False,
                        False,
                        scale_layout,
                        layout_tid,
                    )
                else:
                    score_tid = indexer_score_topk_native_cube(
                        qr_hadamard_i8,
                        qr_hadamard_scale_dq,
                        weights,
                        idx_native_kv_cache,
                        idx_block_table,
                        position_ids,
                        kv_seq_lens,
                        score_arena,
                        pair_arena,
                        qh_quant_tid,
                        weights_tid,
                        cache_write_tid,
                        2,
                        NATIVE_QLI_QK_COLS,
                        BUFFERED_SCORE_TILE,
                        False,
                        True,
                        0,
                        True,
                        False,
                        False,
                        scale_layout,
                        layout_tid,
                    )
    else:
        with pl.spmd(
            TOPK_SCORE_WORKERS,
            name_hint="indexer_score_topk_leaf",
            deps=[qh_quant_tid, weights_tid, cache_write_tid],
            allow_early_resolve=True,
            optimizations=[pl.cross_core_slot(slot_num=1)],
        ) as direct_leaf_tid:
            worker = pl.tile.get_block_idx()
            query_count = pl.tensor.dim(position_ids, 0)
            max_cache_len = 0
            for batch in pl.range(query_count // S):
                batch_cache_len = pl.read(kv_seq_lens, [batch]) // COMPRESS_RATIO
                max_cache_len = pl.max(max_cache_len, batch_cache_len)
            max_leaves = pl.max(
                (pl.min(max_cache_len, TOPK_MAX_CANDIDATES) + TOPK_CANDIDATES_PER_LEAF - 1) // TOPK_CANDIDATES_PER_LEAF,
                1,
            )
            single_leaf = pl.cast(max_leaves == 1, pl.INDEX)
            for item in pl.range(worker, query_count * max_leaves, TOPK_SCORE_WORKERS):
                query = item // max_leaves
                leaf = item % max_leaves
                batch_idx = query // S
                position = pl.read(position_ids, [query])
                cache_len = pl.read(kv_seq_lens, [batch_idx]) // COMPRESS_RATIO
                cache_bound = pl.min(cache_len, (position + 1) // COMPRESS_RATIO)
                visible_count = pl.max(pl.min(cache_bound, TOPK_MAX_CANDIDATES), 0)
                logical_begin = leaf * TOPK_CANDIDATES_PER_LEAF
                if logical_begin < visible_count:
                    valid_count = pl.min(TOPK_CANDIDATES_PER_LEAF, visible_count - logical_begin)
                    # Each half-leaf must fit the 4096-candidate sort path.
                    lane_span = pl.min(
                        ((valid_count + SCORE_TILE - 1) // SCORE_TILE) * SCORE_LANE_ROWS,
                        TOPK_CANDIDATES_PER_LEAF // 2,
                    )
                    lane_stride = single_leaf * SCORE_LANE_ROWS + (1 - single_leaf) * lane_span
                    query_head_begin = query * IDX_N_HEADS
                    query_vector = qr_hadamard_i8[query_head_begin : query_head_begin + IDX_N_HEADS, 0:IDX_HEAD_DIM]
                    # 两个 Vector lane 共用这一 query 的 head 系数。
                    for _aiv_coeff in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                        query_scale = pl.reshape(
                            qr_hadamard_scale_dq[query_head_begin : query_head_begin + IDX_N_HEADS, 0:1],
                            [1, IDX_N_HEADS],
                        )
                        query_weight = weights[query : query + 1, 0:IDX_N_HEADS]
                        head_coefficient = pl.reshape(pl.mul(query_scale, query_weight), [IDX_N_HEADS, 1])
                    for score_begin in pl.pipeline(0, lane_span, SCORE_LANE_ROWS, stage=2):
                        read_begin = score_begin * (1 + single_leaf)
                        kv_i8 = pl.create_l1([SCORE_TILE, IDX_HEAD_DIM], dtype=pl.INT8)
                        for key_lane in pl.unroll(2):
                            for key_page in pl.range(SCORE_LANE_ROWS // BLOCK_SIZE):
                                key_row = logical_begin + read_begin + key_lane * lane_stride + key_page * BLOCK_SIZE
                                safe_page = pl.min(key_row // BLOCK_SIZE, pl.max((cache_len - 1) // BLOCK_SIZE, 0))
                                physical_page = pl.max(
                                    pl.cast(pl.read(idx_block_table, [batch_idx, safe_page]), pl.INDEX), 0
                                )
                                native_byte = physical_page * native_page_bytes
                                if native_byte % IDX_HEAD_DIM == 0:
                                    kv_i8 = pl.gather_row(
                                        kv_i8,
                                        idx_kv_cache,
                                        [key_lane * SCORE_LANE_ROWS + key_page * BLOCK_SIZE, 0],
                                        [native_byte // IDX_HEAD_DIM, 0],
                                        [BLOCK_SIZE, IDX_HEAD_DIM],
                                    )
                                else:
                                    kv_i8 = pl.gather_row(
                                        kv_i8,
                                        idx_kv_cache_shift64,
                                        [key_lane * SCORE_LANE_ROWS + key_page * BLOCK_SIZE, 0],
                                        [native_byte // IDX_HEAD_DIM, 0],
                                        [BLOCK_SIZE, IDX_HEAD_DIM],
                                    )
                        # 性能版改用 Vector 的 col_sum 规约 head，与上游一致：省掉
                        # 每个 score tile 一次 FP32->FP16 转换和一次 Cube matmul。
                        # 已封存的精度版那条链（NATIVE_QLI_QK_SCALE + FP16 rint + Cube）是为了
                        # 复刻 Native 的 QK tile 与规约精度，本版本刻意放弃该性质。
                        score_i32 = pl.matmul(query_vector, kv_i8, out_dtype=pl.INT32, b_trans=True)
                        # Each lane owns a contiguous candidate-column range.
                        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
                            lane_begin = aiv_id * lane_stride
                            lane_valid_rows = pl.max(pl.min(valid_count - read_begin - lane_begin, SCORE_LANE_ROWS), 0)
                            scale_begin = logical_begin + read_begin + lane_begin
                            scale_bytes = pl.create_tensor([1, SCORE_LANE_ROWS * 2], dtype=pl.INT8)
                            for scale_page in pl.range(SCORE_LANE_ROWS // BLOCK_SIZE):
                                scale_logical_page = pl.min(
                                    (scale_begin + scale_page * BLOCK_SIZE) // BLOCK_SIZE,
                                    pl.max((cache_len - 1) // BLOCK_SIZE, 0),
                                )
                                scale_physical_page = pl.max(
                                    pl.cast(pl.read(idx_block_table, [batch_idx, scale_logical_page]), pl.INDEX), 0
                                )
                                scale_bytes = pl.gather_row(
                                    scale_bytes,
                                    idx_native_kv_cache,
                                    [0, scale_page * BLOCK_SIZE * 2],
                                    [scale_physical_page, INDEXER_KEY_BYTES],
                                    [1, BLOCK_SIZE * 2],
                                )
                            kv_scale = pl.reinterpret_view(scale_bytes, pl.FP16)
                            score_shard = pl.aiv_shard(score_i32)
                            score_fp32 = pl.cast(score_shard, target_type=pl.FP32, mode="none")
                            score_fp32 = pl.maximum(score_fp32, 0.0)
                            score_fp32 = pl.row_expand_mul(score_fp32, head_coefficient)
                            score_row = pl.reshape(pl.col_sum(score_fp32), [1, SCORE_LANE_ROWS])
                            score_row = pl.mul(score_row, pl.cast(kv_scale, pl.FP32))
                            score_row_id = single_leaf * query + (1 - single_leaf) * (worker * 2 + aiv_id)
                            score_col = single_leaf * (read_begin + lane_begin) + (1 - single_leaf) * score_begin
                            # Store only valid scores; the Top-K load pads its own tail.
                            if lane_valid_rows > 0:
                                score_valid = pl.set_validshape(score_row, 1, lane_valid_rows)
                                score_arena[
                                    score_row_id : score_row_id + 1, score_col : score_col + SCORE_LANE_ROWS
                                ] = score_valid

                    if single_leaf == 0:
                        for sort_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                            half_begin = logical_begin + sort_lane * lane_span
                            half_valid = pl.max(pl.min(valid_count - sort_lane * lane_span, lane_span), 0)
                            half_slot = query * TOPK_ROWS_PER_QUERY + leaf * 2 + sort_lane
                            if half_valid > 0:
                                indexer_topk_half_leaf(
                                    score_arena, pair_arena, worker * 2 + sort_lane, half_begin, half_valid, half_slot
                                )
                            else:
                                empty_pairs = pl.tile.full([1, TOPK_PAIR_WIDTH], dtype=pl.FP32, value=FP32_NEG_INF)
                                pl.store(empty_pairs, [half_slot, 0], pair_arena)

        score_tid = direct_leaf_tid

    with pl.scope():
        if max_topk_cache_len < INDEXER_NATIVE_CUBE_MIN_ROWS:
            with pl.spmd(
                TOPK_QUERY_WORKERS,
                name_hint="indexer_topk_single_leaf_publish",
                deps=[score_tid],
                allow_early_resolve=True,
            ):
                indexer_topk_single_leaf_publish(position_ids, kv_seq_lens, score_arena, topk_scores, topk_idxs)
        elif max_topk_cache_len <= TOPK_CANDIDATES_PER_LEAF:
            # Keep the single-leaf kernel free of the multiway branches and
            # their larger UB temporaries. The choice follows actual length.
            with pl.spmd(
                TOPK_QUERY_WORKERS, name_hint="indexer_topk_query_merge", deps=[score_tid], allow_early_resolve=True
            ):
                indexer_topk_query_merge(position_ids, kv_seq_lens, pair_arena, topk_scores, topk_idxs, False)
        else:
            with pl.spmd(
                TOPK_QUERY_WORKERS, name_hint="indexer_topk_query_merge", deps=[score_tid], allow_early_resolve=True
            ):
                indexer_topk_query_merge(position_ids, kv_seq_lens, pair_arena, topk_scores, topk_idxs, True)

    return topk_scores, topk_idxs, score_tid


@pl.jit.inline(auto_scope=False)
def indexer_qr_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    qr_bf16: pl.Out[pl.Tensor[[T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], pl.BF16]],
) -> pl.Scalar[pl.TASK_ID]:
    """Indexer query projection, dequant and RoPE -- everything before the hadamard."""

    bs = pl.tensor.dim(x, 0)
    row_blocks = (bs + QR_MM_ROW_TILE - 1) // QR_MM_ROW_TILE
    # QR_MM_T_PAD >= T_PAD，缓冲只会变大不会变小；读者只读到 bs 为止。
    qr_acc_pad = pl.create_tensor([QR_MM_T_PAD, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.INT32)
    with pl.spmd(
        QR_PROJ_WORKERS,
        name_hint="idx_qr_proj_matmul",
        allow_early_resolve=True,
    ) as idx_qr_mm_tid:
        qr_proj_worker = pl.tile.get_block_idx()
        for qr_col in pl.range(qr_proj_worker, IDX_N_HEADS * IDX_HEAD_DIM // QR_MM_N_TILE, QR_PROJ_WORKERS):
            o_base = qr_col * QR_MM_N_TILE
            # 整条 K 的权重块只取一次，供下面所有行块复用。
            wq_tile = wq_b[0:Q_LORA, o_base : o_base + QR_MM_N_TILE]
            for qr_rb in pl.range(row_blocks):
                qr_r0 = qr_rb * QR_MM_ROW_TILE
                qr_rows = pl.min(QR_MM_ROW_TILE, bs - qr_r0)
                qr_tile = pl.slice(qr, [QR_MM_ROW_TILE, Q_LORA], [qr_r0, 0], valid_shape=[qr_rows, Q_LORA])
                qr_acc = pl.matmul(qr_tile, wq_tile, out_dtype=pl.INT32)
                qr_acc_pad[qr_r0 : qr_r0 + QR_MM_ROW_TILE, o_base : o_base + QR_MM_N_TILE] = qr_acc
    # Fused dequant + RoPE: one unit is DEQUANT_T_TILE tokens x DQ_ROPE_H_TILE heads.
    qr_bf16_2d = pl.reshape(qr_bf16, [T_PAD, IDX_N_HEADS * IDX_HEAD_DIM])
    dq_rope_units = ((bs + DEQUANT_T_TILE - 1) // DEQUANT_T_TILE) * (IDX_N_HEADS // DQ_ROPE_H_TILE)
    dq_rope_workers = pl.min(dq_rope_units, DQ_ROPE_WORKERS)
    for dq_rope_worker in pl.spmd(dq_rope_workers, name_hint="idx_qr_dequant_rope", allow_early_resolve=True):
        sw_ones = pl.full([DEQUANT_T_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        sw_index = pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32)
        sw_col = pl.col_expand_mul(sw_ones, sw_index)
        sw_dup_f = pl.cast(pl.cast(pl.mul(sw_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        sw_lane = pl.sub(sw_col, pl.mul(sw_dup_f, 2.0))
        rope_swap_idx = pl.cast(pl.sub(pl.add(sw_col, 1.0), pl.mul(sw_lane, 2.0)), target_type=pl.INT32)
        for dq_unit in pl.range(dq_rope_worker, dq_rope_units, dq_rope_workers):
            hg = (dq_unit % (IDX_N_HEADS // DQ_ROPE_H_TILE)) * DQ_ROPE_H_TILE
            dq_t0 = (dq_unit // (IDX_N_HEADS // DQ_ROPE_H_TILE)) * DEQUANT_T_TILE
            if dq_t0 + DEQUANT_T_TILE <= bs:
                qr_scale_tile = pl.load(qr_scale, [dq_t0, 0], [DEQUANT_T_TILE, 1])
                cos_tile = pl.load(cos, [dq_t0, 0], [DEQUANT_T_TILE, ROPE_HEAD_DIM])
                sin_tile = pl.load(sin, [dq_t0, 0], [DEQUANT_T_TILE, ROPE_HEAD_DIM])
                # The RoPE slice has a 128-element row stride. Gather from the
                # contiguous full head using absolute element indices.
                flat_i = pl.tile.ci(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32)
                flat_tmp = pl.create_tile([1, ROPE_HEAD_DIM], dtype=pl.INT32)
                flat_lane = pl.tile.rems(flat_i, 2, flat_tmp)
                swap_row = pl.tile.adds(pl.tile.sub(flat_i, pl.tile.muls(flat_lane, 2)), IDX_NOPE_HEAD_DIM + 1)
                swap_base = pl.create_tile([DEQUANT_T_TILE, ROPE_HEAD_DIM], dtype=pl.INT32)
                swap_source = pl.col_expand(swap_base, swap_row)
                row_offsets = pl.tile.muls(pl.tile.ci(0, [1, DEQUANT_T_TILE], dtype=pl.INT32), IDX_HEAD_DIM)
                flat_swap = pl.reshape(
                    pl.row_expand_add(swap_source, pl.reshape(row_offsets, [DEQUANT_T_TILE, 1])),
                    [1, DEQUANT_T_TILE * ROPE_HEAD_DIM],
                )
                gather_tmp = pl.create_tile([1, DEQUANT_T_TILE * ROPE_HEAD_DIM], dtype=pl.INT32)
                for h_inner in pl.pipeline(DQ_ROPE_H_TILE, stage=2):
                    h0 = (hg + h_inner) * IDX_HEAD_DIM
                    wq_scale = pl.reshape(pl.load(wq_b_scale, [h0], [IDX_HEAD_DIM]), [1, IDX_HEAD_DIM])
                    acc_fp32 = pl.cast(
                        pl.load(qr_acc_pad, [dq_t0, h0], [DEQUANT_T_TILE, IDX_HEAD_DIM]),
                        target_type=pl.FP32,
                        mode="none",
                    )
                    qr_dequant = pl.col_expand_mul(pl.row_expand_mul(acc_fp32, qr_scale_tile), wq_scale)
                    qr_nope_bf16 = pl.cast(qr_dequant[:, 0:IDX_NOPE_HEAD_DIM], target_type=pl.BF16, mode="rint")
                    qr_rope_slice = qr_dequant[:, IDX_NOPE_HEAD_DIM:IDX_HEAD_DIM]
                    qr_swapped_flat = pl.tile.gather(
                        pl.reshape(qr_dequant, [1, DEQUANT_T_TILE * IDX_HEAD_DIM]),
                        flat_swap, gather_tmp,
                    )
                    qr_swapped = pl.reshape(qr_swapped_flat, [DEQUANT_T_TILE, ROPE_HEAD_DIM])
                    rope_rot = pl.add(pl.mul(qr_rope_slice, cos_tile), pl.mul(qr_swapped, sin_tile))
                    rope_bf16 = pl.cast(rope_rot, target_type=pl.BF16, mode="rint")
                    pl.store(qr_nope_bf16, [dq_t0, h0], qr_bf16_2d)
                    pl.store(rope_bf16, [dq_t0, h0 + IDX_NOPE_HEAD_DIM], qr_bf16_2d)
            else:
                # At most seven rows. Keep all broadcast operands at the same
                # extent; a partial scale tile cannot broadcast into eight rows.
                tail_swap_idx = rope_swap_idx[0:1, :]
                for tail_t0 in pl.range(dq_t0, bs):
                    tail_qr_scale_value = pl.read(qr_scale, [tail_t0, 0])
                    tail_cos_tile = cos[tail_t0 : tail_t0 + 1, 0:ROPE_HEAD_DIM]
                    tail_sin_tile = sin[tail_t0 : tail_t0 + 1, 0:ROPE_HEAD_DIM]
                    for tail_h_inner in pl.pipeline(DQ_ROPE_H_TILE, stage=2):
                        tail_h0 = (hg + tail_h_inner) * IDX_HEAD_DIM
                        tail_wq_scale = pl.reshape(wq_b_scale[tail_h0 : tail_h0 + IDX_HEAD_DIM], [1, IDX_HEAD_DIM])
                        tail_acc_fp32 = pl.cast(
                            qr_acc_pad[tail_t0 : tail_t0 + 1, tail_h0 : tail_h0 + IDX_HEAD_DIM],
                            target_type=pl.FP32,
                            mode="none",
                        )
                        tail_qr_dequant = pl.col_expand_mul(pl.mul(tail_acc_fp32, tail_qr_scale_value), tail_wq_scale)
                        tail_qr_nope_bf16 = pl.cast(
                            tail_qr_dequant[:, 0:IDX_NOPE_HEAD_DIM], target_type=pl.BF16, mode="rint"
                        )
                        tail_qr_rope_slice = tail_qr_dequant[:, IDX_NOPE_HEAD_DIM:IDX_HEAD_DIM]
                        tail_qr_swapped = pl.gather(tail_qr_rope_slice, dim=-1, index=tail_swap_idx)
                        tail_rope_rot = pl.add(
                            pl.mul(tail_qr_rope_slice, tail_cos_tile), pl.mul(tail_qr_swapped, tail_sin_tile)
                        )
                        tail_rope_bf16 = pl.cast(tail_rope_rot, target_type=pl.BF16, mode="rint")
                        qr_bf16_2d[tail_t0 : tail_t0 + 1, tail_h0 : tail_h0 + IDX_NOPE_HEAD_DIM] = tail_qr_nope_bf16
                        qr_bf16_2d[tail_t0 : tail_t0 + 1, tail_h0 + IDX_NOPE_HEAD_DIM : tail_h0 + IDX_HEAD_DIM] = (
                            tail_rope_bf16
                        )

    return idx_qr_mm_tid


@pl.jit.inline(auto_scope=False)
def indexer_qr_hadamard_mm(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr_bf16: pl.Tensor[[T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], pl.BF16],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    qr_hadamard_i8: pl.Out[pl.Tensor[[T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8]],
    qr_hadamard_scale_dq: pl.Out[pl.Tensor[[T_PAD * IDX_N_HEADS, 1], pl.FP32]],
    qh_mm_dep: pl.Scalar[pl.TASK_ID],
) -> tuple[pl.Scalar[pl.TASK_ID], pl.Scalar[pl.TASK_ID]]:
    """q @ hadamard and its INT8 quant, fenced behind the caller's cube ordering."""
    bs = pl.tensor.dim(x, 0)
    bs_heads = bs * IDX_N_HEADS

    # Query Hadamard FP32 intermediate.
    qh_acc_gm = pl.create_tensor([bs_heads, IDX_HEAD_DIM], dtype=pl.FP32)
    with pl.spmd(
        QH_WORKERS,
        name_hint="qr_hadamard_matmul",
        deps=[qh_mm_dep],
        allow_early_resolve=True,
    ) as qh_mm_tid:
        qh_worker = pl.tile.get_block_idx()
        # Shared Hadamard matrix.
        qh_hadamard = hadamard[0:IDX_HEAD_DIM, 0:IDX_HEAD_DIM]
        for idx in pl.range(qh_worker, bs_heads // QH_MM_TILE, QH_WORKERS):
            o0 = idx * QH_MM_TILE
            qh_acc = pl.matmul(qr_bf16[o0 : o0 + QH_MM_TILE, :], qh_hadamard, out_dtype=pl.FP32)
            qh_acc_gm[o0 : o0 + QH_MM_TILE, :] = qh_acc

    with pl.spmd(
        QH_QUANT_WORKERS,
        name_hint="qr_hadamard_quant",
        allow_early_resolve=True,
    ) as qh_quant_tid:
        qh_quant_worker = pl.tile.get_block_idx()
        for idx in pl.range(qh_quant_worker, bs_heads // QH_QUANT_TILE, QH_QUANT_WORKERS):
            o0 = idx * QH_QUANT_TILE
            qh_full_f32 = qh_acc_gm[o0 : o0 + QH_QUANT_TILE, 0:IDX_HEAD_DIM]
            qh_full_f32 = pl.mul(qh_full_f32, HADAMARD_SCALE)
            qh_amax = pl.full([1, QH_QUANT_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for h0 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
                qh_a_f32 = qh_full_f32[:, h0 : h0 + QH_HEAD_DIM_TILE]
                qh_a_neg = pl.neg(qh_a_f32)
                qh_a_abs = pl.maximum(qh_a_f32, qh_a_neg)
                qh_a_max_col = pl.row_max(qh_a_abs)
                qh_a_max = pl.reshape(qh_a_max_col, [1, QH_QUANT_TILE])
                qh_amax = pl.maximum(qh_amax, qh_a_max)
            qh_scale_numerator = pl.full([1, QH_QUANT_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
            qh_scale_quant_row = pl.div(qh_scale_numerator, qh_amax)
            qh_scale_recip = pl.recip(qh_scale_quant_row)
            qh_scale_dq = pl.reshape(qh_scale_recip, [QH_QUANT_TILE, 1])
            qr_hadamard_scale_dq[o0 : o0 + QH_QUANT_TILE, :] = qh_scale_dq
            qh_scale_quant = pl.reshape(qh_scale_quant_row, [QH_QUANT_TILE, 1])
            qh_q_scaled = pl.row_expand_mul(qh_full_f32, qh_scale_quant)
            qh_q_i32 = pl.cast(qh_q_scaled, target_type=pl.INT32, mode="rint")
            qh_q_half = pl.cast(qh_q_i32, target_type=pl.FP16, mode="round")
            qh_i8 = pl.cast(qh_q_half, target_type=pl.INT8, mode="trunc")
            qr_hadamard_i8[o0 : o0 + QH_QUANT_TILE, 0:IDX_HEAD_DIM] = qh_i8

    return qh_mm_tid, qh_quant_tid


@pl.jit.inline(auto_scope=False)
def indexer_qr_hadamard(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    qr_hadamard_i8: pl.Out[pl.Tensor[[T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8]],
    qr_hadamard_scale_dq: pl.Out[pl.Tensor[[T_PAD * IDX_N_HEADS, 1], pl.FP32]],
):
    """Query half of the indexer: qr projection, rope, hadamard, and INT8 quant."""
    qr_bf16 = pl.create_tensor([T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.BF16)
    idx_qr_mm_tid = indexer_qr_rope(
        x,
        qr,
        qr_scale,
        wq_b,
        wq_b_scale,
        cos,
        sin,
        qr_bf16,
    )
    qh_mm_tid, qh_quant_tid = indexer_qr_hadamard_mm(
        x,
        qr_bf16,
        hadamard,
        qr_hadamard_i8,
        qr_hadamard_scale_dq,
        idx_qr_mm_tid,
    )
    return qh_mm_tid, qh_quant_tid, idx_qr_mm_tid


@pl.jit.inline(auto_scope=False)
def indexer_weights_project(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    weights_gate_dep: pl.Scalar[pl.TASK_ID],
    weights_workers: pl.Scalar[pl.INDEX],
) -> tuple[pl.Tensor[[T_PAD, IDX_N_HEADS], pl.FP32], pl.Scalar[pl.TASK_ID]]:
    """Shared production head-weight projection and dtype publication."""
    bs = pl.tensor.dim(x, 0)
    row_blocks = (bs + MM_ROW_TILE - 1) // MM_ROW_TILE

    x_flat = x
    weights = pl.create_tensor([T_PAD, IDX_N_HEADS], dtype=pl.FP32)
    weights_partial = pl.create_tensor([WEIGHTS_OK * T_PAD, IDX_N_HEADS], dtype=pl.FP32)
    # Caller-ordered weights projection.
    with pl.spmd(
        weights_workers, name_hint="weights_proj", deps=[weights_gate_dep], allow_early_resolve=True
    ) as _weights_tid:
        w_worker = pl.tile.get_block_idx()
        for w_unit in pl.range(w_worker, WEIGHTS_OK * row_blocks, weights_workers):
            w_rb = w_unit // WEIGHTS_OK  # row block outermost
            kb = w_unit - w_rb * WEIGHTS_OK
            w_r0 = w_rb * MM_ROW_TILE
            w_rows = pl.min(MM_ROW_TILE, bs - w_r0)
            k_base = kb * WEIGHTS_K_TILE
            weights_acc = pl.create_tensor([MM_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32)
            # 性能版不再按 Native 的 K 遍历序重排（原 k_order 分支）：那是为了复刻
            # CANN9.0 A3 MatMulV2 的累加次序，代价是 K 块必须切到 256、N 块切到 16，
            # 读 weights_proj 变成 256 次 32 字节跨步读。in-core 实测 MTE2 占 60%、
            # Cube 只占 13.4%，改回整行 [D_TILE, IDX_N_HEADS] 后是一段连续 64KiB。
            for db in pl.range(WEIGHTS_K_TILE // D_TILE):
                d0 = k_base + db * D_TILE
                x_tile = pl.slice(x_flat, [MM_ROW_TILE, D_TILE], [w_r0, d0], valid_shape=[w_rows, D_TILE])
                weights_proj_tile = weights_proj[d0 : d0 + D_TILE, :]
                weights_acc = pl.matmul_acc(weights_acc, x_tile, weights_proj_tile, init_cond=(db == 0))
            weights_partial[kb * T_PAD + w_r0 : kb * T_PAD + w_r0 + MM_ROW_TILE, :] = weights_acc

    with pl.spmd(
        row_blocks,
        name_hint="weights_proj_reduce",
        allow_early_resolve=True,
    ) as weights_tid:
        w_rb = pl.tile.get_block_idx()
        w_r0 = w_rb * MM_ROW_TILE
        w_sum = weights_partial[w_r0 : w_r0 + MM_ROW_TILE, :]
        for kb in pl.unroll(1, WEIGHTS_OK):
            partial_r0 = kb * T_PAD + w_r0
            w_sum = pl.add(w_sum, weights_partial[partial_r0 : partial_r0 + MM_ROW_TILE, :])
        # 性能版在此保留FP32，不复刻Native先落BF16的中间舍入。
        # Cube路径在indexer_head_coefficients里转FP16；更短历史的Vector路径仍读FP32。
        weights[w_r0 : w_r0 + MM_ROW_TILE, :] = pl.mul(w_sum, WEIGHTS_SCALE)

    return weights, weights_tid


@pl.jit.inline(auto_scope=False)
def indexer_weights_score(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    qr_hadamard_i8: pl.Tensor[[T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8],
    qr_hadamard_scale_dq: pl.Tensor[[T_PAD * IDX_N_HEADS, 1], pl.FP32],
    idx_native_kv_cache: pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8],
    idx_block_table: pl.Tensor[[B_DYN, INDEXER_TABLE_COLUMNS_DYN], pl.INT32],
    topk_scores: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32]],
    topk_idxs: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32]],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    cache_write_dep: pl.Scalar[pl.TASK_ID],
    weights_gate_dep: pl.Scalar[pl.TASK_ID],
    qh_quant_tid: pl.Scalar[pl.TASK_ID],
    weights_workers: pl.Scalar[pl.INDEX],
    host_max_seq_len: pl.Scalar[pl.INT32],
    scale_layout: pl.Tensor[[SCALE_LAYOUT_CAPACITY], pl.INT32],
    layout_tid: pl.Scalar[pl.TASK_ID],
) -> tuple[pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32], pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32], pl.Scalar[pl.TASK_ID]]:
    """Weights projection and the score/top-k forest over an already-quantized query."""
    weights, weights_tid = indexer_weights_project(x, weights_proj, weights_gate_dep, weights_workers)

    topk_scores, topk_idxs, leaf_tid = indexer_score_topk_forest(
        qr_hadamard_i8,
        qr_hadamard_scale_dq,
        weights,
        idx_native_kv_cache,
        idx_block_table,
        position_ids,
        kv_seq_lens,
        topk_scores,
        topk_idxs,
        qh_quant_tid,
        weights_tid,
        cache_write_dep,
        host_max_seq_len,
        scale_layout,
        layout_tid,
    )
    return topk_scores, topk_idxs, leaf_tid


@pl.jit.inline
def indexer(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_native_kv_cache: pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8],
    idx_block_table: pl.Tensor[[B_DYN, INDEXER_TABLE_COLUMNS_DYN], pl.INT32],
    topk_scores: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32]],
    topk_idxs: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32]],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    late_dep: pl.Scalar[pl.TASK_ID],
    cache_write_dep: pl.Scalar[pl.TASK_ID],
    host_max_seq_len: pl.Scalar[pl.INT32],
    scale_layout: pl.Tensor[[SCALE_LAYOUT_CAPACITY], pl.INT32],
    layout_tid: pl.Scalar[pl.TASK_ID],
):
    qr_hadamard_i8 = pl.create_tensor([T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.INT8)
    qr_hadamard_scale_dq = pl.create_tensor([T_PAD * IDX_N_HEADS, 1], dtype=pl.FP32)
    _qh_mm_tid, qh_quant_tid, _idx_qr_mm_tid = indexer_qr_hadamard(
        x,
        qr,
        qr_scale,
        wq_b,
        wq_b_scale,
        cos,
        sin,
        hadamard,
        qr_hadamard_i8,
        qr_hadamard_scale_dq,
    )
    weights_gate_dep = pl.system.task_dummy(deps=[])
    topk_scores, topk_idxs, _leaf_tid = indexer_weights_score(
        x,
        weights_proj,
        qr_hadamard_i8,
        qr_hadamard_scale_dq,
        idx_native_kv_cache,
        idx_block_table,
        topk_scores,
        topk_idxs,
        position_ids,
        kv_seq_lens,
        cache_write_dep,
        weights_gate_dep,
        qh_quant_tid,
        TP1_WEIGHTS_WORKERS,
        host_max_seq_len,
        scale_layout,
        layout_tid,
    )
    return topk_scores, topk_idxs
