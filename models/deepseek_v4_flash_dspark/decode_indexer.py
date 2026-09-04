# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Indexer (decode). Mirrors model.py Indexer (line 380-433);
golden is a port of forward's decode branch (prefill `start_pos == 0` path is omitted).
The inner Compressor is invoked via golden_compressor (placeholder)."""


import pypto.language as pl

from config import (
    FLASH as M,
    DECODE_BATCH,
    TP,
    DECODE_SEQ,
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    FP32_NEG_INF,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
)
from decode_indexer_compressor import indexer_compressor

# Dynamic shape variables. S stays static: the score/topk scopes divide by it.
B_DYN = pl.dynamic("B_DYN")
T_DYN = pl.dynamic("T_DYN")  # T = B * S

# model config
B = DECODE_BATCH // TP
S = DECODE_SEQ
T = B * S
D = M.hidden_size
Q_LORA = M.q_lora_rank
ROPE_HEAD_DIM = M.qk_rope_head_dim
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_NOPE_HEAD_DIM = M.index_nope_head_dim
WEIGHTS_SCALE = M.index_weights_scale
MAX_SEQ_LEN = M.max_position_embeddings

# kernel-local
COMPRESS_RATIO = 4   # the indexer only runs on ratio-4 layers
IDX_TOPK = M.index_topk
INNER_OVERLAP = COMPRESS_RATIO == 4
INNER_COFF = 1 + int(INNER_OVERLAP)
INNER_HEAD_DIM = IDX_HEAD_DIM
INNER_OUT_DIM = INNER_COFF * INNER_HEAD_DIM
INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
INNER_STATE_LEN = INNER_COFF * COMPRESS_RATIO
INNER_STATE_MAX_BLOCKS = (
    INNER_STATE_LEN + INNER_STATE_BLOCK_SIZE - 1
) // INNER_STATE_BLOCK_SIZE
INNER_STATE_BLOCK_NUM_DYN = pl.dynamic("INNER_STATE_BLOCK_NUM_DYN")
INNER_STATE_DIM = 2 * INNER_OUT_DIM

IDX_MAX_ROWS = MAX_SEQ_LEN // COMPRESS_RATIO
IDX_MAX_BLOCKS = (IDX_MAX_ROWS + BLOCK_SIZE - 1) // BLOCK_SIZE
IDX_CACHE_BLOCK_NUM_DYN = pl.dynamic("IDX_CACHE_BLOCK_NUM_DYN")

# tiling
CACHE_TILE = min(64, BLOCK_SIZE)
assert BLOCK_SIZE % CACHE_TILE == 0, "CACHE_TILE must not cross a paged idx_kv_cache block"
Q_TILE = 256
# Q_OUT_TILE is the per-task N granularity (sets idx_qr_proj task count); MM_N_TILE
# is the Mat-safe cube N-tile. Q_OUT_TILE fans Q_OUT_TILE // MM_N_TILE cube ops per
# task so task count halves without growing the [Q_TILE, MM_N_TILE] L1 wq load.
Q_OUT_TILE = 1024
T_PAD = ((T + 16 - 1) // 16) * 16  # static upper bound on the token axis
# Matmul M at the 16-row cube floor: a tile taller than the dynamic source is not expressible.
MM_ROW_TILE = 16
# INT32 Acc is MM_ROW_TILE * MM_N_TILE * 4B and must stay under the 128KiB L0C wall.
MM_N_TILE = min(512, (128 * 1024) // (MM_ROW_TILE * 4))
QR_OT_COUNT = IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE  # qr_proj N-tasks per row block
assert Q_OUT_TILE % MM_N_TILE == 0
# Dequant token tile: a whole-T [T, Q_OUT_TILE] FP32 tile does not fit UB.
DEQUANT_T_TILE = min(T, 8)
assert T % DEQUANT_T_TILE == 0
HEAD_DIM_TILE = 32
D_TILE = 512
# weights_proj splits K, not N: a [D_TILE, IDX_N_HEADS] row block reads contiguous GM,
# while an N slice would take 32B out of every 128B row. Each task writes its own
# partial row block, summed by a separate reduce scope. Partials are laid out
# [K slice][T_PAD rows] so the reduce adds whole T_PAD-row blocks.
# WEIGHTS_K_SLICE // D_TILE == 2, so the inner loop is a pl.range: a degenerate
# 2-iteration pl.pipeline(stage=2) miscompiles over matmul.
WEIGHTS_OK = 4
WEIGHTS_K_SLICE = D // WEIGHTS_OK
assert WEIGHTS_K_SLICE % D_TILE == 0
QH_QUANT_TILE = 64
# cube tile for q @ hadamard; bs_heads // QH_MM_TILE truncates, so the tile stays
# at the loosest divisibility the kernel already required.
QH_MM_TILE = 64
# qr_hadamard_matmul runs on persistent workers, each striding over the block list.
QH_WORKERS = 24
QH_HEAD_DIM_TILE = 64
ROPE_ROW_BLOCK = IDX_N_HEADS
# qr_rope SPMD tile == row block: one ROPE_ROW_TILE-row block per SPMD tile.
ROPE_ROW_TILE = 32
assert IDX_N_HEADS >= ROPE_ROW_TILE and IDX_N_HEADS % ROPE_ROW_TILE == 0
TOPK_PAIR_WIDTH = 2 * IDX_TOPK

# Exact Top-K geometry for the 1M selector.
TOPK_CANDIDATES_PER_LEAF = 8192
TOPK_MAX_CANDIDATES = IDX_MAX_ROWS
TOPK_MAX_LEAVES = TOPK_MAX_CANDIDATES // TOPK_CANDIDATES_PER_LEAF
TOPK_LEAVES_PER_GROUP = 2
TOPK_GROUPS_PER_QUERY = TOPK_MAX_LEAVES // TOPK_LEAVES_PER_GROUP
# Each AIV reduces one group at a time. Group roots survive until the query
# merge; each worker reuses a two-row scratch for its current group.
TOPK_GROUP_WORKERS = 48
TOPK_GROUP_ROOT_ROWS = T_PAD * TOPK_GROUPS_PER_QUERY
TOPK_GROUP_SCRATCH_ROWS = TOPK_GROUP_WORKERS * TOPK_LEAVES_PER_GROUP
TOPK_ARENA_ROWS = TOPK_GROUP_ROOT_ROWS + TOPK_GROUP_SCRATCH_ROWS
# One persistent mixed worker per physical 910B AIC. The workers share one
# global leaf sequence, so ragged queries cannot strand per-query lanes.
TOPK_SCORE_WORKERS = 24
assert TOPK_MAX_CANDIDATES % TOPK_CANDIDATES_PER_LEAF == 0
assert TOPK_MAX_LEAVES == 32
assert TOPK_MAX_LEAVES % TOPK_LEAVES_PER_GROUP == 0
assert TOPK_GROUPS_PER_QUERY == 16


@pl.jit.inline
def merge2_top512_pairs(
    pair_arena: pl.Tensor,
    left_slot: pl.Scalar[pl.INDEX],
    right_slot: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
) -> None:
    """Merge two arena rows and store their exact Top-512 pair row."""
    left = pl.load(
        pair_arena, [left_slot, 0], [1, TOPK_PAIR_WIDTH]
    )
    right = pl.load(
        pair_arena, [right_slot, 0], [1, TOPK_PAIR_WIDTH]
    )
    merge_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
    merged_all = pl.tile.mrgsort(left, right, tmp=merge_tmp)
    merged = pl.tile.slice(
        merged_all, [1, TOPK_PAIR_WIDTH], [0, 0]
    )
    pl.store(merged, [output_slot, 0], pair_arena)


@pl.jit.inline
def merge_topk_level_pairs(
    pair_arena: pl.Tensor,
    arena_base: pl.Scalar[pl.INDEX],
    input_count: pl.Scalar[pl.INDEX],
    input_base: pl.Scalar[pl.INDEX],
    output_base: pl.Scalar[pl.INDEX],
) -> None:
    """Reduce one exact-Top-K forest level, forwarding an odd final node."""
    output_count = (input_count + 1) // 2
    for output in pl.range(output_count):
        left_slot = arena_base + input_base + 2 * output
        right_slot = left_slot + 1
        output_slot = arena_base + output_base + output
        if right_slot < arena_base + input_base + input_count:
            merge2_top512_pairs(
                pair_arena,
                left_slot,
                right_slot,
                output_slot,
            )
        else:
            forwarded = pl.load(
                pair_arena, [left_slot, 0], [1, TOPK_PAIR_WIDTH]
            )
            pl.store(forwarded, [output_slot, 0], pair_arena)


@pl.jit.inline
def indexer_topk_leaf(
    score_arena: pl.Tensor[[T_DYN, TOPK_MAX_CANDIDATES], pl.FP32],
    pair_arena: pl.Tensor[[TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], pl.FP32],
    query: pl.Scalar[pl.INDEX],
    logical_begin: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
) -> None:
    """Sort one scored 8K leaf and store its exact Top-512 pair row."""
    logical_begin_i32 = pl.cast(logical_begin, pl.INT32)
    leaf_indices = pl.add(
        pl.tile.arange(
            0,
            [1, TOPK_CANDIDATES_PER_LEAF],
            dtype=pl.INT32,
        ),
        logical_begin_i32,
    )
    leaf_scores_raw = pl.load(
        score_arena,
        [query, logical_begin],
        [1, TOPK_CANDIDATES_PER_LEAF],
        valid_shape=[1, valid_count],
    )
    leaf_scores = pl.tile.fillpad(
        leaf_scores_raw,
        pad_value=pl.PadValue.min,
    )
    leaf_scores = pl.maximum(
        leaf_scores,
        pl.tile.full(
            [1, TOPK_CANDIDATES_PER_LEAF],
            dtype=pl.FP32,
            value=FP32_NEG_INF,
        ),
    )
    pairs = pl.sort32(
        leaf_scores,
        pl.reinterpret_view(leaf_indices, pl.UINT32),
    )
    pairs = pl.mrgsort(pairs, block_len=64)
    pairs = pl.mrgsort(pairs, block_len=256)
    pairs = pl.mrgsort(pairs, block_len=1024)
    pairs = pl.mrgsort(pairs, block_len=4096)
    pl.store(
        pl.tile.slice(pairs, [1, TOPK_PAIR_WIDTH], [0, 0]),
        [output_slot, 0],
        pair_arena,
    )


@pl.jit.incore
def indexer_topk_group_wave(
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    score_arena: pl.Tensor[[T_DYN, TOPK_MAX_CANDIDATES], pl.FP32],
    pair_arena: pl.Tensor[[TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], pl.FP32],
):
    """Reduce globally striped two-leaf subtrees into compact roots."""
    worker = pl.tile.get_block_idx()
    query_count = pl.tensor.dim(position_ids, 0)
    global_group_base = 0
    for query in pl.range(query_count):
        batch_idx = query // S
        position = pl.read(position_ids, [query])
        cache_len = pl.read(kv_seq_lens, [batch_idx]) // COMPRESS_RATIO
        visible_count = pl.max(
            pl.min(
                pl.min(cache_len, (position + 1) // COMPRESS_RATIO),
                TOPK_MAX_CANDIDATES,
            ),
            0,
        )
        leaf_count = (
            visible_count + TOPK_CANDIDATES_PER_LEAF - 1
        ) // TOPK_CANDIDATES_PER_LEAF
        group_count = (
            leaf_count + TOPK_LEAVES_PER_GROUP - 1
        ) // TOPK_LEAVES_PER_GROUP
        base_mod = global_group_base % TOPK_GROUP_WORKERS
        first_group = (worker + base_mod) % TOPK_GROUP_WORKERS
        for group in pl.range(
            first_group, group_count, TOPK_GROUP_WORKERS
        ):
            leaf_begin = group * TOPK_LEAVES_PER_GROUP
            group_leaf_count = pl.min(
                TOPK_LEAVES_PER_GROUP,
                leaf_count - leaf_begin,
            )
            group_root_slot = (
                query * TOPK_GROUPS_PER_QUERY + group
            )
            if group_leaf_count == 1:
                logical_begin = (
                    leaf_begin * TOPK_CANDIDATES_PER_LEAF
                )
                valid_count = pl.min(
                    TOPK_CANDIDATES_PER_LEAF,
                    visible_count - logical_begin,
                )
                indexer_topk_leaf(
                    score_arena,
                    pair_arena,
                    query,
                    logical_begin,
                    valid_count,
                    group_root_slot,
                )
            else:
                scratch_base = (
                    TOPK_GROUP_ROOT_ROWS
                    + worker * TOPK_LEAVES_PER_GROUP
                )
                for group_leaf in pl.unroll(TOPK_LEAVES_PER_GROUP):
                    leaf = leaf_begin + group_leaf
                    logical_begin = (
                        leaf * TOPK_CANDIDATES_PER_LEAF
                    )
                    valid_count = pl.min(
                        TOPK_CANDIDATES_PER_LEAF,
                        visible_count - logical_begin,
                    )
                    indexer_topk_leaf(
                        score_arena,
                        pair_arena,
                        query,
                        logical_begin,
                        valid_count,
                        scratch_base + group_leaf,
                    )
                merge2_top512_pairs(
                    pair_arena,
                    scratch_base,
                    scratch_base + 1,
                    group_root_slot,
                )
        global_group_base = global_group_base + group_count


@pl.jit.incore
def indexer_topk_query_merge(
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    pair_arena: pl.Tensor[[TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], pl.FP32],
    topk_scores: pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32],
    topk_indices: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32],
):
    """Merge compact group roots and materialize each query's Top-512."""
    query = pl.tile.get_block_idx()
    batch_idx = query // S
    position = pl.read(position_ids, [query])
    cache_len = pl.read(kv_seq_lens, [batch_idx]) // COMPRESS_RATIO
    visible_count = pl.min(
        pl.min(cache_len, (position + 1) // COMPRESS_RATIO),
        TOPK_MAX_CANDIDATES,
    )
    pl.store(
        pl.tile.full(
            [1, IDX_TOPK],
            dtype=pl.FP32,
            value=FP32_NEG_INF,
        ),
        [query, 0],
        topk_scores,
    )
    pl.store(
        pl.tile.full(
            [1, IDX_TOPK], dtype=pl.INT32, value=-1
        ),
        [query, 0],
        topk_indices,
    )

    if visible_count > 0:
        leaf_count = (
            visible_count + TOPK_CANDIDATES_PER_LEAF - 1
        ) // TOPK_CANDIDATES_PER_LEAF
        group_count = (
            leaf_count + TOPK_LEAVES_PER_GROUP - 1
        ) // TOPK_LEAVES_PER_GROUP
        arena_base = query * TOPK_GROUPS_PER_QUERY
        if group_count > 1:
            level1_count = (group_count + 1) // 2
            merge_topk_level_pairs(
                pair_arena,
                arena_base,
                group_count,
                0,
                0,
            )
            if level1_count > 1:
                level2_count = (level1_count + 1) // 2
                merge_topk_level_pairs(
                    pair_arena,
                    arena_base,
                    level1_count,
                    0,
                    0,
                )
                if level2_count > 1:
                    level3_count = (level2_count + 1) // 2
                    merge_topk_level_pairs(
                        pair_arena,
                        arena_base,
                        level2_count,
                        0,
                        0,
                    )
                    if level3_count > 1:
                        merge_topk_level_pairs(
                            pair_arena,
                            arena_base,
                            level3_count,
                            0,
                            0,
                        )

        root_slot = arena_base
        root_pairs = pl.load(
            pair_arena,
            [root_slot, 0],
            [1, TOPK_PAIR_WIDTH],
        )
        pl.store(
            pl.tile.gather_mask(
                root_pairs,
                mask_pattern=pl.tile.MaskPattern.P0101,
                output_dtype=pl.FP32,
            ),
            [query, 0],
            topk_scores,
        )
        root_indices = pl.tile.gather_mask(
            root_pairs,
            mask_pattern=pl.tile.MaskPattern.P1010,
            output_dtype=pl.INT32,
        )
        output_indices = pl.tile.full(
            [1, IDX_TOPK], dtype=pl.INT32, value=-1
        )
        valid_topk = pl.min(visible_count, IDX_TOPK)
        for lane in pl.range(valid_topk):
            pl.tile.write(
                output_indices,
                [0, lane],
                pl.tile.read(root_indices, [0, lane]),
            )
        pl.store(
            output_indices, [query, 0], topk_indices
        )


@pl.jit.inline(auto_scope=False)
def indexer_score_topk_forest(
    qr_hadamard_i8: pl.Tensor[
        [T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8
    ],
    qr_hadamard_scale_dq: pl.Tensor[
        [T_PAD * IDX_N_HEADS, 1], pl.FP32
    ],
    weights: pl.Tensor[[T_PAD, IDX_N_HEADS], pl.FP32],
    idx_kv_cache: pl.Tensor[
        [IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8
    ],
    idx_kv_scale: pl.Tensor[
        [IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32
    ],
    idx_block_table: pl.Tensor[[B_DYN, IDX_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    topk_scores: pl.Out[
        pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32]
    ],
    topk_idxs: pl.Out[
        pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32]
    ],
    qh_quant_tid: pl.Scalar[pl.TASK_ID],
    weights_tid: pl.Scalar[pl.TASK_ID],
    cache_write_tid: pl.Scalar[pl.TASK_ID],
):
    """Run exact Top-K with the score and pair arenas on separate rings."""
    bs = pl.tensor.dim(position_ids, 0)
    b_dim = pl.tensor.dim(idx_block_table, 0)
    idx_block_num = pl.tensor.dim(idx_kv_cache, 0)
    idx_table_len = b_dim * IDX_MAX_BLOCKS
    kv_cache_i8_flat = pl.reshape(
        idx_kv_cache,
        [idx_block_num * BLOCK_SIZE, IDX_HEAD_DIM],
    )
    kv_scale_flat = pl.reshape(
        idx_kv_scale,
        [idx_block_num * BLOCK_SIZE, 1],
    )
    idx_block_table_flat = pl.reshape(
        idx_block_table,
        [idx_table_len],
    )
    pair_arena = pl.create_tensor(
        [TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], dtype=pl.FP32
    )
    with pl.scope():
        score_arena = pl.create_tensor(
            [bs, TOPK_MAX_CANDIDATES], dtype=pl.FP32
        )
        with pl.spmd(
            TOPK_SCORE_WORKERS,
            name_hint="indexer_score_leaf_wave",
            deps=[qh_quant_tid, weights_tid, cache_write_tid],
            optimizations=[pl.split(pl.SplitMode.NONE, slot_num=2)],
        ) as score_tid:
            worker = pl.tile.get_block_idx()
            query_count = pl.tensor.dim(position_ids, 0)
            global_leaf_base = 0
            for query in pl.range(query_count):
                batch_idx = query // S
                position = pl.read(position_ids, [query])
                cache_len = (
                    pl.read(kv_seq_lens, [batch_idx]) // COMPRESS_RATIO
                )
                visible_count = pl.max(
                    pl.min(
                        pl.min(
                            cache_len,
                            (position + 1) // COMPRESS_RATIO,
                        ),
                        TOPK_MAX_CANDIDATES,
                    ),
                    0,
                )
                leaf_count = (
                    visible_count + TOPK_CANDIDATES_PER_LEAF - 1
                ) // TOPK_CANDIDATES_PER_LEAF
                base_mod = global_leaf_base % TOPK_SCORE_WORKERS
                first_leaf = (worker + base_mod) % TOPK_SCORE_WORKERS
                for leaf in pl.range(
                    first_leaf, leaf_count, TOPK_SCORE_WORKERS
                ):
                    logical_begin = leaf * TOPK_CANDIDATES_PER_LEAF
                    valid_count = pl.min(
                        TOPK_CANDIDATES_PER_LEAF,
                        visible_count - logical_begin,
                    )
                    query_head_begin = query * IDX_N_HEADS
                    query_vector = qr_hadamard_i8[
                        query_head_begin : query_head_begin + IDX_N_HEADS,
                        0:IDX_HEAD_DIM,
                    ]
                    query_scale = pl.reshape(
                        qr_hadamard_scale_dq[
                            query_head_begin : query_head_begin + IDX_N_HEADS,
                            0:1,
                        ],
                        [1, IDX_N_HEADS],
                    )
                    query_weight = weights[
                        query : query + 1,
                        0:IDX_N_HEADS,
                    ]
                    page_count = (
                        valid_count + BLOCK_SIZE - 1
                    ) // BLOCK_SIZE
                    for page in pl.pipeline(0, page_count, stage=2):
                        page_begin = page * BLOCK_SIZE
                        logical_row = logical_begin + page_begin
                        logical_page = logical_row // BLOCK_SIZE
                        physical_block = pl.cast(
                            pl.read(
                                idx_block_table_flat,
                                [
                                    batch_idx * IDX_MAX_BLOCKS
                                    + logical_page
                                ],
                            ),
                            pl.INDEX,
                        )
                        physical_row = physical_block * BLOCK_SIZE
                        kv_i8 = kv_cache_i8_flat[
                            physical_row : physical_row + BLOCK_SIZE,
                            0:IDX_HEAD_DIM,
                        ]
                        score_i32 = pl.matmul(
                            kv_i8,
                            query_vector,
                            out_dtype=pl.INT32,
                            b_trans=True,
                        )
                        score_fp32 = pl.cast(
                            score_i32,
                            target_type=pl.FP32,
                            mode="none",
                        )
                        score_fp32 = pl.col_expand_mul(
                            score_fp32,
                            query_scale,
                        )
                        score_fp32 = pl.maximum(score_fp32, 0.0)
                        score_fp32 = pl.col_expand_mul(
                            score_fp32,
                            query_weight,
                        )
                        kv_scale = kv_scale_flat[
                            physical_row : physical_row + BLOCK_SIZE,
                            0:1,
                        ]
                        score = pl.mul(
                            pl.row_sum(score_fp32),
                            kv_scale,
                        )
                        score_row = pl.reshape(score, [1, BLOCK_SIZE])
                        valid_rows = pl.min(
                            BLOCK_SIZE,
                            valid_count - page_begin,
                        )
                        score_valid = pl.fillpad(
                            pl.set_validshape(score_row, 1, valid_rows),
                            pad_value=pl.PadValue.min,
                        )
                        score_arena[
                            query : query + 1,
                            logical_row : logical_row + BLOCK_SIZE,
                        ] = score_valid
                global_leaf_base = global_leaf_base + leaf_count
        with pl.spmd(
            TOPK_GROUP_WORKERS,
            name_hint="indexer_topk_group_wave",
            deps=[score_tid],
        ) as topk_tid:
            indexer_topk_group_wave(
                position_ids,
                kv_seq_lens,
                score_arena,
                pair_arena,
            )
        with pl.spmd(
            bs,
            name_hint="indexer_topk_query_merge",
            deps=[topk_tid],
        ) as _score_tid:
            indexer_topk_query_merge(
                position_ids,
                kv_seq_lens,
                pair_arena,
                topk_scores,
                topk_idxs,
            )

    return topk_scores, topk_idxs


@pl.jit.inline
def indexer(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    # Interleave-duplicated (j>>1) cos and sign-folded sin, built once by the caller:
    #   cos[j] = cos_half[j>>1];  sin[j] = sin_half[j>>1] * sign[j], sign = [-1,+1,...]
    cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    # C8 indexer cache: INT8 KV (quant-on-write) + per-position FP32 dequant scale; no bf16 cache.
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[B_DYN, IDX_MAX_BLOCKS], pl.INT32],
    topk_scores: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32]],
    topk_idxs: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32]],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    late_dep: pl.Scalar[pl.TASK_ID],
    cache_write_dep: pl.Scalar[pl.TASK_ID],
):
    bs = pl.tensor.dim(x, 0)
    bs_heads = bs * IDX_N_HEADS
    row_blocks = (bs + MM_ROW_TILE - 1) // MM_ROW_TILE
    qr_acc_pad = pl.create_tensor([T_PAD, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.INT32)
    for qr_unit in pl.spmd(QR_OT_COUNT * row_blocks, name_hint="idx_qr_proj_matmul", allow_early_resolve=True):
        qr_rb = qr_unit // QR_OT_COUNT  # row block outermost
        ot = qr_unit - qr_rb * QR_OT_COUNT
        qr_r0 = qr_rb * MM_ROW_TILE
        qr_rows = pl.min(MM_ROW_TILE, bs - qr_r0)
        o_base = ot * Q_OUT_TILE
        for ns in pl.range(0, Q_OUT_TILE, MM_N_TILE):
            qr_acc = pl.create_tensor([MM_ROW_TILE, MM_N_TILE], dtype=pl.INT32)
            for kb in pl.pipeline(0, Q_LORA // Q_TILE, stage=2):
                q0 = kb * Q_TILE
                qr_tile = pl.slice(qr, [MM_ROW_TILE, Q_TILE], [qr_r0, q0], valid_shape=[qr_rows, Q_TILE])
                wq_tile = wq_b[q0 : q0 + Q_TILE, o_base + ns : o_base + ns + MM_N_TILE]
                if q0 == 0:
                    qr_acc = pl.matmul(qr_tile, wq_tile, out_dtype=pl.INT32)
                else:
                    qr_acc = pl.matmul_acc(qr_acc, qr_tile, wq_tile)
            qr_acc_pad[qr_r0 : qr_r0 + MM_ROW_TILE, o_base + ns : o_base + ns + MM_N_TILE] = qr_acc
    qr_proj = pl.create_tensor([bs, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.FP32)
    for ot in pl.spmd(IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE, name_hint="idx_qr_proj_dequant", allow_early_resolve=True):
        o_base = ot * Q_OUT_TILE
        wq_scale = pl.reshape(wq_b_scale[o_base : o_base + Q_OUT_TILE], [1, Q_OUT_TILE])
        for dq_t0 in pl.range(0, bs, DEQUANT_T_TILE):
            acc_fp32 = pl.cast(
                qr_acc_pad[dq_t0 : dq_t0 + DEQUANT_T_TILE, o_base : o_base + Q_OUT_TILE],
                target_type=pl.FP32, mode="none")
            qr_scale_tile = qr_scale[dq_t0 : dq_t0 + DEQUANT_T_TILE, :]
            qr_dequant = pl.col_expand_mul(pl.row_expand_mul(acc_fp32, qr_scale_tile), wq_scale)
            qr_proj[dq_t0 : dq_t0 + DEQUANT_T_TILE, o_base : o_base + Q_OUT_TILE] = qr_dequant

    qr_proj_flat = pl.reshape(qr_proj, [bs_heads, IDX_HEAD_DIM])
    # BF16 q for the Hadamard matmul: nope half rounded from the FP32 dequant, rope
    # half rotated then rounded.
    qr_bf16 = pl.create_tensor([bs_heads, IDX_HEAD_DIM], dtype=pl.BF16)
    # spmd over ROPE_ROW_TILE-row blocks; token_idx = block base // ROPE_ROW_BLOCK
    # picks the token-local cos/sin row. cos/sin arrive already interleave-duplicated and
    # sign-folded (built once by the caller), and col_expand_mul folds the [1, ROPE_HEAD_DIM]
    # row broadcast into the rotation multiply -- so no cos_il/sin_il tile is materialized
    # and no per-block dup-gather runs.
    #   out[j] = x[j]*cos_il[j] + x[j^1]*sin_il_signed[j]
    #
    # The j^1 lane-swap index permutes data, so no host table can hold it -- but it is
    # block-invariant, and rebuilding it inside the spmd cost the same arange/trunc-cast/
    # lane/arithmetic chain on all 16 blocks. Built once here instead (same form as the
    # rope_swap scope in decode_sparse_attn_hca) and loaded per block. No pypto bitwise
    # op is reachable at the tensor level, so the fp32 arithmetic chain is the only form.
    rope_swap_idx_t = pl.create_tensor([ROPE_ROW_TILE, ROPE_HEAD_DIM], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_rope_swap_idx", allow_early_resolve=True):
        sw_col = pl.col_expand_mul(
            pl.full([ROPE_ROW_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0),
            pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        sw_dup_f = pl.cast(pl.cast(pl.mul(sw_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        sw_lane = pl.sub(sw_col, pl.mul(sw_dup_f, 2.0))                                                # j%2
        rope_swap_idx_t[0:ROPE_ROW_TILE, 0:ROPE_HEAD_DIM] = pl.cast(
            pl.sub(pl.add(sw_col, 1.0), pl.mul(sw_lane, 2.0)), target_type=pl.INT32)                   # j^1

    for idx in pl.spmd(bs_heads // ROPE_ROW_TILE, name_hint="qr_rope", allow_early_resolve=True):
        o0 = idx * ROPE_ROW_TILE
        token_idx = o0 // ROPE_ROW_BLOCK
        rope_swap_idx = rope_swap_idx_t[0:ROPE_ROW_TILE, 0:ROPE_HEAD_DIM]
        cos_row = cos[token_idx : token_idx + 1, 0 : ROPE_HEAD_DIM]
        sin_row = sin[token_idx : token_idx + 1, 0 : ROPE_HEAD_DIM]
        qr_nope_slice = qr_proj_flat[o0 : o0 + ROPE_ROW_TILE, 0 : IDX_NOPE_HEAD_DIM]
        qr_rope_slice = qr_proj_flat[o0 : o0 + ROPE_ROW_TILE, IDX_NOPE_HEAD_DIM : IDX_HEAD_DIM]
        qr_swapped = pl.gather(qr_rope_slice, dim=-1, index=rope_swap_idx)
        rope_rot = pl.add(
            pl.col_expand_mul(qr_rope_slice, cos_row), pl.col_expand_mul(qr_swapped, sin_row))
        qr_vec = pl.concat(pl.cast(qr_nope_slice, target_type=pl.BF16, mode="rint"), pl.cast(rope_rot, target_type=pl.BF16, mode="rint"))
        qr_bf16[o0 : o0 + ROPE_ROW_TILE, :] = qr_vec

    # cube-only scope: q @ hadamard lands in GM, keeping the vector amax/quant below
    # in its own scope so the two run as separate cube and vector tasks.
    qh_acc_gm = pl.create_tensor([bs_heads, IDX_HEAD_DIM], dtype=pl.FP32)
    for qh_worker in pl.spmd(QH_WORKERS, name_hint="qr_hadamard_matmul", allow_early_resolve=True):
        # The hadamard matrix is the same for every block, so it is read once per worker.
        qh_hadamard = hadamard[0:IDX_HEAD_DIM, 0:IDX_HEAD_DIM]
        for idx in pl.range(qh_worker, bs_heads // QH_MM_TILE, QH_WORKERS):
            o0 = idx * QH_MM_TILE
            qh_acc = pl.matmul(qr_bf16[o0 : o0 + QH_MM_TILE, :], qh_hadamard, out_dtype=pl.FP32)
            qh_acc_gm[o0 : o0 + QH_MM_TILE, :] = qh_acc

    qr_hadamard_i8 = pl.create_tensor(
        [T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.INT8
    )
    qr_hadamard_scale_dq = pl.create_tensor(
        [T_PAD * IDX_N_HEADS, 1], dtype=pl.FP32
    )
    with pl.spmd(
        bs_heads // QH_QUANT_TILE,
        name_hint="qr_hadamard_quant",
        allow_early_resolve=True,
    ) as qh_quant_tid:
        idx = pl.tile.get_block_idx()
        o0 = idx * QH_QUANT_TILE
        qh_amax = pl.full([1, QH_QUANT_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for h0 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
            qh_a_f32 = qh_acc_gm[o0 : o0 + QH_QUANT_TILE, h0 : h0 + QH_HEAD_DIM_TILE]
            qh_a_abs = pl.maximum(qh_a_f32, pl.neg(qh_a_f32))
            qh_a_max = pl.reshape(pl.row_max(qh_a_abs), [1, QH_QUANT_TILE])
            qh_amax = pl.maximum(qh_amax, qh_a_max)
        qh_scale_quant_row = pl.div(pl.full([1, QH_QUANT_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), qh_amax)
        qh_scale_dq = pl.reshape(pl.recip(qh_scale_quant_row), [QH_QUANT_TILE, 1])
        qr_hadamard_scale_dq[o0 : o0 + QH_QUANT_TILE, :] = qh_scale_dq
        qh_scale_quant = pl.reshape(qh_scale_quant_row, [QH_QUANT_TILE, 1])
        for h1 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
            qh_q_f32 = qh_acc_gm[o0 : o0 + QH_QUANT_TILE, h1 : h1 + QH_HEAD_DIM_TILE]
            qh_q_scaled = pl.row_expand_mul(qh_q_f32, qh_scale_quant)
            qh_q_i32 = pl.cast(qh_q_scaled, target_type=pl.INT32, mode="rint")
            qh_q_half = pl.cast(qh_q_i32, target_type=pl.FP16, mode="round")
            qh_i8 = pl.cast(qh_q_half, target_type=pl.INT8, mode="trunc")
            qr_hadamard_i8[o0 : o0 + QH_QUANT_TILE, h1 : h1 + QH_HEAD_DIM_TILE] = qh_i8

    x_flat = x
    weights = pl.create_tensor([T_PAD, IDX_N_HEADS], dtype=pl.FP32)
    weights_partial = pl.create_tensor([WEIGHTS_OK * T_PAD, IDX_N_HEADS], dtype=pl.FP32)
    # Deferred behind the caller's rms_norm dummy barrier: qkv's qr_proj_matmul is the
    # critical path and must win the cores when rms_norm retires.
    with pl.spmd(WEIGHTS_OK * row_blocks, name_hint="weights_proj", deps=[late_dep]) as _weights_tid:
        w_unit = pl.tile.get_block_idx()
        w_rb = w_unit // WEIGHTS_OK  # row block outermost
        kb = w_unit - w_rb * WEIGHTS_OK
        w_r0 = w_rb * MM_ROW_TILE
        w_rows = pl.min(MM_ROW_TILE, bs - w_r0)
        k_base = kb * WEIGHTS_K_SLICE
        weights_acc = pl.create_tensor([MM_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32)
        for db in pl.range(WEIGHTS_K_SLICE // D_TILE):
            d0 = k_base + db * D_TILE
            x_tile = pl.slice(x_flat, [MM_ROW_TILE, D_TILE], [w_r0, d0], valid_shape=[w_rows, D_TILE])
            weights_proj_tile = weights_proj[d0 : d0 + D_TILE, :]
            if db == 0:
                weights_acc = pl.matmul(x_tile, weights_proj_tile, out_dtype=pl.FP32)
            else:
                weights_acc = pl.matmul_acc(weights_acc, x_tile, weights_proj_tile)
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
        weights[w_r0 : w_r0 + MM_ROW_TILE, :] = pl.mul(w_sum, WEIGHTS_SCALE)

    topk_scores, topk_idxs = indexer_score_topk_forest(
        qr_hadamard_i8,
        qr_hadamard_scale_dq,
        weights,
        idx_kv_cache,
        idx_kv_scale,
        idx_block_table,
        position_ids,
        kv_seq_lens,
        topk_scores,
        topk_idxs,
        qh_quant_tid,
        weights_tid,
        cache_write_dep,
    )
    return topk_scores, topk_idxs


@pl.jit
def indexer_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    cmp_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    cmp_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_kv: pl.Tensor[[T_DYN, INNER_HEAD_DIM], pl.FP32],
    inner_compress_state: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[B_DYN, INNER_STATE_MAX_BLOCKS], pl.INT32],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[B_DYN, IDX_MAX_BLOCKS], pl.INT32],
    topk_scores: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32]],
    topk_idxs: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32]],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    idx_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
):
    x.bind_dynamic(0, T_DYN)
    qr.bind_dynamic(0, T_DYN)
    qr_scale.bind_dynamic(0, T_DYN)
    cos.bind_dynamic(0, T_DYN)
    sin.bind_dynamic(0, T_DYN)
    cmp_cos.bind_dynamic(0, T_DYN)
    cmp_sin.bind_dynamic(0, T_DYN)
    inner_kv.bind_dynamic(0, T_DYN)
    inner_compress_state_block_table.bind_dynamic(0, B_DYN)
    idx_block_table.bind_dynamic(0, B_DYN)
    topk_scores.bind_dynamic(0, T_DYN)
    topk_idxs.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    idx_slot_mapping.bind_dynamic(0, T_DYN)
    inner_state_slot_mapping.bind_dynamic(0, T_DYN)
    kv_seq_lens.bind_dynamic(0, B_DYN)

    # Standalone: no rms_norm producer, so the barrier fences nothing (ready on submit).
    late_dep = pl.system.task_dummy(deps=[])
    cache_write_dep = indexer_compressor(
        x, inner_kv,
        inner_compress_state, inner_compress_state_block_table,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        cmp_cos, cmp_sin, hadamard, idx_kv_cache, idx_kv_scale,
        position_ids, idx_slot_mapping, inner_state_slot_mapping,
        late_dep,
    )
    topk_scores, topk_idxs = indexer(
        x,
        qr,
        qr_scale,
        wq_b,
        wq_b_scale,
        weights_proj,
        cos,
        sin,
        hadamard,
        idx_kv_cache,
        idx_kv_scale,
        idx_block_table,
        topk_scores,
        topk_idxs,
        position_ids,
        kv_seq_lens,
        late_dep,
        cache_write_dep,
    )
    return topk_scores, idx_kv_cache, idx_kv_scale, topk_idxs


def gen_shared_weight(shape, dequant_std, chan_cv):
    """Synthesize a per-output-channel-symmetric INT8 weight + FP32 scale by simulating the
    real DeepSeek-V4-Flash MXFP8 quant grid (e4m3, 128x128-block E8M0 scale), then re-quantizing
    per-output-channel. Used for the indexer ``idx wq_b`` (and shared by decode_csa),
    which follows the same FP8 grid as the shared experts: ~200 discrete levels, ~1.1% zero
    spike, per-channel scale CV ~0.61. A plain randn INT8 misses that level/scale structure.
    ``chan_cv`` (log-space source-gain std) injects the per-output-channel magnitude spread the
    coarse 128-block scale leaves behind; per-channel INT8 is scale-invariant, so the grid sets
    the level shape and ``dequant_std`` only sets the absolute scale magnitude.

    ``shape`` last dim = reduction (in) dim; leading dims map to the per-output-channel scale
    shape ([out, in] -> scale [out]).
    """
    import torch

    FP8_MAX, TINY = 448.0, 1e-20

    def sim_fp8(W, block=128):   # e4m3 + 128x128-block E8M0 (round-up) scale on (out, in)
        out, inn = W.shape
        Wb = W.reshape(out // block, block, inn // block, block)
        scale = torch.exp2(torch.ceil(torch.log2((Wb.abs().amax(dim=(1, 3), keepdim=True) / FP8_MAX).clamp_min(TINY))))
        q = (Wb / scale).to(torch.float8_e4m3fn).float() * scale
        return q.reshape(out, inn)

    W = torch.randn(*shape) * torch.exp(chan_cv * torch.randn(*shape[:-1], 1))  # per-channel gain
    Wq = sim_fp8(W)
    amax = Wq.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale = amax / INT8_SCALE_MAX
    w_i8 = torch.round(Wq / scale).clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
    scale = (scale * (dequant_std / (w_i8.float() * scale).std())).squeeze(-1).float()
    return w_i8, scale


def golden_indexer(tensors, inner_full=None):
    """Torch reference for Indexer.forward decode branch; prefill `start_pos == 0` path is omitted.

    ``inner_full`` supplies the cache half's stream-side inputs, which under CP is
    the whole TP group's token stream rather than the rank's rows.
    """
    import torch
    from decode_indexer_compressor import golden_compressor
    from utils import int8_quant_per_row

    x = tensors["x"].float()
    qr = tensors["qr"]
    qr_scale = tensors["qr_scale"].float()
    wq_b = tensors["wq_b"]
    wq_b_scale = tensors["wq_b_scale"].float()
    weights_proj = tensors["weights_proj"].float()
    cos = tensors["cos"]
    sin = tensors["sin"]
    hadamard = tensors["hadamard"].float()

    kv_seq_lens = tensors["kv_seq_lens"].to(torch.int64)

    tokens = x.shape[0]
    bsz, seqlen = tokens // S, S
    x = x.view(tokens, D)
    ratio, rd = COMPRESS_RATIO, ROPE_HEAD_DIM

    q_i32 = qr.to(torch.int32) @ wq_b.to(torch.int32)
    q = (q_i32.float() * qr_scale * wq_b_scale.view(1, -1)).view(
        tokens, IDX_N_HEADS, IDX_HEAD_DIM
    )
    q_rope = q[..., -rd:]
    q_rope_swapped = q_rope.unflatten(-1, (-1, 2)).flip(-1).flatten(-2)
    q_rope = q_rope * cos[:, None, :] + q_rope_swapped * sin[:, None, :]
    q = torch.cat([q[..., :-rd], q_rope], dim=-1)

    q = q.to(torch.bfloat16).float() @ hadamard
    # W8A8C16: q and Indexer Cache are quantized per row to INT8 for score matmul,
    # then dequantized with q_scale * kv_scale.
    # flash: fp4_act_quant on q (FP4 simulation).

    inner_src = tensors if inner_full is None else inner_full
    inner_tensors = {
        "x": inner_src["x"],
        "kv": inner_src["inner_kv"],
        "wkv": tensors["inner_wkv"],
        "wgate": tensors["inner_wgate"],
        "ape": tensors["inner_ape"],
        "norm_w": tensors["inner_norm_w"],
        "cos": inner_src["cmp_cos"],
        "sin": inner_src["cmp_sin"],
        "hadamard": tensors["hadamard"],
        "compress_state": tensors["inner_compress_state"],
        "compress_state_block_table": inner_src["inner_compress_state_block_table"],
        "idx_kv_cache": tensors["idx_kv_cache"],
        "idx_kv_scale": tensors["idx_kv_scale"],
        "position_ids": inner_src["position_ids"],
        "idx_slot_mapping": inner_src["idx_slot_mapping"],
        "inner_state_slot_mapping": inner_src["inner_state_slot_mapping"],
    }
    golden_compressor(inner_tensors)

    weights = (x @ weights_proj) * WEIGHTS_SCALE

    # C8 cache: pre-quantized INT8 KV + per-position dequant scale (no score-time re-quant)
    idx_kv_cache_i8 = tensors["idx_kv_cache"]
    idx_kv_scale = tensors["idx_kv_scale"].float()
    idx_block_table = tensors["idx_block_table"]
    topk_scores = torch.full(
        (tokens, IDX_TOPK), FP32_NEG_INF, dtype=torch.float32
    )
    topk_idxs = torch.full((tokens, IDX_TOPK), -1, dtype=torch.int32)
    q_i8, q_scale = int8_quant_per_row(
        q.reshape(tokens * IDX_N_HEADS, IDX_HEAD_DIM)
    )
    q_i8 = q_i8.view(tokens, IDX_N_HEADS, IDX_HEAD_DIM)
    q_scale = q_scale.view(tokens, IDX_N_HEADS, 1)
    flat_cache = idx_kv_cache_i8.reshape(-1, IDX_HEAD_DIM)
    flat_scale = idx_kv_scale.reshape(-1, 1)

    for b in range(bsz):
        cache_len = min(
            int(kv_seq_lens[b].item()) // ratio,
            TOPK_MAX_CANDIDATES,
        )
        if cache_len <= 0:
            continue
        logical_rows = torch.arange(cache_len, dtype=torch.int64)
        physical_pages = idx_block_table[
            b, logical_rows // BLOCK_SIZE
        ].to(torch.int64)
        valid_pages = (physical_pages >= 0) & (
            physical_pages < idx_kv_cache_i8.shape[0]
        )
        physical_rows = (
            physical_pages.clamp(min=0) * BLOCK_SIZE
            + logical_rows % BLOCK_SIZE
        )
        kv_i8 = flat_cache[physical_rows]
        kv_scale = flat_scale[physical_rows]
        for s in range(seqlen):
            token = b * S + s
            visible_len = min(
                cache_len,
                int(tensors["position_ids"][token].item() + 1) // ratio,
                TOPK_MAX_CANDIDATES,
            )
            if visible_len <= 0:
                continue
            running_scores = torch.empty(0, dtype=torch.float32)
            running_indices = torch.empty(0, dtype=torch.int64)
            for begin in range(0, visible_len, TOPK_CANDIDATES_PER_LEAF):
                end = min(begin + TOPK_CANDIDATES_PER_LEAF, visible_len)
                score_i32 = torch.einsum(
                    "hd,td->ht",
                    q_i8[token].to(torch.int32),
                    kv_i8[begin:end].to(torch.int32),
                )
                score = score_i32.float() * q_scale[token]
                score = (
                    torch.relu(score) * weights[token].unsqueeze(-1)
                ).sum(dim=0)
                score = score * kv_scale[begin:end, 0]
                score = torch.where(
                    valid_pages[begin:end],
                    score,
                    torch.full_like(score, FP32_NEG_INF),
                )
                indices = torch.arange(begin, end, dtype=torch.int64)
                merged_scores = torch.cat([running_scores, score])
                merged_indices = torch.cat([running_indices, indices])
                keep = min(IDX_TOPK, merged_scores.numel())
                running_scores, selected = torch.topk(
                    merged_scores, keep
                )
                running_indices = merged_indices[selected]
            topk_scores[token, : running_scores.numel()] = running_scores
            topk_idxs[token, : running_indices.numel()] = running_indices.to(
                torch.int32
            )

    tensors["topk_scores"][:] = topk_scores
    tensors["topk_idxs"][:] = topk_idxs


def build_tensor_specs(start_pos=None, batch=B):
    tokens = batch * S
    import torch  # type: ignore[import]
    from utils import (
        block_table,
        compressed_slot_mapping,
        csa_decode_start_set,
        int8_quant_per_row,
        kv_seq_lens_from_starts,
        position_ids_from_starts,
        resolve_start_positions,
        token_local_rope,
    )
    from golden import TensorSpec

    starts = resolve_start_positions(
        start_pos,
        batch=batch,
        seq=S,
        max_seq_len=MAX_SEQ_LEN,
        default_fn=lambda: csa_decode_start_set(
            batch=batch,
            seq=S,
            compress_ratio=COMPRESS_RATIO,
            state_block_size=INNER_STATE_BLOCK_SIZE,
            cache_tile=CACHE_TILE,
        ),
    )
    positions = position_ids_from_starts(starts, seq=S)
    kv_seq_lens = kv_seq_lens_from_starts(starts, seq=S)

    state_block_num = batch * INNER_STATE_MAX_BLOCKS
    state_block_table = torch.arange(
        state_block_num - 1, -1, -1, dtype=torch.int32
    ).reshape(batch, INNER_STATE_MAX_BLOCKS)
    ring_rows = positions.to(torch.int64) % INNER_STATE_LEN
    state_pages = torch.gather(
        state_block_table.to(torch.int64),
        1,
        ring_rows // INNER_STATE_BLOCK_SIZE,
    )
    state_slots = (
        state_pages * INNER_STATE_BLOCK_SIZE
        + ring_rows % INNER_STATE_BLOCK_SIZE
    )

    max_candidate_rows = min(
        int((kv_seq_lens.to(torch.int64) // COMPRESS_RATIO).max()),
        TOPK_MAX_CANDIDATES,
    )
    max_request_pages = max(
        1, (max_candidate_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    )
    idx_physical_blocks = batch * max_request_pages
    idx_block_table = block_table(
        batch=batch,
        table_blocks=IDX_MAX_BLOCKS,
        physical_blocks=idx_physical_blocks,
    )
    idx_slots = compressed_slot_mapping(
        positions,
        idx_block_table,
        compress_ratio=COMPRESS_RATIO,
        block_size=BLOCK_SIZE,
    )

    def interleave_rope(rope_cos, rope_sin):
        rope_cos = rope_cos[:, : ROPE_HEAD_DIM // 2].repeat_interleave(
            2, dim=-1
        )
        rope_sin = rope_sin[:, : ROPE_HEAD_DIM // 2].repeat_interleave(
            2, dim=-1
        )
        rope_sign = torch.ones(ROPE_HEAD_DIM, dtype=torch.float32)
        rope_sign[0::2] = -1.0
        return rope_cos, rope_sin * rope_sign

    rope_cos, rope_sin = token_local_rope(
        M,
        COMPRESS_RATIO,
        positions.reshape(-1),
        max_seq_len=MAX_SEQ_LEN,
        dtype=torch.float32,
    )
    rope_cos, rope_sin = interleave_rope(rope_cos, rope_sin)
    cmp_rope_positions = torch.where(
        (positions.to(torch.int64) + 1) % COMPRESS_RATIO == 0,
        positions.to(torch.int64) - (COMPRESS_RATIO - 1),
        torch.zeros_like(positions, dtype=torch.int64),
    )
    cmp_rope_cos, cmp_rope_sin = token_local_rope(
        M,
        COMPRESS_RATIO,
        cmp_rope_positions.reshape(-1),
        max_seq_len=MAX_SEQ_LEN,
        dtype=torch.float32,
    )
    cmp_rope_cos, cmp_rope_sin = interleave_rope(
        cmp_rope_cos, cmp_rope_sin
    )

    def init_x():
        return torch.rand(batch * S, D)
    def init_qr():
        return torch.rand(tokens, Q_LORA)
    # weights_proj / inner-compressor BF16 weight std and RMSNorm gamma mean/std, averaged
    # over DeepSeek-V4-Flash-0731 layers 8/32. idx wq_b uses the MXFP8 grid below.
    def init_weights_proj():
        return torch.randn(D, IDX_N_HEADS) * 0.2218
    def init_cos():
        return rope_cos.clone()
    def init_sin():
        return rope_sin.clone()
    def init_cmp_cos():
        return cmp_rope_cos.clone()
    def init_cmp_sin():
        return cmp_rope_sin.clone()
    def init_hadamard():
        return torch.rand(IDX_HEAD_DIM, IDX_HEAD_DIM) * (IDX_HEAD_DIM ** -0.5)
    def init_inner_compress_state():
        return torch.randn(
            state_block_num,
            INNER_STATE_BLOCK_SIZE,
            INNER_STATE_DIM,
        ) * 0.05
    def init_inner_compress_state_block_table():
        return state_block_table.clone()
    def init_inner_wkv():
        return torch.randn(INNER_OUT_DIM, D) * 0.0270
    def init_inner_wgate():
        return torch.randn(INNER_OUT_DIM, D) * 0.0513
    def init_inner_ape():
        return torch.randn(COMPRESS_RATIO, INNER_OUT_DIM) * 0.1524
    def init_inner_norm_w():
        return 0.6903 + 0.2663 * torch.randn(INNER_HEAD_DIM)
    def init_idx_block_table():
        return idx_block_table.clone()
    def init_position_ids():
        return positions.clone()
    def init_kv_seq_lens():
        return kv_seq_lens.clone()
    def init_inner_state_slot_mapping():
        return state_slots.clone()
    def init_idx_slot_mapping():
        return idx_slots.clone()

    # idx wq_b: simulate the real MXFP8 (e4m3 + 128x128-block E8M0) grid (~200 levels, scaleCV
    # ~0.61, ~1.1% zero spike) instead of a benign randn INT8. gen_shared_weight reduces over
    # the last (in) dim, so build [out, in] then transpose.
    wq_b_i8_T, wq_b_scale = gen_shared_weight(
        (IDX_N_HEADS * IDX_HEAD_DIM, Q_LORA), dequant_std=0.108, chan_cv=0.56)
    wq_b_i8 = wq_b_i8_T.t().contiguous()
    qr_i8, qr_scale = int8_quant_per_row(init_qr())

    # C8 indexer cache fixture: INT8 + scale from one bf16-rounded random draw
    idx_kv_cache_bf16 = torch.rand(
        idx_physical_blocks, BLOCK_SIZE, 1, IDX_HEAD_DIM
    ).to(torch.bfloat16)
    idx_kv_i8, idx_kv_sc = int8_quant_per_row(
        idx_kv_cache_bf16.float().reshape(
            idx_physical_blocks * BLOCK_SIZE, IDX_HEAD_DIM
        )
    )
    idx_kv_i8 = idx_kv_i8.view(
        idx_physical_blocks, BLOCK_SIZE, 1, IDX_HEAD_DIM
    )
    idx_kv_sc = idx_kv_sc.view(
        idx_physical_blocks, BLOCK_SIZE, 1, 1
    )

    return [
        TensorSpec("x", [batch * S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("qr", [tokens, Q_LORA], torch.int8, init_value=lambda: qr_i8),
        TensorSpec("qr_scale", [tokens, 1], torch.float32, init_value=lambda: qr_scale),
        TensorSpec("wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.int8, init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [IDX_N_HEADS * IDX_HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("weights_proj", [D, IDX_N_HEADS], torch.bfloat16, init_value=init_weights_proj),
        TensorSpec("cos", [tokens, ROPE_HEAD_DIM], torch.float32, init_value=init_cos),
        TensorSpec("sin", [tokens, ROPE_HEAD_DIM], torch.float32, init_value=init_sin),
        TensorSpec("cmp_cos", [tokens, ROPE_HEAD_DIM], torch.float32, init_value=init_cmp_cos),
        TensorSpec("cmp_sin", [tokens, ROPE_HEAD_DIM], torch.float32, init_value=init_cmp_sin),
        TensorSpec("hadamard", [IDX_HEAD_DIM, IDX_HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec("inner_kv", [batch * S, INNER_HEAD_DIM], torch.float32),
        TensorSpec("inner_compress_state", [state_block_num, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], torch.float32, init_value=init_inner_compress_state),
        TensorSpec("inner_compress_state_block_table", [batch, INNER_STATE_MAX_BLOCKS], torch.int32, init_value=init_inner_compress_state_block_table),
        TensorSpec("inner_wkv", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wkv),
        TensorSpec("inner_wgate", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wgate),
        TensorSpec("inner_ape", [COMPRESS_RATIO, INNER_OUT_DIM], torch.float32, init_value=init_inner_ape),
        TensorSpec("inner_norm_w", [INNER_HEAD_DIM], torch.bfloat16, init_value=init_inner_norm_w),
        TensorSpec("idx_kv_cache", [idx_physical_blocks, BLOCK_SIZE, 1, IDX_HEAD_DIM], torch.int8, init_value=lambda: idx_kv_i8),
        TensorSpec("idx_kv_scale", [idx_physical_blocks, BLOCK_SIZE, 1, 1], torch.float32, init_value=lambda: idx_kv_sc),
        TensorSpec("idx_block_table", [batch, IDX_MAX_BLOCKS], torch.int32, init_value=init_idx_block_table),
        TensorSpec("topk_scores", [tokens, IDX_TOPK], torch.float32),
        TensorSpec("topk_idxs", [tokens, IDX_TOPK], torch.int32),
        TensorSpec("position_ids", [batch * S], torch.int32, init_value=lambda: init_position_ids().reshape(-1)),
        TensorSpec("idx_slot_mapping", [batch * S], torch.int64, init_value=lambda: init_idx_slot_mapping().reshape(-1)),
        TensorSpec("inner_state_slot_mapping", [batch * S], torch.int64, init_value=lambda: init_inner_state_slot_mapping().reshape(-1)),
        TensorSpec("kv_seq_lens", [batch], torch.int32, init_value=init_kv_seq_lens),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run, topk_pair_compare

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-b", "--batch", type=int, default=B,
                        help=f"runtime request count up to {B} (the compile-time upper bound). "
                             "The batch axes are pl.dynamic, so one compiled program "
                             "serves every value.")
    parser.add_argument("--enable-chip-swimlane", type=int, default=0, choices=[0, 1, 2],
                        help="chip swimlane level: 0=off, 1=AICore timing, 2=+AICPU timing.")
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--start-pos", type=str, default=None,
                        help="Fixture-only start position: one value for a uniform batch or "
                             "a comma-separated value per request.")
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    if args.batch < 1 or args.batch > B:
        parser.error(f"--batch must be in [1, {B}], got {args.batch}")
    start_pos = None
    if args.start_pos is not None:
        try:
            start_values = [int(value) for value in args.start_pos.split(",")]
        except ValueError:
            parser.error(
                f"--start-pos must contain integers, got {args.start_pos!r}"
            )
        start_pos = start_values[0] if len(start_values) == 1 else start_values

    result = run(
        fn=indexer_test,
        specs=build_tensor_specs(start_pos, batch=args.batch),
        golden_fn=golden_indexer,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "topk_scores": ratio_allclose(
                # Scores are diagnostic; sparse attention consumes the selected
                # indices checked below. A3 reduction order may perturb one
                # root score per query without changing the selected set.
                atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001
            ),
            "topk_idxs": topk_pair_compare("topk_scores"),
            # C8 cache: history is exact; only the <=B boundary rows the compressor rewrote may
            # differ by +/-1 LSB from the bf16 round of a fresh position.
            "idx_kv_cache": ratio_allclose(atol=1, rtol=0, max_error_ratio=0.01),
            "idx_kv_scale": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
