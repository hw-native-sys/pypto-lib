# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 packed prefill indexer: compressed index KV cache + per-token compressed top-k."""

import pypto.language as pl

from config import (
    FLASH as M,
    BLOCK_SIZE,
    CSA_INNER_STATE_PHYSICAL_BLOCKS,
    FP32_NEG_INF,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
    PREFILL_BATCH,
    PREFILL_SEQ,
)

from prefill_indexer_compressor import (
    INNER_STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE,
    INNER_STATE_MAX_BLOCKS,
    STATE_LEN as INNER_STATE_LEN,
    golden_prefill_indexer_compressor,
    prefill_indexer_compressor,
)


PREFILL_MAX_TOKENS = 8192

# Query tile for projection and selection.
PREFILL_DENSE_TILE = 128

# Dynamic shape variables.
T_DYN = pl.dynamic("PREFILL_INDEXER_T_DYN")
# Context-parallel local-query token axis.
Q_T_DYN = pl.dynamic("PREFILL_INDEXER_Q_T_DYN")
IDX_BLOCK_NUM_DYN = pl.dynamic("PREFILL_IDX_BLOCK_NUM_DYN")
INNER_STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_INNER_STATE_BLOCK_NUM_DYN")

# model config
D = M.hidden_size
ROPE_HEAD_DIM = M.qk_rope_head_dim
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_NOPE_HEAD_DIM = M.index_nope_head_dim
Q_LORA = M.q_lora_rank
WEIGHTS_SCALE = M.index_weights_scale
MAX_SEQ_LEN = M.max_position_embeddings
WIN = M.sliding_window

COMPRESS_RATIO = 4  # the indexer only runs on ratio-4 layers
IDX_TOPK = M.index_topk
INNER_OVERLAP = COMPRESS_RATIO == 4
INNER_COFF = 1 + int(INNER_OVERLAP)
INNER_HEAD_DIM = IDX_HEAD_DIM
INNER_OUT_DIM = INNER_COFF * INNER_HEAD_DIM
INNER_COMPRESS_STATE_DIM = 2 * INNER_OUT_DIM
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
START_POS = 0
MAX_CMP_WRITES = PREFILL_MAX_TOKENS // COMPRESS_RATIO
INDEXER_MAX_CANDIDATES = MAX_SEQ_LEN // COMPRESS_RATIO
TOPK_PAIR_WIDTH = 2 * IDX_TOPK

# paged indexer cache
IDX_CACHE_MAX_BLOCKS = (MAX_SEQ_LEN // COMPRESS_RATIO + BLOCK_SIZE - 1) // BLOCK_SIZE
IDX_CACHE_BLOCK_NUM = IDX_CACHE_MAX_BLOCKS

# tiling
Q_TILE = 128
Q_OUT_TILE = 256
QR_PROJ_MM_ROW_TILE = min(128, T)  # A2/A3 L0C-bounded qr-projection row tile
QR_PROJ_TAIL_ROW_TILE = 16
QR_PROJ_ROW_TILE = 16
HEAD_DIM_TILE = 32
D_TILE = 32
WEIGHTS_ROW_TILE = 32
QH_QUANT_ROW_TILE = 64
ROPE_ROW_TILE = IDX_N_HEADS  # one token owns IDX_N_HEADS contiguous q rows + one cos/sin
QH_MM_TILE = 64
WEIGHTS_TAIL_ROW_TILE = 16
TOPK_LEAF_TILE = 8192
TOPK_GROUP_TILE = 2
TOPK_MAX_LEAVES = (INDEXER_MAX_CANDIDATES + TOPK_LEAF_TILE - 1) // TOPK_LEAF_TILE
TOPK_GROUPS_PER_QUERY = (TOPK_MAX_LEAVES + TOPK_GROUP_TILE - 1) // TOPK_GROUP_TILE
TOPK_GROUP_WORKERS = 48
TOPK_GROUP_ROOT_ROWS = PREFILL_DENSE_TILE * TOPK_GROUPS_PER_QUERY
TOPK_GROUP_SCRATCH_ROWS = TOPK_GROUP_WORKERS * TOPK_GROUP_TILE
TOPK_ARENA_ROWS = TOPK_GROUP_ROOT_ROWS + TOPK_GROUP_SCRATCH_ROWS
TOPK_SCORE_WORKERS = 24

# Four pairwise merge levels cover at most 16 roots.
assert TOPK_GROUPS_PER_QUERY <= 16


@pl.jit.inline
def _merge2_top512_pairs(
    pair_arena: pl.Tensor,
    left_slot: pl.Scalar[pl.INDEX],
    right_slot: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
) -> None:
    left = pl.load(pair_arena, [left_slot, 0], [1, TOPK_PAIR_WIDTH])
    right = pl.load(pair_arena, [right_slot, 0], [1, TOPK_PAIR_WIDTH])
    merge_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
    merged_all = pl.tile.mrgsort(left, right, tmp=merge_tmp)
    merged = pl.tile.slice(merged_all, [1, TOPK_PAIR_WIDTH], [0, 0])
    pl.store(merged, [output_slot, 0], pair_arena)


@pl.jit.inline
def _merge_topk_level_pairs(
    pair_arena: pl.Tensor,
    arena_base: pl.Scalar[pl.INDEX],
    input_count: pl.Scalar[pl.INDEX],
    input_base: pl.Scalar[pl.INDEX],
    output_base: pl.Scalar[pl.INDEX],
) -> None:
    output_count = (input_count + 1) // 2
    for output in pl.range(output_count):
        left_slot = arena_base + input_base + 2 * output
        right_slot = left_slot + 1
        output_slot = arena_base + output_base + output
        if right_slot < arena_base + input_base + input_count:
            _merge2_top512_pairs(pair_arena, left_slot, right_slot, output_slot)
        else:
            forwarded = pl.load(pair_arena, [left_slot, 0], [1, TOPK_PAIR_WIDTH])
            pl.store(forwarded, [output_slot, 0], pair_arena)


@pl.jit.inline
def _topk_leaf(
    score_arena: pl.Tensor,
    pair_arena: pl.Tensor,
    query: pl.Scalar[pl.INDEX],
    logical_begin: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
) -> None:
    logical_begin_i32 = pl.cast(logical_begin, pl.INT32)
    leaf_index_ramp = pl.tile.arange(0, [1, TOPK_LEAF_TILE], dtype=pl.INT32)
    leaf_indices = pl.add(leaf_index_ramp, logical_begin_i32)
    leaf_scores_raw = pl.load(score_arena, [query, logical_begin], [1, TOPK_LEAF_TILE], valid_shape=[1, valid_count])
    leaf_scores = pl.tile.fillpad(leaf_scores_raw, pad_value=pl.PadValue.min)
    leaf_floor = pl.tile.full([1, TOPK_LEAF_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
    leaf_scores = pl.maximum(leaf_scores, leaf_floor)
    pairs = pl.tile.sort32(leaf_scores, pl.reinterpret_view(leaf_indices, pl.UINT32))
    pairs = pl.tile.mrgsort(pairs, block_len=64)
    pairs = pl.tile.mrgsort(pairs, block_len=256)
    pairs = pl.tile.mrgsort(pairs, block_len=1024)
    pairs = pl.tile.mrgsort(pairs, block_len=4096)
    top_pairs = pl.tile.slice(pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
    pl.store(top_pairs, [output_slot, 0], pair_arena)


@pl.jit.incore
def _topk_group_wave(
    position_ids: pl.Tensor,
    score_arena: pl.Tensor,
    pair_arena: pl.Tensor,
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
):
    """Reduce striped two-leaf subtrees into compact roots."""
    worker = pl.tile.get_block_idx()
    global_group_base = 0
    for query in pl.range(tile_rows):
        position = pl.read(position_ids, [tile_base + query])
        visible_count = pl.max(pl.min((position + 1) // COMPRESS_RATIO, INDEXER_MAX_CANDIDATES), 0)
        leaf_count = (visible_count + TOPK_LEAF_TILE - 1) // TOPK_LEAF_TILE
        group_count = (leaf_count + TOPK_GROUP_TILE - 1) // TOPK_GROUP_TILE
        base_mod = global_group_base % TOPK_GROUP_WORKERS
        first_group = (worker + base_mod) % TOPK_GROUP_WORKERS
        for group in pl.range(first_group, group_count, TOPK_GROUP_WORKERS):
            leaf_begin = group * TOPK_GROUP_TILE
            group_leaf_count = pl.min(TOPK_GROUP_TILE, leaf_count - leaf_begin)
            group_root_slot = query * TOPK_GROUPS_PER_QUERY + group
            if group_leaf_count == 1:
                logical_begin = leaf_begin * TOPK_LEAF_TILE
                valid_count = pl.min(TOPK_LEAF_TILE, visible_count - logical_begin)
                _topk_leaf(
                    score_arena, pair_arena,
                    query, logical_begin, valid_count,
                    group_root_slot,
                )
            else:
                scratch_base = TOPK_GROUP_ROOT_ROWS + worker * TOPK_GROUP_TILE
                for group_leaf in pl.unroll(TOPK_GROUP_TILE):
                    leaf = leaf_begin + group_leaf
                    logical_begin = leaf * TOPK_LEAF_TILE
                    valid_count = pl.min(TOPK_LEAF_TILE, visible_count - logical_begin)
                    _topk_leaf(
                        score_arena, pair_arena,
                        query, logical_begin, valid_count,
                        scratch_base + group_leaf,
                    )
                _merge2_top512_pairs(pair_arena, scratch_base, scratch_base + 1, group_root_slot)
        global_group_base = global_group_base + group_count


@pl.jit.incore
def _topk_query_merge(
    position_ids: pl.Tensor,
    pair_arena: pl.Tensor,
    topk_indices: pl.Tensor,
    tile_base: pl.Scalar[pl.INDEX],
):
    """Merge compact group roots into one Top-512 row."""
    query = pl.tile.get_block_idx()
    output_query = tile_base + query
    position = pl.read(position_ids, [output_query])
    visible_count = pl.max(pl.min((position + 1) // COMPRESS_RATIO, INDEXER_MAX_CANDIDATES), 0)
    empty_indices = pl.tile.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
    pl.store(empty_indices, [output_query, 0], topk_indices)

    if visible_count > 0:
        leaf_count = (visible_count + TOPK_LEAF_TILE - 1) // TOPK_LEAF_TILE
        group_count = (leaf_count + TOPK_GROUP_TILE - 1) // TOPK_GROUP_TILE
        arena_base = query * TOPK_GROUPS_PER_QUERY
        if group_count > 1:
            level1_count = (group_count + 1) // 2
            _merge_topk_level_pairs(pair_arena, arena_base, group_count, 0, 0)
            if level1_count > 1:
                level2_count = (level1_count + 1) // 2
                _merge_topk_level_pairs(pair_arena, arena_base, level1_count, 0, 0)
                if level2_count > 1:
                    level3_count = (level2_count + 1) // 2
                    _merge_topk_level_pairs(pair_arena, arena_base, level2_count, 0, 0)
                    if level3_count > 1:
                        _merge_topk_level_pairs(pair_arena, arena_base, level3_count, 0, 0)

        root_pairs = pl.load(pair_arena, [arena_base, 0], [1, TOPK_PAIR_WIDTH])
        root_indices = pl.tile.gather_mask(root_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        output_indices = pl.tile.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
        valid_topk = pl.min(visible_count, IDX_TOPK)
        for lane in pl.range(valid_topk):
            output_index = pl.tile.read(root_indices, [0, lane])
            pl.tile.write(output_indices, [0, lane], output_index)
        pl.store(output_indices, [output_query, 0], topk_indices)


@pl.jit.inline(auto_scope=False)
def _prefill_indexer_score_topk(
    qr_hadamard_i8: pl.Tensor,
    qr_hadamard_scale_dq: pl.Tensor,
    weights: pl.Tensor,
    idx_kv_cache: pl.Tensor,
    idx_kv_scale: pl.Tensor,
    idx_block_table: pl.Tensor,
    position_ids: pl.Tensor,
    topk_indices: pl.Tensor,
    score_arena: pl.Tensor,
    pair_arena: pl.Tensor,
    completion: pl.Array[1, pl.TASK_ID],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
):
    """Score one query tile and reduce its Top-K forest."""
    idx_block_num = pl.tensor.dim(idx_kv_cache, 0)
    idx_cache_rows = idx_block_num * BLOCK_SIZE
    kv_cache_i8_flat = pl.reshape(idx_kv_cache, [idx_cache_rows, IDX_HEAD_DIM])
    kv_scale_flat = pl.reshape(idx_kv_scale, [idx_cache_rows, 1])
    with pl.spmd(
        TOPK_SCORE_WORKERS, name_hint="prefill_idx_score_leaf_wave", deps=[completion[0]],
        optimizations=[pl.split(pl.SplitMode.NONE, slot_num=2)],
    ) as score_tid:
        worker = pl.tile.get_block_idx()
        global_leaf_base = 0
        for query in pl.range(tile_rows):
            output_query = tile_base + query
            position = pl.read(position_ids, [output_query])
            visible_count = pl.max(pl.min((position + 1) // COMPRESS_RATIO, INDEXER_MAX_CANDIDATES), 0)
            leaf_count = (visible_count + TOPK_LEAF_TILE - 1) // TOPK_LEAF_TILE
            base_mod = global_leaf_base % TOPK_SCORE_WORKERS
            first_leaf = (worker + base_mod) % TOPK_SCORE_WORKERS
            for leaf in pl.range(first_leaf, leaf_count, TOPK_SCORE_WORKERS):
                logical_begin = leaf * TOPK_LEAF_TILE
                valid_count = pl.min(TOPK_LEAF_TILE, visible_count - logical_begin)
                query_head_begin = query * IDX_N_HEADS
                query_vector = qr_hadamard_i8[query_head_begin : query_head_begin + IDX_N_HEADS, 0:IDX_HEAD_DIM]
                query_scale_heads = qr_hadamard_scale_dq[query_head_begin : query_head_begin + IDX_N_HEADS, 0:1]
                query_scale = pl.reshape(query_scale_heads, [1, IDX_N_HEADS])
                query_weight = weights[query : query + 1, 0:IDX_N_HEADS]
                for page in pl.pipeline(0, (valid_count + BLOCK_SIZE - 1) // BLOCK_SIZE, stage=2):
                    page_begin = page * BLOCK_SIZE
                    logical_row = logical_begin + page_begin
                    logical_page = logical_row // BLOCK_SIZE
                    physical_block_raw = pl.read(idx_block_table, [logical_page])
                    score_valid = pl.full([1, BLOCK_SIZE], dtype=pl.FP32, value=FP32_NEG_INF)
                    if physical_block_raw >= 0 and physical_block_raw < idx_block_num:
                        physical_block = pl.cast(physical_block_raw, pl.INDEX)
                        physical_row = physical_block * BLOCK_SIZE
                        kv_i8 = kv_cache_i8_flat[physical_row : physical_row + BLOCK_SIZE, 0:IDX_HEAD_DIM]
                        score_i32 = pl.matmul(kv_i8, query_vector, out_dtype=pl.INT32, b_trans=True)
                        score_fp32 = pl.cast(score_i32, target_type=pl.FP32, mode="none")
                        score_fp32 = pl.col_expand_mul(score_fp32, query_scale)
                        score_fp32 = pl.maximum(score_fp32, 0.0)
                        score_fp32 = pl.col_expand_mul(score_fp32, query_weight)
                        kv_scale = kv_scale_flat[physical_row : physical_row + BLOCK_SIZE, 0:1]
                        score_sum = pl.row_sum(score_fp32)
                        score_scaled = pl.mul(score_sum, kv_scale)
                        score_row = pl.reshape(score_scaled, [1, BLOCK_SIZE])
                        valid_rows = pl.min(BLOCK_SIZE, valid_count - page_begin)
                        score_valid_view = pl.set_validshape(score_row, 1, valid_rows)
                        score_padded = pl.fillpad(score_valid_view, pad_value=pl.PadValue.min)
                        score_floor = pl.full([1, BLOCK_SIZE], dtype=pl.FP32, value=FP32_NEG_INF)
                        score_valid = pl.maximum(score_padded, score_floor)
                    score_arena[query : query + 1, logical_row : logical_row + BLOCK_SIZE] = score_valid
            global_leaf_base = global_leaf_base + leaf_count

    with pl.spmd(TOPK_GROUP_WORKERS, name_hint="prefill_idx_topk_group_wave", deps=[score_tid]) as topk_tid:
        _topk_group_wave(position_ids, score_arena, pair_arena, tile_base, tile_rows)

    with pl.spmd(tile_rows, name_hint="prefill_idx_topk_query_merge", deps=[topk_tid]) as merge_tid:
        _topk_query_merge(position_ids, pair_arena, topk_indices, tile_base)

    completion[0] = merge_tid
    return topk_indices


@pl.jit.inline(auto_scope=False)
def _prefill_indexer_dense_tile(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale: pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    cmp_topk_indices: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32]],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    rope_dup_idx_template: pl.Tensor[[1, ROPE_HEAD_DIM], pl.INT32],
    rope_swap_idx_template: pl.Tensor[[ROPE_ROW_TILE, ROPE_HEAD_DIM], pl.INT32],
    rope_sign_template: pl.Tensor[[1, ROPE_HEAD_DIM], pl.FP32],
    score_arena: pl.Tensor[[PREFILL_DENSE_TILE, INDEXER_MAX_CANDIDATES], pl.FP32],
    pair_arena: pl.Tensor[[TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], pl.FP32],
    selection_completion: pl.Array[1, pl.TASK_ID],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
):
    """Project and select one bounded query tile."""
    t_dim = pl.tensor.dim(x, 0)
    x_view = pl.reshape(x, [t_dim, D])
    qr_view = pl.reshape(qr, [t_dim, Q_LORA])
    qr_scale_view = pl.reshape(qr_scale, [t_dim, 1])

    # Bounded qr-projection scratch with an M16 tail.
    qr_proj = pl.create_tensor([PREFILL_DENSE_TILE, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.FP32)
    qr_full_rows = (tile_rows // QR_PROJ_MM_ROW_TILE) * QR_PROJ_MM_ROW_TILE
    qr_padded_rows = (
        (tile_rows + QR_PROJ_TAIL_ROW_TILE - 1) // QR_PROJ_TAIL_ROW_TILE
    ) * QR_PROJ_TAIL_ROW_TILE
    qr_full_row_blocks = PREFILL_DENSE_TILE // QR_PROJ_MM_ROW_TILE
    if qr_full_rows > 0:
        for idx in pl.spmd(
            (IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE) * qr_full_row_blocks,
            name_hint="prefill_idx_qr_proj_full",
        ):
            qr_n = idx // qr_full_row_blocks
            qr_row_block = idx - qr_n * qr_full_row_blocks
            o0 = qr_n * Q_OUT_TILE
            local_t0 = qr_row_block * QR_PROJ_MM_ROW_TILE
            if local_t0 < qr_full_rows:
                global_t0 = tile_base + local_t0
                qr_acc = pl.create_tensor([QR_PROJ_MM_ROW_TILE, Q_OUT_TILE], dtype=pl.INT32)
                for kb in pl.pipeline(0, Q_LORA // Q_TILE, stage=2):
                    q0 = kb * Q_TILE
                    qr_tile = qr_view[
                        global_t0 : global_t0 + QR_PROJ_MM_ROW_TILE,
                        q0 : q0 + Q_TILE,
                    ]
                    wq_tile = wq_b[q0 : q0 + Q_TILE, o0 : o0 + Q_OUT_TILE]
                    if q0 == 0:
                        qr_acc = pl.matmul(qr_tile, wq_tile, out_dtype=pl.INT32)
                    else:
                        qr_acc = pl.matmul_acc(qr_acc, qr_tile, wq_tile)
                wq_scale = pl.reshape(wq_b_scale[o0 : o0 + Q_OUT_TILE], [1, Q_OUT_TILE])
                for rl in pl.range(0, QR_PROJ_MM_ROW_TILE, QR_PROJ_ROW_TILE):
                    acc_fp32 = pl.cast(
                        qr_acc[rl : rl + QR_PROJ_ROW_TILE, :],
                        target_type=pl.FP32,
                        mode="none",
                    )
                    scale_dq = qr_scale_view[
                        global_t0 + rl : global_t0 + rl + QR_PROJ_ROW_TILE,
                        :,
                    ]
                    qr_dequant = pl.col_expand_mul(
                        pl.row_expand_mul(acc_fp32, scale_dq),
                        wq_scale,
                    )
                    qr_proj[
                        local_t0 + rl : local_t0 + rl + QR_PROJ_ROW_TILE,
                        o0 : o0 + Q_OUT_TILE,
                    ] = qr_dequant

    if qr_full_rows < qr_padded_rows:
        qr_tail_row_blocks = QR_PROJ_MM_ROW_TILE // QR_PROJ_TAIL_ROW_TILE
        for tail_idx in pl.spmd(
            (IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE) * qr_tail_row_blocks,
            name_hint="prefill_idx_qr_proj_tail",
        ):
            tail_n = tail_idx // qr_tail_row_blocks
            tail_row_block = tail_idx - tail_n * qr_tail_row_blocks
            tail_o0 = tail_n * Q_OUT_TILE
            tail_t0 = qr_full_rows + tail_row_block * QR_PROJ_TAIL_ROW_TILE
            if tail_t0 < qr_padded_rows:
                tail_valid_rows = pl.min(QR_PROJ_TAIL_ROW_TILE, tile_rows - tail_t0)
                tail_global_t0 = tile_base + tail_t0
                qr_tail_acc = pl.create_tensor([QR_PROJ_TAIL_ROW_TILE, Q_OUT_TILE], dtype=pl.INT32)
                for tail_kb in pl.pipeline(0, Q_LORA // Q_TILE, stage=2):
                    tail_q0 = tail_kb * Q_TILE
                    qr_tail = pl.slice(
                        qr_view,
                        [QR_PROJ_TAIL_ROW_TILE, Q_TILE],
                        [tail_global_t0, tail_q0],
                        valid_shape=[tail_valid_rows, Q_TILE],
                    )
                    wq_tail = wq_b[
                        tail_q0 : tail_q0 + Q_TILE,
                        tail_o0 : tail_o0 + Q_OUT_TILE,
                    ]
                    if tail_q0 == 0:
                        qr_tail_acc = pl.matmul(qr_tail, wq_tail, out_dtype=pl.INT32)
                    else:
                        qr_tail_acc = pl.matmul_acc(qr_tail_acc, qr_tail, wq_tail)
                tail_acc_fp32 = pl.cast(qr_tail_acc, target_type=pl.FP32, mode="none")
                tail_scale = pl.slice(
                    qr_scale_view,
                    [QR_PROJ_TAIL_ROW_TILE, 1],
                    [tail_global_t0, 0],
                    valid_shape=[tail_valid_rows, 1],
                )
                tail_wq_scale = pl.reshape(
                    wq_b_scale[tail_o0 : tail_o0 + Q_OUT_TILE],
                    [1, Q_OUT_TILE],
                )
                qr_tail_dequant = pl.col_expand_mul(
                    pl.row_expand_mul(tail_acc_fp32, tail_scale),
                    tail_wq_scale,
                )
                qr_proj[
                    tail_t0 : tail_t0 + QR_PROJ_TAIL_ROW_TILE,
                    tail_o0 : tail_o0 + Q_OUT_TILE,
                ] = qr_tail_dequant

    # Query RoPE, Hadamard, and per-row INT8 quantization.
    qr_proj_flat = pl.reshape(qr_proj, [PREFILL_DENSE_TILE * IDX_N_HEADS, IDX_HEAD_DIM])
    qr_bf16 = pl.create_tensor([PREFILL_DENSE_TILE * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.BF16)
    for token_idx in pl.spmd(tile_rows, name_hint="prefill_idx_qr_rope", allow_early_resolve=True):
        rope_global_t = tile_base + token_idx
        r0 = token_idx * ROPE_ROW_TILE
        qr_nope_fp32 = qr_proj_flat[r0 : r0 + ROPE_ROW_TILE, 0:IDX_NOPE_HEAD_DIM]
        qr_nope = pl.cast(qr_nope_fp32, target_type=pl.BF16, mode="rint")
        qr_rope = qr_proj_flat[r0 : r0 + ROPE_ROW_TILE, IDX_NOPE_HEAD_DIM:IDX_HEAD_DIM]
        qr_swapped = pl.gather(qr_rope, dim=-1, index=rope_swap_idx_template)
        cos_il = pl.gather(
            cos[rope_global_t : rope_global_t + 1, 0 : ROPE_HEAD_DIM // 2],
            dim=-1,
            index=rope_dup_idx_template,
        )
        sin_il = pl.gather(
            sin[rope_global_t : rope_global_t + 1, 0 : ROPE_HEAD_DIM // 2],
            dim=-1,
            index=rope_dup_idx_template,
        )
        rope_rot = pl.add(
            pl.col_expand_mul(qr_rope, cos_il),
            pl.col_expand_mul(qr_swapped, pl.mul(sin_il, rope_sign_template)),
        )
        rope_bf16 = pl.cast(rope_rot, target_type=pl.BF16, mode="rint")
        qr_bf16[r0 : r0 + ROPE_ROW_TILE, :] = pl.concat(qr_nope, rope_bf16)

    qh_acc_gm = pl.create_tensor([PREFILL_DENSE_TILE * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.FP32)
    for mm_idx in pl.spmd(tile_rows, name_hint="prefill_idx_qr_hadamard", allow_early_resolve=True):
        r0 = mm_idx * QH_MM_TILE
        qh_acc = pl.matmul(
            qr_bf16[r0 : r0 + QH_MM_TILE, :],
            hadamard,
            out_dtype=pl.FP32,
        )
        qh_acc_gm[r0 : r0 + QH_MM_TILE, :] = qh_acc

    qr_hadamard_i8 = pl.create_tensor(
        [PREFILL_DENSE_TILE * IDX_N_HEADS, IDX_HEAD_DIM],
        dtype=pl.INT8,
    )
    qr_hadamard_scale_dq = pl.create_tensor(
        [PREFILL_DENSE_TILE * IDX_N_HEADS, 1],
        dtype=pl.FP32,
    )
    for quant_idx in pl.spmd(tile_rows, name_hint="prefill_idx_qr_quant", allow_early_resolve=True):
        r0 = quant_idx * QH_QUANT_ROW_TILE
        qh_amax = pl.full([1, QH_QUANT_ROW_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for h0 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_TILE):
            qh_tile = qh_acc_gm[
                r0 : r0 + QH_QUANT_ROW_TILE,
                h0 : h0 + HEAD_DIM_TILE,
            ]
            qh_abs = pl.maximum(qh_tile, pl.neg(qh_tile))
            qh_amax = pl.maximum(
                qh_amax,
                pl.reshape(pl.row_max(qh_abs), [1, QH_QUANT_ROW_TILE]),
            )
        scale_quant_row = pl.div(
            pl.full([1, QH_QUANT_ROW_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX),
            qh_amax,
        )
        qr_hadamard_scale_dq[r0 : r0 + QH_QUANT_ROW_TILE, :] = pl.reshape(
            pl.recip(scale_quant_row),
            [QH_QUANT_ROW_TILE, 1],
        )
        scale_quant = pl.reshape(scale_quant_row, [QH_QUANT_ROW_TILE, 1])
        for h1 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_TILE):
            qh_quant_tile = qh_acc_gm[
                r0 : r0 + QH_QUANT_ROW_TILE,
                h1 : h1 + HEAD_DIM_TILE,
            ]
            qh_scaled = pl.row_expand_mul(qh_quant_tile, scale_quant)
            qh_i32 = pl.cast(qh_scaled, target_type=pl.INT32, mode="rint")
            qh_half = pl.cast(qh_i32, target_type=pl.FP16, mode="round")
            qr_hadamard_i8[
                r0 : r0 + QH_QUANT_ROW_TILE,
                h1 : h1 + HEAD_DIM_TILE,
            ] = pl.cast(qh_half, target_type=pl.INT8, mode="trunc")

    # Per-head weight projection with an M16 tail.
    weights = pl.create_tensor([PREFILL_DENSE_TILE, IDX_N_HEADS], dtype=pl.FP32)
    weights_full_rows = (tile_rows // WEIGHTS_ROW_TILE) * WEIGHTS_ROW_TILE
    weights_padded_rows = (
        (tile_rows + WEIGHTS_TAIL_ROW_TILE - 1) // WEIGHTS_TAIL_ROW_TILE
    ) * WEIGHTS_TAIL_ROW_TILE
    for weights_idx in pl.spmd(
        PREFILL_DENSE_TILE // WEIGHTS_ROW_TILE,
        name_hint="prefill_idx_weights_proj_full",
    ):
        weights_t0 = weights_idx * WEIGHTS_ROW_TILE
        if weights_t0 < weights_full_rows:
            weights_global_t0 = tile_base + weights_t0
            weights_acc = pl.create_tensor([WEIGHTS_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32)
            for db in pl.pipeline(0, D // D_TILE, stage=2):
                d0 = db * D_TILE
                x_tile = x_view[
                    weights_global_t0 : weights_global_t0 + WEIGHTS_ROW_TILE,
                    d0 : d0 + D_TILE,
                ]
                wp_tile = weights_proj[d0 : d0 + D_TILE, :]
                if d0 == 0:
                    weights_acc = pl.matmul(x_tile, wp_tile, out_dtype=pl.FP32)
                else:
                    weights_acc = pl.matmul_acc(weights_acc, x_tile, wp_tile)
            weights[weights_t0 : weights_t0 + WEIGHTS_ROW_TILE, :] = pl.mul(
                weights_acc,
                WEIGHTS_SCALE,
            )

    if weights_full_rows < weights_padded_rows:
        for weights_tail_idx in pl.spmd(
            WEIGHTS_ROW_TILE // WEIGHTS_TAIL_ROW_TILE,
            name_hint="prefill_idx_weights_proj_tail",
        ):
            weights_tail_t0 = weights_full_rows + weights_tail_idx * WEIGHTS_TAIL_ROW_TILE
            if weights_tail_t0 < weights_padded_rows:
                weights_tail_valid = pl.min(
                    WEIGHTS_TAIL_ROW_TILE,
                    tile_rows - weights_tail_t0,
                )
                weights_tail_global_t0 = tile_base + weights_tail_t0
                weights_tail_acc = pl.create_tensor(
                    [WEIGHTS_TAIL_ROW_TILE, IDX_N_HEADS],
                    dtype=pl.FP32,
                )
                for tail_db in pl.pipeline(0, D // D_TILE, stage=2):
                    tail_d0 = tail_db * D_TILE
                    x_tail = pl.slice(
                        x_view,
                        [WEIGHTS_TAIL_ROW_TILE, D_TILE],
                        [weights_tail_global_t0, tail_d0],
                        valid_shape=[weights_tail_valid, D_TILE],
                    )
                    wp_tail = weights_proj[tail_d0 : tail_d0 + D_TILE, :]
                    if tail_d0 == 0:
                        weights_tail_acc = pl.matmul(x_tail, wp_tail, out_dtype=pl.FP32)
                    else:
                        weights_tail_acc = pl.matmul_acc(weights_tail_acc, x_tail, wp_tail)
                weights[
                    weights_tail_t0 : weights_tail_t0 + WEIGHTS_TAIL_ROW_TILE,
                    :,
                ] = pl.mul(weights_tail_acc, WEIGHTS_SCALE)

    _prefill_indexer_score_topk(
        qr_hadamard_i8, qr_hadamard_scale_dq, weights,
        idx_kv_cache, idx_kv_scale, idx_block_table,
        position_ids,
        cmp_topk_indices,
        score_arena, pair_arena, selection_completion,
        tile_base, tile_rows,
    )


@pl.jit.inline(auto_scope=False)
def prefill_indexer(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.InOut[
        pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM], pl.FP32]
    ],
    inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    cmp_topk_indices: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32]],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    idx_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
):
    """Compress and score one contiguous position-monotonic token run."""
    compressor_completion = pl.array.create(1, pl.TASK_ID)
    prefill_indexer_compressor(
        x,
        inner_compress_state, inner_compress_state_block_table,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        cmp_freqs_cos, cmp_freqs_sin,
        hadamard,
        idx_kv_cache, idx_kv_scale,
        position_ids, idx_slot_mapping, inner_state_slot_mapping,
        compressor_completion,
    )

    cmp_topk_indices = prefill_indexer_query(
        x, qr, qr_scale,
        wq_b, wq_b_scale, weights_proj,
        cos, sin,
        hadamard,
        idx_kv_cache, idx_kv_scale, idx_block_table,
        cmp_topk_indices,
        position_ids,
        compressor_completion,
    )
    return idx_kv_cache, idx_kv_scale, cmp_topk_indices


@pl.jit.inline(auto_scope=False)
def prefill_indexer_query(
    x: pl.Tensor[[Q_T_DYN, D], pl.BF16],
    qr: pl.Tensor[[Q_T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[Q_T_DYN, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[Q_T_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[Q_T_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale: pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    cmp_topk_indices: pl.Out[pl.Tensor[[Q_T_DYN, IDX_TOPK], pl.INT32]],
    position_ids: pl.Tensor[[Q_T_DYN], pl.INT32],
    completion: pl.Array[1, pl.TASK_ID],
):
    """Score local queries against the published indexer cache."""
    t_dim = pl.tensor.dim(x, 0)

    # Invocation-wide RoPE templates.
    rope_dup_idx_template = pl.create_tensor([1, ROPE_HEAD_DIM], dtype=pl.INT32)
    rope_swap_idx_template = pl.create_tensor([ROPE_ROW_TILE, ROPE_HEAD_DIM], dtype=pl.INT32)
    rope_sign_template = pl.create_tensor([1, ROPE_HEAD_DIM], dtype=pl.FP32)
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_idx_rope_index_prepare",
        deps=[completion[0]],
    ) as rope_prepare_tid:
        for rope_c in pl.range(ROPE_HEAD_DIM):
            rope_lane = rope_c % 2
            pl.write(rope_dup_idx_template, [0, rope_c], pl.cast(rope_c // 2, pl.INT32))
            pl.write(rope_sign_template, [0, rope_c], pl.cast(pl.cast(rope_lane * 2 - 1, pl.INT32), pl.FP32))
            for rope_r in pl.range(ROPE_ROW_TILE):
                pl.write(rope_swap_idx_template, [rope_r, rope_c], pl.cast(rope_c + 1 - rope_lane * 2, pl.INT32))

    # Reusable score and pair arenas.
    score_arena = pl.create_tensor([PREFILL_DENSE_TILE, INDEXER_MAX_CANDIDATES], dtype=pl.FP32, manual_dep=True)
    pair_arena = pl.create_tensor([TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], dtype=pl.FP32, manual_dep=True)
    selection_completion = pl.array.create(1, pl.TASK_ID)
    selection_completion[0] = rope_prepare_tid

    for tile_base in pl.range(0, t_dim, PREFILL_DENSE_TILE):
        with pl.scope():
            tile_rows = pl.min(PREFILL_DENSE_TILE, t_dim - tile_base)
            _prefill_indexer_dense_tile(
                x, qr, qr_scale,
                wq_b, wq_b_scale, weights_proj,
                cos, sin,
                hadamard,
                idx_kv_cache, idx_kv_scale, idx_block_table,
                cmp_topk_indices, position_ids,
                rope_dup_idx_template, rope_swap_idx_template, rope_sign_template,
                score_arena, pair_arena, selection_completion,
                tile_base, tile_rows,
            )

    return cmp_topk_indices


def topk_prefix_contract_error(topk_indices, position_ids, num_tokens=None):
    """Return an error string if the top-k prefix contract is broken."""
    import torch

    physical_tokens = int(topk_indices.shape[0])
    if num_tokens is None:
        num_tokens = physical_tokens
    elif hasattr(num_tokens, "item"):
        num_tokens = num_tokens.item()
    num_tokens = int(num_tokens)
    for t in range(physical_tokens):
        row = topk_indices[t]
        if t >= num_tokens:
            non_padding = int((row != -1).count_nonzero().item())
            if non_padding:
                return f"inactive top-k row {t} contains {non_padding} non--1 entries"
            continue
        visible_candidates = min(
            int((int(position_ids[t].item()) + 1) // COMPRESS_RATIO),
            INDEXER_MAX_CANDIDATES,
        )
        selected = min(IDX_TOPK, visible_candidates)
        prefix = row[:selected]
        if selected:
            out_of_range = int(((prefix < 0) | (prefix >= visible_candidates)).count_nonzero().item())
            if out_of_range:
                return (
                    f"top-k row {t} has {out_of_range} entries outside "
                    f"[0, {visible_candidates}) in its selected prefix"
                )
            unique_count = int(torch.unique(prefix).numel())
            if unique_count != selected:
                return f"top-k row {t} selected prefix has {unique_count}/{selected} unique entries"
        tail_non_padding = int((row[selected:] != -1).count_nonzero().item())
        if tail_non_padding:
            return f"top-k row {t} tail contains {tail_non_padding} non--1 entries"
    return None


def golden_prefill_indexer_core(tensors):
    from utils import int8_quant_per_row
    import torch

    token_count = int(tensors["x"].shape[0])
    compressor_tensors = {
        "x": tensors["x"],
        "kv": torch.zeros(
            max(1, (token_count + COMPRESS_RATIO - 1) // COMPRESS_RATIO),
            IDX_HEAD_DIM,
            dtype=torch.int8,
        ),
        "compress_state": tensors["inner_compress_state"],
        "inner_compress_state_block_table": tensors["inner_compress_state_block_table"],
        "wkv": tensors["inner_wkv"],
        "wgate": tensors["inner_wgate"],
        "ape": tensors["inner_ape"],
        "norm_w": tensors["inner_norm_w"],
        "cmp_freqs_cos": tensors["cmp_freqs_cos"],
        "cmp_freqs_sin": tensors["cmp_freqs_sin"],
        "hadamard": tensors["hadamard"],
        "idx_kv_cache": tensors["idx_kv_cache"],
        "idx_kv_scale": tensors["idx_kv_scale"],
        "idx_block_table": tensors["idx_block_table"],
        "position_ids": tensors["position_ids"],
        "idx_slot_mapping": tensors["idx_slot_mapping"],
        "inner_state_slot_mapping": tensors["inner_state_slot_mapping"],
    }
    golden_prefill_indexer_compressor(compressor_tensors)
    tensors["idx_kv_cache"][:] = compressor_tensors["idx_kv_cache"]
    tensors["idx_kv_scale"][:] = compressor_tensors["idx_kv_scale"]

    # Lightning-indexer scores with per-token causal top-k.
    position_ids = tensors["position_ids"].long()
    rd = ROPE_HEAD_DIM
    cmp_topk_indices = torch.full((token_count, IDX_TOPK), -1, dtype=torch.int32)
    visible = ((position_ids + 1) // COMPRESS_RATIO).clamp(max=INDEXER_MAX_CANDIDATES)
    max_visible = int(visible.max().item())
    if max_visible == 0:
        return cmp_topk_indices

    # Quantized queries with interleaved RoPE and Hadamard.
    wq_b = tensors["wq_b"]
    wq_b_scale = tensors["wq_b_scale"].float()
    hadamard = tensors["hadamard"].float()

    # Paged INT8 KV and per-position dequantization scales.
    cache_flat_i8 = tensors["idx_kv_cache"].reshape(-1, IDX_HEAD_DIM)
    scale_flat = tensors["idx_kv_scale"].float().reshape(-1, 1)
    idx_block_table = tensors["idx_block_table"]
    logical_rows = torch.arange(max_visible, dtype=torch.int64)
    physical_pages = idx_block_table[logical_rows // BLOCK_SIZE].to(torch.int64)
    valid_pages = (physical_pages >= 0) & (physical_pages < tensors["idx_kv_cache"].shape[0])
    safe_pages = physical_pages.clamp(min=0, max=tensors["idx_kv_cache"].shape[0] - 1)
    physical_rows = safe_pages * BLOCK_SIZE + logical_rows % BLOCK_SIZE
    kv_i8 = cache_flat_i8[physical_rows]
    kv_sc = scale_flat[physical_rows, 0]

    # Query tiles and 8192-candidate score leaves.
    for tile_base in range(0, token_count, PREFILL_DENSE_TILE):
        tile_end = min(tile_base + PREFILL_DENSE_TILE, token_count)
        tile_rows = tile_end - tile_base
        qr_tile = tensors["qr"][tile_base:tile_end]
        qr_scale_tile = tensors["qr_scale"][tile_base:tile_end].float()
        q_i32 = qr_tile.to(torch.int32) @ wq_b.to(torch.int32)
        q = (q_i32.float() * qr_scale_tile * wq_b_scale.view(1, -1)).view(
            tile_rows, IDX_N_HEADS, IDX_HEAD_DIM
        )
        q_pair = q[..., -rd:].unflatten(-1, (-1, 2))
        q0, q1 = q_pair[..., 0], q_pair[..., 1]
        cos_tile = tensors["cos"][tile_base:tile_end].float().view(tile_rows, 1, -1)
        sin_tile = tensors["sin"][tile_base:tile_end].float().view(tile_rows, 1, -1)
        y0 = (q0 * cos_tile - q1 * sin_tile).to(torch.bfloat16)
        y1 = (q0 * sin_tile + q1 * cos_tile).to(torch.bfloat16)
        q = torch.cat(
            [q[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)],
            dim=-1,
        )
        q = q.to(torch.bfloat16).float() @ hadamard
        weights = (tensors["x"][tile_base:tile_end].float() @ tensors["weights_proj"].float()) * WEIGHTS_SCALE

        # Per-row INT8 query scores against pre-quantized KV.
        q_i8, q_sc = int8_quant_per_row(q.reshape(tile_rows * IDX_N_HEADS, IDX_HEAD_DIM))
        q_i8 = q_i8.view(tile_rows, IDX_N_HEADS, IDX_HEAD_DIM).to(torch.int32)
        q_sc = q_sc.view(tile_rows, IDX_N_HEADS, 1)
        for local_t in range(tile_rows):
            global_t = tile_base + local_t
            visible_t = int(visible[global_t].item())
            if visible_t <= 0:
                continue
            running_scores = torch.empty(0, dtype=torch.float32)
            running_indices = torch.empty(0, dtype=torch.int64)
            for begin in range(0, visible_t, 8192):
                end = min(begin + 8192, visible_t)
                score_i32 = torch.einsum("hd,cd->hc", q_i8[local_t], kv_i8[begin:end].to(torch.int32))
                score_leaf = score_i32.float() * q_sc[local_t]
                score_leaf = (torch.relu(score_leaf) * weights[local_t].unsqueeze(-1)).sum(dim=0)
                score_leaf = score_leaf * kv_sc[begin:end]
                score_leaf = torch.where(valid_pages[begin:end], score_leaf, torch.full_like(score_leaf, FP32_NEG_INF))
                leaf_indices = torch.arange(begin, end, dtype=torch.int64)
                merged_scores = torch.cat([running_scores, score_leaf])
                merged_indices = torch.cat([running_indices, leaf_indices])
                keep = min(IDX_TOPK, merged_scores.numel())
                running_scores, selected = torch.topk(merged_scores, keep)
                running_indices = merged_indices[selected]
            cmp_topk_indices[global_t, : running_indices.numel()] = running_indices.to(torch.int32)
    return cmp_topk_indices


def golden_prefill_indexer(tensors):
    import torch

    cmp_topk_indices = golden_prefill_indexer_core(tensors)
    topk_idxs = cmp_topk_indices[:, 0:IDX_TOPK].to(torch.int32)
    tensors["topk_idxs"][:] = topk_idxs


@pl.jit
def prefill_indexer_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.InOut[
        pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM], pl.FP32]
    ],
    inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    topk_idxs: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32]],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    idx_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
):
    x.bind_dynamic(0, T_DYN)
    qr.bind_dynamic(0, T_DYN)
    qr_scale.bind_dynamic(0, T_DYN)
    cos.bind_dynamic(0, T_DYN)
    sin.bind_dynamic(0, T_DYN)
    cmp_freqs_cos.bind_dynamic(0, T_DYN)
    cmp_freqs_sin.bind_dynamic(0, T_DYN)
    inner_compress_state.bind_dynamic(0, INNER_STATE_BLOCK_NUM_DYN)
    idx_kv_cache.bind_dynamic(0, IDX_BLOCK_NUM_DYN)
    idx_kv_scale.bind_dynamic(0, IDX_BLOCK_NUM_DYN)
    topk_idxs.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    idx_slot_mapping.bind_dynamic(0, T_DYN)
    inner_state_slot_mapping.bind_dynamic(0, T_DYN)

    prefill_indexer(
        x,
        qr,
        qr_scale,
        wq_b,
        wq_b_scale,
        weights_proj,
        cos,
        sin,
        cmp_freqs_cos,
        cmp_freqs_sin,
        hadamard,
        inner_compress_state,
        inner_compress_state_block_table,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        idx_kv_cache,
        idx_kv_scale,
        idx_block_table,
        topk_idxs,
        position_ids,
        idx_slot_mapping,
        inner_state_slot_mapping,
    )
    return idx_kv_cache, idx_kv_scale, topk_idxs


def gen_shared_weight(shape, dequant_std, chan_cv):
    """Synthesize a per-output-channel-symmetric INT8 weight + FP32 scale on the real
    DeepSeek-V4-Flash MXFP8 grid (e4m3 + 128x128-block E8M0 scale), then re-quantize
    per-output-channel. Mirrors decode_indexer.gen_shared_weight; ``shape`` last dim is the
    reduction (in) dim, leading dims map to the per-output-channel scale ([out, in] -> [out]).
    """
    import torch

    FP8_MAX, TINY = 448.0, 1e-20

    def sim_fp8(W, block=128):
        out, inn = W.shape
        Wb = W.reshape(out // block, block, inn // block, block)
        scale = torch.exp2(
            torch.ceil(torch.log2((Wb.abs().amax(dim=(1, 3), keepdim=True) / FP8_MAX).clamp_min(TINY)))
        )
        q = (Wb / scale).to(torch.float8_e4m3fn).float() * scale
        return q.reshape(out, inn)

    W = torch.randn(*shape) * torch.exp(chan_cv * torch.randn(*shape[:-1], 1))
    Wq = sim_fp8(W)
    amax = Wq.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale = amax / INT8_SCALE_MAX
    w_i8 = torch.round(Wq / scale).clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
    scale = (scale * (dequant_std / (w_i8.float() * scale).std())).squeeze(-1).float()
    return w_i8, scale


def build_tensor_specs(start_pos: int = START_POS, token_count: int = PREFILL_SEQ):
    from utils import int8_quant_per_row
    import torch
    from golden import TensorSpec
    from utils import token_local_rope

    if token_count <= 0 or token_count > PREFILL_MAX_TOKENS:
        raise ValueError(f"token_count must be in [1, {PREFILL_MAX_TOKENS}], got {token_count}")
    if start_pos < 0 or start_pos + token_count > MAX_SEQ_LEN:
        raise ValueError(
            f"start_pos must satisfy 0 <= start_pos <= {MAX_SEQ_LEN - token_count}, got {start_pos}"
        )
    max_visible = (start_pos + token_count) // COMPRESS_RATIO
    if max_visible > INDEXER_MAX_CANDIDATES:
        raise ValueError(
            f"prefill_indexer needs max_visible={max_visible} compressed slots for start_pos={start_pos}, "
            f"but the exact selector cap is INDEXER_MAX_CANDIDATES={INDEXER_MAX_CANDIDATES}."
        )
    write_count = sum(1 for t in range(token_count) if (start_pos + t + 1) % COMPRESS_RATIO == 0)
    if write_count > MAX_CMP_WRITES:
        raise ValueError(f"fixture generated {write_count} compressed writes, cap is {MAX_CMP_WRITES}")

    def init_inner_compress_state_block_table():
        blocks = torch.arange(INNER_STATE_MAX_BLOCKS, dtype=torch.int64)
        return ((blocks * 17 + 3) % CSA_INNER_STATE_PHYSICAL_BLOCKS).to(torch.int32)

    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        block = abs_pos // INNER_STATE_BLOCK_SIZE
        intra = abs_pos % INNER_STATE_BLOCK_SIZE
        physical_block = (block * 17 + 3) % CSA_INNER_STATE_PHYSICAL_BLOCKS
        return physical_block * INNER_STATE_BLOCK_SIZE + intra

    def init_x():
        return ((torch.rand(token_count, D) - 0.5) * 0.1).to(torch.bfloat16)

    def init_hadamard():
        h = torch.ones((1, 1))
        while h.shape[0] < IDX_HEAD_DIM:
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        return (h * (IDX_HEAD_DIM**-0.5)).to(torch.bfloat16)

    def init_inner_compress_state():
        state = torch.zeros(INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM)
        flat = state.view(-1, INNER_COMPRESS_STATE_DIM)
        for abs_pos in range(max(0, start_pos - INNER_STATE_LEN), start_pos):
            row = state_row(abs_pos)
            if row >= 0:
                flat[row] = (torch.rand(INNER_COMPRESS_STATE_DIM) - 0.5) * 0.05
        return state

    # Indexer-compressor BF16 weight and RMSNorm statistics.
    def init_inner_wkv():
        return torch.randn(INNER_OUT_DIM, D) * 0.0270

    def init_inner_wgate():
        return torch.randn(INNER_OUT_DIM, D) * 0.0513

    def init_inner_ape():
        return torch.randn(COMPRESS_RATIO, INNER_OUT_DIM) * 0.1524

    def init_inner_norm_w():
        return 0.6903 + 0.2663 * torch.randn(INNER_HEAD_DIM)

    # Historical INT8 index KV and per-position dequantization scales.
    _idx_hist = {}

    def _build_idx_hist():
        if "cache" in _idx_hist:
            return
        cache_i8 = torch.zeros(IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM, dtype=torch.int8)
        scale = torch.zeros(IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1)
        c_flat = cache_i8.view(IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, IDX_HEAD_DIM)
        s_flat = scale.view(IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, 1)
        completed = start_pos // COMPRESS_RATIO
        table = init_idx_block_table().to(torch.int64)
        if completed > table.numel() * BLOCK_SIZE:
            raise ValueError("fixture historical compressed slots exceed the standalone idx block table")
        history_chunk = 16 * 1024
        for begin in range(0, completed, history_chunk):
            end = min(begin + history_chunk, completed)
            cmp_slots = torch.arange(begin, end, dtype=torch.int64)
            physical_pages = table[cmp_slots // BLOCK_SIZE]
            valid = physical_pages >= 0
            rows = physical_pages.clamp(min=0) * BLOCK_SIZE + cmp_slots % BLOCK_SIZE
            if valid.any() and int(rows[valid].max().item()) >= c_flat.shape[0]:
                raise ValueError("fixture historical compressed slot exceeds standalone idx_kv_cache capacity")
            hist_bf16 = ((torch.rand(end - begin, IDX_HEAD_DIM) - 0.5) * 0.05).to(torch.bfloat16)
            hi8, hsc = int8_quant_per_row(hist_bf16.float())
            if valid.any():
                valid_rows = rows[valid]
                c_flat.index_copy_(0, valid_rows, hi8[valid])
                s_flat.index_copy_(0, valid_rows, hsc[valid])
        _idx_hist["cache"] = cache_i8
        _idx_hist["scale"] = scale

    def init_idx_kv_cache():
        _build_idx_hist()
        return _idx_hist["cache"].clone()

    def init_idx_kv_scale():
        _build_idx_hist()
        return _idx_hist["scale"].clone()

    def init_idx_block_table():
        return torch.arange(IDX_CACHE_MAX_BLOCKS, dtype=torch.int32)

    def init_position_ids():
        return torch.arange(start_pos, start_pos + token_count, dtype=torch.int32)

    def init_cmp_rope_positions():
        positions = init_position_ids().to(torch.int64)
        boundary = (positions + 1) % COMPRESS_RATIO == 0
        return torch.where(boundary, positions - (COMPRESS_RATIO - 1), torch.zeros_like(positions))

    def init_cmp_freqs_cos():
        cos_cmp, _ = token_local_rope(
            M, COMPRESS_RATIO, init_cmp_rope_positions(),
            max_seq_len=MAX_SEQ_LEN, dtype=torch.bfloat16,
        )
        return cos_cmp.contiguous()

    def init_cmp_freqs_sin():
        _, sin_cmp = token_local_rope(
            M, COMPRESS_RATIO, init_cmp_rope_positions(),
            max_seq_len=MAX_SEQ_LEN, dtype=torch.bfloat16,
        )
        return sin_cmp.contiguous()

    def init_idx_slot_mapping():
        mapping = torch.full((token_count,), -1, dtype=torch.int64)
        positions = torch.arange(start_pos, start_pos + token_count, dtype=torch.int64)
        write_mask = (positions + 1) % COMPRESS_RATIO == 0
        if write_mask.any():
            compressed_slots = (positions[write_mask] + 1) // COMPRESS_RATIO - 1
            table = init_idx_block_table().to(torch.int64)
            physical_pages = table[compressed_slots // BLOCK_SIZE]
            rows = physical_pages * BLOCK_SIZE + compressed_slots % BLOCK_SIZE
            if (physical_pages < 0).any():
                raise ValueError("fixture compressed slot maps to an invalid idx cache block")
            if int(rows.max().item()) >= IDX_CACHE_BLOCK_NUM * BLOCK_SIZE:
                raise ValueError("fixture compressed slot exceeds standalone idx_kv_cache capacity")
            mapping[write_mask] = rows
        return mapping

    def init_inner_state_slot_mapping():
        mapping = torch.full((token_count,), -1, dtype=torch.int64)
        for t in range(token_count):
            mapping[t] = state_row(start_pos + t)
        return mapping

    def init_weights_proj():
        # weights_proj std, averaged over DeepSeek-V4-Flash-0731 layers 8/32.
        return torch.randn(D, IDX_N_HEADS) * 0.2218

    def init_cos():
        cos_query, _ = token_local_rope(
            M, COMPRESS_RATIO, init_position_ids(),
            max_seq_len=MAX_SEQ_LEN, dtype=torch.bfloat16,
        )
        return cos_query[:, : ROPE_HEAD_DIM // 2].float().contiguous()

    def init_sin():
        _, sin_query = token_local_rope(
            M, COMPRESS_RATIO, init_position_ids(),
            max_seq_len=MAX_SEQ_LEN, dtype=torch.bfloat16,
        )
        return sin_query[:, : ROPE_HEAD_DIM // 2].float().contiguous()

    # idx wq_b uses the real MXFP8 grid (not a benign randn int8); qr is per-row int8 like the
    # runtime W8A8C16 activation path.
    wq_b_i8_T, wq_b_scale = gen_shared_weight(
        (IDX_N_HEADS * IDX_HEAD_DIM, Q_LORA), dequant_std=0.108, chan_cv=0.56
    )
    wq_b_i8 = wq_b_i8_T.t().contiguous()
    qr_i8, qr_scale = int8_quant_per_row(torch.rand(token_count, Q_LORA))

    return [
        TensorSpec("x", [token_count, D], torch.bfloat16, init_value=init_x),
        TensorSpec("qr", [token_count, Q_LORA], torch.int8, init_value=lambda: qr_i8),
        TensorSpec("qr_scale", [token_count, 1], torch.float32, init_value=lambda: qr_scale),
        TensorSpec("wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.int8, init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [IDX_N_HEADS * IDX_HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("weights_proj", [D, IDX_N_HEADS], torch.bfloat16, init_value=init_weights_proj),
        TensorSpec("cos", [token_count, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [token_count, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec("cmp_freqs_cos", [token_count, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_cmp_freqs_cos),
        TensorSpec("cmp_freqs_sin", [token_count, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_cmp_freqs_sin),
        TensorSpec("hadamard", [IDX_HEAD_DIM, IDX_HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec(
            "inner_compress_state",
            [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM],
            torch.float32,
            init_value=init_inner_compress_state,
            is_output=True,
        ),
        TensorSpec(
            "inner_compress_state_block_table",
            [INNER_STATE_MAX_BLOCKS],
            torch.int32,
            init_value=init_inner_compress_state_block_table,
        ),
        TensorSpec("inner_wkv", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wkv),
        TensorSpec("inner_wgate", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wgate),
        TensorSpec("inner_ape", [COMPRESS_RATIO, INNER_OUT_DIM], torch.float32, init_value=init_inner_ape),
        TensorSpec("inner_norm_w", [INNER_HEAD_DIM], torch.bfloat16, init_value=init_inner_norm_w),
        TensorSpec(
            "idx_kv_cache",
            [IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM],
            torch.int8,
            init_value=init_idx_kv_cache,
            is_output=True,
        ),
        TensorSpec(
            "idx_kv_scale",
            [IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1],
            torch.float32,
            init_value=init_idx_kv_scale,
            is_output=True,
        ),
        TensorSpec("idx_block_table", [IDX_CACHE_MAX_BLOCKS], torch.int32, init_value=init_idx_block_table),
        TensorSpec("topk_idxs", [token_count, IDX_TOPK], torch.int32, is_output=True),
        TensorSpec("position_ids", [token_count], torch.int32, init_value=init_position_ids),
        TensorSpec(
            "idx_slot_mapping",
            [token_count],
            torch.int64,
            init_value=init_idx_slot_mapping,
        ),
        TensorSpec(
            "inner_state_slot_mapping",
            [token_count],
            torch.int64,
            init_value=init_inner_state_slot_mapping,
        ),
    ]


if __name__ == "__main__":
    import argparse
    import torch
    from golden import ratio_allclose, run_jit
    from utils import int8_quant_per_row

    parser = argparse.ArgumentParser(
        description="Standalone token-major DeepSeek V4 prefill indexer validation."
    )
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"]
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--compile-only",
        action="store_true",
        default=False,
        help="Compile/codegen only; also the implicit behavior on the *sim platforms CI uses.",
    )
    parser.add_argument(
        "--start-pos",
        type=int,
        default=START_POS,
        help="Fixture-only absolute position for token 0; lowered into position_ids and dense idx_slot_mapping.",
    )
    parser.add_argument(
        "--token-count",
        "--num-tokens",
        dest="token_count",
        type=int,
        default=PREFILL_SEQ,
        help=f"Physical token length in [1, {PREFILL_MAX_TOKENS}].",
    )
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    def score_selected_indices(token_id, indices, expected_outputs, inputs):
        qr = inputs["qr"][token_id : token_id + 1]
        qr_scale = inputs["qr_scale"][token_id : token_id + 1].float()
        q_i32 = qr.to(torch.int32) @ inputs["wq_b"].to(torch.int32)
        q = (q_i32.float() * qr_scale * inputs["wq_b_scale"].float().view(1, -1)).view(1, IDX_N_HEADS, IDX_HEAD_DIM)
        q_pair = q[..., -ROPE_HEAD_DIM:].unflatten(-1, (-1, 2))
        cos = inputs["cos"][token_id : token_id + 1].float().view(1, 1, -1)
        sin = inputs["sin"][token_id : token_id + 1].float().view(1, 1, -1)
        rope_even = (q_pair[..., 0] * cos - q_pair[..., 1] * sin).to(torch.bfloat16)
        rope_odd = (q_pair[..., 0] * sin + q_pair[..., 1] * cos).to(torch.bfloat16)
        q = torch.cat(
            [q[..., :-ROPE_HEAD_DIM], torch.stack([rope_even, rope_odd], dim=-1).flatten(-2)],
            dim=-1,
        )
        q = q.to(torch.bfloat16).float() @ inputs["hadamard"].float()
        q_i8, q_scale = int8_quant_per_row(q.reshape(IDX_N_HEADS, IDX_HEAD_DIM))
        weights = (inputs["x"][token_id].float() @ inputs["weights_proj"].float()) * WEIGHTS_SCALE

        logical_rows = indices.to(torch.int64)
        physical_pages = inputs["idx_block_table"][logical_rows // BLOCK_SIZE].to(torch.int64)
        physical_rows = physical_pages * BLOCK_SIZE + logical_rows % BLOCK_SIZE
        cache = expected_outputs["idx_kv_cache"].reshape(-1, IDX_HEAD_DIM)
        scales = expected_outputs["idx_kv_scale"].float().reshape(-1, 1)
        score_i32 = torch.einsum("hd,kd->hk", q_i8.to(torch.int32), cache[physical_rows].to(torch.int32))
        scores = torch.relu(score_i32.float() * q_scale.view(IDX_N_HEADS, 1))
        return (scores * weights.view(-1, 1)).sum(dim=0) * scales[physical_rows, 0]

    def topk_idxs_compare(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        del actual_outputs, rtol, atol
        a_top = actual[..., :IDX_TOPK]
        contract_error = topk_prefix_contract_error(
            a_top,
            inputs["position_ids"],
        )
        if contract_error:
            return False, f"    {contract_error}"
        positions = inputs["position_ids"].cpu().to(torch.int64)
        visible = ((positions + 1) // COMPRESS_RATIO).clamp(max=INDEXER_MAX_CANDIDATES)
        for token_id, visible_count in enumerate(visible.tolist()):
            selected = min(visible_count, IDX_TOPK)
            if selected == 0:
                continue
            indices = a_top[token_id, :selected].long()
            expected_indices = expected[token_id, :selected].long()
            actual_set = set(indices.tolist())
            expected_set = set(expected_indices.tolist())
            if actual_set != expected_set:
                missing = torch.tensor(sorted(expected_set - actual_set), dtype=torch.int64)
                extra = torch.tensor(sorted(actual_set - expected_set), dtype=torch.int64)
                missing_scores = torch.sort(
                    score_selected_indices(token_id, missing, expected_outputs, inputs),
                    descending=True,
                ).values
                extra_scores = torch.sort(
                    score_selected_indices(token_id, extra, expected_outputs, inputs),
                    descending=True,
                ).values
                # Device/CPU score tolerance at the TopK cutoff.
                tolerance = 5e-6 + 1e-5 * missing_scores.abs()
                score_gap = missing_scores - extra_scores
                if torch.all(score_gap <= tolerance):
                    continue
                worst = int(torch.argmax(score_gap - tolerance).item())
                return False, (
                    f"    top-k row {token_id} differs from the streaming "
                    f"full-prefix golden over {visible_count} candidates: "
                    f"missing score {float(missing_scores[worst]):.8g}, "
                    f"replacement score {float(extra_scores[worst]):.8g}"
                )
        return True, ""

    topk_idxs_compare.__name__ = "numerical_cutoff_topk_compare"

    def mapped_active_rows(mapping_name, point_compare):
        def compare(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
            physical_rows = actual.shape[0] * actual.shape[1]
            mapping = inputs[mapping_name].cpu().reshape(-1).to(torch.int64)
            active = torch.unique(mapping[mapping >= 0], sorted=True)
            if active.numel() and int(active[-1]) >= physical_rows:
                return False, f"    {mapping_name} row {int(active[-1])} exceeds {physical_rows}"
            actual_rows = actual.reshape(physical_rows, *actual.shape[2:])
            expected_rows = expected.reshape(physical_rows, *expected.shape[2:])
            if active.numel():
                ok, detail = point_compare(
                    actual_rows.index_select(0, active),
                    expected_rows.index_select(0, active),
                    actual_outputs=actual_outputs,
                    expected_outputs=expected_outputs,
                    inputs=inputs,
                    rtol=rtol,
                    atol=atol,
                )
                if not ok:
                    return False, f"    active rows from {mapping_name}:\n{detail}"
            inactive = torch.ones(physical_rows, dtype=torch.bool)
            inactive[active] = False
            if not torch.equal(actual_rows[inactive], expected_rows[inactive]):
                return False, f"    unmapped physical rows changed for {mapping_name}"
            return True, ""

        compare.__name__ = f"mapped_active_rows({mapping_name})"
        return compare

    result = run_jit(
        fn=prefill_indexer_test,
        specs=build_tensor_specs(args.start_pos, args.token_count),
        golden_fn=golden_prefill_indexer,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform, device_id=args.device, enable_chip_swimlane=args.enable_chip_swimlane
        ),
        rtol=1e-3,
        atol=1e-3,
        compile_only=args.compile_only,
        compare_fn={
            "topk_idxs": topk_idxs_compare,
            # C8 cache: INT8 rows exact bar boundary +/-1 LSB; scale rides alongside.
            "idx_kv_cache": mapped_active_rows(
                "idx_slot_mapping",
                ratio_allclose(atol=1, rtol=0, max_error_ratio=0.01),
            ),
            "idx_kv_scale": mapped_active_rows(
                "idx_slot_mapping",
                ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.01),
            ),
            "inner_compress_state": mapped_active_rows(
                "inner_state_slot_mapping",
                ratio_allclose(
                    atol=1e-4,
                    rtol=1.0 / 128,
                    max_error_ratio=0.01,
                ),
            ),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
