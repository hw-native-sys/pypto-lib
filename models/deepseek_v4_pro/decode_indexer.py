# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 decode indexer with compressed-cache scoring and top-k selection."""

import pypto.language as pl

from config import (
    ACTIVE as M,
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    DECODE_BATCH,
    DECODE_IDX_BLOCK_NUM,
    DECODE_SEQ,
    FP32_NEG_INF,
    IDX_CACHE_MAX_BLOCKS,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
)
from decode_indexer_compressor import indexer_compressor

# model config
B = DECODE_BATCH
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
OFFSET = M.sliding_window

# indexer config
COMPRESS_RATIO = 4
IDX_TOPK = M.index_topk
INNER_OVERLAP = COMPRESS_RATIO == 4
INNER_COFF = 1 + int(INNER_OVERLAP)
INNER_HEAD_DIM = IDX_HEAD_DIM
INNER_OUT_DIM = INNER_COFF * INNER_HEAD_DIM
INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
INNER_STATE_PHYSICAL_BLOCKS = 65
INNER_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + INNER_STATE_BLOCK_SIZE - 1) // INNER_STATE_BLOCK_SIZE
INNER_STATE_BLOCK_NUM = INNER_STATE_PHYSICAL_BLOCKS
INNER_STATE_DIM = 2 * INNER_OUT_DIM

IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
IDX_CACHE_BLOCK_NUM = DECODE_IDX_BLOCK_NUM
SCORE_LEN = IDX_KV_LEN
SCORE_ATOL = 1e-4
SCORE_RTOL = 1.0 / 128
SCORE_HARD_MULTIPLIER = 4.0
SCORE_HARD_ATOL = 5e-3

# tiling
CACHE_TILE = 64
Q_TILE = 256
Q_OUT_TILE = 1024
MM_N_TILE = 512
MM_ROW_TILE = 16
T_PAD = ((T + MM_ROW_TILE - 1) // MM_ROW_TILE) * MM_ROW_TILE
D_TILE = 512
WEIGHTS_K_TILE = D // (4 if M.name == "flash" else 7)
Q_HEAD_TILE = 32
Q_HEAD_FLAT_LEN = Q_HEAD_TILE * IDX_HEAD_DIM
assert IDX_N_HEADS % Q_HEAD_TILE == 0
SCORE_LANE_TILE = 8
TOPK_HALF_LEN = SCORE_LEN // 2
TOPK_HALF_PAIR_OFFSET = 2 * TOPK_HALF_LEN
TOPK_PAIR_WIDTH = 2 * IDX_TOPK


@pl.jit.inline
def indexer(
    x: pl.Tensor[[B, S, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_kv: pl.Tensor[[B, S, INNER_HEAD_DIM], pl.FP32],
    inner_compress_state: pl.InOut[
        pl.Tensor[
            [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM],
            pl.FP32,
        ]
    ],
    inner_compress_state_block_table: pl.Tensor[[B, INNER_STATE_MAX_BLOCKS], pl.INT32],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.BF16],
    # INT8 index cache and per-position FP32 dequantization scale.
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[B, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    score: pl.Tensor[[B, S, SCORE_LEN], pl.FP32],
    topk_idxs: pl.Tensor[[B, S, SCORE_LEN], pl.INT32],
    position_ids: pl.Tensor[[B, S], pl.INT32],
    idx_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    offset: pl.Scalar[pl.INT32],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    qr_acc_pad = pl.create_tensor([T_PAD, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.INT32)
    for ot in pl.spmd(IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE, name_hint="idx_qr_proj_matmul"):
        o_base = ot * Q_OUT_TILE
        for ns in pl.range(0, Q_OUT_TILE, MM_N_TILE):
            qr_acc = pl.create_tensor([MM_ROW_TILE, MM_N_TILE], dtype=pl.INT32)
            for kb in pl.pipeline(0, Q_LORA // Q_TILE, stage=2):
                q0 = kb * Q_TILE
                qr_tile = pl.slice(qr, [T_PAD, Q_TILE], [0, q0], valid_shape=[T, Q_TILE])
                wq_tile = wq_b[q0 : q0 + Q_TILE, o_base + ns : o_base + ns + MM_N_TILE]
                if q0 == 0:
                    qr_acc = pl.matmul(qr_tile, wq_tile, out_dtype=pl.INT32)
                else:
                    qr_acc = pl.matmul_acc(qr_acc, qr_tile, wq_tile)
            qr_acc_pad[0:T_PAD, o_base + ns : o_base + ns + MM_N_TILE] = qr_acc

    # out[j] = x[j] * cos[j] + x[j ^ 1] * sin_signed[j]
    cos_full_t = pl.create_tensor([B, IDX_HEAD_DIM], dtype=pl.FP32)
    sin_full_t = pl.create_tensor([B, IDX_HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_rope_tables"):
        cos_tile = cos[0:B, 0 : ROPE_HEAD_DIM // 2]
        sin_tile = sin[0:B, 0 : ROPE_HEAD_DIM // 2]
        cos_il = pl.full([B, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        cos_il = pl.tensor.scatter(cos_tile, mask_pattern=pl.tile.MaskPattern.P0101, dst=cos_il)
        cos_il = pl.tensor.scatter(cos_tile, mask_pattern=pl.tile.MaskPattern.P1010, dst=cos_il)
        sin_neg = pl.neg(sin_tile)
        sin_signed = pl.full([B, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        sin_signed = pl.tensor.scatter(sin_neg, mask_pattern=pl.tile.MaskPattern.P0101, dst=sin_signed)
        sin_signed = pl.tensor.scatter(sin_tile, mask_pattern=pl.tile.MaskPattern.P1010, dst=sin_signed)
        nope_cos = pl.full([B, IDX_NOPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        cos_full = pl.concat(nope_cos, cos_il)
        cos_full_t[0:B, 0:IDX_HEAD_DIM] = cos_full
        nope_sin = pl.full([B, IDX_NOPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        sin_full = pl.concat(nope_sin, sin_signed)
        sin_full_t[0:B, 0:IDX_HEAD_DIM] = sin_full

    topk_idx_init_t = pl.create_tensor([1, SCORE_LEN], dtype=pl.UINT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="topk_idx_table"):
        topk_idx_values = pl.arange(0, [1, SCORE_LEN], dtype=pl.UINT32)
        topk_idx_init_t[0:1, 0:SCORE_LEN] = topk_idx_values

    qr_hadamard_i8 = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.INT8)
    qr_hadamard_scale_dq = pl.create_tensor([T * IDX_N_HEADS, 1], dtype=pl.FP32)
    for idx in pl.spmd(T * IDX_N_HEADS // Q_HEAD_TILE, name_hint="qr_head"):
        o0 = idx * Q_HEAD_TILE
        tok = o0 // IDX_N_HEADS
        c0 = (o0 - tok * IDX_N_HEADS) * IDX_HEAD_DIM
        qr_batch_idx = tok // S
        wq_scale_slice = wq_b_scale[c0 : c0 + Q_HEAD_FLAT_LEN]
        wq_scale_flat = pl.reshape(wq_scale_slice, [1, Q_HEAD_FLAT_LEN])
        acc_i32 = qr_acc_pad[tok : tok + 1, c0 : c0 + Q_HEAD_FLAT_LEN]
        acc_flat = pl.cast(acc_i32, target_type=pl.FP32, mode="none")
        qr_scale_scalar = pl.read(qr_scale, [tok, 0])
        qr_dequant = pl.mul(acc_flat, qr_scale_scalar)
        qr_dq_flat = pl.mul(qr_dequant, wq_scale_flat)
        qh_rows = pl.reshape(qr_dq_flat, [Q_HEAD_TILE, IDX_HEAD_DIM])
        qh_even = pl.gather(qh_rows, mask_pattern=pl.tile.MaskPattern.P0101)
        qh_odd = pl.gather(qh_rows, mask_pattern=pl.tile.MaskPattern.P1010)
        qh_swapped = pl.full([Q_HEAD_TILE, IDX_HEAD_DIM], dtype=pl.FP32, value=0.0)
        qh_swapped = pl.tensor.scatter(qh_odd, mask_pattern=pl.tile.MaskPattern.P0101, dst=qh_swapped)
        qh_swapped = pl.tensor.scatter(qh_even, mask_pattern=pl.tile.MaskPattern.P1010, dst=qh_swapped)
        cos_full_row = cos_full_t[qr_batch_idx : qr_batch_idx + 1, 0:IDX_HEAD_DIM]
        qh_cos = pl.col_expand_mul(qh_rows, cos_full_row)
        sin_full_row = sin_full_t[qr_batch_idx : qr_batch_idx + 1, 0:IDX_HEAD_DIM]
        qh_sin = pl.col_expand_mul(qh_swapped, sin_full_row)
        qh_rot = pl.add(qh_cos, qh_sin)
        qh_rot_bf16 = pl.cast(qh_rot, target_type=pl.BF16, mode="rint")
        qh_acc = pl.matmul(qh_rot_bf16, hadamard, out_dtype=pl.FP32)
        qh_a_neg = pl.neg(qh_acc)
        qh_a_abs = pl.maximum(qh_acc, qh_a_neg)
        qh_row_max = pl.row_max(qh_a_abs)
        qh_amax_row = pl.reshape(qh_row_max, [1, Q_HEAD_TILE])
        qh_amax_floor = pl.full([1, Q_HEAD_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        qh_amax = pl.maximum(qh_amax_row, qh_amax_floor)
        qh_scale_max = pl.full([1, Q_HEAD_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
        qh_scale_quant_row = pl.div(qh_scale_max, qh_amax)
        qh_scale_dq_row = pl.recip(qh_scale_quant_row)
        qh_scale_dq = pl.reshape(qh_scale_dq_row, [Q_HEAD_TILE, 1])
        qr_hadamard_scale_dq[o0 : o0 + Q_HEAD_TILE, :] = qh_scale_dq
        qh_scale_quant = pl.reshape(qh_scale_quant_row, [Q_HEAD_TILE, 1])
        qh_q_scaled = pl.row_expand_mul(qh_acc, qh_scale_quant)
        qh_q_i32 = pl.cast(qh_q_scaled, target_type=pl.INT32, mode="rint")
        qh_q_half = pl.cast(qh_q_i32, target_type=pl.FP16, mode="round")
        qh_q_i8 = pl.cast(qh_q_half, target_type=pl.INT8, mode="trunc")
        qr_hadamard_i8[o0 : o0 + Q_HEAD_TILE, :] = qh_q_i8

    x_flat = pl.reshape(x, [T, D])
    weights_partial = pl.create_tensor([(D // WEIGHTS_K_TILE) * MM_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32)
    with pl.spmd(D // WEIGHTS_K_TILE, name_hint="weights_proj", deps=[late_dep]) as _weights_tid:
        kb = pl.tile.get_block_idx()
        k_base = kb * WEIGHTS_K_TILE
        weights_acc = pl.create_tensor([MM_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32)
        for db in pl.range(WEIGHTS_K_TILE // D_TILE):
            d0 = k_base + db * D_TILE
            x_rows = pl.min(MM_ROW_TILE, T)
            x_tile = pl.slice(x_flat, [MM_ROW_TILE, D_TILE], [0, d0], valid_shape=[x_rows, D_TILE])
            weights_proj_tile = weights_proj[d0 : d0 + D_TILE, :]
            if db == 0:
                weights_acc = pl.matmul(x_tile, weights_proj_tile, out_dtype=pl.FP32)
            else:
                weights_acc = pl.matmul_acc(weights_acc, x_tile, weights_proj_tile)
        weights_partial[kb * MM_ROW_TILE : kb * MM_ROW_TILE + MM_ROW_TILE, :] = weights_acc

    weights = pl.create_tensor([T_PAD, IDX_N_HEADS], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="weights_proj_reduce"):
        w_sum = weights_partial[0:MM_ROW_TILE, :]
        for kb in pl.unroll(1, D // WEIGHTS_K_TILE):
            weights_partial_tile = weights_partial[kb * MM_ROW_TILE : kb * MM_ROW_TILE + MM_ROW_TILE, :]
            w_sum = pl.add(w_sum, weights_partial_tile)
        weights_scaled = pl.mul(w_sum, WEIGHTS_SCALE)
        weights[0:MM_ROW_TILE, :] = weights_scaled

    indexer_compressor(
        x, inner_kv,
        inner_compress_state, inner_compress_state_block_table,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        cos, sin, hadamard, idx_kv_cache, idx_kv_scale,
        position_ids, idx_slot_mapping, inner_state_slot_mapping,
        late_dep,
    )

    kv_cache_i8_flat = pl.reshape(idx_kv_cache, [IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, IDX_HEAD_DIM])
    idx_block_table_flat = pl.reshape(idx_block_table, [B * IDX_CACHE_MAX_BLOCKS])
    score_acc_gm = pl.create_tensor([T * IDX_N_HEADS, SCORE_LEN], dtype=pl.INT32)

    with pl.spmd(SCORE_LANE_TILE, name_hint="score_mat") as score_mat_tid:
        mat_lane = pl.tile.get_block_idx()
        for mat_tg in pl.unroll(T):
            mat_b = mat_tg // S
            mat_s = mat_tg - mat_b * S
            mat_cache_len = pl.read(kv_seq_lens, [mat_b]) // COMPRESS_RATIO
            mat_score_pos = pl.read(position_ids, [mat_b, mat_s])
            mat_visible_cache = pl.min(mat_cache_len, (mat_score_pos + 1) // COMPRESS_RATIO)
            mat_visible_len = pl.min(mat_visible_cache, SCORE_LEN)
            mat_cache_blocks = (mat_visible_len + BLOCK_SIZE - 1) // BLOCK_SIZE
            mat_q0 = (mat_b * S + mat_s) * IDX_N_HEADS
            mat_lane_rot = mat_lane + (SCORE_LANE_TILE - mat_tg % SCORE_LANE_TILE) % SCORE_LANE_TILE
            mat_first_block = mat_lane_rot - SCORE_LANE_TILE * (mat_lane_rot // SCORE_LANE_TILE)
            mat_lane_iters = (mat_cache_blocks - mat_first_block + SCORE_LANE_TILE - 1) // SCORE_LANE_TILE
            if mat_lane_iters > 0:
                mat_query = qr_hadamard_i8[mat_q0 : mat_q0 + IDX_N_HEADS, 0:IDX_HEAD_DIM]
                for mat_local_block in pl.pipeline(0, mat_lane_iters, stage=2):
                    mat_cache_block = mat_first_block + mat_local_block * SCORE_LANE_TILE
                    mat_c0 = mat_cache_block * BLOCK_SIZE
                    mat_block_raw = pl.read(idx_block_table_flat, [mat_b * IDX_CACHE_MAX_BLOCKS + mat_cache_block])
                    mat_block_id = pl.cast(mat_block_raw, pl.INDEX)
                    mat_kv0 = mat_block_id * BLOCK_SIZE
                    mat_kv_tile = kv_cache_i8_flat[mat_kv0 : mat_kv0 + BLOCK_SIZE, :]
                    mat_score_acc = pl.matmul(mat_query, mat_kv_tile, out_dtype=pl.INT32, b_trans=True)
                    score_acc_gm[mat_q0 : mat_q0 + IDX_N_HEADS, mat_c0 : mat_c0 + BLOCK_SIZE] = mat_score_acc

    kv_scale_rows = pl.reshape(idx_kv_scale, [IDX_CACHE_BLOCK_NUM, BLOCK_SIZE])
    score_flat = pl.reshape(score, [T, SCORE_LEN])
    with pl.spmd(SCORE_LANE_TILE, name_hint="score_reduce", deps=[score_mat_tid]) as score_reduce_tid:
        reduce_lane = pl.tile.get_block_idx()
        for reduce_tg in pl.unroll(T):
            reduce_b = reduce_tg // S
            reduce_s = reduce_tg - reduce_b * S
            reduce_cache_len = pl.read(kv_seq_lens, [reduce_b]) // COMPRESS_RATIO
            reduce_score_pos = pl.read(position_ids, [reduce_b, reduce_s])
            reduce_visible_cache = pl.min(reduce_cache_len, (reduce_score_pos + 1) // COMPRESS_RATIO)
            reduce_visible_len = pl.min(reduce_visible_cache, SCORE_LEN)
            reduce_cache_blocks = (reduce_visible_len + BLOCK_SIZE - 1) // BLOCK_SIZE
            reduce_t = reduce_b * S + reduce_s
            reduce_q0 = reduce_t * IDX_N_HEADS
            reduce_lane_rot = reduce_lane + (SCORE_LANE_TILE - reduce_tg % SCORE_LANE_TILE) % SCORE_LANE_TILE
            reduce_first_block = reduce_lane_rot - SCORE_LANE_TILE * (reduce_lane_rot // SCORE_LANE_TILE)
            reduce_lane_iters = (reduce_cache_blocks - reduce_first_block + SCORE_LANE_TILE - 1) // SCORE_LANE_TILE
            if reduce_lane_iters > 0:
                reduce_query_scale = qr_hadamard_scale_dq[reduce_q0 : reduce_q0 + IDX_N_HEADS, :]
                reduce_weight_row = weights[reduce_t : reduce_t + 1, 0:IDX_N_HEADS]
                reduce_weight_col = pl.reshape(reduce_weight_row, [IDX_N_HEADS, 1])
                reduce_head_scale = pl.mul(reduce_query_scale, reduce_weight_col)
                for reduce_local_block in pl.pipeline(0, reduce_lane_iters, stage=2):
                    reduce_cache_block = reduce_first_block + reduce_local_block * SCORE_LANE_TILE
                    reduce_c0 = reduce_cache_block * BLOCK_SIZE
                    reduce_valid_len = pl.min(BLOCK_SIZE, reduce_visible_len - reduce_c0)
                    reduce_block_index = reduce_b * IDX_CACHE_MAX_BLOCKS + reduce_cache_block
                    reduce_block_raw = pl.read(idx_block_table_flat, [reduce_block_index])
                    reduce_block_id = pl.cast(reduce_block_raw, pl.INDEX)
                    reduce_acc = score_acc_gm[reduce_q0 : reduce_q0 + IDX_N_HEADS, reduce_c0 : reduce_c0 + BLOCK_SIZE]
                    reduce_score_fp32 = pl.cast(reduce_acc, target_type=pl.FP32, mode="none")
                    reduce_score_relu = pl.maximum(reduce_score_fp32, 0.0)
                    reduce_score_weighted = pl.row_expand_mul(reduce_score_relu, reduce_head_scale)
                    reduce_score_head_sum = pl.col_sum(reduce_score_weighted)
                    reduce_kv_scale = kv_scale_rows[reduce_block_id : reduce_block_id + 1, 0:BLOCK_SIZE]
                    reduce_score_dequant = pl.mul(reduce_score_head_sum, reduce_kv_scale)
                    reduce_score_shape = pl.set_validshape(reduce_score_dequant, 1, reduce_valid_len)
                    reduce_score_padded = pl.fillpad(reduce_score_shape, pad_value=pl.PadValue.min)
                    reduce_score_valid = pl.maximum(reduce_score_padded, FP32_NEG_INF)
                    score_flat[reduce_t : reduce_t + 1, reduce_c0 : reduce_c0 + BLOCK_SIZE] = reduce_score_valid

    topk_idxs_flat = pl.reshape(topk_idxs, [T, SCORE_LEN])
    with pl.spmd(T, name_hint="topk", deps=[score_reduce_tid]) as _topk_tid:
        t = pl.tile.get_block_idx()
        invalid_idxs = pl.full([1, SCORE_LEN], dtype=pl.INT32, value=-1)
        topk_idxs_flat[t : t + 1, :] = invalid_idxs
        topk_batch_idx = t // S
        token_s = t - topk_batch_idx * S
        cache_len_b = pl.read(kv_seq_lens, [topk_batch_idx]) // COMPRESS_RATIO
        topk_pos = pl.read(position_ids, [topk_batch_idx, token_s])
        topk_visible_cache = pl.min(cache_len_b, (topk_pos + 1) // COMPRESS_RATIO)
        topk_visible_len = pl.min(topk_visible_cache, SCORE_LEN)
        if topk_visible_len > 0:
            offset_i32 = pl.cast(offset, target_type=pl.INT32)
            score_full_raw = score_flat[t : t + 1, 0:SCORE_LEN]
            score_full_shape = pl.set_validshape(score_full_raw, 1, topk_visible_len)
            score_full_padded = pl.fillpad(score_full_shape, pad_value=pl.PadValue.min)
            score_floor = pl.full([1, SCORE_LEN], dtype=pl.FP32, value=FP32_NEG_INF)
            score_full = pl.maximum(score_full_padded, score_floor)
            topk_index_row = topk_idx_init_t[0:1, 0:SCORE_LEN]
            sorted_32 = pl.sort32(score_full, topk_index_row)
            sorted_64 = pl.mrgsort(sorted_32, block_len=64)
            sorted_256 = pl.mrgsort(sorted_64, block_len=256)
            sorted_full = pl.mrgsort(sorted_256, block_len=1024)

            # Merge the two sorted 2048-score runs.
            half0_candidates = sorted_full[:, 0:TOPK_PAIR_WIDTH]
            half1_candidates = sorted_full[:, TOPK_HALF_PAIR_OFFSET : TOPK_HALF_PAIR_OFFSET + TOPK_PAIR_WIDTH]
            merged_candidates = pl.mrgsort(half0_candidates, half1_candidates)
            topk_pairs = merged_candidates[:, 0:TOPK_PAIR_WIDTH]
            topk_idxs_tile = pl.gather(topk_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            valid_topk = pl.min(IDX_TOPK, topk_visible_len)
            topk_idxs_valid = pl.set_validshape(topk_idxs_tile, 1, valid_topk)
            topk_idxs_offset = pl.add(topk_idxs_valid, offset_i32)
            topk_idxs_flat[t : t + 1, 0:IDX_TOPK] = topk_idxs_offset

    return idx_kv_cache, idx_kv_scale, inner_compress_state, score, topk_idxs


@pl.jit
def indexer_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_kv: pl.Tensor[[B, S, INNER_HEAD_DIM], pl.FP32],
    inner_compress_state: pl.InOut[
        pl.Tensor[
            [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM],
            pl.FP32,
        ]
    ],
    inner_compress_state_block_table: pl.Tensor[[B, INNER_STATE_MAX_BLOCKS], pl.INT32],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[B, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    score: pl.Out[pl.Tensor[[B, S, SCORE_LEN], pl.FP32]],
    topk_idxs: pl.Out[pl.Tensor[[B, S, SCORE_LEN], pl.INT32]],
    position_ids: pl.Tensor[[B, S], pl.INT32],
    idx_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    offset: pl.Scalar[pl.INT32],
):
    late_dep = pl.system.task_dummy(deps=[])
    indexer(
        x, qr, qr_scale, wq_b, wq_b_scale, weights_proj,
        cos, sin, hadamard,
        inner_kv, inner_compress_state, inner_compress_state_block_table,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        idx_kv_cache, idx_kv_scale, idx_block_table,
        score, topk_idxs,
        position_ids, idx_slot_mapping, inner_state_slot_mapping, kv_seq_lens,
        offset, late_dep,
    )
    return score, inner_compress_state, idx_kv_cache, idx_kv_scale, topk_idxs


def _int8_quant_per_row(x):
    """Per-row INT8 symmetric quant matching the runtime W8A8C16 activation path."""
    import torch

    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


def gen_shared_weight(shape, dequant_std, chan_cv):
    """Generate per-output-channel INT8 weights from an MXFP8 E4M3/E8M0 grid."""
    import torch

    FP8_MAX, TINY = 448.0, 1e-20

    def sim_fp8(W, block=128):
        out, inn = W.shape
        Wb = W.reshape(out // block, block, inn // block, block)
        scale = torch.exp2(torch.ceil(torch.log2((Wb.abs().amax(dim=(1, 3), keepdim=True) / FP8_MAX).clamp_min(TINY))))
        q = (Wb / scale).to(torch.float8_e4m3fn).float() * scale
        return q.reshape(out, inn)

    W = torch.randn(*shape) * torch.exp(chan_cv * torch.randn(*shape[:-1], 1))
    Wq = sim_fp8(W)
    amax = Wq.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale = amax / INT8_SCALE_MAX
    w_i8 = torch.round(Wq / scale).clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
    scale = (scale * (dequant_std / (w_i8.float() * scale).std())).squeeze(-1).float()
    return w_i8, scale


def golden_indexer(tensors):
    """Torch reference for Indexer.forward decode branch; prefill `start_pos == 0` path is omitted."""
    import torch
    from decode_indexer_compressor import golden_compressor

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
    offset = int(tensors["offset"])

    bsz, seqlen, _ = x.shape
    ratio, rd = COMPRESS_RATIO, ROPE_HEAD_DIM

    q_i32 = qr.to(torch.int32) @ wq_b.to(torch.int32)
    q = (q_i32.float() * qr_scale * wq_b_scale.view(1, -1)).view(B, S, IDX_N_HEADS, IDX_HEAD_DIM)

    x_pair = q[..., -rd:].unflatten(-1, (-1, 2))
    x0, x1 = x_pair[..., 0], x_pair[..., 1]
    cos_v = cos.view(B, 1, 1, -1)
    sin_v = sin.view(B, 1, 1, -1)
    y0 = (x0 * cos_v - x1 * sin_v).to(torch.bfloat16)
    y1 = (x0 * sin_v + x1 * cos_v).to(torch.bfloat16)

    q = torch.cat([q[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)

    q = q.to(torch.bfloat16).float() @ hadamard
    # Per-row INT8 query values and scales feed the score matmul.

    inner_tensors = {
        "x": tensors["x"],
        "kv": tensors["inner_kv"],
        "wkv": tensors["inner_wkv"],
        "wgate": tensors["inner_wgate"],
        "ape": tensors["inner_ape"],
        "norm_w": tensors["inner_norm_w"],
        "cos": tensors["cos"],
        "sin": tensors["sin"],
        "hadamard": tensors["hadamard"],
        "compress_state": tensors["inner_compress_state"],
        "compress_state_block_table": tensors["inner_compress_state_block_table"],
        "idx_kv_cache": tensors["idx_kv_cache"],
        "idx_kv_scale": tensors["idx_kv_scale"],
        "position_ids": tensors["position_ids"],
        "idx_slot_mapping": tensors["idx_slot_mapping"],
        "inner_state_slot_mapping": tensors["inner_state_slot_mapping"],
    }
    golden_compressor(inner_tensors)

    weights = (x @ weights_proj) * WEIGHTS_SCALE

    # The index cache stores INT8 KV rows and per-position dequantization scales.
    idx_kv_cache_i8 = tensors["idx_kv_cache"]
    idx_kv_scale = tensors["idx_kv_scale"].float()
    idx_block_table = tensors["idx_block_table"]
    score_full = torch.full((bsz, seqlen, SCORE_LEN), FP32_NEG_INF, dtype=torch.float32)
    topk_idxs = torch.full((bsz, seqlen, SCORE_LEN), -1, dtype=torch.int32)
    q_i8, q_scale = _int8_quant_per_row(q.reshape(B * S * IDX_N_HEADS, IDX_HEAD_DIM))
    q_i8 = q_i8.view(B, S, IDX_N_HEADS, IDX_HEAD_DIM)
    q_scale = q_scale.view(B, S, IDX_N_HEADS, 1)

    for b in range(bsz):
        cache_len = int(kv_seq_lens[b].item()) // ratio
        if cache_len <= 0:
            continue

        kv_i8_rows = []
        kv_scale_rows = []
        for slot in range(cache_len):
            blk_id = int(idx_block_table[b, slot // BLOCK_SIZE].item())
            kv_i8_rows.append(idx_kv_cache_i8[blk_id, slot % BLOCK_SIZE, 0])
            kv_scale_rows.append(idx_kv_scale[blk_id, slot % BLOCK_SIZE, 0, 0])
        kv_i8 = torch.stack(kv_i8_rows, dim=0).view(cache_len, IDX_HEAD_DIM)
        kv_scale = torch.stack(kv_scale_rows, dim=0).view(cache_len, 1)
        score_i32 = torch.einsum("shd,td->sht", q_i8[b].to(torch.int32), kv_i8.to(torch.int32))
        score = score_i32.float() * q_scale[b]
        score = (torch.relu(score) * weights[b].unsqueeze(-1)).sum(dim=1)
        score = score * kv_scale.view(1, cache_len)
        for s in range(seqlen):
            visible_len = min(cache_len, int(tensors["position_ids"][b, s].item() + 1) // ratio, SCORE_LEN)
            if visible_len <= 0:
                continue
            score_full[b, s, :visible_len] = score[s, :visible_len].to(torch.float32)
            k = min(IDX_TOPK, visible_len)
            _, idx = score[s, :visible_len].topk(k, dim=-1)
            topk_idxs[b, s, :k] = idx.to(torch.int32)
            topk_idxs[b, s, :k] += offset

    tensors["score"][:] = score_full

    tensors["topk_idxs"][:] = topk_idxs.view(B, S, SCORE_LEN)


def _scalar_input_as_int(value, name):
    """Decode a scalar harness input and return either its value or an error string."""
    if hasattr(value, "numel") and value.numel() != 1:
        return None, f"{name} must be scalar, got shape {tuple(value.shape)}"
    if hasattr(value, "item"):
        value = value.item()
    elif hasattr(value, "value"):
        value = value.value
    try:
        return int(value), None
    except (TypeError, ValueError):
        return None, f"{name} must be integer-like, got {value!r}"


def topk_prefix_contract_error(
    topk_indices,
    position_ids,
    kv_seq_lens,
    offset,
):
    """Return an error string for an invalid decode top-k prefix or ``-1`` tail."""
    import torch

    if topk_indices.ndim != 3 or topk_indices.shape != (B, S, SCORE_LEN):
        return (
            f"top-k tensor must have shape {(B, S, SCORE_LEN)}, "
            f"got {tuple(topk_indices.shape)}"
        )
    if position_ids.ndim != 2 or position_ids.shape != (B, S):
        return f"position_ids must have shape {(B, S)}, got {tuple(position_ids.shape)}"
    if kv_seq_lens.ndim != 1 or kv_seq_lens.shape[0] != B:
        return f"kv_seq_lens must have shape {(B,)}, got {tuple(kv_seq_lens.shape)}"

    for b in range(B):
        cache_len = max(int(kv_seq_lens[b].item()) // COMPRESS_RATIO, 0)
        for s in range(S):
            row = topk_indices[b, s]
            position_visible = max(
                (int(position_ids[b, s].item()) + 1) // COMPRESS_RATIO,
                0,
            )
            visible = min(cache_len, position_visible, SCORE_LEN)
            prefix_len = min(IDX_TOPK, visible)
            prefix = row[:prefix_len]
            if prefix_len:
                out_of_range = (prefix < offset) | (prefix >= offset + visible)
                if out_of_range.any().item():
                    return (
                        f"top-k row [{b},{s}] has "
                        f"{int(out_of_range.count_nonzero().item())} entries outside "
                        f"[{offset}, {offset + visible}) in its active prefix"
                    )
                unique_count = int(torch.unique(prefix).numel())
                if unique_count != prefix_len:
                    return (
                        f"top-k row [{b},{s}] active prefix has "
                        f"{unique_count}/{prefix_len} unique entries"
                    )
            tail_non_padding = int((row[prefix_len:] != -1).count_nonzero().item())
            if tail_non_padding:
                return (
                    f"top-k row [{b},{s}] tail contains "
                    f"{tail_non_padding} non--1 entries"
                )
    return None


def decode_topk_compare(
    actual,
    expected,
    *,
    actual_outputs,
    expected_outputs,
    inputs,
    rtol,
    atol,
):
    """Validate decode top-k structure and allow only score-bounded tie changes."""
    import torch

    max_show = 10

    position_ids = inputs.get("position_ids")
    kv_seq_lens = inputs.get("kv_seq_lens")
    offset_raw = inputs.get("offset", OFFSET)
    if position_ids is None or kv_seq_lens is None:
        return False, (
            "    compare_fn requires position_ids and kv_seq_lens inputs"
        )
    offset, scalar_error = _scalar_input_as_int(offset_raw, "offset")
    if scalar_error:
        return False, f"    {scalar_error}"

    actual = actual.cpu()
    expected = expected.cpu()
    position_ids = position_ids.cpu()
    kv_seq_lens = kv_seq_lens.cpu()
    for label, indices in (("actual", actual), ("expected", expected)):
        contract_error = topk_prefix_contract_error(
            indices,
            position_ids,
            kv_seq_lens,
            offset,
        )
        if contract_error:
            return False, f"    {label} {contract_error}"

    scores = {}
    for label, outputs in (("actual", actual_outputs), ("expected", expected_outputs)):
        score = outputs.get("score")
        if score is None:
            return False, f"    compare_fn misconfigured: missing {label} output 'score'"
        score = score.cpu().to(torch.float32)
        if score.shape != (B, S, SCORE_LEN):
            return False, (
                f"    {label} score must have shape {(B, S, SCORE_LEN)}, "
                f"got {tuple(score.shape)}"
            )
        nonfinite = ~torch.isfinite(score)
        if nonfinite.any().item():
            return False, (
                f"    {label} score contains "
                f"{int(nonfinite.count_nonzero().item())} non-finite value(s)"
            )
        scores[label] = score

    actual_score = scores["actual"]
    expected_score = scores["expected"]
    failures = []
    failure_count = 0

    def record_failure(detail):
        nonlocal failure_count
        failure_count += 1
        if len(failures) < max_show:
            failures.append(detail)

    for b in range(B):
        cache_len = max(int(kv_seq_lens[b].item()) // COMPRESS_RATIO, 0)
        for s in range(S):
            position_visible = max(
                (int(position_ids[b, s].item()) + 1) // COMPRESS_RATIO,
                0,
            )
            visible = min(cache_len, position_visible, SCORE_LEN)
            k = min(IDX_TOPK, visible)
            if k <= 0:
                continue

            actual_order = actual[b, s, :k].to(torch.int64) - offset
            expected_order = expected[b, s, :k].to(torch.int64) - offset
            mismatch = actual_order != expected_order
            if not mismatch.any().item():
                continue

            actual_row = actual_score[b, s, :visible]
            expected_row = expected_score[b, s, :visible]
            candidate_tolerance = SCORE_ATOL + SCORE_RTOL * torch.maximum(
                actual_row.abs(),
                expected_row.abs(),
            )
            candidate_hard_tolerance = torch.maximum(
                SCORE_HARD_MULTIPLIER * candidate_tolerance,
                torch.full_like(candidate_tolerance, SCORE_HARD_ATOL),
            )
            candidate_error = (actual_row - expected_row).abs()

            # Check pointwise score errors for displaced candidates.
            displaced = torch.unique(
                torch.cat((actual_order[mismatch], expected_order[mismatch]))
            )
            bad_candidates = displaced[
                candidate_error[displaced]
                > candidate_hard_tolerance[displaced]
            ]
            for candidate_tensor in bad_candidates:
                candidate = int(candidate_tensor.item())
                record_failure(
                    f"row=[{b},{s}] candidate={candidate} score error "
                    f"{float(candidate_error[candidate].item()):.6g} exceeds "
                    f"candidate hard bound "
                    f"{float(candidate_hard_tolerance[candidate].item()):.6g} "
                    f"(actual={float(actual_row[candidate].item()):.6g}, "
                    f"expected={float(expected_row[candidate].item()):.6g})"
                )

            # Check selected candidates remain descending within score uncertainty.
            if k >= 2:
                for label, row in (
                    ("expected", expected_row),
                    ("actual", actual_row),
                ):
                    selected_lower = (
                        row[actual_order]
                        - candidate_hard_tolerance[actual_order]
                    )
                    selected_upper = (
                        row[actual_order]
                        + candidate_hard_tolerance[actual_order]
                    )
                    prior_min_upper = torch.cummin(
                        selected_upper[:-1],
                        dim=0,
                    ).values
                    clear_reverse = (
                        selected_lower[1:] > prior_min_upper
                    ).nonzero(as_tuple=False).flatten()
                    for position_tensor in clear_reverse:
                        later_position = int(position_tensor.item()) + 1
                        earlier_position = int(
                            selected_upper[:later_position].argmin().item()
                        )
                        earlier_candidate = int(
                            actual_order[earlier_position].item()
                        )
                        later_candidate = int(
                            actual_order[later_position].item()
                        )
                        reverse_gap = float(
                            (
                                row[later_candidate]
                                - row[earlier_candidate]
                            ).item()
                        )
                        joint_bound = float(
                            (
                                candidate_hard_tolerance[earlier_candidate]
                                + candidate_hard_tolerance[later_candidate]
                            ).item()
                        )
                        record_failure(
                            f"row=[{b},{s}] clear inversion on {label} "
                            f"scores at positions {earlier_position}/"
                            f"{later_position}: candidate "
                            f"{earlier_candidate} precedes {later_candidate}; "
                            f"reverse_gap={reverse_gap:.6g}, "
                            f"joint_bound={joint_bound:.6g}"
                        )

            # Check candidate changes across the top-k boundary.
            if visible > k:
                missing_mask = ~torch.isin(expected_order, actual_order)
                added_mask = ~torch.isin(actual_order, expected_order)
                missing = expected_order[missing_mask]
                added = actual_order[added_mask]
                if missing.numel() and added.numel():
                    for label, row in (
                        ("expected", expected_row),
                        ("actual", actual_row),
                    ):
                        missing_lower = (
                            row[missing] - candidate_hard_tolerance[missing]
                        )
                        added_upper = (
                            row[added] + candidate_hard_tolerance[added]
                        )
                        best_missing_pos = int(missing_lower.argmax().item())
                        worst_added_pos = int(added_upper.argmin().item())
                        boundary_gap = float(
                            (
                                missing_lower[best_missing_pos]
                                - added_upper[worst_added_pos]
                            ).item()
                        )
                        if boundary_gap > 0:
                            missing_candidate = int(missing[best_missing_pos].item())
                            added_candidate = int(added[worst_added_pos].item())
                            record_failure(
                                f"row=[{b},{s}] clear top-k boundary miss on "
                                f"{label} scores: omitted candidate "
                                f"{missing_candidate}, selected candidate "
                                f"{added_candidate}, uncertainty-adjusted "
                                f"gap={boundary_gap:.6g}"
                            )

    if not failure_count:
        return True, ""
    lines = [
        "    decode top-k differs outside candidate-specific score bounds: "
        f"{failure_count} failure(s) "
        f"(score_atol={SCORE_ATOL} score_rtol={SCORE_RTOL} "
        f"score_hard_multiplier={SCORE_HARD_MULTIPLIER} "
        f"score_hard_atol={SCORE_HARD_ATOL})"
    ]
    lines.extend(f"      {detail}" for detail in failures)
    if failure_count > len(failures):
        lines.append(f"      ... and {failure_count - len(failures)} more")
    return False, "\n".join(lines)


decode_topk_compare.__name__ = "decode_topk_pair_compare"


def score_valid_compare(
    actual,
    expected,
    *,
    actual_outputs,
    expected_outputs,
    inputs,
    rtol,
    atol,
):
    """Compare visible scores with a ratio budget and hard pointwise ceiling."""
    import torch

    from golden import ratio_allclose

    if actual.shape != expected.shape:
        return False, (
            f"    score shape mismatch: actual={tuple(actual.shape)} "
            f"expected={tuple(expected.shape)}"
        )
    actual_f = actual.cpu().to(torch.float32)
    expected_f = expected.cpu().to(torch.float32)
    for label, value in (("actual", actual_f), ("expected", expected_f)):
        nonfinite = ~torch.isfinite(value)
        if nonfinite.any().item():
            return False, (
                f"    {label} score contains "
                f"{int(nonfinite.count_nonzero().item())} non-finite value(s)"
            )
    valid = expected_f != FP32_NEG_INF
    actual_valid = actual_f[valid]
    expected_valid = expected_f[valid]
    base_ok, base_detail = ratio_allclose(atol=SCORE_ATOL, rtol=SCORE_RTOL)(
        actual_valid,
        expected_valid,
        actual_outputs=actual_outputs,
        expected_outputs=expected_outputs,
        inputs=inputs,
        rtol=rtol,
        atol=atol,
    )
    if not base_ok or actual_valid.numel() == 0:
        return base_ok, base_detail

    diff = (actual_valid - expected_valid).abs()
    hard_tolerance = torch.maximum(
        SCORE_HARD_MULTIPLIER * (
            SCORE_ATOL
            + SCORE_RTOL * torch.maximum(
                actual_valid.abs(),
                expected_valid.abs(),
            )
        ),
        torch.full_like(actual_valid, SCORE_HARD_ATOL),
    )
    hard_bad = diff > hard_tolerance
    hard_bad_count = int(hard_bad.count_nonzero().item())
    if not hard_bad_count:
        return True, ""

    flat_bad = hard_bad.nonzero(as_tuple=False).flatten()
    lines = []
    for index in flat_bad[:10].tolist():
        lines.append(
            f"      [{index}] actual={float(actual_valid[index]):.8g} "
            f"expected={float(expected_valid[index]):.8g} "
            f"diff={float(diff[index]):.4g} "
            f"hard_tol={float(hard_tolerance[index]):.4g}"
        )
    return False, (
        f"    score hard bound exceeded at {hard_bad_count} point(s): "
        f"hard_multiplier={SCORE_HARD_MULTIPLIER}, score_atol={SCORE_ATOL}, "
        f"score_rtol={SCORE_RTOL}, hard_atol={SCORE_HARD_ATOL}\n"
        + "\n".join(lines)
    )


score_valid_compare.__name__ = "score_valid_region_compare"


def build_tensor_specs(start_pos=None):
    import torch
    from decode_metadata import (
        block_table,
        compressed_slot_mapping,
        csa_decode_start_set,
        kv_seq_lens_from_starts,
        position_ids_from_starts,
        resolve_start_positions,
        state_slot_mapping,
    )
    from golden import ScalarSpec, TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables, materialize_half_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

    def init_x():
        return torch.rand(B, S, D)
    def init_qr():
        return torch.rand(T, Q_LORA)
    def init_weights_proj():
        return torch.randn(D, IDX_N_HEADS) * 0.2313
    def init_rope_positions():
        return init_position_ids().to(torch.int64)[:, 0]
    def init_cos():
        return materialize_half_rope_tables(shared_freqs_cos, shared_freqs_sin, init_rope_positions())[0]
    def init_sin():
        return materialize_half_rope_tables(shared_freqs_cos, shared_freqs_sin, init_rope_positions())[1]
    def init_hadamard():
        return torch.rand(IDX_HEAD_DIM, IDX_HEAD_DIM) * (IDX_HEAD_DIM ** -0.5)
    def init_inner_compress_state():
        state = torch.zeros(INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM)
        state[:, :, INNER_OUT_DIM:] = FP32_NEG_INF
        return state
    def init_inner_compress_state_block_table():
        return block_table(
            batch=B,
            table_blocks=INNER_STATE_MAX_BLOCKS,
            physical_blocks=INNER_STATE_PHYSICAL_BLOCKS,
        )
    def init_inner_wkv():
        return torch.randn(INNER_OUT_DIM, D) * 0.0293
    def init_inner_wgate():
        return torch.randn(INNER_OUT_DIM, D) * 0.0512
    def init_inner_ape():
        return torch.randn(COMPRESS_RATIO, INNER_OUT_DIM) * 0.1528
    def init_inner_norm_w():
        return 0.6850 + 0.2610 * torch.randn(INNER_HEAD_DIM)
    def init_idx_block_table():
        return block_table(
            batch=B,
            table_blocks=IDX_CACHE_MAX_BLOCKS,
            physical_blocks=IDX_CACHE_MAX_BLOCKS,
        )
    def init_default_start_pos():
        return csa_decode_start_set(
            batch=B, seq=S, compress_ratio=COMPRESS_RATIO,
            state_block_size=INNER_STATE_BLOCK_SIZE, cache_tile=CACHE_TILE)
    def init_start_pos():
        return resolve_start_positions(
            start_pos,
            batch=B,
            seq=S,
            max_seq_len=MAX_SEQ_LEN,
            default_fn=init_default_start_pos,
        )
    def init_position_ids():
        return position_ids_from_starts(init_start_pos(), seq=S)
    def init_kv_seq_lens():
        return kv_seq_lens_from_starts(init_start_pos(), seq=S)
    def init_inner_state_slot_mapping():
        return state_slot_mapping(
            init_position_ids(),
            init_inner_compress_state_block_table(),
            state_block_size=INNER_STATE_BLOCK_SIZE,
        )
    def init_idx_slot_mapping():
        positions = init_position_ids()
        return compressed_slot_mapping(
            positions,
            init_idx_block_table(),
            compress_ratio=COMPRESS_RATIO,
            block_size=BLOCK_SIZE,
        )

    wq_b_i8_T, wq_b_scale = gen_shared_weight(
        (IDX_N_HEADS * IDX_HEAD_DIM, Q_LORA), dequant_std=0.108, chan_cv=0.56)
    wq_b_i8 = wq_b_i8_T.t().contiguous()
    qr_i8, qr_scale = _int8_quant_per_row(init_qr())

    idx_kv_cache_bf16 = torch.rand(IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM).to(torch.bfloat16)
    idx_kv_i8, idx_kv_sc = _int8_quant_per_row(
        idx_kv_cache_bf16.float().reshape(IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, IDX_HEAD_DIM))
    idx_kv_i8 = idx_kv_i8.view(IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM)
    idx_kv_sc = idx_kv_sc.view(IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1)

    return [
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("qr", [T, Q_LORA], torch.int8, init_value=lambda: qr_i8),
        TensorSpec("qr_scale", [T, 1], torch.float32, init_value=lambda: qr_scale),
        TensorSpec("wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.int8, init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [IDX_N_HEADS * IDX_HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("weights_proj", [D, IDX_N_HEADS], torch.bfloat16, init_value=init_weights_proj),
        TensorSpec("cos", [B, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [B, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec("hadamard", [IDX_HEAD_DIM, IDX_HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec("inner_kv", [B, S, INNER_HEAD_DIM], torch.float32),
        TensorSpec("inner_compress_state", [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], torch.float32, init_value=init_inner_compress_state, is_output=True),
        TensorSpec("inner_compress_state_block_table", [B, INNER_STATE_MAX_BLOCKS], torch.int32, init_value=init_inner_compress_state_block_table),
        TensorSpec("inner_wkv", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wkv),
        TensorSpec("inner_wgate", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wgate),
        TensorSpec("inner_ape", [COMPRESS_RATIO, INNER_OUT_DIM], torch.float32, init_value=init_inner_ape),
        TensorSpec("inner_norm_w", [INNER_HEAD_DIM], torch.bfloat16, init_value=init_inner_norm_w),
        TensorSpec("idx_kv_cache", [IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], torch.int8, init_value=lambda: idx_kv_i8, is_output=True),
        TensorSpec("idx_kv_scale", [IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1], torch.float32, init_value=lambda: idx_kv_sc, is_output=True),
        TensorSpec("idx_block_table", [B, IDX_CACHE_MAX_BLOCKS], torch.int32, init_value=init_idx_block_table),
        # Output tails use -inf scores and -1 indices.
        TensorSpec("score", [B, S, SCORE_LEN], torch.float32, is_output=True),
        TensorSpec("topk_idxs", [B, S, SCORE_LEN], torch.int32, is_output=True),
        TensorSpec("position_ids", [B, S], torch.int32, init_value=init_position_ids),
        TensorSpec("idx_slot_mapping", [B, S], torch.int64, init_value=init_idx_slot_mapping),
        TensorSpec("inner_state_slot_mapping", [B, S], torch.int64, init_value=init_inner_state_slot_mapping),
        TensorSpec("kv_seq_lens", [B], torch.int32, init_value=init_kv_seq_lens),
        ScalarSpec("offset", torch.int32, OFFSET),
    ]


if __name__ == "__main__":
    import argparse
    from decode_indexer_compressor import (
        mapped_idx_cache_ratio_allclose,
        mapped_inner_state_ratio_allclose,
    )
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, default=0, choices=[0, 1, 2],
                        help="chip swimlane level: 0=off, 1=AICore timing, 2=+AICPU timing.")
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--start-pos", type=int, default=None,
                        help="Uniform fixture-only start_pos override for all batches; "
                             "default (unset) uses the canonical per-batch CSA set that includes the 8k point.")
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=indexer_test,
        specs=build_tensor_specs(args.start_pos),
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
            "score":        score_valid_compare,
            "topk_idxs":    decode_topk_compare,
            "inner_compress_state": mapped_inner_state_ratio_allclose(
                atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            # Compare rows selected by the current slot mappings.
            "idx_kv_cache": mapped_idx_cache_ratio_allclose(
                atol=1, rtol=0, max_error_ratio=0.01),
            "idx_kv_scale": mapped_idx_cache_ratio_allclose(
                atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
