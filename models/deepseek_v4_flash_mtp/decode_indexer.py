# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 decode Indexer: q projection, Hadamard INT8 quant, inner compressor, paged C8 score and Top-K."""


import pypto.language as pl

from config import (
    FLASH as M,
    DECODE_BATCH,
    DECODE_SEQ,
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    DECODE_IDX_BLOCK_NUM,
    FP32_NEG_INF,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
)
from decode_indexer_compressor import COMPRESS_STATE_TABLE_BLOCKS_DYN as INNER_STATE_TABLE_BLOCKS_DYN
from decode_indexer_compressor import indexer_compressor
from rope_interleave import rope_interleave

# Dynamic shape variables.
INNER_STATE_BLOCK_NUM_DYN = pl.dynamic("INNER_STATE_BLOCK_NUM_DYN")
IDX_CACHE_BLOCK_NUM_DYN = pl.dynamic("IDX_CACHE_BLOCK_NUM_DYN")
IDX_TABLE_BLOCKS_DYN = pl.dynamic("CSA_IDX_TABLE_BLOCKS_DYN")
SCORE_COLS_DYN = pl.dynamic("CSA_IDX_SCORE_COLS_DYN")

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
IDX_Q_DIM = IDX_N_HEADS * IDX_HEAD_DIM
WEIGHTS_SCALE = M.index_weights_scale
MAX_SEQ_LEN = M.max_position_embeddings
OFFSET = M.sliding_window
COMPRESS_RATIO = 4   # the indexer only runs on ratio-4 layers
IDX_STORAGE_BLOCK_SIZE = BLOCK_SIZE // COMPRESS_RATIO
IDX_TOPK = M.index_topk
IDX_MAX_ROWS = MAX_SEQ_LEN // COMPRESS_RATIO
IDX_CACHE_BLOCK_NUM = DECODE_IDX_BLOCK_NUM
INNER_OVERLAP = COMPRESS_RATIO == 4
INNER_COFF = 1 + int(INNER_OVERLAP)
INNER_HEAD_DIM = IDX_HEAD_DIM
INNER_OUT_DIM = INNER_COFF * INNER_HEAD_DIM
INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
INNER_STATE_PHYSICAL_BLOCKS = 65
INNER_STATE_BLOCK_NUM = INNER_STATE_PHYSICAL_BLOCKS
INNER_STATE_DIM = 2 * INNER_OUT_DIM
ROPE_ROW_BLOCK = S * IDX_N_HEADS   # q rows per batch

# tiling
Q_TILE = 256          # idx qr_proj K tile
Q_OUT_TILE = 1024     # idx qr_proj N tile per task
MM_N_TILE = 512       # idx qr_proj Mat N tile
MM_ROW_TILE = 16
T_PAD = ((T + MM_ROW_TILE - 1) // MM_ROW_TILE) * MM_ROW_TILE
assert T_PAD == MM_ROW_TILE, "weights_proj single-row-tile scope assumes decode T <= MM_ROW_TILE"
D_TILE = 512
# weights_proj: WEIGHTS_OK tasks, each a [MM_ROW_TILE, IDX_N_HEADS] matmul over one WEIGHTS_K_TILE K range,
# summed by a separate reduce scope (a zero-seed + atomic-add assemble races on the full-extent seed).
# The inner K loop is a pl.range: a 2-iteration pl.pipeline(stage=2) miscompiles over matmul.
WEIGHTS_OK = 4
WEIGHTS_K_TILE = D // WEIGHTS_OK
assert WEIGHTS_K_TILE % D_TILE == 0
QH_MM_TILE = 64       # q @ hadamard row tile; L0C caps QH_MM_TILE * IDX_HEAD_DIM * 4B <= 64KiB
QH_QUANT_TILE = 64
assert QH_QUANT_TILE == IDX_N_HEADS, "qr_hadamard_quant writes one per-token scale row per task"
QH_HEAD_DIM_TILE = 64
ROPE_ROW_TILE = 32    # qr_rope rows per SPMD block
CACHE_TILE = 64
REDUCE_TILE = BLOCK_SIZE // COMPRESS_RATIO   # C4 rows per cache page
SCORE_STACK_PAGES = 4
SCORE_STACK_TILE = SCORE_STACK_PAGES * REDUCE_TILE   # score rows per INT32 matmul + reduce
SCORE_AIV_LANES = 2
SCORE_TASK_LEAVES = 4   # max Top-K leaves per multi-leaf score task
# Top-K leaf: an exact Top-IDX_TOPK over TOPK_LEAF_TILE candidates; score columns span whole leaves.
TOPK_LEAF_TILE = 4096
TOPK_HALF_TILE = TOPK_LEAF_TILE // 2
TOPK_HALF_PAIR_OFFSET = 2 * TOPK_HALF_TILE
TOPK_PAIR_WIDTH = 2 * IDX_TOPK
assert IDX_TOPK <= TOPK_HALF_TILE, "per-half candidate list must cover the final topk width"
LEAF_STACKS = TOPK_LEAF_TILE // SCORE_STACK_TILE
assert LEAF_STACKS * SCORE_STACK_TILE == TOPK_LEAF_TILE, "score stacks must tile a Top-K leaf"
assert SCORE_AIV_LANES * TOPK_HALF_TILE == TOPK_LEAF_TILE, "a single-leaf AIV lane sorts at most one Top-K half"

# score cube/vector handshake
SCORE_TRANSFER_SLOTS = 3   # GM ring depth, in stack pairs
SCORE_READY_EVENT = 0
SCORE_DONE_EVENT = 1


@pl.jit.inline
def indexer(
    x: pl.Tensor[[B, S, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    # Interleave-duplicated (j>>1) cos and sign-folded sin, built once by the caller:
    #   cos[j] = cos_half[j>>1];  sin[j] = sin_half[j>>1] * sign[j], sign = [-1,+1,...]
    cos: pl.Tensor[[B, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],  # shared by q rotation and inner Compressor
    inner_kv: pl.Tensor[[B, S, INNER_HEAD_DIM], pl.FP32],
    inner_compress_state: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[B, INNER_STATE_TABLE_BLOCKS_DYN], pl.INT32],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.BF16],
    # C8 indexer cache: INT8 KV (quant-on-write) + per-position FP32 dequant scale.
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[B, IDX_TABLE_BLOCKS_DYN], pl.INT32],
    score: pl.Tensor[[T, SCORE_COLS_DYN], pl.FP32],
    topk_idxs: pl.Tensor[[T, IDX_TOPK], pl.INT32],
    position_ids: pl.Tensor[[B, S], pl.INT32],
    idx_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    offset: pl.Scalar[pl.INT32],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    qr_acc_pad = pl.create_tensor([T_PAD, IDX_Q_DIM], dtype=pl.INT32)
    for ot in pl.spmd(IDX_Q_DIM // Q_OUT_TILE, name_hint="idx_qr_proj_matmul", allow_early_resolve=True):
        o_base = ot * Q_OUT_TILE
        for ns in pl.range(0, Q_OUT_TILE, MM_N_TILE):
            qr_acc = pl.create_tensor([MM_ROW_TILE, MM_N_TILE], dtype=pl.INT32)
            for kb in pl.pipeline(0, Q_LORA // Q_TILE, stage=2):
                q0 = kb * Q_TILE
                qr_tile = pl.slice(qr, [T_PAD, Q_TILE], [0, q0], valid_shape=[T, Q_TILE])
                wq_tile = wq_b[q0 : q0 + Q_TILE, o_base + ns : o_base + ns + MM_N_TILE]
                qr_acc = pl.matmul_acc(qr_acc, qr_tile, wq_tile, init_cond=(q0 == 0))
            qr_acc_pad[0:T_PAD, o_base + ns : o_base + ns + MM_N_TILE] = qr_acc

    qr_proj = pl.create_tensor([T, IDX_Q_DIM], dtype=pl.FP32)
    for ot in pl.spmd(IDX_Q_DIM // Q_OUT_TILE, name_hint="idx_qr_proj_dequant", allow_early_resolve=True):
        o_base = ot * Q_OUT_TILE
        wq_scale = pl.reshape(wq_b_scale[o_base : o_base + Q_OUT_TILE], [1, Q_OUT_TILE])
        acc_fp32 = pl.cast(qr_acc_pad[0:T, o_base : o_base + Q_OUT_TILE], target_type=pl.FP32, mode="none")
        qr_dequant = pl.col_expand_mul(pl.row_expand_mul(acc_fp32, qr_scale[0:T, :]), wq_scale)
        qr_proj[0:T, o_base : o_base + Q_OUT_TILE] = qr_dequant

    # j^1 lane-swap gather index, shared by every qr_rope block
    rope_swap_idx_t = pl.create_tensor([ROPE_ROW_TILE, ROPE_HEAD_DIM], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_rope_swap_idx", allow_early_resolve=True):
        sw_ones = pl.full([ROPE_ROW_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        sw_j_i32 = pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32)
        sw_col = pl.col_expand_mul(sw_ones, pl.cast(sw_j_i32, target_type=pl.FP32))
        sw_dup_i32 = pl.cast(pl.mul(sw_col, 0.5), target_type=pl.INT32, mode="trunc")
        sw_dup_f = pl.cast(sw_dup_i32, target_type=pl.FP32)
        sw_lane = pl.sub(sw_col, pl.mul(sw_dup_f, 2.0))              # j%2
        sw_swap_f = pl.sub(pl.add(sw_col, 1.0), pl.mul(sw_lane, 2.0))   # j^1
        rope_swap_idx_t[0:ROPE_ROW_TILE, 0:ROPE_HEAD_DIM] = pl.cast(sw_swap_f, target_type=pl.INT32)

    # BF16 q: nope half rounded from the FP32 dequant, rope half rotated then rounded.
    #   out[j] = x[j]*cos_il[j] + x[j^1]*sin_il_signed[j]
    qr_proj_flat = pl.reshape(qr_proj, [T * IDX_N_HEADS, IDX_HEAD_DIM])
    qr_bf16 = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.BF16)
    for idx in pl.spmd(T * IDX_N_HEADS // ROPE_ROW_TILE, name_hint="qr_rope", allow_early_resolve=True):
        o0 = idx * ROPE_ROW_TILE
        batch_idx = o0 // ROPE_ROW_BLOCK
        rope_swap_idx = rope_swap_idx_t[0:ROPE_ROW_TILE, 0:ROPE_HEAD_DIM]
        cos_row = cos[batch_idx : batch_idx + 1, 0 : ROPE_HEAD_DIM]
        sin_row = sin[batch_idx : batch_idx + 1, 0 : ROPE_HEAD_DIM]
        qr_nope_slice = qr_proj_flat[o0 : o0 + ROPE_ROW_TILE, 0 : IDX_NOPE_HEAD_DIM]
        qr_rope_slice = qr_proj_flat[o0 : o0 + ROPE_ROW_TILE, IDX_NOPE_HEAD_DIM : IDX_HEAD_DIM]
        qr_swapped = pl.gather(qr_rope_slice, dim=-1, index=rope_swap_idx)
        rope_rot = pl.add(pl.col_expand_mul(qr_rope_slice, cos_row), pl.col_expand_mul(qr_swapped, sin_row))
        qr_nope_bf16 = pl.cast(qr_nope_slice, target_type=pl.BF16, mode="rint")
        qr_vec = pl.concat(qr_nope_bf16, pl.cast(rope_rot, target_type=pl.BF16, mode="rint"))
        qr_bf16[o0 : o0 + ROPE_ROW_TILE, :] = qr_vec

    # cube-only q @ hadamard into GM; the vector amax/quant runs as its own scope
    qh_acc_gm = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.FP32)
    for idx in pl.spmd(T * IDX_N_HEADS // QH_MM_TILE, name_hint="qr_hadamard_matmul", allow_early_resolve=True):
        o0 = idx * QH_MM_TILE
        qh_acc = pl.matmul(qr_bf16[o0 : o0 + QH_MM_TILE, :], hadamard, out_dtype=pl.FP32)
        qh_acc_gm[o0 : o0 + QH_MM_TILE, :] = qh_acc

    qr_hadamard_i8 = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.INT8)
    qr_hadamard_scale_dq = pl.create_tensor([T, IDX_N_HEADS], dtype=pl.FP32)
    for idx in pl.spmd(T * IDX_N_HEADS // QH_QUANT_TILE, name_hint="qr_hadamard_quant", allow_early_resolve=True):
        o0 = idx * QH_QUANT_TILE
        qh_amax = pl.full([1, QH_QUANT_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for h0 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
            qh_a_f32 = qh_acc_gm[o0 : o0 + QH_QUANT_TILE, h0 : h0 + QH_HEAD_DIM_TILE]
            qh_a_abs = pl.maximum(qh_a_f32, pl.neg(qh_a_f32))
            qh_a_max = pl.reshape(pl.row_max(qh_a_abs), [1, QH_QUANT_TILE])
            qh_amax = pl.maximum(qh_amax, qh_a_max)
        qh_scale_quant_row = pl.div(pl.full([1, QH_QUANT_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), qh_amax)
        qr_hadamard_scale_dq[idx : idx + 1, :] = pl.recip(qh_scale_quant_row)
        qh_scale_quant = pl.reshape(qh_scale_quant_row, [QH_QUANT_TILE, 1])
        for h1 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
            qh_q_f32 = qh_acc_gm[o0 : o0 + QH_QUANT_TILE, h1 : h1 + QH_HEAD_DIM_TILE]
            qh_q_scaled = pl.row_expand_mul(qh_q_f32, qh_scale_quant)
            qh_q_i32 = pl.cast(qh_q_scaled, target_type=pl.INT32, mode="rint")
            qh_q_half = pl.cast(qh_q_i32, target_type=pl.FP16, mode="round")
            qh_i8 = pl.cast(qh_q_half, target_type=pl.INT8, mode="trunc")
            qr_hadamard_i8[o0 : o0 + QH_QUANT_TILE, h1 : h1 + QH_HEAD_DIM_TILE] = qh_i8

    # deferred behind the caller's rms_norm barrier (late_dep)
    x_flat = pl.reshape(x, [T, D])
    weights_partial = pl.create_tensor([WEIGHTS_OK * MM_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32)
    with pl.spmd(WEIGHTS_OK, name_hint="weights_proj", deps=[late_dep]) as _weights_tid:
        kb = pl.tile.get_block_idx()
        k_base = kb * WEIGHTS_K_TILE
        weights_acc = pl.create_tensor([MM_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32)
        for db in pl.range(WEIGHTS_K_TILE // D_TILE):
            d0 = k_base + db * D_TILE
            x_tile = pl.slice(x_flat, [MM_ROW_TILE, D_TILE], [0, d0], valid_shape=[pl.min(MM_ROW_TILE, T), D_TILE])
            weights_proj_tile = weights_proj[d0 : d0 + D_TILE, :]
            weights_acc = pl.matmul_acc(weights_acc, x_tile, weights_proj_tile, init_cond=(db == 0))
        weights_partial[kb * MM_ROW_TILE : kb * MM_ROW_TILE + MM_ROW_TILE, :] = weights_acc

    weights = pl.create_tensor([T_PAD, IDX_N_HEADS], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="weights_proj_reduce", allow_early_resolve=True):
        w_sum = weights_partial[0:MM_ROW_TILE, :]
        for kb in pl.unroll(1, WEIGHTS_OK):
            w_sum = pl.add(w_sum, weights_partial[kb * MM_ROW_TILE : kb * MM_ROW_TILE + MM_ROW_TILE, :])
        weights[0:MM_ROW_TILE, :] = pl.mul(w_sum, WEIGHTS_SCALE)

    indexer_compressor(
        x, inner_kv,
        inner_compress_state, inner_compress_state_block_table,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        cos, sin, hadamard, idx_kv_cache, idx_kv_scale,
        position_ids, idx_slot_mapping, inner_state_slot_mapping,
        late_dep,
    )

    idx_block_num = pl.tensor.dim(idx_kv_cache, 0)
    idx_table_blocks = pl.tensor.dim(idx_block_table, 1)
    score_cols = pl.tensor.dim(score, 1)
    score_rows = pl.min(pl.min(idx_table_blocks * REDUCE_TILE, IDX_MAX_ROWS), score_cols)
    idx_table_len = B * idx_table_blocks
    kv_cache_i8_flat = pl.reshape(idx_kv_cache, [idx_block_num * IDX_STORAGE_BLOCK_SIZE, IDX_HEAD_DIM])
    kv_scale_flat = pl.reshape(idx_kv_scale, [idx_block_num * IDX_STORAGE_BLOCK_SIZE, 1])
    idx_block_table_flat = pl.reshape(idx_block_table, [idx_table_len])

    # score: each stack gathers SCORE_STACK_PAGES paged C8 pages into one INT32 matmul; tail stacks repeat
    # the last real page, and entries past the visible length are masked by Top-K.
    topk_leaves = score_cols // TOPK_LEAF_TILE
    if topk_leaves == 1:
        # Single leaf: one task per query. The AIC loop scores the query's stacks into a GM ring; AIV lane h
        # reduces half h of the stacks and sorts it into a Top-IDX_TOPK pair row; lane 0 then merges both rows.
        half_pairs = pl.create_tensor([T * SCORE_AIV_LANES, TOPK_PAIR_WIDTH], dtype=pl.FP32)
        q_transfer_rows = T * SCORE_TRANSFER_SLOTS * SCORE_AIV_LANES * SCORE_STACK_TILE
        q_acc_transfer = pl.create_tensor([q_transfer_rows, IDX_N_HEADS], dtype=pl.INT32)
        q_ffts_workspace = pl.create_tensor([256], dtype=pl.INT64)
        with pl.spmd(T, name_hint="score_topk", allow_early_resolve=True):
            q_t = pl.tile.get_block_idx()
            pl.system.set_ffts(q_ffts_workspace)
            q_b = q_t // S
            q_s = q_t - q_b * S
            q_cache_len = pl.read(kv_seq_lens, [q_b]) // COMPRESS_RATIO
            q_pos = pl.read(position_ids, [q_b, q_s])
            q_visible = pl.min(pl.min(q_cache_len, (q_pos + 1) // COMPRESS_RATIO), score_rows)
            q_cblk = (q_visible + REDUCE_TILE - 1) // REDUCE_TILE
            q_sblk = (q_cblk + SCORE_STACK_PAGES - 1) // SCORE_STACK_PAGES
            q_lane0_stacks = (q_sblk + 1) // SCORE_AIV_LANES
            q_lane1_stacks = q_sblk - q_lane0_stacks
            q_lane0_valid = pl.min(q_visible, q_lane0_stacks * SCORE_STACK_TILE)
            q_lane1_valid = q_visible - q_lane0_valid
            q_qb = q_b * S * IDX_N_HEADS
            q_tb = q_b * S
            q_slot_base = q_t * SCORE_TRANSFER_SLOTS * SCORE_AIV_LANES * SCORE_STACK_TILE

            q_qr_full = qr_hadamard_i8[q_qb + q_s * IDX_N_HEADS : q_qb + (q_s + 1) * IDX_N_HEADS, 0 : IDX_HEAD_DIM]
            for qm_it in pl.range(q_lane0_stacks):
                if qm_it >= SCORE_TRANSFER_SLOTS:
                    pl.system.sync_wait(SCORE_DONE_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                for qm_lane in pl.unroll(SCORE_AIV_LANES):
                    qm_stacks = q_lane0_stacks + qm_lane * (q_lane1_stacks - q_lane0_stacks)
                    if qm_it < qm_stacks:
                        qm_sb = qm_lane * q_lane0_stacks + qm_it
                        q_kv_i8_mat = pl.create_l1([SCORE_STACK_TILE, IDX_HEAD_DIM], pl.INT8)
                        for q_p in pl.unroll(SCORE_STACK_PAGES):
                            q_cb = pl.min(qm_sb * SCORE_STACK_PAGES + q_p, q_cblk - 1)
                            q_blk_id = pl.cast(pl.read(idx_block_table_flat, [q_b * idx_table_blocks + q_cb]), pl.INDEX)
                            q_kv0 = q_blk_id * IDX_STORAGE_BLOCK_SIZE
                            q_kv_i8_mat = pl.gather_row(
                                q_kv_i8_mat, kv_cache_i8_flat,
                                [q_p * REDUCE_TILE, 0], [q_kv0, 0], [REDUCE_TILE, IDX_HEAD_DIM],
                            )
                        q_score_acc = pl.matmul(q_kv_i8_mat, q_qr_full, out_dtype=pl.INT32, b_trans=True)
                        qm_slot_row = q_slot_base + ((qm_it % SCORE_TRANSFER_SLOTS) * SCORE_AIV_LANES + qm_lane) * SCORE_STACK_TILE
                        q_acc_transfer[qm_slot_row : qm_slot_row + SCORE_STACK_TILE, :] = q_score_acc
                pl.system.sync_set(SCORE_READY_EVENT, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
            for qd_it in pl.range(pl.min(q_lane0_stacks, SCORE_TRANSFER_SLOTS)):
                pl.system.sync_wait(SCORE_DONE_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
            # both halves sorted -> release lane 0's merge
            pl.system.sync_wait(SCORE_DONE_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
            pl.system.sync_set(SCORE_READY_EVENT, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)

            for q_aiv in pl.split_aiv(SCORE_AIV_LANES, mode=pl.SplitMode.NONE):
                pl.system.set_ffts(q_ffts_workspace)
                qv_sb0 = q_aiv * q_lane0_stacks
                qv_stacks = q_lane0_stacks + q_aiv * (q_lane1_stacks - q_lane0_stacks)
                qv_valid = q_lane0_valid + q_aiv * (q_lane1_valid - q_lane0_valid)
                q_row = q_tb + q_s
                q_qh_scale_s = pl.load(
                    qr_hadamard_scale_dq, [q_row, 0], [1, IDX_N_HEADS], target_memory=pl.MemorySpace.Vec,
                )
                q_weights_row_s = pl.load(weights, [q_row, 0], [1, IDX_N_HEADS], target_memory=pl.MemorySpace.Vec)
                # relu(x * qh_scale) * weights == relu(x) * (qh_scale * weights), qh_scale > 0
                q_head_coef_s = pl.mul(q_qh_scale_s, q_weights_row_s)
                q_reduce_tmp = pl.create_tile(
                    [SCORE_STACK_TILE, IDX_N_HEADS], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec,
                )
                for qv_it in pl.range(q_lane0_stacks):
                    pl.system.sync_wait(SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    if qv_it < qv_stacks:
                        qv_sb = qv_sb0 + qv_it
                        qv_slot_row = q_slot_base + ((qv_it % SCORE_TRANSFER_SLOTS) * SCORE_AIV_LANES + q_aiv) * SCORE_STACK_TILE
                        q_acc_vec = pl.load(
                            q_acc_transfer, [qv_slot_row, 0], [SCORE_STACK_TILE, IDX_N_HEADS],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        q_kv_dq_vec = pl.create_tile(
                            [SCORE_STACK_TILE, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec,
                        )
                        for q_vp in pl.unroll(SCORE_STACK_PAGES):
                            qv_cb = pl.min(qv_sb * SCORE_STACK_PAGES + q_vp, q_cblk - 1)
                            qv_blk_id = pl.cast(pl.read(idx_block_table_flat, [q_b * idx_table_blocks + qv_cb]), pl.INDEX)
                            qv_kv0 = qv_blk_id * IDX_STORAGE_BLOCK_SIZE
                            # paged per-position dequant scale
                            q_kv_dq_vec = pl.gather_row(
                                q_kv_dq_vec, kv_scale_flat,
                                [q_vp * REDUCE_TILE, 0], [qv_kv0, 0], [REDUCE_TILE, 1],
                            )
                        q_score_tile = pl.cast(q_acc_vec, target_type=pl.FP32, mode="none")
                        q_relu_score = pl.maximum(q_score_tile, 0.0)
                        q_weighted_score = pl.col_expand_mul(q_relu_score, q_head_coef_s)
                        # per-position dequant q_kv_dq_vec applied after the head-sum
                        q_weighted_row = pl.mul(pl.row_sum(q_weighted_score, q_reduce_tmp), q_kv_dq_vec)
                        q_weighted_s = pl.reshape(q_weighted_row, [1, SCORE_STACK_TILE])
                        pl.store(q_weighted_s, [q_t, qv_sb * SCORE_STACK_TILE], score)
                    pl.system.sync_set(SCORE_DONE_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)

                half_row = q_t * SCORE_AIV_LANES + q_aiv
                if qv_valid > 0:
                    half0 = qv_sb0 * SCORE_STACK_TILE
                    half_score_raw = score[q_t : q_t + 1, half0 : half0 + TOPK_HALF_TILE]
                    half_score = pl.fillpad(pl.set_validshape(half_score_raw, 1, qv_valid), pad_value=pl.PadValue.min)
                    half_neg_inf = pl.full([1, TOPK_HALF_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
                    half_score = pl.maximum(half_score, half_neg_inf)
                    half_idx_base = pl.arange(0, [1, TOPK_HALF_TILE], dtype=pl.INT32)
                    half_idx_i32 = pl.add(half_idx_base, pl.cast(half0, pl.INT32))
                    half_sorted = pl.sort32(half_score, pl.reinterpret_view(half_idx_i32, pl.UINT32))
                    half_sorted = pl.mrgsort(half_sorted, block_len=64)
                    half_sorted = pl.mrgsort(half_sorted, block_len=256)
                    half_sorted = pl.mrgsort(half_sorted, block_len=1024)
                    half_pairs[half_row : half_row + 1, 0:TOPK_PAIR_WIDTH] = half_sorted[:, 0:TOPK_PAIR_WIDTH]
                pl.system.sync_set(SCORE_DONE_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                pl.system.sync_wait(SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)

                if q_aiv == 0:
                    q_invalid = pl.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
                    topk_idxs[q_t : q_t + 1, :] = q_invalid
                    if q_visible > 0:
                        q_pairs = half_pairs[half_row : half_row + 1, 0:TOPK_PAIR_WIDTH]
                        if q_lane1_valid > 0:
                            q_pairs1 = half_pairs[half_row + 1 : half_row + 2, 0:TOPK_PAIR_WIDTH]
                            q_merged = pl.mrgsort(q_pairs, q_pairs1)
                            q_pairs = q_merged[:, 0:TOPK_PAIR_WIDTH]
                        q_idxs = pl.gather(q_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                        q_valid_topk = pl.min(IDX_TOPK, q_visible)
                        q_idxs_valid = pl.set_validshape(q_idxs, 1, q_valid_topk)
                        q_topk_row = pl.add(q_idxs_valid, pl.cast(offset, target_type=pl.INT32))
                        topk_idxs[q_t : q_t + 1, 0:IDX_TOPK] = q_topk_row
    else:
        # Multi-leaf: one task per SCORE_TASK_LEAVES of the batch's longest visible leaf count per query. A
        # task's two AIV lanes split the query's leaves into contiguous runs; the AIC loop scores both lanes'
        # stacks into a GM ring, and a lane reduces its stacks and sorts each finished leaf into its pair row.
        for bi, (max_kv_len,) in pl.range(B, init_values=(0,)):
            max_kv_len = pl.yield_(pl.max(max_kv_len, pl.read(kv_seq_lens, [bi])))
        max_visible = pl.min(max_kv_len // COMPRESS_RATIO, score_rows)
        active_leaves = pl.max((max_visible + TOPK_LEAF_TILE - 1) // TOPK_LEAF_TILE, 1)
        query_tasks = (active_leaves + SCORE_TASK_LEAVES - 1) // SCORE_TASK_LEAVES
        query_lanes = query_tasks * SCORE_AIV_LANES
        leaf_tasks = T * query_tasks
        leaf_rows = T * topk_leaves
        leaf_pairs = pl.create_tensor([leaf_rows, TOPK_PAIR_WIDTH], dtype=pl.FP32)
        leaf_transfer_rows = leaf_tasks * SCORE_TRANSFER_SLOTS * SCORE_AIV_LANES * SCORE_STACK_TILE
        leaf_acc_transfer = pl.create_tensor([leaf_transfer_rows, IDX_N_HEADS], dtype=pl.INT32)
        leaf_ffts_workspace = pl.create_tensor([256], dtype=pl.INT64)
        with pl.spmd(leaf_tasks, name_hint="score_topk_leaf", allow_early_resolve=True) as leaf_tid:
            task_item = pl.tile.get_block_idx()
            pl.system.set_ffts(leaf_ffts_workspace)
            leaf_t = task_item // query_tasks
            task_g = task_item - leaf_t * query_tasks
            leaf_b = leaf_t // S
            leaf_s = leaf_t - leaf_b * S
            leaf_cache_len = pl.read(kv_seq_lens, [leaf_b]) // COMPRESS_RATIO
            leaf_pos = pl.read(position_ids, [leaf_b, leaf_s])
            leaf_visible = pl.min(pl.min(leaf_cache_len, (leaf_pos + 1) // COMPRESS_RATIO), score_rows)
            leaf_cblk = (leaf_visible + REDUCE_TILE - 1) // REDUCE_TILE
            leaf_sblk = (leaf_cblk + SCORE_STACK_PAGES - 1) // SCORE_STACK_PAGES
            query_leaves = (leaf_visible + TOPK_LEAF_TILE - 1) // TOPK_LEAF_TILE
            lane_base = query_leaves // query_lanes
            lane_extra = query_leaves - lane_base * query_lanes
            lane0 = task_g * SCORE_AIV_LANES
            lane0_leaf0 = lane0 * lane_base + pl.min(lane0, lane_extra)
            lane0_leaves = lane_base + pl.min(pl.max(lane_extra - lane0, 0), 1)
            lane0_stacks = pl.max(pl.min(leaf_sblk - lane0_leaf0 * LEAF_STACKS, lane0_leaves * LEAF_STACKS), 0)
            lane1 = lane0 + 1
            lane1_leaf0 = lane1 * lane_base + pl.min(lane1, lane_extra)
            lane1_leaves = lane_base + pl.min(pl.max(lane_extra - lane1, 0), 1)
            lane1_stacks = pl.max(pl.min(leaf_sblk - lane1_leaf0 * LEAF_STACKS, lane1_leaves * LEAF_STACKS), 0)
            task_iters = pl.max(lane0_stacks, lane1_stacks)
            leaf_qb = leaf_b * S * IDX_N_HEADS
            leaf_tb = leaf_b * S
            task_slot_base = task_item * SCORE_TRANSFER_SLOTS * SCORE_AIV_LANES * SCORE_STACK_TILE

            qr_full = qr_hadamard_i8[leaf_qb + leaf_s * IDX_N_HEADS : leaf_qb + (leaf_s + 1) * IDX_N_HEADS, 0 : IDX_HEAD_DIM]
            for mm_it in pl.range(task_iters):
                if mm_it >= SCORE_TRANSFER_SLOTS:
                    pl.system.sync_wait(SCORE_DONE_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                for mm_lane in pl.unroll(SCORE_AIV_LANES):
                    mm_leaf0 = lane0_leaf0 + mm_lane * (lane1_leaf0 - lane0_leaf0)
                    mm_stacks = lane0_stacks + mm_lane * (lane1_stacks - lane0_stacks)
                    if mm_it < mm_stacks:
                        mm_sb = mm_leaf0 * LEAF_STACKS + mm_it
                        lf_kv_i8_mat = pl.create_l1([SCORE_STACK_TILE, IDX_HEAD_DIM], pl.INT8)
                        for lf_p in pl.unroll(SCORE_STACK_PAGES):
                            lf_cb = pl.min(mm_sb * SCORE_STACK_PAGES + lf_p, leaf_cblk - 1)
                            lf_idx_blk_id = pl.cast(pl.read(idx_block_table_flat, [leaf_b * idx_table_blocks + lf_cb]), pl.INDEX)
                            lf_kv0 = lf_idx_blk_id * IDX_STORAGE_BLOCK_SIZE
                            lf_kv_i8_mat = pl.gather_row(
                                lf_kv_i8_mat, kv_cache_i8_flat,
                                [lf_p * REDUCE_TILE, 0], [lf_kv0, 0], [REDUCE_TILE, IDX_HEAD_DIM],
                            )
                        lf_score_acc_red = pl.matmul(lf_kv_i8_mat, qr_full, out_dtype=pl.INT32, b_trans=True)
                        mm_slot_row = task_slot_base + ((mm_it % SCORE_TRANSFER_SLOTS) * SCORE_AIV_LANES + mm_lane) * SCORE_STACK_TILE
                        leaf_acc_transfer[mm_slot_row : mm_slot_row + SCORE_STACK_TILE, :] = lf_score_acc_red
                pl.system.sync_set(SCORE_READY_EVENT, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
            for drain_it in pl.range(pl.min(task_iters, SCORE_TRANSFER_SLOTS)):
                pl.system.sync_wait(SCORE_DONE_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)

            for aiv_id in pl.split_aiv(SCORE_AIV_LANES, mode=pl.SplitMode.NONE):
                pl.system.set_ffts(leaf_ffts_workspace)
                lane_leaf0 = lane0_leaf0 + aiv_id * (lane1_leaf0 - lane0_leaf0)
                lane_stacks = lane0_stacks + aiv_id * (lane1_stacks - lane0_stacks)
                lf_row = leaf_tb + leaf_s
                lf_qh_scale_s = pl.load(
                    qr_hadamard_scale_dq, [lf_row, 0], [1, IDX_N_HEADS], target_memory=pl.MemorySpace.Vec,
                )
                lf_weights_row_s = pl.load(weights, [lf_row, 0], [1, IDX_N_HEADS], target_memory=pl.MemorySpace.Vec)
                # relu(x * qh_scale) * weights == relu(x) * (qh_scale * weights), qh_scale > 0
                lf_head_coef_s = pl.mul(lf_qh_scale_s, lf_weights_row_s)
                lf_score_reduce_tmp = pl.create_tile(
                    [SCORE_STACK_TILE, IDX_N_HEADS], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec,
                )
                for red_it in pl.range(task_iters):
                    pl.system.sync_wait(SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    red_sb = lane_leaf0 * LEAF_STACKS + red_it
                    if red_it < lane_stacks:
                        red_slot_row = task_slot_base + ((red_it % SCORE_TRANSFER_SLOTS) * SCORE_AIV_LANES + aiv_id) * SCORE_STACK_TILE
                        lf_score_acc_vec = pl.load(
                            leaf_acc_transfer, [red_slot_row, 0], [SCORE_STACK_TILE, IDX_N_HEADS],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        lf_kv_dq_vec = pl.create_tile(
                            [SCORE_STACK_TILE, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec,
                        )
                        for lf_p in pl.unroll(SCORE_STACK_PAGES):
                            lf_red_cb = pl.min(red_sb * SCORE_STACK_PAGES + lf_p, leaf_cblk - 1)
                            lf_red_blk_id = pl.cast(pl.read(idx_block_table_flat, [leaf_b * idx_table_blocks + lf_red_cb]), pl.INDEX)
                            lf_red_kv0 = lf_red_blk_id * IDX_STORAGE_BLOCK_SIZE
                            # paged per-position dequant scale
                            lf_kv_dq_vec = pl.gather_row(
                                lf_kv_dq_vec, kv_scale_flat,
                                [lf_p * REDUCE_TILE, 0], [lf_red_kv0, 0], [REDUCE_TILE, 1],
                            )
                        lf_score_tile_vec = pl.cast(lf_score_acc_vec, target_type=pl.FP32, mode="none")
                        lf_relu_score_vec = pl.maximum(lf_score_tile_vec, 0.0)
                        lf_weighted_score_vec = pl.col_expand_mul(lf_relu_score_vec, lf_head_coef_s)
                        # per-position dequant lf_kv_dq_vec applied after the head-sum
                        lf_head_sum = pl.row_sum(lf_weighted_score_vec, lf_score_reduce_tmp)
                        lf_weighted_score_row = pl.mul(lf_head_sum, lf_kv_dq_vec)
                        lf_weighted_score_s = pl.reshape(lf_weighted_score_row, [1, SCORE_STACK_TILE])
                        pl.store(lf_weighted_score_s, [leaf_t, red_sb * SCORE_STACK_TILE], score)
                    pl.system.sync_set(SCORE_DONE_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)

                    if red_it < lane_stacks:
                        # 0 once red_it closes a leaf or the lane's last stack
                        leaf_boundary = pl.min((red_it + 1) % LEAF_STACKS, lane_stacks - red_it - 1)
                        if leaf_boundary == 0:
                            leaf0 = (red_sb // LEAF_STACKS) * TOPK_LEAF_TILE
                            leaf_valid = pl.min(TOPK_LEAF_TILE, leaf_visible - leaf0)
                            leaf_score_raw = score[leaf_t : leaf_t + 1, leaf0 : leaf0 + TOPK_LEAF_TILE]
                            leaf_score_valid = pl.set_validshape(leaf_score_raw, 1, leaf_valid)
                            leaf_score = pl.fillpad(leaf_score_valid, pad_value=pl.PadValue.min)
                            leaf_neg_inf = pl.full([1, TOPK_LEAF_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
                            leaf_score = pl.maximum(leaf_score, leaf_neg_inf)
                            leaf_idx_base = pl.arange(0, [1, TOPK_LEAF_TILE], dtype=pl.INT32)
                            leaf_idx_i32 = pl.add(leaf_idx_base, pl.cast(leaf0, pl.INT32))
                            leaf_sorted = pl.sort32(leaf_score, pl.reinterpret_view(leaf_idx_i32, pl.UINT32))
                            leaf_sorted = pl.mrgsort(leaf_sorted, block_len=64)
                            leaf_sorted = pl.mrgsort(leaf_sorted, block_len=256)
                            leaf_sorted = pl.mrgsort(leaf_sorted, block_len=1024)
                            leaf_half0 = leaf_sorted[:, 0:TOPK_PAIR_WIDTH]
                            leaf_half1 = leaf_sorted[:, TOPK_HALF_PAIR_OFFSET : TOPK_HALF_PAIR_OFFSET + TOPK_PAIR_WIDTH]
                            leaf_merged = pl.mrgsort(leaf_half0, leaf_half1)
                            leaf_row = leaf_t * topk_leaves + red_sb // LEAF_STACKS
                            leaf_pairs[leaf_row : leaf_row + 1, 0:TOPK_PAIR_WIDTH] = leaf_merged[:, 0:TOPK_PAIR_WIDTH]

        for merge_t in pl.spmd(T, name_hint="topk_merge", deps=[leaf_tid], allow_early_resolve=True):
            merge_invalid = pl.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
            topk_idxs[merge_t : merge_t + 1, :] = merge_invalid
            merge_b = merge_t // S
            merge_s = merge_t - merge_b * S
            merge_cache_len = pl.read(kv_seq_lens, [merge_b]) // COMPRESS_RATIO
            merge_pos = pl.read(position_ids, [merge_b, merge_s])
            merge_visible = pl.min(pl.min(merge_cache_len, (merge_pos + 1) // COMPRESS_RATIO), score_rows)
            if merge_visible > 0:
                merge_base = merge_t * topk_leaves
                merge_leaf_count = (merge_visible + TOPK_LEAF_TILE - 1) // TOPK_LEAF_TILE
                for merge_leaf in pl.range(1, merge_leaf_count):
                    merge_row = merge_base + merge_leaf
                    merge_left = leaf_pairs[merge_base : merge_base + 1, 0:TOPK_PAIR_WIDTH]
                    merge_right = leaf_pairs[merge_row : merge_row + 1, 0:TOPK_PAIR_WIDTH]
                    merge_both = pl.mrgsort(merge_left, merge_right)
                    leaf_pairs[merge_base : merge_base + 1, 0:TOPK_PAIR_WIDTH] = merge_both[:, 0:TOPK_PAIR_WIDTH]
                merge_pairs = leaf_pairs[merge_base : merge_base + 1, 0:TOPK_PAIR_WIDTH]
                merge_idxs = pl.gather(merge_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                merge_valid_topk = pl.min(IDX_TOPK, merge_visible)
                merge_idxs_valid = pl.set_validshape(merge_idxs, 1, merge_valid_topk)
                merge_topk_row = pl.add(merge_idxs_valid, pl.cast(offset, target_type=pl.INT32))
                topk_idxs[merge_t : merge_t + 1, 0:IDX_TOPK] = merge_topk_row

    return score, topk_idxs


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
    inner_compress_state: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[B, INNER_STATE_TABLE_BLOCKS_DYN], pl.INT32],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[B, IDX_TABLE_BLOCKS_DYN], pl.INT32],
    score: pl.Out[pl.Tensor[[T, SCORE_COLS_DYN], pl.FP32]],
    topk_idxs: pl.Out[pl.Tensor[[T, IDX_TOPK], pl.INT32]],
    position_ids: pl.Tensor[[B, S], pl.INT32],
    idx_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    offset: pl.Scalar[pl.INT32],
):
    inner_compress_state_block_table.bind_dynamic(1, INNER_STATE_TABLE_BLOCKS_DYN)
    idx_block_table.bind_dynamic(1, IDX_TABLE_BLOCKS_DYN)
    score.bind_dynamic(1, SCORE_COLS_DYN)
    # standalone rms_norm barrier with no producer
    late_dep = pl.system.task_dummy(deps=[])
    # interleave-duplicated cos and sign-folded sin from the half-width cos/sin inputs
    cos_il = pl.create_tensor([B, ROPE_HEAD_DIM], dtype=pl.FP32)
    sin_signed = pl.create_tensor([B, ROPE_HEAD_DIM], dtype=pl.FP32)
    rope_interleave(cos, sin, cos_il, sin_signed)
    indexer(
        x, qr, qr_scale,
        wq_b, wq_b_scale, weights_proj,
        cos_il, sin_signed, hadamard,
        inner_kv, inner_compress_state, inner_compress_state_block_table,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        idx_kv_cache, idx_kv_scale, idx_block_table,
        score, topk_idxs,
        position_ids, idx_slot_mapping, inner_state_slot_mapping, kv_seq_lens,
        offset,
        late_dep,
    )
    return score, idx_kv_cache, idx_kv_scale, topk_idxs


def gen_shared_weight(shape, dequant_std, chan_cv):
    """Per-output-channel INT8 weight + FP32 scale re-quantized from a simulated MXFP8 (e4m3, 128x128 E8M0) grid.

    ``shape`` last dim = reduction (in) dim; leading dims map to the scale shape ([out, in] -> scale [out]).
    ``chan_cv`` is the log-space per-output-channel gain std; ``dequant_std`` sets the dequantized weight std.
    """
    import torch

    FP8_MAX, TINY = 448.0, 1e-20

    def sim_fp8(W, block=128):   # e4m3 + 128x128-block E8M0 (round-up) scale on (out, in)
        out, inn = W.shape
        Wb = W.reshape(out // block, block, inn // block, block)
        block_amax = Wb.abs().amax(dim=(1, 3), keepdim=True)
        scale = torch.exp2(torch.ceil(torch.log2((block_amax / FP8_MAX).clamp_min(TINY))))
        q = (Wb / scale).to(torch.float8_e4m3fn).float() * scale
        return q.reshape(out, inn)

    W = torch.randn(*shape) * torch.exp(chan_cv * torch.randn(*shape[:-1], 1))  # per-channel gain
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
    # W8A8C16: q and the Indexer cache are INT8 per row for the score matmul, dequantized with q_scale * kv_scale.

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

    # C8 cache: INT8 KV + per-position dequant scale
    idx_kv_cache_i8 = tensors["idx_kv_cache"]
    idx_kv_scale = tensors["idx_kv_scale"].float()
    idx_block_table = tensors["idx_block_table"]
    score_cols = tensors["score"].shape[-1]
    score_rows = min(idx_block_table.shape[1] * REDUCE_TILE, IDX_MAX_ROWS, score_cols)
    score_full = torch.full((bsz, seqlen, score_cols), FP32_NEG_INF, dtype=torch.float32)
    topk_idxs = torch.full((bsz, seqlen, IDX_TOPK), -1, dtype=torch.int32)
    q_i8, q_scale = int8_quant_per_row(q.reshape(B * S * IDX_N_HEADS, IDX_HEAD_DIM))
    q_i8 = q_i8.view(B, S, IDX_N_HEADS, IDX_HEAD_DIM)
    q_scale = q_scale.view(B, S, IDX_N_HEADS, 1)

    for b in range(bsz):
        cache_len = int(kv_seq_lens[b].item()) // ratio
        if cache_len <= 0:
            continue

        kv_i8_rows = []
        kv_scale_rows = []
        for slot in range(cache_len):
            block_id = int(idx_block_table[b, slot // REDUCE_TILE].item())
            kv_i8_rows.append(idx_kv_cache_i8[block_id, slot % REDUCE_TILE, 0])
            kv_scale_rows.append(idx_kv_scale[block_id, slot % REDUCE_TILE, 0, 0])
        kv_i8 = torch.stack(kv_i8_rows, dim=0).view(cache_len, IDX_HEAD_DIM)
        kv_scale = torch.stack(kv_scale_rows, dim=0).view(cache_len, 1)
        score_i32 = torch.einsum("shd,td->sht", q_i8[b].to(torch.int32), kv_i8.to(torch.int32))
        score = score_i32.float() * q_scale[b]
        score = (torch.relu(score) * weights[b].unsqueeze(-1)).sum(dim=1)
        score = score * kv_scale.view(1, cache_len)
        for s in range(seqlen):
            visible_len = min(cache_len, int(tensors["position_ids"][b, s].item() + 1) // ratio, score_rows)
            if visible_len <= 0:
                continue
            score_full[b, s, :visible_len] = score[s, :visible_len].to(torch.float32)
            k = min(IDX_TOPK, visible_len)
            _, idx = score[s, :visible_len].topk(k, dim=-1)
            topk_idxs[b, s, :k] = idx.to(torch.int32)
            topk_idxs[b, s, :k] += offset

    tensors["score"][:] = score_full.view(T, score_cols)
    tensors["topk_idxs"][:] = topk_idxs.view(T, IDX_TOPK)


def build_tensor_specs(start_pos=None):
    import torch
    from utils import (
        block_table,
        compressed_slot_mapping,
        csa_decode_start_set,
        int8_quant_per_row,
        kv_seq_lens_from_starts,
        position_ids_from_starts,
        resolve_start_positions,
        state_slot_mapping,
    )
    from golden import ScalarSpec, TensorSpec
    from utils import logical_table_blocks, token_local_rope

    def init_x():
        return torch.rand(B, S, D)
    def init_qr():
        return torch.rand(T, Q_LORA)
    # weights_proj / inner compressor: zero-mean Gaussians at the real CSA indexer std, gamma near its mean.
    def init_weights_proj():
        return torch.randn(D, IDX_N_HEADS) * 0.2313
    def init_rope_rows():
        cos, sin = token_local_rope(M, COMPRESS_RATIO, init_position_ids()[:, 0])
        return cos[:, : ROPE_HEAD_DIM // 2].float().contiguous(), sin[:, : ROPE_HEAD_DIM // 2].float().contiguous()
    def init_cos():
        return init_rope_rows()[0]
    def init_sin():
        return init_rope_rows()[1]
    def init_hadamard():
        return torch.rand(IDX_HEAD_DIM, IDX_HEAD_DIM) * (IDX_HEAD_DIM ** -0.5)
    def init_inner_compress_state():
        state = torch.zeros(INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM)
        state[:, :, INNER_OUT_DIM:] = FP32_NEG_INF
        return state
    def init_inner_compress_state_block_table():
        return block_table(batch=B, table_blocks=inner_state_table_blocks(), physical_blocks=INNER_STATE_PHYSICAL_BLOCKS)
    def init_inner_wkv():
        return torch.randn(INNER_OUT_DIM, D) * 0.0293
    def init_inner_wgate():
        return torch.randn(INNER_OUT_DIM, D) * 0.0512
    def init_inner_ape():
        return torch.randn(COMPRESS_RATIO, INNER_OUT_DIM) * 0.1528
    def init_inner_norm_w():
        return 0.6850 + 0.2610 * torch.randn(INNER_HEAD_DIM)
    def init_idx_block_table():
        return block_table(batch=B, table_blocks=idx_table_blocks(), physical_blocks=IDX_CACHE_BLOCK_NUM)
    def init_default_start_pos():
        # canonical CSA start-position set (ratio-4 compressor + indexer + sliding-window + 8k)
        return csa_decode_start_set(
            batch=B, seq=S,
            compress_ratio=COMPRESS_RATIO, state_block_size=INNER_STATE_BLOCK_SIZE, cache_tile=CACHE_TILE,
        )
    def init_start_pos():
        return resolve_start_positions(
            start_pos, batch=B, seq=S, max_seq_len=MAX_SEQ_LEN, default_fn=init_default_start_pos,
        )
    def init_position_ids():
        return position_ids_from_starts(init_start_pos(), seq=S)
    def inner_state_table_blocks():
        return logical_table_blocks(init_start_pos(), seq=S, block_size=INNER_STATE_BLOCK_SIZE)
    def idx_table_blocks():
        return logical_table_blocks(init_start_pos(), seq=S, block_size=BLOCK_SIZE)
    def score_cols():
        return (idx_table_blocks() * REDUCE_TILE + TOPK_LEAF_TILE - 1) // TOPK_LEAF_TILE * TOPK_LEAF_TILE
    def init_kv_seq_lens():
        return kv_seq_lens_from_starts(init_start_pos(), seq=S)
    def init_inner_state_slot_mapping():
        return state_slot_mapping(
            init_position_ids(), init_inner_compress_state_block_table(),
            state_block_size=INNER_STATE_BLOCK_SIZE,
        )
    def init_idx_slot_mapping():
        positions = init_position_ids()
        return compressed_slot_mapping(
            positions, init_idx_block_table(),
            compress_ratio=COMPRESS_RATIO, block_size=IDX_STORAGE_BLOCK_SIZE,
        )

    # idx wq_b: simulated MXFP8 grid, built [out, in] then transposed
    wq_b_i8_T, wq_b_scale = gen_shared_weight((IDX_N_HEADS * IDX_HEAD_DIM, Q_LORA), dequant_std=0.108, chan_cv=0.56)
    wq_b_i8 = wq_b_i8_T.t().contiguous()
    qr_i8, qr_scale = int8_quant_per_row(init_qr())

    # C8 indexer cache fixture: INT8 + scale from one bf16-rounded random draw
    idx_kv_cache_bf16 = torch.rand(IDX_CACHE_BLOCK_NUM, IDX_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM).to(torch.bfloat16)
    idx_kv_rows = idx_kv_cache_bf16.float().reshape(IDX_CACHE_BLOCK_NUM * IDX_STORAGE_BLOCK_SIZE, IDX_HEAD_DIM)
    idx_kv_i8, idx_kv_sc = int8_quant_per_row(idx_kv_rows)
    idx_kv_i8 = idx_kv_i8.view(IDX_CACHE_BLOCK_NUM, IDX_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM)
    idx_kv_sc = idx_kv_sc.view(IDX_CACHE_BLOCK_NUM, IDX_STORAGE_BLOCK_SIZE, 1, 1)

    inner_state_shape = [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM]
    inner_state_table_shape = [B, inner_state_table_blocks()]
    idx_kv_cache_shape = [IDX_CACHE_BLOCK_NUM, IDX_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM]
    idx_kv_scale_shape = [IDX_CACHE_BLOCK_NUM, IDX_STORAGE_BLOCK_SIZE, 1, 1]
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
        TensorSpec("inner_compress_state", inner_state_shape, torch.float32, init_value=init_inner_compress_state),
        TensorSpec(
            "inner_compress_state_block_table", inner_state_table_shape, torch.int32,
            init_value=init_inner_compress_state_block_table,
        ),
        TensorSpec("inner_wkv", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wkv),
        TensorSpec("inner_wgate", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wgate),
        TensorSpec("inner_ape", [COMPRESS_RATIO, INNER_OUT_DIM], torch.float32, init_value=init_inner_ape),
        TensorSpec("inner_norm_w", [INNER_HEAD_DIM], torch.bfloat16, init_value=init_inner_norm_w),
        TensorSpec("idx_kv_cache", idx_kv_cache_shape, torch.int8, init_value=lambda: idx_kv_i8),
        TensorSpec("idx_kv_scale", idx_kv_scale_shape, torch.float32, init_value=lambda: idx_kv_sc),
        TensorSpec("idx_block_table", [B, idx_table_blocks()], torch.int32, init_value=init_idx_block_table),
        # score spans whole Top-K leaves; past the visible rows score is -inf and topk_idxs is -1
        TensorSpec("score", [T, score_cols()], torch.float32),
        TensorSpec("topk_idxs", [T, IDX_TOPK], torch.int32),
        TensorSpec("position_ids", [B, S], torch.int32, init_value=init_position_ids),
        TensorSpec("idx_slot_mapping", [B, S], torch.int64, init_value=init_idx_slot_mapping),
        TensorSpec("inner_state_slot_mapping", [B, S], torch.int64, init_value=init_inner_state_slot_mapping),
        TensorSpec("kv_seq_lens", [B], torch.int32, init_value=init_kv_seq_lens),
        ScalarSpec("offset", torch.int32, OFFSET),
    ]


if __name__ == "__main__":
    import argparse
    import torch
    from golden import ratio_allclose, run, topk_pair_compare
    from utils import parse_start_pos_arg

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--start-pos", type=str, default=None, help="one start_pos or a comma-separated per-request list")
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    # pair each returned index with score[topk_idxs - OFFSET] for topk_pair_compare
    def topk_idxs_compare(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        score = actual_outputs["score"]
        a_top = actual[..., :IDX_TOPK]
        e_top = expected[..., :IDX_TOPK]
        a_orig = (a_top.long() - OFFSET).clamp(min=0, max=score.shape[-1] - 1)
        paired = torch.gather(score, dim=-1, index=a_orig)
        synth_actual = {**actual_outputs, "_topk_paired_scores": paired}
        return topk_pair_compare("_topk_paired_scores")(
            a_top, e_top,
            actual_outputs=synth_actual, expected_outputs=expected_outputs, inputs=inputs,
            rtol=rtol, atol=atol,
        )
    topk_idxs_compare.__name__ = "topk_pair_compare"

    # compare `score` over the valid region (golden tail is FP32_NEG_INF)
    def score_valid_compare(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        expected_f = expected.cpu().to(torch.float32)
        valid = expected_f != FP32_NEG_INF
        return ratio_allclose(atol=1e-4, rtol=1.0 / 128)(
            actual.cpu().to(torch.float32)[valid], expected_f[valid],
            actual_outputs=actual_outputs, expected_outputs=expected_outputs, inputs=inputs,
            rtol=rtol, atol=atol,
        )
    score_valid_compare.__name__ = "score_valid_region_compare"

    result = run(
        fn=indexer_test,
        specs=build_tensor_specs(parse_start_pos_arg(args.start_pos)),
        golden_fn=golden_indexer,
        runtime_dir=args.runtime_dir,
        config=dict(
            dump_passes=args.dump_passes,
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "score":        score_valid_compare,
            "topk_idxs":    topk_idxs_compare,
            # C8 cache: history is exact; compressor-rewritten boundary rows may differ by +/-1 LSB.
            "idx_kv_cache": ratio_allclose(atol=1, rtol=0, max_error_ratio=0.01),
            "idx_kv_scale": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
