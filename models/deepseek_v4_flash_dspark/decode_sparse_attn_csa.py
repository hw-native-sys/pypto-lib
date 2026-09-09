# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 CSA decode sparse attention over sliding-window and compressed KV caches."""


import pypto.language as pl

from config import (
    FLASH as M,
    DECODE_BATCH,
    TP,
    DECODE_SEQ,
    BLOCK_SIZE,
    KV_ORI_BLOCK_NUM,
)


# Dynamic shape variables.
B_DYN = pl.dynamic("B_DYN")  # per-request axis (block tables)
T_DYN = pl.dynamic("T_DYN")  # T = B * S
ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")

# model config
B = DECODE_BATCH // TP
S = DECODE_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
WIN = M.sliding_window
MAX_SEQ_LEN = M.max_position_embeddings
IDX_TOPK = M.index_topk
CMP_TOPK = IDX_TOPK
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
COMPRESS_RATIO = 4
COMPRESS_RATIO_INV = 1.0 / COMPRESS_RATIO
CSA_CMP_GE_BIAS = 1.0  # raw + 1, folded for the ge clamp
NEG_INF = -1.0e20

# cache geometry
ORI_MAX_BLOCKS = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
ORI_BLOCK_NUM = KV_ORI_BLOCK_NUM
CMP_MAX_BLOCKS = (MAX_SEQ_LEN // COMPRESS_RATIO + BLOCK_SIZE - 1) // BLOCK_SIZE

# tiling
H_TILE = 16
QK_PRE_LAUNCH = 2
QK_TRANSFER_SLOTS = QK_PRE_LAUNCH + 1
QK_KV_READY_EVENT = 0
QK_SCORE_READY_EVENT = 1
QK_PROB_READY_EVENT = 2
QK_PV_READY_EVENT = 3
ATTN_K_TILE = 128
NUM_QK_CORES = 24  # qk_pv dispatch lanes
T_PAD = ((T + 16 - 1) // 16) * 16  # Cube M floor
ATTENTION_PUBLISH_WORKERS = 48
ATTENTION_PUBLISH_T_TILE = 4
LOCAL_O_GROUPS = O_GROUPS // TP
GROUP_T_PAD = TP * T_PAD
ATTENTION_WINDOW_ROWS = LOCAL_O_GROUPS * GROUP_T_PAD
PUBLISH_GROUPS = H_TILE // HEADS_PER_GROUP
ROPE_CS_T_TILE = 8
TOPK = WIN + CMP_TOPK
SPARSE_BLOCKS = max(2, (TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE)  # Sparse-K block floor
PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE
SWA_TILE_WIN_ROWS = min(ATTN_K_TILE, WIN)
SWA_RUNS = (SWA_TILE_WIN_ROWS + 2 * (BLOCK_SIZE - 1)) // BLOCK_SIZE  # Sliding-window page runs
BIAS_T_TILE = min(T, 8)
if T % BIAS_T_TILE != 0:
    raise ValueError("CSA token capacity must contain complete bias tiles")
if H_TILE % HEADS_PER_GROUP != 0:
    raise ValueError(f"CSA head tile {H_TILE} must contain complete output groups")
if O_GROUPS % TP != 0:
    raise ValueError(f"output groups {O_GROUPS} must be divisible by TP size {TP}")
if LOCAL_O_GROUPS % PUBLISH_GROUPS != 0:
    raise ValueError("local output groups must contain complete CSA publish tiles")
if T % ATTENTION_PUBLISH_T_TILE != 0:
    raise ValueError("local token capacity must contain complete attention publish tiles")


@pl.jit.inline(auto_scope=False)
def sparse_attn_csa(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_MAX_BLOCKS], pl.INT32],
    idx_topk: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32],
    position_ids: pl.Tensor[[T_DYN, 1], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
):
    """Plan and run CSA QK/PV over sparse blocks, and build inverse-RoPE metadata."""
    # Compressed index contract.
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    t_dim = pl.tensor.dim(q, 0)
    t_heads = t_dim * H
    rope_cs_blocks = t_dim // ROPE_CS_T_TILE
    ori_kv_flat = pl.reshape(ori_kv, [ori_block_num * BLOCK_SIZE, HEAD_DIM])

    # pypto-lib#481 original-cache WAR marker.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_touch", allow_early_resolve=True):
        ori_kv_flat[0:1, 0:HEAD_DIM] = ori_kv_flat[0:1, 0:HEAD_DIM]

    # Sparse slot indices, additive softmax bias, and per-block validity.
    sparse_bias = pl.create_tensor([t_dim, PADDED_TOPK], dtype=pl.FP32)
    cmp_sparse_indices = pl.create_tensor([t_dim, CMP_TOPK], dtype=pl.INT32)
    valid_block_mask = pl.create_tensor([t_dim, SPARSE_BLOCKS], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_slots_build_valid_qk_plan") as qk_plan_tid:
        # Valid compressed slots.
        for bias_t0 in pl.range(0, t_dim, BIAS_T_TILE):
            c_raw = pl.cast(idx_topk[bias_t0 : bias_t0 + BIAS_T_TILE, 0:IDX_TOPK], target_type=pl.FP32)
            c_pos = pl.cast(position_ids[bias_t0 : bias_t0 + BIAS_T_TILE, 0:1], target_type=pl.FP32)
            c_pos_scaled = pl.mul(pl.add(c_pos, 1.0), COMPRESS_RATIO_INV)
            c_pos_i32 = pl.cast(c_pos_scaled, target_type=pl.INT32, mode="trunc")
            c_pos_q = pl.cast(c_pos_i32, target_type=pl.FP32)
            # Per-token compressed-slot bound.
            c_upper_b = pl.row_expand_mul(pl.full([BIAS_T_TILE, IDX_TOPK], dtype=pl.FP32, value=1.0), c_pos_q)
            c_ge = pl.minimum(pl.maximum(pl.add(c_raw, CSA_CMP_GE_BIAS), 0.0), 1.0)
            c_lt = pl.minimum(pl.maximum(pl.sub(c_upper_b, c_raw), 0.0), 1.0)
            c_mask = pl.mul(c_ge, c_lt)
            c_out = pl.sub(pl.mul(c_mask, pl.add(c_raw, 1.0)), 1.0)
            cmp_sparse_indices[bias_t0 : bias_t0 + BIAS_T_TILE, 0:IDX_TOPK] = pl.cast(c_out, target_type=pl.INT32)
            v_win_f = pl.cast(window_swa_indices[bias_t0 : bias_t0 + BIAS_T_TILE, 0:WIN], target_type=pl.FP32)
            v_win_valid = pl.minimum(pl.maximum(pl.add(v_win_f, 1.0), 0.0), 1.0)
            raw_block_valid = pl.row_max(v_win_valid)
            for c_t0 in pl.range(BIAS_T_TILE):
                c_valid = pl.cast(pl.read(raw_block_valid, [c_t0, 0]), target_type=pl.INT32)
                pl.write(valid_block_mask, [bias_t0 + c_t0, 0], c_valid)
            for c_sb in pl.range(1, SPARSE_BLOCKS):
                c_s0 = (c_sb - 1) * ATTN_K_TILE
                c_blk_valid = pl.row_max(c_mask[:, c_s0 : c_s0 + ATTN_K_TILE])
                for c_dt in pl.range(BIAS_T_TILE):
                    c_valid = pl.cast(pl.read(c_blk_valid, [c_dt, 0]), target_type=pl.INT32)
                    pl.write(valid_block_mask, [bias_t0 + c_dt, c_sb], c_valid)

            # Additive sparse softmax bias.
            sparse_bias[bias_t0 : bias_t0 + BIAS_T_TILE, 0:WIN] = pl.mul(pl.sub(v_win_valid, 1.0), -NEG_INF)
            sparse_bias[bias_t0 : bias_t0 + BIAS_T_TILE, WIN:TOPK] = pl.mul(pl.minimum(c_out, 0.0), -NEG_INF)
            if PADDED_TOPK > TOPK:
                bias_pad = pl.full([BIAS_T_TILE, PADDED_TOPK - TOPK], dtype=pl.FP32, value=NEG_INF)
                sparse_bias[bias_t0 : bias_t0 + BIAS_T_TILE, TOPK:PADDED_TOPK] = bias_pad

    # QK/PV scratch tensors.
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_block_num * BLOCK_SIZE, HEAD_DIM])
    q_flat = pl.reshape(q, [t_heads, HEAD_DIM])
    attn_sink_col = pl.reshape(attn_sink, [H, 1])
    attn_mi = pl.create_tensor([t_heads, 1], dtype=pl.FP32)
    attn_li = pl.create_tensor([t_heads, 1], dtype=pl.FP32)
    attn_oi = pl.create_tensor([t_heads, HEAD_DIM], dtype=pl.FP32)

    transfer_slots = NUM_QK_CORES * QK_TRANSFER_SLOTS
    transfer_heads = transfer_slots * H
    transfer_kv_rows = transfer_slots * ATTN_K_TILE
    kv_transfer = pl.create_tensor([transfer_kv_rows, HEAD_DIM], dtype=pl.BF16)
    score_transfer = pl.create_tensor([transfer_heads, ATTN_K_TILE], dtype=pl.FP32)
    probability_transfer = pl.create_tensor([transfer_heads, ATTN_K_TILE], dtype=pl.BF16)
    pv_transfer = pl.create_tensor([transfer_heads, HEAD_DIM], dtype=pl.FP32)
    mi_transfer = pl.create_tensor([transfer_heads, 1], dtype=pl.FP32)
    li_transfer = pl.create_tensor([transfer_heads, 1], dtype=pl.FP32)
    ffts_workspace = pl.create_tensor([256], dtype=pl.INT64)
    with pl.spmd(NUM_QK_CORES, name_hint="qk_pv", deps=[qk_plan_tid], allow_early_resolve=True) as qk_tid:
        qk_core = pl.tile.get_block_idx()
        pl.system.set_ffts(ffts_workspace)
        for qk_t in pl.range(qk_core, t_dim, NUM_QK_CORES):
            qk_b = qk_t // S
            qk_q = pl.load(
                q_flat, [qk_t * H, 0], [H, HEAD_DIM], target_memory=pl.MemorySpace.Mat,
            )
            for qk_tick in pl.range(SPARSE_BLOCKS + QK_PRE_LAUNCH):
                if qk_tick < SPARSE_BLOCKS:
                    qk_sb = qk_tick
                    if pl.read(valid_block_mask, [qk_t, qk_sb]) > 0:
                        qk_slot = qk_core * QK_TRANSFER_SLOTS + qk_sb % QK_TRANSFER_SLOTS
                        qk_kv_row = qk_slot * ATTN_K_TILE
                        qk_transfer_row = qk_slot * H
                        pl.system.sync_wait(QK_KV_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                        qk_kv = pl.load(
                            kv_transfer, [qk_kv_row, 0], [ATTN_K_TILE, HEAD_DIM],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        qk_scores = pl.matmul(qk_q, pl.tile.transpose_view(qk_kv), out_dtype=pl.FP32)
                        pl.store(qk_scores, [qk_transfer_row, 0], score_transfer)
                        pl.system.sync_set(
                            QK_SCORE_READY_EVENT, pipe=pl.PipeType.FIX,
                            ffts_mode=2, core_type=pl.KernelType.AIC,
                        )
                if qk_tick >= QK_PRE_LAUNCH:
                    pv_sb = qk_tick - QK_PRE_LAUNCH
                    if pl.read(valid_block_mask, [qk_t, pv_sb]) > 0:
                        pv_slot = qk_core * QK_TRANSFER_SLOTS + pv_sb % QK_TRANSFER_SLOTS
                        pv_kv_row = pv_slot * ATTN_K_TILE
                        pv_transfer_row = pv_slot * H
                        pl.system.sync_wait(QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                        pv_probability = pl.load(
                            probability_transfer, [pv_transfer_row, 0], [H, ATTN_K_TILE],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        pv_kv = pl.load(
                            kv_transfer, [pv_kv_row, 0], [ATTN_K_TILE, HEAD_DIM],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        pv_output = pl.matmul(pv_probability, pv_kv, out_dtype=pl.FP32)
                        pl.store(pv_output, [pv_transfer_row, 0], pv_transfer)
                        pl.system.sync_set(
                            QK_PV_READY_EVENT, pipe=pl.PipeType.FIX,
                            ffts_mode=2, core_type=pl.KernelType.AIC,
                        )

            for qk_aiv in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                pl.system.set_ffts(ffts_workspace)
                qk_lane_head = qk_aiv * (H // 2)
                qk_lane_kv = qk_aiv * (ATTN_K_TILE // 2)
                qk_reduce_tmp = pl.create_tile([H // 2, ATTN_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                running_m = pl.load(attn_sink_col, [qk_lane_head, 0], [H // 2, 1], target_memory=pl.MemorySpace.Vec)
                running_l = pl.tile.muls(running_m, 0.0)
                running_left = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
                running_right = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
                for qk_tick, (m_iter, l_iter, left_iter, right_iter) in pl.range(
                    SPARSE_BLOCKS + QK_PRE_LAUNCH,
                    init_values=(running_m, running_l, running_left, running_right),
                ):
                    if qk_tick < SPARSE_BLOCKS:
                        qk_sb = qk_tick
                        if pl.read(valid_block_mask, [qk_t, qk_sb]) > 0:
                            qk_slot = qk_core * QK_TRANSFER_SLOTS + qk_sb % QK_TRANSFER_SLOTS
                            qk_kv_row = qk_slot * ATTN_K_TILE
                            qk_transfer_row = qk_slot * H
                            qk_s0 = qk_sb * ATTN_K_TILE
                            qk_kv_half = pl.tile.full([ATTN_K_TILE // 2, HEAD_DIM], dtype=pl.BF16, value=0.0)
                            if qk_s0 < WIN:
                                qk_pos = pl.cast(pl.read(position_ids, [qk_t, 0]), pl.INDEX)
                                qk_win_len = pl.min(qk_pos + 1, WIN)
                                qk_win_start = qk_pos - qk_win_len + 1
                                qk_head = (qk_win_start + qk_s0) % BLOCK_SIZE
                                qk_rows = pl.min(pl.max(qk_win_len - qk_s0 - qk_lane_kv, 0), ATTN_K_TILE // 2)
                                for qk_run in pl.unroll(SWA_RUNS):
                                    qk_lo = pl.max(qk_run * BLOCK_SIZE - qk_head - qk_lane_kv, 0)
                                    qk_hi = pl.min((qk_run + 1) * BLOCK_SIZE - qk_head - qk_lane_kv, qk_rows)
                                    if qk_hi > qk_lo:
                                        qk_raw_row = pl.read(window_swa_indices, [qk_t, qk_s0 + qk_lane_kv + qk_lo])
                                        if qk_raw_row >= 0:
                                            qk_kv_half = pl.gather_row(
                                                qk_kv_half, ori_kv_flat, [qk_lo, 0], [qk_raw_row, 0],
                                                [ATTN_K_TILE // 2, HEAD_DIM], valid_shape=[qk_hi - qk_lo, HEAD_DIM],
                                            )
                            else:
                                for qk_row in pl.range(ATTN_K_TILE // 2):
                                    qk_cmp_k = qk_s0 + qk_lane_kv + qk_row - WIN
                                    if qk_cmp_k < CMP_TOPK:
                                        qk_ridx = pl.read(cmp_sparse_indices, [qk_t, qk_cmp_k])
                                        if qk_ridx >= 0:
                                            qk_page = pl.cast(pl.read(cmp_block_table, [qk_b, qk_ridx // BLOCK_SIZE]), pl.INDEX)
                                            qk_src = qk_page * BLOCK_SIZE + qk_ridx % BLOCK_SIZE
                                            qk_kv_half = pl.gather_row(
                                                qk_kv_half, cmp_kv_flat, [qk_row, 0], [qk_src, 0], [1, HEAD_DIM],
                                            )
                            pl.store(qk_kv_half, [qk_kv_row + qk_lane_kv, 0], kv_transfer)
                            pl.system.sync_set(
                                QK_KV_READY_EVENT, pipe=pl.PipeType.MTE3,
                                ffts_mode=2, core_type=pl.KernelType.AIV,
                            )
                            pl.system.sync_wait(QK_SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                            qk_scores_half = pl.load(
                                score_transfer, [qk_transfer_row + qk_lane_head, 0], [H // 2, ATTN_K_TILE],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            qk_bias = pl.load(
                                sparse_bias, [qk_t, qk_s0], [1, ATTN_K_TILE], target_memory=pl.MemorySpace.Vec,
                            )
                            qk_scaled = pl.mul(qk_scores_half, SOFTMAX_SCALE)
                            qk_masked = pl.col_expand_add(qk_scaled, qk_bias)
                            qk_mi = pl.row_max(qk_masked, qk_reduce_tmp)
                            qk_exp = pl.exp(pl.row_expand_sub(qk_masked, qk_mi))
                            qk_li = pl.row_sum(qk_exp, qk_reduce_tmp)
                            qk_probability = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
                            pl.store(qk_probability, [qk_transfer_row + qk_lane_head, 0], probability_transfer)
                            pl.store(qk_mi, [qk_transfer_row + qk_lane_head, 0], mi_transfer)
                            pl.store(qk_li, [qk_transfer_row + qk_lane_head, 0], li_transfer)
                            pl.system.sync_set(
                                QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE3,
                                ffts_mode=2, core_type=pl.KernelType.AIV,
                            )
                    if qk_tick >= QK_PRE_LAUNCH:
                        pv_sb = qk_tick - QK_PRE_LAUNCH
                        if pl.read(valid_block_mask, [qk_t, pv_sb]) > 0:
                            pv_slot = qk_core * QK_TRANSFER_SLOTS + pv_sb % QK_TRANSFER_SLOTS
                            pv_transfer_row = pv_slot * H
                            pl.system.sync_wait(QK_PV_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                            pv_m = pl.load(mi_transfer, [pv_transfer_row + qk_lane_head, 0], [H // 2, 1], target_memory=pl.MemorySpace.Vec)
                            pv_l = pl.load(li_transfer, [pv_transfer_row + qk_lane_head, 0], [H // 2, 1], target_memory=pl.MemorySpace.Vec)
                            next_m = pl.maximum(m_iter, pv_m)
                            alpha = pl.exp(pl.sub(m_iter, next_m))
                            beta = pl.exp(pl.sub(pv_m, next_m))
                            next_l = pl.add(pl.mul(alpha, l_iter), pl.mul(beta, pv_l))
                            pv_left = pl.load(
                                pv_transfer, [pv_transfer_row + qk_lane_head, 0], [H // 2, HEAD_DIM // 2],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            next_left = pl.add(pl.row_expand_mul(left_iter, alpha), pl.row_expand_mul(pv_left, beta))
                            pv_right = pl.load(
                                pv_transfer, [pv_transfer_row + qk_lane_head, HEAD_DIM // 2], [H // 2, HEAD_DIM // 2],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            next_right = pl.add(pl.row_expand_mul(right_iter, alpha), pl.row_expand_mul(pv_right, beta))
                            m_valid, l_valid, left_valid, right_valid = pl.yield_(next_m, next_l, next_left, next_right)
                        else:
                            m_valid, l_valid, left_valid, right_valid = pl.yield_(m_iter, l_iter, left_iter, right_iter)
                        m_after, l_after, left_after, right_after = pl.yield_(m_valid, l_valid, left_valid, right_valid)
                    else:
                        m_after, l_after, left_after, right_after = pl.yield_(m_iter, l_iter, left_iter, right_iter)
                    running_m, running_l, running_left, running_right = pl.yield_(m_after, l_after, left_after, right_after)
                qk_output_row = qk_t * H + qk_lane_head
                pl.store(running_m, [qk_output_row, 0], attn_mi)
                pl.store(running_l, [qk_output_row, 0], attn_li)
                pl.store(running_left, [qk_output_row, 0], attn_oi)
                pl.store(running_right, [qk_output_row, HEAD_DIM // 2], attn_oi)

    # Interleaved inverse-RoPE frequency rows.
    rope_cos_il = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
    # Inverse-RoPE lane-swap index.
    rope_swap_idx = pl.create_tensor([H_TILE, ROPE_DIM], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_cs", allow_early_resolve=True) as rope_tid:
        sw_ones = pl.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        sw_idx_f = pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32)
        sw_col = pl.col_expand_mul(sw_ones, sw_idx_f)
        sw_dup_i32 = pl.cast(pl.mul(sw_col, 0.5), target_type=pl.INT32, mode="trunc")
        sw_dup_f = pl.cast(sw_dup_i32, target_type=pl.FP32)
        sw_lane = pl.sub(sw_col, pl.mul(sw_dup_f, 2.0))
        sw_swap_f = pl.sub(pl.add(sw_col, 1.0), pl.mul(sw_lane, 2.0))
        rope_swap_idx[0:H_TILE, 0:ROPE_DIM] = pl.cast(sw_swap_f, target_type=pl.INT32)

        cs_ones = pl.full([ROPE_CS_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        cs_idx_f = pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32)
        cs_col = pl.col_expand_mul(cs_ones, cs_idx_f)
        cs_dup_i32 = pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode="trunc")
        cs_dup_f = pl.cast(cs_dup_i32, target_type=pl.FP32)
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))
        for cs_rb in pl.range(rope_cs_blocks):
            cs_t0 = cs_rb * ROPE_CS_T_TILE
            cs_cos = pl.cast(freqs_cos[cs_t0 : cs_t0 + ROPE_CS_T_TILE, 0:HALF_ROPE], target_type=pl.FP32)
            cs_sin = pl.cast(freqs_sin[cs_t0 : cs_t0 + ROPE_CS_T_TILE, 0:HALF_ROPE], target_type=pl.FP32)
            rope_cos_il[cs_t0 : cs_t0 + ROPE_CS_T_TILE, 0:ROPE_DIM] = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
            cs_sin_il = pl.gather(cs_sin, dim=-1, index=cs_dup_idx)
            rope_sin_signed[cs_t0 : cs_t0 + ROPE_CS_T_TILE, 0:ROPE_DIM] = pl.mul(cs_sin_il, cs_sign)

    return (
        attn_mi, attn_li, attn_oi,
        rope_cos_il, rope_sin_signed, rope_swap_idx,
        qk_tid, rope_tid,
    )


@pl.jit.inline
def sparse_attn_csa_tp1(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_MAX_BLOCKS], pl.INT32],
    idx_topk: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32],
    position_ids: pl.Tensor[[T_DYN, 1], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    o_packed_heads: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
) -> tuple[pl.Tensor, pl.Scalar[pl.TASK_ID]]:
    """Write CSA heads as ``[group, T_PAD, O_GROUP_IN]`` slabs.

    Only the first runtime ``t_dim`` rows in each group are valid. The
    returned task ID covers every write to the packed output tensor.
    """
    (
        attn_mi, attn_li, attn_oi,
        rope_cos_il, rope_sin_signed, rope_swap_idx,
        qk_tid, rope_tid,
    ) = sparse_attn_csa(
        q, ori_kv, window_swa_indices,
        cmp_kv, cmp_block_table, idx_topk,
        position_ids, attn_sink, freqs_cos, freqs_sin,
    )
    t_dim = pl.tensor.dim(q, 0)

    with pl.spmd(t_dim * (H // H_TILE), name_hint="merge_norm", deps=[qk_tid, rope_tid]) as merge_tid:
        m_idx = pl.tile.get_block_idx()
        m_t = m_idx // (H // H_TILE)
        m_h_idx = m_idx - m_t * (H // H_TILE)
        m_h0 = m_h_idx * H_TILE
        m_row = m_idx * H_TILE
        m_mi = attn_mi[m_row : m_row + H_TILE, 0:1]
        m_li = attn_li[m_row : m_row + H_TILE, 0:1]
        m_oi = attn_oi[m_row : m_row + H_TILE, 0:HEAD_DIM]

        n_sink_bias = pl.reshape(attn_sink[m_h0 : m_h0 + H_TILE], [H_TILE, 1])
        n_sink_tile = pl.add(pl.sub(m_mi, m_mi), n_sink_bias)
        n_denom = pl.add(m_li, pl.exp(pl.sub(n_sink_tile, m_mi)))
        n_full = pl.row_expand_div(m_oi, n_denom)[0 : H_TILE, 0 : HEAD_DIM]
        n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")

        # Inverse-RoPE head tile.
        m_rope = n_full[0 : H_TILE, NOPE_DIM : HEAD_DIM]
        m_cos_il = rope_cos_il[m_t : m_t + 1, 0 : ROPE_DIM]
        m_sin_signed = rope_sin_signed[m_t : m_t + 1, 0 : ROPE_DIM]
        m_swapped = pl.gather(m_rope, dim=-1, index=rope_swap_idx[0:H_TILE, 0:ROPE_DIM])
        m_rot = pl.add(pl.col_expand_mul(m_rope, m_cos_il), pl.col_expand_mul(m_swapped, m_sin_signed))
        n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode="rint")
        n_full_bf16 = pl.concat(n_bf16[:, : NOPE_DIM], n_rope_bf16)

        n_group_bf16 = pl.reshape(n_full_bf16, [PUBLISH_GROUPS, O_GROUP_IN])
        for n_group in pl.unroll(PUBLISH_GROUPS):
            n_pack_row = (m_h0 // HEADS_PER_GROUP + n_group) * T_PAD + m_t
            o_packed_heads[n_pack_row : n_pack_row + 1, 0:O_GROUP_IN] = n_group_bf16[n_group : n_group + 1, :]

    return o_packed_heads, merge_tid


@pl.jit
def sparse_attn_csa_test(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_MAX_BLOCKS], pl.INT32],
    idx_topk: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32],
    position_ids: pl.Tensor[[T_DYN, 1], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    o_packed_heads: pl.Out[pl.Tensor[[O_GROUPS, T_PAD, O_GROUP_IN], pl.BF16]],
):
    q.bind_dynamic(0, T_DYN)
    window_swa_indices.bind_dynamic(0, T_DYN)
    cmp_block_table.bind_dynamic(0, B_DYN)
    idx_topk.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)

    o_packed_flat = pl.reshape(o_packed_heads, [O_GROUPS * T_PAD, O_GROUP_IN])
    o_packed_flat, _ = sparse_attn_csa_tp1(
        q, ori_kv, window_swa_indices,
        cmp_kv, cmp_block_table, idx_topk,
        position_ids, attn_sink, freqs_cos, freqs_sin,
        o_packed_flat,
    )
    return o_packed_heads


def golden_sparse_attn(tensors):
    """Torch reference for the CSA sparse-attention heads."""
    import torch

    q = tensors["q"].float()
    tokens = q.shape[0]
    ori_kv = tensors["ori_kv"].float()
    window_swa_indices = tensors["window_swa_indices"]
    cmp_kv = tensors["cmp_kv"].float()
    cmp_block_table = tensors["cmp_block_table"]
    # Valid compressed slots.
    raw = tensors["idx_topk"][:, :CMP_TOPK].to(torch.int64)
    bound = ((tensors["position_ids"][:, 0].to(torch.int64) + 1) // COMPRESS_RATIO).unsqueeze(1)
    keep = (raw >= 0) & (raw < bound)
    cmp_sparse_indices = torch.where(keep, raw, torch.full_like(raw, -1)).to(torch.int32)
    attn_sink = tensors["attn_sink"].float()
    cos = tensors["freqs_cos"].float()
    sin = tensors["freqs_sin"].float()

    o = torch.zeros(tokens, H, HEAD_DIM)

    # Per-token sparse attention.
    for t in range(tokens):
        b = t // S
        kv_rows = []
        valid = []

        for raw in window_swa_indices[t].tolist():
            slot = int(raw)
            if slot >= 0:
                blk_id = slot // BLOCK_SIZE
                intra = slot % BLOCK_SIZE
                kv_rows.append(ori_kv[blk_id, intra, 0])
                valid.append(True)
            else:
                kv_rows.append(torch.zeros(HEAD_DIM, dtype=ori_kv.dtype))
                valid.append(False)

        for raw in cmp_sparse_indices[t].tolist():
            if raw < 0:
                kv_rows.append(torch.zeros(HEAD_DIM, dtype=ori_kv.dtype))
                valid.append(False)
                continue
            cmp_slot = int(raw)
            blk_id = int(cmp_block_table[b, cmp_slot // BLOCK_SIZE].item())
            intra = cmp_slot % BLOCK_SIZE
            kv_rows.append(cmp_kv[blk_id, intra, 0])
            valid.append(True)

        if not any(valid):
            continue

        pad_k = PADDED_TOPK - TOPK
        if pad_k:
            kv_rows.extend(torch.zeros(HEAD_DIM, dtype=ori_kv.dtype) for _ in range(pad_k))
            valid.extend(False for _ in range(pad_k))

        kv_b = torch.stack(kv_rows, dim=0)
        valid_b = torch.tensor(valid, dtype=torch.bool)
        q_t = q[t]

        block_mi = []
        block_li = []
        block_oi = []
        for tile_start in range(0, PADDED_TOPK, ATTN_K_TILE):
            kv_tile = kv_b[tile_start:tile_start + ATTN_K_TILE]
            valid_tile = valid_b[tile_start:tile_start + ATTN_K_TILE]
            scores = (q_t @ kv_tile.T) * SOFTMAX_SCALE
            scores = scores.masked_fill(~valid_tile.unsqueeze(0), NEG_INF)
            mi = scores.max(dim=-1, keepdim=True).values
            exp_scores = torch.exp(scores - mi).masked_fill(~valid_tile.unsqueeze(0), 0.0)
            li = exp_scores.sum(dim=-1, keepdim=True)
            oi = exp_scores.to(torch.bfloat16).float() @ kv_tile.to(torch.bfloat16).float()
            block_mi.append(mi)
            block_li.append(li)
            block_oi.append(oi)

        score_max = block_mi[0]
        li = block_li[0]
        oi_num = block_oi[0]
        for mi_cur, li_cur, oi_cur in zip(block_mi[1:], block_li[1:], block_oi[1:]):
            score_max_new = torch.maximum(score_max, mi_cur)
            alpha = torch.exp(score_max - score_max_new)
            beta = torch.exp(mi_cur - score_max_new)
            li = alpha * li + beta * li_cur
            oi_num = alpha * oi_num + beta * oi_cur
            score_max = score_max_new

        denom = li + torch.exp(attn_sink.unsqueeze(-1) - score_max)
        o[t] = oi_num / denom

    rope_pair = o[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    rope_even = rope_pair[..., 0]
    rope_odd = rope_pair[..., 1]
    cos_half = cos[:, :HALF_ROPE].unsqueeze(1)
    sin_half = sin[:, :HALF_ROPE].unsqueeze(1)
    inv_even = (rope_even * cos_half + rope_odd * sin_half).to(torch.bfloat16).float()
    inv_odd = (rope_odd * cos_half - rope_even * sin_half).to(torch.bfloat16).float()
    o_rope = torch.stack([inv_even, inv_odd], dim=-1).flatten(-2)
    o = torch.cat([o[..., :NOPE_DIM], o_rope], dim=-1).to(torch.bfloat16)

    # Grouped attention output packing.
    packed = tensors["o_packed_heads"]
    packed[:, :tokens] = o.float().view(tokens, O_GROUPS, O_GROUP_IN).permute(1, 0, 2).to(torch.bfloat16)

def build_tensor_specs(
    causal_regression_fixture: bool = False,
    short_window_fixture: bool = False,
    mixed_topk_fixture: bool = False,
    cache_window_replacement_fixture: bool = False,
    all_invalid_fixture: bool = False,
    start_pos=None,
    batch: int = B,
):
    """Build deterministic demo tensors for the CSA standalone harness."""
    import torch
    from golden import TensorSpec
    from utils import (
        block_table,
        csa_decode_start_set,
        position_ids_from_starts,
        resolve_start_positions,
        swa_indices_and_lens,
        token_local_rope,
    )

    tokens = batch * S
    def default_starts():
        return csa_decode_start_set(batch=batch, seq=S, compress_ratio=COMPRESS_RATIO)

    starts = resolve_start_positions(
        start_pos, batch=batch, seq=S, max_seq_len=MAX_SEQ_LEN, default_fn=default_starts,
    )
    positions = position_ids_from_starts(starts, seq=S)
    visible_rows = ((positions.to(torch.int64) + 1) // COMPRESS_RATIO).reshape(-1)
    max_visible_rows = int(visible_rows.max().item())
    active_cmp_pages = max(1, (max_visible_rows + BLOCK_SIZE - 1) // BLOCK_SIZE)
    cmp_block_num = batch * active_cmp_pages
    shared_rope_cos, shared_rope_sin = token_local_rope(
        M, COMPRESS_RATIO, positions.reshape(-1),
        max_seq_len=MAX_SEQ_LEN, dtype=torch.bfloat16,
    )
    shared_window_block_table = block_table(batch=batch, table_blocks=ORI_MAX_BLOCKS, physical_blocks=ORI_BLOCK_NUM)
    shared_swa_metadata = swa_indices_and_lens(positions, shared_window_block_table, block_size=BLOCK_SIZE, window=WIN)
    shared_swa_indices = shared_swa_metadata[0].contiguous()
    if all_invalid_fixture:
        shared_swa_indices.fill_(-1)

    def init_q():
        """Initialize the query tensor used by the decode attention stage."""
        q = torch.rand(tokens, H, HEAD_DIM) - 0.5
        if causal_regression_fixture:
            q[0].fill_(1.0)
        return q

    def init_ori_kv():
        """Initialize the sliding-window KV cache pages."""
        kv = torch.rand(ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM) - 0.5
        if causal_regression_fixture:
            sentinel_row = int(shared_swa_indices[0, -1].item())
            if sentinel_row >= 0:
                kv.reshape(-1, 1, HEAD_DIM)[sentinel_row, 0].fill_(8.0)
        if cache_window_replacement_fixture:
            kv[0, 16, 0].fill_(0.0)
            kv[0, 16, 0, 0] = 4.0
        return kv

    def init_window_swa_indices():
        """Lower the window through the same producer the model uses.

        Indexing the block table by window slot instead of absolute position
        would keep every row of a WIN == BLOCK_SIZE window inside one page, so
        the fixture could not tell a correct page-run split from a broken one.
        Going through swa_indices_and_lens straddles a page boundary whenever
        init_position_ids is not page-aligned, which it is not.
        """
        return shared_swa_indices.clone()

    def init_cmp_kv():
        """Initialize the compressed-cache KV pages."""
        return torch.rand(cmp_block_num, BLOCK_SIZE, 1, HEAD_DIM) - 0.5

    def init_attn_sink():
        """Initialize the per-head sink logits to zero."""
        return torch.zeros(H)

    def init_window_block_table():
        """Build the demo block table for the sliding-window cache pages."""
        return shared_window_block_table.clone()

    def init_cmp_block_table():
        """Build the demo block table for the compressed-cache pages."""
        return block_table(batch=batch, table_blocks=CMP_MAX_BLOCKS, physical_blocks=cmp_block_num)

    def init_cmp_sparse_indices():
        """Build length-aware logical Top-K candidates across each visible range."""
        indices = torch.full((tokens, CMP_TOPK), -1, dtype=torch.int32)
        for token, visible in enumerate(visible_rows.tolist()):
            valid = min(CMP_TOPK, int(visible))
            if short_window_fixture:
                valid = min(valid, 17)
            if valid == 0:
                continue
            if valid == int(visible) or mixed_topk_fixture:
                candidates = torch.arange(valid, dtype=torch.int64)
            elif valid == 1:
                candidates = torch.zeros(1, dtype=torch.int64)
            else:
                spread = torch.arange(valid, dtype=torch.int64) * (int(visible) - 1)
                candidates = torch.div(spread, valid - 1, rounding_mode="floor")
            indices[token, :valid] = candidates.to(torch.int32)
        if cache_window_replacement_fixture:
            indices[:, :] = -1
        if causal_regression_fixture:
            indices[0, :] = -1
        if all_invalid_fixture:
            indices.fill_(-1)
        return indices

    def init_idx_topk():
        """Raw logical Top-K output produced by the standalone indexer."""
        return init_cmp_sparse_indices()

    def init_position_ids():
        return positions.reshape(tokens, 1).contiguous()

    def init_cos():
        """Build the split-half cosine table used by the inverse-RoPE reference."""
        return shared_rope_cos.clone()

    def init_sin():
        """Build the split-half sine table used by the inverse-RoPE reference."""
        return shared_rope_sin.clone()

    return [
        TensorSpec("q", [tokens, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("window_swa_indices", [tokens, WIN], torch.int32, init_value=init_window_swa_indices),
        TensorSpec("cmp_kv", [cmp_block_num, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [batch, CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("idx_topk", [tokens, IDX_TOPK], torch.int32, init_value=init_idx_topk),
        TensorSpec("position_ids", [tokens, 1], torch.int32, init_value=init_position_ids),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("freqs_cos", [tokens, ROPE_DIM], torch.bfloat16, init_value=init_cos),
        TensorSpec("freqs_sin", [tokens, ROPE_DIM], torch.bfloat16, init_value=init_sin),
        TensorSpec("o_packed_heads", [O_GROUPS, T_PAD, O_GROUP_IN], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "-b", "--batch", type=int, default=B,
        help=f"runtime request count up to {B} (the compile-time upper bound). The token axis "
             "is pl.dynamic, so one compiled program serves every value.",
    )
    parser.add_argument(
        "--start-pos", type=str, default=None,
        help="Fixture-only start position: one value for a uniform batch or "
             "a comma-separated value per request.",
    )
    parser.add_argument(
        "--causal-regression-fixture", action="store_true", default=False,
        help="Amplify the S=2 future-window-slot regression.",
    )
    parser.add_argument(
        "--short-window-fixture", action="store_true", default=False,
        help="Use a short-window topk row with valid prefix + -1 padding.",
    )
    parser.add_argument(
        "--mixed-topk-fixture", action="store_true", default=False,
        help="Use -1-padded window slots with valid compressed raw indices.",
    )
    parser.add_argument(
        "--cache-window-replacement-fixture", action="store_true", default=False,
        help="Place a sentinel row inside the cache window prefix.",
    )
    parser.add_argument(
        "--all-invalid-fixture", action="store_true", default=False,
        help="Mask every raw and compressed row.",
    )
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    parser.add_argument(
        "--enable-dep-gen", action="store_true", default=False,
        help="Capture PTO2 dependency edges (deps.json) for the swimlane converter.",
    )
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    if args.batch < 1 or args.batch > B:
        parser.error(f"--batch must be in [1, {B}], got {args.batch}")
    start_pos = None
    if args.start_pos is not None:
        try:
            start_values = [int(value.strip()) for value in args.start_pos.split(",") if value.strip() != ""]
        except ValueError:
            parser.error(f"--start-pos must contain integers, got {args.start_pos!r}")
        if not start_values:
            parser.error("--start-pos must contain at least one integer")
        if len(start_values) not in (1, args.batch):
            parser.error(f"--start-pos needs 1 or {args.batch} values, got {len(start_values)}")
        start_pos = start_values[0] if len(start_values) == 1 else start_values

    print(f"compress_ratio={COMPRESS_RATIO} -> TOPK={TOPK} SPARSE_BLOCKS={SPARSE_BLOCKS} PADDED_TOPK={PADDED_TOPK}", flush=True)

    result = run(
        fn=sparse_attn_csa_test,
        specs=build_tensor_specs(
            args.causal_regression_fixture, args.short_window_fixture, args.mixed_topk_fixture,
            args.cache_window_replacement_fixture, args.all_invalid_fixture,
            start_pos=start_pos, batch=args.batch,
        ),
        golden_fn=golden_sparse_attn,
        golden_data=args.golden_data,
        config=dict(
            dump_passes=args.dump_passes,
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_dep_gen=args.enable_dep_gen,
            enable_pmu=args.enable_pmu,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "o_packed_heads": ratio_allclose(
                atol=1e-4, rtol=1.0 / 128,
                valid_rows=args.batch * S, valid_axis=1,
            ),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
