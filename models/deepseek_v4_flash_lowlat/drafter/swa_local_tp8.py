# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""One padded D attention panel from the original online-softmax body."""

import pypto.language as pl
from .config import FLASH as M

T=5
S=T
B=1
H=16
HEAD_DIM=M.head_dim
ROPE_DIM=M.qk_rope_head_dim
NOPE_DIM=M.nope_head_dim
HEADS_PER_GROUP=8
O_GROUPS=H//HEADS_PER_GROUP
LOCAL_T_PAD=16
SOFTMAX_SCALE=M.softmax_scale
ATTN_K_TILE=64
VISIBLE_ROWS=M.sliding_window+S
SPARSE_BLOCKS=(VISIBLE_ROWS+ATTN_K_TILE-1)//ATTN_K_TILE
INDEX_WIDTH=SPARSE_BLOCKS*ATTN_K_TILE
NUM_QK_CORES=T
QK_PRE_LAUNCH=2
QK_TRANSFER_SLOTS=QK_PRE_LAUNCH+1
QK_SCORE_READY_EVENT=0
QK_PROB_READY_EVENT=1
QK_PV_READY_EVENT=2
H_TILE=8


@pl.jit.inline
def local_attention(
    q: pl.Tensor[[T,H,HEAD_DIM],pl.BF16],
    visible_kv: pl.Tensor[[B*INDEX_WIDTH,HEAD_DIM],pl.BF16],
    sparse_bias: pl.Tensor[[B,INDEX_WIDTH],pl.FP32],
    swa_lens: pl.Tensor[[B],pl.INT32],
    attn_sink: pl.Tensor[[H],pl.FP32],
    rope_cos_il: pl.Tensor[[T,ROPE_DIM],pl.FP32],
    rope_sin_signed: pl.Tensor[[T,ROPE_DIM],pl.FP32],
    rope_swap_idx: pl.Tensor[[T,ROPE_DIM],pl.INT32],
    o_packed_heads: pl.Out[pl.Tensor[[O_GROUPS*LOCAL_T_PAD*HEADS_PER_GROUP,HEAD_DIM],pl.BF16]],
):
    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    # Save online-softmax state for the fused publish phase after QK/PV.
    sparse_mi = pl.create_tensor([T * H, 1], dtype=pl.FP32)
    sparse_li = pl.create_tensor([T * H, 1], dtype=pl.FP32)
    sparse_oi = pl.create_tensor([T * H, HEAD_DIM], dtype=pl.FP32)
    attn_sink_col = pl.reshape(attn_sink, [H, 1])
    o_packed_flat = pl.reshape(
        o_packed_heads,
        [O_GROUPS * LOCAL_T_PAD * HEADS_PER_GROUP, HEAD_DIM],
    )
    transfer_slots = NUM_QK_CORES * QK_TRANSFER_SLOTS
    transfer_heads = transfer_slots * H
    score_transfer = pl.create_tensor([transfer_heads, ATTN_K_TILE], dtype=pl.FP32)
    probability_transfer = pl.create_tensor([transfer_heads, ATTN_K_TILE], dtype=pl.BF16)
    pv_transfer = pl.create_tensor([transfer_heads, HEAD_DIM], dtype=pl.FP32)
    mi_transfer = pl.create_tensor([transfer_heads, 1], dtype=pl.FP32)
    li_transfer = pl.create_tensor([transfer_heads, 1], dtype=pl.FP32)
    ffts_workspace = pl.create_tensor([256], dtype=pl.INT64)
    with pl.spmd(NUM_QK_CORES, name_hint="dspark_qk_pv") as qk_tid:
        qk_core = pl.tile.get_block_idx()
        pl.system.set_ffts(ffts_workspace)
        for qk_t in pl.range(qk_core, T, NUM_QK_CORES):
            qk_b = qk_t // S
            qk_q = pl.load(
                q_flat,
                [qk_t * H, 0],
                [H, HEAD_DIM],
                target_memory=pl.MemorySpace.Mat,
            )
            for qk_tick in pl.range(SPARSE_BLOCKS + QK_PRE_LAUNCH):
                if qk_tick < SPARSE_BLOCKS:
                    qk_sb = qk_tick
                    if qk_sb * ATTN_K_TILE < pl.read(swa_lens, [qk_b]):
                        qk_slot = qk_core * QK_TRANSFER_SLOTS + qk_sb % QK_TRANSFER_SLOTS
                        qk_kv_row = qk_b * INDEX_WIDTH + qk_sb * ATTN_K_TILE
                        qk_transfer_row = qk_slot * H
                        qk_kv = pl.load(
                            visible_kv,
                            [qk_kv_row, 0],
                            [ATTN_K_TILE, HEAD_DIM],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        qk_scores = pl.matmul(qk_q, pl.tile.transpose_view(qk_kv), out_dtype=pl.FP32)
                        pl.store(qk_scores, [qk_transfer_row, 0], score_transfer)
                        pl.system.sync_set(
                            QK_SCORE_READY_EVENT,
                            pipe=pl.PipeType.FIX,
                            ffts_mode=2,
                            core_type=pl.KernelType.AIC,
                        )
                if qk_tick >= QK_PRE_LAUNCH:
                    pv_sb = qk_tick - QK_PRE_LAUNCH
                    if pv_sb * ATTN_K_TILE < pl.read(swa_lens, [qk_b]):
                        pv_slot = qk_core * QK_TRANSFER_SLOTS + pv_sb % QK_TRANSFER_SLOTS
                        pv_kv_row = qk_b * INDEX_WIDTH + pv_sb * ATTN_K_TILE
                        pv_transfer_row = pv_slot * H
                        pl.system.sync_wait(
                            QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC
                        )
                        pv_probability = pl.load(
                            probability_transfer,
                            [pv_transfer_row, 0],
                            [H, ATTN_K_TILE],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        pv_kv = pl.load(
                            visible_kv,
                            [pv_kv_row, 0],
                            [ATTN_K_TILE, HEAD_DIM],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        pv_output = pl.matmul(pv_probability, pv_kv, out_dtype=pl.FP32)
                        pl.store(pv_output, [pv_transfer_row, 0], pv_transfer)
                        pl.system.sync_set(
                            QK_PV_READY_EVENT,
                            pipe=pl.PipeType.FIX,
                            ffts_mode=2,
                            core_type=pl.KernelType.AIC,
                        )

            for qk_aiv in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                pl.system.set_ffts(ffts_workspace)
                qk_lane_head = qk_aiv * (H // 2)
                qk_reduce_tmp = pl.create_tile(
                    [H // 2, ATTN_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                running_m = pl.load(
                    attn_sink_col, [qk_lane_head, 0], [H // 2, 1], target_memory=pl.MemorySpace.Vec
                )
                running_l = pl.tile.muls(running_m, 0.0)
                running_left = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
                running_right = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
                for qk_tick, (m_iter, l_iter, left_iter, right_iter) in pl.range(
                    SPARSE_BLOCKS + QK_PRE_LAUNCH,
                    init_values=(running_m, running_l, running_left, running_right),
                ):
                    if qk_tick < SPARSE_BLOCKS:
                        qk_sb = qk_tick
                        if qk_sb * ATTN_K_TILE < pl.read(swa_lens, [qk_b]):
                            qk_slot = qk_core * QK_TRANSFER_SLOTS + qk_sb % QK_TRANSFER_SLOTS
                            qk_transfer_row = qk_slot * H
                            qk_s0 = qk_sb * ATTN_K_TILE
                            pl.system.sync_wait(
                                QK_SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV
                            )
                            qk_scores_half = pl.load(
                                score_transfer,
                                [qk_transfer_row + qk_lane_head, 0],
                                [H // 2, ATTN_K_TILE],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            qk_bias = pl.load(
                                sparse_bias,
                                [qk_b, qk_s0],
                                [1, ATTN_K_TILE],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            qk_scaled = pl.mul(qk_scores_half, SOFTMAX_SCALE)
                            qk_masked = pl.col_expand_add(qk_scaled, qk_bias)
                            qk_mi = pl.row_max(qk_masked, qk_reduce_tmp)
                            qk_exp = pl.exp(pl.row_expand_sub(qk_masked, qk_mi))
                            qk_li = pl.row_sum(qk_exp, qk_reduce_tmp)
                            qk_probability = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
                            pl.store(
                                qk_probability, [qk_transfer_row + qk_lane_head, 0], probability_transfer
                            )
                            pl.store(qk_mi, [qk_transfer_row + qk_lane_head, 0], mi_transfer)
                            pl.store(qk_li, [qk_transfer_row + qk_lane_head, 0], li_transfer)
                            pl.system.sync_set(
                                QK_PROB_READY_EVENT,
                                pipe=pl.PipeType.MTE3,
                                ffts_mode=2,
                                core_type=pl.KernelType.AIV,
                            )
                    if qk_tick >= QK_PRE_LAUNCH:
                        pv_sb = qk_tick - QK_PRE_LAUNCH
                        if pv_sb * ATTN_K_TILE < pl.read(swa_lens, [qk_b]):
                            pv_slot = qk_core * QK_TRANSFER_SLOTS + pv_sb % QK_TRANSFER_SLOTS
                            pv_transfer_row = pv_slot * H
                            pl.system.sync_wait(
                                QK_PV_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV
                            )
                            pv_m = pl.load(
                                mi_transfer,
                                [pv_transfer_row + qk_lane_head, 0],
                                [H // 2, 1],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            pv_l = pl.load(
                                li_transfer,
                                [pv_transfer_row + qk_lane_head, 0],
                                [H // 2, 1],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            next_m = pl.maximum(m_iter, pv_m)
                            alpha = pl.exp(pl.sub(m_iter, next_m))
                            beta = pl.exp(pl.sub(pv_m, next_m))
                            next_l = pl.add(pl.mul(alpha, l_iter), pl.mul(beta, pv_l))
                            pv_left = pl.load(
                                pv_transfer,
                                [pv_transfer_row + qk_lane_head, 0],
                                [H // 2, HEAD_DIM // 2],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            next_left = pl.add(
                                pl.row_expand_mul(left_iter, alpha), pl.row_expand_mul(pv_left, beta)
                            )
                            pv_right = pl.load(
                                pv_transfer,
                                [pv_transfer_row + qk_lane_head, HEAD_DIM // 2],
                                [H // 2, HEAD_DIM // 2],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            next_right = pl.add(
                                pl.row_expand_mul(right_iter, alpha), pl.row_expand_mul(pv_right, beta)
                            )
                            m_valid, l_valid, left_valid, right_valid = pl.yield_(
                                next_m, next_l, next_left, next_right
                            )
                        else:
                            m_valid, l_valid, left_valid, right_valid = pl.yield_(
                                m_iter, l_iter, left_iter, right_iter
                            )
                        m_after, l_after, left_after, right_after = pl.yield_(
                            m_valid, l_valid, left_valid, right_valid
                        )
                    else:
                        m_after, l_after, left_after, right_after = pl.yield_(
                            m_iter, l_iter, left_iter, right_iter
                        )
                    running_m, running_l, running_left, running_right = pl.yield_(
                        m_after, l_after, left_after, right_after
                    )
                qk_output_row = qk_t * H + qk_lane_head
                pl.store(running_m, [qk_output_row, 0], sparse_mi)
                pl.store(running_l, [qk_output_row, 0], sparse_li)
                pl.store(running_left, [qk_output_row, 0], sparse_oi)
                pl.store(running_right, [qk_output_row, HEAD_DIM // 2], sparse_oi)

        # Publish this core's completed query rows without launching another task.
        for merge_aiv in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            merge_lane_head0 = merge_aiv * (H // 2)
            merge_swap_col = pl.load(rope_swap_idx, [0, 0], [1, ROPE_DIM])
            merge_swap_offset = pl.add(pl.cast(merge_swap_col, target_type=pl.FP32), NOPE_DIM)
            merge_swap_grid = pl.col_expand_add(
                pl.tile.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0),
                merge_swap_offset,
            )
            merge_row_ids = pl.tile.arange(0, [1, H_TILE], dtype=pl.INT32)
            merge_row_offsets = pl.reshape(
                pl.mul(pl.cast(merge_row_ids, target_type=pl.FP32), HEAD_DIM),
                [H_TILE, 1],
            )
            merge_swap_flat = pl.row_expand_add(merge_swap_grid, merge_row_offsets)
            merge_swap_indices = pl.cast(merge_swap_flat, target_type=pl.INT32)
            merge_gather_tmp = pl.create_tile([H_TILE, ROPE_DIM], dtype=pl.INT32)
            for merge_t in pl.range(qk_core, T, NUM_QK_CORES):
                merge_cos = pl.load(rope_cos_il, [merge_t, 0], [1, ROPE_DIM])
                merge_sin = pl.load(rope_sin_signed, [merge_t, 0], [1, ROPE_DIM])
                for merge_tile in pl.unroll((H // 2) // H_TILE):
                    merge_lane_head = merge_tile * H_TILE
                    merge_head0 = merge_lane_head0 + merge_lane_head
                    merge_partial_row0 = merge_t * H + merge_head0
                    merge_m = pl.load(sparse_mi, [merge_partial_row0, 0], [H_TILE, 1])
                    merge_l = pl.load(sparse_li, [merge_partial_row0, 0], [H_TILE, 1])
                    merge_out = pl.load(sparse_oi, [merge_partial_row0, 0], [H_TILE, HEAD_DIM])
                    merge_sink = pl.load(attn_sink_col, [merge_head0, 0], [H_TILE, 1])
                    merge_sink_exp = pl.exp(pl.sub(merge_sink, merge_m))
                    merge_denom = pl.add(merge_l, merge_sink_exp)
                    merge_normed = pl.row_expand_div(merge_out, merge_denom)
                    merge_normed_bf16 = pl.cast(merge_normed, target_type=pl.BF16, mode="rint")
                    merge_nope = merge_normed_bf16[:, 0:NOPE_DIM]
                    merge_rope = merge_normed[:, NOPE_DIM:HEAD_DIM]
                    merge_swapped = pl.tile.gather(
                        merge_normed,
                        merge_swap_indices,
                        merge_gather_tmp,
                    )
                    inverse_cos = pl.col_expand_mul(merge_rope, merge_cos)
                    inverse_sin = pl.col_expand_mul(merge_swapped, merge_sin)
                    inverse_rope = pl.sub(inverse_cos, inverse_sin)
                    inverse_rope_bf16 = pl.cast(inverse_rope, target_type=pl.BF16, mode="rint")

                    for merge_group in pl.unroll(H_TILE // HEADS_PER_GROUP):
                        group = merge_head0 // HEADS_PER_GROUP + merge_group
                        group_head = merge_group * HEADS_PER_GROUP
                        packed_row = (group * LOCAL_T_PAD + merge_t) * HEADS_PER_GROUP
                        packed_nope = pl.slice(
                            merge_nope,
                            [HEADS_PER_GROUP, NOPE_DIM],
                            [group_head, 0],
                        )
                        packed_rope = pl.slice(
                            inverse_rope_bf16,
                            [HEADS_PER_GROUP, ROPE_DIM],
                            [group_head, 0],
                        )
                        pl.store(packed_nope, [packed_row, 0], o_packed_flat)
                        pl.store(packed_rope, [packed_row, NOPE_DIM], o_packed_flat)

    return o_packed_heads
