# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl

from .config import BLOCK_SIZE, DECODE_SEQ, FLASH as M, MOE_TOKENS

from .decode_o_proj import LOCAL_T_PAD

from .qkv_proj_rope import kv_proj_rope, q_proj_rope, rope_prepare

ORI_BLOCK_NUM_DYN = pl.dynamic("NATIVE_D_DSPARK_ATTENTION_ORI_BLOCK_NUM_DYN")

KV_T_DYN = pl.dynamic("NATIVE_D_DSPARK_ATTENTION_KV_T_DYN")

B = MOE_TOKENS // DECODE_SEQ

S = 5  # anchor-first draft query rows per request

T = B * S

D = M.hidden_size

H = M.num_attention_heads

HEAD_DIM = M.head_dim

ROPE_DIM = M.qk_rope_head_dim

NOPE_DIM = M.nope_head_dim

Q_LORA = M.q_lora_rank

WIN = M.sliding_window

O_GROUPS = M.o_groups

HEADS_PER_GROUP = H // O_GROUPS

SOFTMAX_SCALE = M.softmax_scale

VISIBLE_ROWS = WIN + S  # trailing window + whole draft block

ATTN_K_TILE = 64

NUM_QK_CORES = 24

QK_PRE_LAUNCH = 2

QK_TRANSFER_SLOTS = QK_PRE_LAUNCH + 1

QK_SCORE_READY_EVENT = 0

QK_PROB_READY_EVENT = 1

QK_PV_READY_EVENT = 2

SPARSE_BLOCKS = (VISIBLE_ROWS + ATTN_K_TILE - 1) // ATTN_K_TILE

INDEX_WIDTH = SPARSE_BLOCKS * ATTN_K_TILE

BIAS_B_TILE = 8  # 2 request blocks

H_TILE = 16  # 4 head blocks

NEG_INF = -1.0e20


@pl.jit.inline
def dspark_attention(
    x: pl.Tensor[[T, D], pl.BF16],
    kv_x: pl.Tensor[[KV_T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16, pl.NZ],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos_local: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin_local: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[KV_T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[KV_T_DYN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_position_ids: pl.Tensor[[KV_T_DYN], pl.INT32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    kv_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    swa_indices: pl.Tensor[[B, INDEX_WIDTH], pl.INT32],
    swa_lens: pl.Tensor[[B], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    o_packed_heads: pl.Tensor[[O_GROUPS * LOCAL_T_PAD * HEADS_PER_GROUP, HEAD_DIM], pl.BF16],
):
    kv_tokens = pl.tensor.dim(kv_position_ids, 0)
    rope_cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_swap_idx = pl.create_tensor([T, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(
        freqs_cos_local,
        freqs_sin_local,
        rope_cos_il,
        rope_sin_signed,
        rope_swap_idx,
    )

    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    q_proj_rope(
        x,
        wq_a,
        wq_b,
        wq_b_scale,
        gamma_cq,
        rope_cos_il,
        rope_sin_signed,
        rope_swap_idx,
        q,
        qr,
        qr_scale,
    )

    kv_rope_cos_il = pl.create_tensor([kv_tokens, ROPE_DIM], dtype=pl.FP32)
    kv_rope_sin_signed = pl.create_tensor([kv_tokens, ROPE_DIM], dtype=pl.FP32)
    kv_rope_swap_idx = pl.create_tensor([kv_tokens, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(
        freqs_cos,
        freqs_sin,
        kv_rope_cos_il,
        kv_rope_sin_signed,
        kv_rope_swap_idx,
    )

    # DSA-CP keeps Q on its token owner while every rank updates the complete
    # group KV stream before running attention for its local queries.
    late_dep = pl.system.task_dummy(deps=[])
    kv = pl.create_tensor([kv_tokens, HEAD_DIM], dtype=pl.BF16)
    kv_proj_rope(
        kv_x,
        wkv,
        gamma_ckv,
        kv_rope_cos_il,
        kv_rope_sin_signed,
        kv_rope_swap_idx,
        kv,
        late_dep,
    )

    # Commit the block's own KV and build the visible-length mask in one task; the
    # gather below reads those rows back through swa_indices.
    ori_block_num = pl.tensor.dim(kv_cache, 0)
    kv_cache_flat = pl.reshape(kv_cache, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    sparse_bias = pl.create_tensor([B, INDEX_WIDTH], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_kv_commit_valid_bias"):
        for write_t in pl.range(kv_tokens):
            write_row_i64 = pl.read(kv_slot_mapping, [write_t])
            if write_row_i64 >= 0:
                write_row = pl.cast(write_row_i64, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, 0:HEAD_DIM] = kv[write_t : write_t + 1, 0:HEAD_DIM]
        v_col = pl.cast(pl.arange(0, [1, INDEX_WIDTH], dtype=pl.INT32), target_type=pl.FP32)
        for v_blk in pl.range((B + BIAS_B_TILE - 1) // BIAS_B_TILE):
            v_b0 = v_blk * BIAS_B_TILE
            v_rows = pl.min(BIAS_B_TILE, B - v_b0)
            v_col_m = pl.col_expand(pl.full([BIAS_B_TILE, INDEX_WIDTH], dtype=pl.FP32, value=0.0), v_col)
            v_lens_i32 = pl.slice(swa_lens, [BIAS_B_TILE], [v_b0], valid_shape=[v_rows])
            v_lens = pl.cast(pl.reshape(v_lens_i32, [BIAS_B_TILE, 1]), target_type=pl.FP32)
            v_valid = pl.minimum(pl.maximum(pl.neg(pl.row_expand_sub(v_col_m, v_lens)), 0.0), 1.0)
            v_idx_i32 = pl.slice(
                swa_indices, [BIAS_B_TILE, INDEX_WIDTH], [v_b0, 0], valid_shape=[v_rows, INDEX_WIDTH]
            )
            v_idx = pl.cast(v_idx_i32, target_type=pl.FP32)
            v_mapped = pl.minimum(pl.maximum(pl.add(v_idx, 1.0), 0.0), 1.0)
            v_valid = pl.mul(v_valid, v_mapped)
            v_bias = pl.mul(pl.sub(v_valid, 1.0), -NEG_INF)
            v_bias = pl.set_validshape(v_bias, v_rows, INDEX_WIDTH)
            sparse_bias[v_b0 : v_b0 + BIAS_B_TILE, 0:INDEX_WIDTH] = v_bias

    # One index row per request: build_dspark_swa_indices repeats it across the
    # block's query rows, so the gather runs B times, not T.
    visible_kv = pl.create_tensor([B * INDEX_WIDTH, HEAD_DIM], dtype=pl.BF16)
    for g_task in pl.spmd(B * SPARSE_BLOCKS, name_hint="dspark_gather_kv"):
        g_b = g_task // SPARSE_BLOCKS
        g_c0 = (g_task % SPARSE_BLOCKS) * ATTN_K_TILE
        g_base = g_b * INDEX_WIDTH
        if g_c0 < pl.read(swa_lens, [g_b]):
            for g_c in pl.range(g_c0, g_c0 + ATTN_K_TILE):
                g_slot_i32 = pl.read(swa_indices, [g_b, g_c])
                g_dst = g_base + g_c
                if g_slot_i32 >= 0:
                    g_slot = pl.cast(g_slot_i32, pl.INDEX)
                    visible_kv[g_dst : g_dst + 1, 0:HEAD_DIM] = kv_cache_flat[g_slot : g_slot + 1, 0:HEAD_DIM]
                else:
                    visible_kv[g_dst : g_dst + 1, 0:HEAD_DIM] = pl.full(
                        [1, HEAD_DIM], dtype=pl.BF16, value=0.0
                    )

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

    return kv_cache, o_packed_heads
