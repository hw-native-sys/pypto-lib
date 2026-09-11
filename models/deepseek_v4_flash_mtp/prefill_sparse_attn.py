# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 token-major prefill sparse attention with grouped output projection.

Sparse index contract: ``swa_indices`` holds physical ori-KV rows, ``cmp_indices``
holds compressed logical slots lowered through ``cmp_block_table``, ``-1`` invalid.
"""

import pypto.language as pl

from config import (
    BLOCK_SIZE,
    FLASH as M,
    FP32_NEG_INF,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    PREFILL_BATCH,
    PREFILL_CMP_BLOCK_NUM,
    PREFILL_CMP_MAX_BLOCKS,
    PREFILL_ORI_BLOCK_NUM,
    PREFILL_SEQ,
)
from prefill_indexer import INDEXER_TOPK_CAP
from hc_post import hc_post_prefill

# Dynamic shape variables.
ORI_BLOCK_NUM_DYN = pl.dynamic("PREFILL_ORI_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CMP_BLOCK_NUM_DYN")
CMP_STORAGE_BLOCK_SIZE_DYN = pl.dynamic("PREFILL_CMP_STORAGE_BLOCK_SIZE_DYN")
PHYSICAL_ORI_BLOCK_NUM_DYN = pl.dynamic("PREFILL_PHYSICAL_ORI_BLOCK_NUM_DYN")
PREFILL_ATTN_STATS_ROWS_DYN = pl.dynamic("PREFILL_ATTN_STATS_ROWS_DYN")
PREFILL_ATTN_ROWS_DYN = pl.dynamic("PREFILL_ATTN_ROWS_DYN")
PREFILL_ATTN_SOURCE_ROWS_DYN = pl.dynamic("PREFILL_ATTN_SOURCE_ROWS_DYN")
PREFILL_ATTN_PACKED_ROWS_DYN = pl.dynamic("PREFILL_ATTN_PACKED_ROWS_DYN")
PREFILL_ATTN_ROPE_ROWS_DYN = pl.dynamic("PREFILL_ATTN_ROPE_ROWS_DYN")

# HCA scratch extents are independent of compressed-cache capacity.
HCA_FULL_BLOCK_NUM_DYN = pl.dynamic("HCA_FULL_BLOCK_NUM_DYN")
HCA_RAW_ROWS_DYN = pl.dynamic("HCA_RAW_ROWS_DYN")
HCA_HEAD_ROWS_DYN = pl.dynamic("HCA_HEAD_ROWS_DYN")
HCA_PARTIAL_ROWS_DYN = pl.dynamic("HCA_PARTIAL_ROWS_DYN")

# model config
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
D = M.hidden_size
HC_MULT = M.hc_mult
HC_COMB = HC_MULT * HC_MULT
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_HALF = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
IDX_TOPK = M.index_topk
WIN = M.sliding_window
TOPK = WIN + IDX_TOPK
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
SUPPORTED_COMPRESS_RATIOS = (0, 4, 128)
DEFAULT_COMPRESS_RATIO = 4

# paged KV cache
PREFILL_MAX_COMPRESSED = INDEXER_TOPK_CAP
PREFILL_SPARSE_TOPK = min(WIN, S) + PREFILL_MAX_COMPRESSED
ORI_BLOCK_NUM = PREFILL_ORI_BLOCK_NUM
CMP_MAX_BLOCKS = PREFILL_CMP_MAX_BLOCKS
CMP_BLOCK_NUM = PREFILL_CMP_BLOCK_NUM

# tiling
HEAD_TILE = 16               # storage / merge head-tile
QK_M_TILE = 32               # head rows cube-batched per QK/PV matmul
QK_TASKS = min(24, T)
QK_PRE_LAUNCH = 2
QK_TRANSFER_SLOTS = QK_PRE_LAUNCH + 1
QK_SCORE_READY_EVENT = 0
QK_PROB_READY_EVENT = 1
QK_PV_READY_EVENT = 2
GATHER_TOKEN_TILE = 2
BIAS_TOKEN_TILE = 16
QUANT_TOKEN_TILE = 8
ROPE_TILE = 16
ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE
A_K_TILE = 256               # proj_a cube K frag
B_K_TILE = 256               # proj_b_mm cube K frag
QUANT_TILE = 512             # quant column tile over O_LORA
PROJ_A_MM_N_TILE = 128       # proj_a cube N frag
PROJ_B_MM_N_TILE = 256       # proj_b_mm cube N frag; Acc = T*N*4 sits on the a2a3 L0C wall
PROJ_B_ACT_N_TILE = 512      # proj_b_act vector N frag
PROJ_B_ACT_T_TILE = 16       # proj_b_act inner token tile for the O_GROUPS-way INT32->FP32 accumulate
PROJ_B_ACT_TASK_T_TILE = 32  # proj_b_act token block per task
# Task-array fan-outs; named because deps= list comprehensions cannot be hoisted out of the call.
PA_NFRAGS = O_LORA // PROJ_A_MM_N_TILE
PREFILL_ATTN_TILE = 128      # sparse-K rows per compile-time block
PREFILL_ATTN_BLOCKS = (PREFILL_SPARSE_TOPK + PREFILL_ATTN_TILE - 1) // PREFILL_ATTN_TILE
PREFILL_SPARSE_PAD = PREFILL_ATTN_BLOCKS * PREFILL_ATTN_TILE
MASK_ALIGN_ELEMS = 32 // 4
VALID_BLOCK_MASK_COLS = ((PREFILL_ATTN_BLOCKS + MASK_ALIGN_ELEMS - 1) // MASK_ALIGN_ELEMS) * MASK_ALIGN_ELEMS
# Padded sparse-window columns carrying real metadata entries.
SPARSE_BIAS_COLS = min(TOPK, PREFILL_SPARSE_PAD)
SPARSE_CMP_BIAS_COLS = max(0, SPARSE_BIAS_COLS - WIN)

# Shared attention derives query extents from tensors. The query tile only
# bounds reusable scratch in the paged sparse gather path.
STAGED_SWA_QUERY_TILE = 128
STAGED_SWA_QUERY_STATS_ROWS = STAGED_SWA_QUERY_TILE * H
STAGED_SWA_ROPE_CS_T_TILE = 8
STAGED_SWA_PROJ_A_ROW_TILE = 256
STAGED_SWA_PROJ_B_ROW_TILE = 128
STAGED_SWA_PROJ_B_D_TILE = 512
STAGED_SWA_PB_DSLABS = D // STAGED_SWA_PROJ_B_D_TILE

# Ratio-128 HCA reads the predecessor window and current runtime segment
# from a physical raw cache, plus compressed history in BLOCK_SIZE-row pages.
HCA_COMPRESS_RATIO = 128
HCA_GATHER_TOKEN_TILE = 2
HCA_ATTN_TILE = 128
HCA_MAX_COMPRESSED_ROWS = (M.max_position_embeddings + HCA_COMPRESS_RATIO - 1) // HCA_COMPRESS_RATIO
HCA_CMP_WORK_COUNT = (HCA_MAX_COMPRESSED_ROWS + HCA_ATTN_TILE - 1) // HCA_ATTN_TILE
HCA_CMP_PAD_ROWS = HCA_CMP_WORK_COUNT * HCA_ATTN_TILE

assert WIN == PREFILL_ATTN_TILE, f"Sparse prefill expects WIN ({WIN}) == PREFILL_ATTN_TILE ({PREFILL_ATTN_TILE})"

@pl.jit.inline(auto_scope=False)
def _staged_attn_prepare_rope(
    freqs_cos: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, ROPE_DIM], pl.BF16],
    rope_cos_il: pl.Tensor[[PREFILL_ATTN_ROPE_ROWS_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[PREFILL_ATTN_ROPE_ROWS_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[HEAD_TILE, ROPE_DIM], pl.INT32],
    active_rows: pl.Scalar[pl.INDEX],
    prior_dep: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Build inverse-RoPE tables for one staged segment."""
    rope_cs_blocks = (active_rows + STAGED_SWA_ROPE_CS_T_TILE - 1) // STAGED_SWA_ROPE_CS_T_TILE
    with pl.spmd(ROPE_HALF // ROPE_TILE, name_hint="staged_swa_rope_cs", deps=[prior_dep]) as rope_cs_tid:
        cp = pl.tile.get_block_idx()
        cp_r0 = cp * ROPE_TILE
        cp_c0 = 2 * cp_r0

        swap_ones = pl.full([HEAD_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        swap_ramp = pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32)
        swap_col = pl.col_expand_mul(swap_ones, swap_ramp)
        swap_dup_i32 = pl.cast(pl.mul(swap_col, 0.5), target_type=pl.INT32, mode="trunc")
        swap_dup_f = pl.cast(swap_dup_i32, target_type=pl.FP32)
        swap_lane = pl.sub(swap_col, pl.mul(swap_dup_f, 2.0))
        swap_pair_f = pl.sub(pl.add(swap_col, 1.0), pl.mul(swap_lane, 2.0))
        swap_idx = pl.cast(swap_pair_f, target_type=pl.INT32)
        rope_swap_idx[:, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = swap_idx[:, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE]

        cs_ones = pl.full([STAGED_SWA_ROPE_CS_T_TILE, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=1.0)
        cs_ramp_i32 = pl.arange(0, [1, ROPE_INTERLEAVE_TILE], dtype=pl.INT32)
        cs_ramp = pl.cast(cs_ramp_i32, target_type=pl.FP32)
        cs_col = pl.col_expand_mul(cs_ones, cs_ramp)
        cs_dup_i32 = pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode="trunc")
        cs_dup_f = pl.cast(cs_dup_i32, target_type=pl.FP32)
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))
        for cs_rb in pl.range(rope_cs_blocks):
            cs_t0 = cs_rb * STAGED_SWA_ROPE_CS_T_TILE
            cs_rows = pl.min(STAGED_SWA_ROPE_CS_T_TILE, active_rows - cs_t0)
            cs_cos_rows = pl.slice(
                freqs_cos,
                [STAGED_SWA_ROPE_CS_T_TILE, ROPE_TILE],
                [cs_t0, cp_r0],
                valid_shape=[cs_rows, ROPE_TILE],
            )
            cs_sin_rows = pl.slice(
                freqs_sin,
                [STAGED_SWA_ROPE_CS_T_TILE, ROPE_TILE],
                [cs_t0, cp_r0],
                valid_shape=[cs_rows, ROPE_TILE],
            )
            cs_cos = pl.cast(cs_cos_rows, target_type=pl.FP32)
            cs_sin = pl.cast(cs_sin_rows, target_type=pl.FP32)
            cs_cos_il = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
            cs_sin_il = pl.gather(cs_sin, dim=-1, index=cs_dup_idx)
            cs_sin_signed = pl.mul(cs_sin_il, cs_sign)
            rope_cos_il[cs_t0 : cs_t0 + STAGED_SWA_ROPE_CS_T_TILE, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = cs_cos_il
            rope_sin_signed[
                cs_t0 : cs_t0 + STAGED_SWA_ROPE_CS_T_TILE,
                cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE,
            ] = cs_sin_signed

    return rope_cs_tid


@pl.jit.inline(auto_scope=False)
def _staged_sparse_wave(
    q: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, H, HEAD_DIM], pl.BF16],
    sparse_kv: pl.Tensor[[PREFILL_ATTN_SOURCE_ROWS_DYN, HEAD_DIM], pl.BF16],
    sparse_bias: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, PREFILL_SPARSE_PAD], pl.FP32],
    valid_block_mask: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, VALID_BLOCK_MASK_COLS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    active_rows: pl.Scalar[pl.INDEX],
    o_packed_heads: pl.Tensor[[PREFILL_ATTN_PACKED_ROWS_DYN, HEAD_DIM], pl.BF16],
    sparse_blk_mi: pl.Tensor[[PREFILL_ATTN_STATS_ROWS_DYN, 1], pl.FP32],
    sparse_blk_li: pl.Tensor[[PREFILL_ATTN_STATS_ROWS_DYN, 1], pl.FP32],
    sparse_blk_oi: pl.Tensor[[PREFILL_ATTN_STATS_ROWS_DYN, HEAD_DIM], pl.FP32],
    rope_cos_il: pl.Tensor[[PREFILL_ATTN_ROPE_ROWS_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[PREFILL_ATTN_ROPE_ROWS_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[HEAD_TILE, ROPE_DIM], pl.INT32],
    source_ready_tid: pl.Scalar[pl.TASK_ID],
    merge_ready_tid: pl.Scalar[pl.TASK_ID],
    query_base: pl.Scalar[pl.INDEX],
) -> pl.Scalar[pl.TASK_ID]:
    """Run a QK/PV/merge wave bounded by caller-owned statistics scratch."""
    wave_rows = pl.tensor.dim(sparse_blk_mi, 0) // H
    wave_end = pl.min(active_rows, query_base + wave_rows)
    token_rows = pl.tensor.dim(q, 0)
    packed_rows = pl.tensor.dim(o_packed_heads, 0) // H
    query_head_rows = token_rows * H
    output_group_rows = O_GROUPS * packed_rows
    q_flat = pl.reshape(q, [query_head_rows, HEAD_DIM])
    attn_sink_col = pl.reshape(attn_sink, [H, 1])
    o_packed = pl.reshape(o_packed_heads, [output_group_rows, O_GROUP_IN])

    # Pipeline K blocks and keep the online-softmax state local to each query.
    transfer_slots = QK_TASKS * QK_TRANSFER_SLOTS
    transfer_heads = transfer_slots * H
    score_transfer = pl.create_tensor([transfer_heads, PREFILL_ATTN_TILE], dtype=pl.FP32)
    probability_transfer = pl.create_tensor([transfer_heads, PREFILL_ATTN_TILE], dtype=pl.BF16)
    pv_transfer = pl.create_tensor([transfer_heads, HEAD_DIM], dtype=pl.FP32)
    mi_transfer = pl.create_tensor([transfer_heads, 1], dtype=pl.FP32)
    li_transfer = pl.create_tensor([transfer_heads, 1], dtype=pl.FP32)
    ffts_workspace = pl.create_tensor([256], dtype=pl.INT64)
    with pl.spmd(QK_TASKS, name_hint="staged_swa_qk_pv", deps=[source_ready_tid], allow_early_resolve=True) as qk_tid:
        qk_core = pl.tile.get_block_idx()
        pl.system.set_ffts(ffts_workspace)
        for qk_t in pl.range(query_base + qk_core, wave_end, QK_TASKS):
            qk_q = pl.load(
                q_flat, [qk_t * H, 0], [H, HEAD_DIM], target_memory=pl.MemorySpace.Mat,
            )
            for qk_tick in pl.range(PREFILL_ATTN_BLOCKS + QK_PRE_LAUNCH):
                if qk_tick < PREFILL_ATTN_BLOCKS:
                    qk_sb = qk_tick
                    if pl.read(valid_block_mask, [qk_t, qk_sb]) > 0:
                        qk_slot = qk_core * QK_TRANSFER_SLOTS + qk_sb % QK_TRANSFER_SLOTS
                        qk_kv_row = qk_t * PREFILL_SPARSE_PAD + qk_sb * PREFILL_ATTN_TILE
                        qk_transfer_row = qk_slot * H
                        qk_kv = pl.load(
                            sparse_kv, [qk_kv_row, 0], [PREFILL_ATTN_TILE, HEAD_DIM],
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
                        pv_kv_row = qk_t * PREFILL_SPARSE_PAD + pv_sb * PREFILL_ATTN_TILE
                        pv_transfer_row = pv_slot * H
                        pl.system.sync_wait(QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                        pv_probability = pl.load(
                            probability_transfer, [pv_transfer_row, 0], [H, PREFILL_ATTN_TILE],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        pv_kv = pl.load(
                            sparse_kv, [pv_kv_row, 0], [PREFILL_ATTN_TILE, HEAD_DIM],
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
                qk_reduce_tmp = pl.create_tile([H // 2, PREFILL_ATTN_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                running_m = pl.load(attn_sink_col, [qk_lane_head, 0], [H // 2, 1], target_memory=pl.MemorySpace.Vec)
                running_l = pl.tile.muls(running_m, 0.0)
                running_left = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
                running_right = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
                for qk_tick, (m_iter, l_iter, left_iter, right_iter) in pl.range(
                    PREFILL_ATTN_BLOCKS + QK_PRE_LAUNCH,
                    init_values=(running_m, running_l, running_left, running_right),
                ):
                    if qk_tick < PREFILL_ATTN_BLOCKS:
                        qk_sb = qk_tick
                        if pl.read(valid_block_mask, [qk_t, qk_sb]) > 0:
                            qk_slot = qk_core * QK_TRANSFER_SLOTS + qk_sb % QK_TRANSFER_SLOTS
                            qk_transfer_row = qk_slot * H
                            qk_s0 = qk_sb * PREFILL_ATTN_TILE
                            pl.system.sync_wait(QK_SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                            qk_scores_half = pl.load(
                                score_transfer, [qk_transfer_row + qk_lane_head, 0], [H // 2, PREFILL_ATTN_TILE],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            qk_bias = pl.load(
                                sparse_bias, [qk_t, qk_s0], [1, PREFILL_ATTN_TILE], target_memory=pl.MemorySpace.Vec,
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
                qk_output_row = (qk_t - query_base) * H + qk_lane_head
                pl.store(running_m, [qk_output_row, 0], sparse_blk_mi)
                pl.store(running_l, [qk_output_row, 0], sparse_blk_li)
                pl.store(running_left, [qk_output_row, 0], sparse_blk_oi)
                pl.store(running_right, [qk_output_row, HEAD_DIM // 2], sparse_blk_oi)

    with pl.spmd(wave_rows, name_hint="staged_swa_merge_rope_pack", deps=[qk_tid, merge_ready_tid]) as merge_tid:
        m_local_t = pl.tile.get_block_idx()
        m_t = query_base + m_local_t
        if m_t < active_rows:
            m_token_base = m_local_t * H
            m_swap_idx = rope_swap_idx[:, :]
            m_cos_il = rope_cos_il[m_t : m_t + 1, :]
            m_sin_signed = rope_sin_signed[m_t : m_t + 1, :]
            for m_h_idx in pl.range(H // HEAD_TILE):
                m_h0 = m_h_idx * HEAD_TILE
                m_blk_base = m_token_base + m_h_idx * HEAD_TILE
                m_mi = sparse_blk_mi[m_blk_base : m_blk_base + HEAD_TILE, :]
                m_li = sparse_blk_li[m_blk_base : m_blk_base + HEAD_TILE, :]
                m_oi = sparse_blk_oi[m_blk_base : m_blk_base + HEAD_TILE, :]
                sink_bias = attn_sink_col[m_h0 : m_h0 + HEAD_TILE, :]
                sink_tile = pl.add(pl.sub(m_mi, m_mi), sink_bias)
                denom = pl.add(m_li, pl.exp(pl.sub(sink_tile, m_mi)))
                n_full = pl.row_expand_div(m_oi, denom)[0:HEAD_TILE, :]
                n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")

                m_rope = n_full[:, NOPE_DIM:HEAD_DIM]
                m_swapped = pl.gather(m_rope, dim=-1, index=m_swap_idx)
                m_rot = pl.add(pl.col_expand_mul(m_rope, m_cos_il), pl.col_expand_mul(m_swapped, m_sin_signed))
                n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode="rint")

                if HEAD_TILE % HEADS_PER_GROUP == 0:
                    m_g0 = m_h0 // HEADS_PER_GROUP
                    for m_sg in pl.unroll(HEAD_TILE // HEADS_PER_GROUP):
                        m_src_h0 = m_sg * HEADS_PER_GROUP
                        m_pack_row = (m_g0 + m_sg) * packed_rows + m_t
                        m_dst_head = m_pack_row * HEADS_PER_GROUP
                        pl.assemble(
                            o_packed_heads,
                            n_bf16[m_src_h0 : m_src_h0 + HEADS_PER_GROUP, 0:NOPE_DIM],
                            [m_dst_head, 0],
                        )
                        pl.assemble(
                            o_packed_heads,
                            n_rope_bf16[m_src_h0 : m_src_h0 + HEADS_PER_GROUP, 0:ROPE_DIM],
                            [m_dst_head, NOPE_DIM],
                        )
                else:
                    for m_hi in pl.range(HEAD_TILE):
                        m_gh = m_h0 + m_hi
                        m_g = m_gh // HEADS_PER_GROUP
                        m_pack_row = m_g * packed_rows + m_t
                        m_col = (m_gh - m_g * HEADS_PER_GROUP) * HEAD_DIM
                        o_packed[
                            m_pack_row : m_pack_row + 1,
                            m_col : m_col + NOPE_DIM,
                        ] = n_bf16[m_hi : m_hi + 1, 0:NOPE_DIM]
                        o_packed[
                            m_pack_row : m_pack_row + 1,
                            m_col + NOPE_DIM : m_col + HEAD_DIM,
                        ] = n_rope_bf16[m_hi : m_hi + 1, 0:ROPE_DIM]

    return merge_tid


@pl.jit.inline(auto_scope=False)
def _staged_swa_heads(
    q: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, H, HEAD_DIM], pl.BF16],
    sparse_kv: pl.Tensor[[PREFILL_ATTN_SOURCE_ROWS_DYN, HEAD_DIM], pl.BF16],
    sparse_bias: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, PREFILL_SPARSE_PAD], pl.FP32],
    valid_block_mask: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, VALID_BLOCK_MASK_COLS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    active_rows: pl.Scalar[pl.INDEX],
    freqs_cos: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, ROPE_DIM], pl.BF16],
    o_packed_heads: pl.Tensor[[PREFILL_ATTN_PACKED_ROWS_DYN, HEAD_DIM], pl.BF16],
    packed_init_tid: pl.Scalar[pl.TASK_ID],
    prior_dep: pl.Scalar[pl.TASK_ID],
) -> tuple[pl.Tensor, pl.Scalar[pl.TASK_ID]]:
    """Write a segment through serial reusable query waves."""
    token_rows = pl.tensor.dim(q, 0)
    rope_rows = ((token_rows + STAGED_SWA_ROPE_CS_T_TILE - 1) // STAGED_SWA_ROPE_CS_T_TILE) * STAGED_SWA_ROPE_CS_T_TILE
    merge_tids = pl.array.create(1, pl.TASK_ID)
    merge_tids[0] = packed_init_tid
    with pl.scope():
        sparse_blk_mi = pl.create_tensor([STAGED_SWA_QUERY_STATS_ROWS, 1], dtype=pl.FP32)
        sparse_blk_li = pl.create_tensor([STAGED_SWA_QUERY_STATS_ROWS, 1], dtype=pl.FP32)
        sparse_blk_oi = pl.create_tensor([STAGED_SWA_QUERY_STATS_ROWS, HEAD_DIM], dtype=pl.FP32)
        rope_cos_il = pl.create_tensor([rope_rows, ROPE_DIM], dtype=pl.FP32, manual_dep=True)
        rope_sin_signed = pl.create_tensor([rope_rows, ROPE_DIM], dtype=pl.FP32, manual_dep=True)
        rope_swap_idx = pl.create_tensor([HEAD_TILE, ROPE_DIM], dtype=pl.INT32, manual_dep=True)
        rope_cs_tid = _staged_attn_prepare_rope(
            freqs_cos,
            freqs_sin,
            rope_cos_il,
            rope_sin_signed,
            rope_swap_idx,
            active_rows,
            prior_dep,
        )
        merge_tids[0] = pl.system.task_dummy(deps=[packed_init_tid, rope_cs_tid])
        for query_block in pl.range((active_rows + STAGED_SWA_QUERY_TILE - 1) // STAGED_SWA_QUERY_TILE):
            query_base = query_block * STAGED_SWA_QUERY_TILE
            prior_merge_tid = merge_tids[0]
            merge_tid = prior_merge_tid
            if query_base < active_rows:
                merge_tid = _staged_sparse_wave(
                    q,
                    sparse_kv,
                    sparse_bias,
                    valid_block_mask,
                    attn_sink,
                    active_rows,
                    o_packed_heads,
                    sparse_blk_mi,
                    sparse_blk_li,
                    sparse_blk_oi,
                    rope_cos_il,
                    rope_sin_signed,
                    rope_swap_idx,
                    prior_merge_tid,
                    prior_merge_tid,
                    query_base,
                )
            merge_tids[0] = merge_tid

    return o_packed_heads, merge_tids[0]


@pl.jit.inline(auto_scope=False)
def _physical_sparse_wave(
    q: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[PHYSICAL_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE_DYN, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[CMP_MAX_BLOCKS], pl.INT32],
    cmp_indices: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, IDX_TOPK], pl.INT32],
    valid_block_mask: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, VALID_BLOCK_MASK_COLS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    active_rows: pl.Scalar[pl.INDEX],
    o_packed_heads: pl.Tensor[[PREFILL_ATTN_PACKED_ROWS_DYN, HEAD_DIM], pl.BF16],
    sparse_kv: pl.Tensor[[PREFILL_ATTN_SOURCE_ROWS_DYN, HEAD_DIM], pl.BF16],
    sparse_bias: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, PREFILL_SPARSE_PAD], pl.FP32],
    sparse_blk_mi: pl.Tensor[[PREFILL_ATTN_STATS_ROWS_DYN, 1], pl.FP32],
    sparse_blk_li: pl.Tensor[[PREFILL_ATTN_STATS_ROWS_DYN, 1], pl.FP32],
    sparse_blk_oi: pl.Tensor[[PREFILL_ATTN_STATS_ROWS_DYN, HEAD_DIM], pl.FP32],
    rope_cos_il: pl.Tensor[[PREFILL_ATTN_ROPE_ROWS_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[PREFILL_ATTN_ROPE_ROWS_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[HEAD_TILE, ROPE_DIM], pl.INT32],
    raw_ready_tid: pl.Scalar[pl.TASK_ID],
    compressed_ready_tid: pl.Scalar[pl.TASK_ID],
    merge_ready_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Run one DSpark-style gather/QK-PV/merge wave from physical roots."""
    token_rows = pl.tensor.dim(q, 0)
    source_rows = pl.tensor.dim(sparse_kv, 0)
    source_view = pl.reshape(sparse_kv, [source_rows, HEAD_DIM])
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    ori_kv_flat = pl.reshape(ori_kv, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    cmp_page_rows = pl.tensor.dim(cmp_kv, 1)
    gather_blocks = (token_rows + GATHER_TOKEN_TILE - 1) // GATHER_TOKEN_TILE
    gather_cmp_blocks = gather_blocks * (PREFILL_ATTN_BLOCKS - 1)

    # This is the post-0.60 DSpark donor contract: ``swa_indices`` already
    # contains physical rows in a page-128 root, with -1 for invalid entries.
    with pl.spmd(gather_blocks, name_hint="physical_sparse_gather_ori", deps=[raw_ready_tid]) as gather_ori_tid:
        gather_schedule_block = pl.tile.get_block_idx()
        gather_token_block = gather_blocks - 1 - gather_schedule_block
        gather_local_t0 = gather_token_block * GATHER_TOKEN_TILE
        for gather_dt in pl.range(GATHER_TOKEN_TILE):
            gather_local_t = gather_local_t0 + gather_dt
            gather_t = gather_local_t
            if gather_t < active_rows:
                block_base = gather_local_t * PREFILL_SPARSE_PAD
                stage = pl.full([PREFILL_ATTN_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                for gather_ki in pl.range(PREFILL_ATTN_TILE):
                    gather_raw = pl.read(swa_indices, [gather_t, gather_ki])
                    if gather_raw >= 0:
                        source = pl.cast(gather_raw, pl.INDEX)
                        stage[gather_ki : gather_ki + 1, 0:HEAD_DIM] = ori_kv_flat[source : source + 1, 0:HEAD_DIM]
                source_view[block_base : block_base + PREFILL_ATTN_TILE, 0:HEAD_DIM] = stage

    with pl.spmd(gather_cmp_blocks, name_hint="physical_sparse_gather_cmp", deps=[compressed_ready_tid]) as gather_cmp_tid:
        gather_block = pl.tile.get_block_idx()
        gather_schedule_block = (gather_block // (PREFILL_ATTN_BLOCKS - 1))
        gather_token_block = gather_blocks - 1 - gather_schedule_block
        gather_sb = (gather_block - gather_schedule_block * (PREFILL_ATTN_BLOCKS - 1) + 1)
        gather_local_t0 = gather_token_block * GATHER_TOKEN_TILE
        gather_k0 = gather_sb * PREFILL_ATTN_TILE
        for gather_dt in pl.range(GATHER_TOKEN_TILE):
            gather_local_t = gather_local_t0 + gather_dt
            gather_t = gather_local_t
            if gather_t < active_rows:
                gather_block_valid = pl.read(valid_block_mask, [gather_t, gather_sb])
                if gather_block_valid > 0:
                    block_base = (gather_local_t * PREFILL_SPARSE_PAD + gather_k0)
                    stage = pl.full([PREFILL_ATTN_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                    for gather_ki in pl.range(PREFILL_ATTN_TILE):
                        gather_cmp_k = gather_k0 + gather_ki - WIN
                        if gather_cmp_k < IDX_TOPK:
                            logical_slot = pl.read(cmp_indices, [gather_t, gather_cmp_k])
                            if logical_slot >= 0:
                                logical_block = logical_slot // cmp_page_rows
                                if logical_block < CMP_MAX_BLOCKS:
                                    physical_block = pl.read(cmp_block_table, [logical_block])
                                    # Validity comes from the indices; physical page 0 is valid
                                    # in caller-owned MTP cache pools.
                                    if physical_block >= 0 and physical_block < cmp_block_num:
                                        page = pl.cast(physical_block, pl.INDEX)
                                        row = logical_slot % cmp_page_rows
                                        kv_row = cmp_kv[page:page + 1, row:row + 1, 0:1, 0:HEAD_DIM]
                                        stage[gather_ki:gather_ki + 1, :] = pl.reshape(kv_row, [1, HEAD_DIM])
                    source_view[block_base : block_base + PREFILL_ATTN_TILE, 0:HEAD_DIM] = stage

    with pl.spmd(
        (token_rows + BIAS_TOKEN_TILE - 1) // BIAS_TOKEN_TILE,
        name_hint="physical_sparse_build_bias",
        deps=[compressed_ready_tid],
    ) as bias_tid:
        bias_block = pl.tile.get_block_idx()
        bias_local_t0 = bias_block * BIAS_TOKEN_TILE
        bias_t0 = bias_local_t0
        if bias_t0 < active_rows:
            bias_rows = pl.min(BIAS_TOKEN_TILE, active_rows - bias_t0)
            raw_rows = pl.slice(swa_indices, [BIAS_TOKEN_TILE, WIN], [bias_t0, 0], valid_shape=[bias_rows, WIN])
            raw_index = pl.cast(raw_rows, target_type=pl.FP32)
            raw_flag = pl.minimum(pl.maximum(pl.add(raw_index, 1.0), 0.0), 1.0)
            raw_bias = pl.mul(pl.sub(raw_flag, 1.0), -FP32_NEG_INF)
            raw_bias_valid = pl.slice(raw_bias, [BIAS_TOKEN_TILE, WIN], [0, 0], valid_shape=[bias_rows, WIN])
            pl.assemble(sparse_bias, raw_bias_valid, [bias_local_t0, 0])
            if SPARSE_CMP_BIAS_COLS > 0:
                cmp_rows = pl.slice(
                    cmp_indices,
                    [BIAS_TOKEN_TILE, SPARSE_CMP_BIAS_COLS],
                    [bias_t0, 0],
                    valid_shape=[bias_rows, SPARSE_CMP_BIAS_COLS],
                )
                cmp_index = pl.cast(cmp_rows, target_type=pl.FP32)
                cmp_flag = pl.minimum(pl.maximum(pl.add(cmp_index, 1.0), 0.0), 1.0)
                cmp_bias = pl.mul(pl.sub(cmp_flag, 1.0), -FP32_NEG_INF)
                cmp_bias_valid = pl.slice(cmp_bias, [BIAS_TOKEN_TILE, SPARSE_CMP_BIAS_COLS], [0, 0], valid_shape=[bias_rows, SPARSE_CMP_BIAS_COLS])
                pl.assemble(sparse_bias, cmp_bias_valid, [bias_local_t0, WIN])
            if PREFILL_SPARSE_PAD > SPARSE_BIAS_COLS:
                pad_bias = pl.full(
                    [BIAS_TOKEN_TILE, PREFILL_SPARSE_PAD - SPARSE_BIAS_COLS],
                    dtype=pl.FP32,
                    value=FP32_NEG_INF,
                )
                pad_bias_valid = pl.slice(pad_bias, [BIAS_TOKEN_TILE, PREFILL_SPARSE_PAD - SPARSE_BIAS_COLS], [0, 0], valid_shape=[bias_rows, PREFILL_SPARSE_PAD - SPARSE_BIAS_COLS])
                pl.assemble(sparse_bias, pad_bias_valid, [bias_local_t0, SPARSE_BIAS_COLS])

    source_ready_tid = pl.system.task_dummy(deps=[gather_ori_tid, gather_cmp_tid, bias_tid])
    staged_source_rows = pl.tensor.dim(sparse_kv, 0)
    bias_rows = pl.tensor.dim(sparse_bias, 0)
    staged_kv = pl.reshape(sparse_kv, [staged_source_rows, HEAD_DIM])
    staged_bias = pl.reshape(sparse_bias, [bias_rows, PREFILL_SPARSE_PAD])
    return _staged_sparse_wave(
        q, staged_kv, staged_bias, valid_block_mask, attn_sink,
        active_rows, o_packed_heads, sparse_blk_mi, sparse_blk_li, sparse_blk_oi,
        rope_cos_il, rope_sin_signed, rope_swap_idx,
        source_ready_tid, merge_ready_tid, pl.cast(0, pl.INDEX),
    )


@pl.jit.inline(auto_scope=False)
def _physical_sparse_heads(
    q: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[PHYSICAL_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE_DYN, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[CMP_MAX_BLOCKS], pl.INT32],
    cmp_indices: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, IDX_TOPK], pl.INT32],
    valid_block_mask: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, VALID_BLOCK_MASK_COLS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    active_rows: pl.Scalar[pl.INDEX],
    freqs_cos: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, ROPE_DIM], pl.BF16],
    o_packed_heads: pl.Tensor[[PREFILL_ATTN_PACKED_ROWS_DYN, HEAD_DIM], pl.BF16],
    packed_init_tid: pl.Scalar[pl.TASK_ID],
    raw_ready_dep: pl.Scalar[pl.TASK_ID],
    compressed_ready_dep: pl.Scalar[pl.TASK_ID],
) -> tuple[pl.Tensor, pl.Scalar[pl.TASK_ID]]:
    """Run a segment-sized physical-gather wave with split source dependencies."""
    token_rows = pl.tensor.dim(q, 0)
    source_rows = token_rows * PREFILL_SPARSE_PAD
    stats_rows = token_rows * H
    rope_rows = ((token_rows + STAGED_SWA_ROPE_CS_T_TILE - 1) // STAGED_SWA_ROPE_CS_T_TILE) * STAGED_SWA_ROPE_CS_T_TILE
    completion = pl.array.create(1, pl.TASK_ID)
    completion[0] = packed_init_tid
    # Keep gathered KV on the caller's ring, separate from the FP32 head
    # partials. A combined allocation can pin wrap padding on the inner ring.
    sparse_kv = pl.create_tensor([source_rows, HEAD_DIM], dtype=pl.BF16)
    with pl.scope():
        sparse_bias = pl.create_tensor([token_rows, PREFILL_SPARSE_PAD], dtype=pl.FP32)
        sparse_blk_mi = pl.create_tensor([stats_rows, 1], dtype=pl.FP32)
        sparse_blk_li = pl.create_tensor([stats_rows, 1], dtype=pl.FP32)
        sparse_blk_oi = pl.create_tensor([stats_rows, HEAD_DIM], dtype=pl.FP32)
        rope_cos_il = pl.create_tensor([rope_rows, ROPE_DIM], dtype=pl.FP32, manual_dep=True)
        rope_sin_signed = pl.create_tensor([rope_rows, ROPE_DIM], dtype=pl.FP32, manual_dep=True)
        rope_swap_idx = pl.create_tensor([HEAD_TILE, ROPE_DIM], dtype=pl.INT32, manual_dep=True)
        rope_tid = _staged_attn_prepare_rope(
            freqs_cos,
            freqs_sin,
            rope_cos_il,
            rope_sin_signed,
            rope_swap_idx,
            active_rows,
            raw_ready_dep,
        )
        merge_ready_tid = pl.system.task_dummy(deps=[packed_init_tid, rope_tid])
        merge_tid = _physical_sparse_wave(
            q,
            ori_kv,
            swa_indices,
            cmp_kv,
            cmp_block_table,
            cmp_indices,
            valid_block_mask,
            attn_sink,
            active_rows,
            o_packed_heads,
            sparse_kv,
            sparse_bias,
            sparse_blk_mi,
            sparse_blk_li,
            sparse_blk_oi,
            rope_cos_il,
            rope_sin_signed,
            rope_swap_idx,
            raw_ready_dep,
            compressed_ready_dep,
            merge_ready_tid,
        )
        completion[0] = merge_tid

    return o_packed_heads, completion[0]


@pl.jit.inline(auto_scope=False)
def _staged_attn_o_proj(
    o_packed_heads: pl.Tensor[[PREFILL_ATTN_PACKED_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, D], pl.BF16],
    heads_dep: pl.Scalar[pl.TASK_ID],
) -> tuple[pl.Tensor, pl.Scalar[pl.TASK_ID]]:
    """Project aligned packed heads and write the logical output rows."""
    projection_rows = pl.tensor.dim(o_packed_heads, 0) // H
    output_rows = pl.tensor.dim(attn_out, 0)
    output_group_rows = O_GROUPS * projection_rows
    attn_out_view = pl.reshape(attn_out, [output_rows, D])
    o_packed = pl.reshape(o_packed_heads, [output_group_rows, O_GROUP_IN])
    o_r = pl.create_tensor([projection_rows, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_i8 = pl.create_tensor([projection_rows, O_GROUPS * O_LORA], dtype=pl.INT8)
    act_scale_dq = pl.create_tensor([O_GROUPS, projection_rows], dtype=pl.FP32)
    partials = pl.create_tensor([projection_rows, O_GROUPS * D], dtype=pl.INT32)
    proj_a_tids = pl.array.create(O_GROUPS * PA_NFRAGS, pl.TASK_ID)
    quant_tids = pl.array.create(O_GROUPS, pl.TASK_ID)
    proj_b_tids = pl.array.create(STAGED_SWA_PB_DSLABS * O_GROUPS, pl.TASK_ID)

    with pl.manual_scope():
        for g in pl.parallel(O_GROUPS):
            row_base_o = g * projection_rows
            out_col_g = g * O_LORA
            for nf in pl.range(PA_NFRAGS):
                n0 = nf * PROJ_A_MM_N_TILE
                with pl.spmd(
                    projection_rows // STAGED_SWA_PROJ_A_ROW_TILE,
                    name_hint="staged_swa_proj_a_mm",
                    deps=[heads_dep],
                ) as pa_tid:
                    pa_rb = pl.tile.get_block_idx()
                    pa_r0 = pa_rb * STAGED_SWA_PROJ_A_ROW_TILE
                    pa_src0 = row_base_o + pa_r0
                    acc_a = pl.create_tensor([1, STAGED_SWA_PROJ_A_ROW_TILE, PROJ_A_MM_N_TILE], dtype=pl.FP32)
                    for kb in pl.pipeline(0, O_GROUP_IN // A_K_TILE, stage=2):
                        k0 = kb * A_K_TILE
                        xa_k_chunk = o_packed[pa_src0 : pa_src0 + STAGED_SWA_PROJ_A_ROW_TILE, k0 : k0 + A_K_TILE]
                        wa_k_chunk = wo_a[g : g + 1, n0 : n0 + PROJ_A_MM_N_TILE, k0 : k0 + A_K_TILE]
                        acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True, init_cond=(kb == 0))
                    o_r = pl.assemble(o_r, acc_a, [pa_r0, out_col_g + n0])
                proj_a_tids[g * PA_NFRAGS + nf] = pa_tid

        for g in pl.parallel(O_GROUPS):
            col_g = g * O_LORA
            with pl.spmd(
                projection_rows // QUANT_TOKEN_TILE,
                name_hint="staged_swa_quant",
                deps=[proj_a_tids[g * PA_NFRAGS + j] for j in range(PA_NFRAGS)],
            ) as q_tid:
                qt = pl.tile.get_block_idx() * QUANT_TOKEN_TILE
                g_amax = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                for k1 in pl.range(0, O_LORA, QUANT_TILE):
                    oc = o_r[qt : qt + QUANT_TOKEN_TILE, col_g + k1 : col_g + k1 + QUANT_TILE]
                    oc_abs = pl.maximum(oc, pl.neg(oc))
                    oc_amax = pl.reshape(pl.row_max(oc_abs), [1, QUANT_TOKEN_TILE])
                    g_amax = pl.maximum(g_amax, oc_amax)
                g_scale_num = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
                g_sq_row = pl.div(g_scale_num, g_amax)
                act_scale_dq[g : g + 1, qt : qt + QUANT_TOKEN_TILE] = pl.recip(g_sq_row)
                g_sq_col = pl.reshape(g_sq_row, [QUANT_TOKEN_TILE, 1])
                for k1 in pl.range(0, O_LORA, QUANT_TILE):
                    oc = o_r[qt : qt + QUANT_TOKEN_TILE, col_g + k1 : col_g + k1 + QUANT_TILE]
                    oq_scaled = pl.row_expand_mul(oc, g_sq_col)
                    oq_i32 = pl.cast(oq_scaled, target_type=pl.INT32, mode="rint")
                    oq_half = pl.cast(oq_i32, target_type=pl.FP16, mode="round")
                    oq_i8 = pl.cast(oq_half, target_type=pl.INT8, mode="trunc")
                    o_r_i8[qt : qt + QUANT_TOKEN_TILE, col_g + k1 : col_g + k1 + QUANT_TILE] = oq_i8
            quant_tids[g] = q_tid

        for dc in pl.parallel(STAGED_SWA_PB_DSLABS):
            d0 = dc * STAGED_SWA_PROJ_B_D_TILE
            for g in pl.range(O_GROUPS):
                col_g = g * O_LORA
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="staged_swa_proj_b_mm", deps=[quant_tids[g]]) as pb_tid:
                    for nf in pl.range(STAGED_SWA_PROJ_B_D_TILE // PROJ_B_MM_N_TILE):
                        n0 = d0 + nf * PROJ_B_MM_N_TILE
                        for pb_rb in pl.range(projection_rows // STAGED_SWA_PROJ_B_ROW_TILE):
                            pb_r0 = pb_rb * STAGED_SWA_PROJ_B_ROW_TILE
                            acc_b = pl.create_tensor([STAGED_SWA_PROJ_B_ROW_TILE, PROJ_B_MM_N_TILE], dtype=pl.INT32)
                            for kb in pl.pipeline(0, O_LORA // B_K_TILE, stage=2):
                                k0 = col_g + kb * B_K_TILE
                                b_act = o_r_i8[pb_r0 : pb_r0 + STAGED_SWA_PROJ_B_ROW_TILE, k0 : k0 + B_K_TILE]
                                b_weight = wo_b[n0 : n0 + PROJ_B_MM_N_TILE, k0 : k0 + B_K_TILE]
                                acc_b = pl.matmul_acc(acc_b, b_act, b_weight, b_trans=True, init_cond=(kb == 0))
                            partials[
                                pb_r0 : pb_r0
                                + STAGED_SWA_PROJ_B_ROW_TILE,
                                g * D + n0 : g * D
                                + n0
                                + PROJ_B_MM_N_TILE,
                            ] = acc_b
                proj_b_tids[dc * O_GROUPS + g] = pb_tid

    act_t_blks = (output_rows + PROJ_B_ACT_TASK_T_TILE - 1) // PROJ_B_ACT_TASK_T_TILE
    with pl.spmd(
        (D // PROJ_B_ACT_N_TILE) * act_t_blks,
        name_hint="staged_swa_proj_b_act",
        deps=[proj_b_tids[i] for i in range(STAGED_SWA_PB_DSLABS * O_GROUPS)],
    ) as act_tid:
        act_idx = pl.tile.get_block_idx()
        nreg = act_idx // act_t_blks
        tblk = act_idx - nreg * act_t_blks
        ob_n0 = nreg * PROJ_B_ACT_N_TILE
        t0 = tblk * PROJ_B_ACT_TASK_T_TILE
        wb_scale = wo_b_scale[ob_n0 : ob_n0 + PROJ_B_ACT_N_TILE]
        wb_scale_chunk = pl.reshape(wb_scale, [1, PROJ_B_ACT_N_TILE])
        for b_tb in pl.range(t0, t0 + PROJ_B_ACT_TASK_T_TILE, PROJ_B_ACT_T_TILE):
            acc = pl.full([PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE], dtype=pl.FP32, value=0.0)
            for g in pl.range(O_GROUPS):
                p_g = partials[b_tb : b_tb + PROJ_B_ACT_T_TILE, g * D + ob_n0 : g * D + ob_n0 + PROJ_B_ACT_N_TILE]
                g_scale_row = act_scale_dq[g : g + 1, b_tb : b_tb + PROJ_B_ACT_T_TILE]
                g_scale = pl.reshape(g_scale_row, [PROJ_B_ACT_T_TILE, 1])
                p_g_f32 = pl.cast(p_g, target_type=pl.FP32, mode="none")
                acc = pl.add(acc, pl.row_expand_mul(p_g_f32, g_scale))
            out_t = pl.col_expand_mul(acc, wb_scale_chunk)
            out_bf16 = pl.cast(out_t, target_type=pl.BF16, mode="rint")
            if b_tb < output_rows:
                valid_rows = pl.min(PROJ_B_ACT_T_TILE, output_rows - b_tb)
                output_tile = pl.slice(
                    out_bf16, [PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE], [0, 0],
                    valid_shape=[valid_rows, PROJ_B_ACT_N_TILE],
                )
                pl.assemble(attn_out_view, output_tile, [b_tb, ob_n0])

    return attn_out, act_tid


@pl.jit.inline(auto_scope=False)
def _hca_segment_heads(
    q: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, H, HEAD_DIM], pl.BF16],
    full_kv: pl.Tensor[[HCA_FULL_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    predecessor_valid: pl.Scalar[pl.INDEX],
    cmp_work_kv: pl.Tensor[[HCA_CMP_PAD_ROWS, HEAD_DIM], pl.BF16],
    cmp_work_valid: pl.Tensor[[HCA_CMP_WORK_COUNT, HCA_ATTN_TILE], pl.FP32],
    position_ids: pl.Tensor[[PREFILL_ATTN_ROWS_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    active_rows: pl.Scalar[pl.INDEX],
    o_packed_heads: pl.Tensor[[PREFILL_ATTN_PACKED_ROWS_DYN, HEAD_DIM], pl.BF16],
    raw_kv: pl.Tensor[[HCA_RAW_ROWS_DYN, HEAD_DIM], pl.BF16],
    raw_valid: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, WIN], pl.FP32],
    stream_state_m: pl.Tensor[[HCA_HEAD_ROWS_DYN, 1], pl.FP32],
    stream_state_l: pl.Tensor[[HCA_HEAD_ROWS_DYN, 1], pl.FP32],
    stream_heads: pl.Tensor[[HCA_HEAD_ROWS_DYN, HEAD_DIM], pl.FP32],
    cmp_partial_m: pl.Tensor[[HCA_PARTIAL_ROWS_DYN, 8], pl.FP32],
    cmp_partial_l: pl.Tensor[[HCA_PARTIAL_ROWS_DYN, 8], pl.FP32],
    cmp_partial_o: pl.Tensor[[HCA_PARTIAL_ROWS_DYN, HEAD_DIM], pl.FP32],
    rope_cos_il: pl.Tensor[[PREFILL_ATTN_ROPE_ROWS_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[PREFILL_ATTN_ROPE_ROWS_DYN, ROPE_DIM], pl.FP32],
    wave_completion: pl.Array[1, pl.TASK_ID],
):
    """Gather raw history and compute one runtime-sized HCA segment."""
    token_rows = pl.tensor.dim(q, 0)
    query_head_rows = token_rows * H
    packed_rows = pl.tensor.dim(o_packed_heads, 0) // H
    full_cache_rows = pl.tensor.dim(full_kv, 0) * BLOCK_SIZE
    full_kv_flat = pl.reshape(full_kv, [full_cache_rows, HEAD_DIM])
    raw_source_rows = pl.tensor.dim(raw_kv, 0)
    raw_mask_rows = pl.tensor.dim(raw_valid, 0)
    raw_kv_view = pl.reshape(raw_kv, [raw_source_rows, HEAD_DIM])
    raw_valid_view = pl.reshape(raw_valid, [raw_mask_rows, WIN])

    with pl.spmd(
        (token_rows + HCA_GATHER_TOKEN_TILE - 1) // HCA_GATHER_TOKEN_TILE,
        name_hint="native_hca_raw_gather",
        deps=[wave_completion[0]],
    ) as raw_gather_tid:
        gather_block = pl.tile.get_block_idx()
        gather_local_t0 = gather_block * HCA_GATHER_TOKEN_TILE
        for gather_dt in pl.range(HCA_GATHER_TOKEN_TILE):
            gather_local_t = gather_local_t0 + gather_dt
            gather_t = gather_local_t
            if gather_local_t < token_rows:
                gather_dst = gather_local_t * WIN
                raw_stage = pl.full([WIN, HEAD_DIM], dtype=pl.BF16, value=0.0)
                valid_stage = pl.full([1, WIN], dtype=pl.FP32, value=0.0)
                if gather_t < active_rows:
                    gather_threshold = WIN - predecessor_valid
                    for gather_k in pl.range(WIN):
                        gather_candidate = gather_t + 1 + gather_k
                        if gather_candidate >= gather_threshold:
                            gather_row = gather_candidate
                            if gather_candidate < WIN:
                                gather_row = gather_candidate - gather_threshold
                            raw_stage[gather_k:gather_k + 1, 0:HEAD_DIM] = full_kv_flat[
                                gather_row:gather_row + 1, 0:HEAD_DIM
                            ]
                            pl.write(valid_stage, [0, gather_k], 1.0)
                raw_kv_view[gather_dst:gather_dst + WIN, 0:HEAD_DIM] = raw_stage
                raw_valid_view[gather_local_t:gather_local_t + 1, 0:WIN] = valid_stage

    q_flat = pl.reshape(q, [query_head_rows, HEAD_DIM])
    with pl.spmd(token_rows, name_hint="native_hca_raw_qk_pv", deps=[raw_gather_tid]) as raw_heads_tid:
        raw_local_t = pl.tile.get_block_idx()
        raw_t = raw_local_t
        if raw_t < active_rows:
            raw_src = raw_local_t * WIN
            raw_kv_tile = raw_kv_view[raw_src:raw_src + WIN, 0:HEAD_DIM]
            raw_valid_row = raw_valid_view[raw_local_t:raw_local_t + 1, 0:WIN]
            raw_valid_zero = pl.full([QK_M_TILE, WIN], dtype=pl.FP32, value=0.0)
            raw_valid_tile = pl.col_expand_add(raw_valid_zero, raw_valid_row)
            raw_bias = pl.mul(pl.sub(raw_valid_tile, 1.0), -FP32_NEG_INF)
            raw_token_row = raw_local_t * H
            for raw_hb in pl.pipeline(H // QK_M_TILE, stage=2):
                raw_h0 = raw_hb * QK_M_TILE
                raw_q_row = raw_t * H + raw_h0
                raw_q = q_flat[raw_q_row:raw_q_row + QK_M_TILE, 0:HEAD_DIM]
                raw_scores = pl.matmul(raw_q, raw_kv_tile, b_trans=True, out_dtype=pl.FP32)
                raw_scores = pl.add(pl.mul(raw_scores, SOFTMAX_SCALE), raw_bias)
                raw_m = pl.row_max(raw_scores)
                raw_exp = pl.exp(pl.row_expand_sub(raw_scores, raw_m))
                raw_exp = pl.mul(raw_exp, raw_valid_tile)
                raw_l = pl.row_sum(raw_exp)
                raw_o = pl.matmul(pl.cast(raw_exp, target_type=pl.BF16, mode="rint"), raw_kv_tile, out_dtype=pl.FP32)
                for raw_sub in pl.unroll(QK_M_TILE // HEAD_TILE):
                    raw_src_h0 = raw_sub * HEAD_TILE
                    raw_dst = raw_token_row + raw_h0 + raw_src_h0
                    stream_state_m[raw_dst:raw_dst + HEAD_TILE, 0:1] = raw_m[raw_src_h0:raw_src_h0 + HEAD_TILE, 0:1]
                    stream_state_l[raw_dst:raw_dst + HEAD_TILE, 0:1] = raw_l[raw_src_h0:raw_src_h0 + HEAD_TILE, 0:1]
                    stream_heads[raw_dst:raw_dst + HEAD_TILE, 0:HEAD_DIM] = raw_o[
                        raw_src_h0:raw_src_h0 + HEAD_TILE, 0:HEAD_DIM
                    ]

    with pl.spmd(token_rows, name_hint="native_hca_cmp_qk_pv", deps=[wave_completion[0]]) as cmp_qk_tid:
        # MTP has one compressed work tile and enough token parallelism.  Fold
        # both QK head blocks into one token task and reuse the compressed KV
        # tile, matching the proven staged-attention task granularity.
        qk_local_t = pl.tile.get_block_idx()
        qk_t = qk_local_t
        qk_token_base = (qk_local_t * (H // HEAD_TILE) * HCA_CMP_WORK_COUNT * HEAD_TILE)
        if qk_t < active_rows:
            qk_position_i32 = pl.read(position_ids, [qk_t])
            if qk_position_i32 >= 0:
                qk_visible_rows = (qk_position_i32 + 1) // HCA_COMPRESS_RATIO
                qk_visible_rows = pl.min(qk_visible_rows, pl.cast(HCA_MAX_COMPRESSED_ROWS, pl.INDEX))
                neutral_m = pl.full([HEAD_TILE, 8], dtype=pl.FP32, value=FP32_NEG_INF)
                neutral_l = pl.full([HEAD_TILE, 8], dtype=pl.FP32, value=0.0)
                neutral_o = pl.full([HEAD_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                for qk_work in pl.range((qk_visible_rows + HCA_ATTN_TILE - 1) // HCA_ATTN_TILE):
                    qk_work_row = qk_work * HCA_ATTN_TILE
                    for qk_hb in pl.range(H // QK_M_TILE):
                        for qk_sub in pl.unroll(QK_M_TILE // HEAD_TILE):
                            qk_h_idx = (qk_hb * (QK_M_TILE // HEAD_TILE) + qk_sub)
                            neutral_row = (
                                qk_token_base
                                + qk_h_idx
                                * HCA_CMP_WORK_COUNT
                                * HEAD_TILE
                                + qk_work * HEAD_TILE
                            )
                            cmp_partial_m[neutral_row:neutral_row + HEAD_TILE, 0:8] = neutral_m
                            cmp_partial_l[neutral_row:neutral_row + HEAD_TILE, 0:8] = neutral_l
                            cmp_partial_o[neutral_row:neutral_row + HEAD_TILE, 0:HEAD_DIM] = neutral_o

                    qk_valid_rows = pl.min(HCA_ATTN_TILE, qk_visible_rows - qk_work_row)
                    qk_kv = cmp_work_kv[qk_work_row:qk_work_row + HCA_ATTN_TILE, 0:HEAD_DIM]
                    qk_valid_row = cmp_work_valid[qk_work:qk_work + 1, :]
                    qk_valid_zero = pl.full([QK_M_TILE, HCA_ATTN_TILE], dtype=pl.FP32, value=0.0)
                    qk_valid = pl.col_expand_add(qk_valid_zero, qk_valid_row)
                    qk_bias = pl.mul(pl.sub(qk_valid, 1.0), -FP32_NEG_INF)
                    for qk_hb in pl.pipeline(H // QK_M_TILE, stage=2):
                        qk_h0 = qk_hb * QK_M_TILE
                        qk_q_row = qk_t * H + qk_h0
                        qk_q = q_flat[qk_q_row:qk_q_row + QK_M_TILE, 0:HEAD_DIM]
                        qk_scores = pl.matmul(qk_q, qk_kv, b_trans=True, out_dtype=pl.FP32)
                        qk_scores = pl.add(pl.mul(qk_scores, SOFTMAX_SCALE), qk_bias)
                        qk_scores = pl.set_validshape(qk_scores, QK_M_TILE, qk_valid_rows)
                        qk_scores = pl.fillpad(qk_scores, pad_value=pl.PadValue.min)
                        qk_m = pl.row_max(qk_scores)
                        qk_exp = pl.exp(pl.row_expand_sub(qk_scores, qk_m))
                        qk_exp = pl.mul(qk_exp, qk_valid)
                        qk_l = pl.row_sum(qk_exp)
                        qk_o = pl.matmul(pl.cast(qk_exp, target_type=pl.BF16, mode="rint"), qk_kv, out_dtype=pl.FP32)
                        for qk_sub in pl.unroll(QK_M_TILE // HEAD_TILE):
                            qk_src_h0 = qk_sub * HEAD_TILE
                            qk_h_idx = (qk_hb * (QK_M_TILE // HEAD_TILE) + qk_sub)
                            qk_row = (qk_token_base + qk_h_idx * HCA_CMP_WORK_COUNT * HEAD_TILE + qk_work * HEAD_TILE)
                            qk_m_column = qk_m[qk_src_h0:qk_src_h0 + HEAD_TILE, 0:1]
                            qk_l_column = qk_l[qk_src_h0:qk_src_h0 + HEAD_TILE, 0:1]
                            # Write whole scratch rows: a narrow column
                            # store ignores the padded row pitch.
                            qk_stat_zeros = pl.full([HEAD_TILE, 8], dtype=pl.FP32, value=0.0)
                            cmp_partial_m[
                                qk_row:qk_row + HEAD_TILE, 0:8
                            ] = pl.row_expand_add(qk_stat_zeros, qk_m_column)
                            cmp_partial_l[
                                qk_row:qk_row + HEAD_TILE, 0:8
                            ] = pl.row_expand_add(qk_stat_zeros, qk_l_column)
                            cmp_partial_o[qk_row:qk_row + HEAD_TILE, 0:HEAD_DIM] = qk_o[
                                qk_src_h0:qk_src_h0 + HEAD_TILE,
                                0:HEAD_DIM,
                            ]

    attn_sink_col = pl.reshape(attn_sink, [H, 1])
    with pl.spmd(token_rows, name_hint="native_hca_merge_rope_pack", deps=[raw_heads_tid, cmp_qk_tid]) as merge_tid:
        # At MTP's one-work-tile extent, token parallelism alone saturates the
        # device.  Fold the head tiles into each token block so position/RoPE
        # rows stay live and the merge does not fan out four tiny blocks.
        merge_local_t = pl.tile.get_block_idx()
        merge_t = merge_local_t
        if merge_t < active_rows:
            merge_token_base = (merge_local_t * (H // HEAD_TILE) * HCA_CMP_WORK_COUNT * HEAD_TILE)
            merge_position_i32 = pl.read(position_ids, [merge_t])
            merge_visible_rows = (merge_position_i32 + 1) // HCA_COMPRESS_RATIO
            merge_visible_rows = pl.min(merge_visible_rows, pl.cast(HCA_MAX_COMPRESSED_ROWS, pl.INDEX))
            merge_cos = rope_cos_il[merge_t:merge_t + 1, 0:ROPE_DIM]
            merge_sin = rope_sin_signed[merge_t:merge_t + 1, 0:ROPE_DIM]
            for merge_h_idx in pl.range(H // HEAD_TILE):
                merge_h0 = merge_h_idx * HEAD_TILE
                merge_stream_row = merge_local_t * H + merge_h0
                merge_m = stream_state_m[merge_stream_row:merge_stream_row + HEAD_TILE, 0:1]
                merge_l = stream_state_l[merge_stream_row:merge_stream_row + HEAD_TILE, 0:1]
                merge_o = stream_heads[merge_stream_row:merge_stream_row + HEAD_TILE, 0:HEAD_DIM]
                merge_partial_base = (merge_token_base + merge_h_idx * HCA_CMP_WORK_COUNT * HEAD_TILE)
                for merge_work in pl.range((merge_visible_rows + HCA_ATTN_TILE - 1) // HCA_ATTN_TILE):
                    merge_row = (merge_partial_base + merge_work * HEAD_TILE)
                    merge_cmp_m_padded = cmp_partial_m[merge_row:merge_row + HEAD_TILE, 0:8]
                    merge_cmp_l_padded = cmp_partial_l[merge_row:merge_row + HEAD_TILE, 0:8]
                    # Read col0 as a dense [HEAD_TILE, 1] vector: a column
                    # view of a padded tile keeps the row pitch.
                    merge_cmp_m_transposed = pl.transpose(merge_cmp_m_padded, axis1=0, axis2=1)
                    merge_cmp_l_transposed = pl.transpose(merge_cmp_l_padded, axis1=0, axis2=1)
                    merge_cmp_m = pl.reshape(merge_cmp_m_transposed[0:1, :], [HEAD_TILE, 1])
                    merge_cmp_l = pl.reshape(merge_cmp_l_transposed[0:1, :], [HEAD_TILE, 1])
                    merge_cmp_o = cmp_partial_o[merge_row:merge_row + HEAD_TILE, 0:HEAD_DIM]
                    merge_m_new = pl.maximum(merge_m, merge_cmp_m)
                    merge_alpha = pl.exp(pl.sub(merge_m, merge_m_new))
                    merge_beta = pl.exp(pl.sub(merge_cmp_m, merge_m_new))
                    merge_l = pl.add(pl.mul(merge_alpha, merge_l), pl.mul(merge_beta, merge_cmp_l))
                    merge_o = pl.add(
                        pl.row_expand_mul(merge_o, merge_alpha),
                        pl.row_expand_mul(merge_cmp_o, merge_beta),
                    )
                    merge_m = merge_m_new

                merge_sink = attn_sink_col[merge_h0:merge_h0 + HEAD_TILE, 0:1]
                merge_sink_tile = pl.add(pl.sub(merge_m, merge_m), merge_sink)
                merge_denom = pl.add(merge_l, pl.exp(pl.sub(merge_sink_tile, merge_m)))
                merge_full = pl.row_expand_div(merge_o, merge_denom)
                merge_nope_bf16 = pl.cast(merge_full[:, 0:NOPE_DIM], target_type=pl.BF16, mode="rint")
                merge_rope = merge_full[:, NOPE_DIM:HEAD_DIM]
                merge_even = pl.gather(merge_rope, mask_pattern=pl.tile.MaskPattern.P0101)
                merge_odd = pl.gather(merge_rope, mask_pattern=pl.tile.MaskPattern.P1010)
                merge_swapped = pl.full([HEAD_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
                merge_swapped = pl.tensor.scatter(merge_odd, mask_pattern=pl.tile.MaskPattern.P0101, dst=merge_swapped)
                merge_swapped = pl.tensor.scatter(
                    merge_even,
                    mask_pattern=pl.tile.MaskPattern.P1010,
                    dst=merge_swapped,
                )
                merge_rot = pl.add(
                    pl.col_expand_mul(merge_rope, merge_cos),
                    pl.col_expand_mul(merge_swapped, merge_sin),
                )
                merge_rope_bf16 = pl.cast(merge_rot, target_type=pl.BF16, mode="rint")
                merge_group0 = merge_h0 // HEADS_PER_GROUP
                for merge_subgroup in pl.unroll(HEAD_TILE // HEADS_PER_GROUP):
                    merge_src_h0 = merge_subgroup * HEADS_PER_GROUP
                    merge_pack_row = ((merge_group0 + merge_subgroup) * packed_rows + merge_t)
                    merge_dst_head = merge_pack_row * HEADS_PER_GROUP
                    pl.assemble(
                        o_packed_heads,
                        merge_nope_bf16[merge_src_h0: merge_src_h0 + HEADS_PER_GROUP, 0:NOPE_DIM],
                        [merge_dst_head, 0],
                    )
                    pl.assemble(
                        o_packed_heads,
                        merge_rope_bf16[merge_src_h0: merge_src_h0 + HEADS_PER_GROUP, 0:ROPE_DIM],
                        [merge_dst_head, NOPE_DIM],
                    )

    wave_completion[0] = merge_tid


@pl.jit.inline(auto_scope=False)
def _hca_heads(
    q: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, H, HEAD_DIM], pl.BF16],
    full_kv: pl.Tensor[[HCA_FULL_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    predecessor_valid: pl.Scalar[pl.INDEX],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE_DYN, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[CMP_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[PREFILL_ATTN_ROWS_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    active_rows: pl.Scalar[pl.INDEX],
    freqs_cos: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, ROPE_DIM], pl.BF16],
    o_packed_heads: pl.Tensor[[PREFILL_ATTN_PACKED_ROWS_DYN, HEAD_DIM], pl.BF16],
    packed_init_tid: pl.Scalar[pl.TASK_ID],
    cache_ready_dep: pl.Scalar[pl.TASK_ID],
) -> tuple[pl.Tensor, pl.Scalar[pl.TASK_ID]]:
    """Gather compressed history and prepare one runtime-sized HCA segment."""
    token_rows = pl.tensor.dim(q, 0)
    raw_rows = token_rows * WIN
    head_rows = token_rows * H
    partial_rows = head_rows * HCA_CMP_WORK_COUNT
    rope_rows = ((token_rows + STAGED_SWA_ROPE_CS_T_TILE - 1) // STAGED_SWA_ROPE_CS_T_TILE) * STAGED_SWA_ROPE_CS_T_TILE
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    cmp_page_rows = pl.tensor.dim(cmp_kv, 1)
    cmp_cache_rows = cmp_block_num * cmp_page_rows
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_cache_rows, HEAD_DIM])
    completion = pl.array.create(1, pl.TASK_ID)
    completion[0] = packed_init_tid

    with pl.manual_scope():
        tile_cmp_visible_rows = pl.create_tensor([1], dtype=pl.INT32)
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="native_hca_cmp_plan",
            deps=[packed_init_tid, cache_ready_dep],
        ) as cmp_plan_tid:
            plan_visible_rows = pl.cast(0, pl.INDEX)
            if active_rows > 0:
                plan_last_t = active_rows - 1
                plan_max_pos = pl.read(position_ids, [plan_last_t])
                plan_visible_rows = (plan_max_pos + 1) // HCA_COMPRESS_RATIO
            plan_visible_rows = pl.max(plan_visible_rows, 0)
            plan_visible_rows = pl.min(plan_visible_rows, pl.cast(HCA_MAX_COMPRESSED_ROWS, pl.INDEX))
            pl.write(tile_cmp_visible_rows, [0], pl.cast(plan_visible_rows, pl.INT32))

        cmp_work_kv = pl.create_tensor([HCA_CMP_PAD_ROWS, HEAD_DIM], dtype=pl.BF16, manual_dep=True)
        cmp_work_valid = pl.create_tensor([HCA_CMP_WORK_COUNT, HCA_ATTN_TILE], dtype=pl.FP32, manual_dep=True)
        with pl.spmd(HCA_CMP_WORK_COUNT, name_hint="native_hca_cmp_gather", deps=[cmp_plan_tid]) as cmp_gather_tid:
            gather_work = pl.tile.get_block_idx()
            gather_dst0 = gather_work * HCA_ATTN_TILE
            gather_visible = pl.cast(pl.read(tile_cmp_visible_rows, [0]), pl.INDEX)
            gather_rows = pl.max(0, pl.min(HCA_ATTN_TILE, gather_visible - gather_dst0))
            gather_stage = pl.full([HCA_ATTN_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
            gather_valid = pl.full([1, HCA_ATTN_TILE], dtype=pl.FP32, value=0.0)
            for gather_local in pl.range(gather_rows):
                gather_slot = gather_dst0 + gather_local
                gather_table_col = gather_slot // cmp_page_rows
                if gather_table_col < CMP_MAX_BLOCKS:
                    gather_block_i32 = pl.read(cmp_block_table, [gather_table_col])
                    if gather_block_i32 >= 0:
                        if gather_block_i32 < cmp_block_num:
                            gather_block = pl.cast(gather_block_i32, pl.INDEX)
                            gather_src = gather_block * cmp_page_rows + gather_slot % cmp_page_rows
                            gather_stage[gather_local:gather_local + 1, :] = cmp_kv_flat[gather_src:gather_src + 1, :]
                            pl.write(gather_valid, [0, gather_local], 1.0)
            cmp_work_kv[gather_dst0:gather_dst0 + HCA_ATTN_TILE, :] = gather_stage
            cmp_work_valid[gather_work:gather_work + 1, :] = gather_valid

        rope_cos_il = pl.create_tensor([rope_rows, ROPE_DIM], dtype=pl.FP32, manual_dep=True)
        rope_sin_signed = pl.create_tensor([rope_rows, ROPE_DIM], dtype=pl.FP32, manual_dep=True)
        rope_swap_idx = pl.create_tensor([HEAD_TILE, ROPE_DIM], dtype=pl.INT32, manual_dep=True)
        rope_cs_tid = _staged_attn_prepare_rope(
            freqs_cos,
            freqs_sin,
            rope_cos_il,
            rope_sin_signed,
            rope_swap_idx,
            active_rows,
            cache_ready_dep,
        )

        raw_kv = pl.create_tensor([raw_rows, HEAD_DIM], dtype=pl.BF16)
        raw_valid = pl.create_tensor([token_rows, WIN], dtype=pl.FP32)
        stream_state_m = pl.create_tensor([head_rows, 1], dtype=pl.FP32)
        stream_state_l = pl.create_tensor([head_rows, 1], dtype=pl.FP32)
        stream_heads = pl.create_tensor([head_rows, HEAD_DIM], dtype=pl.FP32)
        cmp_partial_m = pl.create_tensor([partial_rows, 8], dtype=pl.FP32)
        cmp_partial_l = pl.create_tensor([partial_rows, 8], dtype=pl.FP32)
        cmp_partial_o = pl.create_tensor([partial_rows, HEAD_DIM], dtype=pl.FP32)
        wave_completion = pl.array.create(1, pl.TASK_ID)
        wave_completion[0] = pl.system.task_dummy(deps=[cmp_gather_tid, rope_cs_tid])
        _hca_segment_heads(
            q,
            full_kv,
            predecessor_valid,
            cmp_work_kv,
            cmp_work_valid,
            position_ids,
            attn_sink,
            active_rows,
            o_packed_heads,
            raw_kv,
            raw_valid,
            stream_state_m,
            stream_state_l,
            stream_heads,
            cmp_partial_m,
            cmp_partial_l,
            cmp_partial_o,
            rope_cos_il,
            rope_sin_signed,
            wave_completion,
        )
        completion[0] = wave_completion[0]

    return o_packed_heads, completion[0]


@pl.jit.inline(auto_scope=False)
def hca_attn(
    q: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, H, HEAD_DIM], pl.BF16],
    full_kv: pl.Tensor[[HCA_FULL_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    predecessor_valid: pl.Scalar[pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE_DYN, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[CMP_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[PREFILL_ATTN_ROWS_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, D], pl.BF16],
    active_rows: pl.Scalar[pl.INT32],
    cache_ready_dep: pl.Scalar[pl.TASK_ID],
    o_proj_weight_dep: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Run variable-length C128 attention over physical cache roots."""
    token_rows = pl.tensor.dim(q, 0)
    projection_rows = ((token_rows + STAGED_SWA_PROJ_A_ROW_TILE - 1) // STAGED_SWA_PROJ_A_ROW_TILE) * STAGED_SWA_PROJ_A_ROW_TILE
    packed_head_rows = projection_rows * H
    active = pl.cast(active_rows, pl.INDEX)
    active = pl.min(token_rows, pl.max(active, 0))
    predecessor_rows = pl.cast(predecessor_valid, pl.INDEX)
    predecessor_rows = pl.min(WIN, pl.max(predecessor_rows, 0))
    completion = pl.array.create(1, pl.TASK_ID)
    completion[0] = cache_ready_dep
    with pl.scope():
        o_packed_heads = pl.create_tensor([packed_head_rows, HEAD_DIM], dtype=pl.BF16, manual_dep=True)
        with pl.spmd(
            packed_head_rows // STAGED_SWA_QUERY_TILE,
            name_hint="native_hca_packed_init",
            deps=[cache_ready_dep],
        ) as packed_init_tid:
            packed_row = (pl.tile.get_block_idx() * STAGED_SWA_QUERY_TILE)
            o_packed_heads[
                packed_row:packed_row + STAGED_SWA_QUERY_TILE,
                0:HEAD_DIM,
            ] = pl.full([STAGED_SWA_QUERY_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)

        o_packed_heads, heads_dep = _hca_heads(
            q,
            full_kv,
            predecessor_rows,
            cmp_kv,
            cmp_block_table,
            position_ids,
            attn_sink,
            active,
            freqs_cos,
            freqs_sin,
            o_packed_heads,
            packed_init_tid,
            cache_ready_dep,
        )
        o_proj_dep = pl.system.task_dummy(deps=[heads_dep, o_proj_weight_dep])
        _attn_out, act_tid = _staged_attn_o_proj(o_packed_heads, wo_a, wo_b, wo_b_scale, attn_out, o_proj_dep)
        completion[0] = act_tid

    return completion[0]


@pl.jit.inline(auto_scope=False)
def physical_sparse_attn(
    q: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[PHYSICAL_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE_DYN, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[CMP_MAX_BLOCKS], pl.INT32],
    cmp_indices: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, IDX_TOPK], pl.INT32],
    valid_block_mask: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, VALID_BLOCK_MASK_COLS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, D], pl.BF16],
    active_rows: pl.Scalar[pl.INT32],
    raw_ready_dep: pl.Scalar[pl.TASK_ID],
    compressed_ready_dep: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Run variable-length TopK attention from Recipes physical roots."""
    token_rows = pl.tensor.dim(q, 0)
    projection_rows = ((token_rows + STAGED_SWA_PROJ_A_ROW_TILE - 1) // STAGED_SWA_PROJ_A_ROW_TILE) * STAGED_SWA_PROJ_A_ROW_TILE
    packed_head_rows = projection_rows * H
    active = pl.cast(active_rows, pl.INDEX)
    active = pl.min(token_rows, pl.max(active, 0))
    completion = pl.array.create(1, pl.TASK_ID)
    completion[0] = compressed_ready_dep
    with pl.scope():
        o_packed_heads = pl.create_tensor([packed_head_rows, HEAD_DIM], dtype=pl.BF16, manual_dep=True)
        with pl.spmd(
            packed_head_rows // STAGED_SWA_QUERY_TILE,
            name_hint="physical_sparse_packed_init",
            deps=[raw_ready_dep],
        ) as packed_init_tid:
            packed_row = (pl.tile.get_block_idx() * STAGED_SWA_QUERY_TILE)
            o_packed_heads[
                packed_row : packed_row + STAGED_SWA_QUERY_TILE,
                0:HEAD_DIM,
            ] = pl.full([STAGED_SWA_QUERY_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)

        o_packed_heads, heads_tid = _physical_sparse_heads(
            q,
            ori_kv,
            swa_indices,
            cmp_kv,
            cmp_block_table,
            cmp_indices,
            valid_block_mask,
            attn_sink,
            active,
            freqs_cos,
            freqs_sin,
            o_packed_heads,
            packed_init_tid,
            raw_ready_dep,
            compressed_ready_dep,
        )
        _attn_out, projection_tid = _staged_attn_o_proj(o_packed_heads, wo_a, wo_b, wo_b_scale, attn_out, heads_tid)
        completion[0] = projection_tid

    # PR #1073 publishes the final output dependency after leaving the
    # tile-local scope.  Keep that exact post-0.60 lifetime pattern here.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="physical_sparse_publish", deps=[completion[0]]) as publish_tid:
        completion_anchor = pl.read(attn_out, [0, 0])
        pl.write(attn_out, [0, 0], completion_anchor)
    return publish_tid


@pl.jit.inline(auto_scope=False)
def staged_sparse_attn(
    q: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, H, HEAD_DIM], pl.BF16],
    sparse_kv: pl.Tensor[[PREFILL_ATTN_SOURCE_ROWS_DYN, HEAD_DIM], pl.BF16],
    sparse_bias: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, PREFILL_SPARSE_PAD], pl.FP32],
    valid_block_mask: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, VALID_BLOCK_MASK_COLS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, D], pl.BF16],
    active_rows: pl.Scalar[pl.INT32],
    prior_dep: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Consume one variable-length staged sparse segment and expose its final projection TaskId.

    The caller owns sparse source staging and may pass this completion TaskId as
    the next segment's ``prior_dep``.  Inactive tail rows are projected from a
    zero-initialized packed-head buffer and therefore remain zero.
    """
    token_rows = pl.tensor.dim(q, 0)
    projection_rows = ((token_rows + STAGED_SWA_PROJ_A_ROW_TILE - 1) // STAGED_SWA_PROJ_A_ROW_TILE) * STAGED_SWA_PROJ_A_ROW_TILE
    packed_head_rows = projection_rows * H
    active = pl.cast(active_rows, pl.INDEX)
    active = pl.min(token_rows, pl.max(active, 0))
    completion = pl.array.create(1, pl.TASK_ID)
    completion[0] = prior_dep
    with pl.scope():
        o_packed_heads = pl.create_tensor([packed_head_rows, HEAD_DIM], dtype=pl.BF16, manual_dep=True)
        with pl.spmd(
            packed_head_rows // STAGED_SWA_QUERY_TILE,
            name_hint="staged_swa_packed_init",
            deps=[prior_dep],
        ) as packed_init_tid:
            packed_row = (pl.tile.get_block_idx() * STAGED_SWA_QUERY_TILE)
            packed_zero = pl.full([STAGED_SWA_QUERY_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
            o_packed_heads[packed_row : packed_row + STAGED_SWA_QUERY_TILE, 0:HEAD_DIM] = packed_zero

        o_packed_heads, heads_dep = _staged_swa_heads(
            q,
            sparse_kv,
            sparse_bias,
            valid_block_mask,
            attn_sink,
            active,
            freqs_cos,
            freqs_sin,
            o_packed_heads,
            packed_init_tid,
            prior_dep,
        )
        _attn_out, act_tid = _staged_attn_o_proj(o_packed_heads, wo_a, wo_b, wo_b_scale, attn_out, heads_dep)
        completion[0] = act_tid

    # ``attn_out`` is caller-owned GM storage written by the projection tasks.
    # Returning its scoped SSA version makes the generated runtime C++ refer to
    # an inner-scope alias after that scope has ended.  Only the completion
    # token needs to cross this boundary.
    return completion[0]


@pl.jit.inline(auto_scope=False)
def prefill_staged_attention(
    q: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, H, HEAD_DIM], pl.BF16],
    sparse_kv: pl.Tensor[[PREFILL_ATTN_SOURCE_ROWS_DYN, HEAD_DIM], pl.BF16],
    sparse_bias: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, PREFILL_SPARSE_PAD], pl.FP32],
    valid_block_mask: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, VALID_BLOCK_MASK_COLS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    residual: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, HC_MULT, D], pl.FP32],
    post: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, HC_COMB], pl.FP32],
    x_out: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, HC_MULT, D], pl.FP32],
    active_rows: pl.Scalar[pl.INT32],
    prior_dep: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Shared staged attention, output projection, and HC residual update."""
    rows = pl.tensor.dim(q, 0)
    active = pl.min(rows, pl.max(pl.cast(active_rows, pl.INDEX), 0))
    compute_rows = pl.max(active, 1)
    source_rows = compute_rows * PREFILL_SPARSE_PAD
    q_active = pl.slice(q, [compute_rows, H, HEAD_DIM], [0, 0, 0])
    kv_active = pl.slice(sparse_kv, [source_rows, HEAD_DIM], [0, 0])
    bias_active = pl.slice(sparse_bias, [compute_rows, PREFILL_SPARSE_PAD], [0, 0])
    mask_active = pl.slice(valid_block_mask, [compute_rows, VALID_BLOCK_MASK_COLS], [0, 0])
    cos_active = pl.slice(freqs_cos, [compute_rows, ROPE_DIM], [0, 0])
    sin_active = pl.slice(freqs_sin, [compute_rows, ROPE_DIM], [0, 0])
    attn_out = pl.create_tensor([rows, D], dtype=pl.BF16)
    attn_active = pl.slice(attn_out, [compute_rows, D], [0, 0])
    attention_done = staged_sparse_attn(
        q_active, kv_active, bias_active, mask_active, attn_sink,
        cos_active, sin_active, wo_a, wo_b, wo_b_scale,
        attn_active, active_rows, prior_dep,
    )
    residual_view = pl.reshape(residual, [rows, HC_MULT, D])
    post_view = pl.reshape(post, [rows, HC_MULT])
    comb_view = pl.reshape(comb, [rows, HC_COMB])
    output_view = pl.reshape(x_out, [rows, HC_MULT, D])
    hc_post_prefill(attn_out, residual_view, post_view, comb_view, output_view, active_rows)
    return attention_done


@pl.jit.inline(auto_scope=False)
def prefill_physical_attention(
    q: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[PHYSICAL_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE_DYN, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[CMP_MAX_BLOCKS], pl.INT32],
    cmp_indices: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, IDX_TOPK], pl.INT32],
    valid_block_mask: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, VALID_BLOCK_MASK_COLS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    residual: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, HC_MULT, D], pl.FP32],
    post: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, HC_COMB], pl.FP32],
    x_out: pl.Tensor[[PREFILL_ATTN_ROWS_DYN, HC_MULT, D], pl.FP32],
    active_rows: pl.Scalar[pl.INT32],
    raw_ready_dep: pl.Scalar[pl.TASK_ID],
    compressed_ready_dep: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Shared physical-cache attention, output projection, and HC residual update."""
    rows = pl.tensor.dim(q, 0)
    active = pl.min(rows, pl.max(pl.cast(active_rows, pl.INDEX), 0))
    compute_rows = pl.max(active, 1)
    q_active = pl.slice(q, [compute_rows, H, HEAD_DIM], [0, 0, 0])
    raw_indices = pl.slice(swa_indices, [compute_rows, WIN], [0, 0])
    compressed_indices = pl.slice(cmp_indices, [compute_rows, IDX_TOPK], [0, 0])
    mask_active = pl.slice(valid_block_mask, [compute_rows, VALID_BLOCK_MASK_COLS], [0, 0])
    cos_active = pl.slice(freqs_cos, [compute_rows, ROPE_DIM], [0, 0])
    sin_active = pl.slice(freqs_sin, [compute_rows, ROPE_DIM], [0, 0])
    attn_out = pl.create_tensor([rows, D], dtype=pl.BF16)
    attn_active = pl.slice(attn_out, [compute_rows, D], [0, 0])
    attention_done = physical_sparse_attn(
        q_active, ori_kv, raw_indices, cmp_kv, cmp_block_table, compressed_indices, mask_active, attn_sink,
        cos_active, sin_active, wo_a, wo_b, wo_b_scale,
        attn_active, active_rows, raw_ready_dep, compressed_ready_dep,
    )
    residual_view = pl.reshape(residual, [rows, HC_MULT, D])
    post_view = pl.reshape(post, [rows, HC_MULT])
    comb_view = pl.reshape(comb, [rows, HC_COMB])
    output_view = pl.reshape(x_out, [rows, HC_MULT, D])
    hc_post_prefill(attn_out, residual_view, post_view, comb_view, output_view, active_rows)
    return attention_done


@pl.jit.inline
def sparse_attn(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE_DYN, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[CMP_MAX_BLOCKS], pl.INT32],
    cmp_storage_block_size: pl.Scalar[pl.INT32],
    cmp_indices: pl.Tensor[[T, IDX_TOPK], pl.INT32],
    valid_block_mask: pl.Tensor[[T, VALID_BLOCK_MASK_COLS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    """Run paged sparse attention with the same cache reader used by CP."""
    # Keep the public scalar argument for existing callers. Physical addressing
    # derives the page size from the cache tensor, as it does in the CP path.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_sparse_sources_ready") as sources_ready:
        _q_ready = pl.read(q, [0, 0, 0])
        _raw_ready = pl.read(ori_kv, [0, 0, 0, 0])
        _compressed_ready = pl.read(cmp_kv, [0, 0, 0, 0])
        _raw_indices_ready = pl.read(swa_indices, [0, 0])
        _compressed_indices_ready = pl.read(cmp_indices, [0, 0])
        _mask_ready = pl.read(valid_block_mask, [0, 0])
    physical_sparse_attn(
        q, ori_kv, swa_indices, cmp_kv, cmp_block_table,
        cmp_indices, valid_block_mask, attn_sink,
        freqs_cos, freqs_sin, wo_a, wo_b, wo_b_scale,
        attn_out, num_tokens, sources_ready, sources_ready,
    )
    return attn_out


@pl.jit
def prefill_sparse_attn_test(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE_DYN, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[CMP_MAX_BLOCKS], pl.INT32],
    cmp_storage_block_size: pl.Scalar[pl.INT32],
    cmp_indices: pl.Tensor[[T, IDX_TOPK], pl.INT32],
    valid_block_mask: pl.Tensor[[T, VALID_BLOCK_MASK_COLS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    ori_kv.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    cmp_kv.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    cmp_kv.bind_dynamic(1, CMP_STORAGE_BLOCK_SIZE_DYN)
    return sparse_attn(
        q, ori_kv, swa_indices,
        cmp_kv, cmp_block_table, cmp_storage_block_size, cmp_indices,
        valid_block_mask, attn_sink, num_tokens,
        freqs_cos, freqs_sin,
        wo_a, wo_b, wo_b_scale,
        attn_out,
    )

def golden_prefill_sparse_attn(tensors):
    """Self-contained torch reference for the cache-first sparse-attn entry."""
    import torch

    num_tokens = int(tensors["num_tokens"])
    q = tensors["q"].float()
    ori_kv = tensors["ori_kv"].float()
    cmp_kv = tensors["cmp_kv"].float()
    cmp_block_table = tensors["cmp_block_table"]
    cmp_storage_block_size = int(tensors.get("cmp_storage_block_size", BLOCK_SIZE))
    swa_indices = tensors["swa_indices"]
    cmp_indices = tensors["cmp_indices"]
    attn_sink = tensors["attn_sink"].float()
    cos = tensors["freqs_cos"].float()
    sin = tensors["freqs_sin"].float()
    wo_a = tensors["wo_a"].float()
    wo_b_i8 = tensors["wo_b"]
    wo_b_scale = tensors["wo_b_scale"].float()

    o = torch.zeros(T, H, HEAD_DIM)
    for t in range(num_tokens):
        gathered = []
        for row_i in swa_indices[t].tolist():
            row = int(row_i)
            if row < 0:
                continue
            gathered.append(ori_kv.reshape(-1, HEAD_DIM)[row])
        for raw_i in cmp_indices[t].tolist():
            cmp_slot = int(raw_i)
            if cmp_slot < 0 or cmp_slot >= CMP_MAX_BLOCKS * cmp_storage_block_size:
                continue
            block_id = int(cmp_block_table[cmp_slot // cmp_storage_block_size].item())
            if block_id >= 0:
                gathered.append(cmp_kv[block_id, cmp_slot % cmp_storage_block_size, 0])

        if not gathered:
            continue
        kv_rows = torch.stack(gathered, dim=0)

        mi = None
        li = None
        oi = None
        for tile_start in range(0, kv_rows.shape[0], PREFILL_ATTN_TILE):
            kv_tile = kv_rows[tile_start : tile_start + PREFILL_ATTN_TILE]
            scores = (q[t] @ kv_tile.T) * SOFTMAX_SCALE
            cur_mi = scores.max(dim=-1, keepdim=True).values
            exp_scores = torch.exp(scores - cur_mi)
            cur_li = exp_scores.sum(dim=-1, keepdim=True)
            exp_scores_bf16 = exp_scores.to(torch.bfloat16)
            cur_oi = exp_scores_bf16.float() @ kv_tile.to(torch.bfloat16).float()
            if mi is None:
                mi = cur_mi
                li = cur_li
                oi = cur_oi
            else:
                mi_new = torch.maximum(mi, cur_mi)
                alpha = torch.exp(mi - mi_new)
                beta = torch.exp(cur_mi - mi_new)
                li = alpha * li + beta * cur_li
                oi = oi * alpha + cur_oi * beta
                mi = mi_new

        if mi is not None:
            denom = li + torch.exp(attn_sink.unsqueeze(-1) - mi)
            o[t] = oi / denom

    rope_pair = o[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    rope_even = rope_pair[..., 0]
    rope_odd = rope_pair[..., 1]
    cos_half = cos[:, :ROPE_HALF].unsqueeze(1)
    sin_half = sin[:, :ROPE_HALF].unsqueeze(1)
    inv_even = (rope_even * cos_half + rope_odd * sin_half).to(torch.bfloat16).float()
    inv_odd = (rope_odd * cos_half - rope_even * sin_half).to(torch.bfloat16).float()
    o_rope = torch.stack([inv_even, inv_odd], dim=-1).flatten(-2)
    o = torch.cat([o[..., :NOPE_DIM], o_rope], dim=-1).to(torch.bfloat16)

    o_model = o.float().view(T, O_GROUPS, O_GROUP_IN)
    o_r = torch.einsum("tgd,grd->tgr", o_model, wo_a)   # [T, G, O_LORA]
    # PER-GROUP INT8 activation quant (one amax per O_LORA group, not per full row) --
    # mirrors the decoupled proj_a[g]->quant[g]->proj_b[g] kernel pipeline. Each group's
    # INT32 partial is dequantized by its OWN per-row act scale (the per-group scale cannot
    # factor out of the K-sum), then the per-channel weight scale is applied.
    o_r_g = o_r.reshape(T, O_GROUPS, O_LORA)
    amax_g = o_r_g.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)   # [T, G, 1]
    scale_q_g = INT8_SCALE_MAX / amax_g
    o_r_i8_g = torch.round(o_r_g * scale_q_g).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dq_g = 1.0 / scale_q_g                                              # [T, G, 1]
    wo_b_g = wo_b_i8.reshape(D, O_GROUPS, O_LORA)
    out = torch.zeros(T, D, dtype=torch.float32)
    for g in range(O_GROUPS):
        p_g = o_r_i8_g[:, g].to(torch.int32) @ wo_b_g[:, g].to(torch.int32).T   # [T, D]
        out = out + p_g.float() * scale_dq_g[:, g]                             # per-row group scale
    out = out * wo_b_scale.unsqueeze(0)                                        # per-channel weight scale
    tensors["attn_out"][:] = out.to(torch.bfloat16)

def get_prefill_cmp_valid(compress_ratio: int) -> int:
    """Map standalone ratio modes to visible compressed-cache length."""
    if compress_ratio == 0:
        return 0
    if compress_ratio in (4, 128):
        storage_block_size = BLOCK_SIZE // compress_ratio
        return min(IDX_TOPK, S // compress_ratio, CMP_MAX_BLOCKS * storage_block_size)
    raise ValueError(f"Unsupported compress_ratio={compress_ratio}; expected one of {SUPPORTED_COMPRESS_RATIOS}")

def build_tensor_specs(
    compress_ratio: int = DEFAULT_COMPRESS_RATIO,
    num_tokens: int = T,
    ori_block_num: int = ORI_BLOCK_NUM,
    cmp_block_num: int = CMP_BLOCK_NUM,
):
    import torch
    from golden import ScalarSpec, TensorSpec
    from utils import build_rope_tables, materialize_token_rope_tables, quant_w_per_channel

    if not 0 < num_tokens <= T:
        raise ValueError(f"num_tokens must be in [1, {T}], got {num_tokens}")
    if ori_block_num <= 0 or cmp_block_num <= 0:
        raise ValueError("dynamic cache block counts must be positive")
    cmp_valid = get_prefill_cmp_valid(compress_ratio)
    storage_block_size = BLOCK_SIZE // compress_ratio if compress_ratio else BLOCK_SIZE
    shared_freqs_cos, shared_freqs_sin = build_rope_tables(M, compress_ratio, dtype=torch.bfloat16)
    rope_positions = torch.arange(T, dtype=torch.int32)
    shared_rope_cos, shared_rope_sin = materialize_token_rope_tables(shared_freqs_cos, shared_freqs_sin, rope_positions)

    def init_q():
        return ((torch.rand(T, H, HEAD_DIM) - 0.5) * 0.05).to(torch.bfloat16)
    def init_ori_kv():
        return ((torch.rand(ori_block_num, BLOCK_SIZE, 1, HEAD_DIM) - 0.5) * 0.05).to(torch.bfloat16)
    def init_cmp_kv():
        return ((torch.rand(cmp_block_num, storage_block_size, 1, HEAD_DIM) - 0.5) * 0.05).to(torch.bfloat16)
    def init_cmp_block_table():
        table = torch.zeros(CMP_MAX_BLOCKS, dtype=torch.int32)
        for blk in range(CMP_MAX_BLOCKS):
            table[blk] = blk % cmp_block_num
        return table
    def init_swa_indices():
        idx = torch.full((T, WIN), -1, dtype=torch.int32)
        for t in range(num_tokens):
            window_start = max(0, t - WIN + 1)
            window = torch.arange(window_start, t + 1, dtype=torch.int32)
            idx[t, :window.numel()] = window
        return idx
    def init_cmp_indices():
        idx = torch.full((T, IDX_TOPK), -1, dtype=torch.int32)
        if compress_ratio:
            for t in range(num_tokens):
                comp_count = min(cmp_valid, (t + 1) // compress_ratio, IDX_TOPK)
                if comp_count > 0:
                    idx[t, :comp_count] = torch.arange(comp_count, dtype=torch.int32)
        return idx
    def init_valid_block_mask():
        """Flag sparse blocks holding at least one valid index, as the real callers do."""
        mask = torch.zeros(T, VALID_BLOCK_MASK_COLS, dtype=torch.int32)
        swa = init_swa_indices()
        cmp = init_cmp_indices()
        for t in range(num_tokens):
            for sb in range(PREFILL_ATTN_BLOCKS):
                for ki in range(PREFILL_ATTN_TILE):
                    k = sb * PREFILL_ATTN_TILE + ki
                    if k >= SPARSE_BIAS_COLS:
                        continue
                    if k < WIN:
                        raw = int(swa[t, k])
                    elif k - WIN < SPARSE_CMP_BIAS_COLS:
                        raw = int(cmp[t, k - WIN])
                    else:
                        continue
                    if raw >= 0:
                        mask[t, sb] = 1
                        break
        return mask

    def init_attn_sink():
        return torch.zeros(H)
    def init_freqs_cos():
        return shared_rope_cos.clone()
    def init_freqs_sin():
        return shared_rope_sin.clone()
    def init_wo_a():
        return ((torch.rand(O_GROUPS, O_LORA, O_GROUP_IN) - 0.5) * O_GROUP_IN ** -0.5).to(torch.bfloat16)
    def init_wo_b():
        return ((torch.rand(D, O_GROUPS * O_LORA) - 0.5) * (O_GROUPS * O_LORA) ** -0.5).to(torch.bfloat16)

    wo_b_i8, wo_b_scale = quant_w_per_channel(init_wo_b())

    return [
        TensorSpec("q", [T, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", [ori_block_num, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("swa_indices", [T, WIN], torch.int32, init_value=init_swa_indices),
        TensorSpec("cmp_kv", [cmp_block_num, storage_block_size, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        ScalarSpec("cmp_storage_block_size", torch.int32, storage_block_size),
        TensorSpec("cmp_indices", [T, IDX_TOPK], torch.int32, init_value=init_cmp_indices),
        TensorSpec("valid_block_mask", [T, VALID_BLOCK_MASK_COLS], torch.int32, init_value=init_valid_block_mask),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        TensorSpec("freqs_cos", [T, ROPE_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [T, ROPE_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("attn_out", [T, D], torch.bfloat16),
    ]

if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true", default=False)
    ratio_choices = list(SUPPORTED_COMPRESS_RATIOS)
    parser.add_argument("--compress-ratio", type=int, default=DEFAULT_COMPRESS_RATIO, choices=ratio_choices)
    parser.add_argument("--num-tokens", type=int, default=T, help="Active prefix; inactive output rows stay zero.")
    parser.add_argument("--ori-block-num", type=int, default=ORI_BLOCK_NUM)
    parser.add_argument("--cmp-block-num", type=int, default=CMP_BLOCK_NUM)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    parser.add_argument("--save-data", action="store_true", help="Save inputs and golden outputs for replay.")
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        fn=prefill_sparse_attn_test,
        specs=build_tensor_specs(args.compress_ratio, args.num_tokens, args.ori_block_num, args.cmp_block_num),
        golden_fn=golden_prefill_sparse_attn,
        save_data=args.save_data,
        golden_data=args.golden_data,
        config=dict(
            dump_passes=args.dump_passes,
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_pmu=args.enable_pmu,
        ),
        rtol=1e-3,
        atol=1e-3,
        compile_only=args.compile_only,
        compare_fn={
            "attn_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, valid_rows=args.num_tokens, zero_tail=True),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
