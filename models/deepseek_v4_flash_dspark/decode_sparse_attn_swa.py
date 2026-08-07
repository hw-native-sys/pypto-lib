# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 SWA sparse attention with grouped output projection (decode).

Sliding window only -- no compressed cache and no indexer. The CSA and HCA
variants live in sibling modules.
"""


import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from attention_tp import GROUP_T_MAX, TP_CHOICES, TP_SIZE, reduce_scatter_fp32
from config import (
    FLASH as M,
    DECODE_BATCH,
    TP,
    DECODE_SEQ,
    BLOCK_SIZE,
    DECODE_ORI_BLOCK_NUM,
    KV_ORI_MAX_BLOCKS,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
)


# Dynamic shape variables.
T_DYN = pl.dynamic("SWA_ATTN_T_DYN")  # T = B * S
ORI_BLOCK_NUM_DYN = pl.dynamic("SWA_ATTN_ORI_BLOCK_NUM_DYN")

# model config
B = DECODE_BATCH // TP if TP_SIZE == 1 else DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
GLOBAL_H = M.num_attention_heads
H = GLOBAL_H // TP_SIZE
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
WIN = M.sliding_window
MAX_SEQ_LEN = M.max_position_embeddings
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
GLOBAL_O_GROUPS = M.o_groups
O_GROUPS = GLOBAL_O_GROUPS // TP_SIZE
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
NEG_INF = -1.0e20
SP_T = GROUP_T_MAX // TP_SIZE

# paged KV cache
ORI_MAX_BLOCKS = KV_ORI_MAX_BLOCKS
ORI_BLOCK_NUM = DECODE_ORI_BLOCK_NUM

# tiling
GATHER_SPLITS = 4
GATHER_ROWS_PER_TASK = WIN // GATHER_SPLITS
GATHER_RUN = 16          # window sub-tile probed for physical contiguity -> one bulk DMA
H_TILE = 16
H_VALID_TILE = min(H_TILE, H)
MERGE_GROUPS_PER_TILE = H_VALID_TILE // HEADS_PER_GROUP
QK_M_TILE = min(32, max(16, H))  # qk_pv M rows; TP8 pads its eight local heads to the cube floor
QK_VALID_HEADS = min(QK_M_TILE, H)
QK_RESULT_H_TILE = 16
QK_H_PAD = ((H + QK_RESULT_H_TILE - 1) // QK_RESULT_H_TILE) * QK_RESULT_H_TILE
QK_PIPELINE_STAGE = min(2, (H + QK_M_TILE - 1) // QK_M_TILE)
ATTN_K_TILE = 128
ROPE_TILE = 16
ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE
A_K_TILE = 256           # proj_a cube K frag
PROJ_A_MM_N_TILE = 128   # proj_a cube N frag
T_PAD = ((T + 16 - 1) // 16) * 16  # T padded up to the 16-row cube M floor
MM_T_TILE = T_PAD  # one cube M tile spans every token row of the T_PAD-strided scratch
ROPE_CS_T_TILE = 8  # rope cos/sin row block; T is a multiple of 8 by the batch contract
BIAS_T_TILE = 8     # swa_valid_bias row block, same contract
PROJ_A_ROW_TILE = 16  # proj_a cube M; row-blocked so uninitialized pad rows never enter the matmul
PA_N_FRAGS = O_LORA // PROJ_A_MM_N_TILE
B_K_TILE = 256           # proj_b_mm cube K frag
# proj_b_mm cube N frag; Acc = MM_T_TILE*N*4 = 128KB sits exactly on the a2a3 L0C wall.
PROJ_B_MM_N_TILE = 256
PROJ_B_ACT_N_TILE = 512  # proj_b_act vector N frag; keeps the O_GROUPS-way accumulate inside UB
# Fused amax+quant token tile. 8 keeps the [1, QUANT_TOKEN_TILE] fp32 amax tile
# 32-byte aligned (8*4=32B, the alloc-tile row floor).
QUANT_TOKEN_TILE = 8
PROJ_B_D_TILE = 512      # proj_b_mm D chunk per task; its N frags loop inside the task
PROJ_B_ACT_T_TILE = 8    # proj_b_act inner token tile for the O_GROUPS-way INT32->FP32 accumulate
PROJ_B_ACT_TASK_T_TILE = 8   # proj_b_act token block per task
PROJ_B_ACT_N_REGS = D // PROJ_B_ACT_N_TILE
PROJ_B_ACT_PIPELINE_STAGE = min(2, O_GROUPS)
TOPK = WIN               # SWA sparse-K width: sliding window only
SPARSE_BLOCKS = 1        # the SWA window fits one attention K tile
PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE

assert BLOCK_SIZE % GATHER_RUN == 0, "a contiguous run must not straddle two paged blocks by construction"
assert WIN == ATTN_K_TILE, f"SWA decode expects WIN ({WIN}) == ATTN_K_TILE ({ATTN_K_TILE})"


@pl.jit.inline
def sparse_attn_swa_partial(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    sparse_bias: pl.Tensor[[T_DYN, PADDED_TOPK], pl.FP32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_partial: pl.Tensor[[T_DYN, D], pl.FP32],
):
    """SWA attention with one rank-local FP32 output-projection partial."""
    t_dim = pl.tensor.dim(q, 0)
    t_heads = t_dim * H
    t_win = t_dim * WIN
    t_blk = t_dim * QK_H_PAD * SPARSE_BLOCKS
    t_hblocks = t_dim * (QK_H_PAD // H_TILE)
    t_gather = t_dim * GATHER_SPLITS
    rope_cs_blocks = t_dim // ROPE_CS_T_TILE
    act_t_blks = t_dim // PROJ_B_ACT_TASK_T_TILE
    proj_a_rows = (t_dim + PROJ_A_ROW_TILE - 1) // PROJ_A_ROW_TILE
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    ori_kv_flat = pl.reshape(ori_kv, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    partials = pl.create_tensor([T_PAD, O_GROUPS * D], dtype=pl.INT32)
    act_scale_dq = pl.create_tensor([O_GROUPS, T_PAD], dtype=pl.FP32)
    proj_b_tids = pl.array.create(O_GROUPS, pl.TASK_ID)
    # SWA metadata already lowered each logical window row to a physical cache
    # slot. Current decode tokens must be inserted into ori_kv by the caller
    # before this function runs; there is no MTP overlay path here.

    swa_kv_flat = pl.create_tensor([t_win, HEAD_DIM], dtype=pl.BF16)
    gather_tids = pl.array.create(1, pl.TASK_ID)
    with pl.spmd(t_gather, name_hint="swa_gather_kv") as gather_tid:
        g_task = pl.tile.get_block_idx()
        g_t = g_task // GATHER_SPLITS
        g_split = g_task - g_t * GATHER_SPLITS
        g_r0 = g_split * GATHER_ROWS_PER_TASK
        g_base = g_t * WIN
        # Probe each sub-tile's first/last slot: endpoints GATHER_RUN-1 apart mean
        # the whole run sits in one paged block and moves as one bulk copy.
        for g_sub in pl.range(GATHER_ROWS_PER_TASK // GATHER_RUN):
            g_sr0 = g_r0 + g_sub * GATHER_RUN
            g_sdst = g_base + g_sr0
            g_first = pl.read(swa_indices, [g_t, g_sr0])
            g_last = pl.read(swa_indices, [g_t, g_sr0 + GATHER_RUN - 1])
            # A -1 slot anywhere in the run pins g_run_ok below the match value,
            # so an invalid or block-straddling run takes the per-row path.
            g_run_ok = (g_last - g_first) + pl.min(g_first, 0) * GATHER_RUN
            if g_run_ok == GATHER_RUN - 1:
                g_run_src = pl.cast(g_first, pl.INDEX)
                swa_kv_flat[g_sdst : g_sdst + GATHER_RUN, 0 : HEAD_DIM] = ori_kv_flat[
                    g_run_src : g_run_src + GATHER_RUN, 0 : HEAD_DIM
                ]
            else:
                for g_dr in pl.range(GATHER_RUN):
                    g_dst = g_sdst + g_dr
                    g_slot_i32 = pl.read(swa_indices, [g_t, g_sr0 + g_dr])
                    if g_slot_i32 >= 0:
                        g_slot = pl.cast(g_slot_i32, pl.INDEX)
                        swa_kv_flat[g_dst : g_dst + 1, 0 : HEAD_DIM] = ori_kv_flat[g_slot : g_slot + 1, 0 : HEAD_DIM]
                    else:
                        swa_kv_flat[g_dst : g_dst + 1, 0 : HEAD_DIM] = pl.full(
                            [1, HEAD_DIM], dtype=pl.BF16, value=0.0)
    gather_tids[0] = gather_tid

    # qk_pv writes per-tile (mi, li, oi) to GM; merge_norm reads them back. Not
    # fused on a2a3: the PV output (Acc) -> online rescale (Vec) needs an
    # unsupported tmov, and a [H_TILE, HEAD_DIM] carry overflows the Vec buffer.
    q_flat = pl.reshape(q, [t_heads, HEAD_DIM])
    o_packed_heads = pl.create_tensor([O_GROUPS * T_PAD * HEADS_PER_GROUP, HEAD_DIM], dtype=pl.BF16)
    o_packed = pl.reshape(o_packed_heads, [O_GROUPS * T_PAD, O_GROUP_IN])
    sparse_blk_mi = pl.create_tensor([t_blk, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([t_blk, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([t_blk, HEAD_DIM], dtype=pl.FP32)

    with pl.spmd(t_dim, name_hint="qk_pv", deps=[gather_tids[0]], allow_early_resolve=True) as qk_tid:
        qk_t = pl.tile.get_block_idx()
        qk_token_base = qk_t * QK_H_PAD * SPARSE_BLOCKS
        for qk_sb in pl.unroll(SPARSE_BLOCKS):
            qk_s0 = qk_sb * ATTN_K_TILE
            qk_bias_row = sparse_bias[qk_t : qk_t + 1, qk_s0 : qk_s0 + ATTN_K_TILE]
            qk_base = qk_t * WIN + qk_s0
            qk_kv = swa_kv_flat[qk_base : qk_base + ATTN_K_TILE, 0 : HEAD_DIM]

            # Keep both 32-head batches in one token task so they reuse the KV
            # tile already resident in L1 instead of loading it once per block.
            for qk_hb in pl.pipeline((H + QK_M_TILE - 1) // QK_M_TILE, stage=QK_PIPELINE_STAGE):
                qk_h0 = qk_hb * QK_M_TILE
                qk_head_row = qk_t * H + qk_h0
                if H < QK_M_TILE:
                    qk_q_valid = q_flat[qk_head_row : qk_head_row + QK_VALID_HEADS, 0:HEAD_DIM]
                    qk_q_padded = pl.fillpad_expand(
                        qk_q_valid,
                        [QK_M_TILE, HEAD_DIM],
                        pad_value=pl.PadValue.zero,
                    )
                    qk_raw = pl.matmul(qk_q_padded, qk_kv, b_trans=True, out_dtype=pl.FP32)
                else:
                    qk_q_full = q_flat[qk_head_row : qk_head_row + QK_M_TILE, 0:HEAD_DIM]
                    qk_raw = pl.matmul(qk_q_full, qk_kv, b_trans=True, out_dtype=pl.FP32)
                qk_scaled = pl.mul(qk_raw, SOFTMAX_SCALE)
                qk_scores = pl.col_expand_add(qk_scaled, qk_bias_row)
                qk_mi = pl.row_max(qk_scores)
                # Invalid lanes (NEG_INF bias, zero kv rows) exp to ~0; all-invalid
                # blocks die in the merge alpha/beta -- no mask multiply needed.
                qk_exp = pl.exp(pl.row_expand_sub(qk_scores, qk_mi))
                qk_li = pl.row_sum(qk_exp)
                qk_exp_bf16 = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
                qk_oi = pl.matmul(qk_exp_bf16, qk_kv, out_dtype=pl.FP32)
                for qk_sub in pl.unroll(QK_M_TILE // QK_RESULT_H_TILE):
                    qk_h_idx = qk_hb * (QK_M_TILE // QK_RESULT_H_TILE) + qk_sub
                    qk_r0 = qk_sub * QK_RESULT_H_TILE
                    qk_blk_base = qk_token_base + qk_h_idx * SPARSE_BLOCKS * QK_RESULT_H_TILE
                    qk_row = qk_blk_base + qk_sb * QK_RESULT_H_TILE
                    sparse_blk_mi[qk_row : qk_row + QK_RESULT_H_TILE, 0 : 1] = qk_mi[
                        qk_r0 : qk_r0 + QK_RESULT_H_TILE,
                        0 : 1,
                    ]
                    sparse_blk_li[qk_row : qk_row + QK_RESULT_H_TILE, 0 : 1] = qk_li[
                        qk_r0 : qk_r0 + QK_RESULT_H_TILE,
                        0 : 1,
                    ]
                    sparse_blk_oi[qk_row : qk_row + QK_RESULT_H_TILE, 0 : HEAD_DIM] = qk_oi[
                        qk_r0 : qk_r0 + QK_RESULT_H_TILE,
                        0 : HEAD_DIM,
                    ]

    # Materialize the head-invariant interleaved cos and signed-sin rows once.
    # This runs alongside qk_pv and keeps the exact indexed RoPE arithmetic used
    # by the reference path while the group merge below changes only scheduling
    # and store granularity.
    rope_cos_il = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
    rope_swap_idx = pl.create_tensor([H_TILE, ROPE_DIM], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_cs") as rope_tid:
        swap_ones = pl.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        swap_range_i32 = pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32)
        swap_range = pl.cast(swap_range_i32, target_type=pl.FP32)
        swap_col = pl.col_expand_mul(swap_ones, swap_range)
        swap_half = pl.mul(swap_col, 0.5)
        swap_dup_i32 = pl.cast(swap_half, target_type=pl.INT32, mode="trunc")
        swap_dup_f = pl.cast(swap_dup_i32, target_type=pl.FP32)
        swap_lane = pl.sub(swap_col, pl.mul(swap_dup_f, 2.0))
        swap_next = pl.add(swap_col, 1.0)
        swap_stride = pl.mul(swap_lane, 2.0)
        swap_idx_f = pl.sub(swap_next, swap_stride)
        rope_swap_idx[:, :] = pl.cast(swap_idx_f, target_type=pl.INT32)

        cs_ones = pl.full([ROPE_CS_T_TILE, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=1.0)
        cs_range_i32 = pl.arange(0, [1, ROPE_INTERLEAVE_TILE], dtype=pl.INT32)
        cs_range = pl.cast(cs_range_i32, target_type=pl.FP32)
        cs_col = pl.col_expand_mul(cs_ones, cs_range)
        cs_half = pl.mul(cs_col, 0.5)
        cs_dup_i32 = pl.cast(cs_half, target_type=pl.INT32, mode="trunc")
        cs_dup_f = pl.cast(cs_dup_i32, target_type=pl.FP32)
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))
        cs_sign_base = pl.sub(pl.mul(cs_lane, 2.0), 1.0)
        cs_sign = pl.neg(cs_sign_base)
        for cp in pl.range(HALF_ROPE // ROPE_TILE):
            cp_r0 = cp * ROPE_TILE
            cp_c0 = 2 * cp_r0
            for cs_rb in pl.range(rope_cs_blocks):
                cs_t0 = cs_rb * ROPE_CS_T_TILE
                cs_cos = pl.cast(freqs_cos[cs_t0 : cs_t0 + ROPE_CS_T_TILE, cp_r0 : cp_r0 + ROPE_TILE], target_type=pl.FP32)
                cs_sin = pl.cast(freqs_sin[cs_t0 : cs_t0 + ROPE_CS_T_TILE, cp_r0 : cp_r0 + ROPE_TILE], target_type=pl.FP32)
                cs_cos_dup = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
                cs_sin_dup = pl.gather(cs_sin, dim=-1, index=cs_dup_idx)
                cs_sin_signed = pl.mul(cs_sin_dup, cs_sign)
                rope_cos_il[cs_t0 : cs_t0 + ROPE_CS_T_TILE, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = cs_cos_dup
                rope_sin_signed[cs_t0 : cs_t0 + ROPE_CS_T_TILE, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = cs_sin_signed

    # Flatten the one-block SWA merge over token/head tiles into a single
    # 32-block grid, which fits in one AIV wave and avoids eight group-grid
    # submissions. Each block writes two output-projection groups using the
    # same contiguous per-group stores.
    with pl.spmd(t_hblocks, name_hint="merge_norm", deps=[qk_tid, rope_tid],
                 allow_early_resolve=True) as merge_tid:
        m_idx = pl.tile.get_block_idx()
        m_t = m_idx // (QK_H_PAD // H_TILE)
        m_h_idx = m_idx - m_t * (QK_H_PAD // H_TILE)
        m_h0 = m_h_idx * H_TILE
        m_blk_base = m_t * QK_H_PAD + m_h0
        m_mi = sparse_blk_mi[m_blk_base : m_blk_base + H_TILE, 0:1]
        m_li = sparse_blk_li[m_blk_base : m_blk_base + H_TILE, 0:1]
        m_oi = sparse_blk_oi[m_blk_base : m_blk_base + H_TILE, 0:HEAD_DIM]

        n_sink_source = pl.reshape(attn_sink, [1, H])
        n_sink_valid = n_sink_source[0:1, m_h0 : m_h0 + H_VALID_TILE]
        n_sink_zeros = pl.full([1, H_VALID_TILE], dtype=pl.FP32, value=0.0)
        n_sink_materialized = pl.add(n_sink_valid, n_sink_zeros)
        n_sink_padded = pl.fillpad_expand(n_sink_materialized, [1, H_TILE], pad_value=pl.PadValue.zero)
        n_sink = pl.reshape(n_sink_padded, [H_TILE, 1])
        n_sink_delta = pl.sub(n_sink, m_mi)
        n_sink_exp = pl.exp(n_sink_delta)
        n_denom = pl.add(m_li, n_sink_exp)
        n_normalized = pl.row_expand_div(m_oi, n_denom)
        n_full = n_normalized[0:H_TILE, 0:HEAD_DIM]
        n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")

        m_rope = n_full[:, NOPE_DIM:HEAD_DIM]
        m_swapped = pl.gather(m_rope, dim=-1, index=rope_swap_idx[:, :])
        m_cos_il = rope_cos_il[m_t : m_t + 1, 0:ROPE_DIM]
        m_sin_signed = rope_sin_signed[m_t : m_t + 1, 0:ROPE_DIM]
        m_rope_cos = pl.col_expand_mul(m_rope, m_cos_il)
        m_swap_sin = pl.col_expand_mul(m_swapped, m_sin_signed)
        m_rot = pl.add(m_rope_cos, m_swap_sin)
        n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode="rint")

        m_g0 = m_h0 // HEADS_PER_GROUP
        for m_sg in pl.unroll(MERGE_GROUPS_PER_TILE):
            m_src_h0 = m_sg * HEADS_PER_GROUP
            n_pack_row = (m_g0 + m_sg) * T_PAD + m_t
            n_dst_head = n_pack_row * HEADS_PER_GROUP
            o_packed_heads[n_dst_head : n_dst_head + HEADS_PER_GROUP, 0:NOPE_DIM] = n_bf16[
                m_src_h0 : m_src_h0 + HEADS_PER_GROUP, 0:NOPE_DIM
            ]
            o_packed_heads[n_dst_head : n_dst_head + HEADS_PER_GROUP, NOPE_DIM:HEAD_DIM] = n_rope_bf16[
                m_src_h0 : m_src_h0 + HEADS_PER_GROUP, 0:ROPE_DIM
            ]

    # ========================================================================
    # Back-to-back grouped output projection (manual scope, PER-GROUP INT8 quant).
    #
    # Per-GROUP amax localizes the quant reduction to each O_LORA group (vs the
    # per-ROW-amax form, where a full-8192-row reduction is a hard barrier between
    # proj_a and proj_b), so the three stages PIPELINE per group with qwen3-style
    # fine-grained deps: proj_b[*, g] waits only on quant[g], which waits only on
    # proj_a[g, *] -- so proj_b's cube for group g runs while proj_a/quant of later
    # groups are still in flight (a genuine proj_a<->proj_b back-to-back GEMM).
    #
    # manual_scope SUPPRESSES auto-dep, so every edge is explicit: proj_a[g]
    # reads only its o_packed slab -> deps=[merge_tid]; quant[g] deps on group
    # g's proj_a task; proj_b depends directly on quant[g] and writes a disjoint
    # group partial. proj_b_act combines those partials with their group row scales,
    # applies the per-channel weight scale, and is the consolidated attn_out writer.
    # ========================================================================
    o_r_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_i8_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.INT8)
    # [G, T] keeps each group's per-row scale as one contiguous row;
    # column reads would become unsupported strided GM->VecTile loads.
    # Per-group INT32 partials: proj_b_mm (pure cube) writes group g's contribution to
    # output channel n at partials[:, g*D + n]; proj_b_act (pure vector) sums the
    # O_GROUPS partials with their per-group act scales. No atomic-add -> no zero-seed.
    # Package each group's fragments into one grid. The group TaskId is the
    # exact dependency granularity needed by quant/proj_b, while 80 individual
    # orchestration submissions disappear from the critical projection tail.
    with pl.manual_scope():
        for g in pl.parallel(O_GROUPS):
            row_base_o = g * T_PAD
            out_col_g = g * O_LORA

            with pl.spmd(proj_a_rows * PA_N_FRAGS, name_hint="proj_a_mm", deps=[merge_tid],
                         allow_early_resolve=True) as pa_tid:
                pa_unit = pl.tile.get_block_idx()
                pa_rb = pa_unit // PA_N_FRAGS  # row block outermost
                nf = pa_unit - pa_rb * PA_N_FRAGS
                pa_r0 = pa_rb * PROJ_A_ROW_TILE
                pa_rows = pl.min(PROJ_A_ROW_TILE, t_dim - pa_r0)
                pa_src0 = row_base_o + pa_r0
                n0 = nf * PROJ_A_MM_N_TILE
                xa0_chunk = pl.slice(o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, 0], valid_shape=[pa_rows, A_K_TILE])
                wa0_chunk = wo_a[g : g + 1, n0 : n0 + PROJ_A_MM_N_TILE, 0:A_K_TILE]
                acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
                    k0 = kb * A_K_TILE
                    xa_k_chunk = pl.slice(o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, k0], valid_shape=[pa_rows, A_K_TILE])
                    wa_k_chunk = wo_a[g : g + 1, n0 : n0 + PROJ_A_MM_N_TILE, k0 : k0 + A_K_TILE]
                    acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)
                # acc_a is 3D (wo_a keeps its group axis), which subscript-write cannot express.
                o_r_pad = pl.assemble(o_r_pad, acc_a, [pa_r0, out_col_g + n0])

            col_g = g * O_LORA
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="quant", deps=[pa_tid], allow_early_resolve=True) as q_tid:
                for qt in pl.pipeline(0, t_dim, QUANT_TOKEN_TILE, stage=2):
                    oc_amax = o_r_pad[qt : qt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA]
                    g_abs = pl.abs(oc_amax)
                    g_row_max = pl.row_max(g_abs)
                    g_row_max = pl.reshape(g_row_max, [1, QUANT_TOKEN_TILE])
                    g_amax_floor = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                    g_amax = pl.maximum(g_amax_floor, g_row_max)
                    g_scale_num = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
                    g_sq_row = pl.div(g_scale_num, g_amax)
                    g_scale_dq = pl.mul(g_amax, 1.0 / INT8_SCALE_MAX)
                    act_scale_dq[g : g + 1, qt : qt + QUANT_TOKEN_TILE] = g_scale_dq
                    g_sq_col = pl.reshape(g_sq_row, [QUANT_TOKEN_TILE, 1])
                    oc_q = o_r_pad[qt : qt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA]
                    oq_scaled = pl.row_expand_mul(oc_q, g_sq_col)
                    oq_i32 = pl.cast(oq_scaled, target_type=pl.INT32, mode="rint")
                    oq_half = pl.cast(oq_i32, target_type=pl.FP16, mode="round")
                    oq_i8 = pl.cast(oq_half, target_type=pl.INT8, mode="trunc")
                    o_r_i8_pad[qt : qt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA] = oq_i8
                # Zero the rows past the runtime token count; proj_b_mm reads the full T_PAD extent.
                for zt in pl.range(t_dim, T_PAD, QUANT_TOKEN_TILE):
                    zero_half = pl.full([QUANT_TOKEN_TILE, O_LORA], dtype=pl.FP16, value=0.0)
                    o_r_i8_pad[zt : zt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA] = pl.cast(
                        zero_half, target_type=pl.INT8, mode="trunc")

            with pl.spmd(D // PROJ_B_D_TILE, name_hint="proj_b_mm", deps=[q_tid], allow_early_resolve=True) as pb_tid:
                dc = pl.tile.get_block_idx()
                d0 = dc * PROJ_B_D_TILE
                for nf in pl.range(PROJ_B_D_TILE // PROJ_B_MM_N_TILE):
                    n0 = d0 + nf * PROJ_B_MM_N_TILE
                    acc_b = pl.create_tensor([MM_T_TILE, PROJ_B_MM_N_TILE], dtype=pl.INT32)
                    for kb in pl.pipeline(0, O_LORA // B_K_TILE, stage=2):
                        k0 = col_g + kb * B_K_TILE
                        if kb == 0:
                            b_act = o_r_i8_pad[:, col_g : col_g + B_K_TILE]
                            b_weight = wo_b[n0 : n0 + PROJ_B_MM_N_TILE, col_g : col_g + B_K_TILE]
                            acc_b = pl.matmul(b_act, b_weight, b_trans=True, out_dtype=pl.INT32)
                        else:
                            b_act = o_r_i8_pad[:, k0 : k0 + B_K_TILE]
                            b_weight = wo_b[n0 : n0 + PROJ_B_MM_N_TILE, k0 : k0 + B_K_TILE]
                            acc_b = pl.matmul_acc(acc_b, b_act, b_weight, b_trans=True)
                    partials[0:MM_T_TILE, g * D + n0 : g * D + n0 + PROJ_B_MM_N_TILE] = acc_b
            proj_b_tids[g] = pb_tid

    # Consolidate the rank-local grouped INT32 partials in one vector epilogue. Keep
    # the direct per-group task dependencies so there is no synthetic join task
    # between the output-projection cubes and dequantization.
    with pl.spmd(act_t_blks * PROJ_B_ACT_N_REGS, name_hint="proj_b_act",
                 deps=[proj_b_tids[i] for i in range(O_GROUPS)], allow_early_resolve=True) as act_tid:
        act_idx = pl.tile.get_block_idx()
        tblk = act_idx // PROJ_B_ACT_N_REGS  # token block outermost
        nreg = act_idx - tblk * PROJ_B_ACT_N_REGS
        ob_n0 = nreg * PROJ_B_ACT_N_TILE
        t0 = tblk * PROJ_B_ACT_TASK_T_TILE
        wb_scale = wo_b_scale[ob_n0 : ob_n0 + PROJ_B_ACT_N_TILE]
        wb_scale_chunk = pl.reshape(wb_scale, [1, PROJ_B_ACT_N_TILE])
        for b_tb in pl.range(t0, t0 + PROJ_B_ACT_TASK_T_TILE, PROJ_B_ACT_T_TILE):
            acc = pl.full([PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE], dtype=pl.FP32, value=0.0)
            for act_g in pl.pipeline(O_GROUPS, stage=PROJ_B_ACT_PIPELINE_STAGE):
                p_col0 = act_g * D + ob_n0
                p_g = partials[b_tb : b_tb + PROJ_B_ACT_T_TILE, p_col0 : p_col0 + PROJ_B_ACT_N_TILE]
                g_scale_row = act_scale_dq[act_g : act_g + 1, b_tb : b_tb + PROJ_B_ACT_T_TILE]
                g_scale = pl.reshape(g_scale_row, [PROJ_B_ACT_T_TILE, 1])
                p_g_f32 = pl.cast(p_g, target_type=pl.FP32, mode="none")
                p_g_scaled = pl.row_expand_mul(p_g_f32, g_scale)
                acc = pl.add(acc, p_g_scaled)
            out_t = pl.col_expand_mul(acc, wb_scale_chunk)
            attn_partial[b_tb : b_tb + PROJ_B_ACT_T_TILE, ob_n0 : ob_n0 + PROJ_B_ACT_N_TILE] = out_t
    return act_tid


@pl.jit.inline
def sparse_attn_swa(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    sparse_bias: pl.Tensor[[T_DYN, PADDED_TOPK], pl.FP32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[T_DYN, D], pl.BF16],
):
    """Standalone sparse attention with a BF16 projected output."""
    t_dim = pl.tensor.dim(q, 0)
    attn_partial = pl.create_tensor([t_dim, D], dtype=pl.FP32)
    partial_ready = sparse_attn_swa_partial(
        q,
        ori_kv,
        swa_indices,
        sparse_bias,
        attn_sink,
        freqs_cos,
        freqs_sin,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_partial,
    )
    with pl.spmd(
        t_dim * PROJ_B_ACT_N_REGS,
        name_hint="swa_out_cast",
        deps=[partial_ready],
    ) as _cast_tid:
        index = pl.tile.get_block_idx()
        token = index // PROJ_B_ACT_N_REGS
        nreg = index - token * PROJ_B_ACT_N_REGS
        n0 = nreg * PROJ_B_ACT_N_TILE
        partial = attn_partial[token : token + 1, n0 : n0 + PROJ_B_ACT_N_TILE]
        attn_out[token : token + 1, n0 : n0 + PROJ_B_ACT_N_TILE] = pl.cast(
            partial,
            target_type=pl.BF16,
            mode="rint",
        )
    return attn_out


@pl.jit.inline(auto_scope=False)
def sparse_attn_swa_tp(
    q: pl.Tensor[[GROUP_T_MAX, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[GROUP_T_MAX, WIN], pl.INT32],
    sparse_bias: pl.Tensor[[GROUP_T_MAX, PADDED_TOPK], pl.FP32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[GROUP_T_MAX, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[GROUP_T_MAX, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[SP_T, D], pl.BF16],
    scatter_window: pld.DistributedTensor[[GROUP_T_MAX, D], pl.FP32],
    scatter_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Project local attention heads and reduce-scatter their FP32 partial."""
    attn_partial = pl.create_tensor([GROUP_T_MAX, D], dtype=pl.FP32)
    partial_ready = sparse_attn_swa_partial(
        q,
        ori_kv,
        swa_indices,
        sparse_bias,
        attn_sink,
        freqs_cos,
        freqs_sin,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_partial,
    )
    return reduce_scatter_fp32(
        attn_partial,
        attn_out,
        scatter_window,
        scatter_signal,
        my_rank,
        partial_ready,
    )


@pl.jit
def sparse_attn_tp_test(
    q: pl.Tensor[[GROUP_T_MAX, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[GROUP_T_MAX, WIN], pl.INT32],
    swa_lens: pl.Tensor[[GROUP_T_MAX], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[GROUP_T_MAX, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[GROUP_T_MAX, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[SP_T, D], pl.BF16]],
    scatter_window: pld.DistributedTensor[[GROUP_T_MAX, D], pl.FP32],
    scatter_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    sparse_bias = pl.create_tensor([GROUP_T_MAX, PADDED_TOPK], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_valid_bias"):
        v_col = pl.cast(pl.arange(0, [1, ATTN_K_TILE], dtype=pl.INT32), target_type=pl.FP32)
        for vb in pl.range(GROUP_T_MAX // BIAS_T_TILE):
            v_t0 = vb * BIAS_T_TILE
            v_col_m = pl.col_expand(
                pl.full([BIAS_T_TILE, ATTN_K_TILE], dtype=pl.FP32, value=0.0),
                v_col,
            )
            v_lens = pl.cast(
                pl.reshape(swa_lens[v_t0 : v_t0 + BIAS_T_TILE], [BIAS_T_TILE, 1]),
                target_type=pl.FP32,
            )
            v_valid = pl.minimum(
                pl.maximum(pl.neg(pl.row_expand_sub(v_col_m, v_lens)), 0.0),
                1.0,
            )
            sparse_bias[v_t0 : v_t0 + BIAS_T_TILE, 0:ATTN_K_TILE] = pl.mul(
                pl.sub(v_valid, 1.0),
                -NEG_INF,
            )
    return sparse_attn_swa_tp(
        q,
        ori_kv,
        swa_indices,
        sparse_bias,
        attn_sink,
        freqs_cos,
        freqs_sin,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
        scatter_window,
        scatter_signal,
        my_rank,
    )


@pl.jit.host
def l3_sparse_attn_swa_tp(
    q: pl.Tensor[[TP_SIZE, GROUP_T_MAX, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[TP_SIZE, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[TP_SIZE, GROUP_T_MAX, WIN], pl.INT32],
    swa_lens: pl.Tensor[[TP_SIZE, GROUP_T_MAX], pl.INT32],
    attn_sink: pl.Tensor[[TP_SIZE, H], pl.FP32],
    freqs_cos: pl.Tensor[[TP_SIZE, GROUP_T_MAX, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[TP_SIZE, GROUP_T_MAX, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[TP_SIZE, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[TP_SIZE, D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[TP_SIZE, D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[TP_SIZE, SP_T, D], pl.BF16]],
):
    scatter_window_buffer = pld.alloc_window_buffer([GROUP_T_MAX, D], dtype=pl.FP32)
    scatter_signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        scatter_window = pld.window(scatter_window_buffer, [GROUP_T_MAX, D], dtype=pl.FP32)
        scatter_signal = pld.window(scatter_signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        sparse_attn_tp_test(
            q[rank],
            ori_kv[rank],
            swa_indices[rank],
            swa_lens[rank],
            attn_sink[rank],
            freqs_cos[rank],
            freqs_sin[rank],
            wo_a[rank],
            wo_b[rank],
            wo_b_scale[rank],
            attn_out[rank],
            scatter_window,
            scatter_signal,
            rank,
            device=rank,
        )
    return attn_out


@pl.jit
def sparse_attn_test(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    q.bind_dynamic(0, T_DYN)
    swa_indices.bind_dynamic(0, T_DYN)
    swa_lens.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    attn_out.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(q, 0)
    bias_blocks = t_dim // BIAS_T_TILE
    sparse_bias = pl.create_tensor([t_dim, PADDED_TOPK], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_valid_bias"):
        v_col = pl.cast(pl.arange(0, [1, ATTN_K_TILE], dtype=pl.INT32), target_type=pl.FP32)
        for vb in pl.range(bias_blocks):
            v_t0 = vb * BIAS_T_TILE
            v_col_m = pl.col_expand(pl.full([BIAS_T_TILE, ATTN_K_TILE], dtype=pl.FP32, value=0.0), v_col)
            v_lens = pl.cast(pl.reshape(swa_lens[v_t0 : v_t0 + BIAS_T_TILE], [BIAS_T_TILE, 1]), target_type=pl.FP32)
            v_valid = pl.minimum(
                pl.maximum(pl.neg(pl.row_expand_sub(v_col_m, v_lens)), 0.0),
                1.0,
            )
            sparse_bias[v_t0 : v_t0 + BIAS_T_TILE, 0:ATTN_K_TILE] = pl.mul(pl.sub(v_valid, 1.0), -NEG_INF)
    sparse_attn_swa(
        q,
        ori_kv,
        swa_indices,
        sparse_bias,
        attn_sink,
        freqs_cos,
        freqs_sin,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
    )
    return attn_out


def _golden_sparse_attn_partial(tensors):
    """Return one head shard's FP32 grouped output-projection partial."""
    import torch

    q = tensors["q"].float()
    ori_kv = tensors["ori_kv"].float()
    ori_kv_flat = ori_kv.reshape(ori_kv.shape[0] * BLOCK_SIZE, HEAD_DIM)
    swa_indices = tensors["swa_indices"]
    swa_lens = tensors["swa_lens"]
    attn_sink = tensors["attn_sink"].float()
    cos = tensors["freqs_cos"].float()
    sin = tensors["freqs_sin"].float()
    wo_a = tensors["wo_a"].float()
    wo_b_i8 = tensors["wo_b"]
    wo_b_scale = tensors["wo_b_scale"].float()

    tokens = q.shape[0]
    batch = tokens // S
    o = torch.zeros(tokens, H, HEAD_DIM)

    # Per-query-token attention. swa_indices is the authoritative physical
    # cache-row list; invalid tail columns are -1 and swa_lens gives the valid
    # prefix length.
    for t in range(tokens):
        valid_len = int(swa_lens[t].item())
        valid_slots = [int(v) for v in swa_indices[t, :valid_len].tolist() if int(v) >= 0]
        if not valid_slots:
            continue

        q_t = q[t]

        block_mi = []
        block_li = []
        block_oi = []
        for sb in range(SPARSE_BLOCKS):
            start = sb * ATTN_K_TILE
            end = min(start + ATTN_K_TILE, WIN)
            slots = swa_indices[t, start:end].tolist()
            valid_tile = torch.tensor(
                [start + i < valid_len and int(slot) >= 0 for i, slot in enumerate(slots)],
                dtype=torch.bool,
            )
            if end - start < ATTN_K_TILE:
                valid_tile = torch.cat([
                    valid_tile,
                    torch.zeros(ATTN_K_TILE - (end - start), dtype=torch.bool),
                ])
            valid_tile = valid_tile.to(device=ori_kv.device)
            kv_tile = torch.zeros(ATTN_K_TILE, HEAD_DIM, dtype=ori_kv.dtype, device=ori_kv.device)
            for r, slot in enumerate(slots):
                if r >= ATTN_K_TILE:
                    break
                slot_i = int(slot)
                if slot_i >= 0:
                    kv_tile[r] = ori_kv_flat[slot_i]
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

    seq_per_batch = tokens // batch
    o_model = o.float().view(batch, seq_per_batch, O_GROUPS, O_GROUP_IN)
    o_r = torch.einsum("bsgd,grd->bsgr", o_model, wo_a)
    # PER-GROUP INT8 activation quant (one amax per O_LORA group, not per full row):
    # this localizes the reduction so proj_a[g]->quant[g]->proj_b[g] can pipeline
    # back-to-back. Each group's INT32 partial is dequantized by its OWN per-row
    # activation scale before the groups are summed (the per-group scale cannot
    # factor out of the K-sum), then the per-channel weight scale is applied.
    o_r_g = o_r.reshape(tokens, O_GROUPS, O_LORA)
    amax_g = o_r_g.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)   # [tokens, G, 1]
    scale_q_g = INT8_SCALE_MAX / amax_g
    o_r_i8_g = torch.round(o_r_g * scale_q_g).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dq_g = 1.0 / scale_q_g                                              # [tokens, G, 1]
    wo_b_g = wo_b_i8.reshape(D, O_GROUPS, O_LORA)
    out = torch.zeros(tokens, D, dtype=torch.float32)
    for g in range(O_GROUPS):
        p_g = o_r_i8_g[:, g].to(torch.int32) @ wo_b_g[:, g].to(torch.int32).T   # [tokens, D]
        out = out + p_g.float() * scale_dq_g[:, g]                             # per-row group scale
    out = out * wo_b_scale.unsqueeze(0)                                        # per-channel weight scale

    return out


def golden_sparse_attn(tensors):
    """Torch reference: sparse_attn decode path followed by grouped o_proj."""
    import torch

    tensors["attn_out"][:] = _golden_sparse_attn_partial(tensors).to(torch.bfloat16)


def golden_sparse_attn_tp(tensors):
    """Torch reference for local attention heads and FP32 reduce-scatter."""
    import torch

    reduced = torch.zeros(GROUP_T_MAX, D, dtype=torch.float32)
    for rank in range(TP_SIZE):
        reduced += _golden_sparse_attn_partial({
            "q": tensors["q"][rank],
            "ori_kv": tensors["ori_kv"][rank],
            "swa_indices": tensors["swa_indices"][rank],
            "swa_lens": tensors["swa_lens"][rank],
            "attn_sink": tensors["attn_sink"][rank],
            "freqs_cos": tensors["freqs_cos"][rank],
            "freqs_sin": tensors["freqs_sin"][rank],
            "wo_a": tensors["wo_a"][rank],
            "wo_b": tensors["wo_b"][rank],
            "wo_b_scale": tensors["wo_b_scale"][rank],
        })

    for rank in range(TP_SIZE):
        row_start = rank * SP_T
        tensors["attn_out"][rank] = reduced[row_start : row_start + SP_T].to(torch.bfloat16)


def build_tensor_specs(
    causal_regression_fixture: bool = False,
    short_window_fixture: bool = False,
    batch: int = B,
):
    """Build deterministic demo tensors for the merged standalone harness."""
    tokens = batch * S
    import torch
    from golden import TensorSpec
    from utils import block_table, quant_w_per_channel

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
            kv[0, WIN - 1, 0].fill_(8.0)
        return kv

    def init_attn_sink():
        """Initialize the per-head sink logits to zero."""
        return torch.zeros(H)

    def init_ori_block_table():
        """Build the demo block table for the sliding-window cache pages."""
        return block_table(batch=batch, table_blocks=ORI_MAX_BLOCKS, physical_blocks=ORI_BLOCK_NUM)

    def init_swa_lens():
        lens = torch.full((tokens,), WIN, dtype=torch.int32)
        if short_window_fixture:
            lens.fill_(17)
        return lens

    def init_swa_indices():
        """Build physical cache-row indices for the standalone SWA fixture."""
        tbl = init_ori_block_table()
        indices = torch.full((tokens, WIN), -1, dtype=torch.int32)
        lens = init_swa_lens()
        for t in range(tokens):
            b = t // S
            valid_len = int(lens[t].item())
            for w in range(valid_len):
                logical_blk = w // BLOCK_SIZE
                intra = w % BLOCK_SIZE
                blk = int(tbl[b, logical_blk].item())
                if blk >= 0:
                    indices[t, w] = blk * BLOCK_SIZE + intra
        return indices

    def init_cos():
        """Build the split-half cosine table used by the inverse-RoPE reference."""
        angles = torch.arange(tokens * HALF_ROPE).reshape(tokens, HALF_ROPE) * 1e-3
        cos_half = torch.cos(angles)
        return torch.cat([cos_half, cos_half], dim=-1)

    def init_sin():
        """Build the split-half sine table used by the inverse-RoPE reference."""
        angles = torch.arange(tokens * HALF_ROPE).reshape(tokens, HALF_ROPE) * 1e-3
        sin_half = torch.sin(angles)
        return torch.cat([sin_half, sin_half], dim=-1)

    def init_wo_a():
        """Initialize the grouped first-stage output-projection weights."""
        return (torch.rand(O_GROUPS, O_LORA, O_GROUP_IN) - 0.5) / (O_GROUP_IN ** 0.5)

    wo_b_bf16 = ((torch.rand(D, O_GROUPS * O_LORA) - 0.5) / ((O_GROUPS * O_LORA) ** 0.5)).to(torch.bfloat16)
    wo_b_i8, wo_b_scale = quant_w_per_channel(wo_b_bf16)

    def init_wo_b():
        """Initialize the second-stage output-projection weights in per-channel INT8 form."""
        return wo_b_i8

    def init_wo_b_scale():
        """Initialize the dequant scales paired with the INT8 second-stage weights."""
        return wo_b_scale

    return [
        TensorSpec("q", [tokens, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("swa_indices", [tokens, WIN], torch.int32, init_value=init_swa_indices),
        TensorSpec("swa_lens", [tokens], torch.int32, init_value=init_swa_lens),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("freqs_cos", [tokens, ROPE_DIM], torch.bfloat16, init_value=init_cos),
        TensorSpec("freqs_sin", [tokens, ROPE_DIM], torch.bfloat16, init_value=init_sin),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=init_wo_b),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=init_wo_b_scale),
        TensorSpec("attn_out", [tokens, D], torch.bfloat16, is_output=True),
    ]


def build_tp_tensor_specs(
    causal_regression_fixture: bool = False,
    short_window_fixture: bool = False,
):
    """Build contiguous head and output-group shards for distributed validation."""
    import torch
    from golden import TensorSpec
    from utils import block_table, quant_w_per_channel

    tokens = GROUP_T_MAX
    batch = tokens // S

    q_full = torch.rand(tokens, GLOBAL_H, HEAD_DIM) - 0.5
    ori_kv_one = torch.rand(ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM) - 0.5
    if causal_regression_fixture:
        q_full[0].fill_(1.0)
        ori_kv_one[0, WIN - 1, 0].fill_(8.0)

    lens_one = torch.full((tokens,), WIN, dtype=torch.int32)
    if short_window_fixture:
        lens_one.fill_(17)

    table = block_table(batch=batch, table_blocks=ORI_MAX_BLOCKS, physical_blocks=ORI_BLOCK_NUM)
    indices_one = torch.full((tokens, WIN), -1, dtype=torch.int32)
    for token in range(tokens):
        request = token // S
        valid_len = int(lens_one[token].item())
        for window_index in range(valid_len):
            logical_block = window_index // BLOCK_SIZE
            block_offset = window_index % BLOCK_SIZE
            physical_block = int(table[request, logical_block].item())
            if physical_block >= 0:
                indices_one[token, window_index] = physical_block * BLOCK_SIZE + block_offset

    angles = torch.arange(tokens * HALF_ROPE).reshape(tokens, HALF_ROPE) * 1e-3
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    cos_one = torch.cat([cos_half, cos_half], dim=-1)
    sin_one = torch.cat([sin_half, sin_half], dim=-1)

    wo_a_full = (torch.rand(GLOBAL_O_GROUPS, O_LORA, O_GROUP_IN) - 0.5) / (O_GROUP_IN ** 0.5)
    wo_b_full = (
        (torch.rand(D, GLOBAL_O_GROUPS * O_LORA) - 0.5)
        / ((GLOBAL_O_GROUPS * O_LORA) ** 0.5)
    ).to(torch.bfloat16)
    wo_b_full_i8, wo_b_scale_one = quant_w_per_channel(wo_b_full)

    q = torch.stack([chunk.contiguous() for chunk in torch.chunk(q_full, TP_SIZE, dim=1)])
    ori_kv = ori_kv_one.unsqueeze(0).repeat(TP_SIZE, 1, 1, 1, 1).contiguous()
    swa_indices = indices_one.unsqueeze(0).repeat(TP_SIZE, 1, 1).contiguous()
    swa_lens = lens_one.unsqueeze(0).repeat(TP_SIZE, 1).contiguous()
    attn_sink = torch.zeros(TP_SIZE, H, dtype=torch.float32)
    freqs_cos = cos_one.unsqueeze(0).repeat(TP_SIZE, 1, 1).contiguous()
    freqs_sin = sin_one.unsqueeze(0).repeat(TP_SIZE, 1, 1).contiguous()
    wo_a = torch.stack([chunk.contiguous() for chunk in torch.chunk(wo_a_full, TP_SIZE, dim=0)])
    wo_b = torch.stack([chunk.contiguous() for chunk in torch.chunk(wo_b_full_i8, TP_SIZE, dim=1)])
    wo_b_scale = wo_b_scale_one.unsqueeze(0).repeat(TP_SIZE, 1).contiguous()

    return [
        TensorSpec("q", [TP_SIZE, tokens, H, HEAD_DIM], torch.bfloat16, init_value=lambda: q),
        TensorSpec(
            "ori_kv",
            [TP_SIZE, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            init_value=lambda: ori_kv,
        ),
        TensorSpec(
            "swa_indices",
            [TP_SIZE, tokens, WIN],
            torch.int32,
            init_value=lambda: swa_indices,
        ),
        TensorSpec("swa_lens", [TP_SIZE, tokens], torch.int32, init_value=lambda: swa_lens),
        TensorSpec("attn_sink", [TP_SIZE, H], torch.float32, init_value=lambda: attn_sink),
        TensorSpec(
            "freqs_cos",
            [TP_SIZE, tokens, ROPE_DIM],
            torch.bfloat16,
            init_value=lambda: freqs_cos,
        ),
        TensorSpec(
            "freqs_sin",
            [TP_SIZE, tokens, ROPE_DIM],
            torch.bfloat16,
            init_value=lambda: freqs_sin,
        ),
        TensorSpec(
            "wo_a",
            [TP_SIZE, O_GROUPS, O_LORA, O_GROUP_IN],
            torch.bfloat16,
            init_value=lambda: wo_a,
            resident="stacked",
        ),
        TensorSpec(
            "wo_b",
            [TP_SIZE, D, O_GROUPS * O_LORA],
            torch.int8,
            init_value=lambda: wo_b,
            resident="stacked",
        ),
        TensorSpec(
            "wo_b_scale",
            [TP_SIZE, D],
            torch.float32,
            init_value=lambda: wo_b_scale,
            resident="stacked",
        ),
        TensorSpec("attn_out", [TP_SIZE, SP_T, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=list(TP_CHOICES))
    parser.add_argument("-d", "--device", type=str, default=",".join(str(rank) for rank in range(TP_SIZE)))
    parser.add_argument("-b", "--batch", type=int, default=B,
                        help=f"runtime request count; a multiple of 4 up to {B} (the compile-time "
                             "upper bound). The token axis is pl.dynamic, so one compiled program "
                             "serves every value.")
    parser.add_argument("--causal-regression-fixture", action="store_true", default=False,
                        help="Amplify the S=2 future-window-slot regression.")
    parser.add_argument("--short-window-fixture", action="store_true", default=False,
                        help="Use a short-window topk row with valid prefix + -1 padding.")
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2, 4))
    parser.add_argument("--enable-dep-gen", action="store_true", default=False,
                        help="Capture PTO2 dependency edges (deps.json); the swimlane "
                             "converter draws fanout/fanin arrows from the sibling file.")
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    if args.tp != TP_SIZE:
        raise ValueError(f"import-time TP size {TP_SIZE} does not match --tp {args.tp}")
    if args.batch < 4 or args.batch > B or args.batch % 4 != 0:
        parser.error(f"--batch must be a multiple of 4 in [4, {B}], got {args.batch}")
    if TP_SIZE > 1 and args.batch != DECODE_BATCH:
        parser.error(f"distributed TP validation requires --batch={DECODE_BATCH}, got {args.batch}")

    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) < TP_SIZE:
        raise ValueError(f"need at least {TP_SIZE} devices, got {device_ids}")

    print(f"TOPK={TOPK} SPARSE_BLOCKS={SPARSE_BLOCKS} PADDED_TOPK={PADDED_TOPK}", flush=True)

    distributed = TP_SIZE > 1
    compile_cfg = dict(dump_passes=args.dump_passes)
    runtime_cfg = dict(
        platform=args.platform,
        enable_l2_swimlane=args.enable_l2_swimlane,
        enable_dep_gen=args.enable_dep_gen,
        enable_pmu=args.enable_pmu,
    )
    if distributed:
        compile_cfg["distributed_config"] = DistributedConfig(
            device_ids=device_ids[:TP_SIZE],
            num_sub_workers=0,
        )
    else:
        runtime_cfg["device_id"] = device_ids[0]

    result = run_jit(
        fn=l3_sparse_attn_swa_tp if distributed else sparse_attn_test,
        specs=(
            build_tp_tensor_specs(
                args.causal_regression_fixture,
                args.short_window_fixture,
            )
            if distributed
            else build_tensor_specs(
                args.causal_regression_fixture,
                args.short_window_fixture,
                batch=args.batch,
            )
        ),
        golden_fn=golden_sparse_attn_tp if distributed else golden_sparse_attn,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        compile_cfg=compile_cfg,
        runtime_cfg=runtime_cfg,
        compile_only=args.compile_only,
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "attn_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
