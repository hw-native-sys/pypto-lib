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

# Dynamic shape variables.
ORI_BLOCK_NUM_DYN = pl.dynamic("PREFILL_ORI_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CMP_BLOCK_NUM_DYN")
CMP_STORAGE_BLOCK_SIZE_DYN = pl.dynamic("PREFILL_CMP_STORAGE_BLOCK_SIZE_DYN")

# model config
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
D = M.hidden_size
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
PROJ_B_D_TILE = 1024         # proj_b_mm D slab per task; its N frags loop inside the task
PROJ_B_ACT_N_TILE = 512      # proj_b_act vector N frag
QUANT_CHUNKS = 4             # quant dispatch width per group
QUANT_T_TILE = T // QUANT_CHUNKS
PROJ_B_ACT_T_TILE = 16       # proj_b_act inner token tile for the O_GROUPS-way INT32->FP32 accumulate
PROJ_B_ACT_TASK_T_TILE = 32  # proj_b_act token block per task
# Task-array fan-outs; named because deps= list comprehensions cannot be hoisted out of the call.
PA_NFRAGS = O_LORA // PROJ_A_MM_N_TILE
PB_DSLABS = D // PROJ_B_D_TILE
PREFILL_ATTN_TILE = 128      # sparse-K rows per compile-time block
PREFILL_ATTN_BLOCKS = (PREFILL_SPARSE_TOPK + PREFILL_ATTN_TILE - 1) // PREFILL_ATTN_TILE
PREFILL_SPARSE_PAD = PREFILL_ATTN_BLOCKS * PREFILL_ATTN_TILE
MASK_ALIGN_ELEMS = 32 // 4
VALID_BLOCK_MASK_COLS = ((PREFILL_ATTN_BLOCKS + MASK_ALIGN_ELEMS - 1) // MASK_ALIGN_ELEMS) * MASK_ALIGN_ELEMS
# Padded sparse-window columns carrying real metadata entries.
SPARSE_BIAS_COLS = min(TOPK, PREFILL_SPARSE_PAD)
SPARSE_CMP_BIAS_COLS = max(0, SPARSE_BIAS_COLS - WIN)

assert WIN == PREFILL_ATTN_TILE, f"Sparse prefill expects WIN ({WIN}) == PREFILL_ATTN_TILE ({PREFILL_ATTN_TILE})"

@pl.jit.inline
def sparse_attn_math(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    sparse_kv: pl.Tensor[[T * PREFILL_SPARSE_PAD, HEAD_DIM], pl.BF16],
    sparse_bias: pl.Tensor[[T, PREFILL_SPARSE_PAD], pl.FP32],
    valid_block_mask: pl.Tensor[[T, VALID_BLOCK_MASK_COLS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Run source-independent sparse QK/PV, merge, inverse RoPE, and projection."""

    # Block 0 stores all-masked fallback statistics.
    blk_rows = T * (H // HEAD_TILE) * PREFILL_ATTN_BLOCKS * HEAD_TILE
    sparse_blk_mi = pl.create_tensor([blk_rows, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([blk_rows, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([blk_rows, HEAD_DIM], dtype=pl.FP32)
    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    # Reshape sink bias for head-tile broadcast.
    attn_sink_col = pl.reshape(attn_sink, [H, 1])

    # Build head-invariant inverse-RoPE tables in disjoint column tiles.
    rope_cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_swap_idx = pl.create_tensor([HEAD_TILE, ROPE_DIM], dtype=pl.INT32)
    with pl.spmd(ROPE_HALF // ROPE_TILE, name_hint="rope_cs") as rope_cs_tid:
        cp = pl.tile.get_block_idx()
        cp_r0 = cp * ROPE_TILE
        cp_c0 = 2 * cp_r0

        swap_ones = pl.full([HEAD_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        swap_ramp = pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32)
        swap_col = pl.col_expand_mul(swap_ones, swap_ramp)
        swap_dup_i32 = pl.cast(pl.mul(swap_col, 0.5), target_type=pl.INT32, mode="trunc")
        swap_dup_f = pl.cast(swap_dup_i32, target_type=pl.FP32)
        swap_lane = pl.sub(swap_col, pl.mul(swap_dup_f, 2.0))                                   # j%2
        swap_pair_f = pl.sub(pl.add(swap_col, 1.0), pl.mul(swap_lane, 2.0))                     # j^1
        swap_idx = pl.cast(swap_pair_f, target_type=pl.INT32)
        rope_swap_idx[:, cp_c0:cp_c0 + ROPE_INTERLEAVE_TILE] = swap_idx[:, cp_c0:cp_c0 + ROPE_INTERLEAVE_TILE]

        cs_ones = pl.full([T, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=1.0)
        cs_ramp_i32 = pl.arange(0, [1, ROPE_INTERLEAVE_TILE], dtype=pl.INT32)
        cs_ramp = pl.cast(cs_ramp_i32, target_type=pl.FP32)
        cs_col = pl.col_expand_mul(cs_ones, cs_ramp)
        cs_dup_i32 = pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode="trunc")
        cs_dup_f = pl.cast(cs_dup_i32, target_type=pl.FP32)
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)                                    # j>>1
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))                                         # j%2
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))                                     # [+1,-1,...]
        cs_cos = pl.cast(freqs_cos[0:T, cp_r0:cp_r0 + ROPE_TILE], target_type=pl.FP32)
        cs_sin = pl.cast(freqs_sin[0:T, cp_r0:cp_r0 + ROPE_TILE], target_type=pl.FP32)
        cs_cos_il = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
        cs_sin_il = pl.gather(cs_sin, dim=-1, index=cs_dup_idx)
        rope_cos_il[0:T, cp_c0:cp_c0 + ROPE_INTERLEAVE_TILE] = cs_cos_il
        rope_sin_signed[0:T, cp_c0:cp_c0 + ROPE_INTERLEAVE_TILE] = pl.mul(cs_sin_il, cs_sign)

    # Consume staged sparse sources for QK/PV.
    with pl.spmd(T, name_hint="qk_pv") as qk_tid:
        qk_t = pl.tile.get_block_idx()
        if qk_t < num_tokens:
            qk_kv_base = qk_t * PREFILL_SPARSE_PAD
            qk_token_base = qk_t * (H // HEAD_TILE) * PREFILL_ATTN_BLOCKS * HEAD_TILE
            for qk_sb in pl.range(PREFILL_ATTN_BLOCKS):
                qk_s0 = qk_kv_base + qk_sb * PREFILL_ATTN_TILE
                qk_b0 = qk_sb * PREFILL_ATTN_TILE
                qk_bias_row = sparse_bias[qk_t:qk_t + 1, qk_b0:qk_b0 + PREFILL_ATTN_TILE]
                qk_block_valid = pl.read(valid_block_mask, [qk_t, qk_sb])
                if qk_sb == 0:
                    qk_block_valid = pl.cast(1, pl.INT32)
                if qk_block_valid > 0:
                    # Separate QK and PV cache views over the same rows.
                    qk_kv_k = sparse_kv[qk_s0:qk_s0 + PREFILL_ATTN_TILE, :]
                    qk_kv_v = sparse_kv[qk_s0:qk_s0 + PREFILL_ATTN_TILE, :]
                    for qk_hb in pl.pipeline(H // QK_M_TILE, stage=2):
                        qk_head_row = qk_t * H + qk_hb * QK_M_TILE
                        qk_q_tile = q_flat[qk_head_row:qk_head_row + QK_M_TILE, :]
                        qk_raw = pl.matmul(qk_q_tile, qk_kv_k, b_trans=True, out_dtype=pl.FP32)
                        qk_scaled = pl.mul(qk_raw, SOFTMAX_SCALE)
                        qk_scores = pl.col_expand_add(qk_scaled, qk_bias_row)
                        qk_mi = pl.row_max(qk_scores)
                        qk_exp = pl.exp(pl.row_expand_sub(qk_scores, qk_mi))
                        qk_li = pl.row_sum(qk_exp)
                        qk_exp_bf16 = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
                        qk_oi = pl.matmul(qk_exp_bf16, qk_kv_v, out_dtype=pl.FP32)
                        for qk_sub in pl.unroll(QK_M_TILE // HEAD_TILE):
                            qk_h_idx = qk_hb * (QK_M_TILE // HEAD_TILE) + qk_sub
                            qk_r0 = qk_sub * HEAD_TILE
                            qk_blk_base = qk_token_base + qk_h_idx * PREFILL_ATTN_BLOCKS * HEAD_TILE
                            qk_row = qk_blk_base + qk_sb * HEAD_TILE
                            sparse_blk_mi[qk_row:qk_row + HEAD_TILE, :] = qk_mi[qk_r0:qk_r0 + HEAD_TILE, :]
                            sparse_blk_li[qk_row:qk_row + HEAD_TILE, :] = qk_li[qk_r0:qk_r0 + HEAD_TILE, :]
                            sparse_blk_oi[qk_row:qk_row + HEAD_TILE, :] = qk_oi[qk_r0:qk_r0 + HEAD_TILE, :]

    # Merge in FP32 and apply inverse RoPE before BF16 rounding.
    # o_packed aliases grouped-head storage in projection layout.
    o_packed_heads = pl.create_tensor([O_GROUPS * T * HEADS_PER_GROUP, HEAD_DIM], dtype=pl.BF16)
    o_packed = pl.reshape(o_packed_heads, [O_GROUPS * T, O_GROUP_IN])
    # merge_tid orders proj_a after writes to o_packed.
    with pl.spmd(T, name_hint="merge_rope_pack", deps=[qk_tid, rope_cs_tid]) as merge_tid:
        m_t = pl.tile.get_block_idx()
        m_token_base = m_t * (H // HEAD_TILE) * PREFILL_ATTN_BLOCKS * HEAD_TILE
        # Load the aligned mask row; only set blocks have statistics.
        m_mask_row = valid_block_mask[m_t:m_t + 1, :]
        m_swap_idx = rope_swap_idx[:, :]
        m_cos_il = rope_cos_il[m_t:m_t + 1, :]
        m_sin_signed = rope_sin_signed[m_t:m_t + 1, :]
        for m_h_idx in pl.range(H // HEAD_TILE):
            m_h0 = m_h_idx * HEAD_TILE
            if m_t < num_tokens:
                m_blk_base = m_token_base + m_h_idx * PREFILL_ATTN_BLOCKS * HEAD_TILE
                m_mi = sparse_blk_mi[m_blk_base:m_blk_base + HEAD_TILE, :]
                m_li = sparse_blk_li[m_blk_base:m_blk_base + HEAD_TILE, :]
                m_oi = sparse_blk_oi[m_blk_base:m_blk_base + HEAD_TILE, :]
                for m_sb in pl.unroll(1, PREFILL_ATTN_BLOCKS):
                    m_block_valid = pl.read(m_mask_row, [0, m_sb])
                    if m_block_valid > 0:
                        m_row = m_blk_base + m_sb * HEAD_TILE
                        cur_mi = sparse_blk_mi[m_row:m_row + HEAD_TILE, :]
                        cur_li = sparse_blk_li[m_row:m_row + HEAD_TILE, :]
                        cur_oi = sparse_blk_oi[m_row:m_row + HEAD_TILE, :]
                        mi_new = pl.maximum(m_mi, cur_mi)
                        alpha = pl.exp(pl.sub(m_mi, mi_new))
                        beta = pl.exp(pl.sub(cur_mi, mi_new))
                        m_li = pl.add(pl.mul(alpha, m_li), pl.mul(beta, cur_li))
                        m_oi_alpha = pl.row_expand_mul(m_oi, alpha)
                        cur_oi_beta = pl.row_expand_mul(cur_oi, beta)
                        m_oi = pl.add(m_oi_alpha, cur_oi_beta)
                        m_mi = mi_new
                sink_bias = attn_sink_col[m_h0:m_h0 + HEAD_TILE, :]
                sink_tile = pl.add(pl.sub(m_mi, m_mi), sink_bias)
                denom = pl.add(m_li, pl.exp(pl.sub(sink_tile, m_mi)))
                n_full = pl.row_expand_div(m_oi, denom)[0:HEAD_TILE, :]
                n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")
            else:
                n_full = pl.full([HEAD_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                n_bf16 = pl.full([HEAD_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)

            m_rope = n_full[:, NOPE_DIM:HEAD_DIM]
            m_swapped = pl.gather(m_rope, dim=-1, index=m_swap_idx)
            m_rope_cos = pl.col_expand_mul(m_rope, m_cos_il)
            m_swapped_sin = pl.col_expand_mul(m_swapped, m_sin_signed)
            m_rot = pl.add(m_rope_cos, m_swapped_sin)
            n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode="rint")

            if HEAD_TILE % HEADS_PER_GROUP == 0:
                m_g0 = m_h0 // HEADS_PER_GROUP
                for m_sg in pl.unroll(HEAD_TILE // HEADS_PER_GROUP):
                    m_src_h0 = m_sg * HEADS_PER_GROUP
                    m_pack_row = (m_g0 + m_sg) * T + m_t
                    m_dst_head = m_pack_row * HEADS_PER_GROUP
                    # Slice bounds stay inline: hoisting the end offset loses the static dim,
                    # and naming the source materializes an extra tile.
                    o_packed_heads[
                        m_dst_head:m_dst_head + HEADS_PER_GROUP, 0:NOPE_DIM
                    ] = n_bf16[
                        m_src_h0:m_src_h0 + HEADS_PER_GROUP, 0:NOPE_DIM
                    ]
                    o_packed_heads[
                        m_dst_head:m_dst_head + HEADS_PER_GROUP, NOPE_DIM:HEAD_DIM
                    ] = n_rope_bf16[
                        m_src_h0:m_src_h0 + HEADS_PER_GROUP, 0:ROPE_DIM
                    ]
            else:
                # Store groups crossing a head tile one head at a time.
                for m_hi in pl.range(HEAD_TILE):
                    m_gh = m_h0 + m_hi
                    m_g = m_gh // HEADS_PER_GROUP
                    m_pack_row = m_g * T + m_t
                    m_col = (m_gh - m_g * HEADS_PER_GROUP) * HEAD_DIM
                    o_packed[m_pack_row:m_pack_row + 1, m_col:m_col + NOPE_DIM] = n_bf16[m_hi:m_hi + 1, 0:NOPE_DIM]
                    m_rope_row = n_rope_bf16[m_hi:m_hi + 1, 0:ROPE_DIM]
                    o_packed[m_pack_row:m_pack_row + 1, m_col + NOPE_DIM:m_col + HEAD_DIM] = m_rope_row

    # Chain merge, quantization, and projection through explicit task dependencies.
    o_r = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_i8 = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.INT8)
    act_scale_dq = pl.create_tensor([O_GROUPS, T], dtype=pl.FP32)
    partials = pl.create_tensor([T, O_GROUPS * D], dtype=pl.INT32)
    proj_a_tids = pl.array.create(O_GROUPS * PA_NFRAGS, pl.TASK_ID)
    quant_tids = pl.array.create(O_GROUPS * QUANT_CHUNKS, pl.TASK_ID)
    proj_b_tids = pl.array.create(PB_DSLABS * O_GROUPS, pl.TASK_ID)

    with pl.manual_scope():
        # proj_a[g, nf]: BF16 grouped GEMM -> o_r[:, group g], peel-first-iter form.
        for g in pl.parallel(O_GROUPS):
            row_base_o = g * T
            out_col_g = g * O_LORA
            for nf in pl.range(PA_NFRAGS):
                n0 = nf * PROJ_A_MM_N_TILE
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_a_mm", deps=[merge_tid]) as pa_tid:
                    xa0_chunk = o_packed[row_base_o:row_base_o + T, 0:A_K_TILE]
                    wa0_chunk = wo_a[g:g + 1, n0:n0 + PROJ_A_MM_N_TILE, 0:A_K_TILE]
                    acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                    for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
                        k0 = kb * A_K_TILE
                        xa_k_chunk = o_packed[row_base_o:row_base_o + T, k0:k0 + A_K_TILE]
                        wa_k_chunk = wo_a[g:g + 1, n0:n0 + PROJ_A_MM_N_TILE, k0:k0 + A_K_TILE]
                        acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)
                    acc_a_2d = pl.reshape(acc_a, [T, PROJ_A_MM_N_TILE])
                    o_r[0:T, out_col_g + n0:out_col_g + n0 + PROJ_A_MM_N_TILE] = acc_a_2d
                proj_a_tids[g * PA_NFRAGS + nf] = pa_tid

        # quant[g, tc]: per-group amax + symmetric INT8 quant of o_r[:, group g] over a
        # token-chunk, storing the per-row group dequant scale into act_scale_dq[g, :].
        for tc in pl.parallel(QUANT_CHUNKS):
            t_base = tc * QUANT_T_TILE
            for g in pl.range(O_GROUPS):
                col_g = g * O_LORA
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="quant",
                           deps=[proj_a_tids[g * PA_NFRAGS + j] for j in range(PA_NFRAGS)]) as q_tid:
                    for qt in pl.range(t_base, t_base + QUANT_T_TILE, QUANT_TOKEN_TILE):
                        g_amax = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                        for k1 in pl.range(0, O_LORA, QUANT_TILE):
                            oc = o_r[qt:qt + QUANT_TOKEN_TILE, col_g + k1:col_g + k1 + QUANT_TILE]
                            oc_abs = pl.maximum(oc, pl.neg(oc))
                            oc_amax = pl.reshape(pl.row_max(oc_abs), [1, QUANT_TOKEN_TILE])
                            g_amax = pl.maximum(g_amax, oc_amax)
                        g_scale_num = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
                        g_sq_row = pl.div(g_scale_num, g_amax)
                        act_scale_dq[g:g + 1, qt:qt + QUANT_TOKEN_TILE] = pl.recip(g_sq_row)
                        g_sq_col = pl.reshape(g_sq_row, [QUANT_TOKEN_TILE, 1])
                        for k1 in pl.range(0, O_LORA, QUANT_TILE):
                            oc = o_r[qt:qt + QUANT_TOKEN_TILE, col_g + k1:col_g + k1 + QUANT_TILE]
                            oq_scaled = pl.row_expand_mul(oc, g_sq_col)
                            oq_i32 = pl.cast(oq_scaled, target_type=pl.INT32, mode="rint")
                            oq_half = pl.cast(oq_i32, target_type=pl.FP16, mode="round")
                            oq_i8 = pl.cast(oq_half, target_type=pl.INT8, mode="trunc")
                            o_r_i8[qt:qt + QUANT_TOKEN_TILE, col_g + k1:col_g + k1 + QUANT_TILE] = oq_i8
                quant_tids[g * QUANT_CHUNKS + tc] = q_tid

        # proj_b_mm[dc, g]: INT8 GEMM of group g's contribution to a PROJ_B_D_TILE-wide slab
        # of D, written as INT32 partials[:, g*D+n]. Peel-first matmul: matmul_acc from a zero
        # carry trips TLOAD DN->NZ (pypto#1540).
        for dc in pl.parallel(PB_DSLABS):
            d0 = dc * PROJ_B_D_TILE
            for g in pl.range(O_GROUPS):
                col_g = g * O_LORA
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_b_mm",
                           deps=[quant_tids[g * QUANT_CHUNKS + tc] for tc in range(QUANT_CHUNKS)]) as pb_tid:
                    for nf in pl.range(PROJ_B_D_TILE // PROJ_B_MM_N_TILE):
                        n0 = d0 + nf * PROJ_B_MM_N_TILE
                        b_act0 = o_r_i8[:, col_g:col_g + B_K_TILE]
                        b_weight0 = wo_b[n0:n0 + PROJ_B_MM_N_TILE, col_g:col_g + B_K_TILE]
                        acc_b = pl.matmul(b_act0, b_weight0, b_trans=True, out_dtype=pl.INT32)
                        for kb in pl.pipeline(1, O_LORA // B_K_TILE, stage=2):
                            k0 = col_g + kb * B_K_TILE
                            b_act = o_r_i8[:, k0:k0 + B_K_TILE]
                            b_weight = wo_b[n0:n0 + PROJ_B_MM_N_TILE, k0:k0 + B_K_TILE]
                            acc_b = pl.matmul_acc(acc_b, b_act, b_weight, b_trans=True)
                        partials[0:T, g * D + n0:g * D + n0 + PROJ_B_MM_N_TILE] = acc_b
                proj_b_tids[dc * O_GROUPS + g] = pb_tid

    # Dequantize and sum per-group INT32 partials into the BF16 output.
    act_blocks = (D // PROJ_B_ACT_N_TILE) * (T // PROJ_B_ACT_TASK_T_TILE)
    with pl.spmd(act_blocks, name_hint="proj_b_act",
                 deps=[proj_b_tids[i] for i in range(PB_DSLABS * O_GROUPS)]) as act_tid:
        act_idx = pl.tile.get_block_idx()
        nreg = act_idx // (T // PROJ_B_ACT_TASK_T_TILE)
        tblk = act_idx - nreg * (T // PROJ_B_ACT_TASK_T_TILE)
        ob_n0 = nreg * PROJ_B_ACT_N_TILE
        t0 = tblk * PROJ_B_ACT_TASK_T_TILE
        wb_scale = wo_b_scale[ob_n0:ob_n0 + PROJ_B_ACT_N_TILE]
        wb_scale_chunk = pl.reshape(wb_scale, [1, PROJ_B_ACT_N_TILE])
        for b_tb in pl.range(t0, t0 + PROJ_B_ACT_TASK_T_TILE, PROJ_B_ACT_T_TILE):
            acc = pl.full([PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE], dtype=pl.FP32, value=0.0)
            for g in pl.range(O_GROUPS):
                p_g = partials[b_tb:b_tb + PROJ_B_ACT_T_TILE, g * D + ob_n0:g * D + ob_n0 + PROJ_B_ACT_N_TILE]
                g_scale_row = act_scale_dq[g:g + 1, b_tb:b_tb + PROJ_B_ACT_T_TILE]
                g_scale = pl.reshape(g_scale_row, [PROJ_B_ACT_T_TILE, 1])
                p_g_f32 = pl.cast(p_g, target_type=pl.FP32, mode="none")
                p_g_scaled = pl.row_expand_mul(p_g_f32, g_scale)
                acc = pl.add(acc, p_g_scaled)
            out_t = pl.col_expand_mul(acc, wb_scale_chunk)
            out_bf16 = pl.cast(out_t, target_type=pl.BF16, mode="rint")
            attn_out[b_tb:b_tb + PROJ_B_ACT_T_TILE, ob_n0:ob_n0 + PROJ_B_ACT_N_TILE] = out_bf16

    return attn_out


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
    """Stage sparse sources and per-token bias, then run the sparse-attention math."""
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    ori_cache_rows = ori_block_num * BLOCK_SIZE
    cmp_cache_rows = cmp_block_num * pl.tensor.dim(cmp_kv, 1)
    ori_kv_flat = pl.reshape(ori_kv, [ori_cache_rows, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_cache_rows, HEAD_DIM])
    sparse_kv = pl.create_tensor([T * PREFILL_SPARSE_PAD, HEAD_DIM], dtype=pl.BF16)
    sparse_bias = pl.create_tensor([T, PREFILL_SPARSE_PAD], dtype=pl.FP32)

    # Gather sparse sources in reverse token-tile order.
    with pl.spmd((T + GATHER_TOKEN_TILE - 1) // GATHER_TOKEN_TILE, name_hint="gather_ori_kv") as gather_ori_tid:
        gather_schedule_block = pl.tile.get_block_idx()
        gather_token_block = (T + GATHER_TOKEN_TILE - 1) // GATHER_TOKEN_TILE - 1 - gather_schedule_block
        gather_t0 = gather_token_block * GATHER_TOKEN_TILE
        for gather_dt in pl.range(GATHER_TOKEN_TILE):
            gather_t = gather_t0 + gather_dt
            if gather_t < T:
                if gather_t < num_tokens:
                    block_base = gather_t * PREFILL_SPARSE_PAD
                    stage = pl.full([PREFILL_ATTN_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                    for gather_ki in pl.range(PREFILL_ATTN_TILE):
                        gather_raw = pl.read(swa_indices, [gather_t, gather_ki])
                        if gather_raw >= 0:
                            src = pl.cast(gather_raw, pl.INDEX)
                            stage[gather_ki:gather_ki + 1, :] = ori_kv_flat[src:src + 1, :]
                    sparse_kv[block_base:block_base + PREFILL_ATTN_TILE, :] = stage

    gather_cmp_blocks = ((T + GATHER_TOKEN_TILE - 1) // GATHER_TOKEN_TILE) * (PREFILL_ATTN_BLOCKS - 1)
    with pl.spmd(gather_cmp_blocks, name_hint="gather_cmp_kv") as gather_cmp_tid:
        gather_block = pl.tile.get_block_idx()
        gather_schedule_block = gather_block // (PREFILL_ATTN_BLOCKS - 1)
        gather_token_block = (T + GATHER_TOKEN_TILE - 1) // GATHER_TOKEN_TILE - 1 - gather_schedule_block
        gather_sb = gather_block - gather_schedule_block * (PREFILL_ATTN_BLOCKS - 1) + 1
        gather_t0 = gather_token_block * GATHER_TOKEN_TILE
        gather_k0 = gather_sb * PREFILL_ATTN_TILE
        for gather_dt in pl.range(GATHER_TOKEN_TILE):
            gather_t = gather_t0 + gather_dt
            if gather_t < T:
                if gather_t < num_tokens:
                    gather_block_valid = pl.read(valid_block_mask, [gather_t, gather_sb])
                    if gather_block_valid > 0:
                        block_base = gather_t * PREFILL_SPARSE_PAD + gather_k0
                        stage = pl.full([PREFILL_ATTN_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                        for gather_ki in pl.range(PREFILL_ATTN_TILE):
                            gather_cmp_k = gather_k0 + gather_ki - WIN
                            if gather_cmp_k < IDX_TOPK:
                                gather_raw = pl.read(cmp_indices, [gather_t, gather_cmp_k])
                                if gather_raw >= 0:
                                    cmp_slot = gather_raw
                                    blk_slot = cmp_slot // cmp_storage_block_size
                                    blk = pl.cast(pl.read(cmp_block_table, [blk_slot]), pl.INDEX)
                                    src = blk * cmp_storage_block_size + (
                                        cmp_slot - blk_slot * cmp_storage_block_size
                                    )
                                    stage[gather_ki:gather_ki + 1, :] = cmp_kv_flat[src:src + 1, :]
                        sparse_kv[block_base:block_base + PREFILL_ATTN_TILE, :] = stage

    with pl.spmd(T // BIAS_TOKEN_TILE, name_hint="build_bias") as bias_tid:
        bias_blk = pl.tile.get_block_idx()
        bias_t0 = bias_blk * BIAS_TOKEN_TILE
        bias_win_rows = swa_indices[bias_t0:bias_t0 + BIAS_TOKEN_TILE, 0:WIN]
        bias_win_idx = pl.cast(bias_win_rows, target_type=pl.FP32)
        bias_win_raw_flag = pl.minimum(pl.maximum(pl.add(bias_win_idx, 1.0), 0.0), 1.0)
        bias_win = pl.mul(pl.sub(bias_win_raw_flag, 1.0), -FP32_NEG_INF)
        sparse_bias[bias_t0:bias_t0 + BIAS_TOKEN_TILE, 0:WIN] = bias_win
        if SPARSE_CMP_BIAS_COLS > 0:
            bias_cmp_rows = cmp_indices[bias_t0:bias_t0 + BIAS_TOKEN_TILE, 0:SPARSE_CMP_BIAS_COLS]
            bias_cmp_idx = pl.cast(bias_cmp_rows, target_type=pl.FP32)
            bias_cmp_raw_flag = pl.minimum(pl.maximum(pl.add(bias_cmp_idx, 1.0), 0.0), 1.0)
            bias_cmp = pl.mul(pl.sub(bias_cmp_raw_flag, 1.0), -FP32_NEG_INF)
            sparse_bias[bias_t0:bias_t0 + BIAS_TOKEN_TILE, WIN:SPARSE_BIAS_COLS] = bias_cmp
        if PREFILL_SPARSE_PAD > SPARSE_BIAS_COLS:
            bias_pad_cols = PREFILL_SPARSE_PAD - SPARSE_BIAS_COLS
            bias_pad = pl.full([BIAS_TOKEN_TILE, bias_pad_cols], dtype=pl.FP32, value=FP32_NEG_INF)
            sparse_bias[bias_t0:bias_t0 + BIAS_TOKEN_TILE, SPARSE_BIAS_COLS:PREFILL_SPARSE_PAD] = bias_pad

    return sparse_attn_math(
        q,
        sparse_kv, sparse_bias,
        valid_block_mask, attn_sink,
        freqs_cos, freqs_sin,
        wo_a, wo_b, wo_b_scale,
        attn_out, num_tokens,
    )


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
        return min(
            IDX_TOPK,
            S // compress_ratio,
            CMP_MAX_BLOCKS * storage_block_size,
        )
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
        return (
            (torch.rand(cmp_block_num, storage_block_size, 1, HEAD_DIM) - 0.5)
            * 0.05
        ).to(torch.bfloat16)
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
        TensorSpec(
            "cmp_kv",
            [cmp_block_num, storage_block_size, 1, HEAD_DIM],
            torch.bfloat16,
            init_value=init_cmp_kv,
        ),
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
        TensorSpec("attn_out", [T, D], torch.bfloat16, is_output=True),
    ]

if __name__ == "__main__":
    import argparse
    import torch
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible inputs and golden.")
    parser.add_argument("--compile-only", action="store_true", default=False)
    ratio_choices = list(SUPPORTED_COMPRESS_RATIOS)
    parser.add_argument("--compress-ratio", type=int, default=DEFAULT_COMPRESS_RATIO, choices=ratio_choices)
    parser.add_argument("--num-tokens", type=int, default=T, help="Active prefix; inactive output rows stay zero.")
    parser.add_argument("--ori-block-num", type=int, default=ORI_BLOCK_NUM)
    parser.add_argument("--cmp-block-num", type=int, default=CMP_BLOCK_NUM)
    parser.add_argument("--enable-chip-swimlane", nargs="?", const=4, default=0, type=int)
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    result = run_jit(
        fn=prefill_sparse_attn_test,
        specs=build_tensor_specs(args.compress_ratio, args.num_tokens, args.ori_block_num, args.cmp_block_num),
        golden_fn=golden_prefill_sparse_attn,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_pmu=args.enable_pmu,
        ),
        rtol=1e-3,
        atol=1e-3,
        compile_only=args.compile_only,
        compare_fn={
            "attn_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128,
                                       valid_rows=args.num_tokens, zero_tail=True),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
