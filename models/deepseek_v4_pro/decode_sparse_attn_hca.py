# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 sparse attention with grouped output projection (decode)."""


import pypto.language as pl

from config import (
    ACTIVE as M,
    DECODE_BATCH,
    DECODE_SEQ,
    BLOCK_SIZE,
    DECODE_CMP_BLOCK_NUM,
    DECODE_ORI_BLOCK_NUM,
    KV_CMP_MAX_BLOCKS,
    KV_ORI_MAX_BLOCKS,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
)


# model config
B = DECODE_BATCH
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
CMP_TOPK = min(MAX_SEQ_LEN // 128, IDX_TOPK)
TOPK = WIN + CMP_TOPK
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
SUPPORTED_COMPRESS_RATIOS = (0, 4, 128)
DEFAULT_COMPRESS_RATIO = 128
ORI_MAX_BLOCKS = KV_ORI_MAX_BLOCKS
ORI_BLOCK_NUM = DECODE_ORI_BLOCK_NUM
CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM
NEG_INF = -1.0e20

# tiling
VALID_TOKEN_TILE = 8
GATHER_SEG_TILE = 4
GATHER_RUN_TILE = 16
H_TILE = 16
QK_M_TILE = 32
ATTN_K_TILE = 128
A_K_TILE = 256
PROJ_A_MM_N_TILE = 128
MM_T_TILE = 16
T_PAD = ((T + MM_T_TILE - 1) // MM_T_TILE) * MM_T_TILE
B_K_TILE = 256
PROJ_B_MM_N_TILE = 256
PROJ_B_ACT_N_TILE = 512
QUANT_TOKEN_TILE = 8
O_GROUP_TILE = 2
assert H % H_TILE == 0
assert H % O_GROUPS == 0
assert O_GROUPS % O_GROUP_TILE == 0
assert H // H_TILE == O_GROUPS // O_GROUP_TILE
PA_NF_TILE = 2
PROJ_B_D_TILE = 512 if M.name == "flash" else 1792
PROJ_B_ACT_T_TILE = 8
PROJ_B_ACT_TASK_T_TILE = 8
# HCA uses at least two sparse blocks.
SPARSE_BLOCKS = max(2, (TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE)
PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE
GATHER_WIN_ROW_TILE = WIN // GATHER_SEG_TILE
GATHER_CMP_ROW_TILE = (PADDED_TOPK - WIN) // GATHER_SEG_TILE


def get_standalone_cmp_valid(compress_ratio: int) -> int:
    """Map demo compress-ratio modes to the valid compressed-cache tail length."""
    if compress_ratio == 0:
        return 0
    if compress_ratio == 4:
        return IDX_TOPK
    if compress_ratio == 128:
        return MAX_SEQ_LEN // compress_ratio
    raise ValueError(f"Unsupported compress_ratio={compress_ratio}; expected one of {SUPPORTED_COMPRESS_RATIOS}")

@pl.jit.inline
def sparse_attn_hca(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, CMP_TOPK], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[T, D], pl.BF16],
):
    """Run sparse decode attention, inverse RoPE, and grouped output projection."""
    sparse_bias = pl.create_tensor([T, PADDED_TOPK], dtype=pl.FP32)
    for v_blk in pl.spmd(T // VALID_TOKEN_TILE, name_hint="build_valid", allow_early_resolve=True):
        v_t0 = v_blk * VALID_TOKEN_TILE
        v_win_f = pl.cast(window_swa_indices[v_t0 : v_t0 + VALID_TOKEN_TILE, 0 : WIN], target_type=pl.FP32)
        v_idx_f = pl.cast(cmp_sparse_indices[v_t0 : v_t0 + VALID_TOKEN_TILE, 0 : CMP_TOPK], target_type=pl.FP32)
        v_win_one = pl.add(v_win_f, 1.0)
        v_win_floor = pl.maximum(v_win_one, 0.0)
        v_win_valid = pl.minimum(v_win_floor, 1.0)
        v_cmp_one = pl.add(v_idx_f, 1.0)
        v_cmp_floor = pl.maximum(v_cmp_one, 0.0)
        v_cmp_valid = pl.minimum(v_cmp_floor, 1.0)
        v_win_mask = pl.sub(v_win_valid, 1.0)
        sparse_bias[v_t0 : v_t0 + VALID_TOKEN_TILE, 0 : WIN] = pl.mul(v_win_mask, -NEG_INF)
        v_cmp_mask = pl.sub(v_cmp_valid, 1.0)
        sparse_bias[v_t0 : v_t0 + VALID_TOKEN_TILE, WIN : TOPK] = pl.mul(v_cmp_mask, -NEG_INF)
        if PADDED_TOPK > TOPK:
            v_pad = pl.full([VALID_TOKEN_TILE, PADDED_TOPK - TOPK], dtype=pl.FP32, value=NEG_INF)
            sparse_bias[v_t0 : v_t0 + VALID_TOKEN_TILE, TOPK : PADDED_TOPK] = v_pad

    ori_kv_flat = pl.reshape(ori_kv, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    hca_kv_flat = pl.create_tensor([T * PADDED_TOPK, HEAD_DIM], dtype=pl.BF16)
    with pl.spmd(T * GATHER_SEG_TILE, name_hint="hca_gather_kv") as gather_tid:
        g_task = pl.tile.get_block_idx()
        g_t = g_task // GATHER_SEG_TILE
        g_seg = g_task - g_t * GATHER_SEG_TILE
        g_b = g_t // S
        g_row0 = g_t * PADDED_TOPK

        g_wk0 = g_seg * GATHER_WIN_ROW_TILE
        for g_sub in pl.range(GATHER_WIN_ROW_TILE // GATHER_RUN_TILE):
            g_sk0 = g_wk0 + g_sub * GATHER_RUN_TILE
            g_sdst = g_row0 + g_sk0
            g_first = pl.read(window_swa_indices, [g_t, g_sk0])
            g_last = pl.read(window_swa_indices, [g_t, g_sk0 + GATHER_RUN_TILE - 1])
            g_run_ok = (g_last - g_first) + pl.min(g_first, 0) * GATHER_RUN_TILE
            if g_run_ok == GATHER_RUN_TILE - 1:
                g_run_src = pl.cast(g_first, pl.INDEX)
                g_run_tile = ori_kv_flat[g_run_src : g_run_src + GATHER_RUN_TILE, 0:HEAD_DIM]
                hca_kv_flat[g_sdst : g_sdst + GATHER_RUN_TILE, 0:HEAD_DIM] = g_run_tile
            else:
                for g_dr in pl.range(GATHER_RUN_TILE):
                    g_wdst = g_sdst + g_dr
                    g_win_slot_i32 = pl.read(window_swa_indices, [g_t, g_sk0 + g_dr])
                    if g_win_slot_i32 >= 0:
                        g_win_slot = pl.cast(g_win_slot_i32, pl.INDEX)
                        g_win_tile = ori_kv_flat[g_win_slot : g_win_slot + 1, 0:HEAD_DIM]
                        hca_kv_flat[g_wdst : g_wdst + 1, 0:HEAD_DIM] = g_win_tile
                    else:
                        hca_kv_flat[g_wdst : g_wdst + 1, 0:HEAD_DIM] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)

        g_ck0 = g_seg * GATHER_CMP_ROW_TILE
        g_cdst0 = g_row0 + WIN + g_ck0
        for g_dr in pl.range(GATHER_CMP_ROW_TILE):
            g_dst = g_cdst0 + g_dr
            g_cmp_k = g_ck0 + g_dr
            if g_cmp_k < CMP_TOPK:
                g_ridx = pl.read(cmp_sparse_indices, [g_t, g_cmp_k])
                if g_ridx >= 0:
                    g_cblk_i32 = pl.read(cmp_block_table, [g_b, g_ridx // BLOCK_SIZE])
                    g_cblk = pl.cast(g_cblk_i32, pl.INDEX)
                    g_csrc = g_cblk * BLOCK_SIZE + g_ridx % BLOCK_SIZE
                    hca_kv_flat[g_dst : g_dst + 1, 0:HEAD_DIM] = cmp_kv_flat[g_csrc : g_csrc + 1, 0:HEAD_DIM]
                else:
                    hca_kv_flat[g_dst : g_dst + 1, 0:HEAD_DIM] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
            else:
                hca_kv_flat[g_dst : g_dst + 1, 0:HEAD_DIM] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)

    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    sparse_blk_mi = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, HEAD_DIM], dtype=pl.FP32)

    with pl.spmd(T * SPARSE_BLOCKS, name_hint="qk_pv", deps=[gather_tid], allow_early_resolve=True) as _qk_tid:
        qk_item = pl.tile.get_block_idx()
        qk_t = qk_item // SPARSE_BLOCKS
        qk_sb = qk_item - qk_t * SPARSE_BLOCKS
        qk_token_base = qk_t * (H // H_TILE) * SPARSE_BLOCKS * H_TILE
        qk_s0 = qk_sb * ATTN_K_TILE
        qk_bias_row = sparse_bias[qk_t : qk_t + 1, qk_s0 : qk_s0 + ATTN_K_TILE]
        qk_base = qk_t * PADDED_TOPK + qk_s0
        qk_kv = hca_kv_flat[qk_base : qk_base + ATTN_K_TILE, 0:HEAD_DIM]

        for qk_hb in pl.pipeline(H // QK_M_TILE, stage=2):
            qk_h0 = qk_hb * QK_M_TILE
            qk_head_row = qk_t * H + qk_h0
            qk_q_tile = q_flat[qk_head_row : qk_head_row + QK_M_TILE, 0 : HEAD_DIM]
            qk_raw = pl.matmul(qk_q_tile, qk_kv, b_trans=True, out_dtype=pl.FP32)
            qk_scaled = pl.mul(qk_raw, SOFTMAX_SCALE)
            qk_bias_base = pl.full([QK_M_TILE, ATTN_K_TILE], dtype=pl.FP32, value=0.0)
            qk_bias = pl.col_expand(qk_bias_base, qk_bias_row)
            qk_scores = pl.add(qk_scaled, qk_bias)
            qk_mi = pl.row_max(qk_scores)
            qk_centered = pl.row_expand_sub(qk_scores, qk_mi)
            qk_exp = pl.exp(qk_centered)
            qk_li = pl.row_sum(qk_exp)
            qk_exp_bf16 = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
            qk_oi = pl.matmul(qk_exp_bf16, qk_kv, out_dtype=pl.FP32)
            for qk_sub in pl.unroll(QK_M_TILE // H_TILE):
                qk_h_idx = qk_hb * (QK_M_TILE // H_TILE) + qk_sub
                qk_r0 = qk_sub * H_TILE
                qk_blk_base = qk_token_base + qk_h_idx * SPARSE_BLOCKS * H_TILE
                qk_row = qk_blk_base + qk_sb * H_TILE
                sparse_blk_mi[qk_row : qk_row + H_TILE, 0 : 1] = qk_mi[qk_r0 : qk_r0 + H_TILE, 0 : 1]
                sparse_blk_li[qk_row : qk_row + H_TILE, 0 : 1] = qk_li[qk_r0 : qk_r0 + H_TILE, 0 : 1]
                sparse_blk_oi[qk_row : qk_row + H_TILE, 0 : HEAD_DIM] = qk_oi[qk_r0 : qk_r0 + H_TILE, 0 : HEAD_DIM]

    # Inverse RoPE: out[j] = x[j] * cos[j] + x[j ^ 1] * sign[j] * sin[j].
    rope_cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_cs", allow_early_resolve=True):
        cs_cos_f32 = pl.cast(freqs_cos[0:T, 0:HALF_ROPE], target_type=pl.FP32)
        cs_sin_f32 = pl.cast(freqs_sin[0:T, 0:HALF_ROPE], target_type=pl.FP32)
        cs_cos_il = pl.full([T, ROPE_DIM], dtype=pl.FP32, value=0.0)
        cs_cos_il = pl.tensor.scatter(cs_cos_f32, mask_pattern=pl.tile.MaskPattern.P0101, dst=cs_cos_il)
        cs_cos_il = pl.tensor.scatter(cs_cos_f32, mask_pattern=pl.tile.MaskPattern.P1010, dst=cs_cos_il)
        rope_cos_il[0:T, 0:ROPE_DIM] = cs_cos_il
        cs_sin_neg = pl.neg(cs_sin_f32)
        cs_sin_signed = pl.full([T, ROPE_DIM], dtype=pl.FP32, value=0.0)
        cs_sin_signed = pl.tensor.scatter(cs_sin_f32, mask_pattern=pl.tile.MaskPattern.P0101, dst=cs_sin_signed)
        cs_sin_signed = pl.tensor.scatter(cs_sin_neg, mask_pattern=pl.tile.MaskPattern.P1010, dst=cs_sin_signed)
        rope_sin_signed[0:T, 0:ROPE_DIM] = cs_sin_signed

    o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)
    merge_tids = pl.array.create(O_GROUPS // O_GROUP_TILE, pl.TASK_ID)

    for merge_group in pl.parallel(O_GROUPS // O_GROUP_TILE):
        with pl.spmd(T, name_hint="merge_norm") as merge_tid:
            m_t = pl.tile.get_block_idx()
            m_h_idx = merge_group
            m_h0 = m_h_idx * H_TILE
            m_idx = m_t * (H // H_TILE) + m_h_idx
            m_blk_base = m_idx * SPARSE_BLOCKS * H_TILE
            m_mi = sparse_blk_mi[m_blk_base : m_blk_base + H_TILE, 0 : 1]
            m_li = sparse_blk_li[m_blk_base : m_blk_base + H_TILE, 0 : 1]
            m_oi = sparse_blk_oi[m_blk_base : m_blk_base + H_TILE, 0 : HEAD_DIM]

            if SPARSE_BLOCKS > 1:
                for m_sb in pl.range(1, SPARSE_BLOCKS):
                    m_row = m_blk_base + m_sb * H_TILE
                    m_cur_mi = sparse_blk_mi[m_row : m_row + H_TILE, 0 : 1]
                    m_cur_li = sparse_blk_li[m_row : m_row + H_TILE, 0 : 1]
                    m_cur_oi = sparse_blk_oi[m_row : m_row + H_TILE, 0 : HEAD_DIM]
                    m_mi_new = pl.maximum(m_mi, m_cur_mi)
                    m_alpha_delta = pl.sub(m_mi, m_mi_new)
                    m_alpha = pl.exp(m_alpha_delta)
                    m_beta_delta = pl.sub(m_cur_mi, m_mi_new)
                    m_beta = pl.exp(m_beta_delta)
                    m_li_prev = pl.mul(m_alpha, m_li)
                    m_li_cur = pl.mul(m_beta, m_cur_li)
                    m_li = pl.add(m_li_prev, m_li_cur)
                    m_oi_prev = pl.row_expand_mul(m_oi, m_alpha)
                    m_oi_cur = pl.row_expand_mul(m_cur_oi, m_beta)
                    m_oi = pl.add(m_oi_prev, m_oi_cur)
                    m_mi = m_mi_new

            n_sink_bias = pl.reshape(attn_sink[m_h0 : m_h0 + H_TILE], [H_TILE, 1])
            n_zero = pl.sub(m_mi, m_mi)
            n_sink_tile = pl.add(n_zero, n_sink_bias)
            n_sink_delta = pl.sub(n_sink_tile, m_mi)
            n_sink_exp = pl.exp(n_sink_delta)
            n_denom = pl.add(m_li, n_sink_exp)
            n_full = pl.row_expand_div(m_oi, n_denom)[0 : H_TILE, 0 : HEAD_DIM]
            n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")

            m_rope = n_full[0 : H_TILE, NOPE_DIM : HEAD_DIM]
            m_cos_il = rope_cos_il[m_t : m_t + 1, 0 : ROPE_DIM]
            m_sin_signed = rope_sin_signed[m_t : m_t + 1, 0 : ROPE_DIM]
            m_even = pl.gather(m_rope, mask_pattern=pl.tile.MaskPattern.P0101)
            m_odd = pl.gather(m_rope, mask_pattern=pl.tile.MaskPattern.P1010)
            m_swapped = pl.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
            m_swapped = pl.tensor.scatter(m_odd, mask_pattern=pl.tile.MaskPattern.P0101, dst=m_swapped)
            m_swapped = pl.tensor.scatter(m_even, mask_pattern=pl.tile.MaskPattern.P1010, dst=m_swapped)
            m_rope_cos = pl.col_expand_mul(m_rope, m_cos_il)
            m_rope_sin = pl.col_expand_mul(m_swapped, m_sin_signed)
            m_rot = pl.add(m_rope_cos, m_rope_sin)
            n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode="rint")

            for n_hi in pl.range(H_TILE):
                n_gh = m_h0 + n_hi
                n_g = n_gh // HEADS_PER_GROUP
                n_hh = n_gh - n_g * HEADS_PER_GROUP
                n_pack_row = n_g * T + m_t
                n_col = n_hh * HEAD_DIM
                o_packed[n_pack_row : n_pack_row + 1, n_col : n_col + NOPE_DIM] = n_bf16[n_hi : n_hi + 1, 0 : NOPE_DIM]
                n_rope_head = n_rope_bf16[n_hi : n_hi + 1, 0 : ROPE_DIM]
                o_packed[n_pack_row : n_pack_row + 1, n_col + NOPE_DIM : n_col + HEAD_DIM] = n_rope_head
        merge_tids[merge_group] = merge_tid

    o_r_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_i8_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.INT8)
    act_scale_dq = pl.create_tensor([O_GROUPS, T], dtype=pl.FP32)
    partials = pl.create_tensor([T_PAD, O_GROUPS * D], dtype=pl.INT32)
    proj_b_tids = pl.array.create(O_GROUPS // O_GROUP_TILE, pl.TASK_ID)

    with pl.manual_scope():
        for group_bundle in pl.parallel(O_GROUPS // O_GROUP_TILE):
            with pl.spmd(
                O_GROUP_TILE * (O_LORA // PROJ_A_MM_N_TILE // PA_NF_TILE),
                name_hint="proj_a_mm",
                deps=[merge_tids[group_bundle]],
                allow_early_resolve=True,
            ) as pa_tid:
                pa_idx = pl.tile.get_block_idx()
                pa_local_group = pa_idx // (O_LORA // PROJ_A_MM_N_TILE // PA_NF_TILE)
                pa_blk = pa_idx - pa_local_group * (O_LORA // PROJ_A_MM_N_TILE // PA_NF_TILE)
                g = group_bundle * O_GROUP_TILE + pa_local_group
                row_base_o = g * T
                out_col_g = g * O_LORA
                for anf in pl.range(PA_NF_TILE):
                    n0 = pa_blk * (PA_NF_TILE * PROJ_A_MM_N_TILE) + anf * PROJ_A_MM_N_TILE
                    xa0_chunk = pl.slice(o_packed, [MM_T_TILE, A_K_TILE], [row_base_o, 0], valid_shape=[T, A_K_TILE])
                    wa0_chunk = wo_a[g : g + 1, n0 : n0 + PROJ_A_MM_N_TILE, 0:A_K_TILE]
                    acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                    for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
                        k0 = kb * A_K_TILE
                        xa_k_chunk = pl.slice(
                            o_packed,
                            [MM_T_TILE, A_K_TILE],
                            [row_base_o, k0],
                            valid_shape=[T, A_K_TILE],
                        )
                        wa_k_chunk = wo_a[g : g + 1, n0 : n0 + PROJ_A_MM_N_TILE, k0 : k0 + A_K_TILE]
                        acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)
                    o_r_pad = pl.assemble(o_r_pad, acc_a, [0, out_col_g + n0])

            with pl.spmd(
                O_GROUP_TILE,
                name_hint="quant",
                deps=[pa_tid],
                allow_early_resolve=True,
            ) as q_tid:
                q_local_group = pl.tile.get_block_idx()
                g = group_bundle * O_GROUP_TILE + q_local_group
                col_g = g * O_LORA
                for qt in pl.pipeline(0, T, QUANT_TOKEN_TILE, stage=2):
                    oc_amax = o_r_pad[qt : qt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA]
                    g_abs = pl.abs(oc_amax)
                    g_row_max = pl.row_max(g_abs)
                    g_row_max = pl.reshape(g_row_max, [1, QUANT_TOKEN_TILE])
                    g_amax_floor = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                    g_amax = pl.maximum(g_amax_floor, g_row_max)
                    g_scale_num = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
                    g_sq_row = pl.div(g_scale_num, g_amax)
                    g_scale_dq = pl.recip(g_sq_row)
                    act_scale_dq[g : g + 1, qt : qt + QUANT_TOKEN_TILE] = g_scale_dq
                    g_sq_col = pl.reshape(g_sq_row, [QUANT_TOKEN_TILE, 1])
                    oc_q = o_r_pad[qt : qt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA]
                    oq_scaled = pl.row_expand_mul(oc_q, g_sq_col)
                    oq_i32 = pl.cast(oq_scaled, target_type=pl.INT32, mode="rint")
                    oq_half = pl.cast(oq_i32, target_type=pl.FP16, mode="round")
                    oq_i8 = pl.cast(oq_half, target_type=pl.INT8, mode="trunc")
                    o_r_i8_pad[qt : qt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA] = oq_i8
                    if T_PAD > T:
                        zero_half = pl.full([T_PAD - T, O_LORA], dtype=pl.FP16, value=0.0)
                        zero_i8 = pl.cast(zero_half, target_type=pl.INT8, mode="trunc")
                        o_r_i8_pad[T:T_PAD, col_g : col_g + O_LORA] = zero_i8

            with pl.spmd(
                O_GROUP_TILE * (D // PROJ_B_D_TILE),
                name_hint="proj_b_mm",
                deps=[q_tid],
                allow_early_resolve=True,
            ) as pb_tid:
                pb_idx = pl.tile.get_block_idx()
                pb_local_group = pb_idx // (D // PROJ_B_D_TILE)
                dc = pb_idx - pb_local_group * (D // PROJ_B_D_TILE)
                g = group_bundle * O_GROUP_TILE + pb_local_group
                col_g = g * O_LORA
                d0 = dc * PROJ_B_D_TILE
                for nf in pl.range(PROJ_B_D_TILE // PROJ_B_MM_N_TILE):
                    n0 = d0 + nf * PROJ_B_MM_N_TILE
                    b_act = o_r_i8_pad[:, col_g : col_g + B_K_TILE]
                    b_weight = wo_b[n0 : n0 + PROJ_B_MM_N_TILE, col_g : col_g + B_K_TILE]
                    acc_b = pl.matmul(b_act, b_weight, b_trans=True, out_dtype=pl.INT32)
                    for kb in pl.pipeline(1, O_LORA // B_K_TILE, stage=2):
                        k0 = col_g + kb * B_K_TILE
                        b_act = o_r_i8_pad[:, k0 : k0 + B_K_TILE]
                        b_weight = wo_b[n0 : n0 + PROJ_B_MM_N_TILE, k0 : k0 + B_K_TILE]
                        acc_b = pl.matmul_acc(acc_b, b_act, b_weight, b_trans=True)
                    partial_n0 = g * D + n0
                    partials[0:MM_T_TILE, partial_n0 : partial_n0 + PROJ_B_MM_N_TILE] = acc_b
            proj_b_tids[group_bundle] = pb_tid

    with pl.spmd(
        (D // PROJ_B_ACT_N_TILE) * (T // PROJ_B_ACT_TASK_T_TILE),
        name_hint="proj_b_act",
        deps=[proj_b_tids[i] for i in range(O_GROUPS // O_GROUP_TILE)],
        allow_early_resolve=True,
    ) as _act_tid:
        act_idx = pl.tile.get_block_idx()
        nreg = act_idx // (T // PROJ_B_ACT_TASK_T_TILE)
        tblk = act_idx - nreg * (T // PROJ_B_ACT_TASK_T_TILE)
        ob_n0 = nreg * PROJ_B_ACT_N_TILE
        t0 = tblk * PROJ_B_ACT_TASK_T_TILE
        wb_scale = wo_b_scale[ob_n0 : ob_n0 + PROJ_B_ACT_N_TILE]
        wb_scale_tile = pl.reshape(wb_scale, [1, PROJ_B_ACT_N_TILE])
        for b_tb in pl.range(t0, t0 + PROJ_B_ACT_TASK_T_TILE, PROJ_B_ACT_T_TILE):
            acc = pl.full([PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE], dtype=pl.FP32, value=0.0)
            for act_g in pl.pipeline(O_GROUPS, stage=2):
                p_col0 = act_g * D + ob_n0
                p_g = partials[b_tb : b_tb + PROJ_B_ACT_T_TILE, p_col0 : p_col0 + PROJ_B_ACT_N_TILE]
                g_scale_row = act_scale_dq[act_g : act_g + 1, b_tb : b_tb + PROJ_B_ACT_T_TILE]
                g_scale = pl.reshape(g_scale_row, [PROJ_B_ACT_T_TILE, 1])
                p_g_f32 = pl.cast(p_g, target_type=pl.FP32, mode="none")
                p_g_scaled = pl.row_expand_mul(p_g_f32, g_scale)
                acc = pl.add(acc, p_g_scaled)
            out_t = pl.col_expand_mul(acc, wb_scale_tile)
            out_bf16 = pl.cast(out_t, target_type=pl.BF16, mode="rint")
            attn_out[b_tb : b_tb + PROJ_B_ACT_T_TILE, ob_n0 : ob_n0 + PROJ_B_ACT_N_TILE] = out_bf16

    return attn_out

@pl.jit
def sparse_attn_test(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, CMP_TOPK], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    sparse_attn_hca(
        q, ori_kv, window_swa_indices,
        cmp_kv, cmp_block_table, cmp_sparse_indices,
        attn_sink, freqs_cos, freqs_sin,
        wo_a, wo_b, wo_b_scale,
        attn_out,
    )
    return attn_out


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


def _quant_w_per_channel(w):
    """Per-output-channel INT8 quant on the last axis."""
    import torch

    amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.unsqueeze(-1)
    w_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


def golden_sparse_attn(tensors):
    """Torch reference: sparse_attn decode path followed by grouped o_proj."""
    import torch

    q = tensors["q"].float()
    ori_kv = tensors["ori_kv"].float()
    window_swa_indices = tensors["window_swa_indices"]
    cmp_kv = tensors["cmp_kv"].float()
    cmp_block_table = tensors["cmp_block_table"]
    cmp_sparse_indices = tensors["cmp_sparse_indices"]
    attn_sink = tensors["attn_sink"].float()
    cos = tensors["freqs_cos"].float()
    sin = tensors["freqs_sin"].float()
    wo_a = tensors["wo_a"].float()
    wo_b_i8 = tensors["wo_b"]
    wo_b_scale = tensors["wo_b_scale"].float()

    o = torch.zeros(T, H, HEAD_DIM)

    for t in range(T):
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

    seq_per_batch = T // B
    o_model = o.float().view(B, seq_per_batch, O_GROUPS, O_GROUP_IN)
    o_r = torch.einsum("bsgd,grd->bsgr", o_model, wo_a)
    # Quantize each output-projection group independently.
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

def build_tensor_specs(
    compress_ratio: int = DEFAULT_COMPRESS_RATIO,
    causal_regression_fixture: bool = False,
    short_window_fixture: bool = False,
    mixed_topk_fixture: bool = False,
    cache_window_replacement_fixture: bool = False,
):
    """Build deterministic demo tensors for the merged standalone harness."""
    import torch
    from decode_metadata import block_table
    from golden import TensorSpec

    cmp_valid = min(get_standalone_cmp_valid(compress_ratio), TOPK - WIN)

    def init_q():
        """Initialize the query tensor used by the decode attention stage."""
        q = torch.rand(T, H, HEAD_DIM) - 0.5
        if causal_regression_fixture:
            q[0].fill_(1.0)
        return q

    def init_ori_kv():
        """Initialize the sliding-window KV cache pages."""
        kv = torch.rand(ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM) - 0.5
        if causal_regression_fixture:
            kv[0, WIN - 1, 0].fill_(8.0)
        if cache_window_replacement_fixture:
            kv[0, 16, 0].fill_(0.0)
            kv[0, 16, 0, 0] = 4.0
        return kv

    def init_window_swa_indices():
        """Build physical cache-row indices for standalone window raw slots."""
        tbl = init_window_block_table()
        indices = torch.full((T, WIN), -1, dtype=torch.int32)
        for t in range(T):
            b = t // S
            for raw in range(WIN):
                blk = int(tbl[b, raw // BLOCK_SIZE].item())
                if blk >= 0:
                    indices[t, raw] = blk * BLOCK_SIZE + raw % BLOCK_SIZE
        return indices

    def init_cmp_kv():
        """Initialize the compressed-cache KV pages."""
        return torch.rand(CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM) - 0.5

    def init_attn_sink():
        """Initialize the per-head sink logits to zero."""
        return torch.zeros(H)

    def init_window_block_table():
        """Build the demo block table for the sliding-window cache pages."""
        return block_table(batch=B, table_blocks=ORI_MAX_BLOCKS, physical_blocks=ORI_BLOCK_NUM)

    def init_cmp_block_table():
        """Build the demo block table for the compressed-cache pages."""
        return block_table(batch=B, table_blocks=CMP_MAX_BLOCKS, physical_blocks=CMP_BLOCK_NUM)

    def init_cmp_sparse_indices():
        """Build the padded compressed sparse-index list."""
        indices = torch.full((T, CMP_TOPK), -1, dtype=torch.int32)
        if cmp_valid:
            indices[:, :cmp_valid] = torch.arange(cmp_valid, dtype=torch.int32)
        if short_window_fixture:
            indices[:, :] = -1
        if mixed_topk_fixture:
            indices[:, :] = -1
            mixed_cmp_valid = min(cmp_valid, IDX_TOPK)
            if mixed_cmp_valid:
                indices[:, :mixed_cmp_valid] = torch.arange(mixed_cmp_valid, dtype=torch.int32)
        if cache_window_replacement_fixture:
            indices[:, :] = -1
        if causal_regression_fixture:
            indices[0, :] = -1
        return indices

    def init_cos():
        """Build the split-half cosine table used by the inverse-RoPE reference."""
        angles = torch.arange(T * HALF_ROPE).reshape(T, HALF_ROPE) * 1e-3
        cos_half = torch.cos(angles)
        return torch.cat([cos_half, cos_half], dim=-1)

    def init_sin():
        """Build the split-half sine table used by the inverse-RoPE reference."""
        angles = torch.arange(T * HALF_ROPE).reshape(T, HALF_ROPE) * 1e-3
        sin_half = torch.sin(angles)
        return torch.cat([sin_half, sin_half], dim=-1)

    def init_wo_a():
        """Initialize the grouped first-stage output-projection weights."""
        return (torch.rand(O_GROUPS, O_LORA, O_GROUP_IN) - 0.5) / (O_GROUP_IN ** 0.5)

    wo_b_bf16 = ((torch.rand(D, O_GROUPS * O_LORA) - 0.5) / ((O_GROUPS * O_LORA) ** 0.5)).to(torch.bfloat16)
    wo_b_i8, wo_b_scale = _quant_w_per_channel(wo_b_bf16)

    def init_wo_b():
        """Initialize the second-stage output-projection weights in per-channel INT8 form."""
        return wo_b_i8

    def init_wo_b_scale():
        """Initialize the dequant scales paired with the INT8 second-stage weights."""
        return wo_b_scale

    return [
        TensorSpec("q", [T, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("window_swa_indices", [T, WIN], torch.int32, init_value=init_window_swa_indices),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [B, CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("cmp_sparse_indices", [T, CMP_TOPK], torch.int32, init_value=init_cmp_sparse_indices),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("freqs_cos", [T, ROPE_DIM], torch.bfloat16, init_value=init_cos),
        TensorSpec("freqs_sin", [T, ROPE_DIM], torch.bfloat16, init_value=init_sin),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=init_wo_b),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=init_wo_b_scale),
        TensorSpec("attn_out", [T, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    compress_choices = list(SUPPORTED_COMPRESS_RATIOS)
    parser.add_argument("--compress-ratio", type=int, default=DEFAULT_COMPRESS_RATIO, choices=compress_choices)
    causal_help = "Amplify the S=2 future-window-slot regression; use with --compress-ratio 0."
    parser.add_argument("--causal-regression-fixture", action="store_true", default=False, help=causal_help)
    short_help = "Use a short-window topk row with valid prefix + -1 padding."
    parser.add_argument("--short-window-fixture", action="store_true", default=False, help=short_help)
    mixed_help = "Use -1-padded window slots with valid compressed raw indices."
    parser.add_argument("--mixed-topk-fixture", action="store_true", default=False, help=mixed_help)
    cache_help = "Place a sentinel row inside the cache window prefix."
    parser.add_argument("--cache-window-replacement-fixture", action="store_true", default=False, help=cache_help)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
    dep_help = "Capture PTO2 dependency edges (deps.json); the swimlane "
    dep_help += "converter draws fanout/fanin arrows from the sibling file."
    parser.add_argument("--enable-dep-gen", action="store_true", default=False, help=dep_help)
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    compress_ratio = args.compress_ratio
    summary = f"compress_ratio={compress_ratio} -> TOPK={TOPK} SPARSE_BLOCKS={SPARSE_BLOCKS} PADDED_TOPK={PADDED_TOPK}"
    print(summary, flush=True)

    result = run_jit(
        fn=sparse_attn_test,
        specs=build_tensor_specs(
            compress_ratio,
            args.causal_regression_fixture,
            args.short_window_fixture,
            args.mixed_topk_fixture,
            args.cache_window_replacement_fixture,
        ),
        golden_fn=golden_sparse_attn,
        golden_data=args.golden_data,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_dep_gen=args.enable_dep_gen,
            enable_pmu=args.enable_pmu,
        ),
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
