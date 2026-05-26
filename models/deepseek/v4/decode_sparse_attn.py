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

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE, INT8_SCALE_MAX, INT8_AMAX_EPS


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
TOPK = WIN + IDX_TOPK
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

# kernel-local
SUPPORTED_COMPRESS_RATIOS = (0, 4, 128)
DEFAULT_COMPRESS_RATIO = 0
ORI_MAX_BLOCKS = 1  # paged-KV pool: ori (sliding-window) blocks per batch
ORI_BLOCK_NUM = B * ORI_MAX_BLOCKS
CMP_MAX_BLOCKS = 64  # paged-KV pool: compressed blocks per batch
CMP_BLOCK_NUM = B * CMP_MAX_BLOCKS

# tiling
ROPE_TOKEN_TILE = 4
ROPE_PACK_TOKEN_TILE = 32
ROPE_PACK_GROUP_TILE = 1
H_TILE = 16
ATTN_K_TILE = 32
ROPE_TILE = 16
ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE
A_T_TILE = 16
A_K_TILE = 128
A_N_TILE = 128
B_T_TILE = 16
B_K_TILE = 128
B_N_TILE = 128
QUANT_TILE = 32
QUANT_TOKEN_TILE = 8
QUANT_K_TILE = O_GROUPS * O_LORA // 2


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
def sparse_attn(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, TOPK], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B], pl.INT32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    even_select_local: pl.Tensor[[ROPE_INTERLEAVE_TILE, ROPE_TILE], pl.BF16],
    odd_select_local: pl.Tensor[[ROPE_INTERLEAVE_TILE, ROPE_TILE], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[T, D], pl.BF16],
):
    """Run sparse decode attention, inverse RoPE, and grouped output projection."""
    # Gather the sliding-window + compressed-cache rows into a per-token packed KV list.
    ori_kv_flat = pl.reshape(ori_kv, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    sparse_kv = pl.create_tensor([T * TOPK, HEAD_DIM], dtype=pl.BF16)
    for g_t in pl.spmd(T, name_hint="gather_kv"):
        g_b = g_t // S
        g_s = g_t - g_b * S
        g_seq_end = pl.read(seqused_kv, [g_b])
        g_seq_len = g_seq_end - S + 1 + g_s
        g_win_v = pl.min(WIN, g_seq_len)
        g_cmp_v = g_seq_len - g_win_v
        g_tk_v = pl.min(IDX_TOPK, g_cmp_v)
        g_sparse_k = g_win_v + g_tk_v
        g_kv_base = g_t * TOPK

        # Window prefix: contiguous, copy as one row block.
        g_ori_blk = pl.cast(pl.read(ori_block_table, [g_b, 0]), pl.INDEX)
        g_ori_row = g_ori_blk * BLOCK_SIZE
        window_rows = pl.set_validshape(ori_kv_flat[g_ori_row : g_ori_row + WIN, 0 : HEAD_DIM], g_win_v, HEAD_DIM)
        sparse_kv[g_kv_base : g_kv_base + WIN, 0 : HEAD_DIM] = window_rows

        # Compressed-cache hits after the window prefix (sparse row-gather).
        for g_kk in pl.range(g_tk_v):
            g_raw = pl.read(cmp_sparse_indices, [g_t, g_win_v + g_kk])
            g_slot = g_raw - WIN
            g_blk = pl.cast(pl.read(cmp_block_table, [g_b, g_slot // BLOCK_SIZE]), pl.INDEX)
            g_src_row = g_blk * BLOCK_SIZE + g_slot % BLOCK_SIZE
            g_dst_row = g_kv_base + g_win_v + g_kk
            sparse_kv[g_dst_row : g_dst_row + 1, 0 : HEAD_DIM] = cmp_kv_flat[g_src_row : g_src_row + 1, 0 : HEAD_DIM]

        # Zero-pad the tail so ratio-0/128 sanity modes stay deterministic.
        zero_kv_row = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
        for g_pad_kk in pl.range(g_sparse_k, TOPK):
            g_pad_row = g_kv_base + g_pad_kk
            sparse_kv[g_pad_row : g_pad_row + 1, 0 : HEAD_DIM] = zero_kv_row

    # Sparse-K attention: qk_pv writes per-tile (mi, li, oi) into GM scratch,
    # merge_norm reads them back. ATTN_K_TILE keeps K and V right-buffer
    # copies together under the 64KB L1B limit.
    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    attn_rope_stage = pl.create_tensor([T * H, ROPE_DIM], dtype=pl.BF16)
    o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)
    sparse_blk_mi = pl.create_tensor([T * (H // H_TILE) * ((TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE) * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([T * (H // H_TILE) * ((TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE) * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([T * (H // H_TILE) * ((TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE) * H_TILE, HEAD_DIM], dtype=pl.FP32)

    for qk_t in pl.spmd(T, name_hint="qk_pv"):
        qk_b = qk_t // S
        qk_s = qk_t - qk_b * S
        qk_seq_end = pl.read(seqused_kv, [qk_b])
        qk_seq_len = qk_seq_end - S + 1 + qk_s
        qk_win_v = pl.min(WIN, qk_seq_len)
        qk_tk_v = pl.min(IDX_TOPK, qk_seq_len - qk_win_v)
        qk_sparse_k = qk_win_v + qk_tk_v
        qk_kv_base = qk_t * TOPK
        qk_token_base = qk_t * (H // H_TILE) * ((TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE) * H_TILE
        for qk_h_idx in pl.range((H // H_TILE)):
            qk_h0 = qk_h_idx * H_TILE
            qk_head_row = qk_t * H + qk_h0
            qk_q_tile = q_flat[qk_head_row : qk_head_row + H_TILE, 0 : HEAD_DIM]
            qk_blk_base = qk_token_base + qk_h_idx * ((TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE) * H_TILE

            for qk_sb in pl.range(((TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE)):
                qk_s0 = qk_sb * ATTN_K_TILE
                if qk_s0 < qk_sparse_k:
                    qk_s_v = pl.min(ATTN_K_TILE, qk_sparse_k - qk_s0)
                    qk_kv_k = sparse_kv[qk_kv_base + qk_s0 : qk_kv_base + qk_s0 + ATTN_K_TILE, 0 : HEAD_DIM]
                    qk_raw = pl.matmul(qk_q_tile, qk_kv_k, b_trans=True, out_dtype=pl.FP32)
                    qk_scores_v = pl.set_validshape(pl.mul(qk_raw, SOFTMAX_SCALE), H_TILE, qk_s_v)
                    qk_scores = pl.fillpad(qk_scores_v, pad_value=pl.PadValue.min)
                    qk_mi = pl.row_max(qk_scores)
                    qk_exp = pl.exp(pl.row_expand_sub(qk_scores, qk_mi))
                    qk_exp_bf16 = pl.cast(qk_exp, target_type=pl.BF16)
                    qk_li = pl.row_sum(pl.cast(qk_exp_bf16, target_type=pl.FP32))
                    qk_kv_v = sparse_kv[qk_kv_base + qk_s0 : qk_kv_base + qk_s0 + ATTN_K_TILE, 0 : HEAD_DIM]
                    qk_oi = pl.matmul(qk_exp_bf16, qk_kv_v, out_dtype=pl.FP32)
                    qk_row = qk_blk_base + qk_sb * H_TILE
                    sparse_blk_mi[qk_row : qk_row + H_TILE, 0 : 1] = qk_mi
                    sparse_blk_li[qk_row : qk_row + H_TILE, 0 : 1] = qk_li
                    sparse_blk_oi[qk_row : qk_row + H_TILE, 0 : HEAD_DIM] = qk_oi

    # Online-softmax merge across sparse-K tiles, then sink-norm.
    for m_t in pl.spmd(T, name_hint="merge_norm"):
        m_b = m_t // S
        m_s = m_t - m_b * S
        m_seq_end = pl.read(seqused_kv, [m_b])
        m_seq_len = m_seq_end - S + 1 + m_s
        m_win_v = pl.min(WIN, m_seq_len)
        m_tk_v = pl.min(IDX_TOPK, m_seq_len - m_win_v)
        m_sparse_k = m_win_v + m_tk_v
        m_token_base = m_t * (H // H_TILE) * ((TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE) * H_TILE

        for m_h_idx in pl.range((H // H_TILE)):
            m_h0 = m_h_idx * H_TILE
            m_blk_base = m_token_base + m_h_idx * ((TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE) * H_TILE
            m_mi = sparse_blk_mi[m_blk_base : m_blk_base + H_TILE, 0 : 1]
            m_li = sparse_blk_li[m_blk_base : m_blk_base + H_TILE, 0 : 1]
            m_oi = sparse_blk_oi[m_blk_base : m_blk_base + H_TILE, 0 : HEAD_DIM]

            for m_sb in pl.range(1, ((TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE)):
                m_s0 = m_sb * ATTN_K_TILE
                if m_s0 < m_sparse_k:
                    m_row = m_blk_base + m_sb * H_TILE
                    m_cur_mi = sparse_blk_mi[m_row : m_row + H_TILE, 0 : 1]
                    m_cur_li = sparse_blk_li[m_row : m_row + H_TILE, 0 : 1]
                    m_cur_oi = sparse_blk_oi[m_row : m_row + H_TILE, 0 : HEAD_DIM]
                    m_mi_new = pl.maximum(m_mi, m_cur_mi)
                    m_alpha = pl.exp(pl.sub(m_mi, m_mi_new))
                    m_beta = pl.exp(pl.sub(m_cur_mi, m_mi_new))
                    m_li = pl.add(pl.mul(m_alpha, m_li), pl.mul(m_beta, m_cur_li))
                    m_oi = pl.add(pl.row_expand_mul(m_oi, m_alpha), pl.row_expand_mul(m_cur_oi, m_beta))
                    m_mi = m_mi_new

            n_sink_bias = pl.reshape(attn_sink[m_h0 : m_h0 + H_TILE], [H_TILE, 1])
            n_sink_tile = pl.add(pl.sub(m_mi, m_mi), n_sink_bias)
            n_denom = pl.add(m_li, pl.exp(pl.sub(n_sink_tile, m_mi)))
            n_out = pl.cast(pl.row_expand_div(m_oi, n_denom)[0 : H_TILE, 0 : HEAD_DIM], target_type=pl.BF16)
            n_rope_row = m_t * H + m_h0
            attn_rope_stage[n_rope_row : n_rope_row + H_TILE, 0 : ROPE_DIM] = n_out[0 : H_TILE, NOPE_DIM : HEAD_DIM]

            for n_hi in pl.range(H_TILE):
                n_gh = m_h0 + n_hi
                n_g = n_gh // HEADS_PER_GROUP
                n_hh = n_gh - n_g * HEADS_PER_GROUP
                n_pack_row = n_g * T + m_t
                n_col = n_hh * HEAD_DIM
                o_packed[n_pack_row : n_pack_row + 1, n_col : n_col + NOPE_DIM] = n_out[n_hi : n_hi + 1, 0 : NOPE_DIM]

    # Inverse RoPE: deinterleave even/odd lanes, rotate with cos/sin, reinterleave.
    rope_even_buf = pl.create_tensor([T * H, ROPE_DIM], dtype=pl.FP32)
    rope_odd_buf = pl.create_tensor([T * H, ROPE_DIM], dtype=pl.FP32)
    for r_idx in pl.spmd(T // ROPE_TOKEN_TILE, name_hint="rope"):
        r_t0 = r_idx * ROPE_TOKEN_TILE
        for r_dt in pl.range(ROPE_TOKEN_TILE):
            r_t = r_t0 + r_dt
            r_row = r_t * H

            for r_r0 in pl.range(0, HALF_ROPE, ROPE_TILE):
                # Split interleaved rope lanes into even and odd halves.
                r_tile = attn_rope_stage[r_row : r_row + H, 2 * r_r0 : 2 * r_r0 + ROPE_INTERLEAVE_TILE]
                r_even = pl.matmul(r_tile, even_select_local, out_dtype=pl.FP32)
                r_odd = pl.matmul(r_tile, odd_select_local, out_dtype=pl.FP32)

                # Rotate the even/odd halves with cos/sin.
                r_cos = pl.cast(freqs_cos[r_t : r_t + 1, r_r0 : r_r0 + ROPE_TILE], target_type=pl.FP32)
                r_sin = pl.cast(freqs_sin[r_t : r_t + 1, r_r0 : r_r0 + ROPE_TILE], target_type=pl.FP32)
                r_even_rot = pl.add(pl.col_expand_mul(r_even, r_cos), pl.col_expand_mul(r_odd, r_sin))
                r_odd_rot = pl.sub(pl.col_expand_mul(r_odd, r_cos), pl.col_expand_mul(r_even, r_sin))
                r_even_bf16 = pl.cast(r_even_rot, target_type=pl.BF16, mode="rint")
                r_odd_bf16 = pl.cast(r_odd_rot, target_type=pl.BF16, mode="rint")

                # Reinterleave the rotated halves back to rope-lane order.
                r_even_il = pl.matmul(r_even_bf16, even_select_local, b_trans=True, out_dtype=pl.FP32)
                r_odd_il = pl.matmul(r_odd_bf16, odd_select_local, b_trans=True, out_dtype=pl.FP32)
                rope_even_buf[r_row : r_row + H, 2 * r_r0 : 2 * r_r0 + ROPE_INTERLEAVE_TILE] = r_even_il
                rope_odd_buf[r_row : r_row + H, 2 * r_r0 : 2 * r_r0 + ROPE_INTERLEAVE_TILE] = r_odd_il

    for rp_block in pl.spmd((T // ROPE_PACK_TOKEN_TILE) * O_GROUPS, name_hint="rope_pack"):
        rp_tb = rp_block // O_GROUPS
        rp_g = rp_block - rp_tb * O_GROUPS
        rp_t0 = rp_tb * ROPE_PACK_TOKEN_TILE

        for rp_dt in pl.range(ROPE_PACK_TOKEN_TILE):
            rp_t = rp_t0 + rp_dt
            rp_row = rp_t * H + rp_g * HEADS_PER_GROUP

            # Merge and write only this group's inverse-RoPE tail of o_packed.
            rp_even = rope_even_buf[rp_row : rp_row + HEADS_PER_GROUP, 0 : ROPE_DIM]
            rp_odd = rope_odd_buf[rp_row : rp_row + HEADS_PER_GROUP, 0 : ROPE_DIM]
            rp_full = pl.cast(pl.add(rp_even, rp_odd), target_type=pl.BF16)
            rp_pack_row = rp_g * T + rp_t
            for rp_hh in pl.range(HEADS_PER_GROUP):
                rp_col = rp_hh * HEAD_DIM + NOPE_DIM
                o_packed[rp_pack_row : rp_pack_row + 1, rp_col : rp_col + ROPE_DIM] = rp_full[rp_hh : rp_hh + 1, 0 : ROPE_DIM]

    # Grouped BF16 projection `o_packed @ wo_a^T` -> `o_r`. Vec post-process
    # (BF16 store + per-row partial amax) is T-tiled inside the scope as a
    # pypto#1472 workaround — without it the fused proj_a AIV side oversizes
    # UB and AllocateMemoryAddr rejects the kernel.
    o_r = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.BF16)
    o_r_amax_parts = pl.create_tensor([O_GROUPS * (O_LORA // A_N_TILE), T], dtype=pl.FP32)
    for proj_a_block in pl.spmd(O_GROUPS * (O_LORA // A_N_TILE), name_hint="proj_a"):
        # K-split BF16 matmul for one wo_a output tile. Stays in
        # peel-first-iter form: the `pl.create_tensor` + `if k0 == 0`
        # carry hits pypto#1540 on the 3D wo_a slice.
        g = proj_a_block // (O_LORA // A_N_TILE)
        nb = proj_a_block - g * (O_LORA // A_N_TILE)
        row_base_o = g * T
        out_col_g = g * O_LORA
        n0 = nb * A_N_TILE
        amax_part_row = g * (O_LORA // A_N_TILE) + nb

        xa0_chunk = o_packed[row_base_o:row_base_o + T, 0:A_K_TILE]
        wa0_chunk = wo_a[g:g + 1, n0:n0 + A_N_TILE, 0:A_K_TILE]
        acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
        for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
            k0 = kb * A_K_TILE
            xa_k_chunk = o_packed[row_base_o:row_base_o + T, k0:k0 + A_K_TILE]
            wa_k_chunk = wo_a[g:g + 1, n0:n0 + A_N_TILE, k0:k0 + A_K_TILE]
            acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)

        acc_a_2d = pl.reshape(acc_a, [T, A_N_TILE])
        for tb in pl.range(0, T, A_T_TILE):
            acc_t = acc_a_2d[tb:tb + A_T_TILE, 0:A_N_TILE]
            acc_t_bf16 = pl.cast(acc_t, target_type=pl.BF16)
            o_r[tb:tb + A_T_TILE, out_col_g + n0:out_col_g + n0 + A_N_TILE] = acc_t_bf16
            acc_t_f32 = pl.cast(acc_t_bf16, target_type=pl.FP32)
            acc_t_abs = pl.maximum(acc_t_f32, pl.neg(acc_t_f32))
            acc_t_amax = pl.reshape(pl.row_max(acc_t_abs), [1, A_T_TILE])
            o_r_amax_parts[amax_part_row:amax_part_row + 1, tb:tb + A_T_TILE] = acc_t_amax

    # Per-row symmetric INT8 quant of `o_r`, K-tiled as a second parallel axis.
    o_r_i8 = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.INT8)
    o_r_scale_dq = pl.create_tensor([T, 1], dtype=pl.FP32)
    for q_block in pl.spmd((T // QUANT_TOKEN_TILE) * ((O_GROUPS * O_LORA) // QUANT_K_TILE), name_hint="quant"):
        qt_idx = q_block // ((O_GROUPS * O_LORA) // QUANT_K_TILE)
        qk_idx = q_block - qt_idx * ((O_GROUPS * O_LORA) // QUANT_K_TILE)
        quant_t0 = qt_idx * QUANT_TOKEN_TILE
        k0 = qk_idx * QUANT_K_TILE

        or_amax = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for ab in pl.range(0, O_GROUPS * (O_LORA // A_N_TILE), 1):
            or_a_part = o_r_amax_parts[ab:ab + 1, quant_t0:quant_t0 + QUANT_TOKEN_TILE]
            or_amax = pl.maximum(or_amax, or_a_part)
        or_sq_row = pl.div(pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), or_amax)
        or_scale_dq = pl.reshape(pl.recip(or_sq_row), [QUANT_TOKEN_TILE, 1])
        o_r_scale_dq[quant_t0:quant_t0 + QUANT_TOKEN_TILE, 0:1] = or_scale_dq
        or_sq_col = pl.reshape(or_sq_row, [QUANT_TOKEN_TILE, 1])
        for k1 in pl.range(k0, k0 + QUANT_K_TILE, QUANT_TILE):
            or_q_f32 = pl.cast(o_r[quant_t0:quant_t0 + QUANT_TOKEN_TILE, k1:k1 + QUANT_TILE], target_type=pl.FP32)
            or_q_scaled = pl.row_expand_mul(or_q_f32, or_sq_col)
            or_q_i32 = pl.cast(or_q_scaled, target_type=pl.INT32, mode="rint")
            or_q_half = pl.cast(or_q_i32, target_type=pl.FP16, mode="round")
            o_r_i8[quant_t0:quant_t0 + QUANT_TOKEN_TILE, k1:k1 + QUANT_TILE] = pl.cast(or_q_half, target_type=pl.INT8, mode="trunc")

    # INT8 projection `o_r_i8 @ wo_b^T`, then dequantize -> final BF16 output.
    for nb in pl.spmd(D // B_N_TILE, name_hint="proj_b"):
        # K-split INT8 GEMM + dequant in one scope. T-tiled vec post-process
        # is the pypto#1472 workaround (same as proj_a).
        n0 = nb * B_N_TILE
        acc_b = pl.create_tensor([T, B_N_TILE], dtype=pl.INT32)
        for kb in pl.pipeline(0, (O_GROUPS * O_LORA) // B_K_TILE, stage=2):
            k0 = kb * B_K_TILE
            xb_k_chunk = o_r_i8[:, k0:k0 + B_K_TILE]
            wb_k_chunk = wo_b[n0:n0 + B_N_TILE, k0:k0 + B_K_TILE]
            if k0 == 0:
                acc_b = pl.matmul(xb_k_chunk, wb_k_chunk, b_trans=True, out_dtype=pl.INT32)
            else:
                acc_b = pl.matmul_acc(acc_b, xb_k_chunk, wb_k_chunk, b_trans=True)

        wb_scale_chunk = pl.reshape(wo_b_scale[n0:n0 + B_N_TILE], [1, B_N_TILE])
        for b_tb in pl.range(0, T, B_T_TILE):
            acc_b_t = acc_b[b_tb:b_tb + B_T_TILE, 0:B_N_TILE]
            b_scale_t = o_r_scale_dq[b_tb:b_tb + B_T_TILE, 0:1]
            attn_t = pl.cast(acc_b_t, target_type=pl.FP32, mode="none")
            attn_t = pl.col_expand_mul(pl.row_expand_mul(attn_t, b_scale_t), wb_scale_chunk)
            attn_out[b_tb:b_tb + B_T_TILE, n0:n0 + B_N_TILE] = pl.cast(attn_t, target_type=pl.BF16, mode="rint")

    return attn_out


@pl.jit
def sparse_attn_test(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, TOPK], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B], pl.INT32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    even_select_local: pl.Tensor[[ROPE_INTERLEAVE_TILE, ROPE_TILE], pl.BF16],
    odd_select_local: pl.Tensor[[ROPE_INTERLEAVE_TILE, ROPE_TILE], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    attn_out = sparse_attn(
        q,
        ori_kv,
        ori_block_table,
        cmp_kv,
        cmp_block_table,
        cmp_sparse_indices,
        attn_sink,
        seqused_kv,
        freqs_cos,
        freqs_sin,
        even_select_local,
        odd_select_local,
        wo_a,
        wo_b,
        wo_b_scale,
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
    ori_block_table = tensors["ori_block_table"]
    cmp_kv = tensors["cmp_kv"].float()
    cmp_block_table = tensors["cmp_block_table"]
    cmp_sparse_indices = tensors["cmp_sparse_indices"]
    attn_sink = tensors["attn_sink"].float()
    seqused_kv = tensors["seqused_kv"]
    cos = tensors["freqs_cos"].float()
    sin = tensors["freqs_sin"].float()
    wo_a = tensors["wo_a"].float()
    wo_b_i8 = tensors["wo_b"]
    wo_b_scale = tensors["wo_b_scale"].float()

    o = torch.zeros(T, H, HEAD_DIM)

    # Per-query-token attention. seqused_kv stores each batch's final sparse
    # length for this decode chunk; token t derives its causal length from s.
    for t in range(T):
        b = t // S
        s = t - b * S
        seq_used = int(seqused_kv[b].item()) - S + 1 + s
        window_valid = min(WIN, seq_used)
        cmp_valid = max(seq_used - window_valid, 0)
        gathered = []

        for raw in cmp_sparse_indices[t].tolist():
            if raw < 0:
                continue
            if raw < WIN:
                if raw >= window_valid:
                    continue
                blk_id = int(ori_block_table[b, raw // BLOCK_SIZE].item())
                intra = raw % BLOCK_SIZE
                gathered.append(ori_kv[blk_id, intra, 0])
            else:
                cmp_slot = raw - WIN
                if cmp_slot >= cmp_valid:
                    continue
                blk_id = int(cmp_block_table[b, cmp_slot // BLOCK_SIZE].item())
                intra = cmp_slot % BLOCK_SIZE
                gathered.append(cmp_kv[blk_id, intra, 0])

        if not gathered:
            continue

        kv_b = torch.stack(gathered, dim=0)
        q_t = q[t]

        block_mi = []
        block_li = []
        block_oi = []
        for tile_start in range(0, kv_b.shape[0], ATTN_K_TILE):
            kv_tile = kv_b[tile_start:tile_start + ATTN_K_TILE]
            scores = (q_t @ kv_tile.T) * SOFTMAX_SCALE
            mi = scores.max(dim=-1, keepdim=True).values
            exp_scores = torch.exp(scores - mi).to(torch.bfloat16).float()
            li = exp_scores.sum(dim=-1, keepdim=True)
            oi = exp_scores @ kv_tile.to(torch.bfloat16).float()
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
    o_r = o_r.to(torch.bfloat16).float()
    o_r_q = o_r.flatten(2).view(T, O_GROUPS * O_LORA)
    o_r_i8, o_r_scale = _int8_quant_per_row(o_r_q)
    acc = o_r_i8.to(torch.int32) @ wo_b_i8.to(torch.int32).T
    out = acc.float() * o_r_scale * wo_b_scale.unsqueeze(0)

    tensors["attn_out"][:] = out.to(torch.bfloat16)


def build_tensor_specs(
    compress_ratio: int = DEFAULT_COMPRESS_RATIO,
    causal_regression_fixture: bool = False,
):
    """Build deterministic demo tensors for the merged standalone harness."""
    import torch
    from golden import TensorSpec

    cmp_valid = get_standalone_cmp_valid(compress_ratio)
    sparse_k = WIN + cmp_valid

    def seeded_uniform(shape, seed):
        """Create a deterministic centered uniform tensor for repeatable tests."""
        generator = torch.Generator()
        generator.manual_seed(seed)
        return torch.rand(*shape, generator=generator) - 0.5

    def init_q():
        """Initialize the query tensor used by the decode attention stage."""
        q = seeded_uniform((T, H, HEAD_DIM), 1)
        if causal_regression_fixture:
            q[0].fill_(1.0)
        return q

    def init_ori_kv():
        """Initialize the sliding-window KV cache pages."""
        kv = seeded_uniform((ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM), 2)
        if causal_regression_fixture:
            kv[0, WIN - 1, 0].fill_(8.0)
        return kv

    def init_cmp_kv():
        """Initialize the compressed-cache KV pages."""
        return seeded_uniform((CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM), 3)

    def init_attn_sink():
        """Initialize the per-head sink logits to zero."""
        return torch.zeros(H)

    def init_ori_block_table():
        """Build the demo block table for the sliding-window cache pages."""
        tbl = torch.full((B, ORI_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(ORI_MAX_BLOCKS):
                tbl[b, j] = b * ORI_MAX_BLOCKS + j
        return tbl

    def init_cmp_block_table():
        """Build the demo block table for the compressed-cache pages."""
        tbl = torch.full((B, CMP_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(CMP_MAX_BLOCKS):
                tbl[b, j] = b * CMP_MAX_BLOCKS + j
        return tbl

    def init_cmp_sparse_indices():
        """Build the sparse index list with a full window prefix and padded compressed tail."""
        win_part = torch.arange(WIN, dtype=torch.int32).unsqueeze(0).expand(T, -1)
        cmp_part = torch.full((T, IDX_TOPK), -1, dtype=torch.int32)
        cmp_part[:, :cmp_valid] = (torch.arange(cmp_valid, dtype=torch.int32) + WIN).unsqueeze(0).expand(T, -1)
        indices = torch.cat([win_part, cmp_part], dim=-1).contiguous()
        if causal_regression_fixture:
            indices[0, WIN - 1] = WIN - 1
        return indices

    def init_seqused_kv():
        """Expose the demo sequence-used length that matches the chosen ratio mode."""
        return torch.full((B,), sparse_k, dtype=torch.int32)

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

    def init_odd_select_local():
        """Build the chunk-local selector that extracts odd rope lanes from interleaved inputs."""
        matrix = torch.zeros((ROPE_INTERLEAVE_TILE, ROPE_TILE))
        for i in range(ROPE_TILE):
            matrix[2 * i + 1, i] = 1
        return matrix

    def init_even_select_local():
        """Build the chunk-local selector that extracts even rope lanes from interleaved inputs."""
        matrix = torch.zeros((ROPE_INTERLEAVE_TILE, ROPE_TILE))
        for i in range(ROPE_TILE):
            matrix[2 * i, i] = 1
        return matrix

    def init_wo_a():
        """Initialize the grouped first-stage output-projection weights."""
        return seeded_uniform((O_GROUPS, O_LORA, O_GROUP_IN), 4) / (O_GROUP_IN ** 0.5)

    wo_b_bf16 = (seeded_uniform((D, O_GROUPS * O_LORA), 5) / ((O_GROUPS * O_LORA) ** 0.5)).to(torch.bfloat16)
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
        TensorSpec("ori_block_table", [B, ORI_MAX_BLOCKS], torch.int32, init_value=init_ori_block_table),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [B, CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("cmp_sparse_indices", [T, TOPK], torch.int32, init_value=init_cmp_sparse_indices),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("seqused_kv", [B], torch.int32, init_value=init_seqused_kv),
        TensorSpec("freqs_cos", [T, ROPE_DIM], torch.bfloat16, init_value=init_cos),
        TensorSpec("freqs_sin", [T, ROPE_DIM], torch.bfloat16, init_value=init_sin),
        TensorSpec("even_select_local", [ROPE_INTERLEAVE_TILE, ROPE_TILE], torch.bfloat16, init_value=init_even_select_local),
        TensorSpec("odd_select_local", [ROPE_INTERLEAVE_TILE, ROPE_TILE], torch.bfloat16, init_value=init_odd_select_local),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=init_wo_b),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=init_wo_b_scale),
        TensorSpec("attn_out", [T, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compress-ratio", type=int, default=DEFAULT_COMPRESS_RATIO,
                        choices=list(SUPPORTED_COMPRESS_RATIOS))
    parser.add_argument("--causal-regression-fixture", action="store_true", default=False,
                        help="Amplify the S=2 future-window-slot regression; use with --compress-ratio 0.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    args = parser.parse_args()

    result = run_jit(
        fn=sparse_attn_test,
        specs=build_tensor_specs(args.compress_ratio, args.causal_regression_fixture),
        golden_fn=golden_sparse_attn,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
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
