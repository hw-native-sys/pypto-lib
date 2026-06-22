# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 token-major prefill sparse attention with grouped output projection.

Raw-index contract for cmp_sparse_indices: -1 invalid; [0, WIN) ring KV;
[WIN, WIN + T) current-suffix overlay; [WIN + T, ...) compressed KV. cmp_sparse_lens[t]
is the usable prefix length; --compress-ratio {0,4,128} is a standalone fixture knob only.
"""

import pypto.language as pl

from config import BLOCK_SIZE, FLASH as M, FP32_NEG_INF, INT8_AMAX_EPS, INT8_SCALE_MAX, PREFILL_BATCH, PREFILL_SEQ

# Prefill target shape. T is fixed at 128.
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S

# Model config.
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
ROPE_HALF = HALF_ROPE
NOPE_DIM = M.nope_head_dim
IDX_TOPK = M.index_topk
WIN = M.sliding_window
TOPK = WIN + IDX_TOPK
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

# Cache shapes.
SUPPORTED_COMPRESS_RATIOS = (0, 4, 128)
DEFAULT_COMPRESS_RATIO = 4
PREFILL_MAX_COMPRESSED = max(1, min(IDX_TOPK, WIN + WIN // 2))
PREFILL_SPARSE_TOPK = min(TOPK, min(WIN, S) + PREFILL_MAX_COMPRESSED)
ORI_MAX_BLOCKS = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
ORI_BLOCK_NUM = B * ORI_MAX_BLOCKS
CMP_MAX_BLOCKS = max(1, (PREFILL_MAX_COMPRESSED + BLOCK_SIZE - 1) // BLOCK_SIZE)
CMP_BLOCK_NUM = B * CMP_MAX_BLOCKS

# Kernel tiling (mirrors decode sparse-attn).
HEAD_TILE = 16                       # head-tile granularity for storage / merge
QK_M_TILE = 32                       # head rows cube-batched per QK/PV matmul
GATHER_TOKEN_TILE = 4
BIAS_TOKEN_TILE = 16
QUANT_TOKEN_TILE = 8
ROPE_OUT_TOK_TILE = T // 2
ROPE_TILE = 16
ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE
A_K_TILE = 128
A_N_TILE = 128
A_T_TILE = 32                        # token tile for the proj_a/proj_b vec post-process
B_K_TILE = 256
B_N_TILE = 128
B_T_TILE = 32
QUANT_TILE = 512
QUANT_K_TILE = O_GROUPS * O_LORA // 2
# Sparse K split into <=3 merge blocks of PREFILL_ATTN_TILE rows.
PREFILL_ATTN_TILE = 128
PREFILL_ATTN_BLOCKS = (PREFILL_SPARSE_TOPK + PREFILL_ATTN_TILE - 1) // PREFILL_ATTN_TILE
PREFILL_SPARSE_PAD = PREFILL_ATTN_BLOCKS * PREFILL_ATTN_TILE
# Columns of the padded sparse window that carry a real cmp_sparse_indices entry
# (the rest of the [TOPK, PREFILL_SPARSE_PAD) tail, if any, is always masked).
SPARSE_BIAS_COLS = min(TOPK, PREFILL_SPARSE_PAD)

@pl.jit.inline
def prefill_sparse_attn(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[ORI_MAX_BLOCKS], pl.INT32],
    kv_overlay: pl.Tensor[[T, HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[CMP_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[T], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    # Half-width unsigned inverse-RoPE tables for the gather-free split-half rotate:
    #   rope_cos_half = [c0,c1,...,c_{HALF-1}]   rope_sin_half = [s0,s1,...]  (one per freq)
    # The rope segment is split-half ([x_lo | x_hi] = dims [0:HALF | HALF:ROPE_DIM]),
    # so the rotation partner of lane k is lane k+HALF -- a contiguous slice, not a
    # j^1 swap gather. FP32 so the rope loop reads them straight into its FP32 rotate.
    rope_cos_half: pl.Tensor[[T, HALF_ROPE], pl.FP32],
    rope_sin_half: pl.Tensor[[T, HALF_ROPE], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    """Token-major sparse attention with current-suffix KV overlay: gather the
    sliding-window / overlay / compressed KV rows, then run causal sparse attention,
    inverse RoPE, and the grouped INT8 output projection -- all in one scope set,
    mirroring decode_sparse_attn.

    Raw index contract: -1 invalid; [0, WIN) ring KV; [WIN, WIN + T) overlay row;
    [WIN + T, ...) compressed slot. HCA/CSA use all three sources; SWA passes dummy
    compressed tensors with compressed raw indices unreachable. cmp_sparse_lens is the
    usable prefix per row (pass TOPK when the rows are already -1 padded, e.g. CSA)."""
    # Gather KV per token: each (token, block) of PREFILL_ATTN_TILE slots is staged into one
    # UB tile (scattered 1-row loads on MTE2, invalid slots stay zero) then flushed with a
    # single wide MTE3 store. cmp_sparse_lens clamps the usable prefix.
    ori_kv_flat = pl.reshape(ori_kv, [ORI_MAX_BLOCKS * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [CMP_MAX_BLOCKS * BLOCK_SIZE, HEAD_DIM])
    sparse_kv = pl.create_tensor([T * PREFILL_SPARSE_PAD, HEAD_DIM], dtype=pl.BF16)
    for gather_block in pl.spmd(((T + GATHER_TOKEN_TILE - 1) // GATHER_TOKEN_TILE) * PREFILL_ATTN_BLOCKS, name_hint="gather_kv"):
        gather_token_block = gather_block // PREFILL_ATTN_BLOCKS
        gather_sb = gather_block - gather_token_block * PREFILL_ATTN_BLOCKS
        gather_t0 = gather_token_block * GATHER_TOKEN_TILE
        gather_k0 = gather_sb * PREFILL_ATTN_TILE
        for gather_dt in pl.range(GATHER_TOKEN_TILE):
            gather_t = gather_t0 + gather_dt
            if gather_t < T:
                block_base = gather_t * PREFILL_SPARSE_PAD + gather_k0
                stage = pl.full([PREFILL_ATTN_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                if gather_t < num_tokens:
                    gather_len = pl.read(cmp_sparse_lens, [gather_t])
                    gather_len_eff = pl.cast(0, pl.INT32)
                    if gather_len > 0:
                        gather_len_eff = gather_len
                    for gather_ki in pl.range(PREFILL_ATTN_TILE):
                        gather_k = gather_k0 + gather_ki
                        gather_raw = pl.cast(-1, pl.INT32)
                        if gather_k < TOPK:
                            if gather_k < gather_len_eff:
                                gather_raw = pl.read(cmp_sparse_indices, [gather_t, gather_k])
                        if gather_raw >= 0:
                            if gather_raw < WIN:
                                blk_slot = gather_raw // BLOCK_SIZE
                                blk = pl.cast(pl.read(ori_block_table, [blk_slot]), pl.INDEX)
                                src = blk * BLOCK_SIZE + (gather_raw - blk_slot * BLOCK_SIZE)
                                stage[gather_ki:gather_ki + 1, :] = ori_kv_flat[src:src + 1, :]
                            elif gather_raw < WIN + T:
                                ov = pl.cast(gather_raw - WIN, pl.INDEX)
                                stage[gather_ki:gather_ki + 1, :] = kv_overlay[ov:ov + 1, :]
                            else:
                                cmp_slot = gather_raw - (WIN + T)
                                blk_slot = cmp_slot // BLOCK_SIZE
                                blk = pl.cast(pl.read(cmp_block_table, [blk_slot]), pl.INDEX)
                                src = blk * BLOCK_SIZE + (cmp_slot - blk_slot * BLOCK_SIZE)
                                stage[gather_ki:gather_ki + 1, :] = cmp_kv_flat[src:src + 1, :]
                sparse_kv[block_base:block_base + PREFILL_ATTN_TILE, :] = stage

    # Additive softmax bias: 0 for valid slots, FP32_NEG_INF for padding, so the QK softmax
    # masks invalid slots without rescanning validity per head. A slot is valid when its raw
    # index is >= 0 AND its column lies inside the per-token cmp_sparse_lens prefix; the
    # [TOPK, PREFILL_SPARSE_PAD) tail (no raw index exists) is always masked.
    cmp_sparse_lens_2d = pl.reshape(cmp_sparse_lens, [T, 1])
    sparse_bias = pl.create_tensor([T, PREFILL_SPARSE_PAD], dtype=pl.FP32)
    for bias_blk in pl.spmd(T // BIAS_TOKEN_TILE, name_hint="build_bias"):
        bias_t0 = bias_blk * BIAS_TOKEN_TILE
        bias_idx = pl.cast(cmp_sparse_indices[bias_t0:bias_t0 + BIAS_TOKEN_TILE, 0:SPARSE_BIAS_COLS], target_type=pl.FP32)
        bias_raw_flag = pl.minimum(pl.maximum(pl.add(bias_idx, 1.0), 0.0), 1.0)
        bias_col = pl.col_expand(
            pl.full([BIAS_TOKEN_TILE, SPARSE_BIAS_COLS], dtype=pl.FP32, value=0.0),
            pl.cast(pl.arange(0, [1, SPARSE_BIAS_COLS], dtype=pl.INT32), target_type=pl.FP32))
        bias_len = pl.cast(cmp_sparse_lens_2d[bias_t0:bias_t0 + BIAS_TOKEN_TILE, 0:1], target_type=pl.FP32)
        bias_len_flag = pl.minimum(pl.maximum(pl.neg(pl.row_expand_sub(bias_col, bias_len)), 0.0), 1.0)
        bias_valid = pl.mul(bias_raw_flag, bias_len_flag)
        sparse_bias[bias_t0:bias_t0 + BIAS_TOKEN_TILE, 0:SPARSE_BIAS_COLS] = pl.mul(pl.sub(bias_valid, 1.0), -FP32_NEG_INF)
        if PREFILL_SPARSE_PAD > SPARSE_BIAS_COLS:
            sparse_bias[bias_t0:bias_t0 + BIAS_TOKEN_TILE, SPARSE_BIAS_COLS:PREFILL_SPARSE_PAD] = pl.full(
                [BIAS_TOKEN_TILE, PREFILL_SPARSE_PAD - SPARSE_BIAS_COLS], dtype=pl.FP32, value=FP32_NEG_INF)

    # Block OUTER (one KV/bias load per block), head-batch INNER (QK_M_TILE rows per matmul,
    # sliced to HEAD_TILE stores). qk_kv_k/qk_kv_v are two views so QK (b_trans) and PV don't
    # collide (#1532). Invalid blocks carry a -inf bias and die via beta == 0 in the merge.
    # sparse_blk_* are the per-(token, head-tile, block) softmax stats for that merge.
    blk_rows = T * (H // HEAD_TILE) * PREFILL_ATTN_BLOCKS * HEAD_TILE
    sparse_blk_mi = pl.create_tensor([blk_rows, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([blk_rows, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([blk_rows, HEAD_DIM], dtype=pl.FP32)
    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    for qk_t in pl.spmd(T, name_hint="qk_pv"):
        if qk_t < num_tokens:
            qk_kv_base = qk_t * PREFILL_SPARSE_PAD
            qk_token_base = qk_t * (H // HEAD_TILE) * PREFILL_ATTN_BLOCKS * HEAD_TILE
            for qk_sb in pl.range(PREFILL_ATTN_BLOCKS):
                qk_s0 = qk_kv_base + qk_sb * PREFILL_ATTN_TILE
                qk_kv_k = sparse_kv[qk_s0:qk_s0 + PREFILL_ATTN_TILE, :]
                qk_kv_v = sparse_kv[qk_s0:qk_s0 + PREFILL_ATTN_TILE, :]
                qk_bias_row = sparse_bias[qk_t:qk_t + 1, qk_sb * PREFILL_ATTN_TILE:qk_sb * PREFILL_ATTN_TILE + PREFILL_ATTN_TILE]
                for qk_hb in pl.pipeline(H // QK_M_TILE, stage=2):
                    qk_head_row = qk_t * H + qk_hb * QK_M_TILE
                    qk_q_tile = q_flat[qk_head_row:qk_head_row + QK_M_TILE, :]
                    qk_raw = pl.matmul(qk_q_tile, qk_kv_k, b_trans=True, out_dtype=pl.FP32)
                    zero_tile = pl.full([QK_M_TILE, PREFILL_ATTN_TILE], dtype=pl.FP32, value=0.0)
                    qk_scores = pl.add(pl.mul(qk_raw, SOFTMAX_SCALE), pl.col_expand(zero_tile, qk_bias_row))
                    qk_mi = pl.row_max(qk_scores)
                    qk_exp = pl.exp(pl.row_expand_sub(qk_scores, qk_mi))
                    # li sums the FP32 exp; only the PV matmul uses the BF16 cast.
                    qk_li = pl.row_sum(qk_exp)
                    qk_exp_bf16 = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
                    qk_oi = pl.matmul(qk_exp_bf16, qk_kv_v, out_dtype=pl.FP32)
                    for qk_sub in pl.unroll(QK_M_TILE // HEAD_TILE):
                        qk_h_idx = qk_hb * (QK_M_TILE // HEAD_TILE) + qk_sub
                        qk_r0 = qk_sub * HEAD_TILE
                        qk_row = qk_token_base + qk_h_idx * PREFILL_ATTN_BLOCKS * HEAD_TILE + qk_sb * HEAD_TILE
                        sparse_blk_mi[qk_row:qk_row + HEAD_TILE, :] = qk_mi[qk_r0:qk_r0 + HEAD_TILE, :]
                        sparse_blk_li[qk_row:qk_row + HEAD_TILE, :] = qk_li[qk_r0:qk_r0 + HEAD_TILE, :]
                        sparse_blk_oi[qk_row:qk_row + HEAD_TILE, :] = qk_oi[qk_r0:qk_r0 + HEAD_TILE, :]

    # Online-softmax merge across blocks, sink-norm, then pack NOPE into o_packed and the
    # FP32 rope slice into attn_rope_stage (full precision for the inverse rotation). Padding
    # tokens (t >= num_tokens) write zeros.
    attn_rope_stage = pl.create_tensor([T * H, ROPE_DIM], dtype=pl.FP32)
    o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)
    for m_t in pl.spmd(T, name_hint="merge_norm"):
        m_token_base = m_t * (H // HEAD_TILE) * PREFILL_ATTN_BLOCKS * HEAD_TILE
        for m_h_idx in pl.range(H // HEAD_TILE):
            m_h0 = m_h_idx * HEAD_TILE
            m_rope_row = m_t * H + m_h0
            if m_t < num_tokens:
                m_blk_base = m_token_base + m_h_idx * PREFILL_ATTN_BLOCKS * HEAD_TILE
                m_mi = sparse_blk_mi[m_blk_base:m_blk_base + HEAD_TILE, :]
                m_li = sparse_blk_li[m_blk_base:m_blk_base + HEAD_TILE, :]
                m_oi = sparse_blk_oi[m_blk_base:m_blk_base + HEAD_TILE, :]
                for m_sb in pl.range(1, PREFILL_ATTN_BLOCKS):
                    m_row = m_blk_base + m_sb * HEAD_TILE
                    cur_mi = sparse_blk_mi[m_row:m_row + HEAD_TILE, :]
                    cur_li = sparse_blk_li[m_row:m_row + HEAD_TILE, :]
                    cur_oi = sparse_blk_oi[m_row:m_row + HEAD_TILE, :]
                    mi_new = pl.maximum(m_mi, cur_mi)
                    alpha = pl.exp(pl.sub(m_mi, mi_new))
                    beta = pl.exp(pl.sub(cur_mi, mi_new))
                    m_li = pl.add(pl.mul(alpha, m_li), pl.mul(beta, cur_li))
                    m_oi = pl.add(pl.row_expand_mul(m_oi, alpha), pl.row_expand_mul(cur_oi, beta))
                    m_mi = mi_new
                sink_bias = pl.reshape(attn_sink[m_h0:m_h0 + HEAD_TILE], [HEAD_TILE, 1])
                sink_tile = pl.add(pl.sub(m_mi, m_mi), sink_bias)
                denom = pl.add(m_li, pl.exp(pl.sub(sink_tile, m_mi)))
                n_full = pl.row_expand_div(m_oi, denom)[0:HEAD_TILE, :]
                n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")
            else:
                n_full = pl.full([HEAD_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                n_bf16 = pl.full([HEAD_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)

            attn_rope_stage[m_rope_row:m_rope_row + HEAD_TILE, :] = n_full[0:HEAD_TILE, NOPE_DIM:HEAD_DIM]
            for n_hi in pl.range(HEAD_TILE):
                gh = m_h0 + n_hi
                g = gh // HEADS_PER_GROUP
                pack_row = g * T + m_t
                col = (gh - g * HEADS_PER_GROUP) * HEAD_DIM
                o_packed[pack_row:pack_row + 1, col:col + NOPE_DIM] = n_bf16[n_hi:n_hi + 1, 0:NOPE_DIM]

    # Gather-free conjugate (inverse) split-half RoPE. The rope segment is laid out
    # split-half ([x_lo | x_hi] = dims [0:HALF_ROPE | HALF_ROPE:ROPE_DIM]), so the
    # rotation partner of lane k is lane k+HALF_ROPE -- a contiguous slice, never a
    # j^1 swap gather. For k in [0, HALF_ROPE):
    #   out_lo[k] = x_lo[k]*cos[k] + x_hi[k]*sin[k]
    #   out_hi[k] = x_hi[k]*cos[k] - x_lo[k]*sin[k]
    # cos/sin are half-width (one value per frequency); no in-kernel index build.
    attn_rope_stage_3d = pl.reshape(attn_rope_stage, [T, H, ROPE_DIM])
    for rp_idx in pl.spmd((H // 4) * (T // ROPE_OUT_TOK_TILE), name_hint="rope"):
        rp_hg = rp_idx // (T // ROPE_OUT_TOK_TILE)
        rp_tt = rp_idx - rp_hg * (T // ROPE_OUT_TOK_TILE)
        rp_t0 = rp_tt * ROPE_OUT_TOK_TILE
        # Head-invariant half-width tables, hoisted once per task.
        r_cos = rope_cos_half[rp_t0 : rp_t0 + ROPE_OUT_TOK_TILE, 0 : HALF_ROPE]
        r_sin = rope_sin_half[rp_t0 : rp_t0 + ROPE_OUT_TOK_TILE, 0 : HALF_ROPE]
        for rp_hl in pl.range(0, 4):
            rp_gh = rp_hg * 4 + rp_hl
            rp_g = rp_gh // HEADS_PER_GROUP
            rp_hh = rp_gh - rp_g * HEADS_PER_GROUP
            rp_col = rp_hh * HEAD_DIM + NOPE_DIM
            rp_o0 = rp_g * T + rp_t0
            # This head's full ROPE_DIM rope rows for this token tile (stride H).
            # attn_rope_stage is already FP32, so read it directly -- no cast.
            r_tile = pl.reshape(
                attn_rope_stage_3d[rp_t0 : rp_t0 + ROPE_OUT_TOK_TILE, rp_gh : rp_gh + 1, 0 : ROPE_DIM], [ROPE_OUT_TOK_TILE, ROPE_DIM])
            r_lo = r_tile[0 : ROPE_OUT_TOK_TILE, 0 : HALF_ROPE]              # x_lo (contiguous, no gather)
            r_hi = r_tile[0 : ROPE_OUT_TOK_TILE, HALF_ROPE : ROPE_DIM]       # x_hi (contiguous, no gather)
            out_lo = pl.add(pl.mul(r_lo, r_cos), pl.mul(r_hi, r_sin))        # x_lo*c + x_hi*s
            out_hi = pl.sub(pl.mul(r_hi, r_cos), pl.mul(r_lo, r_sin))        # x_hi*c - x_lo*s
            # Two contiguous BF16-rounded stores into o_packed's rope columns
            # (golden also rounds inverse-RoPE to bf16).
            o_packed[rp_o0 : rp_o0 + ROPE_OUT_TOK_TILE, rp_col : rp_col + HALF_ROPE] = pl.cast(out_lo, target_type=pl.BF16, mode="rint")
            o_packed[rp_o0 : rp_o0 + ROPE_OUT_TOK_TILE, rp_col + HALF_ROPE : rp_col + ROPE_DIM] = pl.cast(out_hi, target_type=pl.BF16, mode="rint")

    # Grouped BF16 projection o_packed @ wo_a^T -> o_r, with per-row amax for the INT8
    # quant. Flattened (group, n-block) spmd; the BF16 store + amax vec post-process is
    # T-tiled by A_T_TILE to bound UB.
    o_r = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_amax_parts = pl.create_tensor([O_GROUPS * (O_LORA // A_N_TILE), T], dtype=pl.FP32)
    for proj_a_block in pl.spmd(O_GROUPS * (O_LORA // A_N_TILE), name_hint="proj_a"):
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
            o_r[tb:tb + A_T_TILE, out_col_g + n0:out_col_g + n0 + A_N_TILE] = acc_t
            acc_t_abs = pl.maximum(acc_t, pl.neg(acc_t))
            acc_t_amax = pl.reshape(pl.row_max(acc_t_abs), [1, A_T_TILE])
            o_r_amax_parts[amax_part_row:amax_part_row + 1, tb:tb + A_T_TILE] = acc_t_amax

    # Per-row symmetric INT8 quant of o_r, K-dim as a second parallel axis.
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
            or_q_f32 = o_r[quant_t0:quant_t0 + QUANT_TOKEN_TILE, k1:k1 + QUANT_TILE]
            or_q_scaled = pl.row_expand_mul(or_q_f32, or_sq_col)
            or_q_i32 = pl.cast(or_q_scaled, target_type=pl.INT32, mode="rint")
            or_q_half = pl.cast(or_q_i32, target_type=pl.FP16, mode="round")
            o_r_i8[quant_t0:quant_t0 + QUANT_TOKEN_TILE, k1:k1 + QUANT_TILE] = pl.cast(or_q_half, target_type=pl.INT8, mode="trunc")

    # INT8 output projection o_r_i8 @ wo_b^T, then dequant to BF16. UP_DOWN split over
    # the output-channel spmd; dequant vec post-process T-tiled by B_T_TILE.
    for nb in pl.spmd(D // B_N_TILE, name_hint="proj_b", optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
        n0 = nb * B_N_TILE
        xb0_chunk = o_r_i8[0:T, 0:B_K_TILE]
        wb0_chunk = wo_b[n0:n0 + B_N_TILE, 0:B_K_TILE]
        acc_b = pl.matmul(xb0_chunk, wb0_chunk, b_trans=True, out_dtype=pl.INT32)
        for kb in pl.pipeline(1, (O_GROUPS * O_LORA) // B_K_TILE, stage=2):
            k0 = kb * B_K_TILE
            xb_k_chunk = o_r_i8[0:T, k0:k0 + B_K_TILE]
            wb_k_chunk = wo_b[n0:n0 + B_N_TILE, k0:k0 + B_K_TILE]
            acc_b = pl.matmul_acc(acc_b, xb_k_chunk, wb_k_chunk, b_trans=True)

        acc_b_2d = pl.reshape(acc_b, [T, B_N_TILE])
        wb_scale_chunk = pl.reshape(wo_b_scale[n0:n0 + B_N_TILE], [1, B_N_TILE])
        for b_tb in pl.range(0, T, B_T_TILE):
            acc_b_t = acc_b_2d[b_tb:b_tb + B_T_TILE, 0:B_N_TILE]
            b_scale_t = o_r_scale_dq[b_tb:b_tb + B_T_TILE, 0:1]
            attn_t = pl.cast(acc_b_t, target_type=pl.FP32, mode="none")
            attn_t = pl.col_expand_mul(pl.row_expand_mul(attn_t, b_scale_t), wb_scale_chunk)
            attn_out[b_tb:b_tb + B_T_TILE, n0:n0 + B_N_TILE] = pl.cast(attn_t, target_type=pl.BF16, mode="rint")

    return attn_out

@pl.jit
def prefill_sparse_attn_test(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[ORI_MAX_BLOCKS], pl.INT32],
    kv_overlay: pl.Tensor[[T, HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[CMP_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[T], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    rope_cos_half: pl.Tensor[[T, HALF_ROPE], pl.FP32],
    rope_sin_half: pl.Tensor[[T, HALF_ROPE], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    return prefill_sparse_attn(
        q,
        ori_kv,
        ori_block_table,
        kv_overlay,
        cmp_kv,
        cmp_block_table,
        cmp_sparse_indices,
        cmp_sparse_lens,
        attn_sink,
        num_tokens,
        rope_cos_half,
        rope_sin_half,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
    )

def _quant_w_per_channel(w):
    """Per-output-channel INT8 quant on the last axis."""
    import torch

    amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.unsqueeze(-1)
    w_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()

def _int8_quant_per_row(x):
    """Per-row INT8 symmetric quant matching the W8A8C16 activation path."""
    import torch

    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)

def golden_prefill_sparse_attn(tensors):
    """Self-contained torch reference for the unified overlay sparse-attn entry."""
    import torch

    num_tokens = int(tensors["num_tokens"])
    q = tensors["q"].float()
    ori_kv = tensors["ori_kv"].float()
    kv_overlay = tensors["kv_overlay"].float()
    cmp_kv = tensors["cmp_kv"].float()
    ori_block_table = tensors["ori_block_table"]
    cmp_block_table = tensors["cmp_block_table"]
    cmp_sparse_indices = tensors["cmp_sparse_indices"]
    cmp_sparse_lens = tensors["cmp_sparse_lens"]
    attn_sink = tensors["attn_sink"].float()
    cos = tensors["rope_cos_half"].float()
    sin = tensors["rope_sin_half"].float()
    wo_a = tensors["wo_a"].float()
    wo_b_i8 = tensors["wo_b"]
    wo_b_scale = tensors["wo_b_scale"].float()

    o = torch.zeros(T, H, HEAD_DIM)
    for t in range(num_tokens):
        gathered = []
        sparse_len = max(0, min(int(cmp_sparse_lens[t].item()), PREFILL_SPARSE_PAD, TOPK))
        for raw_i in cmp_sparse_indices[t, :sparse_len].tolist():
            raw = int(raw_i)
            if raw < 0:
                continue
            if raw < WIN:
                block_id = int(ori_block_table[raw // BLOCK_SIZE].item())
                intra = raw % BLOCK_SIZE
                gathered.append(ori_kv[block_id, intra, 0])
            elif raw < WIN + T:
                overlay_t = raw - WIN
                if 0 <= overlay_t < num_tokens:
                    gathered.append(kv_overlay[overlay_t])
            else:
                cmp_slot = raw - (WIN + T)
                if cmp_slot < 0 or cmp_slot >= CMP_MAX_BLOCKS * BLOCK_SIZE:
                    continue
                block_id = int(cmp_block_table[cmp_slot // BLOCK_SIZE].item())
                intra = cmp_slot % BLOCK_SIZE
                gathered.append(cmp_kv[block_id, intra, 0])

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

    # Split-half inverse RoPE: lo = dims [:HALF_ROPE], hi = dims [HALF_ROPE:].
    rope_seg = o[..., NOPE_DIM:]
    x_lo = rope_seg[..., :HALF_ROPE]
    x_hi = rope_seg[..., HALF_ROPE:]
    cos_h = cos.unsqueeze(1)  # [T, 1, HALF_ROPE] broadcast over heads
    sin_h = sin.unsqueeze(1)
    inv_lo = (x_lo * cos_h + x_hi * sin_h).to(torch.bfloat16).float()
    inv_hi = (x_hi * cos_h - x_lo * sin_h).to(torch.bfloat16).float()
    o_rope = torch.cat([inv_lo, inv_hi], dim=-1)
    o = torch.cat([o[..., :NOPE_DIM], o_rope], dim=-1).to(torch.bfloat16)

    o_model = o.float().view(T, O_GROUPS, O_GROUP_IN)
    o_r = torch.einsum("tgd,grd->tgr", o_model, wo_a)
    o_r_q = o_r.flatten(1).view(T, O_GROUPS * O_LORA)
    o_r_i8, o_r_scale = _int8_quant_per_row(o_r_q)
    acc = o_r_i8.to(torch.int32) @ wo_b_i8.to(torch.int32).T
    out = acc.float() * o_r_scale * wo_b_scale.unsqueeze(0)
    tensors["attn_out"][:] = out.to(torch.bfloat16)

def get_prefill_cmp_valid(compress_ratio: int) -> int:
    """Map standalone ratio modes to visible compressed-cache length."""
    if compress_ratio == 0:
        return 0
    if compress_ratio in (4, 128):
        return min(IDX_TOPK, S // compress_ratio, CMP_MAX_BLOCKS * BLOCK_SIZE)
    raise ValueError(f"Unsupported compress_ratio={compress_ratio}; expected one of {SUPPORTED_COMPRESS_RATIOS}")

def build_tensor_specs(compress_ratio: int = DEFAULT_COMPRESS_RATIO):
    import torch
    from golden import ScalarSpec, TensorSpec

    num_tokens = T
    cmp_valid = get_prefill_cmp_valid(compress_ratio)
    # Half-width unsigned cos/sin (one value per frequency), rounded to BF16 then back to
    # FP32 so the FP32 kernel input holds exactly the bf16 values (bit-exact with the reference).
    angles = torch.arange(T * HALF_ROPE).reshape(T, HALF_ROPE) * 1e-3
    rope_cos_half_tbl = torch.cos(angles).to(torch.bfloat16).float()
    rope_sin_half_tbl = torch.sin(angles).to(torch.bfloat16).float()

    def init_q():
        return ((torch.rand(T, H, HEAD_DIM) - 0.5) * 0.05).to(torch.bfloat16)
    def init_ori_kv():
        return ((torch.rand(ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM) - 0.5) * 0.05).to(torch.bfloat16)
    def init_ori_block_table():
        table = torch.zeros(ORI_MAX_BLOCKS, dtype=torch.int32)
        for blk in range(ORI_MAX_BLOCKS):
            table[blk] = blk
        return table
    def init_kv_overlay():
        return ((torch.rand(T, HEAD_DIM) - 0.5) * 0.05).to(torch.bfloat16)
    def init_cmp_kv():
        return ((torch.rand(CMP_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM) - 0.5) * 0.05).to(torch.bfloat16)
    def init_cmp_block_table():
        table = torch.zeros(CMP_MAX_BLOCKS, dtype=torch.int32)
        for blk in range(CMP_MAX_BLOCKS):
            table[blk] = blk
        return table
    def init_cmp_sparse_indices():
        idx = torch.full((T, TOPK), -1, dtype=torch.int32)
        for t in range(num_tokens):
            window = torch.arange(t + 1, dtype=torch.int32) + WIN
            cursor = min(window.numel(), PREFILL_SPARSE_PAD)
            idx[t, :cursor] = window[:cursor]
            if compress_ratio:
                comp_count = min(cmp_valid, (t + 1) // compress_ratio)
                comp_count = min(comp_count, PREFILL_SPARSE_PAD - cursor)
                if comp_count > 0:
                    comp = torch.arange(comp_count, dtype=torch.int32) + WIN + T
                    idx[t, cursor : cursor + comp_count] = comp
        return idx
    def init_cmp_sparse_lens():
        idx = init_cmp_sparse_indices()
        lens = torch.zeros(T, dtype=torch.int32)
        for t in range(num_tokens):
            valid = (idx[t] >= 0).nonzero()
            if valid.numel():
                lens[t] = int(valid[-1].item()) + 1
        return lens
    def init_attn_sink():
        return torch.zeros(H)
    def init_rope_cos_half():
        return rope_cos_half_tbl.clone()
    def init_rope_sin_half():
        return rope_sin_half_tbl.clone()
    def init_wo_a():
        return ((torch.rand(O_GROUPS, O_LORA, O_GROUP_IN) - 0.5) * O_GROUP_IN ** -0.5).to(torch.bfloat16)
    def init_wo_b():
        return ((torch.rand(D, O_GROUPS * O_LORA) - 0.5) * (O_GROUPS * O_LORA) ** -0.5).to(torch.bfloat16)

    wo_b_i8, wo_b_scale = _quant_w_per_channel(init_wo_b())

    return [
        TensorSpec("q", [T, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", [ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("ori_block_table", [ORI_MAX_BLOCKS], torch.int32, init_value=init_ori_block_table),
        TensorSpec("kv_overlay", [T, HEAD_DIM], torch.bfloat16, init_value=init_kv_overlay),
        TensorSpec("cmp_kv", [CMP_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("cmp_sparse_indices", [T, TOPK], torch.int32, init_value=init_cmp_sparse_indices),
        TensorSpec("cmp_sparse_lens", [T], torch.int32, init_value=init_cmp_sparse_lens),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        TensorSpec("rope_cos_half", [T, HALF_ROPE], torch.float32, init_value=init_rope_cos_half),
        TensorSpec("rope_sin_half", [T, HALF_ROPE], torch.float32, init_value=init_rope_sin_half),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("attn_out", [T, D], torch.bfloat16, is_output=True),
    ]

if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--compress-ratio", type=int, default=DEFAULT_COMPRESS_RATIO,
                        choices=list(SUPPORTED_COMPRESS_RATIOS))
    parser.add_argument("--enable-l2-swimlane", nargs="?", const=4, default=0, type=int)
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    args = parser.parse_args()

    result = run_jit(
        fn=prefill_sparse_attn_test,
        specs=build_tensor_specs(args.compress_ratio),
        golden_fn=golden_prefill_sparse_attn,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
            enable_pmu=args.enable_pmu,
        ),
        rtol=1e-3,
        atol=1e-3,
        compile_only=args.compile_only,
        compare_fn={"attn_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128)},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
