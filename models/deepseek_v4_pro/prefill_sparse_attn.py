# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 token-major prefill sparse attention with grouped output projection.

Sparse reads are split by cache source. ``swa_indices`` contains physical rows
in the original KV cache; ``cmp_indices`` contains compressed logical slots that
are lowered through ``cmp_block_table``. ``-1`` marks invalid entries.
"""

import pypto.language as pl

from config import (
    BLOCK_SIZE,
    ACTIVE as M,
    FP32_NEG_INF,
    PREFILL_BATCH,
    PREFILL_CMP_BLOCK_NUM,
    PREFILL_CMP_MAX_BLOCKS,
    PREFILL_ORI_BLOCK_NUM,
    PREFILL_ORI_MAX_BLOCKS,
    PREFILL_SEQ,
)
from prefill_mxfp8_o_proj import (
    MX_BLOCK_K,
    MX_K_TILE,
    MX_N_TILE,
    WO_A_SCALE_ROWS_PER_GROUP,
    WO_A_SCALE_ROWS,
    WO_B_SCALE_ROWS,
    prefill_mxfp8_o_proj,
)

# Dynamic physical cache-view dimensions used by sparse attention.
ORI_BLOCK_NUM_DYN = pl.dynamic("PREFILL_ORI_BLOCK_NUM_DYN")
SPARSE_CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_SPARSE_CMP_BLOCK_NUM_DYN")


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
ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
ORI_BLOCK_NUM = PREFILL_ORI_BLOCK_NUM
CMP_MAX_BLOCKS = PREFILL_CMP_MAX_BLOCKS
CMP_BLOCK_NUM = PREFILL_CMP_BLOCK_NUM

# Kernel tiling (mirrors decode sparse-attn).
HEAD_TILE = 16                       # head-tile granularity for storage / merge
QK_M_TILE = 32                       # head rows cube-batched per QK/PV matmul
QK_TOKEN_GROUP = 8                   # keep the mixed QK/PV task within one resident wave
GATHER_TOKEN_TILE = 4
BIAS_TOKEN_TILE = 16
ROPE_OUT_TOK_TILE = T // 2
ROPE_TILE = 16
ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE
# Sparse K split into <=3 merge blocks of PREFILL_ATTN_TILE rows.
PREFILL_ATTN_TILE = 128
PREFILL_ATTN_BLOCKS = (PREFILL_SPARSE_TOPK + PREFILL_ATTN_TILE - 1) // PREFILL_ATTN_TILE
PREFILL_SPARSE_PAD = PREFILL_ATTN_BLOCKS * PREFILL_ATTN_TILE
# Columns of the padded sparse window that carry real metadata entries.
SPARSE_BIAS_COLS = min(TOPK, PREFILL_SPARSE_PAD)
SPARSE_CMP_BIAS_COLS = max(0, SPARSE_BIAS_COLS - WIN)

@pl.jit.inline
def prefill_sparse_rope_tables(
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    rope_cos_il: pl.Tensor[[T, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T, ROPE_DIM], pl.FP32],
):
    """Materialize head-invariant interleaved RoPE tables in GM."""
    for cp in pl.spmd(ROPE_HALF // ROPE_TILE, name_hint="rope_cs"):
        cp_r0 = cp * ROPE_TILE
        cp_c0 = 2 * cp_r0
        cs_col = pl.col_expand_mul(
            pl.full([T, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=1.0),
            pl.cast(pl.arange(0, [1, ROPE_INTERLEAVE_TILE], dtype=pl.INT32), target_type=pl.FP32))
        cs_dup_f = pl.cast(pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)                                      # j>>1
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))                                           # j%2
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))                                       # [+1,-1,...]
        cs_cos = pl.cast(freqs_cos[0:T, cp_r0 : cp_r0 + ROPE_TILE], target_type=pl.FP32)
        cs_sin = pl.cast(freqs_sin[0:T, cp_r0 : cp_r0 + ROPE_TILE], target_type=pl.FP32)
        rope_cos_il[0:T, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
        rope_sin_signed[0:T, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = pl.mul(
            pl.gather(cs_sin, dim=-1, index=cs_dup_idx), cs_sign)


@pl.jit.inline
def prefill_sparse_attn_precomputed_rope(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[SPARSE_CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[CMP_MAX_BLOCKS], pl.INT32],
    cmp_indices: pl.Tensor[[T, IDX_TOPK], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    rope_cos_il: pl.Tensor[[T, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T, ROPE_DIM], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[WO_A_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[WO_B_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    """Gather cache-first sparse rows using RoPE tables already materialized in GM."""

    # Gather KV per token: each (token, block) of PREFILL_ATTN_TILE slots is staged into one
    # UB tile (scattered 1-row loads on MTE2, invalid slots stay zero) then flushed with a
    # single wide MTE3 store. Invalid slots are carried by -1 padding.
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    ori_cache_rows = ori_block_num * BLOCK_SIZE
    ori_kv_flat = pl.reshape(ori_kv, [ori_cache_rows, HEAD_DIM])
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    cmp_cache_rows = cmp_block_num * BLOCK_SIZE
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_cache_rows, HEAD_DIM])
    sparse_kv = pl.create_tensor([T * PREFILL_SPARSE_PAD, HEAD_DIM], dtype=pl.BF16)
    with pl.spmd(
        ((T + GATHER_TOKEN_TILE - 1) // GATHER_TOKEN_TILE) * PREFILL_ATTN_BLOCKS,
        name_hint="gather_kv",
    ) as _gather_tid:
        gather_block = pl.tile.get_block_idx()
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
                    for gather_ki in pl.range(PREFILL_ATTN_TILE):
                        gather_k = gather_k0 + gather_ki
                        gather_raw = pl.cast(-1, pl.INT32)
                        if gather_k < WIN:
                            gather_raw = pl.read(swa_indices, [gather_t, gather_k])
                            if gather_raw >= 0:
                                src = pl.cast(gather_raw, pl.INDEX)
                                stage[gather_ki:gather_ki + 1, :] = ori_kv_flat[src:src + 1, :]
                        else:
                            gather_cmp_k = gather_k - WIN
                            if gather_cmp_k < IDX_TOPK:
                                gather_raw = pl.read(cmp_indices, [gather_t, gather_cmp_k])
                                if gather_raw >= 0:
                                    cmp_slot = gather_raw
                                    blk_slot = cmp_slot // BLOCK_SIZE
                                    blk = pl.cast(pl.read(cmp_block_table, [blk_slot]), pl.INDEX)
                                    src = blk * BLOCK_SIZE + (cmp_slot - blk_slot * BLOCK_SIZE)
                                    stage[gather_ki:gather_ki + 1, :] = cmp_kv_flat[src:src + 1, :]
                sparse_kv[block_base:block_base + PREFILL_ATTN_TILE, :] = stage

    # Additive softmax bias: 0 for valid slots, FP32_NEG_INF for padding, so the QK softmax
    # masks invalid slots without rescanning validity per head. A slot is valid when its raw
    # index is >= 0; the [TOPK, PREFILL_SPARSE_PAD) tail is always masked.
    sparse_bias = pl.create_tensor([T, PREFILL_SPARSE_PAD], dtype=pl.FP32)
    with pl.spmd(T // BIAS_TOKEN_TILE, name_hint="build_bias") as _bias_tid:
        bias_blk = pl.tile.get_block_idx()
        bias_t0 = bias_blk * BIAS_TOKEN_TILE
        bias_win_idx = pl.cast(swa_indices[bias_t0:bias_t0 + BIAS_TOKEN_TILE, 0:WIN], target_type=pl.FP32)
        bias_win_raw_flag = pl.minimum(pl.maximum(pl.add(bias_win_idx, 1.0), 0.0), 1.0)
        sparse_bias[bias_t0:bias_t0 + BIAS_TOKEN_TILE, 0:WIN] = pl.mul(
            pl.sub(bias_win_raw_flag, 1.0), -FP32_NEG_INF)
        if SPARSE_CMP_BIAS_COLS > 0:
            bias_cmp_idx = pl.cast(
                cmp_indices[bias_t0:bias_t0 + BIAS_TOKEN_TILE, 0:SPARSE_CMP_BIAS_COLS],
                target_type=pl.FP32)
            bias_cmp_raw_flag = pl.minimum(pl.maximum(pl.add(bias_cmp_idx, 1.0), 0.0), 1.0)
            sparse_bias[bias_t0:bias_t0 + BIAS_TOKEN_TILE, WIN:SPARSE_BIAS_COLS] = pl.mul(
                pl.sub(bias_cmp_raw_flag, 1.0), -FP32_NEG_INF)
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
    for qk_group in pl.spmd(T // QK_TOKEN_GROUP, name_hint="qk_pv"):
        qk_t0 = qk_group * QK_TOKEN_GROUP
        for qk_dt in pl.range(QK_TOKEN_GROUP):
            qk_t = qk_t0 + qk_dt
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
                        # Broadcast-add the per-block bias directly (col_expand_add) instead of
                        # col_expand into a dead pl.full(0) base + a separate add (mirrors decode).
                        qk_scaled = pl.mul(qk_raw, SOFTMAX_SCALE)
                        qk_scores = pl.col_expand_add(qk_scaled, qk_bias_row)
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
    # The MXFP8 projection helper reads the packed NOPE columns through GM deps.
    with pl.spmd(T, name_hint="merge_norm") as _merge_tid:
        m_t = pl.tile.get_block_idx()
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

    # Inverse RoPE fused with the rope-column pack, against the tables built at the top of the
    # kernel (see the note there for why they are not built here).
    attn_rope_stage_3d = pl.reshape(attn_rope_stage, [T, H, ROPE_DIM])
    # The MXFP8 projection helper also reads these packed RoPE columns through GM deps.
    with pl.spmd((H // 4) * (T // ROPE_OUT_TOK_TILE), name_hint="rope") as _rope_tid:
        rp_idx = pl.tile.get_block_idx()
        rp_hg = rp_idx // (T // ROPE_OUT_TOK_TILE)
        rp_tt = rp_idx - rp_hg * (T // ROPE_OUT_TOK_TILE)
        rp_t0 = rp_tt * ROPE_OUT_TOK_TILE
        # Head-invariant swap index (j^1), built once and reused across the head group.
        sp_col = pl.col_expand_mul(
            pl.full([ROPE_OUT_TOK_TILE, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=1.0),
            pl.cast(pl.arange(0, [1, ROPE_INTERLEAVE_TILE], dtype=pl.INT32), target_type=pl.FP32))
        sp_dup_f = pl.cast(pl.cast(pl.mul(sp_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        sp_lane = pl.sub(sp_col, pl.mul(sp_dup_f, 2.0))                                           # j%2
        sp_swap_idx = pl.cast(pl.sub(pl.add(sp_col, 1.0), pl.mul(sp_lane, 2.0)), target_type=pl.INT32)  # j^1
        for rp_hl in pl.range(0, 4):
            rp_gh = rp_hg * 4 + rp_hl
            rp_g = rp_gh // HEADS_PER_GROUP
            rp_hh = rp_gh - rp_g * HEADS_PER_GROUP
            rp_col = rp_hh * HEAD_DIM + NOPE_DIM
            rp_o0 = rp_g * T + rp_t0
            for r_r0 in pl.range(0, ROPE_HALF, ROPE_TILE):
                c0 = 2 * r_r0
                r_tile_fp32 = pl.reshape(
                    attn_rope_stage_3d[rp_t0 : rp_t0 + ROPE_OUT_TOK_TILE, rp_gh : rp_gh + 1, c0 : c0 + ROPE_INTERLEAVE_TILE],
                    [ROPE_OUT_TOK_TILE, ROPE_INTERLEAVE_TILE])
                r_cos_il = rope_cos_il[rp_t0 : rp_t0 + ROPE_OUT_TOK_TILE, c0 : c0 + ROPE_INTERLEAVE_TILE]
                r_sin_signed = rope_sin_signed[rp_t0 : rp_t0 + ROPE_OUT_TOK_TILE, c0 : c0 + ROPE_INTERLEAVE_TILE]
                r_swapped = pl.gather(r_tile_fp32, dim=-1, index=sp_swap_idx)
                r_rot = pl.add(pl.mul(r_tile_fp32, r_cos_il), pl.mul(r_swapped, r_sin_signed))
                r_rot = pl.cast(r_rot, target_type=pl.BF16, mode="rint")
                o_packed[rp_o0 : rp_o0 + ROPE_OUT_TOK_TILE, rp_col + c0 : rp_col + c0 + ROPE_INTERLEAVE_TILE] = r_rot

    prefill_mxfp8_o_proj(
        o_packed,
        wo_a,
        wo_a_scale,
        wo_b,
        wo_b_scale,
        attn_out,
    )

    return attn_out


@pl.jit.inline
def prefill_sparse_attn(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[SPARSE_CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[CMP_MAX_BLOCKS], pl.INT32],
    cmp_indices: pl.Tensor[[T, IDX_TOPK], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[WO_A_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[WO_B_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    """Compatibility entry that materializes RoPE tables before sparse attention."""
    # Keep this phase at the front of the entry. Placing it next to the later
    # merge/rotate phases can trigger an on-device AIV residency timing fault.
    rope_cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    prefill_sparse_rope_tables(freqs_cos, freqs_sin, rope_cos_il, rope_sin_signed)
    return prefill_sparse_attn_precomputed_rope(
        q,
        ori_kv,
        swa_indices,
        cmp_kv,
        cmp_block_table,
        cmp_indices,
        attn_sink,
        num_tokens,
        rope_cos_il,
        rope_sin_signed,
        wo_a,
        wo_a_scale,
        wo_b,
        wo_b_scale,
        attn_out,
    )

@pl.jit
def prefill_sparse_attn_test(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[SPARSE_CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[CMP_MAX_BLOCKS], pl.INT32],
    cmp_indices: pl.Tensor[[T, IDX_TOPK], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[WO_A_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[WO_B_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    return prefill_sparse_attn(
        q,
        ori_kv,
        swa_indices,
        cmp_kv,
        cmp_block_table,
        cmp_indices,
        attn_sink,
        num_tokens,
        freqs_cos,
        freqs_sin,
        wo_a,
        wo_a_scale,
        wo_b,
        wo_b_scale,
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
    swa_indices = tensors["swa_indices"]
    cmp_indices = tensors["cmp_indices"]
    attn_sink = tensors["attn_sink"].float()
    cos = tensors["freqs_cos"].float()
    sin = tensors["freqs_sin"].float()
    from expert_shared import _dynamic_mxfp8_matmul, _unpack_b_scale_tiled

    wo_a = tensors["wo_a"]
    wo_a_scale_packed = tensors["wo_a_scale"]
    wo_b = tensors["wo_b"]
    wo_b_scale = _unpack_b_scale_tiled(
        tensors["wo_b_scale"],
        O_GROUPS * O_LORA,
        D,
        k_tile=MX_K_TILE,
        n_tile=MX_N_TILE,
    )

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
            if cmp_slot < 0 or cmp_slot >= CMP_MAX_BLOCKS * BLOCK_SIZE:
                continue
            block_id = int(cmp_block_table[cmp_slot // BLOCK_SIZE].item())
            intra = cmp_slot % BLOCK_SIZE
            if block_id >= 0:
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
    o_r_g = torch.empty(T, O_GROUPS, O_LORA, dtype=torch.float32)
    out = torch.zeros(T, D, dtype=torch.float32)
    for g in range(O_GROUPS):
        wa_scale0 = g * WO_A_SCALE_ROWS_PER_GROUP
        wa_scale = _unpack_b_scale_tiled(
            wo_a_scale_packed[
                wa_scale0 : wa_scale0 + WO_A_SCALE_ROWS_PER_GROUP
            ],
            O_GROUP_IN,
            O_LORA,
            k_tile=MX_K_TILE,
            n_tile=MX_N_TILE,
        )
        o_r_g[:, g] = _dynamic_mxfp8_matmul(o_model[:, g], wo_a[g], wa_scale)
        k0 = g * O_LORA
        out += _dynamic_mxfp8_matmul(
            o_r_g[:, g],
            wo_b[k0 : k0 + O_LORA],
            wo_b_scale[k0 // MX_BLOCK_K : (k0 + O_LORA) // MX_BLOCK_K],
        )
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
    from rope_tables import build_deepseek_v4_rope_tables, materialize_token_rope_tables

    num_tokens = T
    cmp_valid = get_prefill_cmp_valid(compress_ratio)
    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, compress_ratio, dtype=torch.bfloat16)
    shared_rope_cos, shared_rope_sin = materialize_token_rope_tables(
        shared_freqs_cos,
        shared_freqs_sin,
        torch.arange(T, dtype=torch.int32),
    )

    def init_q():
        return ((torch.rand(T, H, HEAD_DIM) - 0.5) * 0.05).to(torch.bfloat16)
    def init_ori_kv():
        return ((torch.rand(ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM) - 0.5) * 0.05).to(torch.bfloat16)
    def init_cmp_kv():
        return ((torch.rand(CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM) - 0.5) * 0.05).to(torch.bfloat16)
    def init_cmp_block_table():
        table = torch.zeros(CMP_MAX_BLOCKS, dtype=torch.int32)
        for blk in range(CMP_MAX_BLOCKS):
            table[blk] = blk
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
    def init_attn_sink():
        return torch.zeros(H)
    def init_freqs_cos():
        return shared_rope_cos.clone()
    def init_freqs_sin():
        return shared_rope_sin.clone()
    from expert_shared import _gen_mxfp8_weight_kn

    wo_a_values = []
    wo_a_scales = []
    for _ in range(O_GROUPS):
        wa_value, wa_scale = _gen_mxfp8_weight_kn(
            (O_GROUP_IN, O_LORA),
            dequant_std=0.25 / O_GROUP_IN ** 0.5,
            chan_cv=0.25,
            k_tile=MX_K_TILE,
            n_tile=MX_N_TILE,
        )
        wo_a_values.append(wa_value)
        wo_a_scales.append(wa_scale)
    wo_a_fp8 = torch.stack(wo_a_values)
    wo_a_scale = torch.cat(wo_a_scales, dim=0)
    wo_b_fp8, wo_b_scale = _gen_mxfp8_weight_kn(
        (O_GROUPS * O_LORA, D),
        dequant_std=0.25 / (O_GROUPS * O_LORA) ** 0.5,
        chan_cv=0.25,
        k_tile=MX_K_TILE,
        n_tile=MX_N_TILE,
    )

    def init_wo_a():
        return wo_a_fp8

    def init_wo_a_scale():
        return wo_a_scale

    return [
        TensorSpec("q", [T, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", [ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("swa_indices", [T, WIN], torch.int32, init_value=init_swa_indices),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("cmp_indices", [T, IDX_TOPK], torch.int32, init_value=init_cmp_indices),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        TensorSpec("freqs_cos", [T, ROPE_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [T, ROPE_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("wo_a", [O_GROUPS, O_GROUP_IN, O_LORA], torch.float8_e4m3fn, init_value=init_wo_a),
        TensorSpec("wo_a_scale", [WO_A_SCALE_ROWS, MX_N_TILE], torch.float8_e8m0fnu, init_value=init_wo_a_scale),
        TensorSpec("wo_b", [O_GROUPS * O_LORA, D], torch.float8_e4m3fn, init_value=lambda: wo_b_fp8),
        TensorSpec("wo_b_scale", [WO_B_SCALE_ROWS, MX_N_TILE], torch.float8_e8m0fnu, init_value=lambda: wo_b_scale),
        TensorSpec("attn_out", [T, D], torch.bfloat16, is_output=True),
    ]

if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a5", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--compress-ratio", type=int, default=DEFAULT_COMPRESS_RATIO,
                        choices=list(SUPPORTED_COMPRESS_RATIOS))
    parser.add_argument("--enable-l2-swimlane", nargs="?", const=4, default=0, type=int)
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=prefill_sparse_attn_test,
        specs=build_tensor_specs(args.compress_ratio),
        golden_fn=golden_prefill_sparse_attn,
        compile_cfg=dict(dump_passes=args.dump_passes),
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
