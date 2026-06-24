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
TOPK_FULL = WIN + IDX_TOPK           # full sparse-K width (window + indexer topk)
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

# kernel-local
SUPPORTED_COMPRESS_RATIOS = (0, 4, 128)
DEFAULT_COMPRESS_RATIO = 128
ORI_MAX_BLOCKS = 1  # paged-KV pool: ori (sliding-window) blocks per batch
ORI_BLOCK_NUM = B * ORI_MAX_BLOCKS
CMP_MAX_BLOCKS = 8   # paged-KV pool: compressed blocks per batch
CMP_BLOCK_NUM = B * CMP_MAX_BLOCKS

# tiling
VALID_TOKEN_TILE = 16
GATHER_FILL_TILE = 128
ROPE_OUT_TOK_TILE = 64
H_TILE = 16
# qk_pv cube-batch tile (M for the QK/PV matmuls). Batching QK_M_TILE head rows
# per matmul extracts the shared KV tile L1->L0 once per QK_M_TILE/H_TILE
# head-tiles (2x reuse at 32) instead of per H_TILE head-tile, then slices the
# [QK_M_TILE, ...] result back into H_TILE-row stores so the sparse_blk_* layout
# and merge_norm stay unchanged. 32 keeps the [32,128] softmax inside the 192KB
# Vec budget without a cross-core split. 64 is infeasible without further work
# (its [64,128] softmax and co-resident QK+PV L0C accumulators overflow Vec/L0C).
QK_M_TILE = 32
ATTN_K_TILE = 128
ROPE_TILE = 16
ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE
A_T_TILE = 32
A_K_TILE = 128
A_N_TILE = 128
B_T_TILE = 32
B_K_TILE = 256
B_N_TILE = 128
QUANT_TILE = 512
QUANT_TOKEN_TILE = 8
QUANT_K_TILE = O_GROUPS * O_LORA // 2
NEG_INF = -1.0e20

assert T % VALID_TOKEN_TILE == 0
assert T % 2 == 0
assert H % 4 == 0
assert QK_M_TILE % H_TILE == 0
assert H % QK_M_TILE == 0
assert T % QUANT_TOKEN_TILE == 0
assert H % O_GROUPS == 0
assert (O_GROUPS * O_LORA) % B_K_TILE == 0
assert QUANT_K_TILE % QUANT_TILE == 0


def get_standalone_cmp_valid(compress_ratio: int) -> int:
    """Map demo compress-ratio modes to the valid compressed-cache tail length."""
    if compress_ratio == 0:
        return 0
    if compress_ratio == 4:
        return IDX_TOPK
    if compress_ratio == 128:
        return MAX_SEQ_LEN // compress_ratio
    raise ValueError(f"Unsupported compress_ratio={compress_ratio}; expected one of {SUPPORTED_COMPRESS_RATIOS}")


# HCA sparse-K width: window plus the deterministic ratio-128 compressed tail.
TOPK = WIN + min(MAX_SEQ_LEN // 128, IDX_TOPK)
# Floor to 2: a single sparse-K block miscompiles in pypto (S-stride cross-token
# output mixup); a 2-block build with an all-invalid 2nd block is bit-exact.
SPARSE_BLOCKS = max(2, (TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE)
PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE
assert WIN <= TOPK <= TOPK_FULL, f"TOPK ({TOPK}) must be in [WIN={WIN}, TOPK_FULL={TOPK_FULL}]"
assert PADDED_TOPK % GATHER_FILL_TILE == 0, \
    f"PADDED_TOPK ({PADDED_TOPK}) must be divisible by GATHER_FILL_TILE ({GATHER_FILL_TILE})"  # gather_kv bulk-zero


@pl.jit.inline
def sparse_attn_hca(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    mtp_kv_overlay: pl.Tensor[[T, HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, TOPK], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[T, D], pl.BF16],
):
    """Run sparse decode attention, inverse RoPE, and grouped output projection."""
    # Gather the sliding-window + current-MTP overlay + compressed-cache rows
    # into a per-token packed KV list. Raw index contract:
    #   -1              invalid
    #   [0, WIN)        historical ring/window KV
    #   [WIN, WIN + S)  current MTP overlay KV for this batch
    #   [WIN + S, ...)  compressed KV slots
    ori_kv_flat = pl.reshape(ori_kv, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    sparse_kv = pl.create_tensor([T * PADDED_TOPK, HEAD_DIM], dtype=pl.BF16)
    sparse_bias = pl.create_tensor([T, PADDED_TOPK], dtype=pl.FP32)

    # Additive softmax bias (0 valid / NEG_INF invalid) that qk_pv adds onto the
    # scaled scores, so invalid lanes exp to ~0 with no per-block mask multiply.
    for v_blk in pl.spmd(T // VALID_TOKEN_TILE, name_hint="build_valid"):
        v_t0 = v_blk * VALID_TOKEN_TILE
        v_idx_f = pl.cast(cmp_sparse_indices[v_t0 : v_t0 + VALID_TOKEN_TILE, 0 : TOPK], target_type=pl.FP32)
        v_valid_f = pl.minimum(pl.maximum(pl.add(v_idx_f, 1.0), 0.0), 1.0)
        sparse_bias[v_t0 : v_t0 + VALID_TOKEN_TILE, 0 : TOPK] = pl.mul(pl.sub(v_valid_f, 1.0), -NEG_INF)
        if PADDED_TOPK > TOPK:
            sparse_bias[v_t0 : v_t0 + VALID_TOKEN_TILE, TOPK : PADDED_TOPK] = pl.full(
                [VALID_TOKEN_TILE, PADDED_TOPK - TOPK], dtype=pl.FP32, value=NEG_INF)

    for g_t in pl.spmd(T, name_hint="gather_kv"):
        g_b = g_t // S
        g_kv_base = g_t * PADDED_TOPK
        # No-op self-copy: marks ori_kv add_inout so an in-place cache writeback
        # gets a WAR edge against this read (pypto-lib#481).
        g_self_touch = ori_kv_flat[g_t : g_t + 1, 0 : HEAD_DIM]
        ori_kv_flat[g_t : g_t + 1, 0 : HEAD_DIM] = g_self_touch

        # Bulk-zero the token's whole packed-KV region up front via wide MTE
        # stores, so invalid (-1) slots and the padding tail default to zero;
        # valid slots below overwrite their rows. Finite (zero) invalid KV rows
        # plus the NEG_INF softmax bias let qk_pv skip a per-block mask multiply.
        zero_fill = pl.full([GATHER_FILL_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
        for g_f in pl.range(0, PADDED_TOPK, GATHER_FILL_TILE):
            g_fill_row = g_kv_base + g_f
            sparse_kv[g_fill_row : g_fill_row + GATHER_FILL_TILE, 0 : HEAD_DIM] = zero_fill

        # Slot layout is fixed (window_topk in [0, WIN), compress_topk in
        # [WIN, TOPK)), so split by range to drop the per-row 4-way branch. The
        # window page base is invariant (WIN == BLOCK_SIZE, ORI_MAX_BLOCKS == 1).
        g_ori_base = pl.cast(pl.read(ori_block_table, [g_b, 0]), pl.INDEX) * BLOCK_SIZE
        g_overlay_base = g_b * S

        # Window slots: ring KV, or the MTP overlay when raw in [WIN, WIN + S).
        # Invalid (raw < 0) slots keep the bulk-zero fill. pl.pipeline overlaps
        # block k+1's load with block k's store.
        for g_w in pl.pipeline(WIN, stage=4):
            g_raw = pl.read(cmp_sparse_indices, [g_t, g_w])
            if g_raw >= 0:
                g_dst_row = g_kv_base + g_w
                if g_raw < WIN:
                    g_src_row = g_ori_base + g_raw
                    sparse_kv[g_dst_row : g_dst_row + 1, 0 : HEAD_DIM] = ori_kv_flat[g_src_row : g_src_row + 1, 0 : HEAD_DIM]
                else:
                    g_overlay_row = g_overlay_base + (g_raw - WIN)
                    sparse_kv[g_dst_row : g_dst_row + 1, 0 : HEAD_DIM] = mtp_kv_overlay[g_overlay_row : g_overlay_row + 1, 0 : HEAD_DIM]

        # Compressed slots [WIN, TOPK): paged compressed cache; -1 keeps the fill.
        # Guarded so the SWA (TOPK == WIN) specialization omits the loop entirely.
        # Each GATHER_FILL_TILE block is staged in a UB tile then flushed with one
        # wide MTE3 store: gives each scattered load its own tile row (no
        # buffer-reuse WAR, so loads stream on MTE2 instead of ping-ponging with
        # MTE3 per row) and coalesces the many 1-row stores into one block store.
        if TOPK > WIN:
            for g_cb in pl.range(WIN, PADDED_TOPK, GATHER_FILL_TILE):
                g_stage = pl.full([GATHER_FILL_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                for g_i in pl.range(GATHER_FILL_TILE):
                    g_c = g_cb + g_i
                    if g_c < TOPK:
                        g_raw = pl.read(cmp_sparse_indices, [g_t, g_c])
                        if g_raw >= 0:
                            g_slot = g_raw - (WIN + S)
                            g_blk = pl.cast(pl.read(cmp_block_table, [g_b, g_slot // BLOCK_SIZE]), pl.INDEX)
                            g_src_row = g_blk * BLOCK_SIZE + g_slot % BLOCK_SIZE
                            g_stage[g_i : g_i + 1, 0 : HEAD_DIM] = cmp_kv_flat[g_src_row : g_src_row + 1, 0 : HEAD_DIM]
                g_dst_blk = g_kv_base + g_cb
                sparse_kv[g_dst_blk : g_dst_blk + GATHER_FILL_TILE, 0 : HEAD_DIM] = g_stage

    # qk_pv writes per-tile (mi, li, oi) to GM; merge_norm reads them back. Not
    # fused on a2a3: the PV output (Acc) -> online rescale (Vec) needs an
    # unsupported tmov, and a [H_TILE, HEAD_DIM] carry overflows the Vec buffer.
    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    attn_rope_stage = pl.create_tensor([T * H, ROPE_DIM], dtype=pl.FP32)
    o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)
    sparse_blk_mi = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, HEAD_DIM], dtype=pl.FP32)

    for qk_t in pl.spmd(T, name_hint="qk_pv"):
        qk_kv_base = qk_t * PADDED_TOPK
        qk_token_base = qk_t * (H // H_TILE) * SPARSE_BLOCKS * H_TILE
        # Sparse-block OUTER / head-tile INNER: the KV tile and bias depend only
        # on (token, sparse-block), so hoisting them above the head-tile loop lets
        # one KV/bias load serve all head-tiles instead of re-loading per head.
        for qk_sb in pl.range(SPARSE_BLOCKS):
            qk_s0 = qk_sb * ATTN_K_TILE
            qk_kv_k = sparse_kv[qk_kv_base + qk_s0 : qk_kv_base + qk_s0 + ATTN_K_TILE, 0 : HEAD_DIM]
            qk_kv_v = sparse_kv[qk_kv_base + qk_s0 : qk_kv_base + qk_s0 + ATTN_K_TILE, 0 : HEAD_DIM]
            qk_bias_row = sparse_bias[qk_t : qk_t + 1, qk_s0 : qk_s0 + ATTN_K_TILE]

            # Cube-batch QK_M_TILE head rows per QK/PV matmul so the shared KV
            # tile is extracted L1->L0 once per QK_M_TILE/H_TILE head-tiles
            # (2x reuse at QK_M_TILE=32) instead of per head-tile. The
            # [QK_M_TILE, ...] softmax result is sliced back into H_TILE-row
            # stores at the SAME offsets as the per-head-tile path
            # (qk_h_idx == qk_hb * (QK_M_TILE // H_TILE) + qk_sub), so the
            # sparse_blk_* layout and merge_norm are bit-identical.
            for qk_hb in pl.pipeline(H // QK_M_TILE, stage=2):
                qk_h0 = qk_hb * QK_M_TILE
                qk_head_row = qk_t * H + qk_h0
                qk_q_tile = q_flat[qk_head_row : qk_head_row + QK_M_TILE, 0 : HEAD_DIM]
                qk_raw = pl.matmul(qk_q_tile, qk_kv_k, b_trans=True, out_dtype=pl.FP32)
                qk_scaled = pl.mul(qk_raw, SOFTMAX_SCALE)
                qk_scores = pl.add(qk_scaled, pl.col_expand(pl.full([QK_M_TILE, ATTN_K_TILE], dtype=pl.FP32, value=0.0), qk_bias_row))
                qk_mi = pl.row_max(qk_scores)
                # Invalid lanes (NEG_INF bias, zero kv rows) exp to ~0; all-invalid
                # blocks die in the merge alpha/beta -- no mask multiply needed.
                qk_exp = pl.exp(pl.row_expand_sub(qk_scores, qk_mi))
                qk_li = pl.row_sum(qk_exp)
                qk_exp_bf16 = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
                qk_oi = pl.matmul(qk_exp_bf16, qk_kv_v, out_dtype=pl.FP32)
                for qk_sub in pl.unroll(QK_M_TILE // H_TILE):
                    qk_h_idx = qk_hb * (QK_M_TILE // H_TILE) + qk_sub
                    qk_r0 = qk_sub * H_TILE
                    qk_blk_base = qk_token_base + qk_h_idx * SPARSE_BLOCKS * H_TILE
                    qk_row = qk_blk_base + qk_sb * H_TILE
                    sparse_blk_mi[qk_row : qk_row + H_TILE, 0 : 1] = qk_mi[qk_r0 : qk_r0 + H_TILE, 0 : 1]
                    sparse_blk_li[qk_row : qk_row + H_TILE, 0 : 1] = qk_li[qk_r0 : qk_r0 + H_TILE, 0 : 1]
                    sparse_blk_oi[qk_row : qk_row + H_TILE, 0 : HEAD_DIM] = qk_oi[qk_r0 : qk_r0 + H_TILE, 0 : HEAD_DIM]

    # Online-softmax merge across sparse-K tiles, then sink-norm.
    for m_t in pl.spmd(T, name_hint="merge_norm"):
        m_token_base = m_t * (H // H_TILE) * SPARSE_BLOCKS * H_TILE

        for m_h_idx in pl.range((H // H_TILE)):
            m_h0 = m_h_idx * H_TILE
            m_blk_base = m_token_base + m_h_idx * SPARSE_BLOCKS * H_TILE
            m_mi = sparse_blk_mi[m_blk_base : m_blk_base + H_TILE, 0 : 1]
            m_li = sparse_blk_li[m_blk_base : m_blk_base + H_TILE, 0 : 1]
            m_oi = sparse_blk_oi[m_blk_base : m_blk_base + H_TILE, 0 : HEAD_DIM]

            # Guarded so the SWA (SPARSE_BLOCKS == 1) specialization uses the
            # single block's stats directly instead of an empty merge loop.
            if SPARSE_BLOCKS > 1:
                for m_sb in pl.range(1, SPARSE_BLOCKS):
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
            n_full = pl.row_expand_div(m_oi, n_denom)[0 : H_TILE, 0 : HEAD_DIM]
            n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")
            n_rope_row = m_t * H + m_h0
            attn_rope_stage[n_rope_row : n_rope_row + H_TILE, 0 : ROPE_DIM] = n_full[0 : H_TILE, NOPE_DIM : HEAD_DIM]

            for n_hi in pl.range(H_TILE):
                n_gh = m_h0 + n_hi
                n_g = n_gh // HEADS_PER_GROUP
                n_hh = n_gh - n_g * HEADS_PER_GROUP
                n_pack_row = n_g * T + m_t
                n_col = n_hh * HEAD_DIM
                o_packed[n_pack_row : n_pack_row + 1, n_col : n_col + NOPE_DIM] = n_bf16[n_hi : n_hi + 1, 0 : NOPE_DIM]

    # Precompute the head-invariant interleaved cos and sign*sin once: they depend
    # only on (token, column), not head, so building them per head would repeat the
    # same dup-gather H times on the bottleneck Vec engine. sign is folded into sin
    # (multiply by +/-1). The conjugate (inverse) rotation is:
    #   out[j] = x[j]*cos_il[j] + x[j^1]*sign[j]*sin_il[j]
    rope_cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    for cp in pl.spmd(HALF_ROPE // ROPE_TILE, name_hint="rope_cs"):
        cp_r0 = cp * ROPE_TILE
        cp_c0 = 2 * cp_r0
        cs_col = pl.col_expand_mul(
            pl.full([T, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=1.0),
            pl.cast(pl.arange(0, [1, ROPE_INTERLEAVE_TILE], dtype=pl.INT32), target_type=pl.FP32))
        cs_dup_f = pl.cast(pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)                                      # j>>1
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))                                           # j%2
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))                                       # [+1,-1,...] (conjugate)
        cs_cos = pl.cast(freqs_cos[0:T, cp_r0 : cp_r0 + ROPE_TILE], target_type=pl.FP32)
        cs_sin = pl.cast(freqs_sin[0:T, cp_r0 : cp_r0 + ROPE_TILE], target_type=pl.FP32)
        rope_cos_il[0:T, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
        rope_sin_signed[0:T, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = pl.mul(
            pl.gather(cs_sin, dim=-1, index=cs_dup_idx), cs_sign)

    # Inverse RoPE fused with the rope-column pack: each task rotates its heads'
    # rope segments and stores them straight into o_packed's strided rope columns,
    # dropping a separate rope_pack stage and GM round-trip. cos_il / sign*sin come
    # from the pre-pass above; only swap_idx (j^1) is rebuilt per task.
    attn_rope_stage_3d = pl.reshape(attn_rope_stage, [T, H, ROPE_DIM])
    for rp_idx in pl.spmd((H // 4) * (T // ROPE_OUT_TOK_TILE), name_hint="rope"):
        rp_hg = rp_idx // (T // ROPE_OUT_TOK_TILE)
        rp_tt = rp_idx - rp_hg * (T // ROPE_OUT_TOK_TILE)
        rp_t0 = rp_tt * ROPE_OUT_TOK_TILE
        # Head-invariant swap index (j^1), built once and reused across the head
        # group -- the only per-head input is this head's strided rope slice.
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
            for r_r0 in pl.range(0, HALF_ROPE, ROPE_TILE):
                c0 = 2 * r_r0
                # This head's rope rows for this token tile (stride H).
                r_tile_fp32 = pl.reshape(
                    attn_rope_stage_3d[rp_t0 : rp_t0 + ROPE_OUT_TOK_TILE, rp_gh : rp_gh + 1, c0 : c0 + ROPE_INTERLEAVE_TILE],
                    [ROPE_OUT_TOK_TILE, ROPE_INTERLEAVE_TILE])
                r_cos_il = rope_cos_il[rp_t0 : rp_t0 + ROPE_OUT_TOK_TILE, c0 : c0 + ROPE_INTERLEAVE_TILE]
                r_sin_signed = rope_sin_signed[rp_t0 : rp_t0 + ROPE_OUT_TOK_TILE, c0 : c0 + ROPE_INTERLEAVE_TILE]
                r_swapped = pl.gather(r_tile_fp32, dim=-1, index=sp_swap_idx)
                r_rot = pl.add(pl.mul(r_tile_fp32, r_cos_il), pl.mul(r_swapped, r_sin_signed))
                # Store BF16-rounded rotated values straight into o_packed's rope
                # columns for this head (golden also rounds inverse-RoPE to bf16).
                r_rot = pl.cast(r_rot, target_type=pl.BF16, mode="rint")
                o_packed[rp_o0 : rp_o0 + ROPE_OUT_TOK_TILE, rp_col + c0 : rp_col + c0 + ROPE_INTERLEAVE_TILE] = r_rot

    # Grouped BF16 projection `o_packed @ wo_a^T` -> `o_r`. Vec post-process
    # (BF16 store + per-row amax) is T-tiled to keep the fused AIV side from
    # oversizing UB.
    o_r = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_amax_parts = pl.create_tensor([O_GROUPS * (O_LORA // A_N_TILE), T], dtype=pl.FP32)
    for proj_a_block in pl.spmd(O_GROUPS * (O_LORA // A_N_TILE), name_hint="proj_a"):
        # K-split BF16 matmul for one wo_a output tile, peel-first-iter form.
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
            or_q_f32 = o_r[quant_t0:quant_t0 + QUANT_TOKEN_TILE, k1:k1 + QUANT_TILE]
            or_q_scaled = pl.row_expand_mul(or_q_f32, or_sq_col)
            or_q_i32 = pl.cast(or_q_scaled, target_type=pl.INT32, mode="rint")
            or_q_half = pl.cast(or_q_i32, target_type=pl.FP16, mode="round")
            o_r_i8[quant_t0:quant_t0 + QUANT_TOKEN_TILE, k1:k1 + QUANT_TILE] = pl.cast(or_q_half, target_type=pl.INT8, mode="trunc")

    # INT8 projection `o_r_i8 @ wo_b^T`, then dequantize -> final BF16 output.
    for nb in pl.spmd(D // B_N_TILE, name_hint="proj_b"):
        # K-split INT8 GEMM + dequant in one scope; T-tiled vec post-process
        # to keep the fused AIV side from oversizing UB (same as proj_a).
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
    mtp_kv_overlay: pl.Tensor[[T, HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, TOPK], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    sparse_attn_hca(
        q,
        ori_kv,
        ori_block_table,
        mtp_kv_overlay,
        cmp_kv,
        cmp_block_table,
        cmp_sparse_indices,
        attn_sink,
        freqs_cos,
        freqs_sin,
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
    mtp_kv_overlay = tensors["mtp_kv_overlay"].float()
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

    # Per-query-token attention. cmp_sparse_indices is the authoritative
    # topk list: -1 invalid; raw < WIN selects ring cache;
    # WIN <= raw < WIN+S selects the current MTP overlay; raw >= WIN+S
    # selects the compressed-cache tail.
    for t in range(T):
        b = t // S
        kv_rows = []
        valid = []

        for raw in cmp_sparse_indices[t].tolist():
            if raw < 0:
                kv_rows.append(torch.zeros(HEAD_DIM, dtype=ori_kv.dtype))
                valid.append(False)
                continue
            if raw < WIN:
                blk_id = int(ori_block_table[b, raw // BLOCK_SIZE].item())
                intra = raw % BLOCK_SIZE
                kv_rows.append(ori_kv[blk_id, intra, 0])
                valid.append(True)
            elif raw < WIN + S:
                overlay_s = raw - WIN
                kv_rows.append(mtp_kv_overlay[b * S + overlay_s])
                valid.append(True)
            else:
                cmp_slot = raw - (WIN + S)
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
    o_r_q = o_r.flatten(2).view(T, O_GROUPS * O_LORA)
    o_r_i8, o_r_scale = _int8_quant_per_row(o_r_q)
    acc = o_r_i8.to(torch.int32) @ wo_b_i8.to(torch.int32).T
    out = acc.float() * o_r_scale * wo_b_scale.unsqueeze(0)

    tensors["attn_out"][:] = out.to(torch.bfloat16)

def build_tensor_specs(
    compress_ratio: int = DEFAULT_COMPRESS_RATIO,
    causal_regression_fixture: bool = False,
    short_window_fixture: bool = False,
    mixed_topk_fixture: bool = False,
    overlay_replacement_fixture: bool = False,
):
    """Build deterministic demo tensors for the merged standalone harness."""
    import torch
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
        return kv

    def init_cmp_kv():
        """Initialize the compressed-cache KV pages."""
        return torch.rand(CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM) - 0.5

    def init_mtp_kv_overlay():
        """Initialize the current decode-chunk overlay KV rows."""
        overlay = torch.rand(T, HEAD_DIM) - 0.5
        if overlay_replacement_fixture:
            overlay[:, :] = 0.0
            overlay[:, 0] = 4.0
        return overlay

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
        """Build the sparse index list with a full window prefix and padded compressed tail.

        The compressed tail width follows the active specialization (TOPK - WIN):
        the pruned build narrows it to `cmp_valid` columns, the full-blocks
        baseline keeps the IDX_TOPK-wide padded tail.
        """
        win_part = torch.arange(WIN, dtype=torch.int32).unsqueeze(0).expand(T, -1)
        cmp_width = TOPK - WIN
        cmp_part = torch.full((T, cmp_width), -1, dtype=torch.int32)
        cmp_part[:, :cmp_valid] = (torch.arange(cmp_valid, dtype=torch.int32) + WIN + S).unsqueeze(0).expand(T, -1)
        indices = torch.cat([win_part, cmp_part], dim=-1).contiguous()
        if short_window_fixture:
            indices[:, :] = -1
            indices[:, :17] = torch.arange(17, dtype=torch.int32).unsqueeze(0).expand(T, -1)
        if mixed_topk_fixture:
            indices[:, :] = -1
            indices[:, :17] = torch.arange(17, dtype=torch.int32).unsqueeze(0).expand(T, -1)
            mixed_cmp_valid = min(cmp_valid, IDX_TOPK)
            if mixed_cmp_valid:
                indices[:, WIN:WIN + mixed_cmp_valid] = (
                    torch.arange(mixed_cmp_valid, dtype=torch.int32) + WIN + S
                ).unsqueeze(0).expand(T, -1)
        if overlay_replacement_fixture:
            indices[:, :] = -1
            indices[:, :17] = torch.arange(17, dtype=torch.int32).unsqueeze(0).expand(T, -1)
            indices[:, 16] = WIN
        if causal_regression_fixture:
            indices[0, WIN - 1] = WIN - 1
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
        TensorSpec("ori_block_table", [B, ORI_MAX_BLOCKS], torch.int32, init_value=init_ori_block_table),
        TensorSpec("mtp_kv_overlay", [T, HEAD_DIM], torch.bfloat16, init_value=init_mtp_kv_overlay),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [B, CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("cmp_sparse_indices", [T, TOPK], torch.int32, init_value=init_cmp_sparse_indices),
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
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    # --compress-ratio only selects which compressed-tail data pattern to validate;
    # the pruned widths are covered by the swa/hca variant tests.
    parser.add_argument("--compress-ratio", type=int, default=DEFAULT_COMPRESS_RATIO,
                        choices=list(SUPPORTED_COMPRESS_RATIOS))
    parser.add_argument("--causal-regression-fixture", action="store_true", default=False,
                        help="Amplify the S=2 future-window-slot regression; use with --compress-ratio 0.")
    parser.add_argument("--short-window-fixture", action="store_true", default=False,
                        help="Use a short-window topk row with valid prefix + -1 padding.")
    parser.add_argument("--mixed-topk-fixture", action="store_true", default=False,
                        help="Use -1-padded window slots with valid compressed raw indices.")
    parser.add_argument("--overlay-replacement-fixture", action="store_true", default=False,
                        help="Place a compressed raw index inside the window prefix order.")
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--enable-dep-gen", action="store_true", default=False,
                        help="Capture PTO2 dependency edges (deps.json); the swimlane "
                             "converter draws fanout/fanin arrows from the sibling file.")
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    args = parser.parse_args()

    compress_ratio = args.compress_ratio
    print(f"compress_ratio={compress_ratio} "
          f"-> TOPK={TOPK} SPARSE_BLOCKS={SPARSE_BLOCKS} PADDED_TOPK={PADDED_TOPK}", flush=True)

    result = run_jit(
        fn=sparse_attn_test,
        specs=build_tensor_specs(
            compress_ratio,
            args.causal_regression_fixture,
            args.short_window_fixture,
            args.mixed_topk_fixture,
            args.overlay_replacement_fixture,
        ),
        golden_fn=golden_sparse_attn,
        golden_data=args.golden_data,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
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
