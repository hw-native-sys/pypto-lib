# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 HCA sparse attention with grouped output projection (decode).

Ratio-128 deterministic compressed tail plus the sliding window; no indexer.
The SWA and CSA variants live in sibling modules.
"""


import pypto.language as pl

from config import (
    FLASH as M,
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


# Dynamic shape variables.
ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")

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
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

COMPRESS_RATIO = 128
CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE // COMPRESS_RATIO
NEG_INF = -1.0e20

# paged KV cache
ORI_MAX_BLOCKS = KV_ORI_MAX_BLOCKS
ORI_BLOCK_NUM = DECODE_ORI_BLOCK_NUM
CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM

# tiling
VALID_TOKEN_TILE = 8
GATHER_SEGS = 4          # gather blocks per token; T*GATHER_SEGS co-resides with qproj_dequant
# Each segment carries BOTH a window slice and a compressed-tail slice, whose
# per-row costs are opposite (bulk-run window vs scattered per-row topk).
GATHER_RUN = 16          # window sub-tile probed for physical contiguity -> one bulk DMA
H_TILE = 16
QK_M_TILE = 32           # qk_pv M rows per QK/PV matmul; QK_M_TILE/H_TILE-way KV L1->L0 reuse
ATTN_K_TILE = 128
ROPE_TILE = 16
ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE
A_K_TILE = 256           # proj_a cube K frag
PROJ_A_MM_N_TILE = 128   # proj_a cube N frag
MM_T_TILE = 16
T_PAD = ((T + MM_T_TILE - 1) // MM_T_TILE) * MM_T_TILE
B_K_TILE = 256           # proj_b_mm cube K frag
PROJ_B_MM_N_TILE = 256   # proj_b_mm cube N frag; writes grouped INT32 partials
PROJ_B_ACT_N_TILE = 512  # proj_b_act vector N frag; keeps the O_GROUPS-way accumulate inside UB
QUANT_TOKEN_TILE = 8     # fused per-group amax+quant row tile
PROJ_B_D_TILE = 512      # proj_b_mm D chunk per task; its N frags loop inside the task
PROJ_B_ACT_T_TILE = 8    # proj_b_act inner token tile for the O_GROUPS-way INT32->FP32 accumulate
PROJ_B_ACT_TASK_T_TILE = 8   # proj_b_act token block per task

# Compressed-cache capacity: the ratio-128 layer has no indexer, so its compressed
# tail is the deterministic full compressed cache, one slot per COMPRESS_RATIO
# tokens. `index_topk` is the ratio-4 indexer's budget and does NOT bound this.
CMP_CAPACITY = MAX_SEQ_LEN // COMPRESS_RATIO
# Rounded up to a whole sparse block so TOPK needs no padding (PADDED_TOPK == TOPK).
CMP_TOPK = ((CMP_CAPACITY + ATTN_K_TILE - 1) // ATTN_K_TILE) * ATTN_K_TILE
# Longest context this build serves; past it the tail drops its NEWEST slots and
# leaves a hole between the compressed history and the window.
MAX_SUPPORTED_SEQ = CMP_TOPK * COMPRESS_RATIO
CMP_BLOCKS_PER_REQ = (CMP_TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
TOPK = WIN + CMP_TOPK    # cache-first window slots + the ratio-128 compressed tail
# Floor to 2: a single sparse-K block miscompiles in pypto (S-stride cross-token
# output mixup); a 2-block build with an all-invalid 2nd block is bit-exact.
SPARSE_BLOCKS = max(2, (TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE)
PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE
GATHER_WIN_ROWS = WIN // GATHER_SEGS
GATHER_CMP_ROWS = (PADDED_TOPK - WIN) // GATHER_SEGS

assert CMP_BLOCKS_PER_REQ <= CMP_MAX_BLOCKS, (
    f"compressed block table ({CMP_MAX_BLOCKS} blocks) must index the whole "
    f"{CMP_TOPK}-slot tail; MAX_SUPPORTED_SEQ={MAX_SUPPORTED_SEQ}")
assert B * CMP_BLOCKS_PER_REQ <= CMP_BLOCK_NUM, (
    f"compressed KV pool ({CMP_BLOCK_NUM} blocks) must hold B={B} requests x "
    f"{CMP_BLOCKS_PER_REQ} blocks; MAX_SUPPORTED_SEQ={MAX_SUPPORTED_SEQ}")
assert WIN == ATTN_K_TILE, f"HCA window tile requires WIN ({WIN}) == ATTN_K_TILE ({ATTN_K_TILE})"
assert BLOCK_SIZE % GATHER_RUN == 0, "a contiguous run must not straddle two paged blocks by construction"


@pl.jit.inline
def sparse_attn_hca(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
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
    # Gather the historical/current window + compressed-cache rows.
    # Compressed index contract:
    #   -1              invalid
    #   [0, ...)        compressed KV slots
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    ori_kv_flat = pl.reshape(ori_kv, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_block_num * CMP_STORAGE_BLOCK_SIZE, HEAD_DIM])
    sparse_bias = pl.create_tensor([T, PADDED_TOPK], dtype=pl.FP32)

    # Additive softmax bias (0 valid / NEG_INF invalid) that qk_pv adds onto the
    # scaled scores, so invalid lanes exp to ~0 with no per-block mask multiply.
    for v_blk in pl.spmd(T // VALID_TOKEN_TILE, name_hint="build_valid", allow_early_resolve=True):
        v_t0 = v_blk * VALID_TOKEN_TILE
        v_win_f = pl.cast(window_swa_indices[v_t0 : v_t0 + VALID_TOKEN_TILE, 0 : WIN], target_type=pl.FP32)
        v_idx_f = pl.cast(cmp_sparse_indices[v_t0 : v_t0 + VALID_TOKEN_TILE, 0 : CMP_TOPK], target_type=pl.FP32)
        v_win_valid = pl.minimum(pl.maximum(pl.add(v_win_f, 1.0), 0.0), 1.0)
        v_cmp_valid = pl.minimum(pl.maximum(pl.add(v_idx_f, 1.0), 0.0), 1.0)
        sparse_bias[v_t0 : v_t0 + VALID_TOKEN_TILE, 0 : WIN] = pl.mul(pl.sub(v_win_valid, 1.0), -NEG_INF)
        sparse_bias[v_t0 : v_t0 + VALID_TOKEN_TILE, WIN : TOPK] = pl.mul(pl.sub(v_cmp_valid, 1.0), -NEG_INF)
        if PADDED_TOPK > TOPK:
            sparse_bias[v_t0 : v_t0 + VALID_TOKEN_TILE, TOPK : PADDED_TOPK] = pl.full(
                [VALID_TOKEN_TILE, PADDED_TOPK - TOPK], dtype=pl.FP32, value=NEG_INF)

    # Sparse-K gather, hoisted out of qk_pv into its own grid, writing one token's
    # sparse-K rows into the contiguous hca_kv_flat buffer. Every block carries a
    # GATHER_WIN_ROWS slice of the window AND a GATHER_CMP_ROWS slice of the
    # compressed tail, so the cheap bulk runs and the costly scattered rows are
    # spread evenly. Invalid (-1) and padded lanes are zero-filled to match the
    # golden's zero rows; the NEG_INF bias then kills them in the softmax.
    hca_kv_flat = pl.create_tensor([T * PADDED_TOPK, HEAD_DIM], dtype=pl.BF16)
    with pl.spmd(T * GATHER_SEGS, name_hint="hca_gather_kv") as gather_tid:
        g_task = pl.tile.get_block_idx()
        g_t = g_task // GATHER_SEGS
        g_seg = g_task - g_t * GATHER_SEGS
        g_b = g_t // S
        g_row0 = g_t * PADDED_TOPK

        # Window slice: probe each sub-tile's first/last slot. Endpoints that are
        # GATHER_RUN-1 apart mean the whole run sits in one paged block.
        g_wk0 = g_seg * GATHER_WIN_ROWS
        for g_sub in pl.range(GATHER_WIN_ROWS // GATHER_RUN):
            g_sk0 = g_wk0 + g_sub * GATHER_RUN
            g_sdst = g_row0 + g_sk0
            g_first = pl.read(window_swa_indices, [g_t, g_sk0])
            g_last = pl.read(window_swa_indices, [g_t, g_sk0 + GATHER_RUN - 1])
            # A -1 slot anywhere in the run pins g_run_ok below the match value,
            # so an invalid or block-straddling run takes the per-row path.
            g_run_ok = (g_last - g_first) + pl.min(g_first, 0) * GATHER_RUN
            if g_run_ok == GATHER_RUN - 1:
                g_run_src = pl.cast(g_first, pl.INDEX)
                hca_kv_flat[g_sdst : g_sdst + GATHER_RUN, 0:HEAD_DIM] = ori_kv_flat[
                    g_run_src : g_run_src + GATHER_RUN, 0:HEAD_DIM
                ]
            else:
                for g_dr in pl.range(GATHER_RUN):
                    g_wdst = g_sdst + g_dr
                    g_win_slot_i32 = pl.read(window_swa_indices, [g_t, g_sk0 + g_dr])
                    if g_win_slot_i32 >= 0:
                        g_win_slot = pl.cast(g_win_slot_i32, pl.INDEX)
                        hca_kv_flat[g_wdst : g_wdst + 1, 0:HEAD_DIM] = ori_kv_flat[
                            g_win_slot : g_win_slot + 1, 0:HEAD_DIM
                        ]
                    else:
                        hca_kv_flat[g_wdst : g_wdst + 1, 0:HEAD_DIM] = pl.full(
                            [1, HEAD_DIM], dtype=pl.BF16, value=0.0)

        # Compressed slice: topk slots are scattered, so each row is its own
        # block-table lookup + copy.
        g_ck0 = g_seg * GATHER_CMP_ROWS
        g_cdst0 = g_row0 + WIN + g_ck0
        for g_dr in pl.range(GATHER_CMP_ROWS):
            g_dst = g_cdst0 + g_dr
            g_cmp_k = g_ck0 + g_dr
            if g_cmp_k < CMP_TOPK:
                g_ridx = pl.read(cmp_sparse_indices, [g_t, g_cmp_k])
                if g_ridx >= 0:
                    g_csrc = pl.cast(pl.read(cmp_block_table, [g_b, g_ridx]), pl.INDEX)
                    hca_kv_flat[g_dst : g_dst + 1, 0:HEAD_DIM] = cmp_kv_flat[g_csrc : g_csrc + 1, 0:HEAD_DIM]
                else:
                    hca_kv_flat[g_dst : g_dst + 1, 0:HEAD_DIM] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
            else:
                hca_kv_flat[g_dst : g_dst + 1, 0:HEAD_DIM] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)

    # qk_pv writes per-tile (mi, li, oi) to GM; merge_norm reads them back. Not
    # fused on a2a3: the PV output (Acc) -> online rescale (Vec) needs an
    # unsupported tmov, and a [H_TILE, HEAD_DIM] carry overflows the Vec buffer.
    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)
    sparse_blk_mi = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, HEAD_DIM], dtype=pl.FP32)

    with pl.spmd(T * SPARSE_BLOCKS, name_hint="qk_pv", deps=[gather_tid], allow_early_resolve=True) as qk_tid:
        qk_item = pl.tile.get_block_idx()
        qk_t = qk_item // SPARSE_BLOCKS
        qk_sb = qk_item - qk_t * SPARSE_BLOCKS
        qk_token_base = qk_t * (H // H_TILE) * SPARSE_BLOCKS * H_TILE
        # Sparse-block OUTER / head-tile INNER: both head-batches' QK (b_trans)
        # and PV consume the SAME pre-gathered KV tile.
        qk_s0 = qk_sb * ATTN_K_TILE
        qk_bias_row = sparse_bias[qk_t : qk_t + 1, qk_s0 : qk_s0 + ATTN_K_TILE]
        qk_base = qk_t * PADDED_TOPK + qk_s0
        qk_kv = hca_kv_flat[qk_base : qk_base + ATTN_K_TILE, 0:HEAD_DIM]

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
            qk_raw = pl.matmul(qk_q_tile, qk_kv, b_trans=True, out_dtype=pl.FP32)
            qk_scaled = pl.mul(qk_raw, SOFTMAX_SCALE)
            qk_scores = pl.add(qk_scaled, pl.col_expand(pl.full([QK_M_TILE, ATTN_K_TILE], dtype=pl.FP32, value=0.0), qk_bias_row))
            qk_mi = pl.row_max(qk_scores)
            # Invalid lanes (NEG_INF bias, zero kv rows) exp to ~0; all-invalid
            # blocks die in the merge alpha/beta -- no mask multiply needed.
            qk_exp = pl.exp(pl.row_expand_sub(qk_scores, qk_mi))
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

    # Precompute the head-invariant interleaved cos and sign*sin once: they depend
    # only on (token, column), not head, so building them per head would repeat the
    # same dup-gather H times on the bottleneck Vec engine. sign is folded into sin
    # (multiply by +/-1). The conjugate (inverse) rotation is:
    #   out[j] = x[j]*cos_il[j] + x[j^1]*sign[j]*sin_il[j]
    # Hoisted ABOVE merge_norm (which now fuses the rotation): independent of qk_pv,
    # so it overlaps it and is off merge_norm's critical path.
    rope_cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    # The j^1 lane-swap index for merge_norm's rotation gather is a pure constant
    # (no token/head dependence), so it is built once here instead of rebuilding
    # the same arange/cast chain on each of the T*(H//H_TILE) merge blocks. Shaped
    # [H_TILE, ROPE_DIM] because gather's index must match its source rows. It gets
    # its own single-task scope because rope_cs below is an spmd over rope column
    # tiles -- no single block there owns a column-invariant constant.
    rope_swap_idx = pl.create_tensor([H_TILE, ROPE_DIM], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_swap"):
        sw_col = pl.col_expand_mul(
            pl.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
            pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32))
        sw_dup_f = pl.cast(pl.cast(pl.mul(sw_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        sw_lane = pl.sub(sw_col, pl.mul(sw_dup_f, 2.0))                                           # j%2
        rope_swap_idx[0:H_TILE, 0:ROPE_DIM] = pl.cast(
            pl.sub(pl.add(sw_col, 1.0), pl.mul(sw_lane, 2.0)), target_type=pl.INT32)              # j^1

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

    # Online-softmax merge across sparse-K tiles, sink-norm, then fused inverse RoPE.
    # One spmd block per (token, head-tile) -- T*(H//H_TILE) blocks -- so the merge
    # fans out over that many AIVs instead of T blocks each running a serial head-tile
    # loop. The inverse-RoPE rotation + rope-column pack is fused in (was a separate
    # "rope" spmd reading an attn_rope_stage GM round-trip): the head-tile's fp32 rope
    # segment is rotated in UB and packed straight into o_packed's rope columns.
    # with-form spmd so the dispatch TaskId (merge_tid) can be an explicit dep of
    # the manual-scope proj_a tasks below (which read merge_norm's o_packed cols).
    with pl.spmd(T * (H // H_TILE), name_hint="merge_norm") as merge_tid:
        m_idx = pl.tile.get_block_idx()
        m_t = m_idx // (H // H_TILE)
        m_h_idx = m_idx - m_t * (H // H_TILE)
        m_h0 = m_h_idx * H_TILE
        m_blk_base = m_idx * SPARSE_BLOCKS * H_TILE
        m_mi = sparse_blk_mi[m_blk_base : m_blk_base + H_TILE, 0 : 1]
        m_li = sparse_blk_li[m_blk_base : m_blk_base + H_TILE, 0 : 1]
        m_oi = sparse_blk_oi[m_blk_base : m_blk_base + H_TILE, 0 : HEAD_DIM]

        # SPARSE_BLOCKS is max(2, ...) here, so the merge loop always runs -- no
        # SWA-style guard needed. Software-pipelined so each iteration's
        # sparse_blk_* loads overlap the previous iteration's rescale math.
        for m_sb in pl.pipeline(1, SPARSE_BLOCKS, stage=2):
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

        # Inverse RoPE on this head-tile's fp32 rope segment. cos_il / sign*sin are
        # head-invariant for token m_t, so col_expand them over the H_TILE head rows;
        # rope_swap_idx (j^1, prebuilt above) pairs the interleaved real/imag lanes.
        # Rounded to bf16 (golden also rounds inverse-RoPE to bf16) and packed into
        # o_packed's rope columns.
        m_rope = n_full[0 : H_TILE, NOPE_DIM : HEAD_DIM]
        m_cos_il = rope_cos_il[m_t : m_t + 1, 0 : ROPE_DIM]
        m_sin_signed = rope_sin_signed[m_t : m_t + 1, 0 : ROPE_DIM]
        m_swapped = pl.gather(m_rope, dim=-1, index=rope_swap_idx[0:H_TILE, 0:ROPE_DIM])
        m_rot = pl.add(pl.col_expand_mul(m_rope, m_cos_il), pl.col_expand_mul(m_swapped, m_sin_signed))
        n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode="rint")
        n_full_bf16 = pl.concat(n_bf16[:, : NOPE_DIM], n_rope_bf16)

        for n_hi in pl.unroll(H_TILE):
            n_pack_row = ((m_h0 + n_hi) // HEADS_PER_GROUP) * T + m_t
            n_col = ((m_h0 + n_hi) % HEADS_PER_GROUP) * HEAD_DIM
            # one HEAD_DIM-wide store per head row instead of two: concat the nope and
            # inverse-RoPE halves on chip so o_packed takes a single contiguous write.
            o_packed[n_pack_row : n_pack_row + 1, n_col : n_col + HEAD_DIM] = n_full_bf16[n_hi : n_hi + 1, :]

    # Back-to-back grouped output projection: proj_a[g] -> quant[g] -> proj_b[g]
    # pipelines per group, because the PER-GROUP amax keeps the quant reduction
    # inside one O_LORA group instead of barriering the whole row. manual_scope
    # suppresses auto-dep, so every edge is explicit: proj_a waits on merge_norm,
    # quant[g] on proj_a[g], proj_b[g] on quant[g]. proj_b_act combines the group
    # partials and is the consolidated attn_out writer.
    o_r_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_i8_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.INT8)
    # [G, T] so each group's per-row scale is a contiguous row.
    act_scale_dq = pl.create_tensor([O_GROUPS, T], dtype=pl.FP32)
    # Per-group INT32 partials: proj_b_mm writes group g's contribution to output
    # channel n at partials[:, g*D + n]. No atomic-add -> no zero-seed.
    partials = pl.create_tensor([T_PAD, O_GROUPS * D], dtype=pl.INT32)
    proj_b_tids = pl.array.create(O_GROUPS, pl.TASK_ID)

    with pl.manual_scope():
        # One proj_a SPMD grid per output group.
        for g in pl.parallel(O_GROUPS):
            row_base_o = g * T
            out_col_g = g * O_LORA
            with pl.spmd(O_LORA // PROJ_A_MM_N_TILE, name_hint="proj_a_mm", deps=[merge_tid],
                         allow_early_resolve=True) as pa_tid:
                nf = pl.tile.get_block_idx()
                n0 = nf * PROJ_A_MM_N_TILE
                xa0_chunk = pl.slice(o_packed, [MM_T_TILE, A_K_TILE], [row_base_o, 0], valid_shape=[T, A_K_TILE])
                wa0_chunk = wo_a[g : g + 1, n0 : n0 + PROJ_A_MM_N_TILE, 0:A_K_TILE]
                acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
                    k0 = kb * A_K_TILE
                    xa_k_chunk = pl.slice(o_packed, [MM_T_TILE, A_K_TILE], [row_base_o, k0], valid_shape=[T, A_K_TILE])
                    wa_k_chunk = wo_a[g : g + 1, n0 : n0 + PROJ_A_MM_N_TILE, k0 : k0 + A_K_TILE]
                    acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)
                # acc_a is 3D (wo_a keeps its group axis), which subscript-write cannot express.
                o_r_pad = pl.assemble(o_r_pad, acc_a, [0, out_col_g + n0])

            # Per-group proj_a -> quant -> proj_b dependency chain.
            col_g = g * O_LORA
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="quant", deps=[pa_tid], allow_early_resolve=True) as q_tid:
                for qt in pl.pipeline(0, T, QUANT_TOKEN_TILE, stage=2):
                    oc_amax = o_r_pad[qt : qt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA]
                    g_abs = pl.abs(oc_amax)
                    g_row_max = pl.row_max(g_abs)
                    g_row_max = pl.reshape(g_row_max, [1, QUANT_TOKEN_TILE])
                    g_amax_floor = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                    g_amax = pl.maximum(g_amax_floor, g_row_max)
                    g_scale_num = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
                    g_sq_row = pl.div(g_scale_num, g_amax)
                    act_scale_dq[g : g + 1, qt : qt + QUANT_TOKEN_TILE] = pl.recip(g_sq_row)
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

            # One proj_b SPMD grid per output group.
            with pl.spmd(D // PROJ_B_D_TILE, name_hint="proj_b_mm", deps=[q_tid], allow_early_resolve=True) as pb_tid:
                dc = pl.tile.get_block_idx()
                d0 = dc * PROJ_B_D_TILE
                for nf in pl.range(PROJ_B_D_TILE // PROJ_B_MM_N_TILE):
                    n0 = d0 + nf * PROJ_B_MM_N_TILE
                    acc_b = pl.matmul(
                        o_r_i8_pad[:, col_g : col_g + B_K_TILE],
                        wo_b[n0 : n0 + PROJ_B_MM_N_TILE, col_g : col_g + B_K_TILE],
                        b_trans=True,
                        out_dtype=pl.INT32,
                    )
                    for kb in pl.pipeline(1, O_LORA // B_K_TILE, stage=2):
                        k0 = col_g + kb * B_K_TILE
                        acc_b = pl.matmul_acc(
                            acc_b,
                            o_r_i8_pad[:, k0 : k0 + B_K_TILE],
                            wo_b[n0 : n0 + PROJ_B_MM_N_TILE, k0 : k0 + B_K_TILE],
                            b_trans=True,
                        )
                    partials[0:MM_T_TILE, g * D + n0 : g * D + n0 + PROJ_B_MM_N_TILE] = acc_b
            proj_b_tids[g] = pb_tid

    # proj_b_act sums the O_GROUPS INT32 partials -- each dequantized by its group's
    # per-row act scale -- then applies the per-channel weight scale -> BF16. Explicit
    # deps on the eight proj_b grids bridge manual_scope -> the return's auto-dep.
    with pl.spmd((D // PROJ_B_ACT_N_TILE) * (T // PROJ_B_ACT_TASK_T_TILE), name_hint="proj_b_act",
                 deps=[proj_b_tids[i] for i in range(O_GROUPS)], allow_early_resolve=True) as _act_tid:
        act_idx = pl.tile.get_block_idx()
        nreg = act_idx // (T // PROJ_B_ACT_TASK_T_TILE)
        tblk = act_idx - nreg * (T // PROJ_B_ACT_TASK_T_TILE)
        ob_n0 = nreg * PROJ_B_ACT_N_TILE
        t0 = tblk * PROJ_B_ACT_TASK_T_TILE
        wb_scale = wo_b_scale[ob_n0 : ob_n0 + PROJ_B_ACT_N_TILE]
        wb_scale_chunk = pl.reshape(wb_scale, [1, PROJ_B_ACT_N_TILE])
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
            out_t = pl.col_expand_mul(acc, wb_scale_chunk)
            out_bf16 = pl.cast(out_t, target_type=pl.BF16, mode="rint")
            attn_out[b_tb : b_tb + PROJ_B_ACT_T_TILE, ob_n0 : ob_n0 + PROJ_B_ACT_N_TILE] = out_bf16

    return attn_out

@pl.jit
def sparse_attn_test(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
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
        q,
        ori_kv,
        window_swa_indices,
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

    # Per-query-token attention. The window prefix is driven by window_swa_indices;
    # cmp_sparse_indices contains compressed-cache slots only.
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
            row = int(cmp_block_table[b, cmp_slot].item())
            kv_rows.append(cmp_kv.reshape(-1, HEAD_DIM)[row])
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
    # PER-GROUP INT8 activation quant (one amax per O_LORA group, not per full row):
    # this localizes the reduction so proj_a[g]->quant[g]->proj_b[g] can pipeline
    # back-to-back. Each group's INT32 partial is dequantized by its OWN per-row
    # activation scale before the groups are summed (the per-group scale cannot
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

def build_tensor_specs(
    causal_regression_fixture: bool = False,
    short_window_fixture: bool = False,
    mixed_topk_fixture: bool = False,
    cache_window_replacement_fixture: bool = False,
):
    """Build deterministic demo tensors for the HCA standalone harness."""
    import torch
    from golden import TensorSpec
    from utils import block_table, quant_w_per_channel

    cmp_valid = min(CMP_CAPACITY, TOPK - WIN)

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
        return torch.rand(CMP_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM) - 0.5

    def init_attn_sink():
        """Initialize the per-head sink logits to zero."""
        return torch.zeros(H)

    def init_window_block_table():
        """Build the demo block table for the sliding-window cache pages."""
        return block_table(batch=B, table_blocks=ORI_MAX_BLOCKS, physical_blocks=ORI_BLOCK_NUM)

    def init_cmp_block_table():
        """Build the demo block table for the compressed-cache pages."""
        return block_table(
            batch=B,
            table_blocks=CMP_MAX_BLOCKS,
            physical_blocks=CMP_BLOCK_NUM,
        )

    def init_cmp_sparse_indices():
        """Build the sparse index list with a full window prefix and padded compressed tail.

        The compressed tail width follows the active specialization (TOPK - WIN):
        the pruned build narrows it to `cmp_valid` columns, the full-blocks
        baseline keeps the whole CMP_TOPK-wide tail.
        """
        indices = torch.full((T, CMP_TOPK), -1, dtype=torch.int32)
        if cmp_valid:
            indices[:, :cmp_valid] = torch.arange(cmp_valid, dtype=torch.int32)
        if short_window_fixture:
            indices[:, :] = -1
        if mixed_topk_fixture:
            indices[:, :] = -1
            mixed_cmp_valid = cmp_valid
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
    wo_b_i8, wo_b_scale = quant_w_per_channel(wo_b_bf16)

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
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
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
    parser.add_argument("--causal-regression-fixture", action="store_true", default=False,
                        help="Amplify the S=2 future-window-slot regression.")
    parser.add_argument("--short-window-fixture", action="store_true", default=False,
                        help="Use a short-window topk row with valid prefix + -1 padding.")
    parser.add_argument("--mixed-topk-fixture", action="store_true", default=False,
                        help="Use -1-padded window slots with valid compressed raw indices.")
    parser.add_argument("--cache-window-replacement-fixture", action="store_true", default=False,
                        help="Place a sentinel row inside the cache window prefix.")
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
    parser.add_argument("--enable-dep-gen", action="store_true", default=False,
                        help="Capture PTO2 dependency edges (deps.json); the swimlane "
                             "converter draws fanout/fanin arrows from the sibling file.")
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    print(f"compress_ratio={COMPRESS_RATIO} -> TOPK={TOPK} SPARSE_BLOCKS={SPARSE_BLOCKS} PADDED_TOPK={PADDED_TOPK}", flush=True)

    result = run_jit(
        fn=sparse_attn_test,
        specs=build_tensor_specs(
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
