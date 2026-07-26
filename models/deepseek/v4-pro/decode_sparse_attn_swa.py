# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 sparse SWA attention with grouped MXFP8 output projection (decode).

Attention (FA / RoPE / merge) stays BF16; MLAEpilog ``o_a_proj`` / ``o_b_proj`` are
Hybrid MXFP8 W8A8 (e4m3 + e8m0, block=32, Right ``[K,N]`` layout).
"""


import pypto.language as pl

from config import (
    FLASH as M,
    DECODE_BATCH,
    DECODE_SEQ,
    BLOCK_SIZE,
    DECODE_ORI_BLOCK_NUM,
    KV_ORI_MAX_BLOCKS,
    MX_BLOCK_K,
)
from mx_quant_common import (
    ATOL_RTOL,
    dynamic_mx_quant_e4m3,
    gen_mxfp8_weight_kn,
    mx_matmul_fp8,
    unpack_scale_b_nn_tiled,
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
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
O_GROUP_IN_SCALE = O_GROUP_IN // MX_BLOCK_K
O_LORA_TOTAL = O_GROUPS * O_LORA
O_LORA_TOTAL_SCALE = O_LORA_TOTAL // MX_BLOCK_K

# kernel-local
ORI_MAX_BLOCKS = KV_ORI_MAX_BLOCKS
ORI_BLOCK_NUM = DECODE_ORI_BLOCK_NUM

# tiling
GATHER_SPLITS = 4
GATHER_ROWS_PER_TASK = WIN // GATHER_SPLITS
# Sub-tile probed for physical contiguity: a decode window is a run of consecutive
# logical positions, so a sub-tile that stays inside one paged block maps to
# consecutive cache rows and moves in ONE bulk DMA. Only a block-straddling or -1
# sub-tile falls back to the per-row copy (~0.44us/row vs ~0.02us/row bulk).
GATHER_RUN = 16
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
# A5 FP32 tgather corrupts the last row of an exactly-8-row box; slice gathers to 4.
ROPE_GATHER_T_TILE = 4
# proj_a cube K-frag. 256 (not 128) keeps the B-cache-line floor: B is K-contiguous
# under b_trans, so K*2B(bf16) = 512B == the a2a3 L2 line (K=128 was 256B, half a
# line -> wasted MTE2 DMA). At 256 the cube's L0A/L0B operand staging hits 100%
# (the wall); 512 would spill it for no gain (swept: K=512 net-negative).
A_K_TILE = 256
# proj_a is a pure-cube MX matmul scope writing fp32 o_r_pad for proj_b_mm.
PROJ_A_MM_N_TILE = 128
MM_T_TILE = 16
T_PAD = ((T + MM_T_TILE - 1) // MM_T_TILE) * MM_T_TILE
B_K_TILE = 256
PROJ_B_MM_N_TILE = 256
PROJ_B_ACT_N_TILE = 512
# Per-group back-to-back MXFP8 o_proj: proj_a[g] -> proj_b[g]; proj_b_act sums FP32 partials.
PA_NFRAGS = O_LORA // PROJ_A_MM_N_TILE   # proj_a cube N-frags per group
# proj_b is one task per (D-chunk, group): the D-chunk's N-frags loop INSIDE the task,
# so the per-group split does not multiply the task count by N-frags. A 512-column
# chunk produces 8 * (4096 / 512) = 64 balanced cube blocks.
PROJ_B_D_CHUNK = 512
PB_DCHUNKS = D // PROJ_B_D_CHUNK
# proj_b_act uses one block per 512-column output region, eight blocks in total.
PROJ_B_ACT_T_TILE = 8
PROJ_B_ACT_TBLK = 8      # proj_b_act token block per task
PB_ACT_NREG = D // PROJ_B_ACT_N_TILE
PB_ACT_TBLKS = T // PROJ_B_ACT_TBLK
NEG_INF = -1.0e20

assert T % 2 == 0
assert WIN % GATHER_SPLITS == 0
assert GATHER_ROWS_PER_TASK % GATHER_RUN == 0, "bulk-copy runs must tile the gather task"
assert BLOCK_SIZE % GATHER_RUN == 0, "a contiguous run must not straddle two paged blocks by construction"
assert H % 4 == 0
assert QK_M_TILE % H_TILE == 0
assert H % QK_M_TILE == 0
assert H % O_GROUPS == 0
assert O_GROUP_IN % MX_BLOCK_K == 0
assert O_LORA % MX_BLOCK_K == 0
assert A_K_TILE % MX_BLOCK_K == 0
assert B_K_TILE % MX_BLOCK_K == 0
assert (O_GROUPS * O_LORA) % B_K_TILE == 0
assert D % PROJ_B_MM_N_TILE == 0, "proj_b_mm cube N-loop must cover D"
assert D % PROJ_B_D_CHUNK == 0, "proj_b D-chunk loop must cover D"
assert PROJ_B_D_CHUNK % PROJ_B_MM_N_TILE == 0, "proj_b inner N-frag loop must cover the D-chunk"
assert T % PROJ_B_ACT_TBLK == 0 and PROJ_B_ACT_TBLK % PROJ_B_ACT_T_TILE == 0
assert D % PROJ_B_ACT_N_TILE == 0, "proj_b_act vector N-loop must cover D"
assert O_LORA % B_K_TILE == 0, "proj_b group K-loop covers O_LORA in B_K_TILE iters"
assert T % ROPE_GATHER_T_TILE == 0
assert H_TILE % ROPE_GATHER_T_TILE == 0

_A_K_CHUNKS = O_GROUP_IN // A_K_TILE
_B_K_CHUNKS = O_LORA // B_K_TILE
_A_KS = A_K_TILE // MX_BLOCK_K
_B_KS = B_K_TILE // MX_BLOCK_K
_WO_A_SCALE_ROWS_PER_G = PA_NFRAGS * _A_K_CHUNKS * _A_KS
_WO_B_NUM_N = D // PROJ_B_MM_N_TILE
_WO_B_NUM_K = O_LORA_TOTAL // B_K_TILE
_WO_B_SCALE_ROWS = _WO_B_NUM_N * _WO_B_NUM_K * _B_KS
_A_SLOTS = O_GROUPS * PA_NFRAGS * _A_K_CHUNKS
_B_SLOTS = O_GROUPS * PB_DCHUNKS * _B_K_CHUNKS
_MX_WS_SLOTS = _A_SLOTS + _B_SLOTS
assert _WO_A_SCALE_ROWS_PER_G == PA_NFRAGS * O_GROUP_IN_SCALE
assert _WO_B_SCALE_ROWS == _WO_B_NUM_N * O_LORA_TOTAL_SCALE
assert _A_K_CHUNKS == 16  # pl.unroll literal in proj_a_mm
assert _B_K_CHUNKS == 4   # pl.unroll literal in proj_b_mm


# SWA sparse-K width: sliding window only.
TOPK = WIN
# Decode SWA consumes metadata-expanded physical KV-cache slots. The current
# kernel shape keeps the SWA window in one attention K tile.
SPARSE_BLOCKS = 1
PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE
assert TOPK == WIN, f"SWA decode expects TOPK ({TOPK}) == WIN ({WIN})"
assert WIN == ATTN_K_TILE, f"SWA decode expects WIN ({WIN}) == ATTN_K_TILE ({ATTN_K_TILE})"


@pl.jit.inline
def sparse_attn_swa(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv_flat: pl.Tensor[[ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    sparse_bias: pl.Tensor[[T, PADDED_TOPK], pl.FP32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[O_GROUPS, _WO_A_SCALE_ROWS_PER_G, PROJ_A_MM_N_TILE], pl.FP8E8M0],
    wo_b: pl.Tensor[[O_LORA_TOTAL, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[_WO_B_SCALE_ROWS, PROJ_B_MM_N_TILE], pl.FP8E8M0],
    attn_out: pl.Tensor[[T, D], pl.BF16],
):
    """Standalone sparse SWA attention with MXFP8 grouped output projection."""
    # SWA metadata already lowered each logical window row to a physical cache
    # slot. Current decode tokens must be inserted into ori_kv by the caller
    # before this function runs; there is no MTP overlay path here.

    swa_kv_flat = pl.create_tensor([T * WIN, HEAD_DIM], dtype=pl.BF16)
    gather_tids = pl.array.create(1, pl.TASK_ID)
    with pl.spmd(T * GATHER_SPLITS, name_hint="swa_gather_kv") as gather_tid:
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
    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    o_packed_heads = pl.create_tensor([O_GROUPS * T * HEADS_PER_GROUP, HEAD_DIM], dtype=pl.BF16)
    o_packed = pl.reshape(o_packed_heads, [O_GROUPS * T, O_GROUP_IN])
    sparse_blk_mi = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, HEAD_DIM], dtype=pl.FP32)

    with pl.spmd(T, name_hint="qk_pv", deps=[gather_tids[0]]) as qk_tid:
        qk_t = pl.tile.get_block_idx()
        qk_token_base = qk_t * (H // H_TILE) * SPARSE_BLOCKS * H_TILE
        for qk_sb in pl.unroll(SPARSE_BLOCKS):
            qk_s0 = qk_sb * ATTN_K_TILE
            qk_bias_row = sparse_bias[qk_t : qk_t + 1, qk_s0 : qk_s0 + ATTN_K_TILE]
            qk_base = qk_t * WIN + qk_s0
            qk_kv = swa_kv_flat[qk_base : qk_base + ATTN_K_TILE, 0 : HEAD_DIM]

            # Keep both 32-head batches in one token task so they reuse the KV
            # tile already resident in L1 instead of loading it once per block.
            for qk_hb in pl.pipeline(H // QK_M_TILE, stage=2):
                qk_h0 = qk_hb * QK_M_TILE
                qk_head_row = qk_t * H + qk_h0
                qk_q_tile = q_flat[qk_head_row : qk_head_row + QK_M_TILE, 0 : HEAD_DIM]
                qk_raw = pl.matmul(qk_q_tile, qk_kv, b_trans=True, out_dtype=pl.FP32)
                qk_scaled = pl.mul(qk_raw, SOFTMAX_SCALE)
                qk_scores = pl.col_expand_add(qk_scaled, qk_bias_row)
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

    # Materialize the head-invariant interleaved cos and signed-sin rows once.
    # This runs alongside qk_pv and keeps the exact indexed RoPE arithmetic used
    # by the reference path while the group merge below changes only scheduling
    # and store granularity.
    rope_cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_cs") as rope_tid:
        # Slice T=8 gathers into ROPE_GATHER_T_TILE-row tiles (A5 tgather 8-row bug).
        cs_col = pl.col_expand_mul(
            pl.full([ROPE_GATHER_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
            pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32))
        cs_dup_f = pl.cast(pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))
        for cs_s0 in pl.range(0, T, ROPE_GATHER_T_TILE):
            cs_cos = pl.cast(freqs_cos[cs_s0 : cs_s0 + ROPE_GATHER_T_TILE, 0:HALF_ROPE], target_type=pl.FP32)
            cs_sin = pl.cast(freqs_sin[cs_s0 : cs_s0 + ROPE_GATHER_T_TILE, 0:HALF_ROPE], target_type=pl.FP32)
            rope_cos_il[cs_s0 : cs_s0 + ROPE_GATHER_T_TILE, 0:ROPE_DIM] = pl.gather(
                cs_cos, dim=-1, index=cs_dup_idx
            )
            rope_sin_signed[cs_s0 : cs_s0 + ROPE_GATHER_T_TILE, 0:ROPE_DIM] = pl.mul(
                pl.gather(cs_sin, dim=-1, index=cs_dup_idx), cs_sign
            )

    # Flatten the one-block SWA merge over token/head tiles into a single
    # 32-block grid, which fits in one AIV wave and avoids eight group-grid
    # submissions. Each block writes two output-projection groups using the
    # same contiguous per-group stores.
    with pl.spmd(T * (H // H_TILE), name_hint="merge_norm", deps=[qk_tid, rope_tid]) as merge_tid:
        m_idx = pl.tile.get_block_idx()
        m_t = m_idx // (H // H_TILE)
        m_h_idx = m_idx - m_t * (H // H_TILE)
        m_h0 = m_h_idx * H_TILE
        m_blk_base = m_t * H + m_h0
        m_mi = sparse_blk_mi[m_blk_base : m_blk_base + H_TILE, 0:1]
        m_li = sparse_blk_li[m_blk_base : m_blk_base + H_TILE, 0:1]
        m_oi = sparse_blk_oi[m_blk_base : m_blk_base + H_TILE, 0:HEAD_DIM]

        n_sink = pl.reshape(attn_sink[m_h0 : m_h0 + H_TILE], [H_TILE, 1])
        n_sink_delta = pl.sub(n_sink, m_mi)
        n_sink_exp = pl.exp(n_sink_delta)
        n_denom = pl.add(m_li, n_sink_exp)
        n_normalized = pl.row_expand_div(m_oi, n_denom)
        n_full = n_normalized[0:H_TILE, 0:HEAD_DIM]
        n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")

        # Inverse RoPE gather in ROPE_GATHER_T_TILE-row subtiles.
        m_col = pl.col_expand_mul(
            pl.full([ROPE_GATHER_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
            pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32))
        m_dup_f = pl.cast(pl.cast(pl.mul(m_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        m_lane = pl.sub(m_col, pl.mul(m_dup_f, 2.0))
        m_swap_idx = pl.cast(pl.sub(pl.add(m_col, 1.0), pl.mul(m_lane, 2.0)), target_type=pl.INT32)
        m_cos_il = rope_cos_il[m_t : m_t + 1, 0:ROPE_DIM]
        m_sin_signed = rope_sin_signed[m_t : m_t + 1, 0:ROPE_DIM]
        n_rope_bf16 = pl.full([H_TILE, ROPE_DIM], dtype=pl.BF16, value=0.0)
        for m_sub in pl.unroll(H_TILE // ROPE_GATHER_T_TILE):
            m_s0 = m_sub * ROPE_GATHER_T_TILE
            m_rope_s = n_full[m_s0 : m_s0 + ROPE_GATHER_T_TILE, NOPE_DIM : HEAD_DIM]
            m_swapped_s = pl.gather(m_rope_s, dim=-1, index=m_swap_idx)
            m_rot_s = pl.add(
                pl.col_expand_mul(m_rope_s, m_cos_il),
                pl.col_expand_mul(m_swapped_s, m_sin_signed),
            )
            n_rope_bf16[m_s0 : m_s0 + ROPE_GATHER_T_TILE, 0:ROPE_DIM] = pl.cast(
                m_rot_s, target_type=pl.BF16, mode="rint"
            )

        m_g0 = m_h0 // HEADS_PER_GROUP
        for m_sg in pl.unroll(H_TILE // HEADS_PER_GROUP):
            m_src_h0 = m_sg * HEADS_PER_GROUP
            n_pack_row = (m_g0 + m_sg) * T + m_t
            n_dst_head = n_pack_row * HEADS_PER_GROUP
            o_packed_heads[n_dst_head : n_dst_head + HEADS_PER_GROUP, 0:NOPE_DIM] = n_bf16[
                m_src_h0 : m_src_h0 + HEADS_PER_GROUP, 0:NOPE_DIM
            ]
            o_packed_heads[n_dst_head : n_dst_head + HEADS_PER_GROUP, NOPE_DIM:HEAD_DIM] = n_rope_bf16[
                m_src_h0 : m_src_h0 + HEADS_PER_GROUP, 0:ROPE_DIM
            ]

    # Back-to-back grouped MXFP8 output projection (manual scope).
    o_r_pad = pl.create_tensor([T_PAD, O_LORA_TOTAL], dtype=pl.FP32)
    partials = pl.create_tensor([T_PAD, O_GROUPS * D], dtype=pl.FP32)
    proj_b_tids = pl.array.create(O_GROUPS, pl.TASK_ID)
    wo_a_kn = pl.reshape(wo_a, [O_GROUPS * O_GROUP_IN, O_LORA])
    wo_a_scale_flat = pl.reshape(
        wo_a_scale, [O_GROUPS * _WO_A_SCALE_ROWS_PER_G, PROJ_A_MM_N_TILE]
    )
    mx_scale_ws = pl.create_tensor(
        [_MX_WS_SLOTS * MM_T_TILE, A_K_TILE // MX_BLOCK_K], dtype=pl.FP8E8M0
    )
    with pl.manual_scope():
        for g in pl.parallel(O_GROUPS):
            row_base_o = g * T
            out_col_g = g * O_LORA
            col_g = g * O_LORA
            wa_row0 = g * O_GROUP_IN

            with pl.spmd(PA_NFRAGS, name_hint="proj_a_mm", deps=[merge_tid]) as pa_tid:
                nf = pl.tile.get_block_idx()
                n0 = nf * PROJ_A_MM_N_TILE
                # Peel K=0 with matmul_mx (init Acc); remaining via matmul_mx_acc.
                k0 = 0
                xa_tile = pl.load(
                    o_packed,
                    [row_base_o, k0],
                    [MM_T_TILE, A_K_TILE],
                    valid_shapes=[T, A_K_TILE],
                    target_memory=pl.Mem.Vec,
                )
                xa_f = pl.cast(xa_tile, target_type=pl.FP32, mode="none")
                xa_q, xa_s = pl.mx_quant(xa_f, mode="mxfp8_e4m3")
                wa_tile = pl.load(
                    wo_a_kn,
                    [wa_row0 + k0, n0],
                    [A_K_TILE, PROJ_A_MM_N_TILE],
                    target_memory=pl.Mem.Mat,
                )
                was_tile = pl.load(
                    wo_a_scale_flat,
                    [
                        g * _WO_A_SCALE_ROWS_PER_G
                        + (nf * _A_K_CHUNKS + k0 // A_K_TILE) * _A_KS,
                        0,
                    ],
                    [A_K_TILE // MX_BLOCK_K, PROJ_A_MM_N_TILE],
                    target_memory=pl.Mem.Mat,
                    mx_layout="mx_b_nn",
                )
                srow = (
                    g * PA_NFRAGS * _A_K_CHUNKS
                    + nf * _A_K_CHUNKS
                    + k0 // A_K_TILE
                ) * MM_T_TILE
                la = pl.move(
                    pl.move(pl.tile.reinterpret_view(xa_q, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.Left,
                )
                la = pl.set_validshape(la, T, A_K_TILE)
                pl.store(pl.tile.reinterpret_view(xa_s, pl.FP8E8M0), [srow, 0], mx_scale_ws)
                las = pl.move(
                    pl.load(
                        mx_scale_ws,
                        [srow, 0],
                        [MM_T_TILE, A_K_TILE // MX_BLOCK_K],
                        target_memory=pl.Mem.Mat,
                        mx_layout="mx_a_zz",
                    ),
                    target_memory=pl.Mem.LeftScale,
                )
                las = pl.tget_scale_addr(las, la)
                las = pl.set_validshape(las, T, A_K_TILE // MX_BLOCK_K)
                rb = pl.move(wa_tile, target_memory=pl.Mem.Right)
                rbs = pl.move(was_tile, target_memory=pl.Mem.RightScale)
                rbs = pl.tget_scale_addr(rbs, rb)
                acc_a = pl.matmul_mx(la, las, rb, rbs)
                for db in pl.unroll(15):  # _A_K_CHUNKS - 1
                    k0 = (db + 1) * A_K_TILE
                    xa_tile2 = pl.load(
                        o_packed,
                        [row_base_o, k0],
                        [MM_T_TILE, A_K_TILE],
                        valid_shapes=[T, A_K_TILE],
                        target_memory=pl.Mem.Vec,
                    )
                    xa_f2 = pl.cast(xa_tile2, target_type=pl.FP32, mode="none")
                    xa_q2, xa_s2 = pl.mx_quant(xa_f2, mode="mxfp8_e4m3")
                    wa_tile2 = pl.load(
                        wo_a_kn,
                        [wa_row0 + k0, n0],
                        [A_K_TILE, PROJ_A_MM_N_TILE],
                        target_memory=pl.Mem.Mat,
                    )
                    was_tile2 = pl.load(
                        wo_a_scale_flat,
                        [
                            g * _WO_A_SCALE_ROWS_PER_G
                            + (nf * _A_K_CHUNKS + (db + 1)) * _A_KS,
                            0,
                        ],
                        [A_K_TILE // MX_BLOCK_K, PROJ_A_MM_N_TILE],
                        target_memory=pl.Mem.Mat,
                        mx_layout="mx_b_nn",
                    )
                    srow2 = (
                        g * PA_NFRAGS * _A_K_CHUNKS
                        + nf * _A_K_CHUNKS
                        + (db + 1)
                    ) * MM_T_TILE
                    la2 = pl.move(
                        pl.move(pl.tile.reinterpret_view(xa_q2, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
                        target_memory=pl.Mem.Left,
                    )
                    la2 = pl.set_validshape(la2, T, A_K_TILE)
                    pl.store(pl.tile.reinterpret_view(xa_s2, pl.FP8E8M0), [srow2, 0], mx_scale_ws)
                    las2 = pl.move(
                        pl.load(
                            mx_scale_ws,
                            [srow2, 0],
                            [MM_T_TILE, A_K_TILE // MX_BLOCK_K],
                            target_memory=pl.Mem.Mat,
                            mx_layout="mx_a_zz",
                        ),
                        target_memory=pl.Mem.LeftScale,
                    )
                    las2 = pl.tget_scale_addr(las2, la2)
                    las2 = pl.set_validshape(las2, T, A_K_TILE // MX_BLOCK_K)
                    rb2 = pl.move(wa_tile2, target_memory=pl.Mem.Right)
                    rbs2 = pl.move(was_tile2, target_memory=pl.Mem.RightScale)
                    rbs2 = pl.tget_scale_addr(rbs2, rb2)
                    acc_a = pl.matmul_mx_acc(acc_a, la2, las2, rb2, rbs2)
                pl.store(acc_a, [0, out_col_g + n0], o_r_pad)

            with pl.spmd(PB_DCHUNKS, name_hint="proj_b_mm", deps=[pa_tid]) as pb_tid:
                dc = pl.tile.get_block_idx()
                d0 = dc * PROJ_B_D_CHUNK
                for nf in pl.range(PROJ_B_D_CHUNK // PROJ_B_MM_N_TILE):
                    n0 = d0 + nf * PROJ_B_MM_N_TILE
                    nb = n0 // PROJ_B_MM_N_TILE
                    # Peel K=0 with matmul_mx (init Acc); remaining via matmul_mx_acc.
                    k0 = col_g
                    or_tile = pl.load(
                        o_r_pad,
                        [0, k0],
                        [MM_T_TILE, B_K_TILE],
                        valid_shapes=[T, B_K_TILE],
                        target_memory=pl.Mem.Vec,
                    )
                    or_q, or_s = pl.mx_quant(or_tile, mode="mxfp8_e4m3")
                    wb_tile = pl.load(
                        wo_b,
                        [k0, n0],
                        [B_K_TILE, PROJ_B_MM_N_TILE],
                        target_memory=pl.Mem.Mat,
                    )
                    wbs_tile = pl.load(
                        wo_b_scale,
                        [
                            (nb * _WO_B_NUM_K + k0 // B_K_TILE) * _B_KS,
                            0,
                        ],
                        [B_K_TILE // MX_BLOCK_K, PROJ_B_MM_N_TILE],
                        target_memory=pl.Mem.Mat,
                        mx_layout="mx_b_nn",
                    )
                    srow = (
                        _A_SLOTS
                        + g * PB_DCHUNKS * _B_K_CHUNKS
                        + dc * _B_K_CHUNKS
                        + (k0 - col_g) // B_K_TILE
                    ) * MM_T_TILE
                    ob_la = pl.move(
                        pl.move(pl.tile.reinterpret_view(or_q, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
                        target_memory=pl.Mem.Left,
                    )
                    ob_la = pl.set_validshape(ob_la, T, B_K_TILE)
                    pl.store(pl.tile.reinterpret_view(or_s, pl.FP8E8M0), [srow, 0], mx_scale_ws)
                    ob_las = pl.move(
                        pl.load(
                            mx_scale_ws,
                            [srow, 0],
                            [MM_T_TILE, B_K_TILE // MX_BLOCK_K],
                            target_memory=pl.Mem.Mat,
                            mx_layout="mx_a_zz",
                        ),
                        target_memory=pl.Mem.LeftScale,
                    )
                    ob_las = pl.tget_scale_addr(ob_las, ob_la)
                    ob_las = pl.set_validshape(ob_las, T, B_K_TILE // MX_BLOCK_K)
                    ob_rb = pl.move(wb_tile, target_memory=pl.Mem.Right)
                    ob_rbs = pl.move(wbs_tile, target_memory=pl.Mem.RightScale)
                    ob_rbs = pl.tget_scale_addr(ob_rbs, ob_rb)
                    acc_b = pl.matmul_mx(ob_la, ob_las, ob_rb, ob_rbs)
                    for db in pl.unroll(3):  # _B_K_CHUNKS - 1
                        k0 = col_g + (db + 1) * B_K_TILE
                        or_tile2 = pl.load(
                            o_r_pad,
                            [0, k0],
                            [MM_T_TILE, B_K_TILE],
                            valid_shapes=[T, B_K_TILE],
                            target_memory=pl.Mem.Vec,
                        )
                        or_q2, or_s2 = pl.mx_quant(or_tile2, mode="mxfp8_e4m3")
                        wb_tile2 = pl.load(
                            wo_b,
                            [k0, n0],
                            [B_K_TILE, PROJ_B_MM_N_TILE],
                            target_memory=pl.Mem.Mat,
                        )
                        wbs_tile2 = pl.load(
                            wo_b_scale,
                            [
                                (nb * _WO_B_NUM_K + k0 // B_K_TILE) * _B_KS,
                                0,
                            ],
                            [B_K_TILE // MX_BLOCK_K, PROJ_B_MM_N_TILE],
                            target_memory=pl.Mem.Mat,
                            mx_layout="mx_b_nn",
                        )
                        srow2 = (
                            _A_SLOTS
                            + g * PB_DCHUNKS * _B_K_CHUNKS
                            + dc * _B_K_CHUNKS
                            + (db + 1)
                        ) * MM_T_TILE
                        ob_la2 = pl.move(
                            pl.move(pl.tile.reinterpret_view(or_q2, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
                            target_memory=pl.Mem.Left,
                        )
                        ob_la2 = pl.set_validshape(ob_la2, T, B_K_TILE)
                        pl.store(pl.tile.reinterpret_view(or_s2, pl.FP8E8M0), [srow2, 0], mx_scale_ws)
                        ob_las2 = pl.move(
                            pl.load(
                                mx_scale_ws,
                                [srow2, 0],
                                [MM_T_TILE, B_K_TILE // MX_BLOCK_K],
                                target_memory=pl.Mem.Mat,
                                mx_layout="mx_a_zz",
                            ),
                            target_memory=pl.Mem.LeftScale,
                        )
                        ob_las2 = pl.tget_scale_addr(ob_las2, ob_la2)
                        ob_las2 = pl.set_validshape(ob_las2, T, B_K_TILE // MX_BLOCK_K)
                        ob_rb2 = pl.move(wb_tile2, target_memory=pl.Mem.Right)
                        ob_rbs2 = pl.move(wbs_tile2, target_memory=pl.Mem.RightScale)
                        ob_rbs2 = pl.tget_scale_addr(ob_rbs2, ob_rb2)
                        acc_b = pl.matmul_mx_acc(acc_b, ob_la2, ob_las2, ob_rb2, ob_rbs2)
                    pl.store(acc_b, [0, g * D + n0], partials)
            proj_b_tids[g] = pb_tid

    with pl.spmd(PB_ACT_NREG * PB_ACT_TBLKS, name_hint="proj_b_act",
                 deps=[proj_b_tids[i] for i in range(O_GROUPS)]) as _act_tid:
        act_idx = pl.tile.get_block_idx()
        nreg = act_idx // PB_ACT_TBLKS
        tblk = act_idx - nreg * PB_ACT_TBLKS
        ob_n0 = nreg * PROJ_B_ACT_N_TILE
        t0 = tblk * PROJ_B_ACT_TBLK
        for b_tb in pl.range(t0, t0 + PROJ_B_ACT_TBLK, PROJ_B_ACT_T_TILE):
            acc = pl.full([PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE], dtype=pl.FP32, value=0.0)
            for act_g in pl.pipeline(O_GROUPS, stage=2):
                p_col0 = act_g * D + ob_n0
                p_g = partials[b_tb : b_tb + PROJ_B_ACT_T_TILE, p_col0 : p_col0 + PROJ_B_ACT_N_TILE]
                acc = pl.add(acc, p_g)
            out_bf16 = pl.cast(acc, target_type=pl.BF16, mode="rint")
            attn_out[b_tb : b_tb + PROJ_B_ACT_T_TILE, ob_n0 : ob_n0 + PROJ_B_ACT_N_TILE] = out_bf16
    return attn_out


@pl.jit
def sparse_attn_test(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    swa_lens: pl.Tensor[[T], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[O_GROUPS, _WO_A_SCALE_ROWS_PER_G, PROJ_A_MM_N_TILE], pl.FP8E8M0],
    wo_b: pl.Tensor[[O_LORA_TOTAL, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[_WO_B_SCALE_ROWS, PROJ_B_MM_N_TILE], pl.FP8E8M0],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    ori_kv_flat = pl.reshape(ori_kv, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    sparse_bias = pl.create_tensor([T, PADDED_TOPK], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_valid_bias"):
        v_col = pl.cast(pl.arange(0, [1, ATTN_K_TILE], dtype=pl.INT32), target_type=pl.FP32)
        v_col_m = pl.col_expand(pl.full([T, ATTN_K_TILE], dtype=pl.FP32, value=0.0), v_col)
        v_lens = pl.cast(pl.reshape(swa_lens[0:T], [T, 1]), target_type=pl.FP32)
        v_valid = pl.minimum(
            pl.maximum(pl.neg(pl.row_expand_sub(v_col_m, v_lens)), 0.0),
            1.0,
        )
        sparse_bias[0:T, 0:ATTN_K_TILE] = pl.mul(pl.sub(v_valid, 1.0), -NEG_INF)
    sparse_attn_swa(
        q,
        ori_kv_flat,
        swa_indices,
        sparse_bias,
        attn_sink,
        freqs_cos,
        freqs_sin,
        wo_a,
        wo_a_scale,
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
    ori_kv_flat = ori_kv.reshape(ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
    swa_indices = tensors["swa_indices"]
    swa_lens = tensors["swa_lens"]
    attn_sink = tensors["attn_sink"].float()
    cos = tensors["freqs_cos"].float()
    sin = tensors["freqs_sin"].float()
    wo_a = tensors["wo_a"]
    wo_a_scale = tensors["wo_a_scale"]
    wo_b = tensors["wo_b"]
    wo_b_scale = tensors["wo_b_scale"]

    def _b_scale_a(s):
        return unpack_scale_b_nn_tiled(
            s,
            k_tile_rows=_A_KS,
            n_tile=PROJ_A_MM_N_TILE,
            logical_k=O_GROUP_IN_SCALE,
            logical_n=O_LORA,
        )

    def _b_scale_b(s):
        return unpack_scale_b_nn_tiled(
            s,
            k_tile_rows=_B_KS,
            n_tile=PROJ_B_MM_N_TILE,
            logical_k=O_LORA_TOTAL_SCALE,
            logical_n=D,
        )

    def mx_matmul_act_tiled(x_f, w, w_s, k_tile):
        acc = None
        for k0 in range(0, x_f.shape[-1], k_tile):
            xq, xs = dynamic_mx_quant_e4m3(x_f[..., k0 : k0 + k_tile])
            part = mx_matmul_fp8(
                xq, xs, w[k0 : k0 + k_tile], w_s[k0 // MX_BLOCK_K : (k0 + k_tile) // MX_BLOCK_K]
            )
            acc = part if acc is None else acc + part
        return acc

    o = torch.zeros(T, H, HEAD_DIM)

    # Per-query-token attention. swa_indices is the authoritative physical
    # cache-row list; invalid tail columns are -1 and swa_lens gives the valid
    # prefix length.
    for t in range(T):
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

    seq_per_batch = T // B
    o_model = o.float().view(B, seq_per_batch, O_GROUPS, O_GROUP_IN)
    out = torch.zeros(T, D, dtype=torch.float32)
    wo_b_s = _b_scale_b(wo_b_scale)
    for g in range(O_GROUPS):
        o_g = o_model[:, :, g, :].reshape(T, O_GROUP_IN)
        o_r = mx_matmul_act_tiled(o_g, wo_a[g], _b_scale_a(wo_a_scale[g]), A_K_TILE)
        wo_b_g = wo_b[g * O_LORA : (g + 1) * O_LORA, :]
        wo_b_s_g = wo_b_s[g * O_LORA // MX_BLOCK_K : (g + 1) * O_LORA // MX_BLOCK_K, :]
        out = out + mx_matmul_act_tiled(o_r, wo_b_g, wo_b_s_g, B_K_TILE)

    tensors["attn_out"][:] = out.to(torch.bfloat16)

def build_tensor_specs(
    causal_regression_fixture: bool = False,
    short_window_fixture: bool = False,
):
    """Build deterministic demo tensors for the merged standalone harness."""
    import torch
    from decode_metadata import block_table
    from golden import TensorSpec

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

    def init_attn_sink():
        """Initialize the per-head sink logits to zero."""
        return torch.zeros(H)

    def init_ori_block_table():
        """Build the demo block table for the sliding-window cache pages."""
        return block_table(
            batch=B,
            table_blocks=ORI_MAX_BLOCKS,
            physical_blocks=ORI_BLOCK_NUM,
        )

    def init_swa_lens():
        lens = torch.full((T,), WIN, dtype=torch.int32)
        if short_window_fixture:
            lens.fill_(17)
        return lens

    def init_swa_indices():
        """Build physical cache-row indices for the standalone SWA fixture."""
        tbl = init_ori_block_table()
        indices = torch.full((T, WIN), -1, dtype=torch.int32)
        lens = init_swa_lens()
        for t in range(T):
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
        angles = torch.arange(T * HALF_ROPE).reshape(T, HALF_ROPE) * 1e-3
        cos_half = torch.cos(angles)
        return torch.cat([cos_half, cos_half], dim=-1)

    def init_sin():
        """Build the split-half sine table used by the inverse-RoPE reference."""
        angles = torch.arange(T * HALF_ROPE).reshape(T, HALF_ROPE) * 1e-3
        sin_half = torch.sin(angles)
        return torch.cat([sin_half, sin_half], dim=-1)

    wo_a_stacked = []
    wo_a_scale_stacked = []
    for _g in range(O_GROUPS):
        wa, was = gen_mxfp8_weight_kn(
            (O_GROUP_IN, O_LORA),
            dequant_std=1.0 / (O_GROUP_IN ** 0.5),
            chan_cv=0.50,
            n_tile=PROJ_A_MM_N_TILE,
            k_tile=A_K_TILE,
        )
        wo_a_stacked.append(wa)
        wo_a_scale_stacked.append(was)
    wo_a_tensor = torch.stack(wo_a_stacked, dim=0)
    wo_a_scale_tensor = torch.stack(wo_a_scale_stacked, dim=0)
    wo_b, wo_b_scale = gen_mxfp8_weight_kn(
        (O_LORA_TOTAL, D),
        dequant_std=1.0 / (O_LORA_TOTAL ** 0.5),
        chan_cv=0.50,
        n_tile=PROJ_B_MM_N_TILE,
        k_tile=B_K_TILE,
    )

    def init_wo_a():
        return wo_a_tensor

    def init_wo_a_scale():
        return wo_a_scale_tensor

    def init_wo_b():
        return wo_b

    def init_wo_b_scale():
        return wo_b_scale

    return [
        TensorSpec("q", [T, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("swa_indices", [T, WIN], torch.int32, init_value=init_swa_indices),
        TensorSpec("swa_lens", [T], torch.int32, init_value=init_swa_lens),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("freqs_cos", [T, ROPE_DIM], torch.bfloat16, init_value=init_cos),
        TensorSpec("freqs_sin", [T, ROPE_DIM], torch.bfloat16, init_value=init_sin),
        TensorSpec("wo_a", [O_GROUPS, O_GROUP_IN, O_LORA], torch.float8_e4m3fn, init_value=init_wo_a),
        TensorSpec(
            "wo_a_scale",
            [O_GROUPS, _WO_A_SCALE_ROWS_PER_G, PROJ_A_MM_N_TILE],
            torch.float8_e8m0fnu,
            init_value=init_wo_a_scale,
        ),
        TensorSpec("wo_b", [O_LORA_TOTAL, D], torch.float8_e4m3fn, init_value=init_wo_b),
        TensorSpec(
            "wo_b_scale",
            [_WO_B_SCALE_ROWS, PROJ_B_MM_N_TILE],
            torch.float8_e8m0fnu,
            init_value=init_wo_b_scale,
        ),
        TensorSpec("attn_out", [T, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--causal-regression-fixture", action="store_true", default=False,
                        help="Amplify the S=2 future-window-slot regression.")
    parser.add_argument("--short-window-fixture", action="store_true", default=False,
                        help="Use a short-window topk row with valid prefix + -1 padding.")
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2, 4))
    parser.add_argument("--enable-dep-gen", action="store_true", default=False,
                        help="Capture PTO2 dependency edges (deps.json); the swimlane "
                             "converter draws fanout/fanin arrows from the sibling file.")
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    print(f"TOPK={TOPK} SPARSE_BLOCKS={SPARSE_BLOCKS} PADDED_TOPK={PADDED_TOPK}", flush=True)

    oproj_tol = ATOL_RTOL["oproj_mxfp8"]
    result = run_jit(
        fn=sparse_attn_test,
        specs=build_tensor_specs(
            args.causal_regression_fixture,
            args.short_window_fixture,
        ),
        golden_fn=golden_sparse_attn,
        golden_data=args.golden_data,
        compile_cfg=dict(dump_passes=args.dump_passes),
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
            "attn_out": ratio_allclose(atol=oproj_tol["atol"], rtol=oproj_tol["rtol"]),
        },
        compile_only=args.compile_only,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
