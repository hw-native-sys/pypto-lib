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

from config import (
    FLASH as M,
    DECODE_BATCH,
    TP,
    DECODE_SEQ,
    BLOCK_SIZE,
    KV_ORI_BLOCK_NUM,
    KV_ORI_MAX_BLOCKS,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
)
from decode_o_proj import decode_o_proj_tp1


# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S
ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")

# model config
B = DECODE_BATCH // TP
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
NEG_INF = -1.0e20

# paged KV cache
ORI_MAX_BLOCKS = KV_ORI_MAX_BLOCKS
ORI_BLOCK_NUM = KV_ORI_BLOCK_NUM

# tiling
GATHER_SPLITS = 4
GATHER_ROWS_PER_TASK = WIN // GATHER_SPLITS
GATHER_RUN = 16          # window sub-tile probed for physical contiguity -> one bulk DMA
H_TILE = 16
QK_M_TILE = 32           # qk_pv M rows per QK/PV matmul; QK_M_TILE/H_TILE-way KV L1->L0 reuse
ATTN_K_TILE = 128
ROPE_TILE = 16
ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE
T_PAD = ((T + 16 - 1) // 16) * 16  # T padded up to the 16-row cube M floor
ROPE_CS_T_TILE = 8  # rope cos/sin row block; T is a multiple of 8 by the batch contract
BIAS_T_TILE = 8     # swa_valid_bias row block, same contract
TOPK = WIN               # SWA sparse-K width: sliding window only
SPARSE_BLOCKS = 1        # the SWA window fits one attention K tile
PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE

assert BLOCK_SIZE % GATHER_RUN == 0, "a contiguous run must not straddle two paged blocks by construction"
assert WIN == ATTN_K_TILE, f"SWA decode expects WIN ({WIN}) == ATTN_K_TILE ({ATTN_K_TILE})"


@pl.jit.inline
def sparse_attn_swa(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    sparse_bias: pl.Tensor[[T_DYN, PADDED_TOPK], pl.FP32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    o_packed_heads: pl.Tensor[[O_GROUPS * T_PAD * HEADS_PER_GROUP, HEAD_DIM], pl.BF16],
) -> tuple[pl.Tensor, pl.Scalar[pl.TASK_ID]]:
    """Write SWA heads as ``[group, T_PAD, head-in-group, dim]`` slabs.

    Only the first runtime ``t_dim`` rows in each group are valid. The
    returned task ID covers every write to the packed output tensor.
    """
    t_dim = pl.tensor.dim(q, 0)
    t_heads = t_dim * H
    t_win = t_dim * WIN
    t_blk = t_dim * (H // H_TILE) * SPARSE_BLOCKS * H_TILE
    t_hblocks = t_dim * (H // H_TILE)
    t_gather = t_dim * GATHER_SPLITS
    rope_cs_blocks = t_dim // ROPE_CS_T_TILE
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    ori_kv_flat = pl.reshape(ori_kv, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    # SWA metadata already lowered each logical window row to a physical cache
    # slot. Current decode tokens must be inserted into ori_kv by the caller
    # before this function runs; there is no speculative overlay path here.

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
    sparse_blk_mi = pl.create_tensor([t_blk, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([t_blk, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([t_blk, HEAD_DIM], dtype=pl.FP32)

    with pl.spmd(t_dim, name_hint="qk_pv", deps=[gather_tids[0]], allow_early_resolve=True) as qk_tid:
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

        m_rope = n_full[:, NOPE_DIM:HEAD_DIM]
        m_swapped = pl.gather(m_rope, dim=-1, index=rope_swap_idx[:, :])
        m_cos_il = rope_cos_il[m_t : m_t + 1, 0:ROPE_DIM]
        m_sin_signed = rope_sin_signed[m_t : m_t + 1, 0:ROPE_DIM]
        m_rope_cos = pl.col_expand_mul(m_rope, m_cos_il)
        m_swap_sin = pl.col_expand_mul(m_swapped, m_sin_signed)
        m_rot = pl.add(m_rope_cos, m_swap_sin)
        n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode="rint")

        m_g0 = m_h0 // HEADS_PER_GROUP
        for m_sg in pl.unroll(H_TILE // HEADS_PER_GROUP):
            m_src_h0 = m_sg * HEADS_PER_GROUP
            n_pack_row = (m_g0 + m_sg) * T_PAD + m_t
            n_dst_head = n_pack_row * HEADS_PER_GROUP
            o_packed_heads[n_dst_head : n_dst_head + HEADS_PER_GROUP, 0:NOPE_DIM] = n_bf16[
                m_src_h0 : m_src_h0 + HEADS_PER_GROUP, 0:NOPE_DIM
            ]
            o_packed_heads[n_dst_head : n_dst_head + HEADS_PER_GROUP, NOPE_DIM:HEAD_DIM] = n_rope_bf16[
                m_src_h0 : m_src_h0 + HEADS_PER_GROUP, 0:ROPE_DIM
            ]

    return o_packed_heads, merge_tid


@pl.jit
def sparse_attn_swa_test(
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
    o_packed_heads = pl.create_tensor([O_GROUPS * T_PAD * HEADS_PER_GROUP, HEAD_DIM], dtype=pl.BF16)
    o_packed_heads, heads_dep = sparse_attn_swa(
        q, ori_kv, swa_indices, sparse_bias,
        attn_sink, freqs_cos, freqs_sin,
        o_packed_heads,
    )
    o_packed = pl.reshape(o_packed_heads, [O_GROUPS * T_PAD, O_GROUP_IN])
    attn_out = decode_o_proj_tp1(
        o_packed,
        wo_a, wo_b, wo_b_scale,
        attn_out, heads_dep,
    )
    return attn_out


def golden_sparse_attn(tensors):
    """Torch reference: sparse_attn decode path followed by grouped o_proj."""
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

    tensors["attn_out"][:] = out.to(torch.bfloat16)

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


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-b", "--batch", type=int, default=B,
                        help=f"runtime request count; a multiple of 4 up to {B} (the compile-time "
                             "upper bound). The token axis is pl.dynamic, so one compiled program "
                             "serves every value.")
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
    args = parser.parse_args()
    if args.batch < 4 or args.batch > B or args.batch % 4 != 0:
        parser.error(f"--batch must be a multiple of 4 in [4, {B}], got {args.batch}")

    print(f"TOPK={TOPK} SPARSE_BLOCKS={SPARSE_BLOCKS} PADDED_TOPK={PADDED_TOPK}", flush=True)

    result = run_jit(
        fn=sparse_attn_swa_test,
        specs=build_tensor_specs(
            args.causal_regression_fixture,
            args.short_window_fixture,
            batch=args.batch,
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
            "attn_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
