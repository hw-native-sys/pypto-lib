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

from qkv_proj_rope import make_mxfp8_projection
from output_projection import build_mxfp8_output_specs, golden_mxfp8_output_projection

from config import (
    ACTIVE as M,
    DECODE_BATCH,
    DECODE_SEQ,
    BLOCK_SIZE,
    DECODE_ORI_BLOCK_NUM,
    KV_ORI_MAX_BLOCKS,
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
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
ORI_MAX_BLOCKS = KV_ORI_MAX_BLOCKS
ORI_BLOCK_NUM = DECODE_ORI_BLOCK_NUM
TOPK = WIN
NEG_INF = -1.0e20

# tiling
GATHER_SPLIT_TILE = 4
GATHER_ROW_TILE = WIN // GATHER_SPLIT_TILE
GATHER_RUN_TILE = 16
H_TILE = 16
QK_M_TILE = 32
ATTN_K_TILE = 128
ROPE_TILE = 16
ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE
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
assert H_TILE % HEADS_PER_GROUP == 0

# Sized so the four parallel group bundles fit the AIC grid in a single wave:
# O_GROUP_TILE * (D // PROJ_B_D_TILE) blocks per bundle, times 4 bundles, must stay
# under the 36 cube cores. At 512 the flash variant asked for 64 and proj_b_mm ran
# in two waves, showing up twice on the critical path.
PROJ_B_D_TILE = 1024 if M.name == "flash" else 1792
PROJ_B_ACT_T_TILE = 8
SPARSE_BLOCKS = 1
PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE
assert WIN == ATTN_K_TILE


project_o_a = make_mxfp8_projection(O_GROUP_IN, O_LORA)
project_o_b = make_mxfp8_projection(O_GROUPS * O_LORA, D)
OUTPUT_COPY_ROWS = 4
OUTPUT_COPY_COLS = 512


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
    wo_a_scale: pl.Tensor[[O_GROUPS * (O_GROUP_IN // 32), O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[(O_GROUPS * O_LORA) // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    attn_out: pl.Tensor[[T, D], pl.BF16],
):
    """Standalone sparse attention with a BF16 projected output."""
    swa_kv_flat = pl.create_tensor([T * WIN, HEAD_DIM], dtype=pl.BF16)
    gather_tids = pl.array.create(1, pl.TASK_ID)
    with pl.spmd(T * GATHER_SPLIT_TILE, name_hint="swa_gather_kv") as gather_tid:
        g_task = pl.tile.get_block_idx()
        g_t = g_task // GATHER_SPLIT_TILE
        g_split = g_task - g_t * GATHER_SPLIT_TILE
        g_r0 = g_split * GATHER_ROW_TILE
        g_base = g_t * WIN
        for g_sub in pl.range(GATHER_ROW_TILE // GATHER_RUN_TILE):
            g_sr0 = g_r0 + g_sub * GATHER_RUN_TILE
            g_sdst = g_base + g_sr0
            g_first = pl.read(swa_indices, [g_t, g_sr0])
            g_last = pl.read(swa_indices, [g_t, g_sr0 + GATHER_RUN_TILE - 1])
            g_run_ok = (g_last - g_first) + pl.min(g_first, 0) * GATHER_RUN_TILE
            if g_run_ok == GATHER_RUN_TILE - 1:
                g_run_src = pl.cast(g_first, pl.INDEX)
                g_run_tile = ori_kv_flat[g_run_src : g_run_src + GATHER_RUN_TILE, 0 : HEAD_DIM]
                swa_kv_flat[g_sdst : g_sdst + GATHER_RUN_TILE, 0 : HEAD_DIM] = g_run_tile
            else:
                for g_dr in pl.range(GATHER_RUN_TILE):
                    g_dst = g_sdst + g_dr
                    g_slot_i32 = pl.read(swa_indices, [g_t, g_sr0 + g_dr])
                    if g_slot_i32 >= 0:
                        g_slot = pl.cast(g_slot_i32, pl.INDEX)
                        swa_kv_flat[g_dst : g_dst + 1, 0 : HEAD_DIM] = ori_kv_flat[g_slot : g_slot + 1, 0 : HEAD_DIM]
                    else:
                        swa_kv_flat[g_dst : g_dst + 1, 0 : HEAD_DIM] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
    gather_tids[0] = gather_tid

    rope_cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_cs") as rope_tid:
        for cp in pl.range(HALF_ROPE // ROPE_TILE):
            cp_r0 = cp * ROPE_TILE
            cp_c0 = 2 * cp_r0
            cs_cos = pl.cast(freqs_cos[0:T, cp_r0 : cp_r0 + ROPE_TILE], target_type=pl.FP32)
            cs_sin = pl.cast(freqs_sin[0:T, cp_r0 : cp_r0 + ROPE_TILE], target_type=pl.FP32)
            cs_cos_dup = pl.full([T, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=0.0)
            cs_cos_dup = pl.tensor.scatter(cs_cos, mask_pattern=pl.tile.MaskPattern.P0101, dst=cs_cos_dup)
            cs_cos_dup = pl.tensor.scatter(cs_cos, mask_pattern=pl.tile.MaskPattern.P1010, dst=cs_cos_dup)
            cs_sin_neg = pl.neg(cs_sin)
            cs_sin_signed = pl.full([T, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=0.0)
            cs_sin_signed = pl.tensor.scatter(cs_sin, mask_pattern=pl.tile.MaskPattern.P0101, dst=cs_sin_signed)
            cs_sin_signed = pl.tensor.scatter(cs_sin_neg, mask_pattern=pl.tile.MaskPattern.P1010, dst=cs_sin_signed)
            rope_cos_il[0:T, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = cs_cos_dup
            rope_sin_signed[0:T, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = cs_sin_signed

    o_packed_heads = pl.create_tensor([O_GROUPS * T * HEADS_PER_GROUP, HEAD_DIM], dtype=pl.BF16)
    q_flat = pl.reshape(q, [T * H, HEAD_DIM])

    sparse_blk_mi = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, HEAD_DIM], dtype=pl.FP32)

    with pl.spmd(T, name_hint="qk_pv", deps=[gather_tids[0], rope_tid]) as qk_tid:
        qk_t = pl.tile.get_block_idx()
        qk_token_base = qk_t * (H // H_TILE) * SPARSE_BLOCKS * H_TILE
        for qk_sb in pl.unroll(SPARSE_BLOCKS):
            qk_s0 = qk_sb * ATTN_K_TILE
            qk_bias_row = sparse_bias[qk_t : qk_t + 1, qk_s0 : qk_s0 + ATTN_K_TILE]
            qk_base = qk_t * WIN + qk_s0
            qk_kv = swa_kv_flat[qk_base : qk_base + ATTN_K_TILE, 0 : HEAD_DIM]

            for qk_hb in pl.pipeline(H // QK_M_TILE, stage=2):
                qk_h0 = qk_hb * QK_M_TILE
                qk_head_row = qk_t * H + qk_h0
                qk_q_tile = q_flat[qk_head_row : qk_head_row + QK_M_TILE, 0 : HEAD_DIM]
                qk_raw = pl.matmul(qk_q_tile, qk_kv, b_trans=True, out_dtype=pl.FP32)
                qk_scaled = pl.mul(qk_raw, SOFTMAX_SCALE)
                qk_scores = pl.col_expand_add(qk_scaled, qk_bias_row)
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

        # Sink, normalize, inverse-RoPE and pack, for every head group of this
        # token. SPARSE_BLOCKS is 1 here, so there is no cross-block softmax
        # state to combine and the partials above are already final.
        for m_h_idx in pl.range(H // H_TILE):
            m_h0 = m_h_idx * H_TILE
            m_blk_base = qk_t * H + m_h0
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
            m_even = pl.gather(m_rope, mask_pattern=pl.tile.MaskPattern.P0101)
            m_odd = pl.gather(m_rope, mask_pattern=pl.tile.MaskPattern.P1010)
            m_swapped = pl.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
            m_swapped = pl.tensor.scatter(m_odd, mask_pattern=pl.tile.MaskPattern.P0101, dst=m_swapped)
            m_swapped = pl.tensor.scatter(m_even, mask_pattern=pl.tile.MaskPattern.P1010, dst=m_swapped)
            m_cos_il = rope_cos_il[qk_t : qk_t + 1, 0:ROPE_DIM]
            m_sin_signed = rope_sin_signed[qk_t : qk_t + 1, 0:ROPE_DIM]
            m_rope_cos = pl.col_expand_mul(m_rope, m_cos_il)
            m_swap_sin = pl.col_expand_mul(m_swapped, m_sin_signed)
            m_rot = pl.add(m_rope_cos, m_swap_sin)
            n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode="rint")

            m_g0 = m_h0 // HEADS_PER_GROUP
            for m_sg in pl.unroll(H_TILE // HEADS_PER_GROUP):
                m_src_h0 = m_sg * HEADS_PER_GROUP
                n_pack_row = (m_g0 + m_sg) * T + qk_t
                n_dst_head = n_pack_row * HEADS_PER_GROUP
                n_nope_group = n_bf16[m_src_h0 : m_src_h0 + HEADS_PER_GROUP, 0:NOPE_DIM]
                o_packed_heads[n_dst_head : n_dst_head + HEADS_PER_GROUP, 0:NOPE_DIM] = n_nope_group
                n_rope_group = n_rope_bf16[m_src_h0 : m_src_h0 + HEADS_PER_GROUP, 0:ROPE_DIM]
                o_packed_heads[n_dst_head : n_dst_head + HEADS_PER_GROUP, NOPE_DIM:HEAD_DIM] = n_rope_group


    o_packed = pl.reshape(o_packed_heads, [O_GROUPS * T, O_GROUP_IN])
    o_ready = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)
    with pl.spmd(
        (O_GROUPS * T // OUTPUT_COPY_ROWS) * (O_GROUP_IN // OUTPUT_COPY_COLS),
        name_hint="o_projection_input",
        deps=[qk_tid],
    ) as projection_ready_tid:
        copy_index = pl.tile.get_block_idx()
        row = (copy_index // (O_GROUP_IN // OUTPUT_COPY_COLS)) * OUTPUT_COPY_ROWS
        col = (copy_index % (O_GROUP_IN // OUTPUT_COPY_COLS)) * OUTPUT_COPY_COLS
        o_ready[row:row + OUTPUT_COPY_ROWS, col:col + OUTPUT_COPY_COLS] = o_packed[
            row:row + OUTPUT_COPY_ROWS, col:col + OUTPUT_COPY_COLS
        ]

    o_projected = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.BF16)
    for group in pl.parallel(O_GROUPS):
        group_input: pl.Tensor[[T, O_GROUP_IN], pl.BF16] = pl.slice(o_ready, [T, O_GROUP_IN], [group * T, 0])
        group_weight_3d = pl.slice(wo_a, [1, O_GROUP_IN, O_LORA], [group, 0, 0])
        group_weight = pl.reshape(group_weight_3d, [O_GROUP_IN, O_LORA])
        group_scale: pl.Tensor[[O_GROUP_IN // 32, O_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(
            wo_a_scale, [O_GROUP_IN // 32, O_LORA], [group * (O_GROUP_IN // 32), 0]
        )
        # The native projection takes contiguous output storage. Pack its
        # result into the token-major bank with an explicit outer copy.
        group_output = pl.create_tensor([T, O_LORA], dtype=pl.BF16)
        project_o_a(group_input, group_weight, group_scale, group_output, projection_ready_tid)
        for copy_index in pl.spmd(
            (T // OUTPUT_COPY_ROWS) * (O_LORA // OUTPUT_COPY_COLS),
            name_hint="o_projection_pack",
        ):
            row = (copy_index // (O_LORA // OUTPUT_COPY_COLS)) * OUTPUT_COPY_ROWS
            col = (copy_index % (O_LORA // OUTPUT_COPY_COLS)) * OUTPUT_COPY_COLS
            o_projected[row:row + OUTPUT_COPY_ROWS, group * O_LORA + col:group * O_LORA + col + OUTPUT_COPY_COLS] = group_output[
                row:row + OUTPUT_COPY_ROWS, col:col + OUTPUT_COPY_COLS
            ]
    project_o_b(o_projected, wo_b, wo_b_scale, attn_out, projection_ready_tid)

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
    wo_a_scale: pl.Tensor[[O_GROUPS * (O_GROUP_IN // 32), O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[(O_GROUPS * O_LORA) // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    sparse_bias = pl.create_tensor([T, PADDED_TOPK], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_valid_bias"):
        v_col_i32 = pl.arange(0, [1, ATTN_K_TILE], dtype=pl.INT32)
        v_col = pl.cast(v_col_i32, target_type=pl.FP32)
        v_col_base = pl.full([T, ATTN_K_TILE], dtype=pl.FP32, value=0.0)
        v_col_m = pl.col_expand(v_col_base, v_col)
        v_lens_row = pl.reshape(swa_lens[0:T], [T, 1])
        v_lens = pl.cast(v_lens_row, target_type=pl.FP32)
        v_delta = pl.row_expand_sub(v_col_m, v_lens)
        v_inside = pl.neg(v_delta)
        v_floor = pl.maximum(v_inside, 0.0)
        v_len_valid = pl.minimum(v_floor, 1.0)
        v_idx_f = pl.cast(swa_indices[0:T, 0:ATTN_K_TILE], target_type=pl.FP32)
        v_idx_one = pl.add(v_idx_f, 1.0)
        v_idx_floor = pl.maximum(v_idx_one, 0.0)
        v_idx_valid = pl.minimum(v_idx_floor, 1.0)
        v_valid = pl.mul(v_len_valid, v_idx_valid)
        v_mask = pl.sub(v_valid, 1.0)
        sparse_bias[0:T, 0:ATTN_K_TILE] = pl.mul(v_mask, -NEG_INF)
    ori_kv_flat = pl.reshape(ori_kv, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    sparse_attn_swa(
        q, ori_kv_flat, swa_indices, sparse_bias,
        attn_sink, freqs_cos, freqs_sin,
        wo_a, wo_a_scale, wo_b, wo_b_scale,
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

    o = torch.zeros(T, H, HEAD_DIM)

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
            valid_entries = [start + i < valid_len and int(slot) >= 0 for i, slot in enumerate(slots)]
            valid_tile = torch.tensor(valid_entries, dtype=torch.bool)
            if end - start < ATTN_K_TILE:
                valid_pad = torch.zeros(ATTN_K_TILE - (end - start), dtype=torch.bool)
                valid_tile = torch.cat([valid_tile, valid_pad])
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

    out = golden_mxfp8_output_projection(o.reshape(T, O_GROUPS, O_GROUP_IN), tensors)
    tensors["attn_out"][:] = out

def build_tensor_specs(
    causal_regression_fixture: bool = False,
    short_window_fixture: bool = False,
    unmapped_visible_page_fixture: bool = False,
):
    """Build deterministic demo tensors for the merged standalone harness."""
    import torch
    from decode_metadata import block_table
    from golden import TensorSpec

    fixture_count = sum((causal_regression_fixture, short_window_fixture, unmapped_visible_page_fixture))
    if fixture_count > 1:
        raise ValueError("regression fixtures are mutually exclusive")

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
        return block_table(batch=B, table_blocks=ORI_MAX_BLOCKS, physical_blocks=ORI_BLOCK_NUM)

    def init_swa_lens():
        lens = torch.full((T,), WIN, dtype=torch.int32)
        if short_window_fixture:
            lens.fill_(17)
        return lens

    def init_swa_indices():
        """Build physical cache-row indices for the standalone SWA fixture."""
        tbl = init_ori_block_table().clone()
        window_start = 0
        if unmapped_visible_page_fixture:
            tbl[:, 0] = -1
            window_start = BLOCK_SIZE // 2
        indices = torch.full((T, WIN), -1, dtype=torch.int32)
        lens = init_swa_lens()
        for t in range(T):
            b = t // S
            valid_len = int(lens[t].item())
            for w in range(valid_len):
                pos = window_start + w
                logical_blk = pos // BLOCK_SIZE
                intra = pos % BLOCK_SIZE
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

    return [
        TensorSpec("q", [T, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("swa_indices", [T, WIN], torch.int32, init_value=init_swa_indices),
        TensorSpec("swa_lens", [T], torch.int32, init_value=init_swa_lens),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("freqs_cos", [T, ROPE_DIM], torch.bfloat16, init_value=init_cos),
        TensorSpec("freqs_sin", [T, ROPE_DIM], torch.bfloat16, init_value=init_sin),
        *build_mxfp8_output_specs(O_GROUPS, O_GROUP_IN, O_LORA, D),
        TensorSpec("attn_out", [T, D], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    causal_help = "Amplify the S=2 future-window-slot regression."
    parser.add_argument("--causal-regression-fixture", action="store_true", default=False, help=causal_help)
    short_help = "Use a short-window topk row with valid prefix + -1 padding."
    parser.add_argument("--short-window-fixture", action="store_true", default=False, help=short_help)
    unmapped_help = "Use a visible window whose older cache page is unmapped."
    parser.add_argument(
        "--unmapped-visible-page-fixture",
        action="store_true",
        default=False,
        help=unmapped_help,
    )
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    dep_help = "Capture PTO2 dependency edges (deps.json); the swimlane "
    dep_help += "converter draws fanout/fanin arrows from the sibling file."
    parser.add_argument("--enable-dep-gen", action="store_true", default=False, help=dep_help)
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    print(f"TOPK={TOPK} SPARSE_BLOCKS={SPARSE_BLOCKS} PADDED_TOPK={PADDED_TOPK}", flush=True)

    result = run(
        fn=sparse_attn_test,
        specs=build_tensor_specs(
            args.causal_regression_fixture,
            args.short_window_fixture,
            args.unmapped_visible_page_fixture,
        ),
        golden_fn=golden_sparse_attn,
        golden_data=args.golden_data,
        config=dict(
            dump_passes=args.dump_passes,
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
