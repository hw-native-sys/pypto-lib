# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Replicated grouped output projection, shared by the SWA / HCA / CSA decoders.

The three sparse-attention decoders end at the same packed head output,
``o_packed[O_GROUPS * T, O_GROUP_IN]``, and projected it with three copies of the
same code. This is that code, once.

The model path does not use it: every decoder hands ``o_packed`` to the
TP-by-group projection in ``o_proj_tp``, which shards the groups one per card and
all-reduces. This replicated form -- every card runs all ``O_GROUPS`` chains -- is
what the per-decoder single-card unit entries validate the attention against.
"""

import pypto.language as pl

from config import (
    FLASH as M,
    DECODE_BATCH,
    DECODE_SEQ,
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
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

# tiling -- identical in all three decoders before this module existed
A_K_TILE = 256           # proj_a cube K frag
PROJ_A_MM_N_TILE = 128   # proj_a cube N frag
MM_T_TILE = 16
T_PAD = ((T + MM_T_TILE - 1) // MM_T_TILE) * MM_T_TILE
B_K_TILE = 256           # proj_b_mm cube K frag
PROJ_B_MM_N_TILE = 256   # proj_b_mm cube N frag; writes grouped INT32 partials
PROJ_B_ACT_N_TILE = 512  # proj_b_act vector N frag; keeps the O_GROUPS-way accumulate inside UB
PROJ_B_D_TILE = 512      # proj_b_mm D chunk per task; its N frags loop inside the task
PROJ_B_ACT_T_TILE = 8    # proj_b_act inner token tile for the O_GROUPS-way INT32->FP32 accumulate
PROJ_B_ACT_TASK_T_TILE = 8   # proj_b_act token block per task
QUANT_TOKEN_TILE = 8


@pl.jit.inline
def o_proj_grouped(
    o_packed: pl.Tensor[[O_GROUPS * T, O_GROUP_IN], pl.BF16],
    merge_tid: pl.Scalar[pl.TASK_ID],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[T, D], pl.BF16],
):
    """Replicated grouped output projection: every card runs all O_GROUPS chains."""
    # Grouped output projection pipelined per group as proj_a[g] -> quant[g] ->
    # proj_b[g]; the per-group amax keeps the quant reduction inside one O_LORA
    # group. manual_scope suppresses auto-dep, so every edge is explicit.
    # proj_b_act combines the group partials and is the sole attn_out writer.
    o_r_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_i8_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.INT8)
    act_scale_dq = pl.create_tensor([O_GROUPS, T], dtype=pl.FP32)
    # Per-group INT32 partials: proj_b_mm writes group g's contribution to output
    # channel n at partials[:, g*D + n]. No atomic-add -> no zero-seed.
    partials = pl.create_tensor([T_PAD, O_GROUPS * D], dtype=pl.INT32)
    proj_b_tids = pl.array.create(O_GROUPS, pl.TASK_ID)

    with pl.manual_scope():
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

            with pl.spmd(D // PROJ_B_D_TILE, name_hint="proj_b_mm", deps=[q_tid], allow_early_resolve=True) as pb_tid:
                dc = pl.tile.get_block_idx()
                d0 = dc * PROJ_B_D_TILE
                for nf in pl.range(PROJ_B_D_TILE // PROJ_B_MM_N_TILE):
                    n0 = d0 + nf * PROJ_B_MM_N_TILE
                    acc_b = pl.create_tensor([MM_T_TILE, PROJ_B_MM_N_TILE], dtype=pl.INT32)
                    for kb in pl.pipeline(0, O_LORA // B_K_TILE, stage=2):
                        k0 = col_g + kb * B_K_TILE
                        if kb == 0:
                            b_act = o_r_i8_pad[:, col_g : col_g + B_K_TILE]
                            b_weight = wo_b[n0 : n0 + PROJ_B_MM_N_TILE, col_g : col_g + B_K_TILE]
                            acc_b = pl.matmul(b_act, b_weight, b_trans=True, out_dtype=pl.INT32)
                        else:
                            b_act = o_r_i8_pad[:, k0 : k0 + B_K_TILE]
                            b_weight = wo_b[n0 : n0 + PROJ_B_MM_N_TILE, k0 : k0 + B_K_TILE]
                            acc_b = pl.matmul_acc(acc_b, b_act, b_weight, b_trans=True)
                    partials[0:MM_T_TILE, g * D + n0 : g * D + n0 + PROJ_B_MM_N_TILE] = acc_b
            proj_b_tids[g] = pb_tid

    # proj_b_act sums the O_GROUPS INT32 partials -- each dequantized by its group's
    # per-row act scale -- then applies the per-channel weight scale -> BF16. Explicit
    # deps on all proj_b_mm tasks bridge manual_scope -> the return's auto-dep.
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
