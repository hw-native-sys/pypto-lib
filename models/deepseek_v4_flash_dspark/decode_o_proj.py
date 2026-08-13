# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=4
"""DeepSeek-V4 decode output projections and their TP communication."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import (
    DECODE_TOKENS,
    FLASH as M,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    TP,
)


# parallel layout
TP_SIZE = TP

# model config
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
LOCAL_T = DECODE_TOKENS // TP_SIZE
GROUP_T = TP_SIZE * LOCAL_T
LOCAL_O_GROUPS = O_GROUPS // TP_SIZE
LOCAL_O_WIDTH = LOCAL_O_GROUPS * O_LORA

# dynamic shape variables
T_DYN = pl.dynamic("T_DYN")

# tiling and collective-native layouts
TOKEN_TILE = 16
COMM_ROW_TILE = 8
LOCAL_T_PAD = (LOCAL_T + TOKEN_TILE - 1) // TOKEN_TILE * TOKEN_TILE
T_PAD = LOCAL_T_PAD
GROUP_T_PAD = TP_SIZE * LOCAL_T_PAD
ATTENTION_WINDOW_ROWS = LOCAL_O_GROUPS * GROUP_T_PAD
O_WINDOW_ROWS = TP_SIZE * LOCAL_T_PAD

# local output projection tiling
A_K_TILE = 256
PROJ_A_MM_N_TILE = 128
MM_T_TILE = T_PAD
PROJ_A_ROW_TILE = 16
B_K_TILE = 256
PROJ_B_MM_N_TILE = 256
PROJ_B_ACT_N_TILE = 512
QUANT_TOKEN_TILE = 8
PROJ_B_D_TILE = 512
PROJ_B_ACT_T_TILE = 8
PROJ_B_ACT_TASK_T_TILE = 8

# TP-sharded output projection tiling
O_A_T_TILE = 16
O_A_K_TILE = 256
O_A_N_TILE = 128
QUANT_T_TILE = 8
O_B_T_TILE = 128
O_B_K_TILE = 256
O_B_N_TILE = 256
O_B_D_TILE = 512
ACT_T_TILE = 8
ACT_N_TILE = 512

# fixture
FIXTURE_LOCAL_T = max(1, LOCAL_T - 1)
FIXTURE_OUTPUT_SENTINEL = -7.0

if DECODE_TOKENS % TP_SIZE != 0:
    raise ValueError(f"decode tokens {DECODE_TOKENS} must be divisible by TP size {TP_SIZE}")
if O_GROUPS % TP_SIZE != 0:
    raise ValueError(f"output groups {O_GROUPS} must be divisible by TP size {TP_SIZE}")
if O_GROUP_IN % O_A_K_TILE != 0:
    raise ValueError(f"O-A input {O_GROUP_IN} must be divisible by K tile {O_A_K_TILE}")
if O_LORA % O_A_N_TILE != 0:
    raise ValueError(f"O-A output {O_LORA} must be divisible by N tile {O_A_N_TILE}")
if O_LORA % O_B_K_TILE != 0:
    raise ValueError(f"O-B group width {O_LORA} must be divisible by K tile {O_B_K_TILE}")
if D % O_B_D_TILE != 0 or O_B_D_TILE % O_B_N_TILE != 0:
    raise ValueError("O-B output tiles must divide the hidden dimension")
if D % ACT_N_TILE != 0:
    raise ValueError(f"O-B activation tile {ACT_N_TILE} must divide hidden size {D}")
if GROUP_T_PAD % O_B_T_TILE != 0:
    raise ValueError(f"O-B token tile {O_B_T_TILE} must divide token capacity {GROUP_T_PAD}")


@pl.jit.inline
def decode_o_proj_tp1(
    o_packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[T_DYN, D], pl.BF16],
    heads_dep: pl.Scalar[pl.TASK_ID],
):
    """Project local-token, full-group attention heads into BF16 hidden rows."""
    t_dim = pl.tensor.dim(attn_out, 0)
    act_t_blks = t_dim // PROJ_B_ACT_TASK_T_TILE
    proj_a_rows = (t_dim + PROJ_A_ROW_TILE - 1) // PROJ_A_ROW_TILE

    # Back-to-back grouped output projection: proj_a[g] -> quant[g] -> proj_b[g]
    # pipelines per group, because the PER-GROUP amax keeps the quant reduction
    # inside one O_LORA group instead of barriering the whole row. manual_scope
    # suppresses auto-dep, so every edge is explicit: proj_a waits on merge_norm,
    # quant[g] on proj_a[g], proj_b[g] on quant[g]. proj_b_act combines the group
    # partials with their row scales and is the consolidated attn_out writer.
    o_r_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_i8_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.INT8)
    act_scale_dq = pl.create_tensor([O_GROUPS, T_PAD], dtype=pl.FP32)
    # Per-group INT32 partials: proj_b_mm writes group g's contribution to output
    # channel n at partials[:, g*D + n]. No atomic-add -> no zero-seed.
    partials = pl.create_tensor([T_PAD, O_GROUPS * D], dtype=pl.INT32)
    proj_b_tids = pl.array.create(O_GROUPS, pl.TASK_ID)

    with pl.manual_scope():
        for g in pl.parallel(O_GROUPS):
            row_base_o = g * T_PAD
            out_col_g = g * O_LORA

            with pl.spmd(proj_a_rows * (O_LORA // PROJ_A_MM_N_TILE), name_hint="proj_a_mm", deps=[heads_dep],
                         allow_early_resolve=True) as pa_tid:
                pa_unit = pl.tile.get_block_idx()
                pa_rb = pa_unit // (O_LORA // PROJ_A_MM_N_TILE)  # row block outermost
                nf = pa_unit - pa_rb * (O_LORA // PROJ_A_MM_N_TILE)
                pa_r0 = pa_rb * PROJ_A_ROW_TILE
                pa_rows = pl.min(PROJ_A_ROW_TILE, t_dim - pa_r0)
                pa_src0 = row_base_o + pa_r0
                n0 = nf * PROJ_A_MM_N_TILE
                xa0_chunk = pl.slice(o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, 0], valid_shape=[pa_rows, A_K_TILE])
                wa0_chunk = wo_a[g : g + 1, n0 : n0 + PROJ_A_MM_N_TILE, 0:A_K_TILE]
                acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
                    k0 = kb * A_K_TILE
                    xa_k_chunk = pl.slice(o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, k0], valid_shape=[pa_rows, A_K_TILE])
                    wa_k_chunk = wo_a[g : g + 1, n0 : n0 + PROJ_A_MM_N_TILE, k0 : k0 + A_K_TILE]
                    acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)
                # acc_a is 3D (wo_a keeps its group axis), which subscript-write cannot express.
                o_r_pad = pl.assemble(o_r_pad, acc_a, [pa_r0, out_col_g + n0])

            col_g = g * O_LORA
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="quant", deps=[pa_tid], allow_early_resolve=True) as q_tid:
                for qt in pl.pipeline(0, t_dim, QUANT_TOKEN_TILE, stage=2):
                    oc_amax = o_r_pad[qt : qt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA]
                    g_abs = pl.abs(oc_amax)
                    g_row_max = pl.row_max(g_abs)
                    g_row_max = pl.reshape(g_row_max, [1, QUANT_TOKEN_TILE])
                    g_amax_floor = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                    g_amax = pl.maximum(g_amax_floor, g_row_max)
                    g_scale_num = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
                    g_sq_row = pl.div(g_scale_num, g_amax)
                    g_scale_dq = pl.mul(g_amax, 1.0 / INT8_SCALE_MAX)
                    act_scale_dq[g : g + 1, qt : qt + QUANT_TOKEN_TILE] = g_scale_dq
                    g_sq_col = pl.reshape(g_sq_row, [QUANT_TOKEN_TILE, 1])
                    oc_q = o_r_pad[qt : qt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA]
                    oq_scaled = pl.row_expand_mul(oc_q, g_sq_col)
                    oq_i32 = pl.cast(oq_scaled, target_type=pl.INT32, mode="rint")
                    oq_half = pl.cast(oq_i32, target_type=pl.FP16, mode="round")
                    oq_i8 = pl.cast(oq_half, target_type=pl.INT8, mode="trunc")
                    o_r_i8_pad[qt : qt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA] = oq_i8
                # Zero the rows past the runtime token count; proj_b_mm reads the full T_PAD extent.
                for zt in pl.range(t_dim, T_PAD, QUANT_TOKEN_TILE):
                    zero_half = pl.full([QUANT_TOKEN_TILE, O_LORA], dtype=pl.FP16, value=0.0)
                    o_r_i8_pad[zt : zt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA] = pl.cast(
                        zero_half, target_type=pl.INT8, mode="trunc")

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
    with pl.spmd(act_t_blks * (D // PROJ_B_ACT_N_TILE), name_hint="proj_b_act",
                 deps=[proj_b_tids[i] for i in range(O_GROUPS)], allow_early_resolve=True) as _act_tid:
        act_idx = pl.tile.get_block_idx()
        tblk = act_idx // (D // PROJ_B_ACT_N_TILE)  # token block outermost
        nreg = act_idx - tblk * (D // PROJ_B_ACT_N_TILE)
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


@pl.jit.inline
def tp_group_barrier(
    signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    expected: pl.Scalar[pl.INT32],
):
    """Synchronize one contiguous physical TP group."""
    for peer_tp in pl.range(TP_SIZE):
        if peer_tp != tp_rank:
            pld.system.notify(
                target=signal, peer=group_base + peer_tp,
                offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    for source_tp in pl.range(TP_SIZE):
        if source_tp != tp_rank:
            pld.system.wait(signal=signal, offsets=[source_tp, 0], expected=expected, cmp=pld.WaitCmp.Ge)
    return signal


@pl.jit.inline
def reset_tp_group_signal(
    signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Clear this rank's peer-credit cells after a collective completes."""
    reset_value = pl.cast(-2, pl.INT32)
    self_rank = group_base + tp_rank
    for source_tp in pl.range(TP_SIZE):
        if source_tp != tp_rank:
            pld.system.notify(
                target=signal, peer=self_rank,
                offsets=[source_tp, 0], value=reset_value, op=pld.NotifyOp.AtomicAdd,
            )
    return signal


@pl.jit.incore
def kv_token_allgather_step(
    kv_local: pl.Tensor[[LOCAL_T, D], pl.BF16],
    group_out: pl.InOut[pl.Tensor[[GROUP_T, D], pl.BF16]],
    gather_window: pld.DistributedTensor[[GROUP_T, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
):
    """Gather valid rank-major token rows for the replicated KV projections."""
    group_t = TP_SIZE * local_t
    target_row = tp_rank * local_t
    for peer_tp in pl.range(TP_SIZE):
        pld.tensor.put(
            dst=gather_window, peer=group_base + peer_tp, src=kv_local,
            dst_offsets=[target_row, 0], src_offsets=[0, 0], shape=[local_t, D],
            chunk_rows=COMM_ROW_TILE, chunk_cols=D,
        )

    expected_one = pl.cast(1, pl.INT32)
    gather_signal = tp_group_barrier(gather_signal, group_base, tp_rank, expected_one)
    for group_row in pl.range(group_t):
        group_out[group_row : group_row + 1, 0:D] = gather_window[group_row : group_row + 1, 0:D]
    expected_two = pl.cast(2, pl.INT32)
    gather_signal = tp_group_barrier(gather_signal, group_base, tp_rank, expected_two)
    gather_signal = reset_tp_group_signal(gather_signal, group_base, tp_rank)
    return group_out, gather_signal


@pl.jit.incore
def attention_token_head_all_to_all_step(
    attention_grouped: pl.Tensor[[O_GROUPS * LOCAL_T_PAD, O_GROUP_IN], pl.BF16],
    local_groups_out: pl.InOut[pl.Tensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16]],
    exchange_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    exchange_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
):
    """Exchange valid output-group rows into local-group, group-token order."""
    group_t = TP_SIZE * local_t
    for destination_rank in pl.range(TP_SIZE):
        for local_group in pl.range(LOCAL_O_GROUPS):
            global_group = destination_rank * LOCAL_O_GROUPS + local_group
            source_row = global_group * LOCAL_T_PAD
            target_row = local_group * GROUP_T_PAD + tp_rank * local_t
            pld.tensor.put(
                dst=exchange_window, peer=group_base + destination_rank, src=attention_grouped,
                dst_offsets=[target_row, 0], src_offsets=[source_row, 0], shape=[local_t, O_GROUP_IN],
                chunk_rows=COMM_ROW_TILE, chunk_cols=O_GROUP_IN,
            )

    expected_one = pl.cast(1, pl.INT32)
    exchange_signal = tp_group_barrier(exchange_signal, group_base, tp_rank, expected_one)
    for local_group in pl.range(LOCAL_O_GROUPS):
        group_base_row = local_group * GROUP_T_PAD
        for group_row in pl.range(group_t):
            copy_row = group_base_row + group_row
            window_row = exchange_window[copy_row : copy_row + 1, 0:O_GROUP_IN]
            local_groups_out[copy_row : copy_row + 1, 0:O_GROUP_IN] = window_row
    expected_two = pl.cast(2, pl.INT32)
    exchange_signal = tp_group_barrier(exchange_signal, group_base, tp_rank, expected_two)
    exchange_signal = reset_tp_group_signal(exchange_signal, group_base, tp_rank)
    return local_groups_out, exchange_signal


@pl.jit.incore
def o_projection_reduce_scatter_step(
    o_partial: pl.Tensor[[GROUP_T_PAD, D], pl.FP32],
    local_out: pl.InOut[pl.Tensor[[LOCAL_T_PAD, D], pl.BF16]],
    reduce_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.FP32],
    reduce_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
):
    """Sum O-B rank partials and scatter valid contiguous local token rows."""
    for owner_rank in pl.range(TP_SIZE):
        owner_source_row = owner_rank * local_t
        target_row = tp_rank * local_t
        pld.tensor.put(
            dst=reduce_window, peer=group_base + owner_rank, src=o_partial,
            dst_offsets=[target_row, 0], src_offsets=[owner_source_row, 0], shape=[local_t, D],
            chunk_rows=COMM_ROW_TILE, chunk_cols=D,
        )

    expected_one = pl.cast(1, pl.INT32)
    reduce_signal = tp_group_barrier(reduce_signal, group_base, tp_rank, expected_one)
    for local_row in pl.range(local_t):
        acc = pl.load(reduce_window, [local_row, 0], [1, D])
        for source_tp in pl.range(1, TP_SIZE):
            source_partial_row = source_tp * local_t + local_row
            source_partial = pl.load(reduce_window, [source_partial_row, 0], [1, D])
            acc = pl.add(acc, source_partial)
        reduced_bf16 = pl.cast(acc, target_type=pl.BF16, mode="rint")
        pl.store(reduced_bf16, [local_row, 0], local_out)
    expected_two = pl.cast(2, pl.INT32)
    reduce_signal = tp_group_barrier(reduce_signal, group_base, tp_rank, expected_two)
    reduce_signal = reset_tp_group_signal(reduce_signal, group_base, tp_rank)
    return local_out, reduce_signal


@pl.jit
def decode_attention_collectives_fixture(
    kv_local: pl.Tensor[[LOCAL_T, D], pl.BF16],
    attention_grouped: pl.Tensor[[O_GROUPS * LOCAL_T_PAD, O_GROUP_IN], pl.BF16],
    o_partial: pl.Tensor[[GROUP_T_PAD, D], pl.FP32],
    kv_group: pl.InOut[pl.Tensor[[GROUP_T, D], pl.BF16]],
    attention_local_groups: pl.InOut[pl.Tensor[[LOCAL_O_GROUPS * GROUP_T_PAD, O_GROUP_IN], pl.BF16]],
    o_local: pl.InOut[pl.Tensor[[LOCAL_T_PAD, D], pl.BF16]],
    kv_window: pld.DistributedTensor[[GROUP_T, D], pl.BF16],
    kv_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.FP32],
    o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
):
    """Apply the decode attention communication transitions on one TP rank."""
    kv_group, kv_signal = kv_token_allgather_step(
        kv_local, kv_group, kv_window, kv_signal, group_base, tp_rank, local_t,
    )
    attention_local_groups, attention_signal = attention_token_head_all_to_all_step(
        attention_grouped, attention_local_groups, attention_window, attention_signal, group_base, tp_rank, local_t,
    )
    o_local, o_signal = o_projection_reduce_scatter_step(
        o_partial, o_local, o_window, o_signal, group_base, tp_rank, local_t,
    )
    return kv_group, attention_local_groups, o_local, kv_signal, attention_signal, o_signal


@pl.jit.host
def l3_decode_attention_collectives_fixture(
    kv_local: pl.Tensor[[TP_SIZE, LOCAL_T, D], pl.BF16],
    attention_grouped: pl.Tensor[[TP_SIZE, O_GROUPS * LOCAL_T_PAD, O_GROUP_IN], pl.BF16],
    o_partial: pl.Tensor[[TP_SIZE, GROUP_T_PAD, D], pl.FP32],
    kv_group: pl.InOut[pl.Tensor[[TP_SIZE, GROUP_T, D], pl.BF16]],
    attention_local_groups: pl.InOut[pl.Tensor[[TP_SIZE, ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16]],
    o_local: pl.InOut[pl.Tensor[[TP_SIZE, LOCAL_T_PAD, D], pl.BF16]],
    local_t: pl.Scalar[pl.INT32],
):
    """Launch the decode attention communication fixture on one TP group."""
    kv_window_buf = pld.alloc_window_buffer([GROUP_T, D], dtype=pl.BF16)
    kv_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    attention_window_buf = pld.alloc_window_buffer([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
    attention_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_window_buf = pld.alloc_window_buffer([O_WINDOW_ROWS, D], dtype=pl.FP32)
    o_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        kv_window = pld.window(kv_window_buf, [GROUP_T, D], dtype=pl.BF16)
        kv_signal = pld.window(kv_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        attention_window = pld.window(attention_window_buf, [ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
        attention_signal = pld.window(attention_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_window = pld.window(o_window_buf, [O_WINDOW_ROWS, D], dtype=pl.FP32)
        o_signal = pld.window(o_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        decode_attention_collectives_fixture(
            kv_local[rank], attention_grouped[rank], o_partial[rank],
            kv_group[rank], attention_local_groups[rank], o_local[rank],
            kv_window, kv_signal, attention_window, attention_signal, o_window, o_signal,
            0, rank, local_t, device=rank,
        )


def build_collective_tensor_specs(local_t=FIXTURE_LOCAL_T):
    """Build deterministic four-rank inputs with poisoned capacity rows."""
    import torch

    from golden import ScalarSpec, TensorSpec

    if local_t < 1 or local_t > LOCAL_T:
        raise ValueError(f"local_t must be in [1, {LOCAL_T}], got {local_t}")

    def init_kv_local():
        values = torch.arange(TP_SIZE * LOCAL_T * D, dtype=torch.int32)
        values = values.remainder(251).reshape(TP_SIZE, LOCAL_T, D).to(torch.bfloat16)
        values[:, local_t:] = -1000.0
        return values

    def init_attention_grouped():
        shape = (TP_SIZE, O_GROUPS * LOCAL_T_PAD, O_GROUP_IN)
        values = torch.arange(TP_SIZE * O_GROUPS * LOCAL_T_PAD * O_GROUP_IN, dtype=torch.int32)
        values = values.remainder(127).reshape(shape).to(torch.bfloat16)
        grouped = values.reshape(TP_SIZE, O_GROUPS, LOCAL_T_PAD, O_GROUP_IN)
        grouped[:, :, local_t:] = -2000.0
        return grouped.reshape(shape)

    def init_o_partial():
        shape = (TP_SIZE, GROUP_T_PAD, D)
        values = torch.arange(TP_SIZE * GROUP_T_PAD * D, dtype=torch.int32)
        values = values.remainder(17).reshape(shape).to(torch.float32)
        values[:, TP_SIZE * local_t:] = -3000.0
        return values

    attention_grouped_shape = [TP_SIZE, O_GROUPS * LOCAL_T_PAD, O_GROUP_IN]
    attention_local_shape = [TP_SIZE, LOCAL_O_GROUPS * GROUP_T_PAD, O_GROUP_IN]
    return [
        TensorSpec("kv_local", [TP_SIZE, LOCAL_T, D], torch.bfloat16, init_value=init_kv_local),
        TensorSpec("attention_grouped", attention_grouped_shape, torch.bfloat16, init_value=init_attention_grouped),
        TensorSpec("o_partial", [TP_SIZE, GROUP_T_PAD, D], torch.float32, init_value=init_o_partial),
        TensorSpec(
            "kv_group", [TP_SIZE, GROUP_T, D], torch.bfloat16,
            init_value=FIXTURE_OUTPUT_SENTINEL, is_output=True,
        ),
        TensorSpec(
            "attention_local_groups", attention_local_shape, torch.bfloat16,
            init_value=FIXTURE_OUTPUT_SENTINEL, is_output=True,
        ),
        TensorSpec(
            "o_local", [TP_SIZE, LOCAL_T_PAD, D], torch.bfloat16,
            init_value=FIXTURE_OUTPUT_SENTINEL, is_output=True,
        ),
        ScalarSpec("local_t", torch.int32, local_t),
    ]


def golden_decode_attention_collectives_fixture(tensors):
    """Assemble the rank-major gather, grouped exchange, and reduced token shards."""
    import torch

    local_t = int(tensors["local_t"])
    group_t = TP_SIZE * local_t
    gathered_kv = tensors["kv_local"][:, :local_t].reshape(group_t, D)
    tensors["kv_group"].fill_(FIXTURE_OUTPUT_SENTINEL)
    tensors["kv_group"][:, :group_t] = gathered_kv.unsqueeze(0)

    grouped = tensors["attention_grouped"].reshape(TP_SIZE, O_GROUPS, LOCAL_T_PAD, O_GROUP_IN)
    exchanged = torch.full_like(tensors["attention_local_groups"], FIXTURE_OUTPUT_SENTINEL)
    for destination_rank in range(TP_SIZE):
        for local_group in range(LOCAL_O_GROUPS):
            global_group = destination_rank * LOCAL_O_GROUPS + local_group
            target_row = local_group * GROUP_T_PAD
            group_rows = grouped[:, global_group, :local_t].reshape(group_t, O_GROUP_IN)
            exchanged[destination_rank, target_row : target_row + group_t] = group_rows
    tensors["attention_local_groups"][:] = exchanged

    reduced = tensors["o_partial"][:, :group_t].sum(dim=0)
    tensors["o_local"].fill_(FIXTURE_OUTPUT_SENTINEL)
    for rank in range(TP_SIZE):
        token_start = rank * local_t
        token_end = token_start + local_t
        tensors["o_local"][rank, :local_t] = reduced[token_start:token_end].to(torch.bfloat16)


@pl.jit.inline
def decode_sharded_o_projection(
    attention_local_groups: pl.Tensor[[LOCAL_O_GROUPS, GROUP_T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    local_t: pl.Scalar[pl.INT32],
    o_partial: pl.Tensor[[GROUP_T_PAD, D], pl.FP32],
) -> tuple[pl.Tensor, pl.Scalar[pl.TASK_ID]]:
    """Project the compact A2A receive prefix into one rank's FP32 O-B partial."""
    # A2A receive rows use [local_group, source_rank * local_t] offsets.
    group_t = TP_SIZE * local_t
    o_a_rows = (group_t + O_A_T_TILE - 1) // O_A_T_TILE
    o_b_rows = (group_t + O_B_T_TILE - 1) // O_B_T_TILE
    o_b_group_t = o_b_rows * O_B_T_TILE
    act_rows = (group_t + ACT_T_TILE - 1) // ACT_T_TILE

    attn_2d = pl.reshape(attention_local_groups, [LOCAL_O_GROUPS * GROUP_T_PAD, O_GROUP_IN])
    wo_a_flat = pl.reshape(wo_a, [LOCAL_O_WIDTH, O_GROUP_IN])
    o_a_fp32 = pl.create_tensor([GROUP_T_PAD, LOCAL_O_WIDTH], dtype=pl.FP32)
    o_a_i8 = pl.create_tensor([GROUP_T_PAD, LOCAL_O_WIDTH], dtype=pl.INT8)
    act_scale_dq = pl.create_tensor([LOCAL_O_GROUPS, GROUP_T_PAD], dtype=pl.FP32)
    o_b_i32 = pl.create_tensor([GROUP_T_PAD, LOCAL_O_GROUPS * D], dtype=pl.INT32)
    proj_b_tids = pl.array.create(LOCAL_O_GROUPS, pl.TASK_ID)

    for local_group in pl.parallel(LOCAL_O_GROUPS):
        attention_row = local_group * GROUP_T_PAD
        o_a_col = local_group * O_LORA
        with pl.spmd(o_a_rows * (O_LORA // O_A_N_TILE), name_hint="tp_o_a") as proj_a_tid:
            proj_a_unit = pl.tile.get_block_idx()
            row_block = proj_a_unit // (O_LORA // O_A_N_TILE)
            n_block = proj_a_unit - row_block * (O_LORA // O_A_N_TILE)
            t0 = row_block * O_A_T_TILE
            n0 = n_block * O_A_N_TILE
            a_rows = pl.min(O_A_T_TILE, group_t - t0)
            src_row = attention_row + t0
            weight_row = o_a_col + n0
            o_a_x0 = pl.slice(attn_2d, [O_A_T_TILE, O_A_K_TILE], [src_row, 0], valid_shape=[a_rows, O_A_K_TILE])
            o_a_w0 = wo_a_flat[weight_row : weight_row + O_A_N_TILE, 0:O_A_K_TILE]
            o_a_acc = pl.matmul(o_a_x0, o_a_w0, b_trans=True, out_dtype=pl.FP32)
            for k0 in pl.pipeline(O_A_K_TILE, O_GROUP_IN, O_A_K_TILE, stage=2):
                o_a_xk = pl.slice(attn_2d, [O_A_T_TILE, O_A_K_TILE], [src_row, k0], valid_shape=[a_rows, O_A_K_TILE])
                o_a_wk = wo_a_flat[weight_row : weight_row + O_A_N_TILE, k0 : k0 + O_A_K_TILE]
                o_a_acc = pl.matmul_acc(o_a_acc, o_a_xk, o_a_wk, b_trans=True)
            o_a_valid = pl.set_validshape(o_a_acc, a_rows, O_A_N_TILE)
            o_a_fp32[t0 : t0 + O_A_T_TILE, weight_row : weight_row + O_A_N_TILE] = o_a_valid

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_o_a_quant", deps=[proj_a_tid]) as quant_tid:
            for qt in pl.pipeline(0, group_t, QUANT_T_TILE, stage=2):
                quant_rows = pl.min(QUANT_T_TILE, group_t - qt)
                o_a_tile = pl.slice(o_a_fp32, [QUANT_T_TILE, O_LORA], [qt, o_a_col], valid_shape=[quant_rows, O_LORA])
                o_a_abs = pl.abs(o_a_tile)
                row_amax = pl.reshape(pl.row_max(o_a_abs), [1, QUANT_T_TILE])
                amax_floor = pl.full([1, QUANT_T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                row_amax = pl.maximum(amax_floor, row_amax)
                scale_max = pl.full([1, QUANT_T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
                scale_q_row = pl.div(scale_max, row_amax)
                scale_dq_row = pl.recip(scale_q_row)
                scale_dq_valid = pl.set_validshape(scale_dq_row, 1, quant_rows)
                act_scale_dq[local_group : local_group + 1, qt : qt + QUANT_T_TILE] = scale_dq_valid
                scale_q_col = pl.reshape(scale_q_row, [QUANT_T_TILE, 1])
                o_a_scaled = pl.row_expand_mul(o_a_tile, scale_q_col)
                o_a_i32 = pl.cast(o_a_scaled, target_type=pl.INT32, mode="rint")
                o_a_fp16 = pl.cast(o_a_i32, target_type=pl.FP16, mode="round")
                o_a_quant = pl.cast(o_a_fp16, target_type=pl.INT8, mode="trunc")
                o_a_quant_valid = pl.set_validshape(o_a_quant, quant_rows, O_LORA)
                o_a_i8[qt : qt + QUANT_T_TILE, o_a_col : o_a_col + O_LORA] = o_a_quant_valid
            # Pad the compact O-A prefix to complete 128-row O-B slabs.
            for qt in pl.range(group_t, o_b_group_t, QUANT_T_TILE):
                pad_rows = pl.min(QUANT_T_TILE, o_b_group_t - qt)
                o_a_padding_fp16 = pl.full([QUANT_T_TILE, O_LORA], dtype=pl.FP16, value=0.0)
                o_a_padding = pl.cast(o_a_padding_fp16, target_type=pl.INT8, mode="trunc")
                o_a_padding_valid = pl.set_validshape(o_a_padding, pad_rows, O_LORA)
                o_a_i8[qt : qt + QUANT_T_TILE, o_a_col : o_a_col + O_LORA] = o_a_padding_valid

        with pl.spmd(o_b_rows * (D // O_B_D_TILE), name_hint="tp_o_b", deps=[quant_tid]) as proj_b_tid:
            proj_b_unit = pl.tile.get_block_idx()
            row_block = proj_b_unit // (D // O_B_D_TILE)
            d_block = proj_b_unit - row_block * (D // O_B_D_TILE)
            t0 = row_block * O_B_T_TILE
            d0 = d_block * O_B_D_TILE
            for n0 in pl.range(d0, d0 + O_B_D_TILE, O_B_N_TILE):
                o_b_x0 = o_a_i8[t0 : t0 + O_B_T_TILE, o_a_col : o_a_col + O_B_K_TILE]
                o_b_w0 = wo_b[n0 : n0 + O_B_N_TILE, o_a_col : o_a_col + O_B_K_TILE]
                o_b_acc = pl.matmul(o_b_x0, o_b_w0, b_trans=True, out_dtype=pl.INT32)
                for k0 in pl.pipeline(O_B_K_TILE, O_LORA, O_B_K_TILE, stage=2):
                    b_k0 = o_a_col + k0
                    o_b_xk = o_a_i8[t0 : t0 + O_B_T_TILE, b_k0 : b_k0 + O_B_K_TILE]
                    o_b_wk = wo_b[n0 : n0 + O_B_N_TILE, b_k0 : b_k0 + O_B_K_TILE]
                    o_b_acc = pl.matmul_acc(o_b_acc, o_b_xk, o_b_wk, b_trans=True)
                partial_col = local_group * D + n0
                o_b_i32[t0 : t0 + O_B_T_TILE, partial_col : partial_col + O_B_N_TILE] = o_b_acc
        proj_b_tids[local_group] = proj_b_tid

    with pl.spmd(
        act_rows * (D // ACT_N_TILE), name_hint="tp_o_b_dequant",
        deps=[proj_b_tids[group] for group in range(LOCAL_O_GROUPS)],
    ) as completion_tid:
        act_unit = pl.tile.get_block_idx()
        row_block = act_unit // (D // ACT_N_TILE)
        n_block = act_unit - row_block * (D // ACT_N_TILE)
        t0 = row_block * ACT_T_TILE
        n0 = n_block * ACT_N_TILE
        out_rows = pl.min(ACT_T_TILE, group_t - t0)
        acc = pl.full([ACT_T_TILE, ACT_N_TILE], dtype=pl.FP32, value=0.0)
        for local_group in pl.pipeline(LOCAL_O_GROUPS, stage=2):
            part_col = local_group * D + n0
            group_i32 = pl.slice(o_b_i32, [ACT_T_TILE, ACT_N_TILE], [t0, part_col], valid_shape=[out_rows, ACT_N_TILE])
            group_partial_fp32 = pl.cast(group_i32, target_type=pl.FP32, mode="none")
            scale_row = pl.slice(act_scale_dq, [1, ACT_T_TILE], [local_group, t0], valid_shape=[1, out_rows])
            scale_col = pl.reshape(scale_row, [ACT_T_TILE, 1])
            group_dequant = pl.row_expand_mul(group_partial_fp32, scale_col)
            acc = pl.add(acc, group_dequant)
        weight_scale = pl.reshape(wo_b_scale[n0 : n0 + ACT_N_TILE], [1, ACT_N_TILE])
        o_b_fp32 = pl.col_expand_mul(acc, weight_scale)
        o_partial_valid = pl.set_validshape(o_b_fp32, out_rows, ACT_N_TILE)
        o_partial[t0 : t0 + ACT_T_TILE, n0 : n0 + ACT_N_TILE] = o_partial_valid
    return o_partial, completion_tid


@pl.jit
def decode_sharded_o_projection_test(
    attention_local_groups: pl.Tensor[[LOCAL_O_GROUPS, GROUP_T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    local_t: pl.Scalar[pl.INT32],
    o_partial: pl.InOut[pl.Tensor[[GROUP_T_PAD, D], pl.FP32]],
):
    """Run one receive-side output projection at a runtime token count."""
    o_partial, completion_tid = decode_sharded_o_projection(
        attention_local_groups, wo_a, wo_b, wo_b_scale,
        local_t, o_partial,
    )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_o_projection_complete", deps=[completion_tid]):
        completion_anchor = pl.read(o_partial, [0, 0])
        pl.write(o_partial, [0, 0], completion_anchor)
    return o_partial


def build_sharded_o_proj_tensor_specs(local_t):
    """Build deterministic capacity-static inputs with a poisoned receive tail."""
    import torch

    from golden import ScalarSpec, TensorSpec

    if local_t < 1 or local_t > LOCAL_T:
        raise ValueError(f"local_t must be in [1, {LOCAL_T}], got {local_t}")
    group_t = TP_SIZE * local_t

    def init_attention_local_groups():
        shape = (LOCAL_O_GROUPS, GROUP_T_PAD, O_GROUP_IN)
        values = torch.arange(LOCAL_O_GROUPS * GROUP_T_PAD * O_GROUP_IN, dtype=torch.int32)
        attention = values.remainder(31).sub(15).reshape(shape).to(torch.bfloat16)
        attention[:, :group_t, 0] = 127.0
        attention[:, group_t:, :] = float("nan")
        return attention

    def init_wo_a():
        weight = torch.zeros(LOCAL_O_GROUPS, O_LORA, O_GROUP_IN, dtype=torch.bfloat16)
        diagonal = torch.arange(O_LORA)
        for local_group in range(LOCAL_O_GROUPS):
            group_scale = local_group + 1
            for k_block, coefficient in enumerate((0.5, -0.25, 0.125, -0.0625)):
                k_diagonal = diagonal + k_block * O_LORA
                weight[local_group, diagonal, k_diagonal] = group_scale * coefficient
        return weight

    def init_wo_b():
        values = torch.arange(D * LOCAL_O_WIDTH, dtype=torch.int32)
        return values.remainder(7).sub(3).reshape(D, LOCAL_O_WIDTH).to(torch.int8)

    def init_o_partial():
        return torch.zeros(GROUP_T_PAD, D)

    def init_wo_b_scale():
        values = torch.arange(D, dtype=torch.int32).remainder(4).to(torch.float32)
        return values * 0.25 + 0.5

    attention_shape = [LOCAL_O_GROUPS, GROUP_T_PAD, O_GROUP_IN]
    return [
        TensorSpec("attention_local_groups", attention_shape, torch.bfloat16, init_value=init_attention_local_groups),
        TensorSpec("wo_a", [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, LOCAL_O_WIDTH], torch.int8, init_value=init_wo_b),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=init_wo_b_scale),
        ScalarSpec("local_t", torch.int32, local_t),
        TensorSpec("o_partial", [GROUP_T_PAD, D], torch.float32, init_value=init_o_partial, is_output=True),
    ]


def golden_decode_sharded_o_projection(tensors):
    """Compute the per-group A8 projection over the compact receive prefix."""
    import torch

    group_t = TP_SIZE * int(tensors["local_t"])
    attention = tensors["attention_local_groups"][:, :group_t].float()
    wo_a = tensors["wo_a"].float()
    o_a = torch.einsum("gti,gri->gtr", attention, wo_a)
    row_amax = o_a.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_q = INT8_SCALE_MAX / row_amax
    o_a_i8 = torch.round(o_a * scale_q).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dq = 1.0 / scale_q
    wo_b = tensors["wo_b"].reshape(D, LOCAL_O_GROUPS, O_LORA)
    o_partial = torch.zeros(group_t, D, dtype=torch.float32)
    for local_group in range(LOCAL_O_GROUPS):
        group_i32 = o_a_i8[local_group].to(torch.int32)
        weight_i32 = wo_b[:, local_group].to(torch.int32)
        group_partial = group_i32 @ weight_i32.T
        o_partial = o_partial + group_partial.float() * scale_dq[local_group]
    o_partial = o_partial * tensors["wo_b_scale"].float().unsqueeze(0)
    tensors["o_partial"][:group_t] = o_partial


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run_jit
    from pypto.ir.distributed_compiled_program import DistributedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=("a2a3", "a2a3sim", "a5", "a5sim"))
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(TP_SIZE)))
    parser.add_argument("--mode", choices=("all", "collectives", "sharded"), default="all")
    parser.add_argument(
        "--case", choices=("all", "max", "subcapacity"), default=None,
        help="sharded-mode projection cases",
    )
    parser.add_argument(
        "--local-t", type=int, default=None,
        help="collectives-mode local token count",
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument(
        "--runtime-dir", type=str, default=None,
        help="collectives-mode prebuilt runtime directory",
    )
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(device) for device in args.device.split(",")]
    if args.mode in ("all", "collectives") and len(device_ids) != TP_SIZE:
        parser.error(f"need exactly {TP_SIZE} devices, got {device_ids}")
    if args.mode == "sharded" and args.local_t is not None:
        parser.error("--local-t requires --mode all or --mode collectives")
    if args.mode == "sharded" and args.runtime_dir is not None:
        parser.error("--runtime-dir requires --mode all or --mode collectives")
    if args.mode == "collectives" and args.case is not None:
        parser.error("--case requires --mode all or --mode sharded")

    collective_local_t = FIXTURE_LOCAL_T if args.local_t is None else args.local_t
    if args.mode in ("all", "collectives") and not 1 <= collective_local_t <= LOCAL_T:
        parser.error(f"--local-t must be in [1, {LOCAL_T}], got {collective_local_t}")

    if args.mode in ("all", "collectives"):
        result = run_jit(
            fn=l3_decode_attention_collectives_fixture,
            specs=build_collective_tensor_specs(collective_local_t),
            golden_fn=golden_decode_attention_collectives_fixture,
            compile_only=args.compile_only,
            runtime_dir=args.runtime_dir,
            compile_cfg=dict(
                dump_passes=args.dump_passes,
                distributed_config=DistributedConfig(
                    device_ids=device_ids,
                    num_sub_workers=0,
                ),
            ),
            runtime_cfg=dict(platform=args.platform),
            rtol=0.0,
            atol=0.0,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)

    if args.mode in ("all", "sharded"):
        case_local_t = {"max": LOCAL_T, "subcapacity": LOCAL_T - 1}
        selected_case = "all" if args.case is None else args.case
        selected_cases = tuple(case_local_t) if selected_case == "all" else (selected_case,)
        for case in selected_cases:
            local_t = case_local_t[case]
            group_t = TP_SIZE * local_t
            partial_compare = ratio_allclose(
                atol=1e-4,
                rtol=1.0 / 128,
                max_error_ratio=0.0,
                valid_rows=group_t,
                zero_tail=True,
            )
            result = run_jit(
                fn=decode_sharded_o_projection_test,
                specs=build_sharded_o_proj_tensor_specs(local_t),
                golden_fn=golden_decode_sharded_o_projection,
                compile_only=args.compile_only,
                compile_cfg=dict(dump_passes=args.dump_passes),
                runtime_cfg=dict(platform=args.platform, device_id=device_ids[0]),
                compare_fn={"o_partial": partial_compare},
            )
            if not result.passed:
                if result.error:
                    print(result.error)
                raise SystemExit(1)
