# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from math import gcd

import sys

from . import config

_TP_CHOICES = (1, 2, 4)

_TP_DEFAULT = 2


def _parse_tp_argv():
    for index, arg in enumerate(sys.argv):
        if arg == "--tp" and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if arg.startswith("--tp="):
            return int(arg.split("=", 1)[1])
    return _TP_DEFAULT


TP_SIZE = _parse_tp_argv()

if TP_SIZE not in _TP_CHOICES:
    raise ValueError(f"--tp must be one of {_TP_CHOICES} (got {TP_SIZE})")

config.TP = TP_SIZE

import pypto.language as pl

import pypto.language.distributed as pld

from .config import DECODE_TOKENS, FLASH as M, INT8_AMAX_EPS, INT8_SCALE_MAX

D = M.hidden_size

H = M.num_attention_heads

HEAD_DIM = M.head_dim

O_LORA = M.o_lora_rank

O_GROUPS = M.o_groups

HEADS_PER_GROUP = H // O_GROUPS

O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

LOCAL_T = DECODE_TOKENS // TP_SIZE

LOCAL_O_GROUPS = O_GROUPS // TP_SIZE

LOCAL_O_WIDTH = LOCAL_O_GROUPS * O_LORA

T_DYN = pl.dynamic("NATIVE_D_T_DYN")

TOKEN_TILE = 16

COMM_ROW_TILE = 8

ATTENTION_PUBLISH_WORKERS = 48

O_RS_REDUCE_WORKERS = 48  # 128 row blocks; 8 left 40 of 48 AIV idle

O_RS_PUBLISH_WORKERS = 32

O_RS_DEQUANT_WORKERS = 12  # per owner; 4 owners -> one AIV wave

O_RS_PUT_T_TILE = 8

O_RS_D_TILE = gcd(4096, D)

LOCAL_T_PAD = (LOCAL_T + TOKEN_TILE - 1) // TOKEN_TILE * TOKEN_TILE

T_PAD = LOCAL_T_PAD

GROUP_T_PAD = TP_SIZE * LOCAL_T_PAD

ATTENTION_WINDOW_ROWS = LOCAL_O_GROUPS * GROUP_T_PAD

O_WINDOW_ROWS = TP_SIZE * LOCAL_T_PAD


def _token_tile(capacity, ceiling):
    """Largest TOKEN_TILE multiple up to `ceiling` that divides `capacity`."""
    for tile in range(min(ceiling, capacity) // TOKEN_TILE * TOKEN_TILE, 0, -TOKEN_TILE):
        if capacity % tile == 0:
            return tile
    return TOKEN_TILE


PROJ_B_MM_T_TILE = _token_tile(T_PAD, 128)

O_A_T_TILE = 128

O_A_K_TILE = 256

O_A_N_TILE = 128

QUANT_T_TILE = 8

O_A_QUANT_WORKERS = 6  # per owner-group; 4 x 2 x 6 -> one AIV wave

O_B_T_TILE = _token_tile(LOCAL_T_PAD, 128)

O_B_K_TILE = 256

O_B_N_TILE = 256

O_B_D_TILE = 512

ACT_T_TILE = 16

ACT_N_TILE = 512

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

if D % O_RS_D_TILE != 0:
    raise ValueError(f"O-B ReduceScatter tile {O_RS_D_TILE} must divide hidden size {D}")

if O_RS_REDUCE_WORKERS % TP_SIZE != 0:
    raise ValueError(f"TP size {TP_SIZE} must divide O-B ReduceScatter workers {O_RS_REDUCE_WORKERS}")

if O_RS_PUBLISH_WORKERS % TP_SIZE != 0:
    raise ValueError(f"TP size {TP_SIZE} must divide O-B publish workers {O_RS_PUBLISH_WORKERS}")

if LOCAL_T_PAD % O_B_T_TILE != 0:
    raise ValueError(f"O-B token tile {O_B_T_TILE} must divide local token capacity {LOCAL_T_PAD}")

if GROUP_T_PAD % O_B_T_TILE != 0:
    raise ValueError(f"O-B token tile {O_B_T_TILE} must divide token capacity {GROUP_T_PAD}")

if T_PAD % PROJ_B_MM_T_TILE != 0:
    raise ValueError(f"proj_b_mm token tile {PROJ_B_MM_T_TILE} must divide token capacity {T_PAD}")


@pl.jit.incore
def o_group_a2a_gather(
    local_groups_out: pl.Out[pl.Tensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16]],
    exchange_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    group_t: pl.Scalar[pl.INT32],
):
    """Copy exchanged attention groups into the local projection buffer."""
    worker = pl.tile.get_block_idx()
    for local_group in pl.range(LOCAL_O_GROUPS):
        group_base_row = local_group * GROUP_T_PAD
        for group_row in pl.range(worker, group_t, ATTENTION_PUBLISH_WORKERS):
            copy_row = group_base_row + group_row
            local_groups_out[
                copy_row : copy_row + 1,
                0:O_GROUP_IN,
            ] = exchange_window[
                copy_row : copy_row + 1,
                0:O_GROUP_IN,
            ]
    return local_groups_out


@pl.jit.inline
def o_group_a2a(
    local_groups_out: pl.Tensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    exchange_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    exchange_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
    publish_dep: pl.Scalar[pl.TASK_ID],
    publish_count: pl.Scalar[pl.INT32],
):
    """Finish a non-overlapping producer-fused exchange and release its window."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="o_group_a2a_wait", deps=[publish_dep]) as wait_tid:
        expected = pl.cast(publish_count, pl.INT32)
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.wait(
                    signal=exchange_signal, offsets=[source_tp, 0], expected=expected, cmp=pld.WaitCmp.Ge
                )

    group_t = TP_SIZE * local_t
    # Specialize the explicit SPMD callee on a scratch output.
    if False:
        gather_specialize = pl.create_tensor([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
        o_group_a2a_gather(gather_specialize, exchange_window, group_t)
    local_groups_out, gather_tid = pl.spmd_submit(
        self.o_group_a2a_gather,  # noqa: F821 - materialized by @pl.jit
        local_groups_out,
        pl.no_dep(exchange_window),
        group_t,
        core_num=ATTENTION_PUBLISH_WORKERS,
        deps=[wait_tid],
    )

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="o_group_a2a_complete",
        deps=[gather_tid],
        no_dep_args=[exchange_signal],
    ):
        completion_anchor = pl.read(local_groups_out, [0, 0])
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=exchange_signal,
                    peer=group_base + peer_tp,
                    offsets=[tp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

        completion_expected = pl.cast(publish_count + 1, pl.INT32)
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.wait(
                    signal=exchange_signal,
                    offsets=[source_tp, 0],
                    expected=completion_expected,
                    cmp=pld.WaitCmp.Ge,
                )

        reset_value = pl.cast(-completion_expected, pl.INT32)
        self_rank = group_base + tp_rank
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.notify(
                    target=exchange_signal,
                    peer=self_rank,
                    offsets=[source_tp, 0],
                    value=reset_value,
                    op=pld.NotifyOp.AtomicAdd,
                )
        pl.write(local_groups_out, [0, 0], completion_anchor)
    return local_groups_out, exchange_signal


@pl.jit.incore
def tp_o_rs_reduce(
    local_out: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    reduce_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.BF16],
    local_t: pl.Scalar[pl.INT32],
):
    """Sum the received O-projection partials for the local tokens."""
    worker = pl.tile.get_block_idx()
    for block in pl.range(worker, local_t * (D // O_RS_D_TILE), O_RS_REDUCE_WORKERS):
        local_row = block // (D // O_RS_D_TILE)
        d_block = block - local_row * (D // O_RS_D_TILE)
        d0 = d_block * O_RS_D_TILE
        own_partial = pl.load(reduce_window, [local_row, d0], [1, O_RS_D_TILE])
        reduce_acc = pl.cast(own_partial, target_type=pl.FP32, mode="none")
        for source_tp in pl.range(1, TP_SIZE):
            source_row = source_tp * LOCAL_T_PAD + local_row
            source_partial = pl.load(reduce_window, [source_row, d0], [1, O_RS_D_TILE])
            source_fp32 = pl.cast(source_partial, target_type=pl.FP32, mode="none")
            reduce_acc = pl.add(reduce_acc, source_fp32)
        reduced = pl.cast(reduce_acc, target_type=pl.BF16, mode="rint")
        pl.store(reduced, [local_row, d0], local_out)
    return local_out


@pl.jit.inline
def o_proj_reduce_scatter(
    attention_local_groups: pl.Tensor[[LOCAL_O_GROUPS, GROUP_T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[LOCAL_O_GROUPS, D, O_LORA], pl.INT8, pl.NZ],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    local_t: pl.Scalar[pl.INT32],
    local_out: pl.Tensor[[T_DYN, D], pl.BF16],
    reduce_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.BF16],
    reduce_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Project O-B tiles directly into their ReduceScatter owner windows."""
    group_t = TP_SIZE * local_t
    o_a_rows = (group_t + O_A_T_TILE - 1) // O_A_T_TILE
    o_b_rows = (group_t + O_B_T_TILE - 1) // O_B_T_TILE
    o_b_group_t = o_b_rows * O_B_T_TILE
    owner_rows = (local_t + ACT_T_TILE - 1) // ACT_T_TILE

    attn_2d = pl.reshape(attention_local_groups, [LOCAL_O_GROUPS * GROUP_T_PAD, O_GROUP_IN])
    # Owner-private buffers and group-local A -> quant -> B dependencies.
    publish_all = pl.create_tensor([O_WINDOW_ROWS, D], dtype=pl.BF16)
    put_rows = (local_t + O_RS_PUT_T_TILE - 1) // O_RS_PUT_T_TILE
    own_a_rows = (local_t + O_A_T_TILE - 1) // O_A_T_TILE
    own_b_rows = (local_t + O_B_T_TILE - 1) // O_B_T_TILE
    own_b_t = own_b_rows * O_B_T_TILE
    own_quant_blocks = (local_t + QUANT_T_TILE - 1) // QUANT_T_TILE
    own_pad_blocks = (own_b_t + QUANT_T_TILE - 1) // QUANT_T_TILE
    own_act_rows = (local_t + ACT_T_TILE - 1) // ACT_T_TILE

    for owner in pl.parallel(TP_SIZE):
        own_base = owner * local_t
        own_a_fp32 = pl.create_tensor([LOCAL_T_PAD, LOCAL_O_WIDTH], dtype=pl.FP32, manual_dep=True)
        own_a_i8 = pl.create_tensor([LOCAL_T_PAD, LOCAL_O_WIDTH], dtype=pl.INT8, manual_dep=True)
        # The quant scale rides the own_a_i8 -> tp_o_b -> dequant chain.
        own_scale = pl.create_tensor([LOCAL_O_GROUPS, LOCAL_T_PAD], dtype=pl.FP32, manual_dep=True)
        own_b_i32 = pl.create_tensor([LOCAL_T_PAD, LOCAL_O_GROUPS * D], dtype=pl.INT32)

        for local_group in pl.parallel(LOCAL_O_GROUPS):
            attention_row = local_group * GROUP_T_PAD + own_base
            o_a_col = local_group * O_LORA

            with pl.spmd(own_a_rows * (O_LORA // O_A_N_TILE), name_hint="tp_o_a") as pa_tid:
                # Weight reads bypass L2.
                pl.set_cache_policy(wo_a, pl.CachePolicy.BYPASS)
                pa_unit = pl.tile.get_block_idx()
                pa_rb = pa_unit // (O_LORA // O_A_N_TILE)
                pa_nb = pa_unit % (O_LORA // O_A_N_TILE)
                pa_t0 = pa_rb * O_A_T_TILE
                pa_n0 = pa_nb * O_A_N_TILE
                pa_rows = pl.min(O_A_T_TILE, local_t - pa_t0)
                pa_src = attention_row + pa_t0
                pa_wrow = o_a_col + pa_n0
                pa_x0 = pl.slice(
                    attn_2d, [O_A_T_TILE, O_A_K_TILE], [pa_src, 0], valid_shape=[pa_rows, O_A_K_TILE]
                )
                pa_w0 = wo_a[local_group : local_group + 1, pa_n0 : pa_n0 + O_A_N_TILE, 0:O_A_K_TILE]
                pa_acc = pl.matmul(pa_x0, pa_w0, b_trans=True, out_dtype=pl.FP32)
                for pa_k0 in pl.pipeline(O_A_K_TILE, O_GROUP_IN, O_A_K_TILE, stage=2):
                    pa_xk = pl.slice(
                        attn_2d, [O_A_T_TILE, O_A_K_TILE], [pa_src, pa_k0], valid_shape=[pa_rows, O_A_K_TILE]
                    )
                    pa_wk = wo_a[
                        local_group : local_group + 1, pa_n0 : pa_n0 + O_A_N_TILE, pa_k0 : pa_k0 + O_A_K_TILE
                    ]
                    pa_acc = pl.matmul_acc(pa_acc, pa_xk, pa_wk, b_trans=True)
                pa_acc_2d = pl.reshape(pa_acc, [O_A_T_TILE, O_A_N_TILE])
                pa_valid = pl.set_validshape(pa_acc_2d, pa_rows, O_A_N_TILE)
                own_a_fp32[pa_t0 : pa_t0 + O_A_T_TILE, pa_wrow : pa_wrow + O_A_N_TILE] = pa_valid

            with pl.spmd(O_A_QUANT_WORKERS, name_hint="tp_o_a_quant", deps=[pa_tid]) as q_tid:
                qz_worker = pl.tile.get_block_idx()
                for qz_blk in pl.range(qz_worker, own_quant_blocks, O_A_QUANT_WORKERS):
                    qz_t = qz_blk * QUANT_T_TILE
                    qz_rows = pl.min(QUANT_T_TILE, local_t - qz_t)
                    qz_tile = pl.slice(own_a_fp32, [QUANT_T_TILE, O_LORA], [qz_t, o_a_col])
                    qz_amax = pl.reshape(pl.row_max(pl.abs(qz_tile)), [1, QUANT_T_TILE])
                    qz_floor = pl.full([1, QUANT_T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                    qz_amax = pl.maximum(qz_floor, qz_amax)
                    qz_max = pl.full([1, QUANT_T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
                    qz_sq = pl.div(qz_max, qz_amax)
                    qz_sdq = pl.recip(qz_sq)
                    own_scale[local_group : local_group + 1, qz_t : qz_t + QUANT_T_TILE] = pl.set_validshape(
                        qz_sdq, 1, qz_rows
                    )
                    qz_sq_col = pl.reshape(qz_sq, [QUANT_T_TILE, 1])
                    qz_scaled = pl.row_expand_mul(qz_tile, qz_sq_col)
                    qz_i32 = pl.cast(qz_scaled, target_type=pl.INT32, mode="rint")
                    qz_f16 = pl.cast(qz_i32, target_type=pl.FP16, mode="round")
                    qz_i8 = pl.cast(qz_f16, target_type=pl.INT8, mode="trunc")
                    own_a_i8[qz_t : qz_t + QUANT_T_TILE, o_a_col : o_a_col + O_LORA] = pl.set_validshape(
                        qz_i8, qz_rows, O_LORA
                    )
                for qz_pad in pl.range(own_quant_blocks + qz_worker, own_pad_blocks, O_A_QUANT_WORKERS):
                    qz_pt = qz_pad * QUANT_T_TILE
                    qz_prows = pl.min(QUANT_T_TILE, own_b_t - qz_pt)
                    qz_zero = pl.full([QUANT_T_TILE, O_LORA], dtype=pl.FP16, value=0.0)
                    qz_zero_i8 = pl.cast(qz_zero, target_type=pl.INT8, mode="trunc")
                    own_a_i8[qz_pt : qz_pt + QUANT_T_TILE, o_a_col : o_a_col + O_LORA] = pl.set_validshape(
                        qz_zero_i8, qz_prows, O_LORA
                    )

            with pl.spmd(own_b_rows * (D // O_B_D_TILE), name_hint="tp_o_b", deps=[q_tid]):
                # Weight reads bypass L2.
                pl.set_cache_policy(wo_b, pl.CachePolicy.BYPASS)
                pb_unit = pl.tile.get_block_idx()
                pb_tb = pb_unit // (D // O_B_D_TILE)
                pb_db = pb_unit % (D // O_B_D_TILE)
                pb_t0 = pb_tb * O_B_T_TILE
                pb_d0 = pb_db * O_B_D_TILE
                for pb_n0 in pl.range(pb_d0, pb_d0 + O_B_D_TILE, O_B_N_TILE):
                    pb_x0 = own_a_i8[pb_t0 : pb_t0 + O_B_T_TILE, o_a_col : o_a_col + O_B_K_TILE]
                    pb_w0 = wo_b[local_group : local_group + 1, pb_n0 : pb_n0 + O_B_N_TILE, 0:O_B_K_TILE]
                    pb_acc = pl.matmul(pb_x0, pb_w0, b_trans=True, out_dtype=pl.INT32)
                    for pb_k0 in pl.pipeline(O_B_K_TILE, O_LORA, O_B_K_TILE, stage=2):
                        pb_bk = o_a_col + pb_k0
                        pb_xk = own_a_i8[pb_t0 : pb_t0 + O_B_T_TILE, pb_bk : pb_bk + O_B_K_TILE]
                        pb_wk = wo_b[
                            local_group : local_group + 1,
                            pb_n0 : pb_n0 + O_B_N_TILE,
                            pb_k0 : pb_k0 + O_B_K_TILE,
                        ]
                        pb_acc = pl.matmul_acc(pb_acc, pb_xk, pb_wk, b_trans=True)
                    pb_acc_2d = pl.reshape(pb_acc, [O_B_T_TILE, O_B_N_TILE])
                    pb_col = local_group * D + pb_n0
                    own_b_i32[pb_t0 : pb_t0 + O_B_T_TILE, pb_col : pb_col + O_B_N_TILE] = pb_acc_2d

        with pl.spmd(
            O_RS_DEQUANT_WORKERS,
            name_hint="tp_o_b_dequant",
            optimizations=[pl.cross_core_slot(slot_num=2)],
        ):
            dq_worker = pl.tile.get_block_idx()
            for dq_blk in pl.range(dq_worker, own_act_rows * (D // ACT_N_TILE), O_RS_DEQUANT_WORKERS):
                dq_rb = dq_blk // (D // ACT_N_TILE)
                dq_nb = dq_blk - dq_rb * (D // ACT_N_TILE)
                dq_row = dq_rb * ACT_T_TILE
                dq_n0 = dq_nb * ACT_N_TILE
                dq_rows = pl.min(ACT_T_TILE, local_t - dq_row)
                dq_acc = pl.full([ACT_T_TILE, ACT_N_TILE], dtype=pl.FP32, value=0.0)
                for dq_group in pl.pipeline(LOCAL_O_GROUPS, stage=2):
                    dq_col = dq_group * D + dq_n0
                    dq_i32 = pl.slice(
                        own_b_i32,
                        [ACT_T_TILE, ACT_N_TILE],
                        [dq_row, dq_col],
                        valid_shape=[dq_rows, ACT_N_TILE],
                    )
                    dq_fp32 = pl.cast(dq_i32, target_type=pl.FP32, mode="none")
                    dq_srow = pl.slice(
                        own_scale, [1, ACT_T_TILE], [dq_group, dq_row], valid_shape=[1, dq_rows]
                    )
                    dq_scol = pl.reshape(dq_srow, [ACT_T_TILE, 1])
                    dq_acc = pl.add(dq_acc, pl.row_expand_mul(dq_fp32, dq_scol))
                dq_wscale = pl.reshape(wo_b_scale[dq_n0 : dq_n0 + ACT_N_TILE], [1, ACT_N_TILE])
                dq_bf16 = pl.cast(pl.col_expand_mul(dq_acc, dq_wscale), target_type=pl.BF16, mode="rint")
                dq_stage = owner * LOCAL_T_PAD + dq_row
                publish_all[dq_stage : dq_stage + ACT_T_TILE, dq_n0 : dq_n0 + ACT_N_TILE] = pl.set_validshape(
                    dq_bf16, dq_rows, ACT_N_TILE
                )

    with pl.spmd(
        O_RS_PUBLISH_WORKERS,
        name_hint="tp_o_b_publish",
    ) as publish_tid:
        pub_worker = pl.tile.get_block_idx()
        # Flatten (owner, row block) into one work list: put_rows alone is under
        # the worker count, so an owner-outer loop leaves a third of the workers
        # idle while the rest each issue TP_SIZE puts.
        for pub_blk in pl.range(pub_worker, TP_SIZE * put_rows, O_RS_PUBLISH_WORKERS):
            pub_owner = pub_blk // put_rows
            pub_row_block = pub_blk - pub_owner * put_rows
            pub_owner_row = pub_row_block * O_RS_PUT_T_TILE
            pub_rows = pl.min(O_RS_PUT_T_TILE, local_t - pub_owner_row)
            pub_src_row = pub_owner * LOCAL_T_PAD + pub_owner_row
            pub_dst_row = tp_rank * LOCAL_T_PAD + pub_owner_row
            pld.tensor.put(
                dst=reduce_window,
                peer=group_base + pub_owner,
                src=publish_all,
                dst_offsets=[pub_dst_row, 0],
                src_offsets=[pub_src_row, 0],
                shape=[pub_rows, D],
                chunk_rows=O_RS_PUT_T_TILE,
                chunk_cols=D,
            )

        for notify_owner in pl.range(TP_SIZE):
            if notify_owner != tp_rank:
                pld.system.notify(
                    target=reduce_signal,
                    peer=group_base + notify_owner,
                    offsets=[tp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_o_rs_wait", deps=[publish_tid]) as wait_tid:
        expected = pl.cast(O_RS_PUBLISH_WORKERS, pl.INT32)
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.wait(
                    signal=reduce_signal, offsets=[source_tp, 0], expected=expected, cmp=pld.WaitCmp.Ge
                )

    # Specialize the explicit SPMD callee on a scratch output.
    if False:
        reduce_output_rows = pl.tensor.dim(local_out, 0)
        reduce_specialize = pl.create_tensor([reduce_output_rows, D], dtype=pl.BF16)
        tp_o_rs_reduce(reduce_specialize, reduce_window, local_t)
    local_out, reduce_tid = pl.spmd_submit(
        self.tp_o_rs_reduce,  # noqa: F821 - materialized by @pl.jit
        local_out,
        pl.no_dep(reduce_window),
        local_t,
        core_num=O_RS_REDUCE_WORKERS,
        deps=[wait_tid],
    )

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="tp_o_rs_complete",
        deps=[reduce_tid],
        no_dep_args=[reduce_signal],
    ):
        completion_anchor = pl.read(local_out, [0, 0])
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=reduce_signal,
                    peer=group_base + peer_tp,
                    offsets=[tp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

        completion_expected = pl.cast(O_RS_PUBLISH_WORKERS + 1, pl.INT32)
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.wait(
                    signal=reduce_signal,
                    offsets=[source_tp, 0],
                    expected=completion_expected,
                    cmp=pld.WaitCmp.Ge,
                )

        reset_value = pl.cast(-(O_RS_PUBLISH_WORKERS + 1), pl.INT32)
        self_rank = group_base + tp_rank
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.notify(
                    target=reduce_signal,
                    peer=self_rank,
                    offsets=[source_tp, 0],
                    value=reset_value,
                    op=pld.NotifyOp.AtomicAdd,
                )
        pl.write(local_out, [0, 0], completion_anchor)
    return local_out, reduce_signal
