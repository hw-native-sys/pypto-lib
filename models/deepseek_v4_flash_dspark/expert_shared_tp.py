# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""DeepSeek-V4 shared expert with column-parallel gate/up, local SwiGLU, and row-parallel down all-reduce."""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from config import DECODE_TOKENS, FLASH as M, INT8_AMAX_EPS, INT8_SCALE_MAX


# command-line config
_TP_CHOICES = (2, 4, 8)
_TP_DEFAULT = 2
_ALLREDUCE_MODES = ("parallel-mesh", "parallel-doubling", "mesh", "ring")


def _parse_tp_argv():
    for index, token in enumerate(sys.argv):
        if token == "--tp" and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if token.startswith("--tp="):
            return int(token.split("=", 1)[1])
    return _TP_DEFAULT


def _parse_allreduce_mode_argv(tp_size):
    for index, token in enumerate(sys.argv):
        if token == "--allreduce-mode" and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if token.startswith("--allreduce-mode="):
            return token.split("=", 1)[1]
    return "parallel-doubling" if tp_size == 8 else "parallel-mesh"


# distributed config
TP_SIZE = _parse_tp_argv()
if TP_SIZE not in _TP_CHOICES:
    raise ValueError(f"--tp must be one of {_TP_CHOICES}, got {TP_SIZE}")
ALLREDUCE_MODE = _parse_allreduce_mode_argv(TP_SIZE)
if ALLREDUCE_MODE not in _ALLREDUCE_MODES:
    raise ValueError(f"--allreduce-mode must be one of {_ALLREDUCE_MODES}, got {ALLREDUCE_MODE}")
USE_RING_ALLREDUCE = ALLREDUCE_MODE == "ring"
USE_PARALLEL_MESH = ALLREDUCE_MODE == "parallel-mesh"
USE_PARALLEL_DOUBLING = ALLREDUCE_MODE == "parallel-doubling"

# model config
T = DECODE_TOKENS
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
LOCAL_INTER = MOE_INTER // TP_SIZE
SWIGLU_LIMIT = M.swiglu_limit

# tiling
SH_M_TILE = 32
SH_AMAX_TILE = 8
SH_ACT_M_TILE = 2
T_PAD = ((T + SH_M_TILE - 1) // SH_M_TILE) * SH_M_TILE
SH_VALID_M = T if T < SH_M_TILE else SH_M_TILE

K_TILE = 512
MM_INTER_TILE = 256
ACT_INTER_TILE = min(1024, LOCAL_INTER)
QUANT_TILE = LOCAL_INTER
DOWN_K_TILE = min(512, LOCAL_INTER // 2)
D_OUT_TILE = 256
W2_ACT_TILE = 512
W2_GROUP_TILE = 1024
COMM_STAGE_TILE = 4096

# communication
COMM_WORKERS = min(24, T * (D // COMM_STAGE_TILE))
SIGNAL_ROWS = 2 * (TP_SIZE - 1) if USE_RING_ALLREDUCE else TP_SIZE
SIGNAL_COLS = TP_SIZE if USE_RING_ALLREDUCE else 1

if MOE_INTER % TP_SIZE != 0:
    raise ValueError(f"intermediate size {MOE_INTER} is not divisible by TP {TP_SIZE}")
if T > SH_M_TILE and T % SH_M_TILE != 0:
    raise ValueError("token rows must fit one M tile or be an exact M-tile multiple")
if SH_VALID_M % SH_ACT_M_TILE != 0:
    raise ValueError("valid M rows must be divisible by rows per activation block")
if LOCAL_INTER % MM_INTER_TILE != 0:
    raise ValueError("local intermediate size must be divisible by the matmul N tile")
if LOCAL_INTER % ACT_INTER_TILE != 0:
    raise ValueError("local intermediate size must be divisible by the activation tile")
if LOCAL_INTER % QUANT_TILE != 0:
    raise ValueError("local intermediate size must be divisible by the quantization tile")
if LOCAL_INTER % DOWN_K_TILE != 0:
    raise ValueError("local intermediate size must be divisible by the down K tile")
if D % COMM_STAGE_TILE != 0:
    raise ValueError("hidden size must be divisible by the communication staging tile")
if (T * (D // COMM_STAGE_TILE)) % TP_SIZE != 0:
    raise ValueError("communication blocks must be divisible by TP size")


@pl.jit.inline
def shared_gate_up_local(
    x_local_i8: pl.Tensor[[T, D], pl.INT8],
    x_local_scale_dq: pl.Tensor[[T, 1], pl.FP32],
    shared_w1: pl.Tensor[[LOCAL_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[LOCAL_INTER], pl.FP32],
    shared_w3: pl.Tensor[[LOCAL_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[LOCAL_INTER], pl.FP32],
    h_local: pl.Tensor[[T, LOCAL_INTER], pl.FP32],
    local_amax: pl.Tensor[[T, SH_AMAX_TILE], pl.FP32],
):
    """Compute the local gate/up shard and its row amax values."""
    for mt in pl.parallel(T_PAD // SH_M_TILE):
        ts0 = mt * SH_M_TILE

        gate_i32 = pl.create_tensor([SH_M_TILE, LOCAL_INTER], dtype=pl.INT32)
        for nb_idx in pl.spmd(LOCAL_INTER // MM_INTER_TILE, name_hint="sh_tp_gate_mm", allow_early_resolve=True):
            n0 = nb_idx * MM_INTER_TILE
            gate_acc = pl.create_tensor([SH_M_TILE, MM_INTER_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                xs_k = pl.slice(x_local_i8, [SH_M_TILE, K_TILE], [ts0, k0], valid_shape=[SH_VALID_M, K_TILE])
                sw1_k = shared_w1[n0 : n0 + MM_INTER_TILE, k0 : k0 + K_TILE]
                if k0 == 0:
                    gate_acc = pl.matmul(xs_k, sw1_k, b_trans=True, out_dtype=pl.INT32)
                else:
                    gate_acc = pl.matmul_acc(gate_acc, xs_k, sw1_k, b_trans=True)
            gate_i32[:, n0 : n0 + MM_INTER_TILE] = gate_acc

        up_i32 = pl.create_tensor([SH_M_TILE, LOCAL_INTER], dtype=pl.INT32)
        for nb_idx in pl.spmd(LOCAL_INTER // MM_INTER_TILE, name_hint="sh_tp_up_mm", allow_early_resolve=True):
            n0 = nb_idx * MM_INTER_TILE
            up_acc = pl.create_tensor([SH_M_TILE, MM_INTER_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                xs_k = pl.slice(x_local_i8, [SH_M_TILE, K_TILE], [ts0, k0], valid_shape=[SH_VALID_M, K_TILE])
                sw3_k = shared_w3[n0 : n0 + MM_INTER_TILE, k0 : k0 + K_TILE]
                if k0 == 0:
                    up_acc = pl.matmul(xs_k, sw3_k, b_trans=True, out_dtype=pl.INT32)
                else:
                    up_acc = pl.matmul_acc(up_acc, xs_k, sw3_k, b_trans=True)
            up_i32[:, n0 : n0 + MM_INTER_TILE] = up_acc

        for row_block in pl.spmd(SH_VALID_M // SH_ACT_M_TILE, name_hint="sh_tp_gate_up_act"):
            row0 = row_block * SH_ACT_M_TILE
            row_start = ts0 + row0
            x_scale = pl.slice(x_local_scale_dq, [SH_AMAX_TILE, 1], [row_start, 0], valid_shape=[SH_ACT_M_TILE, 1])
            row_amax = pl.full([1, SH_AMAX_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for part in pl.pipeline(0, LOCAL_INTER // ACT_INTER_TILE, stage=1):
                n0 = part * ACT_INTER_TILE
                gate_rows_i32 = pl.slice(
                    gate_i32, [SH_AMAX_TILE, ACT_INTER_TILE], [row0, n0],
                    valid_shape=[SH_ACT_M_TILE, ACT_INTER_TILE],
                )
                up_rows_i32 = pl.slice(
                    up_i32, [SH_AMAX_TILE, ACT_INTER_TILE], [row0, n0],
                    valid_shape=[SH_ACT_M_TILE, ACT_INTER_TILE],
                )
                w1_scale = pl.reshape(shared_w1_scale[n0 : n0 + ACT_INTER_TILE], [1, ACT_INTER_TILE])
                w3_scale = pl.reshape(shared_w3_scale[n0 : n0 + ACT_INTER_TILE], [1, ACT_INTER_TILE])
                gate_fp32 = pl.cast(gate_rows_i32, target_type=pl.FP32, mode="none")
                up_fp32 = pl.cast(up_rows_i32, target_type=pl.FP32, mode="none")
                gate_fp32 = pl.row_expand_mul(gate_fp32, x_scale)
                gate_fp32 = pl.col_expand_mul(gate_fp32, w1_scale)
                up_fp32 = pl.row_expand_mul(up_fp32, x_scale)
                up_fp32 = pl.col_expand_mul(up_fp32, w3_scale)
                if SWIGLU_LIMIT > 0.0:
                    gate_fp32 = pl.minimum(gate_fp32, SWIGLU_LIMIT)
                    up_fp32 = pl.minimum(up_fp32, SWIGLU_LIMIT)
                    up_fp32 = pl.maximum(up_fp32, -SWIGLU_LIMIT)
                gate_neg = pl.neg(gate_fp32)
                gate_exp = pl.exp(gate_neg)
                sigmoid_den = pl.add(gate_exp, 1.0)
                sigmoid = pl.recip(sigmoid_den)
                silu = pl.mul(gate_fp32, sigmoid)
                gated = pl.mul(silu, up_fp32)
                gated_abs = pl.abs(gated)
                gated_amax = pl.row_max(gated_abs)
                chunk_amax = pl.reshape(gated_amax, [1, SH_AMAX_TILE])
                row_amax = pl.maximum(row_amax, chunk_amax)
                h_local[row_start : row_start + SH_ACT_M_TILE, n0 : n0 + ACT_INTER_TILE] = gated[0:SH_ACT_M_TILE, :]

            row_amax_col = pl.reshape(row_amax, [SH_AMAX_TILE, 1])
            row_amax_zeros = pl.full([SH_AMAX_TILE, SH_AMAX_TILE], dtype=pl.FP32, value=0.0)
            row_amax_matrix = pl.row_expand(row_amax_zeros, row_amax_col)
            local_amax[row_start : row_start + SH_ACT_M_TILE, :] = row_amax_matrix[0:SH_ACT_M_TILE, :]

    return h_local, local_amax


@pl.jit.inline
def shared_down_local(
    h_local: pl.Tensor[[T, LOCAL_INTER], pl.FP32],
    local_amax: pl.Tensor[[T, SH_AMAX_TILE], pl.FP32],
    shared_w2: pl.Tensor[[D, LOCAL_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    partial: pl.Tensor[[T, D], pl.FP32],
):
    """Quantize the local activation shard and emit dequantized partials."""
    for mt in pl.parallel(T_PAD // SH_M_TILE):
        ts0 = mt * SH_M_TILE
        h_tile_i8 = pl.create_tensor([SH_M_TILE, LOCAL_INTER], dtype=pl.INT8, init_value=0)

        for row_block in pl.spmd(SH_VALID_M // SH_ACT_M_TILE, name_hint="sh_tp_act_q"):
            row0 = row_block * SH_ACT_M_TILE
            row_start = ts0 + row0
            row_amax_matrix = pl.slice(
                local_amax, [SH_AMAX_TILE, SH_AMAX_TILE], [row_start, 0],
                valid_shape=[SH_ACT_M_TILE, SH_AMAX_TILE],
            )
            row_amax = pl.row_max(row_amax_matrix)
            row_amax_recip = pl.recip(row_amax)
            row_scale_q = pl.mul(row_amax_recip, INT8_SCALE_MAX)
            for q_idx in pl.pipeline(0, LOCAL_INTER // QUANT_TILE, stage=1):
                k0 = q_idx * QUANT_TILE
                h_fp32 = pl.slice(
                    h_local, [SH_AMAX_TILE, QUANT_TILE], [row_start, k0],
                    valid_shape=[SH_ACT_M_TILE, QUANT_TILE],
                )
                h_scaled = pl.row_expand_mul(h_fp32, row_scale_q)
                h_i32 = pl.cast(h_scaled, target_type=pl.INT32, mode="rint")
                h_fp16 = pl.cast(h_i32, target_type=pl.FP16, mode="round")
                h_i8 = pl.cast(h_fp16, target_type=pl.INT8, mode="trunc")
                h_tile_i8[row0 : row0 + SH_ACT_M_TILE, k0 : k0 + QUANT_TILE] = h_i8[0:SH_ACT_M_TILE, :]

        y_i32 = pl.create_tensor([SH_M_TILE, D], dtype=pl.INT32)
        for db_idx in pl.spmd(D // D_OUT_TILE, name_hint="sh_tp_w2_mm"):
            d0 = db_idx * D_OUT_TILE
            y_acc = pl.create_tensor([SH_M_TILE, D_OUT_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, LOCAL_INTER, DOWN_K_TILE, stage=2):
                hs_k = h_tile_i8[:, k0 : k0 + DOWN_K_TILE]
                sw2_k = shared_w2[d0 : d0 + D_OUT_TILE, k0 : k0 + DOWN_K_TILE]
                if k0 == 0:
                    y_acc = pl.matmul(hs_k, sw2_k, b_trans=True, out_dtype=pl.INT32)
                else:
                    y_acc = pl.matmul_acc(y_acc, hs_k, sw2_k, b_trans=True)
            y_i32[:, d0 : d0 + D_OUT_TILE] = y_acc

        for db_idx in pl.spmd(D // W2_GROUP_TILE, name_hint="sh_tp_w2_partial", allow_early_resolve=True):
            d_base = db_idx * W2_GROUP_TILE
            output_row_amax_matrix = pl.slice(
                local_amax, [SH_M_TILE, SH_AMAX_TILE], [ts0, 0],
                valid_shape=[SH_VALID_M, SH_AMAX_TILE],
            )
            output_row_amax = pl.row_max(output_row_amax_matrix)
            h_scale_dq = pl.div(output_row_amax, INT8_SCALE_MAX)
            for d_offset in pl.pipeline(0, W2_GROUP_TILE, W2_ACT_TILE, stage=2):
                d0 = d_base + d_offset
                y_2d_i32 = y_i32[:, d0 : d0 + W2_ACT_TILE]
                w2_scale = pl.reshape(shared_w2_scale[d0 : d0 + W2_ACT_TILE], [1, W2_ACT_TILE])
                y_2d = pl.cast(y_2d_i32, target_type=pl.FP32, mode="none")
                y_2d = pl.row_expand_mul(y_2d, h_scale_dq)
                y_2d = pl.col_expand_mul(y_2d, w2_scale)
                partial[ts0 : ts0 + SH_VALID_M, d0 : d0 + W2_ACT_TILE] = y_2d[0:SH_VALID_M, :]

    return partial


@pl.jit.inline(auto_scope=False)
def reduce_down_partial_parallel(
    partial: pl.Tensor[[T, D], pl.FP32],
    sh: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    partial_window: pl.InOut[pld.DistributedTensor[[T, D], pl.FP32]],
    signal: pl.InOut[pld.DistributedTensor[[SIGNAL_ROWS, SIGNAL_COLS], pl.INT32]],
    my_rank: pl.Scalar[pl.INT32],
):
    """Sum down partials through a parallel out-of-place mesh."""
    with pl.spmd(COMM_WORKERS, name_hint="sh_tp_partial_publish") as publish_tid:
        comm_core = pl.tile.get_block_idx()
        for block in pl.range(comm_core, T * (D // COMM_STAGE_TILE), COMM_WORKERS):
            row = block // (D // COMM_STAGE_TILE)
            col = block % (D // COMM_STAGE_TILE) * COMM_STAGE_TILE
            pld.tensor.put(
                dst=partial_window, peer=my_rank, src=partial,
                dst_offsets=[row, col], src_offsets=[row, col], shape=[1, COMM_STAGE_TILE],
            )

        for peer in pl.range(TP_SIZE):
            if peer != my_rank:
                pld.system.notify(
                    target=signal, peer=peer, offsets=[my_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_tp_partial_wait", deps=[publish_tid]) as wait_tid:
        for src in pl.range(TP_SIZE):
            if src != my_rank:
                publish_expected = pl.cast(COMM_WORKERS, pl.INT32)
                pld.system.wait(
                    signal=signal, offsets=[src, 0],
                    expected=publish_expected, cmp=pld.WaitCmp.Ge,
                )

    with pl.spmd(COMM_WORKERS, name_hint="sh_tp_partial_reduce", deps=[wait_tid]) as reduce_tid:
        comm_core = pl.tile.get_block_idx()
        for block in pl.range(comm_core, T * (D // COMM_STAGE_TILE), COMM_WORKERS):
            row = block // (D // COMM_STAGE_TILE)
            col = block % (D // COMM_STAGE_TILE) * COMM_STAGE_TILE
            acc = pl.load(partial_window, [row, col], [1, COMM_STAGE_TILE])
            for peer in pl.range(TP_SIZE):
                if peer != my_rank:
                    remote = pld.tile.remote_load(
                        partial_window, peer=peer,
                        offsets=[row, col], shape=[1, COMM_STAGE_TILE],
                    )
                    acc = pl.add(acc, remote)
            reduced_bf16 = pl.cast(acc, target_type=pl.BF16, mode="rint")
            pl.store(reduced_bf16, [row, col], sh)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_tp_partial_complete", deps=[reduce_tid]):
        _mesh_completion_anchor = pl.read(sh, [0, 0])
        for peer in pl.range(TP_SIZE):
            if peer != my_rank:
                pld.system.notify(
                    target=signal, peer=peer, offsets=[my_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )

        for src in pl.range(TP_SIZE):
            if src != my_rank:
                completion_expected = pl.cast(COMM_WORKERS + 1, pl.INT32)
                pld.system.wait(
                    signal=signal, offsets=[src, 0],
                    expected=completion_expected, cmp=pld.WaitCmp.Ge,
                )

        completion_credits = pl.cast(COMM_WORKERS + 1, pl.INT32)
        mesh_reset: pl.Scalar[pl.INT32] = -completion_credits
        for src in pl.range(TP_SIZE):
            if src != my_rank:
                pld.system.notify(
                    target=signal, peer=my_rank, offsets=[src, 0],
                    value=mesh_reset, op=pld.NotifyOp.AtomicAdd,
                )
    return sh


@pl.jit.inline(auto_scope=False)
def doubling_pair_barrier(
    signal: pl.InOut[pld.DistributedTensor[[SIGNAL_ROWS, SIGNAL_COLS], pl.INT32]],
    my_rank: pl.Scalar[pl.INT32],
    partner: pl.Scalar[pl.INT32],
    dep_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Synchronize one recursive-doubling partner pair."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_tp_doubling_pair_barrier", deps=[dep_tid]) as barrier_tid:
        pld.system.notify(
            target=signal, peer=partner, offsets=[my_rank, 0],
            value=1, op=pld.NotifyOp.AtomicAdd,
        )
        pld.system.wait(
            signal=signal, offsets=[partner, 0],
            expected=1, cmp=pld.WaitCmp.Ge,
        )
        pld.system.notify(
            target=signal, peer=my_rank, offsets=[partner, 0],
            value=-1, op=pld.NotifyOp.AtomicAdd,
        )
    return barrier_tid


@pl.jit.inline(auto_scope=False)
def doubling_two_partner_barrier(
    signal: pl.InOut[pld.DistributedTensor[[SIGNAL_ROWS, SIGNAL_COLS], pl.INT32]],
    my_rank: pl.Scalar[pl.INT32],
    current_partner: pl.Scalar[pl.INT32],
    next_partner: pl.Scalar[pl.INT32],
    dep_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Synchronize the current and next recursive-doubling partners."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_tp_doubling_pair2_barrier", deps=[dep_tid]) as barrier_tid:
        pld.system.notify(
            target=signal, peer=current_partner, offsets=[my_rank, 0],
            value=1, op=pld.NotifyOp.AtomicAdd,
        )
        pld.system.notify(
            target=signal, peer=next_partner, offsets=[my_rank, 0],
            value=1, op=pld.NotifyOp.AtomicAdd,
        )
        pld.system.wait(
            signal=signal, offsets=[current_partner, 0],
            expected=1, cmp=pld.WaitCmp.Ge,
        )
        pld.system.wait(
            signal=signal, offsets=[next_partner, 0],
            expected=1, cmp=pld.WaitCmp.Ge,
        )
        pld.system.notify(
            target=signal, peer=my_rank, offsets=[current_partner, 0],
            value=-1, op=pld.NotifyOp.AtomicAdd,
        )
        pld.system.notify(
            target=signal, peer=my_rank, offsets=[next_partner, 0],
            value=-1, op=pld.NotifyOp.AtomicAdd,
        )
    return barrier_tid


@pl.jit.inline(auto_scope=False)
def finish_doubling_pair(
    sh: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    signal: pl.InOut[pld.DistributedTensor[[SIGNAL_ROWS, SIGNAL_COLS], pl.INT32]],
    my_rank: pl.Scalar[pl.INT32],
    partner: pl.Scalar[pl.INT32],
    dep_tid: pl.Scalar[pl.TASK_ID],
):
    """Complete the final partner read and clear its signal credit."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_tp_doubling_finish", deps=[dep_tid]):
        _doubling_completion_anchor = pl.read(sh, [0, 0])
        pld.system.notify(
            target=signal, peer=partner, offsets=[my_rank, 0],
            value=1, op=pld.NotifyOp.AtomicAdd,
        )
        pld.system.wait(
            signal=signal, offsets=[partner, 0],
            expected=1, cmp=pld.WaitCmp.Ge,
        )
        pld.system.notify(
            target=signal, peer=my_rank, offsets=[partner, 0],
            value=-1, op=pld.NotifyOp.AtomicAdd,
        )
    return sh


@pl.jit.inline(auto_scope=False)
def reduce_down_partial_doubling_tp2(
    partial: pl.Tensor[[T, D], pl.FP32],
    sh: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    doubling_window: pl.InOut[pld.DistributedTensor[[2 * T, D], pl.FP32]],
    signal: pl.InOut[pld.DistributedTensor[[SIGNAL_ROWS, SIGNAL_COLS], pl.INT32]],
    my_rank: pl.Scalar[pl.INT32],
):
    """Run the TP2 24-worker recursive-doubling AllReduce."""
    with pl.spmd(COMM_WORKERS, name_hint="sh_tp_doubling_publish") as publish_tid:
        comm_core = pl.tile.get_block_idx()
        for block in pl.range(comm_core, T * (D // COMM_STAGE_TILE), COMM_WORKERS):
            row = block // (D // COMM_STAGE_TILE)
            col = block % (D // COMM_STAGE_TILE) * COMM_STAGE_TILE
            pld.tensor.put(
                dst=doubling_window, peer=my_rank, src=partial,
                dst_offsets=[row, col], src_offsets=[row, col], shape=[1, COMM_STAGE_TILE],
            )

    partner_group = my_rank % 2
    partner = my_rank + 1 - 2 * partner_group
    publish_barrier_tid = doubling_pair_barrier(signal, my_rank, partner, publish_tid)

    with pl.spmd(COMM_WORKERS, name_hint="sh_tp_doubling_reduce_1", deps=[publish_barrier_tid]) as reduce_tid:
        comm_core = pl.tile.get_block_idx()
        for block in pl.range(comm_core, T * (D // COMM_STAGE_TILE), COMM_WORKERS):
            row = block // (D // COMM_STAGE_TILE)
            col = block % (D // COMM_STAGE_TILE) * COMM_STAGE_TILE
            local = pl.load(doubling_window, [row, col], [1, COMM_STAGE_TILE])
            remote = pld.tile.remote_load(
                doubling_window, peer=partner,
                offsets=[row, col], shape=[1, COMM_STAGE_TILE],
            )
            reduced = pl.add(local, remote)
            reduced_bf16 = pl.cast(reduced, target_type=pl.BF16, mode="rint")
            pl.store(reduced_bf16, [row, col], sh)

    return finish_doubling_pair(sh, signal, my_rank, partner, reduce_tid)


@pl.jit.inline(auto_scope=False)
def reduce_down_partial_doubling_tp4(
    partial: pl.Tensor[[T, D], pl.FP32],
    sh: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    doubling_window: pl.InOut[pld.DistributedTensor[[2 * T, D], pl.FP32]],
    signal: pl.InOut[pld.DistributedTensor[[SIGNAL_ROWS, SIGNAL_COLS], pl.INT32]],
    my_rank: pl.Scalar[pl.INT32],
):
    """Run the TP4 24-worker recursive-doubling AllReduce."""
    with pl.spmd(COMM_WORKERS, name_hint="sh_tp_doubling_publish") as publish_tid:
        comm_core = pl.tile.get_block_idx()
        for block in pl.range(comm_core, T * (D // COMM_STAGE_TILE), COMM_WORKERS):
            row = block // (D // COMM_STAGE_TILE)
            col = block % (D // COMM_STAGE_TILE) * COMM_STAGE_TILE
            pld.tensor.put(
                dst=doubling_window, peer=my_rank, src=partial,
                dst_offsets=[row, col], src_offsets=[row, col], shape=[1, COMM_STAGE_TILE],
            )

    partner_group_1 = my_rank % 2
    partner_1 = my_rank + 1 - 2 * partner_group_1
    partner_group_2 = (my_rank // 2) % 2
    partner_2 = my_rank + 2 - 4 * partner_group_2
    publish_barrier_tid = doubling_pair_barrier(signal, my_rank, partner_1, publish_tid)

    with pl.spmd(COMM_WORKERS, name_hint="sh_tp_doubling_reduce_1", deps=[publish_barrier_tid]) as reduce_1_tid:
        comm_core = pl.tile.get_block_idx()
        for block in pl.range(comm_core, T * (D // COMM_STAGE_TILE), COMM_WORKERS):
            row = block // (D // COMM_STAGE_TILE)
            col = block % (D // COMM_STAGE_TILE) * COMM_STAGE_TILE
            local = pl.load(doubling_window, [row, col], [1, COMM_STAGE_TILE])
            remote = pld.tile.remote_load(
                doubling_window, peer=partner_1,
                offsets=[row, col], shape=[1, COMM_STAGE_TILE],
            )
            reduced = pl.add(local, remote)
            pld.tile.remote_store(reduced, target=doubling_window, peer=my_rank, offsets=[T + row, col])

    stage_1_tid = doubling_two_partner_barrier(signal, my_rank, partner_1, partner_2, reduce_1_tid)

    with pl.spmd(COMM_WORKERS, name_hint="sh_tp_doubling_reduce_2", deps=[stage_1_tid]) as reduce_2_tid:
        comm_core = pl.tile.get_block_idx()
        for block in pl.range(comm_core, T * (D // COMM_STAGE_TILE), COMM_WORKERS):
            row = block // (D // COMM_STAGE_TILE)
            col = block % (D // COMM_STAGE_TILE) * COMM_STAGE_TILE
            local = pl.load(doubling_window, [T + row, col], [1, COMM_STAGE_TILE])
            remote = pld.tile.remote_load(
                doubling_window, peer=partner_2,
                offsets=[T + row, col], shape=[1, COMM_STAGE_TILE],
            )
            reduced = pl.add(local, remote)
            reduced_bf16 = pl.cast(reduced, target_type=pl.BF16, mode="rint")
            pl.store(reduced_bf16, [row, col], sh)

    return finish_doubling_pair(sh, signal, my_rank, partner_2, reduce_2_tid)


@pl.jit.inline(auto_scope=False)
def reduce_down_partial_doubling_tp8(
    partial: pl.Tensor[[T, D], pl.FP32],
    sh: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    doubling_window: pl.InOut[pld.DistributedTensor[[2 * T, D], pl.FP32]],
    signal: pl.InOut[pld.DistributedTensor[[SIGNAL_ROWS, SIGNAL_COLS], pl.INT32]],
    my_rank: pl.Scalar[pl.INT32],
):
    """Run the TP8 24-worker recursive-doubling AllReduce."""
    with pl.spmd(COMM_WORKERS, name_hint="sh_tp_doubling_publish") as publish_tid:
        comm_core = pl.tile.get_block_idx()
        for block in pl.range(comm_core, T * (D // COMM_STAGE_TILE), COMM_WORKERS):
            row = block // (D // COMM_STAGE_TILE)
            col = block % (D // COMM_STAGE_TILE) * COMM_STAGE_TILE
            pld.tensor.put(
                dst=doubling_window, peer=my_rank, src=partial,
                dst_offsets=[row, col], src_offsets=[row, col], shape=[1, COMM_STAGE_TILE],
            )

    partner_group_1 = my_rank % 2
    partner_1 = my_rank + 1 - 2 * partner_group_1
    partner_group_2 = (my_rank // 2) % 2
    partner_2 = my_rank + 2 - 4 * partner_group_2
    partner_group_4 = (my_rank // 4) % 2
    partner_4 = my_rank + 4 - 8 * partner_group_4
    publish_barrier_tid = doubling_pair_barrier(signal, my_rank, partner_1, publish_tid)

    with pl.spmd(COMM_WORKERS, name_hint="sh_tp_doubling_reduce_1", deps=[publish_barrier_tid]) as reduce_1_tid:
        comm_core = pl.tile.get_block_idx()
        for block in pl.range(comm_core, T * (D // COMM_STAGE_TILE), COMM_WORKERS):
            row = block // (D // COMM_STAGE_TILE)
            col = block % (D // COMM_STAGE_TILE) * COMM_STAGE_TILE
            local = pl.load(doubling_window, [row, col], [1, COMM_STAGE_TILE])
            remote = pld.tile.remote_load(
                doubling_window, peer=partner_1,
                offsets=[row, col], shape=[1, COMM_STAGE_TILE],
            )
            reduced = pl.add(local, remote)
            pld.tile.remote_store(reduced, target=doubling_window, peer=my_rank, offsets=[T + row, col])

    stage_1_tid = doubling_two_partner_barrier(signal, my_rank, partner_1, partner_2, reduce_1_tid)

    with pl.spmd(COMM_WORKERS, name_hint="sh_tp_doubling_reduce_2", deps=[stage_1_tid]) as reduce_2_tid:
        comm_core = pl.tile.get_block_idx()
        for block in pl.range(comm_core, T * (D // COMM_STAGE_TILE), COMM_WORKERS):
            row = block // (D // COMM_STAGE_TILE)
            col = block % (D // COMM_STAGE_TILE) * COMM_STAGE_TILE
            local = pl.load(doubling_window, [T + row, col], [1, COMM_STAGE_TILE])
            remote = pld.tile.remote_load(
                doubling_window, peer=partner_2,
                offsets=[T + row, col], shape=[1, COMM_STAGE_TILE],
            )
            reduced = pl.add(local, remote)
            pld.tile.remote_store(reduced, target=doubling_window, peer=my_rank, offsets=[row, col])

    stage_2_tid = doubling_two_partner_barrier(signal, my_rank, partner_2, partner_4, reduce_2_tid)

    with pl.spmd(COMM_WORKERS, name_hint="sh_tp_doubling_reduce_4", deps=[stage_2_tid]) as reduce_4_tid:
        comm_core = pl.tile.get_block_idx()
        for block in pl.range(comm_core, T * (D // COMM_STAGE_TILE), COMM_WORKERS):
            row = block // (D // COMM_STAGE_TILE)
            col = block % (D // COMM_STAGE_TILE) * COMM_STAGE_TILE
            local = pl.load(doubling_window, [row, col], [1, COMM_STAGE_TILE])
            remote = pld.tile.remote_load(
                doubling_window, peer=partner_4,
                offsets=[row, col], shape=[1, COMM_STAGE_TILE],
            )
            reduced = pl.add(local, remote)
            reduced_bf16 = pl.cast(reduced, target_type=pl.BF16, mode="rint")
            pl.store(reduced_bf16, [row, col], sh)

    return finish_doubling_pair(sh, signal, my_rank, partner_4, reduce_4_tid)


@pl.jit.incore
def reduce_down_partial(
    partial: pl.Tensor[[T, D], pl.FP32],
    sh: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    partial_window: pl.InOut[pld.DistributedTensor[[T, D], pl.FP32]],
    signal: pl.InOut[pld.DistributedTensor[[SIGNAL_ROWS, SIGNAL_COLS], pl.INT32]],
):
    """Sum local down partials and emit BF16 output."""
    for block, (window_iter,) in pl.range(T * (D // COMM_STAGE_TILE), init_values=(partial_window,)):
        row = block // (D // COMM_STAGE_TILE)
        col = block % (D // COMM_STAGE_TILE) * COMM_STAGE_TILE
        local = pl.load(partial, [row, col], [1, COMM_STAGE_TILE])
        window_iter = pl.store(local, [row, col], window_iter)
        staged_window = pl.yield_(window_iter)

    if USE_RING_ALLREDUCE:
        partial_window = pld.tensor.allreduce(staged_window, signal, op=pld.ReduceOp.Sum, mode="ring")
    else:
        partial_window = pld.tensor.allreduce(staged_window, signal, op=pld.ReduceOp.Sum, mode="mesh")

    for block, (sh_iter,) in pl.range(T * (D // COMM_STAGE_TILE), init_values=(sh,)):
        row = block // (D // COMM_STAGE_TILE)
        col = block % (D // COMM_STAGE_TILE) * COMM_STAGE_TILE
        reduced = pl.load(partial_window, [row, col], [1, COMM_STAGE_TILE])
        reduced_bf16 = pl.cast(reduced, target_type=pl.BF16, mode="rint")
        sh_iter = pl.store(reduced_bf16, [row, col], sh_iter)
        staged_sh = pl.yield_(sh_iter)
    return staged_sh


@pl.jit
def expert_shared_tp(
    x_local_i8: pl.Tensor[[T, D], pl.INT8],
    x_local_scale_dq: pl.Tensor[[T, 1], pl.FP32],
    shared_w1: pl.Tensor[[LOCAL_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[LOCAL_INTER], pl.FP32],
    shared_w3: pl.Tensor[[LOCAL_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[LOCAL_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, LOCAL_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    sh: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    partial_window: pl.InOut[pld.DistributedTensor[[T, D], pl.FP32]],
    doubling_window: pl.InOut[pld.DistributedTensor[[2 * T, D], pl.FP32]],
    signal: pl.InOut[pld.DistributedTensor[[SIGNAL_ROWS, SIGNAL_COLS], pl.INT32]],
    my_rank: pl.Scalar[pl.INT32],
):
    """Run one shared-expert TP rank and the down-output all-reduce."""
    h_local = pl.create_tensor([T, LOCAL_INTER], dtype=pl.FP32)
    local_amax = pl.create_tensor([T, SH_AMAX_TILE], dtype=pl.FP32)
    shared_gate_up_local(
        x_local_i8, x_local_scale_dq,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        h_local, local_amax,
    )

    partial = pl.create_tensor([T, D], dtype=pl.FP32)
    shared_down_local(h_local, local_amax, shared_w2, shared_w2_scale, partial)
    if USE_PARALLEL_DOUBLING:
        if TP_SIZE == 2:
            return reduce_down_partial_doubling_tp2(partial, sh, doubling_window, signal, my_rank)
        elif TP_SIZE == 4:
            return reduce_down_partial_doubling_tp4(partial, sh, doubling_window, signal, my_rank)
        else:
            return reduce_down_partial_doubling_tp8(partial, sh, doubling_window, signal, my_rank)
    elif USE_PARALLEL_MESH:
        return reduce_down_partial_parallel(partial, sh, partial_window, signal, my_rank)
    else:
        return reduce_down_partial(partial, sh, partial_window, signal)


@pl.jit.host
def l3_expert_shared_tp(
    x_local_i8: pl.Tensor[[TP_SIZE, T, D], pl.INT8],
    x_local_scale_dq: pl.Tensor[[TP_SIZE, T, 1], pl.FP32],
    shared_w1: pl.Tensor[[TP_SIZE, LOCAL_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[TP_SIZE, LOCAL_INTER], pl.FP32],
    shared_w3: pl.Tensor[[TP_SIZE, LOCAL_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[TP_SIZE, LOCAL_INTER], pl.FP32],
    shared_w2: pl.Tensor[[TP_SIZE, D, LOCAL_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[TP_SIZE, D], pl.FP32],
    sh: pl.Out[pl.Tensor[[TP_SIZE, T, D], pl.BF16]],
):
    """Launch one rank per TP shard in a single communication domain."""
    partial_window_buf = pld.alloc_window_buffer([T, D], dtype=pl.FP32)
    doubling_window_buf = pld.alloc_window_buffer([2 * T, D], dtype=pl.FP32)
    signal_buf = pld.alloc_window_buffer([SIGNAL_ROWS, SIGNAL_COLS], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        partial_window = pld.window(partial_window_buf, [T, D], dtype=pl.FP32)
        doubling_window = pld.window(doubling_window_buf, [2 * T, D], dtype=pl.FP32)
        signal = pld.window(signal_buf, [SIGNAL_ROWS, SIGNAL_COLS], dtype=pl.INT32)
        expert_shared_tp(
            x_local_i8[rank], x_local_scale_dq[rank],
            shared_w1[rank], shared_w1_scale[rank], shared_w3[rank], shared_w3_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            sh[rank],
            partial_window, doubling_window, signal, rank,
            device=rank,
        )


def golden_expert_shared_tp(tensors):
    """Torch reference with rank-local activation quantization and down sum."""
    import torch
    import torch.nn.functional as F

    from utils import int8_quant_per_row

    def dequant_w(weight_i8, weight_scale):
        return weight_i8.to(torch.float32) * weight_scale.unsqueeze(-1)

    partials = []
    for rank in range(TP_SIZE):
        x_local = tensors["x_local_i8"][rank].float()
        x_scale = tensors["x_local_scale_dq"][rank].float()
        x_local = x_local * x_scale

        sw1 = dequant_w(tensors["shared_w1"][rank], tensors["shared_w1_scale"][rank].float())
        sw3 = dequant_w(tensors["shared_w3"][rank], tensors["shared_w3_scale"][rank].float())
        sw2 = dequant_w(tensors["shared_w2"][rank], tensors["shared_w2_scale"][rank].float())

        sh_gate = x_local @ sw1.T
        sh_up = x_local @ sw3.T
        if SWIGLU_LIMIT > 0:
            sh_gate = sh_gate.clamp(max=SWIGLU_LIMIT)
            sh_up = sh_up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        sh_h = F.silu(sh_gate) * sh_up
        sh_h_i8, sh_h_scale_dq = int8_quant_per_row(sh_h)
        sh_h = sh_h_i8.float() * sh_h_scale_dq
        partials.append(sh_h @ sw2.T)

    sh = torch.stack(partials, dim=0).sum(dim=0)
    tensors["sh"][:] = sh.to(torch.bfloat16).unsqueeze(0).expand_as(tensors["sh"])


def compare_shared_output(actual, expected, **context):
    """Validate every TP rank and communication block, then rank consistency."""
    import torch

    from golden import ratio_reldiff

    rank_compare = ratio_reldiff(diff_thd=2e-3, pct_thd=0.015, max_diff_hd=0.5)
    block_compare = ratio_reldiff(diff_thd=2e-2, pct_thd=0.25, max_diff_hd=0.5)
    for rank in range(TP_SIZE):
        passed, detail = rank_compare(actual[rank], expected[rank], **context)
        if not passed:
            return False, f"TP rank {rank} failed:\n{detail}"
        for row in range(T):
            for col_block in range(D // COMM_STAGE_TILE):
                col = col_block * COMM_STAGE_TILE
                passed, detail = block_compare(
                    actual[rank, row, col : col + COMM_STAGE_TILE], expected[rank, row, col : col + COMM_STAGE_TILE],
                    **context,
                )
                if passed:
                    continue
                block = row * (D // COMM_STAGE_TILE) + col_block
                return False, f"TP rank {rank}, communication block {block} failed:\n{detail}"

    rank_reference = actual[0].cpu().float()
    rank_rtol = torch.finfo(torch.bfloat16).eps
    rank_atol = 2e-3
    for rank in range(1, TP_SIZE):
        rank_actual = actual[rank].cpu().float()
        rank_scale = torch.maximum(rank_reference.abs(), rank_actual.abs())
        rank_diff = (rank_reference - rank_actual).abs()
        mismatch = rank_diff > rank_atol + rank_rtol * rank_scale
        if mismatch.any():
            mismatch_count = int(mismatch.sum().item())
            max_diff = float(rank_diff[mismatch].max().item())
            total_points = actual[0].numel()
            mismatch_ratio = f"{mismatch_count}/{total_points}"
            detail = f"TP rank {rank} differs from rank 0 at {mismatch_ratio} points; max_abs_diff={max_diff:.4g}"
            return False, detail
    return True, ""


def build_tensor_specs():
    """Create replicated inputs and intermediate-sharded shared weights."""
    import torch

    from expert_shared import gen_shared_weight
    from golden import TensorSpec
    from utils import int8_quant_per_row

    x_local_bf16 = torch.randn(T, D, dtype=torch.bfloat16)
    x_local_i8, x_local_scale_dq = int8_quant_per_row(x_local_bf16)
    x_local_i8 = x_local_i8.unsqueeze(0).repeat(TP_SIZE, 1, 1)
    x_local_scale_dq = x_local_scale_dq.float().unsqueeze(0).repeat(TP_SIZE, 1, 1)

    shared_dequant_std = {"w1": 1.71e-2, "w2": 1.68e-2, "w3": 1.70e-2}
    sw1_i8, sw1_scale = gen_shared_weight((MOE_INTER, D), shared_dequant_std["w1"], chan_cv=0.50)
    sw3_i8, sw3_scale = gen_shared_weight((MOE_INTER, D), shared_dequant_std["w3"], chan_cv=0.50)
    sw2_i8, sw2_scale = gen_shared_weight((D, MOE_INTER), shared_dequant_std["w2"], chan_cv=0.33)

    sw1_i8 = sw1_i8.reshape(TP_SIZE, LOCAL_INTER, D).contiguous()
    sw1_scale = sw1_scale.reshape(TP_SIZE, LOCAL_INTER).contiguous()
    sw3_i8 = sw3_i8.reshape(TP_SIZE, LOCAL_INTER, D).contiguous()
    sw3_scale = sw3_scale.reshape(TP_SIZE, LOCAL_INTER).contiguous()
    sw2_chunks = torch.chunk(sw2_i8, TP_SIZE, dim=1)
    sw2_chunks = [chunk.contiguous() for chunk in sw2_chunks]
    sw2_i8 = torch.stack(sw2_chunks, dim=0)
    sw2_scale = sw2_scale.unsqueeze(0).repeat(TP_SIZE, 1).contiguous()

    return [
        TensorSpec("x_local_i8", [TP_SIZE, T, D], torch.int8, init_value=lambda: x_local_i8),
        TensorSpec("x_local_scale_dq", [TP_SIZE, T, 1], torch.float32, init_value=lambda: x_local_scale_dq),
        TensorSpec("shared_w1", [TP_SIZE, LOCAL_INTER, D], torch.int8, init_value=lambda: sw1_i8, resident="stacked"),
        TensorSpec(
            "shared_w1_scale", [TP_SIZE, LOCAL_INTER], torch.float32,
            init_value=lambda: sw1_scale, resident="stacked",
        ),
        TensorSpec("shared_w3", [TP_SIZE, LOCAL_INTER, D], torch.int8, init_value=lambda: sw3_i8, resident="stacked"),
        TensorSpec(
            "shared_w3_scale", [TP_SIZE, LOCAL_INTER], torch.float32,
            init_value=lambda: sw3_scale, resident="stacked",
        ),
        TensorSpec("shared_w2", [TP_SIZE, D, LOCAL_INTER], torch.int8, init_value=lambda: sw2_i8, resident="stacked"),
        TensorSpec("shared_w2_scale", [TP_SIZE, D], torch.float32, init_value=lambda: sw2_scale, resident="stacked"),
        TensorSpec("sh", [TP_SIZE, T, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse

    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument(
        "--tp", type=int, default=TP_SIZE, choices=list(_TP_CHOICES),
        help="shared-expert tensor-parallel world size",
    )
    parser.add_argument(
        "--allreduce-mode", type=str, default=ALLREDUCE_MODE, choices=list(_ALLREDUCE_MODES),
        help="all-reduce schedule used for the row-parallel down output",
    )
    default_devices = ",".join(str(rank) for rank in range(TP_SIZE))
    parser.add_argument(
        "-d", "--device", type=str, default=default_devices,
        help=f"comma-separated device ids; need at least {TP_SIZE}",
    )
    parser.add_argument(
        "--enable-l2-swimlane", type=int, nargs="?", const=1, default=0,
        choices=(0, 1, 2, 4),
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument(
        "--save-data", action="store_true", default=False,
        help="persist inputs and golden outputs for replay",
    )
    parser.add_argument(
        "--golden-data", type=str, default=None,
        help="directory containing cached in/ and out/ tensors",
    )
    args = parser.parse_args()

    device_ids = [int(device) for device in args.device.split(",")]
    if args.tp != TP_SIZE:
        raise ValueError(f"import-time TP size must match --tp, got {TP_SIZE} and {args.tp}")
    if args.allreduce_mode != ALLREDUCE_MODE:
        configured_mode = f"{ALLREDUCE_MODE} and {args.allreduce_mode}"
        raise ValueError(f"import-time all-reduce mode must match --allreduce-mode, got {configured_mode}")
    if len(device_ids) < TP_SIZE:
        raise ValueError(f"need at least {TP_SIZE} devices for TP, got {device_ids}")

    result = run_jit(
        fn=l3_expert_shared_tp,
        specs=build_tensor_specs(),
        golden_fn=golden_expert_shared_tp,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:TP_SIZE],
                num_sub_workers=0,
            ),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "sh": compare_shared_output,
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
