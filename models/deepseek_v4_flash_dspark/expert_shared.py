# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=4
"""DeepSeek-V4 shared-expert TP with SP gather and reduce-scatter boundaries."""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from config import DECODE_TOKENS, FLASH as M, INT8_AMAX_EPS, INT8_SCALE_MAX, TP_SHARED_EXPERT


# command-line config
_TP_CHOICES = (2, 4, 8)
_TP_DEFAULT = TP_SHARED_EXPERT


def _parse_tp_argv():
    for name in ("--tp-shared-expert", "--tp"):
        for index, token in enumerate(sys.argv):
            if token == name and index + 1 < len(sys.argv):
                return int(sys.argv[index + 1])
            if token.startswith(f"{name}="):
                return int(token.split("=", 1)[1])
    return _TP_DEFAULT


# distributed config
TP_SIZE = _parse_tp_argv()
if TP_SIZE not in _TP_CHOICES:
    raise ValueError(f"--tp-shared-expert must be one of {_TP_CHOICES}, got {TP_SIZE}")

# model config
T = DECODE_TOKENS
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
LOCAL_INTER = MOE_INTER // TP_SIZE
SP_T = T // TP_SIZE
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
SP_COMM_WORKERS = min(24, SP_T * (D // COMM_STAGE_TILE))
if MOE_INTER % TP_SIZE != 0:
    raise ValueError(f"intermediate size {MOE_INTER} is not divisible by TP {TP_SIZE}")
if T % TP_SIZE != 0:
    raise ValueError(f"token rows {T} must be divisible by TP {TP_SIZE}")
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
                gate_rows_i32 = pl.slice(gate_i32, [SH_AMAX_TILE, ACT_INTER_TILE], [row0, n0], valid_shape=[SH_ACT_M_TILE, ACT_INTER_TILE])
                up_rows_i32 = pl.slice(up_i32, [SH_AMAX_TILE, ACT_INTER_TILE], [row0, n0], valid_shape=[SH_ACT_M_TILE, ACT_INTER_TILE])
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
            row_amax_matrix = pl.slice(local_amax, [SH_AMAX_TILE, SH_AMAX_TILE], [row_start, 0], valid_shape=[SH_ACT_M_TILE, SH_AMAX_TILE])
            row_amax = pl.row_max(row_amax_matrix)
            row_amax_recip = pl.recip(row_amax)
            row_scale_q = pl.mul(row_amax_recip, INT8_SCALE_MAX)
            for q_idx in pl.pipeline(0, LOCAL_INTER // QUANT_TILE, stage=1):
                k0 = q_idx * QUANT_TILE
                h_fp32 = pl.slice(h_local, [SH_AMAX_TILE, QUANT_TILE], [row_start, k0], valid_shape=[SH_ACT_M_TILE, QUANT_TILE])
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
            output_row_amax_matrix = pl.slice(local_amax, [SH_M_TILE, SH_AMAX_TILE], [ts0, 0], valid_shape=[SH_VALID_M, SH_AMAX_TILE])
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
def gather_sp_input(
    x_local_i8: pl.Tensor[[SP_T, D], pl.INT8],
    x_local_scale_dq: pl.Tensor[[SP_T, 1], pl.FP32],
    gathered_x: pl.Tensor[[T, D], pl.INT8],
    gathered_scale: pl.Tensor[[T, 1], pl.FP32],
    gather_x: pld.DistributedTensor[[T, D], pl.INT8],
    gather_scale: pld.DistributedTensor[[T, 1], pl.FP32],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """All-gather one SP input shard inside its contiguous TP group."""
    group_base = my_rank // TP_SIZE * TP_SIZE
    tp_rank = my_rank % TP_SIZE

    with pl.spmd(SP_COMM_WORKERS, name_hint="sh_sp_tp_gather_publish") as publish_tid:
        comm_core = pl.tile.get_block_idx()
        for block in pl.range(comm_core, SP_T * (D // COMM_STAGE_TILE), SP_COMM_WORKERS):
            local_row = block // (D // COMM_STAGE_TILE)
            col = block % (D // COMM_STAGE_TILE) * COMM_STAGE_TILE
            dst_row = tp_rank * SP_T + local_row
            for peer_tp in pl.range(TP_SIZE):
                peer = group_base + peer_tp
                pld.tensor.put(
                    dst=gather_x, peer=peer, src=x_local_i8,
                    dst_offsets=[dst_row, col], src_offsets=[local_row, col], shape=[1, COMM_STAGE_TILE],
                )

        for scale_block in pl.range(comm_core, SP_T // SH_AMAX_TILE, SP_COMM_WORKERS):
            local_row = scale_block * SH_AMAX_TILE
            dst_row = tp_rank * SP_T + local_row
            for peer_tp in pl.range(TP_SIZE):
                peer = group_base + peer_tp
                pld.tensor.put(
                    dst=gather_scale, peer=peer, src=x_local_scale_dq,
                    dst_offsets=[dst_row, 0], src_offsets=[local_row, 0], shape=[SH_AMAX_TILE, 1],
                )

        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=gather_signal, peer=group_base + peer_tp, offsets=[tp_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_sp_tp_gather_wait", deps=[publish_tid]) as wait_tid:
        publish_expected = pl.cast(SP_COMM_WORKERS, pl.INT32)
        for src_tp in pl.range(TP_SIZE):
            if src_tp != tp_rank:
                pld.system.wait(
                    signal=gather_signal, offsets=[src_tp, 0],
                    expected=publish_expected, cmp=pld.WaitCmp.Ge,
                )

    with pl.spmd(SP_COMM_WORKERS, name_hint="sh_sp_tp_gather_copy", deps=[wait_tid]) as copy_tid:
        comm_core = pl.tile.get_block_idx()
        for block in pl.range(comm_core, T * (D // COMM_STAGE_TILE), SP_COMM_WORKERS):
            row = block // (D // COMM_STAGE_TILE)
            col = block % (D // COMM_STAGE_TILE) * COMM_STAGE_TILE
            gathered_x_tile = gather_x[row : row + 1, col : col + COMM_STAGE_TILE]
            gathered_x[row : row + 1, col : col + COMM_STAGE_TILE] = gathered_x_tile
        for scale_block in pl.range(comm_core, T // SH_AMAX_TILE, SP_COMM_WORKERS):
            row = scale_block * SH_AMAX_TILE
            gathered_scale[row : row + SH_AMAX_TILE, :] = gather_scale[row : row + SH_AMAX_TILE, :]

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_sp_tp_gather_complete", deps=[copy_tid]):
        _gather_completion_anchor = pl.read(gathered_x, [0, 0])
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=gather_signal, peer=group_base + peer_tp, offsets=[tp_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )

        completion_expected = pl.cast(SP_COMM_WORKERS + 1, pl.INT32)
        for src_tp in pl.range(TP_SIZE):
            if src_tp != tp_rank:
                pld.system.wait(
                    signal=gather_signal, offsets=[src_tp, 0],
                    expected=completion_expected, cmp=pld.WaitCmp.Ge,
                )

        gather_credits = pl.cast(SP_COMM_WORKERS + 1, pl.INT32)
        gather_reset: pl.Scalar[pl.INT32] = -gather_credits
        for src_tp in pl.range(TP_SIZE):
            if src_tp != tp_rank:
                pld.system.notify(
                    target=gather_signal, peer=my_rank, offsets=[src_tp, 0],
                    value=gather_reset, op=pld.NotifyOp.AtomicAdd,
                )
        pl.write(gathered_x, [0, 0], _gather_completion_anchor)
    return gathered_x, gathered_scale


@pl.jit.inline(auto_scope=False)
def reduce_scatter_sp_partial(
    partial: pl.Tensor[[T, D], pl.FP32],
    sh_local: pl.Tensor[[SP_T, D], pl.BF16],
    scatter: pld.DistributedTensor[[T, D], pl.FP32],
    scatter_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Reduce-scatter full-token FP32 partials inside a contiguous TP group."""
    group_base = my_rank // TP_SIZE * TP_SIZE
    tp_rank = my_rank % TP_SIZE

    with pl.spmd(COMM_WORKERS, name_hint="sh_sp_tp_scatter_publish") as publish_tid:
        comm_core = pl.tile.get_block_idx()
        for block in pl.range(comm_core, T * (D // COMM_STAGE_TILE), COMM_WORKERS):
            source_row = block // (D // COMM_STAGE_TILE)
            col = block % (D // COMM_STAGE_TILE) * COMM_STAGE_TILE
            owner_tp = source_row // SP_T
            owner_row = source_row % SP_T
            dst_row = tp_rank * SP_T + owner_row
            peer = group_base + owner_tp
            pld.tensor.put(
                dst=scatter, peer=peer, src=partial,
                dst_offsets=[dst_row, col], src_offsets=[source_row, col], shape=[1, COMM_STAGE_TILE],
            )

        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=scatter_signal, peer=group_base + peer_tp, offsets=[tp_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_sp_tp_scatter_wait", deps=[publish_tid]) as wait_tid:
        publish_expected = pl.cast(COMM_WORKERS, pl.INT32)
        for src_tp in pl.range(TP_SIZE):
            if src_tp != tp_rank:
                pld.system.wait(
                    signal=scatter_signal, offsets=[src_tp, 0],
                    expected=publish_expected, cmp=pld.WaitCmp.Ge,
                )

    with pl.spmd(COMM_WORKERS, name_hint="sh_sp_tp_scatter_reduce", deps=[wait_tid]) as reduce_tid:
        comm_core = pl.tile.get_block_idx()
        for block in pl.range(comm_core, SP_T * (D // COMM_STAGE_TILE), COMM_WORKERS):
            local_row = block // (D // COMM_STAGE_TILE)
            col = block % (D // COMM_STAGE_TILE) * COMM_STAGE_TILE
            own_row = tp_rank * SP_T + local_row
            acc = pl.load(scatter, [own_row, col], [1, COMM_STAGE_TILE])
            for src_tp in pl.range(TP_SIZE):
                if src_tp != tp_rank:
                    src_row = src_tp * SP_T + local_row
                    source_partial = pl.load(scatter, [src_row, col], [1, COMM_STAGE_TILE])
                    acc = pl.add(acc, source_partial)
            reduced_bf16 = pl.cast(acc, target_type=pl.BF16, mode="rint")
            pl.store(reduced_bf16, [local_row, col], sh_local)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_sp_tp_scatter_complete", deps=[reduce_tid]):
        _scatter_completion_anchor = pl.read(sh_local, [0, 0])
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=scatter_signal, peer=group_base + peer_tp, offsets=[tp_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )

        completion_expected = pl.cast(COMM_WORKERS + 1, pl.INT32)
        for src_tp in pl.range(TP_SIZE):
            if src_tp != tp_rank:
                pld.system.wait(
                    signal=scatter_signal, offsets=[src_tp, 0],
                    expected=completion_expected, cmp=pld.WaitCmp.Ge,
                )

        scatter_credits = pl.cast(COMM_WORKERS + 1, pl.INT32)
        scatter_reset: pl.Scalar[pl.INT32] = -scatter_credits
        for src_tp in pl.range(TP_SIZE):
            if src_tp != tp_rank:
                pld.system.notify(
                    target=scatter_signal, peer=my_rank, offsets=[src_tp, 0],
                    value=scatter_reset, op=pld.NotifyOp.AtomicAdd,
                )
        pl.write(sh_local, [0, 0], _scatter_completion_anchor)
    return sh_local


@pl.jit.inline(auto_scope=False)
def expert_shared_sp_tp(
    x_local_i8: pl.Tensor[[SP_T, D], pl.INT8],
    x_local_scale_dq: pl.Tensor[[SP_T, 1], pl.FP32],
    shared_w1: pl.Tensor[[LOCAL_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[LOCAL_INTER], pl.FP32],
    shared_w3: pl.Tensor[[LOCAL_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[LOCAL_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, LOCAL_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    sh_local: pl.Tensor[[SP_T, D], pl.BF16],
    gather_x: pld.DistributedTensor[[T, D], pl.INT8],
    gather_scale: pld.DistributedTensor[[T, 1], pl.FP32],
    scatter: pld.DistributedTensor[[T, D], pl.FP32],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    scatter_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Run the shared expert across contiguous SP and TP rank groups."""
    gathered_x = pl.create_tensor([T, D], dtype=pl.INT8)
    gathered_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    gather_sp_input(
        x_local_i8, x_local_scale_dq,
        gathered_x, gathered_scale,
        gather_x, gather_scale, gather_signal,
        my_rank,
    )

    h_local = pl.create_tensor([T, LOCAL_INTER], dtype=pl.FP32)
    local_amax = pl.create_tensor([T, SH_AMAX_TILE], dtype=pl.FP32)
    shared_gate_up_local(
        gathered_x, gathered_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        h_local, local_amax,
    )

    partial = pl.create_tensor([T, D], dtype=pl.FP32)
    shared_down_local(h_local, local_amax, shared_w2, shared_w2_scale, partial)
    return reduce_scatter_sp_partial(partial, sh_local, scatter, scatter_signal, my_rank)


@pl.jit
def expert_shared_sp_tp_test(
    x_local_i8: pl.Tensor[[SP_T, D], pl.INT8],
    x_local_scale_dq: pl.Tensor[[SP_T, 1], pl.FP32],
    shared_w1: pl.Tensor[[LOCAL_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[LOCAL_INTER], pl.FP32],
    shared_w3: pl.Tensor[[LOCAL_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[LOCAL_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, LOCAL_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    sh: pl.Out[pl.Tensor[[SP_T, D], pl.BF16]],
    gather_x: pld.DistributedTensor[[T, D], pl.INT8],
    gather_scale: pld.DistributedTensor[[T, 1], pl.FP32],
    scatter: pld.DistributedTensor[[T, D], pl.FP32],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    scatter_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Run one shared-expert TP rank between SP all-gather and reduce-scatter."""
    sh = expert_shared_sp_tp(
        x_local_i8, x_local_scale_dq,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        sh,
        gather_x, gather_scale, scatter,
        gather_signal, scatter_signal, my_rank,
    )
    return sh


@pl.jit.host
def l3_expert_shared_sp_tp(
    x_local_i8: pl.Tensor[[TP_SIZE, SP_T, D], pl.INT8],
    x_local_scale_dq: pl.Tensor[[TP_SIZE, SP_T, 1], pl.FP32],
    shared_w1: pl.Tensor[[TP_SIZE, LOCAL_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[TP_SIZE, LOCAL_INTER], pl.FP32],
    shared_w3: pl.Tensor[[TP_SIZE, LOCAL_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[TP_SIZE, LOCAL_INTER], pl.FP32],
    shared_w2: pl.Tensor[[TP_SIZE, D, LOCAL_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[TP_SIZE, D], pl.FP32],
    sh: pl.Out[pl.Tensor[[TP_SIZE, SP_T, D], pl.BF16]],
):
    """Launch one SP-local shared-expert rank per TP shard."""
    gather_x_buf = pld.alloc_window_buffer([T, D], dtype=pl.INT8)
    gather_scale_buf = pld.alloc_window_buffer([T, 1], dtype=pl.FP32)
    scatter_buf = pld.alloc_window_buffer([T, D], dtype=pl.FP32)
    gather_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    scatter_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(TP_SIZE):
        gather_x = pld.window(gather_x_buf, [T, D], dtype=pl.INT8)
        gather_scale = pld.window(gather_scale_buf, [T, 1], dtype=pl.FP32)
        scatter = pld.window(scatter_buf, [T, D], dtype=pl.FP32)
        gather_signal = pld.window(gather_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        scatter_signal = pld.window(scatter_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        expert_shared_sp_tp_test(
            x_local_i8[rank], x_local_scale_dq[rank],
            shared_w1[rank], shared_w1_scale[rank], shared_w3[rank], shared_w3_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            sh[rank],
            gather_x, gather_scale, scatter,
            gather_signal, scatter_signal, rank,
            device=rank,
        )


def golden_expert_shared_sp_tp(tensors):
    """Torch reference for SP gather, sharded shared expert, and reduce-scatter."""
    import torch
    import torch.nn.functional as F

    from utils import int8_quant_per_row

    x_group_i8 = tensors["x_local_i8"].reshape(T, D)
    x_group_scale = tensors["x_local_scale_dq"].reshape(T, 1)
    x_group = x_group_i8.float() * x_group_scale.float()

    partials = []
    for rank in range(TP_SIZE):
        sw1_scale = tensors["shared_w1_scale"][rank].float().unsqueeze(-1)
        sw3_scale = tensors["shared_w3_scale"][rank].float().unsqueeze(-1)
        sw2_scale = tensors["shared_w2_scale"][rank].float().unsqueeze(-1)
        sw1 = tensors["shared_w1"][rank].float() * sw1_scale
        sw3 = tensors["shared_w3"][rank].float() * sw3_scale
        sw2 = tensors["shared_w2"][rank].float() * sw2_scale

        sh_gate = x_group @ sw1.T
        sh_up = x_group @ sw3.T
        if SWIGLU_LIMIT > 0:
            sh_gate = sh_gate.clamp(max=SWIGLU_LIMIT)
            sh_up = sh_up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        sh_h = F.silu(sh_gate) * sh_up
        sh_h_i8, sh_h_scale_dq = int8_quant_per_row(sh_h)
        sh_h = sh_h_i8.float() * sh_h_scale_dq
        partials.append(sh_h @ sw2.T)

    sh_group = torch.stack(partials, dim=0).sum(dim=0)
    tensors["sh"][:] = sh_group.to(torch.bfloat16).reshape(TP_SIZE, SP_T, D)


def compare_sp_shared_output(actual, expected, **context):
    """Validate every rank-local SP shard and communication block."""
    from golden import ratio_reldiff

    rank_compare = ratio_reldiff(diff_thd=2e-3, pct_thd=0.015, max_diff_hd=0.5)
    block_compare = ratio_reldiff(diff_thd=2e-2, pct_thd=0.25, max_diff_hd=0.5)
    for rank in range(TP_SIZE):
        passed, detail = rank_compare(actual[rank], expected[rank], **context)
        if not passed:
            return False, f"TP rank {rank} failed:\n{detail}"
        for row in range(SP_T):
            for col_block in range(D // COMM_STAGE_TILE):
                col = col_block * COMM_STAGE_TILE
                passed, detail = block_compare(
                    actual[rank, row, col : col + COMM_STAGE_TILE], expected[rank, row, col : col + COMM_STAGE_TILE],
                    **context,
                )
                if passed:
                    continue
                block = row * (D // COMM_STAGE_TILE) + col_block
                return False, f"TP rank {rank}, SP communication block {block} failed:\n{detail}"
    return True, ""


def gen_shared_weight(shape, dequant_std, chan_cv):
    """Generate per-channel INT8 weights and FP32 scales from an MXFP8 quantization grid."""
    import torch

    FP8_MAX, TINY = 448.0, 1e-20

    def sim_fp8(W, block=128):
        """Simulate E4M3 weights with 128x128-block E8M0 round-up scales."""
        out, inn = W.shape
        Wb = W.reshape(out // block, block, inn // block, block)
        Wb_abs = Wb.abs()
        block_amax = Wb_abs.amax(dim=(1, 3), keepdim=True)
        block_scale = block_amax / FP8_MAX
        block_scale = block_scale.clamp_min(TINY)
        scale_log2 = torch.log2(block_scale)
        scale_exponent = torch.ceil(scale_log2)
        scale = torch.exp2(scale_exponent)
        q = (Wb / scale).to(torch.float8_e4m3fn).float() * scale
        return q.reshape(out, inn)

    W = torch.randn(*shape)
    channel_gain_source = torch.randn(*shape[:-1], 1)
    channel_gain_log = chan_cv * channel_gain_source
    channel_gain = torch.exp(channel_gain_log)
    W = W * channel_gain
    Wq = sim_fp8(W)
    Wq_abs = Wq.abs()
    amax = Wq_abs.amax(dim=-1, keepdim=True)
    amax = amax.clamp_min(INT8_AMAX_EPS)
    scale = amax / INT8_SCALE_MAX
    w_scaled = Wq / scale
    w_rounded = torch.round(w_scaled)
    w_clamped = w_rounded.clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX)
    w_i8 = w_clamped.to(torch.int8)
    w_dequant = w_i8.float() * scale
    scale_adjustment = dequant_std / w_dequant.std()
    scale = scale * scale_adjustment
    scale = scale.squeeze(-1).float()
    return w_i8, scale


def build_sp_tensor_specs():
    """Create distinct SP input shards and intermediate-sharded shared weights."""
    import torch

    from golden import TensorSpec
    from utils import int8_quant_per_row

    x_group_bf16 = torch.randn(T, D, dtype=torch.bfloat16)
    x_group_i8, x_group_scale_dq = int8_quant_per_row(x_group_bf16)
    x_local_i8 = x_group_i8.reshape(TP_SIZE, SP_T, D).contiguous()
    x_local_scale_dq = x_group_scale_dq.float().reshape(TP_SIZE, SP_T, 1).contiguous()

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
        TensorSpec("x_local_i8", [TP_SIZE, SP_T, D], torch.int8, init_value=lambda: x_local_i8),
        TensorSpec("x_local_scale_dq", [TP_SIZE, SP_T, 1], torch.float32, init_value=lambda: x_local_scale_dq),
        TensorSpec("shared_w1", [TP_SIZE, LOCAL_INTER, D], torch.int8, init_value=lambda: sw1_i8, resident="stacked"),
        TensorSpec("shared_w1_scale", [TP_SIZE, LOCAL_INTER], torch.float32, init_value=lambda: sw1_scale, resident="stacked"),
        TensorSpec("shared_w3", [TP_SIZE, LOCAL_INTER, D], torch.int8, init_value=lambda: sw3_i8, resident="stacked"),
        TensorSpec("shared_w3_scale", [TP_SIZE, LOCAL_INTER], torch.float32, init_value=lambda: sw3_scale, resident="stacked"),
        TensorSpec("shared_w2", [TP_SIZE, D, LOCAL_INTER], torch.int8, init_value=lambda: sw2_i8, resident="stacked"),
        TensorSpec("shared_w2_scale", [TP_SIZE, D], torch.float32, init_value=lambda: sw2_scale, resident="stacked"),
        TensorSpec("sh", [TP_SIZE, SP_T, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse

    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument(
        "--tp-shared-expert", "--tp", dest="tp", type=int,
        default=TP_SIZE, choices=list(_TP_CHOICES),
        help="shared-expert tensor-parallel world size",
    )
    default_devices = ",".join(str(rank) for rank in range(TP_SIZE))
    parser.add_argument(
        "-d", "--device", type=str, default=default_devices,
        help=f"comma-separated device ids; need at least {TP_SIZE}",
    )
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2, 4))
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument(
        "--save-data", action="store_true", default=False,
        help="persist inputs and golden outputs for replay",
    )
    parser.add_argument("--golden-data", type=str, default=None, help="directory containing cached in/ and out/ tensors")
    args = parser.parse_args()

    device_ids = [int(device) for device in args.device.split(",")]
    if args.tp != TP_SIZE:
        raise ValueError(f"import-time TP size must match --tp, got {TP_SIZE} and {args.tp}")
    if len(device_ids) < TP_SIZE:
        raise ValueError(f"need at least {TP_SIZE} devices for TP, got {device_ids}")

    run_fn = l3_expert_shared_sp_tp
    specs = build_sp_tensor_specs()
    golden_fn = golden_expert_shared_sp_tp
    output_compare = compare_sp_shared_output

    result = run_jit(
        fn=run_fn,
        specs=specs,
        golden_fn=golden_fn,
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
            "sh": output_compare,
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
