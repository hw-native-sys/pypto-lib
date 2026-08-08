# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 attention TP token gather and output reduce-scatter collectives."""

import sys

import pypto.language as pl
import pypto.language.distributed as pld

from config import (
    DECODE_TOKENS,
    FLASH as M,
    PREFILL_TOKENS,
    TP_ATTN_SINK,
    TP_O_A,
    TP_O_B,
    TP_Q_B,
)


# parallelism
TP_CHOICES = (1, 2, 4, 8)


def _parse_int_argv(name):
    for index, token in enumerate(sys.argv):
        if token == name and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if token.startswith(f"{name}="):
            return int(token.split("=", 1)[1])
    return None


def _resolve_component_tp(name, config_default):
    component_tp = _parse_int_argv(name)
    legacy_tp = _parse_int_argv("--tp")
    resolved_tp = component_tp if component_tp is not None else legacy_tp
    resolved_tp = config_default if resolved_tp is None else resolved_tp
    if resolved_tp not in TP_CHOICES:
        raise ValueError(f"{name} must be one of {TP_CHOICES}, got {resolved_tp}")
    return resolved_tp


TP_Q_B_SIZE = _resolve_component_tp("--tp-q-b", TP_Q_B)
TP_ATTN_SINK_SIZE = _resolve_component_tp("--tp-attn-sink", TP_ATTN_SINK)
TP_O_A_SIZE = _resolve_component_tp("--tp-o-a", TP_O_A)
TP_O_B_SIZE = _resolve_component_tp("--tp-o-b", TP_O_B)

# Compatibility alias for composed attention programs that still expose one TP degree.
TP_SIZE = TP_Q_B_SIZE


def require_fused_attention_tp():
    component_degrees = (TP_Q_B_SIZE, TP_ATTN_SINK_SIZE, TP_O_A_SIZE, TP_O_B_SIZE)
    if len(set(component_degrees)) != 1:
        raise ValueError(
            "fused sparse attention requires equal TP degrees: "
            f"q_b={TP_Q_B_SIZE}, attn_sink={TP_ATTN_SINK_SIZE}, "
            f"o_a={TP_O_A_SIZE}, o_b={TP_O_B_SIZE}"
        )
    return TP_Q_B_SIZE

# model config
D = M.hidden_size
GROUP_T_MAX = max(DECODE_TOKENS, PREFILL_TOKENS)

# tiling
COMM_D_TILE = 4096
COMM_WORKERS = 24

if GROUP_T_MAX % TP_Q_B_SIZE != 0:
    raise ValueError(f"maximum token rows {GROUP_T_MAX} must be divisible by Q-B TP {TP_Q_B_SIZE}")
if GROUP_T_MAX % TP_O_B_SIZE != 0:
    raise ValueError(f"maximum token rows {GROUP_T_MAX} must be divisible by O-B TP {TP_O_B_SIZE}")
if D % COMM_D_TILE != 0:
    raise ValueError(f"hidden size {D} must be divisible by communication tile {COMM_D_TILE}")


@pl.jit.inline(auto_scope=False)
def gather_sp_bf16(
    x_local: pl.Tensor,
    x_group: pl.Tensor,
    gather_window: pld.DistributedTensor[[GROUP_T_MAX, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_Q_B_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    producer_dep: pl.Scalar[pl.TASK_ID],
):
    """All-gather one token shard inside a contiguous TP group."""
    local_t = pl.tensor.dim(x_local, 0)
    group_base = my_rank // TP_Q_B_SIZE * TP_Q_B_SIZE
    tp_rank = my_rank % TP_Q_B_SIZE

    with pl.spmd(COMM_WORKERS, name_hint="attn_tp_gather_publish", deps=[producer_dep]) as publish_tid:
        comm_core = pl.tile.get_block_idx()
        for block in pl.range(comm_core, local_t * (D // COMM_D_TILE), COMM_WORKERS):
            local_row = block // (D // COMM_D_TILE)
            col = block % (D // COMM_D_TILE) * COMM_D_TILE
            dst_row = tp_rank * local_t + local_row
            for peer_tp in pl.range(TP_Q_B_SIZE):
                peer = group_base + peer_tp
                pld.tensor.put(
                    dst=gather_window, peer=peer, src=x_local,
                    dst_offsets=[dst_row, col], src_offsets=[local_row, col], shape=[1, COMM_D_TILE],
                )

        for peer_tp in pl.range(TP_Q_B_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=gather_signal, peer=group_base + peer_tp,
                    offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="attn_tp_gather_wait", deps=[publish_tid]) as wait_tid:
        expected = pl.cast(COMM_WORKERS, pl.INT32)
        for source_tp in pl.range(TP_Q_B_SIZE):
            if source_tp != tp_rank:
                pld.system.wait(
                    signal=gather_signal, offsets=[source_tp, 0],
                    expected=expected, cmp=pld.WaitCmp.Ge,
                )

    group_t = local_t * TP_Q_B_SIZE
    with pl.spmd(COMM_WORKERS, name_hint="attn_tp_gather_copy", deps=[wait_tid]) as copy_tid:
        comm_core = pl.tile.get_block_idx()
        for block in pl.range(comm_core, group_t * (D // COMM_D_TILE), COMM_WORKERS):
            row = block // (D // COMM_D_TILE)
            col = block % (D // COMM_D_TILE) * COMM_D_TILE
            x_group[row : row + 1, col : col + COMM_D_TILE] = gather_window[row : row + 1, col : col + COMM_D_TILE]

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="attn_tp_gather_complete", deps=[copy_tid]):
        _completion_anchor = pl.read(x_group, [0, 0])
        for peer_tp in pl.range(TP_Q_B_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=gather_signal, peer=group_base + peer_tp,
                    offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
                )

        completion_expected = pl.cast(COMM_WORKERS + 1, pl.INT32)
        for source_tp in pl.range(TP_Q_B_SIZE):
            if source_tp != tp_rank:
                pld.system.wait(
                    signal=gather_signal, offsets=[source_tp, 0],
                    expected=completion_expected, cmp=pld.WaitCmp.Ge,
                )

        reset_value = pl.cast(-(COMM_WORKERS + 1), pl.INT32)
        for source_tp in pl.range(TP_Q_B_SIZE):
            if source_tp != tp_rank:
                pld.system.notify(
                    target=gather_signal, peer=my_rank,
                    offsets=[source_tp, 0], value=reset_value, op=pld.NotifyOp.AtomicAdd,
                )
    return x_group


@pl.jit.inline(auto_scope=False)
def reduce_scatter_fp32(
    partial: pl.Tensor,
    local_out: pl.Tensor,
    scatter_window: pld.DistributedTensor[[GROUP_T_MAX, D], pl.FP32],
    scatter_signal: pld.DistributedTensor[[TP_O_B_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    producer_dep: pl.Scalar[pl.TASK_ID],
):
    """Reduce FP32 rank partials and scatter contiguous token shards."""
    group_t = pl.tensor.dim(partial, 0)
    local_t = group_t // TP_O_B_SIZE
    group_base = my_rank // TP_O_B_SIZE * TP_O_B_SIZE
    tp_rank = my_rank % TP_O_B_SIZE

    with pl.spmd(COMM_WORKERS, name_hint="attn_tp_scatter_publish", deps=[producer_dep]) as publish_tid:
        comm_core = pl.tile.get_block_idx()
        for owner_tp in pl.range(TP_O_B_SIZE):
            for block in pl.range(comm_core, local_t * (D // COMM_D_TILE), COMM_WORKERS):
                owner_row = block // (D // COMM_D_TILE)
                col = block % (D // COMM_D_TILE) * COMM_D_TILE
                source_row = owner_tp * local_t + owner_row
                dst_row = tp_rank * local_t + owner_row
                peer = group_base + owner_tp
                pld.tensor.put(
                    dst=scatter_window, peer=peer, src=partial,
                    dst_offsets=[dst_row, col], src_offsets=[source_row, col], shape=[1, COMM_D_TILE],
                )

        for peer_tp in pl.range(TP_O_B_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=scatter_signal, peer=group_base + peer_tp,
                    offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="attn_tp_scatter_wait", deps=[publish_tid]) as wait_tid:
        expected = pl.cast(COMM_WORKERS, pl.INT32)
        for source_tp in pl.range(TP_O_B_SIZE):
            if source_tp != tp_rank:
                pld.system.wait(
                    signal=scatter_signal, offsets=[source_tp, 0],
                    expected=expected, cmp=pld.WaitCmp.Ge,
                )

    with pl.spmd(COMM_WORKERS, name_hint="attn_tp_scatter_reduce", deps=[wait_tid]) as reduce_tid:
        comm_core = pl.tile.get_block_idx()
        for block in pl.range(comm_core, local_t * (D // COMM_D_TILE), COMM_WORKERS):
            local_row = block // (D // COMM_D_TILE)
            col = block % (D // COMM_D_TILE) * COMM_D_TILE
            acc = pl.load(scatter_window, [local_row, col], [1, COMM_D_TILE])
            for source_tp in pl.range(1, TP_O_B_SIZE):
                source_row = source_tp * local_t + local_row
                source_partial = pl.load(scatter_window, [source_row, col], [1, COMM_D_TILE])
                acc = pl.add(acc, source_partial)
            reduced = pl.cast(acc, target_type=pl.BF16, mode="rint")
            pl.store(reduced, [local_row, col], local_out)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="attn_tp_scatter_complete", deps=[reduce_tid]):
        _completion_anchor = pl.read(local_out, [0, 0])
        for peer_tp in pl.range(TP_O_B_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=scatter_signal, peer=group_base + peer_tp,
                    offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
                )

        completion_expected = pl.cast(COMM_WORKERS + 1, pl.INT32)
        for source_tp in pl.range(TP_O_B_SIZE):
            if source_tp != tp_rank:
                pld.system.wait(
                    signal=scatter_signal, offsets=[source_tp, 0],
                    expected=completion_expected, cmp=pld.WaitCmp.Ge,
                )

        reset_value = pl.cast(-(COMM_WORKERS + 1), pl.INT32)
        for source_tp in pl.range(TP_O_B_SIZE):
            if source_tp != tp_rank:
                pld.system.notify(
                    target=scatter_signal, peer=my_rank,
                    offsets=[source_tp, 0], value=reset_value, op=pld.NotifyOp.AtomicAdd,
                )
    return local_out
