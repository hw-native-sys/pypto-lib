# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys

_TP_CHOICES = (1, 2, 4)

_TP_DEFAULT = 2


def _parse_int_argv(flag: str, default: int) -> int:
    for index, arg in enumerate(sys.argv):
        if arg == flag and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if arg.startswith(f"{flag}="):
            return int(arg.split("=", 1)[1])
    return default


TP_SIZE = _parse_int_argv("--tp", _TP_DEFAULT)

if TP_SIZE not in _TP_CHOICES:
    raise ValueError(f"--tp must be one of {_TP_CHOICES} (got {TP_SIZE})")

import pypto.language as pl

import pypto.language.distributed as pld

from .config import FLASH as M

CP_Q_T_DYN = pl.dynamic("NATIVE_D_PREFILL_CP_Q_T_DYN")

CP_KV_T_DYN = pl.dynamic("NATIVE_D_PREFILL_CP_KV_T_DYN")

D = M.hidden_size

PREFILL_GROUP_CAP = 8192

COMM_ROW_TILE = 8

READBACK_ROW_TILE = 16

READBACK_WORKERS = 16


def _prefill_cp_token_allgather_step(
    hidden_local: pl.Tensor[[CP_Q_T_DYN, D], pl.BF16],
    group_out: pl.Out[pl.Tensor[[CP_KV_T_DYN, D], pl.BF16]],
    gather_window: pl.InOut[pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16]],
    gather_signal: pl.InOut[pld.DistributedTensor[[TP_SIZE, 1], pl.INT32]],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    prior_cycle_gate: pl.Scalar[pl.TASK_ID],
):
    """Gather rank-major rows and retire the complete two-phase signal epoch."""
    hidden_local.bind_dynamic(0, CP_Q_T_DYN)
    group_out.bind_dynamic(0, CP_KV_T_DYN)
    local_rows = pl.tensor.dim(hidden_local, 0)
    local_t = pl.cast(local_rows, pl.INT32)
    target_row = tp_rank * local_t

    # Each peer has an independent destination and arrival signal.
    with pl.spmd(
        TP_SIZE,
        name_hint="prefill_cp_token_allgather_push",
        allow_early_resolve=True,
        deps=[prior_cycle_gate],
    ) as _push_tid:
        peer_tp = pl.tile.get_block_idx()
        pld.tensor.put(
            dst=gather_window,
            peer=group_base + peer_tp,
            src=hidden_local,
            dst_offsets=[target_row, 0],
            src_offsets=[0, 0],
            shape=[local_t, D],
            chunk_rows=COMM_ROW_TILE,
            chunk_cols=D,
            pipeline=True,
        )
        if peer_tp != tp_rank:
            pld.system.notify(
                target=gather_signal,
                peer=group_base + peer_tp,
                offsets=[tp_rank, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )

    # Register the peer payload conditions as deferred completion.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_cp_token_allgather_payload_wait",
        deps=[prior_cycle_gate],
    ) as _payload_wait_tid:
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.defer_wait(
                    signal=gather_signal,
                    offsets=[source_tp, 0],
                    expected=pl.cast(1, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    # Fixed readers own strided row tiles; release the window after every reader.
    group_rows = TP_SIZE * local_rows
    full_rows = (group_rows // READBACK_ROW_TILE) * READBACK_ROW_TILE
    with pl.spmd(
        READBACK_WORKERS,
        name_hint="prefill_cp_token_allgather_readback",
        deps=[_push_tid, _payload_wait_tid],
    ) as _readback_tid:
        worker = pl.tile.get_block_idx()
        for tile_row in pl.range(
            worker * READBACK_ROW_TILE,
            full_rows,
            READBACK_WORKERS * READBACK_ROW_TILE,
        ):
            window_tile = gather_window[tile_row : tile_row + READBACK_ROW_TILE, 0:D]
            group_out[tile_row : tile_row + READBACK_ROW_TILE, 0:D] = window_tile
        for tail_row in pl.range(full_rows + worker, group_rows, READBACK_WORKERS):
            window_row = gather_window[tail_row : tail_row + 1, 0:D]
            group_out[tail_row : tail_row + 1, 0:D] = window_row
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_cp_token_allgather_readback_notify",
        deps=[_readback_tid],
    ) as _readback_notify_tid:
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=gather_signal,
                    peer=group_base + peer_tp,
                    offsets=[tp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_cp_token_allgather_readback_wait",
        deps=[prior_cycle_gate],
    ) as _readback_wait_tid:
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.defer_wait(
                    signal=gather_signal,
                    offsets=[source_tp, 0],
                    expected=pl.cast(2, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    # Retire peer credits and anchor output consumption to signal retirement.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_cp_token_allgather_retire",
        deps=[_readback_notify_tid, _readback_wait_tid],
    ):
        completion_anchor = pl.read(group_out, [0, 0])
        reset_value = pl.cast(-2, pl.INT32)
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.notify(
                    target=gather_signal,
                    peer=group_base + tp_rank,
                    offsets=[source_tp, 0],
                    value=reset_value,
                    op=pld.NotifyOp.AtomicAdd,
                )
        pl.write(group_out, [0, 0], completion_anchor)

    return group_out, gather_signal


prefill_cp_token_allgather_step = pl.jit.inline(_prefill_cp_token_allgather_step)
