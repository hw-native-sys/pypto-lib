# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Publish disjoint exact router columns and retire two bounded exchange lanes."""

import pypto.language as pl
import pypto.language.distributed as pld

from .prefill_moe_geometry import GATE_WINDOW_ROWS, N_EXPERTS, N_RANKS, T_PAD

GATE_COPY_TILE = 16
GATE_COLUMNS = N_EXPERTS // N_RANKS


@pl.jit.inline(auto_scope=False)
def exchange_gate_columns(
    logits: pl.Tensor[[T_PAD, N_EXPERTS], pl.FP32],
    window: pld.DistributedTensor[[GATE_WINDOW_ROWS, N_EXPERTS], pl.FP32],
    signal: pld.DistributedTensor[[N_RANKS, 16], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    active: pl.Scalar[pl.INDEX],
    computed: pl.Scalar[pl.TASK_ID],
    prior_copy: pl.Scalar[pl.TASK_ID],
):
    lane_base = pl.cast((epoch - 1) % 2, pl.INDEX) * T_PAD
    column = pl.cast(rank, pl.INDEX) * GATE_COLUMNS
    tasks = pl.array.create(3, pl.TASK_ID)
    with pl.manual_scope():
        with pl.spmd(N_RANKS, name_hint="prefill_gate_publish", deps=[computed, prior_copy]) as published:
            peer = pl.tile.get_block_idx()
            pld.tensor.put(dst=window, peer=peer, src=logits, dst_offsets=[lane_base, column], src_offsets=[0, column], shape=[active, GATE_COLUMNS], chunk_rows=GATE_COPY_TILE, chunk_cols=GATE_COLUMNS)
            pld.system.notify(signal, peer, [rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        tasks[0] = published
    with pl.manual_scope():
        published_scalar = tasks[0]
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_gate_defer", deps=[published_scalar]) as waited:
            for source in pl.range(N_RANKS):
                pld.system.defer_wait(signal, [source, 0], epoch, cmp=pld.WaitCmp.Ge)
        tasks[1] = waited
    with pl.manual_scope():
        waited_scalar = tasks[1]
        blocks = (active + GATE_COPY_TILE - 1) // GATE_COPY_TILE
        with pl.spmd(blocks, name_hint="prefill_gate_collect", deps=[waited_scalar]) as copied:
            start = pl.tile.get_block_idx() * GATE_COPY_TILE
            rows = pl.min(GATE_COPY_TILE, active - start)
            value = pl.load(window, [lane_base + start, 0], [GATE_COPY_TILE, N_EXPERTS])
            valid = pl.set_validshape(value, rows, N_EXPERTS)
            pl.store(valid, [start, 0], logits)
        tasks[2] = copied
    return tasks[2]


@pl.jit.inline(auto_scope=False)
def retire_gate_reservation(
    signal: pld.DistributedTensor[[N_RANKS, 16], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    unused: pl.Scalar[pl.INT32],
    completed: pl.Scalar[pl.TASK_ID],
):
    with pl.spmd(N_RANKS, name_hint="prefill_gate_epoch_padding", deps=[completed]) as retired:
        peer = pl.tile.get_block_idx()
        pld.system.notify(signal, peer, [rank, 0], value=unused, op=pld.NotifyOp.AtomicAdd)
    return retired
