# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generation-checked device storage for prefill bootstrap hidden tails."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import EP, TP, FLASH as MODEL_CONFIG

REQUESTS_DYN = pl.dynamic("DSPARK_TAIL_REQUESTS_DYN")
GROUP_ROWS_DYN = pl.dynamic("DSPARK_TAIL_GROUP_ROWS_DYN")
LEASES_DYN = pl.dynamic("DSPARK_TAIL_LEASES_DYN")
DESCRIPTOR_WIDTH = 11
META_WIDTH = 6
TAIL_ROWS = 128
MAIN_DIM = 3 * MODEL_CONFIG.hidden_size
HIDDEN_TILE = 1024
LOCAL_ROWS_DYN = pl.dynamic("DSPARK_TAIL_LOCAL_ROWS_DYN")
GROUP_CAP = 8192
BOOTSTRAP_BATCH = 4


@pl.jit
def extract_prefill_tail_rank(
    selector: pl.Tensor[[4], pl.INT32],
    state_meta: pl.Tensor[[LEASES_DYN, META_WIDTH], pl.INT32],
    tail: pl.Tensor[[LEASES_DYN, TAIL_ROWS, MAIN_DIM], pl.BF16],
    positions: pl.Tensor[[LEASES_DYN, TAIL_ROWS], pl.INT32],
    hidden: pl.Out[pl.Tensor[[LOCAL_ROWS_DYN, MAIN_DIM], pl.BF16]],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Select one lease's chronological context band for each TP member."""
    state_meta.bind_dynamic(0, LEASES_DYN)
    tail.bind_dynamic(0, LEASES_DYN)
    positions.bind_dynamic(0, LEASES_DYN)
    hidden.bind_dynamic(0, LOCAL_ROWS_DYN)
    leases = pl.tensor.dim(state_meta, 0)
    rows = pl.tensor.dim(hidden, 0)
    flat_tail = pl.reshape(tail, [leases * TAIL_ROWS, MAIN_DIM])
    for core in pl.spmd(48, name_hint="dspark_extract_prefill_tail"):
        slot = pl.read(selector, [0])
        generation = pl.read(selector, [1])
        start = pl.read(selector, [2])
        length = pl.read(selector, [3])
        for index in pl.range(core, rows * (MAIN_DIM // HIDDEN_TILE), 48):
            row = index // (MAIN_DIM // HIDDEN_TILE)
            column = index % (MAIN_DIM // HIDDEN_TILE) * HIDDEN_TILE
            offset = tp_rank * rows + row
            values = pl.full([1, HIDDEN_TILE], dtype=pl.BF16, value=0.0)
            if slot >= 0 and slot < leases and offset < length:
                resident_generation = pl.read(state_meta, [slot, 1])
                position = start + offset
                resident_position = pl.read(positions, [slot, position % TAIL_ROWS])
                if resident_generation == generation and resident_position == position:
                    source_row = slot * TAIL_ROWS + position % TAIL_ROWS
                    values = flat_tail[source_row:source_row + 1, column:column + HIDDEN_TILE]
            hidden[row:row + 1, column:column + HIDDEN_TILE] = values
    return hidden


@pl.jit.host
def l3_extract_prefill_tail(
    selectors: pl.Tensor[[EP, 4], pl.INT32],
    state_meta: pl.Tensor[[EP, LEASES_DYN, META_WIDTH], pl.INT32],
    tail: pl.Tensor[[EP, LEASES_DYN, TAIL_ROWS, MAIN_DIM], pl.BF16],
    positions: pl.Tensor[[EP, LEASES_DYN, TAIL_ROWS], pl.INT32],
    hidden: pl.Out[pl.Tensor[[EP, LOCAL_ROWS_DYN, MAIN_DIM], pl.BF16]],
):
    """Feed ring-backed context to the existing drafter target-hidden ABI."""
    state_meta.bind_dynamic(1, LEASES_DYN)
    tail.bind_dynamic(1, LEASES_DYN)
    positions.bind_dynamic(1, LEASES_DYN)
    hidden.bind_dynamic(1, LOCAL_ROWS_DYN)
    for rank in pl.range(pld.world_size()):
        extract_prefill_tail_rank(
            selectors[rank], state_meta[rank], tail[rank], positions[rank], hidden[rank], rank % TP,
            device=rank,
        )


@pl.jit
def extract_prefill_tails_rank(
    selectors: pl.Tensor[[BOOTSTRAP_BATCH, 4], pl.INT32],
    state_meta: pl.Tensor[[LEASES_DYN, META_WIDTH], pl.INT32],
    tail: pl.Tensor[[LEASES_DYN, TAIL_ROWS, MAIN_DIM], pl.BF16],
    positions: pl.Tensor[[LEASES_DYN, TAIL_ROWS], pl.INT32],
    hidden: pl.Out[pl.Tensor[[LOCAL_ROWS_DYN, MAIN_DIM], pl.BF16]],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Interleave per-request prompt tails in the rank-local context slab."""
    state_meta.bind_dynamic(0, LEASES_DYN)
    tail.bind_dynamic(0, LEASES_DYN)
    positions.bind_dynamic(0, LEASES_DYN)
    hidden.bind_dynamic(0, LOCAL_ROWS_DYN)
    leases = pl.tensor.dim(state_meta, 0)
    rows = pl.tensor.dim(hidden, 0)
    per_request = rows // BOOTSTRAP_BATCH
    flat_tail = pl.reshape(tail, [leases * TAIL_ROWS, MAIN_DIM])
    for core in pl.spmd(48, name_hint="dspark_extract_batched_prefill_tail"):
        for index in pl.range(core, rows * (MAIN_DIM // HIDDEN_TILE), 48):
            row = index // (MAIN_DIM // HIDDEN_TILE)
            column = index % (MAIN_DIM // HIDDEN_TILE) * HIDDEN_TILE
            request = row // per_request
            offset = tp_rank * per_request + row % per_request
            slot = pl.read(selectors, [request, 0])
            generation = pl.read(selectors, [request, 1])
            start = pl.read(selectors, [request, 2])
            length = pl.read(selectors, [request, 3])
            values = pl.full([1, HIDDEN_TILE], dtype=pl.BF16, value=0.0)
            if slot >= 0 and slot < leases and offset < length:
                resident_generation = pl.read(state_meta, [slot, 1])
                position = start + offset
                resident_position = pl.read(positions, [slot, position % TAIL_ROWS])
                if resident_generation == generation and resident_position == position:
                    source_row = slot * TAIL_ROWS + position % TAIL_ROWS
                    values = flat_tail[source_row:source_row + 1, column:column + HIDDEN_TILE]
            hidden[row:row + 1, column:column + HIDDEN_TILE] = values
    return hidden


@pl.jit.host
def l3_extract_prefill_tails(
    selectors: pl.Tensor[[EP, BOOTSTRAP_BATCH, 4], pl.INT32],
    state_meta: pl.Tensor[[EP, LEASES_DYN, META_WIDTH], pl.INT32],
    tail: pl.Tensor[[EP, LEASES_DYN, TAIL_ROWS, MAIN_DIM], pl.BF16],
    positions: pl.Tensor[[EP, LEASES_DYN, TAIL_ROWS], pl.INT32],
    hidden: pl.Out[pl.Tensor[[EP, LOCAL_ROWS_DYN, MAIN_DIM], pl.BF16]],
):
    state_meta.bind_dynamic(1, LEASES_DYN)
    tail.bind_dynamic(1, LEASES_DYN)
    positions.bind_dynamic(1, LEASES_DYN)
    hidden.bind_dynamic(1, LOCAL_ROWS_DYN)
    for rank in pl.range(pld.world_size()):
        extract_prefill_tails_rank(
            selectors[rank], state_meta[rank], tail[rank], positions[rank], hidden[rank], rank % TP,
            device=rank,
        )


@pl.jit.inline
def store_prefill_tail(
    descriptors: pl.Tensor[[REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    state_meta: pl.Tensor[[LEASES_DYN, META_WIDTH], pl.INT32],
    group_hidden: pl.Tensor[[GROUP_ROWS_DYN, MAIN_DIM], pl.BF16],
    tail: pl.Tensor[[LEASES_DYN, TAIL_ROWS, MAIN_DIM], pl.BF16],
    positions: pl.Tensor[[LEASES_DYN, TAIL_ROWS], pl.INT32],
):
    """Store only the last window of live rows owned by the current generation."""
    descriptors.bind_dynamic(0, REQUESTS_DYN)
    state_meta.bind_dynamic(0, LEASES_DYN)
    group_hidden.bind_dynamic(0, GROUP_ROWS_DYN)
    tail.bind_dynamic(0, LEASES_DYN)
    positions.bind_dynamic(0, LEASES_DYN)
    requests = pl.tensor.dim(descriptors, 0)
    leases = pl.tensor.dim(state_meta, 0)
    flat_tail = pl.reshape(tail, [leases * TAIL_ROWS, MAIN_DIM])
    for core in pl.spmd(48, name_hint="dspark_prefill_tail_hidden"):
        for request in pl.range(requests):
            slot = pl.read(descriptors, [request, 0])
            generation = pl.read(descriptors, [request, 1])
            length = pl.read(descriptors, [request, 4])
            if slot >= 0 and slot < leases and length > 0:
                resident_generation = pl.read(state_meta, [slot, 1])
                if generation == resident_generation:
                    start = pl.read(descriptors, [request, 3])
                    packed_start = pl.read(descriptors, [request, 6])
                    first = pl.cast(pl.max(0, length - TAIL_ROWS), pl.INT32)
                    work = (length - first) * (MAIN_DIM // HIDDEN_TILE)
                    for index in pl.range(core, work, 48):
                        offset = first + index // (MAIN_DIM // HIDDEN_TILE)
                        column = index % (MAIN_DIM // HIDDEN_TILE) * HIDDEN_TILE
                        source_row = packed_start + offset
                        target_row = slot * TAIL_ROWS + (start + offset) % TAIL_ROWS
                        flat_tail[target_row:target_row + 1, column:column + HIDDEN_TILE] = group_hidden[
                            source_row:source_row + 1, column:column + HIDDEN_TILE
                        ]
    # One scalar writer owns all position cache lines, including partial windows.
    for _core in pl.spmd(1, name_hint="dspark_prefill_tail_positions"):
        for request in pl.range(requests):
            slot = pl.read(descriptors, [request, 0])
            generation = pl.read(descriptors, [request, 1])
            length = pl.read(descriptors, [request, 4])
            if slot >= 0 and slot < leases and length > 0:
                resident_generation = pl.read(state_meta, [slot, 1])
                if generation == resident_generation:
                    start = pl.read(descriptors, [request, 3])
                    first = pl.cast(pl.max(0, length - TAIL_ROWS), pl.INT32)
                    for offset in pl.range(first, length):
                        position = start + offset
                        pl.write(positions, [slot, position % TAIL_ROWS], pl.cast(position, pl.INT32))
    return pl.reshape(flat_tail, [leases, TAIL_ROWS, MAIN_DIM]), positions


@pl.jit
def capture_prefill_tail_rank(
    descriptors: pl.Tensor[[REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    state_meta: pl.Tensor[[LEASES_DYN, META_WIDTH], pl.INT32],
    hidden_local: pl.Tensor[[LOCAL_ROWS_DYN, MAIN_DIM], pl.BF16],
    tail: pl.InOut[pl.Tensor[[LEASES_DYN, TAIL_ROWS, MAIN_DIM], pl.BF16]],
    positions: pl.InOut[pl.Tensor[[LEASES_DYN, TAIL_ROWS], pl.INT32]],
    gather_window: pl.InOut[pld.DistributedTensor[[GROUP_CAP, MAIN_DIM], pl.BF16]],
    gather_signal: pl.InOut[pld.DistributedTensor[[TP, 1], pl.INT32]],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Gather rank-major taps and retain each lease's tail entirely on device."""
    descriptors.bind_dynamic(0, REQUESTS_DYN)
    state_meta.bind_dynamic(0, LEASES_DYN)
    hidden_local.bind_dynamic(0, LOCAL_ROWS_DYN)
    tail.bind_dynamic(0, LEASES_DYN)
    positions.bind_dynamic(0, LEASES_DYN)
    local_rows = pl.tensor.dim(hidden_local, 0)
    local_t = pl.cast(local_rows, pl.INT32)
    group_hidden = pl.create_tensor([local_rows * TP, MAIN_DIM], dtype=pl.BF16)
    with pl.spmd(TP, name_hint="dspark_tail_push", allow_early_resolve=True) as push_tid:
        peer = pl.tile.get_block_idx()
        pld.tensor.put(
            dst=gather_window, peer=group_base + peer, src=hidden_local,
            dst_offsets=[tp_rank * local_t, 0], src_offsets=[0, 0], shape=[local_t, MAIN_DIM],
            chunk_rows=1, chunk_cols=HIDDEN_TILE, pipeline=True,
        )
        if peer != tp_rank:
            pld.system.notify(
                target=gather_signal, peer=group_base + peer, offsets=[tp_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_tail_wait") as wait_tid:
        for peer in pl.range(TP):
            if peer != tp_rank:
                pld.system.defer_wait(
                    signal=gather_signal, offsets=[peer, 0], expected=pl.cast(1, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )
    with pl.spmd(48, name_hint="dspark_tail_readback", deps=[push_tid, wait_tid]) as read_tid:
        core = pl.tile.get_block_idx()
        for index in pl.range(core, local_rows * TP * (MAIN_DIM // HIDDEN_TILE), 48):
            row = index // (MAIN_DIM // HIDDEN_TILE)
            column = index % (MAIN_DIM // HIDDEN_TILE) * HIDDEN_TILE
            values = pl.slice(gather_window, [1, HIDDEN_TILE], [row, column])
            group_hidden[row:row + 1, column:column + HIDDEN_TILE] = values
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_tail_ack", deps=[read_tid]) as ack_tid:
        for peer in pl.range(TP):
            if peer != tp_rank:
                pld.system.notify(
                    target=gather_signal, peer=group_base + peer, offsets=[tp_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_tail_ack_wait") as ack_wait_tid:
        for peer in pl.range(TP):
            if peer != tp_rank:
                pld.system.defer_wait(
                    signal=gather_signal, offsets=[peer, 0], expected=pl.cast(2, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="dspark_tail_retire", deps=[ack_tid, ack_wait_tid]
    ):
        anchor = pl.read(group_hidden, [0, 0])
        for peer in pl.range(TP):
            if peer != tp_rank:
                pld.system.notify(
                    target=gather_signal, peer=group_base + tp_rank, offsets=[peer, 0],
                    value=pl.cast(-2, pl.INT32), op=pld.NotifyOp.AtomicAdd,
                )
        pl.write(group_hidden, [0, 0], anchor)
    stored_tail, stored_positions = store_prefill_tail(descriptors, state_meta, group_hidden, tail, positions)
    return stored_tail, stored_positions, gather_signal


@pl.jit.host
def l3_capture_prefill_tail(
    descriptors: pl.Tensor[[EP, REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    state_meta: pl.Tensor[[EP, LEASES_DYN, META_WIDTH], pl.INT32],
    hidden_local: pl.Tensor[[EP, LOCAL_ROWS_DYN, MAIN_DIM], pl.BF16],
    tail: pl.InOut[pl.Tensor[[EP, LEASES_DYN, TAIL_ROWS, MAIN_DIM], pl.BF16]],
    positions: pl.InOut[pl.Tensor[[EP, LEASES_DYN, TAIL_ROWS], pl.INT32]],
):
    """Capture every TP group's local prefill taps into replicated lease rings."""
    descriptors.bind_dynamic(1, REQUESTS_DYN)
    state_meta.bind_dynamic(1, LEASES_DYN)
    hidden_local.bind_dynamic(1, LOCAL_ROWS_DYN)
    tail.bind_dynamic(1, LEASES_DYN)
    positions.bind_dynamic(1, LEASES_DYN)
    window_buffer = pld.alloc_window_buffer([GROUP_CAP, MAIN_DIM], dtype=pl.BF16)
    signal_buffer = pld.alloc_window_buffer([TP, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        window = pld.window(window_buffer, [GROUP_CAP, MAIN_DIM], dtype=pl.BF16)
        signal = pld.window(signal_buffer, [TP, 1], dtype=pl.INT32)
        capture_prefill_tail_rank(
            descriptors[rank], state_meta[rank], hidden_local[rank], tail[rank], positions[rank],
            window, signal, rank // TP * TP, rank % TP, device=rank,
        )


@pl.jit
def store_prefill_tail_rank(
    descriptors: pl.Tensor[[REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    state_meta: pl.Tensor[[LEASES_DYN, META_WIDTH], pl.INT32],
    group_hidden: pl.Tensor[[GROUP_ROWS_DYN, MAIN_DIM], pl.BF16],
    tail: pl.InOut[pl.Tensor[[LEASES_DYN, TAIL_ROWS, MAIN_DIM], pl.BF16]],
    positions: pl.InOut[pl.Tensor[[LEASES_DYN, TAIL_ROWS], pl.INT32]],
):
    return store_prefill_tail(descriptors, state_meta, group_hidden, tail, positions)


@pl.jit.host
def l3_store_prefill_tail(
    descriptors: pl.Tensor[[EP, REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    state_meta: pl.Tensor[[EP, LEASES_DYN, META_WIDTH], pl.INT32],
    group_hidden: pl.Tensor[[EP, GROUP_ROWS_DYN, MAIN_DIM], pl.BF16],
    tail: pl.InOut[pl.Tensor[[EP, LEASES_DYN, TAIL_ROWS, MAIN_DIM], pl.BF16]],
    positions: pl.InOut[pl.Tensor[[EP, LEASES_DYN, TAIL_ROWS], pl.INT32]],
):
    """Store TP-replicated hidden rows after the prefill tap gather."""
    descriptors.bind_dynamic(1, REQUESTS_DYN)
    state_meta.bind_dynamic(1, LEASES_DYN)
    group_hidden.bind_dynamic(1, GROUP_ROWS_DYN)
    tail.bind_dynamic(1, LEASES_DYN)
    positions.bind_dynamic(1, LEASES_DYN)
    for rank in pl.range(pld.world_size()):
        store_prefill_tail_rank(
            descriptors[rank], state_meta[rank], group_hidden[rank], tail[rank], positions[rank],
            device=rank,
        )
