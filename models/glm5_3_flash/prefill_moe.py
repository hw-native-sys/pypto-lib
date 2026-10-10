# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""The sparse layer's routed half at prefill scale: pack -> exchange -> experts -> scatter back.

Capacity model. ``--tokens`` is this rank's token capacity ``T``. Routing is
dropless, so one source may legally send all ``T * TOPK`` of its routes to a
single destination: every peer lane is sized ``T * TOPK`` rows, and a rank's
receive windows hold ``N_RANKS`` such lanes. Only live rows cross the wire; the
static capacity is the dropless upper bound and is never computed over.

This is the prefill counterpart of ``decode_moe.py`` — same six stages (pack,
arrive, regroup, experts, return, combine), but the decode entry's serial
per-destination pack scan is replaced by a count/prefix pass with parallel
payload movement, and the transports are capacity-sized:

* ``prefill_moe_dispatch`` counts each route's ``(destination, local expert)``
  lane once in a single InCore task (concurrent INT32 count updates would drop
  cache lines, and one writer gives the pack stable expert-major slots), packs
  every token's INT8 row plus a 4-wide aux row (dequant scale, routing weight,
  route id, source) into the source's contiguous send buffer, exchanges the
  per-destination count rows, pushes the payload in four disjoint row streams
  per destination, and re-sorts each local expert's received rows source-major
  into the slab ``expert_routed`` consumes (expert slabs are ROW_TILE-aligned,
  so the resort's per-field scalar writes own whole cache lines).
* ``expert_routed`` runs the local experts over the slab; its ``expert_offsets``
  are the prefix sums of the exchanged expert counts.
* ``prefill_moe_return`` scatters every live expert row back to the rank that
  routed it (slot ``token * TOPK + k``) and sums the TOPK slots per token.

The barriers are epoch-keyed ``AtomicAdd(1)`` notifies — every rank notifies
each peer exactly once per phase per call — so ``signal[src] >= epoch`` retires
call N+1 only after call N's readers, and one set of windows serves every layer
with no clear. The L3 entry runs two calls to exercise exactly that reuse.

Run (EP2, two ranks)::

    python models/glm5_3_flash/prefill_moe.py -p a2a3sim -d 0,1 --tp 2 --ep 2
    python models/glm5_3_flash/prefill_moe.py -p a2a3    -d 0,1 --tp 2 --ep 2
"""

import sys
from pathlib import Path

import pypto.language as pl
import pypto.language.distributed as pld
import torch

# ``--tp`` / ``--ep`` are read from argv by config.py at import time and every
# sub-kernel inherits the frozen shapes. config.py's own default is the 16-die
# deployment shape, while the CI sweeps run each entry at its default world size
# (ep2 / 2-card, see ``# ci: devices=2``), so pin that bring-up default before
# the first ``models.glm5_3_flash`` import when the command line pins neither axis.
if not any(tok in ("--tp", "--ep") or tok.startswith(("--tp=", "--ep=")) for tok in sys.argv):
    sys.argv += ["--tp", "2", "--ep", "2"]

from models.glm5_3_flash.config import D, EP_SIZE, FLASH, N_LOCAL_EXPERTS, TOPK
from models.glm5_3_flash.expert_routed import expert_routed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _parse_argv_int(flag: str, default: int) -> int:
    for index, tok in enumerate(sys.argv):
        if tok == flag and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if tok.startswith(f"{flag}="):
            return int(tok.split("=", 1)[1])
    return default


N_RANKS = EP_SIZE
MOE_INTER = FLASH.moe_intermediate_size
T = _parse_argv_int("--tokens", 64)
ROUTES_PER_SRC = T * TOPK
PEER_CAP = ROUTES_PER_SRC          # a source may send every route to one peer
TOTAL_CAP = N_RANKS * PEER_CAP     # dropless per-rank receive capacity

# tiling
AUX_SCALE, AUX_W, AUX_ROUTE, AUX_SRC = 0, 1, 2, 3
AUX_PAD = 16              # one 64-byte FP32 line per packed row
# One line-padded INT32 row per (destination, source): the count exchange moves
# this rank's per-local-expert counts, so the row width is the padded local
# expert count, and every source's row owns whole 64-byte cache lines.
COUNT_PAD = ((N_LOCAL_EXPERTS + 15) // 16) * 16
PACK_BLOCKS = 48          # source-side token pack width
ROW_TILE = 16             # bulk transfer / resort row tile
PUT_STREAMS = 4           # disjoint row streams per destination
RESORT_BLOCKS_PER_EXPERT = 1
RETURN_STREAMS = 4        # disjoint row streams per local expert on the way back
COMBINE_TILE = 16

# Every local expert's slab starts on a ROW_TILE boundary. The resort's
# per-field scalar writes (scale, weight, route, source) then own whole 64-byte
# cache lines per expert, so two blocks can never lose each other's stores to a
# shared line; the alignment pad rows the expert kernel computes are zeroed and
# never returned.
GROUPED_CAP = TOTAL_CAP + N_LOCAL_EXPERTS * (ROW_TILE - 1)

assert N_RANKS >= 1
assert T > 0 and T % COMBINE_TILE == 0
assert PEER_CAP >= 1


@pl.jit.inline
def prefill_moe_dispatch(
    route_indices: pl.Tensor[[T, TOPK], pl.INT32],
    route_weights: pl.Tensor[[T, TOPK], pl.FP32],
    x_int8: pl.Tensor[[T, D], pl.INT8],
    x_scale: pl.Tensor[[T, 1], pl.FP32],
    send_counts: pl.Tensor[[N_RANKS, 1], pl.INT32],
    recv_expert_counts: pl.Tensor[[N_RANKS, N_LOCAL_EXPERTS], pl.INT32],
    expert_counts: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    expert_offsets: pl.Tensor[[N_LOCAL_EXPERTS + 1], pl.INT32],
    lane_x: pld.DistributedTensor[[TOTAL_CAP, D], pl.INT8],
    lane_aux: pld.DistributedTensor[[TOTAL_CAP, AUX_PAD], pl.FP32],
    lane_count: pld.DistributedTensor[[N_RANKS, COUNT_PAD], pl.INT32],
    count_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    x_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    slab_x: pl.Tensor[[GROUPED_CAP, D], pl.INT8],
    slab_scale: pl.Tensor[[GROUPED_CAP, 1], pl.FP32],
    slab_weight: pl.Tensor[[GROUPED_CAP, 1], pl.FP32],
    slab_route: pl.Tensor[[GROUPED_CAP, 1], pl.INT32],
    slab_src: pl.Tensor[[GROUPED_CAP, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Count/pack this rank's routes, exchange them, and regroup by local expert."""
    x_send = pl.create_tensor([ROUTES_PER_SRC, D], dtype=pl.INT8, manual_dep=True)
    aux_send = pl.create_tensor([ROUTES_PER_SRC, AUX_PAD], dtype=pl.FP32, manual_dep=True)
    slot_map = pl.create_tensor([ROUTES_PER_SRC, 1], dtype=pl.INT32, manual_dep=True)
    send_expert_counts = pl.create_tensor([N_RANKS, N_LOCAL_EXPERTS], dtype=pl.INT32, manual_dep=True)

    active = pl.cast(num_tokens, pl.INDEX)
    if active < 0:
        active = pl.cast(0, pl.INDEX)
    if active > T:
        active = pl.cast(T, pl.INDEX)

    # Count once in a single InCore task: concurrent INT32 count updates would
    # drop cache lines, and one writer gives the pack stable expert-major slots.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_dispatch_count") as count_tid:
        counts = pl.array.create(N_RANKS * N_LOCAL_EXPERTS, pl.INT32)
        for lane in pl.range(N_RANKS * N_LOCAL_EXPERTS):
            counts[lane] = 0
        for token in pl.range(active):
            for k in pl.range(TOPK):
                expert = pl.read(route_indices, [token, k])
                counts[expert] = counts[expert] + 1
        for dst in pl.range(N_RANKS):
            rank_total = pl.const(0, pl.INT32)
            for e in pl.range(N_LOCAL_EXPERTS):
                count = counts[dst * N_LOCAL_EXPERTS + e]
                pl.write(send_expert_counts, [dst, e], count)
                rank_total = rank_total + count
            pl.write(send_counts, [dst, 0], rank_total)
        # Stable expert-major slot per route; the packed order is
        # (destination, local expert, token, k) so one destination's rows are
        # contiguous and its experts grouped.
        cursor = pl.array.create(N_RANKS * N_LOCAL_EXPERTS, pl.INT32)
        prefix = pl.const(0, pl.INT32)
        for lane in pl.range(N_RANKS * N_LOCAL_EXPERTS):
            cursor[lane] = prefix
            prefix = prefix + counts[lane]
        for token in pl.range(active):
            for k in pl.range(TOPK):
                expert = pl.read(route_indices, [token, k])
                slot = cursor[expert]
                pl.write(slot_map, [token * TOPK + k, 0], slot)
                cursor[expert] = slot + 1

    # Pack every token's INT8 row into its routes' slots. The aux row carries
    # the dequant scale, the routing weight, the route id and the source, so the
    # expert side needs no route metadata from the wire structure.
    with pl.spmd(
        PACK_BLOCKS, name_hint="prefill_dispatch_pack", deps=[count_tid], allow_early_resolve=True
    ) as pack_tid:
        pack_block = pl.tile.get_block_idx()
        for token in pl.range(pack_block, active, PACK_BLOCKS):
            hidden_row = x_int8[token : token + 1, :]
            token_scale = pl.read(x_scale, [token, 0])
            for k in pl.range(TOPK):
                route = token * TOPK + k
                slot = pl.read(slot_map, [route, 0])
                row = pl.cast(slot, pl.INDEX)
                x_send[row : row + 1, :] = hidden_row
                aux = pl.tile.full([1, AUX_PAD], dtype=pl.FP32, value=0.0)
                pl.tile.write(aux, [0, AUX_SCALE], token_scale)
                pl.tile.write(aux, [0, AUX_W], pl.read(route_weights, [token, k]))
                pl.tile.write(aux, [0, AUX_ROUTE], pl.cast(pl.cast(route, pl.INT32), pl.FP32))
                pl.tile.write(aux, [0, AUX_SRC], pl.cast(my_rank, pl.FP32))
                pl.tile.store(aux, [row, 0], aux_send)

    # Exchange the per-destination count rows: each destination learns how many
    # rows every source sent for each of its local experts, and this rank's own
    # receive counts, expert totals and offsets come out of the same task.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="prefill_dispatch_count_exchange", deps=[count_tid]
    ) as count_exchange_tid:
        for dest in pl.range(N_RANKS):
            count_row = pl.tile.full([1, COUNT_PAD], dtype=pl.INT32, value=0)
            for e in pl.range(N_LOCAL_EXPERTS):
                pl.tile.write(count_row, [0, e], pl.read(send_expert_counts, [dest, e]))
            pld.tile.remote_store(count_row, target=lane_count, peer=dest, offsets=[my_rank, 0])
            if dest != my_rank:
                pld.system.notify(
                    target=count_ready,
                    peer=dest,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(
                    signal=count_ready, offsets=[src, 0], expected=epoch, cmp=pld.WaitCmp.Ge
                )
        for e in pl.range(N_LOCAL_EXPERTS):
            expert_total = pl.const(0, pl.INT32)
            for src in pl.range(N_RANKS):
                count = pl.read(lane_count, [src, e])
                pl.write(recv_expert_counts, [src, e], count)
                expert_total = expert_total + count
            pl.write(expert_counts, [e, 0], expert_total)
        running = pl.const(0, pl.INT32)
        for e in pl.range(N_LOCAL_EXPERTS):
            pl.write(expert_offsets, [e], running)
            count = pl.read(expert_counts, [e, 0])
            aligned = pl.cast(((count + (ROW_TILE - 1)) // ROW_TILE) * ROW_TILE, pl.INT32)
            running = running + aligned
        pl.write(expert_offsets, [N_LOCAL_EXPERTS], running)

    # Push each destination its contiguous packed range in four disjoint row
    # streams; the aux rail rides the same rows and the same completion notify.
    with pl.spmd(
        N_RANKS * PUT_STREAMS,
        name_hint="prefill_dispatch_put",
        deps=[pack_tid, count_exchange_tid],
        allow_early_resolve=False,
    ) as put_tid:
        put_block = pl.tile.get_block_idx()
        dest = put_block // PUT_STREAMS
        stream = put_block % PUT_STREAMS
        n_rows = pl.cast(pl.read(send_counts, [dest, 0]), pl.INDEX)
        if n_rows < 0:
            n_rows = pl.cast(0, pl.INDEX)
        if n_rows > PEER_CAP:
            n_rows = pl.cast(PEER_CAP, pl.INDEX)
        src_base = pl.cast(0, pl.INDEX)
        for prior in pl.range(dest):
            src_base = src_base + pl.cast(pl.read(send_counts, [prior, 0]), pl.INDEX)
        dst_base = pl.cast(my_rank, pl.INDEX) * PEER_CAP
        bulk = (n_rows // ROW_TILE) * ROW_TILE
        for row in pl.range(stream * ROW_TILE, bulk, PUT_STREAMS * ROW_TILE):
            pld.tensor.put(
                dst=lane_x, peer=dest, src=x_send,
                dst_offsets=[dst_base + row, 0], src_offsets=[src_base + row, 0],
                shape=[ROW_TILE, D],
            )
            pld.tensor.put(
                dst=lane_aux, peer=dest, src=aux_send,
                dst_offsets=[dst_base + row, 0], src_offsets=[src_base + row, 0],
                shape=[ROW_TILE, AUX_PAD],
            )
        for row in pl.range(bulk + stream, n_rows, PUT_STREAMS):
            pld.tensor.put(
                dst=lane_x, peer=dest, src=x_send,
                dst_offsets=[dst_base + row, 0], src_offsets=[src_base + row, 0],
                shape=[1, D],
            )
            pld.tensor.put(
                dst=lane_aux, peer=dest, src=aux_send,
                dst_offsets=[dst_base + row, 0], src_offsets=[src_base + row, 0],
                shape=[1, AUX_PAD],
            )

    # Payload barrier: the counts retired before the payload, so a receiver's
    # count-driven copies can trim every source lane before reading it.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="prefill_dispatch_signal", deps=[put_tid]
    ) as payload_signal_tid:
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.notify(
                    target=x_ready,
                    peer=peer,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(signal=x_ready, offsets=[src, 0], expected=epoch, cmp=pld.WaitCmp.Ge)

    # Re-sort each local expert's received rows source-major into the slab the
    # expert kernel consumes; the aux fields move with their payload row.
    with pl.spmd(
        N_LOCAL_EXPERTS * RESORT_BLOCKS_PER_EXPERT,
        name_hint="prefill_dispatch_resort",
        deps=[count_exchange_tid, payload_signal_tid],
        allow_early_resolve=False,
    ) as resort_tid:
        resort_block = pl.tile.get_block_idx()
        local_e = resort_block // RESORT_BLOCKS_PER_EXPERT
        stream = resort_block % RESORT_BLOCKS_PER_EXPERT
        n_expert = pl.cast(pl.read(expert_counts, [local_e, 0]), pl.INDEX)
        if n_expert > 0:
            expert_base = pl.cast(pl.read(expert_offsets, [local_e]), pl.INDEX)
            source_prefix = pl.cast(0, pl.INDEX)
            for src in pl.range(N_RANKS):
                n_rows = pl.cast(pl.read(recv_expert_counts, [src, local_e]), pl.INDEX)
                src_expert_base = pl.cast(0, pl.INDEX)
                for prior_e in pl.range(local_e):
                    src_expert_base = src_expert_base + pl.cast(
                        pl.read(recv_expert_counts, [src, prior_e]), pl.INDEX
                    )
                input_base = src * PEER_CAP + src_expert_base
                output_base = expert_base + source_prefix
                bulk = (n_rows // ROW_TILE) * ROW_TILE
                for row in pl.range(stream * ROW_TILE, bulk, RESORT_BLOCKS_PER_EXPERT * ROW_TILE):
                    slab_x[output_base + row : output_base + row + ROW_TILE, :] = lane_x[
                        input_base + row : input_base + row + ROW_TILE, :
                    ]
                    # The expert slab is ROW_TILE-aligned, so this block's scalar
                    # field writes own whole 64-byte lines and cannot lose stores
                    # to another block. The compiler's static heuristic cannot
                    # prove the dynamic alignment and warns; safe by construction.
                    for inner in pl.range(ROW_TILE):
                        row_out = output_base + row + inner
                        row_in = input_base + row + inner
                        pl.write(slab_scale, [row_out, 0], pl.read(lane_aux, [row_in, AUX_SCALE]))
                        pl.write(slab_weight, [row_out, 0], pl.read(lane_aux, [row_in, AUX_W]))
                        pl.write(
                            slab_route, [row_out, 0],
                            pl.cast(pl.read(lane_aux, [row_in, AUX_ROUTE]), pl.INT32),
                        )
                        pl.write(
                            slab_src, [row_out, 0],
                            pl.cast(pl.read(lane_aux, [row_in, AUX_SRC]), pl.INT32),
                        )
                for row in pl.range(bulk + stream, n_rows, RESORT_BLOCKS_PER_EXPERT):
                    row_out = output_base + row
                    row_in = input_base + row
                    slab_x[row_out : row_out + 1, :] = lane_x[row_in : row_in + 1, :]
                    pl.write(slab_scale, [row_out, 0], pl.read(lane_aux, [row_in, AUX_SCALE]))
                    pl.write(slab_weight, [row_out, 0], pl.read(lane_aux, [row_in, AUX_W]))
                    pl.write(
                        slab_route, [row_out, 0],
                        pl.cast(pl.read(lane_aux, [row_in, AUX_ROUTE]), pl.INT32),
                    )
                    pl.write(
                        slab_src, [row_out, 0],
                        pl.cast(pl.read(lane_aux, [row_in, AUX_SRC]), pl.INT32),
                    )
                source_prefix = source_prefix + n_rows
            # Zero the expert's alignment pad: the expert kernel computes these
            # rows through zero scales, and their cache lines belong to this
            # block alone (the expert slab is ROW_TILE-aligned).
            aligned_end = pl.cast(pl.read(expert_offsets, [local_e + 1]), pl.INDEX)
            for row in pl.range(expert_base + n_expert, aligned_end):
                pl.write(slab_scale, [row, 0], pl.cast(0.0, pl.FP32))
                pl.write(slab_weight, [row, 0], pl.cast(0.0, pl.FP32))
                pl.write(slab_route, [row, 0], pl.cast(0, pl.INT32))
                pl.write(slab_src, [row, 0], pl.cast(0, pl.INT32))

    return resort_tid


@pl.jit.inline
def prefill_moe_return(
    expert_y: pl.Tensor[[GROUPED_CAP, D], pl.FP32],
    slab_route: pl.Tensor[[GROUPED_CAP, 1], pl.INT32],
    slab_src: pl.Tensor[[GROUPED_CAP, 1], pl.INT32],
    expert_offsets: pl.Tensor[[N_LOCAL_EXPERTS + 1], pl.INT32],
    expert_counts: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed: pl.Out[pl.Tensor[[T, D], pl.FP32]],
    slot_out: pld.DistributedTensor[[ROUTES_PER_SRC, D], pl.FP32],
    return_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    dispatch_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Scatter every live expert row to its source's route slot, then sum TOPK."""
    with pl.spmd(
        N_LOCAL_EXPERTS * RETURN_STREAMS,
        name_hint="prefill_return_put",
        deps=[dispatch_tid],
        allow_early_resolve=False,
    ) as return_tid:
        return_block = pl.tile.get_block_idx()
        local_e = return_block // RETURN_STREAMS
        stream = return_block % RETURN_STREAMS
        base = pl.cast(pl.read(expert_offsets, [local_e]), pl.INDEX)
        n_rows = pl.cast(pl.read(expert_counts, [local_e, 0]), pl.INDEX)
        for row in pl.range(stream, n_rows, RETURN_STREAMS):
            src = pl.read(slab_src, [base + row, 0])
            slot = pl.cast(pl.read(slab_route, [base + row, 0]), pl.INDEX)
            pld.tensor.put(
                dst=slot_out, peer=src, src=expert_y,
                dst_offsets=[slot, 0], src_offsets=[base + row, 0],
                shape=[1, D],
            )

    # Return barrier, then the local TOPK sum. The per-destination route scans
    # the decode entry runs are gone: the payload carries its own route id, so
    # each expert row is scattered straight to ``token * TOPK + k``.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="prefill_return_signal", deps=[return_tid]
    ) as synced_tid:
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.notify(
                    target=return_ready,
                    peer=peer,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(
                    signal=return_ready, offsets=[src, 0], expected=epoch, cmp=pld.WaitCmp.Ge
                )

    active = pl.cast(num_tokens, pl.INDEX)
    if active < 0:
        active = pl.cast(0, pl.INDEX)
    if active > T:
        active = pl.cast(T, pl.INDEX)
    with pl.spmd(T // COMBINE_TILE, name_hint="prefill_combine", deps=[synced_tid]) as combine_tid:
        t0 = pl.tile.get_block_idx() * COMBINE_TILE
        for lane in pl.range(COMBINE_TILE):
            token = t0 + lane
            acc = pl.full([1, D], dtype=pl.FP32, value=0.0)
            if token < active:
                for k in pl.range(TOPK):
                    slot = token * TOPK + k
                    acc = pl.add(acc, slot_out[slot : slot + 1, :])
            routed[token : token + 1, :] = acc

    return combine_tid


@pl.jit(auto_scope=False)
def l2_prefill_moe(
    route_indices: pl.Tensor[[T, TOPK], pl.INT32],
    route_weights: pl.Tensor[[T, TOPK], pl.FP32],
    x_int8: pl.Tensor[[T, D], pl.INT8],
    x_scale: pl.Tensor[[T, 1], pl.FP32],
    w_gate_up: pl.Tensor[[N_LOCAL_EXPERTS, 2 * MOE_INTER, D], pl.INT8],
    w_gate_up_scale: pl.Tensor[[N_LOCAL_EXPERTS, 2 * MOE_INTER], pl.FP32],
    w_down: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    w_down_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    routed: pl.Out[pl.Tensor[[T, D], pl.FP32]],
    lane_x: pld.DistributedTensor[[TOTAL_CAP, D], pl.INT8],
    lane_aux: pld.DistributedTensor[[TOTAL_CAP, AUX_PAD], pl.FP32],
    lane_count: pld.DistributedTensor[[N_RANKS, COUNT_PAD], pl.INT32],
    count_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    x_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    slot_out: pld.DistributedTensor[[ROUTES_PER_SRC, D], pl.FP32],
    return_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """One rank's orchestration of the three stages."""
    send_counts = pl.create_tensor([N_RANKS, 1], dtype=pl.INT32)
    recv_expert_counts = pl.create_tensor([N_RANKS, N_LOCAL_EXPERTS], dtype=pl.INT32)
    expert_counts = pl.create_tensor([N_LOCAL_EXPERTS, 1], dtype=pl.INT32)
    expert_offsets = pl.create_tensor([N_LOCAL_EXPERTS + 1], dtype=pl.INT32)
    slab_x = pl.create_tensor([GROUPED_CAP, D], dtype=pl.INT8)
    slab_scale = pl.create_tensor([GROUPED_CAP, 1], dtype=pl.FP32)
    slab_weight = pl.create_tensor([GROUPED_CAP, 1], dtype=pl.FP32)
    slab_route = pl.create_tensor([GROUPED_CAP, 1], dtype=pl.INT32)
    slab_src = pl.create_tensor([GROUPED_CAP, 1], dtype=pl.INT32)
    expert_y = pl.create_tensor([GROUPED_CAP, D], dtype=pl.FP32)

    dispatch_tid = prefill_moe_dispatch(
        route_indices, route_weights, x_int8, x_scale,
        send_counts, recv_expert_counts, expert_counts, expert_offsets,
        lane_x, lane_aux, lane_count, count_ready, x_ready,
        slab_x, slab_scale, slab_weight, slab_route, slab_src,
        num_tokens, my_rank, epoch,
    )
    expert_routed(
        slab_x, slab_scale,
        w_gate_up, w_gate_up_scale,
        w_down, w_down_scale,
        expert_offsets,
        slab_weight,
        expert_y,
    )
    prefill_moe_return(
        expert_y, slab_route, slab_src, expert_offsets, expert_counts,
        routed, slot_out, return_ready,
        num_tokens, my_rank, epoch, dispatch_tid,
    )
    return routed


@pl.jit.host
def l3_prefill_moe(
    route_indices: pl.Tensor[[N_RANKS, T, TOPK], pl.INT32],
    route_weights: pl.Tensor[[N_RANKS, T, TOPK], pl.FP32],
    x_int8: pl.Tensor[[N_RANKS, T, D], pl.INT8],
    x_scale: pl.Tensor[[N_RANKS, T, 1], pl.FP32],
    w_gate_up: pl.Tensor[[N_RANKS, N_LOCAL_EXPERTS, 2 * MOE_INTER, D], pl.INT8],
    w_gate_up_scale: pl.Tensor[[N_RANKS, N_LOCAL_EXPERTS, 2 * MOE_INTER], pl.FP32],
    w_down: pl.Tensor[[N_RANKS, N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    w_down_scale: pl.Tensor[[N_RANKS, N_LOCAL_EXPERTS, D], pl.FP32],
    routed: pl.Out[pl.Tensor[[N_RANKS, T, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
    num_ranks: pl.Scalar[pl.INT32],
):
    """Launch one rank's orchestration per chip, sharing the window buffers."""
    lane_x_buf = pld.alloc_window_buffer([TOTAL_CAP, D], dtype=pl.INT8)
    lane_aux_buf = pld.alloc_window_buffer([TOTAL_CAP, AUX_PAD], dtype=pl.FP32)
    lane_count_buf = pld.alloc_window_buffer([N_RANKS, COUNT_PAD], dtype=pl.INT32)
    count_ready_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    x_ready_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    slot_out_buf = pld.alloc_window_buffer([ROUTES_PER_SRC, D], dtype=pl.FP32)
    return_ready_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)

    # Two calls over the same windows: the epoch-keyed barriers only retire
    # call N+1 after call N's readers, so one call would not cover the
    # multi-layer reuse this protocol exists for.
    for call in pl.range(2):
        for r in pl.range(pld.world_size()):
            lane_x = pld.window(lane_x_buf, [TOTAL_CAP, D], dtype=pl.INT8)
            lane_aux = pld.window(lane_aux_buf, [TOTAL_CAP, AUX_PAD], dtype=pl.FP32)
            lane_count = pld.window(lane_count_buf, [N_RANKS, COUNT_PAD], dtype=pl.INT32)
            count_ready = pld.window(count_ready_buf, [N_RANKS, 1], dtype=pl.INT32)
            x_ready = pld.window(x_ready_buf, [N_RANKS, 1], dtype=pl.INT32)
            slot_out = pld.window(slot_out_buf, [ROUTES_PER_SRC, D], dtype=pl.FP32)
            return_ready = pld.window(return_ready_buf, [N_RANKS, 1], dtype=pl.INT32)
            l2_prefill_moe(
                route_indices[r], route_weights[r], x_int8[r], x_scale[r],
                w_gate_up[r], w_gate_up_scale[r], w_down[r], w_down_scale[r],
                routed[r],
                lane_x, lane_aux, lane_count, count_ready, x_ready, slot_out, return_ready,
                num_tokens, r, pl.cast(call + 1, pl.INT32),
                device=r,
            )


def golden_prefill_moe(tensors):
    """Rank-aware torch reference for the routed branch: pack, exchange, experts,
    scatter back, sum.

    Emulates the wire by placing every payload into the destination rank's peer
    lane of the source that computed it, exactly as the kernels move it; the
    expert maths is deferred to ``expert_routed``'s golden so the reference is
    single-sourced.
    """
    from models.glm5_3_flash.expert_routed import golden_expert_routed_fn

    num_tokens = max(0, min(T, int(tensors.get("num_tokens", T))))
    x_next = torch.zeros(N_RANKS, T, D, dtype=torch.float32)
    slot_out = torch.zeros(N_RANKS, ROUTES_PER_SRC, D, dtype=torch.float32)

    for dst in range(N_RANKS):
        # Gather every route bound for this destination, expert-major with the
        # (source, token, k) order preserved inside each expert.
        rows = []
        for src in range(N_RANKS):
            for token in range(num_tokens):
                for k in range(TOPK):
                    expert = int(tensors["route_indices"][src, token, k])
                    if expert // N_LOCAL_EXPERTS == dst:
                        rows.append((expert - dst * N_LOCAL_EXPERTS, src, token, k))
        rows.sort(key=lambda entry: entry[0])
        total = len(rows)
        if total == 0:
            continue

        counts = [0] * N_LOCAL_EXPERTS
        for local_e, _src, _token, _k in rows:
            counts[local_e] += 1
        # Mirror the kernel's ROW_TILE-aligned expert slabs; the pad rows stay
        # zero and are never scattered back.
        aligned = [((count + ROW_TILE - 1) // ROW_TILE) * ROW_TILE for count in counts]
        offsets = [0] * (N_LOCAL_EXPERTS + 1)
        for local_e in range(N_LOCAL_EXPERTS):
            offsets[local_e + 1] = offsets[local_e] + aligned[local_e]

        slab_x = torch.zeros(GROUPED_CAP, D, dtype=torch.int8)
        slab_scale = torch.zeros(GROUPED_CAP, 1, dtype=torch.float32)
        slab_weight = torch.zeros(GROUPED_CAP, 1, dtype=torch.float32)
        cursor = offsets[:N_LOCAL_EXPERTS]
        placed = []
        for local_e, src, token, k in rows:
            slot = cursor[local_e]
            cursor[local_e] += 1
            slab_x[slot] = tensors["x_int8"][src, token]
            slab_scale[slot, 0] = tensors["x_scale"][src, token, 0]
            slab_weight[slot, 0] = tensors["route_weights"][src, token, k]
            placed.append((slot, src, token, k))

        work = {
            "x_int8": slab_x,
            "x_scale": slab_scale,
            "w_gate_up": tensors["w_gate_up"][dst],
            "w_gate_up_scale": tensors["w_gate_up_scale"][dst],
            "w_down": tensors["w_down"][dst],
            "w_down_scale": tensors["w_down_scale"][dst],
            "expert_offsets": torch.tensor(offsets, dtype=torch.int32),
            "route_weights": slab_weight,
            "output": torch.zeros(GROUPED_CAP, D, dtype=torch.float32),
        }
        golden_expert_routed_fn(work)
        expert_out = work["output"].float()
        for slot, src, token, k in placed:
            slot_out[src, token * TOPK + k] = expert_out[slot]

    for src in range(N_RANKS):
        for token in range(num_tokens):
            x_next[src, token] = slot_out[src, token * TOPK : (token + 1) * TOPK].sum(dim=0)

    tensors["routed"][:] = x_next


def build_tensor_specs(num_tokens=T):
    from golden import ScalarSpec, TensorSpec
    from models.glm5_3_flash.decode_moe import gen_routed_weight

    num_tokens = max(0, min(T, int(num_tokens)))

    def init_route_indices():
        # Global expert ids across the whole EP world; every local expert of the
        # EP2 layout is exercised, and both directions of the exchange carry.
        generator = torch.Generator().manual_seed(7)
        return torch.randint(0, N_RANKS * N_LOCAL_EXPERTS, (N_RANKS, T, TOPK), generator=generator).to(
            torch.int32
        )

    x_bf16 = torch.randn(N_RANKS, T, D, dtype=torch.bfloat16)

    def init_x_int8():
        from models.glm5_3_flash.quantization import quantize_per_token_int8

        x_i8, _ = quantize_per_token_int8(x_bf16.reshape(N_RANKS * T, D))
        return x_i8.reshape(N_RANKS, T, D)

    def init_x_scale():
        from models.glm5_3_flash.quantization import quantize_per_token_int8

        _, x_sd = quantize_per_token_int8(x_bf16.reshape(N_RANKS * T, D))
        return x_sd.reshape(N_RANKS, T, 1)

    ROUTED_DEQUANT_STD = {"w1": 2.47e-2, "w2": 2.44e-2}
    w_gu_i8, w_gu_s = gen_routed_weight(
        (N_RANKS, N_LOCAL_EXPERTS, 2 * MOE_INTER, D), ROUTED_DEQUANT_STD["w1"]
    )
    w2_i8, w2_s = gen_routed_weight(
        (N_RANKS, N_LOCAL_EXPERTS, D, MOE_INTER), ROUTED_DEQUANT_STD["w2"]
    )

    return [
        TensorSpec("route_indices", [N_RANKS, T, TOPK], torch.int32, init_value=init_route_indices),
        TensorSpec(
            "route_weights", [N_RANKS, T, TOPK], torch.float32,
            init_value=lambda: torch.rand(N_RANKS, T, TOPK),
        ),
        TensorSpec("x_int8", [N_RANKS, T, D], torch.int8, init_value=init_x_int8),
        TensorSpec("x_scale", [N_RANKS, T, 1], torch.float32, init_value=init_x_scale),
        TensorSpec(
            "w_gate_up", [N_RANKS, N_LOCAL_EXPERTS, 2 * MOE_INTER, D], torch.int8,
            init_value=lambda: w_gu_i8,
        ),
        TensorSpec(
            "w_gate_up_scale", [N_RANKS, N_LOCAL_EXPERTS, 2 * MOE_INTER], torch.float32,
            init_value=lambda: w_gu_s,
        ),
        TensorSpec(
            "w_down", [N_RANKS, N_LOCAL_EXPERTS, D, MOE_INTER], torch.int8, init_value=lambda: w2_i8
        ),
        TensorSpec("w_down_scale", [N_RANKS, N_LOCAL_EXPERTS, D], torch.float32, init_value=lambda: w2_s),
        TensorSpec("routed", [N_RANKS, T, D], torch.float32),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        ScalarSpec("num_ranks", torch.int32, N_RANKS),
    ]


if __name__ == "__main__":
    import argparse

    from golden import ratio_reldiff, run
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--tp", type=int, default=2, help="tensor-parallel size (config.py reads argv)")
    parser.add_argument("--ep", type=int, default=2, help="expert-parallel size / rank count")
    parser.add_argument(
        "-d", "--device", type=str, default=",".join(str(i) for i in range(N_RANKS)),
        help=f"comma-separated device ids (need {N_RANKS})",
    )
    parser.add_argument("--tokens", type=int, default=T, help="per-rank token capacity (module level)")
    parser.add_argument("--num-tokens", type=int, default=T)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) == N_RANKS, f"need exactly {N_RANKS} devices, got {device_ids}"

    result = run(
        fn=l3_prefill_moe,
        specs=build_tensor_specs(args.num_tokens),
        golden_fn=golden_prefill_moe,
        config=dict(
            dump_passes=args.dump_passes,
            platform=args.platform,
            distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            # Deterministic INT8 golden: 3e-3 per point, 1% of points.
            "routed": ratio_reldiff(diff_thd=3e-3, pct_thd=0.01),
        },
        compile_only=args.compile_only,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


__all__ = [
    "golden_prefill_moe",
    "l2_prefill_moe",
    "l3_prefill_moe",
    "prefill_moe_dispatch",
    "prefill_moe_return",
]
