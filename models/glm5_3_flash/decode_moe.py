# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""The sparse layer's routed half: EP pack -> arrive -> regroup -> experts -> return -> combine.

Each route's payload and scalar fields move over the L3 distributed stack, and a
transport-free **loopback** mode runs the very same kernels on one rank
(``--loopback``, ``num_ranks = 1``): every push, count, notify and return then
targets the rank itself, so the stage orderings can be brought up or bisected
without a peer. The stages:

* ``moe_pack`` pushes each route's payload row with ``pld.tensor.put`` and the
  four scalar fields (scale, weight, route id, local expert) as one 8-wide FP32
  ``remote_store`` into the destination rank's window lane ``my_rank * LANE_CAP + slot``.
* The per-source lane count goes to ``lane_count[my_rank, 0]`` and the payload
  barrier ``lane_ready[my_rank, 0]`` is AtomicAdd'ed, so the receiver knows both
  how many rows arrived and that the puts retired.
* ``moe_arrive`` waits on every peer's ``lane_ready``, then flattens its own
  window lanes source-major into the arrival arrays.
* ``moe_return`` puts every expert row to the slot its source rank reserved
  (``route = t * TOPK + k``) and bars with ``return_ready`` before the source
  reduces its slots.

Lane capacity is the dropless bound ``T * TOPK`` per (source, destination), the
same contract ``decode_moe.py`` uses. ``T`` is the decode capacity of the layer
this entry serves, not the active token count, so a run with fewer active tokens
still exercises the full cursor arithmetic.

Run (EP2, two ranks)::

    python models/glm5_3_flash/decode_moe.py -p a2a3sim -d 0,1 --tp 2 --ep 2 --num-tokens 16
    python models/glm5_3_flash/decode_moe.py -p a2a3    -d 0,1 --tp 2 --ep 2 --num-tokens 16

Loopback (one device, no transport)::

    python models/glm5_3_flash/decode_moe.py -p a2a3sim -d 0 --num-tokens 16 --loopback

``N_LOCAL_EXPERTS`` follows ``--ep`` (288 / 2 = 144 experts per rank in the test),
so an EP2 run needs about 3.6 GB of INT8 weights per rank. Verified at EP2 in
``a2a3sim`` and on two Ascend910 dies; the EP16 acceptance run needs 16 dies.

The barriers are epoch-keyed ``Set`` notifies: every rank is the single writer
of its own signal row, so ``lane_ready[src] >= epoch`` retires call N+1 only
after call N's readers, and one set of windows serves every layer with no
``clear``. The L3 entry runs two calls to exercise exactly that reuse.

Open items: ``T`` is the entry's decode capacity, so a 128-token production call
raises it (and the window sizes with it); and the per-destination route scans in
pack/return are serial, which is fine at decode scale but not at prefill.
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
# the first ``models.glm5_3_flash`` import when the command line pins neither
# axis. The EP test additionally asserts that the device count matches EP_SIZE.
if not any(tok in ("--tp", "--ep") or tok.startswith(("--tp=", "--ep=")) for tok in sys.argv):
    sys.argv += ["--tp", "2", "--ep", "2"]
from models.glm5_3_flash.config import D, EP_SIZE, FLASH, N_LOCAL_EXPERTS, TOPK
from models.glm5_3_flash.expert_routed import expert_routed
from models.glm5_3_flash.quantization import INT8_AMAX_EPS, INT8_SCALE_MAX

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


N_RANKS = EP_SIZE
MOE_INTER = FLASH.moe_intermediate_size
T = 16                      # decode capacity exercised by this entry (token rows per rank)
TOPK_ACTIVE = TOPK
N_ROUTES = T * TOPK
LANE_CAP = N_ROUTES          # dropless per (source, destination) lane bound
LANE_ROWS = N_RANKS * LANE_CAP
AUX_SCALE, AUX_W, AUX_ROUTE, AUX_EXPERT = 0, 1, 2, 3
AUX_PAD = 16     # one 64-byte line per lane row: 32-byte rows would share lines
COUNT_PAD = 16   # one 64-byte line per source row for the same reason

assert N_RANKS >= 1
assert N_LOCAL_EXPERTS >= 1


@pl.jit.inline
def moe_pack(
    route_indices: pl.Tensor[[T, TOPK], pl.INT32],
    route_weights: pl.Tensor[[T, TOPK], pl.FP32],
    x_int8: pl.Tensor[[T, D], pl.INT8],
    x_scale: pl.Tensor[[T, 1], pl.FP32],
    lane_x: pld.DistributedTensor[[LANE_ROWS, D], pl.INT8],
    lane_aux: pld.DistributedTensor[[LANE_ROWS, AUX_PAD], pl.FP32],
    lane_count: pld.DistributedTensor[[N_RANKS, COUNT_PAD], pl.INT32],
    lane_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    num_ranks: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Push this rank's routes into the destination lanes.

    One block per destination scans the routes in ``(t, k)`` order and fills its
    lane; a final single-instance block publishes the per-destination count and
    the payload barrier after the push block retires.
    """
    active = pl.cast(num_tokens, pl.INDEX)
    if active < 0:
        active = pl.cast(0, pl.INDEX)
    if active > T:
        active = pl.cast(T, pl.INDEX)
    dst_cols = pl.create_tensor([N_RANKS], dtype=pl.INT32)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_ep_pack_count") as count_tid:
        for d in pl.range(N_RANKS):
            pl.write(dst_cols, [d], pl.cast(0, pl.INT32))
        for t in pl.range(active):
            for k in pl.range(TOPK):
                e = pl.read(route_indices, [t, k])
                dst = pl.cast(e // N_LOCAL_EXPERTS, pl.INDEX)
                pl.write(dst_cols, [dst], pl.cast(pl.read(dst_cols, [dst]) + 1, pl.INT32))

    with pl.spmd(num_ranks, name_hint="moe_ep_pack_push", deps=[count_tid]) as push_tid:
        d = pl.tile.get_block_idx()
        lane_slot = pl.cast(0, pl.INDEX)
        for t in pl.range(active):
            for k in pl.range(TOPK):
                e = pl.read(route_indices, [t, k])
                dst = pl.cast(e // N_LOCAL_EXPERTS, pl.INDEX)
                if dst == d:
                    # The row lands in the *destination's* window, so it is
                    # addressed by the source lane (my_rank), not by d.
                    lane_row = pl.cast(my_rank, pl.INDEX) * LANE_CAP + lane_slot
                    pld.tensor.put(
                        dst=lane_x, peer=d, src=x_int8,
                        dst_offsets=[lane_row, 0], src_offsets=[t, 0], shape=[1, D],
                    )
                    aux = pl.tile.full([1, AUX_PAD], dtype=pl.FP32, value=0.0)
                    pl.tile.write(aux, [0, AUX_SCALE], pl.read(x_scale, [t, 0]))
                    pl.tile.write(aux, [0, AUX_W], pl.read(route_weights, [t, k]))
                    push_route_i32 = pl.cast(t * TOPK + k, pl.INT32)
                    pl.tile.write(aux, [0, AUX_ROUTE], pl.cast(push_route_i32, pl.FP32))
                    d_i32 = pl.cast(d, pl.INT32)
                    push_expert_i32 = pl.cast(e - d_i32 * N_LOCAL_EXPERTS, pl.INT32)
                    pl.tile.write(aux, [0, AUX_EXPERT], pl.cast(push_expert_i32, pl.FP32))
                    pld.tile.remote_store(aux, target=lane_aux, peer=d, offsets=[lane_row, 0])
                    lane_slot = lane_slot + 1

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_ep_pack_notify", deps=[push_tid]) as _notify_tid:
        # Count carries how many rows landed in each destination's lane; the
        # ready flag is the payload barrier the receiver waits on. Both follow
        # the puts in program order.
        for d in pl.range(num_ranks):
            count_tile = pl.tile.full([1, COUNT_PAD], dtype=pl.INT32, value=0)
            pl.tile.write(count_tile, [0, 0], pl.read(dst_cols, [d]))
            pld.tile.remote_store(count_tile, target=lane_count, peer=d, offsets=[my_rank, 0])
            if d != my_rank:
                # Set, not AtomicAdd: every rank is the single writer of its own
                # row, so the value is the epoch and no clear is needed between
                # calls.
                pld.system.notify(
                    target=lane_ready, peer=d, offsets=[my_rank, 0],
                    value=epoch, op=pld.NotifyOp.Set,
                )

    # The notify block must retire before any consumer reads this rank's own
    # lane: the arrival wait only covers peers, so the local count/ready writes
    # ride this TaskId rather than the push's.
    return _notify_tid


@pl.jit.inline
def moe_arrive(
    lane_x: pld.DistributedTensor[[LANE_ROWS, D], pl.INT8],
    lane_aux: pld.DistributedTensor[[LANE_ROWS, AUX_PAD], pl.FP32],
    lane_count: pld.DistributedTensor[[N_RANKS, COUNT_PAD], pl.INT32],
    lane_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    arrive_x: pl.Tensor[[LANE_ROWS, D], pl.INT8],
    arrive_scale: pl.Tensor[[LANE_ROWS, 1], pl.FP32],
    arrive_weight: pl.Tensor[[LANE_ROWS, 1], pl.FP32],
    arrive_route: pl.Tensor[[LANE_ROWS, 1], pl.INT32],
    arrive_expert: pl.Tensor[[LANE_ROWS, 1], pl.INT32],
    arrive_src: pl.Tensor[[LANE_ROWS, 1], pl.INT32],
    arrive_total: pl.Tensor[[1], pl.INT32],
    num_ranks: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    pack_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Wait for the peer payloads, then flatten this rank's lanes source-major."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_ep_arrive_wait", deps=[pack_tid]) as wait_tid:
        for src in pl.range(num_ranks):
            if src != my_rank:
                pld.system.wait(
                    signal=lane_ready, offsets=[src, 0],
                    expected=epoch, cmp=pld.WaitCmp.Ge,
                )

    src_base = pl.create_tensor([N_RANKS], dtype=pl.INT32)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_ep_arrive_prefix", deps=[wait_tid]) as prefix_tid:
        running = pl.cast(0, pl.INDEX)
        for s in pl.range(num_ranks):
            pl.write(src_base, [s], pl.cast(running, pl.INT32))
            n_rows = pl.cast(pl.read(lane_count, [s, 0]), pl.INDEX)
            for i in pl.range(n_rows):
                flat = running + i
                lane_row = s * LANE_CAP + i
                pl.write(arrive_scale, [flat, 0], pl.read(lane_aux, [lane_row, AUX_SCALE]))
                pl.write(arrive_weight, [flat, 0], pl.read(lane_aux, [lane_row, AUX_W]))
                pl.write(
                    arrive_route, [flat, 0],
                    pl.cast(pl.read(lane_aux, [lane_row, AUX_ROUTE]), pl.INT32),
                )
                pl.write(
                    arrive_expert, [flat, 0],
                    pl.cast(pl.read(lane_aux, [lane_row, AUX_EXPERT]), pl.INT32),
                )
                pl.write(arrive_src, [flat, 0], pl.cast(s, pl.INT32))
            running = running + n_rows
        pl.write(arrive_total, [0], pl.cast(running, pl.INT32))

    with pl.spmd(num_ranks, name_hint="moe_ep_arrive_copy", deps=[prefix_tid]) as _arrive_tid:
        s = pl.tile.get_block_idx()
        base = pl.cast(pl.read(src_base, [s]), pl.INDEX)
        n_rows = pl.cast(pl.read(lane_count, [s, 0]), pl.INDEX)
        for i in pl.range(n_rows):
            flat = base + i
            lane_row = s * LANE_CAP + i
            arrive_x[flat : flat + 1, :] = lane_x[lane_row : lane_row + 1, :]

    return prefix_tid


@pl.jit.inline
def moe_dispatch(
    route_expert: pl.Tensor[[LANE_ROWS, 1], pl.INT32],
    route_src: pl.Tensor[[LANE_ROWS, 1], pl.INT32],
    route_id: pl.Tensor[[LANE_ROWS, 1], pl.INT32],
    route_x: pl.Tensor[[LANE_ROWS, D], pl.INT8],
    route_scale: pl.Tensor[[LANE_ROWS, 1], pl.FP32],
    route_weight: pl.Tensor[[LANE_ROWS, 1], pl.FP32],
    route_count: pl.Tensor[[1], pl.INT32],
    recv_x: pl.Tensor[[LANE_ROWS, D], pl.INT8],
    recv_scale: pl.Tensor[[LANE_ROWS, 1], pl.FP32],
    recv_weight: pl.Tensor[[LANE_ROWS, 1], pl.FP32],
    recv_src: pl.Tensor[[LANE_ROWS, 1], pl.INT32],
    recv_route: pl.Tensor[[LANE_ROWS, 1], pl.INT32],
    expert_offsets: pl.Tensor[[N_LOCAL_EXPERTS + 1], pl.INT32],
):
    """Stable expert-major regroup of this rank's arrived routes.

    Capacity tails are never read: every loop here and in ``expert_routed`` is
    bounded by an exact count (``route_count`` here, the expert offsets there),
    so rows a call did not write — which may decode as NaN/Inf — are never
    consumed.
    """
    n_routes = pl.cast(pl.read(route_count, [0]), pl.INDEX)
    if n_routes < 0:
        n_routes = pl.cast(0, pl.INDEX)
    if n_routes > LANE_ROWS:
        n_routes = pl.cast(LANE_ROWS, pl.INDEX)
    expert_cols = pl.create_tensor([N_LOCAL_EXPERTS], dtype=pl.INT32)
    cursor = pl.create_tensor([N_LOCAL_EXPERTS], dtype=pl.INT32)
    bucket = pl.create_tensor([LANE_ROWS], dtype=pl.INT32)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_ep_dispatch_count") as count_tid:
        for e in pl.range(N_LOCAL_EXPERTS):
            pl.write(expert_cols, [e], pl.cast(0, pl.INT32))
        for r in pl.range(n_routes):
            e = pl.cast(pl.read(route_expert, [r, 0]), pl.INDEX)
            pl.write(expert_cols, [e], pl.cast(pl.read(expert_cols, [e]) + 1, pl.INT32))
        running = pl.cast(0, pl.INT32)
        for e in pl.range(N_LOCAL_EXPERTS):
            pl.write(expert_offsets, [e], running)
            pl.write(cursor, [e], running)
            running = running + pl.read(expert_cols, [e])
        pl.write(expert_offsets, [N_LOCAL_EXPERTS], running)
        # One instance owns every scalar aux write: the 4-byte [row, 0] entries
        # share 64-byte lines, and concurrent blocks would drop each other's
        # stores there. The payload rows below are whole 4 KB rows and stay
        # expert-parallel.
        for r in pl.range(n_routes):
            e = pl.cast(pl.read(route_expert, [r, 0]), pl.INDEX)
            slot = pl.read(cursor, [e])
            pl.write(cursor, [e], pl.cast(slot + 1, pl.INT32))
            flat_slot = pl.cast(slot, pl.INDEX)
            pl.write(bucket, [flat_slot], pl.cast(r, pl.INT32))
            pl.write(recv_scale, [flat_slot, 0], pl.read(route_scale, [r, 0]))
            pl.write(recv_weight, [flat_slot, 0], pl.read(route_weight, [r, 0]))
            pl.write(recv_src, [flat_slot, 0], pl.read(route_src, [r, 0]))
            pl.write(recv_route, [flat_slot, 0], pl.read(route_id, [r, 0]))

    with pl.spmd(N_LOCAL_EXPERTS, name_hint="moe_ep_dispatch_gather", deps=[count_tid]) as _gather_tid:
        e = pl.tile.get_block_idx()
        base = pl.cast(pl.read(expert_offsets, [e]), pl.INDEX)
        n_rows = pl.cast(pl.read(expert_cols, [e]), pl.INDEX)
        for i in pl.range(n_rows):
            flat_slot = base + i
            r = pl.cast(pl.read(bucket, [flat_slot]), pl.INDEX)
            recv_x[flat_slot : flat_slot + 1, :] = route_x[r : r + 1, :]

    return recv_x


@pl.jit.inline(auto_scope=False)
def moe_return(
    expert_out: pl.Tensor[[LANE_ROWS, D], pl.FP32],
    recv_src: pl.Tensor[[LANE_ROWS, 1], pl.INT32],
    recv_route: pl.Tensor[[LANE_ROWS, 1], pl.INT32],
    slot_out: pld.DistributedTensor[[N_ROUTES, D], pl.FP32],
    route_count: pl.Tensor[[1], pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Put every expert row back to its owner's route slot."""
    n_rows = pl.cast(pl.read(route_count, [0]), pl.INDEX)
    if n_rows < 0:
        n_rows = pl.cast(0, pl.INDEX)
    if n_rows > LANE_ROWS:
        n_rows = pl.cast(LANE_ROWS, pl.INDEX)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_ep_return", allow_early_resolve=True) as return_tid:
        for r in pl.range(n_rows):
            src = pl.read(recv_src, [r, 0])
            slot = pl.cast(pl.read(recv_route, [r, 0]), pl.INDEX)
            pld.tensor.put(
                dst=slot_out, peer=src, src=expert_out,
                dst_offsets=[slot, 0], src_offsets=[r, 0], shape=[1, D],
            )

    return return_tid


@pl.jit.inline
def moe_return_notify(
    return_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_ranks: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    return_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Bar every peer after the return puts retire (single writer per row, Set)."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_ep_return_notify", deps=[return_tid]) as sync_tid:
        for peer in pl.range(num_ranks):
            if peer != my_rank:
                pld.system.notify(
                    target=return_ready, peer=peer, offsets=[my_rank, 0],
                    value=epoch, op=pld.NotifyOp.Set,
                )
    return sync_tid


@pl.jit.inline
def moe_combine(
    slot_out: pld.DistributedTensor[[N_ROUTES, D], pl.FP32],
    return_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    y: pl.Out[pl.Tensor[[T, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
    num_ranks: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    sync_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Tensor[[T, D], pl.FP32]:
    """Wait for every peer's returns, then sum the TOPK slots of each token."""
    active = pl.cast(num_tokens, pl.INDEX)
    if active < 0:
        active = pl.cast(0, pl.INDEX)
    if active > T:
        active = pl.cast(T, pl.INDEX)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_ep_combine_wait", deps=[sync_tid]) as wait_tid:
        for src in pl.range(num_ranks):
            if src != my_rank:
                pld.system.wait(
                    signal=return_ready, offsets=[src, 0],
                    expected=epoch, cmp=pld.WaitCmp.Ge,
                )

    # One block per capacity row: rows past ``active`` are written as zeros so a
    # run with fewer active tokens still leaves a defined output.
    with pl.spmd(T, name_hint="moe_ep_combine", deps=[wait_tid]) as _combine_tid:
        t = pl.tile.get_block_idx()
        acc = pl.full([1, D], dtype=pl.FP32, value=0.0)
        if t < active:
            for k in pl.range(TOPK):
                slot = t * TOPK + k
                acc = pl.add(acc, slot_out[slot : slot + 1, :])
        y[t : t + 1, :] = acc

    return y


@pl.jit(auto_scope=False)
def l2_decode_moe(
    route_indices: pl.Tensor[[T, TOPK], pl.INT32],
    route_weights: pl.Tensor[[T, TOPK], pl.FP32],
    x_int8: pl.Tensor[[T, D], pl.INT8],
    x_scale: pl.Tensor[[T, 1], pl.FP32],
    w_gate_up: pl.Tensor[[N_LOCAL_EXPERTS, 2 * MOE_INTER, D], pl.INT8],
    w_gate_up_scale: pl.Tensor[[N_LOCAL_EXPERTS, 2 * MOE_INTER], pl.FP32],
    w_down: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    w_down_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    y: pl.Out[pl.Tensor[[T, D], pl.FP32]],
    lane_x: pld.DistributedTensor[[LANE_ROWS, D], pl.INT8],
    lane_aux: pld.DistributedTensor[[LANE_ROWS, AUX_PAD], pl.FP32],
    lane_count: pld.DistributedTensor[[N_RANKS, COUNT_PAD], pl.INT32],
    lane_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    slot_out: pld.DistributedTensor[[N_ROUTES, D], pl.FP32],
    return_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    num_ranks: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """One rank's orchestration of the six stages."""
    recv_x = pl.create_tensor([LANE_ROWS, D], dtype=pl.INT8)
    recv_scale = pl.create_tensor([LANE_ROWS, 1], dtype=pl.FP32)
    recv_weight = pl.create_tensor([LANE_ROWS, 1], dtype=pl.FP32)
    recv_src = pl.create_tensor([LANE_ROWS, 1], dtype=pl.INT32)
    recv_route = pl.create_tensor([LANE_ROWS, 1], dtype=pl.INT32)
    expert_offsets = pl.create_tensor([N_LOCAL_EXPERTS + 1], dtype=pl.INT32)
    expert_out = pl.create_tensor([LANE_ROWS, D], dtype=pl.FP32)
    arrive_x = pl.create_tensor([LANE_ROWS, D], dtype=pl.INT8)
    arrive_scale = pl.create_tensor([LANE_ROWS, 1], dtype=pl.FP32)
    arrive_weight = pl.create_tensor([LANE_ROWS, 1], dtype=pl.FP32)
    arrive_route = pl.create_tensor([LANE_ROWS, 1], dtype=pl.INT32)
    arrive_expert = pl.create_tensor([LANE_ROWS, 1], dtype=pl.INT32)
    arrive_src = pl.create_tensor([LANE_ROWS, 1], dtype=pl.INT32)
    arrive_total = pl.create_tensor([1], dtype=pl.INT32)

    pack_tid = moe_pack(
        route_indices, route_weights, x_int8, x_scale,
        lane_x, lane_aux, lane_count, lane_ready,
        num_tokens, num_ranks, my_rank, epoch,
    )
    moe_arrive(
        lane_x, lane_aux, lane_count, lane_ready,
        arrive_x, arrive_scale, arrive_weight, arrive_route, arrive_expert, arrive_src,
        arrive_total, num_ranks, my_rank, epoch, pack_tid,
    )
    moe_dispatch(
        arrive_expert, arrive_src, arrive_route, arrive_x, arrive_scale, arrive_weight,
        arrive_total,
        recv_x, recv_scale, recv_weight, recv_src, recv_route, expert_offsets,
    )
    expert_routed(
        recv_x, recv_scale,
        w_gate_up, w_gate_up_scale,
        w_down, w_down_scale,
        expert_offsets,
        recv_weight,
        expert_out,
    )
    return_tid = moe_return(expert_out, recv_src, recv_route, slot_out, arrive_total)
    sync_tid = moe_return_notify(return_ready, num_ranks, my_rank, epoch, return_tid)
    moe_combine(slot_out, return_ready, y, num_tokens, num_ranks, my_rank, epoch, sync_tid)
    return y


@pl.jit.host
def l3_decode_moe(
    route_indices: pl.Tensor[[N_RANKS, T, TOPK], pl.INT32],
    route_weights: pl.Tensor[[N_RANKS, T, TOPK], pl.FP32],
    x_int8: pl.Tensor[[N_RANKS, T, D], pl.INT8],
    x_scale: pl.Tensor[[N_RANKS, T, 1], pl.FP32],
    w_gate_up: pl.Tensor[[N_RANKS, N_LOCAL_EXPERTS, 2 * MOE_INTER, D], pl.INT8],
    w_gate_up_scale: pl.Tensor[[N_RANKS, N_LOCAL_EXPERTS, 2 * MOE_INTER], pl.FP32],
    w_down: pl.Tensor[[N_RANKS, N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    w_down_scale: pl.Tensor[[N_RANKS, N_LOCAL_EXPERTS, D], pl.FP32],
    y: pl.Out[pl.Tensor[[N_RANKS, T, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
    num_ranks: pl.Scalar[pl.INT32],
):
    """Launch one rank's orchestration per chip, sharing the window buffers."""
    lane_x_buf = pld.alloc_window_buffer([LANE_ROWS, D], dtype=pl.INT8)
    lane_aux_buf = pld.alloc_window_buffer([LANE_ROWS, AUX_PAD], dtype=pl.FP32)
    lane_count_buf = pld.alloc_window_buffer([N_RANKS, COUNT_PAD], dtype=pl.INT32)
    lane_ready_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    slot_out_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.FP32)
    return_ready_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)

    # Two calls over the same windows: the epoch-keyed barriers only retire
    # call N+1 after call N's readers, so one call would not cover the
    # multi-layer reuse this protocol exists for.
    for call in pl.range(2):
        for r in pl.range(pld.world_size()):
            lane_x = pld.window(lane_x_buf, [LANE_ROWS, D], dtype=pl.INT8)
            lane_aux = pld.window(lane_aux_buf, [LANE_ROWS, AUX_PAD], dtype=pl.FP32)
            lane_count = pld.window(lane_count_buf, [N_RANKS, COUNT_PAD], dtype=pl.INT32)
            lane_ready = pld.window(lane_ready_buf, [N_RANKS, 1], dtype=pl.INT32)
            slot_out = pld.window(slot_out_buf, [N_ROUTES, D], dtype=pl.FP32)
            return_ready = pld.window(return_ready_buf, [N_RANKS, 1], dtype=pl.INT32)
            l2_decode_moe(
                route_indices[r], route_weights[r], x_int8[r], x_scale[r],
                w_gate_up[r], w_gate_up_scale[r], w_down[r], w_down_scale[r],
                y[r],
                lane_x, lane_aux, lane_count, lane_ready, slot_out, return_ready,
                num_tokens, num_ranks, r, pl.cast(call + 1, pl.INT32),
                device=r,
            )


def golden_routed_ep(tensors):
    """Rank-aware torch reference for the routed branch: pack, arrive, regroup,
    experts, return, combine.

    The transport is emulated by placing every payload into the destination
    rank's lane of the source that computed it, which is exactly what the two
    kernels move over the wire. The expert maths is deferred to
    ``expert_routed``'s golden so the reference stays single-sourced.

    Returns the per-rank routed output ``[N_RANKS, T, D]``; the EP entry writes
    it to ``y``, while the layer composition adds the shared-expert result.
    """
    from models.glm5_3_flash.expert_routed import golden_expert_routed_fn

    num_tokens = max(0, min(T, int(tensors.get("num_tokens", T))))
    x_next = torch.zeros(N_RANKS, T, D, dtype=torch.float32)

    # Stage 1: pack every source rank's routes into destination lanes.
    lane_x = torch.zeros(N_RANKS, LANE_ROWS, D, dtype=torch.int8)
    lane_aux = torch.zeros(N_RANKS, LANE_ROWS, AUX_PAD, dtype=torch.float32)
    lane_count = torch.zeros(N_RANKS, N_RANKS, dtype=torch.int64)
    for src in range(N_RANKS):
        for t in range(num_tokens):
            for k in range(TOPK):
                e = int(tensors["route_indices"][src, t, k])
                dst = e // N_LOCAL_EXPERTS
                slot = int(lane_count[src, dst])
                lane_count[src, dst] += 1
                row = src * LANE_CAP + slot
                lane_x[dst, row] = tensors["x_int8"][src, t]
                lane_aux[dst, row, AUX_SCALE] = tensors["x_scale"][src, t, 0]
                lane_aux[dst, row, AUX_W] = tensors["route_weights"][src, t, k]
                lane_aux[dst, row, AUX_ROUTE] = t * TOPK + k
                lane_aux[dst, row, AUX_EXPERT] = e - dst * N_LOCAL_EXPERTS

    # Stage 2-5: per destination, flatten source-major, regroup by expert,
    # evaluate the local experts, and return each row to its owner's slot.
    slot_out = torch.zeros(N_RANKS, N_ROUTES, D, dtype=torch.float32)
    for dst in range(N_RANKS):
        flat_expert = []
        flat_route = []
        flat_src = []
        recv_x = []
        recv_scale = []
        recv_weight = []
        expert_offsets = [0]
        for src in range(N_RANKS):
            for i in range(int(lane_count[src, dst])):
                row = src * LANE_CAP + i
                flat_expert.append(int(lane_aux[dst, row, AUX_EXPERT]))
                flat_route.append(int(lane_aux[dst, row, AUX_ROUTE]))
                flat_src.append(src)
                recv_x.append(lane_x[dst, row])
                recv_scale.append(lane_aux[dst, row, AUX_SCALE])
                recv_weight.append(lane_aux[dst, row, AUX_W])
        total = len(flat_expert)
        counts = [0] * N_LOCAL_EXPERTS
        for e in flat_expert:
            counts[e] += 1
        for e in range(N_LOCAL_EXPERTS):
            expert_offsets.append(expert_offsets[-1] + counts[e])
        order = sorted(range(total), key=lambda r: flat_expert[r])
        x_flat = torch.zeros(LANE_ROWS, D, dtype=torch.int8)
        s_flat = torch.zeros(LANE_ROWS, 1, dtype=torch.float32)
        w_flat = torch.zeros(LANE_ROWS, 1, dtype=torch.float32)
        for slot, r in enumerate(order):
            x_flat[slot] = recv_x[r]
            s_flat[slot] = recv_scale[r]
            w_flat[slot] = recv_weight[r]
        work = {
            "x_int8": x_flat,
            "x_scale": s_flat,
            "w_gate_up": tensors["w_gate_up"][dst],
            "w_gate_up_scale": tensors["w_gate_up_scale"][dst],
            "w_down": tensors["w_down"][dst],
            "w_down_scale": tensors["w_down_scale"][dst],
            "expert_offsets": torch.tensor(expert_offsets, dtype=torch.int32),
            "route_weights": w_flat,
            "output": torch.zeros(LANE_ROWS, D, dtype=torch.float32),
        }
        golden_expert_routed_fn(work)
        expert_out = work["output"].float()
        for slot, r in enumerate(order):
            slot_out[flat_src[r], flat_route[r]] = expert_out[slot]

    # Stage 6: token sum of the TOPK slots.
    for r in range(N_RANKS):
        for t in range(num_tokens):
            x_next[r, t] = slot_out[r, t * TOPK : (t + 1) * TOPK].sum(dim=0)

    return x_next


def golden_decode_moe(tensors):
    """Routed-branch expected output for the EP entry."""
    tensors["y"][:] = golden_routed_ep(tensors)


def gen_routed_weight(shape, dequant_std, chunk_experts=16):
    """Per-channel INT8 routed weights, generated rank/expert chunk by chunk.

    Same distribution and rescale contract as ``expert_routed.gen_routed_weight``
    (per-output-channel amax scale, rescaled to ``dequant_std``), but the FP32
    transient stays bounded instead of materialising the whole tensor family.
    Shared by this entry and ``decode_layer``'s fixture.
    """
    n_lead = shape[0] * shape[1]
    out_features, in_features = shape[-2], shape[-1]
    w_i8 = torch.empty(shape, dtype=torch.int8)
    scale = torch.empty(*shape[:-1], dtype=torch.float32)
    w_flat = w_i8.reshape(n_lead, out_features, in_features)
    s_flat = scale.reshape(n_lead, out_features)
    for i0 in range(0, n_lead, chunk_experts):
        i1 = min(i0 + chunk_experts, n_lead)
        w = torch.randn(i1 - i0, out_features, in_features)
        amax = w.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
        s = amax / INT8_SCALE_MAX
        w_flat[i0:i1] = torch.round(w / s).clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
        del w
        s_ch = s.squeeze(-1)
        w_deq = w_flat[i0:i1].float() * s_ch.unsqueeze(-1)
        s_flat[i0:i1] = s_ch * (dequant_std / w_deq.std())
        del w_deq
    return w_i8, scale


def build_tensor_specs(num_tokens=T, *, loopback=False, num_ranks=EP_SIZE):
    from golden import ScalarSpec, TensorSpec

    num_tokens = max(0, min(T, int(num_tokens)))

    def init_route_indices():
        # Global expert ids spread across both ranks so both lanes move data.
        # Loopback has no peers, so every route has to land on this rank.
        gen = torch.Generator().manual_seed(7)
        bound = N_LOCAL_EXPERTS if loopback else N_RANKS * N_LOCAL_EXPERTS
        return torch.randint(0, bound, (N_RANKS, T, TOPK), generator=gen).to(torch.int32)

    ROUTED_DEQUANT_STD = {"w1": 2.47e-2, "w2": 2.44e-2}
    w_gu_i8, w_gu_s = gen_routed_weight(
        (N_RANKS, N_LOCAL_EXPERTS, 2 * MOE_INTER, D), ROUTED_DEQUANT_STD["w1"]
    )
    w2_i8, w2_s = gen_routed_weight(
        (N_RANKS, N_LOCAL_EXPERTS, D, MOE_INTER), ROUTED_DEQUANT_STD["w2"]
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

    return [
        TensorSpec("route_indices", [N_RANKS, T, TOPK], torch.int32, init_value=init_route_indices),
        TensorSpec("route_weights", [N_RANKS, T, TOPK], torch.float32,
                   init_value=lambda: torch.rand(N_RANKS, T, TOPK)),
        TensorSpec("x_int8", [N_RANKS, T, D], torch.int8, init_value=init_x_int8),
        TensorSpec("x_scale", [N_RANKS, T, 1], torch.float32, init_value=init_x_scale),
        TensorSpec("w_gate_up", [N_RANKS, N_LOCAL_EXPERTS, 2 * MOE_INTER, D], torch.int8,
                   init_value=lambda: w_gu_i8),
        TensorSpec("w_gate_up_scale", [N_RANKS, N_LOCAL_EXPERTS, 2 * MOE_INTER], torch.float32,
                   init_value=lambda: w_gu_s),
        TensorSpec("w_down", [N_RANKS, N_LOCAL_EXPERTS, D, MOE_INTER], torch.int8,
                   init_value=lambda: w2_i8),
        TensorSpec("w_down_scale", [N_RANKS, N_LOCAL_EXPERTS, D], torch.float32,
                   init_value=lambda: w2_s),
        TensorSpec("y", [N_RANKS, T, D], torch.float32),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        ScalarSpec("num_ranks", torch.int32, num_ranks),
    ]


if __name__ == "__main__":
    import argparse

    from golden import ratio_reldiff, run
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--tp", type=int, default=2, help="tensor-parallel size (config.py reads argv)")
    parser.add_argument("--ep", type=int, default=2, help="expert-parallel size / rank count")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(N_RANKS)),
                        help=f"comma-separated device ids (need {N_RANKS})")
    parser.add_argument("--num-tokens", type=int, default=T)
    parser.add_argument("--loopback", action="store_true", default=False,
                        help="transport-free single-rank path (needs one device, num_ranks=1)")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    if args.loopback:
        assert len(device_ids) == 1, f"loopback needs exactly one device, got {device_ids}"
        num_ranks = 1
    else:
        assert len(device_ids) == N_RANKS, f"need exactly {N_RANKS} devices, got {device_ids}"
        num_ranks = len(device_ids)

    result = run(
        fn=l3_decode_moe,
        specs=build_tensor_specs(args.num_tokens, loopback=args.loopback, num_ranks=num_ranks),
        golden_fn=golden_decode_moe,
        config=dict(
            dump_passes=args.dump_passes,
            platform=args.platform,
            distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            # Deterministic INT8 golden, so align with the sibling EP MoE bar
            # (3e-3 per point) but allow only 1% of points: the measured margin
            # is orders of magnitude inside it, and a 1% budget still catches a
            # single garbled expert row (4096 / 65536 = 6% of the output).
            "y": ratio_reldiff(
                diff_thd=3e-3, pct_thd=0.01,
                valid_rows=1 if args.loopback else None,
            ),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


__all__ = [
    "l2_decode_moe",
    "l3_decode_moe",
    "moe_arrive",
    "moe_combine",
    "moe_dispatch",
    "moe_pack",
    "moe_return",
]
