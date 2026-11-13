# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""FP32 projection broadcast with two prepared, ready, and consumed slots shared across projection families."""

import pypto.language as pl
import pypto.language.distributed as pld

# Communication bounds.
PROJECTION_RANKS = 8
PROJECTION_KINDS = 6
PROJECTION_SLOTS = 2
PROJECTION_TILE = 512
PROJECTION_WINDOW_ROWS = PROJECTION_SLOTS * PROJECTION_TILE
PROJECTION_WINDOW_COLS = 2048
PROJECTION_SIGNAL_ROWS = PROJECTION_SLOTS

# Tiling.
PROJECTION_ROW_TILE = 16
PROJECTION_COL_TILE = 64


@pl.jit.inline(auto_scope=False)
def projection_prepare(
    prepared: pld.DistributedTensor[[PROJECTION_SIGNAL_ROWS, PROJECTION_RANKS], pl.INT32],
    consumed: pld.DistributedTensor[[PROJECTION_SIGNAL_ROWS, PROJECTION_RANKS], pl.INT32],
    peers: pl.Tensor[[PROJECTION_RANKS], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    previous_local_consumer: pl.Scalar[pl.TASK_ID],
    kind: pl.constexpr,
    replicas: pl.constexpr,
):
    """Prepare the destination slot after its previous distributed consumers retire."""
    lane = (epoch - 1) % PROJECTION_SLOTS
    generation = pl.cast((epoch - 1) // PROJECTION_SLOTS + 1, pl.INT32)
    owner = pl.read(peers, [(epoch - 1) % (PROJECTION_RANKS // replicas)])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="projection_slot_defer_wait", deps=[previous_local_consumer], no_dep_args=[consumed], allow_early_resolve=False) as reusable:
        if epoch > PROJECTION_SLOTS:
            for consumed_peer in pl.range(PROJECTION_RANKS):
                pld.system.defer_wait(consumed, [lane, consumed_peer], generation - 1, cmp=pld.WaitCmp.Ge)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="projection_target_prepare", deps=[reusable], no_dep_args=[prepared], allow_early_resolve=False) as space_ready:
        for prepared_peer in pl.range(PROJECTION_RANKS):
            peer_rank = pl.cast(prepared_peer, pl.INT32)
            pld.system.notify(prepared, peer_rank, [lane, my_rank], value=1, op=pld.NotifyOp.AtomicAdd)
    target_peer_0 = pl.read(peers, [0])
    target_peer_1 = pl.read(peers, [1])
    target_peer_2 = pl.read(peers, [2])
    target_peer_3 = pl.read(peers, [3])
    target_peer_4 = pl.read(peers, [4])
    target_peer_5 = pl.read(peers, [5])
    target_peer_6 = pl.read(peers, [6])
    target_peer_7 = pl.read(peers, [7])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="projection_prepared_defer_wait", deps=[space_ready], no_dep_args=[prepared], allow_early_resolve=False) as targets_ready:
        if my_rank == owner:
            if PROJECTION_RANKS // replicas > 0:
                pld.system.defer_wait(prepared, [lane, target_peer_0], generation, cmp=pld.WaitCmp.Ge)
            if PROJECTION_RANKS // replicas > 1:
                pld.system.defer_wait(prepared, [lane, target_peer_1], generation, cmp=pld.WaitCmp.Ge)
            if PROJECTION_RANKS // replicas > 2:
                pld.system.defer_wait(prepared, [lane, target_peer_2], generation, cmp=pld.WaitCmp.Ge)
            if PROJECTION_RANKS // replicas > 3:
                pld.system.defer_wait(prepared, [lane, target_peer_3], generation, cmp=pld.WaitCmp.Ge)
            if PROJECTION_RANKS // replicas > 4:
                pld.system.defer_wait(prepared, [lane, target_peer_4], generation, cmp=pld.WaitCmp.Ge)
            if PROJECTION_RANKS // replicas > 5:
                pld.system.defer_wait(prepared, [lane, target_peer_5], generation, cmp=pld.WaitCmp.Ge)
            if PROJECTION_RANKS // replicas > 6:
                pld.system.defer_wait(prepared, [lane, target_peer_6], generation, cmp=pld.WaitCmp.Ge)
            if PROJECTION_RANKS // replicas > 7:
                pld.system.defer_wait(prepared, [lane, target_peer_7], generation, cmp=pld.WaitCmp.Ge)
    return reusable, targets_ready


@pl.jit.inline(auto_scope=False)
def projection_exchange(
    kv: pl.Tensor,
    gate: pl.Tensor,
    window: pld.DistributedTensor[[PROJECTION_WINDOW_ROWS, PROJECTION_WINDOW_COLS], pl.FP32],
    ready: pld.DistributedTensor[[PROJECTION_SIGNAL_ROWS, PROJECTION_RANKS], pl.INT32],
    peers: pl.Tensor[[PROJECTION_RANKS], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    rows: pl.Scalar[pl.INDEX],
    producer_dep: pl.Scalar[pl.TASK_ID],
    targets_ready: pl.Scalar[pl.TASK_ID],
    kind: pl.constexpr,
    replicas: pl.constexpr,
    columns: pl.constexpr,
    has_gate: pl.constexpr,
):
    """Publish pure projections and make their FP32 rows available to local consumers."""
    lane = (epoch - 1) % PROJECTION_SLOTS
    generation = pl.cast((epoch - 1) // PROJECTION_SLOTS + 1, pl.INT32)
    row = lane * PROJECTION_TILE
    owner = pl.read(peers, [(epoch - 1) % (PROJECTION_RANKS // replicas)])
    published_tasks = pl.array.create(1, pl.TASK_ID)
    with pl.manual_scope():
        with pl.spmd(PROJECTION_RANKS // replicas, name_hint="projection_put_notify", deps=[producer_dep, targets_ready], allow_early_resolve=False) as pushed_inner:
            if my_rank == owner:
                member = pl.tile.get_block_idx()
                peer = pl.read(peers, [member])
                pld.tensor.put(window, peer, kv, dst_offsets=[row, 0], src_offsets=[0, 0], shape=[rows, columns], chunk_rows=PROJECTION_ROW_TILE, chunk_cols=columns)
                if has_gate:
                    pld.tensor.put(window, peer, gate, dst_offsets=[row, columns], src_offsets=[0, 0], shape=[rows, columns], chunk_rows=PROJECTION_ROW_TILE, chunk_cols=columns)
                pld.system.notify(ready, peer, [lane, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        published_tasks[0] = pushed_inner
    pushed = published_tasks[0]
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="projection_payload_defer_wait", deps=[pushed], no_dep_args=[ready], allow_early_resolve=False) as payload_ready:
        pld.system.defer_wait(ready, [lane, 0], generation, cmp=pld.WaitCmp.Ge)
    copied_tasks = pl.array.create(1, pl.TASK_ID)
    with pl.manual_scope():
        with pl.spmd((rows + PROJECTION_ROW_TILE - 1) // PROJECTION_ROW_TILE, name_hint="projection_consumer_ready", deps=[payload_ready, pushed], allow_early_resolve=False) as consumer_ready_inner:
            t0 = pl.tile.get_block_idx() * PROJECTION_ROW_TILE
            valid = pl.min(PROJECTION_ROW_TILE, rows - t0)
            for c0 in pl.range(0, columns, PROJECTION_COL_TILE):
                values = pl.slice(window, [PROJECTION_ROW_TILE, PROJECTION_COL_TILE], [row + t0, c0], valid_shape=[valid, PROJECTION_COL_TILE])
                kv[t0:t0 + PROJECTION_ROW_TILE, c0:c0 + PROJECTION_COL_TILE] = values
                if has_gate:
                    gates = pl.slice(window, [PROJECTION_ROW_TILE, PROJECTION_COL_TILE], [row + t0, columns + c0], valid_shape=[valid, PROJECTION_COL_TILE])
                    gate[t0:t0 + PROJECTION_ROW_TILE, c0:c0 + PROJECTION_COL_TILE] = gates
        copied_tasks[0] = consumer_ready_inner
    consumer_ready = copied_tasks[0]
    return consumer_ready


@pl.jit.inline(auto_scope=False)
def projection_release(
    consumed: pld.DistributedTensor[[PROJECTION_SIGNAL_ROWS, PROJECTION_RANKS], pl.INT32],
    peers: pl.Tensor[[PROJECTION_RANKS], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    consumer: pl.Scalar[pl.TASK_ID],
    kind: pl.constexpr,
    replicas: pl.constexpr,
):
    """Retire every local scratch reader before granting the next slot overwrite."""
    lane = (epoch - 1) % PROJECTION_SLOTS
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="projection_consumed", deps=[consumer], no_dep_args=[consumed], allow_early_resolve=False) as released:
        for member in pl.range(PROJECTION_RANKS):
            peer = pl.cast(member, pl.INT32)
            pld.system.notify(consumed, peer, [lane, my_rank], value=1, op=pld.NotifyOp.AtomicAdd)
    return released


def projection_groups(topology, replicas):
    """Partition measured rank links into equal groups with minimum cross-group broadcast cost.

    Each topology row supplies rank, board, chip, and a complete peer_cost mapping.
    Rank identifiers are communication ranks, independent of device and page identifiers.
    """
    import itertools

    if replicas not in (1, 2, 4, 8):
        raise ValueError("Projection replicas must be 1, 2, 4, or 8")
    records = {int(row["rank"]): row for row in topology}
    if set(records) != set(range(PROJECTION_RANKS)):
        raise ValueError("A measured topology record for each communication rank is required")
    for row in records.values():
        if "board" not in row or "chip" not in row or set(map(int, row["peer_cost"])) != set(records):
            raise ValueError("Complete board/chip identity and peer costs are required")
    size = PROJECTION_RANKS // replicas

    def cost(group):
        return sum(float(records[a]["peer_cost"].get(b, records[a]["peer_cost"].get(str(b)))) for a in group for b in group if a != b)

    def partitions(remaining):
        if not remaining:
            yield ()
        else:
            first, *rest = remaining
            for tail in itertools.combinations(rest, size - 1):
                group = (first, *tail)
                rest_set = set(group)
                for others in partitions(tuple(r for r in remaining if r not in rest_set)):
                    yield (group, *others)

    groups = min(partitions(tuple(sorted(records))), key=lambda p: (sum(map(cost, p)), p))
    return [list(next(g for g in groups if r in g)) + [-1] * (PROJECTION_RANKS - size) for r in range(PROJECTION_RANKS)]


def select_projection_degree(measurements, *, attention, chunk_size):
    """Select a qualified paired wall-time winner for one attention and chunk size."""
    import math

    identity = ("metric", "baseline_id", "topology_id", "input_id", "rounds", "warmup")
    rows = [r for r in measurements if r.get("attention") == attention and r.get("chunk_size") == chunk_size]
    rows = [r for r in rows if r.get("qualified") is True and all(key in r for key in identity)]
    rows = [r for r in rows if r["metric"] in ("public_submit_to_result_median_us", "fastest_rank_device_median_us")]
    rows = [r for r in rows if math.isfinite(float(r.get("wall_median_us", math.nan))) and r["wall_median_us"] > 0]
    baselines = [r for r in rows if r.get("d") == 8]
    winners = []
    for baseline in baselines:
        for candidate in rows:
            paired = all(candidate[key] == baseline[key] for key in identity)
            verified = candidate.get("mechanism_verified") is True
            if candidate.get("d") in (1, 2, 4) and paired and verified:
                if candidate["wall_median_us"] < baseline["wall_median_us"]:
                    winners.append(candidate)
    return int(min(winners, key=lambda row: row["wall_median_us"])["d"]) if winners else 8


def projection_degree(attention):
    """Read the offline-selected degree for one attention family; retain eight replicas by default."""
    import sys

    flag = "--prefill-projection-" + attention + "-d"
    result = 8
    for index, value in enumerate(sys.argv):
        if value == flag and index + 1 < len(sys.argv):
            result = int(sys.argv[index + 1])
        elif value.startswith(flag + "="):
            result = int(value.split("=", 1)[1])
    if result not in (1, 2, 4, 8):
        raise ValueError("Projection replicas must be 1, 2, 4, or 8")
    return result


def projection_tensor_specs(tp_size):
    """Build fixture peer metadata from an explicitly supplied measured topology."""
    import json
    import sys
    from pathlib import Path

    import torch
    from golden import TensorSpec

    topology_path = None
    for index, value in enumerate(sys.argv):
        if value == "--prefill-projection-topology" and index + 1 < len(sys.argv):
            topology_path = Path(sys.argv[index + 1])
    degrees = [projection_degree(family) for family in ("swa", "csa", "hca")]
    if any(degree < 8 for degree in degrees) and (tp_size != 8 or topology_path is None):
        raise ValueError("Shared projection fixtures require TP8 and an explicit measured topology")
    peers = torch.full((tp_size, 3, PROJECTION_RANKS), -1, dtype=torch.int32)
    for family, degree in enumerate(degrees):
        if degree == 8:
            peers[:, family, 0] = torch.arange(tp_size, dtype=torch.int32)
        else:
            topology = json.loads(topology_path.read_text())
            peers[:, family] = torch.tensor(projection_groups(topology, degree), dtype=torch.int32)
    epochs = torch.zeros((tp_size, PROJECTION_KINDS), dtype=torch.int32)
    return [
        TensorSpec("projection_peers", list(peers.shape), torch.int32, init_value=peers),
        TensorSpec("projection_epoch_bases", list(epochs.shape), torch.int32, init_value=epochs),
    ]


def projection_epoch_advance(bases, request_lengths, *, layers=43):
    """Advance exact tile counts from physical packed request lengths without a device result."""
    if type(layers) is not int or not 1 <= layers <= 43:
        raise ValueError("Projection epoch accounting supports one through 43 model layers")
    if len(bases) != PROJECTION_KINDS or any(type(base) is not int or base < 0 for base in bases):
        raise ValueError("Six nonnegative integer epoch bases are required")
    if any(type(length) is not int or length < 0 for length in request_lengths):
        raise ValueError("Six nonnegative epoch bases and nonnegative request lengths are required")
    raw_tiles = (sum(request_lengths) + PROJECTION_TILE - 1) // PROJECTION_TILE
    state_tiles = sum((length + PROJECTION_TILE - 1) // PROJECTION_TILE for length in request_lengths)
    counts = [min(layers, 2), sum(i >= 2 and i % 2 == 0 for i in range(layers)), sum(i >= 3 and i % 2 == 1 for i in range(layers))]
    increments = [raw_tiles * n for n in counts] + [state_tiles * counts[1], state_tiles * counts[2], state_tiles * counts[1]]
    result = [int(base) + count for base, count in zip(bases, increments)]
    if any(base < 0 for base in result) or sum(result) >= 2**31 - 1:
        raise ValueError("Projection epochs require a new communication domain before INT32 overflow")
    return result
