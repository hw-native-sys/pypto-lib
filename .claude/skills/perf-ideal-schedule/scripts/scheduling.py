# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Wave expansion, demand diagnostics and bounded-core list scheduling."""

from collections import defaultdict, deque
import heapq
import math

SCALE = 1000
CAPACITIES = (32, 64)


def demand(task):
    kernels = task["kernel_ids"]
    blocks = task["block_num"]
    return blocks * int(kernels[0] >= 0), blocks * sum(k >= 0 for k in kernels[1:])


def expand_waves(deps, stats):
    """Retain original task IDs as completion nodes; redirect incoming edges."""
    info = {str(t["task_id"]): dict(t) for t in deps["tasks"]}
    stats = {str(t["task_id"]): dict(t) for t in stats}
    used = set(info) | {str(e[k]) for e in deps["edges"] for k in ("pred", "succ")}
    first, extra_edges, expansions = {}, [], []
    for tid, task in list(info.items()):
        old = stats.get(tid)
        if not old or old["max_us"] <= 0:
            continue
        nc, nv = demand(task)
        blocks = task["block_num"]
        limits = ([32] if nc else []) + ([64 // (nv // blocks)] if nv else [])
        if not limits:
            raise ValueError(f"Positive-duration task without active kernels: {tid}")
        capacity = min(limits)
        if blocks <= capacity:
            continue
        wave_ids, wave_blocks = [], []
        for wave, offset in enumerate(range(0, blocks, capacity)):
            wid = f"{tid}:wave{wave}"
            if wid in used:
                raise ValueError(f"Wave ID collision: {wid}")
            used.add(wid)
            count = min(capacity, blocks - offset)
            info[wid] = {**task, "task_id": wid, "block_num": count}
            stats[wid] = {**old, "task_id": wid, "blocks": count, "original_task_id": tid}
            wave_ids.append(wid)
            wave_blocks.append(count)
        stats[tid] = {**old, "max_us": 0}
        first[tid] = wave_ids[0]
        extra_edges.extend(zip(wave_ids, wave_ids[1:] + [tid]))
        expansions.append(
            {
                "task_id": tid,
                "name": old["name"],
                "blocks": blocks,
                "aic_demand": nc,
                "aiv_demand": nv,
                "blocks_per_wave": capacity,
                "wave_ids": wave_ids,
                "wave_blocks": wave_blocks,
                "worker_max_us": old["max_us"],
            }
        )
    edges = {(str(e["pred"]), first.get(str(e["succ"]), str(e["succ"]))) for e in deps["edges"]}
    edges.update(extra_edges)
    return info, stats, sorted(edges), expansions


def graph_times(info, stats, edges):
    nodes = set(info) | {t for edge in edges for t in edge}
    pred, succ = defaultdict(set), defaultdict(set)
    for a, b in edges:
        pred[b].add(a)
        succ[a].add(b)
    duration = {t: math.ceil(stats.get(t, {}).get("max_us", 0) * SCALE - 1e-6) for t in nodes}
    if any(w < 0 for w in duration.values()):
        raise ValueError("Negative task duration")
    indegree = {t: len(pred[t]) for t in nodes}
    queue = deque(sorted(t for t in nodes if not indegree[t]))
    start, end, order = {}, {}, []
    while queue:
        t = queue.popleft()
        start[t] = max((end[p] for p in pred[t]), default=0)
        end[t] = start[t] + duration[t]
        order.append(t)
        for child in sorted(succ[t]):
            indegree[child] -= 1
            if not indegree[child]:
                queue.append(child)
    if len(order) != len(nodes):
        raise ValueError("Dependency cycle")
    tail = {}
    for t in reversed(order):
        tail[t] = duration[t] + max((tail[c] for c in succ[t]), default=0)
    return duration, pred, succ, start, end, order, tail


def pressure_report(info, stats, starts, ends):
    """Sweep the unlimited-core ASAP timeline; this is potential demand, not observed occupancy."""
    events = defaultdict(lambda: [0, 0, 0])
    for tid in starts:
        if ends[tid] == starts[tid]:
            continue
        nc, nv = demand(info[tid])
        for when, sign in ((starts[tid], 1), (ends[tid], -1)):
            for i, value in enumerate((nc, nv, 1)):
                events[when][i] += sign * value
    active = [0, 0, 0]
    peaks = [0, 0]
    at = [0, 0]
    overloaded_ticks = 0
    windows = []
    times = sorted(events)
    for index, when in enumerate(times[:-1]):
        active = [n + d for n, d in zip(active, events[when])]
        if active[0] > 32 or active[1] > 64:
            overloaded_ticks += times[index + 1] - when
        for engine, capacity in enumerate(CAPACITIES):
            if active[engine] > capacity:
                until = times[index + 1]
                windows.append(
                    ((active[engine] - capacity) * (until - when), when, until, engine, active[engine])
                )
            if active[engine] > peaks[engine]:
                peaks[engine], at[engine] = active[engine], when

    def describe(engine, when, core_demand):
        tasks = [t for t in starts if starts[t] <= when < ends[t] and demand(info[t])[engine]]
        groups = defaultdict(lambda: {"tasks": 0, "core_demand": 0, "task_ids": []})
        for tid in sorted(tasks):
            group = groups[stats[tid]["name"]]
            group["tasks"] += 1
            group["core_demand"] += demand(info[tid])[engine]
            if len(group["task_ids"]) < 8:
                group["task_ids"].append(tid)
        return {
            "engine": ("AIC", "AIV")[engine],
            "capacity": CAPACITIES[engine],
            "peak_demand": core_demand,
            "time_us": when / SCALE,
            "parallel_tasks": len(tasks),
            "groups": [
                {"name": name, **group}
                for name, group in sorted(groups.items(), key=lambda item: (-item[1]["core_demand"], item[0]))
            ],
        }

    reports = [
        describe(engine, at[engine], peaks[engine])
        for engine, capacity in enumerate(CAPACITIES)
        if peaks[engine] > capacity
    ]
    # Retain numeric windows cheaply; only expand contributors for a bounded sample.
    top_windows = []
    for excess, when, until, engine, count in heapq.nsmallest(8, windows, key=lambda w: (-w[0], w[1], w[3])):
        top_windows.append(
            {**describe(engine, when, count), "end_us": until / SCALE, "excess_core_us": excess / SCALE}
        )
    return {
        "basis": "unlimited-core dependency ASAP timeline after wave expansion",
        "overloaded_us": overloaded_ticks / SCALE,
        "peaks": reports,
        "overloaded_engine_intervals": len(windows),
        "top_windows": top_windows,
    }


def choose_cores(task, busy):
    kernels = task["kernel_ids"]
    blocks = task["block_num"]
    if kernels[0] >= 0:
        groups = [[c] + [32 + 2 * c + slot for slot in range(2) if kernels[slot + 1] >= 0] for c in range(32)]
        free = [group for group in groups if not any(c in busy for c in group)]
        return [c for group in free[:blocks] for c in group] if len(free) >= blocks else None
    count = demand(task)[1]
    free = [c for c in range(32, 96) if c not in busy]
    # Fill fragmented clusters first to leave whole MIX clusters available.
    free.sort(key=lambda c: (not (32 + ((c - 32) ^ 1) in busy or (c - 32) // 2 in busy), c))
    return free[:count] if len(free) >= count else None


def list_schedule(info, stats, graph):
    """Critical-tail priority, with ready heaps grouped by identical core demand."""
    duration, pred, succ, _, _, order, tail = graph
    indegree = {t: len(pred[t]) for t in order}
    ready = defaultdict(list)
    running, busy, starts, ends, assigned = [], set(), {}, {}, {}

    def enqueue(t):
        signature = (
            (info[t]["block_num"], tuple(k >= 0 for k in info[t]["kernel_ids"])) if duration[t] else (0, ())
        )
        heapq.heappush(ready[signature], (-tail[t], t))

    def complete(t):
        for child in sorted(succ[t]):
            indegree[child] -= 1
            if not indegree[child]:
                enqueue(child)

    for t in order:
        if not indegree[t]:
            enqueue(t)
    now = 0
    while ready or running:
        selected = None
        for signature, heap in ready.items():
            _, tid = heap[0]
            cores = choose_cores(info[tid], busy) if duration[tid] else []
            if cores is not None and (selected is None or heap[0] < selected[0]):
                selected = (heap[0], signature, cores)
        if selected is not None:
            (_, tid), signature, cores = selected
            heapq.heappop(ready[signature])
            if not ready[signature]:
                del ready[signature]
            starts[tid], ends[tid] = now, now + duration[tid]
            assigned[tid] = cores
            if duration[tid]:
                busy.update(cores)
                heapq.heappush(running, (ends[tid], tid))
            else:
                complete(tid)
            continue
        if not running:
            raise ValueError("No physically feasible ready task")
        now = running[0][0]
        while running and running[0][0] == now:
            _, tid = heapq.heappop(running)
            busy.difference_update(assigned[tid])
            complete(tid)
    if len(ends) != len(order):
        raise ValueError("Incomplete list schedule")
    return starts, ends, assigned


def verify_schedule(info, duration, edges, starts, ends, assigned):
    """Independently check precedence, exact demand, MIX slots and half-open core occupancy."""
    cores = defaultdict(list)
    if set(starts) != set(duration) or set(ends) != set(duration):
        raise ValueError("Incomplete schedule")
    for a, b in edges:
        if ends[a] > starts[b]:
            raise ValueError(f"Dependency violation: {a} -> {b}")
    for tid, weight in duration.items():
        if starts[tid] < 0 or ends[tid] - starts[tid] != weight:
            raise ValueError(f"Invalid task timing: {tid}")
        if not weight:
            continue
        selected = assigned.get(tid, [])
        nc, nv = demand(info[tid])
        if (
            len(set(selected)) != len(selected)
            or any(c < 0 or c >= 96 for c in selected)
            or sum(c < 32 for c in selected) != nc
            or sum(c >= 32 for c in selected) != nv
        ):
            raise ValueError(f"Invalid core demand: {tid}")
        if nc:
            expected = {
                32 + 2 * c + slot
                for c in selected
                if c < 32
                for slot in range(2)
                if info[tid]["kernel_ids"][slot + 1] >= 0
            }
            if expected != {c for c in selected if c >= 32}:
                raise ValueError(f"Invalid MIX pairing: {tid}")
        for core in selected:
            cores[core].append((starts[tid], ends[tid], tid))
    for core, intervals in cores.items():
        intervals.sort()
        for a, b in zip(intervals, intervals[1:]):
            if a[1] > b[0]:
                raise ValueError(f"Core overlap: {core}, {a[2]}, {b[2]}")
    return {str(core): [t for _, _, t in intervals] for core, intervals in sorted(cores.items())}
