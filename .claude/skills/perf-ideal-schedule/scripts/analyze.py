# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Solve and validate conservative task-gang schedules from A5 level-1 captures."""

import argparse
from collections import defaultdict, deque
import json
import math
import time
from pathlib import Path

try:
    from ortools.sat.python import cp_model
except ImportError as exc:
    raise SystemExit(
        "Install ortools in an isolated target and add it to PYTHONPATH; retain the active PyPTO environment."
    ) from exc

from scheduling import (
    SCALE,
    demand,
    expand_waves,
    graph_times,
    list_schedule,
    pressure_report,
    verify_schedule,
)


def solve_exact(info, stats, edges, graph, timeout, hint):
    duration = graph[0]
    horizon = max(hint[1].values())
    model = cp_model.CpModel()
    starts, ends, intervals = {}, {}, {}
    resource, demands = {"aic": [], "aiv": []}, {"aic": [], "aiv": []}
    for tid in sorted(duration):
        w = duration[tid]
        starts[tid] = model.new_int_var(0, horizon, "s" + tid)
        ends[tid] = model.new_int_var(0, horizon, "e" + tid)
        model.add(ends[tid] == starts[tid] + w)
        model.add_hint(starts[tid], hint[0][tid])
        model.add_hint(ends[tid], hint[1][tid])
        if not w:
            continue
        iv = model.new_interval_var(starts[tid], w, ends[tid], "iv" + tid)
        intervals[tid] = iv
        for engine, n in zip(("aic", "aiv"), demand(info[tid])):
            if n:
                resource[engine].append(iv)
                demands[engine].append(n)
    for a, b in edges:
        model.add(starts[b] >= ends[a])
    for engine, cap in [("aic", 32), ("aiv", 64)]:
        model.add_cumulative(resource[engine], demands[engine], cap)
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, [ends[t] for t in duration])
    model.minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_search_workers = 4
    solver.parameters.random_seed = 0
    status = solver.solve(model)
    result = {
        "engine_status": solver.status_name(status),
        "engine_schedule_us": None,
        "solver_bound_us": solver.best_objective_bound / SCALE,
        "paired_core_assignment": "NOT_RUN",
        "candidate": None,
    }
    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        if status != cp_model.UNKNOWN:
            raise ValueError(("Unexpected solver failure with verified feasible hint", result))
        return result
    result["engine_schedule_us"] = solver.value(makespan) / SCALE
    # Verify the relaxed engine capacities have a physically paired core assignment.
    placement = cp_model.CpModel()
    cores = defaultdict(list)
    presence = defaultdict(list)
    for tid in intervals:
        ks = info[tid]["kernel_ids"]
        b = info[tid]["block_num"]
        st = solver.value(starts[tid])
        en = solver.value(ends[tid])
        candidates = []
        if ks[0] >= 0:
            for cluster in range(32):
                v = placement.new_bool_var(f"{tid}c{cluster}")
                iv = placement.new_optional_fixed_size_interval_var(st, en - st, v, f"p{tid}c{cluster}")
                cores[cluster].append(iv)
                presence[cluster].append((tid, v))
                candidates.append(v)
                if ks[1] >= 0 and ks[2] >= 0:
                    cores[32 + 2 * cluster].append(iv)
                    cores[33 + 2 * cluster].append(iv)
                    presence[32 + 2 * cluster].append((tid, v))
                    presence[33 + 2 * cluster].append((tid, v))
                elif ks[1] >= 0 or ks[2] >= 0:
                    # Preserve the active AIV slot within the selected MIX cluster.
                    pair = []
                    for slot in [i for i in range(2) if ks[i + 1] >= 0]:
                        a = placement.new_bool_var(f"{tid}v{cluster}_{slot}")
                        vi = placement.new_optional_fixed_size_interval_var(
                            st, en - st, a, f"pv{tid}{cluster}_{slot}"
                        )
                        cores[32 + 2 * cluster + slot].append(vi)
                        presence[32 + 2 * cluster + slot].append((tid, a))
                        pair.append(a)
                    placement.add(sum(pair) == v)
            placement.add(sum(candidates) == b)
        else:
            for core in range(32, 96):
                v = placement.new_bool_var(f"{tid}v{core}")
                iv = placement.new_optional_fixed_size_interval_var(st, en - st, v, f"p{tid}v{core}")
                cores[core].append(iv)
                presence[core].append((tid, v))
                candidates.append(v)
            placement.add(sum(candidates) == b * sum(k >= 0 for k in ks[1:]))
    for ivs in cores.values():
        placement.add_no_overlap(ivs)
    ps = cp_model.CpSolver()
    ps.parameters.max_time_in_seconds = timeout
    ps.parameters.num_search_workers = 4
    pstatus = ps.solve(placement)
    has_placement = pstatus in [cp_model.OPTIMAL, cp_model.FEASIBLE]
    result["paired_core_assignment"] = ps.status_name(pstatus)
    if has_placement:
        assigned = defaultdict(list)
        for core, entries in presence.items():
            for tid, present in entries:
                if ps.value(present):
                    assigned[tid].append(core)
        result["candidate"] = (
            {tid: solver.value(v) for tid, v in starts.items()},
            {tid: solver.value(v) for tid, v in ends.items()},
            dict(assigned),
        )
    return result


def run(key, record):
    began = time.monotonic()
    r = record["ranks"][record["rank"]]
    p = Path(r["records"])
    deps = json.loads((p.parent / "deps.json").read_text())
    info, stats, edges, expansions = expand_waves(deps, r["tasks"])
    graph = graph_times(info, stats, edges)
    duration, _, _, asap_start, asap_end, _, _ = graph
    included = [t for t in duration if duration[t]]
    if not included:
        raise ValueError("No positive-duration compute tasks")
    path_us = max(asap_end.values()) / SCALE
    pressure = pressure_report(info, stats, asap_start, asap_end)
    lower = (
        max(
            max(asap_end.values()),
            *(
                math.ceil(sum(duration[t] * demand(info[t])[engine] for t in included) / capacity)
                for engine, capacity in enumerate((32, 64))
            ),
        )
        / SCALE
    )
    fallback = list_schedule(info, stats, graph)
    verify_schedule(info, duration, edges, *fallback)
    mode = record.get("solver", "auto")
    limit = record.get("exact_task_limit", 500)
    use_exact = mode == "exact" or (mode == "auto" and len(duration) <= limit)
    exact = {
        "engine_status": "NOT_RUN",
        "engine_schedule_us": None,
        "solver_bound_us": lower,
        "paired_core_assignment": "NOT_RUN",
        "candidate": None,
    }
    reason = (
        "requested heuristic"
        if mode == "heuristic"
        else f"{len(duration)} graph nodes exceed exact limit {limit}"
    )
    if use_exact:
        exact = solve_exact(info, stats, edges, graph, record.get("timeout", 30), fallback)
        reason = f"engine {exact['engine_status']}; placement {exact['paired_core_assignment']}"
    candidate = exact.pop("candidate")
    chosen = fallback
    method = "heuristic"
    if candidate is not None:
        verify_schedule(info, duration, edges, *candidate)
        if max(candidate[1].values()) <= max(fallback[1].values()):
            chosen, method = candidate, "cp-sat"
        else:
            reason = "verified heuristic is shorter than the placed engine candidate"
    starts, ends, assigned = chosen
    physical = verify_schedule(info, duration, edges, starts, ends, assigned)
    feasible = max(ends.values()) / SCALE
    lower = max(lower, exact["solver_bound_us"])
    if lower > feasible + 1e-6:
        raise ValueError("Model bound exceeds verified feasible schedule")
    status = exact["engine_status"] if method == "cp-sat" else "HEURISTIC"
    schedule = [
        {
            "task_id": tid,
            "original_task_id": stats[tid].get("original_task_id", tid),
            "name": stats[tid]["name"],
            "blocks": info[tid]["block_num"],
            "start_us": starts[tid] / SCALE,
            "end_us": ends[tid] / SCALE,
            "weight_us": duration[tid] / SCALE,
            "kernel_ids": info[tid]["kernel_ids"],
        }
        for tid in included
    ]
    schedule.sort(key=lambda t: (t["start_us"], t["task_id"]))
    return {
        "case": key,
        "previous_path_us": path_us,
        "single_wave_dependency_path_us": r["kernel_us"],
        "resource_schedule_us": feasible,
        "resource_lower_bound_us": lower,
        "optimality_gap_us": feasible - lower,
        "status": status,
        **exact,
        "schedule_method": method,
        "fallback_reason": reason if method == "heuristic" else None,
        "physical_assignment": "VERIFIED",
        "physical_cores": physical,
        "solver_wall_s": time.monotonic() - began,
        "schedule": schedule,
        "records": str(p),
        "include_setup": record.get("include_setup", False),
        "extra_us": feasible - path_us,
        "wave_expansions": expansions,
        "parallel_pressure": pressure,
        "graph_nodes": len(duration),
        "compute_tasks": len(included),
        "model": "ordered-wave task-gang; each wave charges original max worker duration; paired MIX cores; ignores gaps and bandwidth changes",
    }


def load_capture(path, include_setup=False, exclusions=()):
    from simpler_setup.tools.swimlane_converter import read_perf_data

    raw = json.loads(path.read_text())
    decoded = read_perf_data(path)
    if decoded.get("chip_swimlane_level") != 1:
        raise ValueError("perf-ideal-schedule requires level 1")
    types = raw["metadata"]["core_types"]
    if types != ["aic"] * 32 + ["aiv"] * 64:
        raise ValueError("This model supports the A5 32 AIC / 64 AIV topology only")
    rows = decoded["tasks"]
    if not rows or len(rows) != len(raw["aicore_tasks"]):
        raise ValueError("Raw/decoded row count mismatch or empty capture")
    if len({r.get("run_epoch") for r in rows}) != 1:
        raise ValueError("Multiple execution epochs")
    deps = json.loads((path.parent / "deps.json").read_text())
    names_path = path.parent / "name_map.json"
    if not names_path.exists():
        candidates = list(path.parent.glob("name_map*.json"))
        if len(candidates) != 1:
            raise ValueError("Missing or ambiguous name map")
        names_path = candidates[0]
    names = json.loads(names_path.read_text())["callable_id_to_name"]
    by_task = defaultdict(list)
    for row in rows:
        if abs(row["duration_us"] - (row["end_time_us"] - row["start_time_us"])) > 1e-7:
            raise ValueError("Duration mismatch")
        if not 0 <= row["core_id"] < 96 or types[row["core_id"]] != row["core_type"]:
            raise ValueError("Invalid physical core")
        if include_setup and row.get("receive_time_us") is None:
            raise ValueError("Setup requested but receive timestamp missing")
        if include_setup and row["receive_time_us"] > row["start_time_us"] + 1e-7:
            raise ValueError("Receive follows kernel start")
        by_task[str(row["task_id"])].append(row)
    known = {str(t["task_id"]) for t in deps["tasks"]}
    if set(by_task) - known:
        raise ValueError("Decoded tasks missing from dependencies")
    stats = []
    included = []
    for task in deps["tasks"]:
        tid = str(task["task_id"])
        rr = by_task[tid]
        slots = [k for k in task["kernel_ids"] if k >= 0]
        if len(rr) != task["block_num"] * len(slots):
            raise ValueError(f"Incomplete worker rows for {tid}")
        expected_c = task["block_num"] * int(task["kernel_ids"][0] >= 0)
        expected_v = task["block_num"] * sum(k >= 0 for k in task["kernel_ids"][1:])
        if (
            sum(r["core_type"] == "aic" for r in rr) != expected_c
            or sum(r["core_type"] == "aiv" for r in rr) != expected_v
        ):
            raise ValueError(f"AIC/AIV worker counts disagree with kernel slots for {tid}")
        if not rr:
            continue
        name = "+".join(dict.fromkeys(names[str(k)] for k in slots))
        weight = max(
            r["end_time_us"] - (r["receive_time_us"] if include_setup else r["start_time_us"]) for r in rr
        )
        excluded = any(n in exclusions for n in name.split("+"))
        if excluded:
            weight = 0
        else:
            included.extend(rr)
        stats.append({"task_id": tid, "name": name, "blocks": task["block_num"], "max_us": weight})
    if not included:
        raise ValueError("No included tasks")
    # Check the graph independently, including zero-duration dummy/alloc nodes.
    weights = {s["task_id"]: math.ceil(s["max_us"] * SCALE - 1e-6) for s in stats}
    pred = defaultdict(set)
    succ = defaultdict(set)
    nodes = set(known)
    for edge in deps["edges"]:
        a, b = str(edge["pred"]), str(edge["succ"])
        nodes.update([a, b])
        pred[b].add(a)
        succ[a].add(b)
    indegree = {t: len(pred[t]) for t in nodes}
    queue = deque(sorted(t for t in nodes if not indegree[t]))
    finish = {}
    while queue:
        t = queue.popleft()
        finish[t] = max((finish[p] for p in pred[t]), default=0) + weights.get(t, 0)
        for child in succ[t]:
            indegree[child] -= 1
            if not indegree[child]:
                queue.append(child)
    if len(finish) != len(nodes):
        raise ValueError("Dependency cycle")
    first = min(r["receive_time_us"] if include_setup else r["start_time_us"] for r in included)
    return {
        "records": str(path),
        "tasks": stats,
        "kernel_us": max(finish.values()) / SCALE,
        "observed_us": max(r["end_time_us"] for r in included) - first,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="Build directory, dfx_outputs, or one records JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-ranks", type=int, help="Required for a directory containing rankN/dK captures"
    )
    parser.add_argument(
        "--include-setup", action="store_true", help="Use max(receive -> end) on the same worker"
    )
    parser.add_argument(
        "--exclude-name",
        action="append",
        default=[],
        help="Additional zero-weight tasks; dependencies retained",
    )
    parser.add_argument("--keep-signal-clear", action="store_true")
    parser.add_argument("--solver", choices=("auto", "exact", "heuristic"), default="auto")
    parser.add_argument(
        "--exact-task-limit",
        type=int,
        default=500,
        help="Auto mode uses the verified heuristic above this expanded graph size",
    )
    parser.add_argument("--timeout", type=float, default=30, help="Seconds per optimization/placement solve")
    args = parser.parse_args()
    if (
        args.exact_task_limit < 1
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or (args.expected_ranks is not None and args.expected_ranks < 1)
    ):
        parser.error("Timeout must be finite and positive; task limit and expected-ranks must be positive")
    source = args.capture.resolve()
    compared_ranks = False
    if source.is_file():
        paths = [source]
    else:
        source = source / "dfx_outputs" if (source / "dfx_outputs").is_dir() else source
        paths = sorted(source.glob("rank*/d*/chip_swimlane_records.json"))
        if paths:
            compared_ranks = True
            if args.expected_ranks is None:
                parser.error(
                    "Pass --expected-ranks for distributed captures; incomplete ranks cannot be ignored"
                )
            ranks = {p.parent.parent.name for p in paths}
            if ranks != {f"rank{i}" for i in range(args.expected_ranks)}:
                parser.error("Missing or unexpected ranks")
            layouts = [{p.parent.name for p in paths if p.parent.parent.name == r} for r in ranks]
            if any(s != layouts[0] for s in layouts):
                parser.error("Ranks have different dispatch sets; inspect dispatch_program.json")
            for dispatch in layouts[0]:
                programs = set()
                for path in paths:
                    if path.parent.name != dispatch:
                        continue
                    identity_path = path.parent / "dispatch_program.json"
                    if not identity_path.exists():
                        parser.error("Distributed comparison requires dispatch_program.json for every rank")
                    program = json.loads(identity_path.read_text()).get("program")
                    if not isinstance(program, str) or not program:
                        parser.error(f"Missing program identity: {identity_path}")
                    programs.add(program)
                if len(programs) != 1:
                    parser.error(f"Ranks execute different programs in {dispatch}")
        else:
            paths = [source / "chip_swimlane_records.json"]
            if args.expected_ranks not in (None, 1):
                parser.error("Expected multiple ranks but no distributed layout exists")
    exclusions = set(args.exclude_name)
    if not args.keep_signal_clear:
        exclusions.add("moe_signal_clear")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for path in paths:
        print(f"Analyzing {path}", flush=True)
        r = load_capture(path, args.include_setup, exclusions)
        key = (
            path.parent.parent.name + "/" + path.parent.name
            if path.parent.parent.name.startswith("rank")
            else "capture"
        )
        result = run(
            key,
            {
                "rank": "selected",
                "ranks": {"selected": r},
                "timeout": args.timeout,
                "solver": args.solver,
                "exact_task_limit": args.exact_task_limit,
                "include_setup": args.include_setup,
            },
        )
        result["observed_us"] = r["observed_us"]
        result["exclusions"] = sorted(exclusions)
        results[key] = result
    totals = defaultdict(float)
    for key, r in results.items():
        totals[key.split("/")[0]] += r["observed_us"]
    selected = min(totals, key=totals.get)
    rank_summaries = {}
    for rank, observed in totals.items():
        dispatches = [r for key, r in results.items() if key.split("/")[0] == rank]
        all_placed = all(r["resource_schedule_us"] is not None for r in dispatches)
        rank_summaries[rank] = {
            "observed_span_us": observed,
            "dependency_path_us": sum(r["previous_path_us"] for r in dispatches),
            "feasible_schedule_us": sum(r["resource_schedule_us"] for r in dispatches)
            if all_placed
            else None,
            "engine_schedule_us": (
                sum(r["engine_schedule_us"] for r in dispatches)
                if all(r["engine_schedule_us"] is not None for r in dispatches)
                else None
            ),
            "model_bound_us": sum(r["resource_lower_bound_us"] for r in dispatches),
            "all_optimal": all_placed and all(r["status"] == "OPTIMAL" for r in dispatches),
        }
    payload = {
        "selected_rank": selected if compared_ranks else None,
        "supplied_rank": None if compared_ranks else selected,
        "scope": "validated rank set" if compared_ranks else "one supplied rank",
        "rank_spans_us": dict(totals),
        "rank_summaries": rank_summaries,
        "results": results,
    }
    (args.output_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Ideal schedule estimate",
        "",
        f"Weight: {'receive-to-end including local setup' if args.include_setup else 'execution only; local setup excluded'}.",
        (
            f"Selected rank: {selected}, by summed included AICore spans."
            if compared_ranks
            else f"Supplied rank: {selected}; fastest rank was not established."
        ),
        "",
        "Conservative ordered-wave task-gang model, not measured latency or hardware-optimal execution. "
        "Oversized launches become sequential capacity-sized waves; each wave holds cores for the original maximum worker duration. "
        "Inter-task gaps and fixed-core ordering are removed; bandwidth/cache effects and remote-rank progress are not simulated.",
        "",
        "| Dispatch | Dependency path us | Feasible schedule us | Model bound us | Status |",
        "|---|---:|---:|---:|---|",
    ]

    def timing(value):
        return "unavailable" if value is None else f"{value:.3f}"

    for key, r in results.items():
        lines.append(
            f"| {key} | {r['previous_path_us']:.3f} | {timing(r['resource_schedule_us'])} | {r['resource_lower_bound_us']:.3f} | {r['status']}; {r['schedule_method']}; physical {r['physical_assignment']} |"
        )
    for key, r in results.items():
        lines += [
            "",
            f"### {key}: capacity diagnostics",
            "",
            f"Graph: {r['graph_nodes']} nodes / {r['compute_tasks']} compute gangs. "
            f"Feasible-minus-bound gap: {r['optimality_gap_us']:.3f} us. "
            f"Finite-core schedule minus dependency path: {r['extra_us']:.3f} us.",
        ]
        if r["fallback_reason"]:
            lines.append(f"Verified heuristic: {r['fallback_reason']}. No optimality claim.")
        lines.append(
            f"Engine status: {r['engine_status']}; candidate: {timing(r['engine_schedule_us'])} us; "
            f"engine placement: {r['paired_core_assignment']}."
        )
        lines.append("")
        for wave in r["wave_expansions"]:
            lines.append(
                f"- Oversized launch `{wave['name']}` (task {wave['task_id']}): {wave['blocks']} blocks, "
                f"AIC/AIV demand {wave['aic_demand']}/{wave['aiv_demand']} versus 32/64 cores; "
                f"{len(wave['wave_ids'])} ordered waves, {wave['blocks_per_wave']} blocks per full wave, "
                f"{wave['wave_blocks'][-1]} in the last, each {wave['worker_max_us']:.3f} us. "
                "Inspect tiling and consider coarser blocks or looping over more work per block; "
                "preserve all work and check load balance and buffer limits."
            )
        pressure = r["parallel_pressure"]
        for peak in pressure["peaks"]:
            lines.append(
                f"- Potential {peak['engine']} pressure at dependency-ASAP {peak['time_us']:.3f} us: "
                f"{peak['parallel_tasks']} tasks demand {peak['peak_demand']} / {peak['capacity']} cores. "
                "Inspect noncritical branches for fewer workers or coarser tasks, keeping critical work ready."
            )
            for group in peak["groups"][:10]:
                lines.append(
                    f"  - `{group['name']}`: {group['tasks']} tasks, "
                    f"{group['core_demand']} cores; sample IDs {', '.join(group['task_ids'])}."
                )
        if pressure["top_windows"]:
            lines += [
                "",
                "Largest dependency-ASAP demand intervals by excess core-time "
                "(not a latency penalty or predicted speedup):",
                "",
                "| Engine | Interval us | Demand / cores | Tasks | Contributors |",
                "|---|---:|---:|---:|---|",
            ]
            for window in pressure["top_windows"]:
                contributors = "; ".join(
                    f"{g['name']} ({g['core_demand']}, IDs {', '.join(g['task_ids'])})"
                    for g in window["groups"][:5]
                )
                if len(window["groups"]) > 5:
                    contributors += "; additional groups in results.json"
                lines.append(
                    f"| {window['engine']} | {window['time_us']:.3f} - {window['end_us']:.3f} | "
                    f"{window['peak_demand']} / {window['capacity']} | "
                    f"{window['parallel_tasks']} | {contributors} |"
                )
            lines.append("")
        if not r["wave_expansions"] and not pressure["peaks"]:
            lines.append("No exposed core demand exceeds capacity in the dependency-ASAP timeline.")
        if r["wave_expansions"] or pressure["peaks"]:
            lines.append(
                "These are optimization candidates, not measured occupancy or proof of wasted parallelism. "
                "Extra waves can be useful for load balance. Do not clamp block counts or serialize branches "
                "without preserving work; validate candidate changes and benchmark device wall time. "
                "Inspect generated inner-loop tiling separately: logical work can exceed cores even when "
                "the exposed launch fits in one wave."
            )
    lines += [
        "",
        "## Per-rank dispatch totals",
        "",
        "| Rank | Observed span us | Dependency path us | Feasible schedule us | Model bound us | All optimal |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for rank, r in rank_summaries.items():
        lines.append(
            f"| {rank} | {r['observed_span_us']:.3f} | {r['dependency_path_us']:.3f} | "
            f"{timing(r['feasible_schedule_us'])} | {r['model_bound_us']:.3f} | {r['all_optimal']} |"
        )
    lines += [
        "",
        "Engine OPTIMAL proves model optimality only with a paired-core witness; "
        "engine FEASIBLE is an upper bound only with that witness. HEURISTIC is a verified upper bound, not an optimum. "
        "Every reported feasible schedule has independently checked dependencies, core demand, MIX pairing and core overlap. "
        "Durations round upward to 0.001 us. All non-excluded compute tasks contribute to makespan.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
