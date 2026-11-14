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
from pathlib import Path

try:
    from ortools.sat.python import cp_model
except ImportError as exc:
    raise SystemExit(
        "Install ortools in an isolated target and add it to PYTHONPATH; retain the active PyPTO environment."
    ) from exc

SCALE = 1000


def run(key, record):
    r = record["ranks"][record["rank"]]
    p = Path(r["records"])
    deps = json.loads((p.parent / "deps.json").read_text())
    info = {str(t["task_id"]): t for t in deps["tasks"]}
    stats = {t["task_id"]: t for t in r["tasks"]}
    nodes = set(info)
    for e in deps["edges"]:
        nodes.update([str(e["pred"]), str(e["succ"])])
    horizon = sum(math.ceil(t["max_us"] * SCALE - 1e-6) for t in stats.values())
    model = cp_model.CpModel()
    starts = {}
    ends = {}
    intervals = {}
    demands = {"aic": [], "aiv": []}
    resource = {"aic": [], "aiv": []}
    dur = {}
    for tid in sorted(nodes):
        t = stats.get(tid)
        w = math.ceil(t["max_us"] * SCALE - 1e-6) if t else 0
        dur[tid] = w
        starts[tid] = model.new_int_var(0, horizon, "s" + tid)
        ends[tid] = model.new_int_var(0, horizon, "e" + tid)
        model.add(ends[tid] == starts[tid] + w)
        if not w:
            continue
        iv = model.new_interval_var(starts[tid], w, ends[tid], "iv" + tid)
        intervals[tid] = iv
        ks = info[tid]["kernel_ids"]
        b = info[tid]["block_num"]
        n_c = int(ks[0] >= 0) * b
        n_v = sum(k >= 0 for k in ks[1:]) * b
        if n_c > 32 or n_v > 64:
            raise ValueError(("multi-wave task", tid, n_c, n_v))
        for engine, n in [("aic", n_c), ("aiv", n_v)]:
            if n:
                resource[engine].append(iv)
                demands[engine].append(n)
    for e in deps["edges"]:
        model.add(starts[str(e["succ"])] >= ends[str(e["pred"])])
    for engine, cap in [("aic", 32), ("aiv", 64)]:
        model.add_cumulative(resource[engine], demands[engine], cap)
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, [ends[t] for t in intervals])
    model.minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = record.get("timeout", 30)
    solver.parameters.num_search_workers = 4
    solver.parameters.random_seed = 0
    status = solver.solve(model)
    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        raise ValueError(("no schedule", key, solver.status_name(status)))
    # Verify the relaxed engine capacities have a physically paired core assignment.
    placement = cp_model.CpModel()
    cores = defaultdict(list)
    choices = {}
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
        choices[tid] = candidates
    for ivs in cores.values():
        placement.add_no_overlap(ivs)
    ps = cp_model.CpSolver()
    ps.parameters.max_time_in_seconds = record.get("timeout", 30)
    ps.parameters.num_search_workers = 4
    pstatus = ps.solve(placement)
    has_placement = pstatus in [cp_model.OPTIMAL, cp_model.FEASIBLE]
    schedule = []
    for tid in intervals:
        t = stats[tid]
        schedule.append(
            {
                "task_id": tid,
                "name": t["name"],
                "blocks": t["blocks"],
                "start_us": solver.value(starts[tid]) / SCALE,
                "end_us": solver.value(ends[tid]) / SCALE,
                "weight_us": dur[tid] / SCALE,
                "kernel_ids": info[tid]["kernel_ids"],
            }
        )
    schedule.sort(key=lambda t: (t["start_us"], t["task_id"]))
    events = []
    for t in schedule:
        ks = t["kernel_ids"]
        b = t["blocks"]
        events.extend(
            [
                (t["start_us"], b * int(ks[0] >= 0), b * sum(k >= 0 for k in ks[1:])),
                (t["end_us"], -b * int(ks[0] >= 0), -b * sum(k >= 0 for k in ks[1:])),
            ]
        )
    # Independent sweep check of every half-open resource interval.
    for a in sorted({e[0] for e in events}):
        active = [t for t in schedule if t["start_us"] <= a < t["end_us"]]
        nc = sum(t["blocks"] * int(t["kernel_ids"][0] >= 0) for t in active)
        nv = sum(t["blocks"] * sum(k >= 0 for k in t["kernel_ids"][1:]) for t in active)
        if nc > 32 or nv > 64:
            raise ValueError("resource violation")
    result = {
        "case": key,
        "previous_path_us": r["kernel_us"],
        "resource_schedule_us": solver.value(makespan) / SCALE if has_placement else None,
        "engine_schedule_us": solver.value(makespan) / SCALE,
        "resource_lower_bound_us": solver.best_objective_bound / SCALE,
        "status": solver.status_name(status),
        "paired_core_assignment": ps.status_name(pstatus),
        "solver_wall_s": solver.wall_time,
        "schedule": schedule,
        "records": str(p),
        "include_setup": record.get("include_setup", False),
        "model": "task-gang; all physical workers charged task max duration; paired MIX cores; ignores gaps and bandwidth changes",
    }
    if not has_placement:
        result["physical_cores"] = None
        result["extra_us"] = None
        return result
    result["physical_cores"] = {
        str(core): [tid for tid, v in vv if ps.value(v)] for core, vv in presence.items()
    }
    for core, tids in result["physical_cores"].items():
        ordered = sorted(tids, key=lambda t: solver.value(starts[t]))
        for a, b in zip(ordered, ordered[1:]):
            if solver.value(ends[a]) > solver.value(starts[b]):
                raise ValueError(("core overlap", core, a, b))
    result["extra_us"] = result["resource_schedule_us"] - r["kernel_us"]
    return result


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
    parser.add_argument("--timeout", type=float, default=30, help="Seconds per optimization/placement solve")
    args = parser.parse_args()
    if args.timeout <= 0 or (args.expected_ranks is not None and args.expected_ranks < 1):
        parser.error("Timeout and expected-ranks must be positive")
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
            "feasible_schedule_us": sum(r["resource_schedule_us"] for r in dispatches) if all_placed else None,
            "engine_schedule_us": sum(r["engine_schedule_us"] for r in dispatches),
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
        "Conservative task-gang model, not measured latency or hardware-optimal execution. "
        "All blocks start together and hold cores until the task maximum worker duration. "
        "Inter-task gaps and fixed-core ordering are removed; bandwidth/cache effects and remote-rank progress are not simulated.",
        "",
        "| Dispatch | Dependency path us | Feasible schedule us | Model bound us | Status |",
        "|---|---:|---:|---:|---|",
    ]
    def timing(value):
        return "unavailable" if value is None else f"{value:.3f}"

    for key, r in results.items():
        lines.append(
            f"| {key} | {r['previous_path_us']:.3f} | {timing(r['resource_schedule_us'])} | {r['resource_lower_bound_us']:.3f} | {r['status']}; placement {r['paired_core_assignment']} |"
        )
    for key, r in results.items():
        if r["resource_schedule_us"] is None:
            lines.append("")
            lines.append(
                f"{key}: placement {r['paired_core_assignment']}; engine candidate "
                f"{r['engine_schedule_us']:.3f} us is diagnostic only, not a feasible schedule."
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
        "engine FEASIBLE is an upper bound only with that witness. "
        "Every reported feasible schedule has a paired-core assignment with independently checked core overlap. "
        "Durations round upward to 0.001 us. All non-excluded compute tasks contribute to makespan.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
