# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Render an operator-focused report from level-4 L2 swimlane artifacts.

The observed path itself comes from ``simpler_setup.tools.critical_path``.
This companion adds:

* fastest-rank selection by dispatch-to-finish elapsed time;
* the requested per-task gap table and strict ``gap > threshold`` marker;
* proof of actual early dispatch from AICPU timestamps; and
* conservative named-blocker evidence without turning scheduler correlation
  into a causal claim.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from simpler_setup.tools import critical_path
    from simpler_setup.tools.swimlane_converter import read_perf_data
except ImportError as exc:  # pragma: no cover - exercised by the environment preflight
    raise SystemExit(
        "Could not import simpler_setup. Activate the pypto-lib environment first "
        "(for this worktree: source temp/set_env.sh)."
    ) from exc


RECORD_NAMES = ("l2_swimlane_records.json", "l2_perf_records.json")
RANK_RE = re.compile(r"rank\d+$")


@dataclass
class RunAnalysis:
    directory: Path
    rank: str
    program: str | None
    graph: Any
    result: Any
    rows: list[dict[str, Any]]
    rows_by_task: dict[str, list[dict[str, Any]]]
    rows_by_core: dict[int, list[dict[str, Any]]]
    deps: dict[str, Any]
    task_table: dict[str, dict[str, Any]]
    preds: dict[str, set[str]]
    name_map: dict[str, str]
    core_types: list[str]
    elapsed_us: float


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _records_file(directory: Path) -> Path | None:
    for name in RECORD_NAMES:
        path = directory / name
        if path.is_file():
            return path
    return None


def _record_directories(root: Path) -> list[Path]:
    if root.is_file():
        return [root.parent] if root.name in RECORD_NAMES else []
    directories: set[Path] = set()
    for name in RECORD_NAMES:
        directories.update(path.parent for path in root.rglob(name))
    return sorted(directories)


def _program(directory: Path) -> str | None:
    marker = directory / "dispatch_program.json"
    if not marker.is_file():
        return None
    data = _read_json(marker)
    value = data.get("program") if isinstance(data, dict) else None
    return str(value) if value is not None else None


def _operator_matches(operator: str, program: str) -> bool:
    query = operator.casefold()
    candidate = program.casefold()
    return candidate == query or Path(program).stem.casefold() == Path(operator).stem.casefold()


def _rank_label(directory: Path) -> str:
    for part in reversed(directory.parts):
        if RANK_RE.fullmatch(part):
            return part
    return "single"


def _rank_sort_key(rank: str) -> tuple[int, str]:
    match = re.fullmatch(r"rank(\d+)", rank)
    return (int(match.group(1)), rank) if match else (-1, rank)


def _dispatch_sort_key(analysis: RunAnalysis) -> tuple[int, str]:
    match = re.fullmatch(r"d(\d+)", analysis.directory.name)
    return (int(match.group(1)), str(analysis.directory)) if match else (-1, str(analysis.directory))


def _name_map(directory: Path) -> dict[str, str]:
    candidates = sorted(
        directory.glob("name_map*.json"),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    if not candidates:
        return {}
    data = _read_json(candidates[-1])
    if isinstance(data, dict) and isinstance(data.get("callable_id_to_name"), dict):
        data = data["callable_id_to_name"]
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def _task_id(value: Any) -> str:
    return str(int(value))


def _float(row: dict[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key} for task {row.get('task_id')}: {value}")
    return value


def _aggregate_min(rows_by_task: dict[str, list[dict[str, Any]]], key: str) -> dict[str, float]:
    return {task: min(_float(row, key) for row in rows) for task, rows in rows_by_task.items()}


def _aggregate_max(rows_by_task: dict[str, list[dict[str, Any]]], key: str) -> dict[str, float]:
    return {task: max(_float(row, key) for row in rows) for task, rows in rows_by_task.items()}


def _build_analysis(directory: Path, root: Path, tol: int) -> RunAnalysis:
    records = _records_file(directory)
    if records is None:
        raise ValueError(f"missing swimlane records in {directory}")
    raw = _read_json(records)
    level = raw.get("l2_swimlane_level") if isinstance(raw, dict) else None
    if level != 4:
        raise ValueError(f"{records}: expected l2_swimlane_level=4, got {level!r}")
    metadata = raw.get("metadata") or {}
    frequency = int(metadata.get("clock_freq_hz") or 0)
    if frequency <= 0:
        raise ValueError(f"{records}: invalid clock frequency {frequency}")

    data = read_perf_data(records)
    rows = data.get("tasks") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{records}: no joined AICore/AICPU task rows")
    raw_aicore = raw.get("aicore_tasks")
    raw_aicpu = raw.get("aicpu_tasks")
    if not isinstance(raw_aicore, list) or not isinstance(raw_aicpu, list):
        raise ValueError(f"{records}: missing raw AICore/AICPU task streams")
    if len(raw_aicore) != len(raw_aicpu) or len(rows) != len(raw_aicore):
        raise ValueError(
            f"{records}: incomplete AICore/AICPU join "
            f"(aicore={len(raw_aicore)}, aicpu={len(raw_aicpu)}, joined={len(rows)})"
        )
    required = (
        "task_id",
        "core_id",
        "core_type",
        "start_time_us",
        "end_time_us",
        "dispatch_time_us",
        "finish_time_us",
    )
    for row in rows:
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"{records}: task {row.get('task_id')} missing {', '.join(missing)}")
        task = _task_id(row["task_id"])
        if _float(row, "dispatch_time_us") < 0 or _float(row, "finish_time_us") <= 0:
            raise ValueError(f"{records}: task {task} has missing level-4 AICPU timing")

    rows_by_task: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    rows_by_core: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        rows_by_task[_task_id(row["task_id"])].append(row)
        rows_by_core[int(row["core_id"])].append(row)
    for core_rows in rows_by_core.values():
        core_rows.sort(key=lambda row: (_float(row, "start_time_us"), _float(row, "end_time_us")))

    deps = _read_json(directory / "deps.json")
    if not isinstance(deps, dict):
        raise ValueError(f"{directory / 'deps.json'}: expected a JSON object")
    task_table = {
        _task_id(task["task_id"]): task
        for task in deps.get("tasks", [])
        if isinstance(task, dict) and task.get("task_id") is not None
    }
    for task, info in task_table.items():
        try:
            active_slots = sum(int(kernel_id) >= 0 for kernel_id in (info.get("kernel_ids") or []))
            logical_blocks = int(info.get("block_num"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{directory / 'deps.json'}: task {task} has invalid kernel_ids/block_num") from exc
        expected_rows = logical_blocks * active_slots
        actual_rows = len(rows_by_task.get(task, []))
        if actual_rows != expected_rows:
            raise ValueError(
                f"{records}: incomplete task {task} rows "
                f"(joined={actual_rows}, expected={logical_blocks}×{active_slots}={expected_rows})"
            )
    preds: dict[str, set[str]] = collections.defaultdict(set)
    for edge in deps.get("edges", []):
        if not isinstance(edge, dict) or edge.get("pred") is None or edge.get("succ") is None:
            continue
        pred, succ = _task_id(edge["pred"]), _task_id(edge["succ"])
        if pred != succ:
            preds[succ].add(pred)

    graph = critical_path.build_graph(directory, root, tol)
    result = critical_path.analyze_rank(graph, tol)
    dispatch = _aggregate_min(rows_by_task, "dispatch_time_us")
    finish = _aggregate_max(rows_by_task, "finish_time_us")
    elapsed_us = max(finish.values()) - min(dispatch.values())
    return RunAnalysis(
        directory=directory,
        rank=_rank_label(directory),
        program=_program(directory),
        graph=graph,
        result=result,
        rows=rows,
        rows_by_task=dict(rows_by_task),
        rows_by_core=dict(rows_by_core),
        deps=deps,
        task_table=task_table,
        preds=dict(preds),
        name_map=_name_map(directory),
        core_types=list(metadata.get("core_types") or []),
        elapsed_us=elapsed_us,
    )


def _is_alloc(task: str, analysis: RunAnalysis) -> bool:
    return task not in analysis.task_table


def _early_producer(task: str, analysis: RunAnalysis) -> bool:
    return _is_alloc(task, analysis) or bool(analysis.task_table.get(task, {}).get("early_dispatch"))


def _early_eligible(task: str, analysis: RunAnalysis) -> bool:
    predecessors = analysis.preds.get(task, set())
    return bool(
        predecessors
        and all(_early_producer(pred, analysis) for pred in predecessors)
        and any(not _is_alloc(pred, analysis) for pred in predecessors)
    )


def _observed_early(task: str, analysis: RunAnalysis, tol_us: float) -> tuple[str, int, int]:
    """Return proof status and early/total physical-row counts."""
    rows = analysis.rows_by_task.get(task, [])
    total = len(rows)
    finish = _aggregate_max(analysis.rows_by_task, "finish_time_us")
    direct = analysis.preds.get(task, set())
    untimed_non_alloc = [
        pred for pred in direct if not _is_alloc(pred, analysis) and pred not in finish
    ]
    if untimed_non_alloc:
        return "unverifiable", 0, total
    predecessors = [pred for pred in direct if pred in finish]
    if not predecessors:
        return "none", 0, total
    observed_ready = max(finish[pred] for pred in predecessors)
    early_count = sum(_float(row, "dispatch_time_us") + tol_us < observed_ready for row in rows)
    if not _early_eligible(task, analysis):
        return ("mismatch" if early_count else "none"), early_count, total
    if early_count == total and total:
        return "full", early_count, total
    if early_count:
        return "partial", early_count, total
    return "none", 0, total


def _logical_name(task: str, analysis: RunAnalysis, row: dict[str, Any] | None = None) -> str:
    if row is not None:
        func_id = row.get("func_id")
        if func_id is not None:
            mapped = analysis.name_map.get(str(func_id))
            if mapped:
                return mapped
    return analysis.graph.name.get(task, "unknown")


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _us(ticks: int, frequency: int) -> float:
    return ticks / frequency * 1e6


def _path_table(analysis: RunAnalysis, gap_threshold_us: float, tol: int) -> tuple[list[str], list[int]]:
    lines = [
        "| # | operator/task | task id | task wall span µs | Observed compute contribution µs | "
        "gap from previous µs | gap kind | markers |",
        "|---:|---|---|---:|---:|---:|---|---|",
    ]
    snail_indices: list[int] = []
    tol_us = tol / analysis.graph.freq * 1e6
    for index, segment in enumerate(analysis.result.segments):
        duration_us = _us(segment.dur, analysis.graph.freq)
        compute_us = _us(segment.compute, analysis.graph.freq)
        gap_us = _us(segment.stall, analysis.graph.freq)
        is_snail = index > 0 and gap_us > gap_threshold_us
        early_status, early_count, total_count = _observed_early(segment.task, analysis, tol_us)
        markers = []
        if is_snail:
            markers.append("🐌")
            snail_indices.append(index)
        if early_status in {"full", "partial"}:
            markers.append(
                "⭐" if early_status == "full" else f"⭐ partial rows {early_count}/{total_count}"
            )
        elif early_status == "mismatch":
            markers.append(f"⚠ dependency/timing mismatch rows {early_count}/{total_count}")
        elif early_status == "unverifiable":
            markers.append("⚠ early-dispatch unverifiable")
        gap = "—" if index == 0 else f"{gap_us:.3f}"
        kind = "—" if index == 0 or not segment.stall else segment.kind
        lines.append(
            f"| {index} | {_md(segment.name)} | `{segment.task}` | {duration_us:.3f} | "
            f"{compute_us:.3f} | {gap} | {kind} | {' '.join(markers)} |"
        )
    return lines, snail_indices


def _unflagged_predecessors(task: str, analysis: RunAnalysis) -> list[str]:
    return sorted(
        pred
        for pred in analysis.preds.get(task, set())
        if not _is_alloc(pred, analysis) and not analysis.task_table.get(pred, {}).get("early_dispatch")
    )


def _same_core_blockers(
    task: str,
    analysis: RunAnalysis,
    tol_us: float,
    data_ready_us: float,
) -> list[tuple[str, str, int, float]]:
    """Return core occupancy that continues into the task's earliest start.

    A task that ran earlier in a speculative dispatch-to-start window but freed
    the core before the target's data was ready did not gate the target's
    critical-path start.  Use each earliest row's own dispatch timestamp and
    require the other row to reach the target start within tolerance.
    """
    target_rows = analysis.rows_by_task.get(task, [])
    if not target_rows:
        return []
    start = min(_float(row, "start_time_us") for row in target_rows)

    overlap_intervals: dict[tuple[str, int], list[tuple[float, float, dict[str, Any]]]] = (
        collections.defaultdict(list)
    )
    earliest_rows = [
        row for row in target_rows if abs(_float(row, "start_time_us") - start) <= tol_us
    ]
    for target in earliest_rows:
        core = int(target["core_id"])
        window_start = max(data_ready_us, _float(target, "dispatch_time_us"))
        if start <= window_start + tol_us:
            continue
        for other in analysis.rows_by_core.get(core, []):
            other_task = _task_id(other["task_id"])
            if other_task == task:
                continue
            other_end = _float(other, "end_time_us")
            if other_end + tol_us < start:
                continue
            overlap_start = max(window_start, _float(other, "start_time_us"))
            overlap_end = min(start, other_end)
            if overlap_end <= overlap_start:
                continue
            overlap_intervals[(other_task, core)].append((overlap_start, overlap_end, other))

    blockers: list[tuple[str, str, int, float]] = []
    for (other_task, core), intervals in overlap_intervals.items():
        ordered = sorted((left, right) for left, right, _row in intervals)
        merged: list[list[float]] = []
        for left, right in ordered:
            if not merged or left > merged[-1][1]:
                merged.append([left, right])
            else:
                merged[-1][1] = max(merged[-1][1], right)
        overlap = sum(right - left for left, right in merged)
        blockers.append(
            (other_task, _logical_name(other_task, analysis, intervals[0][2]), core, overlap)
        )
    return sorted(blockers, key=lambda item: (-item[3], item[2], item[0]))


def _resource_saturation(
    analysis: RunAnalysis,
    task: str,
    window_start_us: float,
    window_end_us: float,
) -> tuple[bool, list[str], str]:
    """Prove two-descriptor-slot saturation on every compatible AIC or AIV core."""
    if window_end_us <= window_start_us:
        return False, [], ""
    target_types = {str(row["core_type"]) for row in analysis.rows_by_task.get(task, [])}
    if len(target_types) != 1:
        return False, [], "MIX/heterogeneous launch requires cluster-aware manual inspection"
    target_type = next(iter(target_types))
    compatible = [
        core for core, core_type in enumerate(analysis.core_types) if str(core_type) == target_type
    ]
    if not compatible:
        return False, [], f"no compatible {target_type} core inventory in metadata"

    intervals: dict[int, list[tuple[float, float, str]]] = collections.defaultdict(list)
    boundaries = {window_start_us, window_end_us}
    for row in analysis.rows:
        if str(row["core_type"]) != target_type:
            continue
        dispatch = _float(row, "dispatch_time_us")
        finish = _float(row, "finish_time_us")
        if finish <= window_start_us or dispatch >= window_end_us:
            continue
        core = int(row["core_id"])
        left = max(dispatch, window_start_us)
        right = min(finish, window_end_us)
        intervals[core].append((left, right, _task_id(row["task_id"])))
        boundaries.update((left, right))

    blockers: set[str] = set()
    ordered = sorted(boundaries)
    for left, right in zip(ordered, ordered[1:]):
        if right <= left:
            continue
        probe = (left + right) / 2
        for core in compatible:
            active = [
                other
                for interval_start, interval_end, other in intervals.get(core, [])
                if interval_start <= probe < interval_end
            ]
            if len(active) < 2:  # one running plus one pending descriptor slot
                return False, [], (
                    f"full-engine saturation not proven: {target_type} core {core} had a free "
                    f"descriptor slot during {left:.3f}–{right:.3f} µs"
                )
            blockers.update(active)
    blockers.discard(task)
    evidence = (
        f"all {len(compatible)} compatible {target_type} cores had both running/pending "
        f"descriptor slots occupied throughout {window_start_us:.3f}–{window_end_us:.3f} µs"
    )
    return True, sorted(blockers, key=int), evidence


def _dispatch_diagnostics(
    analysis: RunAnalysis,
    snail_indices: list[int],
    gap_threshold_us: float,
    tol: int,
) -> list[str]:
    if not snail_indices:
        return ["No path task has a gap strictly greater than the threshold."]

    starts = _aggregate_min(analysis.rows_by_task, "start_time_us")
    ends = _aggregate_max(analysis.rows_by_task, "end_time_us")
    dispatches = _aggregate_min(analysis.rows_by_task, "dispatch_time_us")
    finishes = _aggregate_max(analysis.rows_by_task, "finish_time_us")
    tol_us = tol / analysis.graph.freq * 1e6
    lines = [
        "| 🐌 task | data ready µs | last predecessor FIN µs | dispatch µs | start µs | attribution |",
        "|---|---:|---:|---:|---:|---|",
    ]
    named_blockers: list[tuple[str, str, str, str, str]] = []
    unattributed_scheduler_delays: list[tuple[str, float]] = []

    for index in snail_indices:
        segment = analysis.result.segments[index]
        task = segment.task
        predecessor_ids = [pred for pred in analysis.preds.get(task, set()) if pred in ends]
        data_ready = max((ends[pred] for pred in predecessor_ids), default=dispatches[task])
        observed_ready = max((finishes[pred] for pred in predecessor_ids), default=dispatches[task])
        dispatch = dispatches[task]
        start = starts[task]
        early_status, _, _ = _observed_early(task, analysis, tol_us)
        is_early = early_status in {"full", "partial"}
        policy_blockers = _unflagged_predecessors(task, analysis)
        scheduler_ready = max(data_ready, observed_ready)
        delay = max(0.0, dispatch - scheduler_ready)

        attribution: list[str] = []
        if observed_ready > data_ready + tol_us:
            attribution.append(f"producer end→FIN {observed_ready - data_ready:.3f} µs")
        if is_early:
            attribution.append("actually early-dispatched; no predecessor blocked dispatch")
        elif early_status == "mismatch":
            attribution.append("dispatch timing precedes predecessor FIN but dependency policy is ineligible")
        elif early_status == "unverifiable":
            attribution.append("early-dispatch proof is unverifiable because a direct producer lacks timing")
        elif policy_blockers:
            attribution.append("not early-eligible: unflagged direct producer(s)")
            for blocker in policy_blockers:
                named_blockers.append(
                    (
                        task,
                        blocker,
                        _logical_name(blocker, analysis),
                        "early-dispatch policy",
                        "direct producer lacks `early_dispatch=true`",
                    )
                )
        if delay > gap_threshold_us:
            attribution.append(f"post-FIN ready→dispatch scheduler delay {delay:.3f} µs")
            saturated, resource_tasks, resource_evidence = _resource_saturation(
                analysis, task, scheduler_ready, dispatch
            )
            if saturated:
                attribution.append("dispatch resource saturation proven")
                for blocker in resource_tasks:
                    named_blockers.append(
                        (
                            task,
                            blocker,
                            _logical_name(blocker, analysis),
                            "dispatch resource",
                            resource_evidence,
                        )
                    )
            else:
                attribution.append(f"dispatch resource blocker unproven: {resource_evidence}")
                unattributed_scheduler_delays.append((task, delay))
        if start > dispatch + tol_us:
            attribution.append(f"dispatch→start {start - dispatch:.3f} µs (post-dispatch)")
        if not attribution:
            attribution.append("no dispatch delay proven")

        for blocker, name, core, overlap in _same_core_blockers(
            task, analysis, tol_us, data_ready
        ):
            named_blockers.append(
                (
                    task,
                    blocker,
                    name,
                    "post-dispatch same-core",
                    f"core {core} occupied for {overlap:.3f} µs and remained occupied "
                    "until the task's earliest start (within tolerance) inside the "
                    "row-local max(data-ready, dispatch)→start window",
                )
            )

        lines.append(
            f"| `{task}` {_md(segment.name)} | {data_ready:.3f} | {observed_ready:.3f} | "
            f"{dispatch:.3f} | {start:.3f} | {_md('; '.join(attribution))} |"
        )

    lines.extend(["", "### Named blocker evidence", ""])
    if named_blockers:
        lines.extend(
            [
                "| 🐌 task | blocker task | blocker operator | evidence type | evidence |",
                "|---|---|---|---|---|",
            ]
        )
        for task, blocker, name, kind, evidence in sorted(set(named_blockers)):
            lines.append(f"| `{task}` | `{blocker}` | {_md(name)} | {_md(kind)} | {_md(evidence)} |")
    else:
        lines.append("No named task blocker is proven by the captured task-level evidence.")

    if unattributed_scheduler_delays:
        lines.extend(
            [
                "",
                "> The post-FIN ready→dispatch rows above prove scheduler delay, but no named dispatch blocker "
                "was proven for those rows. Level-4 scheduler phase records do not carry ordinary task IDs; a "
                "causal task is named only when every compatible core's running and pending descriptor slots "
                "are saturated throughout the interval.",
            ]
        )
    return lines


def _render(
    analyses: list[RunAnalysis],
    selected_rank: str,
    gap_threshold_us: float,
    tol: int,
    requested_operator: str | None,
) -> str:
    grouped: dict[str, list[RunAnalysis]] = collections.defaultdict(list)
    for analysis in analyses:
        grouped[analysis.rank].append(analysis)
    totals = {rank: sum(item.elapsed_us for item in items) for rank, items in grouped.items()}
    aicore_totals = {
        rank: sum(_us(item.result.makespan, item.result.freq) for item in items)
        for rank, items in grouped.items()
    }

    lines = [
        "# Operator critical-path report",
        "",
        f"- Selected rank/device: **`{selected_rank}`** (minimum summed dispatch→finish elapsed)",
        f"- Selected operator elapsed: **{totals[selected_rank]:.3f} µs**",
        f"- Selected AICore makespan coverage: **{aicore_totals[selected_rank]:.3f} µs**",
        f"- Gap marker: **🐌 when gap > {gap_threshold_us:.3f} µs**",
        "- Early marker: **⭐ only when structurally eligible and dispatch precedes the last predecessor FIN**",
        "",
        "## Rank comparison",
        "",
        "| rank/device | dispatches | dispatch→finish elapsed µs | AICore makespan µs | selected |",
        "|---|---:|---:|---:|---|",
    ]
    for rank in sorted(grouped, key=_rank_sort_key):
        lines.append(
            f"| `{rank}` | {len(grouped[rank])} | {totals[rank]:.3f} | "
            f"{aicore_totals[rank]:.3f} | {'✓' if rank == selected_rank else ''} |"
        )

    selected = sorted(grouped[selected_rank], key=_dispatch_sort_key)
    for dispatch_index, analysis in enumerate(selected):
        title = analysis.program or requested_operator or analysis.directory.name
        identity = (
            f"dispatch_program.json = `{analysis.program}`"
            if analysis.program is not None
            else "unambiguous direct dispatch (no dispatch_program.json)"
        )
        lines.extend(
            [
                "",
                f"## Dispatch {dispatch_index}: `{_md(title)}`",
                "",
                f"- Artifact directory: `{analysis.directory}`",
                f"- Program identity: {identity}",
                f"- Dispatch→finish elapsed: **{analysis.elapsed_us:.3f} µs**",
                f"- AICore makespan: **{_us(analysis.result.makespan, analysis.result.freq):.3f} µs**",
                f"- Static CPM cross-check: **{_us(analysis.result.cpm_len, analysis.result.freq):.3f} µs**",
                f"- Primary table: **Observed critical path**, {len(analysis.result.segments)} tasks",
                "- `task wall span` can overlap earlier path work; only the Observed compute contribution plus "
                "canonical stall contributions tile the AICore makespan.",
                "",
                "### Observed path",
                "",
            ]
        )
        table, snail_indices = _path_table(analysis, gap_threshold_us, tol)
        lines.extend(table)
        lines.extend(["", "### Dispatch investigation for 🐌 tasks", ""])
        lines.extend(_dispatch_diagnostics(analysis, snail_indices, gap_threshold_us, tol))

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Dispatch→finish is the closest per-rank elapsed available in the swimlane; it is not the DFX-off benchmark.",
            "- The critical-path makespan covers first AICore start through last AICore end, excluding host/orchestrator front "
            "time and AICPU/host tail time.",
            "- `post-dispatch same-core` is restricted to core row(s) that realize the task-global earliest start "
            "and occupancy continuing through the row-local effective-ready→start window. It proves start "
            "contention there, not that all cores were unable to accept dispatch.",
            "- MIX, SPMD, and sync-start capacity may require cluster-aware manual inspection; the report leaves a "
            "scheduler delay unattributed unless full compatible-engine descriptor saturation is proven.",
            "- Level-4 collection has observer cost. Keep the unprofiled benchmark as the production performance headline.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a gap/early-dispatch/blocker report from a level-4 L2 swimlane run."
    )
    parser.add_argument("build_dir", type=Path, help="Build/run directory containing dfx_outputs artifacts")
    parser.add_argument("--operator", help="Filter distributed dispatch_program.json values")
    parser.add_argument("--rank", help="Select this rank instead of the fastest rank")
    parser.add_argument("--gap-threshold-us", type=float, default=1.0)
    parser.add_argument("--tol", type=int, default=2, help="Critical-path timestamp tolerance in clock ticks")
    parser.add_argument("-o", "--output", type=Path, help="Write Markdown here instead of stdout")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = args.build_dir.expanduser().resolve()
    if not root.exists():
        print(f"error: path not found: {root}", file=sys.stderr)
        return 2
    if args.gap_threshold_us < 0:
        print("error: --gap-threshold-us must be non-negative", file=sys.stderr)
        return 2
    if args.tol < 0:
        print("error: --tol must be non-negative", file=sys.stderr)
        return 2
    if root.is_file():
        root = root.parent

    record_dirs = _record_directories(root)
    try:
        programs = {directory: _program(directory) for directory in record_dirs}
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: invalid dispatch_program.json: {exc}", file=sys.stderr)
        return 2
    known_programs = {program for program in programs.values() if program is not None}
    dispatches_per_rank = collections.Counter(_rank_label(directory) for directory in record_dirs)
    if known_programs and any(program is None for program in programs.values()):
        print("error: some dispatches are missing dispatch_program.json; refusing mixed program identity", file=sys.stderr)
        return 2
    if any(count > 1 for count in dispatches_per_rank.values()) and not known_programs:
        print("error: multiple dispatches per rank require dispatch_program.json", file=sys.stderr)
        return 2
    if not args.operator and len(known_programs) > 1:
        choices = ", ".join(sorted(known_programs))
        print(f"error: multiple dispatch programs found; select one with --operator ({choices})", file=sys.stderr)
        return 2
    if args.operator and known_programs:
        matched_programs = {
            program for program in known_programs if _operator_matches(args.operator, program)
        }
        if not matched_programs:
            choices = ", ".join(sorted(known_programs))
            print(
                f"error: no exact dispatch program matches {args.operator!r}; available: {choices}",
                file=sys.stderr,
            )
            return 2
        if len(matched_programs) > 1:
            choices = ", ".join(sorted(matched_programs))
            print(
                f"error: {args.operator!r} matches multiple dispatch programs by stem ({choices}); "
                "use the exact program identity",
                file=sys.stderr,
            )
            return 2
        record_dirs = [
            directory for directory, program in programs.items() if program in matched_programs
        ]
    if args.rank:
        record_dirs = [directory for directory in record_dirs if _rank_label(directory) == args.rank]
    if not record_dirs:
        print("error: no matching swimlane record directories found", file=sys.stderr)
        return 2

    dispatches_per_rank = collections.Counter(_rank_label(directory) for directory in record_dirs)
    if not args.rank and len(set(dispatches_per_rank.values())) > 1:
        detail = ", ".join(f"{rank}={count}" for rank, count in sorted(dispatches_per_rank.items()))
        print(
            "error: matching dispatch counts differ across ranks; refusing an unequal fast-rank comparison "
            f"({detail})",
            file=sys.stderr,
        )
        return 2

    incomplete = []
    for directory in record_dirs:
        missing = []
        if not (directory / "deps.json").is_file():
            missing.append("deps.json")
        if not list(directory.glob("name_map*.json")):
            missing.append("name_map*.json")
        if missing:
            incomplete.append(f"{directory}: {', '.join(missing)}")
    if incomplete:
        print("error: refusing to compare ranks with incomplete critical-path artifacts:", file=sys.stderr)
        for item in incomplete:
            print(f"  {item}", file=sys.stderr)
        return 2

    try:
        analyses = [_build_analysis(directory, root, args.tol) for directory in record_dirs]
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    grouped: dict[str, list[RunAnalysis]] = collections.defaultdict(list)
    for analysis in analyses:
        grouped[analysis.rank].append(analysis)
    if args.rank:
        if args.rank not in grouped:
            print(f"error: selected rank not found: {args.rank}", file=sys.stderr)
            return 2
        selected_rank = args.rank
    else:
        selected_rank = min(
            grouped,
            key=lambda rank: (sum(item.elapsed_us for item in grouped[rank]), _rank_sort_key(rank)),
        )

    report = _render(analyses, selected_rank, args.gap_threshold_us, args.tol, args.operator)
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(f"Report written to: {output}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
