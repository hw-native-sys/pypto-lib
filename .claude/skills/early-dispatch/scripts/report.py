# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Inspect and compare actual early dispatch in level-4 L2 swimlane artifacts.

The source-side ``allow_early_resolve`` flag belongs to a producer. A consumer
is considered actually early-dispatched only when every direct producer is
eligible and at least one of the consumer's AICore blocks was dispatched before
the latest producer FIN timestamp.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from simpler_setup.tools.swimlane_converter import read_perf_data
except ImportError as exc:  # pragma: no cover - environment preflight
    raise SystemExit(
        "Could not import simpler_setup. Activate the pypto-lib environment first "
        "(for this worktree: source temp/set_env.sh)."
    ) from exc


RECORD_NAMES = ("l2_swimlane_records.json", "l2_perf_records.json")
RANK_RE = re.compile(r"rank\d+$")
BASELINE_VERSION = 2


@dataclass
class Timing:
    start_us: float
    end_us: float
    dispatch_us: float
    latest_dispatch_us: float
    finish_us: float
    blocks: int


@dataclass
class TargetSnapshot:
    dispatch: str
    dispatch_index: int
    dispatch_program: str | None
    occurrence: int
    task_id: str
    names: list[str]
    task_signature: str
    predecessor_ids: list[str]
    predecessor_names: list[str]
    predecessor_signatures: list[str]
    submit_start_us: float | None
    submit_end_us: float | None
    earliest_dispatch_us: float
    earliest_start_us: float
    latest_predecessor_end_us: float | None
    latest_predecessor_finish_us: float | None
    start_gap_us: float | None
    post_fin_gap_us: float | None
    early_lead_us: float | None
    early_blocks: int
    total_blocks: int
    structural_status: str
    observed_status: str
    unflagged_predecessors: list[str]
    untimed_predecessors: list[str]
    blocker_summary: list[str]


@dataclass
class Run:
    directory: Path
    rank: str
    program: str | None
    rows: list[dict[str, Any]]
    rows_by_task: dict[str, list[dict[str, Any]]]
    timings: dict[str, Timing]
    task_table: dict[str, dict[str, Any]]
    predecessors: dict[str, set[str]]
    names: dict[str, list[str]]
    scheduler_phases: list[dict[str, Any]]
    orchestrator_phases: list[dict[str, Any]]
    core_types: list[str]
    elapsed_us: float
    tol_us: float


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_id(value: Any) -> str:
    return str(int(value))


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {result}")
    return result


def _records_file(directory: Path) -> Path | None:
    for name in RECORD_NAMES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _record_directories(root: Path) -> list[Path]:
    if root.is_file():
        return [root.parent] if root.name in RECORD_NAMES else []
    search_root = root / "dfx_outputs" if (root / "dfx_outputs").is_dir() else root
    directories: set[Path] = set()
    for name in RECORD_NAMES:
        directories.update(path.parent for path in search_root.rglob(name))
    return sorted(directories)


def _program(directory: Path) -> str | None:
    marker = directory / "dispatch_program.json"
    if not marker.is_file():
        return None
    data = _read_json(marker)
    value = data.get("program") if isinstance(data, dict) else None
    return str(value) if value is not None else None


def _operator_matches(query: str, program: str) -> bool:
    needle = query.casefold()
    candidate = program.casefold()
    return candidate == needle or Path(program).stem.casefold() == Path(query).stem.casefold()


def _rank_label(directory: Path) -> str:
    for part in reversed(directory.parts):
        if RANK_RE.fullmatch(part):
            return part
    return "single"


def _rank_sort_key(rank: str) -> tuple[int, str]:
    match = re.fullmatch(r"rank(\d+)", rank)
    return (int(match.group(1)), rank) if match else (-1, rank)


def _dispatch_sort_key(run: Run) -> tuple[int, str]:
    match = re.fullmatch(r"d(\d+)", run.directory.name)
    return (int(match.group(1)), str(run.directory)) if match else (-1, str(run.directory))


def _name_map(directory: Path) -> dict[str, str]:
    candidates = sorted(
        directory.glob("name_map*.json"),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    if not candidates:
        raise ValueError(f"{directory}: missing name_map*.json")
    data = _read_json(candidates[-1])
    if isinstance(data, dict) and isinstance(data.get("callable_id_to_name"), dict):
        data = data["callable_id_to_name"]
    if not isinstance(data, dict):
        raise ValueError(f"{candidates[-1]}: expected a callable-id mapping")
    return {str(key): str(value) for key, value in data.items()}


def _task_names(task: dict[str, Any], names: dict[str, str]) -> list[str]:
    result: list[str] = []
    for kernel_id in task.get("kernel_ids") or []:
        try:
            numeric = int(kernel_id)
        except (TypeError, ValueError):
            continue
        if numeric < 0:
            continue
        name = names.get(str(numeric), f"cid{numeric}")
        if name not in result:
            result.append(name)
    return result or ["unknown"]


def _aggregate_timing(rows: list[dict[str, Any]], task: str) -> Timing:
    starts = [_finite(row["start_time_us"], f"{task}.start_time_us") for row in rows]
    ends = [_finite(row["end_time_us"], f"{task}.end_time_us") for row in rows]
    dispatches = [_finite(row["dispatch_time_us"], f"{task}.dispatch_time_us") for row in rows]
    finishes = [_finite(row["finish_time_us"], f"{task}.finish_time_us") for row in rows]
    if min(dispatches) < 0 or min(finishes) <= 0:
        raise ValueError(f"task {task}: level-4 AICPU timing is missing")
    return Timing(
        start_us=min(starts),
        end_us=max(ends),
        dispatch_us=min(dispatches),
        latest_dispatch_us=max(dispatches),
        finish_us=max(finishes),
        blocks=len(rows),
    )


def _build_run(directory: Path) -> Run:
    records = _records_file(directory)
    if records is None:
        raise ValueError(f"{directory}: missing swimlane records")
    raw = _read_json(records)
    level = raw.get("l2_swimlane_level") if isinstance(raw, dict) else None
    if level != 4:
        raise ValueError(f"{records}: expected l2_swimlane_level=4, got {level!r}")
    frequency = int((raw.get("metadata") or {}).get("clock_freq_hz") or 0)
    if frequency <= 0:
        raise ValueError(f"{records}: invalid clock frequency {frequency}")

    joined = read_perf_data(records)
    rows = joined.get("tasks") if isinstance(joined, dict) else None
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
    rows_by_task: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"{records}: task row missing {', '.join(missing)}")
        rows_by_task[_task_id(row["task_id"])].append(row)
    timings = {task: _aggregate_timing(task_rows, task) for task, task_rows in rows_by_task.items()}

    scheduler_phases = [
        record
        for thread in joined.get("aicpu_scheduler_phases", [])
        for record in thread
        if isinstance(record, dict)
    ]
    orchestrator_phases = [
        record
        for thread in joined.get("aicpu_orchestrator_phases", [])
        for record in thread
        if isinstance(record, dict) and record.get("task_id") is not None
    ]

    deps_path = directory / "deps.json"
    deps = _read_json(deps_path)
    if not isinstance(deps, dict):
        raise ValueError(f"{deps_path}: expected a JSON object")
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
            raise ValueError(f"{deps_path}: task {task} has invalid kernel_ids/block_num") from exc
        expected_rows = active_slots * logical_blocks
        actual_rows = len(rows_by_task.get(task, []))
        if actual_rows != expected_rows:
            raise ValueError(
                f"{records}: incomplete task {task} rows "
                f"(joined={actual_rows}, expected={logical_blocks}×{active_slots}={expected_rows})"
            )
    predecessors: dict[str, set[str]] = collections.defaultdict(set)
    for edge in deps.get("edges", []):
        if not isinstance(edge, dict) or edge.get("pred") is None or edge.get("succ") is None:
            continue
        pred, succ = _task_id(edge["pred"]), _task_id(edge["succ"])
        if pred != succ:
            predecessors[succ].add(pred)

    callable_names = _name_map(directory)
    names = {task: _task_names(info, callable_names) for task, info in task_table.items()}
    elapsed_us = max(t.finish_us for t in timings.values()) - min(t.dispatch_us for t in timings.values())
    return Run(
        directory=directory,
        rank=_rank_label(directory),
        program=_program(directory),
        rows=rows,
        rows_by_task=dict(rows_by_task),
        timings=timings,
        task_table=task_table,
        predecessors=dict(predecessors),
        names=names,
        scheduler_phases=scheduler_phases,
        orchestrator_phases=orchestrator_phases,
        core_types=list((raw.get("metadata") or {}).get("core_types") or []),
        elapsed_us=elapsed_us,
        tol_us=2 / frequency * 1e6,
    )


def _matching_tasks(
    run: Run,
    query: str,
    task_id: str | None,
    occurrence: int | None,
) -> list[tuple[int, str]]:
    exact = sorted(
        (
            task
            for task, names in run.names.items()
            if task in run.rows_by_task and any(name.casefold() == query.casefold() for name in names)
        ),
        key=int,
    )
    candidates = exact
    if not candidates:
        expected_mix_names = {f"{query.casefold()}_aic", f"{query.casefold()}_aiv"}
        candidates = sorted(
            (
                task
                for task, names in run.names.items()
                if task in run.rows_by_task and {name.casefold() for name in names} == expected_mix_names
            ),
            key=int,
        )
    if not candidates:
        available = sorted(
            {name for task in run.rows_by_task for name in run.names.get(task, []) if name != "unknown"}
        )
        close = ", ".join(name for name in available if query.casefold() in name.casefold()) or ", ".join(
            available[:20]
        )
        raise ValueError(
            f"{run.directory}: no exact compiled task or safe MIX alias matches {query!r}; "
            f"available examples: {close}"
        )

    if task_id is not None:
        normalized = _task_id(task_id)
        if normalized not in candidates:
            raise ValueError(
                f"{run.directory}: task id {normalized} is not a timed occurrence of target {query!r}"
            )
        index = candidates.index(normalized)
        return [(index, normalized)]
    if occurrence is not None:
        if occurrence < 0 or occurrence >= len(candidates):
            raise ValueError(
                f"{run.directory}: occurrence {occurrence} is outside the target range 0..{len(candidates) - 1}"
            )
        return [(occurrence, candidates[occurrence])]
    return list(enumerate(candidates))


def _is_alloc(task: str, run: Run) -> bool:
    return task not in run.task_table


def _flagged(task: str, run: Run) -> bool:
    return _is_alloc(task, run) or bool(run.task_table[task].get("early_dispatch"))


def _task_signature(run: Run, task: str) -> str:
    if _is_alloc(task, run):
        graph_tasks = set(run.timings)
        graph_tasks.update(
            predecessor for predecessors in run.predecessors.values() for predecessor in predecessors
        )
        allocations = sorted((candidate for candidate in graph_tasks if _is_alloc(candidate, run)), key=int)
        occurrence = allocations.index(task)
        rows = run.rows_by_task.get(task, [])
        func_ids = ",".join(sorted({str(row.get("func_id", "?")) for row in rows}))
        core_types = ",".join(sorted({str(row.get("core_type", "?")) for row in rows}))
        return f"alloc#{occurrence}|func={func_ids}|cores={core_types}|rows={len(rows)}"

    info = run.task_table[task]
    active_slots = sum(int(kernel_id) >= 0 for kernel_id in (info.get("kernel_ids") or []))
    logical_blocks = int(info.get("block_num"))
    names = run.names.get(task, ["unknown"])
    peers = sorted(
        (candidate for candidate, candidate_names in run.names.items() if candidate_names == names),
        key=int,
    )
    occurrence = peers.index(task)
    rows = len(run.rows_by_task.get(task, []))
    return f"{'/'.join(names)}#{occurrence}|logical={logical_blocks}|slots={active_slots}|rows={rows}"


def _predecessor_signatures(run: Run, task: str) -> list[str]:
    return sorted(_task_signature(run, pred) for pred in run.predecessors.get(task, set()))


def _scheduler_evidence(run: Run, start_us: float, end_us: float) -> tuple[bool, int]:
    ready_backlog = False
    staged = 0
    for phase in run.scheduler_phases:
        phase_start = _finite(phase.get("start_time_us", 0), "scheduler phase start")
        phase_end = _finite(phase.get("end_time_us", 0), "scheduler phase end")
        if phase_end < start_us or phase_start > end_us:
            continue
        if phase.get("phase") == "early_dispatch":
            staged += int(phase.get("tasks_processed") or 0)
        for field in ("shared_at_start", "shared_at_end"):
            depths = phase.get(field)
            if isinstance(depths, list) and any(int(value) > 0 for value in depths):
                ready_backlog = True
    return ready_backlog, staged


def _resource_saturation(
    run: Run,
    task: str,
    window_start_us: float,
    window_end_us: float,
) -> tuple[bool, list[str]]:
    """Prove two-slot saturation on every compatible AIC or AIV core."""
    if window_end_us <= window_start_us:
        return False, []
    target_types = {str(row["core_type"]) for row in run.rows_by_task[task]}
    if len(target_types) != 1:
        return False, []  # MIX/heterogeneous placement needs cluster-aware inspection.
    target_type = next(iter(target_types))
    compatible = [core for core, core_type in enumerate(run.core_types) if core_type == target_type]
    if not compatible:
        return False, []

    intervals: dict[int, list[tuple[float, float, str]]] = collections.defaultdict(list)
    boundaries = {window_start_us, window_end_us}
    for row in run.rows:
        if str(row["core_type"]) != target_type:
            continue
        dispatch = _finite(row["dispatch_time_us"], "resource dispatch")
        finish = _finite(row["finish_time_us"], "resource finish")
        if finish <= window_start_us or dispatch >= window_end_us:
            continue
        core = int(row["core_id"])
        other = _task_id(row["task_id"])
        start = max(dispatch, window_start_us)
        end = min(finish, window_end_us)
        intervals[core].append((start, end, other))
        boundaries.update((start, end))

    ordered = sorted(boundaries)
    if len(ordered) < 2:
        return False, []
    blockers: set[str] = set()
    for left, right in zip(ordered, ordered[1:]):
        if right <= left:
            continue
        probe = (left + right) / 2
        for core in compatible:
            active = [other for start, end, other in intervals.get(core, []) if start <= probe < end]
            if len(active) < 2:  # one running plus one pending descriptor slot
                return False, []
            blockers.update(active)
    blockers.discard(task)
    return True, sorted(blockers, key=int)


def _snapshot(run: Run, task: str, occurrence: int, dispatch_index: int) -> TargetSnapshot:
    target = run.timings[task]
    predecessors = sorted(run.predecessors.get(task, set()), key=int)
    non_alloc = [pred for pred in predecessors if not _is_alloc(pred, run)]
    unflagged = [pred for pred in non_alloc if not _flagged(pred, run)]
    untimed = [pred for pred in non_alloc if pred not in run.rows_by_task]
    timed = [pred for pred in predecessors if pred in run.rows_by_task]

    structural_status = "root"
    if predecessors:
        if unflagged:
            structural_status = "policy-blocked"
        elif non_alloc:
            structural_status = "eligible"
        else:
            structural_status = "alloc-only"

    latest_end = None if untimed else max((run.timings[pred].end_us for pred in timed), default=None)
    latest_finish = None if untimed else max((run.timings[pred].finish_us for pred in timed), default=None)
    start_gap = target.start_us - latest_end if latest_end is not None else None
    post_fin_gap = target.start_us - latest_finish if latest_finish is not None else None
    early_lead = latest_finish - target.dispatch_us if latest_finish is not None else None

    early_blocks = 0
    if latest_finish is not None and not untimed:
        early_blocks = sum(
            _finite(row["dispatch_time_us"], f"{task}.dispatch_time_us") + run.tol_us < latest_finish
            for row in run.rows_by_task[task]
        )
    raw_early = early_blocks > 0
    if untimed or latest_finish is None:
        observed_status = "unverifiable"
    elif structural_status == "eligible":
        if early_blocks == target.blocks:
            observed_status = "full"
        elif raw_early:
            observed_status = "partial"
        else:
            observed_status = "none"
    elif raw_early:
        observed_status = "mismatch"
    else:
        observed_status = "none"

    submit_records = [record for record in run.orchestrator_phases if _task_id(record["task_id"]) == task]
    submit_start = (
        min(_finite(record["start_time_us"], f"{task}.submit_start") for record in submit_records)
        if submit_records
        else None
    )
    submit_end = (
        max(_finite(record["end_time_us"], f"{task}.submit_end") for record in submit_records)
        if submit_records
        else None
    )

    blockers: list[str] = []
    if unflagged:
        details = ", ".join(f"{pred} ({'/'.join(run.names.get(pred, ['unknown']))})" for pred in unflagged)
        blockers.append(f"early-dispatch policy: unflagged direct producer(s): {details}")
    elif untimed:
        blockers.append(f"unverifiable predecessor timing: {', '.join(untimed)}")
    elif observed_status in {"full", "partial"}:
        blockers.append("none: scheduler timestamps prove actual early dispatch")
    elif structural_status == "eligible" and latest_finish is not None:
        producer_dispatches = [
            run.timings[pred].latest_dispatch_us for pred in non_alloc if pred in run.rows_by_task
        ]
        candidate_lower_bound = max(producer_dispatches, default=latest_finish)
        if submit_start is not None:
            candidate_lower_bound = max(candidate_lower_bound, submit_start)
        opportunity = latest_finish - candidate_lower_bound
        if submit_start is not None and submit_start + run.tol_us >= latest_finish:
            blockers.append(
                f"target submission started after the predecessor FIN window closed "
                f"({submit_start:.3f} ≥ {latest_finish:.3f} µs)"
            )
        elif opportunity <= run.tol_us:
            blockers.append(
                f"no measurable pre-stage window after producer publication ({opportunity:.3f} µs)"
            )
        else:
            saturated, resource_tasks = _resource_saturation(run, task, candidate_lower_bound, latest_finish)
            ready_backlog, staged_elsewhere = _scheduler_evidence(run, candidate_lower_bound, latest_finish)
            if saturated:
                details = ", ".join(
                    f"{other} ({'/'.join(run.names.get(other, ['unknown']))})"
                    for other in resource_tasks[:16]
                )
                suffix = "" if len(resource_tasks) <= 16 else f" and {len(resource_tasks) - 16} more"
                blockers.append(f"proven AICore running+pending slot saturation: {details}{suffix}")
            if ready_backlog:
                blockers.append(
                    "candidate observation: ordinary ready-queue backlog was sampled in the opportunity window; "
                    "normal ready work has priority, but the aggregate phase has no target task ID"
                )
            if staged_elsewhere:
                blockers.append(
                    f"context only: aggregate scheduler phases staged {staged_elsewhere} early-dispatch physical "
                    "row(s) "
                    "in overlapping intervals; those phases do not identify this target"
                )
            if not blockers:
                blockers.append(
                    "unattributed scheduler/launch constraint; inspect predicate, sync_start, PMU, resource shape, "
                    "and scheduler lanes because phase records do not carry ordinary task IDs"
                )
    else:
        blockers.append(f"not structurally eligible ({structural_status})")

    return TargetSnapshot(
        dispatch=run.directory.name,
        dispatch_index=dispatch_index,
        dispatch_program=run.program,
        occurrence=occurrence,
        task_id=task,
        names=run.names.get(task, ["unknown"]),
        task_signature=_task_signature(run, task),
        predecessor_ids=predecessors,
        predecessor_names=sorted(
            "alloc" if _is_alloc(pred, run) else "/".join(run.names.get(pred, ["unknown"]))
            for pred in predecessors
        ),
        predecessor_signatures=_predecessor_signatures(run, task),
        submit_start_us=submit_start,
        submit_end_us=submit_end,
        earliest_dispatch_us=target.dispatch_us,
        earliest_start_us=target.start_us,
        latest_predecessor_end_us=latest_end,
        latest_predecessor_finish_us=latest_finish,
        start_gap_us=start_gap,
        post_fin_gap_us=post_fin_gap,
        early_lead_us=early_lead,
        early_blocks=early_blocks,
        total_blocks=target.blocks,
        structural_status=structural_status,
        observed_status=observed_status,
        unflagged_predecessors=unflagged,
        untimed_predecessors=untimed,
        blocker_summary=blockers,
    )


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _run_payload(
    root: Path,
    target: str,
    operator: str | None,
    rank: str | None,
    task_id: str | None,
    occurrence: int | None,
) -> dict[str, Any]:
    directories = _record_directories(root)
    programs = {directory: _program(directory) for directory in directories}
    known_programs = {program for program in programs.values() if program is not None}
    dispatches_per_rank = collections.Counter(_rank_label(directory) for directory in directories)
    if known_programs and any(program is None for program in programs.values()):
        raise ValueError("some dispatches are missing dispatch_program.json; refusing mixed program identity")
    if any(count > 1 for count in dispatches_per_rank.values()) and any(
        program is None for program in programs.values()
    ):
        raise ValueError("multiple dispatches per rank require dispatch_program.json")
    if not operator and len(known_programs) > 1:
        choices = ", ".join(sorted(known_programs))
        raise ValueError(f"multiple dispatch programs found; select one with --operator ({choices})")
    if operator:
        if not known_programs:
            raise ValueError("--operator cannot be verified because dispatch_program.json is missing")
        matched_programs = {
            program for program in known_programs if _operator_matches(operator, program)
        }
        if not matched_programs:
            raise ValueError(
                f"no exact dispatch program matches {operator!r}; available: "
                + ", ".join(sorted(known_programs))
            )
        if len(matched_programs) > 1:
            raise ValueError(
                f"{operator!r} matches multiple dispatch programs by stem "
                f"({', '.join(sorted(matched_programs))}); use the exact program identity"
            )
        directories = [
            directory
            for directory, program in programs.items()
            if program in matched_programs
        ]
    if rank:
        directories = [directory for directory in directories if _rank_label(directory) == rank]
    if not directories:
        raise ValueError("no matching level-4 swimlane directories found")

    incomplete: list[str] = []
    for directory in directories:
        missing = []
        if not (directory / "deps.json").is_file():
            missing.append("deps.json")
        if not list(directory.glob("name_map*.json")):
            missing.append("name_map*.json")
        if missing:
            incomplete.append(f"{directory}: {', '.join(missing)}")
    if incomplete:
        raise ValueError("incomplete artifact set(s): " + "; ".join(incomplete))

    runs = [_build_run(directory) for directory in directories]
    grouped: dict[str, list[Run]] = collections.defaultdict(list)
    for run in runs:
        grouped[run.rank].append(run)
    counts = {candidate_rank: len(items) for candidate_rank, items in grouped.items()}
    if not rank and len(set(counts.values())) > 1:
        detail = ", ".join(f"{candidate_rank}={count}" for candidate_rank, count in sorted(counts.items()))
        raise ValueError(f"matching dispatch counts differ across ranks ({detail})")
    selected_rank = rank or min(
        grouped,
        key=lambda candidate_rank: (
            sum(item.elapsed_us for item in grouped[candidate_rank]),
            _rank_sort_key(candidate_rank),
        ),
    )
    if selected_rank not in grouped:
        raise ValueError(f"selected rank not found: {selected_rank}")

    selected_runs = sorted(grouped[selected_rank], key=_dispatch_sort_key)
    dispatch_identities = collections.Counter(
        (run.program or "", run.directory.name) for run in selected_runs
    )
    duplicates = sorted(identity for identity, count in dispatch_identities.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"selected rank contains duplicate program/dispatch identities {duplicates}; pass a narrower build root"
        )
    snapshots: list[TargetSnapshot] = []
    artifacts: list[str] = []
    selected_programs: set[str] = set()
    for dispatch_index, run in enumerate(selected_runs):
        artifacts.append(str(run.directory))
        if run.program is not None:
            selected_programs.add(run.program)
        for match_occurrence, match in _matching_tasks(run, target, task_id, occurrence):
            snapshots.append(_snapshot(run, match, match_occurrence, dispatch_index))
    if not snapshots:
        raise ValueError(f"target {target!r} has no matching task in selected rank {selected_rank}")

    per_dispatch_matches = collections.Counter(snapshot.dispatch_index for snapshot in snapshots)
    ambiguous_occurrences = (
        task_id is None
        and occurrence is None
        and any(count > 1 for count in per_dispatch_matches.values())
    )
    expected_mix_names = {f"{target.casefold()}_aic", f"{target.casefold()}_aiv"}
    resolved_as_mix_alias = bool(snapshots) and all(
        {name.casefold() for name in snapshot.names} == expected_mix_names for snapshot in snapshots
    )

    rank_elapsed = {
        candidate_rank: sum(item.elapsed_us for item in items) for candidate_rank, items in grouped.items()
    }
    return {
        "version": BASELINE_VERSION,
        "root": str(root),
        "target": target,
        "operator": operator,
        "selected_rank": selected_rank,
        "rank_elapsed_us": dict(sorted(rank_elapsed.items(), key=lambda item: _rank_sort_key(item[0]))),
        "programs": sorted(selected_programs),
        "program_identity": (
            ", ".join(sorted(selected_programs))
            if selected_programs
            else "unambiguous direct dispatch (no dispatch_program.json)"
        ),
        "selection_explicit": task_id is not None or occurrence is not None,
        "ambiguous_occurrences": ambiguous_occurrences,
        "resolved_as_mix_alias": resolved_as_mix_alias,
        "artifacts": artifacts,
        "snapshots": [asdict(snapshot) for snapshot in snapshots],
    }


def _render_single(payload: dict[str, Any]) -> str:
    snapshots = payload["snapshots"]
    actual = [snapshot["observed_status"] in {"full", "partial"} for snapshot in snapshots]
    policy_edits = [snapshot["unflagged_predecessors"] for snapshot in snapshots]
    lines = [
        "# Early-dispatch baseline",
        "",
        f"- Target: **`{_md(payload['target'])}`**",
        f"- Selected rank/device: **`{payload['selected_rank']}`** (minimum dispatch→finish elapsed)",
        f"- Program identity: **{_md(payload.get('program_identity', 'unknown'))}**",
        "- Signed start gap: `earliest target start − latest direct-predecessor end`; negative means overlap.",
        "- Actual early dispatch: every direct producer is opted in and target dispatch precedes the latest producer FIN.",
        "- Graph-fidelity prerequisite: the separate deps capture must have the same value-routed task topology as "
        "the timing pass; structural row-count checks cannot prove this for device-resident control inputs.",
    ]
    if payload.get("resolved_as_mix_alias"):
        lines.append(
            "- Target resolution: the unsuffixed source name matched exactly one logical MIX identity whose compiled "
            "names are the `_aic` and `_aiv` variants shown below."
        )
    if all(actual):
        lines.append("- Result: **already early-dispatched; stop without modifying source.**")
    elif any(actual):
        lines.append(
            "- Result: **only some matching task occurrences were actually early-dispatched; do not apply a blanket "
            "source edit without selecting the intended task ID.**"
        )
    elif payload.get("ambiguous_occurrences"):
        lines.append(
            "- Result: **multiple same-name occurrences exist in one dispatch; select the intended source occurrence "
            "with `--occurrence N` before modifying source.**"
        )
    elif all(snapshot["structural_status"] == "eligible" for snapshot in snapshots) and not any(policy_edits):
        lines.append(
            "- Result: **all direct producers are already opted in; source annotation is not the missing condition. "
            "Diagnose the scheduler/launch blocker instead of adding duplicate flags.**"
        )
    elif all(snapshot["structural_status"] in {"root", "alloc-only"} for snapshot in snapshots):
        lines.append(
            "- Result: **the target has no non-allocation direct producer to mark, so this mechanism cannot make it "
            "early-dispatch eligible.**"
        )
    if any(snapshot["start_gap_us"] is not None and snapshot["start_gap_us"] < 1.0 for snapshot in snapshots):
        lines.append(
            "- ⚠️ Baseline start gap is below 1 µs for at least one target instance; the change may have little benefit."
        )

    lines.extend(
        [
            "",
            "## Rank comparison",
            "",
            "| rank/device | dispatch→finish elapsed µs | selected |",
            "|---|---:|---|",
        ]
    )
    for rank, elapsed in payload["rank_elapsed_us"].items():
        lines.append(f"| `{rank}` | {elapsed:.3f} | {'✓' if rank == payload['selected_rank'] else ''} |")

    lines.extend(
        [
            "",
            "## Target timing",
            "",
            "| dispatch | occurrence | task id | name | submit start–end µs | earliest dispatch µs | earliest start µs | "
            "latest pred end µs | signed start gap µs | post-FIN gap µs | early physical rows | structural | observed |",
            "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for snapshot in snapshots:
        submit_range = (
            f"{snapshot['submit_start_us']:.3f}–{snapshot['submit_end_us']:.3f}"
            if snapshot["submit_start_us"] is not None and snapshot["submit_end_us"] is not None
            else "—"
        )
        lines.append(
            f"| `{snapshot['dispatch']}` | {snapshot['occurrence']} | `{snapshot['task_id']}` | "
            f"{_md('/'.join(snapshot['names']))} | {submit_range} | {snapshot['earliest_dispatch_us']:.3f} | "
            f"{snapshot['earliest_start_us']:.3f} | "
            f"{_number(snapshot['latest_predecessor_end_us'])} | {_number(snapshot['start_gap_us'])} | "
            f"{_number(snapshot['post_fin_gap_us'])} | "
            f"{snapshot['early_blocks']}/{snapshot['total_blocks']} | {snapshot['structural_status']} | "
            f"{snapshot['observed_status']} |"
        )

    lines.extend(["", "## Direct predecessors and blockers", ""])
    for snapshot in snapshots:
        lines.append(f"### `{snapshot['task_id']}` {'/'.join(snapshot['names'])}")
        lines.append("")
        lines.append(
            "- Direct predecessor task IDs: "
            + (", ".join(f"`{task}`" for task in snapshot["predecessor_ids"]) or "none")
        )
        lines.append("- Direct predecessor names: " + (", ".join(snapshot["predecessor_names"]) or "none"))
        for blocker in snapshot["blocker_summary"]:
            lines.append(f"- {_md(blocker)}")
        lines.append("")

    lines.extend(["## Artifact directories", ""])
    lines.extend(f"- `{path}`" for path in payload["artifacts"])
    return "\n".join(lines).rstrip() + "\n"


def _snapshot_key(snapshot: dict[str, Any]) -> tuple[int, str, str, str, int]:
    return (
        int(snapshot.get("dispatch_index", 0)),
        str(snapshot["dispatch"]),
        str(snapshot.get("dispatch_program") or ""),
        "/".join(snapshot.get("names", ["unknown"])),
        int(snapshot.get("occurrence", 0)),
    )


def _index_snapshots(
    payload: dict[str, Any], label: str
) -> dict[tuple[int, str, str, str, int], dict[str, Any]]:
    indexed: dict[tuple[int, str, str, str, int], dict[str, Any]] = {}
    for snapshot in payload["snapshots"]:
        key = _snapshot_key(snapshot)
        if key in indexed:
            raise ValueError(f"{label} capture has duplicate target identity {key}")
        indexed[key] = snapshot
    return indexed


def _render_comparison(before: dict[str, Any], after: dict[str, Any]) -> str:
    if before.get("selected_rank") != after.get("selected_rank"):
        raise ValueError(
            "before/after selected ranks differ; rerun the after capture with the baseline physical rank"
        )
    before_by_key = _index_snapshots(before, "baseline")
    after_by_key = _index_snapshots(after, "after")
    if before_by_key.keys() != after_by_key.keys():
        missing_after = sorted(before_by_key.keys() - after_by_key.keys())
        added_after = sorted(after_by_key.keys() - before_by_key.keys())
        raise ValueError(
            "before/after target occurrence sets differ; refusing comparison "
            f"(missing after={missing_after}, added after={added_after})"
        )
    keys = sorted(before_by_key)
    for key in keys:
        old_signature = before_by_key[key].get("predecessor_names")
        new_signature = after_by_key[key].get("predecessor_names")
        if old_signature != new_signature:
            raise ValueError(
                f"target occurrence {key} changed direct-predecessor signature; refusing a non-equivalent comparison"
            )
        if before_by_key[key].get("total_blocks") != after_by_key[key].get("total_blocks"):
            raise ValueError(
                f"target occurrence {key} changed physical-row count; refusing a non-equivalent comparison"
            )
        if before_by_key[key].get("task_signature") != after_by_key[key].get("task_signature"):
            raise ValueError(
                f"target occurrence {key} changed logical block/kernel-slot signature; "
                "refusing a non-equivalent comparison"
            )
        if before_by_key[key].get("predecessor_signatures") != after_by_key[key].get(
            "predecessor_signatures"
        ):
            raise ValueError(
                f"target occurrence {key} changed direct-predecessor identity/shape; "
                "refusing a non-equivalent comparison"
            )
    lines = [
        "# Early-dispatch before/after comparison",
        "",
        f"- Target: **`{_md(after['target'])}`**",
        f"- Rank/device held fixed: **`{after['selected_rank']}`**",
        "- Signed start gap: `earliest target start − latest direct-predecessor end`; values below 1 µs satisfy the "
        "requested keep criterion.",
        "- The comparison shows both timestamps used by that calculation: the latest end across all timed direct "
        "predecessors and the earliest start across all physical rows of the target successor.",
        "- Graph-fidelity prerequisite: both deps captures must match their timing passes, including any "
        "value-routed task topology.",
        "",
        "| dispatch | occurrence | state | task id | target successor | latest direct-pred end µs | "
        "earliest target start µs | signed start gap µs | gap saved vs before µs | observed |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for key in keys:
        old = before_by_key[key]
        new = after_by_key[key]
        old_gap = old.get("start_gap_us")
        new_gap = new.get("start_gap_us")
        saved = old_gap - new_gap if old_gap is not None and new_gap is not None else None
        names = new.get("names", ["unknown"])
        lines.append(
            f"| `{old['dispatch']}` | {old['occurrence']} | before | `{old.get('task_id')}` | "
            f"{_md('/'.join(names))} | {_number(old.get('latest_predecessor_end_us'))} | "
            f"{_number(old.get('earliest_start_us'))} | {_number(old_gap)} | — | "
            f"{old.get('observed_status', 'missing')} |"
        )
        lines.append(
            f"| `{new['dispatch']}` | {new['occurrence']} | after | `{new.get('task_id')}` | "
            f"{_md('/'.join(names))} | {_number(new.get('latest_predecessor_end_us'))} | "
            f"{_number(new.get('earliest_start_us'))} | {_number(new_gap)} | {_number(saved)} | "
            f"{new.get('observed_status', 'missing')} |"
        )

    after_snapshots = [after_by_key[key] for key in keys]
    observed = [snapshot["observed_status"] in {"full", "partial"} for snapshot in after_snapshots]
    lines.extend(["", "## Decision", ""])
    if not all(observed):
        if any(observed):
            lines.append(
                "The scheduler early-dispatched only some matching target occurrences. The after artifacts are "
                "listed below; do not claim complete success."
            )
        else:
            lines.append(
                "The scheduler did **not** actually early-dispatch the target. The after artifacts are listed "
                "below; the proven blockers or narrowest supported diagnoses follow."
            )
        lines.append("")
        lines.append("Blocker and diagnostic evidence:")
        for snapshot, was_observed in zip(after_snapshots, observed, strict=True):
            if was_observed:
                continue
            for blocker in snapshot["blocker_summary"]:
                lines.append(f"- `{snapshot['task_id']}`: {_md(blocker)}")
    elif all(
        snapshot["start_gap_us"] is not None and snapshot["start_gap_us"] < 1.0
        for snapshot in after_snapshots
    ):
        lines.append(
            "The modification is effective under the requested target-gap criterion; **keep it after the required "
            "correctness checks pass**. Report any separate fan-out or end-to-end regression."
        )
    else:
        lines.append(
            "The target was early-dispatched, but at least one resulting signed start gap is not below 1 µs. "
            "**Do you want to retain this source change?**"
        )

    lines.extend(["", "## After artifacts", ""])
    lines.extend(f"- `{path}`" for path in after["artifacts"])
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect actual early dispatch for a named task in level-4 L2 swimlane artifacts."
    )
    parser.add_argument("build_dir", type=Path, help="Build/run directory containing fresh dfx_outputs")
    parser.add_argument("--target", required=True, help="Exact task/operator name from name_map")
    parser.add_argument("--operator", help="Filter dispatch_program.json values for the enclosing program")
    parser.add_argument(
        "--rank", help="Select a rank; comparisons automatically pin the baseline physical rank"
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task-id", help="Select one logical task ID in this capture")
    selection.add_argument(
        "--occurrence",
        type=int,
        help="Select the zero-based same-name occurrence; stable across equivalent before/after captures",
    )
    parser.add_argument(
        "--baseline-json", type=Path, help="Compare this run with a prior --json-out snapshot"
    )
    parser.add_argument("--json-out", type=Path, help="Write this run's machine-readable snapshot")
    parser.add_argument("-o", "--output", type=Path, help="Write Markdown here instead of stdout")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = args.build_dir.expanduser().resolve()
    if not root.exists():
        print(f"error: path not found: {root}", file=sys.stderr)
        return 2
    if root.is_file():
        root = root.parent

    try:
        baseline = None
        effective_rank = args.rank
        effective_operator = args.operator
        if args.baseline_json:
            baseline_path = args.baseline_json.expanduser().resolve()
            baseline = _read_json(baseline_path)
            if baseline.get("version") != BASELINE_VERSION:
                raise ValueError(f"{baseline_path}: unsupported baseline version")
            if str(baseline.get("target", "")).casefold() != args.target.casefold():
                raise ValueError(
                    f"{baseline_path}: target {baseline.get('target')!r} does not match {args.target!r}"
                )
            baseline_rank = str(baseline.get("selected_rank"))
            if args.rank is not None and args.rank != baseline_rank:
                raise ValueError(
                    f"--rank {args.rank!r} differs from baseline rank {baseline_rank!r}; "
                    "before/after comparison must use the same physical rank"
                )
            effective_rank = baseline_rank
            if effective_operator is None:
                effective_operator = baseline.get("operator")

        payload = _run_payload(
            root,
            args.target,
            effective_operator,
            effective_rank,
            args.task_id,
            args.occurrence,
        )
        if baseline is not None:
            if baseline.get("programs", []) != payload.get("programs", []):
                raise ValueError(
                    f"{baseline_path}: dispatch programs {baseline.get('programs', [])!r} do not match "
                    f"{payload.get('programs', [])!r}"
                )
        report = _render_comparison(baseline, payload) if baseline else _render_single(payload)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json_out:
        json_out = args.json_out.expanduser().resolve()
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Snapshot written to: {json_out}", file=sys.stderr)
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(f"Report written to: {output}")
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
