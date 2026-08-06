# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Select and compare an early-dispatch sibling-suppression experiment.

The positional skill argument is the sibling that receives an unflagged
``pl.system.task_dummy(deps=[])`` predecessor.  The protected task and shared
producer are selected from the canonical Observed critical path.
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
    from simpler_setup.tools import critical_path
    from simpler_setup.tools.swimlane_converter import read_perf_data
except ImportError as exc:  # pragma: no cover - environment preflight
    raise SystemExit(
        "Could not import simpler_setup. Activate the pypto-lib environment first "
        "(for this worktree: source temp/set_env.sh)."
    ) from exc


RECORD_NAMES = ("l2_swimlane_records.json", "l2_perf_records.json")
RANK_RE = re.compile(r"rank\d+$")
SNAPSHOT_VERSION = 1


@dataclass
class Timing:
    start_us: float
    end_us: float
    dispatch_us: float
    finish_us: float
    blocks: int


@dataclass
class EarlyStatus:
    structural: str
    observed: str
    early_blocks: int
    total_blocks: int
    latest_predecessor_finish_us: float | None
    unflagged_predecessors: list[str]
    untimed_predecessors: list[str]


@dataclass
class PairSnapshot:
    dispatch: str
    dispatch_index: int
    dispatch_program: str | None
    total_dummy_tasks: int
    total_scheduler_dummy_tasks: int
    suppressed_occurrence: int
    suppressed_task_id: str
    suppressed_names: list[str]
    suppressed_total_occurrences: int
    suppressed_task_signature: str
    suppressed_block_count: int
    suppressed_predecessor_names: list[str]
    suppressed_predecessor_signatures: list[str]
    suppressed_dummy_predecessors: list[str]
    suppressed_root_dummy_predecessors: list[str]
    suppressed_unflagged_dummy_predecessors: list[str]
    suppressed_explicit_dummy_predecessors: list[str]
    suppressed_scheduler_dummy_predecessors: list[str]
    suppressed_status: EarlyStatus
    protected_occurrence: int
    protected_task_id: str
    protected_names: list[str]
    protected_total_occurrences: int
    protected_task_signature: str
    protected_block_count: int
    protected_predecessor_names: list[str]
    protected_predecessor_signatures: list[str]
    protected_status: EarlyStatus
    producer_occurrence: int
    producer_task_id: str
    producer_names: list[str]
    producer_total_occurrences: int
    producer_task_signature: str
    producer_block_count: int
    producer_predecessor_signatures: list[str]
    producer_end_us: float
    protected_end_us: float
    producer_to_protected_end_us: float
    protected_edge_on_observed_path: bool
    suppressed_on_observed_path: bool
    other_early_siblings: list[str]


@dataclass
class Run:
    directory: Path
    rank: str
    program: str | None
    graph: Any
    result: Any
    rows_by_task: dict[str, list[dict[str, Any]]]
    timings: dict[str, Timing]
    task_table: dict[str, dict[str, Any]]
    predecessors: dict[str, set[str]]
    edge_sources: dict[tuple[str, str], set[str]]
    names: dict[str, list[str]]
    scheduler_dummy_tasks: set[str]
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


def _is_dummy_info(task: dict[str, Any]) -> bool:
    kernel_ids = task.get("kernel_ids") or []
    return bool(kernel_ids) and all(int(kernel_id) < 0 for kernel_id in kernel_ids) and not task.get("args")


def _task_names(task: dict[str, Any], callable_names: dict[str, str]) -> list[str]:
    result: list[str] = []
    for kernel_id in task.get("kernel_ids") or []:
        try:
            numeric = int(kernel_id)
        except (TypeError, ValueError):
            continue
        if numeric < 0:
            continue
        name = callable_names.get(str(numeric), f"cid{numeric}")
        if name not in result:
            result.append(name)
    if result:
        return result
    return ["dummy"] if _is_dummy_info(task) else ["unknown"]


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
        finish_us=max(finishes),
        blocks=len(rows),
    )


def _build_run(directory: Path, graph_root: Path, tol_ticks: int) -> Run:
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
    edge_sources: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for edge in deps.get("edges", []):
        if not isinstance(edge, dict) or edge.get("pred") is None or edge.get("succ") is None:
            continue
        pred, succ = _task_id(edge["pred"]), _task_id(edge["succ"])
        if pred != succ:
            predecessors[succ].add(pred)
            edge_sources[(pred, succ)].add(str(edge.get("source") or "unknown"))

    callable_names = _name_map(directory)
    names = {task: _task_names(info, callable_names) for task, info in task_table.items()}
    scheduler_dummy_tasks = {
        _task_id(phase["task_id"])
        for thread in joined.get("aicpu_scheduler_phases", [])
        for phase in thread
        if isinstance(phase, dict) and phase.get("phase") == "dummy_task" and phase.get("task_id") is not None
    }
    graph = critical_path.build_graph(directory, graph_root, tol_ticks)
    result = critical_path.analyze_rank(graph, tol_ticks)
    elapsed_us = max(t.finish_us for t in timings.values()) - min(t.dispatch_us for t in timings.values())
    return Run(
        directory=directory,
        rank=_rank_label(directory),
        program=_program(directory),
        graph=graph,
        result=result,
        rows_by_task=dict(rows_by_task),
        timings=timings,
        task_table=task_table,
        predecessors=dict(predecessors),
        edge_sources=dict(edge_sources),
        names=names,
        scheduler_dummy_tasks=scheduler_dummy_tasks,
        elapsed_us=elapsed_us,
        tol_us=tol_ticks / frequency * 1e6,
    )


def _is_alloc(task: str, run: Run) -> bool:
    return task not in run.task_table


def _is_dummy(task: str, run: Run) -> bool:
    info = run.task_table.get(task)
    return bool(info and _is_dummy_info(info))


def _name(task: str, run: Run) -> str:
    if _is_alloc(task, run):
        return "alloc"
    return "/".join(run.names.get(task, ["unknown"]))


def _query_matches(query: str, names: list[str]) -> bool:
    needle = query.casefold()
    folded = [name.casefold() for name in names]
    if needle in folded:
        return True
    suffixes = ("_aic", "_aiv")
    return len(folded) > 1 and all(
        name.startswith(needle) and name.removeprefix(needle) in suffixes for name in folded
    )


def _matching_tasks(run: Run, query: str) -> list[str]:
    return sorted(
        (task for task, names in run.names.items() if task in run.timings and _query_matches(query, names)),
        key=int,
    )


def _select_occurrence(run: Run, query: str, occurrence: int | None, role: str) -> tuple[int, str]:
    matches = _matching_tasks(run, query)
    if not matches:
        available = sorted(
            {
                name
                for task in run.timings
                for name in run.names.get(task, [])
                if name not in {"unknown", "dummy"}
            }
        )
        examples = ", ".join(name for name in available if query.casefold() in name.casefold())
        raise ValueError(
            f"{run.directory}: no exact timed {role} matches {query!r}; "
            f"available examples: {examples or ', '.join(available[:20])}"
        )
    if occurrence is None:
        if len(matches) != 1:
            raise ValueError(
                f"{run.directory}: {role} {query!r} has {len(matches)} occurrences "
                f"({', '.join(matches)}); select one with --{role}-occurrence"
            )
        return 0, matches[0]
    if occurrence < 0 or occurrence >= len(matches):
        raise ValueError(
            f"{run.directory}: --{role}-occurrence {occurrence} is outside 0..{len(matches) - 1}"
        )
    return occurrence, matches[occurrence]


def _identity_count(run: Run, names: list[str]) -> int:
    return sum(candidate_names == names for candidate_names in run.names.values())


def _select_identity(
    run: Run,
    names: list[str],
    occurrence: int,
    expected_total: int,
    role: str,
) -> tuple[int, str]:
    actual_total = _identity_count(run, names)
    if actual_total != expected_total:
        raise ValueError(
            f"{run.directory}: {role} identity {'/'.join(names)!r} occurrence count changed "
            f"from {expected_total} to {actual_total}"
        )
    matches = sorted(
        (
            task
            for task, candidate_names in run.names.items()
            if task in run.timings and candidate_names == names
        ),
        key=int,
    )
    if occurrence < 0 or occurrence >= len(matches):
        raise ValueError(
            f"{run.directory}: {role} identity {'/'.join(names)!r} occurrence {occurrence} "
            f"is outside 0..{len(matches) - 1}"
        )
    return occurrence, matches[occurrence]


def _occurrence(run: Run, task: str) -> int:
    names = run.names.get(task, [])
    if not names:
        raise ValueError(f"{run.directory}: task {task} has no callable name")
    matches = sorted(
        (
            candidate
            for candidate, candidate_names in run.names.items()
            if candidate in run.timings and candidate_names == names
        ),
        key=int,
    )
    return matches.index(task)


def _predecessor_names(run: Run, task: str) -> list[str]:
    return sorted(_name(pred, run) for pred in run.predecessors.get(task, set()))


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
    kernel_ids = info.get("kernel_ids") or []
    active_slots = sum(int(kernel_id) >= 0 for kernel_id in kernel_ids)
    logical_blocks = int(info.get("block_num"))
    rows = len(run.rows_by_task.get(task, []))
    if _is_dummy(task, run):
        fanin = len(run.predecessors.get(task, set()))
        return f"dummy|fanin={fanin}|logical={logical_blocks}|slots={active_slots}|rows={rows}"

    names = run.names.get(task, ["unknown"])
    peers = sorted(
        (candidate for candidate, candidate_names in run.names.items() if candidate_names == names),
        key=int,
    )
    occurrence = peers.index(task)
    return f"{'/'.join(names)}#{occurrence}|logical={logical_blocks}|slots={active_slots}|rows={rows}"


def _predecessor_signatures(run: Run, task: str) -> list[str]:
    return sorted(_task_signature(run, pred) for pred in run.predecessors.get(task, set()))


def _early_status(run: Run, task: str) -> EarlyStatus:
    predecessors = sorted(run.predecessors.get(task, set()), key=int)
    non_alloc = [pred for pred in predecessors if not _is_alloc(pred, run)]
    unflagged = [pred for pred in non_alloc if not bool(run.task_table[pred].get("early_dispatch"))]
    # Allocation sources are graph roots and may intentionally have no joined
    # timing row.  Missing timing is fatal only for a real non-allocation
    # producer whose FIN is required for target-specific early-dispatch proof.
    untimed = [pred for pred in non_alloc if pred not in run.timings]

    if not predecessors:
        structural = "root"
    elif unflagged:
        structural = "policy-blocked"
    elif non_alloc:
        structural = "eligible"
    else:
        structural = "alloc-only"

    target_rows = run.rows_by_task.get(task, [])
    total_blocks = len(target_rows)
    latest_finish = None
    early_blocks = 0
    if predecessors:
        timed_predecessors = [pred for pred in predecessors if pred in run.timings]
        latest_finish = max((run.timings[pred].finish_us for pred in timed_predecessors), default=None)
    if latest_finish is not None:
        early_blocks = sum(
            _finite(row["dispatch_time_us"], f"{task}.dispatch_time_us") + run.tol_us < latest_finish
            for row in target_rows
        )

    if structural == "policy-blocked":
        # An unflagged producer (notably task_dummy) proves ineligibility, but
        # still compare against every timed real producer.  Early-looking
        # dispatch timestamps indicate stale/mismatched deps rather than a
        # successful suppression.
        observed = "mismatch" if early_blocks else "none"
    elif untimed or latest_finish is None:
        observed = "unverifiable"
    elif structural != "eligible":
        observed = "mismatch" if early_blocks else "none"
    elif early_blocks == total_blocks and total_blocks:
        observed = "full"
    elif early_blocks:
        observed = "partial"
    else:
        observed = "none"

    return EarlyStatus(
        structural=structural,
        observed=observed,
        early_blocks=early_blocks,
        total_blocks=total_blocks,
        latest_predecessor_finish_us=latest_finish,
        unflagged_predecessors=unflagged,
        untimed_predecessors=untimed,
    )


def _observed_edges(run: Run) -> list[tuple[str, str]]:
    path = [segment.task for segment in run.result.segments]
    return list(zip(path, path[1:]))


def _observed_data_edges(run: Run) -> list[tuple[str, str]]:
    segments = run.result.segments
    return [
        (previous.task, current.task)
        for previous, current in zip(segments, segments[1:])
        if current.kind == "data-wait"
    ]


def _candidate_description(run: Run, producer: str, protected: str) -> str:
    return f"producer {producer} ({_name(producer, run)}) -> protected {protected} ({_name(protected, run)})"


def _discover_pair(
    run: Run,
    suppressed: str,
    suppressed_occurrence: int | None,
    protected: str | None,
    protected_occurrence: int | None,
    producer: str | None,
    producer_occurrence: int | None,
) -> tuple[int, str, int, str, int, str]:
    suppressed_index, suppressed_task = _select_occurrence(
        run, suppressed, suppressed_occurrence, "suppressed"
    )
    suppressed_status = _early_status(run, suppressed_task)
    if suppressed_status.observed not in {"full", "partial"}:
        raise ValueError(
            f"{run.directory}: suppressed task {suppressed_task} ({_name(suppressed_task, run)}) "
            f"is not actually early-dispatched ({suppressed_status.observed}); do not add a dummy"
        )

    path_tasks = {segment.task for segment in run.result.segments}
    if suppressed_task in path_tasks:
        raise ValueError(
            f"{run.directory}: suppressed task {suppressed_task} ({_name(suppressed_task, run)}) is on the "
            "Observed critical path; adding a dummy to it is unsafe"
        )

    protected_selection: tuple[int, str] | None = None
    if protected is not None:
        protected_selection = _select_occurrence(run, protected, protected_occurrence, "protected")
    producer_selection: tuple[int, str] | None = None
    if producer is not None:
        producer_selection = _select_occurrence(run, producer, producer_occurrence, "producer")

    candidates: list[tuple[str, str]] = []
    for candidate_producer, candidate_protected in _observed_data_edges(run):
        if candidate_producer not in run.predecessors.get(candidate_protected, set()):
            continue  # Observed same-core predecessor, not the requested shared dependency edge.
        if candidate_producer not in run.predecessors.get(suppressed_task, set()):
            continue
        if _is_alloc(candidate_producer, run) or candidate_protected == suppressed_task:
            continue
        if protected_selection and candidate_protected != protected_selection[1]:
            continue
        if producer_selection and candidate_producer != producer_selection[1]:
            continue
        candidates.append((candidate_producer, candidate_protected))

    if not candidates:
        qualifier = " using the requested protected task/producer" if protected or producer else ""
        raise ValueError(
            f"{run.directory}: no Observed-path data edge c->a shares c with suppressed task "
            f"{suppressed_task}{qualifier}"
        )
    if len(candidates) != 1:
        detail = "; ".join(_candidate_description(run, c, a) for c, a in candidates)
        raise ValueError(
            f"{run.directory}: multiple protected sibling pairs match; select --protected and/or --producer "
            f"with occurrence selectors if needed ({detail})"
        )

    producer_task, protected_task = candidates[0]
    protected_status = _early_status(run, protected_task)
    if protected_status.structural != "eligible":
        raise ValueError(
            f"{run.directory}: protected task {protected_task} ({_name(protected_task, run)}) is "
            f"{protected_status.structural}; suppressing its sibling cannot make it early-dispatch eligible"
        )
    return (
        suppressed_index,
        suppressed_task,
        _occurrence(run, protected_task),
        protected_task,
        _occurrence(run, producer_task),
        producer_task,
    )


def _snapshot(
    run: Run,
    dispatch_index: int,
    suppressed_occurrence: int,
    suppressed_task: str,
    protected_occurrence: int,
    protected_task: str,
    producer_occurrence: int,
    producer_task: str,
) -> PairSnapshot:
    if producer_task not in run.predecessors.get(suppressed_task, set()):
        raise ValueError(
            f"{run.directory}: shared producer is no longer a direct predecessor of suppressed task"
        )
    if producer_task not in run.predecessors.get(protected_task, set()):
        raise ValueError(
            f"{run.directory}: shared producer is no longer a direct predecessor of protected task"
        )
    producer_timing = run.timings.get(producer_task)
    protected_timing = run.timings.get(protected_task)
    suppressed_timing = run.timings.get(suppressed_task)
    if producer_timing is None or protected_timing is None or suppressed_timing is None:
        raise ValueError(f"{run.directory}: selected c/a/b task timing is incomplete")

    observed_edges = set(_observed_edges(run))
    observed_tasks = {segment.task for segment in run.result.segments}
    dummy_predecessors = sorted(
        (pred for pred in run.predecessors.get(suppressed_task, set()) if _is_dummy(pred, run)),
        key=int,
    )
    explicit_dummy_predecessors = [
        pred
        for pred in dummy_predecessors
        if "explicit" in run.edge_sources.get((pred, suppressed_task), set())
    ]
    unflagged_dummy_predecessors = [
        pred for pred in dummy_predecessors if not bool(run.task_table[pred].get("early_dispatch"))
    ]
    root_dummy_predecessors = [pred for pred in dummy_predecessors if not run.predecessors.get(pred, set())]
    scheduler_dummy_predecessors = [pred for pred in dummy_predecessors if pred in run.scheduler_dummy_tasks]
    other_early_siblings: list[str] = []
    for sibling, sibling_predecessors in run.predecessors.items():
        if sibling in {suppressed_task, protected_task} or producer_task not in sibling_predecessors:
            continue
        if sibling not in run.timings:
            continue
        sibling_status = _early_status(run, sibling)
        if sibling_status.observed in {"full", "partial"}:
            other_early_siblings.append(
                f"{sibling} ({_name(sibling, run)}): {sibling_status.observed} "
                f"{sibling_status.early_blocks}/{sibling_status.total_blocks}"
            )
    return PairSnapshot(
        dispatch=run.directory.name,
        dispatch_index=dispatch_index,
        dispatch_program=run.program,
        total_dummy_tasks=sum(_is_dummy(task, run) for task in run.task_table),
        total_scheduler_dummy_tasks=len(run.scheduler_dummy_tasks),
        suppressed_occurrence=suppressed_occurrence,
        suppressed_task_id=suppressed_task,
        suppressed_names=run.names.get(suppressed_task, ["unknown"]),
        suppressed_total_occurrences=_identity_count(run, run.names.get(suppressed_task, ["unknown"])),
        suppressed_task_signature=_task_signature(run, suppressed_task),
        suppressed_block_count=suppressed_timing.blocks,
        suppressed_predecessor_names=_predecessor_names(run, suppressed_task),
        suppressed_predecessor_signatures=_predecessor_signatures(run, suppressed_task),
        suppressed_dummy_predecessors=dummy_predecessors,
        suppressed_root_dummy_predecessors=root_dummy_predecessors,
        suppressed_unflagged_dummy_predecessors=unflagged_dummy_predecessors,
        suppressed_explicit_dummy_predecessors=explicit_dummy_predecessors,
        suppressed_scheduler_dummy_predecessors=scheduler_dummy_predecessors,
        suppressed_status=_early_status(run, suppressed_task),
        protected_occurrence=protected_occurrence,
        protected_task_id=protected_task,
        protected_names=run.names.get(protected_task, ["unknown"]),
        protected_total_occurrences=_identity_count(run, run.names.get(protected_task, ["unknown"])),
        protected_task_signature=_task_signature(run, protected_task),
        protected_block_count=protected_timing.blocks,
        protected_predecessor_names=_predecessor_names(run, protected_task),
        protected_predecessor_signatures=_predecessor_signatures(run, protected_task),
        protected_status=_early_status(run, protected_task),
        producer_occurrence=producer_occurrence,
        producer_task_id=producer_task,
        producer_names=run.names.get(producer_task, ["unknown"]),
        producer_total_occurrences=_identity_count(run, run.names.get(producer_task, ["unknown"])),
        producer_task_signature=_task_signature(run, producer_task),
        producer_block_count=producer_timing.blocks,
        producer_predecessor_signatures=_predecessor_signatures(run, producer_task),
        producer_end_us=producer_timing.end_us,
        protected_end_us=protected_timing.end_us,
        producer_to_protected_end_us=protected_timing.end_us - producer_timing.end_us,
        protected_edge_on_observed_path=(producer_task, protected_task) in observed_edges,
        suppressed_on_observed_path=suppressed_task in observed_tasks,
        other_early_siblings=sorted(other_early_siblings),
    )


def _select_runs(
    root: Path,
    operator: str | None,
    rank: str | None,
    tol_ticks: int,
) -> tuple[list[Run], str, dict[str, float]]:
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
        raise ValueError(
            "multiple dispatch programs found; select one with --operator "
            + ", ".join(sorted(known_programs))
        )
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

    runs = [_build_run(directory, root, tol_ticks) for directory in directories]
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
    selected = sorted(grouped[selected_rank], key=_dispatch_sort_key)
    identities = collections.Counter((run.program or "", run.directory.name) for run in selected)
    duplicates = sorted(identity for identity, count in identities.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"selected rank contains duplicate program/dispatch identities {duplicates}; pass a narrower root"
        )
    elapsed = {
        candidate_rank: sum(item.elapsed_us for item in items) for candidate_rank, items in grouped.items()
    }
    return selected, selected_rank, elapsed


def _baseline_payload(
    root: Path,
    suppressed: str,
    operator: str | None,
    rank: str | None,
    suppressed_occurrence: int | None,
    protected: str | None,
    protected_occurrence: int | None,
    producer: str | None,
    producer_occurrence: int | None,
    tol_ticks: int,
) -> dict[str, Any]:
    runs, selected_rank, rank_elapsed = _select_runs(root, operator, rank, tol_ticks)
    snapshots: list[PairSnapshot] = []
    artifacts: list[str] = []
    programs: set[str] = set()
    for dispatch_index, run in enumerate(runs):
        artifacts.append(str(run.directory))
        if run.program:
            programs.add(run.program)
        selection = _discover_pair(
            run,
            suppressed,
            suppressed_occurrence,
            protected,
            protected_occurrence,
            producer,
            producer_occurrence,
        )
        snapshots.append(_snapshot(run, dispatch_index, *selection))
    return {
        "version": SNAPSHOT_VERSION,
        "root": str(root),
        "suppressed": suppressed,
        "operator": operator,
        "selected_rank": selected_rank,
        "rank_elapsed_us": dict(sorted(rank_elapsed.items(), key=lambda item: _rank_sort_key(item[0]))),
        "programs": sorted(programs),
        "program_identity": (
            ", ".join(sorted(programs))
            if programs
            else "unambiguous direct dispatch (no dispatch_program.json)"
        ),
        "artifacts": artifacts,
        "snapshots": [asdict(snapshot) for snapshot in snapshots],
    }


def _snapshot_key(snapshot: dict[str, Any]) -> tuple[int, str, str]:
    return (
        int(snapshot.get("dispatch_index", 0)),
        str(snapshot["dispatch"]),
        str(snapshot.get("dispatch_program") or ""),
    )


def _comparison_payload(
    root: Path,
    baseline: dict[str, Any],
    operator: str | None,
    rank: str | None,
    tol_ticks: int,
) -> dict[str, Any]:
    baseline_rank = str(baseline["selected_rank"])
    if rank is not None and rank != baseline_rank:
        raise ValueError(f"--rank {rank!r} differs from baseline rank {baseline_rank!r}")
    effective_operator = operator if operator is not None else baseline.get("operator")
    runs, selected_rank, rank_elapsed = _select_runs(root, effective_operator, baseline_rank, tol_ticks)
    baseline_by_key = {_snapshot_key(snapshot): snapshot for snapshot in baseline["snapshots"]}
    if len(baseline_by_key) != len(baseline["snapshots"]):
        raise ValueError("baseline contains duplicate dispatch identities")

    snapshots: list[PairSnapshot] = []
    artifacts: list[str] = []
    programs: set[str] = set()
    for dispatch_index, run in enumerate(runs):
        artifacts.append(str(run.directory))
        if run.program:
            programs.add(run.program)
        key = (dispatch_index, run.directory.name, run.program or "")
        old = baseline_by_key.get(key)
        if old is None:
            raise ValueError(f"after capture has no matching baseline dispatch identity {key}")
        suppressed_selection = _select_identity(
            run,
            list(old["suppressed_names"]),
            int(old["suppressed_occurrence"]),
            int(old["suppressed_total_occurrences"]),
            "suppressed",
        )
        protected_selection = _select_identity(
            run,
            list(old["protected_names"]),
            int(old["protected_occurrence"]),
            int(old["protected_total_occurrences"]),
            "protected",
        )
        producer_selection = _select_identity(
            run,
            list(old["producer_names"]),
            int(old["producer_occurrence"]),
            int(old["producer_total_occurrences"]),
            "producer",
        )
        snapshots.append(
            _snapshot(
                run,
                dispatch_index,
                suppressed_selection[0],
                suppressed_selection[1],
                protected_selection[0],
                protected_selection[1],
                producer_selection[0],
                producer_selection[1],
            )
        )
    if set(baseline_by_key) != {_snapshot_key(asdict(snapshot)) for snapshot in snapshots}:
        raise ValueError("before/after dispatch identity sets differ")
    return {
        "version": SNAPSHOT_VERSION,
        "root": str(root),
        "suppressed": baseline["suppressed"],
        "operator": effective_operator,
        "selected_rank": selected_rank,
        "rank_elapsed_us": dict(sorted(rank_elapsed.items(), key=lambda item: _rank_sort_key(item[0]))),
        "programs": sorted(programs),
        "program_identity": (
            ", ".join(sorted(programs))
            if programs
            else "unambiguous direct dispatch (no dispatch_program.json)"
        ),
        "artifacts": artifacts,
        "snapshots": [asdict(snapshot) for snapshot in snapshots],
    }


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _status_text(status: dict[str, Any]) -> str:
    return (
        f"{status['observed']} {status['early_blocks']}/{status['total_blocks']} physical rows"
    )


def _render_baseline(payload: dict[str, Any]) -> str:
    lines = [
        "# Add-dummy baseline",
        "",
        f"- Suppressed sibling `b`: **`{_md(payload['suppressed'])}`**",
        f"- Selected rank/device: **`{payload['selected_rank']}`** (minimum dispatch→finish elapsed)",
        f"- Program identity: **{_md(payload.get('program_identity', 'unknown'))}**",
        "- Pair selection: `c -> a` must be an actual dependency edge on the canonical Observed path, and `b` "
        "must share `c` without itself appearing on that path.",
        "- Metric: `latest a AICore end - latest c AICore end`; smaller is better.",
        "- Baseline result: the requested sibling is actually early-dispatched and is safe to suppress with one "
        "empty-dependency dummy, subject to source mapping and correctness checks.",
        "- Graph-fidelity prerequisite: the deps and timing passes must represent the same value-routed task topology.",
        "",
        "## Rank comparison",
        "",
        "| rank/device | dispatch→finish elapsed µs | selected |",
        "|---|---:|---|",
    ]
    for rank, elapsed in payload["rank_elapsed_us"].items():
        lines.append(f"| `{rank}` | {elapsed:.3f} | {'✓' if rank == payload['selected_rank'] else ''} |")
    lines.extend(
        [
            "",
            "## Selected sibling pair",
            "",
            "| dispatch | b: suppress | c: shared producer | a: protect | b early | a early | "
            "c end µs | a end µs | c end→a end µs |",
            "|---|---|---|---|---|---|---:|---:|---:|",
        ]
    )
    for snapshot in payload["snapshots"]:
        lines.append(
            f"| `{snapshot['dispatch']}` | `{snapshot['suppressed_task_id']}` "
            f"{_md('/'.join(snapshot['suppressed_names']))} | `{snapshot['producer_task_id']}` "
            f"{_md('/'.join(snapshot['producer_names']))} | `{snapshot['protected_task_id']}` "
            f"{_md('/'.join(snapshot['protected_names']))} | {_status_text(snapshot['suppressed_status'])} | "
            f"{_status_text(snapshot['protected_status'])} | {snapshot['producer_end_us']:.3f} | "
            f"{snapshot['protected_end_us']:.3f} | {snapshot['producer_to_protected_end_us']:.3f} |"
        )
    other_siblings = [
        f"`{snapshot['dispatch']}`: {', '.join(snapshot['other_early_siblings'])}"
        for snapshot in payload["snapshots"]
        if snapshot["other_early_siblings"]
    ]
    if other_siblings:
        lines.extend(["", "### Other early-dispatched consumers of `c`", ""])
        lines.extend(f"- {item}" for item in other_siblings)
        lines.append("")
        lines.append(
            "This invocation suppresses only the named `b`; do not claim that `a` is the only early-dispatched "
            "consumer while these tasks remain."
        )
    lines.extend(["", "## Required source change", ""])
    lines.append(
        "Create `seed_dummy = pl.system.task_dummy(deps=[])` in the same live scope before `b`, then append "
        "`seed_dummy` to `b`'s existing `deps` without changing `a`, `c`, or any other dependency."
    )
    lines.extend(["", "## Artifact directories", ""])
    lines.extend(f"- `{path}`" for path in payload["artifacts"])
    return "\n".join(lines).rstrip() + "\n"


def _indexed(payload: dict[str, Any], label: str) -> dict[tuple[int, str, str], dict[str, Any]]:
    result: dict[tuple[int, str, str], dict[str, Any]] = {}
    for snapshot in payload["snapshots"]:
        key = _snapshot_key(snapshot)
        if key in result:
            raise ValueError(f"{label} capture contains duplicate dispatch identity {key}")
        result[key] = snapshot
    return result


def _validate_comparison(before: dict[str, Any], after: dict[str, Any]) -> list[tuple[int, str, str]]:
    if before.get("programs") != after.get("programs"):
        raise ValueError("before/after dispatch programs differ")
    if before.get("selected_rank") != after.get("selected_rank"):
        raise ValueError("before/after selected ranks differ")
    before_by_key = _indexed(before, "baseline")
    after_by_key = _indexed(after, "after")
    if before_by_key.keys() != after_by_key.keys():
        raise ValueError("before/after dispatch identity sets differ")
    keys = sorted(before_by_key)
    for key in keys:
        old, new = before_by_key[key], after_by_key[key]
        identity_fields = (
            "suppressed_names",
            "suppressed_occurrence",
            "suppressed_total_occurrences",
            "suppressed_task_signature",
            "suppressed_block_count",
            "protected_names",
            "protected_occurrence",
            "protected_total_occurrences",
            "protected_task_signature",
            "protected_block_count",
            "protected_predecessor_names",
            "protected_predecessor_signatures",
            "producer_names",
            "producer_occurrence",
            "producer_total_occurrences",
            "producer_task_signature",
            "producer_block_count",
            "producer_predecessor_signatures",
        )
        for field in identity_fields:
            if old.get(field) != new.get(field):
                raise ValueError(f"{key}: before/after {field} differs; refusing a non-equivalent comparison")
        expected_suppressed_preds = sorted([*old["suppressed_predecessor_names"], "dummy"])
        if new["suppressed_predecessor_names"] != expected_suppressed_preds:
            raise ValueError(
                f"{key}: suppressed predecessor signature must gain exactly one dummy "
                f"(expected {expected_suppressed_preds}, got {new['suppressed_predecessor_names']})"
            )
        if int(new["total_dummy_tasks"]) <= int(old["total_dummy_tasks"]):
            raise ValueError(f"{key}: no newly created task_dummy is present in the after graph")
        if int(new["total_scheduler_dummy_tasks"]) <= int(old["total_scheduler_dummy_tasks"]):
            raise ValueError(f"{key}: no newly created scheduler dummy_task record is present")
        new_real_predecessors = [
            signature
            for signature in new["suppressed_predecessor_signatures"]
            if not signature.startswith("dummy|")
        ]
        if new_real_predecessors != old["suppressed_predecessor_signatures"]:
            raise ValueError(f"{key}: suppressed task's original predecessor identities/topology changed")
        if len(new["suppressed_dummy_predecessors"]) != 1:
            raise ValueError(f"{key}: expected exactly one direct dummy predecessor after the edit")
        if new["suppressed_root_dummy_predecessors"] != new["suppressed_dummy_predecessors"]:
            raise ValueError(f"{key}: the new dummy predecessor does not have deps=[]")
        if new["suppressed_unflagged_dummy_predecessors"] != new["suppressed_dummy_predecessors"]:
            raise ValueError(f"{key}: the new dummy predecessor has early_dispatch=true")
        if new["suppressed_explicit_dummy_predecessors"] != new["suppressed_dummy_predecessors"]:
            raise ValueError(f"{key}: the new dummy predecessor edge is not marked source=explicit")
        if new["suppressed_scheduler_dummy_predecessors"] != new["suppressed_dummy_predecessors"]:
            raise ValueError(f"{key}: the direct dummy predecessor lacks a task-ID-bearing scheduler marker")
        if new["suppressed_on_observed_path"]:
            raise ValueError(f"{key}: suppressed sibling moved onto the Observed critical path")
    return keys


def _render_comparison(before: dict[str, Any], after: dict[str, Any]) -> str:
    keys = _validate_comparison(before, after)
    before_by_key = _indexed(before, "baseline")
    after_by_key = _indexed(after, "after")
    lines = [
        "# Add-dummy before/after comparison",
        "",
        f"- Suppressed sibling `b`: **`{_md(after['suppressed'])}`**",
        f"- Rank/device held fixed: **`{after['selected_rank']}`**",
        f"- Program identity: **{_md(after.get('program_identity', 'unknown'))}**",
        "- Metric: `latest a AICore end - latest c AICore end`; improvement is `before - after`.",
        "- Target-specific early dispatch is proved from dependency policy plus per-block dispatch versus producer FIN; "
        "the scheduler's aggregate early-dispatch phase does not contain task IDs.",
        "",
        "| dispatch | c → a | before c→a-end µs | after c→a-end µs | improvement µs | "
        "b early before→after | a early before→after | a still on Observed edge |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    improvements: list[float] = []
    protected_success: list[bool] = []
    suppressed_success: list[bool] = []
    for key in keys:
        old, new = before_by_key[key], after_by_key[key]
        old_window = float(old["producer_to_protected_end_us"])
        new_window = float(new["producer_to_protected_end_us"])
        improvement = old_window - new_window
        improvements.append(improvement)
        protected_ok = new["protected_status"]["observed"] in {"full", "partial"}
        suppressed_ok = new["suppressed_status"]["observed"] == "none"
        protected_success.append(protected_ok)
        suppressed_success.append(suppressed_ok)
        edge = "yes" if new["protected_edge_on_observed_path"] else "no (path changed)"
        lines.append(
            f"| `{new['dispatch']}` | {_md('/'.join(new['producer_names']))} → "
            f"{_md('/'.join(new['protected_names']))} | {old_window:.3f} | {new_window:.3f} | "
            f"{improvement:.3f} | {_status_text(old['suppressed_status'])} → "
            f"{_status_text(new['suppressed_status'])} | {_status_text(old['protected_status'])} → "
            f"{_status_text(new['protected_status'])} | {edge} |"
        )

    other_siblings = [
        f"`{snapshot['dispatch']}`: {', '.join(snapshot['other_early_siblings'])}"
        for snapshot in after_by_key.values()
        if snapshot["other_early_siblings"]
    ]
    if other_siblings:
        lines.extend(["", "Other early-dispatched consumers of `c` remain after this one-sibling edit:", ""])
        lines.extend(f"- {item}" for item in other_siblings)

    lines.extend(["", "## Decision", ""])
    if not all(suppressed_success):
        lines.append(
            "The dummy did **not** reliably suppress early dispatch for `b`; the rebuilt dependency or capture is "
            "inconsistent. Do not claim success."
        )
    elif not all(protected_success):
        lines.append(
            "`b` was suppressed, but `a` was **not actually early-dispatched for every selected occurrence**. "
            "Report the after artifacts and diagnose `a` before deciding to keep the edit."
        )
    elif all(improvement > 0.0 for improvement in improvements):
        lines.append(
            "Every L4-observed `c end → a end` interval became strictly smaller. **The modification has measured "
            "benefit; keep it after correctness checks and repeated DFX-off performance validation pass.**"
        )
    else:
        lines.append(
            "At least one L4-observed `c end → a end` interval did not become smaller. **No benefit was measured. "
            "Do you want me to restore the source change? I have not reverted it.**"
        )

    lines.extend(["", "## After artifacts", ""])
    lines.extend(f"- `{path}`" for path in after["artifacts"])
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze an add-dummy sibling-suppression experiment from level-4 L2 swimlanes."
    )
    parser.add_argument("build_dir", type=Path, help="Build/run directory containing fresh dfx_outputs")
    parser.add_argument("--suppressed", required=True, help="Exact name of sibling b that receives the dummy")
    parser.add_argument(
        "--protected", help="Exact name of Observed-path sibling a; auto-selected when unique"
    )
    parser.add_argument("--producer", help="Exact name of shared producer c; auto-selected when unique")
    parser.add_argument("--operator", help="Exact enclosing dispatch_program.json program")
    parser.add_argument("--rank", help="Select a rank; comparison defaults to the baseline rank")
    parser.add_argument("--suppressed-occurrence", type=int)
    parser.add_argument("--protected-occurrence", type=int)
    parser.add_argument("--producer-occurrence", type=int)
    parser.add_argument("--tol", type=int, default=2, help="Timestamp tolerance in clock ticks")
    parser.add_argument("--baseline-json", type=Path, help="Compare this run with a baseline snapshot")
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
    if args.tol < 0:
        print("error: --tol must be non-negative", file=sys.stderr)
        return 2

    try:
        baseline = None
        if args.baseline_json:
            baseline_path = args.baseline_json.expanduser().resolve()
            baseline = _read_json(baseline_path)
            if baseline.get("version") != SNAPSHOT_VERSION:
                raise ValueError(f"{baseline_path}: unsupported baseline version")
            if str(baseline.get("suppressed", "")).casefold() != args.suppressed.casefold():
                raise ValueError(
                    f"{baseline_path}: suppressed task {baseline.get('suppressed')!r} does not match "
                    f"{args.suppressed!r}"
                )
            payload = _comparison_payload(root, baseline, args.operator, args.rank, args.tol)
            report = _render_comparison(baseline, payload)
        else:
            payload = _baseline_payload(
                root,
                args.suppressed,
                args.operator,
                args.rank,
                args.suppressed_occurrence,
                args.protected,
                args.protected_occurrence,
                args.producer,
                args.producer_occurrence,
                args.tol,
            )
            report = _render_baseline(payload)
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
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
