# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Report task existence and producer-to-consumer reachability from deps.json."""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SNAPSHOT_VERSION = 1


@dataclass
class Task:
    task_id: str
    names: list[str]
    block_num: int
    kernel_ids: list[int]


@dataclass
class PathNode:
    task_id: str
    names: list[str]


@dataclass
class PathEdge:
    pred: str
    succ: str
    sources: list[str]


@dataclass
class PairResult:
    producer_occurrence: int | None
    consumer_occurrence: int | None
    producer: Task | None
    consumer: Task | None
    status: str
    path: list[PathNode]
    edges: list[PathEdge]


@dataclass
class DispatchResult:
    directory: str
    label: str
    program: str | None
    producer_matches: int
    consumer_matches: int
    pairs: list[PairResult]


@dataclass
class Graph:
    directory: Path
    label: str
    program: str | None
    tasks: dict[str, Task]
    successors: dict[str, set[str]]
    edge_sources: dict[tuple[str, str], set[str]]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc


def _task_id(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid task id {value!r}") from exc


def _task_sort_key(task_id: str) -> tuple[int, str]:
    try:
        return int(task_id), task_id
    except ValueError:
        return sys.maxsize, task_id


def _dispatch_sort_key(directory: Path) -> tuple[tuple[int, str], ...]:
    result = []
    for part in directory.parts:
        match = re.fullmatch(r"(?:rank|d)(\d+)", part)
        result.append((int(match.group(1)), part) if match else (sys.maxsize, part))
    return tuple(result)


def _program(directory: Path) -> str | None:
    marker = directory / "dispatch_program.json"
    if not marker.is_file():
        return None
    data = _read_json(marker)
    if not isinstance(data, dict):
        raise ValueError(f"{marker}: expected a JSON object")
    value = data.get("program")
    return str(value) if value is not None else None


def _program_matches(query: str, program: str | None) -> bool:
    if program is None:
        return False
    needle = query.casefold()
    candidate = program.casefold()
    return candidate == needle or Path(candidate).stem == Path(needle).stem


def _name_map(directory: Path) -> dict[str, str]:
    candidates = sorted(directory.glob("name_map*.json"), key=lambda path: (path.stat().st_mtime, path.name))
    if not candidates:
        raise ValueError(f"{directory}: missing name_map*.json")
    source = candidates[-1]
    data = _read_json(source)
    if isinstance(data, dict) and isinstance(data.get("callable_id_to_name"), dict):
        data = data["callable_id_to_name"]
    if not isinstance(data, dict):
        raise ValueError(f"{source}: expected a callable-id mapping")
    return {str(key): str(value) for key, value in data.items()}


def _task_names(info: dict[str, Any], names: dict[str, str]) -> tuple[list[str], list[int]]:
    kernel_ids = []
    result = []
    for raw_kernel_id in info.get("kernel_ids") or []:
        try:
            kernel_id = int(raw_kernel_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"task {info.get('task_id')}: invalid kernel id {raw_kernel_id!r}") from exc
        kernel_ids.append(kernel_id)
        if kernel_id < 0:
            continue
        name = names.get(str(kernel_id), f"cid{kernel_id}")
        if name not in result:
            result.append(name)
    if not result:
        result.append("dummy" if kernel_ids and all(kernel_id < 0 for kernel_id in kernel_ids) else "unknown")
    return result, kernel_ids


def _load_graph(directory: Path, root: Path) -> Graph:
    deps_path = directory / "deps.json"
    data = _read_json(deps_path)
    if not isinstance(data, dict):
        raise ValueError(f"{deps_path}: expected a JSON object")
    callable_names = _name_map(directory)
    tasks = {}
    for info in data.get("tasks") or []:
        if not isinstance(info, dict) or info.get("task_id") is None:
            continue
        task_id = _task_id(info["task_id"])
        names, kernel_ids = _task_names(info, callable_names)
        try:
            block_num = int(info.get("block_num", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{deps_path}: task {task_id} has invalid block_num") from exc
        tasks[task_id] = Task(task_id, names, block_num, kernel_ids)

    successors: dict[str, set[str]] = collections.defaultdict(set)
    edge_sources: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for edge in data.get("edges") or []:
        if not isinstance(edge, dict) or edge.get("pred") is None or edge.get("succ") is None:
            continue
        pred = _task_id(edge["pred"])
        succ = _task_id(edge["succ"])
        if pred not in tasks or succ not in tasks:
            continue
        successors[pred].add(succ)
        edge_sources[(pred, succ)].add(str(edge.get("source") or "unknown"))

    try:
        label = str(directory.relative_to(root)) or "."
    except ValueError:
        label = str(directory)
    return Graph(directory, label, _program(directory), tasks, dict(successors), dict(edge_sources))


def _dispatch_directories(root: Path) -> list[Path]:
    if root.is_file():
        if root.name != "deps.json":
            raise ValueError(f"{root}: expected deps.json or a build directory")
        return [root.parent]
    if not root.is_dir():
        raise ValueError(f"{root}: path does not exist")
    direct = root / "deps.json"
    if direct.is_file():
        return [root]
    return sorted({path.parent for path in root.rglob("deps.json")}, key=_dispatch_sort_key)


def _compiled_name_matches(query: str, name: str) -> bool:
    if query == name:
        return True
    suffix = name[len(query) :] if name.startswith(query) else ""
    return bool(suffix and re.fullmatch(r"(?:_(?:aic|aiv))?(?:_\d+)?", suffix))


def _matching_tasks(graph: Graph, query: str) -> list[Task]:
    result = [
        task
        for task in graph.tasks.values()
        if any(_compiled_name_matches(query, name) for name in task.names)
    ]
    return sorted(result, key=lambda task: _task_sort_key(task.task_id))


def _task_by_id(graph: Graph, task_id: str | None, role: str) -> Task | None:
    if task_id is None:
        return None
    normalized = _task_id(task_id)
    if normalized not in graph.tasks:
        raise ValueError(f"{graph.directory}: {role} task id {normalized} does not exist")
    return graph.tasks[normalized]


def _shortest_path(graph: Graph, producer: Task, consumer: Task) -> list[str]:
    if producer.task_id == consumer.task_id:
        return [producer.task_id]
    queue = collections.deque([producer.task_id])
    previous = {producer.task_id: None}
    while queue:
        current = queue.popleft()
        for successor in sorted(graph.successors.get(current, ()), key=_task_sort_key):
            if successor in previous:
                continue
            previous[successor] = current
            if successor == consumer.task_id:
                path = [successor]
                while previous[path[-1]] is not None:
                    path.append(previous[path[-1]])
                return list(reversed(path))
            queue.append(successor)
    return []


def _path_nodes(graph: Graph, path: list[str]) -> list[PathNode]:
    return [PathNode(task_id, graph.tasks[task_id].names) for task_id in path]


def _path_edges(graph: Graph, path: list[str]) -> list[PathEdge]:
    result = []
    for pred, succ in zip(path, path[1:]):
        sources = sorted(graph.edge_sources.get((pred, succ), {"unknown"}))
        result.append(PathEdge(pred, succ, sources))
    return result


def _pair_result(
    graph: Graph,
    producer: Task | None,
    consumer: Task | None,
    producer_occurrence: int | None,
    consumer_occurrence: int | None,
) -> PairResult:
    if producer is None:
        return PairResult(producer_occurrence, consumer_occurrence, None, consumer, "producer_absent", [], [])
    if consumer is None:
        return PairResult(producer_occurrence, consumer_occurrence, producer, None, "consumer_absent", [], [])
    path = _shortest_path(graph, producer, consumer)
    if not path:
        return PairResult(producer_occurrence, consumer_occurrence, producer, consumer, "no_path", [], [])
    status = "direct" if len(path) == 2 else "same_task" if len(path) == 1 else "transitive"
    return PairResult(
        producer_occurrence,
        consumer_occurrence,
        producer,
        consumer,
        status,
        _path_nodes(graph, path),
        _path_edges(graph, path),
    )


def _select_occurrence(tasks: list[Task], occurrence: int, role: str, directory: Path) -> Task:
    if occurrence < 0 or occurrence >= len(tasks):
        raise ValueError(
            f"{directory}: {role} occurrence {occurrence} is out of range for {len(tasks)} matches",
        )
    return tasks[occurrence]


def _analyze_graph(graph: Graph, args: argparse.Namespace) -> DispatchResult:
    producers = _matching_tasks(graph, args.producer)
    consumers = _matching_tasks(graph, args.consumer)
    producer_by_id = _task_by_id(graph, args.producer_task_id, "producer")
    consumer_by_id = _task_by_id(graph, args.consumer_task_id, "consumer")
    pairs = []

    if producer_by_id is not None or consumer_by_id is not None:
        if producer_by_id is None or consumer_by_id is None:
            raise ValueError("--producer-task-id and --consumer-task-id must be used together")
        pairs.append(_pair_result(graph, producer_by_id, consumer_by_id, None, None))
    elif args.producer_occurrence is not None or args.consumer_occurrence is not None:
        if args.producer_occurrence is None or args.consumer_occurrence is None:
            raise ValueError("--producer-occurrence and --consumer-occurrence must be used together")
        producer = _select_occurrence(producers, args.producer_occurrence, "producer", graph.directory)
        consumer = _select_occurrence(consumers, args.consumer_occurrence, "consumer", graph.directory)
        pairs.append(
            _pair_result(
                graph,
                producer,
                consumer,
                args.producer_occurrence,
                args.consumer_occurrence,
            ),
        )
    elif not producers:
        consumer = consumers[0] if consumers else None
        consumer_occurrence = 0 if consumers else None
        pairs.append(_pair_result(graph, None, consumer, None, consumer_occurrence))
    elif not consumers:
        pairs.append(_pair_result(graph, producers[0], None, 0, None))
    else:
        for producer_occurrence, producer in enumerate(producers):
            consumer_occurrence = producer_occurrence + args.pair_offset
            if 0 <= consumer_occurrence < len(consumers):
                pairs.append(
                    _pair_result(
                        graph,
                        producer,
                        consumers[consumer_occurrence],
                        producer_occurrence,
                        consumer_occurrence,
                    ),
                )
        if not pairs:
            pairs.append(_pair_result(graph, producers[0], None, 0, args.pair_offset))

    return DispatchResult(
        str(graph.directory),
        graph.label,
        graph.program,
        len(producers),
        len(consumers),
        pairs,
    )


def _name(task: Task | PathNode | None) -> str:
    return "missing" if task is None else "/".join(task.names)


def _md(value: str) -> str:
    return value.replace("|", "\\|").replace("`", "\\`")


def _path_text(pair: PairResult, max_nodes: int) -> str:
    nodes = pair.path
    if not nodes:
        return "-"
    selected = nodes
    omitted = 0
    if len(nodes) > max_nodes:
        keep = max_nodes // 2
        selected = [*nodes[:keep], *nodes[-keep:]]
        omitted = len(nodes) - len(selected)
    parts = [f"{_name(node)}[{node.task_id}]" for node in selected]
    if omitted:
        parts.insert(len(parts) // 2, f"... {omitted} task(s) ...")
    return " -> ".join(parts)


def _edge_text(pair: PairResult) -> str:
    if not pair.edges:
        return "-"
    return ", ".join(f"{edge.pred}->{edge.succ} ({'/'.join(edge.sources)})" for edge in pair.edges)


def _status_counts(results: list[DispatchResult]) -> collections.Counter[str]:
    return collections.Counter(pair.status for result in results for pair in result.pairs)


def _recommendations(results: list[DispatchResult]) -> list[str]:
    statuses = _status_counts(results)
    recommendations = []
    if statuses["producer_absent"]:
        recommendations.append(
            "At least one producer is absent. Check predicates, zero-trip loops, unused inline returns, "
            "and DCE before adding an edge.",
        )
    if statuses["consumer_absent"]:
        recommendations.append(
            "At least one consumer is absent. Confirm that the selected branch and dispatch should "
            "materialize it.",
        )
    if statuses["no_path"]:
        recommendations.append(
            "Both endpoints exist without a path. Repair the tensor direction/slice or add the narrowest "
            "explicit TaskId dependency.",
        )
    reachable = statuses["direct"] + statuses["transitive"] + statuses["same_task"]
    if reachable and not (statuses["producer_absent"] or statuses["consumer_absent"] or statuses["no_path"]):
        recommendations.append(
            "All selected pairs are reachable. Do not add a duplicate edge; use level-4 scheduler evidence "
            "to test for a core-holding wait or another resource stall.",
        )
    fingerprints = {
        tuple(sorted(collections.Counter(pair.status for pair in result.pairs).items())) for result in results
    }
    if len(fingerprints) > 1:
        recommendations.append(
            "Reachability differs across dispatches or ranks. Map the differing graph to dynamic occupancy "
            "or branch control before editing source.",
        )
    return recommendations


def _markdown(root: Path, args: argparse.Namespace, results: list[DispatchResult]) -> str:
    counts = _status_counts(results)
    lines = [
        "# Dependency Liveness Report",
        "",
        f"- Build root: `{root}`",
        f"- Producer query: `{_md(args.producer)}`",
        f"- Consumer query: `{_md(args.consumer)}`",
        f"- Pair offset: `{args.pair_offset}`",
        f"- Selected dispatches: `{len(results)}`",
        "",
        "## Summary",
        "",
        "| Direct | Transitive | Same task | No path | Producer absent | Consumer absent |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {counts['direct']} | {counts['transitive']} | {counts['same_task']} | "
            f"{counts['no_path']} | {counts['producer_absent']} | {counts['consumer_absent']} |"
        ),
        "",
        "| Dispatch | Program | Producer matches | Consumer matches | Reachable pairs | Total pairs |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        reachable = sum(pair.status in {"direct", "transitive", "same_task"} for pair in result.pairs)
        lines.append(
            f"| `{_md(result.label)}` | `{_md(result.program or '-')}` | {result.producer_matches} | "
            f"{result.consumer_matches} | {reachable} | {len(result.pairs)} |",
        )

    lines.extend(["", "## Pair details", ""])
    for result in results:
        lines.extend([f"### `{_md(result.label)}`", ""])
        for pair in result.pairs:
            producer_occurrence = "-" if pair.producer_occurrence is None else str(pair.producer_occurrence)
            consumer_occurrence = "-" if pair.consumer_occurrence is None else str(pair.consumer_occurrence)
            producer_id = "-" if pair.producer is None else pair.producer.task_id
            consumer_id = "-" if pair.consumer is None else pair.consumer.task_id
            lines.extend(
                [
                    (
                        f"- Pair `{producer_occurrence} -> {consumer_occurrence}`: **{pair.status}**; "
                        f"producer `{_md(_name(pair.producer))}[{producer_id}]`; "
                        f"consumer `{_md(_name(pair.consumer))}[{consumer_id}]`"
                    ),
                    f"  - Shortest path: `{_md(_path_text(pair, args.max_path_nodes))}`",
                    f"  - Edge sources: `{_md(_edge_text(pair))}`",
                ],
            )

    lines.extend(["", "## Decision", ""])
    for recommendation in _recommendations(results):
        lines.append(f"- {recommendation}")
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path, help="build root, dispatch directory, or deps.json")
    parser.add_argument("--producer", required=True, help="producer callable or source name")
    parser.add_argument("--consumer", required=True, help="consumer callable or source name")
    parser.add_argument("--operator", help="exact enclosing dispatch program")
    parser.add_argument(
        "--pair-offset",
        type=int,
        default=0,
        help="consumer occurrence minus producer occurrence",
    )
    parser.add_argument("--producer-occurrence", type=int)
    parser.add_argument("--consumer-occurrence", type=int)
    parser.add_argument("--producer-task-id")
    parser.add_argument("--consumer-task-id")
    parser.add_argument("--max-path-nodes", type=int, default=16)
    parser.add_argument("--require-all-reachable", action="store_true")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.max_path_nodes < 4:
        raise ValueError("--max-path-nodes must be at least 4")
    if (args.producer_occurrence is None) != (args.consumer_occurrence is None):
        raise ValueError("--producer-occurrence and --consumer-occurrence must be used together")
    if (args.producer_task_id is None) != (args.consumer_task_id is None):
        raise ValueError("--producer-task-id and --consumer-task-id must be used together")

    root = args.build.resolve()
    directories = _dispatch_directories(root)
    if not directories:
        raise ValueError(f"{root}: no deps.json found")
    programs = {
        program for program in (_program(directory) for directory in directories) if program is not None
    }
    if args.operator:
        directories = [
            directory for directory in directories if _program_matches(args.operator, _program(directory))
        ]
        if not directories:
            raise ValueError(f"{root}: no dispatch program matches {args.operator!r}")
    elif len(directories) > 1 and (
        len(programs) != 1 or any(_program(directory) is None for directory in directories)
    ):
        rendered = ", ".join(sorted(programs)) or "no dispatch_program.json markers"
        raise ValueError(f"{root}: multiple dispatch programs are present ({rendered}); pass --operator")

    graphs = [_load_graph(directory, root) for directory in directories]
    results = [_analyze_graph(graph, args) for graph in graphs]
    markdown = _markdown(root, args, results)
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
    else:
        sys.stdout.write(markdown)
    if args.json_out:
        payload = {
            "version": SNAPSHOT_VERSION,
            "build_root": str(root),
            "producer_query": args.producer,
            "consumer_query": args.consumer,
            "pair_offset": args.pair_offset,
            "results": [asdict(result) for result in results],
        }
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.require_all_reachable:
        statuses = _status_counts(results)
        failing_statuses = ("producer_absent", "consumer_absent", "no_path")
        if not results or any(statuses[status] for status in failing_statuses):
            return 3
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
