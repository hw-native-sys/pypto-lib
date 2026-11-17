# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Execution-structure gate: reject token loops that enclose the complete decoder stack."""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FunctionInfo:
    name: str
    file: Path
    calls: set[str] = field(default_factory=set)
    loops: list[ast.AST] = field(default_factory=list)
    node: ast.AST | None = None


def _parse(root: Path) -> dict[str, FunctionInfo]:
    functions: dict[str, FunctionInfo] = {}
    for path in sorted(root.rglob("*.py")):
        if path.name.startswith("_") and path.name not in ("__init__.py",):
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                info = FunctionInfo(node.name, path, node=node)
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        target = child.func
                        if isinstance(target, ast.Name):
                            info.calls.add(target.id)
                        elif isinstance(target, ast.Attribute):
                            info.calls.add(target.attr)
                    if isinstance(child, (ast.For, ast.AsyncFor, ast.While)):
                        info.loops.append(child)
                functions[node.name] = info
    return functions


def _iter_names(loop: ast.AST) -> set[str]:
    name = loop.target.id if isinstance(loop, ast.For) and isinstance(loop.target, ast.Name) else ""
    return {name} if name else set()


def _is_rank_loop(loop: ast.AST) -> bool:
    """True when the loop iterates the distributed rank axis (host-driver dispatch).

    A complete-model L3 host driver legitimately loops one entry call per rank
    (``for rank in pl.range(pld.world_size())``). That loop is not a prompt-token
    loop through the stack; the per-rank entry it calls is inspected separately,
    so both the rank dispatch and any token loop nested inside it are checked.
    """
    if not isinstance(loop, ast.For):
        return False
    try:
        source = ast.unparse(loop.iter)
    except Exception:  # pragma: no cover - ast.unparse is available on 3.10+
        return False
    return "world_size" in source


def _loop_body_calls(loop: ast.AST) -> set[str]:
    calls = set()
    for child in ast.walk(loop):
        if isinstance(child, ast.Call):
            target = child.func
            if isinstance(target, ast.Name):
                calls.add(target.id)
            elif isinstance(target, ast.Attribute):
                calls.add(target.attr)
    return calls


def _reachable_calls(functions: dict[str, FunctionInfo], start: str, seen: set[str] | None = None) -> set[str]:
    seen = set() if seen is None else seen
    if start in seen or start not in functions:
        return seen
    seen.add(start)
    for callee in functions[start].calls:
        _reachable_calls(functions, callee, seen)
    return seen


def _is_layer_work(functions: dict[str, FunctionInfo], name: str) -> bool:
    """A function performs layer work when it calls both attention and feed-forward operators."""
    reachable = _reachable_calls(functions, name)
    has_attention = any("mla_proj" in item or "attention_core" in item for item in reachable)
    has_feed_forward = any("moe_gate" in item or "expert_routed" in item or "expert_shared" in item for item in reachable)
    return has_attention and has_feed_forward


def _contains_layer_loop(functions: dict[str, FunctionInfo], name: str, seen: set[str] | None = None) -> bool:
    """True when the function stack contains a loop whose body performs layer work."""
    seen = set() if seen is None else seen
    if name in seen or name not in functions:
        return False
    seen.add(name)
    info = functions[name]
    for loop in info.loops:
        body_calls = _loop_body_calls(loop)
        for callee in body_calls:
            if callee in functions and _is_layer_work(functions, callee):
                return True
        # A5-literal loop bodies perform the attention and feed-forward work
        # inline through separate calls (no single per-layer helper).  Judge
        # the loop body collectively: the union of everything it reaches must
        # contain both attention and feed-forward operators.
        reachable: set[str] = set()
        for callee in body_calls:
            if callee in functions:
                reachable |= _reachable_calls(functions, callee)
        has_attention = any("mla_proj" in item or "attention_core" in item for item in reachable)
        has_feed_forward = any(
            "moe_gate" in item or "expert_routed" in item or "expert_shared" in item
            for item in reachable
        )
        if has_attention and has_feed_forward:
            return True
    for callee in info.calls:
        if _contains_layer_loop(functions, callee, seen):
            return True
    return False


def _token_loop_enclosing_stack(
    functions: dict[str, FunctionInfo], entry: str, seen: set[str] | None = None
) -> str | None:
    """Return a description when a loop above the layer stack also encloses it.

    The call graph may contain cycles across operator files, so the walk keeps a
    visited set; a cycle is not itself evidence of a token loop.
    """
    seen = set() if seen is None else seen
    if entry in seen:
        return None
    seen.add(entry)
    info = functions.get(entry)
    if info is None:
        return f"entry symbol {entry!r} not found in source"
    for loop in info.loops:
        body_calls = _loop_body_calls(loop)
        for callee in body_calls:
            if callee in functions and _contains_layer_loop(functions, callee):
                if _is_rank_loop(loop):
                    # Legitimate rank dispatch; any token loop must live below
                    # the per-rank entry and is checked by this recursion.
                    nested = _token_loop_enclosing_stack(functions, callee, seen)
                    if nested:
                        return nested
                    continue
                return (
                    f"loop over {_iter_names(loop) or '?'} in {entry} calls {callee}, "
                    "which reaches a decoder-layer loop; this serializes the complete stack per token"
                )
    for callee in info.calls:
        if callee in functions:
            nested = _token_loop_enclosing_stack(functions, callee, seen)
            if nested and not _is_layer_work(functions, callee):
                return nested
    return None


def _pto_tiles(build_root: Path, entry: str) -> tuple[int, int, list[str]]:
    """Return (max matmul rows, dynamic loops with matmul, evidence paths) from generated PTO IR.

    Complete-model L3 entries lower their rank kernels into
    ``next_levels/<rank>/ptoas/`` (distributed host drivers may also emit a
    top-level ``ptoas/``), so the search is recursive over the build directory.

    Dynamic orchestration loops carrying matmul work are expected after the
    scope-revision-2 migration: the token axis is a DynVar, so per-layer tiling
    and the attention query-chunk loop lower to runtime-bound loops. The
    structural prohibition is a token loop *above the decoder stack*, which the
    source call-graph check (`_token_loop_enclosing_stack`) proves directly; the
    compiled count is therefore evidence, not a failure condition.
    """
    candidates = sorted(build_root.glob(f"_jit_{entry}_*"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        return 0, 0, []
    # A complete entry compiles several programs (seed, per-mode layer prologs,
    # exchange, epilogue, head).  Which one is newest is an artifact of compile
    # order, and an embedding/seed artifact carries no matmul.  Evaluate every
    # matching artifact in the standard build root and keep the strongest
    # evidence, so the compiled leg is reproducible regardless of compile order.
    max_rows = 0
    dynamic_stack_loops = 0
    evidence: list[str] = []
    for build in candidates:
        files = sorted(build.rglob("*.pto"))
        evidence.extend(str(path) for path in files)
        build_rows = 0
        build_loops = 0
        for path in files:
            text = path.read_text()
            for match in re.finditer(r"tmatmul ins\([^\n]*?rows=(\d+)", text):
                build_rows = max(build_rows, int(match.group(1)))
            # Tiled scheduling inside a child InCore kernel is allowed (the GEMM M
            # tiling over a runtime row count is not a prompt-token loop). A loop
            # that encloses the complete decoder stack lowers into the
            # orchestration, so only orchestration-level IR is counted here; the
            # source call-graph check still rejects a token loop above the stack.
            if "next_levels" in path.parts:
                continue
            for loop in re.finditer(r"scf\.for .*? to %arg\d+ .*?\{(.*?)\n  \}", text, re.S):
                if "tmatmul" in loop.group(1):
                    build_loops += 1
        if build_rows > max_rows:
            max_rows = build_rows
            dynamic_stack_loops = build_loops
    return max_rows, dynamic_stack_loops, evidence


def run_check(
    model_root: Path,
    entry_file: Path,
    entry_symbol: str,
    build_root: Path,
    source_only: bool = False,
    complete_entry: bool = False,
) -> dict:
    functions = _parse(model_root)
    result = {
        "entry_file": str(entry_file),
        "entry_symbol": entry_symbol,
        "source_parse_ok": entry_symbol in functions,
        "tokenwise_stack": _token_loop_enclosing_stack(functions, entry_symbol),
        "layer_work": _is_layer_work(functions, entry_symbol) if entry_symbol in functions else False,
        "layer_loop": _contains_layer_loop(functions, entry_symbol) if entry_symbol in functions else False,
        "max_matmul_rows": 0,
        "dynamic_matmul_loops": 0,
        "ir_evidence": [],
        "passed": False,
    }
    max_rows, dynamic_loops, evidence = _pto_tiles(build_root, entry_symbol)
    result["max_matmul_rows"] = max_rows
    result["dynamic_matmul_loops"] = dynamic_loops
    result["ir_evidence"] = evidence
    if source_only:
        result["passed"] = result["source_parse_ok"] and result["tokenwise_stack"] is None
    else:
        # Missing source or generated evidence fails: unrecognized evidence is not a pass.
        # Dynamic orchestration loops are allowed under pure-DYN token axes
        # (per-layer tiling / query chunks). The source call-graph check above
        # remains the gate against a tokenwise complete stack, and the negative
        # controls exercise exactly that property.
        result["passed"] = (
            result["source_parse_ok"]
            and result["tokenwise_stack"] is None
            and max_rows > 1
        )
        if complete_entry:
            # A complete-model entry must additionally contain a decoder-layer
            # loop below the entry (positive evidence that the stack is batched
            # as layers, not serialized per token).
            result["passed"] = result["passed"] and result["layer_loop"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--entry-file", required=True)
    parser.add_argument("--entry-symbol", required=True)
    parser.add_argument("--build-root", default=str(Path(__file__).resolve().parents[2] / "build_output"))
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument(
        "--complete-entry",
        action="store_true",
        help="also require positive evidence of a decoder-layer loop below the entry",
    )
    args = parser.parse_args()
    result = run_check(
        Path(args.model_root),
        Path(args.entry_file),
        args.entry_symbol,
        Path(args.build_root),
        args.source_only,
        args.complete_entry,
    )
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
