# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Select the runnable kernel files CI must exercise.

Given a list of changed files on stdin, decide
which runnable scripts CI should execute. A "runnable" file is one that has a
`__main__` guard — the harness invokes it as `python <file> -p <platform>`.

Selection rules
---------------
1. A changed file under ``models/`` pulls in not just itself but every file
   that (transitively) imports it. Model kernels are split across many sibling
   modules (``config``, ``rmsnorm``, ``qkv_proj_rope`` …) and a leaf change can
   break any downstream kernel, so we walk the reverse-import graph and run
   every runnable dependent. Only ``models/`` needs this: any ``examples/``
   change is already covered by rule 2's full-suite run.
2. Any runtime-affecting change **outside** ``models/`` selects *all* runnable
   ``examples/`` files as a smoke test. Documentation and documentation-only
   control files are explicitly exempt because they cannot change generated
   kernels or runtime behavior. Unknown paths remain runtime-affecting so the
   safe default is still the full smoke suite.

Platform ownership comes from ``.github/ci_platforms.json``. Directory rules
use the longest matching prefix; an unmapped source directory is an error.
Runnable entries have a script guard and a platform CLI, or implementation-local
pytest tests. Golden-only helpers and checkpoint-specific drivers are excluded.
Simulator selection additionally excludes ``# ci: no-sim`` entries.

Imports resolve two ways. A bare module name (``from qkv_proj_rope import ...``)
resolves against the importer's own directory, so that part of the graph is
keyed by file basename. A repository-rooted name
(``from models.deepseek_v4_1_flash.config import ...``, or
``from models.deepseek_v4_1_flash import config``) resolves against the
repository root. Both spellings are in use, and a directory written entirely in
the rooted spelling has no sibling-name edges at all, so ignoring it would
silently select nothing for every leaf change there.

The selected, deduplicated, sorted file list is printed space-separated on a
single line to stdout.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
import sys
from collections import defaultdict

# Directories whose .py files participate in the bare-name sibling-import graph.
SOURCE_ROOTS = ("examples", "models")

PLATFORM_CONFIG = Path(__file__).resolve().parents[1] / "ci_platforms.json"


def platforms_for(path):
    """Resolve platform families without importing model configuration."""
    rules = json.loads(PLATFORM_CONFIG.read_text())
    matches = [prefix for prefix in rules if path.startswith(prefix)]
    if not matches:
        raise ValueError(f"No CI platform rule for {path}")
    return rules[max(matches, key=len)]


def has_pytest_entry(path):
    """Recognize implementation-local precision tests without importing kernels."""
    return bool(re.search(r"^\s*def test_\w+\(", _read(path), re.MULTILINE))


def supports_platform(path, platform):
    family = platform.removesuffix("sim")
    return family in platforms_for(path) and not (
        platform.endswith("sim")
        and re.search(r"^#\s*ci:\s*no-sim\b", _read(path), re.MULTILINE)
    )


def all_entries(platform, root=None):
    return sorted(path for path in _iter_source_files()
                  if (root is None or path.startswith(root + "/"))
                  and _has_main(path) and supports_platform(path, platform))


# Paths that can change documentation or repository guidance but cannot change
# generated kernels or runtime behavior. Keep this list explicit: an unknown
# path must continue to select the full examples suite.
NON_RUNTIME_FILES = {
    ".gitignore",
    ".pre-commit-config.yaml",
    "AGENTS.md",
    "README.md",
    "mkdocs.yml",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/workflows/docs.yml",
    "tests/lint/check_docs_nav.py",
    "tests/lint/check_english_only.py",
    "tests/lint/check_public_docs.py",
}
NON_RUNTIME_PREFIXES = (
    "docs/",
    "tests/docs/",
)

def _iter_source_files():
    for root in SOURCE_ROOTS:
        for dirpath, _, files in os.walk(root):
            for name in files:
                if name.endswith(".py") and not name.endswith("_draft.py"):
                    yield os.path.join(dirpath, name)


def _read(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


def _root_names():
    """Top-level names importable from the repository root.

    A sibling file cannot be the target of a bare import whose name also exists
    at the root: these scripts put the repository root ahead of their own
    directory on ``sys.path``, so the root entry wins. ``from golden import ...``
    next to a ``models/pkg/golden.py`` is the live instance of that shadowing.
    """
    names = set()
    for entry in os.listdir("."):
        if entry.endswith(".py"):
            names.add(entry[: -len(".py")])
        elif os.path.isfile(os.path.join(entry, "__init__.py")):
            names.add(entry)
    return names


def _candidates(dotted):
    """Resolution candidates for one imported dotted name.

    ``("sibling", head)`` is the importer-directory lookup. ``("rooted", path)``
    is the repository-root lookup, emitted only under a source root so that
    third-party imports cost nothing.
    """
    head = dotted.partition(".")[0]
    found = {("sibling", head)}
    if head in SOURCE_ROOTS:
        found.add(("rooted", os.path.join(*dotted.split(".")) + ".py"))
    return found


def _imported_names(path):
    """Resolution candidates for every module ``path`` imports.

    ``from pkg import name`` may name a submodule or an attribute of ``pkg``,
    so both readings are offered and the caller keeps whichever resolves to a
    real file.
    """
    try:
        tree = ast.parse(_read(path))
    except SyntaxError:
        return set()
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found |= _candidates(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found |= _candidates(node.module)
            for alias in node.names:
                found |= _candidates(f"{node.module}.{alias.name}")
    return found


def has_platform_cli(path):
    """Recognize the platform argument required by the generic script runners."""
    try:
        tree = ast.parse(_read(path))
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and any(isinstance(arg, ast.Constant) and arg.value == "--platform" for arg in node.args)
        for node in ast.walk(tree)
    )


def _has_main(path):
    source = _read(path)
    script_guard = "__main__" in source or "if __name__ == _SCRIPT_ENTRY_POINT:" in source
    return has_pytest_entry(path) or (script_guard and has_platform_cli(path))


def build_reverse_graph():
    """Map each source file -> set of files that import it.

    A bare import of ``foo`` from a file in ``dir`` resolves to ``dir/foo.py``;
    a rooted import of ``models.pkg.foo`` resolves to ``models/pkg/foo.py`` from
    anywhere in the tree.
    """
    files = list(_iter_source_files())
    # (dir, basename-without-.py) -> file path, for resolving sibling imports.
    module_of = {
        (os.path.dirname(f), os.path.splitext(os.path.basename(f))[0]): f
        for f in files
    }
    rooted = set(files)
    shadowed = _root_names()
    reverse = defaultdict(set)
    for f in files:
        d = os.path.dirname(f)
        for kind, name in _imported_names(f):
            if kind == "sibling":
                target = None if name in shadowed else module_of.get((d, name))
            else:
                target = name
            if target in rooted and target != f:
                reverse[target].add(f)
    return reverse


def closure(seeds, reverse):
    """All files reachable from ``seeds`` by following reverse-import edges."""
    seen = set()
    stack = list(seeds)
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(reverse.get(cur, ()))
    return seen


def _is_non_runtime_path(path):
    if path in NON_RUNTIME_FILES or path.startswith(NON_RUNTIME_PREFIXES):
        return True
    return path.endswith(".md") and path.startswith((".claude/", ".agents/"))


def has_runtime_impact(changed):
    """Whether the change set can affect generated kernels or runtime behavior."""
    return any(path and not _is_non_runtime_path(path) for path in changed)


def select_runnable(changed, platform="a2a3"):
    """Select affected entries, then filter by directory platform ownership."""
    changed = [path for path in changed if path]
    seeds = [path for path in changed if path.startswith("models/")
             and path.endswith(".py") and not path.endswith("_draft.py")
             and os.path.isfile(path)]
    selected = closure(seeds, build_reverse_graph())
    for path in changed:
        if path.startswith("models/") and path.endswith("/conftest.py"):
            selected.update(f for f in _iter_source_files()
                            if f.startswith(os.path.dirname(path) + "/"))
    if any(not path.startswith("models/") and not _is_non_runtime_path(path) for path in changed):
        selected.update(f for f in _iter_source_files() if f.startswith("examples/"))
    return sorted(f for f in selected if os.path.isfile(f) and _has_main(f)
                  and supports_platform(f, platform))


def select_a5(changed):
    return select_runnable(changed, "a5")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-impact", action="store_true")
    parser.add_argument("--a5-entries", action="store_true")
    parser.add_argument("--platform", choices=("a2a3", "a2a3sim", "a5", "a5sim"), default="a2a3")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--root", choices=SOURCE_ROOTS)
    parser.add_argument("--filter", action="store_true", help="filter supplied entry paths by platform")
    args = parser.parse_args()
    changed = [line.strip() for line in sys.stdin if line.strip()] if not args.all else []
    platform = "a5" if args.a5_entries else args.platform
    if args.runtime_impact:
        print("true" if has_runtime_impact(changed) else "false")
        return
    if args.all:
        selected = all_entries(platform, args.root)
    elif args.filter:
        selected = [path for path in changed if supports_platform(path, platform)]
    else:
        selected = select_runnable(changed, platform)
    print(" ".join(selected))


if __name__ == "__main__":
    main()
