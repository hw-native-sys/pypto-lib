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

``--a5-entries`` applies rule 1 to a different entry marker. Files tagged
``# ci: a5`` are device entries the A2/A3 and simulator sweeps must not pick
up, so they carry no ``__main__`` sentinel and rule 1 cannot see them; the
dedicated A5 pull-request job asks for them by name. Rule 2 has no counterpart
there, because A5 coverage for changes outside ``models/`` is the nightly
sweep's job.

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

import ast
import os
import re
import sys
from collections import defaultdict

# Directories whose .py files participate in the bare-name sibling-import graph.
SOURCE_ROOTS = ("examples", "models")

A5_ONLY_MODEL_PREFIXES = (
    "models/deepseek_v4_pro/",
    "models/deepseek_v4_1_flash/",
)

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

# Device entries claimed by the A5 pull-request job, spelled like the
# repository's other CI hints (`# ci: no-sim`, `# ci: devices=N`).
_A5_ENTRY_RE = re.compile(r"^#\s*ci:\s*a5\s*$", re.MULTILINE)


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


def _has_main(path):
    return "__main__" in _read(path)


def _is_a5_entry(path):
    return bool(_A5_ENTRY_RE.search(_read(path)))


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


def select_runnable(changed):
    """Return runnable scripts required for the supplied changed paths."""
    changed = [path for path in changed if path]

    # Documentation-only paths select no device work. Any unknown or explicitly
    # runtime-affecting non-model path still selects the full examples suite.
    non_models_touched = any(
        not path.startswith("models/") and not _is_non_runtime_path(path)
        for path in changed
    )
    # Only models/ uses the reverse-import graph: a changed examples/ file is
    # already covered by the full-suite run above, so it needs no closure here.
    # A5-only model families have separate device coverage.
    models_changed = [
        c
        for c in changed
        if c.endswith(".py")
        and not c.endswith("_draft.py")
        and c.startswith("models/")
        and not c.startswith(A5_ONLY_MODEL_PREFIXES)
        and os.path.isfile(c)
    ]

    reverse = build_reverse_graph()

    selected = closure(models_changed, reverse)

    if non_models_touched:
        selected.update(
            f for f in _iter_source_files() if f.startswith("examples/")
        )

    return sorted(
        f for f in selected
        if not f.startswith(A5_ONLY_MODEL_PREFIXES)
        and os.path.isfile(f) and _has_main(f)
    )


def select_a5(changed):
    """Return the ``# ci: a5`` device entries required for the changed paths.

    Rule 1 against the A5 entry marker: seed with the changed model sources and
    keep the tagged entries their reverse-import closure reaches. There is no
    per-directory exclusion here — the marker is what makes a file A5 work, so a
    change that reaches no tagged entry selects nothing and the job never starts.
    """
    seeds = [
        c
        for c in changed
        if c.endswith(".py")
        and not c.endswith("_draft.py")
        and c.startswith("models/")
        and os.path.isfile(c)
    ]
    reverse = build_reverse_graph()
    return sorted(f for f in closure(seeds, reverse) if _is_a5_entry(f))


def main():
    changed = [line.strip() for line in sys.stdin if line.strip()]
    if "--runtime-impact" in sys.argv[1:]:
        print("true" if has_runtime_impact(changed) else "false")
        return
    if "--a5-entries" in sys.argv[1:]:
        print(" ".join(select_a5(changed)))
        return
    print(" ".join(select_runnable(changed)))


if __name__ == "__main__":
    main()
