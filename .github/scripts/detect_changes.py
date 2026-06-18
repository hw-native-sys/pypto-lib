# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Select the runnable kernel files a PR must exercise.

Given the list of files a PR changed (on stdin, one path per line), decide
which runnable scripts CI should execute. A "runnable" file is one that has a
`__main__` guard — the harness invokes it as `python <file> -p <platform>`.

Selection rules
---------------
1. Changed source files under ``examples/`` or ``models/`` pull in not just
   themselves but every file that (transitively) imports them. Model kernels
   are split across many sibling modules (``config``, ``rmsnorm``,
   ``qkv_proj_rope`` …) and a leaf change can break any downstream kernel, so
   we walk the reverse-import graph and run every runnable dependent.
2. A change under ``golden/`` (the shared validation harness) touches every
   kernel's correctness check, so it selects *all* runnable ``examples/`` files
   as a smoke test.

Imports in this repo are bare module names (``from qkv_proj_rope import ...``)
resolved against the running script's own directory, so the reverse-import
graph is built per-directory keyed by file basename.

The selected, deduplicated, sorted file list is printed space-separated on a
single line to stdout.
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict

# Directories whose .py files participate in the bare-name sibling-import graph.
SOURCE_ROOTS = ("examples", "models")

# `from <mod> import ...`  or  `import <mod>[ as ...]` — first dotted segment.
_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([A-Za-z_][\w]*)|import\s+([A-Za-z_][\w]*))",
    re.MULTILINE,
)


def _iter_source_files():
    for root in SOURCE_ROOTS:
        for dirpath, _, files in os.walk(root):
            for name in files:
                if name.endswith(".py") and "draft" not in name:
                    yield os.path.join(dirpath, name)


def _imported_modules(path):
    """Bare top-level module names imported by ``path``."""
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return set()
    return {m.group(1) or m.group(2) for m in _IMPORT_RE.finditer(text)}


def _has_main(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return "__main__" in fh.read()
    except OSError:
        return False


def build_reverse_graph():
    """Map each source file -> set of sibling files that import it.

    Bare imports resolve to a module in the importer's own directory, so an
    import of ``foo`` from a file in ``dir`` resolves to ``dir/foo.py``.
    """
    files = list(_iter_source_files())
    # (dir, basename-without-.py) -> file path, for resolving sibling imports.
    module_of = {
        (os.path.dirname(f), os.path.splitext(os.path.basename(f))[0]): f
        for f in files
    }
    reverse = defaultdict(set)
    for f in files:
        d = os.path.dirname(f)
        for mod in _imported_modules(f):
            target = module_of.get((d, mod))
            if target and target != f:
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


def main():
    changed = [line.strip() for line in sys.stdin if line.strip()]

    golden_touched = any(c.startswith("golden/") for c in changed)
    src_changed = [
        c
        for c in changed
        if c.endswith(".py")
        and "draft" not in os.path.basename(c)
        and c.split("/", 1)[0] in SOURCE_ROOTS
        and os.path.isfile(c)
    ]

    reverse = build_reverse_graph()

    selected = closure(src_changed, reverse)

    if golden_touched:
        selected.update(
            f for f in _iter_source_files() if f.startswith("examples/")
        )

    runnable = sorted(f for f in selected if os.path.isfile(f) and _has_main(f))
    print(" ".join(runnable))


if __name__ == "__main__":
    main()
