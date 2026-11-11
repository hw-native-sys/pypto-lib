# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Require every public Markdown page to appear exactly once in MkDocs nav."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"
MKDOCS_CONFIG = ROOT / "mkdocs.yml"


def _nav_targets(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for item in node:
            yield from _nav_targets(item)
    elif isinstance(node, dict):
        for value in node.values():
            yield from _nav_targets(value)


def _local_markdown_target(target: str) -> str | None:
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or not parsed.path.endswith(".md"):
        return None
    return parsed.path


def main() -> int:
    config = yaml.safe_load(MKDOCS_CONFIG.read_text(encoding="utf-8"))
    nav = config.get("nav")
    if not nav:
        print("mkdocs.yml must define nav", file=sys.stderr)
        return 1

    targets = [
        local
        for target in _nav_targets(nav)
        if (local := _local_markdown_target(target)) is not None
    ]
    counts = Counter(targets)
    failures = []

    for target, count in sorted(counts.items()):
        if count != 1:
            failures.append(f"duplicate nav target ({count} entries): {target}")
        if not (DOCS_DIR / target).is_file():
            failures.append(f"missing nav target: {target}")

    public_pages = {
        path.relative_to(DOCS_DIR).as_posix()
        for path in DOCS_DIR.rglob("*.md")
        if "_hooks" not in path.parts
    }
    omitted = sorted(public_pages - set(targets))
    failures.extend(f"page omitted from nav: {path}" for path in omitted)

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    print(f"Validated {len(public_pages)} documentation pages in mkdocs nav.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
