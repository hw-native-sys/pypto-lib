# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Enforce the one-way dependency from skills to public documentation."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
SKILL_LINK = re.compile(r"(?:\.claude|\.agents)/skills", re.IGNORECASE)
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)\s]+)\)")
REFERENCE_LINK = re.compile(r"^[ \t]{0,3}\[[^\]]+\]:[ \t]*(\S+)", re.MULTILINE)
FENCED_BLOCK = re.compile(
    r"(^```.*?^```[ \t]*$|^~~~.*?^~~~[ \t]*$)",
    re.MULTILINE | re.DOTALL,
)


def _public_markdown_files():
    yield ROOT / "README.md"
    yield from sorted((ROOT / "docs").rglob("*.md"))


def _check_public_dependencies() -> list[str]:
    failures = []
    for path in _public_markdown_files():
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if SKILL_LINK.search(line):
                relative = path.relative_to(ROOT)
                failures.append(
                    f"{relative}:{line_number}: public documentation must not depend on skills"
                )
    return failures


def _link_targets(markdown: str):
    for index, segment in enumerate(FENCED_BLOCK.split(markdown)):
        if index % 2:
            continue
        yield from MARKDOWN_LINK.findall(segment)
        yield from REFERENCE_LINK.findall(segment)


def _check_public_link_targets() -> list[str]:
    failures = []
    for path in _public_markdown_files():
        for raw_target in _link_targets(path.read_text(encoding="utf-8")):
            target = raw_target.strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or target.startswith("#"):
                continue
            if parsed.path.startswith("/"):
                failures.append(
                    f"{path.relative_to(ROOT)}: absolute documentation link: {raw_target}"
                )
                continue
            if not parsed.path:
                continue

            resolved = (path.parent / unquote(parsed.path)).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                failures.append(
                    f"{path.relative_to(ROOT)}: link escapes repository: {raw_target}"
                )
                continue
            if not resolved.exists():
                failures.append(
                    f"{path.relative_to(ROOT)}: missing link target: {raw_target}"
                )
    return failures


def _check_skill_doc_links() -> list[str]:
    failures = []
    for skill in sorted((ROOT / ".claude" / "skills").glob("*/SKILL.md")):
        text = skill.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.split("#", maxsplit=1)[0]
            if "docs/" not in target or "://" in target:
                continue
            resolved = (skill.parent / target).resolve()
            try:
                resolved.relative_to(ROOT / "docs")
            except ValueError:
                failures.append(
                    f"{skill.relative_to(ROOT)}: documentation link escapes docs/: {raw_target}"
                )
                continue
            if not resolved.is_file():
                failures.append(
                    f"{skill.relative_to(ROOT)}: missing documentation link: {raw_target}"
                )
    return failures


def main() -> int:
    failures = (
        _check_public_dependencies()
        + _check_public_link_targets()
        + _check_skill_doc_links()
    )
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    print("Validated public boundaries, link targets, and skill-to-doc links.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
