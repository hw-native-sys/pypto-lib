# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check copyright headers in git-tracked source files."""

import subprocess
import sys
from pathlib import Path

PY_HEADER = """\
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------"""

EXCLUDED_PREFIXES = [
    "junk_models/",
    "junk_tensor_functions/",
    "projects/",
    "examples/docs/",
]

EXCLUDED_SUFFIXES = [
    "_build/",
]


def get_git_tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    )
    return [
        root / line
        for line in result.stdout.strip().split("\n")
        if line and (root / line).is_file()
    ]


def is_excluded(rel: str) -> bool:
    return any(rel.startswith(prefix) for prefix in EXCLUDED_PREFIXES) or any(
        suffix in rel for suffix in EXCLUDED_SUFFIXES
    )


def check_header(path: Path) -> tuple[bool, str]:
    expected_lines = PY_HEADER.split("\n")
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError) as e:
        return False, f"Error reading: {e}"
    if not content:
        return False, "File is empty"
    actual_lines = content.split("\n")
    if len(actual_lines) < len(expected_lines):
        return False, f"Too few lines ({len(actual_lines)} < {len(expected_lines)})"
    for i, expected in enumerate(expected_lines):
        if actual_lines[i] != expected:
            return False, f"Line {i + 1}: expected: {expected!r}, got: {actual_lines[i]!r}"
    return True, ""


def main() -> int:
    root = Path(__file__).resolve().parent.parent.parent
    all_files = get_git_tracked_files(root)

    py_files = [
        f
        for f in all_files
        if f.suffix == ".py" and not is_excluded(str(f.relative_to(root)))
    ]

    if not py_files:
        print("No Python files to check.")
        return 0

    print(f"Checking {len(py_files)} file(s)...")
    failed = []
    for f in py_files:
        ok, msg = check_header(f)
        if not ok:
            failed.append((f, msg))
            print(f"  FAIL {f.relative_to(root)}: {msg}")

    if failed:
        print(f"\n{len(failed)} file(s) missing or incorrect copyright header.")
        return 1

    print("All files have correct headers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
