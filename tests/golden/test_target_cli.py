# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Target CLI declarations across runnable examples and model entries."""

import argparse
import ast
from pathlib import Path

import pytest

from golden.runner import _normalize_config


ROOT = Path(__file__).resolve().parents[2]
ENTRIES = []
for directory in ("examples", "models"):
    for path in sorted((ROOT / directory).rglob("*.py")):
        tree = ast.parse(path.read_text())
        declarations = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and any(isinstance(arg, ast.Constant) and arg.value == "--platform" for arg in node.args)
        ]
        for declaration in declarations:
            ENTRIES.append((path.relative_to(ROOT).as_posix(), tree, declaration))


@pytest.mark.parametrize("path,tree,platform", ENTRIES, ids=[entry[0] for entry in ENTRIES])
def test_entry_target_cli(path, tree, platform):
    """Exercise the actual target argument declarations without importing model kernels."""
    parser = argparse.ArgumentParser(prog=path)
    simulator = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument" and ast.dump(node.func.value) == ast.dump(platform.func.value)
        and any(isinstance(arg, ast.Constant) and arg.value == "--sim" for arg in node.args)
    )
    for node in (platform, simulator):
        parser.add_argument(
            *(ast.literal_eval(arg) for arg in node.args),
            **{keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords},
        )
    defaults = parser.parse_args([])
    assert defaults.execution_mode is False
    for backend in next(keyword.value for keyword in platform.keywords if keyword.arg == "choices").elts:
        for sim in (False, True):
            args = parser.parse_args(["-p", backend.value] + (["--sim"] if sim else []))
            cfg = _normalize_config(dict(backend_type=args.backend_type, execution_mode=args.execution_mode))
            assert cfg.platform == backend.value + ("sim" if sim else "")
