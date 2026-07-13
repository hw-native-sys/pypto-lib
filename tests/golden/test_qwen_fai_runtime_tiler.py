# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side coverage for the Qwen FAI runtime tiler."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
QWEN = ROOT / "models" / "qwen3" / "14b"


def _int_constant(name: str) -> int:
    tree = ast.parse((QWEN / "paged_attention_cce.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not defined")


def test_attention_entry_selects_flash_decode_from_tiler_metadata() -> None:
    source = (
        QWEN / "kernels" / "paged_attention_cce" / "attention" / "entry.cpp"
    ).read_text(encoding="utf-8")

    assert "if (tiling->needCoreNum != 0)" in source
    assert "if (tiling->batch == 1)" not in source
