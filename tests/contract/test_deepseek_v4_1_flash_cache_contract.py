# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU contracts for bounded end-to-end E2M1 cache differences."""

import ast
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def compare_cache():
    # Load the actual helper without importing the model's PyPTO DSL functions.
    source = Path(__file__).resolve().parents[2] / "models/deepseek_v4_1_flash/decode_indexer.py"
    tree = ast.parse(source.read_text())
    helper = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "mapped_fp4_cache_compare")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["mapped_fp4_cache_compare"](
        "slots", mapping_shape=(2,), block_size=16, pool_name="cache",
    )


@pytest.mark.parametrize(
    "expected_byte,actual_byte,count,row,accepted",
    [
        pytest.param(0x33, 0x33, 1, 1, True, id="exact"),
        pytest.param(0x33, 0x34, 1, 1, True, id="adjacent-low"),
        pytest.param(0x33, 0x43, 1, 17, True, id="adjacent-high"),
        pytest.param(0x33, 0x44, 1, 1, True, id="adjacent-both"),
        pytest.param(0x33, 0x22, 1, 1, True, id="adjacent-decrease"),
        pytest.param(0xBB, 0xCC, 1, 1, True, id="negative-adjacent"),
        pytest.param(0x88, 0x88, 1, 1, True, id="negative-zero-exact"),
        pytest.param(0x33, 0x35, 1, 1, False, id="low-skips-level"),
        pytest.param(0x33, 0x53, 1, 1, False, id="high-skips-level"),
        pytest.param(0x33, 0x3B, 1, 1, False, id="low-sign-change"),
        pytest.param(0x33, 0xB3, 1, 1, False, id="high-sign-change"),
        pytest.param(0x00, 0x08, 1, 1, False, id="low-signed-zero"),
        pytest.param(0x80, 0x00, 1, 1, False, id="high-signed-zero"),
        # The mapped comparator permits round(512 * 0.005) == 3 changed bytes.
        pytest.param(0x33, 0x44, 3, 1, True, id="byte-budget-boundary"),
        pytest.param(0x33, 0x34, 4, 1, False, id="byte-budget-exceeded"),
        pytest.param(0x33, 0x34, 1, 0, False, id="unmapped-write"),
    ],
)
def test_mapped_fp4_cache_contract(compare_cache, expected_byte, actual_byte, count, row, accepted):
    expected = torch.full((2, 16, 1, 256), expected_byte, dtype=torch.uint8)
    actual = expected.clone()
    actual.reshape(32, 256)[row, :count] = actual_byte
    passed, detail = compare_cache(
        actual, expected, inputs={"slots": torch.tensor([1, 17], dtype=torch.int64)},
        actual_outputs={}, expected_outputs={}, rtol=0.0, atol=0.0,
    )
    assert passed is accepted, detail
    if not accepted:
        assert detail
