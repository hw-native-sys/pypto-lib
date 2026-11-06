# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Check the ordinary Torch reference projections without importing a device SDK."""

import ast
from pathlib import Path
import unittest

import torch


def _load_projections():
    path = Path(__file__).resolve().parents[2] / "models/deepseek_v4_flash_lowlat/utils.py"
    tree = ast.parse(path.read_text())
    names = {"fp32_matmul_fp32", "bf16_matmul_fp32"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise RuntimeError("Reference projections are missing")
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(path), "exec"), namespace)
    return namespace


class ReferenceMatmulTests(unittest.TestCase):
    def test_batched_fp32_and_bf16_are_close_to_fp64_products(self):
        torch.manual_seed(1234)
        a, b = torch.randn(2, 3, 64), torch.randn(64, 7)
        methods = _load_projections()
        for name, rounded in (("fp32_matmul_fp32", False), ("bf16_matmul_fp32", True)):
            lhs = a.to(torch.bfloat16).double() if rounded else a.double()
            rhs = b.to(torch.bfloat16).double() if rounded else b.double()
            expected = (lhs @ rhs).float()
            for split in (1, 2, 4):
                with self.subTest(name=name, split=split):
                    actual = methods[name](a, b, split_k=split)
                    self.assertEqual(tuple(actual.shape), (2, 3, 7))
                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_rejects_invalid_shapes_and_split(self):
        methods = _load_projections()
        for name in ("fp32_matmul_fp32", "bf16_matmul_fp32"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                methods[name](torch.ones(2, 32), torch.ones(31, 4))
            with self.subTest(name=name), self.assertRaises(ValueError):
                methods[name](torch.ones(2, 32), torch.ones(32, 4), split_k=3)


if __name__ == "__main__":
    unittest.main()
