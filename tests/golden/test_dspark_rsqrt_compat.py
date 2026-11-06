# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise the actual RMS JIT compatibility helper without a device SDK."""

import ast
import inspect
import linecache
from pathlib import Path
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

pl = None  # Replaced with the selected SDK-shaped fake by each test.


def positional_scratch(x, scratch):
    return pl.tile.rsqrt(x, scratch)


def keyword_scratch(x, scratch):
    return pl.tile.rsqrt(x, tmp=scratch)


def missing_scratch(x, scratch):
    return pl.tile.rsqrt(x)


def _load_helper(pl):
    path = Path(__file__).resolve().parents[2] / "models" / "deepseek_v4_flash_lowlat" / "rmsnorm.py"
    tree = ast.parse(path.read_text())
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_jit_rsqrt_compatible")
    namespace = {"ast": ast, "inspect": inspect, "linecache": linecache, "textwrap": textwrap, "pl": pl}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["_jit_rsqrt_compatible"]


class RmsRsqrtCompatibilityTests(unittest.TestCase):
    def test_modern_api_keeps_either_scratch_spelling(self):
        def rsqrt(x, tmp=None, *, high_precision=False):
            return x, tmp, high_precision

        pl = SimpleNamespace(tile=SimpleNamespace(rsqrt=rsqrt), jit=SimpleNamespace(inline=lambda func: func))
        with patch.dict(globals(), {"pl": pl}):
            select = _load_helper(pl)
            for function in (positional_scratch, keyword_scratch):
                with self.subTest(function=function.__name__):
                    self.assertEqual(select(function)("input", "scratch"), ("input", "scratch", True))
            with self.assertRaisesRegex(RuntimeError, "scratch-rsqrt contract"):
                select(missing_scratch)

    def test_legacy_api_keeps_original_function(self):
        def rsqrt(x, tmp=None):
            return x, tmp

        pl = SimpleNamespace(tile=SimpleNamespace(rsqrt=rsqrt), jit=SimpleNamespace(inline=lambda func: func))
        with patch.dict(globals(), {"pl": pl}):
            selected = _load_helper(pl)(positional_scratch)
            self.assertIs(selected, positional_scratch)
            self.assertEqual(selected("input", "scratch"), ("input", "scratch"))


if __name__ == "__main__":
    unittest.main()
