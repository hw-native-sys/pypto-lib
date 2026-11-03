# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""High-precision model adapters accept both old and explicit rsqrt APIs."""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_MODELS = [
    "models/deepseek_v4_flash_dspark/qkv_proj_rope.py",
    "models/deepseek_v4_flash_lowlat/rmsnorm.py",
]


class _TileAnnotation:
    def __class_getitem__(cls, _args):
        return cls


@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("explicit_precision", [False, True])
def test_rsqrt_adapter_preserves_high_precision(model, explicit_precision):
    # Execute only the Python-level adapter selection, without importing model
    # weights/config or a compiler. Strict fake signatures catch an unsupported
    # keyword on old PyPTO and an omitted precision request on the new API.
    calls = []

    if explicit_precision:
        def rsqrt(value, tmp=None, *, high_precision=False):
            assert high_precision is True
            calls.append((value, tmp))
            return value
    else:
        def rsqrt(value, tmp=None):
            calls.append((value, tmp))
            return value

    source = ast.parse((_ROOT / model).read_text())
    selection = next(
        node for node in source.body
        if isinstance(node, ast.If)
        and any(isinstance(child, ast.FunctionDef) and child.name == "_rsqrt_high_precision"
                for child in node.body)
    )
    fake_pl = SimpleNamespace(
        Tile=_TileAnnotation,
        FP32=object(),
        tile=SimpleNamespace(rsqrt=rsqrt),
        jit=SimpleNamespace(inline=lambda fn: fn),
    )
    namespace = {"inspect": inspect, "pl": fake_pl,
                 "Q_ROPE_T_TILE": 16, "REDUCE_ROW_TILE": 16}
    exec(compile(ast.Module(body=[selection], type_ignores=[]), model, "exec"), namespace)
    value, workspace = object(), object()
    assert namespace["_rsqrt_high_precision"](value, workspace) is value
    assert calls == [(value, workspace)]
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "_rsqrt_high_precision" for node in ast.walk(source))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
