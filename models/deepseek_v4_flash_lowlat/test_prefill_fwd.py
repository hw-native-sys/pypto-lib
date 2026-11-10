# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib
import sys
from types import ModuleType

import pytest

from models.deepseek_v4_flash_lowlat.prefill_fwd import load_prefill_program


def test_prefill_uses_ordered_kv_without_changing_existing_module(monkeypatch):
    import pypto

    if getattr(pypto, "__pypto_stub__", False):
        pytest.skip("requires the real PyPTO frontend")
    existing = ModuleType("qkv_proj_rope")
    existing.kv_proj_rope = object()
    monkeypatch.setitem(sys.modules, "qkv_proj_rope", existing)
    old_projection = existing.kv_proj_rope
    old_path, old_argv = list(sys.path), list(sys.argv)
    _, module = load_prefill_program()
    ordered = importlib.import_module("models.deepseek_v4_flash_lowlat.drafter.qkv_proj_rope").kv_proj_rope
    for name in ("prefill_attention_swa_cp", "prefill_attention_csa_cp", "prefill_attention_hca_cp"):
        assert getattr(module, name)._func.__globals__["kv_proj_rope"] is ordered
    assert sys.modules["qkv_proj_rope"] is existing
    assert existing.kv_proj_rope is old_projection
    assert sys.path == old_path and sys.argv == old_argv


def test_prefill_rejects_incompatible_ordered_kv_geometry(monkeypatch):
    import pypto

    if getattr(pypto, "__pypto_stub__", False):
        pytest.skip("requires the real PyPTO frontend")
    ordered = importlib.import_module("models.deepseek_v4_flash_lowlat.drafter.qkv_proj_rope")
    monkeypatch.setattr(ordered, "KV_OK", ordered.KV_OK + 1)
    old_path, old_argv = list(sys.path), list(sys.argv)
    with pytest.raises(RuntimeError, match="KV_OK"):
        load_prefill_program()
    assert sys.path == old_path and sys.argv == old_argv
