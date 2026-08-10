# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "deepseek_v4_flash_mtp"
_LM_HEAD_PATH = _MODEL_DIR / "lm_head.py"
_MOE_PATH = _MODEL_DIR / "moe.py"

_MOE_IMPORT_PROBE = r"""
import json
import sys
import types


class _DslSymbol:
    def __getattr__(self, _name):
        return self

    def __getitem__(self, _key):
        return self

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return self


dsl_symbol = _DslSymbol()
pypto = types.ModuleType("pypto")
pypto.__path__ = []
language = types.ModuleType("pypto.language")
language.__path__ = []
language.__getattr__ = lambda _name: dsl_symbol
language.jit = dsl_symbol
distributed = types.ModuleType("pypto.language.distributed")
distributed.__getattr__ = lambda _name: dsl_symbol
ir = types.ModuleType("pypto.ir")
ir.__path__ = []
compiled_program = types.ModuleType("pypto.ir.distributed_compiled_program")
compiled_program.DistributedConfig = type("DistributedConfig", (), {})

pypto.language = language
pypto.ir = ir
language.distributed = distributed
ir.distributed_compiled_program = compiled_program
sys.modules["pypto"] = pypto
sys.modules["pypto.language"] = language
sys.modules["pypto.language.distributed"] = distributed
sys.modules["pypto.ir"] = ir
sys.modules["pypto.ir.distributed_compiled_program"] = compiled_program

for module_name in ("hc_pre", "hc_post", "gate", "expert_shared", "expert_routed"):
    module = types.ModuleType(module_name)
    setattr(module, module_name, lambda *_args, **_kwargs: None)
    sys.modules[module_name] = module

import moe

print(json.dumps({
    "ep": moe.EP,
    "experts_per_rank": moe.EXPERTS_PER_RANK,
    "global_experts": moe.N_EXPERTS_GLOBAL,
    "local_experts": moe.N_LOCAL,
    "config_global_experts": moe.M.n_routed_experts,
}))
"""


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _arg_names(function: ast.FunctionDef) -> list[str]:
    return [arg.arg for arg in function.args.args]


def _call_name(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    if not isinstance(statement.value.func, ast.Name):
        return None
    return statement.value.func.id


def _return_name(function: ast.FunctionDef) -> str | None:
    statement = function.body[-1]
    if not isinstance(statement, ast.Return) or not isinstance(statement.value, ast.Name):
        return None
    return statement.value.id


def _probe_moe_import(*argv: str) -> dict[str, int]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_MODEL_DIR)
    result = subprocess.run(
        [sys.executable, "-c", _MOE_IMPORT_PROBE, *argv],
        cwd=_MODEL_DIR,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(result.stdout)


def test_lm_head_wrapper_preserves_core_then_cleanup_order() -> None:
    tree = ast.parse(_LM_HEAD_PATH.read_text())
    core = _function(tree, "lm_head_core")
    cleanup = _function(tree, "clear_lm_head_signals")
    wrapper = _function(tree, "lm_head")
    expected_args = [
        "hidden_states", "lm_head_weight", "logit_row_indices", "logits",
        "hidden_window", "hidden_done", "logits_window", "logits_done",
        "group_base", "tp_rank", "done_epoch",
    ]

    assert _arg_names(core) == expected_args
    assert _arg_names(wrapper) == expected_args
    assert _arg_names(cleanup) == ["completion_anchor", "hidden_done", "logits_done"]
    assert _return_name(core) == "logits"
    assert _return_name(cleanup) == "completion_anchor"
    assert [_call_name(statement) for statement in wrapper.body[:-1]] == [
        "lm_head_core",
        "clear_lm_head_signals",
    ]

    core_call = wrapper.body[0].value
    cleanup_call = wrapper.body[1].value
    assert isinstance(core_call, ast.Call)
    assert isinstance(cleanup_call, ast.Call)
    assert [arg.id for arg in core_call.args if isinstance(arg, ast.Name)] == expected_args
    assert [arg.id for arg in cleanup_call.args if isinstance(arg, ast.Name)] == [
        "logits", "hidden_done", "logits_done",
    ]
    assert isinstance(wrapper.body[-1], ast.Return)
    assert isinstance(wrapper.body[-1].value, ast.Name)
    assert wrapper.body[-1].value.id == "logits"


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            ("--ep", "8"),
            {"ep": 8, "experts_per_rank": 32, "global_experts": 256, "local_experts": 32},
        ),
        (
            ("--ep=8", "--experts-per-rank", "16"),
            {"ep": 8, "experts_per_rank": 16, "global_experts": 128, "local_experts": 16},
        ),
    ],
    ids=("ep8-default", "ep8-eplb"),
)
def test_moe_import_time_topology(argv: tuple[str, ...], expected: dict[str, int]) -> None:
    topology = _probe_moe_import(*argv)

    assert {key: topology[key] for key in expected} == expected
    assert topology["config_global_experts"] == expected["global_experts"]


def test_moe_standalone_parser_accepts_experts_per_rank() -> None:
    tree = ast.parse(_MOE_PATH.read_text())
    option_names = {
        arg.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for arg in node.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    }

    assert "--experts-per-rank" in option_names
