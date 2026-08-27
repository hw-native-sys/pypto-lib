# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Detect main-orchestration drift in the additive EPLB decode benchmark."""

from __future__ import annotations

import ast
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "deepseek_v4_flash_mtp"
_MAIN_PATH = _MODEL_DIR / "decode_fwd.py"
_EPLB_PATH = _MODEL_DIR / "eplb_decode_logits.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _function(path: Path, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in _tree(path).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _integer_assignment(path: Path, name: str) -> int:
    assignment = next(
        node
        for node in _tree(path).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )
    assert isinstance(assignment.value, ast.Constant)
    assert isinstance(assignment.value.value, int)
    return assignment.value.value


def _bare_call_sequence(path: Path, function_name: str) -> list[str]:
    function = _function(path, function_name)
    calls = [
        (node.lineno, node.col_offset, node.func.id)
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    return [name for _, _, name in sorted(calls)]


def _main_compute_sequence() -> list[str]:
    ignored_host_prep = {"build_decode_metadata", "pack_x_hc"}
    return [
        "lm_head_tail" if name == "lm_head_with_sampling" else name
        for name in _bare_call_sequence(_MAIN_PATH, "decode_fwd")
        if name not in ignored_host_prep
    ]


def _eplb_compute_sequence() -> list[str]:
    return [
        "lm_head_tail" if name == "lm_head_core" else name
        for name in _bare_call_sequence(_EPLB_PATH, "eplb_decode_logits_inline")
        if name != "clear_lm_head_signals"
    ]


def test_eplb_decode_tracks_main_layer_topology() -> None:
    for name in ("FWD_NUM_LAYERS", "CSA_NUM_LAYERS", "HCA_NUM_LAYERS"):
        assert _integer_assignment(_EPLB_PATH, name) == _integer_assignment(_MAIN_PATH, name)


def test_eplb_decode_tracks_main_compute_stage_skeleton() -> None:
    expected = [
        "attention_swa",
        "moe",
        "attention_swa",
        "moe",
        "attention_csa",
        "moe",
        "attention_hca",
        "moe",
        "attention_csa",
        "moe",
        "clear_moe_signals",
        "hc_head",
        "rms_norm",
        "lm_head_tail",
    ]

    assert _main_compute_sequence() == expected
    assert _eplb_compute_sequence() == expected


def test_eplb_decode_specs_follow_the_host_abi_without_materializing_weights() -> None:
    host_names = [
        parameter.arg
        for parameter in _function(_EPLB_PATH, "l3_eplb_decode_logits").args.args
    ]
    build_specs = _function(_EPLB_PATH, "build_tensor_specs")
    ordered_names_assignment = next(
        node
        for node in ast.walk(build_specs)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ordered_names" for target in node.targets)
    )
    ordered_names = ast.literal_eval(ordered_names_assignment.value)
    appended_names = []
    for statement in build_specs.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            continue
        append_call = statement.value
        if not isinstance(append_call.func, ast.Attribute):
            continue
        if ast.unparse(append_call.func) != "specs.append":
            continue
        tensor_spec = append_call.args[0]
        if not isinstance(tensor_spec, ast.Call) or not tensor_spec.args:
            continue
        name = tensor_spec.args[0]
        if isinstance(name, ast.Constant) and isinstance(name.value, str):
            appended_names.append(name.value)

    assert ordered_names + appended_names == host_names
