# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Source-level checks for the DeepSeek V4 dynamic-token Gate contract."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GATE_SOURCE = ROOT / "models" / "deepseek" / "v4-flash" / "gate.py"


def _module() -> ast.Module:
    return ast.parse(GATE_SOURCE.read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _parameter_names(function: ast.FunctionDef) -> tuple[str, ...]:
    return tuple(arg.arg for arg in function.args.args)


def _calls_named(function: ast.FunctionDef, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == name
    ]


def test_gate_derives_token_dimension_without_leaking_entry_symbol() -> None:
    tree = _module()
    gate = _function(tree, "gate")
    gate_test = _function(tree, "gate_test")

    assert "num_tokens" not in _parameter_names(gate)
    assert "num_tokens" not in _parameter_names(gate_test)

    token_aligned = {
        "x_mixed",
        "input_ids",
        "x_norm",
        "x_norm_i8",
        "x_norm_scale",
        "indices",
        "weights",
    }
    gate_annotations = {
        arg.arg: ast.unparse(arg.annotation)
        for arg in gate.args.args
        if arg.annotation is not None and arg.arg in token_aligned
    }
    assert gate_annotations == {name: "pl.Tensor" for name in token_aligned}

    entry_annotations = {
        arg.arg: ast.unparse(arg.annotation)
        for arg in gate_test.args.args
        if arg.annotation is not None and arg.arg in token_aligned
    }
    assert set(entry_annotations) == token_aligned
    assert all("T_DYN" in annotation for annotation in entry_annotations.values())

    dim_calls = _calls_named(gate, "pl.tensor.dim")
    assert len(dim_calls) == 1
    assert ast.unparse(dim_calls[0]) == "pl.tensor.dim(x_mixed, 0)"


def test_gate_test_binds_every_token_aligned_tensor() -> None:
    gate_test = _function(_module(), "gate_test")
    bound_names = {
        node.func.value.id
        for node in ast.walk(gate_test)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "bind_dynamic"
        and isinstance(node.func.value, ast.Name)
    }

    assert bound_names == {
        "x_mixed",
        "input_ids",
        "x_norm",
        "x_norm_i8",
        "x_norm_scale",
        "indices",
        "weights",
    }


def test_gate_fixture_uses_physical_token_shapes_instead_of_a_scalar_prefix() -> None:
    tree = _module()
    build_specs = _function(tree, "build_tensor_specs")
    source = ast.unparse(build_specs)

    assert "ScalarSpec('num_tokens'" not in source
    assert "[token_count, D]" in source
    assert "[token_count, TOPK]" in source
