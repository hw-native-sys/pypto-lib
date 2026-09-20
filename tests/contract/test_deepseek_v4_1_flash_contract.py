# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Contract tests for the DeepSeek-V4.1 Flash decode compositions."""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

MODEL_DIR = Path(__file__).parents[2] / "models" / "deepseek_v4_1_flash"
module = sys.modules.get("pypto")
HAS_PYPTO = (
    not getattr(module, "__pypto_stub__", False)
    if module is not None
    else importlib.util.find_spec("pypto") is not None
)
requires_pypto = pytest.mark.skipif(not HAS_PYPTO, reason="model imports require real PyPTO")


def _tree(name: str) -> ast.Module:
    return ast.parse((MODEL_DIR / name).read_text())


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def _call_names(function: ast.FunctionDef) -> list[str]:
    names = []
    for node in sorted(ast.walk(function), key=lambda item: getattr(item, "lineno", 0)):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.append(node.func.attr)
    return names


def test_swa_composition_is_split_from_leaf():
    composition = _tree("decode_swa.py")
    calls = _call_names(_function(composition, "decode_swa"))
    assert [name for name in calls if name in ("attention_pre", "decode_attn_swa", "mhc_post")] == [
        "attention_pre",
        "decode_attn_swa",
        "mhc_post",
    ]
    assert (MODEL_DIR / "decode_attn_swa.py").is_file()


def test_swa_composition_keeps_leaf_call_contract():
    composition = _function(_tree("decode_swa.py"), "decode_swa")
    call = next(
        node
        for node in ast.walk(composition)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "decode_attn_swa"
    )
    parameters = _function(_tree("decode_attn_swa.py"), "decode_attn_swa").args.args
    overrides = {"x": "normalized_attention", "output": "attention_output"}
    assert [ast.unparse(argument) for argument in call.args] == [
        overrides.get(parameter.arg, parameter.arg) for parameter in parameters
    ]


def test_swa_composition_has_explicit_abi_and_ci_entry():
    source = (MODEL_DIR / "decode_swa.py").read_text()
    assert "# ci: no-sim" in source
    assert "# ci: a5" in source
    tree = ast.parse(source)
    for name in ("decode_swa", "decode_swa_rank"):
        annotations = [ast.unparse(argument.annotation) for argument in _function(tree, name).args.args]
        assert all(annotation != "pl.Tensor" for annotation in annotations)
    production = _function(tree, "decode_swa")
    assert all("pl.InOut" not in ast.unparse(argument.annotation) for argument in production.args.args)
    assert all("pl.Out" not in ast.unparse(argument.annotation) for argument in production.args.args)


def test_swa_rank_forwards_the_production_contract():
    tree = _tree("decode_swa.py")
    production = _function(tree, "decode_swa")
    rank = _function(tree, "decode_swa_rank")
    production_args = [argument.arg for argument in production.args.args]
    assert [argument.arg for argument in rank.args.args] == production_args
    call = next(node for node in ast.walk(rank) if isinstance(node, ast.Call))
    assert [ast.unparse(argument) for argument in call.args] == production_args


@requires_pypto
def test_swa_public_names_match_file_ownership():
    from models.deepseek_v4_1_flash import decode_attn_swa, decode_swa

    assert decode_attn_swa.decode_attn_swa
    assert decode_attn_swa.golden_decode_attn_swa
    assert not hasattr(decode_attn_swa, "decode_swa")
    assert not hasattr(decode_attn_swa, "golden_decode_swa")
    assert decode_swa.ATTENTION_GOLDEN is decode_attn_swa.golden_decode_attn_swa
    assert not hasattr(decode_swa, "GOLDEN")


def test_c2a_full_composition_is_split_from_leaf():
    composition = _tree("decode_c2a_full.py")
    calls = _call_names(_function(composition, "decode_c2a_full"))
    assert [
        name for name in calls if name in ("attention_pre", "decode_attn_c2a_full", "mhc_post")
    ] == ["attention_pre", "decode_attn_c2a_full", "mhc_post"]
    assert (MODEL_DIR / "decode_attn_c2a_full.py").is_file()


def test_c2a_full_composition_keeps_leaf_call_contract():
    composition = _function(_tree("decode_c2a_full.py"), "decode_c2a_full")
    call = next(
        node
        for node in ast.walk(composition)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "decode_attn_c2a_full"
    )
    parameters = _function(_tree("decode_attn_c2a_full.py"), "decode_attn_c2a_full").args.args
    overrides = {
        "x": "normalized_attention",
        "compressed_indices": "topk_indices",
        "output": "attention_output",
    }
    assert [ast.unparse(argument) for argument in call.args] == [
        overrides.get(parameter.arg, parameter.arg) for parameter in parameters
    ]


def test_c2a_full_composition_has_explicit_abi_and_ci_entry():
    source = (MODEL_DIR / "decode_c2a_full.py").read_text()
    assert "# ci: no-sim" in source
    assert "# ci: a5" in source
    tree = ast.parse(source)
    for name in ("decode_c2a_full", "decode_c2a_full_rank"):
        annotations = [ast.unparse(argument.annotation) for argument in _function(tree, name).args.args]
        assert all(annotation != "pl.Tensor" for annotation in annotations)
    production = _function(tree, "decode_c2a_full")
    assert all("pl.InOut" not in ast.unparse(argument.annotation) for argument in production.args.args)
    assert all("pl.Out" not in ast.unparse(argument.annotation) for argument in production.args.args)


def test_c2a_full_rank_forwards_the_production_contract():
    tree = _tree("decode_c2a_full.py")
    production = _function(tree, "decode_c2a_full")
    rank = _function(tree, "decode_c2a_full_rank")
    production_args = [argument.arg for argument in production.args.args]
    assert [argument.arg for argument in rank.args.args] == production_args
    call = next(node for node in ast.walk(rank) if isinstance(node, ast.Call))
    assert [ast.unparse(argument) for argument in call.args] == production_args


@requires_pypto
def test_c2a_full_public_names_match_file_ownership():
    from models.deepseek_v4_1_flash import decode_attn_c2a_full, decode_c2a_full

    assert decode_attn_c2a_full.decode_attn_c2a_full
    assert decode_attn_c2a_full.golden_decode_attn_c2a_full
    assert not hasattr(decode_attn_c2a_full, "decode_c2a_full")
    assert not hasattr(decode_attn_c2a_full, "golden_decode_c2a_full")
    assert decode_c2a_full.ATTENTION_GOLDEN is decode_attn_c2a_full.golden_decode_attn_c2a_full
    assert not hasattr(decode_c2a_full, "GOLDEN")


def test_c2a_reuse_composition_is_split_from_leaf():
    composition = _tree("decode_c2a_reuse.py")
    calls = _call_names(_function(composition, "decode_c2a_reuse"))
    assert [
        name for name in calls if name in ("attention_pre", "decode_attn_c2a_reuse", "mhc_post")
    ] == ["attention_pre", "decode_attn_c2a_reuse", "mhc_post"]
    imports = {
        node.module
        for node in _tree("decode_attn_c2a_reuse.py").body
        if isinstance(node, ast.ImportFrom)
    }
    assert "models.deepseek_v4_1_flash.decode_attn_c2a_full" in imports


def test_c2a_reuse_composition_keeps_leaf_call_contract():
    composition = _function(_tree("decode_c2a_reuse.py"), "decode_c2a_reuse")
    call = next(
        node
        for node in ast.walk(composition)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "decode_attn_c2a_reuse"
    )
    parameters = _function(_tree("decode_attn_c2a_reuse.py"), "decode_attn_c2a_reuse").args.args
    overrides = {
        "x": "normalized_attention",
        "compressed_indices": "topk_indices",
        "output": "attention_output",
    }
    assert [ast.unparse(argument) for argument in call.args] == [
        overrides.get(parameter.arg, parameter.arg) for parameter in parameters
    ]


def test_c2a_reuse_composition_has_explicit_abi_and_ci_entry():
    source = (MODEL_DIR / "decode_c2a_reuse.py").read_text()
    assert "# ci: no-sim" in source
    assert "# ci: a5" in source
    tree = ast.parse(source)
    for name in ("decode_c2a_reuse", "decode_c2a_reuse_rank"):
        annotations = [ast.unparse(argument.annotation) for argument in _function(tree, name).args.args]
        assert all(annotation != "pl.Tensor" for annotation in annotations)
    production = _function(tree, "decode_c2a_reuse")
    assert all("pl.InOut" not in ast.unparse(argument.annotation) for argument in production.args.args)
    assert all("pl.Out" not in ast.unparse(argument.annotation) for argument in production.args.args)


@requires_pypto
def test_c2a_reuse_public_names_and_source_owned_state_contract():
    from models.deepseek_v4_1_flash import decode_attn_c2a_reuse, decode_c2a_reuse, decode_common

    assert decode_attn_c2a_reuse.decode_attn_c2a_reuse
    assert decode_attn_c2a_reuse.golden_decode_attn_c2a_reuse
    assert not hasattr(decode_attn_c2a_reuse, "decode_c2a_reuse")
    assert not hasattr(decode_attn_c2a_reuse, "golden_decode_c2a_reuse")
    assert decode_c2a_reuse.ATTENTION_GOLDEN is decode_attn_c2a_reuse.golden_decode_attn_c2a_reuse
    comparator = decode_common.compare_unchanged("source_state")
    expected = torch.tensor([1.0, 2.0], dtype=torch.float32)
    assert comparator(expected.clone(), expected)[0]
    changed = expected.clone()
    changed[1] = 3.0
    assert not comparator(changed, expected)[0]


@requires_pypto
def test_c2a_reuse_inactive_suffix_is_compared_independently():
    from models.deepseek_v4_1_flash import decode_common

    calls = []

    def compare(actual, expected, **kwargs):
        calls.append(actual.shape[0])
        return torch.equal(actual, expected), "exact"

    comparator = decode_common.make_compare_attention_hidden(compare)
    expected = torch.zeros([1, 3, 4, 2])
    actual = expected.clone()
    assert comparator(actual, expected, inputs={"num_tokens": 2})[0]
    assert calls == [2, 1]
    actual[:, 2] = 1.0
    assert not comparator(actual, expected, inputs={"num_tokens": 2})[0]
