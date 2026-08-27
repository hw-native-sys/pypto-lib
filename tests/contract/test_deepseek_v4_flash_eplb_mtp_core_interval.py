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
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORE_PATH = _REPO_ROOT / "models" / "deepseek_v4_flash_mtp" / "eplb_mtp_core.py"
_TREE = ast.parse(_CORE_PATH.read_text())


def _function(name: str) -> ast.FunctionDef:
    return next(node for node in _TREE.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _call_names(function: ast.FunctionDef) -> list[str]:
    names = []
    for statement in function.body:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.append(node.func.attr)
    return names


def test_eplb_mtp_core_interval_has_only_the_six_compute_leaves() -> None:
    compute = _function("eplb_mtp_core_logits")
    leaf_names = {"mtp_projection", "attention_swa", "moe", "hc_head", "rms_norm", "lm_head_core"}
    calls = [name for name in _call_names(compute) if name in leaf_names]

    assert calls == ["mtp_projection", "attention_swa", "moe", "hc_head", "rms_norm", "lm_head_core"]

    parameters = {parameter.arg for parameter in compute.args.args}
    excluded_parameters = {
        "embed_weight",
        "main_pre_hc_hidden",
        "tail_pre_hc_pool",
        "accepted_counts",
        "tail_slot_ids",
        "ori_block_table",
        "sampled_ids",
    }
    assert parameters.isdisjoint(excluded_parameters)
    assert {"hidden_states", "prev_pre_hc_hidden", "swa_slot_mapping", "swa_indices", "swa_lens"} <= parameters


def test_eplb_mtp_core_cleanup_is_a_separate_host_child() -> None:
    cleanup = _function("eplb_mtp_core_cleanup")
    cleanup_calls = [name for name in _call_names(cleanup) if name.startswith("clear_")]
    assert cleanup_calls == ["clear_moe_signals", "clear_lm_head_signals"]

    host = _function("l3_eplb_mtp_core")
    child_calls = [
        name
        for name in _call_names(host)
        if name in {"eplb_mtp_core_logits", "eplb_mtp_core_cleanup"}
    ]
    assert child_calls == ["eplb_mtp_core_logits", "eplb_mtp_core_cleanup"]

    rank_loops = [
        node
        for node in host.body
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "pl.range(pld.world_size())"
    ]
    assert len(rank_loops) == 2
    loop_calls = [
        [
            name
            for name in _call_names(loop)
            if name in {"eplb_mtp_core_logits", "eplb_mtp_core_cleanup"}
        ]
        for loop in rank_loops
    ]
    assert loop_calls == [["eplb_mtp_core_logits"], ["eplb_mtp_core_cleanup"]]


def test_eplb_mtp_core_specs_follow_the_host_abi_without_materializing_weights() -> None:
    host_names = [parameter.arg for parameter in _function("l3_eplb_mtp_core").args.args]
    build_specs = _function("build_tensor_specs")
    ordered_names_assignment = next(
        node
        for node in ast.walk(build_specs)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ordered_names" for target in node.targets)
    )
    ordered_names = ast.literal_eval(ordered_names_assignment.value)

    assert ordered_names == host_names


def test_eplb_mtp_core_lm_head_window_uses_shape_and_dtype_allocation() -> None:
    host = _function("l3_eplb_mtp_core")
    allocation = next(
        node.value
        for node in host.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "lm_head_logits_window_buf"
            for target in node.targets
        )
    )

    assert isinstance(allocation, ast.Call)
    assert ast.unparse(allocation.func) == "pld.alloc_window_buffer"
    assert [ast.unparse(argument) for argument in allocation.args] == ["[MAX_LOGIT_ROWS, LM_HEAD_VOCAB]"]
    assert {keyword.arg: ast.unparse(keyword.value) for keyword in allocation.keywords} == {
        "dtype": "pl.FP32"
    }


def test_eplb_mtp_core_uses_only_the_fixed_routing_adapter() -> None:
    source = _CORE_PATH.read_text()
    assert "configure_eplb_argv()" in source
    assert "replace_eplb_routing_specs(base_specs, active_tokens=num_tokens)" in source
    assert "routing_mode" not in source

    forbidden_calls = {
        "lookup_embedding",
        "pack_mtp_hidden",
        "build_swa_metadata",
        "greedy_sample",
        "lm_head_with_sampling",
        "mtp_decode_layer",
        "l3_mtp_decode_layer",
    }
    all_calls = set()
    for function_name in ("eplb_mtp_core_logits", "eplb_mtp_core_cleanup"):
        all_calls.update(_call_names(_function(function_name)))
    assert all_calls.isdisjoint(forbidden_calls)


def test_eplb_mtp_core_routes_with_hash_layer_zero() -> None:
    layer_id = next(
        node
        for node in _TREE.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "EPLB_MTP_LAYER_ID" for target in node.targets)
    )
    assert isinstance(layer_id.value, ast.Constant)
    assert layer_id.value.value == 0

    compute = _function("eplb_mtp_core_logits")
    moe_call = next(
        node
        for node in ast.walk(compute)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "moe"
    )
    assert ast.unparse(moe_call.args[-4]) == "pl.cast(EPLB_MTP_LAYER_ID, pl.INT32)"

    golden = _function("golden_eplb_mtp_core")
    golden_layer_id = next(
        node
        for node in ast.walk(golden)
        if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == "moe_tensors['layer_id']" for target in node.targets)
    )
    assert ast.unparse(golden_layer_id.value) == "EPLB_MTP_LAYER_ID"
