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
import importlib
import inspect
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "deepseek" / "v4-flash"
sys.path.insert(0, str(_MODEL_DIR))


def _function_node(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _called_function_names(function: ast.FunctionDef) -> list[str]:
    return [
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]


def test_lm_head_keeps_cleanup_out_of_the_reusable_core() -> None:
    path = _MODEL_DIR / "lm_head.py"
    core = _function_node(path, "lm_head_core")
    cleanup = _function_node(path, "clear_lm_head_signals")
    wrapper = _function_node(path, "lm_head")

    assert "clear_lm_head_signals" not in _called_function_names(core)
    core_write_destinations = {
        ast.unparse(node.args[0])
        for node in ast.walk(core)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write"
    }
    cleanup_write_destinations = {
        ast.unparse(node.args[0])
        for node in ast.walk(cleanup)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write"
    }
    assert core_write_destinations.isdisjoint({"hidden_done", "logits_done"})
    assert cleanup_write_destinations == {"hidden_done", "logits_done"}
    wrapper_calls = [
        name
        for name in _called_function_names(wrapper)
        if name in {"lm_head_core", "clear_lm_head_signals"}
    ]
    assert wrapper_calls == ["lm_head_core", "clear_lm_head_signals"]

    module = importlib.import_module("lm_head")
    assert inspect.signature(module.lm_head._func) == inspect.signature(module.lm_head_core._func)
    cleanup_parameters = tuple(inspect.signature(module.clear_lm_head_signals._func).parameters)
    assert cleanup_parameters == ("completion_anchor", "hidden_done", "logits_done")


def test_mtp_core_contains_only_the_requested_model_stages() -> None:
    path = _MODEL_DIR / "decode_mtp_core.py"
    function = _function_node(path, "mtp_decode_core_logits")
    module = importlib.import_module("decode_mtp_core")
    core_signature = inspect.signature(module.mtp_decode_core_logits._func)
    host_signature = inspect.signature(module.l3_mtp_decode_core._func)
    prepared = (
        "hidden_states", "prev_pre_hc_hidden", "swa_slot_mapping", "swa_indices", "swa_lens",
    )
    assert tuple(core_signature.parameters)[:5] == prepared
    assert tuple(host_signature.parameters)[:5] == prepared
    required = {"position_ids", "routing_input_ids", "hidden_out", "next_pre_hc_hidden", "logits"}
    assert required <= set(core_signature.parameters)
    assert required <= set(host_signature.parameters)
    forbidden = {
        "embed_weight", "main_pre_hc_hidden", "tail_pre_hc_pool",
        "accepted_counts", "tail_slot_ids", "input_ids", "ori_block_table", "sampled_ids",
    }
    assert forbidden.isdisjoint(core_signature.parameters)
    assert forbidden.isdisjoint(host_signature.parameters)
    expected_dtypes = ("bfloat16", "fp32", "int64", "int32", "int32")
    actual_dtypes = tuple(str(core_signature.parameters[name].annotation).split(", ")[-1][:-1] for name in prepared)
    assert actual_dtypes == expected_dtypes
    assert str(core_signature.parameters["logits"].annotation).endswith(", fp32]")

    tracked = {
        "mtp_projection", "attention_swa", "moe", "hc_head", "rms_norm",
        "lm_head_core", "lm_head", "lookup_embedding", "pack_mtp_hidden",
        "build_swa_metadata", "clear_moe_signals", "clear_lm_head_signals",
        "lm_head_with_sampling", "greedy_sample",
    }
    calls = sorted(
        (
            (node.lineno, node.col_offset, node.func.id)
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in tracked
        ),
    )
    assert [name for _, _, name in calls] == [
        "mtp_projection", "attention_swa", "moe", "hc_head", "rms_norm", "lm_head_core",
    ]


def test_mtp_core_cleanup_is_a_separate_child() -> None:
    path = _MODEL_DIR / "decode_mtp_core.py"
    cleanup = _function_node(path, "mtp_decode_core_cleanup")
    module = importlib.import_module("decode_mtp_core")
    cleanup_parameters = tuple(inspect.signature(module.mtp_decode_core_cleanup._func).parameters)
    assert cleanup_parameters == (
        "next_pre_hc_hidden", "logits", "arrived", "data_arrived", "combine_arrived",
        "lm_head_hidden_done", "lm_head_logits_done",
    )
    cleanup_calls = sorted(
        (
            (node.lineno, node.col_offset, node.func.id)
            for node in ast.walk(cleanup)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"clear_moe_signals", "clear_lm_head_signals"}
        ),
    )
    assert [name for _, _, name in cleanup_calls] == ["clear_moe_signals", "clear_lm_head_signals"]

    host = _function_node(path, "l3_mtp_decode_core")
    rank_loops = [
        node
        for node in host.body
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "pl.range(pld.world_size())"
    ]
    assert len(rank_loops) == 2
    loop_dispatches = [
        [
            name
            for name in _called_function_names(loop)
            if name in {"mtp_decode_core_logits", "mtp_decode_core_cleanup"}
        ]
        for loop in rank_loops
    ]
    assert loop_dispatches == [["mtp_decode_core_logits"], ["mtp_decode_core_cleanup"]]

    main = _function_node(path, "main")
    run_call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_jit"
    )
    run_keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in run_call.keywords}
    assert run_keywords["fn"] == "l3_mtp_decode_core"


def test_mtp_core_lm_head_logits_window_uses_shape_and_dtype_allocation() -> None:
    path = _MODEL_DIR / "decode_mtp_core.py"
    host = _function_node(path, "l3_mtp_decode_core")
    allocation = next(
        node.value
        for node in host.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "lm_head_logits_window_buf" for target in node.targets)
    )
    assert isinstance(allocation, ast.Call)
    assert ast.unparse(allocation.func) == "pld.alloc_window_buffer"
    assert [ast.unparse(argument) for argument in allocation.args] == ["[MAX_LOGIT_ROWS, LM_HEAD_VOCAB]"]
    assert {keyword.arg: ast.unparse(keyword.value) for keyword in allocation.keywords} == {"dtype": "pl.FP32"}


def test_mtp_core_specs_match_the_host_abi(monkeypatch) -> None:
    module = importlib.import_module("decode_mtp_core")
    original_build_tensor_specs = module.decode_mtp.build_tensor_specs
    captured = {}

    def capture_base_specs(*args, **kwargs):
        base_specs = original_build_tensor_specs(*args, **kwargs)
        captured["specs"] = base_specs
        return base_specs

    monkeypatch.setattr(module.decode_mtp, "build_tensor_specs", capture_base_specs)
    host_parameters = tuple(inspect.signature(module.l3_mtp_decode_core._func).parameters)
    specs = module.build_tensor_specs(start_pos=8192, num_tokens=8)
    assert tuple(spec.name for spec in specs) == host_parameters

    tensor_specs = {spec.name: spec for spec in specs if hasattr(spec, "shape")}
    prepared = {
        "hidden_states": ([module.N_RANKS, module.T, module.D], "torch.bfloat16"),
        "prev_pre_hc_hidden": ([module.N_RANKS, module.T, module.HC_MULT, module.D], "torch.float32"),
        "swa_slot_mapping": ([module.N_RANKS, module.T], "torch.int64"),
        "swa_indices": ([module.N_RANKS, module.T, module.WIN], "torch.int32"),
        "swa_lens": ([module.N_RANKS, module.T], "torch.int32"),
    }
    for name, (shape, dtype) in prepared.items():
        assert tensor_specs[name].shape == shape
        assert str(tensor_specs[name].dtype) == dtype
        assert tensor_specs[name].resident == "stacked"

    base_by_name = {spec.name: spec for spec in captured["specs"]}
    for spec in specs:
        if spec.name not in prepared:
            assert spec is base_by_name[spec.name]

    forbidden = {
        "embed_weight", "main_pre_hc_hidden", "tail_pre_hc_pool",
        "accepted_counts", "tail_slot_ids", "input_ids", "ori_block_table", "sampled_ids",
    }
    assert forbidden.isdisjoint(tensor_specs)
    assert tensor_specs["kv_cache"].resident == "stacked"
    assert tensor_specs["kv_cache"].is_output
    assert {
        name for name, spec in tensor_specs.items() if spec.is_output
    } == {"kv_cache", "hidden_out", "next_pre_hc_hidden", "logits"}


def test_mtp_core_prepared_metadata_matches_requested_cache_capacity() -> None:
    import torch
    from decode_metadata import block_table, paged_slot_mapping, swa_indices_and_lens

    module = importlib.import_module("decode_mtp_core")
    ori_block_num = 73
    specs = {
        spec.name: spec
        for spec in module.build_tensor_specs(start_pos=8192, num_tokens=8, ori_block_num=ori_block_num)
    }
    positions = specs["position_ids"].create_tensor()
    actual_slot_mapping = specs["swa_slot_mapping"].create_tensor()
    actual_indices = specs["swa_indices"].create_tensor()
    actual_lens = specs["swa_lens"].create_tensor()

    assert actual_slot_mapping.shape == (module.N_RANKS, module.T)
    assert actual_slot_mapping.dtype == torch.int64
    assert actual_indices.shape == (module.N_RANKS, module.T, module.WIN)
    assert actual_indices.dtype == torch.int32
    assert actual_lens.shape == (module.N_RANKS, module.T)
    assert actual_lens.dtype == torch.int32

    table = block_table(
        batch=module.B,
        table_blocks=module.ORI_TABLE_MAX_BLOCKS,
        physical_blocks=ori_block_num,
    )
    for rank in range(module.N_RANKS):
        rank_positions = positions[rank].reshape(module.B, module.T // module.B)
        expected_slot_mapping = paged_slot_mapping(
            rank_positions, table, block_size=module.BLOCK_SIZE,
        ).reshape(-1)
        expected_indices, expected_lens = swa_indices_and_lens(
            rank_positions, table, block_size=module.BLOCK_SIZE, window=module.WIN,
        )
        assert torch.equal(actual_slot_mapping[rank], expected_slot_mapping)
        assert torch.equal(actual_indices[rank], expected_indices)
        assert torch.equal(actual_lens[rank], expected_lens)


def test_mtp_core_finite_only_checks_every_output() -> None:
    from types import SimpleNamespace

    import torch

    module = importlib.import_module("decode_mtp_core")
    output_names = ("kv_cache", "hidden_out", "next_pre_hc_hidden", "logits")
    specs = [SimpleNamespace(name=name, is_output=True) for name in output_names]
    specs.append(SimpleNamespace(name="position_ids", is_output=False))
    compare_map = module.finite_output_compare_map(specs)
    assert set(compare_map) == set(output_names)
    for compare in compare_map.values():
        passed, _ = compare(torch.tensor([0.0]), None)
        failed, message = compare(torch.tensor([float("inf")]), None)
        assert passed
        assert not failed
        assert "non-finite" in message


def test_mtp_core_exact_golden_covers_every_tp_group(monkeypatch) -> None:
    import torch

    module = importlib.import_module("decode_mtp_core")
    monkeypatch.setattr(module, "N_RANKS", 8)
    monkeypatch.setattr(module, "LM_HEAD_TP_SIZE", 4)
    hidden_states = torch.arange(module.N_RANKS, dtype=torch.float32).reshape(module.N_RANKS, 1, 1)
    tensors = {
        "hidden_out": hidden_states,
        "lm_head_weight": torch.zeros(module.N_RANKS, 1, 1),
        "logit_row_indices": torch.zeros(module.N_RANKS, 1, dtype=torch.int32),
        "logits": torch.zeros(module.N_RANKS, 1, 1),
    }
    owner_groups = []

    def fake_golden_lm_head(group_tensors):
        owner_groups.append(group_tensors["hidden_states"][:, 0, 0].tolist())
        group_tensors["logits"].fill_(len(owner_groups))

    monkeypatch.setattr(module, "golden_lm_head", fake_golden_lm_head)
    module._golden_lm_head_groups(tensors)

    assert owner_groups == [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]
    expected_logits = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    assert torch.equal(tensors["logits"][:, 0, 0], expected_logits)
