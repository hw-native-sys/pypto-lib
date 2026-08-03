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
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "deepseek" / "v4-flash"
sys.path.insert(0, str(_MODEL_DIR))


def test_handoff_packs_both_acceptance_paths_and_updates_tails() -> None:
    module = importlib.import_module("decode_input_pack")
    golden = getattr(module, "golden_pack_mtp_inputs")
    sampled = torch.full((8, 8), -777, dtype=torch.int32)
    sampled[:, 0] = torch.tensor(
        [100, 101, 200, 201, 300, 301, 400, 401],
        dtype=torch.int32,
    )
    tensors = {
        "main_sampled_ids": sampled,
        "main_position_ids": torch.tensor(
            [10, 11, 20, 21, 30, 31, 40, 41],
            dtype=torch.int32,
        ),
        "accepted_counts": torch.tensor([1, 2, 1, 2], dtype=torch.int32),
        "tail_slot_ids": torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        "tail_token_pool": torch.tensor([900, 901, 902, 903], dtype=torch.int64),
        "tail_position_pool": torch.tensor([9, 19, 29, 39], dtype=torch.int32),
        "mtp_input_ids": torch.zeros(8, dtype=torch.int64),
        "mtp_position_ids": torch.zeros(8, dtype=torch.int32),
    }

    golden(tensors)

    assert tensors["mtp_input_ids"].tolist() == [
        900, 100, 200, 201, 902, 300, 400, 401,
    ]
    assert tensors["mtp_position_ids"].tolist() == [
        9, 10, 20, 21, 29, 30, 40, 41,
    ]
    assert tensors["tail_token_pool"].tolist() == [100, 201, 300, 401]
    assert tensors["tail_position_pool"].tolist() == [10, 21, 30, 41]


def test_handoff_golden_honors_permuted_tail_slots() -> None:
    module = importlib.import_module("decode_input_pack")
    sampled = torch.full((8, 8), -777, dtype=torch.int32)
    sampled[:, 0] = torch.tensor(
        [100, 101, 200, 201, 300, 301, 400, 401],
        dtype=torch.int32,
    )
    tensors = {
        "main_sampled_ids": sampled,
        "main_position_ids": torch.tensor(
            [10, 11, 20, 21, 30, 31, 40, 41],
            dtype=torch.int32,
        ),
        "accepted_counts": torch.tensor([1, 2, 1, 2], dtype=torch.int32),
        "tail_slot_ids": torch.tensor([2, 0, 3, 1], dtype=torch.int32),
        "tail_token_pool": torch.tensor([900, 901, 902, 903], dtype=torch.int64),
        "tail_position_pool": torch.tensor([9, 19, 29, 39], dtype=torch.int32),
        "mtp_input_ids": torch.zeros(8, dtype=torch.int64),
        "mtp_position_ids": torch.zeros(8, dtype=torch.int32),
    }

    module.golden_pack_mtp_inputs(tensors)

    assert tensors["mtp_input_ids"].tolist() == [
        902, 100, 200, 201, 903, 300, 400, 401,
    ]
    assert tensors["mtp_position_ids"].tolist() == [
        29, 10, 20, 21, 39, 30, 40, 41,
    ]
    assert tensors["tail_token_pool"].tolist() == [201, 401, 100, 300]
    assert tensors["tail_position_pool"].tolist() == [21, 41, 10, 30]


def test_handoff_device_entry_has_the_stateful_contract() -> None:
    module = importlib.import_module("decode_input_pack")
    function = getattr(module, "pack_mtp_inputs")
    assert list(inspect.signature(function._func).parameters) == [
        "main_sampled_ids",
        "main_position_ids",
        "accepted_counts",
        "tail_slot_ids",
        "tail_token_pool",
        "tail_position_pool",
        "mtp_input_ids",
        "mtp_position_ids",
    ]
    tree = ast.parse(inspect.getsource(function._func))
    definition = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
    annotation_wrappers = [
        argument.annotation.value.attr
        for argument in definition.args.args
    ]
    assert annotation_wrappers == [
        "Tensor",
        "Tensor",
        "Tensor",
        "Tensor",
        "InOut",
        "InOut",
        "Out",
        "Out",
    ]

    specs = module.build_handoff_tensor_specs()
    assert [
        (spec.name, spec.shape, spec.dtype, spec.is_output, spec.init_value is not None)
        for spec in specs
    ] == [
        ("main_sampled_ids", [8, 8], torch.int32, False, True),
        ("main_position_ids", [8], torch.int32, False, True),
        ("accepted_counts", [4], torch.int32, False, True),
        ("tail_slot_ids", [4], torch.int32, False, True),
        ("tail_token_pool", [4], torch.int64, True, True),
        ("tail_position_pool", [4], torch.int32, True, True),
        ("mtp_input_ids", [8], torch.int64, True, False),
        ("mtp_position_ids", [8], torch.int32, True, False),
    ]


def test_handoff_fixture_accepts_valid_vocab_boundaries() -> None:
    module = importlib.import_module("decode_input_pack")
    sampled = torch.full((8, 8), -777, dtype=torch.int32)
    sampled[:, 0] = torch.tensor(
        [0, 129279, 1, 2, 3, 4, 5, 6],
        dtype=torch.int32,
    )

    result = module.validate_handoff_fixture(
        torch.tensor([1, 2, 1, 2], dtype=torch.int32),
        torch.tensor([2, 0, 3, 1], dtype=torch.int32),
        sampled,
        torch.tensor([0, 129279, 7, 8], dtype=torch.int64),
    )

    assert result is None


@pytest.mark.parametrize(
    ("accepted_counts", "tail_slot_ids", "sampled_token", "tail_token"),
    [
        ([0, 2, 1, 2], [0, 1, 2, 3], 100, 900),
        ([1, 2, 1, 2], [0, 1, 1, 3], 100, 900),
        ([1, 2, 1, 2], [0, 1, 2, 4], 100, 900),
        ([1, 2, 1, 2], [0, 1, 2, 3], -1, 900),
        ([1, 2, 1, 2], [0, 1, 2, 3], 100, 129280),
    ],
)
def test_handoff_fixture_rejects_invalid_metadata(
    accepted_counts,
    tail_slot_ids,
    sampled_token,
    tail_token,
) -> None:
    module = importlib.import_module("decode_input_pack")
    sampled = torch.full((8, 8), sampled_token, dtype=torch.int32)
    tails = torch.full((4,), tail_token, dtype=torch.int64)
    with pytest.raises(ValueError):
        module.validate_handoff_fixture(
            torch.tensor(accepted_counts, dtype=torch.int32),
            torch.tensor(tail_slot_ids, dtype=torch.int32),
            sampled,
            tails,
        )


def _load_routing_module():
    return importlib.import_module("decode_routing")


def _function_node(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _call_node(function: ast.FunctionDef, name: str) -> ast.Call:
    return next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    )


def _called_function_names(function: ast.FunctionDef) -> list[str]:
    names = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.append(node.func.id)
    return names


def test_mtp_logits_child_has_no_sampled_ids_or_sampler() -> None:
    path = _MODEL_DIR / "decode_mtp.py"
    function = _function_node(path, "mtp_decode_layer_logits")
    parameters = [arg.arg for arg in function.args.args]
    calls = _called_function_names(function)
    assert "sampled_ids" not in parameters
    assert "lm_head" in calls
    assert "lm_head_with_sampling" not in calls
    assert "greedy_sample" not in calls


def test_standalone_mtp_keeps_sampling() -> None:
    path = _MODEL_DIR / "decode_mtp.py"
    function = _function_node(path, "mtp_decode_layer")
    parameters = [arg.arg for arg in function.args.args]
    calls = _called_function_names(function)
    assert "sampled_ids" in parameters
    assert "lm_head_with_sampling" in calls


def test_routing_modes_preserve_or_override_layer_identity() -> None:
    routing = _load_routing_module()
    actual_layers = list(range(44))
    assert [
        routing.resolve_routing_layer_id(layer_id, "model")
        for layer_id in actual_layers
    ] == actual_layers
    assert [
        routing.resolve_routing_layer_id(layer_id, "trace-hash")
        for layer_id in actual_layers
    ] == [0] * 44


def test_trace_hash_ep8_routes_balance_exactly() -> None:
    routing = _load_routing_module()
    table = routing.build_trace_hash_tid2eid(
        num_layers=1,
        first_layer_id=0,
        n_ranks=8,
        tokens_per_rank=8,
        vocab_size=32,
        topk=6,
        n_experts=128,
    )
    input_ids = torch.arange(8, dtype=torch.int64).expand(8, -1)
    active = torch.stack(
        [table[rank, input_ids[rank]] for rank in range(8)],
        dim=0,
    )
    counts = torch.bincount(active.reshape(-1).long(), minlength=128)
    assert counts.tolist() == [3] * 128


def test_trace_hash_keeps_ep8_expert_dimensions() -> None:
    routing = _load_routing_module()
    assert routing.target_expert_topology(ep=8) == (128, 16)


def test_ep8_runtime_topology_uses_128_global_and_16_local() -> None:
    env = os.environ.copy()
    inherited_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(_MODEL_DIR)
    if inherited_pythonpath:
        env["PYTHONPATH"] += os.pathsep + inherited_pythonpath
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import moe; print(moe.N_RANKS, moe.N_EXPERTS_GLOBAL, moe.N_LOCAL)",
            "--ep",
            "8",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert "8 128 16" in result.stdout.splitlines()


def test_layer1_routing_scalar_is_declared_in_its_consuming_scope() -> None:
    path = _MODEL_DIR / "decode_fwd.py"
    function = _function_node(path, "decode_fwd")
    moe_calls = sorted(
        [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "moe"
        ],
        key=lambda node: node.lineno,
    )
    layer1_moe = moe_calls[1]
    consuming_scope = min(
        [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.With)
            and node.lineno < layer1_moe.lineno <= node.end_lineno
            and any(ast.unparse(item.context_expr) == "pl.scope()" for item in node.items)
        ],
        key=lambda node: node.end_lineno - node.lineno,
    )
    routing_assignments = []
    for node in ast.walk(consuming_scope):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "layer1_routing_layer":
                routing_assignments.append(node)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "layer1_routing_layer"
            for target in node.targets
        ):
            routing_assignments.append(node)

    routing_assignments.sort(key=lambda node: node.lineno)
    assert [type(node) for node in routing_assignments] == [ast.AnnAssign, ast.Assign]
    assert all(node.lineno < layer1_moe.lineno for node in routing_assignments)


def test_main_decode_keeps_actual_slice_ids_separate_from_routing_ids() -> None:
    path = _MODEL_DIR / "decode_fwd.py"
    function = _function_node(path, "decode_fwd")
    parameter_names = [arg.arg for arg in function.args.args]
    assert parameter_names[-1] == "routing_mode"
    source = ast.get_source_segment(path.read_text(), function)
    assert "csa_layer * N_EXPERTS_GLOBAL" in source
    assert "hca_layer * N_EXPERTS_GLOBAL" in source
    assert "csa_layer_last * N_EXPERTS_GLOBAL" in source
    assert "csa_routing_layer" in source
    assert "hca_routing_layer" in source
    assert "last_routing_layer" in source
    moe_calls = sorted(
        [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "moe"
        ],
        key=lambda node: node.lineno,
    )
    assert len(moe_calls) == 5
    assert [ast.unparse(call.args[-4]) for call in moe_calls] == [
        "pl.cast(0, pl.INT32)",
        "layer1_routing_layer",
        "csa_routing_layer",
        "hca_routing_layer",
        "last_routing_layer",
    ]


def test_mtp_moe_uses_a_separate_routing_identity() -> None:
    path = _MODEL_DIR / "decode_mtp.py"
    try:
        function = _function_node(path, "_mtp_decode_body")
    except StopIteration:
        function = _function_node(path, "mtp_decode_layer")
    parameter_names = [arg.arg for arg in function.args.args]
    input_ids_index = parameter_names.index("input_ids")
    assert parameter_names[input_ids_index + 1] == "routing_input_ids"
    moe_call = _call_node(function, "moe")
    assert ast.unparse(moe_call.args[8]) == "routing_input_ids"
    assert ast.unparse(moe_call.args[-4]) == "mtp_routing_layer"
    source = ast.get_source_segment(path.read_text(), function)
    assert "lookup_embedding(input_ids, embed_weight, hidden_states)" in source
    assert "pl.cast(MTP_LAYER_ID, pl.INT32)" in source
    assert "routing_mode == ROUTING_TRACE_HASH" in source

    host = _function_node(path, "l3_mtp_decode_layer")
    host_parameter_names = [arg.arg for arg in host.args.args]
    host_input_ids_index = host_parameter_names.index("input_ids")
    assert host_parameter_names[host_input_ids_index + 1] == "routing_input_ids"


def test_mtp_trace_routing_fixture_balances_independently(monkeypatch) -> None:
    mtp = importlib.import_module("decode_mtp")
    routing = _load_routing_module()
    monkeypatch.setattr(mtp, "N_RANKS", 8)
    spec = mtp._routing_input_ids_spec()
    routing_input_ids = spec.create_tensor()
    assert routing_input_ids.tolist() == [list(range(8))] * 8

    table = routing.build_trace_hash_tid2eid(
        num_layers=1,
        first_layer_id=43,
        n_ranks=8,
        tokens_per_rank=8,
        vocab_size=32,
        topk=6,
        n_experts=128,
    )
    active = torch.stack(
        [table[rank, routing_input_ids[rank]] for rank in range(8)],
        dim=0,
    )
    counts = torch.bincount(active.reshape(-1).long(), minlength=128)
    assert counts.tolist() == [3] * 128


def test_composite_dispatches_decode_handoff_then_logits_mtp() -> None:
    path = _MODEL_DIR / "decode_fwd_mtp.py"
    function = _function_node(path, "l3_decode_fwd_mtp")
    calls = [
        (node.lineno, node.func.id)
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {
            "decode_fwd",
            "pack_mtp_inputs",
            "mtp_decode_layer_logits",
        }
    ]
    ordered = [name for _, name in sorted(calls)]
    assert ordered == [
        "decode_fwd",
        "pack_mtp_inputs",
        "mtp_decode_layer_logits",
    ]


def test_composite_has_no_mtp_sampled_output() -> None:
    path = _MODEL_DIR / "decode_fwd_mtp.py"
    function = _function_node(path, "l3_decode_fwd_mtp")
    parameters = [arg.arg for arg in function.args.args]
    assert "sampled_ids" in parameters
    assert "mtp_sampled_ids" not in parameters
    assert "mtp_logits" in parameters


def test_composite_signature_deduplicates_shared_tensors() -> None:
    path = _MODEL_DIR / "decode_fwd_mtp.py"
    function = _function_node(path, "l3_decode_fwd_mtp")
    parameters = [arg.arg for arg in function.args.args]
    assert len(parameters) == 140
    assert len(set(parameters)) == 140
    assert parameters[-2:] == ["num_tokens", "routing_mode"]
    for omitted in [
        "mtp_embed_weight",
        "mtp_main_pre_hc_hidden",
        "mtp_freqs_cos",
        "mtp_freqs_sin",
        "mtp_lm_head_weight",
        "mtp_routing_input_ids",
        "mtp_sampled_ids",
    ]:
        assert omitted not in parameters
    for required in [
        "mtp_tail_token_pool",
        "mtp_tail_position_pool",
        "mtp_input_ids",
        "mtp_position_ids",
        "mtp_moe_norm_w",
        "mtp_final_norm_w",
        "mtp_hidden_out",
        "mtp_next_pre_hc_hidden",
        "mtp_logits",
        "mtp_logit_row_indices",
    ]:
        assert required in parameters


def test_composite_wires_shared_inputs_and_disjoint_windows() -> None:
    path = _MODEL_DIR / "decode_fwd_mtp.py"
    host = _function_node(path, "l3_decode_fwd_mtp")
    handoff_call = _call_node(host, "pack_mtp_inputs")
    assert [ast.unparse(arg) for arg in handoff_call.args] == [
        "sampled_ids[r]",
        "position_ids[r]",
        "mtp_accepted_counts[r]",
        "mtp_tail_slot_ids[r]",
        "mtp_tail_token_pool[r]",
        "mtp_tail_position_pool[r]",
        "mtp_input_ids[r]",
        "mtp_position_ids[r]",
    ]

    main_child = _function_node(_MODEL_DIR / "decode_fwd.py", "decode_fwd")
    main_call = _call_node(host, "decode_fwd")
    assert len(main_call.args) == len(main_child.args.args)
    main_mapping = {
        parameter.arg: ast.unparse(argument)
        for parameter, argument in zip(
            main_child.args.args,
            main_call.args,
            strict=True,
        )
    }

    mtp_child = _function_node(
        _MODEL_DIR / "decode_mtp.py",
        "mtp_decode_layer_logits",
    )
    mtp_call = _call_node(host, "mtp_decode_layer_logits")
    assert len(mtp_call.args) == len(mtp_child.args.args)
    mtp_mapping = {
        parameter.arg: ast.unparse(argument)
        for parameter, argument in zip(
            mtp_child.args.args,
            mtp_call.args,
            strict=True,
        )
    }
    assert {
        name: mtp_mapping[name]
        for name in [
            "embed_weight",
            "main_pre_hc_hidden",
            "position_ids",
            "freqs_cos",
            "freqs_sin",
            "input_ids",
            "routing_input_ids",
            "lm_head_weight",
        ]
    } == {
        "embed_weight": "embed_weight[r]",
        "main_pre_hc_hidden": "pre_hc_hidden_out[r]",
        "position_ids": "mtp_position_ids[r]",
        "freqs_cos": "freqs_cos[r]",
        "freqs_sin": "freqs_sin[r]",
        "input_ids": "mtp_input_ids[r]",
        "routing_input_ids": "input_ids[r]",
        "lm_head_weight": "lm_head_weight[r]",
    }

    window_names = [
        "recv_meta",
        "recv_x",
        "recv_aux",
        "recv_route",
        "arrived",
        "data_arrived",
        "routed_y_buf",
        "combine_arrived",
        "lm_head_hidden_window",
        "lm_head_hidden_done",
        "lm_head_logits_window",
        "lm_head_logits_done",
    ]
    for name in window_names:
        assert main_mapping[name] == f"main_{name}"
        assert mtp_mapping[name] == f"mtp_{name}"


def test_composite_topology_validation_targets_ep8_tp4() -> None:
    module = importlib.import_module("decode_fwd_mtp")
    module.validate_topology(ep=8, tp=4, device_ids=list(range(8)))
    for ep, tp, devices in [
        (4, 4, list(range(4))),
        (8, 8, list(range(8))),
        (8, 4, list(range(7))),
    ]:
        try:
            module.validate_topology(ep=ep, tp=tp, device_ids=devices)
        except ValueError:
            continue
        raise AssertionError((ep, tp, devices))


def test_composite_finite_oracle_rejects_invalid_logits() -> None:
    module = importlib.import_module("decode_fwd_mtp")
    expected = torch.zeros(2, dtype=torch.float32)
    assert module.finite_tensor_compare(
        torch.tensor([1.0, -2.0]),
        expected,
    )[0]
    ok, detail = module.finite_tensor_compare(
        torch.tensor([1.0, float("nan")]),
        expected,
    )
    assert not ok
    assert "1/2" in detail
