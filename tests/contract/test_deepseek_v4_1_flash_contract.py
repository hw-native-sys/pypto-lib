# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Contract tests for the DeepSeek-V4.1 Flash decode layer composition."""

import ast
import importlib.util
import inspect
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

module = sys.modules.get("pypto")
HAS_PYPTO = (
    not getattr(module, "__pypto_stub__", False)
    if module is not None
    else importlib.util.find_spec("pypto") is not None
)
requires_pypto = pytest.mark.skipif(not HAS_PYPTO, reason="model imports require real PyPTO")


@pytest.fixture
def composition():
    from models.deepseek_v4_1_flash import decode_layer

    return decode_layer


@pytest.fixture
def attention_common():
    from models.deepseek_v4_1_flash import decode_common

    return decode_common


MODEL_DIR = Path(__file__).parents[2] / "models" / "deepseek_v4_1_flash"


def _tree(name: str) -> ast.Module:
    return ast.parse((MODEL_DIR / name).read_text())


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def _top_level_functions(name: str) -> set[str]:
    return {node.name for node in _tree(name).body if isinstance(node, ast.FunctionDef)}


def _string_list_assignment(name: str, variable: str) -> set[str]:
    tree = _tree(name)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == variable for target in node.targets)
    )
    return {element.value for element in assignment.value.elts}


@requires_pypto
def test_decode_layer_schedule_covers_all_40_layers(composition):
    resolve_decode_layer_plan = composition.resolve_decode_layer_plan
    DecodeLayerKind = composition.DecodeLayerKind
    REPRESENTATIVE_LAYER_IDS = composition.REPRESENTATIVE_LAYER_IDS
    plans = [resolve_decode_layer_plan(layer_id) for layer_id in range(40)]
    assert Counter(plan.kind for plan in plans) == {
        DecodeLayerKind.SWA: 2,
        DecodeLayerKind.C2A_FULL: 3,
        DecodeLayerKind.C2A_REUSE: 15,
        DecodeLayerKind.C1A_FULL: 1,
        DecodeLayerKind.C1A_REINDEX: 4,
        DecodeLayerKind.C1A_REUSE: 15,
    }
    assert {
        kind: resolve_decode_layer_plan(layer_id).kind for kind, layer_id in REPRESENTATIVE_LAYER_IDS.items()
    } == {kind: kind for kind in DecodeLayerKind}


@requires_pypto
def test_decode_layer_schedule_resolves_cache_sources(composition):
    resolve_decode_layer_plan = composition.resolve_decode_layer_plan
    assert resolve_decode_layer_plan(2).kv_source_layer_id == 2
    assert resolve_decode_layer_plan(7).index_source_layer_id == 2
    assert resolve_decode_layer_plan(8).kv_source_layer_id == 8
    assert resolve_decode_layer_plan(19).index_source_layer_id == 14
    assert resolve_decode_layer_plan(20).is_candidate_source
    assert resolve_decode_layer_plan(23).index_source_layer_id == 20
    assert resolve_decode_layer_plan(24).index_source_layer_id == 24
    assert resolve_decode_layer_plan(39).index_source_layer_id == 36
    with pytest.raises(ValueError, match="layer_id must be"):
        resolve_decode_layer_plan(40)


def test_decode_attention_modes_are_split_from_block_composition():
    common_tree = _tree("decode_common.py")
    layer_tree = _tree("decode_layer.py")
    block = _function(layer_tree, "golden_decode_layer")

    def call_names(function):
        names = []
        for node in sorted(ast.walk(function), key=lambda n: getattr(n, "lineno", 0)):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.append(node.func.attr)
        return names

    common_imports = {
        alias.name
        for node in common_tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(name.startswith("decode_swa") or name.startswith("decode_c") for name in common_imports)
    assert "DecodeLayerKind" not in {node.id for node in ast.walk(common_tree) if isinstance(node, ast.Name)}
    for module_name, leaf in (
        ("decode_swa", "decode_attn_swa"),
        ("decode_c2a_full", "decode_attn_c2a_full"),
        ("decode_c2a_reuse", "decode_attn_c2a_reuse"),
    ):
        tree = _tree(f"{module_name}.py")
        half = _function(tree, module_name)
        half_calls = call_names(half)
        assert [name for name in half_calls if name in ("attention_pre", leaf, "mhc_post")] == [
            "attention_pre",
            leaf,
            "mhc_post",
        ]
        rank = _function(tree, f"{module_name}_rank")
        assert call_names(rank) == [module_name]
        production_args = [arg.arg for arg in half.args.args]
        rank_args = [arg.arg for arg in rank.args.args]
        assert rank_args == production_args
        rank_call = next(node for node in ast.walk(rank) if isinstance(node, ast.Call))
        assert [ast.unparse(arg) for arg in rank_call.args] == production_args
    block_calls = call_names(block)
    assert [
        n for n in block_calls if n in ("golden_mhc_mixes", "golden_mhc_pre", "golden_moe", "golden_mhc_post")
    ] == [
        "golden_mhc_mixes",
        "golden_mhc_pre",
        "golden_mhc_post",
        "golden_mhc_mixes",
        "golden_mhc_pre",
        "golden_moe",
        "golden_mhc_post",
    ]
    assert "load_decode_attention_module" in block_calls
    assert not any(name.startswith("decode_c") or name == "decode_swa" for name in block_calls)
    assert not (MODEL_DIR / "decode_attention.py").exists()
    for module_name in (
        "decode_swa",
        "decode_c2a_full",
        "decode_c2a_reuse",
    ):
        tree = _tree(f"{module_name}.py")
        names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        assert {"skip_reason", "make_program", "main"} <= names
        assert "make_rank" not in names
    for module_name in ("decode_attn_c1a_full", "decode_attn_c1a_reindex", "decode_attn_c1a_reuse"):
        names = {node.name for node in _tree(f"{module_name}.py").body if isinstance(node, ast.FunctionDef)}
        assert {"make_program", "main"} <= names


def test_attention_pre_and_post_operators_have_single_public_owners():
    assert {
        "make_mx_projection",
        "make_bf16_projection",
        "make_bf16_projection_with_deps",
        "make_norm",
        "make_norm_with_deps",
        "make_rope",
        "make_rope_with_deps",
    } <= _top_level_functions("attention_ops.py")
    assert {
        "q_proj_qr",
        "q_proj_rope",
        "kv_proj_rope",
        "qkv_proj_rope",
        "prefill_q_proj_qr",
        "prefill_q_proj_rope",
        "prefill_kv_proj_rope",
    } <= _string_list_assignment("qkv_proj_rope.py", "__all__")
    assert {
        "grouped_output",
        "grouped_output_with_deps",
        "o_proj",
        "prefill_o_proj",
    } <= _string_list_assignment(
        "o_proj.py", "__all__"
    )

    basic_factories = {"make_mx_projection", "make_norm", "make_rope", "make_bf16_projection"}
    for module_name in ("decode_attn_swa.py", "prefill_attn_swa.py", "prefill_c1a_common.py"):
        assert _top_level_functions(module_name).isdisjoint(basic_factories)

    for module_name in ("decode_attn_swa.py", "decode_attn_c2a_full.py", "decode_attn_c2a_reuse.py"):
        names = {node.id for node in ast.walk(_tree(module_name)) if isinstance(node, ast.Name)}
        assert {"qkv_proj_rope", "o_proj"} <= names

    # C1A decode keeps only its distinct MX projection and uses shared TaskId-aware primitives.
    c1a_names = _top_level_functions("decode_attn_c1a_full.py")
    assert "make_mx_projection_with_deps" in c1a_names
    assert c1a_names.isdisjoint(
        {"grouped_output", "make_norm", "make_rope", "make_bf16_projection"}
    )
    c1a_tree = _tree("decode_attn_c1a_full.py")
    referenced_names = {node.id for node in ast.walk(c1a_tree) if isinstance(node, ast.Name)}
    assert {
        "make_bf16_projection_with_deps",
        "make_norm_with_deps",
        "make_rope_with_deps",
    } <= referenced_names
    assigned_names = {
        target.id
        for node in c1a_tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert {"qkv_proj_rope_with_deps", "o_proj_with_deps"} <= assigned_names


@requires_pypto
@pytest.mark.parametrize("layer_id", (0, 2, 3, 20, 24, 21))
def test_decode_layer_unfinished_device_dependencies_are_explicit(layer_id, composition):
    decode_layer_kernel_skip_reason = composition.decode_layer_kernel_skip_reason
    reason = decode_layer_kernel_skip_reason(layer_id)
    assert reason is not None and "EP8 MoE kernel" in reason
    assert "attention kernel" not in reason


@requires_pypto
@pytest.mark.parametrize("layer_id", (0, 2, 3, 20, 24, 21))
def test_decode_layer_golden_runs_every_mode(layer_id, composition):
    from models.deepseek_v4_1_flash._golden_smoke import make_decode_layer_golden_inputs
    from models.deepseek_v4_1_flash.hc_mixes import golden_mhc_mixes
    from models.deepseek_v4_1_flash.hc_pre import golden_mhc_pre

    golden_decode_layer = composition.golden_decode_layer
    decode_layer_attention_inputs = composition.decode_layer_attention_inputs
    inputs = make_decode_layer_golden_inputs(layer_id)
    original_compressed = inputs["attention_inputs"]["compressed_cache"].clone()
    original_index = inputs["attention_inputs"]["index_cache"].clone()
    result = golden_decode_layer(**inputs)
    expected_next_pre_mix = golden_mhc_mixes(
        result.attention_hidden,
        inputs["hc_ffn_fn"],
        inputs["hc_ffn_scale"],
        inputs["hc_ffn_base"],
    )[0]
    assert result.output.shape == inputs["x_hc"].shape
    assert result.output.dtype is torch.float32
    assert result.next_pre_mix.shape == inputs["incoming_pre_mix"].shape
    assert torch.isfinite(result.output).all()
    assert torch.equal(result.next_pre_mix, expected_next_pre_mix)
    attn_pre = golden_mhc_mixes(
        inputs["x_hc"], inputs["hc_attn_fn"], inputs["hc_attn_scale"], inputs["hc_attn_base"]
    )[0]
    assert torch.equal(result.attention_input, golden_mhc_pre(inputs["x_hc"], inputs["incoming_pre_mix"]))
    assert torch.equal(result.ffn_input, golden_mhc_pre(result.attention_hidden, attn_pre))
    assert not torch.equal(attn_pre, inputs["incoming_pre_mix"])
    assert torch.isfinite(result.next_pre_mix).all()
    assert set(decode_layer_attention_inputs(layer_id)) <= {"x", *inputs["attention_inputs"]}
    if layer_id in (3, 21, 24):
        assert torch.equal(result.attention.compressed_cache, original_compressed)
    if layer_id == 24:
        assert torch.equal(result.attention.index_cache, original_index)


def test_decode_sequence_parallel_validation_reports_either_failure():
    from types import SimpleNamespace

    from models.deepseek_v4_1_flash.decode_sp_integration import combine_validation

    def result(passed):
        return SimpleNamespace(passed=passed)

    replicated, sharded = result(True), result(False)
    assert combine_validation([replicated, sharded]) is sharded, (
        "a passing replicated run must not hide a sequence-parallel mismatch"
    )
    replicated, sharded = result(False), result(True)
    assert combine_validation([replicated, sharded]) is replicated, (
        "a passing sequence-parallel run must not hide a replicated mismatch"
    )
    replicated, sharded = result(True), result(True)
    assert combine_validation([replicated, sharded]) is sharded
    both_failed = [result(False), result(False)]
    assert combine_validation(both_failed) is both_failed[0]
    with pytest.raises(ValueError):
        combine_validation([])


def test_decode_sequence_parallel_metadata_supports_empty_owner_slabs():
    from models.deepseek_v4_1_flash.decode_sp_integration import (
        sequence_parallel_bounds,
        validate_two_layer_metadata,
    )

    assert [sequence_parallel_bounds(2, 4, rank) for rank in range(4)] == [
        (0, 1, 1),
        (1, 1, 1),
        (2, 0, 1),
        (2, 0, 1),
    ]
    gathered = validate_two_layer_metadata(num_tokens=2, tp_size=4)
    assert gathered.hidden.shape == (2, 4)
    assert [sequence_parallel_bounds(5, 4, rank) for rank in range(4)] == [
        (0, 2, 2),
        (2, 2, 2),
        (4, 1, 2),
        (5, 0, 2),
    ]
    # A padded batch keeps the physical-slab mapping: capacity 8 with 4 active rows
    # still publishes two-row slabs, while ceil(active / tp) would say one row and
    # shift every later rank by a row.
    assert [sequence_parallel_bounds(4, 4, rank, capacity=8) for rank in range(4)] == [
        (0, 2, 2),
        (2, 2, 2),
        (4, 0, 2),
        (4, 0, 2),
    ]


@requires_pypto
def test_decode_sequence_parallel_fixtures_differ_per_rank():
    from types import SimpleNamespace

    from models.deepseek_v4_1_flash.decode_common import make_boundary_specs

    def args(**overrides):
        values = dict(tokens=8, tp=4, seed=17, active_tokens=8, epochs=1, bench=False)
        values.update(overrides)
        return SimpleNamespace(**values)

    def stacked(specs, name):
        value = next(spec.init_value for spec in specs if spec.name == name)
        return value() if callable(value) else value

    for name in ("x_hc", "incoming_pre_mix"):
        replicated = stacked(make_boundary_specs(args(), sharded=False), name)
        assert all(torch.equal(replicated[0], replicated[rank]) for rank in range(1, 4)), (
            f"{name}: the replicated wiring must replicate the batch"
        )
        sharded = stacked(make_boundary_specs(args(), sharded=True), name)
        assert not any(torch.equal(sharded[0], sharded[rank]) for rank in range(1, 4)), (
            f"{name}: sequence-parallel slabs must differ per rank, otherwise a gather "
            "that mis-orders the slabs still compares equal"
        )


@requires_pypto
@pytest.mark.parametrize("layer_id", (0, 2, 3, 20, 24, 21))
def test_attention_half_readiness_is_independent_of_moe(layer_id, composition):
    attention_half_skip_reason = composition.attention_half_skip_reason
    reason = attention_half_skip_reason(layer_id)
    assert "MoE" not in (reason or "")
    assert reason is None
    mode = composition.load_decode_attention_module(composition.resolve_decode_layer_plan(layer_id).kind)
    assert getattr(mode, "KERNEL_READY", True)


@pytest.mark.parametrize(
    "module_name,leaf",
    (
        ("decode_swa", "decode_attn_swa"),
        ("decode_c2a_full", "decode_attn_c2a_full"),
        ("decode_c2a_reuse", "decode_attn_c2a_reuse"),
    ),
)
def test_attention_half_adapters_keep_leaf_call_contracts(module_name, leaf):
    function = _function(_tree(f"{module_name}.py"), module_name)
    call = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == leaf)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == leaf)
        )
    )
    parameters = _function(_tree(f"{leaf}.py"), leaf).args.args
    overrides = {
        "x": "normalized_attention",
        "compressed_indices": "topk_indices",
        "output": "attention_output",
        "group_base": "group_base",
        "tp_rank": "tp_rank",
        "attention_epoch": "attention_epoch",
    }
    assert [ast.unparse(arg) for arg in call.args] == [overrides.get(arg.arg, arg.arg) for arg in parameters]
    annotations = {arg.arg: ast.unparse(arg.annotation) for arg in function.args.args}
    for name in ("wq_a_scale", "wq_b_scale", "wkv_scale", "wo_b_scale", "index_wq_b_scale"):
        if name in annotations:
            assert "pl.MX_B_NN" in annotations[name]


@pytest.mark.parametrize("module_name", ("decode_swa", "decode_c2a_full", "decode_c2a_reuse"))
def test_attention_half_compositions_have_explicit_abi_and_ci_entry(module_name):
    path = MODEL_DIR / f"{module_name}.py"
    source = path.read_text()
    assert "# ci: no-sim" in source
    assert "# ci: a5" in source

    tree = ast.parse(source)
    production = _function(tree, module_name)
    rank = _function(tree, f"{module_name}_rank")
    for function in (production, rank):
        annotations = {arg.arg: ast.unparse(arg.annotation) for arg in function.args.args}
        assert all(annotation != "pl.Tensor" for annotation in annotations.values())
    assert all("pl.InOut" not in ast.unparse(arg.annotation) for arg in production.args.args)
    assert all("pl.Out" not in ast.unparse(arg.annotation) for arg in production.args.args)

    host = _function(tree, "attention_group")
    host_annotations = [ast.unparse(arg.annotation) for arg in host.args.args]
    assert not any("shapes[" in annotation or "dtypes[" in annotation for annotation in host_annotations)
    assert any("tensor_specs.x_hc" in annotation for annotation in host_annotations)


@requires_pypto
@pytest.mark.parametrize("module_name", ("decode_swa", "decode_c2a_full", "decode_c2a_reuse"))
def test_attention_half_compositions_name_leaf_golden_explicitly(module_name):
    from importlib import import_module

    module = import_module(f"models.deepseek_v4_1_flash.{module_name}")
    assert module.ATTENTION_GOLDEN
    assert not hasattr(module, "GOLDEN")


@requires_pypto
@pytest.mark.parametrize(
    "module_name,entry,golden,old_entry,old_golden",
    (
        ("decode_attn_swa", "decode_attn_swa", "golden_decode_attn_swa", "decode_swa", "golden_decode_swa"),
        (
            "decode_attn_c2a_full",
            "decode_attn_c2a_full",
            "golden_decode_attn_c2a_full",
            "decode_c2a_full",
            "golden_decode_c2a_full",
        ),
        (
            "decode_attn_c2a_reuse",
            "decode_attn_c2a_reuse",
            "golden_decode_attn_c2a_reuse",
            "decode_c2a_reuse",
            "golden_decode_c2a_reuse",
        ),
    ),
)
def test_attention_leaf_public_entries_match_file_ownership(
    module_name, entry, golden, old_entry, old_golden
):
    from importlib import import_module

    module = import_module(f"models.deepseek_v4_1_flash.{module_name}")
    assert getattr(module, entry)
    assert getattr(module, golden)
    assert not hasattr(module, old_entry)
    assert not hasattr(module, old_golden)


@requires_pypto
@pytest.mark.parametrize("layer_id", (0, 2, 3))
def test_attention_half_specs_match_host_and_do_not_generate_weights(
    layer_id, monkeypatch, composition, attention_common
):
    C = attention_common.C
    resolve_decode_layer_plan = composition.resolve_decode_layer_plan
    mode = composition.load_decode_attention_module(resolve_decode_layer_plan(layer_id).kind)
    # Full's existing spec builder is eager; the SWA/Reuse builders stay lazy.
    if layer_id != 2:

        def unexpected(*args, **kwargs):
            raise AssertionError("spec creation generated weights")

        monkeypatch.setattr("models.deepseek_v4_1_flash.decode_attn_swa.make_inputs", unexpected)
        monkeypatch.setattr(
            "models.deepseek_v4_1_flash.decode_attn_c2a_reuse.make_c2a_reuse_inputs", unexpected
        )
    args = SimpleNamespace(
        tp=C.TP_SIZE,
        dp=1,
        tokens=2,
        active_tokens=2,
        requests=2,
        epochs=2,
        seed=17,
        case="mixed",
        bench=False,
    )
    specs = mode.build_specs(args, {})
    program = mode.make_program(C.TP_SIZE, 2, specs)
    assert [spec.name for spec in specs] == list(inspect.signature(program._func).parameters)
    parameters = inspect.signature(program._func).parameters
    for spec in specs:
        if hasattr(spec, "shape"):
            assert list(parameters[spec.name].annotation.shape) == spec.shape
    assert "compressed_indices" not in {spec.name for spec in specs}
    assert "x" not in {spec.name for spec in specs}
    with pytest.raises(ValueError, match="host parameter names and order"):
        composition.make_decode_layer_program(layer_id, C.TP_SIZE, 2, specs=specs[::-1])


@requires_pypto
def test_attention_half_non_owner_scale_comparison_is_byte_exact(attention_common):
    compare_unchanged = attention_common.compare_unchanged
    initial = torch.ones(4).to(torch.float8_e4m3fn)
    compare = compare_unchanged("scale")
    assert compare(initial.clone(), initial, inputs={"scale": initial})[0]
    changed = initial.clone()
    changed.view(torch.uint8)[0] = 0
    assert not compare(changed, initial, inputs={"scale": initial})[0]


@requires_pypto
def test_attention_half_partial_outputs_are_inout():
    for module_name in ("decode_swa", "decode_c2a_full", "decode_c2a_reuse"):
        tree = _tree(f"{module_name}.py")
        production = _function(tree, module_name)
        production_names = {arg.arg for arg in production.args.args}
        assert "attention_input" not in production_names
        assert "normalized_attention" not in production_names
        host = _function(tree, "attention_group")
        annotations = {arg.arg: ast.unparse(arg.annotation) for arg in host.args.args}
        assert "attention_input" not in annotations
        assert "normalized_attention" not in annotations
        assert annotations["attention_output"].startswith("pl.InOut[")
        for boundary in ("attention_hidden", "attention_pre_mix"):
            assert annotations[boundary].startswith("pl.Out[")


@requires_pypto
def test_hidden_precision_ignores_inactive_denominator(attention_common):
    from models.deepseek_v4_1_flash.decode_attn_c2a_full import compare_output

    compare_attention_hidden = attention_common.make_compare_attention_hidden(compare_output)

    expected = torch.ones(1, 33, 4, 32)
    expected[:, 31:] = 13
    actual = expected.clone()
    actual[:, :31] *= 1.015
    assert not compare_attention_hidden(actual, expected, inputs={"num_tokens": 31})[0]
    assert compare_attention_hidden(expected.clone(), expected, inputs={"num_tokens": 31})[0]
    changed_tail = expected.clone()
    changed_tail[:, 31:] *= 1.1
    assert not compare_attention_hidden(changed_tail, expected, inputs={"num_tokens": 31})[0]


@requires_pypto
def test_block_golden_rejects_inactive_capacity(composition):
    from models.deepseek_v4_1_flash._golden_smoke import make_decode_layer_golden_inputs

    inputs = make_decode_layer_golden_inputs(0)
    with pytest.raises(ValueError, match="all capacity rows"):
        composition.golden_decode_layer(**dict(inputs, num_tokens=1))


@requires_pypto
def test_stage_selection_checks_only_required_dependencies(composition, attention_common):
    with pytest.raises(NotImplementedError, match="MoE"):
        composition.make_decode_layer_program(0, attention_common.C.EP_SIZE, 1, stage="block")
    with pytest.raises(ValueError, match="unknown decode stage"):
        composition.make_decode_layer_program(0, 1, 1, stage="ffn")
