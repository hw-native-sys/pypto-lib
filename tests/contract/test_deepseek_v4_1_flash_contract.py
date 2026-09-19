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


MODEL_DIR = Path(__file__).parents[2] / "models" / "deepseek_v4_1_flash"


def _tree(name: str) -> ast.Module:
    return ast.parse((MODEL_DIR / name).read_text())


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


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


def test_stages_share_attention_orchestration_and_block_order():
    tree = _tree("decode_layer.py")
    half = _function(tree, "attention_rank")
    block = _function(tree, "decode_layer")

    def call_names(function):
        return [
            node.func.id
            for node in sorted(ast.walk(function), key=lambda n: getattr(n, "lineno", 0))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]

    half_calls = call_names(half)
    assert [
        n for n in half_calls if n in ("mhc_mixes", "mhc_pre", "normalize_attention", "attention", "mhc_post")
    ] == ["mhc_mixes", "mhc_pre", "normalize_attention", "attention", "mhc_post"]
    block_calls = call_names(block)
    assert [n for n in block_calls if n in ("attention_rank", "mhc_mixes", "mhc_pre", "moe", "mhc_post")] == [
        "attention_rank",
        "mhc_mixes",
        "mhc_pre",
        "moe",
        "mhc_post",
    ]
    assert "normalize_attention" not in block_calls
    assert not any(n.startswith("decode_c") or n == "decode_swa" for n in block_calls)
    call = next(
        node
        for node in ast.walk(block)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "attention_rank"
    )
    aliases = {
        "rank": "ep_rank",
        "attention_pre_mix": "attn_pre",
        "output_window": "attention_output_window",
        "output_arrived": "attention_output_arrived",
    }
    assert [ast.unparse(arg) for arg in call.args] == [
        aliases.get(arg.arg, arg.arg) for arg in half.args.args
    ]
    for factory_name in ("make_block_rank", "make_attention_program"):
        assert "make_attention_rank" in call_names(_function(tree, factory_name))
    assert not (MODEL_DIR / "decode_attention.py").exists()


def test_l3_decode_layer_declares_mutable_state_and_outputs():
    host = _function(_tree("decode_layer.py"), "l3_decode_layer")
    inout_names = {
        arg.arg
        for arg in host.args.args
        if arg.annotation is not None and ast.unparse(arg.annotation).startswith("pl.InOut[")
    }
    out_names = {
        arg.arg
        for arg in host.args.args
        if arg.annotation is not None and ast.unparse(arg.annotation).startswith("pl.Out[")
    }
    assert inout_names == {
        "window_cache",
        "window_cache_scale",
        "compressed_cache",
        "compressed_cache_scale",
        "index_cache",
        "index_cache_scale",
        "compressor_state",
        "topk_indices",
        "candidate_mask",
    }
    assert out_names == {"x_next", "next_pre_mix"}


@requires_pypto
@pytest.mark.parametrize("layer_id", (0, 2, 3, 20, 24, 21))
def test_decode_layer_unfinished_device_dependencies_are_explicit(layer_id, composition):
    decode_layer_kernel_skip_reason = composition.decode_layer_kernel_skip_reason
    reason = decode_layer_kernel_skip_reason(layer_id)
    assert reason is not None and "EP8 MoE kernel" in reason
    if layer_id >= 20:
        assert "attention kernel" in reason
        assert "cache ABI agreement" in reason


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


@requires_pypto
@pytest.mark.parametrize("layer_id", (0, 2, 3, 20, 24, 21))
def test_attention_half_readiness_is_independent_of_moe(layer_id, composition):
    attention_half_skip_reason = composition.attention_half_skip_reason
    make_attention_program = composition.make_attention_program
    C = composition.C
    reason = attention_half_skip_reason(layer_id)
    assert "MoE" not in (reason or "")
    if layer_id < 20:
        assert reason is None
    else:
        assert "cache ABI agreement" in reason
        with pytest.raises(NotImplementedError, match="attention kernel"):
            make_attention_program(layer_id, C.TP_SIZE, 1, [])


@pytest.mark.parametrize(
    "adapter,leaf",
    (("_swa", "decode_swa"), ("_c2a_full", "decode_c2a_full"), ("_c2a_reuse", "decode_c2a_reuse")),
)
def test_attention_half_adapters_keep_leaf_call_contracts(adapter, leaf):
    function = _function(_tree("decode_layer.py"), adapter)
    call = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == leaf
    )
    parameters = _function(_tree(f"{leaf}.py"), leaf).args.args
    assert [ast.unparse(arg) for arg in call.args] == [
        "topk_indices" if arg.arg == "compressed_indices" else arg.arg for arg in parameters
    ]
    annotations = {arg.arg: ast.unparse(arg.annotation) for arg in function.args.args}
    for name in ("wq_a_scale", "wq_b_scale", "wkv_scale", "wo_b_scale", "index_wq_b_scale"):
        assert "pl.MX_B_NN" in annotations[name]


@requires_pypto
@pytest.mark.parametrize("layer_id", (0, 2, 3))
def test_attention_half_specs_match_host_and_do_not_generate_weights(layer_id, monkeypatch, composition):
    C = composition.C
    build_specs = composition.build_specs
    resolve_decode_layer_plan = composition.resolve_decode_layer_plan
    make_attention_program = composition.make_attention_program
    # Full's existing spec builder is eager; the SWA/Reuse builders stay lazy.
    if layer_id != 2:

        def unexpected(*args, **kwargs):
            raise AssertionError("spec creation generated weights")

        monkeypatch.setattr("models.deepseek_v4_1_flash.decode_swa.make_inputs", unexpected)
        monkeypatch.setattr("models.deepseek_v4_1_flash.decode_c2a_reuse.make_c2a_reuse_inputs", unexpected)
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
    specs = build_specs(args, resolve_decode_layer_plan(layer_id).kind, {})
    program = make_attention_program(layer_id, C.TP_SIZE, 2, specs)
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
def test_attention_half_non_owner_scale_comparison_is_byte_exact(composition):
    compare_unchanged = composition.compare_unchanged
    initial = torch.ones(4).to(torch.float8_e4m3fn)
    compare = compare_unchanged("scale")
    assert compare(initial.clone(), initial, inputs={"scale": initial})[0]
    changed = initial.clone()
    changed.view(torch.uint8)[0] = 0
    assert not compare(changed, initial, inputs={"scale": initial})[0]


@requires_pypto
def test_attention_half_normalized_suffix_is_byte_exact(composition):
    compare_normalized = composition.compare_normalized
    expected = torch.full((1, 3, 32), 13.0, dtype=torch.bfloat16)
    kwargs = dict(inputs={"num_tokens": 2}, actual_outputs={}, expected_outputs={}, rtol=1e-3, atol=1e-3)
    assert compare_normalized(expected.clone(), expected, **kwargs)[0]
    changed = expected.clone()
    changed[:, 2:] = 13.0625
    assert not compare_normalized(changed, expected, **kwargs)[0]


def test_attention_half_partial_outputs_are_inout():
    tree = _tree("decode_layer.py")
    for name in ("attention_rank", "attention_group"):
        function = next(
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
        )
        annotations = {arg.arg: ast.unparse(arg.annotation) for arg in function.args.args}
        for boundary in ("normalized_attention", "attention_output"):
            assert annotations[boundary].startswith("pl.InOut[")
        for boundary in ("attention_input", "attention_hidden", "attention_pre_mix"):
            assert annotations[boundary].startswith("pl.Out[")


@requires_pypto
def test_normalized_precision_ignores_inactive_denominator(composition):
    expected = torch.ones(1, 192, 5120, dtype=torch.bfloat16)
    actual = expected.clone()
    actual[:, 0, :32] = 2
    kwargs = dict(inputs={"num_tokens": 1}, actual_outputs={}, expected_outputs={}, rtol=1e-3, atol=1e-3)
    assert not composition.compare_normalized(actual, expected, **kwargs)[0]


@requires_pypto
def test_hidden_precision_ignores_inactive_denominator(composition):
    expected = torch.ones(1, 33, 4, 32)
    expected[:, 31:] = 13
    actual = expected.clone()
    actual[:, :31] *= 1.015
    assert not composition.compare_attention_hidden(actual, expected, inputs={"num_tokens": 31})[0]
    assert composition.compare_attention_hidden(expected.clone(), expected, inputs={"num_tokens": 31})[0]
    changed_tail = expected.clone()
    changed_tail[:, 31:] *= 1.1
    assert not composition.compare_attention_hidden(changed_tail, expected, inputs={"num_tokens": 31})[0]


@requires_pypto
def test_block_golden_rejects_inactive_capacity(composition):
    from models.deepseek_v4_1_flash._golden_smoke import make_decode_layer_golden_inputs

    inputs = make_decode_layer_golden_inputs(0)
    with pytest.raises(ValueError, match="all capacity rows"):
        composition.golden_decode_layer(**dict(inputs, num_tokens=1))


@requires_pypto
def test_stage_selection_checks_only_required_dependencies(composition):
    with pytest.raises(NotImplementedError, match="MoE"):
        composition.make_decode_layer_program(0, composition.C.EP_SIZE, 1, stage="block")
    with pytest.raises(ValueError, match="unknown decode stage"):
        composition.make_decode_layer_program(0, 1, 1, stage="ffn")


@requires_pypto
@pytest.mark.parametrize("module_name", ("decode_swa", "decode_c2a_full", "decode_c2a_reuse"))
def test_attention_uses_shared_qkv_stages(module_name):
    from importlib import import_module

    from models.deepseek_v4_1_flash import qkv_proj_rope

    attention = import_module(f"models.deepseek_v4_1_flash.{module_name}")
    for name in ("q_proj_qr", "q_proj_rope", "kv_proj_rope"):
        assert getattr(attention, name) is getattr(qkv_proj_rope, name)
    assert list(inspect.signature(qkv_proj_rope.q_proj_qr._func).parameters)[-3:] == [
        "projected", "normalized", "num_tokens",
    ]


@requires_pypto
def test_prefill_swa_keeps_specialized_qkv_stages():
    from models.deepseek_v4_1_flash import prefill_attn_swa, qkv_proj_rope

    for name in ("q_proj_qr", "q_proj_rope", "kv_proj_rope"):
        specialized = getattr(qkv_proj_rope, f"prefill_{name}")
        assert getattr(prefill_attn_swa, f"prefill_{name}") is specialized
        assert specialized is not getattr(qkv_proj_rope, name)


@requires_pypto
@pytest.mark.parametrize("module_name", ("decode_swa", "decode_c2a_full", "decode_c2a_reuse"))
def test_attention_uses_shared_output_projection(module_name):
    from importlib import import_module

    from models.deepseek_v4_1_flash import o_proj

    attention = import_module(f"models.deepseek_v4_1_flash.{module_name}")
    assert attention.o_proj is o_proj.o_proj
    assert list(inspect.signature(o_proj.o_proj._func).parameters)[-4:] == [
        "unrotated", "latent", "partial", "num_tokens",
    ]


@requires_pypto
def test_prefill_swa_keeps_specialized_output_projection():
    from models.deepseek_v4_1_flash import o_proj, prefill_attn_swa

    assert prefill_attn_swa.prefill_o_proj is o_proj.prefill_o_proj
    assert o_proj.prefill_o_proj is not o_proj.o_proj
