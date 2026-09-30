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
    DecodeLayerKind = composition.DecodeLayerKind
    REPRESENTATIVE_LAYER_IDS = composition.REPRESENTATIVE_LAYER_IDS
    kinds = [composition.decode_layer_kind(layer_id) for layer_id in range(40)]
    assert Counter(kinds) == {
        DecodeLayerKind.SWA: 2,
        DecodeLayerKind.C2A_FULL: 3,
        DecodeLayerKind.C2A_REUSE: 15,
        DecodeLayerKind.C1A_FULL: 1,
        DecodeLayerKind.C1A_REINDEX: 4,
        DecodeLayerKind.C1A_REUSE: 15,
    }
    assert {
        kind: composition.decode_layer_kind(layer_id) for kind, layer_id in REPRESENTATIVE_LAYER_IDS.items()
    } == {kind: kind for kind in DecodeLayerKind}


@requires_pypto
def test_decode_layer_schedule_resolves_cache_sources(composition):
    config = composition.C.FLASH.layer_config
    assert config(2).kv_source_layer_id == 2
    assert config(7).index_source_layer_id == 2
    assert config(8).kv_source_layer_id == 8
    assert config(19).index_source_layer_id == 14
    assert config(20).is_candidate_source
    assert config(23).index_source_layer_id == 20
    assert config(24).index_source_layer_id == 24
    assert config(39).index_source_layer_id == 36
    with pytest.raises(ValueError, match="layer_id must be"):
        config(40)


def test_decode_layer_has_no_duplicate_plan_module():
    assert not (MODEL_DIR / "decode_layer_plan.py").exists()


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
        n
        for n in block_calls
        if n in ("golden_mhc_mixes", "golden_mhc_pre", "golden_decode_moe", "golden_mhc_post")
    ] == [
        "golden_mhc_mixes",
        "golden_mhc_pre",
        "golden_mhc_post",
        "golden_mhc_mixes",
        "golden_mhc_pre",
        "golden_decode_moe",
        "golden_mhc_post",
    ]
    assert "load_decode_attention_module" not in block_calls
    device = _function(layer_tree, "decode_layer")
    assert {
        node.func.id
        for node in ast.walk(device)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } >= {
        "decode_swa_sharded",
        "decode_c2a_full_sharded",
        "decode_c2a_reuse_sharded",
        "decode_c1a_full_sharded",
        "decode_c1a_reindex_sharded",
        "decode_c1a_reuse_sharded",
        "moe",
    }
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
    } <= _string_list_assignment("o_proj.py", "__all__")

    basic_factories = {"make_mx_projection", "make_norm", "make_rope", "make_bf16_projection"}
    for module_name in ("decode_attn_swa.py", "prefill_attn_swa.py", "prefill_c1a_common.py"):
        assert _top_level_functions(module_name).isdisjoint(basic_factories)

    for module_name in ("decode_attn_swa.py", "decode_attn_c2a_full.py", "decode_attn_c2a_reuse.py"):
        names = {node.id for node in ast.walk(_tree(module_name)) if isinstance(node, ast.Name)}
        assert {"qkv_proj_rope", "o_proj"} <= names

    # C1A decode keeps only its distinct MX projection and uses shared TaskId-aware primitives.
    c1a_names = _top_level_functions("decode_attn_c1a_full.py")
    assert "make_mx_projection_with_deps" in c1a_names
    assert c1a_names.isdisjoint({"grouped_output", "make_norm", "make_rope", "make_bf16_projection"})
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


def test_decode_layer_is_a_real_device_block():
    tree = _tree("decode_layer.py")
    function = _function(tree, "decode_layer")
    decorators = {ast.unparse(node) for node in function.decorator_list}
    assert "pl.jit" in decorators
    assert "moe" in {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_decode_layer_persistent_attention_state_is_inout():
    tree = _tree("decode_layer.py")
    state_names = {
        "window_cache",
        "window_cache_scale",
        "compressed_cache",
        "compressed_cache_scale",
        "index_cache",
        "index_cache_scale",
        "state_cache",
        "topk_indices",
        "candidate_mask",
    }
    for function_name in ("decode_layer", "l3_decode_layer"):
        function = _function(tree, function_name)
        annotations = {arg.arg: ast.unparse(arg.annotation) for arg in function.args.args}
        assert all(annotations[name].startswith("pl.InOut[") for name in state_names)


def test_decode_layer_validates_every_persistent_attention_state():
    validate = _function(_tree("decode_layer.py"), "validate")
    compare_fn = next(
        keyword.value
        for node in ast.walk(validate)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "compare_fn" and isinstance(keyword.value, ast.Dict)
    )
    compared = {
        key.value
        for key in compare_fn.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert {
        "window_cache",
        "window_cache_scale",
        "compressed_cache",
        "compressed_cache_scale",
        "index_cache",
        "index_cache_scale",
        "state_cache",
        "topk_indices",
        "candidate_mask",
    } <= compared


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


def test_decode_padding_spmd_has_an_explicit_positive_guard():
    function = _function(_tree("decode_common.py"), "zero_bf16_padding")
    launch = next(node for node in ast.walk(function) if isinstance(node, ast.For))
    guard = next(node for node in ast.walk(function) if isinstance(node, ast.If))
    assert ast.unparse(guard.test) == "tokens > 0"
    assert launch in guard.body
    assert ast.unparse(launch.iter) == "pl.spmd(tokens, name_hint='decode_sp_zero_padding')"


def test_decode_layer_avoids_boolean_and_in_orchestration_conditions():
    function = _function(_tree("decode_layer.py"), "decode_layer")
    assert not any(
        isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And)
        for node in ast.walk(function)
    )


def test_decode_sequence_parallel_metadata_covers_padded_and_short_batches():
    from models.deepseek_v4_1_flash.decode_sp_integration import validate_two_layer_metadata

    # T < TP: three empty owners still take part and the gathered batch is intact.
    assert validate_two_layer_metadata(num_tokens=1, tp_size=4).hidden.shape == (1, 4)
    # A padded batch: rank 3's slab starts past the active range, rank 2 owns one
    # row of a two-row slab and must leave the other one zero.  The returned
    # tensor is the second layer's, i.e. the first layer's rows plus one.
    padded = validate_two_layer_metadata(num_tokens=5, tp_size=4, capacity=8)
    assert padded.hidden.shape == (5, 4)
    assert padded.hidden[-1].tolist() == [17.0, 18.0, 19.0, 20.0]


@requires_pypto
def test_decode_moe_token_owners_follow_contiguous_slabs(composition):
    owners = composition._sequence_parallel_token_owners
    assert owners(8, 4).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert owners(5, 4).tolist() == [0, 0, 1, 1, 2]
    assert owners(2, 4).tolist() == [0, 1]
    assert owners(0, 4).numel() == 0
    with pytest.raises(ValueError, match="tp_size must be positive"):
        owners(8, 0)


def test_decode_owner_slab_checks_cover_padding_on_every_layer():
    from models.deepseek_v4_1_flash.decode_sp_integration import (
        DecodeTokenShard,
        check_owner_slabs,
        shard_decode_batch,
        transform_owner_rows,
    )

    num_tokens, tp_size, capacity = 5, 4, 8
    hidden = torch.arange(num_tokens * 4, dtype=torch.float32).reshape(num_tokens, 4)
    token_ids = torch.arange(100, 100 + num_tokens, dtype=torch.int64)
    position_ids = torch.arange(17, 17 + num_tokens, dtype=torch.int32)
    shards = shard_decode_batch(hidden, token_ids, position_ids, tp_size=tp_size, capacity=capacity)

    # A layer updates the rows it owns only: the padding of the two-row slabs
    # survives untouched (rank 2 owns one row, rank 3 none).
    second = transform_owner_rows(
        shards, num_tokens=num_tokens, transform=lambda rows: rows + 1, capacity=capacity
    )
    check_owner_slabs(
        second, hidden + 1, num_tokens, capacity=capacity, name="residual rows", layer="the second layer"
    )
    counts = (2, 2, 1, 0)
    assert [bool((shard.hidden[count:] == 0).all()) for shard, count in zip(second, counts)] == [
        True
    ] * tp_size
    check_owner_slabs(
        second, hidden + 1, num_tokens, capacity=capacity, name="residual rows", layer="the second layer"
    )

    # A layer that rewrites its whole slab -- padding included -- is rejected on
    # that layer, before the gather would have dropped the stale rows silently.
    polluted = tuple(
        DecodeTokenShard(shard.hidden + 1, shard.token_ids, shard.position_ids, shard.valid_mask)
        for shard in shards
    )
    with pytest.raises(RuntimeError, match="non-zero residual rows padding"):
        check_owner_slabs(
            polluted,
            hidden + 1,
            num_tokens,
            capacity=capacity,
            name="residual rows",
            layer="the second layer",
        )


def test_decode_swa_sharded_reduce_comparison_asserts_zero_padding():
    source = (MODEL_DIR / "decode_swa.py").read_text()
    segment = ast.get_source_segment(source, _function(ast.parse(source), "comparisons_sharded"))
    assert segment is not None
    # The ReduceScatter publishes a fully materialized slab, so the sharded
    # comparison has to assert the inactive rows instead of ignoring them.
    assert '"attention_output"' in segment
    assert (
        "make_compare_sharded_rows(swa.compare_output)" in segment or "require_zero_padding=True" in segment
    )


@requires_pypto
def test_decode_swa_sharded_reduce_rejects_stale_padding():
    from models.deepseek_v4_1_flash import decode_swa

    compare = decode_swa.comparisons_sharded()["attention_output"]
    # tokens=7 over TP4 slabs of width 2: rank 3 owns one row and must zero the
    # other, which is what the ReduceScatter publishes.
    ranks, width, dim = 4, 2, 8
    golden = torch.zeros(ranks, width, dim, dtype=torch.float32)
    actual = torch.zeros(ranks, width, dim, dtype=torch.float32)
    for rank, count in enumerate((2, 2, 2, 1)):
        golden[rank, :count] = 1.0
        actual[rank, :count] = 1.0
    assert compare(actual, golden, inputs={})[0]
    stale = actual.clone()
    stale[3, 1:] = 5.0
    passed, detail = compare(stale, golden, inputs={})
    assert not passed and "inactive" in detail


def test_decode_wiring_selection_namespaces_replay_directories():
    from models.deepseek_v4_1_flash.decode_common import (
        selected_wirings,
        wiring_replay_dir,
    )

    assert selected_wirings("both") == ("replicated", "sharded")
    assert selected_wirings("replicated") == ("replicated",)
    assert selected_wirings("sharded") == ("sharded",)
    with pytest.raises(ValueError):
        selected_wirings("nope")
    # The two wirings are different ABIs, so one replay tree is never shared.
    assert wiring_replay_dir("out/data", "replicated") == "out/data/replicated"
    assert wiring_replay_dir("out/data", "sharded") == "out/data/sharded"
    assert wiring_replay_dir(None, "sharded") is None
    with pytest.raises(ValueError):
        wiring_replay_dir("out/data", "both")


def test_decode_owner_slab_comparisons_follow_ownership_not_capacity():
    import torch

    from models.deepseek_v4_1_flash.decode_common import (
        active_rows_from_golden,
        compare_owner_rows,
    )

    def owned_rows(rank, width, num_tokens):
        first = min(rank * width, num_tokens)
        return max(0, min(width, num_tokens - first))

    def slabs(num_tokens, width, ranks=4):
        golden = torch.zeros(ranks, width, 2, 4)
        actual = torch.zeros(ranks, width, 2, 4)
        for rank in range(ranks):
            count = owned_rows(rank, width, num_tokens)
            golden[rank, :count] = 1.0
            actual[rank, :count] = 1.0
            # A kernel need not clear rows it does not own, so the slab's padding
            # stays stale; only ownership decides what is compared.
            actual[rank, count:] = 9.0
        return actual, golden

    def equal(actual, expected, **_kwargs):
        return bool(torch.equal(actual, expected)), "owned rows differ"

    plain = compare_owner_rows(equal, name="HC output")
    zeroed = compare_owner_rows(equal, name="HC output", require_zero_padding=True)
    # (num_tokens, slab width): T < TP, a partially filled slab, and a full one.
    for num_tokens, width in ((1, 1), (7, 2), (32, 8)):
        actual, golden = slabs(num_tokens, width)
        assert plain(actual, golden, inputs={})[0], (num_tokens, width)
        assert active_rows_from_golden(golden) == num_tokens
    actual, golden = slabs(7, 2)
    # Only a ReduceScatter contract reads the inactive rows, and it must see them.
    assert not zeroed(actual, golden, inputs={})[0]
    assert zeroed(*slabs(32, 8), inputs={})[0]
    # An explicit batch size wins over the harness scalar, and the C1A path has
    # only the explicit one: its fixtures never expose ``num_tokens``.
    explicit = compare_owner_rows(equal, name="HC output", active=7)
    assert explicit(actual, golden, inputs={"num_tokens": 8})[0]
    assert not explicit(torch.full((4, 2, 2, 4), 2.0), golden, inputs={})[0]


@requires_pypto
def test_decode_c1a_wiring_flag_selects_the_run_and_the_replay_tree():
    from models.deepseek_v4_1_flash.decode_c1a_full import parse_c1a_hc_args

    args, devices = parse_c1a_hc_args("full", ["--wiring", "sharded", "--tokens", "7"])
    assert args.wiring == "sharded"
    assert args.local_tokens == 2
    # A single wiring still gets the default device set (0 .. TP-1).
    assert devices == list(range(len(devices)))
    args, _ = parse_c1a_hc_args("full", ["--golden-data", "out/data"])
    assert args.wiring == "both"
    with pytest.raises(SystemExit):
        parse_c1a_hc_args("full", ["--wiring", "nope"])


@requires_pypto
def test_decode_sequence_parallel_two_layer_chain(composition):
    from models.deepseek_v4_1_flash._golden_smoke import run_two_layer_decode_chain

    run_two_layer_decode_chain(composition.golden_decode_layer, tp_size=4)


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
def test_decode_layer_golden_selection_is_static(layer_id, composition):
    golden = composition.ATTENTION_GOLDENS[composition.decode_layer_kind(layer_id)]
    assert golden.__name__.startswith("golden_decode_attn_")


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
    from importlib import import_module

    C = attention_common.C
    module_name = {
        composition.DecodeLayerKind.SWA: "decode_swa",
        composition.DecodeLayerKind.C2A_FULL: "decode_c2a_full",
        composition.DecodeLayerKind.C2A_REUSE: "decode_c2a_reuse",
    }[composition.decode_layer_kind(layer_id)]
    mode = import_module(f"models.deepseek_v4_1_flash.{module_name}")
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


def test_decode_layer_has_no_legacy_stage_dispatcher():
    names = _top_level_functions("decode_layer.py")
    assert "make_decode_layer_program" not in names
    assert "decode_layer_kernel_skip_reason" not in names
    assert "attention_half_skip_reason" not in names


def test_decode_fwd_directly_composes_the_backbone():
    tree = _tree("decode_fwd.py")
    imported_names = {
        alias.name for node in tree.body if isinstance(node, ast.ImportFrom) for alias in node.names
    }
    referenced_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "decode_layer" not in imported_names
    assert "decode_layer" not in referenced_names
    assert {
        "decode_swa_sharded",
        "decode_c2a_full_sharded",
        "decode_c2a_reuse_sharded",
        "decode_c1a_full_sharded",
        "decode_c1a_reindex_sharded",
        "decode_c1a_reuse_sharded",
        "moe",
    } <= referenced_names
    assert {"decode_fwd", "l2_decode_fwd"} <= referenced_names
    assert "l3_decode_fwd" in _top_level_functions("decode_fwd.py")


@requires_pypto
def test_decode_moe_capacity_covers_the_largest_local_slab():
    from models.deepseek_v4_1_flash import config, moe

    local_slab = (config.DECODE_MAX_TOKENS + config.TP_SIZE - 1) // config.TP_SIZE
    assert config.MOE_TOKENS >= local_slab
    assert config.MOE_TOKENS % config.MOE_ROW_TILE == 0
    assert config.MOE_RECV_MAX == config.EP_SIZE * config.MOE_TOKENS
    assert int(moe._normalise_num_tokens(17).min()) == 17


@requires_pypto
def test_decode_moe_precision_guard_is_stable_near_zero():
    from models.deepseek_v4_1_flash import config, moe

    shape = (config.EP_SIZE, config.MOE_TOKENS, 1, 1)
    expected = torch.zeros(shape, dtype=torch.float32)
    actual = expected.clone()
    expected[0, 0, 0, 0] = -0.01
    actual[0, 0, 0, 0] = 0.03

    compare = moe._local_mhc_compare(config.MOE_TOKENS)
    compare_kwargs = {
        "actual_outputs": {},
        "expected_outputs": {},
        "inputs": {},
        "rtol": 0.0,
        "atol": 0.0,
    }
    assert compare(actual, expected, **compare_kwargs)[0]

    actual[0, 0, 0, 0] = 0.5
    passed, detail = compare(actual, expected, **compare_kwargs)
    assert not passed
    assert "max_abs_diff=0.25" in detail


def test_decode_moe_orders_output_zero_before_combine_reduce():
    moe_core = _function(_tree("moe.py"), "_moe_core")
    combine_call = next(
        node
        for node in ast.walk(moe_core)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "combine_scattered"
    )
    assert "_output_zero_tid" in {ast.unparse(arg) for arg in combine_call.args}

    combine = _function(_tree("ep_transport.py"), "combine_scattered")
    reduce_spmd = next(
        node
        for node in ast.walk(combine)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and ast.unparse(node.func) == "pl.spmd"
        and any(
            keyword.arg == "name_hint"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "combine_reduce"
            for keyword in node.keywords
        )
    )
    deps = next(keyword.value for keyword in reduce_spmd.keywords if keyword.arg == "deps")
    assert "output_ready" in {ast.unparse(element) for element in deps.elts}


def test_decode_moe_uses_dspark_dispatch_and_accumulated_epochs():
    source = (MODEL_DIR / "ep_transport.py").read_text()

    assert "SIGNAL_PAD = 128" in source
    assert "[N_RANKS, N_LOCAL, SIGNAL_PAD]" in source
    assert "N_RANKS * N_LOCAL," in source
    assert "offsets=[my_rank, 0, 0]" in source
    assert "offsets=[src, 0, 0]" in source
    assert "op=pld.NotifyOp.AtomicAdd" in source
    assert "expected=pl.cast(moe_epoch * N_LOCAL, pl.INT32)" in source
    assert "pld.system.defer_wait(" in source
    assert "op=pld.NotifyOp.Set" not in source
    assert "for tile in pl.range(SCALE_PACK_TILES):" in source
    assert "gather_tile_tids[tile] = pl.system.task_dummy(deps=[])" in source


def test_prefill_layer_uses_decode_moe_transport_signal_layout():
    source = (MODEL_DIR / "prefill_layer.py").read_text()

    assert "from models.deepseek_v4_1_flash.ep_transport import SIGNAL_PAD" in source
    assert "[EP_SIZE, SIGNAL_PAD]" in source
    assert "[EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD]" in source
    assert "arrived_buffer, [EP_SIZE, SIGNAL_PAD]" in source
    assert "data_arrived_buffer,\n                [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD]" in source
    assert "combine_arrived_buffer,\n                [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD]" in source


@pytest.mark.parametrize("module_name", ["decode_layer", "decode_fwd"])
def test_decode_entry_uses_moe_transport_signal_layout(module_name):
    source = (MODEL_DIR / f"{module_name}.py").read_text()

    assert "from models.deepseek_v4_1_flash.ep_transport import SIGNAL_PAD" in source
    assert "[EP_SIZE, SIGNAL_PAD]" in source
    assert "[EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD]" in source
    assert "arrived_buffer, [EP_SIZE, SIGNAL_PAD]" in source
    assert "data_arrived_buffer,\n            [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD]" in source
    assert "combine_arrived_buffer,\n            [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD]" in source


def test_decode_moe_orders_dispatch_and_expert_completion_edges():
    expert_source = (MODEL_DIR / "expert_routed.py").read_text()
    assert expert_source.count("for tile in pl.range(TILES_PER_EXPERT):") == 2
    assert expert_source.count(
        "tile_completion_tids[tile] = pl.system.task_dummy(deps=[])"
    ) == 2
    assert "deps=[gate_mxfp4_aiv_tid, input_ready]" in expert_source
    assert "deps=[up_mxfp4_aiv_tid, input_ready]" in expert_source
    assert "tile_completion_tids[t] = gate_up_act_quant_tid" in expert_source
    assert "deps=[w2_mxfp4_aiv_tid, hidden_completion_tids[local_e]]" in expert_source
    assert "deps=[route_weight_tid]" in expert_source
    assert 'name_hint="expert_tile_scatter"' in expert_source
    assert "src=recv_y_tile" in expert_source
    assert "tile_completion_tids[tt] = scatter_tid" in expert_source
    assert 'name_hint="expert_tile_store"' not in expert_source
    assert "return scatter_done" in expert_source

    moe_source = (MODEL_DIR / "moe.py").read_text()
    assert "input_ready = dispatch(" in moe_source
    assert "scatter_done = expert_routed_scatter(" in moe_source
    assert "combine_scattered(" in moe_source
    assert "_output_zero_tid,\n            scatter_done," in moe_source

    for entry in ("decode_layer.py", "decode_fwd.py", "prefill_layer.py"):
        source = (MODEL_DIR / entry).read_text()
        assert "from models.deepseek_v4_1_flash.moe import (" in source
        assert "moe_dspark" not in source


@requires_pypto
def test_decode_fwd_exposes_a_full_compile_fixture():
    from golden import ScalarSpec
    from models.deepseek_v4_1_flash import decode_fwd

    specs = decode_fwd.build_tensor_specs()
    assert [spec.name for spec in specs] == list(decode_fwd.l3_decode_fwd.param_names)
    by_name = {spec.name: spec for spec in specs}
    assert by_name["x_hc"].shape[1] == decode_fwd.MOE_TOKENS
    assert by_name["x_hc"].shape[1] > 16
    assert by_name["window_cache_pool"].shape[1] == decode_fwd.N_LAYERS
    assert by_name["compressed_cache_pool"].shape[1] == decode_fwd.KV_SOURCE_COUNT
    assert by_name["index_cache_pool"].shape[1] == decode_fwd.INDEX_SOURCE_COUNT
    assert by_name["state_cache_pool"].shape[1] == decode_fwd.C2A_SOURCE_COUNT
    assert isinstance(by_name["attention_num_tokens"], ScalarSpec)
    assert by_name["attention_num_tokens"].compile_runtime
    source = (MODEL_DIR / "decode_fwd.py").read_text()
    assert "# ci: a5" in source
    assert "# ci: no-sim" in source


@requires_pypto
def test_decode_fwd_schedule_and_cache_source_ordinals():
    from models.deepseek_v4_1_flash import decode_fwd

    modes = [layer.mode.value for layer in decode_fwd.BACKBONE_SCHEDULE]
    assert Counter(modes) == {
        "swa": 2,
        "full": 4,
        "reuse": 30,
        "reindex": 4,
    }
    assert decode_fwd.cache_source_ordinals(0) == (None, None, None)
    assert decode_fwd.cache_source_ordinals(2) == (0, 0, 0)
    assert decode_fwd.cache_source_ordinals(7) == (0, 0, 0)
    assert decode_fwd.cache_source_ordinals(8) == (1, 1, 1)
    assert decode_fwd.cache_source_ordinals(19) == (2, 2, 2)
    assert decode_fwd.cache_source_ordinals(20) == (3, 3, None)
    assert decode_fwd.cache_source_ordinals(24) == (3, 4, None)
    assert decode_fwd.cache_source_ordinals(39) == (3, 7, None)


def test_decode_fwd_reuses_one_set_of_tp_and_ep_windows():
    tree = _tree("decode_fwd.py")
    host = _function(tree, "l3_decode_fwd")
    assert [ast.unparse(decorator) for decorator in host.decorator_list] == ["pl.jit.host"]
    allocations = [
        node
        for node in ast.walk(host)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "alloc_window_buffer"
    ]
    assert len(allocations) == 13
    assert (
        sum(
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Attribute)
            and node.iter.func.attr == "range"
            for node in ast.walk(host)
        )
        == 1
    )


def test_decode_fwd_uses_runtime_schedule_blocks():
    tree = _tree("decode_fwd.py")
    device = _function(tree, "_decode_fwd")
    loops = [node for node in device.body if isinstance(node, ast.For)]
    assert [(ast.unparse(loop.target), ast.unparse(loop.iter)) for loop in loops] == [
        ("c2a_block", "pl.range(C2A_SOURCE_COUNT)"),
        ("c1a_block", "pl.range(4)"),
    ]
    calls = Counter(
        node.func.id
        for node in ast.walk(device)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and (node.func.id.startswith("decode_") or node.func.id == "moe")
    )
    assert calls == {
        "decode_swa_sharded": 2,
        "decode_c2a_full_sharded": 1,
        "decode_c2a_reuse_sharded": 5,
        "decode_c1a_full_sharded": 1,
        "decode_c1a_reindex_sharded": 1,
        "decode_c1a_reuse_sharded": 6,
        "moe": 16,
    }


def test_decode_fwd_passes_row_concatenated_mx_scale_banks():
    tree = _tree("decode_fwd.py")
    device = _function(tree, "_decode_fwd")
    host = _function(tree, "l3_decode_fwd")
    scale_names = {
        "wq_a_scale",
        "wq_b_scale",
        "wkv_scale",
        "wo_b_scale",
        "index_wq_b_scale",
        "routed_w1_scale",
        "routed_w2_scale",
        "routed_w3_scale",
        "shared_w1_scale",
        "shared_w2_scale",
        "shared_w3_scale",
    }
    assert len(device.args.args) == 83
    assert scale_names <= {argument.arg for argument in device.args.args}
    assert not any(
        argument.arg.startswith(tuple(f"{name}_l" for name in scale_names))
        or argument.arg.startswith("index_wq_b_scale_s")
        for argument in device.args.args
    )
    host_annotations = {argument.arg: ast.unparse(argument.annotation) for argument in host.args.args}
    for name in scale_names:
        annotation = host_annotations[name]
        assert "pl.MX_B_NN" not in annotation
        assert "EP_SIZE" in annotation
        assert "N_LAYERS," not in annotation
    mx_slices = [
        node
        for node in ast.walk(device)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.annotation, ast.Subscript)
        and "pl.MX_B_NN" in ast.unparse(node.annotation)
    ]
    assert mx_slices
    index_scale_slices = [
        node
        for node in mx_slices
        if isinstance(node.target, ast.Name) and "index_wq_b_scale_source" in node.target.id
    ]
    assert len(index_scale_slices) == 3
    host_loop = next(node for node in host.body if isinstance(node, ast.For))
    assert not any(
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id.endswith("_scale_rank")
        for node in host_loop.body
    )


def test_decode_fwd_hc_scale_slices_use_three_values_per_layer():
    device = _function(_tree("decode_fwd.py"), "_decode_fwd")
    scale_slices = [
        node
        for node in ast.walk(device)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "slice"
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id in {"hc_attn_scale", "hc_ffn_scale"}
    ]
    assert len(scale_slices) == 32
    for scale_slice in scale_slices:
        assert ast.unparse(scale_slice.args[1]) == "[3]"
        offset = scale_slice.args[2].elts[0]
        assert isinstance(offset, ast.BinOp) and isinstance(offset.op, ast.Mult)
        assert isinstance(offset.right, ast.Constant) and offset.right.value == 3


def _ced_metadata(lengths):
    from models.deepseek_v4_1_flash.metadata import build_forward_metadata

    starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    table = torch.arange(len(lengths) * 4, dtype=torch.int32).reshape(-1, 4) + 10
    global_tables = {layer: table + 30 for layer in (2, 8, 14, 20)}
    state_tables = {layer: torch.arange(len(lengths), dtype=torch.int32)[:, None] for layer in (2, 8, 14)}
    return build_forward_metadata(starts, torch.zeros(len(lengths), dtype=torch.int32), table,
                                  global_tables, state_tables), table


@requires_pypto
def test_ced_decoder_replay_gathers_matching_state_and_delayed_mix():
    from models.deepseek_v4_1_flash import config as C
    from models.deepseek_v4_1_flash.metadata import PrefillState
    from models.deepseek_v4_1_flash.metadata import select_decoder_replay

    metadata, table = _ced_metadata([129, 7])
    hidden = torch.zeros(136, C.HC_MULT, C.D)
    hidden[:, 0, 0] = torch.arange(136)
    state = PrefillState(hidden, hidden[:, :, 0].clone())
    replay = select_decoder_replay(metadata, table)
    selected = state.take_rows(replay.source_rows)
    assert state.num_tokens == 136
    assert selected.num_tokens == 135
    assert replay.query_lens.tolist() == [128, 7]
    assert replay.source_rows.tolist() == list(range(1, 136))
    assert torch.equal(selected.x_hc[:, 0, 0], torch.arange(1, 136).float())
    assert torch.equal(selected.pre_mix[:, 0], selected.x_hc[:, 0, 0])
    assert replay.logit_row_indices.tolist() == [127, 134]


@requires_pypto
def test_ced_replay_masks_request_boundaries_and_rejects_prefixes():
    from models.deepseek_v4_1_flash import config as C
    from models.deepseek_v4_1_flash.metadata import PrefillState
    from models.deepseek_v4_1_flash.metadata import select_decoder_replay

    metadata, table = _ced_metadata([0, 1, 127, 128, 129])
    replay = select_decoder_replay(metadata, table)
    assert replay.query_lens.tolist() == [0, 1, 127, 128, 128]
    assert replay.replay_starts.tolist() == [0, 0, 0, 0, 1]
    for row, request in enumerate(replay.token_to_req_indices.tolist()):
        positions = torch.arange(int(replay.replay_starts[request]), int(replay.position_ids[row]) + 1)
        expected = table[request, positions // 128] * 128 + positions % 128
        assert torch.equal(replay.window_indices[row, :len(positions)], expected.int())
        assert bool((replay.window_indices[row, len(positions):] == -1).all())
    metadata.kv_seq_lens[0] = 1
    with pytest.raises(ValueError, match="cached prefixes"):
        select_decoder_replay(metadata, table)
    with pytest.raises(ValueError, match="share"):
        PrefillState(torch.zeros(1, C.HC_MULT, C.D), torch.zeros(1, C.HC_MULT - 1))


@requires_pypto
def test_ced_original_weight_codes_scales_and_selected_rows(tmp_path):
    import json
    import struct

    from models.deepseek_v4_1_flash.prefill_fwd import PrefillCheckpoint, decode_checkpoint_linear

    codes = torch.arange(16, dtype=torch.uint8).repeat(2)
    packed = (codes[::2] | (codes[1::2] << 4)).repeat(2, 1).view(torch.int8)
    scale = torch.tensor([[127], [128]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.]).repeat(2)
    expected = torch.stack((values, values * 2))
    actual = decode_checkpoint_linear(packed, scale)
    assert torch.equal(actual, expected) and torch.equal(actual.signbit(), expected.signbit())
    fp8 = torch.arange(-16, 16).float().repeat(32, 1).to(torch.float8_e4m3fn)
    assert torch.equal(decode_checkpoint_linear(fp8, scale[:1]), fp8.float())
    payload = fp8[:3, :2].contiguous()
    header = json.dumps({"table": {"dtype": "F8_E4M3", "shape": [3, 2], "data_offsets": [0, 6]}}).encode()
    (tmp_path / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(payload.view(torch.uint8).flatten().tolist()))
    reader = object.__new__(PrefillCheckpoint)
    reader.root, reader.weight_map, reader._headers = tmp_path, {"table": "weights.safetensors"}, {}
    ids = torch.tensor([[2, 0], [2, 1]])
    assert torch.equal(reader.tensor_rows("table", ids).view(torch.uint8), payload.view(torch.uint8)[ids])
    with pytest.raises(ValueError, match="invalid table rows"):
        reader.tensor_rows("table", torch.tensor([3]))


@requires_pypto
def test_ced_reference_contractions_match_independent_cancellation_and_sink_oracle():
    from models.deepseek_v4_1_flash.golden import prefill_einsum as einsum
    from models.deepseek_v4_1_flash.golden import prefill_linear as linear
    import math
    from models.deepseek_v4_1_flash.golden import prefill_matmul as matmul
    from models.deepseek_v4_1_flash.golden import prefill_sparse_attention_reference as sparse_attention_reference

    x = torch.tensor([[2.0**24, 1., -(2.0**24)]])
    weight = torch.ones(1, 3)
    expected = torch.tensor([[math.fsum(float(value) for value in x[0])]])
    for result in (linear(x, weight), matmul(x, weight.T), einsum("mk,nk->mn", x, weight)):
        assert result.dtype == torch.float32 and torch.equal(result, expected)
    query = torch.tensor([[[2.0**24, 1., -(2.0**24), 1.]]])
    cache = torch.zeros(1, 128, 4)
    cache[0, 0] = 1
    result = sparse_attention_reference(query, cache, torch.tensor([[0]], dtype=torch.int32), None, None, torch.zeros(1))
    # Exact QK=2, scale=1/sqrt(4); the zero-valued sink contributes exp(0).
    torch.testing.assert_close(result, torch.full_like(query, 1 / (1 + math.exp(-1))))


@requires_pypto
def test_ced_fresh_geometry_limits_fail_before_cache_allocation(monkeypatch):
    from models.deepseek_v4_1_flash.metadata import PrefillContext

    class AllocationReached(Exception):
        pass

    def stop(*_args, **_kwargs):
        raise AllocationReached

    monkeypatch.setattr(torch, "arange", stop)
    for lengths, capacity in (([0], None), ([2], 1), ([4097], None), ([129] * 17, None)):
        with pytest.raises(ValueError):
            PrefillContext(None, lengths, capacity)
    with pytest.raises(AllocationReached):
        PrefillContext(None, [4096])


@requires_pypto
def test_ced_ratio_two_pairs_and_full_encoder_publication_keep_request_ownership():
    from models.deepseek_v4_1_flash.metadata import PrefillContext

    context = PrefillContext(None, [129, 7])
    assert context.precision == "fp32"
    inputs = context.layer_inputs(2, load_weights=False)
    expected = torch.full((136,), -1, dtype=torch.int32)
    expected[1:129:2] = torch.arange(0, 128, 2, dtype=torch.int32)
    expected[130:136:2] = torch.arange(129, 135, 2, dtype=torch.int32)
    assert torch.equal(inputs.previous_rows, expected)
    assert inputs.metadata.compressed_slots[[128, 129]].tolist() == [-1, -1]
    publisher = context.publisher_inputs(load_weights=False).metadata
    assert publisher.compressed_slots.numel() == 136 and bool((publisher.compressed_slots >= 0).all())


@requires_pypto
def test_ced_context_uses_released_rope_profiles_for_every_layer():
    from models.deepseek_v4_1_flash.metadata import PrefillContext
    import math

    context = PrefillContext(None, [129, 7])
    for layer in range(40):
        data = context.layer_inputs(layer, load_weights=False).metadata
        positions = context.metadata.position_ids if layer < 20 else context.replay.position_ids
        base = 10000.0 if layer < 2 else 160000.0
        frequency = torch.tensor([base ** (-2 * k / 64) for k in range(32)], dtype=torch.float64)
        if layer >= 2:
            low = max(math.floor(64 * math.log(65536 / (32 * 2 * math.pi)) / (2 * math.log(base))), 0)
            high = min(math.ceil(64 * math.log(65536 / (2 * math.pi)) / (2 * math.log(base))), 63)
            ramp = ((torch.arange(32, dtype=torch.float64) - low) / max(high - low, 1e-3)).clamp(0, 1)
            frequency = frequency * (1 - ramp) + frequency / 16 * ramp
        phase = positions.double()[:, None] * frequency
        torch.testing.assert_close(data.rope_cos, phase.cos().float(), rtol=1e-6, atol=1e-5)
        torch.testing.assert_close(data.rope_sin, phase.sin().float(), rtol=1e-6, atol=1e-5)


@requires_pypto
def test_ced_precision_modes_preserve_default_and_released_rounding_boundaries():
    from models.deepseek_v4_1_flash.metadata import PrefillContext
    from models.deepseek_v4_1_flash.golden import prefill_linear as linear
    from models.deepseek_v4_1_flash.golden import prefill_quantize_dequantize as quantize_dequantize
    from models.deepseek_v4_1_flash.golden import prefill_round_activation as round_activation

    x = torch.tensor([[1 + 2.0**-9] * 32])
    weight = torch.ones(1, 32)
    assert linear(x, weight).item() == 32 + 2.0**-4
    assert linear(x, weight, precision="official", format="bf16").item() == 32
    assert linear(x, weight, precision="official", format="mxfp8").item() == 32
    assert round_activation(x) is x
    # An amax of six fixes the index scale at one. Halfway values choose even
    # FP4 codes, including signed zero; these are literal released code points.
    values = torch.tensor([[.25, .75, 1.25, 1.75, 2.5, 3.5, 5., -.25, 6.] + [0.] * 23])
    rounded = quantize_dequantize(values, "index_fp4")
    assert rounded[0, :9].tolist() == [0., 1., 1., 2., 2., 4., 4., -0., 6.]
    assert rounded[0, 7].signbit()
    with pytest.raises(ValueError, match="precision"):
        PrefillContext(None, [1], precision="unknown")
    context = PrefillContext(None, [17, 129], precision="official")
    encoder = context.layer_inputs(2, load_weights=False).metadata.attention_extents
    decoder = context.layer_inputs(20, load_weights=False).metadata.attention_extents
    # Global keys follow each request's actual window width, before online64
    # grouping, rather than always starting at padded column128.
    assert encoder[[0, 17]].tolist() == [[17, 8], [128, 64]]
    assert decoder[[0, 17]].tolist() == [[17, 17], [128, 129]]
