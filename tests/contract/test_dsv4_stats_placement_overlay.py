# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from golden import TensorSpec


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "deepseek_v4_flash_mtp"
sys.path.insert(0, str(_MODEL_DIR))

from stats_placement_fixture import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    PLACED_NAMES,
    adapt_mtp_stats_placement_specs,
    adapt_single_layer_stats_placement_specs,
    eplb_replay_placement,
    load_stats_placement_manifest,
    make_stats_placement_spec,
    physical_to_logical_for_placement,
    stats_layer_spec_factory,
)
from stats_route_fixture import apportion_route_counts  # noqa: E402
from eplb_fixture import (  # noqa: E402
    CONTIGUOUS_PLACEMENT,
    EPLB_PLACEMENT,
    STATIC_STATS_PLACEMENT,
)


def _subprocess_check(source: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(_MODEL_DIR), str(_REPO_ROOT), env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=_MODEL_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _logical_expert_tensor(local_experts: int, *tail: int) -> torch.Tensor:
    shape = [8, local_experts, *tail]
    logical_ids = torch.arange(8 * local_experts, dtype=torch.int64).reshape(8, local_experts)
    return logical_ids.reshape(8, local_experts, *([1] * len(tail))).expand(shape).clone()


def _replicated_expert_tensor(*tail: int) -> torch.Tensor:
    shape = [8, 256, *tail]
    logical_ids = torch.arange(256, dtype=torch.int64).reshape(1, 256)
    return logical_ids.reshape(1, 256, *([1] * len(tail))).expand(shape).clone()


def test_checked_in_manifest_is_the_full_decode_profile() -> None:
    manifest = load_stats_placement_manifest(DEFAULT_MANIFEST_PATH)

    assert manifest["source"]["sha256"] == (
        "65d165d9e934f2a273b615b4f6e83b253114a4884da6549d9e2d15a46d2360e3"
    )
    assert manifest["filters"] == {"phases": ["decode_mtp"], "routed_tokens": [384]}
    assert manifest["topology"] == {
        "experts": 256,
        "layers": 44,
        "local_experts": 32,
        "ranks": 8,
    }


def test_checked_in_placement_reduces_every_replayed_layer_peak() -> None:
    manifest = load_stats_placement_manifest()
    contiguous_peaks = []
    stats_peaks = []

    for layer_id in range(44):
        route_counts = apportion_route_counts(
            manifest["expert_loads"][layer_id],
            total_routes=384,
        )
        contiguous_loads = [sum(route_counts[rank * 32 : (rank + 1) * 32]) for rank in range(8)]
        stats_loads = [
            sum(route_counts[expert_id] for expert_id in logical_experts)
            for logical_experts in manifest["layers"][layer_id]["rank_to_logical"]
        ]
        contiguous_peaks.append(max(contiguous_loads))
        stats_peaks.append(max(stats_loads))

    assert all(stats <= contiguous for stats, contiguous in zip(stats_peaks, contiguous_peaks))
    assert sum(stats_peaks) / len(stats_peaks) == pytest.approx(49.79545454545455)
    assert sum(contiguous_peaks) / len(contiguous_peaks) == pytest.approx(85.93181818181819)


def test_manifest_rejects_an_unversioned_algorithm_change(tmp_path: Path) -> None:
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["algorithm"]["version"] = 2
    path = tmp_path / "wrong-algorithm.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="placement manifest algorithm"):
        load_stats_placement_manifest(path)


@pytest.mark.parametrize(
    ("name", "tail"),
    [
        ("routed_w1", (2, 3)),
        ("routed_w1_scale", (2,)),
        ("routed_w3", (2, 3)),
        ("routed_w3_scale", (2,)),
        ("routed_w2", (3, 2)),
        ("routed_w2_scale", (3,)),
    ],
)
def test_routed_weights_and_scales_use_the_same_physical_order(name: str, tail: tuple[int, ...]) -> None:
    manifest = load_stats_placement_manifest()
    source = _logical_expert_tensor(32, *tail)
    base_spec = TensorSpec(
        name,
        list(source.shape),
        source.dtype,
        init_value=lambda: source,
        resident="stacked",
    )

    placed = make_stats_placement_spec(name, base_spec, layer_ids=[0, 43]).create_tensor()

    assert list(placed.shape[:2]) == [8, 64]
    for stack_index, layer_id in enumerate((0, 43)):
        physical_to_logical = torch.tensor(
            manifest["layers"][layer_id]["rank_to_logical"], dtype=torch.int64
        ).reshape(-1)
        layer = placed[:, stack_index * 32 : (stack_index + 1) * 32]
        actual = layer[(...,) + (0,) * len(tail)].reshape(-1)
        assert torch.equal(actual, physical_to_logical)


@pytest.mark.parametrize("name", ["gate_w", "gate_bias"])
def test_gate_rows_follow_the_physical_expert_order(name: str) -> None:
    manifest = load_stats_placement_manifest()
    source = _replicated_expert_tensor(1)
    base_spec = TensorSpec(name, list(source.shape), source.dtype, init_value=lambda: source)

    placed = make_stats_placement_spec(name, base_spec, layer_ids=[1, 42]).create_tensor()

    for stack_index, layer_id in enumerate((1, 42)):
        physical_to_logical = torch.tensor(
            manifest["layers"][layer_id]["rank_to_logical"], dtype=torch.int64
        ).reshape(-1)
        layer = placed[:, stack_index * 256 : (stack_index + 1) * 256, 0]
        assert torch.equal(layer[0], physical_to_logical)
        assert torch.equal(layer, layer[0].unsqueeze(0).expand_as(layer))


@pytest.mark.parametrize("placement", [STATIC_STATS_PLACEMENT, EPLB_PLACEMENT])
def test_placed_tid2eid_preserves_histograms_after_physical_mapping(placement: str) -> None:
    manifest = load_stats_placement_manifest()
    base_spec = TensorSpec("tid2eid", [8, 64, 6], torch.int32, resident="stacked")

    spec = make_stats_placement_spec(
        "tid2eid",
        base_spec,
        layer_ids=[0, 43],
        placement=placement,
    )
    placed = spec.create_tensor()

    assert spec.resident == "stacked"
    assert tuple(placed.shape) == (8, 128, 6)
    for stack_index, layer_id in enumerate((0, 43)):
        physical_routes = placed[0, stack_index * 64 : (stack_index + 1) * 64]
        assert torch.equal(
            placed[:, stack_index * 64 : (stack_index + 1) * 64],
            physical_routes.unsqueeze(0).expand(8, -1, -1),
        )
        assert all(len(set(row.tolist())) == 6 for row in physical_routes)

        physical_to_logical = torch.tensor(
            physical_to_logical_for_placement(manifest, layer_id, placement),
            dtype=torch.int64,
        )
        logical_routes = physical_to_logical[physical_routes.to(torch.int64)]
        actual_counts = torch.bincount(logical_routes.reshape(-1), minlength=256)
        expected_counts = torch.tensor(
            apportion_route_counts(manifest["expert_loads"][layer_id], total_routes=384)
        )
        assert torch.equal(actual_counts, expected_counts)


def test_contiguous_control_preserves_logical_route_ids_and_histograms() -> None:
    manifest = load_stats_placement_manifest()
    base_spec = TensorSpec("tid2eid", [8, 64, 6], torch.int32, resident="stacked")

    spec = make_stats_placement_spec(
        "tid2eid",
        base_spec,
        layer_ids=[0, 43],
        placement=CONTIGUOUS_PLACEMENT,
    )
    routes = spec.create_tensor()

    for stack_index, layer_id in enumerate((0, 43)):
        logical_routes = routes[0, stack_index * 64 : (stack_index + 1) * 64]
        actual_counts = torch.bincount(logical_routes.reshape(-1).to(torch.int64), minlength=256)
        expected_counts = torch.tensor(
            apportion_route_counts(manifest["expert_loads"][layer_id], total_routes=384)
        )
        assert torch.equal(actual_counts, expected_counts)
    assert (
        make_stats_placement_spec(
            "gate_w",
            TensorSpec("gate_w", [8, 256, 1], torch.float32),
            layer_ids=[0],
            placement=CONTIGUOUS_PLACEMENT,
        )
        is None
    )


def test_eplb_route_spec_rejects_a_nonbenchmark_topk() -> None:
    base_spec = TensorSpec("tid2eid", [8, 64, 5], torch.int32, resident="stacked")

    with pytest.raises(ValueError, match="EPLB replay requires topk=6"):
        make_stats_placement_spec(
            "tid2eid",
            base_spec,
            layer_ids=[0],
            placement=EPLB_PLACEMENT,
        )


@pytest.mark.parametrize("placement", [STATIC_STATS_PLACEMENT, EPLB_PLACEMENT])
@pytest.mark.parametrize("layer_id", [0, 42, 43])
def test_route_gate_and_routed_weight_permutations_preserve_logical_experts(
    layer_id: int,
    placement: str,
) -> None:
    route_base = TensorSpec("tid2eid", [8, 64, 6], torch.int32, resident="stacked")
    logical_routes = make_stats_placement_spec(
        "tid2eid",
        route_base,
        layer_ids=[layer_id],
        placement=CONTIGUOUS_PLACEMENT,
    ).create_tensor()[0]
    physical_routes = make_stats_placement_spec(
        "tid2eid",
        route_base,
        layer_ids=[layer_id],
        placement=placement,
    ).create_tensor()[0]

    routed_source = _logical_expert_tensor(32, 1)
    routed_spec = TensorSpec(
        "routed_w1",
        list(routed_source.shape),
        routed_source.dtype,
        init_value=lambda: routed_source,
    )
    routed_physical = make_stats_placement_spec(
        "routed_w1",
        routed_spec,
        layer_ids=[layer_id],
        placement=placement,
    ).create_tensor()
    routed_values = routed_physical.reshape(256, 1)[physical_routes.to(torch.int64), 0]
    assert torch.equal(routed_values, logical_routes.to(routed_values.dtype))

    gate_source = _replicated_expert_tensor(1)
    gate_spec = TensorSpec(
        "gate_w",
        list(gate_source.shape),
        gate_source.dtype,
        init_value=lambda: gate_source,
    )
    gate_physical = make_stats_placement_spec(
        "gate_w",
        gate_spec,
        layer_ids=[layer_id],
        placement=placement,
    ).create_tensor()[0, :, 0]
    gate_values = gate_physical[physical_routes.to(torch.int64)]
    assert torch.equal(gate_values, logical_routes.to(gate_values.dtype))


def test_forward_factory_covers_layers_zero_through_42() -> None:
    manifest = load_stats_placement_manifest()
    source = _replicated_expert_tensor(1)
    base_spec = TensorSpec("gate_w", list(source.shape), source.dtype, init_value=lambda: source)

    placed = stats_layer_spec_factory("gate_w", base_spec, 43).create_tensor()

    assert tuple(placed.shape) == (8, 43 * 256, 1)
    for layer_id in (0, 1, 21, 42):
        expected = torch.tensor(manifest["layers"][layer_id]["rank_to_logical"], dtype=torch.int64).reshape(
            -1
        )
        assert torch.equal(placed[0, layer_id * 256 : (layer_id + 1) * 256, 0], expected)
    assert stats_layer_spec_factory("norm_w", base_spec, 43) is None


def test_mtp_adapter_uses_layer_43_and_preserves_spec_order() -> None:
    manifest = load_stats_placement_manifest()
    routed = _logical_expert_tensor(32, 1)
    specs = [
        TensorSpec("before", [1], torch.float32),
        TensorSpec("routed_w1", list(routed.shape), routed.dtype, init_value=lambda: routed),
        TensorSpec("after", [1], torch.float32),
    ]

    adapted = adapt_mtp_stats_placement_specs(specs)
    placed = adapted[1].create_tensor()

    assert [spec.name for spec in adapted] == ["before", "routed_w1", "after"]
    expected = torch.tensor(manifest["layers"][43]["rank_to_logical"], dtype=torch.int64).reshape(-1)
    assert torch.equal(placed[:, :, 0].reshape(-1), expected)


def test_single_layer_adapter_can_select_the_standalone_moe_layer() -> None:
    manifest = load_stats_placement_manifest()
    routed = _logical_expert_tensor(32, 1)
    specs = [
        TensorSpec("before", [1], torch.float32),
        TensorSpec("routed_w1", list(routed.shape), routed.dtype, init_value=lambda: routed),
        TensorSpec("after", [1], torch.float32),
    ]

    adapted = adapt_single_layer_stats_placement_specs(specs, layer_id=0)
    placed = adapted[1].create_tensor()

    assert [spec.name for spec in adapted] == ["before", "routed_w1", "after"]
    expected = torch.tensor(manifest["layers"][0]["rank_to_logical"], dtype=torch.int64).reshape(-1)
    assert torch.equal(placed[:, :, 0].reshape(-1), expected)


def test_contiguous_mtp_control_only_replaces_the_route_fixture() -> None:
    specs = [
        TensorSpec("gate_w", [8, 256, 1], torch.float32),
        TensorSpec("tid2eid", [8, 64, 6], torch.int32, resident="stacked"),
        TensorSpec("routed_w1", [8, 32, 1], torch.int8),
    ]

    adapted = adapt_mtp_stats_placement_specs(specs, placement=CONTIGUOUS_PLACEMENT)

    assert adapted[0] is specs[0]
    assert adapted[1] is not specs[1]
    assert adapted[2] is specs[2]


def test_eplb_mtp_control_replaces_every_placement_owned_spec() -> None:
    gate = _replicated_expert_tensor(1).to(torch.float32)
    routed = _logical_expert_tensor(32, 1).to(torch.int8)
    specs = [
        TensorSpec("gate_w", [8, 256, 1], torch.float32, init_value=lambda: gate),
        TensorSpec("tid2eid", [8, 64, 6], torch.int32, resident="stacked"),
        TensorSpec("routed_w1", [8, 32, 1], torch.int8, init_value=lambda: routed),
        TensorSpec("unrelated", [1], torch.float32),
    ]

    adapted = adapt_mtp_stats_placement_specs(specs, placement=EPLB_PLACEMENT)

    assert all(adapted[index] is not specs[index] for index in range(3))
    assert adapted[3] is specs[3]
    expected = torch.tensor(
        eplb_replay_placement(load_stats_placement_manifest(), 43).physical_to_logical,
        dtype=torch.int64,
    )
    assert torch.equal(adapted[0].create_tensor()[0, :, 0], expected)


def test_only_the_nine_moe_specs_are_placement_owned() -> None:
    assert PLACED_NAMES == {
        "gate_w",
        "gate_bias",
        "tid2eid",
        "routed_w1",
        "routed_w1_scale",
        "routed_w3",
        "routed_w3_scale",
        "routed_w2",
        "routed_w2_scale",
    }


def test_legacy_and_stats_decode_entrypoints_keep_separate_topologies() -> None:
    _subprocess_check(
        """
import sys
sys.argv = ["eplb_decode_logits.py"]
import eplb_decode_logits as decode
assert (decode.N_RANKS, decode.N_LOCAL, decode.N_EXPERTS_GLOBAL) == (8, 16, 128)
"""
    )
    _subprocess_check(
        """
import sys
sys.argv = ["stats_placement_decode_logits.py"]
import stats_placement_decode_logits
import eplb_decode_logits as decode
assert (decode.N_RANKS, decode.N_LOCAL, decode.N_EXPERTS_GLOBAL) == (8, 32, 256)
"""
    )


def test_stats_mtp_entrypoint_uses_the_256_expert_topology() -> None:
    _subprocess_check(
        """
import sys
sys.argv = ["stats_placement_mtp_core.py"]
import torch
import stats_placement_mtp_core
import eplb_mtp_core as mtp
from golden import mapped_pool_ratio_reldiff, ratio_reldiff
from tools.dsv4_eplb_perf_metrics import STATS_NUMERIC_CASE_CONFIGS
assert (mtp.N_RANKS, mtp.N_LOCAL, mtp.N_EXPERTS_GLOBAL) == (8, 32, 256)
mtp_contract = STATS_NUMERIC_CASE_CONFIGS["mtp-core"]
comparators = mtp.build_numeric_compare_fn(mapped_kv_cache=True)
assert mtp_contract.exact_validation_comparators
assert {
    output_name: comparator.__name__
    for output_name, comparator in comparators.items()
} == dict(mtp_contract.required_validation_comparators)

expected = torch.ones(8, 2, 128, 1)
actual = expected.clone()
mapping = torch.arange(8, dtype=torch.int64).repeat(8, 1)
actual[0].reshape(256, 1)[mapping[0, 0]] = 1.2
kwargs = {
    "actual_outputs": {},
    "expected_outputs": {},
    "inputs": {"swa_slot_mapping": mapping},
    "rtol": 0.0,
    "atol": 0.0,
}
old_ok, _ = ratio_reldiff(diff_thd=0.01, pct_thd=0.05)(
    actual, expected, **kwargs
)
aggregate_ok, aggregate_detail = mapped_pool_ratio_reldiff(
    "swa_slot_mapping",
    mapping_shape=(8, 8),
    block_size=128,
    leading_rank_axis=True,
    pool_name="kv_cache",
    diff_thd=0.01,
    pct_thd=0.05,
    expected_mapped_rows=64,
)(actual, expected, **kwargs)
mapped_ok, mapped_detail = comparators["kv_cache"](
    actual, expected, **kwargs
)
assert old_ok
assert aggregate_ok, aggregate_detail
assert not mapped_ok
assert "mapped" in mapped_detail
"""
    )


def test_stats_mtp_entrypoint_rejects_finite_only_validation() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_MODEL_DIR), str(_REPO_ROOT), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [
            sys.executable,
            str(_MODEL_DIR / "stats_placement_mtp_core.py"),
            "--finite-only",
            "--compile-only",
        ],
        cwd=_MODEL_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )

    assert result.returncode == 2
    assert "--finite-only is not supported by the stats-placement entrypoint" in (
        result.stderr
    )


def test_legacy_and_stats_moe_entrypoints_keep_separate_topologies() -> None:
    _subprocess_check(
        """
import sys
sys.argv = ["moe.py", "--ep", "8", "--experts-per-rank", "16"]
import moe
assert (moe.N_RANKS, moe.N_LOCAL, moe.N_EXPERTS_GLOBAL) == (8, 16, 128)
"""
    )
    _subprocess_check(
        """
import sys
sys.argv = ["stats_placement_moe.py"]
import stats_placement_moe as entry
import moe
import torch
from golden import ratio_reldiff
from tools.dsv4_eplb_perf_metrics import STATS_NUMERIC_CASE_CONFIGS
assert (moe.N_RANKS, moe.N_LOCAL, moe.N_EXPERTS_GLOBAL, moe.RECV_MAX) == (8, 32, 256, 64)
moe_contract = STATS_NUMERIC_CASE_CONFIGS["moe-ep8"]
assert moe_contract.exact_validation_comparators
assert {
    output_name: comparator.__name__
    for output_name, comparator in entry.build_compare_fn().items()
} == dict(moe_contract.required_validation_comparators)

expected = torch.ones(8, 8, 100)
actual = expected.clone()
actual[0, 0] = 1.2
kwargs = {
    "actual_outputs": {},
    "expected_outputs": {},
    "inputs": {},
    "rtol": 0.0,
    "atol": 0.0,
}
global_ok, global_detail = ratio_reldiff(diff_thd=0.003, pct_thd=0.05)(
    actual, expected, **kwargs
)
rowwise_ok, rowwise_detail = entry.build_compare_fn()["x_next"](
    actual, expected, **kwargs
)
assert global_ok, global_detail
assert not rowwise_ok
assert "row [0, 0]" in rowwise_detail
"""
    )


def test_stats_moe_consumes_the_exact_layer_zero_histogram_for_every_placement() -> None:
    _subprocess_check(
        """
import sys
sys.argv = ["stats_placement_moe.py"]

import torch
from golden import ScalarSpec, TensorSpec

import stats_placement_moe as entry
from eplb_fixture import EXPERT_PLACEMENT_CHOICES
from stats_placement_fixture import (
    apportion_route_counts,
    load_stats_placement_manifest,
    physical_to_logical_for_placement,
)

def fake_build_tensor_specs(*, layer_id, num_tokens, balanced_routing):
    assert (layer_id, num_tokens, balanced_routing) == (0, 8, False)
    return [
        TensorSpec("tid2eid", [8, 64, 6], torch.int32, resident="stacked"),
        TensorSpec("input_ids", [8, 8], torch.int64),
        TensorSpec("unrelated", [1], torch.float32, init_value=lambda: torch.ones(1)),
        ScalarSpec("layer_id", torch.int32, layer_id),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
    ]

entry.moe.build_tensor_specs = fake_build_tensor_specs
manifest = load_stats_placement_manifest()
expected_counts = torch.tensor(
    apportion_route_counts(manifest["expert_loads"][0], total_routes=384)
)
expected_input_ids = torch.arange(64, dtype=torch.int64).reshape(8, 8)
reference_logical_routes = None

for placement in EXPERT_PLACEMENT_CHOICES:
    specs = {spec.name: spec for spec in entry.build_tensor_specs(placement=placement)}
    input_ids = specs["input_ids"].create_tensor()
    routes = specs["tid2eid"].create_tensor()
    assert torch.equal(input_ids, expected_input_ids)
    assert specs["tid2eid"].resident == "stacked"

    active_physical_routes = torch.stack(
        [routes[rank, input_ids[rank]] for rank in range(8)]
    )
    physical_to_logical = torch.tensor(
        physical_to_logical_for_placement(manifest, 0, placement),
        dtype=torch.int64,
    )
    logical_routes = physical_to_logical[active_physical_routes.to(torch.int64)]
    actual_counts = torch.bincount(logical_routes.reshape(-1), minlength=256)
    assert torch.equal(actual_counts, expected_counts)
    if reference_logical_routes is None:
        reference_logical_routes = logical_routes
    else:
        assert torch.equal(logical_routes, reference_logical_routes)
"""
    )


def test_legacy_eplb_suite_keeps_the_ep8x16_balanced_moe_anchor() -> None:
    suite = (_REPO_ROOT / "tools" / "run_dsv4_eplb_suite.sh").read_text(encoding="utf-8")

    assert '"$REPO_ROOT/models/deepseek_v4_flash_mtp/moe.py"' in suite
    assert "--experts-per-rank 16" in suite
    assert "--balanced-routing" in suite
    assert "stats_placement_moe.py" not in suite
