# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "deepseek_v4_flash_mtp"


def _run_eplb_check(check_contract: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(_MODEL_DIR), str(_REPO_ROOT), env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-c", check_contract],
        cwd=_MODEL_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_eplb_decode_logits_fixed_topology_and_host_contract() -> None:
    _run_eplb_check(
        """
import inspect
import sys

sys.argv = ["eplb_decode_logits.py"]
import eplb_decode_logits as decode

actual = {
    "ep": decode.N_RANKS,
    "tp": decode.LM_HEAD_TP_SIZE,
    "experts": decode.N_EXPERTS_GLOBAL,
    "experts_per_rank": decode.N_LOCAL,
    "tokens": decode.T,
    "start_pos": decode.DECODE_START_POS,
}
expected = {
    "ep": 8,
    "tp": 4,
    "experts": 128,
    "experts_per_rank": 16,
    "tokens": 8,
    "start_pos": 8192,
}
assert actual == expected

host_prepared = {
    "x_hc",
    "ori_slot_mapping",
    "swa_slot_mapping",
    "swa_indices",
    "swa_lens",
    "hca_cmp_slot_mapping",
    "hca_state_slot_mapping",
    "csa_cmp_slot_mapping",
    "csa_idx_slot_mapping",
    "csa_state_slot_mapping",
    "csa_inner_state_slot_mapping",
    "position_ids",
    "kv_seq_lens",
}
forbidden = {"embed_weight", "block_counts", "sampled_ids"}
for function_name in ("eplb_decode_logits_inline", "eplb_decode_logits", "l3_eplb_decode_logits"):
    function = getattr(decode, function_name)
    parameters = inspect.signature(function._func).parameters
    names = set(parameters)
    assert tuple(parameters)[0] == "x_hc"
    assert host_prepared <= names
    assert not forbidden & names
    assert parameters["x_hc"].annotation.dtype == decode.pl.FP32
    assert parameters["logits"].annotation.dtype == decode.pl.FP32
"""
    )


def test_eplb_decode_logits_reuses_hash_layer_zero_routing_for_all_layers() -> None:
    _run_eplb_check(
        """
import sys

sys.argv = ["eplb_fixture.py"]
import torch
from golden import TensorSpec
from eplb_fixture import make_eplb_input_ids_spec, make_eplb_tid2eid_spec

active_vocab = 64
tid2eid_base = TensorSpec("tid2eid", [8, active_vocab, 6], torch.int32)
input_ids_base = TensorSpec("input_ids", [8, 8], torch.int64)
tid2eid = make_eplb_tid2eid_spec(tid2eid_base, layer_count=43).create_tensor()
input_ids = make_eplb_input_ids_spec(input_ids_base).create_tensor()

assert tuple(tid2eid.shape) == (8, 43 * active_vocab, 6)
assert torch.equal(input_ids, torch.arange(64, dtype=torch.int64).reshape(8, 8))

route_indices = input_ids.unsqueeze(-1).expand(-1, -1, 6)
expected_routes = torch.arange(384, dtype=torch.int32).remainder(128)
expected_counts = torch.full((128,), 3, dtype=torch.int64)
for layer in range(43):
    layer_table = tid2eid[:, layer * active_vocab : (layer + 1) * active_vocab, :]
    active_routes = torch.gather(layer_table, 1, route_indices).reshape(-1)
    assert torch.equal(active_routes, expected_routes)
    assert torch.equal(torch.bincount(active_routes.to(torch.int64), minlength=128), expected_counts)
"""
    )


def test_eplb_decode_logits_orchestration_drift_contract() -> None:
    _run_eplb_check(
        """
import ast
import inspect
import sys
import textwrap

sys.argv = ["eplb_decode_logits.py"]
import eplb_decode_logits as decode

source = textwrap.dedent(inspect.getsource(decode.eplb_decode_logits_inline._func))
tree = ast.parse(source)

def call_name(call):
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None

tracked = {
    "attention_swa",
    "attention_csa",
    "attention_hca",
    "moe",
    "clear_moe_signals",
    "hc_head",
    "rms_norm",
    "lm_head_core",
    "clear_lm_head_signals",
}
calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
ordered = sorted(
    (node for node in calls if call_name(node) in tracked),
    key=lambda node: (node.lineno, node.col_offset),
)
assert [call_name(node) for node in ordered] == [
    "attention_swa", "moe",
    "attention_swa", "moe",
    "attention_csa", "moe",
    "attention_hca", "moe",
    "attention_csa", "moe",
    "clear_moe_signals", "hc_head", "rms_norm", "lm_head_core", "clear_lm_head_signals",
]

moe_calls = [node for node in ordered if call_name(node) == "moe"]
slice_suffixes = ("_l0", "_l1", "_csa", "_hca", "_last")
for call, suffix in zip(moe_calls, slice_suffixes, strict=True):
    assert ast.unparse(call.args[-4]) == "pl.cast(0, pl.INT32)"
    assert f"gate_w{suffix}" in ast.unparse(call)

assert decode.FWD_NUM_LAYERS == 43
assert decode.CSA_NUM_LAYERS == 21
assert decode.HCA_NUM_LAYERS == 20
assert "for loop_i in pl.range(HCA_NUM_LAYERS)" in source
assert "(CSA_NUM_LAYERS - 1) *" in source
assert "lm_head_with_sampling" not in source
assert "greedy_sample" not in source
"""
    )


def test_forward_compressed_mappings_use_each_cache_storage_block_size() -> None:
    _run_eplb_check(
        """
import sys
from types import SimpleNamespace

sys.argv = ["eplb_decode_logits.py"]
import torch
import eplb_decode_logits as decode
from utils import compressed_slot_mapping, mask_uncommitted_compressed_boundaries

base_specs = {
    "hca_compress_state": SimpleNamespace(
        shape=[decode.N_RANKS, decode.HCA_COMPRESS_STATE_BLOCK_NUM, 1],
    ),
    "csa_compress_state": SimpleNamespace(
        shape=[decode.N_RANKS, decode.CSA_MAIN_STATE_BLOCK_NUM, 1],
    ),
    "csa_inner_compress_state": SimpleNamespace(
        shape=[decode.N_RANKS, decode.CSA_INNER_STATE_BLOCK_NUM, 1],
    ),
}
metadata = decode.make_forward_metadata_tensors(base_specs, start_pos=8191)
positions = metadata["position_ids"][0].reshape(decode.B, -1)
inactive_metadata = decode.make_forward_metadata_tensors(
    base_specs,
    start_pos=8192,
)

cases = (
    (
        "hca_cmp_slot_mapping",
        "cmp_block_table",
        decode.HCA_COMPRESS_RATIO,
        decode.HCA_CMP_STORAGE_BLOCK_SIZE,
        decode.CSA_CMP_BLOCK_NUM,
    ),
    (
        "csa_cmp_slot_mapping",
        "cmp_block_table",
        decode.CSA_COMPRESS_RATIO,
        decode.CSA_CMP_STORAGE_BLOCK_SIZE,
        decode.CSA_CMP_BLOCK_NUM,
    ),
    (
        "csa_idx_slot_mapping",
        "idx_block_table",
        decode.CSA_COMPRESS_RATIO,
        decode.CSA_CMP_STORAGE_BLOCK_SIZE,
        decode.CSA_IDX_CACHE_BLOCK_NUM,
    ),
)
for mapping_name, table_name, ratio, storage_block_size, physical_blocks in cases:
    expected = compressed_slot_mapping(
        positions,
        metadata[table_name][0],
        compress_ratio=ratio,
        block_size=storage_block_size,
    )
    expected = mask_uncommitted_compressed_boundaries(
        expected,
        positions,
        compress_ratio=ratio,
        commit_tokens=1,
    ).reshape(-1)
    actual = metadata[mapping_name]
    assert torch.equal(actual, expected.unsqueeze(0).expand_as(actual))
    assert int((actual[0] >= 0).sum()) == decode.B
    assert int((inactive_metadata[mapping_name] >= 0).sum()) == 0
    assert int(actual.max()) < physical_blocks * storage_block_size
"""
    )


def test_compressed_cache_active_write_profile_wires_position_and_golden_counts() -> None:
    _run_eplb_check(
        """
import sys
from types import SimpleNamespace

sys.argv = [
    "eplb_decode_logits.py",
    "--stats-placement",
    "--compressed-cache-active-write",
    "--save-device-output", "/tmp/dsv4-device-output-contract-reference",
    "--device-output-seed", "1807",
    "--ep", "8",
    "--tp", "4",
    "--experts-per-rank", "32",
]
import eplb_decode_logits as decode

captured = {}

def fake_build_tensor_specs(**kwargs):
    captured["start_pos"] = kwargs["start_pos"]
    return []

def fake_run(**kwargs):
    captured["compare_fn"] = kwargs["compare_fn"]
    captured["device_output_capture"] = kwargs["device_output_capture"]
    return SimpleNamespace(passed=True, error=None)

decode.build_tensor_specs = fake_build_tensor_specs
decode.run = fake_run
decode.main()

assert decode.EPLB_START_POS == 8192
assert decode.EPLB_COMPRESSED_WRITE_START_POS == 8191
assert captured["start_pos"] == 8191
comparators = captured["compare_fn"]
assert "expected_active_rows=640" in comparators["hca_cmp_kv"].__name__
assert "expected_active_rows=672" in comparators["csa_cmp_kv"].__name__
assert "expected_active_rows=672" in comparators["idx_kv_cache"].__name__
assert "expected_active_rows=672" in comparators["idx_kv_scale"].__name__
capture = captured["device_output_capture"]
assert capture.metadata.case == "decode-logits"
assert capture.metadata.seed == 1807
assert capture.metadata.placement == "stats"
assert dict(capture.metadata.topology)["start_pos"] == 8191
assert [item.logical_key for item in capture.captures] == [
    "decode.hidden_out",
    "decode.kv_cache",
    "decode.logits",
    "decode.pre_hc_hidden_out",
]
"""
    )


def test_full_decode_golden_schedule_and_last_csa_window_contract() -> None:
    _run_eplb_check(
        """
import sys
from collections import defaultdict

sys.argv = ["stats_placement_decode_logits.py", "--stats-placement", "--ep", "8", "--tp", "4", "--experts-per-rank", "32"]
import torch
import eplb_decode_logits_golden as golden
from tools.dsv4_eplb_perf_metrics import STATS_NUMERIC_CASE_CONFIGS

assert golden.FWD_NUM_LAYERS == 43
assert golden.CSA_NUM_LAYERS == 21
assert golden.HCA_NUM_LAYERS == 20
assert golden.LAYER_KINDS[:2] == ("swa", "swa")
assert golden.LAYER_KINDS[2:] == tuple(
    "csa" if layer % 2 == 0 else "hca"
    for layer in range(2, 43)
)
assert golden.KIND_ORDER[42] == 20
comparators = golden.build_eplb_decode_logits_compare_fn()
assert len(comparators) == 11
decode_contract = STATS_NUMERIC_CASE_CONFIGS["decode-logits"]
assert decode_contract.exact_validation_comparators
assert {
    output_name: comparator.__name__
    for output_name, comparator in comparators.items()
} == dict(decode_contract.required_validation_comparators)
assert comparators["kv_cache"].__name__.startswith(
    "stacked_mapped_pool_cosine_rel_l2"
)
assert "strict_layers=(0,)" in comparators["kv_cache"].__name__
assert comparators["idx_kv_cache"].__name__.startswith(
    "stacked_mapped_pool_ratio_allclose"
)
assert comparators["pre_hc_hidden_out"].__name__.startswith(
    "rank_token_cosine_rel_l2"
)

active_comparators = golden.build_eplb_decode_logits_compare_fn(
    compressed_cache_active_write=True,
)
assert "expected_active_rows=640" in active_comparators["hca_cmp_kv"].__name__
for output_name in ("csa_cmp_kv", "idx_kv_cache", "idx_kv_scale"):
    assert "expected_active_rows=672" in active_comparators[output_name].__name__
assert active_comparators["idx_kv_cache"].__name__.startswith(
    "stacked_mapped_pool_cosine_rel_l2"
)
assert "max_bad_row_ratio=0.035" in active_comparators["idx_kv_cache"].__name__
assert "hard_min_cosine=0.96" in active_comparators["idx_kv_cache"].__name__
assert "hard_max_rel_l2=0.35" in active_comparators["idx_kv_cache"].__name__
assert "max_rel_l2=0.11" in active_comparators["pre_hc_hidden_out"].__name__
assert "max_bad_row_ratio=0.03" in active_comparators["hca_cmp_kv"].__name__
assert "hard_min_cosine=0.97" in active_comparators["hca_cmp_kv"].__name__
assert "max_bad_row_ratio=0.025" in active_comparators["csa_cmp_kv"].__name__
assert "hard_min_cosine=0.96" in active_comparators["csa_cmp_kv"].__name__
assert "hard_max_rel_l2=0.6" in active_comparators["hca_compress_state"].__name__

expected_hidden = torch.ones(8, 8, 4)
actual_hidden = expected_hidden.clone()
ok, detail = comparators["hidden_out"](
    actual_hidden,
    expected_hidden,
)
assert ok, detail
actual_hidden[0, 0].neg_()
ok, detail = comparators["hidden_out"](
    actual_hidden,
    expected_hidden,
)
assert not ok
assert "rank=0 token=0" in detail

zero = torch.zeros(2, dtype=torch.float32)
finite, detail, cosine, rel_l2 = golden._cosine_rel_l2_metrics(zero, zero)
assert finite, detail
assert cosine == 1.0
assert rel_l2 == 0.0

huge = torch.full((2,), 3e38, dtype=torch.float32)
finite, detail, cosine, rel_l2 = golden._cosine_rel_l2_metrics(huge, -huge)
assert finite, detail
assert abs(cosine + 1.0) < 1e-12
assert abs(rel_l2 - 2.0) < 1e-12

logits_compare = golden._logits_cosine_compare()
compare_kwargs = {
    "actual_outputs": {},
    "expected_outputs": {},
    "inputs": {"logit_row_indices": torch.tensor([[0]], dtype=torch.int32)},
    "rtol": 0.0,
    "atol": 0.0,
}
ok, detail = logits_compare(
    zero.reshape(1, 1, 2),
    zero.reshape(1, 1, 2),
    **compare_kwargs,
)
assert ok, detail
ok, detail = logits_compare(
    huge.reshape(1, 1, 2),
    -huge.reshape(1, 1, 2),
    **compare_kwargs,
)
assert not ok

inactive_expected = torch.zeros(1, 2, 2)
inactive_actual = inactive_expected.clone()
inactive_actual[0, 1] = torch.tensor([float("nan"), 7.0])
ok, detail = logits_compare(
    inactive_actual,
    inactive_expected,
    **{
        **compare_kwargs,
        "inputs": {
            "logit_row_indices": torch.tensor([[0, -1]], dtype=torch.int32)
        },
    },
)
assert not ok
assert "inactive logits[0,1] changed" in detail

coverage_logits_compare = golden._logits_cosine_compare(expected_active_rows=1)
ok, detail = coverage_logits_compare(
    torch.zeros(1, 1, 2),
    torch.zeros(1, 1, 2),
    **{
        **compare_kwargs,
        "inputs": {
            "logit_row_indices": torch.tensor([[-1]], dtype=torch.int32)
        },
    },
)
assert not ok
assert "active-row coverage 0 does not match expected 1" in detail

ok, detail = logits_compare(
    torch.zeros(1, 1, 2),
    torch.zeros(1, 1, 2),
    **{
        **compare_kwargs,
        "inputs": {
            "logit_row_indices": torch.tensor([[-2]], dtype=torch.int32)
        },
    },
)
assert not ok
assert "only -1 is a negative sentinel" in detail

strict_pool_compare = golden._stacked_mapped_pool_compare(
    ("mapping",),
    layer_labels=(0,),
    block_size=8,
    pool_name="strict_pool",
    atol=1e-3,
    rtol=3e-2,
    max_error_ratio=0.01,
    min_cosine=0.99,
    max_rel_l2=0.10,
    strict_layer_labels=(0,),
)
expected_pool = torch.ones(8, 1, 8, 512)
actual_pool = expected_pool.clone()
actual_pool[:, 0, 0, :20].neg_()
ok, detail = strict_pool_compare(
    actual_pool,
    expected_pool,
    actual_outputs={},
    expected_outputs={},
    inputs={"mapping": torch.arange(8).repeat(8, 1)},
    rtol=0.0,
    atol=0.0,
)
assert not ok
assert "cosine=" in detail

semantic_pool_compare = golden._stacked_mapped_pool_compare(
    ("mapping",),
    layer_labels=(0,),
    block_size=8,
    pool_name="semantic_pool",
    atol=0.0,
    rtol=0.0,
    max_error_ratio=0.0,
    min_cosine=0.99,
    max_rel_l2=0.10,
    max_bad_row_ratio=0.02,
    hard_min_cosine=0.98,
    hard_max_rel_l2=0.50,
)
expected_pool = torch.ones(8, 1, 8, 4)
semantic_kwargs = {
    "actual_outputs": {},
    "expected_outputs": {},
    "inputs": {"mapping": torch.arange(8).repeat(8, 1)},
    "rtol": 0.0,
    "atol": 0.0,
}

within_budget = expected_pool.clone()
within_budget[0, 0, 0].mul_(1.2)
ok, detail = semantic_pool_compare(
    within_budget,
    expected_pool,
    **semantic_kwargs,
)
assert ok, detail

over_budget = within_budget.clone()
over_budget[0, 0, 1].mul_(1.2)
ok, detail = semantic_pool_compare(
    over_budget,
    expected_pool,
    **semantic_kwargs,
)
assert not ok
assert "bad-row ratio" in detail

outside_hard_envelope = expected_pool.clone()
outside_hard_envelope[0, 0, 0].mul_(1.6)
ok, detail = semantic_pool_compare(
    outside_hard_envelope,
    expected_pool,
    **semantic_kwargs,
)
assert not ok
assert "hard max" in detail

for mapping, expected_rows, observed_rows in (
    (torch.full((8, 8), -1, dtype=torch.int64), 1, 0),
    (
        torch.cat([
            torch.tensor([0], dtype=torch.int64),
            torch.full((63,), -1, dtype=torch.int64),
        ]).reshape(8, 8),
        0,
        1,
    ),
):
    coverage_pool_compare = golden._stacked_mapped_pool_compare(
        ("mapping",),
        layer_labels=(0,),
        block_size=8,
        pool_name="coverage_pool",
        atol=1e-3,
        rtol=3e-2,
        max_error_ratio=0.01,
        min_cosine=0.99,
        max_rel_l2=0.10,
        expected_active_rows=expected_rows,
    )
    ok, detail = coverage_pool_compare(
        torch.zeros(8, 1, 8, 2),
        torch.zeros(8, 1, 8, 2),
        actual_outputs={},
        expected_outputs={},
        inputs={"mapping": mapping},
        rtol=0.0,
        atol=0.0,
    )
    assert not ok
    assert (
        f"active-row coverage {observed_rows} does not match expected {expected_rows}"
        in detail
    )

invalid_mapping = torch.full((8, 8), -1, dtype=torch.int64)
invalid_mapping[0, 0] = -2
ok, detail = semantic_pool_compare(
    torch.zeros(8, 1, 8, 4),
    torch.zeros(8, 1, 8, 4),
    **{
        **semantic_kwargs,
        "inputs": {"mapping": invalid_mapping},
    },
)
assert not ok
assert "only -1 is a negative sentinel" in detail

golden._base_attention_views = lambda *_args: {}
golden._layer_rows = lambda tensor, _count, _index: tensor
tensors = defaultdict(lambda: torch.zeros(8, 1))
tensors["window_swa_indices"] = torch.full((8, 1), 20)
tensors["window_swa_lens"] = torch.full((8, 1), 21)
tensors["swa_indices"] = torch.full((8, 1), 42)
tensors["swa_lens"] = torch.full((8, 1), 43)

middle = golden._csa_attention_views(tensors, 0, 2, torch.empty(0), torch.empty(0))
last = golden._csa_attention_views(tensors, 0, 42, torch.empty(0), torch.empty(0))
assert middle["window_swa_indices"].item() == 20
assert middle["window_swa_lens"].item() == 21
assert last["window_swa_indices"].item() == 42
assert last["window_swa_lens"].item() == 43
"""
    )
