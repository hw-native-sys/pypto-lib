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
