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
_MODEL_DIR = _REPO_ROOT / "models" / "deepseek" / "v4-flash"


def _run_decode_check(check_contract: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(_MODEL_DIR),
            str(_REPO_ROOT),
            env.get("PYTHONPATH", ""),
        ]
    )
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


def test_decode_logits_workload_constants_match_reference() -> None:
    _run_decode_check(
        """
import sys

sys.argv = ["decode_fwd_logits.py", "--ep", "8", "--tp", "4"]
import decode_fwd_logits

actual = {
    "N_RANKS": decode_fwd_logits.N_RANKS,
    "LM_HEAD_TP_SIZE": decode_fwd_logits.LM_HEAD_TP_SIZE,
    "N_EXPERTS_GLOBAL": decode_fwd_logits.N_EXPERTS_GLOBAL,
    "N_LOCAL": decode_fwd_logits.N_LOCAL,
    "B": decode_fwd_logits.B,
    "T": decode_fwd_logits.T,
    "DECODE_START_POS": decode_fwd_logits.DECODE_START_POS,
}
expected = {
    "N_RANKS": 8,
    "LM_HEAD_TP_SIZE": 4,
    "N_EXPERTS_GLOBAL": 128,
    "N_LOCAL": 16,
    "B": 4,
    "T": 8,
    "DECODE_START_POS": 8192,
}
assert actual == expected
"""
    )


def test_decode_signatures_use_host_prepared_inputs() -> None:
    _run_decode_check(
        """
import inspect
import sys

sys.argv = ["decode_fwd_logits.py", "--ep", "8", "--tp", "4"]
import decode_fwd_logits

host_prepared_inputs = {
    "x_hc",
    "ori_slot_mapping",
    "window_swa_indices",
    "window_swa_lens",
    "swa_slot_mapping",
    "swa_indices",
    "swa_lens",
    "hca_cmp_slot_mapping",
    "hca_state_slot_mapping",
    "csa_cmp_slot_mapping",
    "csa_idx_slot_mapping",
    "csa_state_slot_mapping",
    "csa_inner_state_slot_mapping",
}
forbidden_parameters = {"embed_weight", "block_counts"}

for function_name in ("decode_fwd_logits", "l3_decode_fwd_logits"):
    function = getattr(decode_fwd_logits, function_name)
    parameters = inspect.signature(function._func).parameters
    parameter_names = set(parameters)
    missing = host_prepared_inputs - parameter_names
    forbidden_present = forbidden_parameters & parameter_names
    assert tuple(parameters)[0] == "x_hc"
    assert not missing and not forbidden_present, (
        f"{function_name}: missing={sorted(missing)}, "
        f"forbidden_present={sorted(forbidden_present)}"
    )
    assert parameters["x_hc"].annotation.dtype == decode_fwd_logits.pl.FP32
"""
    )


def test_decode_fixture_ends_at_logits_before_sampling() -> None:
    _run_decode_check(
        """
import inspect
import sys

sys.argv = ["decode_fwd_logits.py", "--ep", "8", "--tp", "4"]
import decode_fwd_logits

for function_name in ("decode_fwd_logits", "l3_decode_fwd_logits"):
    function = getattr(decode_fwd_logits, function_name)
    parameters = inspect.signature(function._func).parameters
    assert "logits" in parameters, f"{function_name} must expose logits"
    assert parameters["logits"].annotation.dtype == decode_fwd_logits.pl.FP32, (
        f"{function_name} logits must remain FP32"
    )
    assert "sampled_ids" not in parameters, (
        f"{function_name} must end before greedy sampling"
    )
"""
    )


def test_decode_balanced_routing_fixture_hits_each_expert_three_times_per_layer() -> None:
    _run_decode_check(
        """
import sys

sys.argv = ["decode_fwd_logits.py", "--ep", "8", "--tp", "4"]

import torch
from golden import TensorSpec
import decode_fwd_logits

active_vocab = 64
decode_fwd_logits.VOCAB = active_vocab
base_specs = {
    "tid2eid": TensorSpec(
        "tid2eid",
        [8, active_vocab, 6],
        torch.int32,
        init_value=lambda: None,
    ),
    "input_ids": TensorSpec(
        "input_ids",
        [8, 8],
        torch.int64,
        init_value=lambda: None,
    ),
}

tid2eid = decode_fwd_logits._make_layer_stacked_spec(
    "tid2eid",
    base_specs,
).init_value()
input_ids = decode_fwd_logits._make_balanced_input_ids_spec(
    base_specs["input_ids"],
    8,
).init_value()

assert tuple(tid2eid.shape) == (8, 43 * active_vocab, 6)
assert torch.equal(
    input_ids,
    torch.arange(64, dtype=torch.int64).reshape(8, 8),
)

expected_routes = torch.arange(384, dtype=torch.int32).remainder(128)
expected_counts = torch.full((128,), 3, dtype=torch.int64)
route_indices = input_ids.unsqueeze(-1).expand(-1, -1, 6)

for layer in range(43):
    layer_table = tid2eid[
        :, layer * active_vocab : (layer + 1) * active_vocab, :
    ]
    active_routes = torch.gather(
        layer_table,
        1,
        route_indices,
    ).reshape(-1)
    assert torch.equal(active_routes, expected_routes), (
        f"layer {layer} does not reuse hash-layer-zero round-robin routing"
    )
    assert torch.equal(
        torch.bincount(active_routes.to(torch.int64), minlength=128),
        expected_counts,
    ), f"layer {layer} is not balanced across 128 experts"
"""
    )
