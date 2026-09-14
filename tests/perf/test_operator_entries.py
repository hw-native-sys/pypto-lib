# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU checks for entry specialization and projection-only golden capture."""

import ast
import dataclasses
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from golden import TensorSpec
from tools.perf.deterministic_run import make_capture
from tools.perf.operator_contract import ContractError, ROOT


def function_from_source(path, name, namespace):
    tree = ast.parse(path.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    fn.decorator_list = []
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("ep", [2, 4, 8])
@pytest.mark.parametrize("local", [None, 16, 32])
def test_moe_specializes_before_subkernel_imports(ep, local, monkeypatch):
    path = ROOT / "models/deepseek_v4_flash_mtp/moe.py"
    tree = ast.parse(path.read_text())
    prefix = []
    for node in tree.body:
        if isinstance(node, ast.Import) and node.names[0].name.startswith("pypto"):
            break
        prefix.append(node)
    @dataclasses.dataclass(frozen=True)
    class Config:
        n_routed_experts: int = 256
    import sys
    cfg = SimpleNamespace(FLASH=Config(), MOE_TOKENS=8)
    monkeypatch.setitem(sys.modules, "config", cfg)
    args = [str(path), "--ep", str(ep)]
    if local is not None:
        args += [f"--experts-per-rank={local}"]
    monkeypatch.setattr(sys, "argv", args)
    namespace = {}
    exec(compile(ast.Module(body=prefix, type_ignores=[]), str(path), "exec"), namespace)
    assert cfg.EP_WORLD_SIZE == ep
    assert cfg.FLASH.n_routed_experts == ep * (32 if local is None else local)
    assert cfg.RECV_MAX == ep * 8


def test_projection_host_uses_two_tp4_groups():
    calls = Mock()
    namespace = {
        "WORLD_SIZE": 8, "TEST_TOKENS": 2, "D": 4, "VOCAB_PER_TP": 3,
        "MAX_LOGIT_ROWS": 2, "VOCAB": 12, "GROUP_LOGIT_ROWS": 8, "TP_SIZE": 4, "DONE_VALUE": 1,
        "pl": SimpleNamespace(range=range, BF16="bf16", FP32="fp32", INT32="int32"),
        "pld": SimpleNamespace(alloc_window_buffer=lambda *args: object(), world_size=lambda: 8,
                               window=lambda *args, **kwargs: object()),
        "lm_head_test": calls,
    }
    fn = function_from_source(ROOT / "models/deepseek_v4_flash_mtp/lm_head.py",
                              "l3_lm_head_projection", namespace)
    fn(torch.zeros(8, 2, 4), torch.zeros(8, 3, 4), torch.zeros(8, 2, 12), torch.zeros(8, 2))
    assert calls.call_count == 8
    for rank, call in enumerate(calls.call_args_list):
        assert call.kwargs == {"device": rank}
        assert call.args[-3:] == (rank // 4 * 4, rank % 4, 1)


def test_projection_golden_does_not_require_sampling_inputs():
    sample = Mock(side_effect=AssertionError("sampling must not run"))
    namespace = {"TP_SIZE": 2, "MAX_LOGIT_ROWS": 2, "D": 3, "golden_sample": sample}
    fn = function_from_source(ROOT / "models/deepseek_v4_flash_mtp/lm_head.py", "golden_lm_head", namespace)
    hidden = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    weight = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    values = {"hidden_states": hidden, "lm_head_weight": weight,
              "logit_row_indices": torch.tensor([[1, -1], [0, 1]]), "logits": torch.zeros(2, 2, 4)}
    fn(values)
    full_weight = weight.reshape(4, 3)
    torch.testing.assert_close(values["logits"][0, 0], hidden[0, 1] @ full_weight.T)
    assert torch.equal(values["logits"][0, 1], torch.zeros(4))
    sample.assert_not_called()


def test_capture_preserves_numerics_and_hashes_actual_inputs(tmp_path):
    case = {"case_id": "small", "required_specs": {"out": {"shape": [2], "dtype": "torch.float32"}}}
    cfg = {"device_id": 4, "platform": "a2a3"}
    compare = {"out": object()}
    def original(**kwargs):
        assert kwargs["rtol"] == 0.001
        assert kwargs["atol"] == 0.002
        assert kwargs["compare_fn"] is compare
        assert kwargs["config"] == cfg
        assert kwargs["save_data"] is False
        specs = kwargs["specs"]
        specs[0].direction = "in"
        specs[1].direction = "out"
        values = {"inp": specs[0].create_tensor(), "out": torch.zeros(2)}
        kwargs["golden_fn"](values)
        torch.testing.assert_close(values["out"], values["inp"] * 2)
        # Failed correctness retains hashes but cannot gain a valid metric.
        return SimpleNamespace(passed=False, error="deliberate", work_dir=None)

    hashes = []
    for value in (1, 1, 2):
        specs = [TensorSpec("inp", [2], torch.float32, init_value=value),
                 TensorSpec("out", [2], torch.float32)]
        output = tmp_path / f"capture-{len(hashes)}.json"
        observer = make_capture(original, case, output, [4])
        kwargs = dict(specs=specs, config=cfg, rtol=0.001, atol=0.002, compare_fn=compare,
                      golden_fn=lambda values: values["out"].copy_(values["inp"] * 2))
        observer(**kwargs)
        payload = json.loads(output.read_text())
        assert payload["passed"] is False and "benchmark" not in payload
        hashes.append(payload["fixture_sha256"])
        with pytest.raises(ContractError):
            observer(**kwargs)
        assert not output.exists()
    assert hashes[0] == hashes[1] != hashes[2]
