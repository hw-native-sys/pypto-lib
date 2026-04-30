# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import pytest
from importlib.machinery import ModuleSpec

from llm.testing.hf_compare.cases import qwen3_14b_e2e
from llm.testing.hf_compare.target import _resolve_device_id


def test_qwen3_e2e_case_runs_with_passthrough_weights(monkeypatch):
    fake_state = {
        "model.embed_tokens.weight": torch.randn(32, 16),
        "model.norm.weight": torch.randn(16),
        "lm_head.weight": torch.randn(32, 16),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(16, 16),
    }
    expected_generated = torch.tensor([[11, 12]], dtype=torch.int64)
    expected_logits = torch.tensor([[0.5, -1.0, 3.0]], dtype=torch.float32)
    calls: dict[str, object] = {}

    def fake_ref(inputs, hf_state, **kwargs):
        calls["ref_inputs"] = inputs["input_ids"].clone()
        calls["ref_state_keys"] = sorted(hf_state)
        calls["ref_kwargs"] = kwargs
        return {
            "generated_ids": expected_generated.clone(),
            "last_logits": expected_logits.clone(),
        }

    def fake_target(inputs, weights, **kwargs):
        calls["target_inputs"] = inputs["input_ids"].clone()
        calls["target_state_keys"] = sorted(weights)
        calls["target_kwargs"] = kwargs
        return {
            "generated_ids": expected_generated.clone(),
            "last_logits": expected_logits.clone(),
        }

    monkeypatch.setattr(qwen3_14b_e2e, "_hf_e2e_forward", fake_ref)
    monkeypatch.setattr(qwen3_14b_e2e, "_run_e2e_target", fake_target)

    case = qwen3_14b_e2e.build(
        hf_model_path="/tmp/fake-model",
        prompt_ids="7,8,9",
        max_new_tokens=2,
        num_layers=1,
        max_seq=8,
        cpu_only=True,
    )
    case.hf_weights = fake_state

    report = case.run()

    assert report.passed
    assert [result.selector for result in report.results] == ["generated_ids", "last_logits"]
    assert torch.equal(calls["ref_inputs"], torch.tensor([[7, 8, 9]], dtype=torch.int64))
    assert torch.equal(calls["target_inputs"], torch.tensor([[7, 8, 9]], dtype=torch.int64))
    assert calls["ref_state_keys"] == sorted(fake_state)
    assert calls["target_state_keys"] == sorted(fake_state)
    assert calls["ref_kwargs"] == {
        "model_path": "/tmp/fake-model",
        "num_layers": 1,
        "max_new_tokens": 2,
        "hf_dtype": torch.float32,
    }
    assert calls["target_kwargs"] == {
        "max_new_tokens": 2,
        "num_layers": 1,
        "max_seq": 8,
        "cpu_only": True,
        "platform": "a2a3",
        "device_id": None,
    }


def test_qwen3_e2e_build_validates_max_seq():
    with pytest.raises(ValueError, match="exceeds max_seq=4"):
        qwen3_14b_e2e.build(
            prompt_ids="1,2,3,4",
            max_new_tokens=2,
            max_seq=4,
        )


def test_state_for_num_layers_keeps_requested_prefixes():
    hf_state = {
        "model.embed_tokens.weight": torch.ones(1),
        "model.norm.weight": torch.ones(1),
        "lm_head.weight": torch.ones(1),
        "model.layers.0.a": torch.ones(1),
        "model.layers.1.b": torch.ones(1),
        "model.layers.2.c": torch.ones(1),
        "something.else": torch.ones(1),
    }

    kept = qwen3_14b_e2e._state_for_num_layers(hf_state, num_layers=2)

    assert sorted(kept) == [
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.layers.0.a",
        "model.layers.1.b",
        "model.norm.weight",
    ]


def test_resolve_device_id_prefers_explicit_then_env(monkeypatch):
    monkeypatch.delenv("ASCEND_DEVICE_ID", raising=False)
    monkeypatch.delenv("DEVICE_ID", raising=False)
    assert _resolve_device_id(None) == 0

    monkeypatch.setenv("DEVICE_ID", "6")
    assert _resolve_device_id(None) == 6

    monkeypatch.setenv("ASCEND_DEVICE_ID", "4")
    assert _resolve_device_id(None) == 4
    assert _resolve_device_id("9") == 9


def test_qwen3_e2e_uses_resolved_device_id_for_npu(monkeypatch):
    class _FakeRunner:
        def __init__(self, dec, prefill, *, batch, max_seq, platform, device_id):
            seen["runner_init"] = (batch, max_seq, platform, device_id)

        def run_prefill(self, *args, **kwargs):
            hidden = args[0]
            return hidden

        def run_decode(self, hidden_states, *args, **kwargs):
            return hidden_states

    class _FakeModule:
        HIDDEN = 4
        HEAD_DIM = 2
        NUM_KV_HEADS = 1

    class _FakeDec:
        pass

    seen: dict[str, object] = {}
    monkeypatch.setenv("ASCEND_DEVICE_ID", "7")
    monkeypatch.setattr(qwen3_14b_e2e, "_load_qwen3_modules", lambda: (_FakeDec, _FakeModule))
    monkeypatch.setattr(qwen3_14b_e2e, "_extract_layer_weights", lambda weights, **kwargs: {"wq": torch.zeros(4, 4, dtype=torch.bfloat16)})
    monkeypatch.setattr(qwen3_14b_e2e, "_build_rope", lambda max_seq, head_dim: (torch.zeros(max_seq, head_dim), torch.zeros(max_seq, head_dim)))
    monkeypatch.setattr(qwen3_14b_e2e, "_embed_tokens", lambda input_ids, embed_weight: torch.zeros(input_ids.shape[0], input_ids.shape[1], 4, dtype=torch.bfloat16))
    monkeypatch.setattr(qwen3_14b_e2e, "_final_rmsnorm", lambda hidden, norm_weight: hidden.float())
    monkeypatch.setattr(qwen3_14b_e2e, "_lm_head_forward", lambda hidden, lm_weight: torch.tensor([[1.0, 0.0]], dtype=torch.float32))
    monkeypatch.setattr(qwen3_14b_e2e, "_greedy_sample", lambda logits: torch.tensor([0], dtype=torch.int64))
    monkeypatch.setattr(qwen3_14b_e2e, "_NPURunner", _FakeRunner)

    out = qwen3_14b_e2e._run_e2e_target(
        {"input_ids": torch.tensor([[1, 2]], dtype=torch.int64)},
        {
            "model.embed_tokens.weight": torch.zeros(8, 4),
            "model.norm.weight": torch.ones(4),
            "lm_head.weight": torch.zeros(2, 4),
        },
        max_new_tokens=1,
        num_layers=1,
        max_seq=8,
        cpu_only=False,
        platform="a2a3",
        device_id=None,
    )

    assert seen["runner_init"] == (1, 8, "a2a3", 7)
    assert out["generated_ids"].tolist() == [[0]]


def test_load_qwen3_modules_loads_decode_and_prefill(monkeypatch):
    seen: dict[str, object] = {}

    class _FakeLoader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            seen.setdefault("modules", []).append(module)

    def fake_spec_from_file_location(name, path):
        seen.setdefault("names", []).append(name)
        seen.setdefault("paths", []).append(str(path))
        return ModuleSpec(name=name, loader=_FakeLoader(), origin=str(path))

    monkeypatch.setattr(qwen3_14b_e2e.importlib.util, "spec_from_file_location", fake_spec_from_file_location)

    dec, prefill = qwen3_14b_e2e._load_qwen3_modules()

    assert seen["names"] == ["_qwen3_14b_decode_kernel", "_qwen3_14b_prefill_kernel"]
    assert seen["paths"][0].endswith("examples/models/qwen3/14b/qwen3_14b_decode.py")
    assert seen["paths"][1].endswith("examples/models/qwen3/14b/qwen3_14b_prefill.py")
    assert dec is seen["modules"][0]
    assert prefill is seen["modules"][1]
