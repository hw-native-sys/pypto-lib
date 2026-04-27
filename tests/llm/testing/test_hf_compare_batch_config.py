# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from importlib.machinery import ModuleSpec

import pytest

from llm.testing.hf_compare.cases import qwen3_14b_decode, qwen3_14b_prefill


def _stub_module(monkeypatch, case_module, *, max_seq: int, block_size: int):
    class _FakeLoader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            module.MAX_SEQ = max_seq
            module.BLOCK_SIZE = block_size

    monkeypatch.setattr(
        case_module.importlib.util,
        "spec_from_file_location",
        lambda name, path: ModuleSpec(name=name, loader=_FakeLoader(), origin=str(path)),
    )


def test_decode_case_accepts_batch_override(monkeypatch):
    _stub_module(monkeypatch, qwen3_14b_decode, max_seq=4096, block_size=64)
    case = qwen3_14b_decode.build(cpu_only=True, batch=16, seq_len=128, max_seq=256)

    tensors = case.input_spec.tensors
    assert tensors["hidden_states"].shape == (16, qwen3_14b_decode.HIDDEN)
    assert tensors["seq_lens"].shape == (16,)
    assert tensors["slot_mapping"].shape == (16,)


def test_prefill_case_accepts_batch_override(monkeypatch):
    _stub_module(monkeypatch, qwen3_14b_prefill, max_seq=128, block_size=64)
    case = qwen3_14b_prefill.build(cpu_only=True, batch=16, seq_len=128, max_seq=256)

    tensors = case.input_spec.tensors
    assert tensors["hidden_states"].shape == (16, 256, qwen3_14b_prefill.HIDDEN)
    assert tensors["seq_lens"].shape == (16,)
    assert tensors["slot_mapping"].shape == (16 * 256,)


@pytest.mark.parametrize("builder", [qwen3_14b_decode.build, qwen3_14b_prefill.build])
def test_batch_must_be_positive(builder, monkeypatch):
    case_module = qwen3_14b_decode if builder is qwen3_14b_decode.build else qwen3_14b_prefill
    _stub_module(monkeypatch, case_module, max_seq=4096, block_size=64)
    with pytest.raises(ValueError, match="batch must be positive"):
        builder(cpu_only=True, batch=0)
