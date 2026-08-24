# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the DeepSeek-V4 synthetic token-loop artifact ABI."""

import sys
import types
from unittest.mock import patch

import pytest
import torch

from models.deepseek_v4_pro.synthetic_token_loop import (
    _check_sample,
    _create_persistent_worker,
    _prefill_prompt_row,
    _require_scalar,
    _require_runtime_scalar,
)


class _Compiled:
    def __init__(self, *infos, output_dir=None):
        self._infos = infos
        self.output_dir = output_dir

    def _get_metadata(self):
        return self._infos, None, None


def _info(name, *, shape=None, dtype="i32"):
    return types.SimpleNamespace(name=name, shape=shape, dtype=dtype)


def _dtype_module():
    module = types.ModuleType("pypto.ir.compiled_program")
    module._to_torch_dtype = lambda dtype: {"i32": torch.int32, "i64": torch.int64}[dtype]
    return module


def test_require_scalar_accepts_terminal_ssa_name():
    compiled = _Compiled(_info("moe_epoch_base__ssa_v0"))

    with patch.dict(sys.modules, {"pypto.ir.compiled_program": _dtype_module()}):
        _require_scalar(compiled, "moe_epoch_base", "decode", torch.int32)


def test_require_scalar_rejects_old_artifact():
    compiled = _Compiled(_info("num_tokens"))

    with pytest.raises(ValueError, match="recompile"):
        _require_scalar(compiled, "moe_epoch_base", "decode", torch.int32)


@pytest.mark.parametrize(
    ("info", "error"),
    [
        (_info("moe_epoch_base", shape=[1]), ValueError),
        (_info("moe_epoch_base", dtype="i64"), TypeError),
    ],
)
def test_require_scalar_rejects_wrong_abi(info, error):
    with (
        patch.dict(sys.modules, {"pypto.ir.compiled_program": _dtype_module()}),
        pytest.raises(error),
    ):
        _require_scalar(_Compiled(info), "moe_epoch_base", "decode", torch.int32)


def _write_host_orch(tmp_path, scalar_argument):
    orchestration = tmp_path / "orchestration"
    orchestration.mkdir()
    host_orch = orchestration / "host_orch.py"
    host_orch.write_text(
        "def run(tensors, task_args):\n"
        '    epoch = tensors["moe_epoch_base__ssa_v0"]\n'
        f"    task_args.add_scalar({scalar_argument})\n"
    )


def test_require_runtime_scalar_accepts_forwarded_host_argument(tmp_path):
    _write_host_orch(tmp_path, "epoch")
    compiled = _Compiled(
        _info("moe_epoch_base__ssa_v0"),
        output_dir=tmp_path,
    )

    with patch.dict(sys.modules, {"pypto.ir.compiled_program": _dtype_module()}):
        _require_runtime_scalar(
            compiled, "moe_epoch_base", "decode", torch.int32
        )


def test_require_runtime_scalar_rejects_constant_specialization(tmp_path):
    _write_host_orch(tmp_path, "0")
    compiled = _Compiled(
        _info("moe_epoch_base__ssa_v0"),
        output_dir=tmp_path,
    )

    with (
        patch.dict(sys.modules, {"pypto.ir.compiled_program": _dtype_module()}),
        pytest.raises(ValueError, match="constant-specialized.*compile_runtime=True"),
    ):
        _require_runtime_scalar(
            compiled, "moe_epoch_base", "decode", torch.int32
        )


def test_create_worker_retains_persistent_windows():
    captured = {}

    def worker_type(compiled, **kwargs):
        captured["compiled"] = compiled
        captured["kwargs"] = kwargs
        return "worker"

    programs = [object(), object()]
    config = object()
    inherited = [torch.zeros(1)]
    worker = _create_persistent_worker(worker_type, programs, config, inherited)

    assert worker == "worker"
    assert captured == {
        "compiled": programs,
        "kwargs": {
            "config": config,
            "persistent": True,
            "reset_persistent_windows": False,
            "inherited_host_tensors": inherited,
        },
    }


def test_synthetic_prompt_uses_requested_active_length():
    ids, prompt_len = _prefill_prompt_row(7, None, 5)

    assert prompt_len == 5
    assert ids[:5].tolist() == [0, 1, 2, 3, 4]
    assert torch.equal(ids[5:], torch.zeros_like(ids[5:]))


def test_check_sample_dumps_rank_divergence(tmp_path, monkeypatch, capsys):
    tensors = {
        "input_ids": torch.tensor([[1], [1]], dtype=torch.int64),
        "logit_row_indices": torch.zeros((2, 1), dtype=torch.int32),
        "pre_hc_hidden_out": torch.tensor([[[[1.0, 2.0]]], [[[1.0, 3.0]]]]),
        "hidden_out": torch.tensor([[[1.0, 2.0]], [[1.0, 3.0]]]),
        "logits": torch.tensor([[[0.0, 3.0, 1.0, 2.0]], [[0.0, 1.0, 4.0, 2.0]]]),
        "sampled_ids": torch.tensor([[[1]], [[2]]], dtype=torch.int32),
    }
    monkeypatch.setenv("DSV4_FAIL_DUMP_DIR", str(tmp_path))

    with pytest.raises(AssertionError, match="ranks sampled different tokens"):
        _check_sample(tensors, vocab_size=4, stage="decode[0]@6")

    output = capsys.readouterr().out
    assert "active_rows=[0, 0]" in output
    assert "pre_hc_hidden_out" in output
    assert "max_abs_diff_rank0=[0.0, 1.0]" in output
    assert "logits_top2: ids=[[1, 3], [2, 3]]" in output
    dump = torch.load(tmp_path / "failure_decode_0_6.pt", weights_only=True)
    assert set(dump) == set(tensors)
    assert torch.equal(dump["sampled_ids"], tensors["sampled_ids"])
