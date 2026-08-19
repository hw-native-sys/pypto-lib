# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side numerical contracts for the DeepSeek-V4 sparse-attention golden."""

import os
import sys
from pathlib import Path

import torch

_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "deepseek_v4_pro"
os.environ.setdefault("DEEPSEEK_V4_VARIANT", "flash")
sys.path.insert(0, str(_MODEL_DIR))

from prefill_sparse_attn import (  # noqa: E402
    A_CUBE_ACC_K,
    _golden_a5_cube_bf16_matmul,
    _golden_a5_cube_proj_a,
)


def _manual_cube_matmul(lhs, rhs):
    expected = torch.zeros(lhs.shape[0], rhs.shape[0], dtype=torch.float32)
    for k0 in range(0, lhs.shape[1], A_CUBE_ACC_K):
        group_dot = (
            lhs[:, k0:k0 + A_CUBE_ACC_K].double().unsqueeze(1)
            * rhs[:, k0:k0 + A_CUBE_ACC_K].double().unsqueeze(0)
        ).sum(dim=-1)
        expected = (expected.double() + group_dot).float()
    return expected


def test_golden_qk_uses_a5_cube_k16_accumulation_order():
    generator = torch.Generator().manual_seed(19)
    lhs = (torch.randn(4, 64, generator=generator) * 0.2).to(torch.bfloat16).float()
    rhs = (torch.randn(5, 64, generator=generator) * 0.15).to(torch.bfloat16).float()

    device_order = _manual_cube_matmul(lhs, rhs)
    monolithic = lhs @ rhs.T
    actual = _golden_a5_cube_bf16_matmul(lhs, rhs)

    assert torch.equal(actual, device_order)
    assert not torch.equal(monolithic, device_order)


def test_golden_proj_a_uses_a5_cube_k16_accumulation_order():
    generator = torch.Generator().manual_seed(41)
    t_dim, model_groups, group_in, o_lora = 3, 2, 64, 7
    o_model = (
        torch.randn(t_dim, model_groups, group_in, generator=generator) * 0.2
    ).to(torch.bfloat16).float()
    wo_a = (
        torch.randn(model_groups, o_lora, group_in, generator=generator) * 0.15
    ).to(torch.bfloat16).float()

    expected = torch.zeros(t_dim, model_groups, o_lora, dtype=torch.float32)
    for model_group in range(model_groups):
        expected[:, model_group] = _manual_cube_matmul(
            o_model[:, model_group], wo_a[model_group]
        )

    monolithic = torch.einsum("tgk,gnk->tgn", o_model, wo_a)
    actual = _golden_a5_cube_proj_a(o_model, wo_a)

    assert torch.equal(actual, expected)
    assert not torch.equal(monolithic, expected)
