# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side numerical contracts for the DeepSeek-V4 RMSNorm golden."""

import os
import sys
from pathlib import Path

import torch

_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "deepseek_v4_pro"
os.environ.setdefault("DEEPSEEK_V4_VARIANT", "flash")
sys.path.insert(0, str(_MODEL_DIR))

from rmsnorm import D, D_TILE, EPS, golden_rms_norm  # noqa: E402


def test_golden_rms_norm_uses_device_chunk_reduction_order():
    generator = torch.Generator().manual_seed(26)
    x = (torch.randn(2, D, generator=generator) * 0.13).to(torch.bfloat16)
    norm_w = (torch.randn(D, generator=generator) * 0.1 + 1.0).to(torch.bfloat16)
    x_fp32 = x.float()
    norm_w_fp32 = norm_w.float()

    sq_sum = torch.zeros(2, 1, dtype=torch.float32)
    for d0 in range(0, D, D_TILE):
        chunk = x_fp32[:, d0:d0 + D_TILE]
        sq_sum += (chunk * chunk).sum(dim=-1, keepdim=True)
    device_order = (
        x_fp32
        * torch.rsqrt(sq_sum * (1.0 / D) + EPS)
        * norm_w_fp32
    ).to(torch.bfloat16)

    monolithic = (
        x_fp32
        * torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + EPS)
        * norm_w_fp32
    ).to(torch.bfloat16)

    actual = golden_rms_norm(x, norm_w)
    assert torch.equal(actual, device_order)
    assert not torch.equal(monolithic, device_order)
