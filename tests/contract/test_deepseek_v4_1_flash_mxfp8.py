# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU regressions for grouped output projection and mHC rounding boundaries."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


MODEL = Path(__file__).resolve().parents[2] / "models" / "deepseek_v4_1_flash"


def _load(monkeypatch, name, filename):
    spec = importlib.util.spec_from_file_location(name, MODEL / filename)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def v41(monkeypatch):
    monkeypatch.syspath_prepend(str(MODEL))
    config = _load(monkeypatch, "_v41_mx_config", "config.py")
    monkeypatch.setitem(sys.modules, "config", config)
    quant = _load(monkeypatch, "_v41_mx_quant", "quant.py")
    monkeypatch.setitem(sys.modules, "quant", quant)
    for name in ("rope_tables", "utils", "rmsnorm", "hc_mixes"):
        dependency = _load(monkeypatch, f"_v41_mx_{name}", f"{name}.py")
        monkeypatch.setitem(sys.modules, name, dependency)
    hc_pre = _load(monkeypatch, "_v41_mx_hc_pre", "hc_pre.py")
    return quant, hc_pre


def test_grouped_w8a8_matches_independent_dequantized_gemm(v41):
    quant, _ = v41
    generator = torch.Generator().manual_seed(183)
    tokens, groups, width, outputs = 7, 3, 256, 32
    x = torch.randn(tokens, groups, width, generator=generator).bfloat16()
    x[0].zero_()
    weight = torch.randn(groups, width, outputs, generator=generator).to(torch.float8_e4m3fn)
    codes = torch.randint(115, 124, (groups, width // 32, outputs), generator=generator, dtype=torch.uint8)
    scales = quant.pack_mx_b_scale(codes)
    actual = quant.grouped_mxfp8_linear(x, weight, scales, output_dtype=torch.float32)

    values = x.float().unflatten(-1, (-1, 32))
    a_scale = torch.exp2((torch.floor(torch.log2(values.abs().amax(-1))) - 8).clamp_min(-127))
    payload = (values / a_scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    a = (payload * a_scale.unsqueeze(-1)).flatten(-2)
    b = weight.float() * torch.exp2(codes.float() - 127).repeat_interleave(32, dim=1)
    expected = torch.stack([a[:, g].double() @ b[g].double() for g in range(groups)], dim=1).float()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert actual[0].count_nonzero() == 0
    without_activation_quant = torch.stack([x[:, g].float() @ b[g] for g in range(groups)], dim=1)
    assert (actual - without_activation_quant).norm() / expected.norm() > 0.01


def test_grouped_fixture_transposes_weights_and_scales_together(v41):
    quant, _ = v41
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("grouped fixture storage requires torch E8M0 support")
    generator = torch.Generator().manual_seed(82)
    raw = torch.randn(3, 32, 256, generator=generator)
    raw *= torch.exp2(torch.arange(8).float() - 4).repeat_interleave(32)
    payload, packed = quant.quantize_grouped_mxfp8_weight(raw)
    assert payload.shape == (3, 256, 32)
    assert packed.shape == (3, 8, 32)
    decoded = payload.float() * quant.decode_e8m0(quant.unpack_mx_b_scale(packed)).repeat_interleave(32, -2)
    source = raw.unflatten(-1, (-1, 32))
    scales = torch.exp2(torch.ceil(torch.log2(source.abs().amax(-1).clamp_min(1e-4) / 448.0)))
    expected = ((source / scales.unsqueeze(-1)).to(torch.float8_e4m3fn).float() * scales.unsqueeze(-1))
    torch.testing.assert_close(decoded, expected.flatten(-2).transpose(-1, -2), rtol=0, atol=0)


def test_mhc_pre_preserves_device_pairwise_cancellation(v41):
    _, hc_pre = v41
    x = torch.tensor([1e8, 1.0, -1e8, 1.0]).reshape(1, 4, 1).expand(1, 4, 32)
    result = hc_pre.golden_mhc_pre(x, torch.ones(1, 4))
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result, torch.zeros(1, 32, dtype=torch.bfloat16), rtol=0, atol=0)


def test_ocp_default_saturates_and_preserves_small_activation_scales(v41):
    quant, _ = v41
    x = torch.full((4, 64), 1.875, dtype=torch.bfloat16)
    x[0, 0] = 1.0625
    x[1].zero_()
    x[2].fill_(2.0**-30)
    x[3].fill_(2.0**-120)
    payload, scale = quant._quantize_mxfp8_activation(x, scale_alg=0)
    values = payload.float().flatten(-2)
    assert scale[0, 0] == 2.0**-8
    assert values[0, 0] == 256.0
    assert values[0, 1] == 448.0
    assert values[1].count_nonzero() == 0
    assert scale[2, 0] == 2.0**-38
    assert values[2, 0] == 256.0
    assert scale[3, 0] == 2.0**-127
    assert values[3].count_nonzero() == 0
