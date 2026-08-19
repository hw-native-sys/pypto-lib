# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side A5 numerical contracts for DeepSeek-V4 HC-pre reductions."""

import os
import sys
from pathlib import Path

import torch
import pytest

_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "deepseek_v4_pro"
os.environ.setdefault("DEEPSEEK_V4_VARIANT", "flash")
sys.path.insert(0, str(_MODEL_DIR))

import hc_pre as hc_pre_module  # noqa: E402
from hc_pre import (  # noqa: E402
    _golden_a5_high_precision_rsqrt,
    _golden_a5_pre_raw,
    _golden_a5_post_raw,
    _golden_a5_rms_inv,
    _golden_a5_trowsum_fp32,
    _golden_hc_pre_pre_raw,
    _golden_hc_pre_post_raw,
)


def test_a5_trowsum_uses_64_lane_tree_then_ascending_group_accumulation():
    values = torch.zeros(2, 256, dtype=torch.float32)

    # Row 0 distinguishes the recursive vcadd tree from a scalar left fold.
    values[0, :64] = 2.0**-24
    values[0, 0] = 1.0

    # Row 1 distinguishes ascending accumulation of the four 64-lane results
    # from a second pairwise tree across those results.
    values[1, 0] = 1.0
    values[1, 64] = 2.0**-24
    values[1, 128] = 2.0**-24
    values[1, 192] = 2.0**-24

    actual = _golden_a5_trowsum_fp32(values)
    expected_bits = torch.tensor([[0x3F80001F], [0x3F800000]], dtype=torch.int32)
    assert torch.equal(actual.view(torch.int32), expected_bits)


def test_a5_high_precision_rsqrt_avoids_fp32_sqrt_reciprocal_double_rounding():
    # This exact hc_pre RMS argument was observed at Flash prefill token 4.
    # The A5 high-precision path returns the correctly rounded FP32 reciprocal
    # square root: the golden computes it through float64 and rounds once, so
    # the bit pattern is host-independent. The host FP32 `torch.rsqrt` is NOT
    # asserted here — its rounding depends on the torch build and CPU path
    # (this argument was observed one ULP low, 0x416997DB, on the A5 host).
    rms_arg = torch.tensor([0x3B99BBDE], dtype=torch.int32).view(torch.float32)

    actual = _golden_a5_high_precision_rsqrt(rms_arg)
    assert actual.view(torch.int32).item() == 0x416997DC


@pytest.mark.parametrize("hc_dim", [4 * 4096, 4 * 7168])
def test_a5_rms_golden_accepts_flash_and_pro_hc_widths(monkeypatch, hc_dim):
    monkeypatch.setattr(hc_pre_module, "HC_DIM", hc_dim)
    monkeypatch.setattr(hc_pre_module, "HC_DIM_INV", 1.0 / hc_dim)

    actual = _golden_a5_rms_inv(torch.zeros(1, hc_dim, dtype=torch.float32))
    assert torch.equal(actual, torch.tensor([[1000.0]], dtype=torch.float32))


def test_a5_post_projection_uses_trowsum_and_k256_chunk_order(monkeypatch):
    hc_dim = 4 * 256
    monkeypatch.setattr(hc_pre_module, "HC_DIM", hc_dim)

    x_flat = torch.ones(1, hc_dim, dtype=torch.float32)
    hc_fn = torch.zeros(2 * hc_pre_module.HC_MULT, hc_dim, dtype=torch.float32)
    half_ulp = 2.0**-24

    post_fn = hc_fn[hc_pre_module.HC_MULT:]
    post_fn[0, :64] = half_ulp
    post_fn[0, 0] = 1.0
    post_fn[1, 0] = 1.0
    post_fn[1, 256] = half_ulp
    post_fn[1, 512] = half_ulp
    post_fn[1, 768] = half_ulp

    actual = _golden_a5_post_raw(x_flat, hc_fn)
    expected_bits = torch.tensor(
        [[0x3F80001F, 0x3F800000, 0x00000000, 0x00000000]], dtype=torch.int32
    )
    assert torch.equal(actual.view(torch.int32), expected_bits)

    products = x_flat[:, None, :] * post_fn[None, :, :]
    left_fold = torch.zeros_like(actual)
    for k0 in range(hc_dim):
        left_fold += products[..., k0]
    assert left_fold[0, 0].view(torch.int32).item() == 0x3F800000

    chunks = [
        _golden_a5_trowsum_fp32(products[..., k0:k0 + 256])[..., 0]
        for k0 in range(0, hc_dim, 256)
    ]
    pairwise_chunks = (chunks[0] + chunks[1]) + (chunks[2] + chunks[3])
    assert pairwise_chunks[0, 1].view(torch.int32).item() == 0x3F800001


def test_a5_pre_projection_uses_trowsum_and_k256_chunk_order(monkeypatch):
    hc_dim = 4 * 256
    monkeypatch.setattr(hc_pre_module, "HC_DIM", hc_dim)

    x_flat = torch.ones(1, hc_dim, dtype=torch.float32)
    hc_fn = torch.zeros(2 * hc_pre_module.HC_MULT, hc_dim, dtype=torch.float32)
    half_ulp = 2.0**-24

    pre_fn = hc_fn[:hc_pre_module.HC_MULT]
    pre_fn[0, :64] = half_ulp
    pre_fn[0, 0] = 1.0
    pre_fn[1, 0] = 1.0
    pre_fn[1, 256] = half_ulp
    pre_fn[1, 512] = half_ulp
    pre_fn[1, 768] = half_ulp

    actual = _golden_a5_pre_raw(x_flat, hc_fn)
    expected_bits = torch.tensor(
        [[0x3F80001F, 0x3F800000, 0x00000000, 0x00000000]], dtype=torch.int32
    )
    assert torch.equal(actual.view(torch.int32), expected_bits)

    products = x_flat[:, None, :] * pre_fn[None, :, :]
    chunks = [
        _golden_a5_trowsum_fp32(products[..., k0:k0 + 256])[..., 0]
        for k0 in range(0, hc_dim, 256)
    ]
    pairwise_chunks = (chunks[0] + chunks[1]) + (chunks[2] + chunks[3])
    assert pairwise_chunks[0, 1].view(torch.int32).item() == 0x3F800001


def test_pre_golden_selects_vector_order_only_for_separate(monkeypatch):
    x_flat = torch.empty(1, 0, dtype=torch.float32)
    hc_fn = torch.empty(0, 0, dtype=torch.float32)
    cube_raw = torch.arange(hc_pre_module.MIX_HC, dtype=torch.float32).reshape(1, -1)
    vector_raw = torch.full((1, hc_pre_module.HC_MULT), 11.0, dtype=torch.float32)
    monkeypatch.setattr(hc_pre_module, "_golden_a5_pre_raw", lambda *_args: vector_raw)

    monkeypatch.setattr(hc_pre_module, "HC_PRE_IMPL", "separate")
    assert torch.equal(_golden_hc_pre_pre_raw(x_flat, hc_fn, cube_raw), vector_raw)

    monkeypatch.setattr(hc_pre_module, "HC_PRE_IMPL", "syncall")
    assert torch.equal(
        _golden_hc_pre_pre_raw(x_flat, hc_fn, cube_raw),
        cube_raw[:, :hc_pre_module.HC_MULT],
    )


def test_post_golden_selects_vector_order_only_for_separate(monkeypatch):
    x_flat = torch.empty(1, 0, dtype=torch.float32)
    hc_fn = torch.empty(0, 0, dtype=torch.float32)
    cube_raw = torch.arange(hc_pre_module.MIX_HC, dtype=torch.float32).reshape(1, -1)
    vector_raw = torch.full((1, hc_pre_module.HC_MULT), 7.0, dtype=torch.float32)
    monkeypatch.setattr(hc_pre_module, "_golden_a5_post_raw", lambda *_args: vector_raw)

    monkeypatch.setattr(hc_pre_module, "HC_PRE_IMPL", "separate")
    assert torch.equal(_golden_hc_pre_post_raw(x_flat, hc_fn, cube_raw), vector_raw)

    monkeypatch.setattr(hc_pre_module, "HC_PRE_IMPL", "syncall")
    expected_cube = cube_raw[:, hc_pre_module.HC_MULT:hc_pre_module.HC_MULT * 2]
    assert torch.equal(_golden_hc_pre_post_raw(x_flat, hc_fn, cube_raw), expected_cube)
