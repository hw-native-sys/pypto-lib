# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side A5 numerical contracts for DeepSeek-V4 Q projection golden."""

import os
import sys
from pathlib import Path

import pytest
import torch

_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "deepseek_v4_pro"
os.environ.setdefault("DEEPSEEK_V4_VARIANT", "flash")
sys.path.insert(0, str(_MODEL_DIR))

import qkv_proj_rope as qkv  # noqa: E402


def _manual_a5_trowsum(values):
    groups = values.reshape(*values.shape[:-1], -1, 64)
    while groups.shape[-1] > 1:
        groups = groups.reshape(*groups.shape[:-1], -1, 2)
        groups = groups[..., 0] + groups[..., 1]
    total = torch.zeros_like(groups[..., 0, :1])
    for group in range(groups.shape[-2]):
        total += groups[..., group, 0:1]
    return total


def _manual_a5_cube(lhs, rhs):
    lhs = lhs.to(torch.bfloat16).float()
    rhs = rhs.to(torch.bfloat16).float()
    expected = torch.zeros(lhs.shape[0], rhs.shape[1], dtype=torch.float32)
    for k0 in range(0, lhs.shape[1], 16):
        group_dot = (
            lhs[:, k0:k0 + 16].double().unsqueeze(-1)
            * rhs[k0:k0 + 16].double().unsqueeze(0)
        ).sum(dim=1)
        expected = (expected.double() + group_dot).float()
    return expected


def _small_qkv_tensors(*, rows, num_tokens=None):
    generator = torch.Generator().manual_seed(20260818)
    tensors = {
        "x": (torch.randn(rows, 64, generator=generator) * 0.2).to(torch.bfloat16),
        "wq_a": (torch.randn(64, 64, generator=generator) * 0.15).to(torch.bfloat16),
        "wq_b": torch.randint(-32, 33, (64, 128), generator=generator).to(torch.int8),
        "wq_b_scale": torch.rand(128, generator=generator) * 0.01 + 0.001,
        "wkv": (torch.randn(64, 64, generator=generator) * 0.15).to(torch.bfloat16),
        "rope_cos": (torch.randn(rows, 32, generator=generator) * 0.2).to(torch.bfloat16),
        "rope_sin": (torch.randn(rows, 32, generator=generator) * 0.2).to(torch.bfloat16),
        "gamma_cq": (torch.rand(64, generator=generator) + 0.5).to(torch.bfloat16),
        "gamma_ckv": (torch.rand(64, generator=generator) + 0.5).to(torch.bfloat16),
        "q": torch.full((rows, 2, 64), 7.0, dtype=torch.bfloat16),
        "kv": torch.full((rows, 64), 7.0, dtype=torch.bfloat16),
        "qr": torch.full((rows, 64), 7, dtype=torch.int8),
        "qr_scale": torch.full((rows, 1), 7.0, dtype=torch.float32),
    }
    if num_tokens is not None:
        tensors["num_tokens"] = torch.tensor(num_tokens, dtype=torch.int32)
    return tensors


def _patch_small_shapes(monkeypatch):
    values = {
        "D": 64,
        "H": 2,
        "HEAD_DIM": 64,
        "NOPE_DIM": 32,
        "ROPE_DIM": 32,
        "ROPE_HALF": 16,
        "Q_LORA": 64,
        "Q_LORA_TILE": 64,
        "QR_OK": 2,
        "QR_K_SLICE": 32,
        "QR_K_TILE": 16,
        "KV_OK": 2,
        "KV_K_SLICE": 32,
        "KV_K_TILE": 16,
    }
    for name, value in values.items():
        monkeypatch.setattr(qkv, name, value)


def test_a5_cube_uses_continuous_k16_mads_then_ascending_split_reduce():
    generator = torch.Generator().manual_seed(73)
    lhs = (torch.randn(3, 128, generator=generator) * 0.2).to(torch.bfloat16)
    rhs = (torch.randn(128, 11, generator=generator) * 0.15).to(torch.bfloat16)

    expected = _manual_a5_cube(lhs[:, :64], rhs[:64])
    expected += _manual_a5_cube(lhs[:, 64:], rhs[64:])
    actual = qkv._golden_a5_split_k_bf16_matmul(
        lhs, rhs, splits=2, k_per_split=64
    )
    continuous_without_split = _manual_a5_cube(lhs, rhs)

    assert torch.equal(actual, expected)
    assert not torch.equal(continuous_without_split, expected)


def test_qr_rms_quant_uses_chunked_trowsum_hp_rsqrt_and_raw_amax(monkeypatch):
    monkeypatch.setattr(qkv, "Q_LORA_TILE", 256)
    generator = torch.Generator().manual_seed(91)
    qr_fp32 = torch.randn(3, 512, generator=generator) * 0.1
    gamma = (torch.rand(512, generator=generator) + 0.5).to(torch.bfloat16)

    sq_sum = torch.zeros(3, 1, dtype=torch.float32)
    raw_amax = torch.zeros(3, 1, dtype=torch.float32)
    for k0 in range(0, 512, 256):
        chunk = qr_fp32[:, k0:k0 + 256]
        sq_sum += _manual_a5_trowsum(chunk * chunk)
        raw_amax = torch.maximum(
            raw_amax,
            (chunk * gamma[k0:k0 + 256].float()).abs().amax(-1, keepdim=True),
        )
    inv = torch.rsqrt((sq_sum * (1.0 / 512) + qkv.EPS).double()).float()
    tile_amax = (inv * raw_amax).clamp_min(qkv.INT8_AMAX_EPS)
    scale_quant = torch.full_like(tile_amax, qkv.INT8_SCALE_MAX) / tile_amax
    expected_scale = torch.ones_like(scale_quant) / scale_quant
    expected_qr = torch.round(
        ((qr_fp32 * inv) * gamma.float()) * scale_quant
    ).to(torch.int32).to(torch.float16).to(torch.int8)

    actual_qr, actual_scale = qkv._golden_qr_rms_norm_quant(qr_fp32, gamma)
    legacy_inv = torch.rsqrt(qr_fp32.square().mean(-1, keepdim=True) + qkv.EPS)
    legacy_normed = (qr_fp32 * legacy_inv) * gamma.float()
    legacy_amax = legacy_normed.abs().amax(-1, keepdim=True).clamp_min(
        qkv.INT8_AMAX_EPS
    )
    legacy_scale_quant = torch.full_like(
        legacy_amax, qkv.INT8_SCALE_MAX
    ) / legacy_amax
    legacy_scale = torch.ones_like(legacy_scale_quant) / legacy_scale_quant

    assert torch.equal(actual_qr, expected_qr)
    assert torch.equal(actual_scale, expected_scale)
    assert not torch.equal(actual_scale, legacy_scale)


def test_q_head_rms_uses_a5_trowsum_and_high_precision_rsqrt(monkeypatch):
    monkeypatch.setattr(qkv, "HEAD_DIM", 512)
    generator = torch.Generator().manual_seed(123)
    q_full = torch.randn(2, 3, 512, generator=generator) * 17.0
    sq_sum = _manual_a5_trowsum(q_full * q_full)
    inv = torch.rsqrt((sq_sum * (1.0 / 512) + qkv.EPS).double()).float()
    expected = q_full * inv

    actual = qkv._golden_a5_q_head_rms_norm(q_full)
    legacy = q_full * torch.rsqrt(q_full.square().mean(-1, keepdim=True) + qkv.EPS)

    assert torch.equal(actual, expected)
    assert not torch.equal(actual, legacy)


def test_num_tokens_computes_only_active_rows_and_zeros_tail(monkeypatch):
    _patch_small_shapes(monkeypatch)
    active = _small_qkv_tensors(rows=4, num_tokens=2)
    standalone = _small_qkv_tensors(rows=4)

    qkv.golden_qkv_proj_rope(active)
    qkv.golden_qkv_proj_rope(standalone)

    for name in ("q", "kv", "qr", "qr_scale"):
        assert torch.equal(active[name][:2], standalone[name][:2])
        assert torch.count_nonzero(active[name][2:]) == 0
        assert torch.count_nonzero(standalone[name][2:]) > 0


@pytest.mark.parametrize(
    ("hidden_size", "q_lora"),
    [(4096, 1024), (7168, 1536)],
)
def test_flash_and_pro_q_shapes_satisfy_a5_reduction_contracts(hidden_size, q_lora):
    assert (hidden_size // 2) % qkv._A5_CUBE_ACC_K == 0
    assert q_lora % 256 == 0
    assert q_lora % qkv._A5_FP32_VECTOR_LANES == 0
    assert 512 % qkv._A5_FP32_VECTOR_LANES == 0
