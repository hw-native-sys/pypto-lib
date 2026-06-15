# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for DeepSeek-V4 RoPE/YaRN table generation."""
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


V4_DIR = Path(__file__).resolve().parents[2] / "models" / "deepseek" / "v4"
sys.path.insert(0, str(V4_DIR))

from rope_tables import (  # noqa: E402
    build_deepseek_v4_rope_tables,
    materialize_half_rope_tables,
    materialize_token_rope_tables,
    precompute_freqs_cos_sin,
    rope_profile_for_compress_ratio,
)


@dataclass(frozen=True)
class RopeConfig:
    name: str = "test"
    max_position_embeddings: int = 128
    qk_rope_head_dim: int = 64
    rope_theta: float = 10000.0
    compress_rope_theta: float = 40000.0
    rope_factor: float = 40.0
    beta_fast: int = 32
    beta_slow: int = 1
    original_max_position_embeddings: int = 64
    compress_ratios: tuple[int, ...] = (0, 4, 128)


def reference_precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(low, high, dim):
        if low == high:
            high += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    positions = torch.arange(seqlen)
    angles = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope_real(x, freqs_cos, freqs_sin, positions, inverse=False):
    half = x.shape[-1] // 2
    x_pair = x.float().unflatten(-1, (-1, 2))
    even = x_pair[..., 0]
    odd = x_pair[..., 1]
    cos = freqs_cos.index_select(0, positions.to(torch.long))[:, :half].float()
    sin = freqs_sin.index_select(0, positions.to(torch.long))[:, :half].float()
    while cos.ndim < even.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    if inverse:
        out_even = even * cos + odd * sin
        out_odd = odd * cos - even * sin
    else:
        out_even = even * cos - odd * sin
        out_odd = even * sin + odd * cos
    return torch.stack([out_even, out_odd], dim=-1).flatten(-2)


def apply_rope_complex_reference(x, freqs_cis, positions, inverse=False):
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    freqs = freqs_cis.index_select(0, positions.to(torch.long))
    if inverse:
        freqs = freqs.conj()
    while freqs.ndim < x_complex.ndim:
        freqs = freqs.unsqueeze(-2)
    return torch.view_as_real(x_complex * freqs).flatten(-2)


def assert_tables_match_model_semantics(config, compress_ratio):
    base, original_seq_len = rope_profile_for_compress_ratio(config, compress_ratio)
    freqs_cis = reference_precompute_freqs_cis(
        config.qk_rope_head_dim,
        config.max_position_embeddings,
        original_seq_len,
        base,
        config.rope_factor,
        config.beta_fast,
        config.beta_slow,
    )
    freqs_cos, freqs_sin = build_deepseek_v4_rope_tables(config, compress_ratio, dtype=torch.float32)
    half = config.qk_rope_head_dim // 2
    torch.testing.assert_close(freqs_cos[:, :half], freqs_cis.real)
    torch.testing.assert_close(freqs_sin[:, :half], freqs_cis.imag)
    torch.testing.assert_close(freqs_cos[:, half:], freqs_cos[:, :half])
    torch.testing.assert_close(freqs_sin[:, half:], freqs_sin[:, :half])


def test_rope_tables_match_model_polar_semantics_for_swa_and_compressed_profiles():
    config = RopeConfig()
    assert_tables_match_model_semantics(config, compress_ratio=0)
    assert_tables_match_model_semantics(config, compress_ratio=4)
    assert_tables_match_model_semantics(config, compress_ratio=128)


def test_precompute_freqs_cos_sin_supports_bf16_output_after_fp32_trig():
    freqs_cos, freqs_sin = precompute_freqs_cos_sin(
        dim=64,
        seqlen=32,
        original_seq_len=64,
        base=40000.0,
        factor=40.0,
        beta_fast=32,
        beta_slow=1,
        dtype=torch.bfloat16,
    )
    assert freqs_cos.shape == (32, 64)
    assert freqs_sin.shape == (32, 64)
    assert freqs_cos.dtype == torch.bfloat16
    assert freqs_sin.dtype == torch.bfloat16


def test_real_forward_and_inverse_rope_match_complex_apply_rotary_emb_semantics():
    torch.manual_seed(0)
    config = RopeConfig(max_position_embeddings=64)
    positions = torch.tensor([0, 3, 7, 11], dtype=torch.int64)
    x = torch.randn(4, 3, config.qk_rope_head_dim, dtype=torch.bfloat16)
    base, original_seq_len = rope_profile_for_compress_ratio(config, 4)
    freqs_cis = reference_precompute_freqs_cis(
        config.qk_rope_head_dim,
        config.max_position_embeddings,
        original_seq_len,
        base,
        config.rope_factor,
        config.beta_fast,
        config.beta_slow,
    )
    freqs_cos, freqs_sin = build_deepseek_v4_rope_tables(config, 4, dtype=torch.float32)

    forward_real = apply_rope_real(x, freqs_cos, freqs_sin, positions, inverse=False)
    forward_complex = apply_rope_complex_reference(x, freqs_cis, positions, inverse=False)
    torch.testing.assert_close(forward_real, forward_complex)

    inverse_real = apply_rope_real(x, freqs_cos, freqs_sin, positions, inverse=True)
    inverse_complex = apply_rope_complex_reference(x, freqs_cis, positions, inverse=True)
    torch.testing.assert_close(inverse_real, inverse_complex)


def test_token_and_half_materialization_are_full_table_gathers():
    config = RopeConfig(max_position_embeddings=64)
    freqs_cos, freqs_sin = build_deepseek_v4_rope_tables(config, 4, dtype=torch.bfloat16)
    positions = torch.tensor([1, 5, 9], dtype=torch.int32)
    token_cos, token_sin = materialize_token_rope_tables(freqs_cos, freqs_sin, positions)
    torch.testing.assert_close(token_cos, freqs_cos[positions.long()])
    torch.testing.assert_close(token_sin, freqs_sin[positions.long()])

    half_cos, half_sin = materialize_half_rope_tables(freqs_cos, freqs_sin, positions)
    half = config.qk_rope_head_dim // 2
    torch.testing.assert_close(half_cos, freqs_cos[positions.long(), :half].float())
    torch.testing.assert_close(half_sin, freqs_sin[positions.long(), :half].float())
