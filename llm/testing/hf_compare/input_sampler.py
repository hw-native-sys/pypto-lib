# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Helpers for synthesizing inputs (activations, KV history, RoPE tables)."""
from __future__ import annotations

from collections.abc import Callable

import torch


def uniform(low: float = -0.5, high: float = 0.5) -> Callable[[tuple[int, ...], torch.dtype, torch.Generator], torch.Tensor]:
    """Return a sampler producing uniform ``[low, high)`` values."""
    scale = high - low

    def _fn(shape: tuple[int, ...], dtype: torch.dtype, g: torch.Generator) -> torch.Tensor:
        t = torch.rand(shape, generator=g) * scale + low
        return t.to(dtype)
    return _fn


def normal(mean: float = 0.0, std: float = 1.0) -> Callable[[tuple[int, ...], torch.dtype, torch.Generator], torch.Tensor]:
    def _fn(shape: tuple[int, ...], dtype: torch.dtype, g: torch.Generator) -> torch.Tensor:
        t = torch.randn(shape, generator=g) * std + mean
        return t.to(dtype)
    return _fn


def constant(value: float) -> Callable[[tuple[int, ...], torch.dtype, torch.Generator], torch.Tensor]:
    def _fn(shape: tuple[int, ...], dtype: torch.dtype, g: torch.Generator) -> torch.Tensor:  # noqa: ARG001
        return torch.full(shape, value, dtype=dtype)
    return _fn


def int_fill(value: int) -> Callable[[tuple[int, ...], torch.dtype, torch.Generator], torch.Tensor]:
    def _fn(shape: tuple[int, ...], dtype: torch.dtype, g: torch.Generator) -> torch.Tensor:  # noqa: ARG001
        return torch.full(shape, value, dtype=dtype)
    return _fn


def build_rope_tables(
    max_seq: int,
    head_dim: int,
    base: float = 1_000_000.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin tables compatible with HF rotate_half form."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(max_seq, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)
