# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""YaRN and base-RoPE table generation for text-backbone attention."""

import math

import torch

from .config import FLASH as M, DeepSeekV41Config


# model config
ROPE_DIM = M.qk_rope_head_dim


def precompute_rope_tables(
    sequence_length: int,
    compressed_attention: bool,
    config: DeepSeekV41Config = M,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 cosine and sine tables for adjacent-pair rotary embedding."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    dim = config.qk_rope_head_dim
    base = config.compress_rope_theta if compressed_attention else config.rope_theta
    frequencies = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if compressed_attention:

        def corrected_dimension(rotations: int) -> float:
            numerator = config.original_max_position_embeddings
            return dim * math.log(numerator / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dimension(config.beta_fast)), 0)
        high = min(math.ceil(corrected_dimension(config.beta_slow)), dim - 1)
        ramp = torch.arange(dim // 2, dtype=torch.float32)
        ramp = ((ramp - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        frequencies = frequencies / config.rope_factor * (1 - smooth) + frequencies * smooth
    angles = torch.outer(torch.arange(sequence_length, dtype=torch.float32), frequencies)
    return angles.cos(), angles.sin()


def prepare_forward_rope_tables(
    sequence_length: int,
    tables: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare or reuse full FP32 tables with profile order base, compressed."""
    if not 1 <= sequence_length <= M.max_position_embeddings:
        raise ValueError("RoPE capacity must fit the model position range")
    if tables is None:
        base_cos, base_sin = precompute_rope_tables(sequence_length, False)
        compressed_cos, compressed_sin = precompute_rope_tables(sequence_length, True)
        return torch.stack((base_cos, compressed_cos)), torch.stack((base_sin, compressed_sin))
    freqs_cos, freqs_sin = tables
    for value in (freqs_cos, freqs_sin):
        if value.device.type != "cpu" or value.dtype != torch.float32 or not value.is_contiguous():
            raise ValueError("full RoPE tables must be contiguous CPU FP32 tensors")
        if value.ndim != 3 or value.shape[0] != 2 or value.shape[2] != ROPE_DIM // 2:
            raise ValueError("full RoPE tables must have shape [2, rows, rope_dim // 2]")
        if not sequence_length <= value.shape[1] <= M.max_position_embeddings:
            raise ValueError("full RoPE tables must cover the retained history capacity")
    if freqs_cos.shape != freqs_sin.shape:
        raise ValueError("full RoPE cosine and sine tables must have the same shape")
    return freqs_cos, freqs_sin
