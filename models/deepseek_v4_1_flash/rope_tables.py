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

from models.deepseek_v4_1_flash.config import FLASH, DeepSeekV41Config


def precompute_rope_tables(
    sequence_length: int,
    compressed_attention: bool,
    config: DeepSeekV41Config = FLASH,
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


def materialize_token_rope_tables(
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather from caller-owned tables without recomputing RoPE or moving devices.

    Tables have shape [capacity, rope_dim // 2] and share a floating dtype
    and device with each other. Integer positions must be on that device.
    Outputs preserve the position shape plus the table width; negative
    positions produce identity rotation (cos=1, sin=0). Nonnegative positions
    must be below capacity. The caller owns table allocation and profile choice.
    """
    if freqs_cos.ndim != 2 or freqs_cos.shape != freqs_sin.shape or freqs_cos.shape[0] == 0:
        raise ValueError("RoPE tables must have matching [capacity, width] shapes and nonzero capacity")
    if freqs_cos.dtype != freqs_sin.dtype or not freqs_cos.is_floating_point():
        raise ValueError("RoPE tables must share a floating dtype")
    if freqs_cos.device != freqs_sin.device or position_ids.device != freqs_cos.device:
        raise ValueError("RoPE tables and positions must be on the same device")
    if position_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("RoPE positions must be INT32 or INT64")
    rows = position_ids.clamp_min(0).to(torch.long).reshape(-1)
    shape = (*position_ids.shape, freqs_cos.shape[1])
    invalid = (position_ids < 0).unsqueeze(-1)
    cos = freqs_cos.index_select(0, rows).reshape(shape).masked_fill(invalid, 1.0)
    sin = freqs_sin.index_select(0, rows).reshape(shape).masked_fill(invalid, 0.0)
    return cos, sin


def select_rope_rows(
    position_ids: torch.Tensor,
    compressed_attention: bool,
    config: DeepSeekV41Config = FLASH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RoPE tables then select rows; use materialize_token_rope_tables for cached tables."""
    if position_ids.numel() == 0:
        shape = (*position_ids.shape, config.qk_rope_head_dim // 2)
        empty = torch.empty(shape, dtype=torch.float32, device=position_ids.device)
        return empty, empty.clone()
    valid = position_ids >= 0
    maximum = max(int(position_ids.clamp_min(0).max()) + 1, 1)
    cos, sin = precompute_rope_tables(maximum, compressed_attention, config)
    cos = cos.to(position_ids.device)
    sin = sin.to(position_ids.device)
    rows = position_ids.clamp_min(0).to(torch.long)
    selected_cos = cos[rows].masked_fill(~valid.unsqueeze(-1), 1.0)
    selected_sin = sin[rows].masked_fill(~valid.unsqueeze(-1), 0.0)
    return selected_cos, selected_sin
