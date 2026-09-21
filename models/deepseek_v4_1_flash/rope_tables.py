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


def select_rope_rows(
    position_ids: torch.Tensor,
    compressed_attention: bool,
    config: DeepSeekV41Config = FLASH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize the RoPE rows needed by one packed forward invocation."""
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
