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
import pypto.language as pl

from config import FLASH as M, DeepSeekV41Config, T_DYN


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


ROPE_ROWS_DYN = pl.dynamic("V41_ROPE_ROWS_DYN")


@pl.jit.inline
def materialize_rope_rows(
    freqs_cos: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    freqs_sin: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    pair_start: pl.constexpr,
) -> pl.Scalar[pl.TASK_ID]:
    """Gather active token rows from serving-owned full tables inside the graph.

    Tables are read-only FP32 half-width arrays for one RoPE profile. The caller
    provides 0 <= num_tokens <= T and positions below table capacity. Negative
    positions produce identity rotation for unpublished compressed tokens.
    pair_start selects p-1 for completed C2A pairs and identity otherwise.
    Padding rows are untouched. Return the producer TaskId for consumers that
    use explicit dependencies, as the attention entries do with cache_ready.
    """
    # A fixed worker grid also supports idle ranks: never submit core_num=0.
    with pl.spmd(32, name_hint="v41_rope_rows") as rope_ready:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, num_tokens, 32):
            position = pl.cast(pl.read(position_ids, [token]), pl.INDEX)
            if pair_start:
                if position >= 0 and (position + 1) % 2 == 0:
                    position = position - 1
                else:
                    position = -1
            if position >= 0:
                rope_cos[token : token + 1, :] = freqs_cos[position : position + 1, :]
                rope_sin[token : token + 1, :] = freqs_sin[position : position + 1, :]
            else:
                rope_cos[token : token + 1, :] = pl.full([1, ROPE_DIM // 2], dtype=pl.FP32, value=1.0)
                rope_sin[token : token + 1, :] = pl.full([1, ROPE_DIM // 2], dtype=pl.FP32, value=0.0)
    return rope_ready
