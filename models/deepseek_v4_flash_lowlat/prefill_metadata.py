# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Batch-one prompt metadata for the shared DSpark prefill forward."""

import torch

from .prefill_cache import CP_SIZE, RANKS, prefill_block_tables, prefill_slot_mappings


MAX_PREFILL_TOKENS = 8192
HIDDEN = 4096
HC_MULT = 4
VOCAB = 129280


def _replicate(rows):
    return rows.unsqueeze(0).expand(RANKS, *rows.shape).contiguous()


def prefill_tap_positions(start_pos, token_count):
    """Return one CP group's rank-local tap positions, with padding marked -1."""
    if type(start_pos) is not int or start_pos < 0 or type(token_count) is not int:
        raise ValueError("start_pos and token_count must be nonnegative integers")
    if not 1 <= token_count <= MAX_PREFILL_TOKENS:
        raise ValueError("token_count must be in [1, 8192]")
    padded = (token_count + CP_SIZE - 1) // CP_SIZE * CP_SIZE
    positions = torch.arange(start_pos, start_pos + padded, dtype=torch.int64)
    positions[token_count:] = -1
    return positions.reshape(CP_SIZE, padded // CP_SIZE)


def prepare_prefill_metadata(tokens, *, start_pos, tables, caches, rope_cos, rope_sin, max_logit_rows, terminal,
                             block_counts=None):
    """Build the B1 CP4/EP8 prompt metadata for one chunk.

    ``rope_cos`` and ``rope_sin`` are the existing decode tables with the
    uncompressed and compressed profiles at indices 0 and 1. Cache tables
    must name the currently live worker-owned allocations. Pass the decode
    ``block_counts`` when its compressor tables contain cyclic page mappings.
    """
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 1 or tokens.dtype != torch.int64:
        raise TypeError("tokens must be a one-dimensional INT64 tensor")
    if tokens.device.type != "cpu" or not 1 <= tokens.numel() <= MAX_PREFILL_TOKENS:
        raise ValueError("A CPU token chunk must contain 1..8192 tokens")
    if bool(((tokens < 0) | (tokens >= VOCAB)).any()):
        raise ValueError("A prompt token is outside the vocabulary")
    if type(start_pos) is not int or start_pos < 0:
        raise ValueError("start_pos must be a nonnegative integer")
    if type(max_logit_rows) is not int or max_logit_rows <= 0 or type(terminal) is not bool:
        raise ValueError("A positive logits capacity and a terminal flag are required")
    if (not isinstance(rope_cos, torch.Tensor) or not isinstance(rope_sin, torch.Tensor)
            or rope_cos.dtype != torch.bfloat16 or rope_sin.dtype != torch.bfloat16
            or rope_cos.shape != rope_sin.shape or rope_cos.ndim != 3
            or rope_cos.shape[0] != 2 or rope_cos.shape[2] != 64
            or rope_cos.device.type != "cpu" or rope_sin.device.type != "cpu"):
        raise ValueError("RoPE tables must be matching CPU BF16 [2, positions, 64] tensors")
    logical = tokens.numel()
    padded = (logical + CP_SIZE - 1) // CP_SIZE * CP_SIZE
    if start_pos + padded > rope_cos.shape[1]:
        raise ValueError("The CP-padded prompt chunk exceeds the supplied RoPE tables")
    local = padded // CP_SIZE
    token_rows = torch.zeros(padded, dtype=torch.int64)
    token_rows[:logical] = tokens
    full_positions = torch.arange(start_pos, start_pos + padded, dtype=torch.int32)
    active_positions = full_positions.clone()
    active_positions[logical:] = -1
    inputs = {
        "query_start_loc": _replicate(torch.tensor([0, logical], dtype=torch.int32)),
        "state_checkpoint_ends": torch.zeros((RANKS, 1, (logical + 511) // 512, 6), dtype=torch.int32),
        "position_ids_full": _replicate(full_positions),
        "position_ids_local": torch.stack([full_positions[(rank % CP_SIZE) * local : (rank % CP_SIZE + 1) * local]
                                            for rank in range(RANKS)]),
        "input_ids": torch.stack([token_rows[(rank % CP_SIZE) * local : (rank % CP_SIZE + 1) * local]
                                   for rank in range(RANKS)]),
        "logit_row_indices": torch.full((RANKS, max_logit_rows), -1, dtype=torch.int32),
    }
    if terminal:
        inputs["logit_row_indices"][::CP_SIZE, 0] = logical - 1
    for name, profile, positions in (
        ("swa_freqs", 0, full_positions),
        ("compressed_freqs", 1, full_positions),
        ("hca_cmp_freqs", 1, torch.where((full_positions + 1) % 128 == 0, full_positions - 127, 0)),
        ("csa_cmp_freqs", 1, torch.where((full_positions + 1) % 4 == 0, full_positions - 3, 0)),
    ):
        indices = positions.to(torch.int64)
        inputs[name + "_cos"] = _replicate(rope_cos[profile].index_select(0, indices))
        inputs[name + "_sin"] = _replicate(rope_sin[profile].index_select(0, indices))
    inputs.update(prefill_block_tables(tables, block_counts=block_counts))
    inputs.update(prefill_slot_mappings(_replicate(active_positions), tables, caches, block_counts=block_counts))
    return inputs


def prepare_prefill_hidden(embeddings):
    """Replicate actual BF16 prompt embeddings across HC lanes and CP groups."""
    if (not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2
            or embeddings.shape[1] != HIDDEN or embeddings.dtype != torch.bfloat16):
        raise ValueError("embeddings must have shape [tokens, 4096] and dtype BF16")
    if embeddings.device.type != "cpu" or not 1 <= embeddings.shape[0] <= MAX_PREFILL_TOKENS:
        raise ValueError("embeddings must be a nonempty CPU prompt chunk")
    rows = embeddings.shape[0]
    padded = (rows + CP_SIZE - 1) // CP_SIZE * CP_SIZE
    hidden = torch.zeros((padded, HC_MULT, HIDDEN), dtype=torch.float32)
    hidden[:rows] = embeddings.float().unsqueeze(1)
    return _replicate(hidden)
