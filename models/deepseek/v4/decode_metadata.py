# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode fixture metadata lowering helpers.

The runtime kernels consume lowered metadata: block tables map logical cache
blocks to physical blocks, and slot mappings are flattened physical rows where
``-1`` means no-write.
"""

from typing import Callable

import torch

from config import BLOCK_SIZE, DECODE_BATCH, DECODE_SEQ, FLASH as M


def resolve_start_positions(
    start_pos: int | None,
    *,
    batch: int = DECODE_BATCH,
    seq: int = DECODE_SEQ,
    max_seq_len: int = M.max_position_embeddings,
    default_fn: Callable[[], torch.Tensor] | None = None,
) -> torch.Tensor:
    if start_pos is not None:
        starts = torch.full((batch,), int(start_pos), dtype=torch.int32)
    elif default_fn is not None:
        starts = default_fn().to(torch.int32)
    else:
        starts = torch.zeros(batch, dtype=torch.int32)
    _validate_starts(starts, seq=seq, max_seq_len=max_seq_len)
    return starts


def position_ids_from_starts(starts: torch.Tensor, *, seq: int = DECODE_SEQ) -> torch.Tensor:
    offsets = torch.arange(seq, dtype=torch.int32, device=starts.device)
    return starts.to(torch.int32).unsqueeze(1) + offsets.unsqueeze(0)


def kv_seq_lens_from_starts(
    starts: torch.Tensor,
    *,
    seq: int = DECODE_SEQ,
    commit_tokens: int | None = None,
) -> torch.Tensor:
    visible_tokens = seq if commit_tokens is None else commit_tokens
    if visible_tokens < 0 or visible_tokens > seq:
        raise ValueError(f"commit_tokens must be in [0, {seq}], got {visible_tokens}")
    return (starts.to(torch.int64) + visible_tokens).to(torch.int32)


def block_table(
    *,
    batch: int,
    table_blocks: int,
    physical_blocks: int | None = None,
    permuted: bool = False,
) -> torch.Tensor:
    physical_blocks = table_blocks if physical_blocks is None else physical_blocks
    table_cols = torch.arange(table_blocks, dtype=torch.int32)
    physical_cols = table_cols % physical_blocks
    if permuted and physical_blocks > 1:
        physical_cols = (physical_cols * 7 + 3) % physical_blocks
    batch_offsets = torch.arange(batch, dtype=torch.int32).unsqueeze(1) * physical_blocks
    return batch_offsets + physical_cols.unsqueeze(0)


def ori_slot_mapping(
    positions: torch.Tensor,
    ori_block_table: torch.Tensor,
    *,
    block_size: int = BLOCK_SIZE,
    window: int = M.sliding_window,
) -> torch.Tensor:
    positions_i64 = positions.to(torch.int64)
    table_i64 = ori_block_table.to(device=positions.device, dtype=torch.int64)
    slot = positions_i64 % window
    logical_blk = slot // block_size
    intra = slot % block_size
    in_bounds = logical_blk < table_i64.shape[1]
    clamped_blk = torch.clamp(logical_blk, max=table_i64.shape[1] - 1)
    blk = torch.gather(table_i64, 1, clamped_blk)
    valid = in_bounds & (blk >= 0)
    return torch.where(valid, blk * block_size + intra, -1)


def compressed_slot_mapping(
    positions: torch.Tensor,
    cmp_block_table: torch.Tensor,
    *,
    compress_ratio: int,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    positions_i64 = positions.to(torch.int64)
    table_i64 = cmp_block_table.to(device=positions.device, dtype=torch.int64)
    boundary = (positions_i64 + 1) % compress_ratio == 0
    cache_col = positions_i64 // compress_ratio
    logical_blk = cache_col // block_size
    intra = cache_col % block_size
    in_bounds = logical_blk < table_i64.shape[1]
    clamped_blk = torch.clamp(logical_blk, max=table_i64.shape[1] - 1)
    blk = torch.gather(table_i64, 1, clamped_blk)
    valid = boundary & in_bounds & (blk >= 0)
    return torch.where(valid, blk * block_size + intra, -1)


def mask_uncommitted_compressed_boundaries(
    mapping: torch.Tensor,
    positions: torch.Tensor,
    *,
    compress_ratio: int,
    commit_tokens: int | None,
) -> torch.Tensor:
    if commit_tokens is None:
        return mapping
    if mapping.shape != positions.shape:
        raise ValueError("compressed boundary mask expects mapping and positions to have the same shape")
    if mapping.ndim != 2:
        raise ValueError("compressed boundary mask expects [B, S] tensors")
    if commit_tokens < 0 or commit_tokens > mapping.shape[1]:
        raise ValueError(f"commit_tokens must be in [0, {mapping.shape[1]}], got {commit_tokens}")
    masked = mapping.clone()
    positions_i64 = positions.to(torch.int64)
    token_cols = torch.arange(positions.shape[1], device=positions.device).unsqueeze(0)
    uncommitted = token_cols >= commit_tokens
    boundary = (positions_i64 + 1) % compress_ratio == 0
    masked[uncommitted & boundary] = -1
    return masked


def state_slot_mapping(
    positions: torch.Tensor,
    state_block_table: torch.Tensor,
    *,
    state_block_size: int,
) -> torch.Tensor:
    positions_i64 = positions.to(torch.int64)
    table_i64 = state_block_table.to(device=positions.device, dtype=torch.int64)
    logical_blk = positions_i64 // state_block_size
    intra = positions_i64 % state_block_size
    in_bounds = logical_blk < table_i64.shape[1]
    clamped_blk = torch.clamp(logical_blk, max=table_i64.shape[1] - 1)
    blk = torch.gather(table_i64, 1, clamped_blk)
    valid = in_bounds & (blk >= 0)
    return torch.where(valid, blk * state_block_size + intra, -1)


def _validate_starts(starts: torch.Tensor, *, seq: int, max_seq_len: int) -> None:
    if bool((starts < 0).any()):
        raise ValueError("decode start positions must be non-negative")
    if bool((starts.to(torch.int64) + seq > max_seq_len).any()):
        raise ValueError(f"decode start positions plus seq length must fit MAX_SEQ_LEN={max_seq_len}")
