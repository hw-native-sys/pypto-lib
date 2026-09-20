# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Torch metadata lowering for packed prefill and continuous-batch decode."""

from dataclasses import dataclass
from typing import Mapping

import torch

from models.deepseek_v4_1_flash.config import BLOCK_SIZE, FLASH


@dataclass(frozen=True)
class ForwardMetadata:
    """Canonical token-major metadata consumed by all layer kernels."""

    query_start_loc: torch.Tensor
    query_lens: torch.Tensor
    logit_row_indices: torch.Tensor
    request_ids: torch.Tensor
    position_ids: torch.Tensor
    kv_seq_lens: torch.Tensor
    new_kv_seq_lens: torch.Tensor
    window_slots: torch.Tensor
    window_indices: torch.Tensor
    window_lens: torch.Tensor
    compressed_slots: Mapping[int, torch.Tensor]
    index_slots: Mapping[int, torch.Tensor]
    compressor_state_rows: Mapping[int, torch.Tensor]
    compressed_seq_lens: Mapping[int, torch.Tensor]
    compressed_seq_remainders: Mapping[int, torch.Tensor]
    compressed_lens: Mapping[int, torch.Tensor]
    compressor_output_start_loc: Mapping[int, torch.Tensor]
    compressor_source_token_indices: Mapping[int, torch.Tensor]
    compressor_position_ids: Mapping[int, torch.Tensor]
    compressed_rope_position_ids: Mapping[int, torch.Tensor]


def request_ids_from_starts(query_start_loc: torch.Tensor) -> torch.Tensor:
    """Expand packed cumulative query lengths into one request id per token."""
    lengths = query_start_loc[1:].to(torch.int64) - query_start_loc[:-1].to(torch.int64)
    return torch.repeat_interleave(torch.arange(lengths.numel(), device=lengths.device), lengths).to(
        torch.int32
    )


def paged_slots(
    positions: torch.Tensor,
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    storage_block_size: int = BLOCK_SIZE,
    logical_divisor: int = 1,
    publish_only_complete: bool = False,
) -> torch.Tensor:
    """Map logical token positions to flattened physical cache rows."""
    logical = torch.div(positions, logical_divisor, rounding_mode="floor")
    logical_block = torch.div(logical, storage_block_size, rounding_mode="floor")
    offset = logical.remainder(storage_block_size)
    physical = block_table[request_ids.to(torch.long), logical_block.to(torch.long)]
    slots = physical.to(torch.int64) * storage_block_size + offset.to(torch.int64)
    if publish_only_complete and logical_divisor > 1:
        slots = slots.masked_fill((positions + 1).remainder(logical_divisor) != 0, -1)
    return slots


def compressor_metadata(
    query_start_loc: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    compression_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ragged compressor output starts, source-token rows, and group-first positions."""
    query_lens = query_start_loc[1:].to(torch.int64) - query_start_loc[:-1].to(torch.int64)
    old_groups = torch.div(kv_seq_lens.to(torch.int64), compression_ratio, rounding_mode="floor")
    new_groups = torch.div(kv_seq_lens.to(torch.int64) + query_lens, compression_ratio, rounding_mode="floor")
    output_lens = new_groups - old_groups
    output_starts = torch.cat(
        (torch.zeros(1, dtype=torch.int64, device=query_start_loc.device), output_lens.cumsum(0))
    )
    source_rows: list[torch.Tensor] = []
    output_positions: list[torch.Tensor] = []
    for request in range(query_lens.numel()):
        groups = torch.arange(old_groups[request], new_groups[request], device=query_start_loc.device)
        completed_positions = (groups + 1) * compression_ratio - 1
        local_rows = completed_positions - kv_seq_lens[request].to(torch.int64)
        source_rows.append(query_start_loc[request].to(torch.int64) + local_rows)
        output_positions.append(groups * compression_ratio)
    empty = torch.empty(0, dtype=torch.int64, device=query_start_loc.device)
    return (
        output_starts.to(torch.int32),
        torch.cat(source_rows).to(torch.int32) if source_rows else empty.to(torch.int32),
        torch.cat(output_positions).to(torch.int32) if output_positions else empty.to(torch.int32),
    )


def window_metadata(
    positions: torch.Tensor,
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    window: int = FLASH.sliding_window,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build paged write slots plus causal indices for the visible sliding window."""
    slots = paged_slots(positions, request_ids, block_table)
    lens = torch.minimum(positions + 1, torch.full_like(positions, window)).to(torch.int32)
    offsets = torch.arange(window, device=positions.device)
    starts = positions - lens.to(positions.dtype) + 1
    visible = starts.unsqueeze(-1) + offsets
    valid = offsets.unsqueeze(0) < lens.unsqueeze(-1)
    logical_block = torch.div(visible.clamp_min(0), BLOCK_SIZE, rounding_mode="floor")
    block_offset = visible.clamp_min(0).remainder(BLOCK_SIZE)
    physical = block_table[request_ids.to(torch.long).unsqueeze(-1), logical_block.to(torch.long)]
    indices = physical.to(torch.int64) * BLOCK_SIZE + block_offset.to(torch.int64)
    return slots, indices.masked_fill(~valid, -1).to(torch.int32), lens


def build_forward_metadata(
    query_start_loc: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    window_block_table: torch.Tensor,
    compressed_block_tables: Mapping[int, torch.Tensor],
) -> ForwardMetadata:
    """Lower engine inputs for packed prefill or one-token-per-request decode."""
    if query_start_loc.ndim != 1 or kv_seq_lens.ndim != 1:
        raise ValueError("query_start_loc and kv_seq_lens must be one-dimensional")
    if query_start_loc.numel() != kv_seq_lens.numel() + 1:
        raise ValueError("query_start_loc must contain one more element than kv_seq_lens")
    query_lens = query_start_loc[1:].to(torch.int64) - query_start_loc[:-1].to(torch.int64)
    if bool((query_lens < 0).any()) or int(query_start_loc[0]) != 0:
        raise ValueError("query_start_loc must be nondecreasing and start at zero")
    if bool((kv_seq_lens < 0).any()):
        raise ValueError("kv_seq_lens must be non-negative")
    tables_by_ratio: dict[int, torch.Tensor] = {}
    for source in FLASH.kv_source_layer_ids:
        if source not in compressed_block_tables:
            raise ValueError(f"missing compressed block table for source layer {source}")
        ratio = FLASH.compress_ratios[source]
        previous = tables_by_ratio.setdefault(ratio, compressed_block_tables[source])
        if not torch.equal(previous, compressed_block_tables[source]):
            raise ValueError(f"compression ratio {ratio} sources must share one compressed block table")
    request_ids = request_ids_from_starts(query_start_loc)
    logit_rows = torch.where(query_lens > 0, query_start_loc[1:].to(torch.int64) - 1, -1).to(torch.int32)
    local_offsets = torch.arange(request_ids.numel(), device=request_ids.device)
    local_offsets -= query_start_loc[request_ids.to(torch.long)].to(local_offsets.dtype)
    positions = kv_seq_lens[request_ids.to(torch.long)].to(torch.int64) + local_offsets
    window_slots, window_indices, window_lens = window_metadata(positions, request_ids, window_block_table)
    compressed_slots: dict[int, torch.Tensor] = {}
    index_slots: dict[int, torch.Tensor] = {}
    state_rows: dict[int, torch.Tensor] = {}
    compressed_seq_lens: dict[int, torch.Tensor] = {}
    compressed_seq_remainders: dict[int, torch.Tensor] = {}
    compressed_lens: dict[int, torch.Tensor] = {}
    compressor_output_starts: dict[int, torch.Tensor] = {}
    compressor_source_rows: dict[int, torch.Tensor] = {}
    compressor_positions: dict[int, torch.Tensor] = {}
    compressed_rope_positions: dict[int, torch.Tensor] = {}
    new_kv_seq_lens = kv_seq_lens.to(torch.int64) + query_lens
    for source in FLASH.kv_source_layer_ids:
        ratio = FLASH.compress_ratios[source]
        storage_rows = BLOCK_SIZE
        compressed_slots[source] = paged_slots(
            positions, request_ids, compressed_block_tables[source], storage_rows, ratio, True
        )
        index_slots[source] = paged_slots(
            positions, request_ids, compressed_block_tables[source], storage_rows, ratio, True
        )
        compressed_seq_lens[source] = torch.div(new_kv_seq_lens, ratio, rounding_mode="floor").to(torch.int32)
        compressed_seq_remainders[source] = new_kv_seq_lens.remainder(ratio).to(torch.int32)
        compressed_lens[source] = torch.div(positions + 1, ratio, rounding_mode="floor").to(torch.int32)
        output_starts, source_rows, output_positions = compressor_metadata(
            query_start_loc, kv_seq_lens, ratio
        )
        compressor_output_starts[source] = output_starts
        compressor_source_rows[source] = source_rows
        compressor_positions[source] = output_positions
        complete = (positions + 1).remainder(ratio) == 0
        compressed_rope_positions[source] = torch.where(complete, positions + 1 - ratio, -1).to(torch.int32)
        if ratio > 1:
            state_rows[source] = request_ids.to(torch.int64)
    return ForwardMetadata(
        query_start_loc=query_start_loc.to(torch.int32),
        query_lens=query_lens.to(torch.int32),
        logit_row_indices=logit_rows,
        request_ids=request_ids,
        position_ids=positions.to(torch.int32),
        kv_seq_lens=kv_seq_lens.to(torch.int32),
        new_kv_seq_lens=new_kv_seq_lens.to(torch.int32),
        window_slots=window_slots,
        window_indices=window_indices,
        window_lens=window_lens,
        compressed_slots=compressed_slots,
        index_slots=index_slots,
        compressor_state_rows=state_rows,
        compressed_seq_lens=compressed_seq_lens,
        compressed_seq_remainders=compressed_seq_remainders,
        compressed_lens=compressed_lens,
        compressor_output_start_loc=compressor_output_starts,
        compressor_source_token_indices=compressor_source_rows,
        compressor_position_ids=compressor_positions,
        compressed_rope_position_ids=compressed_rope_positions,
    )
