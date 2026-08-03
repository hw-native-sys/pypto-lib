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

from config import (
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    C128_COMPRESSOR_BLOCK_SIZE,
    DECODE_BATCH,
    DECODE_SEQ,
    DECODE_START_POS,
    FLASH as M,
)


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


# --- Canonical decode fixture start-position sets, one per attention family. ---
# Each set packs the family's distinct position regimes into the batch dimension
# (one start_pos per request); `long_pos` (the 8k target) adds the long-context
# rolling-state / INT64-slot / long-topk path. Sets are order-preserving-deduped
# (S=1 collapses the `-seq`/`-1` boundary pairs; some regimes also coincide at the
# current constants, e.g. window-1 == state_block*32-1 at ratio 4). Coverage is
# capped at `batch` slots, so sets are kept <= batch to avoid silent truncation.

def _tile_starts(pattern: list[int], batch: int) -> torch.Tensor:
    uniq: list[int] = []
    for p in pattern:
        if p not in uniq:
            uniq.append(int(p))
    vals = torch.empty((batch,), dtype=torch.int32)
    for b in range(batch):
        vals[b] = uniq[b % len(uniq)]
    return vals


# `long_pos` (8k) is listed first in each set so it survives truncation even when
# batch < set size (until coverage is decoupled from batch), then the remaining
# regimes in descending importance.

def swa_decode_start_set(
    *,
    batch: int = DECODE_BATCH,
    window: int = M.sliding_window,
    long_pos: int = DECODE_START_POS,
) -> torch.Tensor:
    # long-context wraparound + in-window boundary + one in-window interior slot.
    pattern = [long_pos, window - 1, 31]
    return _tile_starts(pattern, batch)


def hca_decode_start_set(
    *,
    batch: int = DECODE_BATCH,
    compress_ratio: int = 128,
    state_block_size: int = C128_COMPRESSOR_BLOCK_SIZE,
    long_pos: int = DECODE_START_POS,
) -> torch.Tensor:
    R = compress_ratio
    pattern = [
        long_pos,              # 8k long-context
        R - 1,                 # compress boundary, one cache entry
        R,                     # no new boundary on 1st token; 2nd advances window
        2 * R - 1,             # compressed block crossing
        state_block_size - 1,  # last slot of state page 0
        10,                    # pre-compression, state page 1
    ]
    return _tile_starts(pattern, batch)


def csa_decode_start_set(
    *,
    batch: int = DECODE_BATCH,
    seq: int = DECODE_SEQ,
    compress_ratio: int = 4,
    state_block_size: int = C4A_COMPRESSOR_BLOCK_SIZE,
    cache_tile: int = 64,
    window: int = M.sliding_window,
    long_pos: int = DECODE_START_POS,
) -> torch.Tensor:
    R = compress_ratio
    pattern = [
        long_pos,                   # 8k long-context (rolling state, INT64 slot, topk 4096)
        0,                          # cold start, no valid compressed cache
        R - seq,                    # compress boundary on 2nd token (== R-1 at seq=1)
        R - 1,                      # compress boundary on 1st token
        2 * R - 1,                  # 2nd window with previous-window overlap
        window - 1,                 # sliding-window boundary (== state block 31->32 at ratio 4)
        window,                     # post-window ring-cache path
        state_block_size * 32 - 1,  # inner state logical block 31->32 crossing
        R * cache_tile - 1,         # indexer score over exactly one cache tile
        R * 2 * cache_tile - 1,     # indexer score over two cache tiles
    ]
    return _tile_starts(pattern, batch)


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
    # The physical pool is global and does not grow with batch. Interleave the
    # fixture's request-local logical pages inside that fixed pool; production
    # serving supplies allocator-owned block tables under the same contract.
    request_offsets = torch.arange(batch, dtype=torch.int32).unsqueeze(1)
    return (physical_cols.unsqueeze(0) * batch + request_offsets) % physical_blocks


def ori_slot_mapping(
    positions: torch.Tensor,
    table: torch.Tensor,
    *,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """Map absolute positions into the full paged ori-KV pool.

    Sliding-window visibility is lowered separately by
    :func:`swa_indices_and_lens`; it must not alias physical KV write rows.
    """
    positions_i64 = positions.to(torch.int64)
    table_i64 = table.to(device=positions.device, dtype=torch.int64)
    logical_blk = positions_i64 // block_size
    intra = positions_i64 % block_size
    in_bounds = logical_blk < table_i64.shape[1]
    clamped_blk = torch.clamp(logical_blk, max=table_i64.shape[1] - 1)
    blk = torch.gather(table_i64, 1, clamped_blk)
    valid = in_bounds & (blk >= 0)
    return torch.where(valid, blk * block_size + intra, -1)


def paged_slot_mapping(
    positions: torch.Tensor,
    table: torch.Tensor,
    *,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    positions_i64 = positions.to(torch.int64)
    table_i64 = table.to(device=positions.device, dtype=torch.int64)
    logical_blk = positions_i64 // block_size
    intra = positions_i64 % block_size
    in_bounds = logical_blk < table_i64.shape[1]
    clamped_blk = torch.clamp(logical_blk, max=table_i64.shape[1] - 1)
    blk = torch.gather(table_i64, 1, clamped_blk)
    valid = in_bounds & (blk >= 0)
    return torch.where(valid, blk * block_size + intra, -1)


def swa_indices_and_lens(
    positions: torch.Tensor,
    table: torch.Tensor,
    *,
    block_size: int = BLOCK_SIZE,
    window: int = M.sliding_window,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower decode SWA windows to physical KV-cache row indices.

    Each visible absolute logical position is translated with the same paged-KV
    block table contract as vLLM:
    ``physical_slot = block_table[req, pos // block_size] * block_size + pos % block_size``.
    Each row is ordered from the oldest visible token to the current token;
    invalid tail columns are padded with -1 and ``lens`` records the valid
    prefix length.
    """
    if positions.ndim != 2:
        raise ValueError("SWA indices expect positions with shape [B, S]")
    positions_i64 = positions.to(torch.int64)
    table_i64 = table.to(device=positions.device, dtype=torch.int64)
    batch, seq = positions_i64.shape
    indices = torch.full((batch * seq, window), -1, dtype=torch.int32, device=positions.device)
    lens = torch.zeros((batch * seq,), dtype=torch.int32, device=positions.device)

    for b in range(batch):
        for s in range(seq):
            t = b * seq + s
            abs_pos = int(positions_i64[b, s].item())
            start = max(0, abs_pos - window + 1)
            valid_len = abs_pos - start + 1
            lens[t] = valid_len
            for k, pos in enumerate(range(start, abs_pos + 1)):
                logical_blk = pos // block_size
                intra = pos % block_size
                if logical_blk >= table_i64.shape[1]:
                    continue
                blk = int(table_i64[b, logical_blk].item())
                if blk >= 0:
                    indices[t, k] = blk * block_size + intra
    return indices, lens


def history_window_swa_indices_and_lens(
    positions: torch.Tensor,
    window_block_table: torch.Tensor,
    *,
    block_size: int = BLOCK_SIZE,
    window: int = M.sliding_window,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower historical HCA/CSA window rows to physical KV-cache slots.

    Current decode-chunk positions are excluded from this list because HCA/CSA
    still attend current MTP tokens through their overlay raw-index range. The
    returned rows are packed oldest-to-newest; invalid tail columns are -1. The
    block table follows the same vLLM-style absolute logical block contract as
    SWA, while physical blocks may still be a small sliding-window ring.
    """
    if positions.ndim != 2:
        raise ValueError("history window indices expect positions with shape [B, S]")
    positions_i64 = positions.to(torch.int64)
    table_i64 = window_block_table.to(device=positions.device, dtype=torch.int64)
    batch, seq = positions_i64.shape
    indices = torch.full((batch * seq, window), -1, dtype=torch.int32, device=positions.device)
    lens = torch.zeros((batch * seq,), dtype=torch.int32, device=positions.device)

    for b in range(batch):
        for s in range(seq):
            t = b * seq + s
            abs_pos = int(positions_i64[b, s].item())
            overlay_positions = {int(positions_i64[b, os].item()) for os in range(s + 1)}
            start = max(0, abs_pos - window + 1)
            out_k = 0
            for pos in range(start, abs_pos + 1):
                if pos in overlay_positions:
                    continue
                logical_blk = pos // block_size
                intra = pos % block_size
                if logical_blk >= table_i64.shape[1]:
                    continue
                blk = int(table_i64[b, logical_blk].item())
                if blk >= 0:
                    indices[t, out_k] = blk * block_size + intra
                    out_k += 1
            lens[t] = out_k
    return indices, lens


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
