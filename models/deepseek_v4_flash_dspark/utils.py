# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side torch helpers for the decode/prefill test fixtures.

Paged-KV metadata lowering and RoPE/YaRN table generation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import torch

from context_geometry import (
    admit_ragged_page_counts,
    hetero_length_starts_values,
)

from config import (
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    C128_COMPRESSOR_BLOCK_SIZE,
    CSA_COMPRESS_RATIO,
    CSA_INNER_STATE_ROWS_PER_REQUEST,
    CSA_INNER_STATE_PAGES_PER_REQUEST,
    CSA_INNER_STATE_POOL_PAGES,
    CSA_MAIN_PAGES_AT_1M,
    CSA_STATE_ROWS_PER_REQUEST,
    CSA_STATE_PAGES_PER_REQUEST,
    CSA_STATE_POOL_PAGES,
    CSA_TOPK,
    CSA_TOPK_INVALID_INDEX,
    DECODE_BATCH,
    DECODE_SEQ,
    DECODE_START_POS,
    FLASH as M,
    HCA_COMPRESS_RATIO,
    HCA_STATE_ROWS_PER_REQUEST,
    HCA_STATE_PAGES_PER_REQUEST,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    MAX_CSA_CANDIDATES,
    SWA_PERSISTENT_PAGES_PER_REQUEST,
    SWA_PERSISTENT_ROWS_PER_REQUEST,
    SWA_SOURCE_INVALID,
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
        (R - min(seq, 2)) % R,      # compress boundary on 2nd token (1st at seq=1)
        R - 1,                      # compress boundary on 1st token
        2 * R - 1,                  # 2nd window with previous-window overlap
        window - 1,                 # sliding-window boundary
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


def cache_row_from_table(
    table: torch.Tensor,
    slot: int,
    *,
    block_size: int = BLOCK_SIZE,
) -> int:
    """Map one logical slot through a single-request 1D block table.

    Scalar counterpart of :func:`paged_slot_mapping`, which takes a ``[B, blocks]``
    table. Returns ``-1`` for an unmapped page.
    """
    block = slot // block_size
    intra = slot % block_size
    phys_block = int(table[block].item())
    if phys_block < 0:
        return -1
    return phys_block * block_size + intra


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
    return paged_slot_mapping(positions, table, block_size=block_size)


def paged_slot_mapping(
    positions: torch.Tensor,
    table: torch.Tensor,
    *,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """Map absolute positions to flattened physical rows; ``-1`` where unmapped."""
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
    still attend the current speculated tokens through their overlay raw-index
    range. The returned rows are packed oldest-to-newest; invalid tail columns
    are -1. The block table follows the same vLLM-style absolute logical block
    contract as SWA, while physical blocks may still be a small sliding-window
    ring.
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


# --- RoPE/YaRN table generation. ---


def _torch_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    normalized = dtype.lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    if normalized in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    raise ValueError(f"Unsupported RoPE table dtype: {dtype!r}")


def rope_profile_for_compress_ratio(config: Any, compress_ratio: int) -> tuple[float, int]:
    """Return ``(base_theta, original_seq_len)`` for the two DeepSeek-V4 RoPE profiles."""
    if compress_ratio:
        return float(config.compress_rope_theta), int(config.original_max_position_embeddings)
    return float(config.rope_theta), 0


def _linear_ramp_factor(low: int, high: int, dim: int, *, device: torch.device | None = None) -> torch.Tensor:
    if low == high:
        high = high + 0.001
    ramp = (torch.arange(dim, dtype=torch.float32, device=device) - low) / (high - low)
    return torch.clamp(ramp, 0, 1)


def _find_correction_dim(num_rotations: int, dim: int, base: float, max_seq_len: int) -> float:
    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def _find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float,
    max_seq_len: int,
) -> tuple[int, int]:
    low = math.floor(_find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(_find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def precompute_freqs_cos_sin(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
    *,
    dtype: torch.dtype | str = torch.bfloat16,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return real RoPE tables equivalent to ``model.py::precompute_freqs_cis``.

    The returned tensors are shaped ``[seqlen, dim]``.  The first half contains
    the mathematical ``cos(angle)`` / ``sin(angle)`` values; the second half is a
    duplicate so kernels can either read ``:dim//2`` directly or use ``j >> 1``
    frequency duplication over a full-width table.
    """
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"RoPE dim must be a positive even integer, got {dim}")
    if seqlen <= 0:
        raise ValueError(f"RoPE sequence length must be positive, got {seqlen}")

    out_dtype = _torch_dtype(dtype)
    out_device = torch.device(device) if device is not None else None
    half_dim = dim // 2

    inv_freq = 1.0 / (
        float(base) ** (torch.arange(0, dim, 2, dtype=torch.float32, device=out_device) / dim)
    )
    if original_seq_len > 0:
        low, high = _find_correction_range(beta_fast, beta_slow, dim, float(base), int(original_seq_len))
        smooth = 1 - _linear_ramp_factor(low, high, half_dim, device=out_device)
        inv_freq = inv_freq / float(factor) * (1 - smooth) + inv_freq * smooth

    positions = torch.arange(seqlen, dtype=torch.float32, device=out_device)
    angles = torch.outer(positions, inv_freq)
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    freqs_cos = torch.cat([cos_half, cos_half], dim=-1).to(out_dtype)
    freqs_sin = torch.cat([sin_half, sin_half], dim=-1).to(out_dtype)
    return freqs_cos, freqs_sin


def build_rope_tables(
    config: Any,
    compress_ratio: int,
    *,
    max_seq_len: int | None = None,
    rope_dim: int | None = None,
    dtype: torch.dtype | str = torch.bfloat16,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(freqs_cos, freqs_sin)`` shaped ``[max_seq_len, rope_dim]``."""
    base, original_seq_len = rope_profile_for_compress_ratio(config, compress_ratio)
    seq_len = int(max_seq_len if max_seq_len is not None else config.max_position_embeddings)
    dim = int(rope_dim if rope_dim is not None else config.qk_rope_head_dim)

    return precompute_freqs_cos_sin(
        dim,
        seq_len,
        original_seq_len,
        base,
        float(config.rope_factor),
        int(config.beta_fast),
        int(config.beta_slow),
        dtype=dtype,
        device=device,
    )


def materialize_token_rope_tables(
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather token-local RoPE tables using absolute ``position_ids``."""
    positions = position_ids.to(device=freqs_cos.device, dtype=torch.long).reshape(-1)
    return freqs_cos.index_select(0, positions).contiguous(), freqs_sin.index_select(0, positions).contiguous()


def materialize_half_rope_tables(
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather half-width FP32 cos/sin rows for decode submodule fixtures."""
    cos, sin = materialize_token_rope_tables(freqs_cos, freqs_sin, positions)
    half_dim = freqs_cos.shape[-1] // 2
    return cos[:, :half_dim].float().contiguous(), sin[:, :half_dim].float().contiguous()


# --- INT8 quantization. ---
def int8_quant_per_row(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row INT8 symmetric quant matching the runtime W8A8C16 activation path.

    Rounds to int8 through fp16 to match the device rounding; returns the
    per-row dequant scale (``1 / scale_quant``).
    """
    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


def quant_w_per_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel INT8 quant on the last axis."""
    amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.unsqueeze(-1)
    w_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


# ===========================================================================
# Phase A (Run 055): 1M-capable host reference fixtures.
#
# The helpers above are the baseline fixture path consumed by the not-yet-
# migrated SWA/HCA/CSA leaves; they keep their 16K defaults and modulo-based
# block tables so the baseline regression still passes (migration fence,
# run_055 plan §4.4). The section below adds the 1M-capable counterparts:
# ragged / per-request page maps, page permutation / rollover / stale-epoch /
# capacity-clamp fixtures, position 0..M.max_position_embeddings-1, and a token-local
# RoPE helper that does not materialize a full 1M cos/sin table per request.
# Phase B/C/D leaves will migrate onto these; until then both paths coexist.
# ===========================================================================


def validate_positions_1m(positions: torch.Tensor) -> torch.Tensor:
    """Validate that absolute positions fit ``[0, M.max_position_embeddings)``.

    Replaces the legacy ``_validate_starts`` clamp at
    ``FLASH.max_position_embeddings`` (16K) so length fixtures are not
    truncated by the migration-fence default.
    """
    if positions.numel() == 0:
        return positions
    p64 = positions.to(torch.int64)
    if bool((p64 < 0).any()):
        raise ValueError("positions must be non-negative")
    if bool((p64 >= M.max_position_embeddings).any()):
        raise ValueError(
            f"positions must be < M.max_position_embeddings={M.max_position_embeddings}"
        )
    return positions


def ragged_page_map(
    *,
    request_lengths: list[int],
    block_size: int = BLOCK_SIZE,
    physical_pool_blocks: int,
    permuted: bool = False,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a flat/ragged page map for heterogeneous-length requests.

    Returns ``(page_ids, request_page_offsets)`` where:
      - ``page_ids`` is a flat 1D tensor of physical page ids, one per logical
        page across all requests (no ``[B, max_logical_pages]`` dense table);
      - ``request_page_offsets`` is shape ``[B + 1]`` with the past-the-end
        offset of each request's page range in ``page_ids``.

    Physical page ids are assigned from a global pool of
    ``physical_pool_blocks`` and are never derived from ``B``. When
    ``permuted`` is True the assignment is a fixed permutation (seeded) so
    the logical->physical mapping is not identity, exercising the page-
    permutation / rollover fixture. Stale pages are *not* revived by modulo:
    a request only reads the pages the allocator actually assigned it.
    """
    page_counts = admit_ragged_page_counts(
        request_lengths,
        block_size=block_size,
        physical_pool_blocks=physical_pool_blocks,
    )
    total_pages = sum(page_counts)

    offsets = [0]
    page_ids: list[int] = []
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    allocation = (
        torch.randperm(physical_pool_blocks, generator=gen)[:total_pages].tolist()
        if permuted
        else list(range(total_pages))
    )
    cursor = 0
    for n_pages in page_counts:
        ids = allocation[cursor : cursor + n_pages]
        cursor += n_pages
        page_ids.extend(ids)
        offsets.append(len(page_ids))
    return (
        torch.tensor(page_ids, dtype=torch.int32),
        torch.tensor(offsets, dtype=torch.int32),
    )


def stale_epoch_page_map(
    *,
    request_length: int,
    block_size: int = BLOCK_SIZE,
    current_epoch: int,
    stale_epoch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A page map whose mapped physical pages all carry a stale epoch.

    Returns ``(page_ids, page_epochs, request_page_offsets)``. The lowering
    must reject the slot because ``page_epochs[0] != current_epoch``; the
    legacy modulo path would silently revive it.
    """
    n_pages = max(1, (request_length + block_size - 1) // block_size)
    page_ids = torch.tensor([7] * n_pages, dtype=torch.int32)
    page_epochs = torch.tensor([stale_epoch] * n_pages, dtype=torch.int32)
    offsets = torch.tensor([0, n_pages], dtype=torch.int32)
    _ = current_epoch  # validated by the lowering, not the fixture
    return page_ids, page_epochs, offsets


def capacity_clamped_request(
    *,
    committed_tokens: int,
    allocated_blocks: int,
    block_size: int = BLOCK_SIZE,
) -> tuple[int, int]:
    """Return ``(allocated_capacity, valid_end)`` for a capacity-clamped
    request: the allocator granted fewer pages than the committed history
    would fill, so visibility is clamped to the allocated capacity."""
    allocated_capacity = allocated_blocks * block_size
    valid_end = min(committed_tokens, allocated_capacity)
    return allocated_capacity, valid_end


def token_local_rope(
    config: Any,
    compress_ratio: int,
    position_ids: torch.Tensor,
    *,
    rope_dim: int | None = None,
    dtype: torch.dtype | str = torch.bfloat16,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute only the RoPE rows used by the active query set."""
    dim = int(rope_dim if rope_dim is not None else config.qk_rope_head_dim)
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"RoPE dim must be a positive even integer, got {dim}")
    validate_positions_1m(position_ids)
    out_dtype = _torch_dtype(dtype)
    out_device = torch.device(device) if device is not None else position_ids.device
    base, original_seq_len = rope_profile_for_compress_ratio(config, compress_ratio)
    half_dim = dim // 2
    inv_freq = 1.0 / (
        float(base)
        ** (torch.arange(0, dim, 2, dtype=torch.float32, device=out_device) / dim)
    )
    if original_seq_len > 0:
        low, high = _find_correction_range(
            int(config.beta_fast),
            int(config.beta_slow),
            dim,
            float(base),
            int(original_seq_len),
        )
        smooth = 1 - _linear_ramp_factor(low, high, half_dim, device=out_device)
        inv_freq = (
            inv_freq / float(config.rope_factor) * (1 - smooth)
            + inv_freq * smooth
        )
    positions = position_ids.to(device=out_device, dtype=torch.float32).reshape(-1)
    angles = torch.outer(positions, inv_freq)
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    return (
        torch.cat([cos_half, cos_half], dim=-1).to(out_dtype).contiguous(),
        torch.cat([sin_half, sin_half], dim=-1).to(out_dtype).contiguous(),
    )


def hetero_length_starts(
    lengths: list[int],
    *,
    seq: int = DECODE_SEQ,
) -> torch.Tensor:
    """Build per-request start positions for a heterogeneous-length batch.

    ``start[r] = lengths[r] - seq`` so each request's current decode chunk
    ends at ``lengths[r] - 1``. Lengths may span 1..M.max_position_embeddings and
    are not truncated by ``DECODE_BATCH``. Inactive requests use length 0
    (start 0, treated as inactive by the geometry layer).
    """
    starts = torch.tensor(
        hetero_length_starts_values(lengths, seq=seq), dtype=torch.int32
    )
    active_last_positions = torch.tensor(
        [n - 1 for n in lengths if n > 0], dtype=torch.int64
    )
    validate_positions_1m(active_last_positions)
    return starts


def one_m_tail_positions(*, seq: int = DECODE_SEQ) -> torch.Tensor:
    """Position ids for a single 1M-length request's final decode chunk."""
    if seq <= 0 or seq > M.max_position_embeddings:
        raise ValueError(f"seq must be in [1, {M.max_position_embeddings}], got {seq}")
    base = M.max_position_embeddings - seq
    return torch.arange(seq, dtype=torch.int32) + base


@dataclass(frozen=True)
class SwaRingFixture:
    """Token-local raw/SWA descriptors for a bounded four-page ring."""

    raw_page_ids: torch.Tensor
    raw_page_epochs: torch.Tensor
    raw_valid_ranges: torch.Tensor
    raw_page_counts: torch.Tensor
    request_epochs: torch.Tensor


def build_swa_ring(
    raw_valid_ranges: list[tuple[int, int]],
    *,
    request_epochs: list[int],
    page_ids: list[int | None] | None = None,
    page_epochs: list[int] | None = None,
    active: list[bool] | None = None,
    raw_page_counts: list[int] | None = None,
    physical_pool_pages: int | None = None,
    permuted: bool = False,
    seed: int = 0,
) -> SwaRingFixture:
    """Build bounded per-request raw/SWA ring descriptors.

    Every admitted request owns exactly four 32-row physical pages for its
    semantic 128-row ring.  The fixture deliberately does not build an
    absolute-position block table.  A missing request is represented by
    ``raw_page_counts[r] == 0`` and four ``-1`` page ids, which lets metadata
    fixtures exercise rejection without an out-of-bounds cache access.
    """
    request_count = len(raw_valid_ranges)
    if len(request_epochs) != request_count:
        raise ValueError("request_epochs must match raw_valid_ranges")
    if active is None:
        active = [True] * request_count
    if len(active) != request_count:
        raise ValueError("active must match raw_valid_ranges")
    if raw_page_counts is None:
        raw_page_counts = [
            SWA_PERSISTENT_PAGES_PER_REQUEST if is_active else 0
            for is_active in active
        ]
    if len(raw_page_counts) != request_count:
        raise ValueError("raw_page_counts must match raw_valid_ranges")
    if any(count not in (0, SWA_PERSISTENT_PAGES_PER_REQUEST) for count in raw_page_counts):
        raise ValueError("raw/SWA fixtures support zero or four pages per request")
    present = sum(raw_page_counts)
    if page_ids is not None and len(page_ids) != present:
        raise ValueError("page_ids must contain every admitted physical page")
    if page_epochs is not None and len(page_epochs) != present:
        raise ValueError("page_epochs must contain every admitted physical page")

    for begin, end in raw_valid_ranges:
        if begin < 0 or end < begin or end > M.max_position_embeddings:
            raise ValueError(f"raw valid range [{begin}, {end}) is invalid")
        if end - begin > SWA_PERSISTENT_ROWS_PER_REQUEST:
            raise ValueError(
                "raw/SWA range exceeds the semantic 128-row ring: "
                f"[{begin}, {end})"
            )
    if any(epoch < 0 for epoch in request_epochs):
        raise ValueError("request epochs must be non-negative")

    if page_ids is None:
        pool_pages = max(present, 1) if physical_pool_pages is None else physical_pool_pages
        if pool_pages < present:
            raise ValueError(
                f"physical pool has {pool_pages} pages for {present} requests"
            )
        allocation = torch.arange(pool_pages, dtype=torch.int32)
        if permuted and pool_pages > 1:
            generator = torch.Generator()
            generator.manual_seed(int(seed))
            allocation = torch.randperm(pool_pages, generator=generator).to(torch.int32)
            if present == 1 and int(allocation[0]) == 0:
                allocation = torch.roll(allocation, shifts=1)
        page_id_values = [int(value) for value in allocation[:present]]
    else:
        page_id_values = page_ids

    if page_epochs is None:
        page_epoch_values = [
            request_epochs[request]
            for request, count in enumerate(raw_page_counts)
            for _ in range(count)
        ]
    else:
        page_epoch_values = page_epochs
    raw_page_ids = torch.full(
        (request_count, SWA_PERSISTENT_PAGES_PER_REQUEST),
        SWA_SOURCE_INVALID,
        dtype=torch.int32,
    )
    raw_page_epochs = torch.full_like(raw_page_ids, SWA_SOURCE_INVALID)
    cursor = 0
    for request, count in enumerate(raw_page_counts):
        for relative_page in range(count):
            page_id = page_id_values[cursor]
            if page_id is not None:
                raw_page_ids[request, relative_page] = int(page_id)
            raw_page_epochs[request, relative_page] = int(page_epoch_values[cursor])
            cursor += 1
    return SwaRingFixture(
        raw_page_ids=raw_page_ids,
        raw_page_epochs=raw_page_epochs,
        raw_valid_ranges=torch.tensor(raw_valid_ranges, dtype=torch.int32),
        raw_page_counts=torch.tensor(raw_page_counts, dtype=torch.int32),
        request_epochs=torch.tensor(request_epochs, dtype=torch.int32),
    )


# Compatibility aliases for callers that imported the pre-block-32 helper.
SwaOnePageRingFixture = SwaRingFixture
build_swa_one_page_ring = build_swa_ring


def swa_request_query_offsets(
    query_request_ids: torch.Tensor,
    *,
    request_count: int,
) -> torch.Tensor:
    """Build and validate the packed request-to-query offset ABI.

    Queries must be grouped by request in ascending request order. Position
    contiguity is validated by the metadata host/device lowerings because it
    needs the corresponding position ids.
    """
    if request_count < 0:
        raise ValueError(f"request_count must be non-negative, got {request_count}")
    if query_request_ids.ndim != 1:
        raise ValueError("query_request_ids must have shape [Q]")
    ids = query_request_ids.to(dtype=torch.int64).tolist()
    offsets = [0]
    cursor = 0
    for request in range(request_count):
        while cursor < len(ids) and ids[cursor] == request:
            cursor += 1
        offsets.append(cursor)
    if cursor != len(ids):
        raise ValueError(
            "queries must be grouped by ascending request id within "
            "request_count"
        )
    return torch.tensor(offsets, dtype=torch.int32, device=query_request_ids.device)


def swa_token_local_rope(
    config: Any,
    position_ids: torch.Tensor,
    *,
    rope_dim: int | None = None,
    dtype: torch.dtype | str = torch.bfloat16,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate only the active SWA RoPE rows, never a context-length table."""
    return token_local_rope(
        config,
        0,
        position_ids,
        rope_dim=rope_dim,
        dtype=dtype,
        device=device,
    )


@dataclass(frozen=True)
class HcaStateRingFixture:
    """Sixteen-page descriptors for the semantic ratio-128 state ring."""

    state_page_ids: torch.Tensor
    state_page_epochs: torch.Tensor
    state_valid_ranges: torch.Tensor
    request_epochs: torch.Tensor


def build_hca_state_ring(
    state_valid_ranges: list[tuple[int, int]],
    *,
    request_epochs: list[int],
    page_ids: list[int] | None = None,
    page_epochs: list[int] | None = None,
    physical_pool_pages: int | None = None,
    permuted: bool = False,
    seed: int = 0,
) -> HcaStateRingFixture:
    """Build sixteen 8-row physical state pages per admitted request.

    The range is the pre-step partial compression block.  It is always shorter
    than 128 tokens; completed blocks live in HCA main KV instead.
    """
    request_count = len(state_valid_ranges)
    if len(request_epochs) != request_count:
        raise ValueError("request_epochs must match state_valid_ranges")
    physical_pages = request_count * HCA_STATE_PAGES_PER_REQUEST
    if page_epochs is not None and len(page_epochs) != physical_pages:
        raise ValueError("page_epochs must contain every HCA state page")
    for begin, end in state_valid_ranges:
        if not 0 <= begin <= end <= M.max_position_embeddings:
            raise ValueError(f"invalid HCA state range [{begin}, {end})")
        if end - begin >= HCA_STATE_ROWS_PER_REQUEST:
            raise ValueError("HCA state range must be shorter than one ratio-128 ring")
        if end > begin and begin // HCA_COMPRESS_RATIO != (end - 1) // HCA_COMPRESS_RATIO:
            raise ValueError("HCA state range may not cross a ratio-128 block")
    if page_ids is None:
        pool_pages = physical_pages if physical_pool_pages is None else physical_pool_pages
        if pool_pages < physical_pages:
            raise ValueError(
                f"HCA state pool has {pool_pages} pages for {physical_pages} required pages"
            )
        allocation = torch.arange(pool_pages, dtype=torch.int32)
        if permuted and pool_pages > 1:
            generator = torch.Generator().manual_seed(int(seed))
            allocation = torch.randperm(pool_pages, generator=generator).to(torch.int32)
        page_ids = [int(value) for value in allocation[:physical_pages]]
    if len(page_ids) != physical_pages or any(page < 0 for page in page_ids):
        raise ValueError("page_ids must contain every non-negative HCA state page")
    stored_epochs = (
        [epoch for epoch in request_epochs for _ in range(HCA_STATE_PAGES_PER_REQUEST)]
        if page_epochs is None
        else page_epochs
    )
    return HcaStateRingFixture(
        state_page_ids=torch.tensor(page_ids, dtype=torch.int32).reshape(
            request_count, HCA_STATE_PAGES_PER_REQUEST
        ),
        state_page_epochs=torch.tensor(stored_epochs, dtype=torch.int32).reshape(
            request_count, HCA_STATE_PAGES_PER_REQUEST
        ),
        state_valid_ranges=torch.tensor(state_valid_ranges, dtype=torch.int32),
        request_epochs=torch.tensor(request_epochs, dtype=torch.int32),
    )


def hca_event_rows(position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return current-step boundary query ids and compressed row ids."""
    validate_positions_1m(position_ids)
    flat = position_ids.to(torch.int64).reshape(-1)
    event_query_ids = torch.nonzero(
        (flat + 1).remainder(HCA_COMPRESS_RATIO) == 0,
        as_tuple=False,
    ).reshape(-1)
    return event_query_ids.to(torch.int32), (
        flat[event_query_ids] // HCA_COMPRESS_RATIO
    ).to(torch.int32)


def hca_query_local_rope(
    config: Any,
    position_ids: torch.Tensor,
    *,
    dtype: torch.dtype | str = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate only active query RoPE rows for HCA."""
    return token_local_rope(config, HCA_COMPRESS_RATIO, position_ids, dtype=dtype)


def hca_event_local_rope(
    config: Any,
    event_rows: torch.Tensor,
    *,
    dtype: torch.dtype | str = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate RoPE at each compression block's model-defined start row."""
    event_positions = event_rows.to(torch.int64) * HCA_COMPRESS_RATIO
    validate_positions_1m(event_positions)
    return token_local_rope(
        config,
        HCA_COMPRESS_RATIO,
        event_positions,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Phase D CSA fixtures. Main KV and index KV intentionally use independent
# physical page assignments while sharing one logical candidate window.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CsaPairedPageFixture:
    """Flat ragged descriptors for synchronized CSA main/index pools."""

    main_pages: torch.Tensor
    main_page_offsets: torch.Tensor
    main_windows: torch.Tensor
    index_pages: torch.Tensor
    index_page_offsets: torch.Tensor
    index_windows: torch.Tensor
    request_epochs: torch.Tensor


def _csa_page_counts_for_ranges(
    valid_ranges: list[tuple[int, int]],
    *,
    include_next_candidate: bool,
) -> list[int]:
    counts: list[int] = []
    for begin, end in valid_ranges:
        if not 0 <= begin <= end <= MAX_CSA_CANDIDATES:
            raise ValueError(f"invalid CSA candidate range [{begin}, {end})")
        capacity_end = end
        if include_next_candidate and end < MAX_CSA_CANDIDATES:
            capacity_end += 1
        if capacity_end <= begin:
            counts.append(0)
            continue
        first_page = begin // BLOCK_SIZE
        last_page = (capacity_end - 1) // BLOCK_SIZE
        counts.append(last_page - first_page + 1)
    return counts


def build_csa_paired_pages(
    valid_ranges: list[tuple[int, int]],
    *,
    request_epochs: list[int],
    main_page_ids: list[list[int]] | None = None,
    index_page_ids: list[list[int]] | None = None,
    main_page_epochs: list[list[int]] | None = None,
    index_page_epochs: list[list[int]] | None = None,
    main_heads: list[int] | None = None,
    index_heads: list[int] | None = None,
    physical_main_pages: int | None = None,
    physical_index_pages: int | None = None,
    include_next_candidate: bool = True,
    permuted: bool = False,
    seed: int = 0,
) -> CsaPairedPageFixture:
    """Build synchronized logical windows over independent ragged pools.

    The page count covers the declared live range and, by default, the next
    candidate write. Pool admission is aggregate across requests and is not
    multiplied by a fixed decode batch.
    """
    request_count = len(valid_ranges)
    if len(request_epochs) != request_count:
        raise ValueError("request_epochs must match valid_ranges")
    if any(epoch < 0 for epoch in request_epochs):
        raise ValueError("request epochs must be non-negative")
    counts = _csa_page_counts_for_ranges(
        valid_ranges,
        include_next_candidate=include_next_candidate,
    )
    total_pages = sum(counts)
    if total_pages > CSA_MAIN_PAGES_AT_1M * max(request_count, 1):
        raise ValueError("CSA fixture exceeds the per-request 1M page ceiling")

    def allocate(
        explicit: list[list[int]] | None,
        pool_pages: int | None,
        allocation_seed: int,
    ) -> list[list[int]]:
        if explicit is not None:
            if len(explicit) != request_count:
                raise ValueError("explicit CSA page ids must match valid_ranges")
            if any(len(ids) != count for ids, count in zip(explicit, counts)):
                raise ValueError("explicit CSA page spans do not match required counts")
            return [[int(page) for page in ids] for ids in explicit]
        pool = CSA_MAIN_PAGES_AT_1M if pool_pages is None else int(pool_pages)
        if pool < total_pages:
            raise ValueError(
                f"CSA pool has {pool} pages but the ragged batch needs {total_pages}"
            )
        allocation = torch.arange(pool, dtype=torch.int32)
        if permuted and pool > 1:
            generator = torch.Generator().manual_seed(int(allocation_seed))
            allocation = torch.randperm(pool, generator=generator).to(torch.int32)
        result: list[list[int]] = []
        cursor = 0
        for count in counts:
            result.append([int(v) for v in allocation[cursor : cursor + count]])
            cursor += count
        return result

    main_ids = allocate(main_page_ids, physical_main_pages, seed)
    index_ids = allocate(index_page_ids, physical_index_pages, seed + 97)
    if main_heads is None:
        main_heads = [0] * request_count
    if index_heads is None:
        index_heads = [0] * request_count
    if len(main_heads) != request_count or len(index_heads) != request_count:
        raise ValueError("CSA heads must match valid_ranges")
    for count, main_head, index_head in zip(counts, main_heads, index_heads):
        if count == 0:
            if main_head != 0 or index_head != 0:
                raise ValueError("empty CSA page spans require head=0")
        elif not (0 <= main_head < count and 0 <= index_head < count):
            raise ValueError("CSA page head is outside its ragged span")

    def epochs_or_default(explicit: list[list[int]] | None) -> list[list[int]]:
        if explicit is None:
            return [[request_epochs[r]] * counts[r] for r in range(request_count)]
        if len(explicit) != request_count:
            raise ValueError("CSA page epochs must match valid_ranges")
        if any(len(values) != count for values, count in zip(explicit, counts)):
            raise ValueError("CSA page epoch spans do not match page ids")
        return [[int(value) for value in values] for values in explicit]

    main_epochs = epochs_or_default(main_page_epochs)
    index_epochs = epochs_or_default(index_page_epochs)

    def pack(ids: list[list[int]], epochs: list[list[int]]):
        rows: list[tuple[int, int]] = []
        offsets = [0]
        for page_ids, page_epochs in zip(ids, epochs):
            rows.extend(zip(page_ids, page_epochs))
            offsets.append(len(rows))
        return (
            torch.tensor(rows, dtype=torch.int32).reshape(-1, 2),
            torch.tensor(offsets, dtype=torch.int32),
        )

    main_pages, main_offsets = pack(main_ids, main_epochs)
    index_pages, index_offsets = pack(index_ids, index_epochs)
    return CsaPairedPageFixture(
        main_pages=main_pages,
        main_page_offsets=main_offsets,
        main_windows=torch.tensor(
            [
                [begin, end, int(head)]
                for (begin, end), head in zip(valid_ranges, main_heads)
            ],
            dtype=torch.int32,
        ),
        index_pages=index_pages,
        index_page_offsets=index_offsets,
        index_windows=torch.tensor(
            [
                [begin, end, int(head)]
                for (begin, end), head in zip(valid_ranges, index_heads)
            ],
            dtype=torch.int32,
        ),
        request_epochs=torch.tensor(request_epochs, dtype=torch.int32),
    )


@dataclass(frozen=True)
class CsaStateRingsFixture:
    """Independent four-page main and inner ratio-4 state descriptors."""

    main_state_pages: torch.Tensor
    main_state_valid_ranges: torch.Tensor
    inner_state_pages: torch.Tensor
    inner_state_valid_ranges: torch.Tensor


def build_csa_state_rings(
    main_valid_ranges: list[tuple[int, int]],
    inner_valid_ranges: list[tuple[int, int]],
    *,
    request_epochs: list[int],
    main_page_ids: list[int] | None = None,
    inner_page_ids: list[int] | None = None,
    main_page_epochs: list[int] | None = None,
    inner_page_epochs: list[int] | None = None,
    physical_main_pages: int | None = None,
    physical_inner_pages: int | None = None,
    permuted: bool = False,
    seed: int = 0,
) -> CsaStateRingsFixture:
    """Build two independent eight-row state rings for every request."""
    request_count = len(request_epochs)
    if len(main_valid_ranges) != request_count or len(inner_valid_ranges) != request_count:
        raise ValueError("CSA state valid ranges must match request_epochs")
    if CSA_STATE_ROWS_PER_REQUEST != 8 or CSA_INNER_STATE_ROWS_PER_REQUEST != 8:
        raise AssertionError("Phase D requires two eight-row state rings")

    def validate_ranges(ranges: list[tuple[int, int]], name: str) -> None:
        for begin, end in ranges:
            if not 0 <= begin <= end <= M.max_position_embeddings:
                raise ValueError(f"invalid {name} state range [{begin}, {end})")
            if end - begin > CSA_STATE_ROWS_PER_REQUEST:
                raise ValueError(f"{name} state range exceeds the eight-row ring")

    validate_ranges(main_valid_ranges, "main")
    validate_ranges(inner_valid_ranges, "inner")

    def allocate(
        explicit: list[int] | None,
        pool_pages: int | None,
        allocation_seed: int,
        default_pool_pages: int,
        pages_per_request: int,
    ) -> list[int]:
        required_pages = request_count * pages_per_request
        if explicit is not None:
            if len(explicit) != required_pages or any(page < 0 for page in explicit):
                raise ValueError("CSA state page ids must contain every physical ring page")
            return [int(page) for page in explicit]
        pool = default_pool_pages if pool_pages is None else int(pool_pages)
        if pool < required_pages:
            raise ValueError(
                f"CSA state pool has {pool} pages for {required_pages} required pages"
            )
        allocation = torch.arange(pool, dtype=torch.int32)
        if permuted and pool > 1:
            generator = torch.Generator().manual_seed(int(allocation_seed))
            allocation = torch.randperm(pool, generator=generator).to(torch.int32)
        return [int(value) for value in allocation[:required_pages]]

    main_ids = allocate(
        main_page_ids,
        physical_main_pages,
        seed,
        CSA_STATE_POOL_PAGES,
        CSA_STATE_PAGES_PER_REQUEST,
    )
    inner_ids = allocate(
        inner_page_ids,
        physical_inner_pages,
        seed + 193,
        CSA_INNER_STATE_POOL_PAGES,
        CSA_INNER_STATE_PAGES_PER_REQUEST,
    )
    main_epochs = (
        [epoch for epoch in request_epochs for _ in range(CSA_STATE_PAGES_PER_REQUEST)]
        if main_page_epochs is None
        else main_page_epochs
    )
    inner_epochs = (
        [epoch for epoch in request_epochs for _ in range(CSA_INNER_STATE_PAGES_PER_REQUEST)]
        if inner_page_epochs is None
        else inner_page_epochs
    )
    if len(main_epochs) != len(main_ids) or len(inner_epochs) != len(inner_ids):
        raise ValueError("CSA state page epochs must match physical page counts")
    return CsaStateRingsFixture(
        main_state_pages=torch.tensor(
            list(zip(main_ids, main_epochs)), dtype=torch.int32
        ).reshape(request_count, CSA_STATE_PAGES_PER_REQUEST, 2),
        main_state_valid_ranges=torch.tensor(main_valid_ranges, dtype=torch.int32),
        inner_state_pages=torch.tensor(
            list(zip(inner_ids, inner_epochs)), dtype=torch.int32
        ).reshape(request_count, CSA_INNER_STATE_PAGES_PER_REQUEST, 2),
        inner_state_valid_ranges=torch.tensor(inner_valid_ranges, dtype=torch.int32),
    )


def csa_event_rows(position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return query ids and logical rows for active ratio-4 boundaries."""
    validate_positions_1m(position_ids)
    flat = position_ids.to(torch.int64).reshape(-1)
    query_ids = torch.nonzero(
        (flat + 1).remainder(CSA_COMPRESS_RATIO) == 0,
        as_tuple=False,
    ).reshape(-1)
    return query_ids.to(torch.int32), (
        flat[query_ids] // CSA_COMPRESS_RATIO
    ).to(torch.int32)


def csa_query_local_rope(
    config: Any,
    position_ids: torch.Tensor,
    *,
    dtype: torch.dtype | str = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one RoPE row for every active CSA index query."""
    return token_local_rope(config, CSA_COMPRESS_RATIO, position_ids, dtype=dtype)


def csa_event_local_rope(
    config: Any,
    event_rows: torch.Tensor,
    *,
    dtype: torch.dtype | str = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate main/inner compressor RoPE at ratio-4 block starts."""
    starts = event_rows.to(torch.int64) * CSA_COMPRESS_RATIO
    validate_positions_1m(starts)
    return token_local_rope(config, CSA_COMPRESS_RATIO, starts, dtype=dtype)


def csa_candidate_boundary_counts() -> tuple[int, ...]:
    """Canonical candidate-count surface for Phase D metadata/Top-K tests."""
    return (
        0,
        1,
        511,
        512,
        513,
        2047,
        2048,
        2049,
        3072,
        4096,
        4097,
        16383,
        16384,
        MAX_CSA_CANDIDATES - 1,
        MAX_CSA_CANDIDATES,
    )


def csa_invalid_topk_rows(query_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return canonical all-invalid fixed-width root rows for zero-leaf queries."""
    if query_count < 0:
        raise ValueError("query_count must be non-negative")
    return (
        torch.full((query_count, CSA_TOPK), float("-inf"), dtype=torch.float32),
        torch.full(
            (query_count, CSA_TOPK),
            CSA_TOPK_INVALID_INDEX,
            dtype=torch.int32,
        ),
    )
