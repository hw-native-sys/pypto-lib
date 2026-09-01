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

import math
from typing import Any, Callable

import torch

from config import (
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    C128_COMPRESSOR_BLOCK_SIZE,
    DECODE_BATCH,
    DECODE_SEQ,
    DECODE_START_POS,
    FLASH as M,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
)


# --- Paged-KV metadata lowering. ---
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
