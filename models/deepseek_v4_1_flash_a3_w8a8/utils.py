# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host utilities: NZ packing, INT8 references, and read-only checkpoint access."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Mapping

import torch

from config import INT8_AMAX_EPS, INT8_SCALE_MAX


NZ_FRACTAL_ROWS = 16
NZ_C0_BYTES = 32


def compressed_source_positions(position_ids: torch.Tensor, ratio: int) -> torch.Tensor:
    """Absolute source positions: C1 token positions or completed C2 pair origins."""
    positions = position_ids.long()
    if ratio == 1:
        return positions.clamp_min(0)
    if ratio != 2:
        raise ValueError(f"unsupported compression ratio {ratio}")
    complete = (positions >= 0) & (positions.remainder(2) == 1)
    return torch.where(complete, positions - 1, torch.zeros_like(positions))


def pack_nz(logical: torch.Tensor) -> torch.Tensor:
    """Reorder the trailing ``[R, C]`` of a row-major tensor into NZ fractal order."""
    rows, cols = logical.shape[-2:]
    c0 = NZ_C0_BYTES // logical.element_size()
    if rows % NZ_FRACTAL_ROWS or cols % c0:
        raise ValueError(f"NZ needs rows%{NZ_FRACTAL_ROWS}==0 and cols%{c0}==0, got {tuple(logical.shape)}")
    blocked = logical.reshape(-1, rows // NZ_FRACTAL_ROWS, NZ_FRACTAL_ROWS, cols // c0, c0)
    return blocked.permute(0, 3, 1, 2, 4).contiguous().reshape(logical.shape)


def unpack_nz(packed: torch.Tensor) -> torch.Tensor:
    """Restore NZ-ordered bytes to the logical row-major ``[..., R, C]`` tensor."""
    rows, cols = packed.shape[-2:]
    c0 = NZ_C0_BYTES // packed.element_size()
    blocked = packed.reshape(-1, cols // c0, rows // NZ_FRACTAL_ROWS, NZ_FRACTAL_ROWS, c0)
    return blocked.permute(0, 2, 3, 1, 4).contiguous().reshape(packed.shape)


def int8_quant_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token INT8 symmetric quant matching the device W8A8 activation path."""
    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = torch.div(torch.full_like(amax, INT8_SCALE_MAX), amax)
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


def quant_w_per_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel INT8 quant of a logical ``[N, K]`` weight."""
    if w.dim() != 2:
        raise ValueError(f"weight must be 2-D, got {tuple(w.shape)}")
    amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.unsqueeze(-1)
    w_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


def dequant_w_per_channel(w_i8: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor | None = None) -> torch.Tensor:
    """Logical FP32 weight ``(q - offset) * scale`` with an optional per-channel offset."""
    value = w_i8.float()
    if offset is not None:
        value = value - offset.float().reshape(-1, 1)
    return value * scale.float().reshape(-1, 1)


class SafeTensorReader:
    """Read tensors from safetensors shards without the ``safetensors`` package."""

    _DTYPES = {
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I8": torch.int8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "U8": torch.uint8,
        "F8_E4M3": torch.uint8,
        "F8_E8M0": torch.uint8,
    }

    def __init__(self, root: str | Path):
        self.root = Path(root)
        index_path = self.root / "quant_model_weights.safetensors.index.json"
        self.weight_map: Mapping[str, str] = json.loads(index_path.read_text())["weight_map"]
        self._headers: dict[str, tuple[Path, int, dict]] = {}

    def _header(self, name: str) -> tuple[Path, int, dict]:
        shard = self.root / self.weight_map[name]
        key = str(shard)
        if key not in self._headers:
            with shard.open("rb") as handle:
                header_size = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(header_size))
            self._headers[key] = (shard, header_size, header)
        return self._headers[key]

    def header(self, name: str) -> tuple[Path, int, dict]:
        """Return the shard path, JSON header size, and parsed header for ``name``."""
        return self._header(name)

    def shape_and_dtype(self, name: str) -> tuple[tuple[int, ...], torch.dtype]:
        _, _, header = self._header(name)
        info = header[name]
        if info["dtype"] not in self._DTYPES:
            raise ValueError(f"unsupported safetensors dtype {info['dtype']} for {name}")
        return tuple(info["shape"]), self._DTYPES[info["dtype"]]

    def nbytes(self, name: str) -> int:
        _, _, header = self._header(name)
        begin, end = header[name]["data_offsets"]
        return end - begin

    def tensor(self, name: str) -> torch.Tensor:
        shard, header_size, header = self._header(name)
        info = header[name]
        begin, end = info["data_offsets"]
        dtype = self._DTYPES[info["dtype"]]
        with shard.open("rb") as handle:
            handle.seek(8 + header_size + begin)
            raw = handle.read(end - begin)
        value = torch.frombuffer(bytearray(raw), dtype=dtype)
        return value.reshape(info["shape"])


def load_quant_description(root: str | Path) -> dict[str, str]:
    """Return the per-tensor quantization tag map (without the ``optional`` entry)."""
    path = Path(root) / "quant_model_description.json"
    description = json.loads(path.read_text())
    description.pop("optional", None)
    return description


def load_embedding_storage(root: str | Path) -> dict:
    """Return the Engram ``embedding_storage`` contract."""
    path = Path(root) / "quant_model_description.json"
    return json.loads(path.read_text())["optional"]["embedding_storage"]


# ---------------------------------------------------------------------------
# Paged batch metadata (host side of the A5-shaped dynamic-batch interface)
# ---------------------------------------------------------------------------


def paged_slots(
    positions: torch.Tensor,
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    logical_divisor: int = 1,
    publish_only_complete: bool = False,
) -> torch.Tensor:
    """Map logical positions to flattened physical cache rows (``-1`` invalid).

    ``logical_divisor`` is the compression ratio of the cache (1 for the raw
    window cache, 2 for C2A compressed/index rows).  With
    ``publish_only_complete`` only positions whose compressed row is complete
    (``position % ratio == ratio - 1``) receive a slot.
    """
    if positions.ndim != 1 or request_ids.shape != positions.shape or block_table.ndim != 2:
        raise ValueError("positions/request_ids must be matching vectors and block_table a matrix")
    if logical_divisor <= 0:
        raise ValueError("logical_divisor must be positive")
    publish = torch.ones_like(positions, dtype=torch.bool)
    if publish_only_complete and logical_divisor > 1:
        publish = positions.remainder(logical_divisor) == logical_divisor - 1
    slots = torch.full_like(positions, -1, dtype=torch.int64)
    logical = torch.div(positions[publish], logical_divisor, rounding_mode="floor")
    logical_page = torch.div(logical, 128, rounding_mode="floor")
    offset = logical.remainder(128)
    requests = request_ids[publish].to(torch.long)
    if bool(((requests < 0) | (requests >= block_table.shape[0])).any()):
        raise ValueError("published request id is outside the block table")
    if bool(((logical_page < 0) | (logical_page >= block_table.shape[1])).any()):
        raise ValueError("block table does not cover a required logical page")
    physical = block_table[requests, logical_page.to(torch.long)]
    if bool((physical < 0).any()):
        raise ValueError("required logical page has no physical allocation")
    slots[publish] = physical.to(torch.int64) * 128 + offset.to(torch.int64)
    return slots


def window_metadata(
    positions: torch.Tensor,
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    window: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Paged write slots plus physical row ids of the causal sliding window.

    Lane ``i`` of the returned ``[T, window]`` matrix holds the physical row of
    absolute position ``position - lens + 1 + i`` (oldest first); lanes past the
    visible prefix are ``-1``.
    """
    slots = paged_slots(positions, request_ids, block_table)
    lens = torch.minimum(positions + 1, torch.full_like(positions, window)).to(torch.int32)
    offsets = torch.arange(window, device=positions.device)
    starts = positions - lens.to(positions.dtype) + 1
    visible = starts.unsqueeze(-1) + offsets
    valid = offsets.unsqueeze(0) < lens.unsqueeze(-1)
    indices = torch.full_like(visible, -1, dtype=torch.int64)
    visible_requests = request_ids.unsqueeze(-1).expand_as(visible)
    indices[valid] = paged_slots(visible[valid], visible_requests[valid], block_table)
    return slots, indices.to(torch.int32), lens


def compressed_metadata(
    positions: torch.Tensor,
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paged write slots and visible lengths of emitted compressed rows."""
    slots = paged_slots(positions, request_ids, block_table, ratio, publish_only_complete=True)
    lens = torch.div(positions.to(torch.int64) + 1, ratio, rounding_mode="floor").to(torch.int32)
    return slots, lens


def build_request_plan(
    requests: list[tuple[int, int]],
    ratio: int = 1,
    window: int = 128,
) -> dict:
    """Build the paged metadata for a batch of ``(start_position, query_length)``.

    The physical page allocator hands out one page per logical page of the
    longest request; the same table addresses the window, compressed and index
    caches of every source, so the returned page counts size all three pools.
    ``state_block_table`` assigns one compressor-state block per request.
    """
    if not requests:
        raise ValueError("a request plan needs at least one request")
    if any(length <= 0 or start < 0 for start, length in requests):
        raise ValueError("request positions must be non-negative and lengths positive")
    max_position = max(start + length - 1 for start, length in requests)
    table_width = (max_position + 1 + 127) // 128
    batch = len(requests)
    table = torch.full((batch, table_width), -1, dtype=torch.int32)
    next_page = 0
    positions = []
    request_rows = []
    for index, (start, length) in enumerate(requests):
        pages = (start + length + 127) // 128
        table[index, :pages] = torch.arange(next_page, next_page + pages, dtype=torch.int32)
        next_page += pages
        positions.append(torch.arange(start, start + length, dtype=torch.int32))
        request_rows.append(torch.full((length,), index, dtype=torch.int32))
    positions = torch.cat(positions)
    request_ids = torch.cat(request_rows)
    window_slots, window_indices, window_lens = window_metadata(positions, request_ids, table, window)
    compressed_slots, compressed_lens = compressed_metadata(positions, request_ids, table, ratio)
    state_table = torch.arange(batch, dtype=torch.int32).reshape(batch, 1)
    return {
        "position_ids": positions,
        "request_ids": request_ids,
        "index_block_table": table,
        "state_block_table": state_table,
        "window_slots": window_slots,
        "window_indices": window_indices,
        "window_lens": window_lens,
        "compressed_slots": compressed_slots,
        "compressed_lens": compressed_lens,
        "table_width": table_width,
        "pages": next_page,
        "tokens": int(positions.numel()),
    }


def gather_cache_rows(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Gather flattened cache rows by physical slot for goldens; ``-1`` yields zeros."""
    if cache.shape[2] != 1:
        raise ValueError("cache must have the [blocks, 128, 1, width] layout")
    flat = cache.reshape(-1, cache.shape[-1])
    valid = slots >= 0
    rows = torch.zeros((slots.numel(), flat.shape[-1]), dtype=cache.dtype)
    rows[valid] = flat[slots[valid].to(torch.long)]
    return rows


def scatter_cache_rows(cache: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Scatter flat rows into cache slots for goldens; ``-1`` slots are skipped."""
    flat = cache.reshape(-1, cache.shape[-1])
    valid = slots >= 0
    flat[slots[valid].to(torch.long)] = rows[valid].to(cache.dtype)
    return cache
