# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare prefill inputs and bind metadata to caller-owned cache storage."""

import logging
from math import prod
import torch
from .prefill_ops.prefill_shapes import DECODE_SEQ as PREFILL_STATE_ROWS
from dataclasses import dataclass


RANKS = 8
CP_SIZE = 8

# name: (layers, decode rows per page, prefill rows per page, trailing shape, dtype)
CACHE_LAYOUTS = {
    "kv_cache": (43, 128, 32, (1, 512), torch.bfloat16),
    "hca_cmp_kv": (20, 1, 1, (1, 512), torch.bfloat16),
    "csa_cmp_kv": (21, 32, 32, (1, 512), torch.bfloat16),
    "idx_kv_cache": (21, 32, 32, (1, 128), torch.int8),
    "idx_kv_scale": (21, 32, 32, (1, 1), torch.float32),
    "hca_compress_state": (20, 8, 8, (1024,), torch.float32),
    "csa_compress_state": (21, 4, 2, (2048,), torch.float32),
    "csa_inner_compress_state": (21, 4, 2, (512,), torch.float32),
}

# prefill table name: (decode table name, subpages per decode page, prefill depth)
TABLE_LAYOUTS = {
    "ori_block_table": ("block_table", 4, 32768),
    "hca_cmp_block_table": ("hca_cmp_block_table", 1, 8192),
    "csa_cmp_block_table": ("csa_cmp_block_table", 1, 8192),
    "idx_block_table": ("idx_block_table", 1, 8192),
    "hca_compress_state_block_table": ("hca_compress_state_block_table", 1, 131072),
    "csa_compress_state_block_table": ("csa_compress_state_block_table", 2, 524288),
    "csa_inner_compress_state_block_table": ("csa_inner_compress_state_block_table", 2, 524288),
}

_COUNT_GROUPS = {
    "hca_cmp_block_table": 1,
    "csa_cmp_block_table": 2,
    "idx_block_table": 3,
    "hca_compress_state_block_table": 4,
    "csa_compress_state_block_table": 5,
    "csa_inner_compress_state_block_table": 6,
}


def _page_counts(block_counts):
    if block_counts is None:
        return {}
    if (not isinstance(block_counts, torch.Tensor) or block_counts.dtype != torch.int32
            or tuple(block_counts.shape) != (RANKS, 1, 7)):
        raise ValueError("block_counts must be an INT32 [8, 1, 7] tensor")
    if bool((block_counts < 0).any()):
        raise ValueError("block_counts cannot be negative")
    return {name: block_counts[:, 0, index].to(torch.int64) for name, index in _COUNT_GROUPS.items()}


def _cyclic_pages(table, counts, length):
    if bool((counts > table.shape[-1]).any()):
        raise ValueError("A cache page count exceeds its table")
    logical = torch.arange(length, dtype=torch.int64, device=table.device).unsqueeze(0)
    physical_index = logical.remainder(counts.clamp(min=1).unsqueeze(1))
    pages = torch.gather(table[:, 0, :], 1, physical_index)
    return torch.where(counts.unsqueeze(1) > 0, pages, -1).unsqueeze(1)


def _cache_shape(name, shape, dtype):
    layers, decode_page, prefill_page, tail, expected_dtype = CACHE_LAYOUTS[name]
    shape = tuple(shape)
    if len(shape) != 3 + len(tail) or shape[0] != RANKS or shape[2:] != (decode_page, *tail):
        raise ValueError(f"{name} has an incompatible decode cache shape: {shape}")
    if shape[1] <= 0 or shape[1] % layers or dtype != expected_dtype:
        raise ValueError(f"{name} has an incompatible layer count or dtype")
    return (RANKS, shape[1] * decode_page // prefill_page, prefill_page, *tail)


def prefill_cache_views(caches, *, worker=None):
    """Return prefill-shaped views of the same owning decode cache allocations.

    The original buffers must remain live until both programs finish. A view is
    never an independent allocation and must not be freed separately.
    """
    views = {}
    for name in CACHE_LAYOUTS:
        root = caches[name]
        shape = _cache_shape(name, root.shape, root.dtype)
        if isinstance(root, torch.Tensor):
            if not root.is_contiguous():
                raise ValueError(f"{name} must have contiguous storage")
            views[name] = root.view(shape)
            continue
        from pypto.runtime import StackedDeviceTensor

        if not isinstance(root, StackedDeviceTensor) or root.worker_ids != tuple(range(RANKS)):
            raise TypeError(f"{name} must be a rank-ordered StackedDeviceTensor")
        if worker is None or not callable(getattr(worker, "view_tensor", None)):
            raise TypeError("Device cache views require their owning Worker's view_tensor API")
        shards = []
        for rank, shard in enumerate(root.shards):
            alias = worker.view_tensor(shard, shape[1:], worker_id=rank)
            if alias.nbytes != shard.nbytes:
                raise ValueError(f"{name} view does not cover the original allocation")
            shards.append(alias)
        views[name] = StackedDeviceTensor(shards, shape, root.worker_ids)
    return views


def initialize_cache_pages(worker, name, root, pages_by_rank):
    """Initialize only newly owned pages of one decode cache family.

    The caller supplies pages exclusively owned by this request. Shared prefix
    pages and retained pages from a previous prefill chunk must not be listed.
    The distributed Worker's copy_to stages non-inherited CPU buffers.
    """
    if name not in CACHE_LAYOUTS:
        raise ValueError(f"Unknown cache family: {name}")
    _cache_shape(name, root.shape, root.dtype)
    if tuple(root.worker_ids) != tuple(range(RANKS)) or len(pages_by_rank) != RANKS:
        raise ValueError("Cache initialization requires eight rank-local page lists")
    layers, page_rows, _, tail, dtype = CACHE_LAYOUTS[name]
    blocks_per_layer = root.shape[1] // layers
    selected = []
    for pages in pages_by_rank:
        pages = list(pages)
        if (any(type(page) is not int or not 0 <= page < blocks_per_layer for page in pages)
                or len(pages) != len(set(pages))):
            raise ValueError("Owned cache pages must be unique and within the allocation")
        selected.append(sorted(pages))
    for rank, pages in enumerate(selected):
        if not pages:
            continue
        start = previous = pages[0]
        runs = []
        for page in pages[1:]:
            if page != previous + 1:
                runs.append((start, previous + 1))
                start = page
            previous = page
        runs.append((start, previous + 1))
        page_bytes = prod((page_rows, *tail)) * torch.empty((), dtype=dtype).element_size()
        pages_per_copy = max(1, (64 * 1024 * 1024) // page_bytes)
        for run_first, run_end in runs:
            for first in range(run_first, run_end, pages_per_copy):
                count = min(pages_per_copy, run_end - first)
                host = torch.zeros((count, page_rows, *tail), dtype=dtype)
                if name.endswith("compress_state"):
                    host[..., host.shape[-1] // 2 :] = torch.finfo(torch.float32).min
                for layer in range(layers):
                    offset = (layer * blocks_per_layer + first) * page_bytes
                    worker.copy_to(root.shards[rank].data_ptr, host.data_ptr(), count * page_bytes,
                                   dst_offset=offset, worker_id=rank)


def _table(table, name):
    if not isinstance(table, torch.Tensor) or table.dtype != torch.int32:
        raise TypeError(f"{name} must be an INT32 tensor")
    if table.ndim != 3 or tuple(table.shape[:2]) != (RANKS, 1) or table.shape[-1] == 0:
        raise ValueError(f"{name} must have shape [8, 1, pages]")
    if bool((table < -1).any()):
        raise ValueError(f"{name} contains an invalid page id")
    return table


def prefill_block_tables(decode_tables, *, block_counts=None):
    """Split decode pages into prefill pages without changing physical rows."""
    result = {}
    counts = _page_counts(block_counts)
    for name, (decode_name, factor, depth) in TABLE_LAYOUTS.items():
        table = _table(decode_tables[decode_name], decode_name)
        if decode_name in counts:
            table = _cyclic_pages(table, counts[decode_name], depth // factor)
        if table.shape[-1] * factor > depth:
            raise ValueError(f"{decode_name} exceeds the prefill table depth")
        if factor != 1:
            maximum = int(table.max())
            if maximum > (torch.iinfo(torch.int32).max - factor + 1) // factor:
                raise ValueError(f"{decode_name} page ids overflow INT32")
            offsets = torch.arange(factor, dtype=torch.int32, device=table.device)
            expanded = table.unsqueeze(-1) * factor + offsets
            table = torch.where(table.unsqueeze(-1) >= 0, expanded, -1).reshape(RANKS, 1, -1)
        if table.shape[-1] != depth:
            padded = torch.full((RANKS, 1, depth), -1, dtype=torch.int32, device=table.device)
            padded[:, :, : table.shape[-1]] = table
            table = padded
        result[name] = table
    return result


def _physical_rows(table, positions, page_tokens, rows_per_page, compression=1, counts=None):
    valid = positions >= 0
    logical_page = positions.clamp(min=0).to(torch.int64) // page_tokens
    if counts is not None:
        if bool((counts > table.shape[-1]).any()):
            raise ValueError("A cache page count exceeds its table")
        logical_page = logical_page.remainder(counts.clamp(min=1).unsqueeze(1))
    if bool((logical_page[valid] >= table.shape[-1]).any()):
        raise ValueError("A prefill position exceeds its cache table")
    page = torch.gather(table[:, 0, :], 1, logical_page.clamp(max=table.shape[-1] - 1))
    if counts is not None:
        page = torch.where(counts.unsqueeze(1) > 0, page, -1)
    if bool((page[valid] < 0).any()):
        raise ValueError("An active prefill position has no cache page")
    row = page.to(torch.int64) * rows_per_page + positions.clamp(min=0) % page_tokens // compression
    return torch.where(valid, row, -1)


def _check_rows(name, rows, cache, cache_name):
    _cache_shape(cache_name, cache.shape, cache.dtype)
    layers, page_rows, _, _, _ = CACHE_LAYOUTS[cache_name]
    capacity = cache.shape[1] // layers * page_rows
    if bool((rows >= capacity).any()):
        raise ValueError(f"{name} exceeds the {cache_name} allocation")


def prefill_slot_mappings(positions, decode_tables, caches, *, block_counts=None):
    """Map absolute positions to the shared physical rows; -1 marks padding."""
    if not isinstance(positions, torch.Tensor) or positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("positions must be an integer tensor")
    if positions.ndim != 2 or positions.shape[0] != RANKS:
        raise ValueError("positions must have shape [8, tokens]")
    if bool((positions < -1).any()):
        raise ValueError("Only -1 may mark a padded position")
    if not torch.equal(positions, positions[:1].expand(RANKS, -1)):
        raise ValueError("All prefill ranks must process the same positions")
    valid = positions >= 0
    if bool((valid[:, 1:] & ~valid[:, :-1]).any()):
        raise ValueError("Padding must follow every active position")
    for name in {item[0] for item in TABLE_LAYOUTS.values()}:
        _table(decode_tables[name], name)
    counts = _page_counts(block_counts)
    mapping = {
        "ori_slot_mapping_full": _physical_rows(decode_tables["block_table"], positions, 128, 128),
        "hca_state_slot_mapping_full": _physical_rows(
            decode_tables["hca_compress_state_block_table"], positions, 8, 8,
            counts=counts.get("hca_compress_state_block_table")
        ),
        "csa_state_slot_mapping_full": _physical_rows(
            decode_tables["csa_compress_state_block_table"], positions, 4, 4,
            counts=counts.get("csa_compress_state_block_table")
        ),
        "csa_inner_state_slot_mapping_full": _physical_rows(
            decode_tables["csa_inner_compress_state_block_table"], positions, 4, 4,
            counts=counts.get("csa_inner_compress_state_block_table")
        ),
    }
    for name, table_name, ratio, page_rows in (
        ("hca_cmp_slot_mapping_full", "hca_cmp_block_table", 128, 1),
        ("csa_cmp_slot_mapping_full", "csa_cmp_block_table", 4, 32),
        ("csa_idx_slot_mapping_full", "idx_block_table", 4, 32),
    ):
        boundary = (positions >= 0) & ((positions + 1) % ratio == 0)
        boundary_positions = torch.where(boundary, positions, -1)
        mapping[name] = _physical_rows(decode_tables[table_name], boundary_positions, 128, page_rows, ratio,
                                       counts=counts.get(table_name))
    for name, rows in mapping.items():
        for rank in range(RANKS):
            active = rows[rank][rows[rank] >= 0]
            # Compressor states publish a bounded tail after each 512-row
            # projection tile. Their physical pages may be reused by a later
            # tile once its predecessor's tail has been consumed.
            # The HCA prefill kernel publishes its 128-row history plus one
            # DSpark decode transaction in the same parallel state update.
            tail = 128 + PREFILL_STATE_ROWS if name == "hca_state_slot_mapping_full" else 8
            windows = ([active[start : start + 512][-tail:] for start in range(0, active.numel(), 512)]
                       if "state_slot_mapping" in name else [active])
            if any(window.numel() != torch.unique(window).numel() for window in windows):
                raise ValueError(f"{name} aliases two writes on rank {rank}")
    mapping_caches = {
        "ori_slot_mapping_full": ("kv_cache",),
        "hca_cmp_slot_mapping_full": ("hca_cmp_kv",),
        "csa_cmp_slot_mapping_full": ("csa_cmp_kv",),
        "csa_idx_slot_mapping_full": ("idx_kv_cache", "idx_kv_scale"),
        "hca_state_slot_mapping_full": ("hca_compress_state",),
        "csa_state_slot_mapping_full": ("csa_compress_state",),
        "csa_inner_state_slot_mapping_full": ("csa_inner_compress_state",),
    }
    for mapping_name, cache_names in mapping_caches.items():
        rows = mapping[mapping_name]
        for cache_name in cache_names:
            _check_rows(mapping_name, rows, caches[cache_name], cache_name)
    valid_positions = positions[0][positions[0] >= 0].to(torch.int64)
    if valid_positions.numel():
        start, end = int(valid_positions[0]), int(valid_positions[-1]) + 1
        if not torch.equal(valid_positions, torch.arange(start, end, device=positions.device)):
            raise ValueError("Prefill positions must be consecutive and followed by padding")
        # A later chunk can read the preceding 128 HCA projections. Check both
        # the incoming history and the history retained after each 512-row tile.
        # The caller must also keep their logical-to-physical mapping unchanged
        # between chunks; uniqueness alone cannot establish the stored values.
        boundaries = list(range(start, end, 512)) + [end]
        for boundary in boundaries:
            history = torch.arange(max(0, boundary - 128), boundary, dtype=torch.int64, device=positions.device)
            history = history.unsqueeze(0).expand(RANKS, -1)
            physical = _physical_rows(decode_tables["hca_compress_state_block_table"], history, 8, 8,
                                      counts=counts.get("hca_compress_state_block_table"))
            _check_rows("HCA history", physical, caches["hca_compress_state"], "hca_compress_state")
            for rank in range(RANKS):
                if physical.shape[1] != torch.unique(physical[rank]).numel():
                    raise ValueError(f"HCA history aliases live rows on rank {rank}")
        old_start = max(0, start - 127)
        window = torch.arange(old_start, end, dtype=torch.int64, device=positions.device)
        window = window.unsqueeze(0).expand(RANKS, -1)
        physical = _physical_rows(decode_tables["block_table"], window, 128, 128)
        _check_rows("SWA history", physical, caches["kv_cache"], "kv_cache")
        for rank in range(RANKS):
            if physical.shape[1] != torch.unique(physical[rank]).numel():
                raise ValueError("The prefill chunk would overwrite its live SWA history")
        for table_name, cache_name, ratio, page_rows in (
            ("hca_cmp_block_table", "hca_cmp_kv", 128, 1),
            ("csa_cmp_block_table", "csa_cmp_kv", 4, 32),
            ("idx_block_table", "idx_kv_cache", 4, 32),
        ):
            if end < ratio:
                continue
            boundaries = torch.arange(ratio - 1, end, ratio, dtype=torch.int64, device=positions.device)
            if not boundaries.numel():
                continue
            boundaries = boundaries.unsqueeze(0).expand(RANKS, -1)
            physical = _physical_rows(decode_tables[table_name], boundaries, 128, page_rows, ratio,
                                      counts=counts.get(table_name))
            _check_rows(table_name, physical, caches[cache_name], cache_name)
            if table_name == "idx_block_table":
                _check_rows(table_name, physical, caches["idx_kv_scale"], "idx_kv_scale")
            for rank in range(RANKS):
                if physical.shape[1] != torch.unique(physical[rank]).numel():
                    raise ValueError(f"{table_name} aliases compressed history")
    return mapping


class SharedInput:
    def __init__(self, max_rows=8192):
        if type(max_rows) is not int or max_rows < 8 or max_rows > 8192 or max_rows % 8:
            raise ValueError("Aligned1..8192-row buffer required")
        self.storage = torch.zeros((max_rows, 4, 4096), dtype=torch.float32).share_memory_()
        self.copying = False
        self.generation = 0

    def fill(self, embeddings):
        if self.copying:
            raise RuntimeError("Do not overwrite shared input while copies are active")
        if (
            not isinstance(embeddings, torch.Tensor)
            or embeddings.device.type != "cpu"
            or embeddings.dtype != torch.bfloat16
            or embeddings.ndim != 2
            or embeddings.shape[1] != 4096
        ):
            raise ValueError("Actual CPU BF16 embeddings required")
        rows = embeddings.shape[0]
        padded = (rows + 7) // 8 * 8
        if rows < 1 or padded > self.storage.shape[0]:
            raise ValueError("Input exceeds the real shared allocation")
        hidden = self.storage[:padded]
        hidden.zero_()
        hidden[:rows] = embeddings.float().unsqueeze(1)
        self.generation += 1
        return hidden.unsqueeze(0).expand(8, *hidden.shape)

    def begin_copy(self):
        if self.copying:
            raise RuntimeError("A copy is already active")
        self.copying = True

    def end_copy(self):
        if not self.copying:
            raise RuntimeError("No active copy to retire")
        self.copying = False


class CompactInput(SharedInput):
    """Register real BF16 prompt rows; HC expansion is a separate device op."""

    def __init__(self, max_rows=8192):
        if type(max_rows) is not int or max_rows < 8 or max_rows > 8192 or max_rows % 8:
            raise ValueError("Aligned1..8192-row buffer required")
        self.storage = torch.zeros((max_rows, 4096), dtype=torch.bfloat16).share_memory_()
        self.copying = False
        self.generation = 0

    def fill(self, embeddings):
        if self.copying:
            raise RuntimeError("Do not overwrite shared input while copies are active")
        if (
            not isinstance(embeddings, torch.Tensor)
            or embeddings.device.type != "cpu"
            or embeddings.dtype != torch.bfloat16
            or embeddings.ndim != 2
            or embeddings.shape[1] != 4096
        ):
            raise ValueError("Actual CPU BF16 embeddings required")
        rows = embeddings.shape[0]
        padded = (rows + 7) // 8 * 8
        if rows < 1 or padded > self.storage.shape[0]:
            raise ValueError("Input exceeds the real shared allocation")
        result = self.storage[:padded]
        result[:rows].copy_(embeddings)
        result[rows:].zero_()
        self.generation += 1
        return result.unsqueeze(0).expand(8, *result.shape)


MAX_PREFILL_TOKENS = 8192
HIDDEN = 4096
HC_MULT = 4
VOCAB = 129280


def advance_gate_epoch(epoch_base, physical_rows, *, layers=43):
    """Check a chunk's reservation and return its endpoint."""
    if any(type(value) is not int or value < 0 for value in (epoch_base, physical_rows, layers)):
        raise ValueError("MoE gate epochs, physical extents, and layer counts must be nonnegative integers")
    end = epoch_base + physical_rows * layers
    if end >= 2**31 - 1:
        raise ValueError("MoE gate counters require a fresh communication domain before INT32 overflow")
    return end


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
    """Build B1 prompt metadata for one eight-rank group and one chunk.

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
        "input_ids_full": _replicate(token_rows),
        "local_request_ids": torch.stack([
            torch.where(torch.arange(rank * local, (rank + 1) * local) < logical, 0, -1).to(torch.int32)
            for rank in range(RANKS)
        ]),
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


ALIGNMENT_BYTES = 64
PLAN_VERSION = 1


@dataclass(frozen=True)
class MetadataField:
    name: str
    shape: tuple
    dtype: torch.dtype
    element_offset: int

    @property
    def elements(self):
        return prod(self.shape)

    @property
    def byte_offset(self):
        return self.element_offset * torch.empty((), dtype=self.dtype).element_size()


class TypedMetadataArena:
    """Own one mutable shared CPU allocation per dtype, with aligned field views."""

    def __init__(self, specs):
        self.fields = {}
        self.elements = {}
        self.copying = False
        for spec in specs:
            name = spec["name"]
            shape = tuple(spec["per_rank_shape"])
            dtype = getattr(torch, spec["dtype"])
            if name in self.fields or not shape or any(type(n) is not int or n <= 0 for n in shape):
                raise ValueError("Metadata fields require unique names and positive per-rank shapes")
            if dtype not in (torch.int16, torch.int32, torch.int64, torch.bfloat16, torch.float32):
                raise ValueError("Unsupported metadata arena dtype")
            alignment = ALIGNMENT_BYTES // torch.empty((), dtype=dtype).element_size()
            start = (self.elements.get(dtype, 0) + alignment - 1) // alignment * alignment
            field = MetadataField(name, shape, dtype, start)
            self.fields[name] = field
            self.elements[dtype] = start + field.elements
        for dtype, elements in self.elements.items():
            alignment = ALIGNMENT_BYTES // torch.empty((), dtype=dtype).element_size()
            self.elements[dtype] = (elements + alignment - 1) // alignment * alignment
        self.roots = {dtype: torch.zeros((RANKS, n), dtype=dtype).share_memory_()
                      for dtype, n in self.elements.items()}
        self.views = {}
        for name, field in self.fields.items():
            root = self.roots[field.dtype]
            self.views[name] = root[:, field.element_offset:field.element_offset + field.elements]
            self.views[name] = self.views[name].view(RANKS, *field.shape)

    @classmethod
    def from_tensors(cls, tensors):
        specs = [{"name": name, "per_rank_shape": list(value.shape[1:]),
                  "dtype": str(value.dtype).removeprefix("torch.")}
                 for name, value in tensors.items()]
        result = cls(specs)
        result.copy_from(tensors)
        return result

    def copy_from(self, tensors):
        if self.copying:
            raise RuntimeError("Metadata upload must retire before host refill")
        if tensors.keys() != self.views.keys():
            raise ValueError("Metadata fields differ from the prepared arena")
        for name, value in tensors.items():
            target = self.views[name]
            if value.shape != target.shape or value.dtype != target.dtype or value.device.type != "cpu":
                raise ValueError("Metadata shape or dtype differs: " + name)
            target.copy_(value)

    def begin_copy(self):
        if self.copying:
            raise RuntimeError("Metadata copy is already active")
        self.copying = True

    def end_copy(self):
        if not self.copying:
            raise RuntimeError("No metadata copy is active")
        self.copying = False

    @property
    def bytes_per_rank(self):
        return sum(root[0].numel() * root.element_size() for root in self.roots.values())

    def allocation_specs(self):
        return [{"name": "metadata_arena_" + str(dtype).removeprefix("torch."),
                 "per_rank_shape": [n], "dtype": str(dtype).removeprefix("torch.")}
                for dtype, n in self.elements.items()]


def _tensor_stamp(value):
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise ValueError("Prepared metadata sources must be CPU tensors")
    return (value.data_ptr(), tuple(value.shape), tuple(value.stride()), str(value.dtype), value._version)


def _signature(owner, revision, token_count, start_pos, terminal, tables, caches, rope_cos, rope_sin,
               max_logit_rows, block_counts, extras):
    if not isinstance(owner, str) or not owner or not isinstance(revision, str) or not revision:
        raise ValueError("An explicit cache owner and metadata revision are required")
    cache_specs = tuple((name, tuple(value.shape), str(value.dtype)) for name, value in sorted(caches.items()))
    table_specs = tuple((name, _tensor_stamp(value)) for name, value in sorted(tables.items()))
    extra_specs = tuple((name, _tensor_stamp(value)) for name, value in sorted(extras.items()))
    return (PLAN_VERSION, owner, revision, token_count, start_pos, terminal, max_logit_rows,
            cache_specs, table_specs, _tensor_stamp(rope_cos), _tensor_stamp(rope_sin),
            None if block_counts is None else _tensor_stamp(block_counts), extra_specs)


class PreparedInputPlan:
    """Retain prompt-independent geometry; fill actual token IDs during each request."""

    def __init__(self, *, owner, revision, token_count, start_pos, terminal, tables, caches,
                 rope_cos, rope_sin, max_logit_rows, block_counts=None, extras):
        for name, shape in (("projection_epoch_bases", (RANKS, 6)), ("gate_epoch_bases", (RANKS, 1))):
            value = extras.get(name)
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                    or value.dtype != torch.int32 or tuple(value.shape) != shape):
                raise ValueError(f"{name} must be a CPU INT32 tensor with shape {shape}")
        self.context = dict(owner=owner, revision=revision, token_count=token_count, start_pos=start_pos,
                            terminal=terminal, tables=tables, caches=caches, rope_cos=rope_cos,
                            rope_sin=rope_sin, max_logit_rows=max_logit_rows,
                            block_counts=block_counts, extras=extras)
        self.signature = _signature(**self.context)
        placeholder = torch.zeros(token_count, dtype=torch.int64)
        values = prepare_prefill_metadata(placeholder, start_pos=start_pos, terminal=terminal,
                                          tables=tables, caches=caches, rope_cos=rope_cos, rope_sin=rope_sin,
                                          max_logit_rows=max_logit_rows, block_counts=block_counts)
        if values.keys() & extras.keys():
            raise ValueError("Extra metadata must not replace native geometry fields")
        values.update(extras)
        self.arena = TypedMetadataArena.from_tensors(values)
        self.generation = 0

    def validate(self, **changes):
        context = {**self.context, **changes}
        if _signature(**context) != self.signature:
            raise ValueError("Prepared input owner/page/position/shape/RoPE/grammar/revision changed")

    def fill(self, tokens, *, projection_epochs, gate_epoch, **changes):
        if self.arena.copying:
            raise RuntimeError("Metadata upload must retire before filling token IDs")
        self.validate(**changes)
        if (not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.int64
                or tokens.device.type != "cpu" or tokens.ndim != 1
                or tokens.numel() != self.context["token_count"]):
            raise ValueError("Actual INT64 token IDs must match the prepared token extent")
        if bool(((tokens < 0) | (tokens >= 129280)).any()):
            raise ValueError("Token ID outside the vocabulary")
        views = self.arena.views
        full = views["input_ids_full"]
        advance_gate_epoch(gate_epoch, full.shape[1])
        if len(projection_epochs) != 6 or any(type(v) is not int or not 0 <= v < 2**31 for v in projection_epochs):
            raise ValueError("Projection epochs must be six nonnegative INT32 values")
        full[:, :tokens.numel()].copy_(tokens.unsqueeze(0))
        full[:, tokens.numel():].zero_()
        views["input_ids"].copy_(full[0].reshape(RANKS, -1))
        views["projection_epoch_bases"].copy_(torch.tensor(projection_epochs, dtype=torch.int32).unsqueeze(0))
        views["gate_epoch_bases"].fill_(gate_epoch)
        self.generation += 1
        return views


class DeviceMetadataArena:
    """Own real typed device roots and borrow same-dtype public field views."""

    def __init__(self, worker, layout):
        from pypto.runtime import StackedDeviceTensor

        self.worker = worker
        self.busy = False
        self.roots = {}
        self.fields = {}
        allocations = []
        try:
            for dtype, n in layout.elements.items():
                shards = []
                for rank in range(RANKS):
                    shard = worker.alloc_tensor((n,), dtype, worker_id=rank)
                    allocations.append((rank, shard))
                    shards.append(shard)
                self.roots[dtype] = StackedDeviceTensor(shards, (RANKS, n), range(RANKS))
            for name, field in layout.fields.items():
                root = self.roots[field.dtype]
                shards = [worker.view_tensor(shard, field.shape, byte_offset=field.byte_offset, worker_id=rank)
                          for rank, shard in enumerate(root.shards)]
                self.fields[name] = StackedDeviceTensor(shards, (RANKS, *field.shape), range(RANKS))
        except BaseException:
            for rank, shard in reversed(allocations):
                try:
                    worker.free_tensor(shard, worker_id=rank)
                except Exception:
                    logging.getLogger(__name__).exception("Could not free failed metadata arena root on rank %d", rank)
            raise

    def require_idle(self, worker):
        if worker is not self.worker:
            raise ValueError("Metadata arenas belong to a different Worker")
        if self.busy:
            raise RuntimeError("Native metadata consumer must finish before arena reuse")

    def begin_consume(self, worker):
        self.require_idle(worker)
        self.busy = True

    def end_consume(self, worker):
        if worker is not self.worker or not self.busy:
            raise RuntimeError("No native metadata consumer is active for this Worker")
        self.busy = False
