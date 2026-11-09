# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Views and metadata shared by DSpark prefill and batch-one decode."""

from math import prod

import torch


RANKS = 8
CP_SIZE = 4

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
        raise ValueError("The two prefill CP groups must process the same positions")
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
            tail = 128 if name == "hca_state_slot_mapping_full" else 8
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
            for rank in range(RANKS):
                if physical.shape[1] != torch.unique(physical[rank]).numel():
                    raise ValueError(f"{table_name} aliases compressed history")
    return mapping
