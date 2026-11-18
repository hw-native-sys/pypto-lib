# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Publish completed cache prefixes and gate queries by their logical causal ranges."""

import sys

import pypto.language as pl
import pypto.language.distributed as pld


# Dynamic shape variables.
CAUSAL_BLOCKS_DYN = pl.dynamic("CAUSAL_BLOCKS_DYN")

# Scheduling options.
CAUSAL_READY = "--prefill-causal-ready" in sys.argv and sys.argv[sys.argv.index("--prefill-causal-ready") + 1] == "1"
CAUSAL_FULL_WAIT = "--prefill-causal-full-wait" in sys.argv and sys.argv[sys.argv.index("--prefill-causal-full-wait") + 1] == "1"
CAUSAL_FAMILIES = 3
CAUSAL_REQUIRED_COLS = 16

# Tiling.
CAUSAL_QUERY_TILE = 128
CAUSAL_CACHE_TILE = 512


@pl.jit.inline(auto_scope=False)
def causal_required_prefixes(
    query_start_loc: pl.Tensor,
    position_ids_full: pl.Tensor,
    position_ids_local: pl.Tensor,
    request_ids_local: pl.Tensor,
    required: pl.Tensor[[CAUSAL_BLOCKS_DYN, CAUSAL_REQUIRED_COLS], pl.INT32],
    seed: pl.Scalar[pl.TASK_ID],
    compress_ratio: pl.constexpr,
    has_indexer: pl.constexpr,
):
    """Map valid query positions to current-chunk row prefixes for three cache families."""
    query_rows = pl.tensor.dim(position_ids_local, 0)
    full_rows = pl.tensor.dim(position_ids_full, 0)
    with pl.spmd((query_rows + CAUSAL_QUERY_TILE - 1) // CAUSAL_QUERY_TILE, name_hint="causal_required_prefixes", deps=[seed]) as metadata_ready:
        block = pl.tile.get_block_idx()
        raw_prefix = pl.cast(0, pl.INDEX)
        compressed_prefix = pl.cast(0, pl.INDEX)
        indexer_prefix = pl.cast(0, pl.INDEX)
        for offset in pl.range(CAUSAL_QUERY_TILE):
            query = block * CAUSAL_QUERY_TILE + offset
            if query < query_rows:
                request = pl.read(request_ids_local, [query])
                if request >= 0:
                    begin = pl.read(query_start_loc, [request])
                    end = pl.read(query_start_loc, [request + 1])
                    if begin < end:
                        origin = pl.read(position_ids_full, [begin])
                        position = pl.read(position_ids_local, [query])
                        raw_rows = pl.max(pl.min(position - origin + 1, end - begin), 0)
                        if raw_rows > 0:
                            raw_prefix = pl.max(raw_prefix, begin + raw_rows)
                        if compress_ratio > 0:
                            compressed_rows = ((position + 1) // compress_ratio) * compress_ratio - origin
                            compressed_rows = pl.max(pl.min(compressed_rows, end - begin), 0)
                            if compressed_rows > 0:
                                compressed_prefix = pl.max(compressed_prefix, begin + compressed_rows)
                                if has_indexer:
                                    indexer_prefix = pl.max(indexer_prefix, begin + compressed_rows)
        if CAUSAL_FULL_WAIT:
            if raw_prefix > 0:
                raw_prefix = pl.cast(full_rows, pl.INDEX)
            if compressed_prefix > 0:
                compressed_prefix = pl.cast(full_rows, pl.INDEX)
            if indexer_prefix > 0:
                indexer_prefix = pl.cast(full_rows, pl.INDEX)
        prefix_row = pl.tile.full([1, CAUSAL_REQUIRED_COLS], dtype=pl.INT32, value=0)
        pl.tile.write(prefix_row, [0, 0], pl.cast(raw_prefix, pl.INT32))
        pl.tile.write(prefix_row, [0, 1], pl.cast(compressed_prefix, pl.INT32))
        pl.tile.write(prefix_row, [0, 2], pl.cast(indexer_prefix, pl.INT32))
        pl.store(prefix_row, [block, 0], required)
    return metadata_ready


@pl.jit.inline(auto_scope=False)
def causal_publish_prefix(
    ready: pld.DistributedTensor[[CAUSAL_FAMILIES, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    rows: pl.Scalar[pl.INDEX],
    cache_complete: pl.Scalar[pl.TASK_ID],
    previous_publication: pl.Scalar[pl.TASK_ID],
    family: pl.constexpr,
):
    """Advance one local prefix after its physical cache writes complete."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="causal_cache_prefix_publish", deps=[cache_complete, previous_publication], no_dep_args=[ready], allow_early_resolve=False) as published:
        pld.system.notify(ready, my_rank, [family, 0], value=pl.cast(rows, pl.INT32), op=pld.NotifyOp.AtomicAdd)
    return published


@pl.jit.inline(auto_scope=False)
def causal_query_ready(
    ready: pld.DistributedTensor[[CAUSAL_FAMILIES, 1], pl.INT32],
    required: pl.Tensor[[CAUSAL_BLOCKS_DYN, CAUSAL_REQUIRED_COLS], pl.INT32],
    epoch_base: pl.Scalar[pl.INT32],
    block: pl.Scalar[pl.INDEX],
    prior: pl.Scalar[pl.TASK_ID],
):
    """Register core-releasing waits for only the query block's required cache prefixes."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="causal_query_defer_wait", deps=[prior], no_dep_args=[ready], allow_early_resolve=False) as cache_ready:
        raw_prefix = pl.read(required, [block, 0])
        compressed_prefix = pl.read(required, [block, 1])
        indexer_prefix = pl.read(required, [block, 2])
        raw_expected = epoch_base + raw_prefix
        compressed_expected = epoch_base + compressed_prefix
        indexer_expected = epoch_base + indexer_prefix
        if raw_prefix > 0:
            pld.system.defer_wait(ready, [0, 0], raw_expected, cmp=pld.WaitCmp.Ge)
        if compressed_prefix > 0:
            pld.system.defer_wait(ready, [1, 0], compressed_expected, cmp=pld.WaitCmp.Ge)
        if indexer_prefix > 0:
            pld.system.defer_wait(ready, [2, 0], indexer_expected, cmp=pld.WaitCmp.Ge)
    return cache_ready


@pl.jit.inline(auto_scope=False)
def causal_layer_ready(
    ready: pld.DistributedTensor[[CAUSAL_FAMILIES, 1], pl.INT32],
    epoch_base: pl.Scalar[pl.INT32],
    full_rows: pl.Scalar[pl.INDEX],
    prior: pl.Scalar[pl.TASK_ID],
):
    """Retire all reserved cache-prefix rows before the next layer or chunk."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="causal_layer_defer_wait", deps=[prior], no_dep_args=[ready], allow_early_resolve=False) as layer_ready:
        expected = epoch_base + pl.cast(full_rows, pl.INT32)
        pld.system.defer_wait(ready, [0, 0], expected, cmp=pld.WaitCmp.Ge)
        pld.system.defer_wait(ready, [1, 0], expected, cmp=pld.WaitCmp.Ge)
        pld.system.defer_wait(ready, [2, 0], expected, cmp=pld.WaitCmp.Ge)
    return layer_ready


def causal_tensor_specs(tp_size):
    """Build rank-stacked initial cache-prefix epochs for component validation."""
    import torch
    from golden import TensorSpec

    return [TensorSpec("causal_epoch_bases", [tp_size, 1], torch.int32, init_value=torch.zeros(tp_size, 1, dtype=torch.int32))]
