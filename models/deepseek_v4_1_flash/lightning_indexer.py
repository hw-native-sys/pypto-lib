# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared bindings for the native paged lightning indexer."""
import os
from pathlib import Path

import pypto.language as pl
from pypto.runtime import pto_isa_include_dir


KERNEL_DIR = Path(__file__).parent / "kernels"
LI_SOURCE = KERNEL_DIR / "lightning_indexer.cpp"
CANN = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
WORKERS = 32


LI_INCLUDES = (pto_isa_include_dir(), CANN / "asc", CANN / "asc/include",
               CANN / "asc/include/basic_api", CANN / "include")
# Stage tags in the lightning_indexer.cpp task ABI.
METADATA, ATTENTION, SCORE, POOL, SCORE_CANDIDATES = range(5)


@pl.jit.inline
def _request_work(
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    start: pl.Scalar[pl.INDEX],
    active: pl.Scalar[pl.INDEX],
):
    request = pl.read(request_ids, [start])
    end = start + 1
    context = pl.cast(pl.read(lengths, [start]), pl.INDEX)
    scanning = True
    while end < active and scanning:
        next_request = pl.read(request_ids, [end])
        if next_request == request:
            context = pl.max(context, pl.cast(pl.read(lengths, [end]), pl.INDEX))
            end = end + 1
        else:
            scanning = False
    splits = (context + 511) // 512
    tile_cost = pl.cast(2 + 2 * ((end - start + 1) // 2), pl.INDEX)
    split_cost = 4 * tile_cost + 2
    cost = pl.cast(0, pl.INDEX)
    if splits > 0:
        last_tokens = context - (splits - 1) * 512
        cost = (splits - 1) * split_cost + ((last_tokens + 127) // 128) * tile_cost + 2
    return end, splits, split_cost, cost


@pl.jit.incore
def _score_prepare(
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    metadata: pl.Out[pl.Tensor],
    active: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    packed_bounds = pl.tile.create([1, 72], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec)
    zero = pl.cast(0, pl.INDEX)
    for (row, accumulated_cost) in pl.while_(init_values=(zero, zero)):
        pl.cond(row < active)
        next_row, _splits, _split_cost, cost = _request_work(request_ids, lengths, row, active)
        _, total_cost = pl.yield_(next_row, accumulated_cost + cost)

    cost_per_worker = total_cost // WORKERS
    extra_workers = total_cost % WORKERS
    for (start, prefix, worker) in pl.while_(init_values=(zero, zero, zero)):
        pl.cond(start < active and worker < WORKERS)
        end, splits, split_cost, cost = _request_work(request_ids, lengths, start, active)
        next_prefix = prefix + cost
        target = worker * cost_per_worker + pl.min(worker, extra_workers)
        for (slot, boundary) in pl.while_(init_values=(worker, target)):
            pl.cond(boundary < next_prefix and slot < WORKERS)
            split = pl.min((boundary - prefix) // split_cost, splits - 1)
            pl.tile.write(packed_bounds, [0, 2 * slot], pl.cast(start, pl.INT32))
            pl.tile.write(packed_bounds, [0, 2 * slot + 1], pl.cast(split, pl.INT32))
            next_slot = slot + 1
            next_boundary = next_slot * cost_per_worker + pl.min(next_slot, extra_workers)
            next_worker, _ = pl.yield_(next_slot, next_boundary)
        _, _, worker_end = pl.yield_(end, next_prefix, next_worker)

    for unused_worker in pl.range(worker_end, WORKERS + 1):
        pl.tile.write(packed_bounds, [0, 2 * unused_worker], pl.cast(active, pl.INT32))
        pl.tile.write(packed_bounds, [0, 2 * unused_worker + 1], pl.cast(0, pl.INT32))
    return pl.store(pl.set_validshape(packed_bounds, 1, 66), [0, 0], metadata)


@pl.jit.extern(core_type="mixed", aic_source=LI_SOURCE, aiv_source=LI_SOURCE,
               dual_aiv_dispatch=True, include_dirs=LI_INCLUDES)
def _score(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    key: pl.Tensor,
    key_scale: pl.Tensor,
    table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    metadata: pl.Tensor,
    scores: pl.Out[pl.Tensor],
    active: pl.Scalar[pl.INT64],
    mode: pl.Scalar[pl.INT64],
) -> pl.Tensor:
    ...


@pl.jit.extern(core_type="mixed", aic_source=LI_SOURCE, aiv_source=KERNEL_DIR / "prefill_lightning_indexer.cpp",
               dual_aiv_dispatch=True, include_dirs=(*LI_INCLUDES, KERNEL_DIR))
def _prefill_score(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    key: pl.Tensor,
    key_scale: pl.Tensor,
    table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    metadata: pl.Tensor,
    scores: pl.Out[pl.Tensor],
    active: pl.Scalar[pl.INT64],
    mode: pl.Scalar[pl.INT64],
) -> pl.Tensor:
    ...


@pl.jit.extern(core_type="mixed", aic_source=LI_SOURCE, aiv_source=LI_SOURCE,
               dual_aiv_dispatch=True, include_dirs=LI_INCLUDES)
def _score_candidates(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    key: pl.Tensor,
    key_scale: pl.Tensor,
    table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    metadata: pl.Tensor,
    scores: pl.Out[pl.Tensor],
    block_scores: pl.Out[pl.Tensor],
    mode: pl.Scalar[pl.INT64],
    active: pl.Scalar[pl.INT64],
) -> pl.Tensor:
    ...


@pl.jit.inline(auto_scope=False)
def prepare_score_metadata(
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    active_rows = pl.cast(num_tokens, pl.INDEX)
    metadata = pl.create_tensor([1, 66], dtype=pl.INT32)
    with pl.spmd(1, name_hint="qli_score_metadata"):
        metadata = _score_prepare(request_ids, lengths, metadata, active_rows)
    return pl.reshape(metadata, [33, 2])


@pl.jit.inline(auto_scope=False)
def score_paged(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    key: pl.Tensor,
    key_scale: pl.Tensor,
    table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    metadata: pl.Tensor,
    scores: pl.Tensor,
    active: pl.Scalar[pl.INT32],
    block_scores: pl.Tensor,
    candidates: pl.constexpr = False,
):
    active_rows = pl.cast(active, pl.INT64)
    with pl.spmd(WORKERS, name_hint="qli_score"):
        if candidates:
            _score_candidates(query, query_scale, weights, key, key_scale, table, request_ids, lengths, metadata, scores, block_scores, SCORE_CANDIDATES, active_rows)
        else:
            _score(query, query_scale, weights, key, key_scale, table, request_ids, lengths, metadata, scores, active_rows, SCORE)
