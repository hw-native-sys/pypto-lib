# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Paged 128-row cache publication and sparse row gathering for every attention mode.

A cache is stored as ``[blocks, 128, 1, width]`` but consumed as a flat row
matrix, so a per-request block table and the host slot maps address it with a
single 64-bit slot.  Publication skips negative slots (rows a token does not
emit).  Gathering produces the per-token packed key set ``[window 128 |
Top-K 512]`` with the physical row ids kept beside it for masking.
"""

import pypto.language as pl
import torch

from config import (
    BLOCK_SIZE,
    CMP_BLOCKS_DYN,
    FLASH as M,
    ORI_BLOCKS_DYN,
    SELECTED_KEYS,
    T_DYN,
    WINDOW_KEYS,
)

HEAD_DIM = M.head_dim
INDEX_TOPK = M.index_topk

GATHER_WORKERS = 16


def compare_published_source_cache(actual, expected, *, inputs, **kwargs):
    """Compare the full cache and the physical rows published by this source."""
    from golden import ratio_allclose

    compare = ratio_allclose(atol=2e-2, rtol=1.0 / 64, max_error_ratio=1e-2)
    ok, detail = compare(actual, expected, inputs=inputs, **kwargs)
    if not ok:
        return ok, detail
    slots = inputs["compressed_slots"].long()
    slots = slots[slots >= 0].unique()
    if slots.numel() == 0:
        return ok, detail
    actual_rows = actual.reshape(-1, actual.shape[-1])[slots]
    expected_rows = expected.reshape(-1, expected.shape[-1])[slots]
    ok, rows_detail = compare(actual_rows, expected_rows, inputs=inputs, **kwargs)
    return ok, detail + "\n    published source rows: " + rows_detail


@pl.jit.inline(auto_scope=False)
def publish_cache_row(
    rows: pl.Tensor,
    slots: pl.Tensor,
    cache: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.constexpr,
    ready: pl.Scalar[pl.TASK_ID],
):
    """Write each ``rows[t]`` at physical slot ``slots[t]`` (negative skips)."""
    blocks = pl.tensor.dim(cache, 0)
    cache_rows = blocks * BLOCK_SIZE
    flat = pl.reshape(cache, [cache_rows, width])
    workers = pl.max(1, pl.min(num_tokens, GATHER_WORKERS))
    step = workers
    with pl.spmd(workers, name_hint="cache_publish", deps=[ready]) as publish_tid:
        slot_index = pl.tile.get_block_idx()
        for token in pl.range(slot_index, num_tokens, step):
            slot = pl.read(slots, [token])
            if slot >= 0:
                value = pl.load(rows, [token, 0], [1, width])
                pl.store(value, [pl.cast(slot, pl.INDEX), 0], flat)
    return cache, publish_tid


@pl.jit.inline(auto_scope=False)
def snapshot_rows(
    source: pl.Tensor,
    width: pl.constexpr,
):
    """Copy rows immediately so a later task cannot read a reused buffer."""
    rows = pl.tensor.dim(source, 0)
    target = pl.create_tensor([rows, width], dtype=pl.BF16)
    with pl.spmd((rows + 7) // 8, name_hint="snapshot_rows") as snapshot_tid:
        worker = pl.tile.get_block_idx()
        begin = worker * 8
        active = pl.min(8, rows - begin)
        value = pl.load(source, [begin, 0], [8, width], valid_shape=[active, width])
        pl.store(pl.set_validshape(value, active, width), [begin, 0], target)
    return target, snapshot_tid


@pl.jit.inline(auto_scope=False)
def gather_selected_keys(
    window_cache: pl.Tensor,
    window_indices: pl.Tensor,
    compressed_cache: pl.Tensor,
    topk_indices: pl.Tensor,
    selected: pl.Tensor,
    combined: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Gather the window lanes and the Top-K compressed lanes per token."""
    window_rows = pl.tensor.dim(window_cache, 0) * BLOCK_SIZE
    compressed_rows = pl.tensor.dim(compressed_cache, 0) * BLOCK_SIZE
    window_flat = pl.reshape(window_cache, [window_rows, HEAD_DIM])
    compressed_flat = pl.reshape(compressed_cache, [compressed_rows, HEAD_DIM])
    tokens = pl.tensor.dim(selected, 0)
    selected_rows = tokens * SELECTED_KEYS
    destination = pl.reshape(selected, [selected_rows, HEAD_DIM])
    workers = pl.max(1, pl.min(num_tokens, GATHER_WORKERS))
    step = workers
    with pl.spmd(workers, name_hint="gather_selected", deps=[ready]) as gathered:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, num_tokens, step):
            base = token * SELECTED_KEYS
            for lane in pl.range(WINDOW_KEYS):
                row_id = pl.read(window_indices, [token, lane])
                pl.write(combined, [token, lane], row_id)
                value = pl.tile.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
                if row_id >= 0:
                    value = pl.load(window_flat, [pl.cast(row_id, pl.INDEX), 0], [1, HEAD_DIM])
                pl.store(value, [base + lane, 0], destination)
            for lane in pl.range(INDEX_TOPK):
                slot = WINDOW_KEYS + lane
                row_id = pl.read(topk_indices, [token, lane])
                pl.write(combined, [token, slot], row_id)
                value = pl.tile.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
                if row_id >= 0:
                    value = pl.load(compressed_flat, [pl.cast(row_id, pl.INDEX), 0], [1, HEAD_DIM])
                pl.store(value, [base + slot, 0], destination)
    return selected, combined, gathered


@pl.jit.inline(auto_scope=False)
def gather_window_keys(
    window_cache: pl.Tensor,
    window_indices: pl.Tensor,
    selected: pl.Tensor,
    combined: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Gather only the raw sliding window (SWA layers)."""
    window_rows = pl.tensor.dim(window_cache, 0) * BLOCK_SIZE
    window_flat = pl.reshape(window_cache, [window_rows, HEAD_DIM])
    tokens = pl.tensor.dim(selected, 0)
    selected_rows = tokens * WINDOW_KEYS
    destination = pl.reshape(selected, [selected_rows, HEAD_DIM])
    workers = pl.max(1, pl.min(num_tokens, GATHER_WORKERS))
    step = workers
    with pl.spmd(workers, name_hint="gather_window", deps=[ready]) as gathered:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, num_tokens, step):
            base = token * WINDOW_KEYS
            for lane in pl.range(WINDOW_KEYS):
                row_id = pl.read(window_indices, [token, lane])
                pl.write(combined, [token, lane], row_id)
                value = pl.tile.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
                if row_id >= 0:
                    value = pl.load(window_flat, [pl.cast(row_id, pl.INDEX), 0], [1, HEAD_DIM])
                pl.store(value, [base + lane, 0], destination)
    return selected, combined, gathered


def golden_publish_cache_row(tensors):
    """Scatter rows into the cache at their slots; negative slots stay untouched."""
    from utils import scatter_cache_rows

    tensors["cache"] = scatter_cache_rows(tensors["cache"], tensors["slots"], tensors["rows"])
    return tensors["cache"]


def golden_gather_selected_keys(tensors):
    """Torch reference for the packed per-token key gather."""
    from utils import gather_cache_rows

    window = gather_cache_rows(tensors["window_cache"], tensors["window_indices"].reshape(-1))
    window = window.reshape(tensors["window_indices"].shape[0], WINDOW_KEYS, HEAD_DIM)
    compressed = gather_cache_rows(tensors["compressed_cache"], tensors["topk_indices"].reshape(-1))
    compressed = compressed.reshape(tensors["topk_indices"].shape[0], INDEX_TOPK, HEAD_DIM)
    tensors["selected"][:] = torch.cat((window, compressed), dim=1)
    combined = torch.cat((tensors["window_indices"], tensors["topk_indices"]), dim=1)
    tensors["combined"][:] = combined


def golden_gather_window_keys(tensors):
    """Torch reference for the SWA window gather."""
    from utils import gather_cache_rows

    rows = gather_cache_rows(tensors["window_cache"], tensors["window_indices"].reshape(-1))
    tensors["selected"][:] = rows.reshape(tensors["window_indices"].shape[0], WINDOW_KEYS, HEAD_DIM)
    tensors["combined"][:] = tensors["window_indices"]


@pl.jit
def cache_publish_test(
    rows: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
):
    """Bounded publication entry."""
    rows.bind_dynamic(0, T_DYN)
    slots.bind_dynamic(0, T_DYN)
    cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    start = pl.system.task_dummy(deps=[])
    _, publish_tid = publish_cache_row(rows, slots, cache, pl.tensor.dim(rows, 0), width=HEAD_DIM, ready=start)
    return cache, publish_tid


@pl.jit
def selected_gather_test(
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_indices: pl.Tensor[[T_DYN, WINDOW_KEYS], pl.INT32],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    selected: pl.Out[pl.Tensor[[T_DYN, SELECTED_KEYS, HEAD_DIM], pl.BF16]],
    combined: pl.Out[pl.Tensor[[T_DYN, SELECTED_KEYS], pl.INT32]],
):
    """Bounded packed-key gather entry."""
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    window_indices.bind_dynamic(0, T_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    selected.bind_dynamic(0, T_DYN)
    combined.bind_dynamic(0, T_DYN)
    ready = pl.system.task_dummy(deps=[])
    gather_selected_keys(
        window_cache, window_indices, compressed_cache, topk_indices,
        selected, combined, pl.tensor.dim(window_indices, 0), ready,
    )
    return selected, combined


@pl.jit
def window_gather_test(
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_indices: pl.Tensor[[T_DYN, WINDOW_KEYS], pl.INT32],
    selected: pl.Out[pl.Tensor[[T_DYN, WINDOW_KEYS, HEAD_DIM], pl.BF16]],
    combined: pl.Out[pl.Tensor[[T_DYN, WINDOW_KEYS], pl.INT32]],
):
    """Bounded SWA window gather entry."""
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    window_indices.bind_dynamic(0, T_DYN)
    selected.bind_dynamic(0, T_DYN)
    combined.bind_dynamic(0, T_DYN)
    ready = pl.system.task_dummy(deps=[])
    gather_window_keys(
        window_cache, window_indices, selected, combined, pl.tensor.dim(window_indices, 0), ready,
    )
    return selected, combined


def _fixture(tokens: int = 8, seed: int = 0):
    """Shared multi-request paged fixture (two requests, one shared block table)."""
    from utils import build_request_plan

    generator = torch.Generator().manual_seed(seed)
    plan = build_request_plan([(0, tokens // 2), (tokens // 2 + 3, tokens - tokens // 2)])
    pages = plan["pages"]
    window = torch.randn(pages * BLOCK_SIZE, HEAD_DIM, generator=generator).to(torch.bfloat16)
    compressed = torch.randn(pages * BLOCK_SIZE, HEAD_DIM, generator=generator).to(torch.bfloat16)
    return plan, window, compressed


def _cache_spec(names, plan, values):
    from golden import TensorSpec

    return [
        TensorSpec(name, [plan["pages"], BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16,
                   init_value=value.reshape(plan["pages"], BLOCK_SIZE, 1, HEAD_DIM))
        for name, value in zip(names, values, strict=True)
    ]


def build_publish_specs(tokens: int = 8):
    from golden import TensorSpec

    plan, _, _ = _fixture(tokens)
    rows = torch.randn(tokens, HEAD_DIM).to(torch.bfloat16)
    return [
        TensorSpec("rows", [tokens, HEAD_DIM], torch.bfloat16, init_value=rows),
        TensorSpec("slots", [tokens], torch.int64, init_value=plan["window_slots"]),
        TensorSpec("cache", [plan["pages"], BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=0.0),
    ]


def _topk_fixture(plan, tokens):
    """Use every token's compressed slot as Top-K lane 0; other lanes masked."""
    padded = torch.full((tokens, INDEX_TOPK), -1, dtype=torch.int32)
    slots = plan["compressed_slots"]
    padded[:, 0] = torch.where(slots >= 0, slots, torch.full_like(slots, -1)).to(torch.int32)
    return padded


def build_gather_specs(tokens: int = 8):
    from golden import TensorSpec

    plan, window, compressed = _fixture(tokens)
    caches = _cache_spec(("window_cache", "compressed_cache"), plan, (window, compressed))
    return [
        caches[0],
        TensorSpec("window_indices", [tokens, WINDOW_KEYS], torch.int32, init_value=plan["window_indices"]),
        caches[1],
        TensorSpec("topk_indices", [tokens, INDEX_TOPK], torch.int32, init_value=_topk_fixture(plan, tokens)),
        TensorSpec("selected", [tokens, SELECTED_KEYS, HEAD_DIM], torch.bfloat16),
        TensorSpec("combined", [tokens, SELECTED_KEYS], torch.int32),
    ]


def build_window_gather_specs(tokens: int = 8):
    from golden import TensorSpec

    plan, window, _ = _fixture(tokens)
    specs = _cache_spec(("window_cache",), plan, (window,))
    return specs + [
        TensorSpec("window_indices", [tokens, WINDOW_KEYS], torch.int32, init_value=plan["window_indices"]),
        TensorSpec("selected", [tokens, WINDOW_KEYS, HEAD_DIM], torch.bfloat16),
        TensorSpec("combined", [tokens, WINDOW_KEYS], torch.int32),
    ]


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    from config import add_parallel_arguments
    add_parallel_arguments(parser)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    exact = {"cache": ratio_allclose(atol=0.0, rtol=0.0)}
    gather_compare = {
        "selected": ratio_allclose(atol=0.0, rtol=0.0),
        "combined": ratio_allclose(atol=0.0, rtol=0.0),
    }
    cases = [
        ("publish", cache_publish_test, build_publish_specs(args.tokens), golden_publish_cache_row, exact),
        ("selected_gather", selected_gather_test, build_gather_specs(args.tokens), golden_gather_selected_keys, gather_compare),
        ("window_gather", window_gather_test, build_window_gather_specs(args.tokens), golden_gather_window_keys, gather_compare),
    ]
    failures = []
    for name, fn, specs, golden_fn, compare in cases:
        print(f"[PAGED] {name}", flush=True)
        result = run(
            fn=fn,
            specs=specs,
            golden_fn=golden_fn,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            rtol=0.0,
            atol=0.0,
            compare_fn=compare,
            compile_only=args.compile_only,
        )
        if not result.passed:
            failures.append(f"{name}: {result.error or 'validation failed'}")
    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)
