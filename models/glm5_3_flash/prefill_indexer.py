# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill indexer composition and validation."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import BLOCK_SIZE, D
from models.glm5_3_flash.config import INDEX_BLOCKS_DYN, INDEX_DIM, INDEX_H
from models.glm5_3_flash.config import INDEX_KPOOL, INDEX_STATE_BLOCK_SIZE
from models.glm5_3_flash.config import INDEX_STATE_WIDTH
from models.glm5_3_flash.config import KPOOL_SELECT_K, POOL_EVENTS_DYN, Q_LORA, TABLE_DYN, T_DYN
from models.glm5_3_flash.config import TOPK_INDEX_WIDTH
from models.glm5_3_flash.config import POOLS_DYN, POOL_PAGES_DYN
from models.glm5_3_flash.indexer_cache import POOL_SCALE_WIDTH, POOL_VALID_WIDTH
from models.glm5_3_flash.indexer_cache import golden_indexer_cache_write, golden_indexer_pool_write
from models.glm5_3_flash.indexer_cache import indexer_cache_write, indexer_pool_write
from models.glm5_3_flash.indexer_cache import sylvester_hadamard
from models.glm5_3_flash.quantization import quantize_per_token_int8

from models.glm5_3_flash.indexer import (
    PAIR_WIDTH,
    LEAF, K_NORM_EPS, PROJ_T_TILE, golden_indexer_proj,
    indexer_proj, golden_indexer_score, indexer_score, golden_indexer_topk,
    indexer_topk, golden_indexer_expand, indexer_expand,
)


@pl.jit
def indexer_proj_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    q_resid: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    w_q_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.BF16],
    w_k: pl.Tensor[[D, INDEX_DIM], pl.BF16],
    k_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    k_norm_bias: pl.Tensor[[INDEX_DIM], pl.BF16],
    w_weights: pl.Tensor[[D, INDEX_H], pl.BF16],
    w_compress_gate: pl.Tensor[[D, INDEX_DIM], pl.BF16],
    index_q: pl.Out[pl.Tensor[[T_DYN, INDEX_H, INDEX_DIM], pl.BF16]],
    index_k: pl.Out[pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16]],
    head_weights: pl.Out[pl.Tensor[[T_DYN, INDEX_H], pl.FP32]],
    gate_scores: pl.Out[pl.Tensor[[T_DYN, INDEX_DIM], pl.FP32]],
):
    """Run the four projections for golden.run validation."""
    x.bind_dynamic(0, T_DYN)
    q_resid.bind_dynamic(0, T_DYN)
    index_q.bind_dynamic(0, T_DYN)
    index_k.bind_dynamic(0, T_DYN)
    head_weights.bind_dynamic(0, T_DYN)
    gate_scores.bind_dynamic(0, T_DYN)
    indexer_proj(
        x,
        q_resid,
        w_q_b,
        w_k,
        k_norm_weight,
        k_norm_bias,
        w_weights,
        w_compress_gate,
        index_q,
        index_k,
        head_weights,
        gate_scores, pl.cast(0, pl.INT32),
    )
    return index_q, index_k, head_weights, gate_scores


@pl.jit
def indexer_score_test(
    index_q: pl.Tensor[[T_DYN, INDEX_H, INDEX_DIM], pl.BF16],
    hadamard: pl.Tensor[[INDEX_DIM, INDEX_DIM], pl.BF16],
    pool_cache: pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, INDEX_DIM], pl.INT8],
    pool_scale: pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, POOL_SCALE_WIDTH], pl.FP32],
    pool_blocks: pl.Tensor[[POOL_PAGES_DYN], pl.INT32],
    head_weights: pl.Tensor[[T_DYN, INDEX_H], pl.FP32],
    seg_start: pl.Tensor[[T_DYN], pl.INT32],
    pool_count: pl.Tensor[[T_DYN], pl.INT32],
    index_scores: pl.Out[pl.Tensor[[T_DYN, POOLS_DYN], pl.FP32]],
):
    """Score one packed batch for golden.run validation."""
    index_q.bind_dynamic(0, T_DYN)
    head_weights.bind_dynamic(0, T_DYN)
    seg_start.bind_dynamic(0, T_DYN)
    pool_count.bind_dynamic(0, T_DYN)
    pool_blocks.bind_dynamic(0, POOL_PAGES_DYN)
    index_scores.bind_dynamic(0, T_DYN)
    index_scores.bind_dynamic(1, POOLS_DYN)
    indexer_score(
        index_q,
        hadamard,
        pool_cache,
        pool_scale,
        pool_blocks,
        head_weights,
        seg_start,
        pool_count,
        index_scores,
    )
    return index_scores


@pl.jit
def indexer_topk_test(
    index_scores: pl.Tensor[[T_DYN, POOLS_DYN], pl.FP32],
    seg_start: pl.Tensor[[T_DYN], pl.INT32],
    pool_count: pl.Tensor[[T_DYN], pl.INT32],
    selected_pools: pl.Out[pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32]],
    selected_valid: pl.Out[pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32]],
):
    """Select the top pools for one batch for golden.run validation."""
    index_scores.bind_dynamic(0, T_DYN)
    index_scores.bind_dynamic(1, POOLS_DYN)
    seg_start.bind_dynamic(0, T_DYN)
    pool_count.bind_dynamic(0, T_DYN)
    selected_pools.bind_dynamic(0, T_DYN)
    selected_valid.bind_dynamic(0, T_DYN)
    indexer_topk(index_scores, seg_start, pool_count, selected_pools, selected_valid)
    return selected_pools, selected_valid


@pl.jit
def indexer_expand_test(
    selected_pools: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    selected_valid: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    tail_start: pl.Tensor[[T_DYN], pl.INT32],
    tail_count: pl.Tensor[[T_DYN], pl.INT32],
    kv_len: pl.Tensor[[T_DYN], pl.INT32],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32]],
):
    """Expand one batch of selections for golden.run validation."""
    selected_pools.bind_dynamic(0, T_DYN)
    selected_valid.bind_dynamic(0, T_DYN)
    tail_start.bind_dynamic(0, T_DYN)
    tail_count.bind_dynamic(0, T_DYN)
    kv_len.bind_dynamic(0, T_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    indexer_expand(selected_pools, selected_valid, tail_start, tail_count, kv_len, topk_indices)
    return topk_indices


@pl.jit
def indexer_prefill_step_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    q_resid: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    w_q_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.BF16],
    w_k: pl.Tensor[[D, INDEX_DIM], pl.BF16],
    k_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    k_norm_bias: pl.Tensor[[INDEX_DIM], pl.BF16],
    w_weights: pl.Tensor[[D, INDEX_H], pl.BF16],
    w_compress_gate: pl.Tensor[[D, INDEX_DIM], pl.BF16],
    hadamard: pl.Tensor[[INDEX_DIM, INDEX_DIM], pl.BF16],
    compress_ape: pl.Tensor[[INDEX_KPOOL, INDEX_DIM], pl.BF16],
    index_slots: pl.Tensor[[T_DYN], pl.INT32],
    pool_token_slots: pl.Tensor[[POOL_EVENTS_DYN, INDEX_KPOOL], pl.INT32],
    pool_slots: pl.Tensor[[POOL_EVENTS_DYN], pl.INT32],
    pool_blocks: pl.Tensor[[POOL_PAGES_DYN], pl.INT32],
    seg_start: pl.Tensor[[T_DYN], pl.INT32],
    pool_count: pl.Tensor[[T_DYN], pl.INT32],
    tail_start: pl.Tensor[[T_DYN], pl.INT32],
    tail_count: pl.Tensor[[T_DYN], pl.INT32],
    kv_len: pl.Tensor[[T_DYN], pl.INT32],
    raw_cache: pl.InOut[pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32]],
    pool_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, INDEX_DIM], pl.INT8]],
    pool_scale: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, POOL_SCALE_WIDTH], pl.FP32]],
    pool_valid: pl.Out[pl.Tensor[[POOL_EVENTS_DYN, POOL_VALID_WIDTH], pl.INT32]],
    index_q: pl.Out[pl.Tensor[[T_DYN, INDEX_H, INDEX_DIM], pl.BF16]],
    index_k: pl.Out[pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16]],
    head_weights: pl.Out[pl.Tensor[[T_DYN, INDEX_H], pl.FP32]],
    gate_scores: pl.Out[pl.Tensor[[T_DYN, INDEX_DIM], pl.FP32]],
    index_scores: pl.Out[pl.Tensor[[T_DYN, POOLS_DYN], pl.FP32]],
    selected_pools: pl.Out[pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32]],
    selected_valid: pl.Out[pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32]],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32]],
):
    """Run one whole prefill chunk for golden.run validation.

    One request's chunk walks the pipeline the way serving does: project the
    chunk, scatter its raw rows, close every pool the chunk completes, score
    each query against the request's whole visible segment, and expand the
    front-packed index list. Every stage's output is exposed so the harness
    validates the composition, not just the tail.
    """
    x.bind_dynamic(0, T_DYN)
    q_resid.bind_dynamic(0, T_DYN)
    index_slots.bind_dynamic(0, T_DYN)
    seg_start.bind_dynamic(0, T_DYN)
    pool_count.bind_dynamic(0, T_DYN)
    tail_start.bind_dynamic(0, T_DYN)
    tail_count.bind_dynamic(0, T_DYN)
    kv_len.bind_dynamic(0, T_DYN)
    pool_token_slots.bind_dynamic(0, POOL_EVENTS_DYN)
    pool_slots.bind_dynamic(0, POOL_EVENTS_DYN)
    pool_blocks.bind_dynamic(0, POOL_PAGES_DYN)
    pool_valid.bind_dynamic(0, POOL_EVENTS_DYN)
    index_q.bind_dynamic(0, T_DYN)
    index_k.bind_dynamic(0, T_DYN)
    head_weights.bind_dynamic(0, T_DYN)
    gate_scores.bind_dynamic(0, T_DYN)
    index_scores.bind_dynamic(0, T_DYN)
    selected_pools.bind_dynamic(0, T_DYN)
    selected_valid.bind_dynamic(0, T_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    index_scores.bind_dynamic(1, POOLS_DYN)
    indexer_proj(
        x,
        q_resid,
        w_q_b,
        w_k,
        k_norm_weight,
        k_norm_bias,
        w_weights,
        w_compress_gate,
        index_q,
        index_k,
        head_weights,
        gate_scores, pl.cast(0, pl.INT32),
    )
    indexer_cache_write(index_k, gate_scores, index_slots, raw_cache)
    indexer_pool_write(
        raw_cache, compress_ape, hadamard, pool_token_slots, pool_slots, pool_cache, pool_scale, pool_valid
    )
    indexer_score(
        index_q,
        hadamard,
        pool_cache,
        pool_scale,
        pool_blocks,
        head_weights,
        seg_start,
        pool_count,
        index_scores,
    )
    indexer_topk(index_scores, seg_start, pool_count, selected_pools, selected_valid)
    indexer_expand(selected_pools, selected_valid, tail_start, tail_count, kv_len, topk_indices)
    return (
        index_q,
        index_k,
        head_weights,
        gate_scores,
        raw_cache,
        pool_cache,
        pool_scale,
        pool_valid,
        index_scores,
        selected_pools,
        selected_valid,
        topk_indices,
    )


def build_indexer_proj_specs(tokens: int = 64):
    """Build one deterministic projection batch with all four weight groups."""
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(67)

    def init_x():
        return torch.randn(tokens, D, generator=generator, dtype=torch.float32).bfloat16()

    def init_q_resid():
        return torch.randn(tokens, Q_LORA, generator=generator, dtype=torch.float32).bfloat16()

    def init_w_q_b():
        weight = torch.randn(Q_LORA, INDEX_H * INDEX_DIM, generator=generator) * 0.02
        return weight.bfloat16()

    def init_w_k():
        return (torch.randn(D, INDEX_DIM, generator=generator) * 0.02).bfloat16()

    def init_k_norm_weight():
        return (1.0 + 0.1 * torch.randn(INDEX_DIM, generator=generator)).bfloat16()

    def init_k_norm_bias():
        return (0.1 * torch.randn(INDEX_DIM, generator=generator)).bfloat16()

    def init_w_weights():
        return (torch.randn(D, INDEX_H, generator=generator) * 0.05).bfloat16()

    def init_w_compress_gate():
        return (torch.randn(D, INDEX_DIM, generator=generator) * 0.02).bfloat16()

    return [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=init_x),
        TensorSpec("q_resid", [tokens, Q_LORA], torch.bfloat16, init_value=init_q_resid),
        TensorSpec("w_q_b", [Q_LORA, INDEX_H * INDEX_DIM], torch.bfloat16, init_value=init_w_q_b),
        TensorSpec("w_k", [D, INDEX_DIM], torch.bfloat16, init_value=init_w_k),
        TensorSpec("k_norm_weight", [INDEX_DIM], torch.bfloat16, init_value=init_k_norm_weight),
        TensorSpec("k_norm_bias", [INDEX_DIM], torch.bfloat16, init_value=init_k_norm_bias),
        TensorSpec("w_weights", [D, INDEX_H], torch.bfloat16, init_value=init_w_weights),
        TensorSpec("w_compress_gate", [D, INDEX_DIM], torch.bfloat16, init_value=init_w_compress_gate),
        TensorSpec("index_q", [tokens, INDEX_H, INDEX_DIM], torch.bfloat16),
        TensorSpec("index_k", [tokens, INDEX_DIM], torch.bfloat16),
        TensorSpec("head_weights", [tokens, INDEX_H], torch.float32),
        TensorSpec("gate_scores", [tokens, INDEX_DIM], torch.float32),
    ]


def _pool_batch(tokens: int, counts: tuple[int, ...], positions: tuple[int, ...], *, pad_pages: bool = True):
    """Derive the packed score/topk batch geometry from per-request pool counts.

    Returns the compacted width (leaf-aligned with slack), the pooled-table row
    count, the per-query ``seg_start`` and causally clamped ``pool_count``, and
    the pool row indirection. ``positions`` is each query's request-local
    position. ``pad_pages=False`` ends the page table at the last live page —
    the production contract — so the gathered row count is not a multiple of
    the score tile and the pack loop's final tile is partial.
    """
    pools_true = sum(counts)
    width = ((pools_true + 2 * LEAF - 2) // LEAF) * LEAF
    per_request = tokens // len(counts)
    if len(positions) != tokens or per_request * len(counts) != tokens:
        raise ValueError("positions must cover every token; counts must tile the batch")
    seg_start = torch.zeros(tokens, dtype=torch.int32)
    pool_count = torch.zeros(tokens, dtype=torch.int32)
    for index, count in enumerate(counts):
        for row in range(per_request):
            token = index * per_request + row
            seg_start[token] = sum(counts[:index])
            pool_count[token] = min((positions[token] + 1) // INDEX_KPOOL, count)
    # The pooled table is paged at INDEX_STATE_BLOCK_SIZE rows, the metadata
    # module's granularity: each compacted page lands at its own physical base
    # row, so the scorer's gather walks one entry per page. The page permutation
    # makes a compacted span physically discontinuous — the gather's real
    # workload. Table rows are therefore page-aligned. The default pads the
    # table to the leaf-aligned width; the production page table instead ends
    # at the last live page (``pad_pages=False``).
    if pad_pages:
        pages = width // INDEX_STATE_BLOCK_SIZE
    else:
        pages = (pools_true + INDEX_STATE_BLOCK_SIZE - 1) // INDEX_STATE_BLOCK_SIZE
    pool_table_rows = pages * INDEX_STATE_BLOCK_SIZE
    generator = torch.Generator().manual_seed(71)
    pool_blocks = torch.randperm(pages, generator=generator).to(torch.int32) * INDEX_STATE_BLOCK_SIZE
    return width, pool_table_rows, seg_start, pool_count, pool_blocks


def build_indexer_score_specs(
    tokens: int = 8,
    counts: tuple[int, ...] | None = None,
    positions: tuple[int, ...] | None = None,
    *,
    pad_pages: bool = True,
):
    """Build one deterministic two-request score batch.

    Request 0 owns 3000 pools so its longest query spans two sort leaves, and
    the query positions ramp from a first token (visible count 0) to a full
    window, so the causal clamp and the leaf-crossing window both execute.
    ``counts``/``positions`` override the fixture geometry for the
    business-shape benchmark points: one request whose positions are the
    ``history`` ramp in front of the measured chunk. ``pad_pages=False`` ends
    the pooled page table at the last live page instead of the leaf boundary.
    """
    from golden import TensorSpec

    counts = (3000, 140) if counts is None else counts
    positions = (2, 6, 5000, 11999, 3, 7, 500, 559) if positions is None else positions
    width, pool_table_rows, seg_start, pool_count, pool_blocks = _pool_batch(
        tokens, counts, positions, pad_pages=pad_pages
    )

    generator = torch.Generator().manual_seed(73)

    def init_index_q():
        return torch.randn(tokens, INDEX_H, INDEX_DIM, generator=generator).bfloat16()

    # One draw feeds both tables so every INT8 row and its scale stay paired;
    # the rows read as already Hadamard-rotated prior state, the way earlier
    # pool writes would have left them.
    prior_i8, prior_scale_col = quantize_per_token_int8(
        torch.randn(pool_table_rows, INDEX_DIM, generator=generator, dtype=torch.float32)
    )

    def init_pool_cache():
        return prior_i8

    def init_pool_scale():
        # The writer replicates the scale across all POOL_SCALE_WIDTH lanes and
        # the scorer reads the row sum scaled by 1/8, so the prior table must
        # carry the replication too.
        return prior_scale_col.expand(-1, POOL_SCALE_WIDTH).contiguous().clone()

    def init_head_weights():
        return torch.randn(tokens, INDEX_H, generator=generator)

    return [
        TensorSpec("index_q", [tokens, INDEX_H, INDEX_DIM], torch.bfloat16, init_value=init_index_q),
        TensorSpec("hadamard", [INDEX_DIM, INDEX_DIM], torch.bfloat16, init_value=sylvester_hadamard),
        TensorSpec("pool_cache", [pool_table_rows, INDEX_DIM], torch.int8, init_value=init_pool_cache),
        TensorSpec(
            "pool_scale", [pool_table_rows, POOL_SCALE_WIDTH], torch.float32, init_value=init_pool_scale
        ),
        TensorSpec(
            "pool_blocks",
            [width // INDEX_STATE_BLOCK_SIZE],
            torch.int32,
            init_value=lambda: pool_blocks,
        ),
        TensorSpec("head_weights", [tokens, INDEX_H], torch.float32, init_value=init_head_weights),
        TensorSpec("seg_start", [tokens], torch.int32, init_value=lambda: seg_start),
        TensorSpec("pool_count", [tokens], torch.int32, init_value=lambda: pool_count),
        TensorSpec("index_scores", [tokens, width], torch.float32),
    ]


def build_indexer_score_tail_specs():
    """Score batch whose gathered row count misses both score tile boundaries.

    Thirty-three live four-row pages hold 132 pooled rows, so the scale pack
    and per-query score loops both end with a partial 64-row tile. This is
    the production page-table tail shape that the leaf-padded fixtures above
    cannot reach. The position ramp climbs from an empty window through the
    64-row boundary to the full 132-row one.
    """
    return build_indexer_score_specs(8, (132,), (1, 3, 7, 15, 31, 63, 255, 527), pad_pages=False)


def build_indexer_topk_specs(
    tokens: int = 8,
    counts: tuple[int, ...] | None = None,
    positions: tuple[int, ...] | None = None,
):
    """Score-and-select on one batch: synthetic scores, real windows.

    The score rows are random values inside each query's window and
    ``FP32_NEG_INF`` outside it, which is exactly the shape :func:`indexer_score`
    leaves behind, so the sort sees realistic value structure without depending
    on the scorer's numerics. ``counts``/``positions`` override the fixture
    geometry for the business-shape benchmark points.
    """
    from golden import TensorSpec

    from models.glm5_3_flash.config import FP32_NEG_INF as NEG_INF

    counts = (3000, 140) if counts is None else counts
    positions = (2, 6, 5000, 11999, 3, 7, 500, 559) if positions is None else positions
    width, _, seg_start, pool_count, _ = _pool_batch(tokens, counts, positions)

    generator = torch.Generator().manual_seed(79)

    def init_index_scores():
        rows = torch.rand(tokens, width, generator=generator) * 10.0
        mask = torch.full((tokens, width), NEG_INF)
        for token in range(tokens):
            seg0 = int(seg_start[token])
            mask[token, seg0 : seg0 + int(pool_count[token])] = 1.0
        return rows * mask

    return [
        TensorSpec("index_scores", [tokens, width], torch.float32, init_value=init_index_scores),
        TensorSpec("seg_start", [tokens], torch.int32, init_value=lambda: seg_start),
        TensorSpec("pool_count", [tokens], torch.int32, init_value=lambda: pool_count),
        TensorSpec("selected_pools", [tokens, KPOOL_SELECT_K], torch.int32),
        TensorSpec("selected_valid", [tokens, KPOOL_SELECT_K], torch.int32),
    ]


def build_indexer_expand_specs(tokens: int = 8, lengths: tuple[int, ...] | None = None):
    """Expand selections across short rows, partial groups and a full row.

    The lengths ramp from a context too short to close one pool to one that
    fills all 512 selections, so the packing suffix, the tail window and the
    full-prefix case all execute in one batch. ``lengths`` overrides the ramp
    for the business-shape benchmark points (one full-width row per token).
    """
    from golden import TensorSpec

    lengths = (2, 3, 6, 15, 63, 1039, 2048, 2051) if lengths is None else lengths
    if len(lengths) != tokens:
        raise ValueError("lengths must cover every token")
    generator = torch.Generator().manual_seed(83)

    def init_selected_pools():
        picked = torch.randint(0, 64, (tokens, KPOOL_SELECT_K), generator=generator)
        return picked.to(torch.int32)

    def init_selected_valid():
        valid = torch.zeros(tokens, KPOOL_SELECT_K, dtype=torch.int32)
        for token, length in enumerate(lengths):
            keep = min(KPOOL_SELECT_K, length // INDEX_KPOOL)
            valid[token, :keep] = 1
        return valid

    def init_tail_start():
        return torch.tensor([length - length % INDEX_KPOOL for length in lengths], dtype=torch.int32)

    def init_tail_count():
        return torch.tensor([length % INDEX_KPOOL for length in lengths], dtype=torch.int32)

    def init_kv_len():
        return torch.tensor(lengths, dtype=torch.int32)

    return [
        TensorSpec(
            "selected_pools",
            [tokens, KPOOL_SELECT_K],
            torch.int32,
            init_value=init_selected_pools,
        ),
        TensorSpec(
            "selected_valid",
            [tokens, KPOOL_SELECT_K],
            torch.int32,
            init_value=init_selected_valid,
        ),
        TensorSpec("tail_start", [tokens], torch.int32, init_value=init_tail_start),
        TensorSpec("tail_count", [tokens], torch.int32, init_value=init_tail_count),
        TensorSpec("kv_len", [tokens], torch.int32, init_value=init_kv_len),
        TensorSpec("topk_indices", [tokens, TOPK_INDEX_WIDTH], torch.int32),
    ]


def golden_indexer_proj_case(tensors):
    """Fill the expected projections for :func:`build_indexer_proj_specs`."""
    index_q, index_k, head_weights, gate_scores = golden_indexer_proj(
        tensors["x"],
        tensors["q_resid"],
        tensors["w_q_b"],
        tensors["w_k"],
        tensors["k_norm_weight"],
        tensors["k_norm_bias"],
        tensors["w_weights"],
        tensors["w_compress_gate"],
    )
    tensors["index_q"][:] = index_q
    tensors["index_k"][:] = index_k
    tensors["head_weights"][:] = head_weights
    tensors["gate_scores"][:] = gate_scores


def golden_indexer_score_case(tensors):
    """Fill the expected scores for :func:`build_indexer_score_specs`.

    The golden emits one column per gathered row; lanes past that width stay
    ``FP32_NEG_INF`` under the same contract the kernel leaves them — the
    top-k clamps through its valid shapes, so only windows are compared.
    """
    from models.glm5_3_flash.config import FP32_NEG_INF as NEG_INF

    expected = golden_indexer_score(
        tensors["index_q"],
        tensors["hadamard"],
        tensors["pool_cache"],
        tensors["pool_scale"],
        tensors["pool_blocks"],
        tensors["head_weights"],
        tensors["seg_start"],
        tensors["pool_count"],
    )
    scores = tensors["index_scores"]
    scores[:] = NEG_INF
    scores[:, : expected.shape[1]] = expected


def golden_indexer_topk_case(tensors):
    """Fill the expected selections for :func:`build_indexer_topk_specs`."""
    selected, valid = golden_indexer_topk(
        tensors["index_scores"], tensors["seg_start"], tensors["pool_count"]
    )
    tensors["selected_pools"][:] = selected
    tensors["selected_valid"][:] = valid


def golden_indexer_expand_case(tensors):
    """Fill the expected expansion for :func:`build_indexer_expand_specs`."""
    tensors["topk_indices"][:] = golden_indexer_expand(
        tensors["selected_pools"],
        tensors["selected_valid"],
        tensors["tail_start"],
        tensors["tail_count"],
        tensors["kv_len"],
    )


def build_indexer_step_specs(tokens: int = 64, history: int = 0):
    """Build one deterministic single-request prefill chunk, end to end.

    One request holding ``history`` prior tokens receives a ``tokens``-row
    chunk: the chunk's rows scatter into the raw support table through a paged
    row map, every pool the chunk completes closes into the block-paged pooled
    table, and each chunk query scores the request's whole visible segment
    before the expansion. ``history=0`` is a fresh request's first chunk;
    otherwise the first closing pool mixes cached rows with fresh ones and the
    pooled table starts with the prior pools' rows — the checkpointed-chunk
    shape. ``tokens``/``history`` drive the business-shape benchmark points the
    same way the standalone cases' overrides do.
    """
    from golden import TensorSpec

    if tokens % PROJ_T_TILE:
        raise ValueError(f"the projection stage needs a token count that is a multiple of {PROJ_T_TILE}")
    total = history + tokens
    prior_pools = history // INDEX_KPOOL
    pools_true = total // INDEX_KPOOL
    events = pools_true - prior_pools
    width = ((pools_true + 2 * LEAF - 2) // LEAF) * LEAF

    generator = torch.Generator().manual_seed(97)

    # Paged raw rows for every logical position, past and new, the way the
    # metadata module hands them out: unordered pages, in-order rows.
    needed_pages = (total + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
    pages = torch.randperm(max(8, needed_pages + 3), generator=generator)[:needed_pages]

    def row_of(position):
        return int(pages[position // BLOCK_SIZE]) * BLOCK_SIZE + position % BLOCK_SIZE

    index_slots = torch.tensor([row_of(history + t) for t in range(tokens)], dtype=torch.int32)
    pool_token_slots = torch.tensor(
        [
            [row_of(INDEX_KPOOL * pool + k) for k in range(INDEX_KPOOL)]
            for pool in range(prior_pools, pools_true)
        ],
        dtype=torch.int32,
    )

    # The pooled table is paged at INDEX_STATE_BLOCK_SIZE rows like the
    # standalone fixtures: each compacted page lands at its own physical base,
    # and a pool's physical row derives from its compacted id.
    pages = width // INDEX_STATE_BLOCK_SIZE
    pool_table_rows = pages * INDEX_STATE_BLOCK_SIZE
    page_bases = torch.randperm(pages, generator=generator).to(torch.int32) * INDEX_STATE_BLOCK_SIZE

    def pool_row_of(compacted):
        return int(page_bases[compacted // INDEX_STATE_BLOCK_SIZE]) + compacted % INDEX_STATE_BLOCK_SIZE

    pool_slots = torch.tensor(
        [pool_row_of(pool) for pool in range(prior_pools, pools_true)], dtype=torch.int32
    )

    seg_start = torch.zeros(tokens, dtype=torch.int32)
    pool_count = torch.zeros(tokens, dtype=torch.int32)
    tail_start = torch.zeros(tokens, dtype=torch.int32)
    tail_count = torch.zeros(tokens, dtype=torch.int32)
    kv_len = torch.zeros(tokens, dtype=torch.int32)
    for t in range(tokens):
        length = history + t + 1
        tail_count[t] = length % INDEX_KPOOL
        tail_start[t] = length - int(tail_count[t])
        pool_count[t] = min(length // INDEX_KPOOL, pools_true)
        kv_len[t] = length

    shapes = {
        "w_q_b": (Q_LORA, INDEX_H * INDEX_DIM),
        "w_k": (D, INDEX_DIM),
        "w_weights": (D, INDEX_H),
        "w_compress_gate": (D, INDEX_DIM),
    }

    def init_weights(name, scale):
        def build():
            return (torch.randn(*shapes[name], generator=generator) * scale).bfloat16()

        return build

    raw_rows = int(max(index_slots.max(), pool_token_slots.max())) + 1

    def init_raw_cache():
        # History rows arrive as earlier steps left them: BF16-projected keys
        # widened back into the FP32 storage ABI beside FP32 gates.
        cache = torch.randn(raw_rows, INDEX_STATE_WIDTH, generator=generator)
        cache[:, :INDEX_DIM] = cache[:, :INDEX_DIM].bfloat16().float()
        return cache

    # The pooled table starts as the quantized form earlier steps would have
    # left behind: INT8 rows already rotated into the Hadamard basis, paired
    # with their dequant scales, one draw feeding both so the pairs stay
    # consistent.
    prior_i8, prior_scale_col = quantize_per_token_int8(
        torch.randn(pool_table_rows, INDEX_DIM, generator=generator, dtype=torch.float32)
    )

    def init_pool_scale():
        # The writer replicates the scale across all POOL_SCALE_WIDTH lanes and
        # the scorer reads the row sum scaled by 1/8, so the prior table must
        # carry the replication too.
        return prior_scale_col.expand(-1, POOL_SCALE_WIDTH).contiguous().clone()

    specs = [
        TensorSpec(
            "x",
            [tokens, D],
            torch.bfloat16,
            init_value=lambda: torch.randn(tokens, D, generator=generator, dtype=torch.float32).bfloat16(),
        ),
        TensorSpec(
            "q_resid",
            [tokens, Q_LORA],
            torch.bfloat16,
            init_value=lambda: torch.randn(
                tokens, Q_LORA, generator=generator, dtype=torch.float32
            ).bfloat16(),
        ),
        TensorSpec(
            "w_q_b",
            [Q_LORA, INDEX_H * INDEX_DIM],
            torch.bfloat16,
            init_value=init_weights("w_q_b", 0.02),
        ),
        TensorSpec("w_k", [D, INDEX_DIM], torch.bfloat16, init_value=init_weights("w_k", 0.02)),
        TensorSpec(
            "k_norm_weight",
            [INDEX_DIM],
            torch.bfloat16,
            init_value=lambda: (1.0 + 0.1 * torch.randn(INDEX_DIM, generator=generator)).bfloat16(),
        ),
        TensorSpec(
            "k_norm_bias",
            [INDEX_DIM],
            torch.bfloat16,
            init_value=lambda: (0.1 * torch.randn(INDEX_DIM, generator=generator)).bfloat16(),
        ),
        TensorSpec("w_weights", [D, INDEX_H], torch.bfloat16, init_value=init_weights("w_weights", 0.05)),
        TensorSpec(
            "w_compress_gate",
            [D, INDEX_DIM],
            torch.bfloat16,
            init_value=init_weights("w_compress_gate", 0.02),
        ),
        TensorSpec("hadamard", [INDEX_DIM, INDEX_DIM], torch.bfloat16, init_value=sylvester_hadamard),
        TensorSpec(
            "compress_ape",
            [INDEX_KPOOL, INDEX_DIM],
            torch.bfloat16,
            init_value=lambda: torch.randn(INDEX_KPOOL, INDEX_DIM, generator=generator).bfloat16(),
        ),
        TensorSpec("index_slots", [tokens], torch.int32, init_value=lambda: index_slots),
        TensorSpec(
            "pool_token_slots",
            [events, INDEX_KPOOL],
            torch.int32,
            init_value=lambda: pool_token_slots,
        ),
        TensorSpec("pool_slots", [events], torch.int32, init_value=lambda: pool_slots),
        TensorSpec("pool_blocks", [pages], torch.int32, init_value=lambda: page_bases),
        TensorSpec("seg_start", [tokens], torch.int32, init_value=lambda: seg_start),
        TensorSpec("pool_count", [tokens], torch.int32, init_value=lambda: pool_count),
        TensorSpec("tail_start", [tokens], torch.int32, init_value=lambda: tail_start),
        TensorSpec("tail_count", [tokens], torch.int32, init_value=lambda: tail_count),
        TensorSpec("kv_len", [tokens], torch.int32, init_value=lambda: kv_len),
        TensorSpec("raw_cache", [raw_rows, INDEX_STATE_WIDTH], torch.float32, init_value=init_raw_cache),
        TensorSpec("pool_cache", [pool_table_rows, INDEX_DIM], torch.int8, init_value=lambda: prior_i8),
        TensorSpec(
            "pool_scale",
            [pool_table_rows, POOL_SCALE_WIDTH],
            torch.float32,
            init_value=init_pool_scale,
        ),
        TensorSpec("pool_valid", [events, POOL_VALID_WIDTH], torch.int32),
        TensorSpec("index_q", [tokens, INDEX_H, INDEX_DIM], torch.bfloat16),
        TensorSpec("index_k", [tokens, INDEX_DIM], torch.bfloat16),
        TensorSpec("head_weights", [tokens, INDEX_H], torch.float32),
        TensorSpec("gate_scores", [tokens, INDEX_DIM], torch.float32),
        TensorSpec("index_scores", [tokens, width], torch.float32),
        TensorSpec("selected_pools", [tokens, KPOOL_SELECT_K], torch.int32),
        TensorSpec("selected_valid", [tokens, KPOOL_SELECT_K], torch.int32),
        TensorSpec("topk_indices", [tokens, TOPK_INDEX_WIDTH], torch.int32),
    ]
    return specs


def golden_indexer_prefill_step_case(tensors):
    """Fill every expected output of one prefill chunk by composing the goldens."""
    index_q, index_k, head_weights, gate_scores = golden_indexer_proj(
        tensors["x"],
        tensors["q_resid"],
        tensors["w_q_b"],
        tensors["w_k"],
        tensors["k_norm_weight"],
        tensors["k_norm_bias"],
        tensors["w_weights"],
        tensors["w_compress_gate"],
    )
    tensors["index_q"][:] = index_q
    tensors["index_k"][:] = index_k
    tensors["head_weights"][:] = head_weights
    tensors["gate_scores"][:] = gate_scores
    tensors["raw_cache"][:] = golden_indexer_cache_write(
        tensors["raw_cache"], index_k, gate_scores, tensors["index_slots"]
    )
    pool_cache, pool_scale, pool_valid = golden_indexer_pool_write(
        tensors["pool_cache"],
        tensors["pool_scale"],
        tensors["raw_cache"],
        tensors["compress_ape"],
        tensors["hadamard"],
        tensors["pool_token_slots"],
        tensors["pool_slots"],
    )
    tensors["pool_cache"][:] = pool_cache
    tensors["pool_scale"][:] = pool_scale
    tensors["pool_valid"][:] = pool_valid
    tensors["index_scores"][:] = golden_indexer_score(
        tensors["index_q"],
        tensors["hadamard"],
        tensors["pool_cache"],
        tensors["pool_scale"],
        tensors["pool_blocks"],
        tensors["head_weights"],
        tensors["seg_start"],
        tensors["pool_count"],
    )
    selected, valid = golden_indexer_topk(
        tensors["index_scores"], tensors["seg_start"], tensors["pool_count"]
    )
    tensors["selected_pools"][:] = selected
    tensors["selected_valid"][:] = valid
    tensors["topk_indices"][:] = golden_indexer_expand(
        tensors["selected_pools"],
        tensors["selected_valid"],
        tensors["tail_start"],
        tensors["tail_count"],
        tensors["kv_len"],
    )
    for token in range(tensors["topk_indices"].shape[0]):
        row = tensors["topk_indices"][token]
        live = int((row >= 0).sum())
        assert (row[:live] >= 0).all() and (row[live:] == -1).all(), (
            f"row {token} violates the front-packed index ABI"
        )


def _self_check() -> None:
    """Exercise the four goldens on CPU with inlined fixtures.

    Kept local rather than in ``_golden_smoke.py`` because that module is shared
    across the whole model directory.
    """
    torch.manual_seed(85)
    heads, dim, hidden = 4, 16, 64
    x = torch.randn(8, hidden).bfloat16()
    q_resid = torch.randn(8, 32).bfloat16()
    w_q_b = torch.randn(32, heads * dim).bfloat16() * 0.1
    w_k = torch.randn(hidden, dim).bfloat16() * 0.1
    k_w = torch.randn(dim).bfloat16()
    k_b = torch.randn(dim).bfloat16()
    index_q, index_k, _, _ = golden_indexer_proj(
        x,
        q_resid,
        w_q_b,
        w_k,
        k_w,
        k_b,
        torch.randn(hidden, heads).bfloat16() * 0.1,
        torch.randn(hidden, dim).bfloat16() * 0.1,
    )
    assert index_q.shape == (8, heads, dim), index_q.shape
    assert index_k.dtype is torch.bfloat16
    manual_q = torch.nn.functional.linear(q_resid.float(), w_q_b.t().float())
    assert torch.allclose(index_q.float().flatten(1), manual_q, atol=2e-2), "q projection drifted"
    raw = torch.nn.functional.linear(x.float(), w_k.t().float())
    manual_k = torch.nn.functional.layer_norm(raw, (dim,), k_w.float(), k_b.float(), K_NORM_EPS)
    assert torch.allclose(index_k.float(), manual_k.to(torch.bfloat16).float(), atol=2e-2)

    scores = torch.tensor([[9.0, 9.0, 5.0, 1.0, 3.0, 0.0], [0.0, 0.0, 9.0, 1.0, 5.0, 1.0]])
    seg = torch.tensor([0, 2], dtype=torch.int32)
    count = torch.tensor([3, 4], dtype=torch.int32)
    selected, valid = golden_indexer_topk(scores, seg, count)
    assert valid[:, 0].tolist() == [1, 1] and valid[0, 3].item() == 0
    assert selected[0, 0] in (0, 1) and selected[0, 2] == 2, selected[0]

    picked = torch.tensor([[2, 1, 0, 0]], dtype=torch.int32)
    picked_valid = torch.tensor([[1, 1, 1, 0]], dtype=torch.int32)
    tail_start = torch.tensor([12], dtype=torch.int32)
    tail_count = torch.tensor([2], dtype=torch.int32)
    kv = torch.tensor([14], dtype=torch.int32)
    expanded = golden_indexer_expand(picked, picked_valid, tail_start, tail_count, kv)
    expected = [8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 12, 13]
    assert expanded[0, :14].tolist() == expected, expanded[0, :14]
    assert (expanded[0, 14:] == -1).all(), "padding is not a pure suffix"
    print("[GOLDEN] PASS prefill_indexer self-check")


def main():
    """Prove the goldens on CPU, then validate each stage on device.

    The projections round once to BF16 or hold FP32, so their budget is one
    BF16 ulp with a small outlier allowance. The scorer carries the INT8
    quantization of both operands, so a tie-swapped top-k is judged through the
    paired-score comparator rather than index equality. The expansion is pure
    integer lane math and compares exactly.
    """
    import argparse

    from golden import ratio_allclose, run, topk_pair_compare

    def score_window_compare(actual, expected, *, inputs, rtol, atol, **_kwargs):
        """Compare only each query's visible window: the kernel leaves every
        other lane as undefined scratch (the top-k clamps through its valid
        shapes), so out-of-window contents carry no contract to check."""

        seg = inputs["seg_start"].long().unsqueeze(1)
        end = seg + inputs["pool_count"].long().clamp(min=0).unsqueeze(1)
        lanes = torch.arange(actual.shape[-1]).unsqueeze(0)
        window = (lanes >= seg) & (lanes < end)
        picked_a = actual.float()[window]
        picked_e = expected.float()[window]
        if picked_a.numel() == 0:
            return True, ""
        bad = ~torch.isfinite(picked_a)
        bad |= (picked_a - picked_e).abs() > (atol + rtol * picked_e.abs())
        ratio = bad.float().mean().item()
        ok = ratio <= 0.01
        return ok, "" if ok else f"    in-window mismatch ratio {ratio:.4f}"

    _self_check()

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--case",
        default="all",
        choices=["proj", "score", "topk", "expand", "step", "all"],
    )
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument(
        "--bench",
        action="store_true",
        help="run the case at a business shape without golden validation; the shape "
        "comes from --tokens/--history, timing from PYPTO_BENCH=1",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=0,
        help="history tokens in front of the --tokens chunk; defines the bench geometry",
    )
    args = parser.parse_args()

    def print_bench(label: str, result) -> None:
        stats = result.bench
        if stats is None:
            print(f"[BENCH] {label}: no timing (run with PYPTO_BENCH=1)")
            return
        print(
            f"[BENCH] {label}: device_us median={stats.device_us_median:.1f} "
            f"min={stats.device_us_min:.1f} mean={stats.device_us_mean:.1f} "
            f"max={stats.device_us_max:.1f} rounds={stats.rounds}"
        )

    def bench_label(name: str, pools: int = 0) -> str:
        label = f"{name} T={args.tokens} history={args.history}"
        if pools:
            width = ((pools + 2 * LEAF - 2) // LEAF) * LEAF
            label += f" pools={pools} width={width}"
        return label

    bench_counts = None
    bench_positions = None
    if args.bench:
        pools = (args.tokens + args.history) // INDEX_KPOOL
        bench_counts = (pools,)
        bench_positions = tuple(range(args.history, args.history + args.tokens))
        if args.tokens % 64:
            parser.error("--bench needs a token count that is a multiple of 64")

    def _paired_selected_scores(scores, seg0, selected):
        """Gather each row's selected scores; padding lanes read as ``-inf``."""
        cols = (selected.long() + seg0).clamp(0, scores.shape[-1] - 1)
        paired = torch.gather(scores, 1, cols)
        return torch.where(selected < 0, torch.full_like(paired, -torch.inf), paired)

    def _selection_compare(scores, actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        """Tie-aware selection compare that also audits the picked scores.

        :func:`topk_pair_compare` adjudicates index mismatches through the
        actual ordering alone, so a sorted strictly-worse subset would pass
        it. The picked scores must therefore also match the scores the golden
        picked, under the same tolerance: a legal tie swap keeps the paired
        values equal, a real miss breaks them.
        """
        seg0 = inputs["seg_start"].long().unsqueeze(1)
        paired = _paired_selected_scores(scores, seg0, actual)
        synth_outputs = {**actual_outputs, "_selected_paired_scores": paired}
        ok, msg = topk_pair_compare("_selected_paired_scores")(
            actual,
            expected,
            actual_outputs=synth_outputs,
            expected_outputs=expected_outputs,
            inputs=inputs,
            rtol=rtol,
            atol=atol,
        )
        if not ok:
            return ok, msg
        expected_paired = _paired_selected_scores(scores, seg0, expected)
        if not torch.allclose(paired, expected_paired, rtol=rtol, atol=atol):
            diff = (paired - expected_paired).abs()
            diff = torch.where(torch.isfinite(diff), diff, torch.full_like(diff, torch.inf))
            return False, (
                "    selected scores differ from the golden-selected scores "
                f"(max |diff|={diff.max().item():.6g})"
            )
        return True, ""

    def selected_pools_compare(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        return _selection_compare(
            inputs["index_scores"].float(),
            actual,
            expected,
            actual_outputs=actual_outputs,
            expected_outputs=expected_outputs,
            inputs=inputs,
            rtol=rtol,
            atol=atol,
        )

    def exact_compare(actual, expected, **_kwargs):
        exact = torch.equal(actual.cpu(), expected.cpu())
        return exact, "" if exact else "    integer output differs from golden"

    def fused_selected_compare(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        """Tie-aware selection compare reading the scores the run produced.

        The composed step emits ``index_scores`` as an output, so the paired
        scores for the top-k check come from ``actual_outputs`` — the same
        rule :func:`selected_pools_compare` applies with an input matrix.
        """
        return _selection_compare(
            actual_outputs["index_scores"].float(),
            actual,
            expected,
            actual_outputs=actual_outputs,
            expected_outputs=expected_outputs,
            inputs=inputs,
            rtol=rtol,
            atol=atol,
        )

    def step_topk_indices_compare(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        """Tie-aware position compare for the composed step's expanded rows.

        A near-boundary score cluster can reshuffle two pools whose scores each
        sit inside the scorer's own tolerance; the swap is legal exactly when
        the swapped positions' paired scores agree under the same allclose rule
        that certified ``index_scores``. Tail and padding lanes are
        deterministic given the inputs, so any mismatch there fails outright,
        and every actual row must keep the front-packed ABI.
        """
        a = actual.cpu()
        e = expected.cpu()
        if torch.equal(a, e):
            return True, ""
        scores = actual_outputs["index_scores"].float().cpu()
        seg0 = inputs["seg_start"].long().unsqueeze(1)
        kv_len = inputs["kv_len"].long().unsqueeze(1)
        hist_len = (kv_len // INDEX_KPOOL) * INDEX_KPOOL
        live = (a >= 0).sum(dim=1)
        rows = a.shape[0]
        for row in range(rows):
            if not ((a[row, : live[row]] >= 0).all() and (a[row, live[row] :] == -1).all()):
                return False, f"    row {row} violates the front-packed ABI"

        def lane_scores(x):
            pool = x.clamp(min=0) // INDEX_KPOOL
            cols = (pool + seg0).clamp(0, scores.shape[1] - 1)
            paired = torch.gather(scores, 1, cols)
            is_hist = (x >= 0) & (x < hist_len)
            return torch.where(is_hist, paired, torch.full_like(paired, -torch.inf))

        mismatch = a != e
        both_hist = (a >= 0) & (a < hist_len) & (e >= 0) & (e < hist_len)
        sa, se = lane_scores(a), lane_scores(e)
        tie_legal = (sa - se).abs() <= atol + rtol * torch.maximum(sa.abs(), se.abs())
        illegal = mismatch & ~(both_hist & tie_legal)
        if illegal.any():
            n = int(illegal.sum())
            first = illegal.nonzero()[0].tolist()
            return False, f"    {n} lane(s) differ beyond the score tolerance, first at {first}"
        return True, ""

    results = []
    if args.case in ("proj", "all"):
        if args.bench:
            extra = dict(golden_fn=None)
            label = bench_label("proj")
        else:
            extra = dict(
                golden_fn=golden_indexer_proj_case,
                rtol=1.0 / 64,
                atol=1e-3,
                compare_fn={
                    "index_q": ratio_allclose(atol=1e-3, rtol=1.0 / 64, max_error_ratio=0.01),
                    "index_k": ratio_allclose(atol=1e-3, rtol=1.0 / 64, max_error_ratio=0.01),
                    "head_weights": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.01),
                    "gate_scores": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.01),
                },
            )
        results.append(
            run(
                fn=indexer_proj_test,
                specs=build_indexer_proj_specs(args.tokens),
                config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
                compile_only=args.compile_only,
                **extra,
            )
        )
        if args.bench:
            print_bench(label, results[-1])
    if args.case in ("score", "all"):
        if args.bench:
            specs = build_indexer_score_specs(args.tokens, bench_counts, bench_positions)
            extra = dict(golden_fn=None)
            label = bench_label("score", bench_counts[0])
        else:
            specs = build_indexer_score_specs()
            extra = dict(
                golden_fn=golden_indexer_score_case,
                rtol=1.0 / 128,
                atol=1e-4,
                compare_fn={
                    "index_scores": score_window_compare,
                },
            )
        results.append(
            run(
                fn=indexer_score_test,
                specs=specs,
                config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
                compile_only=args.compile_only,
                **extra,
            )
        )
        if args.bench:
            print_bench(label, results[-1])
        else:
            # The production page table ends at the last live page, so the
            # gathered row count need not be a SCORE_C_TILE multiple; run the
            # partial-tail-tile shape the padded fixtures cannot reach.
            results.append(
                run(
                    fn=indexer_score_test,
                    specs=build_indexer_score_tail_specs(),
                    golden_fn=golden_indexer_score_case,
                    rtol=1.0 / 128,
                    atol=1e-4,
                    compare_fn={
                        "index_scores": score_window_compare,
                    },
                    config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
                    compile_only=args.compile_only,
                )
            )
            # Stable request slots can leave a large leading pooled-key gap.
            # Only sixteen queries are live in this capacity-192 batch.
            sparse_specs = build_indexer_score_specs(192, (388,), (0,) * 192)
            sparse_segments = torch.zeros(192, dtype=torch.int32)
            sparse_segments[:16] = 384
            sparse_counts = torch.zeros(192, dtype=torch.int32)
            sparse_counts[:16] = torch.arange(1, 17, dtype=torch.int32) // INDEX_KPOOL
            for spec in sparse_specs:
                if spec.name == "seg_start":
                    spec.init_value = sparse_segments
                elif spec.name == "pool_count":
                    spec.init_value = sparse_counts
            results.append(
                run(
                    fn=indexer_score_test,
                    specs=sparse_specs,
                    config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
                    compile_only=args.compile_only,
                    **extra,
                )
            )
    if args.case in ("topk", "all"):
        if args.bench:
            specs = build_indexer_topk_specs(args.tokens, bench_counts, bench_positions)
            extra = dict(golden_fn=None)
            label = bench_label("topk", bench_counts[0])
        else:
            specs = build_indexer_topk_specs()
            extra = dict(
                golden_fn=golden_indexer_topk_case,
                rtol=1e-3,
                atol=1e-3,
                compare_fn={
                    "selected_pools": selected_pools_compare,
                    "selected_valid": exact_compare,
                },
            )
        results.append(
            run(
                fn=indexer_topk_test,
                specs=specs,
                config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
                compile_only=args.compile_only,
                **extra,
            )
        )
        if args.bench:
            print_bench(label, results[-1])
    if args.case in ("expand", "all"):
        if args.bench:
            specs = build_indexer_expand_specs(args.tokens, (args.tokens + args.history,) * args.tokens)
            extra = dict(golden_fn=None)
            label = bench_label("expand")
        else:
            specs = build_indexer_expand_specs()
            extra = dict(
                golden_fn=golden_indexer_expand_case,
                rtol=0.0,
                atol=0.0,
            )
        results.append(
            run(
                fn=indexer_expand_test,
                specs=specs,
                config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
                compile_only=args.compile_only,
                **extra,
            )
        )
        if args.bench:
            print_bench(label, results[-1])
    if args.case in ("step", "all"):
        step_compare = {
            "index_q": ratio_allclose(atol=1e-3, rtol=1.0 / 64, max_error_ratio=0.01),
            "index_k": ratio_allclose(atol=1e-3, rtol=1.0 / 64, max_error_ratio=0.01),
            "head_weights": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.01),
            "gate_scores": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.01),
            # The raw rows carry the projection rounding of the composed chunk,
            # unlike the standalone scatter, which moves bytes at zero tolerance.
            "raw_cache": ratio_allclose(atol=1e-3, rtol=1.0 / 64, max_error_ratio=0.01),
            # The pooled table is quantized on write; the softmax transcendentals
            # can flip a rint boundary, so allow a tiny tail of one-quantum flips.
            "pool_cache": ratio_allclose(atol=1.0, rtol=0.0, max_error_ratio=0.005),
            "pool_scale": ratio_allclose(atol=1e-6, rtol=1e-3, max_error_ratio=0.005),
            "pool_valid": exact_compare,
            "index_scores": score_window_compare,
            "selected_pools": fused_selected_compare,
            "selected_valid": exact_compare,
            "topk_indices": step_topk_indices_compare,
        }
        # Two validation geometries: the request's first chunk, then a
        # checkpointed chunk whose first closing pool mixes two cached rows
        # with two fresh ones (history not a multiple of the pool size).
        if args.bench or args.history:
            tiers = [(args.tokens, args.history)]
        else:
            tiers = [(args.tokens, 0), (args.tokens, 4 * args.tokens - 6)]
        for tier_tokens, tier_history in tiers:
            specs = build_indexer_step_specs(tier_tokens, tier_history)
            if args.bench:
                pools = (tier_tokens + tier_history) // INDEX_KPOOL
                extra = dict(golden_fn=None)
                label = bench_label("step", pools)
            else:
                extra = dict(
                    golden_fn=golden_indexer_prefill_step_case,
                    rtol=1.0 / 64,
                    atol=1e-3,
                    compare_fn=step_compare,
                )
            results.append(
                run(
                    fn=indexer_prefill_step_test,
                    specs=specs,
                    config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
                    compile_only=args.compile_only,
                    **extra,
                )
            )
            if args.bench:
                print_bench(label, results[-1])
    for result in results:
        print(result)
        if not result.passed:
            raise SystemExit(result.error or 1)


__all__ = [
    "LEAF",
    "PAIR_WIDTH",
    "build_indexer_expand_specs",
    "build_indexer_proj_specs",
    "build_indexer_score_specs",
    "build_indexer_step_specs",
    "build_indexer_topk_specs",
    "golden_indexer_expand",
    "golden_indexer_prefill_step_case",
    "golden_indexer_proj",
    "golden_indexer_score",
    "golden_indexer_topk",
    "indexer_expand",
    "indexer_expand_test",
    "indexer_prefill_step_test",
    "indexer_proj",
    "indexer_proj_test",
    "indexer_score",
    "indexer_score_test",
    "indexer_topk",
    "indexer_topk_test",
    "sylvester_hadamard",
]


if __name__ == "__main__":
    main()
