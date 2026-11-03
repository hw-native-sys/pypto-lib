# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GLM-5.3-Flash indexer projection, pooling and token selection."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from dataclasses import dataclass

import pypto.language as pl
import torch
import torch.nn.functional as F

from models.glm5_3_flash.attention import (
    DECODE_ROWS_PER_REQUEST, decode_sparse_attn_test, golden_decode_sparse_attn_case,
)
from models.glm5_3_flash.config import (
    BLOCK_SIZE, INDEX_DIM, INDEX_H, INDEX_KPOOL, INDEX_STATE_BLOCK_SIZE, INDEX_STATE_WIDTH,
    KPOOL_SELECT_K, KV_LORA, LOCAL_H, Q_LORA, TOPK_INDEX_WIDTH, D, Glm53FlashConfig,
)
from models.glm5_3_flash.elementwise import FP32_NEG_INF
from models.glm5_3_flash.quantization import INT8_AMAX_EPS, INT8_SCALE_MAX, quantize_per_token_int8
from models.glm5_3_flash.utils import (
    _FLOAT32_MIN, _randn, build_forward_metadata, build_indexer_chunk_metadata, sylvester_hadamard,
)

INDEX_BLOCKS_DYN = pl.dynamic("GLM53_INDEX_BLOCKS_DYN")
POOLS_DYN = pl.dynamic("GLM53_POOLS_DYN")
POOL_EVENTS_DYN = pl.dynamic("GLM53_POOL_EVENTS_DYN")
POOL_PAGES_DYN = pl.dynamic("GLM53_POOL_PAGES_DYN")
TABLE_DYN = pl.dynamic("GLM53_TABLE_DYN")
T_DYN = pl.dynamic("GLM53_T_DYN")

SORT_TILE = 2048
PAIR_WIDTH = 2 * KPOOL_SELECT_K
K_NORM_EPS = 1e-6
SOFTMAX_SCALE = INDEX_DIM**-0.5
WEIGHTS_SCALE = INDEX_H**-0.5
PROJ_T_TILE = 16
PROJ_K_TILE = 128
Q_OUT_TILE = 256
QH_MM_TILE = 64
SCORE_C_TILE = 64
EXPAND_W_TILE = 256
EXPAND_TAIL_TILE = 8  # covers the last lanes in one aligned tile
EXPAND_TOKEN_TILE = 4
TOPK_QUERY_WORKERS = 48
POOL_GATHER_PAGE_TILE = 16
POOL_GATHER_ROW_TILE = POOL_GATHER_PAGE_TILE * INDEX_STATE_BLOCK_SIZE
POOL_GATHER_WORKERS = 48
POOL_SCALE_WORKERS = 48
# Batches at or below this token count take the decode-shaped dispatch:
# persistent workers own pool-gather chunks and scale blocks. Larger prefill
# chunks dispatch one task per chunk/block instead. The threshold matches the
# resident decode batch of 32 requests of ``DECODE_ROWS_PER_REQUEST`` rows.
INDEXER_PERSISTENT_MAX_TOKENS = 32 * DECODE_ROWS_PER_REQUEST

POOL_STAGE_DYN = pl.dynamic("GLM53_POOL_STAGE_DYN")
POOL_VALID_WIDTH = 8
POOL_SCALE_WIDTH = 8
POOL_QUANT_TILE = 64
# Rows per cache-write block: the key half loads as one wide BF16 tile and
# widens to FP32 once, then each row scatters by its own slot.
CACHE_WRITE_TILE = 8


@pl.jit
def decode_indexer_step_test(
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
    """Run one whole decode step for golden.run validation.

    The raw scatter, the pool close, the score, the selection and the expansion
    order themselves through their shared tensors; every stage's output is
    exposed so the harness can validate the composition, not just the tail.
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
        x, q_resid, w_q_b, w_k, k_norm_weight, k_norm_bias, w_weights, w_compress_gate, index_q, index_k,
        head_weights, gate_scores, pl.cast(0, pl.INT32),
    )
    indexer_cache_write(index_k, gate_scores, index_slots, raw_cache)
    indexer_pool_write(
        raw_cache, compress_ape, hadamard, pool_token_slots, pool_slots, pool_cache, pool_scale, pool_valid,
    )
    indexer_score(
        index_q, hadamard, pool_cache, pool_scale, pool_blocks, head_weights, seg_start, pool_count,
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
        x, q_resid, w_q_b, w_k, k_norm_weight, k_norm_bias, w_weights, w_compress_gate, index_q, index_k,
        head_weights, gate_scores, pl.cast(0, pl.INT32),
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
        index_q, hadamard, pool_cache, pool_scale, pool_blocks, head_weights, seg_start, pool_count,
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
        x, q_resid, w_q_b, w_k, k_norm_weight, k_norm_bias, w_weights, w_compress_gate, index_q, index_k,
        head_weights, gate_scores, pl.cast(0, pl.INT32),
    )
    indexer_cache_write(index_k, gate_scores, index_slots, raw_cache)
    indexer_pool_write(
        raw_cache, compress_ape, hadamard, pool_token_slots, pool_slots, pool_cache, pool_scale, pool_valid,
    )
    indexer_score(
        index_q, hadamard, pool_cache, pool_scale, pool_blocks, head_weights, seg_start, pool_count,
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


@pl.jit.inline
def indexer_proj(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    q_resid: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    w_q_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.BF16],
    w_k: pl.Tensor[[D, INDEX_DIM], pl.BF16],
    k_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    k_norm_bias: pl.Tensor[[INDEX_DIM], pl.BF16],
    w_weights: pl.Tensor[[D, INDEX_H], pl.BF16],
    w_compress_gate: pl.Tensor[[D, INDEX_DIM], pl.BF16],
    index_q: pl.Tensor[[T_DYN, INDEX_H, INDEX_DIM], pl.BF16],
    index_k: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    head_weights: pl.Tensor[[T_DYN, INDEX_H], pl.FP32],
    gate_scores: pl.Tensor[[T_DYN, INDEX_DIM], pl.FP32],
    skip_query: pl.Scalar[pl.INT32],
):
    """Project one dispatch's rows: queries, biased LayerNorm keys, both gates.

    ``T`` must be a multiple of ``PROJ_T_TILE`` = 16; the scheduler pads a
    packed batch up to it and feeds the padding rows ordinary metadata. Four
    independent scopes, one cube accumulator each, so the compiler can lay them
    out concurrently.
    """
    t_dim = pl.tensor.dim(x, 0)
    if skip_query == 0:
        q_flat = pl.reshape(index_q, [t_dim, INDEX_H * INDEX_DIM])
        q_blocks = (INDEX_H * INDEX_DIM) // Q_OUT_TILE

        for idx in pl.spmd((t_dim // PROJ_T_TILE) * q_blocks, name_hint="indexer_q_proj"):
            t_block = idx // q_blocks
            t0 = t_block * PROJ_T_TILE
            o0 = (idx - t_block * q_blocks) * Q_OUT_TILE
            q_acc = pl.create_tensor([PROJ_T_TILE, Q_OUT_TILE], dtype=pl.FP32)
            for kb in pl.pipeline(0, Q_LORA // PROJ_K_TILE, stage=2):
                k0 = kb * PROJ_K_TILE
                q_acc = pl.matmul_acc(q_acc, q_resid[t0:t0 + PROJ_T_TILE, k0:k0 + PROJ_K_TILE], w_q_b[k0:k0 + PROJ_K_TILE, o0:o0 + Q_OUT_TILE], init_cond=k0 == 0)
            q_flat[t0 : t0 + PROJ_T_TILE, o0 : o0 + Q_OUT_TILE] = pl.cast(q_acc, pl.BF16, mode="rint")

    for idx in pl.spmd(t_dim // PROJ_T_TILE, name_hint="indexer_k_proj"):
        t0 = idx * PROJ_T_TILE
        k_acc = pl.create_tensor([PROJ_T_TILE, INDEX_DIM], dtype=pl.FP32)
        for db in pl.pipeline(0, D // PROJ_K_TILE, stage=2):
            d0 = db * PROJ_K_TILE
            k_acc = pl.matmul_acc(k_acc, x[t0:t0 + PROJ_T_TILE, d0:d0 + PROJ_K_TILE], w_k[d0:d0 + PROJ_K_TILE, 0:INDEX_DIM], init_cond=d0 == 0)
        mean = pl.mul(pl.row_sum(k_acc), 1.0 / INDEX_DIM)
        centered = pl.row_expand_sub(k_acc, mean)
        variance = pl.mul(pl.row_sum(pl.mul(centered, centered)), 1.0 / INDEX_DIM)
        rstd = pl.recip(pl.sqrt(pl.add(variance, K_NORM_EPS)))
        normed = pl.row_expand_mul(centered, rstd)
        weight_row = pl.cast(pl.reshape(k_norm_weight, [1, INDEX_DIM])[0:1, 0:INDEX_DIM], pl.FP32, mode="none")
        bias_row = pl.cast(pl.reshape(k_norm_bias, [1, INDEX_DIM])[0:1, 0:INDEX_DIM], pl.FP32, mode="none")
        index_k_value_col_expand_mul = pl.col_expand_mul(normed, weight_row)
        index_k_value_col_expand_add = pl.col_expand_add(index_k_value_col_expand_mul, bias_row)
        index_k[t0:t0 + PROJ_T_TILE, 0:INDEX_DIM] = pl.cast(index_k_value_col_expand_add, pl.BF16, mode="rint")

    if skip_query == 0:
        for idx in pl.spmd(t_dim // PROJ_T_TILE, name_hint="indexer_weights_proj"):
            t0 = idx * PROJ_T_TILE
            w_acc = pl.create_tensor([PROJ_T_TILE, INDEX_H], dtype=pl.FP32)
            for db in pl.pipeline(0, D // PROJ_K_TILE, stage=2):
                d0 = db * PROJ_K_TILE
                w_acc = pl.matmul_acc(w_acc, x[t0:t0 + PROJ_T_TILE, d0:d0 + PROJ_K_TILE], w_weights[d0:d0 + PROJ_K_TILE, 0:INDEX_H], init_cond=d0 == 0)
            head_weights[t0 : t0 + PROJ_T_TILE, 0:INDEX_H] = pl.mul(w_acc, WEIGHTS_SCALE)

    for idx in pl.spmd(t_dim // PROJ_T_TILE, name_hint="indexer_gate_proj"):
        t0 = idx * PROJ_T_TILE
        g_acc = pl.create_tensor([PROJ_T_TILE, INDEX_DIM], dtype=pl.FP32)
        for db in pl.pipeline(0, D // PROJ_K_TILE, stage=2):
            d0 = db * PROJ_K_TILE
            g_acc = pl.matmul_acc(g_acc, x[t0:t0 + PROJ_T_TILE, d0:d0 + PROJ_K_TILE], w_compress_gate[d0:d0 + PROJ_K_TILE, 0:INDEX_DIM], init_cond=d0 == 0)
        gate_scores[t0 : t0 + PROJ_T_TILE, 0:INDEX_DIM] = g_acc

    return index_q, index_k, head_weights, gate_scores


@pl.jit.inline
def _indexer_pool_gather_chunk(
    pool_cache: pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, INDEX_DIM], pl.INT8],
    pool_scale: pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, POOL_SCALE_WIDTH], pl.FP32],
    pool_blocks: pl.Tensor[[POOL_PAGES_DYN], pl.INT32],
    pool_i8: pl.Tensor[[POOL_PAGES_DYN * INDEX_STATE_BLOCK_SIZE + SCORE_C_TILE, INDEX_DIM], pl.INT8],
    pool_scale_g: pl.Tensor[[POOL_PAGES_DYN * INDEX_STATE_BLOCK_SIZE + SCORE_C_TILE, POOL_SCALE_WIDTH], pl.FP32],
    chunk: pl.Scalar[pl.INDEX],
    page_count: pl.Scalar[pl.INDEX],
):
    """Gather one 16-page chunk into contiguous score scratch."""
    chunk_base = pl.cast(chunk, pl.INT32) * POOL_GATHER_PAGE_TILE
    chunk_ok = 1
    for j in pl.range(1, POOL_GATHER_PAGE_TILE):
        step = chunk_base + j
        cur = pl.read(pool_blocks, [pl.min(step, page_count - 1)])
        prv = pl.read(pool_blocks, [pl.max(pl.min(step, page_count - 1) - 1, 0)])
        good = pl.cast(cur == pl.add(prv, INDEX_STATE_BLOCK_SIZE), pl.INT32)
        good = good * pl.cast(step < page_count, pl.INT32)
        chunk_ok = chunk_ok * good
    if chunk_ok == 1:
        p0_idx = pl.cast(chunk_base, pl.INDEX) * INDEX_STATE_BLOCK_SIZE
        base_idx = pl.cast(pl.read(pool_blocks, [chunk_base]), pl.INDEX)
        pool_i8[p0_idx : p0_idx + POOL_GATHER_ROW_TILE, :] = pool_cache[
            base_idx : base_idx + POOL_GATHER_ROW_TILE, :
        ]
        pool_scale_g[p0_idx : p0_idx + POOL_GATHER_ROW_TILE, :] = pool_scale[
            base_idx : base_idx + POOL_GATHER_ROW_TILE, :
        ]
    else:
        for j in pl.range(POOL_GATHER_PAGE_TILE):
            page_i = chunk_base + j
            if page_i < page_count:
                p0_idx = pl.cast(page_i, pl.INDEX) * INDEX_STATE_BLOCK_SIZE
                base_idx = pl.cast(pl.read(pool_blocks, [page_i]), pl.INDEX)
                pool_i8[p0_idx : p0_idx + INDEX_STATE_BLOCK_SIZE, :] = pool_cache[
                    base_idx : base_idx + INDEX_STATE_BLOCK_SIZE, :
                ]
                pool_scale_g[p0_idx : p0_idx + INDEX_STATE_BLOCK_SIZE, :] = pool_scale[
                    base_idx : base_idx + INDEX_STATE_BLOCK_SIZE, :
                ]


@pl.jit.inline
def _indexer_pool_scale_block(
    pool_scale_g: pl.Tensor[[POOL_PAGES_DYN * INDEX_STATE_BLOCK_SIZE + SCORE_C_TILE, POOL_SCALE_WIDTH], pl.FP32],
    pool_scale_col: pl.Tensor[[POOL_PAGES_DYN * INDEX_STATE_BLOCK_SIZE + SCORE_C_TILE, 1], pl.FP32],
    scale_block: pl.Scalar[pl.INDEX],
):
    """Pack one 64-row scale block and apply the score scale."""
    c0 = pl.cast(scale_block, pl.INT32) * SCORE_C_TILE
    c0_idx = pl.cast(c0, pl.INDEX)
    sc8 = pool_scale_g[c0_idx : c0_idx + SCORE_C_TILE, :]
    summed = pl.reshape(pl.row_sum(sc8), [SCORE_C_TILE, 1])
    pool_scale_col[c0_idx : c0_idx + SCORE_C_TILE, :] = pl.mul(summed, 0.125 * SOFTMAX_SCALE)


@pl.jit.inline
def _indexer_score_prepare(
    index_q: pl.Tensor[[T_DYN, INDEX_H, INDEX_DIM], pl.BF16],
    hadamard: pl.Tensor[[INDEX_DIM, INDEX_DIM], pl.BF16],
    pool_cache: pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, INDEX_DIM], pl.INT8],
    pool_scale: pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, POOL_SCALE_WIDTH], pl.FP32],
    pool_blocks: pl.Tensor[[POOL_PAGES_DYN], pl.INT32],
):
    """Shared scorer prologue: query rotation/quant and the pool gather.

    Returns the INT8 query rows with their dequant scales and the contiguous
    INT8/scale scratch the score loops read. ``pool_blocks`` holds one physical
    base row per INDEX_STATE_BLOCK_SIZE-row page; the gather walks it page by
    page, which is the granularity the metadata module lowers the pool page
    table at — the host lowers paging, the kernel gathers. The score matrix
    itself needs no preparation: prefill tiles store their whole lane span and
    the top-k clamps through its valid-shaped leaf loads, so there is no init
    fill.
    """
    t_dim = pl.tensor.dim(index_q, 0)
    page_count = pl.tensor.dim(pool_blocks, 0)
    pools = page_count * INDEX_STATE_BLOCK_SIZE

    qh_acc = pl.create_tensor([t_dim * INDEX_H, INDEX_DIM], dtype=pl.FP32)
    q_flat = pl.reshape(index_q, [t_dim * INDEX_H, INDEX_DIM])
    for idx in pl.spmd(t_dim * INDEX_H // QH_MM_TILE, name_hint="indexer_q_hadamard", allow_early_resolve=True):
        r0 = idx * QH_MM_TILE
        qh_acc[r0:r0 + QH_MM_TILE, :] = pl.matmul(q_flat[r0:r0 + QH_MM_TILE, :], hadamard, out_dtype=pl.FP32)

    q_scale_dq = pl.create_tensor([t_dim * INDEX_H, 1], dtype=pl.FP32)
    q_i8 = pl.create_tensor([t_dim * INDEX_H, INDEX_DIM], dtype=pl.INT8)
    for idx in pl.spmd(t_dim * INDEX_H // QH_MM_TILE, name_hint="indexer_q_quant", allow_early_resolve=True):
        r0 = idx * QH_MM_TILE
        qh_tile = qh_acc[r0 : r0 + QH_MM_TILE, :]
        qh_amax = pl.full([1, QH_MM_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        qh_abs = pl.maximum(qh_tile, pl.neg(qh_tile))
        qh_amax = pl.maximum(qh_amax, pl.reshape(pl.row_max(qh_abs), [1, QH_MM_TILE]))
        scale_quant_row = pl.div(pl.full([1, QH_MM_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), qh_amax)
        q_scale_dq[r0 : r0 + QH_MM_TILE, :] = pl.reshape(pl.recip(scale_quant_row), [QH_MM_TILE, 1])
        qh_scaled = pl.row_expand_mul(qh_tile, pl.reshape(scale_quant_row, [QH_MM_TILE, 1]))
        qh_i32 = pl.cast(qh_scaled, target_type=pl.INT32, mode="rint")
        qh_half = pl.cast(qh_i32, target_type=pl.FP16, mode="round")
        q_i8[r0 : r0 + QH_MM_TILE, :] = pl.cast(qh_half, target_type=pl.INT8, mode="trunc")

    pool_scale_g = pl.create_tensor([pools + SCORE_C_TILE, POOL_SCALE_WIDTH], dtype=pl.FP32)
    pool_i8 = pl.create_tensor([pools + SCORE_C_TILE, INDEX_DIM], dtype=pl.INT8)
    if t_dim <= INDEXER_PERSISTENT_MAX_TOKENS:
        # Persistent workers own 16-page chunks. A contiguous chunk is copied once
        # as 64 rows; a scattered chunk retains per-page moves with identical bytes.
        chunk_count = (page_count + POOL_GATHER_PAGE_TILE - 1) // POOL_GATHER_PAGE_TILE
        gather_workers = pl.max(pl.min(chunk_count, POOL_GATHER_WORKERS), 1)
        with pl.spmd(gather_workers, name_hint="indexer_pool_gather"):
            worker = pl.tile.get_block_idx()
            for chunk in pl.range(worker, chunk_count, POOL_GATHER_WORKERS):
                _indexer_pool_gather_chunk(
                    pool_cache, pool_scale, pool_blocks, pool_i8, pool_scale_g, chunk, page_count
                )
    else:
        # Larger prefill batches dispatch one task per 16-page chunk with the
        # same contiguous and scattered copy paths.
        prefill_chunk_count = (page_count + POOL_GATHER_PAGE_TILE - 1) // POOL_GATHER_PAGE_TILE
        for prefill_chunk in pl.spmd(prefill_chunk_count, name_hint="indexer_pool_gather"):
            _indexer_pool_gather_chunk(
                pool_cache, pool_scale, pool_blocks, pool_i8, pool_scale_g, prefill_chunk, page_count
            )
    # Collapse the replicated scale lanes once, width-proportional work outside
    # the per-token loop: a strided lane-0 column read is not a legal TLOAD, and
    # reducing inside the score loop costs ~40-70 ns per (token, tile) step.
    # The softmax scale rides along here (it is per-row constant), so the score
    # loops never multiply it per tile.
    pool_scale_col = pl.create_tensor([pools + SCORE_C_TILE, 1], dtype=pl.FP32)
    # pools is the page-table length times INDEX_STATE_BLOCK_SIZE and is NOT
    # padded to SCORE_C_TILE — the production page table ends at the last live
    # page — so the dispatch is ceiled: the final partial tile still packs its
    # real rows. Lanes at or beyond pools stay undefined under the same
    # clamped-by-valid-shape contract as the score matrix, and the scratch
    # slack above covers every reader's widest tile, so the tail reads in
    # bounds.
    scale_blocks = (pools + SCORE_C_TILE - 1) // SCORE_C_TILE
    if t_dim <= INDEXER_PERSISTENT_MAX_TOKENS:
        scale_workers = pl.max(pl.min(scale_blocks, POOL_SCALE_WORKERS), 1)
        with pl.spmd(scale_workers, name_hint="indexer_pool_scale_pack"):
            worker = pl.tile.get_block_idx()
            for scale_block in pl.range(worker, scale_blocks, POOL_SCALE_WORKERS):
                _indexer_pool_scale_block(pool_scale_g, pool_scale_col, scale_block)
    else:
        for c_block in pl.spmd(scale_blocks, name_hint="indexer_pool_scale_pack"):
            _indexer_pool_scale_block(pool_scale_g, pool_scale_col, c_block)
    return q_i8, q_scale_dq, pool_i8, pool_scale_col


@pl.jit.inline
def indexer_score(
    index_q: pl.Tensor[[T_DYN, INDEX_H, INDEX_DIM], pl.BF16],
    hadamard: pl.Tensor[[INDEX_DIM, INDEX_DIM], pl.BF16],
    pool_cache: pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, INDEX_DIM], pl.INT8],
    pool_scale: pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, POOL_SCALE_WIDTH], pl.FP32],
    pool_blocks: pl.Tensor[[POOL_PAGES_DYN], pl.INT32],
    head_weights: pl.Tensor[[T_DYN, INDEX_H], pl.FP32],
    seg_start: pl.Tensor[[T_DYN], pl.INT32],
    pool_count: pl.Tensor[[T_DYN], pl.INT32],
    index_scores: pl.Tensor[[T_DYN, POOLS_DYN], pl.FP32],
):
    """Score each query's own pooled-key segment for prefill and decode."""
    t_dim = pl.tensor.dim(index_q, 0)
    (q_i8, q_scale_dq, pool_i8, pool_scale_col) = _indexer_score_prepare(
        index_q, hadamard, pool_cache, pool_scale, pool_blocks,
    )

    for token in pl.spmd(t_dim, name_hint="indexer_score"):
        visible = pl.max(pl.read(pool_count, [token]), 0)
        if visible > 0:
            seg0 = pl.read(seg_start, [token])
            q_tile_i8 = q_i8[token * INDEX_H : token * INDEX_H + INDEX_H, :]
            q_scale_row = pl.reshape(q_scale_dq[token * INDEX_H : token * INDEX_H + INDEX_H, :], [1, INDEX_H])
            # Fused per-head multiplier; the softmax scale already rides on the
            # packed key scale, and the positive scale commutes with the relu.
            qsw = pl.mul(q_scale_row, head_weights[token : token + 1, 0:INDEX_H])
            for c0 in pl.range(0, visible, SCORE_C_TILE):
                kv_q_i8 = pool_i8[seg0 + c0 : seg0 + c0 + SCORE_C_TILE, :]
                kv_cache_scale_dq = pool_scale_col[seg0 + c0 : seg0 + c0 + SCORE_C_TILE, :]
                dots_i32 = pl.matmul(kv_q_i8, q_tile_i8, out_dtype=pl.INT32, b_trans=True)
                dots = pl.cast(dots_i32, target_type=pl.FP32, mode="none")
                dots = pl.row_expand_mul(dots, kv_cache_scale_dq)
                relu = pl.maximum(dots, 0.0)
                weighted = pl.col_expand_mul(relu, qsw)
                row_score = pl.reshape(pl.row_sum(weighted), [1, SCORE_C_TILE])
                valid_len = pl.min(SCORE_C_TILE, visible - c0)
                masked = pl.fillpad(pl.set_validshape(row_score, 1, valid_len), pad_value=pl.PadValue.min)
                masked = pl.maximum(masked, pl.full([1, SCORE_C_TILE], dtype=pl.FP32, value=FP32_NEG_INF))
                index_scores[token : token + 1, seg0 + c0 : seg0 + c0 + SCORE_C_TILE] = masked


@pl.jit.incore
def _indexer_topk_query(
    index_scores: pl.Tensor[[T_DYN, POOLS_DYN], pl.FP32],
    seg_start: pl.Tensor[[T_DYN], pl.INT32],
    pool_count: pl.Tensor[[T_DYN], pl.INT32],
    selected_pools: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    selected_valid: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
) -> None:
    """Stripe queries round-robin over the dispatched workers.

    ``min(t_dim, TOPK_QUERY_WORKERS)`` blocks are dispatched. Worker ``w``
    processes ``w, w + TOPK_QUERY_WORKERS, ...`` while the query is below
    ``t_dim``.
    """
    worker = pl.tile.get_block_idx()
    t_dim = pl.tensor.dim(index_scores, 0)
    for query in pl.range(worker, t_dim, TOPK_QUERY_WORKERS):
        _indexer_topk_one_query(index_scores, seg_start, pool_count, selected_pools, selected_valid, query)


@pl.jit.inline
def indexer_topk(
    index_scores: pl.Tensor[[T_DYN, POOLS_DYN], pl.FP32],
    seg_start: pl.Tensor[[T_DYN], pl.INT32],
    pool_count: pl.Tensor[[T_DYN], pl.INT32],
    selected_pools: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    selected_valid: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
):
    """Top-512 over each query's visible window, persistent-worker dispatch.

    ``min(t_dim, TOPK_QUERY_WORKERS)`` blocks stripe the queries. A window of
    at most ``KPOOL_SELECT_K`` pools selects every visible pool in compact id
    order; larger windows take the exact sorted path.

    Selections are segment-local pool ids: the sort payload carries the offset
    from ``seg_start``, so ``indexer_expand`` can map pool ``j`` to raw
    positions ``4j .. 4j+3`` without any batch knowledge.

    The sorted path's running top pairs live in a loop-carried on-chip tile.

    **Sharing ABI** (``index_share_for_mtp_iteration``, settled in issue
    #1267): ``selected_pools``/``selected_valid`` are the shareable selection
    surface. A request's MTP draft steps 1+ reuse the selection their
    proposal computed on step 0 — per request, not per row — with the query
    projection, scoring and this selection skipped on those steps. The
    expansion is *not* shareable and is rebuilt per row by
    :func:`indexer_expand`. The resident MTP scheduler retains the selection
    per request and skips this entry on recycled proposal steps.

    Returns the region's TaskId so the expansion can order against it.
    """
    t_dim = pl.tensor.dim(index_scores, 0)
    with pl.spmd(
        pl.min(t_dim, TOPK_QUERY_WORKERS), name_hint="indexer_topk"
    ) as topk_tid:
        _indexer_topk_query(index_scores, seg_start, pool_count, selected_pools, selected_valid)
    return topk_tid


@pl.jit.inline
def indexer_expand(
    selected_pools: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    selected_valid: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    tail_start: pl.Tensor[[T_DYN], pl.INT32],
    tail_count: pl.Tensor[[T_DYN], pl.INT32],
    kv_len: pl.Tensor[[T_DYN], pl.INT32],
    topk_indices: pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32],
):
    """Expand the selected pools into raw cache rows, front packed per row.

    **ABI**: ``topk_indices`` is front packed. Each row holds its valid logical
    per-request positions in its leading lanes and pads the remaining suffix
    with ``-1``; a ``-1`` never sits between two valid entries. Both sparse
    attention kernels read this as a contract rather than a convention: they
    test one lane to decide that a 128-wide block, or a whole row, carries no
    selection, so an interleaved ``-1`` would silently drop the live entries
    behind it. Both sparse attention paths consume this layout in
    :mod:`models.glm5_3_flash.attention`.

    The prefix length is ``min(KPOOL_SELECT_K, kv_len // 4) * 4`` — selection
    order within it is free because the attention is a permutation-invariant
    softmax, so this constrains only where the padding goes. The tail pool is
    always appended right behind the selected groups, which is what keeps the
    packing tight for short contexts; the reference leaves holes there instead.

    **Sharing ABI**: only the selection (``selected_pools``/``selected_valid``)
    may be shared across a request's rows; the expansion is rebuilt per row
    because ``tail_start``/``tail_count``/``kv_len`` differ per row — a decode
    request's four rows (one live plus three MTP spec tokens) carry different
    tails, and the front-packed row bakes its own tail in.
    """
    t_dim = pl.tensor.dim(selected_pools, 0)
    with pl.spmd(
        (t_dim + EXPAND_TOKEN_TILE - 1) // EXPAND_TOKEN_TILE, name_hint="indexer_expand"
    ) as expand_tid:
        tok_blk = pl.tile.get_block_idx()
        for tk_i in pl.range(EXPAND_TOKEN_TILE):
            token = tok_blk * EXPAND_TOKEN_TILE + tk_i
            if token < t_dim:
                # Scalar bookkeeping stays INT32: pl.read preserves the tensor dtype but
                # pl.min promotes to index, which scalar instructions reject.
                groups = pl.cast(pl.min(pl.read(kv_len, [token]) // INDEX_KPOOL, KPOOL_SELECT_K), pl.INT32)
                hist_len = pl.cast(groups * INDEX_KPOOL, pl.INT32)
                base = pl.cast(pl.read(tail_start, [token]), pl.INT32)
                tail = pl.cast(pl.read(tail_count, [token]), pl.INT32)
                for w0 in pl.unroll(0, TOPK_INDEX_WIDTH - (INDEX_KPOOL - 1), EXPAND_W_TILE):
                    lane = pl.arange(pl.cast(w0, pl.INT32), [1, EXPAND_W_TILE], dtype=pl.INT32)
                    pool_raw = pl.cast(pl.mul(pl.cast(lane, pl.FP32, mode="none"), 1.0 / INDEX_KPOOL), pl.INT32, mode="trunc")
                    pool_id = pl.minimum(pool_raw, KPOOL_SELECT_K - 1)
                    sel_row = pl.gather(selected_pools[token : token + 1, 0:KPOOL_SELECT_K], dim=-1, index=pool_id)
                    sel_validity = pl.gather(selected_valid[token:token + 1, 0:KPOOL_SELECT_K], dim=-1, index=pool_id)
                    hist_slack = pl.neg(pl.sub(lane, hist_len))
                    hist_mask = pl.minimum(pl.maximum(hist_slack, 0), 1)
                    hist_active = pl.mul(hist_mask, pl.minimum(sel_validity, 1))
                    tail_offset = pl.sub(lane, hist_len)
                    tail_in_range = pl.minimum(pl.maximum(pl.neg(pl.sub(tail_offset, tail)), 0), 1)
                    tail_ge_zero = pl.minimum(pl.maximum(pl.add(tail_offset, 1), 0), 1)
                    tail_active = pl.mul(tail_in_range, tail_ge_zero)
                    # Position math in FP32 for the float-constant multiplies, rounded
                    # once at the store.
                    lane_f = pl.cast(lane, pl.FP32, mode="none")
                    pool_f = pl.cast(pool_raw, pl.FP32, mode="none")
                    rem_f = pl.sub(lane_f, pl.mul(pool_f, 4.0))
                    hist_pos_f = pl.add(pl.mul(pl.cast(sel_row, pl.FP32, mode="none"), 4.0), rem_f)
                    tail_pos_f = pl.cast(pl.add(pl.add(tail_offset, base), 1), pl.FP32, mode="none")
                    positions_f_cast = pl.cast(hist_active, pl.FP32, mode="none")
                    positions_f_add = pl.add(hist_pos_f, 1.0)
                    positions_f_mul = pl.mul(positions_f_cast, positions_f_add)
                    positions_f_cast_2 = pl.cast(tail_active, pl.FP32, mode="none")
                    positions_f_mul_2 = pl.mul(positions_f_cast_2, tail_pos_f)
                    positions_f_add_2 = pl.add(positions_f_mul, positions_f_mul_2)
                    positions_f = pl.sub(positions_f_add_2, 1.0)
                    topk_indices[token:token + 1, w0:w0 + EXPAND_W_TILE] = pl.cast(positions_f, pl.INT32, mode="none")
                # The odd 2051-wide row cannot end on an aligned tile, so the last chunk
                # is an eight-lane tile starting five lanes early; the overlap is written
                # twice with identical values, which keeps every store on the MTE3 path.
                lane_t = pl.arange(pl.const(2043, pl.INT32), [1, EXPAND_TAIL_TILE], dtype=pl.INT32)
                pool_t = pl.cast(pl.mul(pl.cast(lane_t, pl.FP32, mode="none"), 1.0 / INDEX_KPOOL), pl.INT32, mode="trunc")
                pool_t = pl.minimum(pool_t, KPOOL_SELECT_K - 1)
                sel_t = pl.gather(selected_pools[token : token + 1, 0:KPOOL_SELECT_K], dim=-1, index=pool_t)
                valid_t = pl.gather(selected_valid[token : token + 1, 0:KPOOL_SELECT_K], dim=-1, index=pool_t)
                hslack_t = pl.neg(pl.sub(lane_t, hist_len))
                hmask_t = pl.minimum(pl.maximum(hslack_t, 0), 1)
                hactive_t = pl.mul(hmask_t, pl.minimum(valid_t, 1))
                offset_t = pl.sub(lane_t, hist_len)
                trange_t = pl.minimum(pl.maximum(pl.neg(pl.sub(offset_t, tail)), 0), 1)
                tge_t = pl.minimum(pl.maximum(pl.add(offset_t, 1), 0), 1)
                tactive_t = pl.mul(trange_t, tge_t)
                lane_tf = pl.cast(lane_t, pl.FP32, mode="none")
                pool_tf = pl.cast(pool_t, pl.FP32, mode="none")
                rem_tf = pl.sub(lane_tf, pl.mul(pool_tf, 4.0))
                hpos_tf = pl.add(pl.mul(pl.cast(sel_t, pl.FP32, mode="none"), 4.0), rem_tf)
                tpos_tf = pl.cast(pl.add(pl.add(offset_t, base), 1), pl.FP32, mode="none")
                pos_tf_cast = pl.cast(hactive_t, pl.FP32, mode="none")
                pos_tf_add = pl.add(hpos_tf, 1.0)
                pos_tf_mul = pl.mul(pos_tf_cast, pos_tf_add)
                pos_tf_cast_2 = pl.cast(tactive_t, pl.FP32, mode="none")
                pos_tf_mul_2 = pl.mul(pos_tf_cast_2, tpos_tf)
                pos_tf_add_2 = pl.add(pos_tf_mul, pos_tf_mul_2)
                pos_tf = pl.sub(pos_tf_add_2, 1.0)
                topk_indices[token:token + 1, TOPK_INDEX_WIDTH - EXPAND_TAIL_TILE:TOPK_INDEX_WIDTH] = pl.cast(pos_tf, pl.INT32, mode="none")
    return expand_tid


@pl.jit.inline
def indexer_cache_write(
    index_k: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    gate_scores: pl.Tensor[[T_DYN, INDEX_DIM], pl.FP32],
    slots: pl.Tensor[[T_DYN], pl.INT32],
    cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32],
):
    """Scatter one step's raw indexer rows into the paged pool, eight rows a block.

    A slot of ``-1`` marks a row that owns no cache position — a padded row in a
    packed batch — and is skipped, so a negative slot cannot wrap into the tail of
    the pool. The key half is widened to FP32 on the way in: the pooled-table writer
    re-reads these rows in FP32, and the storage ABI is FP32 end to end. One block
    moves :data:`CACHE_WRITE_TILE` rows: a single ``[tile, dim]`` key load and one
    widening cast feed the per-row slot scatters. A tail block with fewer rows
    than the tile clamps its load to the table's edge and skips the rows past the
    token count, so any token count is legal.

    Returns the region's TaskId so a later reader can order against the write.
    """
    tokens = pl.tensor.dim(index_k, 0)
    with pl.spmd(
        (tokens + CACHE_WRITE_TILE - 1) // CACHE_WRITE_TILE, name_hint="indexer_cache_write"
    ) as write_tid:
        base = pl.tile.get_block_idx() * CACHE_WRITE_TILE
        # clamp=True cuts the tail block's window back to the table's last row
        # and fillpad restores the tile to full width with zeros; the per-row
        # bounds check below drops the window's phantom rows from the slot read
        # and both stores. Full blocks are unaffected.
        keys_f32 = pl.tile.fillpad(
            pl.cast(pl.load(index_k, [base, 0], [CACHE_WRITE_TILE, INDEX_DIM], clamp=True), pl.FP32, mode="none")
        )
        gates = pl.tile.fillpad(pl.load(gate_scores, [base, 0], [CACHE_WRITE_TILE, INDEX_DIM], clamp=True))
        for row in pl.unroll(0, CACHE_WRITE_TILE):
            if base + row < tokens:
                slot_i32 = pl.read(slots, [base + row])
                if slot_i32 >= 0:
                    slot = pl.cast(slot_i32, pl.INDEX)
                    pl.store(pl.tile.slice(keys_f32, [1, INDEX_DIM], [row, 0]), [slot, 0], cache)
                    pl.store(pl.tile.slice(gates, [1, INDEX_DIM], [row, 0]), [slot, INDEX_DIM], cache)
    return write_tid


@pl.jit.inline
def _indexer_topk_one_query(
    index_scores: pl.Tensor[[T_DYN, POOLS_DYN], pl.FP32],
    seg_start: pl.Tensor[[T_DYN], pl.INT32],
    pool_count: pl.Tensor[[T_DYN], pl.INT32],
    selected_pools: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    selected_valid: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    query: pl.Scalar[pl.INDEX],
) -> None:
    """Select one query's top visible pools (the per-query sort pipeline)."""
    pl.store(pl.tile.full([1, KPOOL_SELECT_K], dtype=pl.INT32, value=-1), [query, 0], selected_pools)
    pl.store(pl.tile.full([1, KPOOL_SELECT_K], dtype=pl.INT32, value=0), [query, 0], selected_valid)
    visible_count = pl.max(pl.read(pool_count, [query]), 0)
    if visible_count > 0:
        seg0 = pl.read(seg_start, [query])
        if visible_count <= KPOOL_SELECT_K:
            # At or below the select width, materialize every visible pool
            # directly from the compact id ramp without reading scores or
            # sorting. The selection prefix order is ABI-free.
            valid_topk = pl.cast(visible_count, pl.INT32)
            lane = pl.tile.arange(0, [1, KPOOL_SELECT_K], dtype=pl.INT32)
            keep = pl.minimum(pl.maximum(pl.neg(pl.sub(lane, valid_topk)), 0), 1)
            indices_out = pl.sub(pl.add(pl.mul(lane, keep), keep), 1)
            pl.store(indices_out, [query, 0], selected_pools)
            pl.store(keep, [query, 0], selected_valid)
        else:
            # Carry a pad-free root tile across the leaf loop. Extract each
            # leaf's sorted top pairs and merge them into the root with
            # assemble; a re-sliced merge result is a view whose pad-carrying
            # type is rejected as a loop iter-arg. A one-leaf window publishes
            # directly from the seeded root.
            leaf_valid = pl.min(SORT_TILE, visible_count)
            row_raw = pl.load(index_scores, [query, seg0], [1, SORT_TILE], valid_shape=[1, leaf_valid])
            row = pl.tile.fillpad(row_raw, pad_value=pl.PadValue.min)
            row = pl.maximum(row, pl.tile.full([1, SORT_TILE], dtype=pl.FP32, value=FP32_NEG_INF))
            index_ramp = pl.tile.arange(0, [1, SORT_TILE], dtype=pl.INT32)
            leaf_pairs = pl.tile.sort32(row, pl.reinterpret_view(index_ramp, pl.UINT32))
            leaf_pairs = pl.tile.mrgsort(leaf_pairs, block_len=64)
            leaf_pairs = pl.tile.mrgsort(leaf_pairs, block_len=256)
            leaf_pairs = pl.tile.mrgsort(leaf_pairs, block_len=1024)
            root = pl.tile.create([1, PAIR_WIDTH], dtype=pl.FP32)
            seed = pl.tile.extract(leaf_pairs, 0, 0, [1, PAIR_WIDTH], target_memory=pl.Mem.Vec)
            root = pl.tile.assemble(root, seed, [0, 0])
            leaf_count = (visible_count + SORT_TILE - 1) // SORT_TILE
            for leaf in pl.range(1, leaf_count):
                leaf0 = leaf * SORT_TILE
                lv = pl.min(SORT_TILE, visible_count - leaf0)
                row2_raw = pl.load(index_scores, [query, seg0 + leaf0], [1, SORT_TILE], valid_shape=[1, lv])
                row2 = pl.tile.fillpad(row2_raw, pad_value=pl.PadValue.min)
                row2 = pl.maximum(row2, pl.tile.full([1, SORT_TILE], dtype=pl.FP32, value=FP32_NEG_INF))
                ramp2 = pl.tile.arange(0, [1, SORT_TILE], dtype=pl.INT32)
                lp2 = pl.tile.sort32(row2, pl.reinterpret_view(pl.tile.add(ramp2, pl.cast(leaf0, pl.INT32)), pl.UINT32))
                lp2 = pl.tile.mrgsort(lp2, block_len=64)
                lp2 = pl.tile.mrgsort(lp2, block_len=256)
                lp2 = pl.tile.mrgsort(lp2, block_len=1024)
                lp2_top = pl.tile.extract(lp2, 0, 0, [1, PAIR_WIDTH], target_memory=pl.Mem.Vec)
                old_root = pl.tile.extract(root, 0, 0, [1, PAIR_WIDTH], target_memory=pl.Mem.Vec)
                merge_tmp = pl.tile.create([1, 2 * PAIR_WIDTH], dtype=pl.FP32)
                merged = pl.tile.mrgsort(old_root, lp2_top, tmp=merge_tmp)
                next_root = pl.tile.extract(merged, 0, 0, [1, PAIR_WIDTH], target_memory=pl.Mem.Vec)
                root = pl.tile.assemble(root, next_root, [0, 0])
            pairs = pl.tile.extract(root, 0, 0, [1, PAIR_WIDTH], target_memory=pl.Mem.Vec)
            picked = pl.tile.gather_mask(pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            valid_topk = pl.cast(pl.min(visible_count, KPOOL_SELECT_K), pl.INT32)
            # Materialize the picked payload in one integer lane pass. keep is
            # the 0/1 window mask: picked*keep + keep - 1 maps valid lanes to
            # picked ids and padding lanes to -1, while keep is the validity
            # output. fillpad cannot emit the arbitrary integer -1 pad.
            lane = pl.tile.arange(0, [1, KPOOL_SELECT_K], dtype=pl.INT32)
            keep = pl.minimum(pl.maximum(pl.neg(pl.sub(lane, valid_topk)), 0), 1)
            indices_out = pl.sub(pl.add(pl.mul(picked, keep), keep), 1)
            pl.store(indices_out, [query, 0], selected_pools)
            pl.store(keep, [query, 0], selected_valid)


@pl.jit.incore
def _indexer_pool_write_event(
    raw_cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32],
    compress_ape: pl.Tensor[[INDEX_KPOOL, INDEX_DIM], pl.BF16],
    pool_token_slots: pl.Tensor[[POOL_EVENTS_DYN, INDEX_KPOOL], pl.INT32],
    pool_stage: pl.Tensor[[POOL_STAGE_DYN, INDEX_DIM], pl.FP32],
    pool_valid: pl.Tensor[[POOL_EVENTS_DYN, POOL_VALID_WIDTH], pl.INT32],
) -> None:
    """Close one pool event; the caller gives this scope one block per event.

    The softmax runs per dimension over the four pool lanes in FP32, matching the
    reference numerics: ``weights = softmax(gate + ape)`` along the lane axis, then
    ``pool_key = sum(weights * key)``. The FP32 pooled key lands in the
    contiguous ``pool_stage`` row of its event; a second scope quantizes the
    staged rows to INT8 plus a dequant scale and scatters them to the paged
    destinations. An event with a ``-1`` member slot is a padded pool: it stages a
    zero row and reports ``pool_valid = 0`` so the scorer can drop it. The staged
    row is always written, which keeps the epilogue's input fully defined.
    """
    event = pl.tile.get_block_idx()
    slot0 = pl.read(pool_token_slots, [event, 0])
    slot1 = pl.read(pool_token_slots, [event, 1])
    slot2 = pl.read(pool_token_slots, [event, 2])
    slot3 = pl.read(pool_token_slots, [event, 3])
    members_present = pl.min(pl.min(slot0, slot1), pl.min(slot2, slot3))
    # 0/1 flag in column 0 of a [1, 8] INT32 tile, the narrowest store the
    # 32-byte vector alignment accepts. Slot 0 is a legal raw row, so the flag
    # comes from the branch condition, not from clamping the slot value.
    flag_tile = pl.tile.full([1, POOL_VALID_WIDTH], dtype=pl.INT32, value=0)
    if members_present >= 0:
        # Per-lane logits and keys, kept as [1, INDEX_DIM] rows so every step is
        # an elementwise row operation; the softmax reduces over the four lanes.
        key0 = pl.load(raw_cache, [pl.cast(slot0, pl.INDEX), 0], [1, INDEX_DIM])
        key1 = pl.load(raw_cache, [pl.cast(slot1, pl.INDEX), 0], [1, INDEX_DIM])
        key2 = pl.load(raw_cache, [pl.cast(slot2, pl.INDEX), 0], [1, INDEX_DIM])
        key3 = pl.load(raw_cache, [pl.cast(slot3, pl.INDEX), 0], [1, INDEX_DIM])
        gate0 = pl.load(raw_cache, [pl.cast(slot0, pl.INDEX), INDEX_DIM], [1, INDEX_DIM])
        gate1 = pl.load(raw_cache, [pl.cast(slot1, pl.INDEX), INDEX_DIM], [1, INDEX_DIM])
        gate2 = pl.load(raw_cache, [pl.cast(slot2, pl.INDEX), INDEX_DIM], [1, INDEX_DIM])
        gate3 = pl.load(raw_cache, [pl.cast(slot3, pl.INDEX), INDEX_DIM], [1, INDEX_DIM])
        ape0 = pl.cast(pl.load(compress_ape, [0, 0], [1, INDEX_DIM]), pl.FP32, mode="none")
        ape1 = pl.cast(pl.load(compress_ape, [1, 0], [1, INDEX_DIM]), pl.FP32, mode="none")
        ape2 = pl.cast(pl.load(compress_ape, [2, 0], [1, INDEX_DIM]), pl.FP32, mode="none")
        ape3 = pl.cast(pl.load(compress_ape, [3, 0], [1, INDEX_DIM]), pl.FP32, mode="none")
        logit0 = pl.add(gate0, ape0)
        logit1 = pl.add(gate1, ape1)
        logit2 = pl.add(gate2, ape2)
        logit3 = pl.add(gate3, ape3)
        lane_max = pl.maximum(pl.maximum(logit0, logit1), pl.maximum(logit2, logit3))
        weight0 = pl.exp(pl.sub(logit0, lane_max))
        weight1 = pl.exp(pl.sub(logit1, lane_max))
        weight2 = pl.exp(pl.sub(logit2, lane_max))
        weight3 = pl.exp(pl.sub(logit3, lane_max))
        denominator = pl.add(pl.add(weight0, weight1), pl.add(weight2, weight3))
        weighted_sum = pl.mul(weight0, key0)
        weighted_sum = pl.add(weighted_sum, pl.mul(weight1, key1))
        weighted_sum = pl.add(weighted_sum, pl.mul(weight2, key2))
        weighted_sum = pl.add(weighted_sum, pl.mul(weight3, key3))
        pool_key = pl.div(weighted_sum, denominator)
        pl.store(pool_key, [event, 0], pool_stage)
        pl.tile.write(flag_tile, [0, 0], pl.const(1, pl.INT32))
    else:
        zero_row = pl.tile.full([1, INDEX_DIM], dtype=pl.FP32, value=0.0)
        pl.store(zero_row, [event, 0], pool_stage)
    pl.store(flag_tile, [event, 0], pool_valid)


@pl.jit.inline
def indexer_pool_write(
    raw_cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32],
    compress_ape: pl.Tensor[[INDEX_KPOOL, INDEX_DIM], pl.BF16],
    hadamard: pl.Tensor[[INDEX_DIM, INDEX_DIM], pl.BF16],
    pool_token_slots: pl.Tensor[[POOL_EVENTS_DYN, INDEX_KPOOL], pl.INT32],
    pool_slots: pl.Tensor[[POOL_EVENTS_DYN], pl.INT32],
    pool_cache: pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, INDEX_DIM], pl.INT8],
    pool_scale: pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, POOL_SCALE_WIDTH], pl.FP32],
    pool_valid: pl.Tensor[[POOL_EVENTS_DYN, POOL_VALID_WIDTH], pl.INT32],
):
    """Write one quantized pooled key row per pool-close event."""
    events = pl.tensor.dim(pool_token_slots, 0)
    stage_rows = ((events + POOL_QUANT_TILE - 1) // POOL_QUANT_TILE) * POOL_QUANT_TILE
    pool_stage = pl.create_tensor([stage_rows, INDEX_DIM], dtype=pl.FP32)
    with pl.spmd(events, name_hint="indexer_pool_write") as pool_tid:
        _indexer_pool_write_event(raw_cache, compress_ape, pool_token_slots, pool_stage, pool_valid)
    for blk in pl.spmd((events + POOL_QUANT_TILE - 1) // POOL_QUANT_TILE, name_hint="indexer_pool_quant"):
        # Slicing wants INDEX offsets; the INT32 twin feeds the arithmetic.
        e0 = pl.cast(blk, pl.INT32) * POOL_QUANT_TILE
        e0_idx = pl.cast(e0, pl.INDEX)
        staged = pool_stage[e0_idx : e0_idx + POOL_QUANT_TILE, :]
        for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            staged_bf16 = pl.cast(staged, pl.BF16, mode="rint")
            staged_mat = pl.aic_gather(staged_bf16)
        staged_rot = pl.matmul(staged_mat, hadamard, out_dtype=pl.FP32)
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            rotated = pl.aiv_shard(staged_rot)
            if aiv_id == 0:
                staged_abs = pl.maximum(rotated, pl.neg(rotated))
                amax = pl.full([1, POOL_QUANT_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                amax = pl.maximum(amax, pl.reshape(pl.row_max(staged_abs), [1, POOL_QUANT_TILE]))
                scale_quant_row = pl.div(pl.full([1, POOL_QUANT_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), amax)
                staged_scaled = pl.row_expand_mul(rotated, pl.reshape(scale_quant_row, [POOL_QUANT_TILE, 1]))
                staged_i32 = pl.cast(staged_scaled, target_type=pl.INT32, mode="rint")
                staged_half = pl.cast(staged_i32, target_type=pl.FP16, mode="round")
                staged_i8 = pl.cast(staged_half, target_type=pl.INT8, mode="trunc")
                scale_dq_col = pl.reshape(pl.recip(scale_quant_row), [POOL_QUANT_TILE, 1])
                scale_pad_full = pl.full([POOL_QUANT_TILE, POOL_SCALE_WIDTH], dtype=pl.FP32, value=1.0)
                scale_pad = pl.row_expand_mul(scale_pad_full, scale_dq_col)
                for e in pl.range(POOL_QUANT_TILE):
                    ei = pl.cast(e, pl.INT32)
                    if e0 + ei < events:
                        ei_idx = pl.cast(ei, pl.INDEX)
                        dest = pl.cast(pl.read(pool_slots, [e0 + ei]), pl.INDEX)
                        pool_cache[dest : dest + 1, :] = staged_i8[ei_idx : ei_idx + 1, :]
                        pool_scale[dest : dest + 1, :] = scale_pad[ei_idx : ei_idx + 1, :]
    return pool_tid


@pl.jit
def indexer_cache_write_test(
    index_k: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    gate_scores: pl.Tensor[[T_DYN, INDEX_DIM], pl.FP32],
    slots: pl.Tensor[[T_DYN], pl.INT32],
    cache: pl.InOut[pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32]],
):
    """Run one raw scatter for golden.run validation."""
    index_k.bind_dynamic(0, T_DYN)
    gate_scores.bind_dynamic(0, T_DYN)
    slots.bind_dynamic(0, T_DYN)
    indexer_cache_write(index_k, gate_scores, slots, cache)
    return cache


@pl.jit
def indexer_pool_write_test(
    raw_cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32],
    compress_ape: pl.Tensor[[INDEX_KPOOL, INDEX_DIM], pl.BF16],
    hadamard: pl.Tensor[[INDEX_DIM, INDEX_DIM], pl.BF16],
    pool_token_slots: pl.Tensor[[POOL_EVENTS_DYN, INDEX_KPOOL], pl.INT32],
    pool_slots: pl.Tensor[[POOL_EVENTS_DYN], pl.INT32],
    pool_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, INDEX_DIM], pl.INT8]],
    pool_scale: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, POOL_SCALE_WIDTH], pl.FP32]],
    pool_valid: pl.Out[pl.Tensor[[POOL_EVENTS_DYN, POOL_VALID_WIDTH], pl.INT32]],
):
    """Close every pool event of one dispatch for golden.run validation."""
    pool_token_slots.bind_dynamic(0, POOL_EVENTS_DYN)
    pool_slots.bind_dynamic(0, POOL_EVENTS_DYN)
    pool_valid.bind_dynamic(0, POOL_EVENTS_DYN)
    indexer_pool_write(
        raw_cache, compress_ape, hadamard, pool_token_slots, pool_slots, pool_cache, pool_scale, pool_valid,
    )
    return pool_cache, pool_scale, pool_valid


def build_decode_indexer_step_specs(requests: int = 16, prior_counts: int | tuple[int, ...] | None = None):
    """Build one deterministic decode step: sixteen requests, one closing pool each.

    The requests carry prior contexts of different lengths, so one closing pool
    is made of four fresh rows, another of fresh rows mixed with rows cached by
    earlier steps, and the pooled table starts with rows only earlier steps
    could have written. Every request contributes exactly
    ``DECODE_ROWS_PER_REQUEST`` query rows. ``prior_counts`` overrides the
    fixture's per-request pool history for the business-shape benchmark points;
    an int means the same history for every request.
    """
    from golden import TensorSpec

    rows_per_request = DECODE_ROWS_PER_REQUEST
    tokens = requests * rows_per_request
    if tokens % PROJ_T_TILE:
        raise ValueError(f"the projection stage needs a token count that is a multiple of {PROJ_T_TILE}")
    if prior_counts is None:
        prior_counts = (13, 5, 40, 2, 9, 21, 60, 3, 33, 7, 50, 17, 26, 11, 44, 6)[:requests]
    elif isinstance(prior_counts, int):
        prior_counts = (prior_counts,) * requests
    if len(prior_counts) != requests:
        raise ValueError("prior_counts must hold one pool count per request")
    new_counts = tuple(count + 1 for count in prior_counts)
    pools_true = sum(new_counts)
    width = ((pools_true + 2 * SORT_TILE - 2) // SORT_TILE) * SORT_TILE

    generator = torch.Generator().manual_seed(89)
    prior_tokens = [4 * count for count in prior_counts]

    def row_map():
        """Physical raw row of every request's logical token, past and new.

        Pages come from one global permutation, partitioned request by
        request: per-request draws from a shared page pool can hand two
        requests the same physical slots, which makes the cache writer race
        and lets a pool close read another request's keys and gates.
        """
        rows = {}
        pages_per_request = [(prior_tokens[request] + rows_per_request + 127) // 128 + 1 for request in range(requests)]
        total_pages = sum(pages_per_request)
        page_pool = max(8, total_pages + 4)
        permutation = torch.randperm(page_pool, generator=generator)
        cursor = 0
        for request in range(requests):
            take = pages_per_request[request]
            pages = permutation[cursor : cursor + take]
            cursor += take
            total = prior_tokens[request] + rows_per_request
            for position in range(total):
                rows[(request, position)] = int(pages[position // 128]) * 128 + position % 128
        all_rows = list(rows.values())
        if len(set(all_rows)) != len(all_rows):
            raise AssertionError("raw cache rows collide across requests")
        return rows

    mapping = row_map()

    def raw_rows_for(request, first, count):
        return [mapping[(request, position)] for position in range(first, first + count)]

    # The pooled table is paged at INDEX_STATE_BLOCK_SIZE rows like the prefill
    # fixtures: each compacted page lands at its own physical base, and a
    # pool's physical row derives from its compacted id.
    pages = width // INDEX_STATE_BLOCK_SIZE
    pool_table_rows = pages * INDEX_STATE_BLOCK_SIZE
    page_bases = torch.randperm(pages, generator=generator).to(torch.int32) * INDEX_STATE_BLOCK_SIZE
    seg_offsets = []
    offset = 0
    for count in new_counts:
        seg_offsets.append(offset)
        offset += count

    def pool_row_of_id(compacted):
        return int(page_bases[compacted // INDEX_STATE_BLOCK_SIZE]) + compacted % INDEX_STATE_BLOCK_SIZE

    pool_row_of = {}
    for request in range(requests):
        for pool in range(new_counts[request]):
            pool_row_of[(request, pool)] = pool_row_of_id(seg_offsets[request] + pool)

    seg_start = torch.zeros(tokens, dtype=torch.int32)
    pool_count = torch.zeros(tokens, dtype=torch.int32)
    tail_start = torch.zeros(tokens, dtype=torch.int32)
    tail_count = torch.zeros(tokens, dtype=torch.int32)
    kv_len = torch.zeros(tokens, dtype=torch.int32)
    for request in range(requests):
        for row in range(rows_per_request):
            token = request * rows_per_request + row
            position = prior_tokens[request] + row
            length = position + 1
            seg_start[token] = sum(new_counts[:request])
            pool_count[token] = min(length // INDEX_KPOOL, new_counts[request])
            tail_count[token] = length % INDEX_KPOOL
            tail_start[token] = length - int(tail_count[token])
            kv_len[token] = length

    closing_slots = torch.full((requests, INDEX_KPOOL), -1, dtype=torch.int32)
    closing_dest = torch.zeros(requests, dtype=torch.int32)
    for request in range(requests):
        closing = new_counts[request] - 1
        first = closing * INDEX_KPOOL
        closing_slots[request] = torch.tensor(raw_rows_for(request, first, INDEX_KPOOL), dtype=torch.int32)
        closing_dest[request] = pool_row_of[(request, closing)]

    index_slots = torch.zeros(tokens, dtype=torch.int32)
    for request in range(requests):
        for row in range(rows_per_request):
            index_slots[request * rows_per_request + row] = mapping[(request, prior_tokens[request] + row)]

    def init_x():
        return torch.randn(tokens, D, generator=generator, dtype=torch.float32).bfloat16()

    def init_q_resid():
        return torch.randn(tokens, Q_LORA, generator=generator, dtype=torch.float32).bfloat16()

    def init_raw_cache():
        rows = max(mapping.values()) + 1
        cache = torch.randn(rows, INDEX_STATE_WIDTH, generator=generator)
        cache[:, :INDEX_DIM] = cache[:, :INDEX_DIM].bfloat16().float()
        return cache

    # The pooled table starts as the quantized form earlier steps would have
    # left behind: INT8 rows already rotated into the Hadamard basis, paired
    # with their dequant scales, one draw feeding both so the pairs stay
    # consistent.
    (prior_i8, prior_scale_col) = quantize_per_token_int8(
        torch.randn(pool_table_rows, INDEX_DIM, generator=generator, dtype=torch.float32),
    )

    def init_pool_cache():
        return prior_i8

    def init_pool_scale():
        # The writer replicates the scale across all POOL_SCALE_WIDTH lanes and
        # the scorer reads the row sum scaled by 1/8, so the prior table must
        # carry the replication too.
        return prior_scale_col.expand(-1, POOL_SCALE_WIDTH).contiguous().clone()

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

    specs = [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=init_x),
        TensorSpec("q_resid", [tokens, Q_LORA], torch.bfloat16, init_value=init_q_resid),
        TensorSpec("w_q_b", [Q_LORA, INDEX_H * INDEX_DIM], torch.bfloat16, init_value=init_weights("w_q_b", 0.02)),
        TensorSpec("w_k", [D, INDEX_DIM], torch.bfloat16, init_value=init_weights("w_k", 0.02)),
        TensorSpec("k_norm_weight", [INDEX_DIM], torch.bfloat16, init_value=lambda : (1.0 + 0.1 * torch.randn(INDEX_DIM, generator=generator)).bfloat16()),
        TensorSpec("k_norm_bias", [INDEX_DIM], torch.bfloat16, init_value=lambda : (0.1 * torch.randn(INDEX_DIM, generator=generator)).bfloat16()),
        TensorSpec("w_weights", [D, INDEX_H], torch.bfloat16, init_value=init_weights("w_weights", 0.05)),
        TensorSpec("w_compress_gate", [D, INDEX_DIM], torch.bfloat16, init_value=init_weights("w_compress_gate", 0.02)),
        TensorSpec("hadamard", [INDEX_DIM, INDEX_DIM], torch.bfloat16, init_value=sylvester_hadamard),
        TensorSpec("compress_ape", [INDEX_KPOOL, INDEX_DIM], torch.bfloat16, init_value=lambda : torch.randn(INDEX_KPOOL, INDEX_DIM, generator=generator).bfloat16()),
        TensorSpec("index_slots", [tokens], torch.int32, init_value=lambda: index_slots),
        TensorSpec("pool_token_slots", [requests, INDEX_KPOOL], torch.int32, init_value=lambda : closing_slots),
        TensorSpec("pool_slots", [requests], torch.int32, init_value=lambda: closing_dest),
        TensorSpec("pool_blocks", [pages], torch.int32, init_value=lambda: page_bases),
        TensorSpec("seg_start", [tokens], torch.int32, init_value=lambda: seg_start),
        TensorSpec("pool_count", [tokens], torch.int32, init_value=lambda: pool_count),
        TensorSpec("tail_start", [tokens], torch.int32, init_value=lambda: tail_start),
        TensorSpec("tail_count", [tokens], torch.int32, init_value=lambda: tail_count),
        TensorSpec("kv_len", [tokens], torch.int32, init_value=lambda: kv_len),
        TensorSpec("raw_cache", [max(mapping.values()) + 1, INDEX_STATE_WIDTH], torch.float32, init_value=init_raw_cache),
        TensorSpec("pool_cache", [pool_table_rows, INDEX_DIM], torch.int8, init_value=init_pool_cache),
        TensorSpec("pool_scale", [pool_table_rows, POOL_SCALE_WIDTH], torch.float32, init_value=init_pool_scale),
        TensorSpec("pool_valid", [requests, POOL_VALID_WIDTH], torch.int32),
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


def golden_decode_indexer_step_case(tensors):
    """Fill every expected output of one decode step by composing the goldens."""
    (index_q, index_k, head_weights, gate_scores) = golden_indexer_proj(
        tensors["x"], tensors["q_resid"], tensors["w_q_b"], tensors["w_k"], tensors["k_norm_weight"],
        tensors["k_norm_bias"], tensors["w_weights"], tensors["w_compress_gate"],
    )
    tensors["index_q"][:] = index_q
    tensors["index_k"][:] = index_k
    tensors["head_weights"][:] = head_weights
    tensors["gate_scores"][:] = gate_scores
    tensors["raw_cache"][:] = golden_indexer_cache_write(
        tensors["raw_cache"],
        index_k, gate_scores, tensors["index_slots"],
    )
    (pool_cache, pool_scale, pool_valid) = golden_indexer_pool_write(
        tensors["pool_cache"], tensors["pool_scale"], tensors["raw_cache"], tensors["compress_ape"],
        tensors["hadamard"], tensors["pool_token_slots"], tensors["pool_slots"],
    )
    tensors["pool_cache"][:] = pool_cache
    tensors["pool_scale"][:] = pool_scale
    tensors["pool_valid"][:] = pool_valid
    tensors["index_scores"][:] = golden_indexer_score(
        tensors["index_q"], tensors["hadamard"], tensors["pool_cache"], tensors["pool_scale"],
        tensors["pool_blocks"], tensors["head_weights"], tensors["seg_start"], tensors["pool_count"],
    )
    (selected, valid) = golden_indexer_topk(tensors["index_scores"], tensors["seg_start"], tensors["pool_count"])
    tensors["selected_pools"][:] = selected
    tensors["selected_valid"][:] = valid
    tensors["topk_indices"][:] = golden_indexer_expand(
        tensors["selected_pools"], tensors["selected_valid"], tensors["tail_start"], tensors["tail_count"],
        tensors["kv_len"],
    )
    for token in range(tensors["topk_indices"].shape[0]):
        row = tensors["topk_indices"][token]
        live = int((row >= 0).sum())
        assert (row[:live] >= 0).all() and (row[live:] == -1).all(), (
            f"row {token} violates the front-packed index ABI"
        )


def build_decode_indexer_abi_specs(topk_indices: torch.Tensor, kv_len: torch.Tensor):
    """Build the sparse-attention consumer specs around a real index list.

    Takes the decode step's actual front-packed ``topk_indices`` together with
    the per-row ``kv_len`` it was produced under, and pages each request's
    logical positions into a fresh latent pool the way the MLA cache would, so
    the stream B gather resolves exactly the rows serving will resolve. The
    latent pool is its own table: the indexer's raw row map and the attention
    page map are independent layouts over the same logical positions, which is
    the point of the resolution contract.
    """
    from golden import TensorSpec

    tokens = topk_indices.shape[0]
    requests = tokens // DECODE_ROWS_PER_REQUEST
    pages_per_request = (int(kv_len.max()) + BLOCK_SIZE - 1) // BLOCK_SIZE

    generator = torch.Generator().manual_seed(101)
    pages = torch.randperm(requests * pages_per_request, generator=generator)
    block_table = pages.reshape(requests, pages_per_request).to(torch.int32)
    cache_rows = requests * pages_per_request * BLOCK_SIZE
    request_ids = torch.arange(tokens, dtype=torch.int32) // DECODE_ROWS_PER_REQUEST

    return [
        TensorSpec("absorbed_query", [tokens, LOCAL_H, KV_LORA], torch.bfloat16, init_value=lambda : torch.randn(tokens, LOCAL_H, KV_LORA, generator=generator, dtype=torch.float32).bfloat16()),
        TensorSpec("latent_cache", [cache_rows, KV_LORA], torch.bfloat16, init_value=lambda : torch.randn(cache_rows, KV_LORA, generator=generator).bfloat16()),
        TensorSpec("block_table", [requests, pages_per_request], torch.int32, init_value=lambda : block_table),
        TensorSpec("request_ids", [tokens], torch.int32, init_value=lambda: request_ids),
        TensorSpec("topk_indices", [tokens, TOPK_INDEX_WIDTH], torch.int32, init_value=lambda : topk_indices.clone()),
        TensorSpec("output", [tokens, LOCAL_H, KV_LORA], torch.bfloat16),
    ]


@dataclass
class IndexerTensors:
    """k-pool indexer weights of one DSA layer (FP32, checkpoint orientation)."""

    wq_b: torch.Tensor  # [n_heads * head_dim, q_lora]
    wk: torch.Tensor  # [head_dim, D]
    k_norm_w: torch.Tensor  # [head_dim]
    k_norm_b: torch.Tensor  # [head_dim]
    w_weights: torch.Tensor  # [n_heads, D]
    w_compress_gate: torch.Tensor  # [head_dim, D]
    ape: torch.Tensor  # [kpool, head_dim]


def pooled_states(
    packed_states: torch.Tensor, ape: torch.Tensor, kpool: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build compressed k-pool candidates (port of ``Glm5NextTextIndexer.get_pooled_states``).

    Pooling starts at each row's first valid token; a pool is selectable only
    when all its ``kpool`` tokens are valid. Returns ``(pool_keys [B, P, hd],
    pool_indices [B, P, kpool] with -1 holes, pool_valid [B, P])``.
    """
    head_dim = packed_states.shape[-1] // 2  # packed layout is [key | gate | valid]
    keys, gate_scores, valid_keys = torch.split(packed_states, [head_dim, head_dim, 1], dim=-1)
    valid_keys = valid_keys.bool().squeeze(-1)

    batch_size, seq_len = keys.shape[:2]
    number_of_pools = (seq_len + kpool - 1) // kpool

    first_key = torch.where(
        valid_keys.any(-1), valid_keys.long().argmax(-1),
        torch.full((batch_size,), seq_len, dtype=torch.long),
    )
    pool_offsets = torch.arange(number_of_pools * kpool)
    pool_offsets = pool_offsets.view(1, number_of_pools, kpool)
    pool_indices = first_key[:, None, None] + pool_offsets

    batch_idx = torch.arange(batch_size)[:, None, None]
    safe_indices = pool_indices.clamp(0, seq_len - 1)

    grouped_keys = keys[batch_idx, safe_indices]
    grouped_gate_scores = gate_scores[batch_idx, safe_indices]
    grouped_valid_keys = valid_keys[batch_idx, safe_indices]

    grouped_valid_keys = grouped_valid_keys & (pool_indices < seq_len)
    pool_valid = grouped_valid_keys.all(-1)
    pool_indices = pool_indices.masked_fill(~grouped_valid_keys, -1)

    # Learned weighted average over the tokens of each complete pool.
    logits = grouped_gate_scores + ape[None, None]
    logits = logits.masked_fill(~grouped_valid_keys[..., None], float("-inf"))
    probabilities = torch.nan_to_num(logits.softmax(dim=2))
    pool_keys = (probabilities * grouped_keys).sum(dim=2)

    keep = pool_valid.any(0)  # keep pools valid in any batch row
    return pool_keys[:, keep], pool_indices[:, keep], pool_valid[:, keep]


def append_visible_tail(
    topk_indices: torch.Tensor, token_visible: torch.Tensor, key_valid: torch.Tensor, kpool: int
) -> torch.Tensor:
    """Append the incomplete tail pool as raw indices (HF ``append_visible_tail``)."""
    max_tail_width = kpool - 1
    if max_tail_width == 0:
        return topk_indices
    batch_size, _, kv_length = token_visible.shape

    first_key = torch.where(
        key_valid.any(-1), key_valid.long().argmax(-1),
        torch.full((batch_size,), kv_length, dtype=torch.long),
    )
    visible_count = token_visible.long().sum(-1)
    tail_count = visible_count.remainder(kpool)
    tail_offsets = torch.arange(max_tail_width)

    tail_start = first_key[:, None] + visible_count - tail_count
    tail_indices = tail_start[..., None] + tail_offsets

    tail_valid = (tail_offsets[None, None, :] < tail_count[..., None]) & tail_indices.lt(kv_length)
    kv_idx = tail_indices.clamp(0, kv_length - 1)
    tail_visible = token_visible.gather(dim=-1, index=kv_idx)
    tail_indices = tail_indices.masked_fill(~(tail_valid & tail_visible), -1)
    return torch.cat([topk_indices, tail_indices], dim=-1)


def reference_indexer_topk(
    hidden: torch.Tensor,
    q_resid: torch.Tensor,
    weights: IndexerTensors,
    attention_mask: torch.Tensor,
    model: Glm53FlashConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """k-pool DSA indexer (port of ``Glm5NextTextIndexer.forward``, no cache).

    Returns ``(topk_indices int32 [B, S, index_topk + kpool - 1], internals)``;
    ``-1`` marks invalid slots. Valid values are raw key positions.
    """
    batch_size, seq_len = hidden.shape[:2]
    head_dim = model.index_head_dim
    kv_len = seq_len  # full prefill: no past cache

    q = (q_resid @ weights.wq_b.t()).view(batch_size, seq_len, -1, head_dim)
    k = F.layer_norm(hidden @ weights.wk.t(), (head_dim,), weights.k_norm_w, weights.k_norm_b, 1e-6)
    gate_scores = hidden @ weights.w_compress_gate.t()
    valid_channel = attention_mask.to(k.dtype)[..., None]
    packed_states = torch.cat([k, gate_scores, valid_channel], dim=-1)

    valid_keys = packed_states[..., -1].bool()
    kv_positions = torch.arange(kv_len)
    q_positions = torch.arange(seq_len)
    visible_tokens = (kv_positions[None, None, :] <= q_positions[None, :, None]) & valid_keys[:, None, :]

    pool_keys, pool_indices, pool_valid = pooled_states(packed_states, weights.ape, model.index_kpool)
    scores = torch.matmul(q, pool_keys.transpose(-1, -2).unsqueeze(1))
    scores = F.relu(scores * head_dim**-0.5)

    # Per-head weights, summed over heads: [B, S, 1, H] @ [B, S, H, P] -> [B, S, P].
    head_weights = (hidden @ weights.w_weights.t()) * (model.index_n_heads**-0.5)
    index_scores = torch.matmul(head_weights.unsqueeze(-2), scores).squeeze(-2)

    # A pool is selectable only if its final token is visible to the query.
    pool_end = pool_indices[..., -1].clamp(0, kv_len - 1)
    pool_visible = visible_tokens.gather(dim=-1, index=pool_end[:, None, :].expand(batch_size, seq_len, -1))
    valid_candidates = pool_visible & pool_valid[:, None]
    index_scores = index_scores.masked_fill(~valid_candidates, _FLOAT32_MIN)

    select_k = min(model.index_topk // model.index_kpool, index_scores.shape[-1])
    selected = index_scores.topk(select_k, dim=-1).indices
    batch_idx = torch.arange(batch_size)[:, None, None]
    selected_valid = valid_candidates.gather(-1, selected)
    selected_indices = pool_indices[batch_idx, selected]

    topk_indices = selected_indices.flatten(-2)
    invalid = ~selected_valid[..., None].expand_as(selected_indices).flatten(-2)
    topk_indices = topk_indices.masked_fill(invalid, -1)

    output_width = model.index_topk
    if model.index_kpool_always_select_tail:
        topk_indices = append_visible_tail(topk_indices, visible_tokens, valid_keys, model.index_kpool)
        output_width += model.index_kpool - 1
    topk_indices = F.pad(topk_indices, (0, output_width - topk_indices.shape[-1]), value=-1)
    topk_indices = topk_indices[..., :output_width]
    topk_indices = topk_indices.masked_fill(~attention_mask[..., None], -1)

    internals = {
        "pool_keys": pool_keys,
        "pool_indices": pool_indices,
        "pool_valid": pool_valid,
        "index_scores": index_scores,
    }
    return topk_indices.to(torch.int32), internals


def synthetic_indexer(model: Glm53FlashConfig, generator: torch.Generator) -> IndexerTensors:
    return IndexerTensors(
        wq_b=_randn((model.index_n_heads * model.index_head_dim, model.q_lora_rank), generator),
        wk=_randn((model.index_head_dim, model.hidden_size), generator),
        k_norm_w=1.0 + torch.randn(model.index_head_dim, generator=generator) * 0.05,
        k_norm_b=torch.randn(model.index_head_dim, generator=generator) * 0.05,
        w_weights=_randn((model.index_n_heads, model.hidden_size), generator),
        w_compress_gate=_randn((model.index_head_dim, model.hidden_size), generator),
        ape=torch.zeros(model.index_kpool, model.index_head_dim),
    )


def _test_indexer(model: Glm53FlashConfig) -> None:
    torch.manual_seed(17)
    b, s = 2, 18  # 4 full pools + a 2-token tail per row
    hidden = torch.randn(b, s, model.hidden_size) * 0.5
    q_resid = torch.randn(b, s, model.q_lora_rank) * 0.5
    weights = synthetic_indexer(model, torch.Generator().manual_seed(5))
    mask = torch.ones(b, s, dtype=torch.bool)
    mask[1, 15:] = False  # second row carries right padding

    topk, internals = reference_indexer_topk(hidden, q_resid, weights, mask, model)
    width = model.index_topk + model.index_kpool - 1
    assert topk.dtype == torch.int32 and topk.shape == (b, s, width)

    valid = topk.ge(0)
    assert int(topk.min()) >= -1 and int(topk.max()) < s
    # Causality: every selected key position is <= its query position.
    positions = torch.arange(s)
    assert bool((topk <= positions[None, :, None]).all()), "indexer selected a future key"
    # Padded query rows must be fully masked.
    assert not bool(valid[1, 15:].any()), "padded query rows must be -1"
    # Pooling starts at slot 0; with S=18 exactly 4 complete pools exist.
    assert int(internals["pool_valid"].any(0).sum()) == s // model.index_kpool
    # Query 17 of row 0 selects every pool: leading select_k*kpool slots hold the
    # pool quartets [4j, 4j+1, 4j+2, 4j+3]; the appended tail covers the last
    # S % kpool = 2 tokens (16, 17), then padding.
    row = topk[0, 17]
    select_k = min(model.index_topk // model.index_kpool, internals["pool_keys"].shape[1])
    pool_part = row[: select_k * model.index_kpool]
    assert bool(pool_part.ge(0).all())
    quartets = pool_part.reshape(-1, model.index_kpool)
    assert bool((quartets[:, 0] % model.index_kpool == 0).all())
    assert bool((quartets[:, 1:] - quartets[:, :-1] == 1).all()), f"quartets broken: {quartets}"
    tail = row[select_k * model.index_kpool : select_k * model.index_kpool + model.index_kpool - 1]
    assert tail.tolist() == [s - 2, s - 1, -1], f"tail is {tail.tolist()}, want [16, 17, -1]"
    print(f"  indexer: width {width}, causal, padding-masked, {select_k} quartet pools, tail {tail.tolist()}")


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


def _pool_axis_width(
    tokens: int,
    counts: tuple[int, ...],
    *,
    pool_stride: int | None = None,
    request_slots: int | None = None,
):
    """Return the leaf-padded score width for compact or stable-slot pools."""
    if pool_stride is None:
        axis = sum(counts)
    else:
        slots = len(counts) if request_slots is None else request_slots
        if slots < len(counts):
            raise ValueError("request_slots must cover every active request")
        if pool_stride <= 0 or pool_stride % INDEX_STATE_BLOCK_SIZE:
            raise ValueError("pool_stride must be a positive multiple of the pool page rows")
        if any(count < 0 or count > pool_stride for count in counts):
            raise ValueError("a request pool count exceeds pool_stride")
        axis = max(slots * pool_stride, tokens // INDEX_KPOOL + slots)
    return ((axis + 2 * SORT_TILE - 2) // SORT_TILE) * SORT_TILE


def _resident_default_tables(request_slots: int, table_max: int, pool_stride: int):
    """Return the resident forward's default ``cache_pages=None`` tables.

    ``block_table``/``pool_block_table`` are the explicit page tables the
    resident caller builds when it owns the whole shared cache, ``pool_offsets``
    the per-slot stable strides, and ``pool_rows`` the physical pooled-cache
    rows the resident ``POOL_TABLE_ROWS`` formula derives from the raw extent
    (the final raw-page span is the shared scratch). Passing these tables to
    :func:`models.glm5_3_flash.utils.build_indexer_chunk_metadata` reproduces
    the exact production lowering the resident forward runs on every chunk.
    """
    if request_slots <= 0:
        raise ValueError("resident fixtures need a positive request slot count")
    if table_max is None or table_max <= 0:
        raise ValueError("resident fixtures need a positive resident table_max")
    if pool_stride <= 0 or pool_stride % INDEX_STATE_BLOCK_SIZE:
        raise ValueError("pool_stride must be a positive multiple of the pool page rows")
    raw_rows = request_slots * table_max * BLOCK_SIZE
    pool_rows = raw_rows // INDEX_KPOOL + BLOCK_SIZE // INDEX_KPOOL
    block_table = torch.arange(request_slots * table_max, dtype=torch.int32).reshape(request_slots, table_max)
    pool_block_table = torch.arange(
        request_slots * (pool_stride // INDEX_STATE_BLOCK_SIZE), dtype=torch.int32,
    ).reshape(request_slots, pool_stride // INDEX_STATE_BLOCK_SIZE)
    pool_offsets = torch.arange(request_slots, dtype=torch.int32) * pool_stride
    return block_table, pool_block_table, pool_offsets, pool_rows


def _pool_batch(
    tokens: int,
    counts: tuple[int, ...],
    positions: tuple[int, ...],
    *,
    pad_pages: bool = True,
    page_map: str | None = None,
    pool_stride: int | None = None,
    request_slots: int | None = None,
    table_max: int | None = None,
):
    """Derive the score/topk batch geometry from per-request pool counts.

    Returns the leaf-aligned width, pooled-table row count, leaf-padded draw
    row count, per-query segment metadata, and pool row indirection. ``positions``
    contains request-local query positions. ``pool_stride`` places request
    segments at fixed stable slot offsets and sizes the full resident pool
    axis; without it, segments are compacted. ``pad_pages=False`` ends the
    page table at the last live page. ``page_map`` selects the page table's
    physical layout: the default resolves per layout — stable-slot batches
    lower the current resident explicit ``pool_block_table`` through the
    production metadata path (live pages mapped per slot, every unused entry
    aliasing the shared scratch page and the pooled cache sized by the
    resident row formula), while compact batches keep ``'perm'`` — and
    ``'perm'``/``'identity'`` remain bench-only diagnostics for either layout
    (a scattered permutation and the fully contiguous identity
    table, not the production layout).
    """
    if page_map is None:
        # Stable-slot batches default to the resident production layout; the
        # compact batch uses the scattered perm diagnostic.
        page_map = "resident" if pool_stride is not None else "perm"
    pools_true = sum(counts)
    width = _pool_axis_width(
        tokens, counts, pool_stride=pool_stride, request_slots=request_slots
    )
    per_request = tokens // len(counts)
    if len(positions) != tokens or per_request * len(counts) != tokens:
        raise ValueError("positions must cover every token; counts must tile the batch")
    seg_start = torch.zeros(tokens, dtype=torch.int32)
    pool_count = torch.zeros(tokens, dtype=torch.int32)
    for index, count in enumerate(counts):
        for row in range(per_request):
            token = index * per_request + row
            seg_start[token] = index * pool_stride if pool_stride is not None else sum(counts[:index])
            pool_count[token] = min((positions[token] + 1) // INDEX_KPOOL, count)
    if page_map == "resident":
        # Stable-slot default: derive the page table from the production
        # resident lowering rather than a hand-rolled layout. Every slot owns
        # explicit pooled pages, each active request's live pages map through
        # its table row, and every unused entry aliases the shared scratch
        # page — the layout the resident forward passes on every chunk. The
        # pooled cache carries the resident physical row count, not one page
        # per score-axis entry.
        if pool_stride is None:
            raise ValueError("the resident page map needs the stable-slot pool layout")
        slots = len(counts) if request_slots is None else request_slots
        (block_table, pool_block_table, pool_offsets, pool_table_rows) = _resident_default_tables(
            slots, table_max, pool_stride
        )
        lengths = torch.tensor(
            [count * INDEX_KPOOL for count in counts] + [0] * (slots - len(counts)), dtype=torch.int64,
        )
        query_start_loc = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.int64), lengths]), 0)
        metadata = build_forward_metadata(
            query_start_loc, kv_seq_lens=torch.zeros(slots, dtype=torch.int32),
            mla_block_table=block_table, index_block_table=block_table,
            pool_block_table=pool_block_table, kda_state_rows=torch.arange(slots, dtype=torch.int32),
        )
        lowered = build_indexer_chunk_metadata(
            metadata, block_table, pool_offsets, pools_cap=width, pool_rows=pool_table_rows,
            pool_block_table=pool_block_table,
        )
        # Draw enough rows for both the leaf-padded table and resident cache;
        # return the draw extent alongside the resident storage extent.
        legacy_pages = width if pad_pages else (pools_true + INDEX_STATE_BLOCK_SIZE - 1) // INDEX_STATE_BLOCK_SIZE
        legacy_table_rows = max(pool_table_rows, legacy_pages * INDEX_STATE_BLOCK_SIZE)
        return width, pool_table_rows, legacy_table_rows, seg_start, pool_count, lowered["pool_blocks"]
    # The pooled table is paged at INDEX_STATE_BLOCK_SIZE rows, the metadata
    # module's granularity: each compacted page lands at its own physical base
    # row, so the scorer's gather walks one entry per page. The default
    # permutation makes a compacted span physically discontinuous — the
    # gather's scattered worst case; ``page_map`` above picks the layout. Table
    # rows are therefore page-aligned. The default pads the table to the
    # leaf-aligned width; the production page table instead ends at the last
    # live page (``pad_pages=False``).
    if pad_pages:
        pages = width if pool_stride is not None else width // INDEX_STATE_BLOCK_SIZE
    else:
        pages = (pools_true + INDEX_STATE_BLOCK_SIZE - 1) // INDEX_STATE_BLOCK_SIZE
    pool_table_rows = pages * INDEX_STATE_BLOCK_SIZE
    if page_map == "perm":
        generator = torch.Generator().manual_seed(71)
        pool_blocks = torch.randperm(pages, generator=generator).to(torch.int32) * INDEX_STATE_BLOCK_SIZE
    elif page_map == "identity":
        pool_blocks = torch.arange(pages, dtype=torch.int32) * INDEX_STATE_BLOCK_SIZE
    elif page_map == "runs":
        pool_blocks = torch.arange(pages, dtype=torch.int32) * INDEX_STATE_BLOCK_SIZE
        if pages > 10:
            pool_blocks[[5, 9]] = pool_blocks[[9, 5]]
    else:
        raise ValueError(f"unknown page map {page_map!r}")
    return width, pool_table_rows, pool_table_rows, seg_start, pool_count, pool_blocks


def build_indexer_score_specs(
    tokens: int = 8,
    counts: tuple[int, ...] | None = None,
    positions: tuple[int, ...] | None = None,
    *,
    pad_pages: bool = True,
    page_map: str | None = None,
    pool_stride: int | None = None,
    request_slots: int | None = None,
    table_max: int | None = None,
):
    """Build one deterministic two-request score batch.

    Request 0 owns 3000 pools so its longest query spans two sort leaves, and
    the query positions ramp from a first token (visible count 0) to a full
    window, so the causal clamp and the leaf-crossing window both execute.
    ``counts``/``positions`` override the fixture geometry for the
    business-shape benchmark points: one request whose positions are the
    ``history`` ramp in front of the measured chunk. ``pad_pages=False`` ends
    the pooled page table at the last live page instead of the leaf boundary.
    ``page_map`` selects the page table's physical layout (see
    :func:`_pool_batch`; the default is the resident table for stable-slot
    batches and ``'perm'`` for compact ones) and with ``pool_stride`` the
    resident layout also needs the resident ``table_max`` (raw block-table
    pages per request slot) to size the shared pooled cache.
    """
    from golden import TensorSpec

    counts = (3000, 140) if counts is None else counts
    positions = (2, 6, 5000, 11999, 3, 7, 500, 559) if positions is None else positions
    (width, pool_table_rows, prior_draw_rows, seg_start, pool_count, pool_blocks) = _pool_batch(
        tokens,
        counts,
        positions,
        pad_pages=pad_pages,
        page_map=page_map,
        pool_stride=pool_stride,
        request_slots=request_slots,
        table_max=table_max,
    )

    generator = torch.Generator().manual_seed(73)

    def init_index_q():
        return torch.randn(tokens, INDEX_H, INDEX_DIM, generator=generator).bfloat16()

    # One leaf-padded draw supplies paired INT8 keys and dequant scales in
    # the Hadamard basis; the resident cache retains its row prefix.
    (prior_i8, prior_scale_col) = quantize_per_token_int8(
        torch.randn(prior_draw_rows, INDEX_DIM, generator=generator, dtype=torch.float32),
    )
    if prior_draw_rows > pool_table_rows:
        prior_i8 = prior_i8[:pool_table_rows]
        prior_scale_col = prior_scale_col[:pool_table_rows]

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
        TensorSpec("pool_scale", [pool_table_rows, POOL_SCALE_WIDTH], torch.float32, init_value=init_pool_scale),
        TensorSpec("pool_blocks", [len(pool_blocks)], torch.int32, init_value=lambda : pool_blocks),
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


def build_indexer_score_run_specs():
    """Score batch whose page table is mostly one physical run.

    A ``pool_block_table=None`` metadata lowering is the identity map
    (:func:`models.glm5_3_flash.utils.build_indexer_chunk_metadata`), so
    whole-table contiguity is that path's common case; the fixtures above
    permute pages instead and never exercise a run longer than one page. This
    batch keeps the default two-request geometry but swaps two interior pages
    of an identity table: two short runs, two break fallbacks, then one long
    run spanning several gather chunks — every branch of the gather's run
    merge under the same window/golden contract.
    """
    return build_indexer_score_specs(page_map="runs")


def build_indexer_topk_specs(
    tokens: int = 8,
    counts: tuple[int, ...] | None = None,
    positions: tuple[int, ...] | None = None,
    *,
    pool_stride: int | None = None,
    request_slots: int | None = None,
):
    """Score-and-select on one batch: synthetic scores, real windows.

    The score rows are random values inside each query's window and
    ``FP32_NEG_INF`` outside it, which is exactly the shape :func:`indexer_score`
    leaves behind, so the sort sees realistic value structure without depending
    on the scorer's numerics. ``counts``/``positions`` override the fixture
    geometry for the business-shape benchmark points.
    """
    from golden import TensorSpec

    counts = (3000, 140) if counts is None else counts
    positions = (2, 6, 5000, 11999, 3, 7, 500, 559) if positions is None else positions
    # Synthetic-score batches use the permuted page-map geometry without a
    # resident raw block-table extent.
    width, _, _, seg_start, pool_count, _ = _pool_batch(
        tokens,
        counts,
        positions,
        page_map="perm",
        pool_stride=pool_stride,
        request_slots=request_slots,
    )

    generator = torch.Generator().manual_seed(79)

    def init_index_scores():
        rows = torch.rand(tokens, width, generator=generator) * 10.0
        mask = torch.full((tokens, width), FP32_NEG_INF)
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
        TensorSpec("selected_pools", [tokens, KPOOL_SELECT_K], torch.int32, init_value=init_selected_pools),
        TensorSpec("selected_valid", [tokens, KPOOL_SELECT_K], torch.int32, init_value=init_selected_valid),
        TensorSpec("tail_start", [tokens], torch.int32, init_value=init_tail_start),
        TensorSpec("tail_count", [tokens], torch.int32, init_value=init_tail_count),
        TensorSpec("kv_len", [tokens], torch.int32, init_value=init_kv_len),
        TensorSpec("topk_indices", [tokens, TOPK_INDEX_WIDTH], torch.int32),
    ]


def golden_indexer_proj_case(tensors):
    """Fill the expected projections for :func:`build_indexer_proj_specs`."""
    (index_q, index_k, head_weights, gate_scores) = golden_indexer_proj(
        tensors["x"], tensors["q_resid"], tensors["w_q_b"], tensors["w_k"], tensors["k_norm_weight"],
        tensors["k_norm_bias"], tensors["w_weights"], tensors["w_compress_gate"],
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

    expected = golden_indexer_score(
        tensors["index_q"], tensors["hadamard"], tensors["pool_cache"], tensors["pool_scale"],
        tensors["pool_blocks"], tensors["head_weights"], tensors["seg_start"], tensors["pool_count"],
    )
    scores = tensors["index_scores"]
    scores[:] = FP32_NEG_INF
    width = min(expected.shape[1], scores.shape[1])
    scores[:, :width] = expected[:, :width]


def golden_indexer_topk_case(tensors):
    """Fill the expected selections for :func:`build_indexer_topk_specs`."""
    (selected, valid) = golden_indexer_topk(tensors["index_scores"], tensors["seg_start"], tensors["pool_count"])
    tensors["selected_pools"][:] = selected
    tensors["selected_valid"][:] = valid


def golden_indexer_expand_case(tensors):
    """Fill the expected expansion for :func:`build_indexer_expand_specs`."""
    tensors["topk_indices"][:] = golden_indexer_expand(
        tensors["selected_pools"], tensors["selected_valid"], tensors["tail_start"], tensors["tail_count"],
        tensors["kv_len"],
    )


def build_indexer_step_specs(
    tokens: int = 64,
    history: int = 0,
    requests: int = 1,
    *,
    pool_stride: int | None = None,
    request_slots: int | None = None,
    table_max: int | None = None,
    page_map: str = "resident",
):
    """Build one deterministic packed prefill chunk, end to end.

    Equal requests divide the history and chunk rows. With ``pool_stride``,
    request segments use stable slot offsets and the full resident pool axis,
    matching ``build_indexer_chunk_metadata``; otherwise they use compact
    offsets. Per-token lengths and causal windows remain request-local. The
    stable-slot default (``page_map='resident'``) lowers the batch through the
    production ``build_forward_metadata`` + ``build_indexer_chunk_metadata(
    ..., pool_block_table=...)`` pair, so the pooled page table carries the
    resident explicit per-slot live pages with every unused entry aliasing the
    shared scratch page and the pooled cache keeps the resident physical row
    count. The generator consumes one raw-page permutation followed by a
    leaf-padded prior-pool draw. ``page_map='identity'`` uses a contiguous
    identity table as an explicit non-production diagnostic. The compact
    (``pool_stride=None``) layout ignores ``page_map``.
    """
    from golden import TensorSpec

    if tokens % PROJ_T_TILE:
        raise ValueError(f"the projection stage needs a token count that is a multiple of {PROJ_T_TILE}")
    if requests < 1:
        raise ValueError("requests must be >= 1")
    if page_map not in ("resident", "identity"):
        raise ValueError(f"unknown page map {page_map!r}")
    per_tokens = tokens // requests if tokens % requests == 0 else None
    per_history = history // requests if history % requests == 0 else None
    if per_tokens is None or per_history is None:
        raise ValueError("requests must divide tokens and history evenly")
    per_total = per_history + per_tokens
    if requests > 1 and per_total % INDEX_KPOOL:
        raise ValueError(
            "the multi-request geometry needs each request's span to be a multiple of "
            f"{INDEX_KPOOL} so the packed pool count matches the single-request batch"
        )
    prior_pools = per_history // INDEX_KPOOL
    pools_per_request = per_total // INDEX_KPOOL
    actual_events = (pools_per_request - prior_pools) * requests
    width = _pool_axis_width(
        tokens,
        (pools_per_request,) * requests,
        pool_stride=pool_stride,
        request_slots=request_slots,
    )
    events = width if pool_stride is not None else actual_events
    if pool_stride is not None and (table_max is None or table_max <= 0):
        raise ValueError("stable-slot fixtures need a positive resident table_max")

    generator = torch.Generator().manual_seed(97)

    if pool_stride is not None and page_map == "resident":
        # Lower the batch through the production resident metadata path: the
        # explicit pool_block_table maps each active slot's live pooled pages,
        # every unused entry aliases the shared scratch page, the pooled cache
        # carries the resident physical row count, and padded events target
        # the scratch row with -1 members — exactly what the metadata module
        # emits for the capacity the score axis reserves.
        #
        # Advance the shared generator by one raw-page permutation, then
        # draw the leaf-padded prior-pool rows before the spec inputs.
        total = history + tokens
        needed_pages = (total + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
        torch.randperm(max(8, needed_pages + 3), generator=generator)[:needed_pages]
        slots = requests if request_slots is None else request_slots
        (block_table, pool_block_table, pool_offsets, pool_table_rows) = _resident_default_tables(
            slots, table_max, pool_stride
        )
        prior_draw_rows = max(pool_table_rows, width * INDEX_STATE_BLOCK_SIZE)
        lengths = torch.tensor(
            [per_tokens if request < requests else 0 for request in range(slots)], dtype=torch.int64,
        )
        previous = torch.tensor(
            [per_history if request < requests else 0 for request in range(slots)], dtype=torch.int64,
        )
        query_start_loc = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.int64), lengths]), 0)
        metadata = build_forward_metadata(
            query_start_loc, kv_seq_lens=previous, mla_block_table=block_table,
            index_block_table=block_table, pool_block_table=pool_block_table,
            kda_state_rows=torch.arange(slots, dtype=torch.int32),
        )
        lowered = build_indexer_chunk_metadata(
            metadata, block_table, pool_offsets, pools_cap=width, pool_rows=pool_table_rows,
            pool_block_table=pool_block_table,
        )
        index_slots = metadata.index_slots
        pool_token_slots = lowered["pool_token_slots"]
        pool_slots = lowered["pool_slots"]
        page_bases = lowered["pool_blocks"]
        seg_start = lowered["seg_start"]
        pool_count = lowered["pool_count_tok"]
        tail_start = lowered["tail_start"]
        tail_count = lowered["tail_count"]
        kv_len = lowered["kv_len_tok"]
        pages = width
    else:
        # Paged raw rows for every logical position of every request, past and
        # new, the way the metadata module hands them out: unordered pages,
        # in-order rows. Each request owns a contiguous span of logical
        # positions (its own prior tokens plus its chunk), mirroring
        # production's per-request sequential physical block assignment.
        total = history + tokens
        needed_pages = (total + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
        pages = torch.randperm(max(8, needed_pages + 3), generator=generator)[:needed_pages]

        def row_of(request, position):
            if pool_stride is None:
                position += request * per_total
            logical_page = position // BLOCK_SIZE
            if pool_stride is not None:
                if logical_page >= table_max:
                    raise ValueError("request position exceeds the resident block table")
                return (request * table_max + logical_page) * BLOCK_SIZE + position % BLOCK_SIZE
            return int(pages[logical_page]) * BLOCK_SIZE + position % BLOCK_SIZE

        index_slots = torch.tensor(
            [
                row_of(request, per_history + row)
                for request in range(requests)
                for row in range(per_tokens)
            ],
            dtype=torch.int32,
        )
        pool_token_slots = torch.full((events, INDEX_KPOOL), -1, dtype=torch.int32)
        pool_token_slots[:actual_events] = torch.tensor(
            [
                [row_of(request, INDEX_KPOOL * pool + k) for k in range(INDEX_KPOOL)]
                for request in range(requests)
                for pool in range(prior_pools, pools_per_request)
            ],
            dtype=torch.int32,
        )

        # The pooled table is paged at INDEX_STATE_BLOCK_SIZE rows like the
        # standalone fixtures: each compacted page lands at its own physical
        # base, and a pool's physical row derives from its compacted id. The
        # stable-slot identity branch uses a contiguous diagnostic layout.
        pages = width if pool_stride is not None else width // INDEX_STATE_BLOCK_SIZE
        pool_table_rows = pages * INDEX_STATE_BLOCK_SIZE
        prior_draw_rows = pool_table_rows
        if pool_stride is None:
            page_bases = torch.randperm(pages, generator=generator).to(torch.int32) * INDEX_STATE_BLOCK_SIZE
        else:
            page_bases = torch.arange(pages, dtype=torch.int32) * INDEX_STATE_BLOCK_SIZE

        def pool_row_of(compacted):
            return int(page_bases[compacted // INDEX_STATE_BLOCK_SIZE]) + compacted % INDEX_STATE_BLOCK_SIZE

        pool_slots = torch.full((events,), pool_table_rows - 1, dtype=torch.int32)
        pool_slots[:actual_events] = torch.tensor(
            [
                pool_row_of(
                    (request * pool_stride if pool_stride is not None else request * pools_per_request) + pool
                )
                for request in range(requests)
                for pool in range(prior_pools, pools_per_request)
            ],
            dtype=torch.int32,
        )

        seg_start = torch.zeros(tokens, dtype=torch.int32)
        pool_count = torch.zeros(tokens, dtype=torch.int32)
        tail_start = torch.zeros(tokens, dtype=torch.int32)
        tail_count = torch.zeros(tokens, dtype=torch.int32)
        kv_len = torch.zeros(tokens, dtype=torch.int32)
        for t in range(tokens):
            request, row = divmod(t, per_tokens)
            length = per_history + row + 1
            tail_count[t] = length % INDEX_KPOOL
            tail_start[t] = length - int(tail_count[t])
            seg_start[t] = request * pool_stride if pool_stride is not None else request * pools_per_request
            pool_count[t] = min(length // INDEX_KPOOL, pools_per_request)
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

    if pool_stride is None:
        raw_rows = int(max(index_slots.max(), pool_token_slots.max())) + 1
    else:
        raw_slots = requests if request_slots is None else request_slots
        raw_rows = raw_slots * table_max * BLOCK_SIZE

    def init_raw_cache():
        # History rows arrive as earlier steps left them: BF16-projected keys
        # widened back into the FP32 storage ABI beside FP32 gates.
        cache = torch.randn(raw_rows, INDEX_STATE_WIDTH, generator=generator)
        cache[:, :INDEX_DIM] = cache[:, :INDEX_DIM].bfloat16().float()
        return cache

    # One leaf-padded draw supplies paired INT8 keys and dequant scales in
    # the Hadamard basis; the resident cache retains its row prefix.
    (prior_i8, prior_scale_col) = quantize_per_token_int8(
        torch.randn(prior_draw_rows, INDEX_DIM, generator=generator, dtype=torch.float32),
    )
    if prior_draw_rows > pool_table_rows:
        prior_i8 = prior_i8[:pool_table_rows]
        prior_scale_col = prior_scale_col[:pool_table_rows]

    def init_pool_scale():
        # The writer replicates the scale across all POOL_SCALE_WIDTH lanes and
        # the scorer reads the row sum scaled by 1/8, so the prior table must
        # carry the replication too.
        return prior_scale_col.expand(-1, POOL_SCALE_WIDTH).contiguous().clone()

    specs = [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=lambda : torch.randn(tokens, D, generator=generator, dtype=torch.float32).bfloat16()),
        TensorSpec("q_resid", [tokens, Q_LORA], torch.bfloat16, init_value=lambda : torch.randn(tokens, Q_LORA, generator=generator, dtype=torch.float32).bfloat16()),
        TensorSpec("w_q_b", [Q_LORA, INDEX_H * INDEX_DIM], torch.bfloat16, init_value=init_weights("w_q_b", 0.02)),
        TensorSpec("w_k", [D, INDEX_DIM], torch.bfloat16, init_value=init_weights("w_k", 0.02)),
        TensorSpec("k_norm_weight", [INDEX_DIM], torch.bfloat16, init_value=lambda : (1.0 + 0.1 * torch.randn(INDEX_DIM, generator=generator)).bfloat16()),
        TensorSpec("k_norm_bias", [INDEX_DIM], torch.bfloat16, init_value=lambda : (0.1 * torch.randn(INDEX_DIM, generator=generator)).bfloat16()),
        TensorSpec("w_weights", [D, INDEX_H], torch.bfloat16, init_value=init_weights("w_weights", 0.05)),
        TensorSpec("w_compress_gate", [D, INDEX_DIM], torch.bfloat16, init_value=init_weights("w_compress_gate", 0.02)),
        TensorSpec("hadamard", [INDEX_DIM, INDEX_DIM], torch.bfloat16, init_value=sylvester_hadamard),
        TensorSpec("compress_ape", [INDEX_KPOOL, INDEX_DIM], torch.bfloat16, init_value=lambda : torch.randn(INDEX_KPOOL, INDEX_DIM, generator=generator).bfloat16()),
        TensorSpec("index_slots", [tokens], torch.int32, init_value=lambda: index_slots),
        TensorSpec("pool_token_slots", [events, INDEX_KPOOL], torch.int32, init_value=lambda : pool_token_slots),
        TensorSpec("pool_slots", [events], torch.int32, init_value=lambda: pool_slots),
        TensorSpec("pool_blocks", [pages], torch.int32, init_value=lambda: page_bases),
        TensorSpec("seg_start", [tokens], torch.int32, init_value=lambda: seg_start),
        TensorSpec("pool_count", [tokens], torch.int32, init_value=lambda: pool_count),
        TensorSpec("tail_start", [tokens], torch.int32, init_value=lambda: tail_start),
        TensorSpec("tail_count", [tokens], torch.int32, init_value=lambda: tail_count),
        TensorSpec("kv_len", [tokens], torch.int32, init_value=lambda: kv_len),
        TensorSpec("raw_cache", [raw_rows, INDEX_STATE_WIDTH], torch.float32, init_value=init_raw_cache),
        TensorSpec("pool_cache", [pool_table_rows, INDEX_DIM], torch.int8, init_value=lambda: prior_i8),
        TensorSpec("pool_scale", [pool_table_rows, POOL_SCALE_WIDTH], torch.float32, init_value=init_pool_scale),
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
    (index_q, index_k, head_weights, gate_scores) = golden_indexer_proj(
        tensors["x"], tensors["q_resid"], tensors["w_q_b"], tensors["w_k"], tensors["k_norm_weight"],
        tensors["k_norm_bias"], tensors["w_weights"], tensors["w_compress_gate"],
    )
    tensors["index_q"][:] = index_q
    tensors["index_k"][:] = index_k
    tensors["head_weights"][:] = head_weights
    tensors["gate_scores"][:] = gate_scores
    tensors["raw_cache"][:] = golden_indexer_cache_write(
        tensors["raw_cache"],
        index_k, gate_scores, tensors["index_slots"],
    )
    (pool_cache, pool_scale, pool_valid) = golden_indexer_pool_write(
        tensors["pool_cache"], tensors["pool_scale"], tensors["raw_cache"], tensors["compress_ape"],
        tensors["hadamard"], tensors["pool_token_slots"], tensors["pool_slots"],
    )
    tensors["pool_cache"][:] = pool_cache
    tensors["pool_scale"][:] = pool_scale
    tensors["pool_valid"][:] = pool_valid
    expected_scores = golden_indexer_score(
        tensors["index_q"], tensors["hadamard"], tensors["pool_cache"], tensors["pool_scale"],
        tensors["pool_blocks"], tensors["head_weights"], tensors["seg_start"], tensors["pool_count"],
    )
    score_width = tensors["index_scores"].shape[1]
    tensors["index_scores"][:] = expected_scores[:, :score_width]
    (selected, valid) = golden_indexer_topk(tensors["index_scores"], tensors["seg_start"], tensors["pool_count"])
    tensors["selected_pools"][:] = selected
    tensors["selected_valid"][:] = valid
    tensors["topk_indices"][:] = golden_indexer_expand(
        tensors["selected_pools"], tensors["selected_valid"], tensors["tail_start"], tensors["tail_count"],
        tensors["kv_len"],
    )
    for token in range(tensors["topk_indices"].shape[0]):
        row = tensors["topk_indices"][token]
        live = int((row >= 0).sum())
        assert (row[:live] >= 0).all() and (row[live:] == -1).all(), (
            f"row {token} violates the front-packed index ABI"
        )


def _prefill_self_check() -> None:
    """Exercise the four goldens on CPU with inlined fixtures.
    """
    torch.manual_seed(85)
    heads, dim, hidden = 4, 16, 64
    x = torch.randn(8, hidden).bfloat16()
    q_resid = torch.randn(8, 32).bfloat16()
    w_q_b = torch.randn(32, heads * dim).bfloat16() * 0.1
    w_k = torch.randn(hidden, dim).bfloat16() * 0.1
    k_w = torch.randn(dim).bfloat16()
    k_b = torch.randn(dim).bfloat16()
    (index_q, index_k, _, _) = golden_indexer_proj(
        x, q_resid, w_q_b, w_k, k_w, k_b, torch.randn(hidden, heads).bfloat16() * 0.1,
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


def golden_indexer_proj(
    x: torch.Tensor,
    q_resid: torch.Tensor,
    w_q_b: torch.Tensor,
    w_k: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    w_weights: torch.Tensor,
    w_compress_gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The four indexer projections in FP32, rounded once per output dtype.

    Weights arrive K-major ``[in, out]``; the reference's ``nn.Linear`` layout
    is the transpose, applied once at load time.
    """
    tokens = x.shape[0]
    heads = w_weights.shape[1]
    dim = w_k.shape[1]
    index_q = torch.nn.functional.linear(q_resid.float(), w_q_b.t().float())
    index_q = index_q.reshape(tokens, heads, dim).to(torch.bfloat16)

    projected = torch.nn.functional.linear(x.float(), w_k.t().float())
    mean = projected.mean(dim=-1, keepdim=True)
    centered = projected - mean
    variance = centered.square().mean(dim=-1, keepdim=True)
    normalized = centered * (1.0 / torch.sqrt(variance + K_NORM_EPS))
    index_k = (normalized * k_norm_weight.float() + k_norm_bias.float()).to(torch.bfloat16)

    head_weights = torch.nn.functional.linear(x.float(), w_weights.t().float()) * WEIGHTS_SCALE
    gate_scores = torch.nn.functional.linear(x.float(), w_compress_gate.t().float())
    return index_q, index_k, head_weights, gate_scores


def golden_indexer_score(
    index_q: torch.Tensor,
    hadamard: torch.Tensor,
    pool_cache: torch.Tensor,
    pool_scale: torch.Tensor,
    pool_blocks: torch.Tensor,
    head_weights: torch.Tensor,
    seg_start: torch.Tensor,
    pool_count: torch.Tensor,
) -> torch.Tensor:
    """Head-weighted ReLU scores of every query against the pooled table.

    The pooled table arrives already quantized — INT8 keys with their per-row
    dequant scales, rotated into this Hadamard basis and quantized once at pool
    close — and the query-side Hadamard rotation and INT8 quantization mirror
    the kernel op for op. ``pool_blocks`` holds one physical base row per
    compacted INDEX_STATE_BLOCK_SIZE-row page, the same page granularity the
    metadata module maps pool slots at. The reference
    ``Glm5NextTextIndexer`` scores in plain FP32, and the quantized path is
    the a2a3 deployment numerics, so the golden carries the quantization
    rather than approximating around it.
    """
    tokens = index_q.shape[0]
    pools = pool_blocks.shape[0] * INDEX_STATE_BLOCK_SIZE
    scores = torch.full((tokens, pools), FP32_NEG_INF, dtype=torch.float32)
    if pools == 0 or tokens == 0:
        return scores
    rotated = index_q.float().reshape(tokens * INDEX_H, INDEX_DIM) @ hadamard.float()
    q_i8, q_scale = quantize_per_token_int8(rotated)
    gathered = pool_blocks.to(torch.long).repeat_interleave(INDEX_STATE_BLOCK_SIZE) + torch.arange(
        INDEX_STATE_BLOCK_SIZE
    ).repeat(pool_blocks.shape[0])
    k_i8 = pool_cache[gathered]
    k_scale = pool_scale[gathered].sum(dim=-1, keepdim=True) * (0.125 * SOFTMAX_SCALE)
    for token in range(tokens):
        visible = int(pool_count[token])
        if visible <= 0:
            continue
        seg0 = int(seg_start[token])
        q_rows = slice(token * INDEX_H, (token + 1) * INDEX_H)
        dots = torch.matmul(k_i8[seg0:seg0 + visible].to(torch.int32), q_i8[q_rows].to(torch.int32).t()).float()
        dots = torch.relu(dots * k_scale[seg0 : seg0 + visible])
        query_weights = q_scale[q_rows].t() * head_weights[token]
        scores[token, seg0 : seg0 + visible] = (dots * query_weights).sum(dim=1)
    return scores


def golden_indexer_topk(
    index_scores: torch.Tensor,
    seg_start: torch.Tensor,
    pool_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the selected segment-local pool ids and which of them are real.

    A window at or below ``KPOOL_SELECT_K`` pools takes the kernel's short
    route: every visible pool is selected, materialized in compact id order.
    The order inside the prefix is ABI-free (the consumer is a permutation-
    invariant softmax), so this golden adopts the same identity convention and
    the selection compares exactly against the short route.
    """
    tokens = index_scores.shape[0]
    selected = torch.full((tokens, KPOOL_SELECT_K), -1, dtype=torch.int32)
    valid = torch.zeros(tokens, KPOOL_SELECT_K, dtype=torch.int32)
    for token in range(tokens):
        visible = max(int(pool_count[token]), 0)
        if visible > 0:
            keep = min(KPOOL_SELECT_K, visible)
            if visible <= KPOOL_SELECT_K:
                picked = torch.arange(keep)
            else:
                seg0 = int(seg_start[token])
                picked = torch.topk(index_scores[token, seg0 : seg0 + visible], keep).indices
            selected[token, :keep] = picked.to(torch.int32)
            valid[token, :keep] = 1
    return selected, valid


def golden_indexer_expand(
    selected_pools: torch.Tensor,
    selected_valid: torch.Tensor,
    tail_start: torch.Tensor,
    tail_count: torch.Tensor,
    kv_len: torch.Tensor,
) -> torch.Tensor:
    """Expand the selections into front-packed raw positions, one row per query."""
    tokens = selected_pools.shape[0]
    expanded = torch.full((tokens, TOPK_INDEX_WIDTH), -1, dtype=torch.int32)
    lanes = torch.arange(TOPK_INDEX_WIDTH)
    pool_raw = lanes // INDEX_KPOOL
    pool_id = pool_raw.clamp(max=selected_pools.shape[1] - 1)
    rem = lanes - pool_raw * INDEX_KPOOL
    for token in range(tokens):
        groups = min(int(kv_len[token]) // INDEX_KPOOL, KPOOL_SELECT_K)
        hist_len = groups * INDEX_KPOOL
        base = int(tail_start[token])
        tail = int(tail_count[token])
        hist_pos = selected_pools[token].to(torch.int64)[pool_id] * INDEX_KPOOL + rem
        tail_offset = lanes - hist_len
        hist_active = ((lanes < hist_len) & (selected_valid[token].to(torch.int64)[pool_id] > 0)).to(torch.int64)
        tail_active = ((tail_offset >= 0) & (tail_offset < tail)).to(torch.int64)
        positions = hist_active * (hist_pos + 1) + tail_active * (base + tail_offset + 1) - 1
        expanded[token] = positions.to(torch.int32)
    return expanded


def golden_indexer_cache_write(
    cache: torch.Tensor,
    index_k: torch.Tensor,
    gate_scores: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """Scatter one step's raw rows, skipping ``-1`` slots.

    The guard is the ABI, not politeness: an unmasked ``-1`` indexes the last row of
    the pool through Python's negative indexing and silently corrupts it.
    """
    packed = torch.cat([index_k, gate_scores], dim=-1)
    updated = cache.clone()
    written = slots >= 0
    updated[slots.to(torch.long)[written]] = packed.to(cache.dtype)[written]
    return updated


def golden_indexer_pool_write(
    pool_cache: torch.Tensor,
    pool_scale: torch.Tensor,
    raw_cache: torch.Tensor,
    compress_ape: torch.Tensor,
    hadamard: torch.Tensor,
    pool_token_slots: torch.Tensor,
    pool_slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Close each pool event: per-dimension softmax over its four gate lanes.

    The pooled key is rotated into the scorer's Hadamard basis (with the same
    BF16 round the kernel's operand takes) and quantized to INT8 plus a per-row
    dequant scale at write time. FP32 softmax and reduction use ordinary PyTorch
    operations; device rounding can differ at quantization boundaries.

    Args:
        pool_cache: ``[POOL_TABLE * INDEX_STATE_BLOCK_SIZE, INDEX_DIM]`` INT8
            pooled-key table holding the prior state.
        pool_scale: ``[POOL_TABLE * INDEX_STATE_BLOCK_SIZE, POOL_SCALE_WIDTH]``
            FP32 per-row dequant scales holding the prior state.
        raw_cache: ``[TABLE * BLOCK_SIZE, INDEX_STATE_WIDTH]`` raw support table.
        compress_ape: ``[INDEX_KPOOL, INDEX_DIM]`` learned pool-lane bias.
        hadamard: ``[INDEX_DIM, INDEX_DIM]`` BF16 scorer-basis rotation.
        pool_token_slots: ``[events, INDEX_KPOOL]`` physical raw row of each member
            token, ``-1`` for a padded member.
        pool_slots: ``[events]`` destination pooled row of each event.

    Returns:
        The updated INT8 table, the updated scale table and the per-event
        validity (all four members present).
    """
    updated = pool_cache.clone()
    updated_scale = pool_scale.clone()
    valid = torch.zeros(pool_slots.numel(), POOL_VALID_WIDTH, dtype=torch.int32)
    for event in range(pool_slots.numel()):
        members = pool_token_slots[event].to(torch.long)
        destination = int(pool_slots[event])
        present = bool((members >= 0).all())
        if present:
            keys = raw_cache[members, 0:INDEX_DIM].float()
            gates = raw_cache[members, INDEX_DIM:].float()
            probabilities = torch.softmax(gates + compress_ape.float(), dim=0)
            pooled = (probabilities * keys).sum(dim=0, keepdim=True)
        else:
            pooled = torch.zeros(1, INDEX_DIM)
        pooled = pooled.to(torch.bfloat16).float() @ hadamard.float()
        key_i8, key_scale = quantize_per_token_int8(pooled)
        updated[destination] = key_i8[0]
        updated_scale[destination, :] = key_scale[0, 0]
        valid[event, 0] = 1 if present else 0
    return updated, updated_scale, valid


def build_indexer_cache_write_specs(tokens: int = 24, pages: int = 4):
    """Build one deterministic raw scatter: unordered distinct slots and a padded row.

    Row 3 carries ``-1``, and the destination rows are deliberately out of order
    across pages so the write cannot pass by walking the pool linearly. ``cache`` is
    ``InOut`` so its untouched rows are uploaded and can be asserted, rather than
    read back as allocator residue.
    """
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(59)
    cache_rows = pages * BLOCK_SIZE

    def init_index_k():
        return torch.randn(tokens, INDEX_DIM, generator=generator, dtype=torch.float32).bfloat16()

    def init_gate_scores():
        return torch.randn(tokens, INDEX_DIM, generator=generator, dtype=torch.float32)

    def init_cache():
        return torch.randn(cache_rows, INDEX_STATE_WIDTH, generator=generator, dtype=torch.float32)

    def init_slots():
        chosen = torch.randperm(cache_rows, generator=generator)[:tokens]
        slots = chosen.to(torch.int32)
        slots[3] = -1
        return slots

    return [
        TensorSpec("index_k", [tokens, INDEX_DIM], torch.bfloat16, init_value=init_index_k),
        TensorSpec("gate_scores", [tokens, INDEX_DIM], torch.float32, init_value=init_gate_scores),
        TensorSpec("slots", [tokens], torch.int32, init_value=init_slots),
        TensorSpec("cache", [cache_rows, INDEX_STATE_WIDTH], torch.float32, init_value=init_cache),
    ]


def build_indexer_pool_write_specs(events: int = 6, raw_pages: int = 2, pool_pages: int = 8):
    """Build one deterministic pool-close batch: fresh events and one padded pool.

    Member slots are grouped four-per-event out of consecutive raw rows so the
    golden's pooling window is unambiguous, the destinations are a permutation of
    pooled rows, event 0's members are forced onto physical rows 0-3 (a slot of 0
    is valid — the regression corner of the validity flag) and event 4 carries a
    ``-1`` member to exercise the padded-pool rejection path. The raw table is
    built as the cache write would leave it: BF16 keys widened to FP32 beside
    FP32 gates.
    """
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(61)
    raw_rows = raw_pages * BLOCK_SIZE
    pool_rows = pool_pages * INDEX_STATE_BLOCK_SIZE
    member_rows = events * INDEX_KPOOL
    if member_rows > raw_rows or events > pool_rows:
        raise ValueError("fixture sizes exceed the requested pool geometry")

    def init_raw_cache():
        keys = torch.randn(raw_rows, INDEX_DIM, generator=generator)
        keys = keys.bfloat16().float()
        gates = torch.randn(raw_rows, INDEX_DIM, generator=generator)
        return torch.cat([keys, gates], dim=-1)

    def init_compress_ape():
        return torch.randn(INDEX_KPOOL, INDEX_DIM, generator=generator).bfloat16()

    def init_pool_token_slots():
        base = torch.randperm(raw_rows - INDEX_KPOOL + 1, generator=generator)[:events]
        offsets = torch.arange(INDEX_KPOOL)
        slots = (base.unsqueeze(1) + offsets).to(torch.int32)
        # Event 0's members are physical rows 0-3: row 0 is a legal slot, and a
        # flag derived by clamping the slot value would report it invalid.
        slots[0] = torch.arange(INDEX_KPOOL, dtype=torch.int32)
        slots[4, 3] = -1
        return slots

    def init_pool_slots():
        chosen = torch.randperm(pool_rows, generator=generator)[:events]
        return chosen.to(torch.int32)

    # One draw feeds both tables so the prior INT8 rows and their scales match;
    # the rows read as already Hadamard-rotated prior state, the way earlier
    # pool writes would have left them.
    (prior_i8, prior_scale_col) = quantize_per_token_int8(
        torch.randn(pool_rows, INDEX_DIM, generator=generator, dtype=torch.float32),
    )

    def init_pool_cache():
        return prior_i8

    def init_pool_scale():
        scale = torch.zeros(pool_rows, POOL_SCALE_WIDTH)
        scale[:, 0] = prior_scale_col[:, 0]
        return scale

    return [
        TensorSpec("raw_cache", [raw_rows, INDEX_STATE_WIDTH], torch.float32, init_value=init_raw_cache),
        TensorSpec("compress_ape", [INDEX_KPOOL, INDEX_DIM], torch.bfloat16, init_value=init_compress_ape),
        TensorSpec("hadamard", [INDEX_DIM, INDEX_DIM], torch.bfloat16, init_value=sylvester_hadamard),
        TensorSpec("pool_token_slots", [events, INDEX_KPOOL], torch.int32, init_value=init_pool_token_slots),
        TensorSpec("pool_slots", [events], torch.int32, init_value=init_pool_slots),
        TensorSpec("pool_cache", [pool_rows, INDEX_DIM], torch.int8, init_value=init_pool_cache),
        TensorSpec("pool_scale", [pool_rows, POOL_SCALE_WIDTH], torch.float32, init_value=init_pool_scale),
        TensorSpec("pool_valid", [events, POOL_VALID_WIDTH], torch.int32),
    ]


def golden_indexer_cache_write_case(tensors):
    """Fill the expected raw pool state for :func:`build_indexer_cache_write_specs`."""
    tensors["cache"][:] = golden_indexer_cache_write(
        tensors["cache"], tensors["index_k"], tensors["gate_scores"], tensors["slots"],
    )


def golden_indexer_pool_write_case(tensors):
    """Fill the expected pooled tables and validity for :func:`build_indexer_pool_write_specs`."""
    (pool_cache, pool_scale, pool_valid) = golden_indexer_pool_write(
        tensors["pool_cache"], tensors["pool_scale"], tensors["raw_cache"], tensors["compress_ape"],
        tensors["hadamard"], tensors["pool_token_slots"], tensors["pool_slots"],
    )
    tensors["pool_cache"][:] = pool_cache
    tensors["pool_scale"][:] = pool_scale
    tensors["pool_valid"][:] = pool_valid


def _cache_self_check() -> None:
    """Prove both goldens on CPU with inlined fixtures, before any device run.
    """
    generator = torch.Generator().manual_seed(53)
    cache = torch.randn(4 * BLOCK_SIZE, INDEX_STATE_WIDTH, generator=generator)
    index_k = torch.randn(8, INDEX_DIM, generator=generator).bfloat16()
    gates = torch.randn(8, INDEX_DIM, generator=generator)
    slots = torch.tensor([5, 0, -1, 11, 300, 128, 257, 3], dtype=torch.int32)
    updated = golden_indexer_cache_write(cache, index_k, gates, slots)
    for row, slot in enumerate(slots.tolist()):
        if slot >= 0:
            expected = torch.cat([index_k[row].float(), gates[row]], dim=-1)
            assert torch.equal(updated[slot], expected), f"slot {slot} did not receive its row"
    assert torch.equal(updated[-1], cache[-1]), "a -1 slot wrapped into the last row"

    pool_i8, pool_scale_col = quantize_per_token_int8(torch.randn(8, INDEX_DIM, generator=generator))
    pool_cache = pool_i8
    pool_scale = torch.zeros(8, POOL_SCALE_WIDTH)
    pool_scale[:, 0] = pool_scale_col[:, 0]
    raw_cache = torch.cat(
        [torch.randn(32, INDEX_DIM, generator=generator).bfloat16().float(), torch.randn(32, INDEX_DIM, generator=generator)],
        dim=-1,
    )
    ape = torch.randn(INDEX_KPOOL, INDEX_DIM, generator=generator).bfloat16()
    hadamard = sylvester_hadamard()
    pool_token_slots = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, -1, 15], [16, 17, 18, 19]], dtype=torch.int32,
    )
    pool_slots = torch.tensor([7, 3, 0, 5, 2], dtype=torch.int32)
    (new_cache, new_scale, valid) = golden_indexer_pool_write(
        pool_cache, pool_scale, raw_cache, ape, hadamard, pool_token_slots, pool_slots,
    )
    assert valid[:, 0].tolist() == [1, 1, 1, 0, 1], valid
    members = pool_token_slots[0].long()
    weights = torch.softmax(raw_cache[members, INDEX_DIM:].float() + ape.float(), dim=0)
    expected = (weights * raw_cache[members, 0:INDEX_DIM].float()).sum(dim=0, keepdim=True)
    expected = expected.to(torch.bfloat16).float() @ hadamard.float()
    expected_i8, expected_scale = quantize_per_token_int8(expected)
    assert torch.equal(new_cache[7], expected_i8[0]), "pool 0 INT8 bytes differ"
    assert (new_scale[7] == expected_scale[0, 0]).all(), "pool 0 scale lanes differ"
    zero_i8, zero_scale = quantize_per_token_int8(torch.zeros(1, INDEX_DIM))
    assert torch.equal(new_cache[5], zero_i8[0].to(torch.int8)), "padded pool is not quantized zero"
    assert (new_scale[5] == zero_scale[0, 0]).all(), "padded pool scale lanes differ"
    assert torch.equal(new_cache[1], pool_cache[1]), "an untouched pooled row changed"
    assert torch.equal(new_scale[1], pool_scale[1]), "an untouched scale row changed"
    print("[GOLDEN] PASS indexer_cache self-check")


def run_decode_case(argv=None):
    """Prove the composed golden on CPU, then validate one decode step on device."""
    import argparse

    from golden import ratio_allclose, run, topk_pair_compare

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument(
        "--case", default="all", choices=["step", "abi", "all"],
        help="step: the indexer composition alone; abi: the composition with the sparse attention consuming its index list; all: both",
    )
    parser.add_argument(
        "--bench", action="store_true",
        help="run the step at a business shape without golden validation; the shape comes from --requests/--prior-tokens, timing from PYPTO_BENCH=1",
    )
    parser.add_argument("--prior-tokens", type=int, default=8192, help="per-request history length in tokens; defines the bench geometry")
    args = parser.parse_args(argv)

    def print_bench(label: str, result) -> None:
        stats = result.bench
        if stats is None:
            print(f"[BENCH] {label}: no timing (run with PYPTO_BENCH=1)")
            return
        print(
            f"[BENCH] {label}: device_us median={stats.device_us_median:.1f} min={stats.device_us_min:.1f} mean={stats.device_us_mean:.1f} max={stats.device_us_max:.1f} rounds={stats.rounds}",
        )

    if args.bench:
        if args.requests * DECODE_ROWS_PER_REQUEST % PROJ_T_TILE:
            parser.error(f"--bench needs a token count that is a multiple of {PROJ_T_TILE}")
        prior_pools = args.prior_tokens // INDEX_KPOOL
        specs = build_decode_indexer_step_specs(args.requests, prior_pools)
        pools_true = args.requests * (prior_pools + 1)
        width = ((pools_true + 2 * SORT_TILE - 2) // SORT_TILE) * SORT_TILE
        label = (
            f"decode_step requests={args.requests} prior_tokens={args.prior_tokens} "
            f"T={args.requests * 4} pools={pools_true} width={width}"
        )
        result = run(
            fn=decode_indexer_step_test,
            specs=specs,
            golden_fn=None,
            config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
            compile_only=args.compile_only,
        )
        print_bench(label, result)
        if not result.passed:
            raise SystemExit(result.error or 1)
        return

    def selected_pools_compare(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        """Tie-aware selection compare that also audits the picked scores.

        :func:`topk_pair_compare` adjudicates index mismatches through the
        actual ordering alone, so a sorted strictly-worse subset would pass
        it. The picked scores must therefore also match the scores the golden
        picked, under the same tolerance: a legal tie swap keeps the paired
        values equal, a real miss breaks them.
        """
        scores = actual_outputs["index_scores"].float()
        seg0 = inputs["seg_start"].long().unsqueeze(1)

        def paired_of(selected):
            cols = (selected.long() + seg0).clamp(0, scores.shape[-1] - 1)
            paired = torch.gather(scores, 1, cols)
            return torch.where(selected < 0, torch.full_like(paired, -torch.inf), paired)

        paired = paired_of(actual)
        synth_outputs = {**actual_outputs, "_selected_paired_scores": paired}
        (ok, msg) = topk_pair_compare("_selected_paired_scores")(
            actual, expected, actual_outputs=synth_outputs, expected_outputs=expected_outputs, inputs=inputs,
            rtol=rtol, atol=atol,
        )
        if not ok:
            return ok, msg
        expected_paired = paired_of(expected)
        if not torch.allclose(paired, expected_paired, rtol=rtol, atol=atol):
            diff = (paired - expected_paired).abs()
            diff = torch.where(torch.isfinite(diff), diff, torch.full_like(diff, torch.inf))
            return False, (
                "    selected scores differ from the golden-selected scores "
                f"(max |diff|={diff.max().item():.6g})"
            )
        return True, ""

    def exact_compare(actual, expected, **_kwargs):
        exact = torch.equal(actual.cpu(), expected.cpu())
        return exact, "" if exact else "    integer output differs from golden"

    def score_window_compare(actual, expected, *, inputs, rtol, atol, **_kwargs):
        """Compare only each query's visible window: lanes outside it are
        undefined scratch under the scorer's contract (the top-k clamps
        through its valid shapes), so they carry nothing to check."""

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

    def topk_indices_compare(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        """Tie-aware position compare for the expanded front-packed rows.

        A near-boundary score cluster can reshuffle two pools whose scores each
        sit inside the scorer's own tolerance; the swap is legal exactly when
        the swapped positions' paired scores agree under the same allclose rule
        that certified ``index_scores``. Tail and padding lanes are deterministic
        given the inputs, so any mismatch there fails outright, and every actual
        row must keep the front-packed ABI.
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

    step_compare = {
        "index_q": ratio_allclose(atol=1e-3, rtol=1.0 / 64, max_error_ratio=0.01),
        "index_k": ratio_allclose(atol=1e-3, rtol=1.0 / 64, max_error_ratio=0.01),
        "head_weights": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.01),
        "gate_scores": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.01),
        # The raw rows carry the projection rounding of the composed step,
        # unlike the standalone scatter, which moves bytes at zero tolerance.
        "raw_cache": ratio_allclose(atol=1e-3, rtol=1.0 / 64, max_error_ratio=0.01),
        # The pooled table is quantized on write; the softmax transcendentals
        # can flip a rint boundary, so allow a tiny tail of one-quantum flips.
        "pool_cache": ratio_allclose(atol=1.0, rtol=0.0, max_error_ratio=0.005),
        "pool_scale": ratio_allclose(atol=1e-6, rtol=1e-3, max_error_ratio=0.005),
        "pool_valid": exact_compare,
        "index_scores": score_window_compare,
        "selected_pools": selected_pools_compare,
        "selected_valid": exact_compare,
        "topk_indices": topk_indices_compare,
    }

    results = []
    captured = {}

    def capture_topk_indices(actual, expected, **kwargs):
        # Stash the device's own expanded list: the ABI stage replays it as an
        # input to the sparse-attention kernel, the way the serving graph
        # would hand the buffer over.
        captured["topk_indices"] = actual.detach().cpu().clone()
        return topk_indices_compare(actual, expected, **kwargs)

    if args.case in ("step", "all"):
        results.append(
            run(
                fn=decode_indexer_step_test,
                specs=build_decode_indexer_step_specs(args.requests),
                golden_fn=golden_decode_indexer_step_case,
                config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
                rtol=1.0 / 64,
                atol=0.001,
                compare_fn=step_compare,
                compile_only=args.compile_only,
            ),
        )
    if args.case in ("abi", "all"):
        step_specs = build_decode_indexer_step_specs(args.requests)
        kv_len = {spec.name: spec for spec in step_specs}["kv_len"].init_value()
        results.append(
            run(
                fn=decode_indexer_step_test,
                specs=step_specs,
                golden_fn=golden_decode_indexer_step_case,
                config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
                rtol=1.0 / 64,
                atol=0.001,
                compare_fn={**step_compare, "topk_indices": capture_topk_indices},
                compile_only=args.compile_only,
            ),
        )
        if results[-1].passed and not args.compile_only:
            indices = captured["topk_indices"]
            live = (indices >= 0).sum(dim=1)
            for row in range(indices.shape[0]):
                assert (indices[row, : live[row]] >= 0).all() and (indices[row, live[row] :] == -1).all(), (
                    f"row {row} violates the front-packed index ABI"
                )
                assert live[row] == 0 or int(indices[row, : live[row]].max()) < int(kv_len[row]), (
                    f"row {row} selects a position past its causal length"
                )
            results.append(
                run(
                    fn=decode_sparse_attn_test,
                    specs=build_decode_indexer_abi_specs(indices, kv_len),
                    golden_fn=golden_decode_sparse_attn_case,
                    config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
                    rtol=1.0 / 64,
                    atol=0.001,
                    compare_fn={"output": ratio_allclose(atol=0.001, rtol=1.0 / 64)},
                    compile_only=args.compile_only,
                ),
            )
    for result in results:
        print(result)
        if not result.passed:
            raise SystemExit(result.error or 1)


def run_prefill_case(argv=None):
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

    _prefill_self_check()

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--case", default="all", choices=["proj", "score", "topk", "expand", "step", "all"])
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument(
        "--bench", action="store_true",
        help="run the case at a business shape without golden validation; the shape comes from --tokens/--history, timing from PYPTO_BENCH=1",
    )
    parser.add_argument("--history", type=int, default=0, help="history tokens in front of the --tokens chunk; defines the bench geometry")
    parser.add_argument(
        "--bench-requests",
        dest="bench_requests",
        type=int,
        default=1,
        metavar="N",
        help="pack the bench as N equal active requests in the resident stable-slot "
        "pool layout: request segments use fixed strides and the score/step fixtures "
        "carry the current resident explicit pool_block_table with shared scratch "
        "aliases (default 1 keeps the frozen shapes)",
    )
    parser.add_argument(
        "--bench-page-map",
        dest="bench_page_map",
        choices=["resident", "perm", "identity"],
        default=None,
        help="bench-only pool page table for --case score: 'resident' lowers the current "
        "resident explicit pool_block_table (live pages per slot, shared scratch aliases) "
        "and is the multi-request default; 'identity' is the fully contiguous pre-rebase "
        "diagnostic, NOT the production layout; 'perm' is the scattered diagnostic. "
        "Single-request benches default to 'perm'",
    )
    args = parser.parse_args(argv)

    bench_pool_stride = None
    bench_request_slots = None
    bench_table_max = None
    if args.bench_requests > 1:
        if args.bench_requests > 16:
            parser.error("--bench-requests cannot exceed the resident batch's 16 request slots")
        per_request_span = (args.tokens + args.history) // args.bench_requests
        max_seq_len = max(args.tokens, per_request_span)
        bench_table_max = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        pool_pages_per_request = (max_seq_len + INDEX_KPOOL - 1) // INDEX_KPOOL
        bench_pool_stride = (
            (pool_pages_per_request + INDEX_STATE_BLOCK_SIZE - 1) // INDEX_STATE_BLOCK_SIZE
        ) * INDEX_STATE_BLOCK_SIZE
        bench_request_slots = 16

    def print_bench(label: str, result) -> None:
        stats = result.bench
        if stats is None:
            print(f"[BENCH] {label}: no timing (run with PYPTO_BENCH=1)")
            return
        print(
            f"[BENCH] {label}: device_us median={stats.device_us_median:.1f} min={stats.device_us_min:.1f} mean={stats.device_us_mean:.1f} max={stats.device_us_max:.1f} rounds={stats.rounds}",
        )

    def bench_label(name: str, pools: int = 0, width: int = 0) -> str:
        label = f"{name} T={args.tokens} history={args.history}"
        if args.bench_requests > 1:
            label += f" reqs={args.bench_requests}"
        if pools:
            width = width or ((pools + 2 * SORT_TILE - 2) // SORT_TILE) * SORT_TILE
            label += f" pools={pools} width={width}"
        return label

    bench_counts = None
    bench_positions = None
    if args.bench:
        if args.tokens % PROJ_T_TILE:
            parser.error(f"--bench needs a token count that is a multiple of {PROJ_T_TILE}")
        requests = args.bench_requests
        if requests < 1:
            parser.error("--bench-requests must be >= 1")
        if requests > 1 and args.case in ("proj", "expand"):
            parser.error(
                f"--case {args.case} has no pool-space geometry (its shape follows "
                "--tokens/--history alone); --bench-requests only packs the "
                "score/topk/step batches"
            )
        span = args.tokens + args.history
        if requests == 1:
            bench_counts = (span // INDEX_KPOOL,)
            bench_positions = tuple(range(args.history, args.history + args.tokens))
        else:
            # Resident multi-request packing: N active requests divide the
            # tokens and history evenly. Each request has request-local
            # metadata and a fixed pool stride in one of 16 stable slots;
            # the explicit raw/pool block tables retain the unused-slot
            # extents, with unused pooled entries aliasing the shared
            # scratch page as in the resident metadata lowering.
            if args.tokens % requests or args.history % requests or span // requests % INDEX_KPOOL:
                parser.error(
                    "--bench-requests must divide --tokens and --history evenly, with "
                    "each request's span a multiple of the pool size "
                    f"({INDEX_KPOOL}), so every resident request has whole-pool metadata"
                )
            per_tokens = args.tokens // requests
            per_history = args.history // requests
            per_pools = (per_tokens + per_history) // INDEX_KPOOL
            bench_counts = (per_pools,) * requests
            bench_positions = tuple(
                position
                for request in range(requests)
                for position in range(per_history, per_history + per_tokens)
            )
        bench_width = _pool_axis_width(
            args.tokens,
            bench_counts,
            pool_stride=bench_pool_stride,
            request_slots=bench_request_slots,
        )
        # Pool-preparation cases use one block per 16 pages for large T and at
        # most 48 persistent workers for small T. Other cases have no pool
        # gather or scale-pack launch.
        if args.case in ("score", "step", "all"):
            bench_pages = bench_width if bench_pool_stride is not None else bench_width // INDEX_STATE_BLOCK_SIZE
            bench_pool_chunks = (bench_pages + POOL_GATHER_PAGE_TILE - 1) // POOL_GATHER_PAGE_TILE
            if args.tokens <= INDEXER_PERSISTENT_MAX_TOKENS:
                bench_pool_blocks = min(bench_pool_chunks, POOL_GATHER_WORKERS)
            else:
                bench_pool_blocks = bench_pool_chunks
            if bench_pool_blocks > 32256:
                parser.error(
                    f"bench pool table width {bench_width} needs {bench_pool_blocks} pool-preparation blocks, "
                    "past the runtime int16 set_block_num limit (blocks ≤ 32256)"
                )
        if args.case in ("step", "all") and bench_pool_stride is not None and bench_width > 32256:
            parser.error(
                f"bench pool table width {bench_width} launches {bench_width} pool-write blocks, "
                "past the runtime int16 set_block_num limit (blocks ≤ 32256)"
            )

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
        (ok, msg) = topk_pair_compare("_selected_paired_scores")(
            actual, expected, actual_outputs=synth_outputs, expected_outputs=expected_outputs, inputs=inputs,
            rtol=rtol, atol=atol,
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
            inputs["index_scores"].float(), actual, expected, actual_outputs=actual_outputs,
            expected_outputs=expected_outputs, inputs=inputs, rtol=rtol, atol=atol,
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
            actual_outputs["index_scores"].float(), actual, expected, actual_outputs=actual_outputs,
            expected_outputs=expected_outputs, inputs=inputs, rtol=rtol, atol=atol,
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
                golden_fn=golden_indexer_proj_case, rtol=1.0 / 64, atol=0.001,
                compare_fn={
                    "index_q": ratio_allclose(atol=0.001, rtol=1.0 / 64, max_error_ratio=0.01),
                    "index_k": ratio_allclose(atol=0.001, rtol=1.0 / 64, max_error_ratio=0.01),
                    "head_weights": ratio_allclose(atol=0.001, rtol=0.001, max_error_ratio=0.01),
                    "gate_scores": ratio_allclose(atol=0.001, rtol=0.001, max_error_ratio=0.01),
                },
            )
        results.append(
            run(
                fn=indexer_proj_test,
                specs=build_indexer_proj_specs(args.tokens),
                config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
                compile_only=args.compile_only,
                **extra,
            ),
        )
        if args.bench:
            print_bench(label, results[-1])
    if args.case in ("score", "all"):
        if args.bench:
            if args.bench_page_map is not None and args.case == "all":
                parser.error("--bench-page-map needs --case score (the probe measures the score stage alone)")
            page_map = args.bench_page_map or ("resident" if requests > 1 else "perm")
            if page_map == "resident" and requests < 2:
                parser.error(
                    "--bench-page-map resident needs --bench-requests > 1 "
                    "(the resident explicit table lives in the stable-slot layout)"
                )
            specs = build_indexer_score_specs(
                args.tokens,
                bench_counts,
                bench_positions,
                page_map=page_map,
                pool_stride=bench_pool_stride,
                request_slots=bench_request_slots,
                table_max=bench_table_max,
            )
            extra = dict(golden_fn=None)
            label = bench_label("score", sum(bench_counts), bench_width)
            if requests > 1 or args.bench_page_map is not None:
                label += f" pages={page_map}"
        else:
            specs = build_indexer_score_specs()
            extra = dict(
                golden_fn=golden_indexer_score_case, rtol=1.0 / 128, atol=0.0001,
                compare_fn={"index_scores": score_window_compare},
            )
        results.append(
            run(
                fn=indexer_score_test,
                specs=specs,
                config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
                compile_only=args.compile_only,
                **extra,
            ),
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
                    atol=0.0001,
                    compare_fn={"index_scores": score_window_compare},
                    config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
                    compile_only=args.compile_only,
                ),
            )
            # Mostly-identity page table: the gather's run-merge branches (wide
            # copies, break fallbacks, chunked long runs) need a contiguous
            # table the permuted fixtures never produce.
            results.append(
                run(
                    fn=indexer_score_test,
                    specs=build_indexer_score_run_specs(),
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
                ),
            )
    if args.case in ("topk", "all"):
        if args.bench:
            specs = build_indexer_topk_specs(
                args.tokens,
                bench_counts,
                bench_positions,
                pool_stride=bench_pool_stride,
                request_slots=bench_request_slots,
            )
            extra = dict(golden_fn=None)
            label = bench_label("topk", sum(bench_counts), bench_width)
        else:
            specs = build_indexer_topk_specs()
            extra = dict(
                golden_fn=golden_indexer_topk_case, rtol=0.001, atol=0.001,
                compare_fn={"selected_pools": selected_pools_compare, "selected_valid": exact_compare},
            )
        results.append(
            run(
                fn=indexer_topk_test,
                specs=specs,
                config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
                compile_only=args.compile_only,
                **extra,
            ),
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
            extra = dict(golden_fn=golden_indexer_expand_case, rtol=0.0, atol=0.0)
        results.append(
            run(
                fn=indexer_expand_test,
                specs=specs,
                config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
                compile_only=args.compile_only,
                **extra,
            ),
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
        # with two fresh ones (history not a multiple of the pool size). An
        # explicit multi-request geometry replaces the ramp with one tier.
        if args.bench or args.history or args.bench_requests > 1:
            tiers = [(args.tokens, args.history)]
        else:
            tiers = [(args.tokens, 0), (args.tokens, 4 * args.tokens - 6)]
        for tier_tokens, tier_history in tiers:
            specs = build_indexer_step_specs(
                tier_tokens,
                tier_history,
                requests=args.bench_requests,
                pool_stride=bench_pool_stride,
                request_slots=bench_request_slots,
                table_max=bench_table_max,
            )
            if args.bench:
                pools = (tier_tokens + tier_history) // INDEX_KPOOL
                extra = dict(golden_fn=None)
                label = bench_label("step", pools, bench_width)
            else:
                extra = dict(
                    golden_fn=golden_indexer_prefill_step_case, rtol=1.0 / 64, atol=0.001,
                    compare_fn=step_compare,
                )
            results.append(
                run(
                    fn=indexer_prefill_step_test,
                    specs=specs,
                    config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
                    compile_only=args.compile_only,
                    **extra,
                ),
            )
            if args.bench:
                print_bench(label, results[-1])
    for result in results:
        print(result)
        if not result.passed:
            raise SystemExit(result.error or 1)


def run_cache_case(argv=None):
    """Prove the goldens on CPU, then validate both writers on device.

    The raw scatter is pure data movement, so it runs at zero tolerance. The
    pooled key is an FP32 softmax quantized once to INT8 plus a dequant scale;
    device and reference differ only in the softmax transcendentals, which
    almost always lands on the same quantized byte — the budget allows a tiny
    tail of one-quantum flips at rint boundaries.
    """
    import argparse

    from golden import ratio_allclose, run

    _cache_self_check()

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--case", default="both", choices=["write", "pool", "both"])
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--events", type=int, default=6)
    parser.add_argument("--raw-pages", type=int, default=2)
    parser.add_argument("--pool-pages", type=int, default=8)
    parser.add_argument(
        "--bench", action="store_true",
        help="skip golden validation; shapes come from --tokens/--pages/--events, timing from PYPTO_BENCH=1",
    )
    args = parser.parse_args(argv)

    def print_bench(label: str, result) -> None:
        stats = result.bench
        if stats is None:
            print(f"[BENCH] {label}: no timing (run with PYPTO_BENCH=1)")
            return
        print(
            f"[BENCH] {label}: device_us median={stats.device_us_median:.1f} min={stats.device_us_min:.1f} mean={stats.device_us_mean:.1f} max={stats.device_us_max:.1f} rounds={stats.rounds}",
        )

    results = []
    if args.case in ("write", "both"):
        results.append(
            run(
                fn=indexer_cache_write_test,
                specs=build_indexer_cache_write_specs(args.tokens, args.pages),
                golden_fn=None if args.bench else golden_indexer_cache_write_case,
                config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
                rtol=0.0,
                atol=0.0,
                compile_only=args.compile_only,
            ),
        )
        if args.bench:
            print_bench(f"cache_write tokens={args.tokens} pages={args.pages}", results[-1])
    if args.case in ("pool", "both"):
        results.append(
            run(
                fn=indexer_pool_write_test,
                specs=build_indexer_pool_write_specs(args.events, args.raw_pages, args.pool_pages),
                golden_fn=None if args.bench else golden_indexer_pool_write_case,
                config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
                rtol=1.0 / 64,
                atol=0.001,
                compare_fn=None if args.bench else {"pool_cache": ratio_allclose(atol=1.0, rtol=0.0, max_error_ratio=0.005), "pool_scale": ratio_allclose(atol=1e-06, rtol=0.001, max_error_ratio=0.005)},
                compile_only=args.compile_only,
            )
        )
        if args.bench:
            print_bench(
                f"pool_write events={args.events} raw_pages={args.raw_pages} pool_pages={args.pool_pages}",
                results[-1],
            )
    for result in results:
        print(result)
        if not result.passed:
            raise SystemExit(result.error or 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    components = ["all", "cache", "prefill", "decode"]
    parser.add_argument("--component", choices=components, default="all")
    options, remaining = parser.parse_known_args(argv)
    if options.component == "all" and any(arg in ("-h", "--help") for arg in remaining):
        help_parser = argparse.ArgumentParser(description=__doc__, parents=[parser])
        help_parser.epilog = "Use --component NAME --help for stage options."
        help_parser.parse_args(argv)
    cases = {
        "cache": run_cache_case,
        "prefill": run_prefill_case,
        "decode": run_decode_case,
    }
    selected = cases if options.component == "all" else {options.component: cases[options.component]}
    for run_case in selected.values():
        run_case(remaining)


if __name__ == "__main__":
    main()
