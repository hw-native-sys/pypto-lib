# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compression projections, rotary embedding, cache publication, and ratio-2 ring state."""

from pathlib import Path

import pypto.language as pl
from pypto.runtime import pto_isa_include_dir
import torch

from config import (
    FLASH as M, STATE_CAPACITY, B_DYN, Q_START_DYN, STATE_BLOCKS_DYN,
    T_DYN, COMPRESSED_CACHE_GROUP, CMP_BLOCKS_DYN,
)
from quant import unpack_bf16_nz
from rmsnorm import golden_rms_norm


# model config
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
EPS = M.rms_norm_eps
STATE_WIDTH = 2 * M.head_dim

# tiling
C1A_PROJ_M_TILE = 48
C1A_PROJ_N_TILE = 64
C1A_PROJ_K_TILE = 512
C2A_PROJ_M_TILE = 80
C2A_PROJ_N_TILE = 64
C2A_PROJ_K_TILE = 256


@pl.jit.inline(auto_scope=False)
def c1a_compressor_project(
    source: pl.Tensor,
    weight: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Project NZ weights with full-height Cube tiles and store active rows."""
    blocks = (num_tokens + C1A_PROJ_M_TILE - 1) // C1A_PROJ_M_TILE * (HEAD_DIM // C1A_PROJ_N_TILE)
    workers = pl.max(blocks, 1)
    with pl.spmd(workers, name_hint="proj_cmp", deps=[ready]) as completed:
        with pl.at(level=pl.Level.CORE_GROUP):
            block = pl.tile.get_block_idx()
            row = block // (HEAD_DIM // C1A_PROJ_N_TILE) * C1A_PROJ_M_TILE
            col = block % (HEAD_DIM // C1A_PROJ_N_TILE) * C1A_PROJ_N_TILE
            rows = pl.max(pl.min(C1A_PROJ_M_TILE, num_tokens - row), 0)
            a = pl.load(source, [row, 0], [C1A_PROJ_M_TILE, C1A_PROJ_K_TILE], valid_shape=[rows, C1A_PROJ_K_TILE], target_memory=pl.Mem.Mat)
            b = pl.load(weight, [0, col], [C1A_PROJ_K_TILE, C1A_PROJ_N_TILE], target_memory=pl.Mem.Mat)
            full_a = pl.set_validshape(a, C1A_PROJ_M_TILE, C1A_PROJ_K_TILE)
            acc = pl.matmul(full_a, b)
            for k in pl.pipeline(C1A_PROJ_K_TILE, D, C1A_PROJ_K_TILE, stage=2):
                a = pl.load(source, [row, k], [C1A_PROJ_M_TILE, C1A_PROJ_K_TILE], valid_shape=[rows, C1A_PROJ_K_TILE], target_memory=pl.Mem.Mat)
                b = pl.load(weight, [k, col], [C1A_PROJ_K_TILE, C1A_PROJ_N_TILE], target_memory=pl.Mem.Mat)
                full_a = pl.set_validshape(a, C1A_PROJ_M_TILE, C1A_PROJ_K_TILE)
                acc = pl.matmul_acc(acc, full_a, b)
            value = pl.set_validshape(acc, rows, C1A_PROJ_N_TILE)
            output = pl.store(value, [row, col], output)
    return output, completed


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels" / "compressor_publish.cpp",
               include_dirs=(pto_isa_include_dir(), str(Path(__file__).parent / "kernels")))
def _c1a_compress(
    source: pl.Tensor,
    slots: pl.Tensor,
    cache: pl.InOut[pl.Tensor],
    scales: pl.InOut[pl.Tensor],
    cosine: pl.Tensor,
    sine: pl.Tensor,
    norm_weight: pl.Tensor,
    latent: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    """Normalize, retain unrotated latent, and publish nearest-even E2M1 rows."""
    ...


@pl.jit.inline(auto_scope=False)
def c1a_compressor_publish(
    source: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    cosine: pl.Tensor,
    sine: pl.Tensor,
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    scales: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    source_ready: pl.Scalar[pl.TASK_ID],
):
    """Normalize and publish MXFP4 rows while retaining the unrotated latent."""
    active_tokens = pl.cast(num_tokens, pl.INT32)
    workers = pl.max(pl.min(active_tokens, 8), 1)
    with pl.spmd(workers, name_hint="c1a_compress", deps=[cache_ready, source_ready]) as published:
        cache = _c1a_compress(source, slots, cache, scales, cosine, sine, norm_weight, latent, active_tokens)
    return published


@pl.jit.inline
def _c2a_compressor_project_tile(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    output: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    row: pl.Scalar[pl.INDEX],
    col: pl.Scalar[pl.INDEX],
    rows: pl.Scalar[pl.INDEX],
):
    a = pl.load(x, [row, 0], [C2A_PROJ_M_TILE, C2A_PROJ_K_TILE], valid_shape=[rows, C2A_PROJ_K_TILE], target_memory=pl.Mem.Mat)
    b = pl.load(weight, [0, col], [C2A_PROJ_K_TILE, C2A_PROJ_N_TILE], target_memory=pl.Mem.Mat)
    a_full = pl.set_validshape(a, C2A_PROJ_M_TILE, C2A_PROJ_K_TILE)
    acc = pl.matmul(a_full, b)
    for k in pl.pipeline(C2A_PROJ_K_TILE, D, C2A_PROJ_K_TILE, stage=2):
        a = pl.load(x, [row, k], [C2A_PROJ_M_TILE, C2A_PROJ_K_TILE], valid_shape=[rows, C2A_PROJ_K_TILE], target_memory=pl.Mem.Mat)
        b = pl.load(weight, [k, col], [C2A_PROJ_K_TILE, C2A_PROJ_N_TILE], target_memory=pl.Mem.Mat)
        a_full = pl.set_validshape(a, C2A_PROJ_M_TILE, C2A_PROJ_K_TILE)
        acc = pl.matmul_acc(acc, a_full, b)
    value = pl.set_validshape(acc, rows, C2A_PROJ_N_TILE)
    output = pl.store(value, [row, col], output)
    return output


@pl.jit.inline
def c2a_compressor_project(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    kv_out: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    score_out: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Project KV and gate with NZ BF16 weights and FP32 Cube output."""
    blocks = (num_tokens + C2A_PROJ_M_TILE - 1) // C2A_PROJ_M_TILE * (2 * HEAD_DIM // C2A_PROJ_N_TILE)
    with pl.spmd(pl.max(blocks, 1), name_hint="proj_cmp"):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_cmp"):
            block = pl.tile.get_block_idx()
            row = block // (2 * HEAD_DIM // C2A_PROJ_N_TILE) * C2A_PROJ_M_TILE
            col = block % (HEAD_DIM // C2A_PROJ_N_TILE) * C2A_PROJ_N_TILE
            rows = pl.max(pl.min(C2A_PROJ_M_TILE, num_tokens - row), 0)
            if block % (2 * HEAD_DIM // C2A_PROJ_N_TILE) < HEAD_DIM // C2A_PROJ_N_TILE:
                kv_out = _c2a_compressor_project_tile(x, wkv, kv_out, row, col, rows)
            else:
                score_out = _c2a_compressor_project_tile(x, wgate, score_out, row, col, rows)
    return kv_out, score_out


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels" / "compressor_pool.cpp",
               include_dirs=(pto_isa_include_dir(), str(Path(__file__).parent / "kernels")))
def _c2a_compress(
    kv_proj: pl.Tensor,
    score_proj: pl.Tensor,
    state_cache: pl.InOut[pl.Tensor],
    position_ids: pl.Tensor,
    norm_weight: pl.Tensor,
    latent: pl.InOut[pl.Tensor],
    query_start_loc: pl.Tensor,
    token_to_req_indices: pl.Tensor,
    state_block_table: pl.Tensor,
    slots: pl.Tensor,
    cosine: pl.Tensor,
    sine: pl.Tensor,
    cache: pl.InOut[pl.Tensor],
    scales: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
    num_state_blocks: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    """Pool, normalize, publish compressed rows, and update request ring state."""
    ...


@pl.jit.inline(auto_scope=False)
def c2a_compressor_publish(
    kv_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    score_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    state_ready: pl.Scalar[pl.TASK_ID],
    cosine: pl.Tensor,
    sine: pl.Tensor,
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    scales: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16],
):
    """Pool and publish rows with state reads and writes owned by physical block."""
    active_tokens = pl.cast(num_tokens, pl.INT32)
    blocks = pl.cast(pl.tensor.dim(state_cache, 0), pl.INT32)
    workers = pl.max(pl.min(pl.tensor.dim(state_block_table, 0), 16), 1)
    with pl.spmd(workers, name_hint="c2a_compress", deps=[state_ready]) as completed:
        _c2a_compress(
            kv_proj, score_proj, state_cache, position_ids, norm_weight, latent,
            query_start_loc, token_to_req_indices, state_block_table,
            slots, cosine, sine, cache, scales, active_tokens, blocks,
        )
    return completed


def golden_c1a_compressor(
    x: torch.Tensor,
    wkv: torch.Tensor,
    norm_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute C1A NZ-weight projection followed by RMSNorm."""
    projected = torch.matmul(x.float(), unpack_bf16_nz(wkv).float()).to(x.dtype)
    return golden_rms_norm(projected, norm_weight)


def golden_c2a_compressor(
    x: torch.Tensor,
    query_start_loc: torch.Tensor,
    position_ids: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    state_block_table: torch.Tensor,
    state_cache: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute C2A compression with request-owned FP32 ring state.

    Query boundaries count valid packed tokens; any tensor tail is padding.
    Negative/out-of-range request or block indices suppress state access and
    publication. Valid chunks contain consecutive absolute positions.
    """
    kv = torch.matmul(x.float(), unpack_bf16_nz(wkv).float())
    score = torch.matmul(x.float(), unpack_bf16_nz(wgate).float())
    latent = torch.zeros_like(kv, dtype=x.dtype)
    publish = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    blocks, capacity, width = state_cache.shape
    if capacity < 1 or width != 2 * kv.shape[-1] or state_cache.dtype != torch.float32:
        raise ValueError("state_cache must be FP32 [blocks, positive capacity, 2 * head_dim]")
    head_dim = kv.shape[-1]
    for request in range(state_block_table.shape[0]):
        start = int(query_start_loc[request])
        end = min(int(query_start_loc[request + 1]), x.shape[0])
        block = int(state_block_table[request, 0])
        if start >= end or not 0 <= block < blocks:
            continue
        for token in range(start, end):
            position = int(position_ids[token])
            if int(token_to_req_indices[token]) != request or position < 0:
                continue
            if position % 2:
                if token > start:
                    previous_kv, previous_score = kv[token - 1], score[token - 1]
                else:
                    previous = state_cache[block, (position - 1) % capacity]
                    previous_kv, previous_score = previous[:head_dim], previous[head_dim:]
                pair_kv = torch.stack((previous_kv, kv[token]))
                pair_score = torch.stack((previous_score, score[token]))
                pooled = (pair_kv * pair_score.softmax(dim=0)).sum(dim=0)
                latent[token] = golden_rms_norm(pooled.to(x.dtype), norm_weight)
                publish[token] = True
        for token in range(max(start, end - capacity), end):
            position = int(position_ids[token])
            if int(token_to_req_indices[token]) == request and position >= 0:
                state_cache[block, position % capacity, :head_dim] = kv[token]
                state_cache[block, position % capacity, head_dim:] = score[token]
    return latent, publish
