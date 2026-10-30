# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compression projections, rotary embedding, cache publication, and ratio-2 ring state."""

import pypto.language as pl
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


@pl.jit.inline
def _compressor_rope_block(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    block: pl.Scalar[pl.INT32],
):
    t = block
    row = pl.tile.load(x, [t, 0], [1, HEAD_DIM])
    tail_bf16 = pl.tile.slice(row, [1, ROPE_DIM], [0, HEAD_DIM - ROPE_DIM])
    tail = pl.cast(tail_bf16, pl.FP32)
    even = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P0101)
    odd = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P1010)
    c = pl.tile.load(cos, [t, 0], [1, ROPE_DIM // 2])
    sine = pl.tile.load(sin, [t, 0], [1, ROPE_DIM // 2])
    s = pl.mul(sine, 1.0)
    even_cos = pl.mul(even, c)
    odd_sin = pl.mul(odd, s)
    re = pl.sub(even_cos, odd_sin)
    even_sin = pl.mul(even, s)
    odd_cos = pl.mul(odd, c)
    im = pl.add(even_sin, odd_cos)
    real_lanes = pl.tile.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
    real_lanes = pl.tile.scatter_mask(real_lanes, re, mask_pattern=pl.tile.MaskPattern.P0101)
    imag_lanes = pl.tile.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
    imag_lanes = pl.tile.scatter_mask(imag_lanes, im, mask_pattern=pl.tile.MaskPattern.P1010)
    real_bits = pl.reinterpret_view(real_lanes, pl.UINT32)
    imag_bits = pl.reinterpret_view(imag_lanes, pl.UINT32)
    rotated_bits = pl.or_(real_bits, imag_bits)
    rotated = pl.reinterpret_view(rotated_bits, pl.FP32)
    rounded = pl.cast(rotated, pl.BF16, mode="rint")
    row = pl.tile.assemble(row, rounded, [0, HEAD_DIM - ROPE_DIM])
    pl.tile.store(row, [block, 0], output)
    return output


@pl.jit.inline(auto_scope=False)
def compressor_rope(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    blocks = num_tokens
    workers = pl.min(blocks, 12)
    with pl.spmd(workers, name_hint="rope_cmp"):
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, blocks, workers):
            _compressor_rope_block(x, cos, sin, output, block)
    return output


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


@pl.jit.inline(auto_scope=False)
def c1a_compressor_publish(
    source: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    scales: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    """Publish group-16 MXFP4 with BF16 scales and nearest-even rounding."""
    cache_rows = pl.tensor.dim(cache, 0) * 128
    flat = pl.reshape(cache, [cache_rows, HEAD_DIM // 2])
    flat_scale = pl.reshape(scales, [cache_rows, HEAD_DIM // COMPRESSED_CACHE_GROUP])
    # Each physical cache page has one publishing worker.
    workers = pl.min(num_tokens, 8)
    with pl.spmd(workers, name_hint="pub_cmp", deps=[cache_ready]) as published:
        worker = pl.tile.get_block_idx()
        for t in pl.range(num_tokens):
            slot = pl.read(slots, [t])
            if slot >= 0 and (slot // 128) % workers == worker:
                physical = pl.cast(slot, pl.INDEX)
                row = pl.tile.load(source, [t, 0], [1, HEAD_DIM])
                values = pl.reshape(pl.cast(row, pl.FP32), [HEAD_DIM // COMPRESSED_CACHE_GROUP, COMPRESSED_CACHE_GROUP])
                temporary = pl.tile.create([HEAD_DIM // COMPRESSED_CACHE_GROUP, COMPRESSED_CACHE_GROUP], dtype=pl.FP32)
                amax = pl.row_max(pl.abs(values), tmp_tile=temporary)
                factors_raw = pl.maximum(pl.mul(amax, 1.0 / 6.0), 0.001953125)
                stored_scale = pl.cast(factors_raw, pl.BF16, mode="rint")
                factors = pl.cast(stored_scale, pl.FP32)
                normalized = pl.minimum(pl.maximum(pl.row_expand_div(values, factors), -6.0), 6.0)
                magnitude = pl.abs(normalized)
                rounded = pl.tile.full([HEAD_DIM // COMPRESSED_CACHE_GROUP, COMPRESSED_CACHE_GROUP], dtype=pl.FP32, value=0.0)
                select_tmp = pl.tile.create([1, 16], dtype=pl.UINT32)
                predicate = pl.tile.cmps(magnitude, 0.25, cmp_type=4)
                level = pl.tile.full([HEAD_DIM // COMPRESSED_CACHE_GROUP, COMPRESSED_CACHE_GROUP], dtype=pl.FP32, value=0.5)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                predicate = pl.tile.cmps(magnitude, 0.75, cmp_type=5)
                level = pl.tile.full([HEAD_DIM // COMPRESSED_CACHE_GROUP, COMPRESSED_CACHE_GROUP], dtype=pl.FP32, value=1.0)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                predicate = pl.tile.cmps(magnitude, 1.25, cmp_type=4)
                level = pl.tile.full([HEAD_DIM // COMPRESSED_CACHE_GROUP, COMPRESSED_CACHE_GROUP], dtype=pl.FP32, value=1.5)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                predicate = pl.tile.cmps(magnitude, 1.75, cmp_type=5)
                level = pl.tile.full([HEAD_DIM // COMPRESSED_CACHE_GROUP, COMPRESSED_CACHE_GROUP], dtype=pl.FP32, value=2.0)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                predicate = pl.tile.cmps(magnitude, 2.5, cmp_type=4)
                level = pl.tile.full([HEAD_DIM // COMPRESSED_CACHE_GROUP, COMPRESSED_CACHE_GROUP], dtype=pl.FP32, value=3.0)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                predicate = pl.tile.cmps(magnitude, 3.5, cmp_type=5)
                level = pl.tile.full([HEAD_DIM // COMPRESSED_CACHE_GROUP, COMPRESSED_CACHE_GROUP], dtype=pl.FP32, value=4.0)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                predicate = pl.tile.cmps(magnitude, 5.0, cmp_type=4)
                level = pl.tile.full([HEAD_DIM // COMPRESSED_CACHE_GROUP, COMPRESSED_CACHE_GROUP], dtype=pl.FP32, value=6.0)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                # Convert the exact RNE levels to E2M1 codes, then pack adjacent lanes.
                lower_codes = pl.minimum(pl.mul(rounded, 2.0), 4.0)
                upper_codes = pl.minimum(pl.maximum(pl.sub(rounded, 2.0), 0.0), 3.0)
                codes = pl.add(lower_codes, upper_codes)
                sign = pl.ands(pl.shrs(pl.reinterpret_view(normalized, pl.INT32), 28), 8)
                codes = pl.reshape(pl.or_(pl.cast(codes, pl.INT32), sign), [1, HEAD_DIM])
                low = pl.tile.gather_mask(codes, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.INT32)
                high = pl.tile.gather_mask(codes, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                packed = pl.add(low, pl.mul(high, 16))
                signed_bytes = pl.sub(packed, pl.mul(pl.shrs(packed, 7), 256))
                payload_bytes = pl.reinterpret_view(pl.cast(signed_bytes, pl.INT8), pl.UINT8)
                payload = pl.set_validshape(payload_bytes, 1, HEAD_DIM // 2)
                scale_row = pl.reshape(stored_scale, [1, HEAD_DIM // COMPRESSED_CACHE_GROUP])
                encoded = pl.set_validshape(scale_row, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP)
                pl.tile.store(payload, [physical, 0], flat)
                pl.tile.store(encoded, [physical, 0], flat_scale)
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


@pl.jit.inline
def c2a_compressor_pool(
    kv_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    score_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    num_state_blocks: pl.Scalar[pl.INT32],
    pair_ready: pl.Scalar[pl.TASK_ID],
):
    """Assemble partners and pool eight rows before updating ring state."""
    state_rows = num_state_blocks * STATE_CAPACITY
    flat = pl.reshape(state_cache, [state_rows, STATE_WIDTH])
    with pl.spmd(pl.max((num_tokens + 7) // 8, 1), name_hint="pool_cmp", deps=[pair_ready]) as pool_tid:
        group = pl.tile.get_block_idx()
        t = group * 8
        if t < num_tokens:
            rows = pl.min(8, num_tokens - t)
            current_kv = pl.slice(kv_proj, [8, HEAD_DIM], [t, 0], valid_shape=[rows, HEAD_DIM])
            current_kv = pl.set_validshape(pl.fillpad(current_kv, pad_value=pl.PadValue.zero), 8, HEAD_DIM)
            current_score = pl.slice(score_proj, [8, HEAD_DIM], [t, 0], valid_shape=[rows, HEAD_DIM])
            current_score = pl.set_validshape(pl.fillpad(current_score, pad_value=pl.PadValue.zero), 8, HEAD_DIM)
            previous_kv = pl.full([8, HEAD_DIM], dtype=pl.FP32, value=0.0)
            previous_score = pl.full([8, HEAD_DIM], dtype=pl.FP32, value=0.0)
            for lane in pl.unroll(8):
                token = t + lane
                if token < num_tokens:
                    request = pl.cast(pl.read(token_to_req_indices, [token]), pl.INDEX)
                    if request >= 0 and request < pl.tensor.dim(state_block_table, 0):
                        block = pl.cast(pl.read(state_block_table, [request, 0]), pl.INDEX)
                        if block >= 0 and block < num_state_blocks:
                            start = pl.read(query_start_loc, [request])
                            end = pl.read(query_start_loc, [request + 1])
                            position = pl.read(position_ids, [token])
                            if token >= start and token < end and position >= 0 and position % 2 == 1:
                                if token > start:
                                    partner_kv = kv_proj[token - 1 : token, :]
                                    partner_score = score_proj[token - 1 : token, :]
                                else:
                                    slot = pl.cast(block * STATE_CAPACITY + (position - 1) % STATE_CAPACITY, pl.INDEX)
                                    partner_kv = flat[slot : slot + 1, :HEAD_DIM]
                                    partner_score = flat[slot : slot + 1, HEAD_DIM:]
                                previous_kv[lane : lane + 1, :] = partner_kv
                                previous_score[lane : lane + 1, :] = partner_score
            highest = pl.maximum(previous_score, current_score)
            previous_weight = pl.exp(pl.sub(previous_score, highest))
            current_weight = pl.exp(pl.sub(current_score, highest))
            previous_weighted = pl.mul(previous_kv, previous_weight)
            current_weighted = pl.mul(current_kv, current_weight)
            weighted = pl.add(previous_weighted, current_weighted)
            pooled = pl.div(weighted, pl.add(previous_weight, current_weight))
            # Round the pooled row to BF16 before normalizing.
            value = pl.cast(pl.cast(pooled, pl.BF16, mode="rint"), pl.FP32)
            square = pl.mul(pl.row_sum(pl.mul(value, value)), 1.0 / HEAD_DIM)
            inv = pl.rsqrt(pl.add(square, EPS), high_precision=True)
            gamma = pl.reshape(pl.cast(norm_weight[:], pl.FP32), [1, HEAD_DIM])
            normalized = pl.col_expand_mul(pl.row_expand_mul(value, inv), gamma)
            parity = pl.full([8, 8], dtype=pl.FP32, value=0.0)
            for lane in pl.unroll(8):
                token = t + lane
                if token < num_tokens:
                    request = pl.cast(pl.read(token_to_req_indices, [token]), pl.INDEX)
                    if request >= 0 and request < pl.tensor.dim(state_block_table, 0):
                        block = pl.cast(pl.read(state_block_table, [request, 0]), pl.INDEX)
                        if block >= 0 and block < num_state_blocks:
                            start = pl.read(query_start_loc, [request])
                            end = pl.read(query_start_loc, [request + 1])
                            position = pl.read(position_ids, [token])
                            if token >= start and token < end and position >= 0 and position % 2 == 1:
                                pl.write(parity, [lane, 0], 1.0)
            parity = pl.row_sum(parity)
            published = pl.row_expand_mul(normalized, parity)
            published_bf16 = pl.cast(published, pl.BF16, mode="rint")
            latent[t : t + 8, :] = pl.set_validshape(published_bf16, rows, HEAD_DIM)
    return pool_tid


@pl.jit.inline
def c2a_compressor_state_write(
    kv_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    score_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    pair_ready: pl.Scalar[pl.TASK_ID],
):
    """Save only the final capacity rows of each chunk after all historical reads."""
    blocks = pl.tensor.dim(state_cache, 0)
    state_rows = blocks * STATE_CAPACITY
    flat = pl.reshape(state_cache, [state_rows, STATE_WIDTH])
    workers = pl.max(pl.min(num_tokens, 16), 1)
    with pl.spmd(workers, name_hint="state_cmp", deps=[pair_ready]) as state_tid:
        worker = pl.tile.get_block_idx()
        for t in pl.range(worker, num_tokens, workers):
            request = pl.cast(pl.read(token_to_req_indices, [t]), pl.INDEX)
            if request >= 0 and request < pl.tensor.dim(state_block_table, 0):
                block = pl.cast(pl.read(state_block_table, [request, 0]), pl.INDEX)
                if block >= 0 and block < blocks:
                    start = pl.read(query_start_loc, [request])
                    end = pl.min(pl.read(query_start_loc, [request + 1]), num_tokens)
                    position = pl.read(position_ids, [t])
                    if t >= start and t < end and t >= end - STATE_CAPACITY and position >= 0:
                        slot = pl.cast(block * STATE_CAPACITY + position % STATE_CAPACITY, pl.INDEX)
                        flat[slot : slot + 1, :HEAD_DIM] = kv_proj[t : t + 1, :]
                        flat[slot : slot + 1, HEAD_DIM:] = score_proj[t : t + 1, :]
    return state_tid


@pl.jit.inline
def c2a_compressor_pool_state(
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
):
    """Pool completed pairs and persist request state using a shared prefill/decode ABI."""
    blocks = pl.cast(pl.tensor.dim(state_cache, 0), pl.INT32)
    pool_tid = c2a_compressor_pool(
        kv_proj, score_proj, state_cache, position_ids, norm_weight, latent,
        num_tokens, query_start_loc, token_to_req_indices,
        state_block_table, blocks, state_ready,
    )
    state_tid = c2a_compressor_state_write(
        kv_proj, score_proj, query_start_loc, position_ids, token_to_req_indices,
        state_block_table, state_cache, num_tokens, pool_tid,
    )
    return pool_tid, state_tid


@pl.jit.inline(auto_scope=False)
def c2a_compressor_publish(
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    cos: pl.Tensor,
    sin: pl.Tensor,
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, (HEAD_DIM // 2)], pl.UINT8],
    scales: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, (HEAD_DIM // 16)], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    source_ready: pl.Scalar[pl.TASK_ID],
    max_workers: pl.constexpr,
):
    """Rotate and publish compressed rows with one worker per physical page."""
    blocks = pl.tensor.dim(cache, 0)
    cache_rows = blocks * 128
    flat = pl.reshape(cache, [cache_rows, (HEAD_DIM // 2)])
    scale_flat = pl.reshape(scales, [cache_rows, (HEAD_DIM // 16)])
    workers = pl.min(num_tokens, max_workers)
    with pl.spmd(workers, name_hint="rope_pub_cmp", deps=[cache_ready, source_ready]) as publish_tid:
        worker = pl.tile.get_block_idx()
        for t in pl.range(num_tokens):
            slot_i64 = pl.read(slots, [t])
            if slot_i64 >= 0 and (slot_i64 // 128) % workers == worker:
                slot = pl.cast(slot_i64, pl.INDEX)
                row = pl.tile.load(latent, [t, 0], [1, HEAD_DIM])
                tail_bf16 = pl.tile.slice(row, [1, ROPE_DIM], [0, HEAD_DIM - ROPE_DIM])
                tail = pl.cast(tail_bf16, pl.FP32)
                even = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P0101)
                odd = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P1010)
                c = pl.tile.load(cos, [t, 0], [1, ROPE_DIM // 2])
                sine = pl.tile.load(sin, [t, 0], [1, ROPE_DIM // 2])
                s = pl.mul(sine, 1.0)
                even_cos = pl.mul(even, c)
                odd_sin = pl.mul(odd, s)
                re = pl.sub(even_cos, odd_sin)
                even_sin = pl.mul(even, s)
                odd_cos = pl.mul(odd, c)
                im = pl.add(even_sin, odd_cos)
                real_lanes = pl.tile.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
                real_lanes = pl.tile.scatter_mask(real_lanes, re, mask_pattern=pl.tile.MaskPattern.P0101)
                imag_lanes = pl.tile.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
                imag_lanes = pl.tile.scatter_mask(imag_lanes, im, mask_pattern=pl.tile.MaskPattern.P1010)
                real_bits = pl.reinterpret_view(real_lanes, pl.UINT32)
                imag_bits = pl.reinterpret_view(imag_lanes, pl.UINT32)
                rotated_bits = pl.or_(real_bits, imag_bits)
                rotated = pl.reinterpret_view(rotated_bits, pl.FP32)
                rounded = pl.cast(rotated, pl.BF16, mode="rint")
                row = pl.tile.assemble(row, rounded, [0, HEAD_DIM - ROPE_DIM])
                source = pl.tile.full([1, HEAD_DIM * 2], dtype=pl.BF16, value=0.0)
                source = pl.tile.assemble(source, row, [0, 0])
                groups = pl.reshape(pl.cast(source, pl.FP32), [HEAD_DIM * 2 // 16, 16])
                scratch = pl.tile.create([HEAD_DIM * 2 // 16, 16], dtype=pl.FP32)
                amax = pl.tile.row_max(pl.abs(groups), tmp_tile=scratch)
                stored = pl.cast(pl.maximum(pl.mul(amax, (1.0 / 6.0)), (2.0**-9)), pl.BF16, mode="rint")
                scale = pl.cast(stored, pl.FP32)
                scaled = pl.row_expand_div(groups, scale)
                clamped = pl.maximum(scaled, -6.0)
                normalized = pl.minimum(clamped, 6.0)
                magnitude = pl.abs(normalized)
                # Nearest E2M1 magnitude, with exact midpoints rounded toward zero.
                delta0 = pl.reinterpret_view(pl.sub(magnitude, 0.25), pl.INT32)
                step0 = pl.shrs(pl.sub(delta0, 1), 31)
                delta1 = pl.reinterpret_view(pl.sub(magnitude, 0.75), pl.INT32)
                step1 = pl.shrs(pl.sub(delta1, 1), 31)
                delta2 = pl.reinterpret_view(pl.sub(magnitude, 1.25), pl.INT32)
                step2 = pl.shrs(pl.sub(delta2, 1), 31)
                delta3 = pl.reinterpret_view(pl.sub(magnitude, 1.75), pl.INT32)
                step3 = pl.shrs(pl.sub(delta3, 1), 31)
                delta4 = pl.reinterpret_view(pl.sub(magnitude, 2.5), pl.INT32)
                step4 = pl.shrs(pl.sub(delta4, 1), 31)
                delta5 = pl.reinterpret_view(pl.sub(magnitude, 3.5), pl.INT32)
                step5 = pl.shrs(pl.sub(delta5, 1), 31)
                delta6 = pl.reinterpret_view(pl.sub(magnitude, 5.0), pl.INT32)
                step6 = pl.shrs(pl.sub(delta6, 1), 31)
                lower = pl.add(pl.add(step0, step1), pl.add(step2, step3))
                upper = pl.add(pl.add(step4, step5), pl.add(step6, 7))
                index = pl.add(lower, upper)
                sign = pl.shrs(pl.reinterpret_view(normalized, pl.INT32), 31)
                codes = pl.reshape(pl.sub(index, pl.mul(sign, 8)), [1, HEAD_DIM * 2])
                low = pl.tile.gather_mask(codes, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.INT32)
                high = pl.tile.gather_mask(codes, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                packed = pl.add(low, pl.mul(high, 16))
                signed = pl.sub(packed, pl.mul(pl.shrs(packed, 7), 256))
                signed_bytes = pl.cast(signed, pl.INT8)
                payload_bytes = pl.reinterpret_view(signed_bytes, pl.UINT8)
                pl.tile.store(pl.set_validshape(payload_bytes, 1, HEAD_DIM // 2), [slot, 0], flat)
                scale_row = pl.reshape(stored, [1, HEAD_DIM * 2 // 16])
                pl.tile.store(pl.set_validshape(scale_row, 1, HEAD_DIM // 16), [slot, 0], scale_flat)
    return publish_tid


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
