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
M_TILE = 16
N_TILE = 128
COMPRESSOR_M_TILE = 80
COMPRESSOR_K_TILE = 64
COMPRESSOR_QUERY_TILE = 128
# Limit Cube rounding before the compensated Vector accumulation.
COMPRESSOR_SEGMENTS = D // COMPRESSOR_K_TILE
C1A_PROJ_M_TILE = 48
C1A_PROJ_N_TILE = 64
C1A_PROJ_K_TILE = 512


@pl.jit.inline
def _compressed_rope_block(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    block: pl.Scalar[pl.INT32],
):
    t = block
    base = 0
    output[t : t + 1, base : base + (HEAD_DIM - ROPE_DIM)] = x[t : t + 1, base : base + (HEAD_DIM - ROPE_DIM)]
    tail = pl.cast(x[t : t + 1, base + (HEAD_DIM - ROPE_DIM) : base + HEAD_DIM], pl.FP32)
    even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
    odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
    c = cos[t : t + 1, :]
    s = pl.mul(sin[t : t + 1, :], 1.0)
    re = pl.sub(pl.mul(even, c), pl.mul(odd, s))
    im = pl.add(pl.mul(even, s), pl.mul(odd, c))
    rotated = pl.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
    rotated = pl.tensor.scatter(re, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
    rotated = pl.tensor.scatter(im, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
    output[t : t + 1, base + (HEAD_DIM - ROPE_DIM) : base + HEAD_DIM] = pl.cast(
        rotated, pl.BF16, mode="rint"
    )
    return output


@pl.jit.inline
def _sum_ratio2_segments(
    high_product: pl.Tensor,
    residual_product: pl.Tensor,
    output: pl.Tensor,
    chunk_rows: pl.Scalar[pl.INDEX],
    begin: pl.Scalar[pl.INDEX],
    sum_t0: pl.Scalar[pl.INDEX],
    sum_n0: pl.Scalar[pl.INDEX],
    sum_rows: pl.Scalar[pl.INDEX],
):
    high_sum = pl.full([M_TILE, N_TILE], dtype=pl.FP32, value=0.0)
    low_sum = pl.full([M_TILE, N_TILE], dtype=pl.FP32, value=0.0)
    high_error = pl.full([M_TILE, N_TILE], dtype=pl.FP32, value=0.0)
    low_error = pl.full([M_TILE, N_TILE], dtype=pl.FP32, value=0.0)
    for sum_segment in pl.range(COMPRESSOR_SEGMENTS):
        high_part = pl.slice(high_product, [M_TILE, N_TILE], [sum_segment * chunk_rows + sum_t0, sum_n0], valid_shape=[sum_rows, N_TILE])
        low_part = pl.slice(residual_product, [M_TILE, N_TILE], [sum_segment * chunk_rows + sum_t0, sum_n0], valid_shape=[sum_rows, N_TILE])
        high_delta = pl.sub(high_part, high_error)
        high_next = pl.add(high_sum, high_delta)
        high_error = pl.sub(pl.sub(high_next, high_sum), high_delta)
        high_sum = high_next
        low_delta = pl.sub(low_part, low_error)
        low_next = pl.add(low_sum, low_delta)
        low_error = pl.sub(pl.sub(low_next, low_sum), low_delta)
        low_sum = low_next
    output[begin + sum_t0 : begin + sum_t0 + M_TILE, sum_n0 : sum_n0 + N_TILE] = pl.set_validshape(pl.add(high_sum, low_sum), sum_rows, N_TILE)
    return output


@pl.jit.inline
def project_ratio2_weights(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    output: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Reuse decomposed weight panels with bounded compensated row staging."""
    completion = pl.array.create(1, pl.TASK_ID)
    completion[0] = pl.system.task_dummy(deps=[])
    for begin in pl.range(0, num_tokens, COMPRESSOR_QUERY_TILE):
        chunk_rows = pl.min(pl.cast(num_tokens, pl.INDEX) - begin, COMPRESSOR_QUERY_TILE)
        high_product = pl.create_tensor([COMPRESSOR_SEGMENTS * chunk_rows, HEAD_DIM], dtype=pl.FP32)
        residual_product = pl.create_tensor([COMPRESSOR_SEGMENTS * chunk_rows, HEAD_DIM], dtype=pl.FP32)
        panel_blocks = (HEAD_DIM // N_TILE) * COMPRESSOR_SEGMENTS
        workers = pl.min(pl.min(panel_blocks, 16), pl.system.available_cluster_count())
        with pl.spmd(workers, name_hint="c2a_compressor_reuse_rows", deps=[completion[0]], sync_start=True) as cube_tid:
            cube_worker = pl.tile.get_block_idx()
            for panel_block in pl.range(cube_worker, panel_blocks, workers):
                cube_segment = panel_block % COMPRESSOR_SEGMENTS
                cube_n0 = panel_block // COMPRESSOR_SEGMENTS * N_TILE
                cube_k0 = cube_segment * COMPRESSOR_K_TILE
                value = pl.load(weight, [cube_k0, cube_n0], [COMPRESSOR_K_TILE, N_TILE], target_memory=pl.MemorySpace.Vec)
                high = pl.cast(value, pl.BF16, mode="rint")
                high_fp32 = pl.cast(high, pl.FP32)
                residual = pl.sub(value, high_fp32)
                low = pl.cast(residual, pl.BF16, mode="rint")
                low_fp32 = pl.cast(low, pl.FP32)
                tail_fp32 = pl.sub(residual, low_fp32)
                tail = pl.cast(tail_fp32, pl.BF16, mode="rint")
                high_mat = pl.tile.move(
                    high, target_memory=pl.MemorySpace.Mat,
                    blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major,
                )
                high_right = pl.tile.move(high_mat, target_memory=pl.MemorySpace.Right)
                low_mat = pl.tile.move(
                    low, target_memory=pl.MemorySpace.Mat,
                    blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major,
                )
                low_right = pl.tile.move(low_mat, target_memory=pl.MemorySpace.Right)
                tail_mat = pl.tile.move(
                    tail, target_memory=pl.MemorySpace.Mat,
                    blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major,
                )
                tail_right = pl.tile.move(tail_mat, target_memory=pl.MemorySpace.Right)
                for cube_t0 in pl.range(0, chunk_rows, COMPRESSOR_M_TILE):
                    cube_rows = pl.min(COMPRESSOR_M_TILE, chunk_rows - cube_t0)
                    a = pl.load(x, [begin + cube_t0, cube_k0], [COMPRESSOR_M_TILE, COMPRESSOR_K_TILE], valid_shape=[cube_rows, COMPRESSOR_K_TILE], target_memory=pl.MemorySpace.Vec)
                    a = pl.fillpad(a, pad_value=pl.PadValue.zero)
                    a = pl.set_validshape(a, COMPRESSOR_M_TILE, COMPRESSOR_K_TILE)
                    a_fp32 = pl.cast(a, pl.FP32)
                    a = pl.cast(a_fp32, pl.BF16, mode="rint")
                    high_acc = pl.matmul(a, high_right)
                    residual_acc = pl.matmul(a, low_right)
                    residual_acc = pl.matmul_acc(residual_acc, a, tail_right)
                    high_value = pl.mul(high_acc, 1.0)
                    high_value = pl.set_validshape(high_value, cube_rows, N_TILE)
                    high_product = pl.store(high_value, [cube_segment * chunk_rows + cube_t0, cube_n0], high_product)
                    residual_value = pl.mul(residual_acc, 1.0)
                    residual_value = pl.set_validshape(residual_value, cube_rows, N_TILE)
                    residual_product = pl.store(residual_value, [cube_segment * chunk_rows + cube_t0, cube_n0], residual_product)
        output_blocks = (chunk_rows + M_TILE - 1) // M_TILE * (HEAD_DIM // N_TILE)
        with pl.spmd(pl.min(output_blocks, 16), name_hint="c2a_compressor_reuse_sum", deps=[cube_tid]) as sum_tid:
            sum_worker = pl.tile.get_block_idx()
            for sum_block in pl.range(sum_worker, output_blocks, pl.min(output_blocks, 16)):
                sum_t0 = sum_block // (HEAD_DIM // N_TILE) * M_TILE
                sum_n0 = sum_block % (HEAD_DIM // N_TILE) * N_TILE
                sum_rows = pl.min(M_TILE, chunk_rows - sum_t0)
                _sum_ratio2_segments(
                    high_product, residual_product, output,
                    chunk_rows, begin, sum_t0, sum_n0, sum_rows,
                )
        completion[0] = sum_tid
    return output


@pl.jit.inline
def project_ratio2(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    wgate: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    kv_out: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    score_out: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    project_ratio2_weights(x, wkv, kv_out, num_tokens)
    project_ratio2_weights(x, wgate, score_out, num_tokens)
    return kv_out, score_out


@pl.jit.inline
def publish_compressed_cache(
    value: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    scales: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP],
        pl.BF16,
    ],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Publish ratio-1 KV rows as group-16 MXFP4 with BF16 scales."""
    cache_rows = pl.tensor.dim(cache, 0) * 128
    cache_flat = pl.reshape(cache, [cache_rows, HEAD_DIM // 2])
    scale_flat = pl.reshape(scales, [cache_rows, HEAD_DIM // COMPRESSED_CACHE_GROUP])
    with pl.spmd(num_tokens, name_hint="pub_cmp") as publish_tid:
        token = pl.tile.get_block_idx()
        slot_i64 = pl.read(slots, [token])
        if slot_i64 >= 0:
            slot = pl.cast(slot_i64, pl.INDEX)
            source = pl.cast(pl.load(value, [token, 0], [1, HEAD_DIM]), pl.FP32)
            grouped = pl.reshape(source, [HEAD_DIM // COMPRESSED_CACHE_GROUP, COMPRESSED_CACHE_GROUP])
            maximum_tmp = pl.create_tile([HEAD_DIM // COMPRESSED_CACHE_GROUP, 128], dtype=pl.FP32)
            maximum = pl.row_max(pl.abs(grouped), tmp_tile=maximum_tmp)
            raw_scale = pl.maximum(pl.mul(maximum, 1.0 / 6.0), 2.0**-9)
            stored_scale = pl.cast(raw_scale, pl.BF16, mode="rint")
            scale = pl.cast(stored_scale, pl.FP32)
            normalized = pl.row_expand_div(grouped, scale)
            normalized = pl.reshape(normalized, [1, HEAD_DIM])
            magnitude = pl.minimum(pl.abs(normalized), 6.0)
            delta0 = pl.reinterpret_view(pl.sub(magnitude, 0.25), pl.INT32)
            delta1 = pl.reinterpret_view(pl.sub(magnitude, 0.75), pl.INT32)
            delta2 = pl.reinterpret_view(pl.sub(magnitude, 1.25), pl.INT32)
            delta3 = pl.reinterpret_view(pl.sub(magnitude, 1.75), pl.INT32)
            delta4 = pl.reinterpret_view(pl.sub(magnitude, 2.5), pl.INT32)
            delta5 = pl.reinterpret_view(pl.sub(magnitude, 3.5), pl.INT32)
            delta6 = pl.reinterpret_view(pl.sub(magnitude, 5.0), pl.INT32)
            step0 = pl.shrs(pl.sub(delta0, 1), 31)
            step1 = pl.shrs(delta1, 31)
            step2 = pl.shrs(pl.sub(delta2, 1), 31)
            step3 = pl.shrs(delta3, 31)
            step4 = pl.shrs(pl.sub(delta4, 1), 31)
            step5 = pl.shrs(delta5, 31)
            step6 = pl.shrs(pl.sub(delta6, 1), 31)
            lower = pl.add(pl.add(step0, step1), pl.add(step2, step3))
            upper = pl.add(pl.add(step4, step5), pl.add(step6, 7))
            unsigned = pl.add(lower, upper)
            sign = pl.ands(pl.shrs(pl.reinterpret_view(normalized, pl.INT32), 28), 8)
            payload_codes = pl.or_(unsigned, sign)
            pair_ids = pl.tile.arange(0, [1, HEAD_DIM // 2], dtype=pl.INT32)
            low_indices = pl.mul(pair_ids, 2)
            high_indices = pl.add(low_indices, 1)
            low_tmp = pl.create_tile([1, HEAD_DIM // 2], dtype=pl.INT32)
            high_tmp = pl.create_tile([1, HEAD_DIM // 2], dtype=pl.INT32)
            low = pl.tile.gather(payload_codes, low_indices, low_tmp)
            high = pl.tile.gather(payload_codes, high_indices, high_tmp)
            payload_bytes = pl.reshape(
                pl.cast(pl.add(low, pl.shls(high, 4)), pl.UINT8),
                [1, HEAD_DIM // 2],
            )
            pl.store(payload_bytes, [slot, 0], cache_flat)
            pl.store(
                pl.reshape(stored_scale, [1, HEAD_DIM // COMPRESSED_CACHE_GROUP]),
                [slot, 0],
                scale_flat,
            )
    return publish_tid


def golden_compressor_ratio1(
    x: torch.Tensor,
    wkv: torch.Tensor,
    norm_weight: torch.Tensor,
) -> torch.Tensor:
    """Reference the ratio-1 NZ-weight projection followed by RMSNorm."""
    projected = torch.matmul(x.float(), unpack_bf16_nz(wkv).float()).to(x.dtype)
    return golden_rms_norm(projected, norm_weight)


def golden_compressor_ratio2_paged(
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
    """Evaluate ratio-2 compression with request-owned FP32 ring state.

    Query boundaries count valid packed tokens; any tensor tail is padding.
    Negative/out-of-range request or block indices suppress state access and
    publication. Valid chunks contain consecutive absolute positions.
    """
    kv = torch.matmul(x.float(), wkv.float())
    score = torch.matmul(x.float(), wgate.float())
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


@pl.jit.inline
def compressor_pool(
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
            weighted = pl.add(pl.mul(previous_kv, previous_weight), pl.mul(current_kv, current_weight))
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
            latent[t : t + 8, :] = pl.set_validshape(pl.cast(published, pl.BF16, mode="rint"), rows, HEAD_DIM)
    return pool_tid


@pl.jit.inline
def compressor_state_write(
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
def compressor_ratio2(
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
    tokens = pl.tensor.dim(kv_proj, 0)
    blocks = pl.cast(pl.tensor.dim(state_cache, 0), pl.INT32)
    pool_tid = compressor_pool(
        kv_proj, score_proj, state_cache, position_ids, norm_weight, latent,
        num_tokens, query_start_loc, token_to_req_indices, state_block_table, blocks, state_ready,
    )
    state_tid = compressor_state_write(
        kv_proj, score_proj, query_start_loc, position_ids, token_to_req_indices,
        state_block_table, state_cache, num_tokens, pool_tid,
    )
    return pool_tid, state_tid


@pl.jit.inline(auto_scope=False)
def prefill_c1a_compressor_rope(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    blocks = num_tokens
    workers = pl.min(blocks, 2147483647)
    for worker in pl.spmd(workers, name_hint='attention_rope'):
        for block in pl.range(worker, blocks, workers):
            _compressed_rope_block(x, cos, sin, output, block)
    return output


@pl.jit.inline
def publish_c2a_compressed(
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, (HEAD_DIM // 2)], pl.UINT8],
    scales: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, (HEAD_DIM // 16)], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    latent_ready: pl.Scalar[pl.TASK_ID],
):
    """Publish rotated compressed KV rows as packed E2M1 with group-16 BF16 scales."""
    blocks = pl.tensor.dim(cache, 0)
    cache_rows = blocks * 128
    flat = pl.reshape(cache, [cache_rows, (HEAD_DIM // 2)])
    scale_flat = pl.reshape(scales, [cache_rows, (HEAD_DIM // 16)])
    with pl.spmd(num_tokens, name_hint="pub_cmp", deps=[cache_ready, latent_ready]) as publish_tid:
        t = pl.tile.get_block_idx()
        slot_i64 = pl.read(slots, [t])
        if slot_i64 >= 0:
            slot = pl.cast(slot_i64, pl.INDEX)
            source = pl.slice(latent, [1, HEAD_DIM * 2], [t, 0], valid_shape=[1, HEAD_DIM])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 1, HEAD_DIM * 2)
            groups = pl.reshape(pl.cast(source, pl.FP32), [HEAD_DIM * 2 // 16, 16])
            amax = pl.row_max(pl.abs(groups))
            stored = pl.cast(pl.maximum(pl.mul(amax, (1.0 / 6.0)), (2.0**-9)), pl.BF16, mode="rint")
            normalized = pl.minimum(
                pl.maximum(pl.row_expand_div(groups, pl.cast(stored, pl.FP32)), -6.0), 6.0
            )
            magnitude = pl.abs(normalized)
            # Nearest E2M1 magnitude index: seven minus the number of table midpoints
            # at or above |v|, which resolves an exact midpoint toward zero.
            step0 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 0.25), pl.INT32), 1), 31)
            step1 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 0.75), pl.INT32), 1), 31)
            step2 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 1.25), pl.INT32), 1), 31)
            step3 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 1.75), pl.INT32), 1), 31)
            step4 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 2.5), pl.INT32), 1), 31)
            step5 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 3.5), pl.INT32), 1), 31)
            step6 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 5.0), pl.INT32), 1), 31)
            index = pl.add(
                pl.add(pl.add(step0, step1), pl.add(step2, step3)),
                pl.add(pl.add(step4, step5), pl.add(step6, 7)),
            )
            sign = pl.shrs(pl.reinterpret_view(normalized, pl.INT32), 31)
            codes = pl.reshape(pl.sub(index, pl.mul(sign, 8)), [1, HEAD_DIM * 2])
            low = pl.gather(codes, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.INT32)
            high = pl.gather(codes, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            packed = pl.add(low, pl.mul(high, 16))
            signed = pl.sub(packed, pl.mul(pl.shrs(packed, 7), 256))
            flat[slot : slot + 1, :] = pl.set_validshape(
                pl.reinterpret_view(pl.cast(signed, pl.INT8), pl.UINT8), 1, (HEAD_DIM // 2)
            )
            scale_flat[slot : slot + 1, :] = pl.set_validshape(
                pl.reshape(stored, [1, HEAD_DIM * 2 // 16]), 1, (HEAD_DIM // 16)
            )
    return publish_tid


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
def decode_c1a_compressor_rope(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    blocks = num_tokens
    workers = pl.min(blocks, 2147483647)
    with pl.spmd(workers, name_hint='rope_cmp', deps=[ready]) as rope_tid:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, blocks, workers):
            _compressed_rope_block(x, cos, sin, output, block)
    return rope_tid


@pl.jit.inline(auto_scope=False)
def decode_c2a_compressor_rope(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    blocks = num_tokens
    workers = pl.min(blocks, 2147483647)
    with pl.spmd(workers, name_hint='rope_cmp', deps=[ready]) as rope_tid:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, blocks, workers):
            _compressed_rope_block(x, cos, sin, output, block)
    return rope_tid


@pl.jit.inline(auto_scope=False)
def prefill_c2a_compressed_rope(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    blocks = num_tokens
    workers = pl.min(blocks, 2147483647)
    for worker in pl.spmd(workers, name_hint='attention_rope'):
        for block in pl.range(worker, blocks, workers):
            _compressed_rope_block(x, cos, sin, output, block)
    return output
