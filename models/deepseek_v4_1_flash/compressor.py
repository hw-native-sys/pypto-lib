# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Ratio-2 compression with request-owned FP32 ring state."""

import pypto.language as pl
import torch

from config import (
    FLASH as M,
    STATE_CAPACITY,
    B_DYN,
    Q_START_DYN,
    STATE_BLOCKS_DYN,
    T_DYN,
    COMPRESSED_CACHE_GROUP,
    CMP_BLOCKS_DYN,
)


STATE_WIDTH = 2 * M.head_dim


from rmsnorm import golden_rms_norm


# model config
HEAD_DIM = M.head_dim


EPS = M.rms_norm_eps


D = M.hidden_size
M_TILE = 16
N_TILE = 128
COMPRESSOR_K_TILE = 64
COMPRESSOR_QUERY_TILE = 128
COMPRESSOR_SEGMENTS = 16
COMPRESSOR_K_CHUNK = D // COMPRESSOR_SEGMENTS


@pl.jit.inline
def project_ratio2_weights(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    output: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Retain FP32 weight residuals through high and residual accumulators.

    Three BF16 weight terms preserve FP32 weights. Bound each Cube reduction
    to one K chunk, then compensate the Vector sum of the staged GM products
    to reduce accumulation error near pooled BF16 rounding boundaries.
    Mixed producers use Vector startup and FP32 Vector stores.
    """
    high_weight = pl.create_tensor([D, HEAD_DIM], dtype=pl.BF16)
    low_weight = pl.create_tensor([D, HEAD_DIM], dtype=pl.BF16)
    tail_weight = pl.create_tensor([D, HEAD_DIM], dtype=pl.BF16)
    split_blocks = D // COMPRESSOR_K_TILE * (HEAD_DIM // N_TILE)
    split_workers = pl.min(split_blocks, 64)
    with pl.spmd(split_workers, name_hint="c2a_compressor_split") as split_tid:
        split_worker = pl.tile.get_block_idx()
        for split_block in pl.range(split_worker, split_blocks, split_workers):
            split_k0 = split_block // (HEAD_DIM // N_TILE) * COMPRESSOR_K_TILE
            split_n0 = split_block % (HEAD_DIM // N_TILE) * N_TILE
            value = weight[split_k0 : split_k0 + COMPRESSOR_K_TILE, split_n0 : split_n0 + N_TILE]
            high = pl.cast(value, pl.BF16, mode="rint")
            residual = pl.sub(value, pl.cast(high, pl.FP32))
            low = pl.cast(residual, pl.BF16, mode="rint")
            tail = pl.cast(pl.sub(residual, pl.cast(low, pl.FP32)), pl.BF16, mode="rint")
            high_weight[split_k0 : split_k0 + COMPRESSOR_K_TILE, split_n0 : split_n0 + N_TILE] = high
            low_weight[split_k0 : split_k0 + COMPRESSOR_K_TILE, split_n0 : split_n0 + N_TILE] = low
            tail_weight[split_k0 : split_k0 + COMPRESSOR_K_TILE, split_n0 : split_n0 + N_TILE] = tail
    completion = pl.array.create(1, pl.TASK_ID)
    completion[0] = split_tid
    if num_tokens > 0:
        chunk_rows = pl.min(pl.cast(num_tokens, pl.INDEX), COMPRESSOR_QUERY_TILE)
        high_product = pl.create_tensor([COMPRESSOR_SEGMENTS * chunk_rows, HEAD_DIM], dtype=pl.FP32)
        residual_product = pl.create_tensor([COMPRESSOR_SEGMENTS * chunk_rows, HEAD_DIM], dtype=pl.FP32)
        for begin in pl.range(0, num_tokens, chunk_rows):
            chunk_active = pl.min(chunk_rows, num_tokens - begin)
            output_blocks = (chunk_active + M_TILE - 1) // M_TILE * (HEAD_DIM // N_TILE)
            blocks = output_blocks * COMPRESSOR_SEGMENTS
            # The synchronous mixed cohort must fit the target A5 core count.
            workers = pl.min(blocks, 16)
            with pl.spmd(workers, name_hint="c2a_compressor_cube", deps=[completion[0]], sync_start=True) as cube_tid:
                cube_worker = pl.tile.get_block_idx()
                for cube_block in pl.range(cube_worker, blocks, workers):
                    cube_segment = cube_block % COMPRESSOR_SEGMENTS
                    cube_tile_block = cube_block // COMPRESSOR_SEGMENTS
                    cube_t0 = cube_tile_block // (HEAD_DIM // N_TILE) * M_TILE
                    cube_n0 = cube_tile_block % (HEAD_DIM // N_TILE) * N_TILE
                    cube_rows = pl.min(M_TILE, chunk_active - cube_t0)
                    high_acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.FP32)
                    residual_acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.FP32)
                    for cube_k0 in pl.range(cube_segment * COMPRESSOR_K_CHUNK, (cube_segment + 1) * COMPRESSOR_K_CHUNK, COMPRESSOR_K_TILE):
                        a = pl.slice(x, [M_TILE, COMPRESSOR_K_TILE], [begin + cube_t0, cube_k0], valid_shape=[cube_rows, COMPRESSOR_K_TILE])
                        a = pl.cast(pl.cast(a, pl.FP32), pl.BF16, mode="rint")
                        high = high_weight[cube_k0 : cube_k0 + COMPRESSOR_K_TILE, cube_n0 : cube_n0 + N_TILE]
                        low = low_weight[cube_k0 : cube_k0 + COMPRESSOR_K_TILE, cube_n0 : cube_n0 + N_TILE]
                        tail = tail_weight[cube_k0 : cube_k0 + COMPRESSOR_K_TILE, cube_n0 : cube_n0 + N_TILE]
                        high_acc = pl.matmul_acc(high_acc, a, high, init_cond=(cube_k0 == cube_segment * COMPRESSOR_K_CHUNK))
                        residual_acc = pl.matmul_acc(residual_acc, a, low, init_cond=(cube_k0 == cube_segment * COMPRESSOR_K_CHUNK))
                        residual_acc = pl.matmul_acc(residual_acc, a, tail)
                    high_product[cube_segment * chunk_rows + cube_t0 : cube_segment * chunk_rows + cube_t0 + M_TILE, cube_n0 : cube_n0 + N_TILE] = pl.set_validshape(pl.mul(high_acc, 1.0), cube_rows, N_TILE)
                    residual_product[cube_segment * chunk_rows + cube_t0 : cube_segment * chunk_rows + cube_t0 + M_TILE, cube_n0 : cube_n0 + N_TILE] = pl.set_validshape(pl.mul(residual_acc, 1.0), cube_rows, N_TILE)
            with pl.spmd(pl.min(output_blocks, 16), name_hint="c2a_compressor_sum", deps=[cube_tid]) as sum_tid:
                sum_worker = pl.tile.get_block_idx()
                for sum_block in pl.range(sum_worker, output_blocks, pl.min(output_blocks, 16)):
                    sum_t0 = sum_block // (HEAD_DIM // N_TILE) * M_TILE
                    sum_n0 = sum_block % (HEAD_DIM // N_TILE) * N_TILE
                    sum_rows = pl.min(M_TILE, chunk_active - sum_t0)
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
    with pl.spmd(num_tokens, name_hint="c1a_compressed_publish") as publish_tid:
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
    """Reference the ratio-1 compressor: projection followed by RMSNorm."""
    projected = torch.matmul(x.float(), wkv.float()).to(x.dtype)
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
    with pl.spmd(pl.max((num_tokens + 7) // 8, 1), name_hint="c2a_compressor_pool", deps=[pair_ready]) as pool_tid:
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
    with pl.spmd(pl.max(num_tokens, 1), name_hint="c2a_compressor_state", deps=[pair_ready]) as state_tid:
        t = pl.tile.get_block_idx()
        if t < num_tokens:
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
