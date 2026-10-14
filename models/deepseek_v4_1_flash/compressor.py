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


@pl.jit.inline
def publish_compressed_cache(
    value: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    scales: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP],
        pl.FP8E4M3FN,
    ],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Publish ratio-1 KV rows as group-16 MXFP4 with E4M3 scales."""
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
            raw_scale = pl.minimum(
                pl.maximum(pl.mul(maximum, 1.0 / 6.0), 2.0**-9),
                448.0,
            )
            stored_scale = pl.cast(raw_scale, pl.FP8E4M3FN, mode="rint")
            scale = pl.cast(stored_scale, pl.FP32)
            normalized = pl.row_expand_div(grouped, scale)
            normalized = pl.reshape(normalized, [1, HEAD_DIM])
            magnitude = pl.minimum(pl.abs(normalized), 6.0)
            lower = pl.cast(
                pl.add(pl.mul(pl.minimum(magnitude, 2.0), 2.0), 0.4999),
                pl.INT32,
                mode="trunc",
            )
            middle = pl.cast(
                pl.add(pl.minimum(pl.maximum(pl.sub(magnitude, 2.0), 0.0), 2.0), 0.4999),
                pl.INT32,
                mode="trunc",
            )
            upper = pl.cast(
                pl.add(pl.mul(pl.maximum(pl.sub(magnitude, 4.0), 0.0), 0.5), 0.4999),
                pl.INT32,
                mode="trunc",
            )
            payload_codes = pl.add(pl.add(lower, middle), upper)
            bits = pl.reinterpret_view(normalized, pl.INT32)
            sign = pl.ands(pl.shrs(bits, 31), 1)
            payload_codes = pl.add(payload_codes, pl.mul(sign, 8))
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
