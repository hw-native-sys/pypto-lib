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

from models.deepseek_v4_1_flash.config import (
    B_DYN,
    FLASH,
    HEAD_DIM,
    Q_START_DYN,
    STATE_BLOCKS_DYN,
    STATE_CAPACITY,
    STATE_WIDTH,
    T_DYN,
)

EPS = FLASH.rms_norm_eps


@pl.jit.inline
def compressor_pair(
    kv_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    score_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    prev_kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    prev_score: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    state_ready: pl.Scalar[pl.TASK_ID],
):
    """Stage odd-position partners from the current chunk or its historical ring row."""
    blocks = pl.tensor.dim(state_cache, 0)
    state_rows = blocks * STATE_CAPACITY
    flat = pl.reshape(state_cache, [state_rows, STATE_WIDTH])
    with pl.spmd(pl.max(num_tokens, 1), name_hint="c2a_compressor_pair", deps=[state_ready]) as pair_tid:
        t = pl.tile.get_block_idx()
        if t < num_tokens:
            kv = pl.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0)
            score = pl.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0)
            request = pl.cast(pl.read(token_to_req_indices, [t]), pl.INDEX)
            if request >= 0 and request < pl.tensor.dim(state_block_table, 0):
                block = pl.cast(pl.read(state_block_table, [request, 0]), pl.INDEX)
                if block >= 0 and block < blocks:
                    start = pl.read(query_start_loc, [request])
                    end = pl.read(query_start_loc, [request + 1])
                    position = pl.read(position_ids, [t])
                    if t >= start and t < end and position >= 0 and position % 2 == 1:
                        if t > start:
                            kv = kv_proj[t - 1 : t, :]
                            score = score_proj[t - 1 : t, :]
                        else:
                            slot = pl.cast(block * STATE_CAPACITY + (position - 1) % STATE_CAPACITY, pl.INDEX)
                            kv = flat[slot : slot + 1, :HEAD_DIM]
                            score = flat[slot : slot + 1, HEAD_DIM:]
            prev_kv[t : t + 1, :] = kv
            prev_score[t : t + 1, :] = score
    return pair_tid


@pl.jit.inline
def compressor_pool(
    kv_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    score_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    prev_kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    prev_score: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
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
    """Pool eight rows with per-channel softmax, BF16 rounding and RMSNorm."""
    with pl.spmd(pl.max((num_tokens + 7) // 8, 1), name_hint="c2a_compressor_pool", deps=[pair_ready]) as pool_tid:
        group = pl.tile.get_block_idx()
        t = group * 8
        if t < num_tokens:
            rows = pl.min(8, num_tokens - t)
            current_kv = pl.slice(kv_proj, [8, HEAD_DIM], [t, 0], valid_shape=[rows, HEAD_DIM])
            current_kv = pl.set_validshape(pl.fillpad(current_kv, pad_value=pl.PadValue.zero), 8, HEAD_DIM)
            current_score = pl.slice(score_proj, [8, HEAD_DIM], [t, 0], valid_shape=[rows, HEAD_DIM])
            current_score = pl.set_validshape(pl.fillpad(current_score, pad_value=pl.PadValue.zero), 8, HEAD_DIM)
            previous_kv = pl.slice(prev_kv, [8, HEAD_DIM], [t, 0], valid_shape=[rows, HEAD_DIM])
            previous_kv = pl.set_validshape(pl.fillpad(previous_kv, pad_value=pl.PadValue.zero), 8, HEAD_DIM)
            previous_score = pl.slice(prev_score, [8, HEAD_DIM], [t, 0], valid_shape=[rows, HEAD_DIM])
            previous_score = pl.set_validshape(pl.fillpad(previous_score, pad_value=pl.PadValue.zero), 8, HEAD_DIM)
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
    previous_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.FP32)
    previous_score = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.FP32)
    pair_tid = compressor_pair(
        kv_proj, score_proj, query_start_loc, position_ids, token_to_req_indices,
        state_block_table, state_cache, previous_kv, previous_score, num_tokens, state_ready,
    )
    pool_tid = compressor_pool(
        kv_proj, score_proj, previous_kv, previous_score, position_ids, norm_weight, latent,
        num_tokens, query_start_loc, token_to_req_indices, state_block_table, blocks, pair_tid,
    )
    state_tid = compressor_state_write(
        kv_proj, score_proj, query_start_loc, position_ids, token_to_req_indices,
        state_block_table, state_cache, num_tokens, pair_tid,
    )
    return pool_tid, state_tid
