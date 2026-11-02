# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pool ratio-2 Prefill pairs and normalize their BF16 pooled rows."""

import pypto.language as pl

from .compressor_core import EPS, HEAD_DIM, RING_ROWS_DYN, STATE_WIDTH, compressor_state_write
from .operator_config import B_DYN, Q_START_DYN, STATE_BLOCKS_DYN, T_DYN


# tiling
T_TILE = 8


@pl.jit.inline
def pool_prefill(
    kv_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    score_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, RING_ROWS_DYN, STATE_WIDTH], pl.FP32],
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
    ring_rows = pl.tensor.dim(state_cache, 1)
    state_rows = num_state_blocks * ring_rows
    flat = pl.reshape(state_cache, [state_rows, STATE_WIDTH])
    with pl.spmd(pl.max((num_tokens + T_TILE - 1) // T_TILE, 1), name_hint="c2_pool", deps=[pair_ready]) as pool_tid:
        group = pl.tile.get_block_idx()
        t = group * T_TILE
        if t < num_tokens:
            rows = pl.min(T_TILE, num_tokens - t)
            current_kv = pl.slice(kv_proj, [T_TILE, HEAD_DIM], [t, 0], valid_shape=[rows, HEAD_DIM])
            current_kv = pl.fillpad(current_kv, pad_value=pl.PadValue.zero)
            current_kv = pl.set_validshape(current_kv, T_TILE, HEAD_DIM)
            current_score = pl.slice(score_proj, [T_TILE, HEAD_DIM], [t, 0], valid_shape=[rows, HEAD_DIM])
            current_score = pl.fillpad(current_score, pad_value=pl.PadValue.zero)
            current_score = pl.set_validshape(current_score, T_TILE, HEAD_DIM)
            previous_kv = pl.full([T_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
            previous_score = pl.full([T_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
            parity = pl.full([T_TILE, 8], dtype=pl.FP32, value=0.0)
            for lane in pl.unroll(T_TILE):
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
                                    slot_offset = block * ring_rows + (position - 1) % ring_rows
                                    slot = pl.cast(slot_offset, pl.INDEX)
                                    partner_kv = flat[slot : slot + 1, :HEAD_DIM]
                                    partner_score = flat[slot : slot + 1, HEAD_DIM:]
                                previous_kv[lane : lane + 1, :] = partner_kv
                                previous_score[lane : lane + 1, :] = partner_score
                                pl.write(parity, [lane, 0], 1.0)
            highest = pl.maximum(previous_score, current_score)
            previous_exp = pl.exp(pl.sub(previous_score, highest))
            current_exp = pl.exp(pl.sub(current_score, highest))
            denominator = pl.add(previous_exp, current_exp)
            previous_probability = pl.div(previous_exp, denominator)
            current_probability = pl.div(current_exp, denominator)
            previous_weighted = pl.mul(previous_kv, previous_probability)
            current_weighted = pl.mul(current_kv, current_probability)
            pooled = pl.add(previous_weighted, current_weighted)
            pooled_bf16 = pl.cast(pooled, pl.BF16, mode="rint")
            value = pl.cast(pooled_bf16, pl.FP32)
            square = pl.mul(value, value)
            square = pl.mul(pl.row_sum(square), 1.0 / HEAD_DIM)
            inverse = pl.rsqrt(pl.add(square, EPS), high_precision=True)
            normalized = pl.row_expand_mul(value, inverse)
            gamma = pl.reshape(pl.cast(norm_weight[:], pl.FP32), [1, HEAD_DIM])
            weighted = pl.col_expand_mul(normalized, gamma)
            parity = pl.row_sum(parity)
            published = pl.row_expand_mul(weighted, parity)
            published = pl.cast(published, pl.BF16, mode="rint")
            latent[t : t + T_TILE, :] = pl.set_validshape(published, rows, HEAD_DIM)
    return pool_tid


@pl.jit.inline
def compressor_ratio2_prefill(
    kv_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    score_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, RING_ROWS_DYN, STATE_WIDTH], pl.FP32],
    norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    state_ready: pl.Scalar[pl.TASK_ID],
):
    blocks = pl.cast(pl.tensor.dim(state_cache, 0), pl.INT32)
    pool_tid = pool_prefill(
        kv_proj, score_proj, state_cache, position_ids, norm_weight, latent,
        num_tokens, query_start_loc, token_to_req_indices, state_block_table, blocks, state_ready,
    )
    state_tid = compressor_state_write(
        kv_proj, score_proj, query_start_loc, position_ids, token_to_req_indices,
        state_block_table, state_cache, num_tokens, pool_tid,
    )
    return pool_tid, state_tid
