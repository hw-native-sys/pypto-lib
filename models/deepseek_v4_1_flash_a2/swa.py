# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""BF16 paged SWA cache publication, window gather and sink attention."""

import pypto.language as pl


# tiling
HEAD_TILE = 16
WINDOW_TILE = 128


def make_swa_inline(capacity, cache_blocks, config):
    """Build a BF16 PA_BBND SWA leaf for the full configured query width."""
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    window = config["sliding_window"]
    if window % WINDOW_TILE:
        raise ValueError("A2 SWA requires a window divisible by its 128-token softmax tile")
    width = heads * head_dim
    cache_rows = cache_blocks * 128
    score_scale = head_dim ** -0.5

    @pl.jit.inline
    def publish(
        kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(cache, [cache_rows, head_dim])
        for token in pl.spmd(active, name_hint="a2_swa_publish"):
            page = pl.read(slots, [token, 0])
            offset = pl.read(slots, [token, 1])
            if page >= 0:
                address = page * 128 + offset
                row = pl.cast(address, pl.INDEX)
                flat[row:row + 1, :] = kv[token:token + 1, :]
        return cache

    @pl.jit.inline
    def gather(
        cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        selected: pl.Tensor[[capacity * window, head_dim], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(cache, [cache_rows, head_dim])
        for block in pl.spmd(active * window, name_hint="a2_swa_gather"):
            token = block // window
            column = block % window
            index = pl.read(indices, [token, column])
            if index >= 0:
                row = pl.cast(index, pl.INDEX)
                selected[block:block + 1, :] = flat[row:row + 1, :]
            else:
                selected[block:block + 1, :] = pl.full([1, head_dim], dtype=pl.BF16, value=0.0)
        return selected

    @pl.jit.inline
    def attend(
        query: pl.Tensor[[capacity, width], pl.BF16],
        selected: pl.Tensor[[capacity * window, head_dim], pl.BF16],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        output: pl.Tensor[[capacity, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        qflat = pl.reshape(query, [capacity * heads, head_dim])
        oflat = pl.reshape(output, [capacity * heads, head_dim])
        for block in pl.spmd(active * (heads // HEAD_TILE), name_hint="a2_swa_attention"):
            token = block // (heads // HEAD_TILE)
            head = block % (heads // HEAD_TILE) * HEAD_TILE
            query_row = token * heads + head
            # Include the zero-value sink before narrowing exp(scores-max) to
            # BF16, matching the native softmax's rounding boundary.
            maximum = pl.reshape(sink[head:head + HEAD_TILE], [1, HEAD_TILE])
            denominator = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=1.0)
            numerator = pl.full([HEAD_TILE, head_dim], dtype=pl.FP32, value=0.0)
            for part in pl.range(window // WINDOW_TILE):
                cache_row = token * window + part * WINDOW_TILE
                q = qflat[query_row:query_row + HEAD_TILE, :]
                kv = selected[cache_row:cache_row + WINDOW_TILE, :]
                dot = pl.matmul(q, kv, b_trans=True)
                scores = pl.mul(dot, score_scale)
                index_tile = indices[token:token + 1, part * WINDOW_TILE:part * WINDOW_TILE + WINDOW_TILE]
                index_float = pl.cast(index_tile, pl.FP32)
                shifted_index = pl.add(index_float, 1.0)
                positive_index = pl.maximum(shifted_index, 0.0)
                valid = pl.minimum(positive_index, 1.0)
                invalid = pl.sub(valid, 1.0)
                bias = pl.mul(invalid, 1e30)
                scores = pl.col_expand_add(scores, bias)
                row_max = pl.row_max(scores)
                local_max = pl.reshape(row_max, [1, HEAD_TILE])
                next_max = pl.maximum(maximum, local_max)
                max_difference = pl.sub(maximum, next_max)
                correction = pl.exp(max_difference)
                max_column = pl.reshape(next_max, [HEAD_TILE, 1])
                shifted = pl.row_expand_sub(scores, max_column)
                exponential = pl.exp(shifted)
                probabilities = pl.col_expand_mul(exponential, valid)
                row_sum = pl.row_sum(probabilities)
                probability_sum = pl.reshape(row_sum, [1, HEAD_TILE])
                old_denominator = pl.mul(denominator, correction)
                denominator = pl.add(old_denominator, probability_sum)
                weights = pl.cast(probabilities, pl.BF16, mode="rint")
                weighted = pl.matmul(weights, kv)
                correction_column = pl.reshape(correction, [HEAD_TILE, 1])
                corrected = pl.row_expand_mul(numerator, correction_column)
                numerator = pl.add(corrected, weighted)
                maximum = next_max
            one = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=1.0)
            normalized = pl.div(one, denominator, high_precision=True)
            normalized_column = pl.reshape(normalized, [HEAD_TILE, 1])
            result = pl.row_expand_mul(numerator, normalized_column)
            oflat[query_row:query_row + HEAD_TILE, :] = pl.cast(result, pl.BF16, mode="rint")
        return output

    @pl.jit.inline
    def swa_inline(
        query: pl.Tensor[[capacity, width], pl.BF16],
        kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
        output: pl.Tensor[[capacity, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        publish(kv, slots, cache, active)
        selected = pl.create_tensor([capacity * window, head_dim], dtype=pl.BF16)
        gather(cache, indices, selected, active)
        attend(query, selected, indices, sink, output, active)
        return cache, output

    return swa_inline


def make_swa(capacity, cache_blocks, config):
    """Build the standalone JIT wrapper for shared inline SWA compute."""
    head_dim = config["head_dim"]
    heads = config["num_attention_heads"]
    width = heads * head_dim
    window = config["sliding_window"]
    compute = make_swa_inline(capacity, cache_blocks, config)

    @pl.jit
    def swa(
        query: pl.Tensor[[capacity, width], pl.BF16],
        kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        cache: pl.InOut[pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16]],
        output: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute(query, kv, slots, indices, sink, cache, output, active)
        return cache, output

    return swa
