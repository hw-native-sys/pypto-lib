# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pipelined rotary position embedding for the A3 attention projections."""

import pypto.language as pl

from .config import FLASH as M
from .dsl_types import T_DYN


# model config
ROPE_DIM = M.qk_rope_head_dim

# tiling
ROPE_WORKER_LIMIT = 48
TOKEN_ROPE_ROWS_TILE = 16


def make_pipelined_token_rope(capacity, head_dim, *, name_hint, inplace):
    """Rotate single-head rows in double-buffered token tiles."""
    nope = head_dim - 64
    load_start = nope if inplace else 0
    load_width = head_dim - load_start
    rows = TOKEN_ROPE_ROWS_TILE
    elements = rows * 64

    @pl.jit.inline
    def rope(
        source: pl.Tensor[[capacity, head_dim], pl.BF16],
        cosine: pl.Tensor[[capacity, 32], pl.FP32],
        sine: pl.Tensor[[capacity, 32], pl.FP32],
        output: pl.Tensor[[capacity, head_dim], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        blocks = (active + rows - 1) // rows
        workers = pl.min(pl.min(blocks, ROPE_WORKER_LIMIT), pl.system.available_aiv_count())
        for worker in pl.spmd(workers, name_hint=name_hint):
            lanes = pl.cast(pl.tile.arange(0, [1, elements], dtype=pl.INT32), pl.FP32)
            pairs = pl.cast(pl.mul(lanes, 0.5), pl.INT32, mode="trunc")
            parity = pl.sub(lanes, pl.mul(pl.cast(pairs, pl.FP32), 2.0))
            swap_idx = pl.cast(pl.sub(pl.add(lanes, 1.0), pl.mul(parity, 2.0)), pl.INT32)
            sign = pl.sub(pl.mul(parity, 2.0), 1.0)
            start = worker * blocks // workers
            end = (worker + 1) * blocks // workers
            for block in pl.pipeline(start, end, stage=2):
                row = block * rows
                valid = pl.min(rows, active - row)
                loaded = pl.load(source, [row, load_start], [rows, load_width], valid_shape=[valid, load_width])
                value = pl.set_validshape(loaded, rows, load_width)
                tail = pl.tile.slice(value, [rows, 64], [0, nope - load_start])
                flat = pl.reshape(pl.cast(tail, pl.FP32), [1, elements])
                scratch = pl.create_tile([1, elements], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                swapped = pl.tile.gather(flat, swap_idx, scratch)
                c_tile = pl.load(cosine, [row, 0], [rows, 32], valid_shape=[valid, 32])
                s_tile = pl.load(sine, [row, 0], [rows, 32], valid_shape=[valid, 32])
                c_table = pl.reshape(pl.set_validshape(c_tile, rows, 32), [1, rows * 32])
                s_table = pl.reshape(pl.set_validshape(s_tile, rows, 32), [1, rows * 32])
                c = pl.tile.gather(c_table, pairs, scratch)
                s = pl.mul(pl.tile.gather(s_table, pairs, scratch), sign)
                direct = pl.mul(flat, c)
                cross = pl.mul(swapped, s)
                result = pl.reshape(pl.add(direct, cross), [rows, 64])
                rotated = pl.cast(result, pl.BF16, mode="rint")
                if inplace:
                    pl.store(pl.set_validshape(rotated, valid, 64), [row, nope], output)
                else:
                    prefix = pl.load(source, [row, 0], [rows, nope], valid_shape=[valid, nope])
                    pl.store(prefix, [row, 0], output)
                    pl.store(pl.set_validshape(rotated, valid, 64), [row, nope], output)
        return output

    return rope


ROPE_ROWS_DYN = pl.dynamic("V41_ROPE_ROWS_DYN")


@pl.jit.inline
def materialize_rope_rows(
    freqs_cos: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    freqs_sin: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    pair_start: pl.constexpr,
) -> pl.Tensor:
    """Gather active RoPE rows, using identity rotation for unpublished positions."""
    workers = pl.max(1, pl.min(num_tokens, pl.system.available_aiv_count()))
    with pl.spmd(workers, name_hint="v41_rope_rows"):
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, num_tokens, workers):
            position = pl.cast(pl.read(position_ids, [token]), pl.INDEX)
            if pair_start:
                if position >= 0 and (position + 1) % 2 == 0:
                    position = position - 1
                else:
                    position = -1
            if position >= 0:
                rope_cos[token : token + 1, :] = freqs_cos[position : position + 1, :]
                rope_sin[token : token + 1, :] = freqs_sin[position : position + 1, :]
            else:
                rope_cos[token : token + 1, :] = pl.full([1, ROPE_DIM // 2], dtype=pl.FP32, value=1.0)
                rope_sin[token : token + 1, :] = pl.full([1, ROPE_DIM // 2], dtype=pl.FP32, value=0.0)
    return rope_cos


def make_prefill_kv_norm_rope(capacity, head_dim, eps):
    """Normalize KV rows and rotate their BF16 tails in double-buffered tiles."""
    nope = head_dim - 64
    rows = TOKEN_ROPE_ROWS_TILE
    elements = rows * 64
    coeffs = rows * 32

    @pl.jit.inline(auto_scope=False)
    def kv_norm_rope(
        source: pl.Tensor[[capacity, head_dim], pl.BF16],
        weight: pl.Tensor[[head_dim], pl.BF16],
        cosine: pl.Tensor[[capacity, 32], pl.FP32],
        sine: pl.Tensor[[capacity, 32], pl.FP32],
        output: pl.Tensor[[capacity, head_dim], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        blocks = (active + rows - 1) // rows
        workers = pl.min(pl.min(blocks, ROPE_WORKER_LIMIT), pl.system.available_aiv_count())
        for worker in pl.spmd(workers, name_hint="rope_kv"):
            gamma = pl.reshape(pl.cast(pl.load(weight, [0], [head_dim]), pl.FP32), [1, head_dim])
            lanes = pl.tile.arange(0, [1, elements], dtype=pl.INT32)
            pairs = pl.shrs(lanes, 1)
            odd = pl.sub(lanes, pl.shls(pairs, 1))
            parity = pl.sub(pl.shls(odd, 1), 1)
            swap_idx = pl.sub(lanes, parity)
            sign = pl.cast(parity, pl.FP32)
            begin = worker * blocks // workers
            end = (worker + 1) * blocks // workers
            for block in pl.pipeline(begin, end, stage=2):
                row = block * rows
                valid = pl.min(rows, active - row)
                raw = pl.load(source, [row, 0], [rows, head_dim], valid_shape=[valid, head_dim])
                padded = pl.fillpad(raw, pad_value=pl.PadValue.zero)
                value = pl.cast(pl.set_validshape(padded, rows, head_dim), pl.FP32)
                squared = pl.mul(value, value)
                sum_scratch = pl.create_tile([rows, head_dim], dtype=pl.FP32)
                sum_squares = pl.reshape(pl.row_sum(squared, tmp_tile=sum_scratch), [1, rows])
                mean = pl.add(pl.mul(sum_squares, 1.0 / head_dim), eps)
                rsqrt_scratch = pl.create_tile([1, rows], dtype=pl.FP32)
                inv_rms = pl.tile.rsqrt(mean, rsqrt_scratch, high_precision=True)
                normalized = pl.row_expand_mul(value, pl.reshape(inv_rms, [rows, 1]))
                weighted = pl.col_expand_mul(normalized, gamma)
                rounded = pl.cast(weighted, pl.BF16, mode="rint")
                prefix = pl.tile.slice(rounded, [rows, nope], [0, 0])
                pl.tile.store(prefix, [row, 0], output, shapes=[valid, nope])
                tail = pl.tile.slice(rounded, [rows, 64], [0, nope])
                flat = pl.reshape(pl.cast(tail, pl.FP32), [1, elements])
                scratch = pl.create_tile([1, elements], dtype=pl.INT32)
                swapped = pl.tile.gather(flat, swap_idx, scratch)
                c_tile = pl.load(cosine, [row, 0], [rows, 32], valid_shape=[valid, 32])
                s_tile = pl.load(sine, [row, 0], [rows, 32], valid_shape=[valid, 32])
                c_table = pl.reshape(pl.set_validshape(c_tile, rows, 32), [1, coeffs])
                s_table = pl.reshape(pl.set_validshape(s_tile, rows, 32), [1, coeffs])
                c = pl.tile.gather(c_table, pairs, scratch)
                s = pl.mul(pl.tile.gather(s_table, pairs, scratch), sign)
                direct = pl.mul(flat, c)
                cross = pl.mul(swapped, s)
                result = pl.reshape(pl.add(direct, cross), [rows, 64])
                rotated = pl.cast(result, pl.BF16, mode="rint")
                pl.store(pl.set_validshape(rotated, valid, 64), [row, nope], output)
        return output

    return kv_norm_rope


def make_pipelined_rope(capacity, heads, head_dim, *, inverse, name_hint, allow_early_resolve, inplace=False):
    """Rotate heads sharing a token's coefficients. In-place callers pass x as output."""
    width = heads * head_dim
    nope = head_dim - 64
    load_start = nope if inplace else 0
    load_width = head_dim - load_start
    tail_start = nope - load_start
    rows = min(heads, 32)
    elements = rows * 64
    direction = -1.0 if inverse else 1.0

    @pl.jit.inline
    def rope(
        x: pl.Tensor[[capacity, width], pl.BF16],
        cosine: pl.Tensor[[capacity, 32], pl.FP32],
        sine: pl.Tensor[[capacity, 32], pl.FP32],
        output: pl.Tensor[[capacity, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        tokens = pl.tensor.dim(x, 0)
        source = pl.reshape(x, [tokens * heads, head_dim])
        target = pl.reshape(output, [tokens * heads, head_dim])
        blocks = active * (heads // rows)
        workers = pl.min(pl.min(blocks, ROPE_WORKER_LIMIT), pl.system.available_aiv_count())
        for worker in pl.spmd(workers, name_hint=name_hint, allow_early_resolve=allow_early_resolve):
            lanes = pl.cast(pl.tile.arange(0, [1, elements], dtype=pl.INT32), pl.FP32)
            pairs = pl.cast(pl.mul(lanes, 0.5), pl.INT32, mode="trunc")
            parity = pl.sub(lanes, pl.mul(pl.cast(pairs, pl.FP32), 2.0))
            swap_idx = pl.cast(pl.sub(pl.add(lanes, 1.0), pl.mul(parity, 2.0)), pl.INT32)
            coeff_idx = pl.tile.slice(pairs, [1, 64], [0, 0])
            head_parity = pl.tile.slice(parity, [1, 64], [0, 0])
            sign = pl.mul(pl.sub(pl.mul(head_parity, 2.0), 1.0), direction)
            start = worker * blocks // workers
            end = (worker + 1) * blocks // workers
            # One token per tile: expand the 32 pairs once, then broadcast
            # over the heads. Smaller coefficient buffers fit both UB slots.
            for block in pl.pipeline(start, end, stage=2):
                row = block * rows
                token = row // heads
                block_value = pl.load(source, [row, load_start], [rows, load_width])
                value = pl.cast(pl.tile.slice(block_value, [rows, 64], [0, tail_start]), pl.FP32)
                swap_tmp = pl.create_tile([1, elements], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                swapped = pl.tile.gather(pl.reshape(value, [1, elements]), swap_idx, swap_tmp)
                coeff_tmp = pl.create_tile([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                c = pl.tile.gather(pl.load(cosine, [token, 0], [1, 32]), coeff_idx, coeff_tmp)
                s = pl.mul(pl.tile.gather(pl.load(sine, [token, 0], [1, 32]), coeff_idx, coeff_tmp), sign)
                direct = pl.col_expand_mul(value, c)
                cross = pl.col_expand_mul(pl.reshape(swapped, [rows, 64]), s)
                rotated = pl.cast(pl.add(direct, cross), pl.BF16, mode="rint")
                if inplace:
                    pl.store(rotated, [row, nope], target)
                else:
                    prefix = pl.tile.slice(block_value, [rows, nope], [0, 0])
                    pl.store(prefix, [row, 0], target)
                    pl.store(rotated, [row, nope], target)
        return output

    return rope
