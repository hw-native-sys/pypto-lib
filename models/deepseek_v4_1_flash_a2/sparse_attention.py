# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2 single-layer dependencies extracted from the shared model operators."""

import pypto.language as pl
from .config import FLASH as M, T_DYN, TP_SIZE


SELECTED_ROWS = pl.dynamic("SPARSE_SELECTED_ROWS")


BF16_HEADS = pl.dynamic("SPARSE_BF16_HEADS")


BF16_PAGES_W = pl.dynamic("SPARSE_BF16_PAGES_W")


BF16_PAGES_C = pl.dynamic("SPARSE_BF16_PAGES_C")


HEAD_DIM = M.head_dim


INDEX_TOPK = M.index_topk


LOCAL_H = M.num_attention_heads // TP_SIZE


SELECTED_KEYS = 128 + INDEX_TOPK


SCALE = HEAD_DIM**-0.5


CUBE_HEADS = max(16, LOCAL_H)


HEAD_TILE = min(16, CUBE_HEADS)


QUERY_HEAD_TILE = min(16, LOCAL_H)


KEY_TILE = 128


TOKEN_TILE = 512


WORKERS = 32


@pl.jit.inline(auto_scope=False)
def _attend_cube(
    query: pl.Tensor[[T_DYN, CUBE_HEADS * HEAD_DIM], pl.BF16],
    selected: pl.Tensor[[SELECTED_ROWS, SELECTED_KEYS, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[SELECTED_ROWS, SELECTED_KEYS], pl.INT32],
    sink: pl.Tensor[[CUBE_HEADS], pl.FP32],
    output: pl.Tensor[[T_DYN, CUBE_HEADS * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    selected_ready: pl.Scalar[pl.TASK_ID],
    query_ready: pl.Scalar[pl.TASK_ID],
):
    tokens = pl.tensor.dim(query, 0)
    head_rows = tokens * CUBE_HEADS
    cache_rows = pl.tensor.dim(selected, 0) * SELECTED_KEYS
    qflat = pl.reshape(query, [head_rows, HEAD_DIM])
    kflat = pl.reshape(selected, [cache_rows, HEAD_DIM])
    oflat = pl.reshape(output, [head_rows, HEAD_DIM])
    workers = pl.max(1, pl.min(num_tokens * ((CUBE_HEADS + HEAD_TILE - 1) // HEAD_TILE), WORKERS))
    with pl.spmd(workers, name_hint="sparse_online_cube", deps=[selected_ready, query_ready], optimizations=[pl.cross_core_slot(slot_num=1)]) as attended:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, num_tokens * ((CUBE_HEADS + HEAD_TILE - 1) // HEAD_TILE), workers):
            token = block // ((CUBE_HEADS + HEAD_TILE - 1) // HEAD_TILE)
            head = block % ((CUBE_HEADS + HEAD_TILE - 1) // HEAD_TILE) * HEAD_TILE
            q0 = token * CUBE_HEADS + head
            q = qflat[q0:q0 + HEAD_TILE, :]
            maximum = pl.reshape(sink[head:head + HEAD_TILE], [1, HEAD_TILE])
            denominator = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=1.0)
            numerator = pl.full([HEAD_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
            for part in pl.range(SELECTED_KEYS // KEY_TILE):
                k0 = token * SELECTED_KEYS + part * KEY_TILE
                kv = kflat[k0:k0 + KEY_TILE, :]
                qk = pl.matmul(q, kv, b_trans=True, out_dtype=pl.FP32)
                scores = pl.mul(qk, SCALE)
                idx0 = part * KEY_TILE
                idx = pl.cast(indices[token:token + 1, idx0:idx0 + KEY_TILE], pl.FP32)
                valid = pl.add(idx, 1.0)
                valid = pl.maximum(valid, 0.0)
                valid = pl.minimum(valid, 1.0)
                penalty = pl.mul(pl.sub(valid, 1.0), 1e30)
                scores = pl.col_expand_add(scores, penalty)
                score_max = pl.reshape(pl.row_max(scores), [1, HEAD_TILE])
                next_max = pl.maximum(maximum, score_max)
                correction = pl.exp(pl.sub(maximum, next_max))
                max_col = pl.reshape(next_max, [HEAD_TILE, 1])
                shifted = pl.row_expand_sub(scores, max_col)
                probabilities = pl.exp(shifted)
                probabilities = pl.col_expand_mul(probabilities, valid)
                previous_denominator = pl.mul(denominator, correction)
                part_denominator = pl.reshape(pl.row_sum(probabilities), [1, HEAD_TILE])
                denominator = pl.add(previous_denominator, part_denominator)
                weights = pl.cast(probabilities, pl.BF16, mode="round")
                weighted = pl.matmul(weights, kv, out_dtype=pl.FP32)
                correction_col = pl.reshape(correction, [HEAD_TILE, 1])
                previous_numerator = pl.row_expand_mul(numerator, correction_col)
                numerator = pl.add(previous_numerator, weighted)
                maximum = next_max
            one = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=1.0)
            normalized = pl.div(one, denominator, high_precision=True)
            multiplier = pl.reshape(normalized, [HEAD_TILE, 1])
            result = pl.row_expand_mul(numerator, multiplier)
            result = pl.cast(result, pl.BF16, mode="rint")
            oflat[q0:q0 + HEAD_TILE, :] = result
    return attended


@pl.jit.inline(auto_scope=False)
def _attend_padded(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    selected: pl.Tensor[[SELECTED_ROWS, SELECTED_KEYS, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[SELECTED_ROWS, SELECTED_KEYS], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    selected_ready: pl.Scalar[pl.TASK_ID],
    query_ready: pl.Scalar[pl.TASK_ID],
):
    # Cube uses at least 16 rows. TP8 retains only its eight real heads.
    rows = pl.tensor.dim(query, 0)
    qflat = pl.reshape(query, [rows * LOCAL_H, HEAD_DIM])
    padded_query = pl.create_tensor([rows, CUBE_HEADS * HEAD_DIM], dtype=pl.BF16)
    pqflat = pl.reshape(padded_query, [rows * CUBE_HEADS, HEAD_DIM])
    padded_sink = pl.create_tensor([CUBE_HEADS], dtype=pl.FP32)
    padded_sink_row = pl.reshape(padded_sink, [1, CUBE_HEADS])
    sink_row = pl.reshape(sink, [1, LOCAL_H])
    workers = pl.max(1, pl.min(num_tokens, WORKERS))
    with pl.spmd(workers, name_hint="sparse_pad_heads", deps=[query_ready]) as padded:
        worker = pl.tile.get_block_idx()
        if worker == 0:
            sinks = pl.full([1, CUBE_HEADS], dtype=pl.FP32, value=0.0)
            sinks[:, 0:LOCAL_H] = sink_row[:, :]
            padded_sink_row[:, :] = sinks
        for token in pl.range(worker, num_tokens, workers):
            q0 = token * LOCAL_H
            pq0 = token * CUBE_HEADS
            q = pl.full([CUBE_HEADS, HEAD_DIM], dtype=pl.BF16, value=0.0)
            q[0:LOCAL_H, :] = qflat[q0:q0 + LOCAL_H, :]
            pqflat[pq0:pq0 + CUBE_HEADS, :] = q

    padded_output = pl.create_tensor([rows, CUBE_HEADS * HEAD_DIM], dtype=pl.BF16)
    cube_ready = _attend_cube(
        padded_query, selected, indices, padded_sink, padded_output,
        num_tokens, selected_ready, padded,
    )
    poflat = pl.reshape(padded_output, [rows * CUBE_HEADS, HEAD_DIM])
    oflat = pl.reshape(output, [rows * LOCAL_H, HEAD_DIM])
    with pl.spmd(workers, name_hint="sparse_unpad_heads", deps=[cube_ready]) as attended:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, num_tokens, workers):
            p0 = token * CUBE_HEADS
            o0 = token * LOCAL_H
            oflat[o0:o0 + LOCAL_H, :] = poflat[p0:p0 + LOCAL_H, :]
    return attended


attend_selected = _attend_cube if LOCAL_H >= 16 else _attend_padded


@pl.jit.inline(auto_scope=False)
def _attend_bf16_block(
    query: pl.Tensor[[T_DYN, BF16_HEADS, HEAD_DIM], pl.FP32],
    window: pl.Tensor[[BF16_PAGES_W, 128, HEAD_DIM], pl.FP32],
    wi: pl.Tensor[[T_DYN, 128], pl.INT32],
    compressed: pl.Tensor[[BF16_PAGES_C, 128, HEAD_DIM], pl.FP32],
    ci: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    sink: pl.Tensor[[BF16_HEADS], pl.FP32],
    extents: pl.Tensor[[T_DYN, 2], pl.INT32],
    output: pl.Tensor[[T_DYN, BF16_HEADS, HEAD_DIM], pl.FP32],
    previous: pl.Scalar[pl.TASK_ID],
):
    rows = pl.tensor.dim(query, 0)
    head_rows = rows * LOCAL_H
    qflat = pl.reshape(query, [head_rows, HEAD_DIM])
    qbf16 = pl.create_tensor([rows, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    qbflat = pl.reshape(qbf16, [head_rows, HEAD_DIM])
    workers_q = pl.max(1, pl.min(rows * (LOCAL_H // QUERY_HEAD_TILE), WORKERS))
    with pl.spmd(workers_q, name_hint="sparse_bf16_query", deps=[previous]) as query_ready:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, rows * (LOCAL_H // QUERY_HEAD_TILE), workers_q):
            q0 = block * QUERY_HEAD_TILE
            q_values = qflat[q0:q0 + QUERY_HEAD_TILE, :]
            qbflat[q0:q0 + QUERY_HEAD_TILE, :] = pl.cast(q_values, pl.BF16, mode="rint")

    wrows = pl.tensor.dim(window, 0) * 128
    crows = pl.tensor.dim(compressed, 0) * 128
    w = pl.reshape(window, [wrows, HEAD_DIM])
    c = pl.reshape(compressed, [crows, HEAD_DIM])
    selected = pl.create_tensor([rows, SELECTED_KEYS, HEAD_DIM], dtype=pl.BF16)
    dst = pl.reshape(selected, [rows * SELECTED_KEYS, HEAD_DIM])
    indices = pl.create_tensor([rows, SELECTED_KEYS], dtype=pl.INT32)
    workers_kv = pl.max(1, pl.min(rows * (SELECTED_KEYS // 16), WORKERS))
    with pl.spmd(workers_kv, name_hint="sparse_bf16_gather", deps=[previous]) as selected_ready:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, rows * (SELECTED_KEYS // 16), workers_kv):
            token = block // (SELECTED_KEYS // 16)
            column = block % (SELECTED_KEYS // 16) * 16
            ww = pl.read(extents, [token, 0])
            cc = pl.read(extents, [token, 1])
            values = pl.tile.full([16, HEAD_DIM], dtype=pl.BF16, value=0.0)
            valid = pl.tile.full([1, 16], dtype=pl.INT32, value=-1)
            for lane in pl.range(16):
                position = column + lane
                row = pl.cast(-1, pl.INT32)
                value = pl.tile.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0)
                # Keep the native SWA and compressed online-softmax partitions.
                if position < 128:
                    if position < ww:
                        row = pl.read(wi, [token, position])
                        if row >= 0:
                            value = pl.tile.load(w, [pl.cast(row, pl.INDEX), 0], [1, HEAD_DIM])
                elif position < 128 + cc:
                    row = pl.read(ci, [token, position - 128])
                    if row >= 0:
                        value = pl.tile.load(c, [pl.cast(row, pl.INDEX), 0], [1, HEAD_DIM])
                converted = pl.cast(value, pl.BF16, mode="rint")
                values = pl.tile.assemble(values, converted, [lane, 0])
                pl.tile.write(valid, [0, lane], row)
            pl.tile.store(values, [token * SELECTED_KEYS + column, 0], dst)
            pl.tile.store(valid, [token, column], indices)

    result = pl.create_tensor([rows, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    attended = attend_selected(qbf16, selected, indices, sink, result, rows, selected_ready, query_ready)
    rflat = pl.reshape(result, [head_rows, HEAD_DIM])
    oflat = pl.reshape(output, [head_rows, HEAD_DIM])
    with pl.spmd(workers_q, name_hint="sparse_bf16_output", deps=[attended]) as converted:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, rows * (LOCAL_H // QUERY_HEAD_TILE), workers_q):
            q0 = block * QUERY_HEAD_TILE
            out_values = rflat[q0:q0 + QUERY_HEAD_TILE, :]
            oflat[q0:q0 + QUERY_HEAD_TILE, :] = pl.cast(out_values, pl.FP32)
    return converted


@pl.jit.inline(auto_scope=False)
def attend_bf16(
    query: pl.Tensor[[T_DYN, BF16_HEADS, HEAD_DIM], pl.FP32],
    window: pl.Tensor[[BF16_PAGES_W, 128, HEAD_DIM], pl.FP32],
    wi: pl.Tensor[[T_DYN, 128], pl.INT32],
    compressed: pl.Tensor[[BF16_PAGES_C, 128, HEAD_DIM], pl.FP32],
    ci: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    sink: pl.Tensor[[BF16_HEADS], pl.FP32],
    extents: pl.Tensor[[T_DYN, 2], pl.INT32],
    output: pl.Tensor[[T_DYN, BF16_HEADS, HEAD_DIM], pl.FP32],
):
    rows = pl.tensor.dim(query, 0)
    heads = pl.tensor.dim(query, 1)
    qflat = pl.reshape(query, [rows * LOCAL_H, HEAD_DIM])
    marker = pl.create_tensor([1, 16], dtype=pl.FP32)
    with pl.spmd(1, name_hint="sparse_bf16_ready") as ready:
        block = pl.tile.get_block_idx()
        marker[block:block + 1, :] = qflat[0:1, 0:16]

    for offset in pl.range(0, rows, TOKEN_TILE):
        count = pl.min(TOKEN_TILE, rows - offset)
        q = pl.slice(query, [count, heads, HEAD_DIM], [offset, 0, 0])
        widx = pl.slice(wi, [count, 128], [offset, 0])
        cidx = pl.slice(ci, [count, INDEX_TOPK], [offset, 0])
        limits = pl.slice(extents, [count, 2], [offset, 0])
        out = pl.slice(output, [count, heads, HEAD_DIM], [offset, 0, 0])
        ready = _attend_bf16_block(q, window, widx, compressed, cidx, sink, limits, out, ready)
    return output
