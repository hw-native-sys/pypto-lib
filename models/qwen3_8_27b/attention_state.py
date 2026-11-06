# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO partial-RoPE and sequential KV state updates for Qwen3.8 attention."""

import pypto.language as pl

from ops import HIDDEN, MLP_WIDTH, linear_into, mlp_into, residual_into, rmsnorm_into

ROWS_DYN = pl.dynamic("ATTN_ROWS_DYN")
HEAD_DIM = 256
ROTARY_DIM = 64
ROW_TILE = 16
CACHE_ROWS = 16
KV_WIDTH = 1024
Q_HEADS = 24
KV_HEADS = 4
Q_WIDTH = Q_HEADS * HEAD_DIM
TOKENS_PAD = 16
EPS = 1e-6


@pl.jit
def rotate_partial(
    x: pl.Tensor[[ROWS_DYN, HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[ROWS_DYN, ROTARY_DIM], pl.BF16],
    sin: pl.Tensor[[ROWS_DYN, ROTARY_DIM], pl.BF16],
    out: pl.Out[pl.Tensor[[ROWS_DYN, HEAD_DIM], pl.BF16]],
):
    rows = pl.tensor.dim(x, 0)
    for block in pl.spmd(rows // ROW_TILE, name_hint="partial_rope"):
        row = block * ROW_TILE
        first = pl.cast(x[row : row + ROW_TILE, 0:ROTARY_DIM], target_type=pl.FP32)
        left = first[:, 0 : ROTARY_DIM // 2]
        right = first[:, ROTARY_DIM // 2 : ROTARY_DIM]
        shifted = pl.concat(pl.neg(right), left)
        cos_chunk = pl.cast(cos[row : row + ROW_TILE, :], target_type=pl.FP32)
        sin_chunk = pl.cast(sin[row : row + ROW_TILE, :], target_type=pl.FP32)
        out[row : row + ROW_TILE, 0:ROTARY_DIM] = pl.cast(
            pl.add(pl.mul(first, cos_chunk), pl.mul(shifted, sin_chunk)),
            target_type=pl.BF16,
            mode="rint",
        )
        out[row : row + ROW_TILE, ROTARY_DIM:HEAD_DIM] = x[row : row + ROW_TILE, ROTARY_DIM:HEAD_DIM]
    return out


@pl.jit
def append_kv(
    previous: pl.Tensor[[CACHE_ROWS, KV_WIDTH], pl.BF16],
    new: pl.Tensor[[CACHE_ROWS, KV_WIDTH], pl.BF16],
    position: pl.Scalar[pl.INDEX],
    out: pl.Out[pl.Tensor[[CACHE_ROWS, KV_WIDTH], pl.BF16]],
):
    for row in pl.spmd(CACHE_ROWS, name_hint="append_kv"):
        if row == position:
            out[row : row + 1, :] = new[0:1, :]
        else:
            out[row : row + 1, :] = previous[row : row + 1, :]
    return out


@pl.jit.inline
def append_kv_into(
    previous: pl.Tensor[[CACHE_ROWS, KV_WIDTH], pl.BF16],
    new: pl.Tensor[[CACHE_ROWS, KV_WIDTH], pl.BF16],
    position: pl.Scalar[pl.INDEX],
    out: pl.Tensor[[CACHE_ROWS, KV_WIDTH], pl.BF16],
):
    for row in pl.spmd(CACHE_ROWS, name_hint="layer_append_kv"):
        if row == position:
            out[row : row + 1, :] = new[0:1, :]
        else:
            out[row : row + 1, :] = previous[row : row + 1, :]
    return out


@pl.jit.inline
def norm_rope_query_into(
    q_gate: pl.Tensor[[TOKENS_PAD, 2 * Q_WIDTH], pl.BF16],
    weight: pl.Tensor[[1, HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[TOKENS_PAD, ROTARY_DIM], pl.BF16],
    sin: pl.Tensor[[TOKENS_PAD, ROTARY_DIM], pl.BF16],
    query: pl.Tensor[[TOKENS_PAD, Q_WIDTH], pl.BF16],
    gate: pl.Tensor[[TOKENS_PAD, Q_WIDTH], pl.BF16],
):
    for head in pl.spmd(Q_HEADS, name_hint="attention_query_norm_rope"):
        source = head * 2 * HEAD_DIM
        target = head * HEAD_DIM
        raw = pl.cast(q_gate[:, source : source + HEAD_DIM], target_type=pl.FP32)
        inv_rms = pl.rsqrt(pl.add(pl.mul(pl.row_sum(pl.mul(raw, raw)), 1.0 / HEAD_DIM), EPS))
        gamma = pl.add(pl.cast(weight, target_type=pl.FP32), 1.0)
        normalized = pl.cast(
            pl.col_expand_mul(pl.row_expand_mul(raw, inv_rms), gamma),
            target_type=pl.BF16,
            mode="rint",
        )
        first = pl.cast(normalized[:, 0:ROTARY_DIM], target_type=pl.FP32)
        shifted = pl.concat(pl.neg(first[:, ROTARY_DIM // 2 : ROTARY_DIM]), first[:, 0 : ROTARY_DIM // 2])
        angle_cos = pl.cast(cos, target_type=pl.FP32)
        angle_sin = pl.cast(sin, target_type=pl.FP32)
        rotated = pl.add(pl.mul(first, angle_cos), pl.mul(shifted, angle_sin))
        query[:, target : target + ROTARY_DIM] = pl.cast(rotated, target_type=pl.BF16, mode="rint")
        query[:, target + ROTARY_DIM : target + HEAD_DIM] = normalized[:, ROTARY_DIM:HEAD_DIM]
        gate[:, target : target + HEAD_DIM] = q_gate[:, source + HEAD_DIM : source + 2 * HEAD_DIM]
    return query, gate


@pl.jit.inline
def norm_rope_key_into(
    projected: pl.Tensor[[TOKENS_PAD, KV_WIDTH], pl.BF16],
    weight: pl.Tensor[[1, HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[TOKENS_PAD, ROTARY_DIM], pl.BF16],
    sin: pl.Tensor[[TOKENS_PAD, ROTARY_DIM], pl.BF16],
    key: pl.Tensor[[TOKENS_PAD, KV_WIDTH], pl.BF16],
):
    for head in pl.spmd(KV_HEADS, name_hint="attention_key_norm_rope"):
        column = head * HEAD_DIM
        raw = pl.cast(projected[:, column : column + HEAD_DIM], target_type=pl.FP32)
        inv_rms = pl.rsqrt(pl.add(pl.mul(pl.row_sum(pl.mul(raw, raw)), 1.0 / HEAD_DIM), EPS))
        gamma = pl.add(pl.cast(weight, target_type=pl.FP32), 1.0)
        normalized = pl.cast(
            pl.col_expand_mul(pl.row_expand_mul(raw, inv_rms), gamma),
            target_type=pl.BF16,
            mode="rint",
        )
        first = pl.cast(normalized[:, 0:ROTARY_DIM], target_type=pl.FP32)
        shifted = pl.concat(pl.neg(first[:, ROTARY_DIM // 2 : ROTARY_DIM]), first[:, 0 : ROTARY_DIM // 2])
        angle_cos = pl.cast(cos, target_type=pl.FP32)
        angle_sin = pl.cast(sin, target_type=pl.FP32)
        rotated = pl.add(pl.mul(first, angle_cos), pl.mul(shifted, angle_sin))
        key[:, column : column + ROTARY_DIM] = pl.cast(rotated, target_type=pl.BF16, mode="rint")
        key[:, column + ROTARY_DIM : column + HEAD_DIM] = normalized[:, ROTARY_DIM:HEAD_DIM]
    return key


@pl.jit.inline
def attention_context_into(
    query: pl.Tensor[[TOKENS_PAD, Q_WIDTH], pl.BF16],
    gate: pl.Tensor[[TOKENS_PAD, Q_WIDTH], pl.BF16],
    key: pl.Tensor[[CACHE_ROWS, KV_WIDTH], pl.BF16],
    value: pl.Tensor[[CACHE_ROWS, KV_WIDTH], pl.BF16],
    start_position: pl.Scalar[pl.INDEX],
    active_tokens: pl.constexpr,
    context: pl.Tensor[[TOKENS_PAD, Q_WIDTH], pl.BF16],
):
    for task in pl.spmd(active_tokens * Q_HEADS, name_hint="attention_causal_core"):
        token = task // Q_HEADS
        head = task % Q_HEADS
        kv_head = head // (Q_HEADS // KV_HEADS)
        query_col = head * HEAD_DIM
        cache_col = kv_head * HEAD_DIM
        q = query[token : token + 1, query_col : query_col + HEAD_DIM]
        k = key[:, cache_col : cache_col + HEAD_DIM]
        v = value[:, cache_col : cache_col + HEAD_DIM]
        logits = pl.matmul(q, k, b_trans=True, out_dtype=pl.FP32)
        bias = pl.full([1, CACHE_ROWS], dtype=pl.FP32, value=-1.0e9)
        for position in pl.range(start_position + token + 1):
            pl.write(bias, [0, position], 0.0)
        scores = pl.col_expand_add(pl.mul(logits, HEAD_DIM**-0.5), bias)
        maximum = pl.row_max(scores)
        exponential = pl.exp(pl.row_expand_sub(scores, maximum))
        probabilities = pl.cast(
            pl.row_expand_div(exponential, pl.row_sum(exponential)),
            target_type=pl.BF16,
            mode="rint",
        )
        result = pl.cast(pl.matmul(probabilities, v, out_dtype=pl.FP32), target_type=pl.BF16, mode="rint")
        gate_f32 = pl.cast(gate[token : token + 1, query_col : query_col + HEAD_DIM], target_type=pl.FP32)
        sigmoid = pl.cast(
            pl.recip(pl.add(pl.exp(pl.neg(gate_f32)), 1.0)),
            target_type=pl.BF16,
            mode="rint",
        )
        context[token : token + 1, query_col : query_col + HEAD_DIM] = pl.cast(
            pl.mul(pl.cast(result, target_type=pl.FP32), pl.cast(sigmoid, target_type=pl.FP32)),
            target_type=pl.BF16,
            mode="rint",
        )
    for head in pl.spmd(Q_HEADS, name_hint="attention_zero_inactive"):
        col = head * HEAD_DIM
        context[active_tokens:TOKENS_PAD, col : col + HEAD_DIM] = pl.full(
            [TOKENS_PAD - active_tokens, HEAD_DIM], dtype=pl.BF16, value=0.0
        )
    return context


@pl.jit.inline
def attention_prefill_into(
    x: pl.Tensor[[TOKENS_PAD, HIDDEN], pl.BF16],
    input_norm_weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    q_weight: pl.Tensor[[2 * Q_WIDTH, HIDDEN], pl.BF16],
    k_weight: pl.Tensor[[KV_WIDTH, HIDDEN], pl.BF16],
    v_weight: pl.Tensor[[KV_WIDTH, HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.BF16],
    k_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.BF16],
    o_weight: pl.Tensor[[HIDDEN, Q_WIDTH], pl.BF16],
    cos: pl.Tensor[[TOKENS_PAD, ROTARY_DIM], pl.BF16],
    sin: pl.Tensor[[TOKENS_PAD, ROTARY_DIM], pl.BF16],
    post_norm_weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    mlp_gate_weight: pl.Tensor[[MLP_WIDTH, HIDDEN], pl.BF16],
    mlp_up_weight: pl.Tensor[[MLP_WIDTH, HIDDEN], pl.BF16],
    mlp_down_weight: pl.Tensor[[HIDDEN, MLP_WIDTH], pl.BF16],
    hidden: pl.Tensor[[TOKENS_PAD, HIDDEN], pl.BF16],
    key_cache: pl.Tensor[[CACHE_ROWS, KV_WIDTH], pl.BF16],
    value_cache: pl.Tensor[[CACHE_ROWS, KV_WIDTH], pl.BF16],
):
    normalized = pl.create_tensor([TOKENS_PAD, HIDDEN], dtype=pl.BF16)
    q_gate = pl.create_tensor([TOKENS_PAD, 2 * Q_WIDTH], dtype=pl.BF16)
    projected_key = pl.create_tensor([TOKENS_PAD, KV_WIDTH], dtype=pl.BF16)
    projected_value = pl.create_tensor([TOKENS_PAD, KV_WIDTH], dtype=pl.BF16)
    query = pl.create_tensor([TOKENS_PAD, Q_WIDTH], dtype=pl.BF16)
    gate = pl.create_tensor([TOKENS_PAD, Q_WIDTH], dtype=pl.BF16)
    context = pl.create_tensor([TOKENS_PAD, Q_WIDTH], dtype=pl.BF16)
    mixed = pl.create_tensor([TOKENS_PAD, HIDDEN], dtype=pl.BF16)
    residual = pl.create_tensor([TOKENS_PAD, HIDDEN], dtype=pl.BF16)

    rmsnorm_into(x, input_norm_weight, 1.0 / HIDDEN, normalized)
    linear_into(normalized, q_weight, q_gate)
    linear_into(normalized, k_weight, projected_key)
    linear_into(normalized, v_weight, projected_value)
    norm_rope_query_into(q_gate, q_norm_weight, cos, sin, query, gate)
    norm_rope_key_into(projected_key, k_norm_weight, cos, sin, key_cache)
    for head in pl.spmd(KV_HEADS, name_hint="attention_prefill_value_cache"):
        col = head * HEAD_DIM
        value_cache[:, col : col + HEAD_DIM] = projected_value[:, col : col + HEAD_DIM]
    attention_context_into(query, gate, key_cache, value_cache, 0, 4, context)
    linear_into(context, o_weight, mixed)
    residual_into(x, mixed, residual)
    mlp_into(residual, post_norm_weight, mlp_gate_weight, mlp_up_weight, mlp_down_weight, hidden)
    return hidden, key_cache, value_cache


@pl.jit.inline
def attention_decode_into(
    x: pl.Tensor[[TOKENS_PAD, HIDDEN], pl.BF16],
    input_norm_weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    q_weight: pl.Tensor[[2 * Q_WIDTH, HIDDEN], pl.BF16],
    k_weight: pl.Tensor[[KV_WIDTH, HIDDEN], pl.BF16],
    v_weight: pl.Tensor[[KV_WIDTH, HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.BF16],
    k_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.BF16],
    o_weight: pl.Tensor[[HIDDEN, Q_WIDTH], pl.BF16],
    cos: pl.Tensor[[TOKENS_PAD, ROTARY_DIM], pl.BF16],
    sin: pl.Tensor[[TOKENS_PAD, ROTARY_DIM], pl.BF16],
    post_norm_weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    mlp_gate_weight: pl.Tensor[[MLP_WIDTH, HIDDEN], pl.BF16],
    mlp_up_weight: pl.Tensor[[MLP_WIDTH, HIDDEN], pl.BF16],
    mlp_down_weight: pl.Tensor[[HIDDEN, MLP_WIDTH], pl.BF16],
    previous_key: pl.Tensor[[CACHE_ROWS, KV_WIDTH], pl.BF16],
    previous_value: pl.Tensor[[CACHE_ROWS, KV_WIDTH], pl.BF16],
    position: pl.Scalar[pl.INDEX],
    hidden: pl.Tensor[[TOKENS_PAD, HIDDEN], pl.BF16],
    key_cache: pl.Tensor[[CACHE_ROWS, KV_WIDTH], pl.BF16],
    value_cache: pl.Tensor[[CACHE_ROWS, KV_WIDTH], pl.BF16],
):
    normalized = pl.create_tensor([TOKENS_PAD, HIDDEN], dtype=pl.BF16)
    q_gate = pl.create_tensor([TOKENS_PAD, 2 * Q_WIDTH], dtype=pl.BF16)
    projected_key = pl.create_tensor([TOKENS_PAD, KV_WIDTH], dtype=pl.BF16)
    projected_value = pl.create_tensor([TOKENS_PAD, KV_WIDTH], dtype=pl.BF16)
    new_key = pl.create_tensor([TOKENS_PAD, KV_WIDTH], dtype=pl.BF16)
    query = pl.create_tensor([TOKENS_PAD, Q_WIDTH], dtype=pl.BF16)
    gate = pl.create_tensor([TOKENS_PAD, Q_WIDTH], dtype=pl.BF16)
    context = pl.create_tensor([TOKENS_PAD, Q_WIDTH], dtype=pl.BF16)
    mixed = pl.create_tensor([TOKENS_PAD, HIDDEN], dtype=pl.BF16)
    residual = pl.create_tensor([TOKENS_PAD, HIDDEN], dtype=pl.BF16)

    rmsnorm_into(x, input_norm_weight, 1.0 / HIDDEN, normalized)
    linear_into(normalized, q_weight, q_gate)
    linear_into(normalized, k_weight, projected_key)
    linear_into(normalized, v_weight, projected_value)
    norm_rope_query_into(q_gate, q_norm_weight, cos, sin, query, gate)
    norm_rope_key_into(projected_key, k_norm_weight, cos, sin, new_key)
    append_kv_into(previous_key, new_key, position, key_cache)
    append_kv_into(previous_value, projected_value, position, value_cache)
    attention_context_into(query, gate, key_cache, value_cache, position, 1, context)
    linear_into(context, o_weight, mixed)
    residual_into(x, mixed, residual)
    mlp_into(residual, post_norm_weight, mlp_gate_weight, mlp_up_weight, mlp_down_weight, hidden)
    return hidden, key_cache, value_cache
