# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared token-major Qwen3.8 whole-network step and persistent-state ABI."""

import pypto.language as pl

from attention_state import (
    HEAD_DIM as ATTN_HEAD_DIM,
    KV_HEADS,
    KV_WIDTH,
    Q_HEADS,
    Q_WIDTH,
    ROTARY_DIM,
    norm_rope_key_into,
    norm_rope_query_into,
)
from decode_layer import delta_decode_into
from delta_state import HEAD_DIM, QKV, VALUE_HEADS, VALUE_WIDTH
from ops import HIDDEN, MLP_WIDTH, residual_into, rmsnorm_into, swiglu_into

NUM_LAYERS = 64
DELTA_LAYERS = 48
ATTN_LAYERS = 16
TOKENS_PAD = 16
MAX_SEQUENCE = 128
VOCAB = 248320
CHANNEL_TILE = 128
@pl.jit.inline
def attention_linear_into(
    x: pl.Tensor,
    weight: pl.Tensor,
    out: pl.Tensor,
):
    inner = pl.tensor.dim(x, 1)
    columns = pl.tensor.dim(weight, 0)
    for block in pl.spmd(columns // CHANNEL_TILE, name_hint="full_attention_linear"):
        column = block * CHANNEL_TILE
        acc = pl.create_tensor([TOKENS_PAD, CHANNEL_TILE], dtype=pl.FP32)
        for kb in pl.range(inner // 256):
            k0 = kb * 256
            lhs = x[:, k0 : k0 + 256]
            rhs = weight[column : column + CHANNEL_TILE, k0 : k0 + 256]
            acc = pl.matmul_acc(acc, lhs, rhs, b_trans=True, init_cond=(kb == 0))
        out[:, column : column + CHANNEL_TILE] = pl.cast(acc, target_type=pl.BF16, mode="rint")
    return out


@pl.jit.inline
def attention_mlp_into(
    x: pl.Tensor[[TOKENS_PAD, HIDDEN], pl.BF16],
    norm_weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    gate_weight: pl.Tensor[[MLP_WIDTH, HIDDEN], pl.BF16],
    up_weight: pl.Tensor[[MLP_WIDTH, HIDDEN], pl.BF16],
    down_weight: pl.Tensor[[HIDDEN, MLP_WIDTH], pl.BF16],
    out: pl.Tensor[[TOKENS_PAD, HIDDEN], pl.BF16],
):
    normalized = pl.create_tensor([TOKENS_PAD, HIDDEN], dtype=pl.BF16)
    gate = pl.create_tensor([TOKENS_PAD, MLP_WIDTH], dtype=pl.BF16)
    up = pl.create_tensor([TOKENS_PAD, MLP_WIDTH], dtype=pl.BF16)
    product = pl.create_tensor([TOKENS_PAD, MLP_WIDTH], dtype=pl.BF16)
    projected = pl.create_tensor([TOKENS_PAD, HIDDEN], dtype=pl.BF16)
    rmsnorm_into(x, norm_weight, 1.0 / HIDDEN, normalized)
    attention_linear_into(normalized, gate_weight, gate)
    attention_linear_into(normalized, up_weight, up)
    swiglu_into(gate, up, product)
    attention_linear_into(product, down_weight, projected)
    residual_into(x, projected, out)
    return out


@pl.jit.inline
def append_kv_full(
    previous: pl.Tensor[[MAX_SEQUENCE, KV_WIDTH], pl.BF16],
    new: pl.Tensor[[TOKENS_PAD, KV_WIDTH], pl.BF16],
    position: pl.Scalar[pl.INDEX],
    out: pl.Tensor[[MAX_SEQUENCE, KV_WIDTH], pl.BF16],
):
    for block in pl.spmd(MAX_SEQUENCE * (KV_WIDTH // CHANNEL_TILE), name_hint="full_append_kv"):
        row = block // (KV_WIDTH // CHANNEL_TILE)
        col = (block % (KV_WIDTH // CHANNEL_TILE)) * CHANNEL_TILE
        if row == position:
            out[row : row + 1, col : col + CHANNEL_TILE] = new[0:1, col : col + CHANNEL_TILE]
        else:
            out[row : row + 1, col : col + CHANNEL_TILE] = previous[
                row : row + 1, col : col + CHANNEL_TILE
            ]
    return out


@pl.jit.inline
def attention_context_full(
    query: pl.Tensor[[TOKENS_PAD, Q_WIDTH], pl.BF16],
    gate: pl.Tensor[[TOKENS_PAD, Q_WIDTH], pl.BF16],
    key: pl.Tensor[[MAX_SEQUENCE, KV_WIDTH], pl.BF16],
    value: pl.Tensor[[MAX_SEQUENCE, KV_WIDTH], pl.BF16],
    position: pl.Scalar[pl.INDEX],
    context: pl.Tensor[[TOKENS_PAD, Q_WIDTH], pl.BF16],
):
    for head in pl.spmd(Q_HEADS, name_hint="full_attention_causal"):
        query_col = head * ATTN_HEAD_DIM
        cache_col = (head // (Q_HEADS // KV_HEADS)) * ATTN_HEAD_DIM
        q = query[0:1, query_col : query_col + ATTN_HEAD_DIM]
        k = key[:, cache_col : cache_col + ATTN_HEAD_DIM]
        v = value[:, cache_col : cache_col + ATTN_HEAD_DIM]
        logits = pl.matmul(q, k, b_trans=True, out_dtype=pl.FP32)
        bias = pl.full([1, MAX_SEQUENCE], dtype=pl.FP32, value=-1.0e9)
        for row in pl.range(position + 1):
            pl.write(bias, [0, row], 0.0)
        scores = pl.col_expand_add(pl.mul(logits, ATTN_HEAD_DIM**-0.5), bias)
        maximum = pl.row_max(scores)
        exponential = pl.exp(pl.row_expand_sub(scores, maximum))
        probabilities = pl.cast(
            pl.row_expand_div(exponential, pl.row_sum(exponential)), target_type=pl.BF16, mode="rint"
        )
        result = pl.cast(pl.matmul(probabilities, v, out_dtype=pl.FP32), target_type=pl.BF16, mode="rint")
        gate_float = pl.cast(gate[0:1, query_col : query_col + ATTN_HEAD_DIM], target_type=pl.FP32)
        sigmoid = pl.cast(
            pl.recip(pl.add(pl.exp(pl.neg(gate_float)), 1.0)), target_type=pl.BF16, mode="rint"
        )
        context[0:1, query_col : query_col + ATTN_HEAD_DIM] = pl.cast(
            pl.mul(pl.cast(result, target_type=pl.FP32), pl.cast(sigmoid, target_type=pl.FP32)),
            target_type=pl.BF16,
            mode="rint",
        )
        context[1:TOKENS_PAD, query_col : query_col + ATTN_HEAD_DIM] = pl.full(
            [TOKENS_PAD - 1, ATTN_HEAD_DIM], dtype=pl.BF16, value=0.0
        )
    return context


@pl.jit.inline
def attention_step_full(
    x: pl.Tensor[[TOKENS_PAD, HIDDEN], pl.BF16],
    input_norm: pl.Tensor[[1, HIDDEN], pl.BF16],
    q_weight: pl.Tensor[[2 * Q_WIDTH, HIDDEN], pl.BF16],
    k_weight: pl.Tensor[[KV_WIDTH, HIDDEN], pl.BF16],
    v_weight: pl.Tensor[[KV_WIDTH, HIDDEN], pl.BF16],
    q_norm: pl.Tensor[[1, ATTN_HEAD_DIM], pl.BF16],
    k_norm: pl.Tensor[[1, ATTN_HEAD_DIM], pl.BF16],
    o_weight: pl.Tensor[[HIDDEN, Q_WIDTH], pl.BF16],
    cos: pl.Tensor[[TOKENS_PAD, ROTARY_DIM], pl.BF16],
    sin: pl.Tensor[[TOKENS_PAD, ROTARY_DIM], pl.BF16],
    post_norm: pl.Tensor[[1, HIDDEN], pl.BF16],
    mlp_gate: pl.Tensor[[MLP_WIDTH, HIDDEN], pl.BF16],
    mlp_up: pl.Tensor[[MLP_WIDTH, HIDDEN], pl.BF16],
    mlp_down: pl.Tensor[[HIDDEN, MLP_WIDTH], pl.BF16],
    previous_key: pl.Tensor[[MAX_SEQUENCE, KV_WIDTH], pl.BF16],
    previous_value: pl.Tensor[[MAX_SEQUENCE, KV_WIDTH], pl.BF16],
    position: pl.Scalar[pl.INDEX],
    hidden: pl.Tensor[[TOKENS_PAD, HIDDEN], pl.BF16],
    key_cache: pl.Tensor[[MAX_SEQUENCE, KV_WIDTH], pl.BF16],
    value_cache: pl.Tensor[[MAX_SEQUENCE, KV_WIDTH], pl.BF16],
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

    rmsnorm_into(x, input_norm, 1.0 / HIDDEN, normalized)
    attention_linear_into(normalized, q_weight, q_gate)
    attention_linear_into(normalized, k_weight, projected_key)
    attention_linear_into(normalized, v_weight, projected_value)
    norm_rope_query_into(q_gate, q_norm, cos, sin, query, gate)
    norm_rope_key_into(projected_key, k_norm, cos, sin, new_key)
    append_kv_full(previous_key, new_key, position, key_cache)
    append_kv_full(previous_value, projected_value, position, value_cache)
    attention_context_full(query, gate, key_cache, value_cache, position, context)
    attention_linear_into(context, o_weight, mixed)
    residual_into(x, mixed, residual)
    attention_mlp_into(residual, post_norm, mlp_gate, mlp_up, mlp_down, hidden)
    return hidden, key_cache, value_cache


@pl.jit.inline
def network_token_step(
    token_ids: pl.Tensor[[MAX_SEQUENCE], pl.INT64],
    position: pl.Scalar[pl.INDEX],
    embedding: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    input_norms: pl.Tensor[[NUM_LAYERS, 1, HIDDEN], pl.BF16],
    post_norms: pl.Tensor[[NUM_LAYERS, 1, HIDDEN], pl.BF16],
    mlp_gates: pl.Tensor[[NUM_LAYERS, MLP_WIDTH, HIDDEN], pl.BF16],
    mlp_ups: pl.Tensor[[NUM_LAYERS, MLP_WIDTH, HIDDEN], pl.BF16],
    mlp_downs: pl.Tensor[[NUM_LAYERS, HIDDEN, MLP_WIDTH], pl.BF16],
    delta_qkvs: pl.Tensor[[DELTA_LAYERS, QKV, HIDDEN], pl.BF16],
    delta_convs: pl.Tensor[[DELTA_LAYERS, 4, QKV], pl.BF16],
    delta_as: pl.Tensor[[DELTA_LAYERS, 128, HIDDEN], pl.BF16],
    delta_bs: pl.Tensor[[DELTA_LAYERS, 128, HIDDEN], pl.BF16],
    delta_zs: pl.Tensor[[DELTA_LAYERS, VALUE_WIDTH, HIDDEN], pl.BF16],
    delta_a_logs: pl.Tensor[[DELTA_LAYERS, 1, 128], pl.BF16],
    delta_dt_biases: pl.Tensor[[DELTA_LAYERS, 1, 128], pl.BF16],
    delta_norms: pl.Tensor[[DELTA_LAYERS, 1, HEAD_DIM], pl.BF16],
    delta_outs: pl.Tensor[[DELTA_LAYERS, HIDDEN, VALUE_WIDTH], pl.BF16],
    attn_qs: pl.Tensor[[ATTN_LAYERS, 2 * Q_WIDTH, HIDDEN], pl.BF16],
    attn_ks: pl.Tensor[[ATTN_LAYERS, KV_WIDTH, HIDDEN], pl.BF16],
    attn_vs: pl.Tensor[[ATTN_LAYERS, KV_WIDTH, HIDDEN], pl.BF16],
    attn_q_norms: pl.Tensor[[ATTN_LAYERS, 1, ATTN_HEAD_DIM], pl.BF16],
    attn_k_norms: pl.Tensor[[ATTN_LAYERS, 1, ATTN_HEAD_DIM], pl.BF16],
    attn_outs: pl.Tensor[[ATTN_LAYERS, HIDDEN, Q_WIDTH], pl.BF16],
    cos_table: pl.Tensor[[MAX_SEQUENCE, ROTARY_DIM], pl.BF16],
    sin_table: pl.Tensor[[MAX_SEQUENCE, ROTARY_DIM], pl.BF16],
    conv_states: pl.Tensor[[DELTA_LAYERS, TOKENS_PAD, QKV], pl.BF16],
    recurrent_states: pl.Tensor[[DELTA_LAYERS, VALUE_HEADS, HEAD_DIM, HEAD_DIM], pl.FP32],
    key_states: pl.Tensor[[ATTN_LAYERS, MAX_SEQUENCE, KV_WIDTH], pl.BF16],
    value_states: pl.Tensor[[ATTN_LAYERS, MAX_SEQUENCE, KV_WIDTH], pl.BF16],
    hidden_out: pl.Tensor[[TOKENS_PAD, HIDDEN], pl.BF16],
):
    activations = pl.create_tensor([(NUM_LAYERS + 1) * TOKENS_PAD, HIDDEN], dtype=pl.BF16)
    for block in pl.spmd(HIDDEN // CHANNEL_TILE, name_hint="full_token_embedding"):
        col = block * CHANNEL_TILE
        token_id = pl.cast(pl.tensor.read(token_ids, [position]), target_type=pl.INDEX)
        activations[0:1, col : col + CHANNEL_TILE] = embedding[
            token_id : token_id + 1, col : col + CHANNEL_TILE
        ]
        activations[1:TOKENS_PAD, col : col + CHANNEL_TILE] = pl.full(
            [TOKENS_PAD - 1, CHANNEL_TILE], dtype=pl.BF16, value=0.0
        )

    cos = pl.slice(cos_table, [TOKENS_PAD, ROTARY_DIM], [position, 0])
    sin = pl.slice(sin_table, [TOKENS_PAD, ROTARY_DIM], [position, 0])
    for group in pl.range(ATTN_LAYERS):
        for phase in pl.range(3):
            layer = group * 4 + phase
            delta_index = group * 3 + phase
            hidden = pl.slice(activations, [TOKENS_PAD, HIDDEN], [layer * TOKENS_PAD, 0])
            next_hidden = pl.create_tensor([TOKENS_PAD, HIDDEN], dtype=pl.BF16)
            next_conv = pl.create_tensor([TOKENS_PAD, QKV], dtype=pl.BF16)
            next_recurrent = pl.create_tensor([VALUE_HEADS, HEAD_DIM, HEAD_DIM], dtype=pl.FP32)
            delta_decode_into(
                hidden,
                input_norms[layer],
                delta_qkvs[delta_index],
                delta_convs[delta_index],
                delta_as[delta_index],
                delta_bs[delta_index],
                delta_zs[delta_index],
                delta_a_logs[delta_index],
                delta_dt_biases[delta_index],
                delta_norms[delta_index],
                delta_outs[delta_index],
                post_norms[layer],
                mlp_gates[layer],
                mlp_ups[layer],
                mlp_downs[layer],
                conv_states[delta_index],
                recurrent_states[delta_index],
                next_hidden,
                next_conv,
                next_recurrent,
            )
            for block in pl.spmd(QKV // CHANNEL_TILE, name_hint="full_commit_conv"):
                col = block * CHANNEL_TILE
                conv_states[delta_index, :, col : col + CHANNEL_TILE] = next_conv[:, col : col + CHANNEL_TILE]
            for head in pl.spmd(VALUE_HEADS, name_hint="full_commit_recurrent"):
                recurrent_states[delta_index, head, :, :] = next_recurrent[head, :, :]

            for block in pl.spmd(HIDDEN // CHANNEL_TILE, name_hint="full_commit_delta_hidden"):
                col = block * CHANNEL_TILE
                next_row = (layer + 1) * TOKENS_PAD
                activations[next_row : next_row + TOKENS_PAD, col : col + CHANNEL_TILE] = next_hidden[
                    :, col : col + CHANNEL_TILE
                ]

        layer = group * 4 + 3
        hidden = pl.slice(activations, [TOKENS_PAD, HIDDEN], [layer * TOKENS_PAD, 0])
        next_hidden = pl.create_tensor([TOKENS_PAD, HIDDEN], dtype=pl.BF16)
        next_key = pl.create_tensor([MAX_SEQUENCE, KV_WIDTH], dtype=pl.BF16)
        next_value = pl.create_tensor([MAX_SEQUENCE, KV_WIDTH], dtype=pl.BF16)
        attention_step_full(
            hidden,
            input_norms[layer],
            attn_qs[group],
            attn_ks[group],
            attn_vs[group],
            attn_q_norms[group],
            attn_k_norms[group],
            attn_outs[group],
            cos,
            sin,
            post_norms[layer],
            mlp_gates[layer],
            mlp_ups[layer],
            mlp_downs[layer],
            key_states[group],
            value_states[group],
            position,
            next_hidden,
            next_key,
            next_value,
        )
        for block in pl.spmd(MAX_SEQUENCE * (KV_WIDTH // CHANNEL_TILE), name_hint="full_commit_kv"):
            row = block // (KV_WIDTH // CHANNEL_TILE)
            col = (block % (KV_WIDTH // CHANNEL_TILE)) * CHANNEL_TILE
            key_states[group, row : row + 1, col : col + CHANNEL_TILE] = next_key[
                row : row + 1, col : col + CHANNEL_TILE
            ]
            value_states[group, row : row + 1, col : col + CHANNEL_TILE] = next_value[
                row : row + 1, col : col + CHANNEL_TILE
            ]
        for block in pl.spmd(HIDDEN // CHANNEL_TILE, name_hint="full_commit_attention_hidden"):
            col = block * CHANNEL_TILE
            next_row = (layer + 1) * TOKENS_PAD
            activations[next_row : next_row + TOKENS_PAD, col : col + CHANNEL_TILE] = next_hidden[
                :, col : col + CHANNEL_TILE
            ]

    last_hidden = pl.slice(activations, [TOKENS_PAD, HIDDEN], [NUM_LAYERS * TOKENS_PAD, 0])
    for block in pl.spmd(HIDDEN // CHANNEL_TILE, name_hint="full_last_hidden"):
        col = block * CHANNEL_TILE
        hidden_out[:, col : col + CHANNEL_TILE] = last_hidden[:, col : col + CHANNEL_TILE]
    return hidden_out


@pl.jit.inline
def final_logits(
    hidden: pl.Tensor[[TOKENS_PAD, HIDDEN], pl.BF16],
    final_norm: pl.Tensor[[1, HIDDEN], pl.BF16],
    lm_head: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    logits: pl.Tensor[[1, VOCAB], pl.FP32],
):
    normalized = pl.create_tensor([TOKENS_PAD, HIDDEN], dtype=pl.BF16)
    rmsnorm_into(hidden, final_norm, 1.0 / HIDDEN, normalized)
    for block in pl.spmd(VOCAB // CHANNEL_TILE, name_hint="full_lm_head"):
        col = block * CHANNEL_TILE
        acc = pl.create_tensor([TOKENS_PAD, CHANNEL_TILE], dtype=pl.FP32)
        for kb in pl.range(HIDDEN // 256):
            inner = kb * 256
            lhs = normalized[:, inner : inner + 256]
            rhs = lm_head[col : col + CHANNEL_TILE, inner : inner + 256]
            acc = pl.matmul_acc(acc, lhs, rhs, b_trans=True, init_cond=(kb == 0))
        rounded = pl.cast(acc, target_type=pl.BF16, mode="rint")
        logits[:, col : col + CHANNEL_TILE] = pl.cast(rounded[0:1, :], target_type=pl.FP32)
    return logits
