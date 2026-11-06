# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3.8-27B layer 0 DeltaNet prefill for four active tokens."""

import pypto.language as pl

from attention_state import KV_WIDTH, Q_WIDTH, ROTARY_DIM, attention_prefill_into
from delta_state import (
    HEAD_DIM,
    VALUE_HEADS,
    VALUE_WIDTH,
    conv_silu_prefill,
    delta_gated_rms,
    delta_gates,
    delta_qk_norm,
    delta_step,
)
from ops import MLP_WIDTH, linear_into, mlp_into, residual_into, rmsnorm_into

HIDDEN = 5120
QKV = 10240
PADDED_TOKENS = 16
PREFILL_TOKENS = 4
CHANNEL_TILE = 128
K_TILE = 256
EPS = 1e-6


@pl.jit.inline
def project_raw_qkv(
    x: pl.Tensor[[PADDED_TOKENS, HIDDEN], pl.BF16],
    norm_weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    qkv_weight: pl.Tensor[[QKV, HIDDEN], pl.BF16],
    raw: pl.Tensor[[PADDED_TOKENS, QKV], pl.BF16],
) -> pl.Tensor[[PADDED_TOKENS, QKV], pl.BF16]:
    normalized = pl.create_tensor([PADDED_TOKENS, HIDDEN], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="input_rmsnorm"):
        square_sum = pl.full([1, PADDED_TOKENS], dtype=pl.FP32, value=0.0)
        for kb in pl.range(HIDDEN // K_TILE):
            k0 = kb * K_TILE
            x_float = pl.cast(x[:, k0 : k0 + K_TILE], target_type=pl.FP32)
            partial = pl.reshape(pl.row_sum(pl.mul(x_float, x_float)), [1, PADDED_TOKENS])
            square_sum = pl.add(square_sum, partial)
        inv_rms = pl.reshape(
            pl.rsqrt(pl.add(pl.mul(square_sum, 1.0 / HIDDEN), EPS)),
            [PADDED_TOKENS, 1],
        )
        for kb in pl.range(HIDDEN // K_TILE):
            k0 = kb * K_TILE
            x_float = pl.cast(x[:, k0 : k0 + K_TILE], target_type=pl.FP32)
            norm_chunk = pl.cast(norm_weight[:, k0 : k0 + K_TILE], target_type=pl.FP32)
            gamma = pl.add(norm_chunk, 1.0)
            scaled = pl.row_expand_mul(x_float, inv_rms)
            normalized[:, k0 : k0 + K_TILE] = pl.cast(pl.col_expand_mul(scaled, gamma), target_type=pl.BF16)

    for block in pl.spmd(QKV // CHANNEL_TILE, name_hint="raw_qkv_projection"):
        channel = block * CHANNEL_TILE
        acc = pl.create_tensor([PADDED_TOKENS, CHANNEL_TILE], dtype=pl.FP32)
        for kb in pl.range(HIDDEN // K_TILE):
            k0 = kb * K_TILE
            a = normalized[:, k0 : k0 + K_TILE]
            qkv_chunk = qkv_weight[channel : channel + CHANNEL_TILE, k0 : k0 + K_TILE]
            acc = pl.matmul_acc(acc, a, qkv_chunk, b_trans=True, init_cond=(kb == 0))
        raw[:, channel : channel + CHANNEL_TILE] = pl.cast(acc, target_type=pl.BF16)
    return raw


@pl.jit.inline
def delta_prefill_into(
    x: pl.Tensor[[PADDED_TOKENS, HIDDEN], pl.BF16],
    norm_weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    qkv_weight: pl.Tensor[[QKV, HIDDEN], pl.BF16],
    conv_weight: pl.Tensor[[4, QKV], pl.BF16],
    a_weight: pl.Tensor[[128, HIDDEN], pl.BF16],
    b_weight: pl.Tensor[[128, HIDDEN], pl.BF16],
    z_weight: pl.Tensor[[VALUE_WIDTH, HIDDEN], pl.BF16],
    a_log: pl.Tensor[[1, 128], pl.BF16],
    dt_bias: pl.Tensor[[1, 128], pl.BF16],
    delta_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.BF16],
    out_weight: pl.Tensor[[HIDDEN, VALUE_WIDTH], pl.BF16],
    post_norm_weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    mlp_gate_weight: pl.Tensor[[MLP_WIDTH, HIDDEN], pl.BF16],
    mlp_up_weight: pl.Tensor[[MLP_WIDTH, HIDDEN], pl.BF16],
    mlp_down_weight: pl.Tensor[[HIDDEN, MLP_WIDTH], pl.BF16],
    hidden: pl.Tensor[[PADDED_TOKENS, HIDDEN], pl.BF16],
    conv: pl.Tensor[[PADDED_TOKENS, QKV], pl.BF16],
    recurrent: pl.Tensor[[VALUE_HEADS, HEAD_DIM, HEAD_DIM], pl.FP32],
):
    normalized = pl.create_tensor([PADDED_TOKENS, HIDDEN], dtype=pl.BF16)
    activated = pl.create_tensor([PADDED_TOKENS, QKV], dtype=pl.BF16)
    a = pl.create_tensor([PADDED_TOKENS, 128], dtype=pl.BF16)
    b = pl.create_tensor([PADDED_TOKENS, 128], dtype=pl.BF16)
    gate = pl.create_tensor([PADDED_TOKENS, VALUE_WIDTH], dtype=pl.BF16)
    beta = pl.create_tensor([PADDED_TOKENS, 128], dtype=pl.FP32)
    decay = pl.create_tensor([PADDED_TOKENS, 128], dtype=pl.FP32)
    query = pl.create_tensor([PADDED_TOKENS, 2048], dtype=pl.FP32)
    key = pl.create_tensor([PADDED_TOKENS, 2048], dtype=pl.FP32)
    core = pl.create_tensor([PADDED_TOKENS, VALUE_WIDTH], dtype=pl.BF16)
    gated_core = pl.create_tensor([PADDED_TOKENS, VALUE_WIDTH], dtype=pl.BF16)
    mixed = pl.create_tensor([PADDED_TOKENS, HIDDEN], dtype=pl.BF16)
    residual = pl.create_tensor([PADDED_TOKENS, HIDDEN], dtype=pl.BF16)
    zero_state = pl.create_tensor([VALUE_HEADS, HEAD_DIM, HEAD_DIM], dtype=pl.FP32)
    state_1 = pl.create_tensor([VALUE_HEADS, HEAD_DIM, HEAD_DIM], dtype=pl.FP32)
    state_2 = pl.create_tensor([VALUE_HEADS, HEAD_DIM, HEAD_DIM], dtype=pl.FP32)
    state_3 = pl.create_tensor([VALUE_HEADS, HEAD_DIM, HEAD_DIM], dtype=pl.FP32)

    rmsnorm_into(x, norm_weight, 1.0 / HIDDEN, normalized)
    project_raw_qkv(x, norm_weight, qkv_weight, conv)
    conv_silu_prefill(conv, conv_weight, activated)
    linear_into(normalized, a_weight, a)
    linear_into(normalized, b_weight, b)
    linear_into(normalized, z_weight, gate)
    delta_gates(a, b, a_log, dt_bias, beta, decay)
    delta_qk_norm(activated, query, key)
    for head in pl.spmd(VALUE_HEADS, name_hint="delta_zero_state"):
        zero_state[head, :, :] = pl.full([HEAD_DIM, HEAD_DIM], dtype=pl.FP32, value=0.0)
    for block in pl.spmd(VALUE_WIDTH // CHANNEL_TILE, name_hint="delta_zero_core"):
        col = block * CHANNEL_TILE
        core[:, col : col + CHANNEL_TILE] = pl.full([PADDED_TOKENS, CHANNEL_TILE], dtype=pl.BF16, value=0.0)
    delta_step(activated, query, key, beta, decay, zero_state, 0, state_1, core)
    delta_step(activated, query, key, beta, decay, state_1, 1, state_2, core)
    delta_step(activated, query, key, beta, decay, state_2, 2, state_3, core)
    delta_step(activated, query, key, beta, decay, state_3, 3, recurrent, core)
    delta_gated_rms(core, gate, delta_norm_weight, gated_core)
    linear_into(gated_core, out_weight, mixed)
    residual_into(x, mixed, residual)
    mlp_into(residual, post_norm_weight, mlp_gate_weight, mlp_up_weight, mlp_down_weight, hidden)
    return hidden, conv, recurrent


@pl.jit
def prefill_layer(
    x: pl.Tensor[[PADDED_TOKENS, HIDDEN], pl.BF16],
    norm_weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    qkv_weight: pl.Tensor[[QKV, HIDDEN], pl.BF16],
    conv_weight: pl.Tensor[[4, QKV], pl.BF16],
    a_weight: pl.Tensor[[128, HIDDEN], pl.BF16],
    b_weight: pl.Tensor[[128, HIDDEN], pl.BF16],
    z_weight: pl.Tensor[[VALUE_WIDTH, HIDDEN], pl.BF16],
    a_log: pl.Tensor[[1, 128], pl.BF16],
    dt_bias: pl.Tensor[[1, 128], pl.BF16],
    delta_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.BF16],
    out_weight: pl.Tensor[[HIDDEN, VALUE_WIDTH], pl.BF16],
    post_norm_weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    mlp_gate_weight: pl.Tensor[[MLP_WIDTH, HIDDEN], pl.BF16],
    mlp_up_weight: pl.Tensor[[MLP_WIDTH, HIDDEN], pl.BF16],
    mlp_down_weight: pl.Tensor[[HIDDEN, MLP_WIDTH], pl.BF16],
    attn_q_weight: pl.Tensor[[2 * Q_WIDTH, HIDDEN], pl.BF16],
    attn_k_weight: pl.Tensor[[KV_WIDTH, HIDDEN], pl.BF16],
    attn_v_weight: pl.Tensor[[KV_WIDTH, HIDDEN], pl.BF16],
    attn_q_norm_weight: pl.Tensor[[1, 256], pl.BF16],
    attn_k_norm_weight: pl.Tensor[[1, 256], pl.BF16],
    attn_o_weight: pl.Tensor[[HIDDEN, Q_WIDTH], pl.BF16],
    cos: pl.Tensor[[PADDED_TOKENS, ROTARY_DIM], pl.BF16],
    sin: pl.Tensor[[PADDED_TOKENS, ROTARY_DIM], pl.BF16],
    hidden: pl.Out[pl.Tensor[[PADDED_TOKENS, HIDDEN], pl.BF16]],
    conv: pl.Out[pl.Tensor[[PADDED_TOKENS, QKV], pl.BF16]],
    recurrent: pl.Out[pl.Tensor[[VALUE_HEADS, HEAD_DIM, HEAD_DIM], pl.FP32]],
    key_cache: pl.Out[pl.Tensor[[PADDED_TOKENS, KV_WIDTH], pl.BF16]],
    value_cache: pl.Out[pl.Tensor[[PADDED_TOKENS, KV_WIDTH], pl.BF16]],
    layer_kind: pl.constexpr,
):
    if layer_kind == 0:
        delta_prefill_into(
            x,
            norm_weight,
            qkv_weight,
            conv_weight,
            a_weight,
            b_weight,
            z_weight,
            a_log,
            dt_bias,
            delta_norm_weight,
            out_weight,
            post_norm_weight,
            mlp_gate_weight,
            mlp_up_weight,
            mlp_down_weight,
            hidden,
            conv,
            recurrent,
        )
    else:
        attention_prefill_into(
            x,
            norm_weight,
            attn_q_weight,
            attn_k_weight,
            attn_v_weight,
            attn_q_norm_weight,
            attn_k_norm_weight,
            attn_o_weight,
            cos,
            sin,
            post_norm_weight,
            mlp_gate_weight,
            mlp_up_weight,
            mlp_down_weight,
            hidden,
            key_cache,
            value_cache,
        )
    return hidden, conv, recurrent, key_cache, value_cache
