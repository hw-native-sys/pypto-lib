# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3.8-27B layer 0 DeltaNet decode with carried convolution and state."""

import pypto.language as pl

from attention_state import KV_WIDTH, Q_WIDTH, ROTARY_DIM, attention_decode_into
from delta_state import (
    HEAD_DIM,
    VALUE_HEADS,
    VALUE_WIDTH,
    conv_silu_decode,
    delta_gated_rms,
    delta_gates,
    delta_qk_norm,
    delta_step,
)
from ops import MLP_WIDTH, linear_into, mlp_into, residual_into, rmsnorm_into
from prefill_layer import CHANNEL_TILE, HIDDEN, PADDED_TOKENS, QKV, project_raw_qkv

HISTORY = 4


@pl.jit.inline
def delta_decode_into(
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
    previous_conv: pl.Tensor[[PADDED_TOKENS, QKV], pl.BF16],
    previous_recurrent: pl.Tensor[[VALUE_HEADS, HEAD_DIM, HEAD_DIM], pl.FP32],
    hidden: pl.Tensor[[PADDED_TOKENS, HIDDEN], pl.BF16],
    conv: pl.Tensor[[PADDED_TOKENS, QKV], pl.BF16],
    recurrent: pl.Tensor[[VALUE_HEADS, HEAD_DIM, HEAD_DIM], pl.FP32],
):
    normalized = pl.create_tensor([PADDED_TOKENS, HIDDEN], dtype=pl.BF16)
    raw = pl.create_tensor([PADDED_TOKENS, QKV], dtype=pl.BF16)
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

    rmsnorm_into(x, norm_weight, 1.0 / HIDDEN, normalized)
    project_raw_qkv(x, norm_weight, qkv_weight, raw)
    for block in pl.spmd(QKV // CHANNEL_TILE, name_hint="decode_raw_history"):
        channel = block * CHANNEL_TILE
        conv[0:3, channel : channel + CHANNEL_TILE] = previous_conv[
            1:HISTORY, channel : channel + CHANNEL_TILE
        ]
        conv[3:4, channel : channel + CHANNEL_TILE] = raw[0:1, channel : channel + CHANNEL_TILE]
        conv[4:PADDED_TOKENS, channel : channel + CHANNEL_TILE] = raw[
            4:PADDED_TOKENS, channel : channel + CHANNEL_TILE
        ]
    conv_silu_decode(conv, conv_weight, activated)
    linear_into(normalized, a_weight, a)
    linear_into(normalized, b_weight, b)
    linear_into(normalized, z_weight, gate)
    delta_gates(a, b, a_log, dt_bias, beta, decay)
    delta_qk_norm(activated, query, key)
    for block in pl.spmd(VALUE_WIDTH // CHANNEL_TILE, name_hint="delta_zero_core"):
        col = block * CHANNEL_TILE
        core[:, col : col + CHANNEL_TILE] = pl.full([PADDED_TOKENS, CHANNEL_TILE], dtype=pl.BF16, value=0.0)
    delta_step(activated, query, key, beta, decay, previous_recurrent, 0, recurrent, core)
    delta_gated_rms(core, gate, delta_norm_weight, gated_core)
    linear_into(gated_core, out_weight, mixed)
    residual_into(x, mixed, residual)
    mlp_into(residual, post_norm_weight, mlp_gate_weight, mlp_up_weight, mlp_down_weight, hidden)
    return hidden, conv, recurrent


@pl.jit
def decode_layer(
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
    previous_conv: pl.Tensor[[PADDED_TOKENS, QKV], pl.BF16],
    previous_recurrent: pl.Tensor[[VALUE_HEADS, HEAD_DIM, HEAD_DIM], pl.FP32],
    previous_key: pl.Tensor[[PADDED_TOKENS, KV_WIDTH], pl.BF16],
    previous_value: pl.Tensor[[PADDED_TOKENS, KV_WIDTH], pl.BF16],
    position: pl.Scalar[pl.INDEX],
    hidden: pl.Out[pl.Tensor[[PADDED_TOKENS, HIDDEN], pl.BF16]],
    conv: pl.Out[pl.Tensor[[PADDED_TOKENS, QKV], pl.BF16]],
    recurrent: pl.Out[pl.Tensor[[VALUE_HEADS, HEAD_DIM, HEAD_DIM], pl.FP32]],
    key_cache: pl.Out[pl.Tensor[[PADDED_TOKENS, KV_WIDTH], pl.BF16]],
    value_cache: pl.Out[pl.Tensor[[PADDED_TOKENS, KV_WIDTH], pl.BF16]],
    layer_kind: pl.constexpr,
):
    if layer_kind == 0:
        delta_decode_into(
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
            previous_conv,
            previous_recurrent,
            hidden,
            conv,
            recurrent,
        )
    else:
        attention_decode_into(
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
            previous_key,
            previous_value,
            position,
            hidden,
            key_cache,
            value_cache,
        )
    return hidden, conv, recurrent, key_cache, value_cache
