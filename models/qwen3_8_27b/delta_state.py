# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO causal convolution and gated DeltaNet recurrent state transitions."""

import pypto.language as pl

TOKENS_PAD = 16
QKV = 10240
QK_HEADS = 16
VALUE_HEADS = 48
HEAD_DIM = 128
VALUE_WIDTH = VALUE_HEADS * HEAD_DIM
CHANNEL_TILE = 128
VALUE_TILE = 64
COL_REP = 16
L2_EPS = 1e-6


@pl.jit.inline
def conv_silu_prefill(
    raw: pl.Tensor[[TOKENS_PAD, QKV], pl.BF16],
    conv_weight: pl.Tensor[[4, QKV], pl.BF16],
    out: pl.Tensor[[TOKENS_PAD, QKV], pl.BF16],
):
    for block in pl.spmd(QKV // CHANNEL_TILE, name_hint="delta_prefill_conv"):
        channel = block * CHANNEL_TILE
        w0 = pl.cast(conv_weight[0:1, channel : channel + CHANNEL_TILE], target_type=pl.FP32)
        w1 = pl.cast(conv_weight[1:2, channel : channel + CHANNEL_TILE], target_type=pl.FP32)
        w2 = pl.cast(conv_weight[2:3, channel : channel + CHANNEL_TILE], target_type=pl.FP32)
        w3 = pl.cast(conv_weight[3:4, channel : channel + CHANNEL_TILE], target_type=pl.FP32)
        r0 = pl.cast(raw[0:1, channel : channel + CHANNEL_TILE], target_type=pl.FP32)
        r1 = pl.cast(raw[1:2, channel : channel + CHANNEL_TILE], target_type=pl.FP32)
        r2 = pl.cast(raw[2:3, channel : channel + CHANNEL_TILE], target_type=pl.FP32)
        r3 = pl.cast(raw[3:4, channel : channel + CHANNEL_TILE], target_type=pl.FP32)
        c0 = pl.mul(r0, w3)
        c1 = pl.add(pl.mul(r0, w2), pl.mul(r1, w3))
        c2 = pl.add(pl.add(pl.mul(r0, w1), pl.mul(r1, w2)), pl.mul(r2, w3))
        c3 = pl.add(
            pl.add(pl.mul(r0, w0), pl.mul(r1, w1)),
            pl.add(pl.mul(r2, w2), pl.mul(r3, w3)),
        )
        f0 = pl.cast(pl.cast(c0, target_type=pl.BF16, mode="rint"), target_type=pl.FP32)
        f1 = pl.cast(pl.cast(c1, target_type=pl.BF16, mode="rint"), target_type=pl.FP32)
        f2 = pl.cast(pl.cast(c2, target_type=pl.BF16, mode="rint"), target_type=pl.FP32)
        f3 = pl.cast(pl.cast(c3, target_type=pl.BF16, mode="rint"), target_type=pl.FP32)
        out[0:1, channel : channel + CHANNEL_TILE] = pl.cast(
            pl.mul(f0, pl.recip(pl.add(pl.exp(pl.neg(f0)), 1.0))), target_type=pl.BF16, mode="rint"
        )
        out[1:2, channel : channel + CHANNEL_TILE] = pl.cast(
            pl.mul(f1, pl.recip(pl.add(pl.exp(pl.neg(f1)), 1.0))), target_type=pl.BF16, mode="rint"
        )
        out[2:3, channel : channel + CHANNEL_TILE] = pl.cast(
            pl.mul(f2, pl.recip(pl.add(pl.exp(pl.neg(f2)), 1.0))), target_type=pl.BF16, mode="rint"
        )
        out[3:4, channel : channel + CHANNEL_TILE] = pl.cast(
            pl.mul(f3, pl.recip(pl.add(pl.exp(pl.neg(f3)), 1.0))), target_type=pl.BF16, mode="rint"
        )
        out[4:TOKENS_PAD, channel : channel + CHANNEL_TILE] = raw[
            4:TOKENS_PAD, channel : channel + CHANNEL_TILE
        ]
    return out


@pl.jit.inline
def conv_silu_decode(
    raw_history: pl.Tensor[[TOKENS_PAD, QKV], pl.BF16],
    conv_weight: pl.Tensor[[4, QKV], pl.BF16],
    out: pl.Tensor[[TOKENS_PAD, QKV], pl.BF16],
):
    for block in pl.spmd(QKV // CHANNEL_TILE, name_hint="delta_decode_conv"):
        channel = block * CHANNEL_TILE
        acc = pl.full([1, CHANNEL_TILE], dtype=pl.FP32, value=0.0)
        for tap in pl.range(4):
            weight = pl.cast(
                conv_weight[tap : tap + 1, channel : channel + CHANNEL_TILE], target_type=pl.FP32
            )
            value = pl.cast(raw_history[tap : tap + 1, channel : channel + CHANNEL_TILE], target_type=pl.FP32)
            acc = pl.add(acc, pl.mul(value, weight))
        rounded = pl.cast(pl.cast(acc, target_type=pl.BF16, mode="rint"), target_type=pl.FP32)
        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(rounded)), 1.0))
        out[0:1, channel : channel + CHANNEL_TILE] = pl.cast(
            pl.mul(rounded, sigmoid), target_type=pl.BF16, mode="rint"
        )
        out[1:TOKENS_PAD, channel : channel + CHANNEL_TILE] = raw_history[
            1:TOKENS_PAD, channel : channel + CHANNEL_TILE
        ]
    return out


@pl.jit.inline
def delta_gates(
    a: pl.Tensor[[TOKENS_PAD, 128], pl.BF16],
    b: pl.Tensor[[TOKENS_PAD, 128], pl.BF16],
    a_log: pl.Tensor[[1, 128], pl.BF16],
    dt_bias: pl.Tensor[[1, 128], pl.BF16],
    beta: pl.Tensor[[TOKENS_PAD, 128], pl.FP32],
    decay: pl.Tensor[[TOKENS_PAD, 128], pl.FP32],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="delta_gates"):
        a_float = pl.cast(a, target_type=pl.FP32)
        b_float = pl.cast(b, target_type=pl.FP32)
        a_plus_dt = pl.col_expand_add(a_float, pl.cast(dt_bias, target_type=pl.FP32))
        softplus = pl.log(pl.add(pl.exp(a_plus_dt), 1.0))
        rate = pl.exp(pl.cast(a_log, target_type=pl.FP32))
        negative_log_decay = pl.neg(pl.col_expand_mul(softplus, rate))
        decay[:, :] = pl.exp(negative_log_decay)
        beta[:, :] = pl.recip(pl.add(pl.exp(pl.neg(b_float)), 1.0))
    return beta, decay


@pl.jit.inline
def delta_qk_norm(
    activated: pl.Tensor[[TOKENS_PAD, QKV], pl.BF16],
    query: pl.Tensor[[TOKENS_PAD, QK_HEADS * HEAD_DIM], pl.FP32],
    key: pl.Tensor[[TOKENS_PAD, QK_HEADS * HEAD_DIM], pl.FP32],
):
    for token in pl.spmd(TOKENS_PAD, name_hint="delta_qk_norm"):
        q = pl.cast(
            pl.reshape(activated[token : token + 1, 0:2048], [QK_HEADS, HEAD_DIM]), target_type=pl.FP32
        )
        k = pl.cast(
            pl.reshape(activated[token : token + 1, 2048:4096], [QK_HEADS, HEAD_DIM]), target_type=pl.FP32
        )
        q_inv = pl.rsqrt(pl.add(pl.row_sum(pl.mul(q, q)), L2_EPS))
        k_inv = pl.rsqrt(pl.add(pl.row_sum(pl.mul(k, k)), L2_EPS))
        q_norm = pl.mul(pl.row_expand_mul(q, q_inv), HEAD_DIM**-0.5)
        k_norm = pl.row_expand_mul(k, k_inv)
        query[token : token + 1, :] = pl.reshape(q_norm, [1, QK_HEADS * HEAD_DIM])
        key[token : token + 1, :] = pl.reshape(k_norm, [1, QK_HEADS * HEAD_DIM])
    return query, key


@pl.jit.inline
def delta_step(
    activated: pl.Tensor[[TOKENS_PAD, QKV], pl.BF16],
    query: pl.Tensor[[TOKENS_PAD, QK_HEADS * HEAD_DIM], pl.FP32],
    key: pl.Tensor[[TOKENS_PAD, QK_HEADS * HEAD_DIM], pl.FP32],
    beta: pl.Tensor[[TOKENS_PAD, 128], pl.FP32],
    decay: pl.Tensor[[TOKENS_PAD, 128], pl.FP32],
    previous: pl.Tensor[[VALUE_HEADS, HEAD_DIM, HEAD_DIM], pl.FP32],
    token_index: pl.Scalar[pl.INDEX],
    state: pl.Tensor[[VALUE_HEADS, HEAD_DIM, HEAD_DIM], pl.FP32],
    core: pl.Tensor[[TOKENS_PAD, VALUE_WIDTH], pl.BF16],
):
    previous_2d = pl.reshape(previous, [VALUE_HEADS * HEAD_DIM, HEAD_DIM])
    state_2d = pl.reshape(state, [VALUE_HEADS * HEAD_DIM, HEAD_DIM])
    for task in pl.spmd(VALUE_HEADS * (HEAD_DIM // VALUE_TILE), name_hint="delta_recurrent"):
        head = task // (HEAD_DIM // VALUE_TILE)
        v0 = (task % (HEAD_DIM // VALUE_TILE)) * VALUE_TILE
        qk_head = head // (VALUE_HEADS // QK_HEADS)
        qk_col = qk_head * HEAD_DIM
        q_row = query[token_index : token_index + 1, qk_col : qk_col + HEAD_DIM]
        k_row = key[token_index : token_index + 1, qk_col : qk_col + HEAD_DIM]
        value_col = 4096 + head * HEAD_DIM + v0
        value_row = pl.cast(
            activated[token_index : token_index + 1, value_col : value_col + VALUE_TILE], target_type=pl.FP32
        )
        state_row = head * HEAD_DIM
        old = previous_2d[state_row : state_row + HEAD_DIM, v0 : v0 + VALUE_TILE]
        decay_scalar = pl.read(decay, [token_index, head])
        beta_scalar = pl.read(beta, [token_index, head])
        decayed = pl.mul(old, decay_scalar)
        k_column = pl.mul(
            pl.row_sum(
                pl.transpose(
                    pl.col_expand_mul(pl.full([COL_REP, HEAD_DIM], dtype=pl.FP32, value=1.0), k_row), 0, 1
                )
            ),
            1.0 / COL_REP,
        )
        memory = pl.col_sum(pl.row_expand_mul(decayed, k_column))
        correction = pl.mul(pl.sub(value_row, memory), beta_scalar)
        update = pl.col_expand_mul(
            pl.row_expand_mul(pl.full([HEAD_DIM, VALUE_TILE], dtype=pl.FP32, value=1.0), k_column),
            correction,
        )
        updated = pl.add(decayed, update)
        state_2d[state_row : state_row + HEAD_DIM, v0 : v0 + VALUE_TILE] = updated
        result = pl.matmul(q_row, updated, out_dtype=pl.FP32)
        output_col = head * HEAD_DIM + v0
        core[token_index : token_index + 1, output_col : output_col + VALUE_TILE] = pl.set_validshape(
            pl.cast(result, target_type=pl.BF16, mode="rint"), 1, VALUE_TILE
        )
    return state, core


@pl.jit.inline
def delta_gated_rms(
    core: pl.Tensor[[TOKENS_PAD, VALUE_WIDTH], pl.BF16],
    gate: pl.Tensor[[TOKENS_PAD, VALUE_WIDTH], pl.BF16],
    weight: pl.Tensor[[1, HEAD_DIM], pl.BF16],
    out: pl.Tensor[[TOKENS_PAD, VALUE_WIDTH], pl.BF16],
):
    core_rows = pl.reshape(core, [TOKENS_PAD * VALUE_HEADS, HEAD_DIM])
    gate_rows = pl.reshape(gate, [TOKENS_PAD * VALUE_HEADS, HEAD_DIM])
    out_rows = pl.reshape(out, [TOKENS_PAD * VALUE_HEADS, HEAD_DIM])
    for block in pl.spmd(TOKENS_PAD * VALUE_HEADS // 16, name_hint="delta_gated_rms"):
        row = block * 16
        x = pl.cast(core_rows[row : row + 16, :], target_type=pl.FP32)
        norm = pl.rsqrt(pl.add(pl.mul(pl.row_sum(pl.mul(x, x)), 1.0 / HEAD_DIM), L2_EPS))
        normalized = pl.cast(pl.row_expand_mul(x, norm), target_type=pl.BF16, mode="rint")
        scaled = pl.cast(
            pl.col_expand_mul(
                pl.cast(normalized, target_type=pl.FP32),
                pl.cast(weight, target_type=pl.FP32),
            ),
            target_type=pl.BF16,
            mode="rint",
        )
        gate_float = pl.cast(gate_rows[row : row + 16, :], target_type=pl.FP32)
        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_float)), 1.0))
        gated = pl.mul(pl.cast(scaled, target_type=pl.FP32), pl.mul(gate_float, sigmoid))
        out_rows[row : row + 16, :] = pl.cast(gated, target_type=pl.BF16, mode="rint")
    return out
