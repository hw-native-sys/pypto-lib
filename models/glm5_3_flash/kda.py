# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GLM-5.3-Flash KDA projections, convolution, recurrence and output."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from dataclasses import dataclass

import pypto.language as pl
import torch
import torch.nn.functional as F

from models.glm5_3_flash.config import (
    FLASH, KDA_CONV_K, KDA_DIM, KDA_GATE_LOWER_BOUND, LOCAL_KDA_H, LOCAL_KDA_QKV_DIM, D, Glm53FlashConfig,
)
from models.glm5_3_flash.elementwise import (
    accurate_divide, accurate_exp, accurate_row_sum, accurate_rsqrt, accurate_sigmoid, accurate_silu,
    accurate_sqrt,
)
from models.glm5_3_flash.linear import bf16_linear, golden_bf16_linear
from models.glm5_3_flash.rmsnorm import l2norm, reference_l2norm, rms_norm_gated
from models.glm5_3_flash.utils import _assert_close, _randn

B_DYN = pl.dynamic("GLM53_B_DYN")
KDA_STATE_DYN = pl.dynamic("GLM53_KDA_STATE_DYN")
Q_START_DYN = pl.dynamic("GLM53_Q_START_DYN")
T_DYN = pl.dynamic("GLM53_T_DYN")

CONV_DIM = 3 * LOCAL_KDA_QKV_DIM
STATE_LEN = KDA_CONV_K - 1
CONV_T_TILE = 16  # tokens per bulk task
C_TILE = 128  # Bounds the corrected SiLU working set alongside the FP32 tap sum
DECODE_C_TILE = 128  # channels per decode convolution task
_PREFILL_LENGTHS = {"ragged": [5, 1, 9, 2, 0], "single": [1], "long": [40]}
CONV_POOL_ROWS = 6

PROJECTION_T_TILE = 16  # tokens per task
BETA_PAD = ((LOCAL_KDA_H + 15) // 16) * 16
N_TILE = min(256, LOCAL_KDA_QKV_DIM)

EPS = FLASH.rms_norm_eps
KDA_DIM_INV = 1.0 / KDA_DIM
OUTPUT_T_TILE = 16  # tokens per task; the norm pass handles T_TILE * LOCAL_KDA_H rows
OUTPUT_NORM_ROWS_TILE = 8
D_TILE = 128  # output columns per projection task

DECODE_L2_EPS = 1e-6
DECODE_Q_SCALE = KDA_DIM**-0.5
DECODE_V_TILE = 16  # value rows per state-update task
DECODE_NORM_ROWS_TILE = 16  # (token, head) rows per normalisation task
COL_REP_TILE = 16  # replication width used to turn a row into a legal column
DECODE_POOL_ROWS = 6

CHUNK_TILE = 16
DECAY_CAP = 80.0  # FP32 exp argument ceiling
DECAY_TILE = CHUNK_TILE  # one local decay pivot per chunk
PREFILL_L2_EPS = 1e-6
PREFILL_Q_SCALE = KDA_DIM**-0.5
PREFILL_V_TILE = 64  # value columns per scan task: the carried state slice is [128, 64] FP32
PREFILL_POOL_ROWS = 6
_LENGTHS = {"ragged": [140, 7, 64], "single": [3], "aligned": [64], "zerostate": [3], "nobeta": [70]}


@pl.jit.inline
def kda_conv_prefill(
    mixed_qkv: pl.Tensor[[T_DYN, CONV_DIM], pl.BF16],
    weight: pl.Tensor[[KDA_CONV_K, CONV_DIM], pl.BF16],
    conv_state: pl.InOut[pl.Tensor[[KDA_STATE_DYN, STATE_LEN, CONV_DIM], pl.BF16]],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_rows: pl.Tensor[[B_DYN], pl.INT32],
    query: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    key: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    value: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
):
    t_dim = pl.tensor.dim(mixed_qkv, 0)
    requests = pl.tensor.dim(query_start_loc, 0) - 1

    # --- pass 1: flat sweep of every token at least STATE_LEN rows into the stream ---
    # Rows below STATE_LEN, and the first STATE_LEN rows of every later request, are
    # recomputed in pass 2; here they would read the preceding request's tail.
    # A batch shorter than the window has no bulk rows, and a zero-task spmd is not
    # dispatchable, so the task count floors at one and the body guards itself. Pass 2
    # covers every row in that case.
    bulk_tiles = pl.max((pl.max(t_dim - STATE_LEN, 0) + CONV_T_TILE - 1) // CONV_T_TILE, 1)
    # A store whose source is a tile reshaped to two leading unit axes lowers to a
    # tile with a single-element row, which ptoas rejects on its 32-byte row
    # alignment. Addressing the pool through a flat view of the same buffer keeps
    # the stored tile the shape the arithmetic already produced.

    v_flat = pl.reshape(value, [t_dim, LOCAL_KDA_QKV_DIM])
    k_flat = pl.reshape(key, [t_dim, LOCAL_KDA_QKV_DIM])
    q_flat = pl.reshape(query, [t_dim, LOCAL_KDA_QKV_DIM])
    with pl.spmd(bulk_tiles * (CONV_DIM // C_TILE), name_hint="kda_conv_bulk") as bulk_tid:
        task = pl.tile.get_block_idx()
        lanes = CONV_DIM // C_TILE
        p0 = STATE_LEN + (task // lanes) * CONV_T_TILE
        c0 = (task % lanes) * C_TILE
        rows = pl.min(CONV_T_TILE, t_dim - p0)
        if rows > 0:
            acc = pl.full([CONV_T_TILE, C_TILE], dtype=pl.FP32, value=0.0)
            for j in pl.unroll(KDA_CONV_K):
                tap = pl.slice(mixed_qkv, [CONV_T_TILE, C_TILE], [p0 - STATE_LEN + j, c0], valid_shape=[rows, C_TILE])
                acc_cast = pl.cast(weight[j:j + 1, c0:c0 + C_TILE], pl.FP32)
                acc_col_expand_mul = pl.col_expand_mul(pl.cast(tap, pl.FP32), acc_cast)
                acc = pl.add(acc, acc_col_expand_mul)
            activation_input = pl.create_tensor([CONV_T_TILE, C_TILE], dtype=pl.FP32)
            activation_input[:, :] = acc
            activated = accurate_silu(activation_input, CONV_T_TILE, C_TILE)
            out_tile = pl.set_validshape(pl.cast(activated, pl.BF16, mode="rint"), rows, C_TILE)
            if c0 < LOCAL_KDA_QKV_DIM:
                q_flat[p0 : p0 + CONV_T_TILE, c0 : c0 + C_TILE] = out_tile
            elif c0 < 2 * LOCAL_KDA_QKV_DIM:
                kc = c0 - LOCAL_KDA_QKV_DIM
                k_flat[p0 : p0 + CONV_T_TILE, kc : kc + C_TILE] = out_tile
            else:
                vc = c0 - 2 * LOCAL_KDA_QKV_DIM
                v_flat[p0 : p0 + CONV_T_TILE, vc : vc + C_TILE] = out_tile

    # --- pass 2: the first STATE_LEN rows of every request, plus the state update ---
    cs2 = pl.reshape(conv_state, [pl.tensor.dim(conv_state, 0) * STATE_LEN, CONV_DIM])
    with pl.spmd(requests * (CONV_DIM // C_TILE), name_hint="kda_conv_head", deps=[bulk_tid]):
        task = pl.tile.get_block_idx()
        lanes = CONV_DIM // C_TILE
        b = task // lanes
        c0 = (task % lanes) * C_TILE
        s = pl.cast(pl.read(query_start_loc, [b]), pl.INDEX)
        e = pl.cast(pl.read(query_start_loc, [b + 1]), pl.INDEX)
        row = pl.cast(pl.read(state_rows, [b]), pl.INDEX)
        srow = row * STATE_LEN
        length = e - s
        # An empty request is legal: query_start_loc is cumulative, and equal adjacent
        # offsets mean a request contributed no tokens -- the reference skips those.
        # The row masks below already write nothing in that case, but the token reads
        # still happen, and for a trailing empty request s equals the token count, so
        # they would run past the end of the packed stream. Skip the body instead.
        if length > 0:
            # full[k] is conv_state[row, k] for k < STATE_LEN and mixed_qkv[s + k - STATE_LEN]
            # otherwise. Written straight through rather than looped: a pl iterator's index
            # is a Scalar, so it cannot drive the trace-time choice between the two sources.
            st0 = pl.cast(cs2[srow + 0:srow + 0 + 1, c0:c0 + C_TILE], pl.FP32)
            st1 = pl.cast(cs2[srow + 1:srow + 1 + 1, c0:c0 + C_TILE], pl.FP32)
            st2 = pl.cast(cs2[srow + 2:srow + 2 + 1, c0:c0 + C_TILE], pl.FP32)
            # k - STATE_LEN <= u < length whenever the row below is written, so these reads
            # stay inside the request.
            x0 = pl.cast(mixed_qkv[s:s + 1, c0:c0 + C_TILE], pl.FP32)
            second = pl.min(s + 1, e - 1)
            x1 = pl.cast(mixed_qkv[second:second + 1, c0:c0 + C_TILE], pl.FP32)
            third = pl.min(s + 2, e - 1)
            x2 = pl.cast(mixed_qkv[third:third + 1, c0:c0 + C_TILE], pl.FP32)
            w0 = pl.cast(weight[0:1, c0 : c0 + C_TILE], pl.FP32)
            w1 = pl.cast(weight[1:2, c0 : c0 + C_TILE], pl.FP32)
            w2 = pl.cast(weight[2:3, c0 : c0 + C_TILE], pl.FP32)
            w3 = pl.cast(weight[3:4, c0 : c0 + C_TILE], pl.FP32)

            # Match the bulk/reference's left-to-right FP32 tap sum. Pairwise
            # sums can cross a BF16 rounding tie when a token starts a new chunk.
            a00 = pl.mul(st0, w0)
            a01 = pl.mul(st1, w1)
            a02 = pl.mul(st2, w2)
            a03 = pl.mul(x0, w3)
            acc0 = pl.add(a00, a01)
            acc0 = pl.add(acc0, a02)
            acc0 = pl.add(acc0, a03)
            a10 = pl.mul(st1, w0)
            a11 = pl.mul(st2, w1)
            a12 = pl.mul(x0, w2)
            a13 = pl.mul(x1, w3)
            acc1 = pl.add(a10, a11)
            acc1 = pl.add(acc1, a12)
            acc1 = pl.add(acc1, a13)
            a20 = pl.mul(st2, w0)
            a21 = pl.mul(x0, w1)
            a22 = pl.mul(x1, w2)
            a23 = pl.mul(x2, w3)
            acc2 = pl.add(a20, a21)
            acc2 = pl.add(acc2, a22)
            acc2 = pl.add(acc2, a23)

            # SiLU and the masked stores, one row at a time. A request shorter than the
            # window leaves the later rows with zero valid rows, so nothing is written.
            input0 = pl.create_tensor([1, C_TILE], dtype=pl.FP32)
            input1 = pl.create_tensor([1, C_TILE], dtype=pl.FP32)
            input2 = pl.create_tensor([1, C_TILE], dtype=pl.FP32)
            input0[:, :] = acc0
            input1[:, :] = acc1
            input2[:, :] = acc2
            act0 = accurate_silu(input0, 1, C_TILE)
            act1 = accurate_silu(input1, 1, C_TILE)
            act2 = accurate_silu(input2, 1, C_TILE)
            v0 = pl.min(pl.max(length, 0), 1)
            v1 = pl.min(pl.max(length - 1, 0), 1)
            v2 = pl.min(pl.max(length - 2, 0), 1)
            o0 = pl.set_validshape(pl.cast(act0, pl.BF16, mode="rint"), v0, C_TILE)
            o1 = pl.set_validshape(pl.cast(act1, pl.BF16, mode="rint"), v1, C_TILE)
            o2 = pl.set_validshape(pl.cast(act2, pl.BF16, mode="rint"), v2, C_TILE)
            # The destination is chosen by writing in each branch, not by rebinding a name
            # to a different tensor: an IfStmt's in-place return must yield one backing
            # value, and three different tensors cannot be unified into it.
            if c0 < LOCAL_KDA_QKV_DIM:
                q_flat[s : s + 1, c0 : c0 + C_TILE] = o0
                q_flat[s + 1 : s + 2, c0 : c0 + C_TILE] = o1
                q_flat[s + 2 : s + 3, c0 : c0 + C_TILE] = o2
            elif c0 < 2 * LOCAL_KDA_QKV_DIM:
                kc0 = c0 - LOCAL_KDA_QKV_DIM
                k_flat[s : s + 1, kc0 : kc0 + C_TILE] = o0
                k_flat[s + 1 : s + 2, kc0 : kc0 + C_TILE] = o1
                k_flat[s + 2 : s + 3, kc0 : kc0 + C_TILE] = o2
            else:
                vc0 = c0 - 2 * LOCAL_KDA_QKV_DIM
                v_flat[s : s + 1, vc0 : vc0 + C_TILE] = o0
                v_flat[s + 1 : s + 2, vc0 : vc0 + C_TILE] = o1
                v_flat[s + 2 : s + 3, vc0 : vc0 + C_TILE] = o2

            for i in pl.unroll(STATE_LEN - 1):
                old_idx = pl.min(length + i, STATE_LEN - 1)
                new_idx = pl.max(e - STATE_LEN + i, s)
                keep_old = pl.min(pl.max(STATE_LEN - length - i, 0), 1)
                take_new = pl.min(pl.max(length + i - STATE_LEN + 1, 0), 1)
                old_row = cs2[srow + old_idx:srow + old_idx + 1, c0:c0 + C_TILE]
                new_row = mixed_qkv[new_idx:new_idx + 1, c0:c0 + C_TILE]
                cs2[srow + i : srow + i + 1, c0 : c0 + C_TILE] = pl.set_validshape(old_row, keep_old, C_TILE)
                cs2[srow + i : srow + i + 1, c0 : c0 + C_TILE] = pl.set_validshape(new_row, take_new, C_TILE)
            last = srow + STATE_LEN - 1
            final_row = pl.max(e - 1, s)
            cs2[last : last + 1, c0 : c0 + C_TILE] = mixed_qkv[final_row:final_row + 1, c0:c0 + C_TILE]

    return query, key, value, conv_state


@pl.jit.inline
def kda_conv_decode(
    mixed_qkv: pl.Tensor[[T_DYN, CONV_DIM], pl.BF16],
    weight: pl.Tensor[[KDA_CONV_K, CONV_DIM], pl.BF16],
    conv_state: pl.InOut[pl.Tensor[[KDA_STATE_DYN, STATE_LEN, CONV_DIM], pl.BF16]],
    state_rows: pl.Tensor[[T_DYN], pl.INT32],
    query: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    key: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    value: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
):
    t_dim = pl.tensor.dim(mixed_qkv, 0)
    cs2 = pl.reshape(conv_state, [pl.tensor.dim(conv_state, 0) * STATE_LEN, CONV_DIM])
    v_flat = pl.reshape(value, [t_dim, LOCAL_KDA_QKV_DIM])
    k_flat = pl.reshape(key, [t_dim, LOCAL_KDA_QKV_DIM])
    q_flat = pl.reshape(query, [t_dim, LOCAL_KDA_QKV_DIM])
    with pl.spmd(t_dim * (CONV_DIM // DECODE_C_TILE), name_hint="kda_conv_decode"):
        task = pl.tile.get_block_idx()
        lanes = CONV_DIM // DECODE_C_TILE
        t = task // lanes
        c0 = task % lanes * DECODE_C_TILE
        row = pl.cast(pl.read(state_rows, [t]), pl.INDEX)
        srow = row * STATE_LEN
        st0 = pl.cast(cs2[srow + 0:srow + 0 + 1, c0:c0 + DECODE_C_TILE], pl.FP32)
        st1 = pl.cast(cs2[srow + 1:srow + 1 + 1, c0:c0 + DECODE_C_TILE], pl.FP32)
        st2 = pl.cast(cs2[srow + 2:srow + 2 + 1, c0:c0 + DECODE_C_TILE], pl.FP32)
        xt = pl.cast(mixed_qkv[t:t + 1, c0:c0 + DECODE_C_TILE], pl.FP32)
        acc_fp32 = pl.cast(weight[0:1, c0 : c0 + DECODE_C_TILE], pl.FP32)
        acc_mul = pl.mul(st0, acc_fp32)
        acc_fp32_2 = pl.cast(weight[1:2, c0 : c0 + DECODE_C_TILE], pl.FP32)
        acc_mul_2 = pl.mul(st1, acc_fp32_2)
        acc_add = pl.add(acc_mul, acc_mul_2)
        acc_fp32_3 = pl.cast(weight[2:3, c0 : c0 + DECODE_C_TILE], pl.FP32)
        acc_mul_3 = pl.mul(st2, acc_fp32_3)
        acc_fp32_4 = pl.cast(weight[3:4, c0 : c0 + DECODE_C_TILE], pl.FP32)
        acc_mul_4 = pl.mul(xt, acc_fp32_4)
        acc_add_2 = pl.add(acc_add, acc_mul_3)
        acc = pl.add(acc_add_2, acc_mul_4)
        activation_arg_activated = pl.create_tensor([1, DECODE_C_TILE], dtype=pl.FP32)
        activation_arg_activated[:, :] = acc
        activated = accurate_silu(activation_arg_activated, 1, DECODE_C_TILE)
        out_tile = pl.cast(activated, pl.BF16, mode="rint")
        if c0 < LOCAL_KDA_QKV_DIM:
            q_flat[t : t + 1, c0 : c0 + DECODE_C_TILE] = out_tile
        elif c0 < 2 * LOCAL_KDA_QKV_DIM:
            k_flat[t : t + 1, c0 - LOCAL_KDA_QKV_DIM : c0 - LOCAL_KDA_QKV_DIM + DECODE_C_TILE] = out_tile
        else:
            v_flat[t:t + 1, c0 - 2 * LOCAL_KDA_QKV_DIM:c0 - 2 * LOCAL_KDA_QKV_DIM + DECODE_C_TILE] = out_tile
        cs2[srow : srow + 1, c0 : c0 + DECODE_C_TILE] = pl.cast(st1, pl.BF16, mode="rint")
        cs2[srow + 1 : srow + 2, c0 : c0 + DECODE_C_TILE] = pl.cast(st2, pl.BF16, mode="rint")
        cs2[srow + 2 : srow + 3, c0 : c0 + DECODE_C_TILE] = mixed_qkv[t:t + 1, c0:c0 + DECODE_C_TILE]
    return (query, key, value, conv_state)


@pl.jit
def kda_conv_prefill_test(
    mixed_qkv: pl.Tensor[[T_DYN, CONV_DIM], pl.BF16],
    weight: pl.Tensor[[KDA_CONV_K, CONV_DIM], pl.BF16],
    conv_state: pl.InOut[pl.Tensor[[KDA_STATE_DYN, STATE_LEN, CONV_DIM], pl.BF16]],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_rows: pl.Tensor[[B_DYN], pl.INT32],
    query: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    key: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    value: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
):
    (query, key, value, conv_state) = kda_conv_prefill(
        mixed_qkv, weight, conv_state, query_start_loc, state_rows, query, key, value,
    )
    return query, key, value, conv_state


@pl.jit
def kda_conv_decode_test(
    mixed_qkv: pl.Tensor[[T_DYN, CONV_DIM], pl.BF16],
    weight: pl.Tensor[[KDA_CONV_K, CONV_DIM], pl.BF16],
    conv_state: pl.InOut[pl.Tensor[[KDA_STATE_DYN, STATE_LEN, CONV_DIM], pl.BF16]],
    state_rows: pl.Tensor[[T_DYN], pl.INT32],
    query: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    key: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    value: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
):
    (query, key, value, conv_state) = kda_conv_decode(mixed_qkv, weight, conv_state, state_rows, query, key, value)
    return query, key, value, conv_state


@pl.jit.inline
def kda_projection(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    w_q: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_k: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_v: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_f_a: pl.Tensor[[KDA_DIM, D], pl.BF16],
    w_f_b: pl.Tensor[[LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    dt_bias: pl.Tensor[[LOCAL_KDA_QKV_DIM], pl.FP32],
    a_log: pl.Tensor[[LOCAL_KDA_H], pl.FP32],
    w_b: pl.Tensor[[LOCAL_KDA_H, D], pl.BF16],
    w_g_a: pl.Tensor[[KDA_DIM, D], pl.BF16],
    w_g_b: pl.Tensor[[LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    mixed_qkv: pl.Tensor[[T_DYN, 3 * LOCAL_KDA_QKV_DIM], pl.BF16],
    decay: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.FP32],
    beta: pl.Tensor[[T_DYN, BETA_PAD], pl.FP32],
    out_gate: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
):
    t_dim = pl.tensor.dim(x, 0)
    tiles = (t_dim + PROJECTION_T_TILE - 1) // PROJECTION_T_TILE
    t_pad = tiles * PROJECTION_T_TILE
    w_b_pad = pl.create_tensor([BETA_PAD, D], dtype=pl.BF16)
    with pl.spmd(1, name_hint="kda_projection_b_pad") as pad_tid:
        zero = pl.tile.get_block_idx() * 0  # single task, block index zero
        for r in pl.unroll(BETA_PAD):
            if r < LOCAL_KDA_H + zero:
                w_b_pad[r : r + 1, 0:D] = w_b[r:r + 1, 0:D]
            else:
                w_b_pad[r : r + 1, 0:D] = pl.full([1, D], dtype=pl.BF16, value=0.0)

    with pl.spmd(tiles * (LOCAL_KDA_QKV_DIM // N_TILE), name_hint="kda_projection_qkv"):
        task = pl.tile.get_block_idx()
        t0 = task // (LOCAL_KDA_QKV_DIM // N_TILE) * PROJECTION_T_TILE
        n0 = task % (LOCAL_KDA_QKV_DIM // N_TILE) * N_TILE
        rows = pl.min(PROJECTION_T_TILE, t_dim - t0)
        acc_q = bf16_linear(x, w_q, t0, n0, rows, pl.cast(N_TILE, pl.INDEX), PROJECTION_T_TILE, N_TILE, D, True)
        acc_q_bf16 = pl.cast(acc_q, pl.BF16, mode="rint")
        mixed_qkv[t0 : t0 + PROJECTION_T_TILE, n0 : n0 + N_TILE] = pl.set_validshape(acc_q_bf16, rows, N_TILE)
        acc_k = bf16_linear(x, w_k, t0, n0, rows, pl.cast(N_TILE, pl.INDEX), PROJECTION_T_TILE, N_TILE, D, True)
        acc_k_bf16 = pl.cast(acc_k, pl.BF16, mode="rint")
        mixed_qkv[t0:t0 + PROJECTION_T_TILE, LOCAL_KDA_QKV_DIM + n0:LOCAL_KDA_QKV_DIM + n0 + N_TILE] = pl.set_validshape(acc_k_bf16, rows, N_TILE)
        acc_v = bf16_linear(x, w_v, t0, n0, rows, pl.cast(N_TILE, pl.INDEX), PROJECTION_T_TILE, N_TILE, D, True)
        acc_v_bf16 = pl.cast(acc_v, pl.BF16, mode="rint")
        mixed_qkv[t0:t0 + PROJECTION_T_TILE, 2 * LOCAL_KDA_QKV_DIM + n0:2 * LOCAL_KDA_QKV_DIM + n0 + N_TILE] = pl.set_validshape(acc_v_bf16, rows, N_TILE)
    b_raw_gm = pl.create_tensor([t_pad, BETA_PAD], dtype=pl.BF16)
    g_a_gm = pl.create_tensor([t_pad, KDA_DIM], dtype=pl.BF16)
    f_a_gm = pl.create_tensor([t_pad, KDA_DIM], dtype=pl.BF16)
    with pl.spmd(tiles, name_hint="kda_projection_bottleneck", deps=[pad_tid]) as bottleneck_tid:
        t0 = pl.tile.get_block_idx() * PROJECTION_T_TILE
        rows = pl.min(PROJECTION_T_TILE, t_dim - t0)
        f_a_index = pl.cast(0, pl.INDEX)
        KDA_DIM_index = pl.cast(KDA_DIM, pl.INDEX)
        f_a = bf16_linear(x, w_f_a, t0, f_a_index, rows, KDA_DIM_index, PROJECTION_T_TILE, KDA_DIM, D, True)
        f_a_gm[t0 : t0 + PROJECTION_T_TILE, :] = pl.set_validshape(pl.cast(f_a, pl.BF16, mode="rint"), rows, KDA_DIM)
        g_a_index = pl.cast(0, pl.INDEX)
        KDA_DIM_index_2 = pl.cast(KDA_DIM, pl.INDEX)
        g_a = bf16_linear(x, w_g_a, t0, g_a_index, rows, KDA_DIM_index_2, PROJECTION_T_TILE, KDA_DIM, D, True)
        g_a_gm[t0 : t0 + PROJECTION_T_TILE, :] = pl.set_validshape(pl.cast(g_a, pl.BF16, mode="rint"), rows, KDA_DIM)
        b_raw_index = pl.cast(0, pl.INDEX)
        BETA_PAD_index = pl.cast(BETA_PAD, pl.INDEX)
        b_raw = bf16_linear(x, w_b_pad, t0, b_raw_index, rows, BETA_PAD_index, PROJECTION_T_TILE, BETA_PAD, D, True)
        b_raw_bf16 = pl.cast(b_raw, pl.BF16, mode="rint")
        b_raw_gm[t0 : t0 + PROJECTION_T_TILE, :] = pl.set_validshape(b_raw_bf16, rows, BETA_PAD)
    out_gate_flat = pl.reshape(out_gate, [t_dim, LOCAL_KDA_QKV_DIM])
    g_raw_gm = pl.create_tensor([t_pad, LOCAL_KDA_QKV_DIM], dtype=pl.BF16)
    with pl.spmd(tiles * (LOCAL_KDA_QKV_DIM // N_TILE), name_hint="kda_projection_output", deps=[bottleneck_tid]) as output_tid:
        task = pl.tile.get_block_idx()
        t0 = task // (LOCAL_KDA_QKV_DIM // N_TILE) * PROJECTION_T_TILE
        n0 = task % (LOCAL_KDA_QKV_DIM // N_TILE) * N_TILE
        rows = pl.min(PROJECTION_T_TILE, t_dim - t0)
        N_TILE_index = pl.cast(N_TILE, pl.INDEX)
        g_raw = bf16_linear(f_a_gm, w_f_b, t0, n0, rows, N_TILE_index, PROJECTION_T_TILE, N_TILE, KDA_DIM, True)
        g_raw_bf16 = pl.cast(g_raw, pl.BF16, mode="rint")
        g_raw_gm[t0 : t0 + PROJECTION_T_TILE, n0 : n0 + N_TILE] = pl.set_validshape(g_raw_bf16, rows, N_TILE)
        N_TILE_index_2 = pl.cast(N_TILE, pl.INDEX)
        out_raw = bf16_linear(g_a_gm, w_g_b, t0, n0, rows, N_TILE_index_2, PROJECTION_T_TILE, N_TILE, KDA_DIM, True)
        out_raw_bf16 = pl.cast(out_raw, pl.BF16, mode="rint")
        out_gate_flat[t0 : t0 + PROJECTION_T_TILE, n0 : n0 + N_TILE] = pl.set_validshape(out_raw_bf16, rows, N_TILE)
    dt_bias_row = pl.reshape(dt_bias, [1, LOCAL_KDA_QKV_DIM])
    decay_flat = pl.reshape(decay, [t_dim, LOCAL_KDA_QKV_DIM])
    with pl.spmd((t_dim + 3) // 4 * (LOCAL_KDA_QKV_DIM // KDA_DIM), name_hint="kda_projection_decay", deps=[output_tid]):
        task = pl.tile.get_block_idx()
        t0 = task // (LOCAL_KDA_QKV_DIM // KDA_DIM) * 4
        n0 = task % (LOCAL_KDA_QKV_DIM // KDA_DIM) * KDA_DIM
        rows = pl.min(4, t_dim - t0)
        g_raw_gm_tile = pl.slice(g_raw_gm, [4, KDA_DIM], [t0, n0], valid_shape=[rows, KDA_DIM])
        g_part = pl.cast(g_raw_gm_tile, pl.FP32)
        biased = pl.col_expand_add(g_part, dt_bias_row[0:1, n0 : n0 + KDA_DIM])
        a_fill = pl.create_tensor([4, KDA_DIM], dtype=pl.FP32)
        a_fill[:, :] = pl.add(pl.mul(biased, 0.0), pl.read(a_log, [n0 // KDA_DIM]))
        rate = accurate_exp(a_fill)
        scaled = pl.create_tensor([4, KDA_DIM], dtype=pl.FP32)
        scaled[:, :] = pl.mul(biased, rate)
        gate_h = accurate_sigmoid(scaled, 4, KDA_DIM)
        part_mul = pl.mul(gate_h, KDA_GATE_LOWER_BOUND)
        decay_flat[t0 : t0 + 4, n0 : n0 + KDA_DIM] = pl.set_validshape(part_mul, rows, KDA_DIM)
    with pl.spmd((t_dim + 7) // 8, name_hint="kda_projection_beta", deps=[bottleneck_tid]):
        t0 = pl.tile.get_block_idx() * 8
        rows = pl.min(8, t_dim - t0)
        logits = pl.create_tensor([8, BETA_PAD], dtype=pl.FP32)
        b_raw_gm_tile = pl.slice(b_raw_gm, [8, BETA_PAD], [t0, 0], valid_shape=[rows, BETA_PAD])
        logits[:, :] = pl.cast(b_raw_gm_tile, pl.FP32)
        sigmoid = accurate_sigmoid(logits, 8, BETA_PAD)
        beta[t0 : t0 + 8, :] = pl.set_validshape(sigmoid, rows, BETA_PAD)
    return (mixed_qkv, decay, beta, out_gate)


@pl.jit
def kda_projection_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    w_q: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_k: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_v: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_f_a: pl.Tensor[[KDA_DIM, D], pl.BF16],
    w_f_b: pl.Tensor[[LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    dt_bias: pl.Tensor[[LOCAL_KDA_QKV_DIM], pl.FP32],
    a_log: pl.Tensor[[LOCAL_KDA_H], pl.FP32],
    w_b: pl.Tensor[[LOCAL_KDA_H, D], pl.BF16],
    w_g_a: pl.Tensor[[KDA_DIM, D], pl.BF16],
    w_g_b: pl.Tensor[[LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    mixed_qkv: pl.Out[pl.Tensor[[T_DYN, 3 * LOCAL_KDA_QKV_DIM], pl.BF16]],
    decay: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.FP32]],
    beta: pl.Out[pl.Tensor[[T_DYN, BETA_PAD], pl.FP32]],
    out_gate: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
):
    (mixed_qkv, decay, beta, out_gate) = kda_projection(
        x, w_q, w_k, w_v, w_f_a, w_f_b, dt_bias, a_log, w_b, w_g_a, w_g_b, mixed_qkv, decay, beta, out_gate,
    )
    return mixed_qkv, decay, beta, out_gate


@pl.jit.inline
def kda_output(
    core_attn_out: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    norm_weight: pl.Tensor[[KDA_DIM], pl.BF16],
    out_gate: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    w_o: pl.Tensor[[D, LOCAL_KDA_QKV_DIM], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.FP32],
):
    t_dim = pl.tensor.dim(core_attn_out, 0)
    tiles = (t_dim + OUTPUT_T_TILE - 1) // OUTPUT_T_TILE
    gated = pl.create_tensor([t_dim, LOCAL_KDA_QKV_DIM], dtype=pl.BF16)
    gated_rows = pl.reshape(gated, [t_dim * LOCAL_KDA_H, KDA_DIM])
    weight_row = pl.reshape(norm_weight, [1, KDA_DIM])
    gate_rows = pl.reshape(out_gate, [t_dim * LOCAL_KDA_H, KDA_DIM])
    core_rows = pl.reshape(core_attn_out, [t_dim * LOCAL_KDA_H, KDA_DIM])
    with pl.spmd((t_dim * LOCAL_KDA_H + OUTPUT_NORM_ROWS_TILE - 1) // OUTPUT_NORM_ROWS_TILE, name_hint="kda_output_norm"):
        task = pl.tile.get_block_idx()
        r0 = task * OUTPUT_NORM_ROWS_TILE
        rows = pl.max(0, pl.min(OUTPUT_NORM_ROWS_TILE, t_dim * LOCAL_KDA_H - r0))
        core_rows_tile = pl.slice(core_rows, [OUTPUT_NORM_ROWS_TILE, KDA_DIM], [r0, 0], valid_shape=[rows, KDA_DIM])
        value = pl.cast(core_rows_tile, pl.FP32)
        square_buffer = pl.create_tensor([OUTPUT_NORM_ROWS_TILE, KDA_DIM], dtype=pl.FP32)
        square_buffer[:, :] = pl.mul(value, value)
        square_sum = accurate_row_sum(square_buffer)
        norm_argument = pl.create_tensor([1, OUTPUT_NORM_ROWS_TILE], dtype=pl.FP32)
        part_mul = pl.mul(square_sum, KDA_DIM_INV)
        part_add = pl.add(part_mul, EPS)
        norm_argument[:, :] = pl.reshape(part_add, [1, OUTPUT_NORM_ROWS_TILE])
        inverse = accurate_rsqrt(norm_argument)
        inv_rms = pl.reshape(inverse, [OUTPUT_NORM_ROWS_TILE, 1])
        normed_row_expand_mul = pl.row_expand_mul(value, inv_rms)
        normed_fp32 = pl.cast(weight_row[0:1, 0:KDA_DIM], pl.FP32)
        normed = pl.col_expand_mul(normed_row_expand_mul, normed_fp32)
        gate_rows_tile = pl.slice(gate_rows, [OUTPUT_NORM_ROWS_TILE, KDA_DIM], [r0, 0], valid_shape=[rows, KDA_DIM])
        gate = pl.cast(gate_rows_tile, pl.FP32)
        gate_buffer = pl.create_tensor([OUTPUT_NORM_ROWS_TILE, KDA_DIM], dtype=pl.FP32)
        gate_buffer[:, :] = gate
        gate_sigmoid = accurate_sigmoid(gate_buffer, OUTPUT_NORM_ROWS_TILE, KDA_DIM)
        activated = pl.mul(normed, gate_sigmoid)
        activated_bf16 = pl.cast(activated, pl.BF16, mode="rint")
        gated_rows[r0 : r0 + OUTPUT_NORM_ROWS_TILE, 0:KDA_DIM] = pl.set_validshape(activated_bf16, rows, KDA_DIM)
    with pl.spmd(tiles * (D // D_TILE), name_hint="kda_output_proj"):
        task = pl.tile.get_block_idx()
        t0 = task // (D // D_TILE) * OUTPUT_T_TILE
        d0 = task % (D // D_TILE) * D_TILE
        rows = pl.min(OUTPUT_T_TILE, t_dim - t0)
        D_TILE_index = pl.cast(D_TILE, pl.INDEX)
        partial = bf16_linear(gated, w_o, t0, d0, rows, D_TILE_index, OUTPUT_T_TILE, D_TILE, LOCAL_KDA_QKV_DIM, True)
        local_bf16 = pl.cast(partial, pl.BF16, mode="rint")
        exchange_value = pl.cast(local_bf16, pl.FP32)
        output[t0 : t0 + OUTPUT_T_TILE, d0 : d0 + D_TILE] = pl.set_validshape(exchange_value, rows, D_TILE)
    return output


@pl.jit
def kda_output_test(
    core_attn_out: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    norm_weight: pl.Tensor[[KDA_DIM], pl.BF16],
    out_gate: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    w_o: pl.Tensor[[D, LOCAL_KDA_QKV_DIM], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.FP32]],
):
    output = kda_output(core_attn_out, norm_weight, out_gate, w_o, output)
    return output


@pl.jit.inline
def decode_kda(
    query: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    key: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    value: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    decay: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.FP32],
    beta: pl.Tensor[[T_DYN, BETA_PAD], pl.FP32],
    recurrent_state: pl.InOut[pl.Tensor[[KDA_STATE_DYN, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]],
    state_rows: pl.Tensor[[T_DYN], pl.INT32],
    output: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
):
    t_dim = pl.tensor.dim(query, 0)
    lanes = KDA_DIM // DECODE_V_TILE
    tasks = t_dim * LOCAL_KDA_H * lanes
    rows_total = t_dim * LOCAL_KDA_H
    norm_tiles = (rows_total + DECODE_NORM_ROWS_TILE - 1) // DECODE_NORM_ROWS_TILE
    kn_gm = pl.create_tensor([norm_tiles * DECODE_NORM_ROWS_TILE, KDA_DIM], dtype=pl.FP32)
    qn_gm = pl.create_tensor([norm_tiles * DECODE_NORM_ROWS_TILE, KDA_DIM], dtype=pl.FP32)
    k_rows = pl.reshape(key, [rows_total, KDA_DIM])
    q_rows = pl.reshape(query, [rows_total, KDA_DIM])
    with pl.spmd(norm_tiles, name_hint="decode_kda_l2norm") as norm_tid:
        r0 = pl.tile.get_block_idx() * DECODE_NORM_ROWS_TILE
        rows = pl.min(DECODE_NORM_ROWS_TILE, rows_total - r0)
        q_rows_tile = pl.slice(q_rows, [DECODE_NORM_ROWS_TILE, KDA_DIM], [r0, 0], valid_shape=[rows, KDA_DIM])
        qt = pl.cast(q_rows_tile, pl.FP32)
        k_rows_tile = pl.slice(k_rows, [DECODE_NORM_ROWS_TILE, KDA_DIM], [r0, 0], valid_shape=[rows, KDA_DIM])
        kt = pl.cast(k_rows_tile, pl.FP32)
        q_buffer = pl.create_tensor([DECODE_NORM_ROWS_TILE, KDA_DIM], dtype=pl.FP32)
        k_buffer = pl.create_tensor([DECODE_NORM_ROWS_TILE, KDA_DIM], dtype=pl.FP32)
        q_buffer[:, :] = qt
        k_buffer[:, :] = kt
        q_squared_norm = pl.create_tensor([DECODE_NORM_ROWS_TILE, 1], dtype=pl.FP32)
        k_squared_norm = pl.create_tensor([DECODE_NORM_ROWS_TILE, 1], dtype=pl.FP32)
        q_square = pl.create_tensor([DECODE_NORM_ROWS_TILE, KDA_DIM], dtype=pl.FP32)
        k_square = pl.create_tensor([DECODE_NORM_ROWS_TILE, KDA_DIM], dtype=pl.FP32)
        q_square[:, :] = pl.mul(q_buffer, q_buffer)
        k_square[:, :] = pl.mul(k_buffer, k_buffer)
        q_sum = accurate_row_sum(q_square)
        k_sum = accurate_row_sum(k_square)
        q_squared_norm[:, :] = pl.add(q_sum, DECODE_L2_EPS)
        k_squared_norm[:, :] = pl.add(k_sum, DECODE_L2_EPS)
        q_root = accurate_sqrt(q_squared_norm)
        k_root = accurate_sqrt(k_squared_norm)
        q_denominator = pl.create_tensor([DECODE_NORM_ROWS_TILE, KDA_DIM], dtype=pl.FP32)
        k_denominator = pl.create_tensor([DECODE_NORM_ROWS_TILE, KDA_DIM], dtype=pl.FP32)
        ones = pl.full([DECODE_NORM_ROWS_TILE, KDA_DIM], dtype=pl.FP32, value=1.0)
        q_denominator[:, :] = pl.row_expand_mul(ones, q_root)
        k_denominator[:, :] = pl.row_expand_mul(ones, k_root)
        q_unit = accurate_divide(q_buffer, q_denominator)
        k_unit = accurate_divide(k_buffer, k_denominator)
        qn_gm[r0 : r0 + DECODE_NORM_ROWS_TILE, :] = pl.mul(q_unit, DECODE_Q_SCALE)
        kn_gm[r0 : r0 + DECODE_NORM_ROWS_TILE, :] = k_unit
    state_2d = pl.reshape(recurrent_state, [pl.tensor.dim(recurrent_state, 0) * LOCAL_KDA_H * KDA_DIM, KDA_DIM])
    out_rows = pl.reshape(output, [t_dim, LOCAL_KDA_H * KDA_DIM])
    with pl.spmd(tasks, name_hint="decode_kda", deps=[norm_tid]):
        task = pl.tile.get_block_idx()
        t = task // (LOCAL_KDA_H * lanes)
        h = task // lanes % LOCAL_KDA_H
        v0 = task % lanes * DECODE_V_TILE
        row = pl.cast(pl.read(state_rows, [t]), pl.INDEX)
        qk_row = t * LOCAL_KDA_H + h
        q_row = qn_gm[qk_row : qk_row + 1, 0:KDA_DIM]
        k_row = kn_gm[qk_row : qk_row + 1, 0:KDA_DIM]
        exp_buffer = pl.create_tensor([1, KDA_DIM], dtype=pl.FP32)
        decay_tile = pl.slice(decay, [1, 1, KDA_DIM], [t, h, 0], drop_dims=[0])
        exp_buffer[:, :] = pl.reshape(decay_tile, [1, KDA_DIM])
        g_row = accurate_exp(exp_buffer)
        beta_s = pl.read(beta, [t, h])
        srow = (row * LOCAL_KDA_H + h) * KDA_DIM + v0
        state_tile = state_2d[srow:srow + DECODE_V_TILE, 0:KDA_DIM]
        decayed = pl.col_expand_mul(state_tile, g_row)
        kv_buffer = pl.create_tensor([DECODE_V_TILE, KDA_DIM], dtype=pl.FP32)
        kv_buffer[:, :] = pl.col_expand_mul(decayed, k_row)
        kv_col = accurate_row_sum(kv_buffer)
        value_tile = pl.slice(value, [1, 1, DECODE_V_TILE], [t, h, v0], drop_dims=[0])
        v_row_v4_rows = pl.reshape(value_tile, [1, DECODE_V_TILE])
        v_row = pl.cast(v_row_v4_rows, pl.FP32)
        v_col_full = pl.full([COL_REP_TILE, DECODE_V_TILE], dtype=pl.FP32, value=1.0)
        v_col_col_expand_mul = pl.col_expand_mul(v_col_full, v_row)
        v_col_transpose = pl.transpose(v_col_col_expand_mul, 0, 1)
        v_col_row_sum = pl.row_sum(v_col_transpose)
        v_col = pl.mul(v_col_row_sum, 1.0 / COL_REP_TILE)
        delta_col = pl.mul(pl.sub(v_col, kv_col), beta_s)
        updated_full = pl.full([DECODE_V_TILE, KDA_DIM], dtype=pl.FP32, value=1.0)
        updated_row_expand_mul = pl.row_expand_mul(updated_full, delta_col)
        updated_col_expand_mul = pl.col_expand_mul(updated_row_expand_mul, k_row)
        updated = pl.add(decayed, updated_col_expand_mul)
        state_2d[srow : srow + DECODE_V_TILE, 0:KDA_DIM] = updated
        # Keep the state/query dot product in the same FP32 reduction domain
        # as the recurrent update.
        out_products = pl.create_tensor([DECODE_V_TILE, KDA_DIM], dtype=pl.FP32)
        out_products[:, :] = pl.col_expand_mul(updated, q_row)
        out_sum = accurate_row_sum(out_products)
        out_row = pl.reshape(out_sum, [1, DECODE_V_TILE])
        ocol = h * KDA_DIM + v0
        out_row_bf16 = pl.cast(out_row, pl.BF16, mode="rint")
        out_rows[t : t + 1, ocol : ocol + DECODE_V_TILE] = pl.set_validshape(out_row_bf16, 1, DECODE_V_TILE)
    return (output, recurrent_state)


@pl.jit
def decode_kda_test(
    query: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    key: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    value: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    decay: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.FP32],
    beta: pl.Tensor[[T_DYN, BETA_PAD], pl.FP32],
    recurrent_state: pl.InOut[pl.Tensor[[KDA_STATE_DYN, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]],
    state_rows: pl.Tensor[[T_DYN], pl.INT32],
    output: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
):
    output, recurrent_state = decode_kda(query, key, value, decay, beta, recurrent_state, state_rows, output)
    return output, recurrent_state


@pl.jit.inline
def prefill_kda(
    query: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    key: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    value: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    decay: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.FP32],
    beta: pl.Tensor[[T_DYN, BETA_PAD], pl.FP32],
    recurrent_state: pl.InOut[pl.Tensor[[KDA_STATE_DYN, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_rows: pl.Tensor[[B_DYN], pl.INT32],
    tri_incl: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP32],
    tri_strict: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP32],
    head_onehot: pl.Tensor[[LOCAL_KDA_H, BETA_PAD], pl.FP32],
    eye_wide: pl.Tensor[[CHUNK_TILE, KDA_DIM], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
):
    t_dim = pl.tensor.dim(query, 0)
    requests = pl.tensor.dim(query_start_loc, 0) - 1
    # Flattened head-major, not token-major: with one row per (token, head) the tokens
    # of one head would sit LOCAL_KDA_H rows apart and a chunk slice would straddle
    # heads. Keeping the row axis as tokens makes a head a contiguous column range.
    head_width = LOCAL_KDA_H * KDA_DIM

    # Per-chunk scratch. A chunk of sequence n starting at token ``off`` is parked at
    # row ``off + n * CHUNK_TILE``: its last chunk runs past the sequence end, and without
    # the per-sequence stride it would land on the next sequence's rows.
    scratch_rows = t_dim + requests * CHUNK_TILE
    # Flat rather than [head, row, dim]: a store whose source is a tile reshaped to
    # two leading unit axes lowers to a tile with a single-element row, which ptoas
    # rejects on its 32-byte row alignment. One row per chunk, so the index is plain.
    # Two alternating slots each, so a squaring never reads the buffer it writes.
    # Kept two-dimensional: storing a matmul result through a reshape into a rank-3
    # view has no producer pipe at this width, while the plain 2D store does.
    # Every slot of these is written whole before it is read, so they need no seeding.
    # a_gram widened to the inverse's working width. Its right-hand columns have to be
    # zero and stay zero, and they are zeroed by their own kernel below: a read that
    # spans two stores from the same task is not ordered against both of them, which is
    # what made a "matrix here, zeros beside it" seed read back as all zeros and
    # collapse the product that followed.
    agw_rows = LOCAL_KDA_H * scratch_rows
    zero_tiles = (agw_rows + CHUNK_TILE - 1) // CHUNK_TILE
    agw_gm = pl.create_tensor([LOCAL_KDA_H * scratch_rows, KDA_DIM], dtype=pl.FP32)
    with pl.spmd(zero_tiles, name_hint="prefill_kda_zero") as zero_tid:
        z0 = pl.tile.get_block_idx() * CHUNK_TILE
        zrows = pl.min(CHUNK_TILE, agw_rows - z0)
        agw_gm_value_full = pl.full([CHUNK_TILE, KDA_DIM], dtype=pl.FP32, value=0.0)
        agw_gm[z0:z0 + CHUNK_TILE, 0:KDA_DIM] = pl.set_validshape(agw_gm_value_full, zrows, KDA_DIM)
    # The cumulative sum comes out of the cube in a boxed layout, whose subviews must
    # be whole inner blocks -- a single row of it is not addressable. Parking it in GM
    # and reading the pivot and last rows back is the way to get at them.
    # Every intermediate gets its own buffer. Reusing one for two values -- parking kg
    # where W will land, or masking a Gram matrix in place -- makes a read and a write
    # of the same rows race inside one task, and a read that wins returns uninitialised
    # memory, which is where the NaNs came from.
    kg_gm = pl.create_tensor([LOCAL_KDA_H, scratch_rows, KDA_DIM], dtype=pl.BF16)

    # --- pass 1: everything that does not touch the carried state ---
    kf_gm = pl.create_tensor([LOCAL_KDA_H, scratch_rows, KDA_DIM], dtype=pl.FP32)
    qf_gm = pl.create_tensor([LOCAL_KDA_H, scratch_rows, KDA_DIM], dtype=pl.FP32)
    gcum_gm = pl.create_tensor([LOCAL_KDA_H, scratch_rows, KDA_DIM], dtype=pl.FP32)
    agram_gm = pl.create_tensor([LOCAL_KDA_H, scratch_rows, CHUNK_TILE], dtype=pl.FP32)
    eglast_gm = pl.create_tensor([LOCAL_KDA_H * scratch_rows, KDA_DIM], dtype=pl.FP32)
    a2_gm = pl.create_tensor([LOCAL_KDA_H, scratch_rows, CHUNK_TILE], dtype=pl.BF16)
    kdec_gm = pl.create_tensor([LOCAL_KDA_H, scratch_rows, KDA_DIM], dtype=pl.BF16)
    qg_gm = pl.create_tensor([LOCAL_KDA_H, scratch_rows, KDA_DIM], dtype=pl.BF16)
    g_rows = pl.reshape(decay, [t_dim, head_width])
    v_rows = pl.reshape(value, [t_dim, head_width])
    k_rows = pl.reshape(key, [t_dim, head_width])
    q_rows = pl.reshape(query, [t_dim, head_width])
    with pl.spmd(LOCAL_KDA_H * requests, name_hint="prefill_kda_chunk", deps=[zero_tid]) as chunk_tid:
        task = pl.tile.get_block_idx()
        h = task // requests
        n = task % requests
        bos = pl.cast(pl.read(query_start_loc, [n]), pl.INDEX)
        eos = pl.cast(pl.read(query_start_loc, [n + 1]), pl.INDEX)
        length = eos - bos

        for s_idx in pl.range(0, length, CHUNK_TILE):
            off = bos + s_idx
            actual = pl.min(length - s_idx, CHUNK_TILE)
            base = off + n * CHUNK_TILE
            # Rows past the chunk's real length must read as zero, and valid_shape does
            # not put them there: it marks which rows are real, while the tile still
            # holds whatever the overread picked up -- which for a chunk that runs past
            # the end of the packed stream is whatever follows it in GM. fillpad zeroes
            # them explicitly. For the decay this is not hygiene but load-bearing: a
            # zero pad keeps the cumulative sum flat, so its last row is still the
            # chunk's true final decay, and the padded rows contribute nothing to the
            # state update. Sequences that are a whole number of chunks never expose
            # this; a ragged tail after a full chunk does.
            # Zeroed, then declared valid to the full chunk. Both halves matter. The
            # zeroing is load-bearing for the decay: a zero pad keeps the cumulative sum
            # flat, so its last row is still the chunk's true final decay, and the pad
            # rows contribute nothing to the state. Restoring the valid extent matters
            # because these tiles end up as transposed matmul operands, and the cube
            # lowers a transposed operand to an extract whose row count must be a
            # multiple of 16 -- a ragged valid extent becomes exactly that row count.
            # The simulator enforces neither, so both show up only on device.
            col0 = h * KDA_DIM
            q_raw_slice = pl.slice(q_rows, [CHUNK_TILE, KDA_DIM], [off, col0], valid_shape=[actual, KDA_DIM])
            q_raw_fillpad = pl.fillpad(q_raw_slice, pad_value=pl.PadValue.zero)
            q_raw_set_validshape = pl.set_validshape(q_raw_fillpad, CHUNK_TILE, KDA_DIM)
            q_raw = pl.cast(q_raw_set_validshape, pl.FP32)
            k_raw_slice = pl.slice(k_rows, [CHUNK_TILE, KDA_DIM], [off, col0], valid_shape=[actual, KDA_DIM])
            k_raw_fillpad = pl.fillpad(k_raw_slice, pad_value=pl.PadValue.zero)
            k_raw_set_validshape = pl.set_validshape(k_raw_fillpad, CHUNK_TILE, KDA_DIM)
            k_raw = pl.cast(k_raw_set_validshape, pl.FP32)
            v_c_slice = pl.slice(v_rows, [CHUNK_TILE, KDA_DIM], [off, col0], valid_shape=[actual, KDA_DIM])
            v_c_fillpad = pl.fillpad(v_c_slice, pad_value=pl.PadValue.zero)
            v_c = pl.set_validshape(v_c_fillpad, CHUNK_TILE, KDA_DIM)
            g_c_slice = pl.slice(g_rows, [CHUNK_TILE, KDA_DIM], [off, col0], valid_shape=[actual, KDA_DIM])
            g_c_fillpad = pl.fillpad(g_c_slice, pad_value=pl.PadValue.zero)
            g_c = pl.set_validshape(g_c_fillpad, CHUNK_TILE, KDA_DIM)

            qf_gm_value_mul = pl.mul(q_raw, q_raw)
            qf_gm_value_row_sum = pl.row_sum(qf_gm_value_mul)
            qf_gm_value_add = pl.add(qf_gm_value_row_sum, PREFILL_L2_EPS)
            qf_gm_value_rsqrt = pl.rsqrt(qf_gm_value_add, high_precision=True)
            qf_gm_value_row_expand_mul = pl.row_expand_mul(q_raw, qf_gm_value_rsqrt)
            qf_gm_value_mul_2 = pl.mul(qf_gm_value_row_expand_mul, PREFILL_Q_SCALE)
            qf_gm[h:h + 1, base:base + CHUNK_TILE, 0:KDA_DIM] = pl.reshape(qf_gm_value_mul_2, [1, CHUNK_TILE, KDA_DIM])
            kf_gm_value_mul = pl.mul(k_raw, k_raw)
            kf_gm_value_row_sum = pl.row_sum(kf_gm_value_mul)
            kf_gm_value_add = pl.add(kf_gm_value_row_sum, PREFILL_L2_EPS)
            kf_gm_value_rsqrt = pl.rsqrt(kf_gm_value_add, high_precision=True)
            kf_gm_value_row_expand_mul = pl.row_expand_mul(k_raw, kf_gm_value_rsqrt)
            kf_gm[h:h + 1, base:base + CHUNK_TILE, 0:KDA_DIM] = pl.reshape(kf_gm_value_row_expand_mul, [1, CHUNK_TILE, KDA_DIM])
            q_f_slice = pl.slice(qf_gm, [1, CHUNK_TILE, KDA_DIM], [h, base, 0], drop_dims=[0])
            q_f = pl.reshape(q_f_slice, [CHUNK_TILE, KDA_DIM])
            k_f_slice = pl.slice(kf_gm, [1, CHUNK_TILE, KDA_DIM], [h, base, 0], drop_dims=[0])
            k_f = pl.reshape(k_f_slice, [CHUNK_TILE, KDA_DIM])

            # Per-channel cumulative sum along the chunk axis, as one GEMM against the
            # inclusive lower-triangular constant: the DSL has no scan primitive.
            gcum_gm_value_matmul = pl.matmul(tri_incl[0:CHUNK_TILE, 0:CHUNK_TILE], g_c, pl.FP32)
            gcum_gm[h:h + 1, base:base + CHUNK_TILE, 0:KDA_DIM] = pl.reshape(gcum_gm_value_matmul, [1, CHUNK_TILE, KDA_DIM])
            gcum_slice = pl.slice(gcum_gm, [1, CHUNK_TILE, KDA_DIM], [h, base, 0], drop_dims=[0])
            gcum = pl.reshape(gcum_slice, [CHUNK_TILE, KDA_DIM])
            eg = pl.exp(gcum)

            # Decay reconstructed around the chunk's first row, which is also its only
            # pivot: CHUNK_TILE is DECAY_TILE. Rows are at or after the pivot, so row_dec's
            # exponent is <= 0; the column factor's exponent is at most
            # 5 * (CHUNK_TILE - 1) = 75 inside the chunk, so the clamp provably touches only
            # the masked-out future. Both Gram matrices stay FP32 -- the decay's
            # dynamic range rounds away in BF16.
            gp = pl.reshape(pl.slice(gcum_gm, [1, 1, KDA_DIM], [h, base, 0], drop_dims=[0]), [1, KDA_DIM])
            gp_full = pl.col_expand_mul(pl.full([CHUNK_TILE, KDA_DIM], dtype=pl.FP32, value=1.0), gp)
            row_dec = pl.exp(pl.sub(gcum, gp_full))
            ki_a = pl.mul(k_f, pl.exp(pl.minimum(pl.sub(gp_full, gcum), DECAY_CAP)))
            a_full = pl.matmul(pl.mul(k_f, row_dec), ki_a, pl.FP32, b_trans=True)
            a2_raw = pl.matmul(pl.mul(q_f, row_dec), ki_a, pl.FP32, b_trans=True)

            # beta as a column, straight out of a masked reduction. beta is stored one
            # value per head in a 16-wide row, so selecting this head's column with a
            # host-supplied one-hot and summing gives the column directly -- no
            # transpose, and a reduction is the only way a column is a legal tile here.
            beta_block_slice = pl.slice(beta, [CHUNK_TILE, BETA_PAD], [off, 0], valid_shape=[actual, BETA_PAD])
            beta_block_fillpad = pl.fillpad(beta_block_slice, pad_value=pl.PadValue.zero)
            beta_block = pl.set_validshape(beta_block_fillpad, CHUNK_TILE, BETA_PAD)
            b_col = pl.row_sum(pl.col_expand_mul(beta_block, head_onehot[h : h + 1, 0:BETA_PAD]))
            a_gram = pl.mul(pl.row_expand_mul(a_full, b_col), tri_strict[0:CHUNK_TILE, 0:CHUNK_TILE])
            agram_gm[h:h + 1, base:base + CHUNK_TILE, 0:CHUNK_TILE] = pl.reshape(a_gram, [1, CHUNK_TILE, CHUNK_TILE])
            a2_gm_value_mul = pl.mul(a2_raw, tri_incl[0:CHUNK_TILE, 0:CHUNK_TILE])
            a2_gm_value_cast = pl.cast(a2_gm_value_mul, pl.BF16, mode="rint")
            a2_gm[h:h + 1, base:base + CHUNK_TILE, 0:CHUNK_TILE] = pl.reshape(a2_gm_value_cast, [1, CHUNK_TILE, CHUNK_TILE])
            qg_gm_value_mul = pl.mul(q_f, eg)
            qg_gm_value_cast = pl.cast(qg_gm_value_mul, pl.BF16, mode="rint")
            qg_gm[h:h + 1, base:base + CHUNK_TILE, 0:KDA_DIM] = pl.reshape(qg_gm_value_cast, [1, CHUNK_TILE, KDA_DIM])
            # kg = k * exp(gcum) feeds W; kdec carries the chunk-final decay.
            glast_slice = pl.slice(gcum_gm, [1, 1, KDA_DIM], [h, base + CHUNK_TILE - 1, 0], drop_dims=[0])
            glast = pl.reshape(glast_slice, [1, KDA_DIM])
            glast_full = pl.col_expand_mul(pl.full([CHUNK_TILE, KDA_DIM], dtype=pl.FP32, value=1.0), glast)
            kdec_gm_value_sub = pl.sub(glast_full, gcum)
            kdec_gm_value_exp = pl.exp(kdec_gm_value_sub)
            kdec_gm_value_mul = pl.mul(k_f, kdec_gm_value_exp)
            kdec_gm_value_cast = pl.cast(kdec_gm_value_mul, pl.BF16, mode="rint")
            kdec_gm[h:h + 1, base:base + CHUNK_TILE, 0:KDA_DIM] = pl.reshape(kdec_gm_value_cast, [1, CHUNK_TILE, KDA_DIM])
            eg_row = h * scratch_rows + base
            eglast_gm[eg_row : eg_row + 1, 0:KDA_DIM] = pl.exp(glast)
            kg_gm_value_mul = pl.mul(k_f, eg)
            kg_gm_value_cast = pl.cast(kg_gm_value_mul, pl.BF16, mode="rint")
            kg_gm[h:h + 1, base:base + CHUNK_TILE, 0:KDA_DIM] = pl.reshape(kg_gm_value_cast, [1, CHUNK_TILE, KDA_DIM])

            rb = h * scratch_rows + base
            agw_gm[rb : rb + CHUNK_TILE, 0:CHUNK_TILE] = a_gram

    # (I + A)^-1 as a finite series. A is strictly lower and therefore nilpotent with
    # A^CHUNK_TILE = 0, so (I + A)^-1 = sum_k (-A)^k, and the sum doubles its reach each
    # step. Iterating E_j = I + M_j rather than the power itself keeps both matmul
    # operands in GM:
    #     E_0 = I - A,               P_0 = I
    #     P_{j+1} = E_j P_j,         E_{j+1} = E_j (E_j - 2I) + 2 I
    # (the second line is M_{j+1} = M_j^2 rewritten in terms of E, factored so that it
    # is one matmul and one add). Four steps cover all 16 terms.
    #
    # Each step is its own kernel, chained by an explicit dependency. Within one kernel
    # a store and a later read of the same buffer are not ordered, and the seed read
    # back as zeros, which collapsed the whole product; across kernels the dependency
    # is what orders them. Everything is carried at KDA_DIM width because a 16-wide
    # FP32 tile has no hand-off between the cube and the vector pipe here, and the
    # padding stays zero through both recurrences.
    #
    # The textbook alternative, row-at-a-time forward substitution, has a matmul with
    # M = 1, which the cube lowers to an extract whose row count must be a multiple
    # of 16.

    prod_gm = pl.create_tensor([LOCAL_KDA_H * scratch_rows, 5 * KDA_DIM], dtype=pl.FP32)
    estep_gm = pl.create_tensor([LOCAL_KDA_H * scratch_rows, 5 * KDA_DIM], dtype=pl.FP32)
    with pl.spmd(LOCAL_KDA_H * requests, name_hint="prefill_kda_inv_seed", deps=[chunk_tid]) as seed_tid:
        task = pl.tile.get_block_idx()
        h = task // requests
        n = task % requests
        bos = pl.cast(pl.read(query_start_loc, [n]), pl.INDEX)
        eos = pl.cast(pl.read(query_start_loc, [n + 1]), pl.INDEX)
        length = eos - bos
        for s_idx in pl.range(0, length, CHUNK_TILE):
            off = bos + s_idx
            base = off + n * CHUNK_TILE
            rb = h * scratch_rows + base
            estep_gm[rb:rb + CHUNK_TILE, 0:KDA_DIM] = pl.sub(eye_wide[0:CHUNK_TILE, 0:KDA_DIM], agw_gm[rb:rb + CHUNK_TILE, 0:KDA_DIM])
            prod_gm[rb : rb + CHUNK_TILE, 0:KDA_DIM] = eye_wide[0:CHUNK_TILE, 0:KDA_DIM]

    with pl.spmd(LOCAL_KDA_H * requests, name_hint="prefill_kda_inv0", deps=[seed_tid]) as inv0_tid:
        task = pl.tile.get_block_idx()
        h = task // requests
        n = task % requests
        bos = pl.cast(pl.read(query_start_loc, [n]), pl.INDEX)
        eos = pl.cast(pl.read(query_start_loc, [n + 1]), pl.INDEX)
        length = eos - bos
        for s_idx in pl.range(0, length, CHUNK_TILE):
            off = bos + s_idx
            base = off + n * CHUNK_TILE
            rb = h * scratch_rows + base
            e16 = estep_gm[rb : rb + CHUNK_TILE, 0 : 0 + CHUNK_TILE]
            e_wide = estep_gm[rb : rb + CHUNK_TILE, 0 : 0 + KDA_DIM]
            two_eye = pl.mul(eye_wide[0:CHUNK_TILE, 0:KDA_DIM], 2.0)
            prod_gm[rb:rb + CHUNK_TILE, 128:128 + KDA_DIM] = pl.matmul(e16, prod_gm[rb:rb + CHUNK_TILE, 0:0 + KDA_DIM], pl.FP32)
            estep_gm_value_sub = pl.sub(e_wide, two_eye)
            estep_gm_value_matmul = pl.matmul(e16, estep_gm_value_sub, pl.FP32)
            estep_gm[rb:rb + CHUNK_TILE, 128:128 + KDA_DIM] = pl.add(estep_gm_value_matmul, two_eye)

    with pl.spmd(LOCAL_KDA_H * requests, name_hint="prefill_kda_inv1", deps=[inv0_tid]) as inv1_tid:
        task = pl.tile.get_block_idx()
        h = task // requests
        n = task % requests
        bos = pl.cast(pl.read(query_start_loc, [n]), pl.INDEX)
        eos = pl.cast(pl.read(query_start_loc, [n + 1]), pl.INDEX)
        length = eos - bos
        for s_idx in pl.range(0, length, CHUNK_TILE):
            off = bos + s_idx
            base = off + n * CHUNK_TILE
            rb = h * scratch_rows + base
            e16 = estep_gm[rb : rb + CHUNK_TILE, 128 : 128 + CHUNK_TILE]
            e_wide = estep_gm[rb : rb + CHUNK_TILE, 128 : 128 + KDA_DIM]
            two_eye = pl.mul(eye_wide[0:CHUNK_TILE, 0:KDA_DIM], 2.0)
            prod_gm[rb:rb + CHUNK_TILE, 256:256 + KDA_DIM] = pl.matmul(e16, prod_gm[rb:rb + CHUNK_TILE, 128:128 + KDA_DIM], pl.FP32)
            estep_gm_value_sub_2 = pl.sub(e_wide, two_eye)
            estep_gm_value_matmul_2 = pl.matmul(e16, estep_gm_value_sub_2, pl.FP32)
            estep_gm[rb:rb + CHUNK_TILE, 256:256 + KDA_DIM] = pl.add(estep_gm_value_matmul_2, two_eye)

    with pl.spmd(LOCAL_KDA_H * requests, name_hint="prefill_kda_inv2", deps=[inv1_tid]) as inv2_tid:
        task = pl.tile.get_block_idx()
        h = task // requests
        n = task % requests
        bos = pl.cast(pl.read(query_start_loc, [n]), pl.INDEX)
        eos = pl.cast(pl.read(query_start_loc, [n + 1]), pl.INDEX)
        length = eos - bos
        for s_idx in pl.range(0, length, CHUNK_TILE):
            off = bos + s_idx
            base = off + n * CHUNK_TILE
            rb = h * scratch_rows + base
            e16 = estep_gm[rb : rb + CHUNK_TILE, 256 : 256 + CHUNK_TILE]
            e_wide = estep_gm[rb : rb + CHUNK_TILE, 256 : 256 + KDA_DIM]
            two_eye = pl.mul(eye_wide[0:CHUNK_TILE, 0:KDA_DIM], 2.0)
            prod_gm[rb:rb + CHUNK_TILE, 384:384 + KDA_DIM] = pl.matmul(e16, prod_gm[rb:rb + CHUNK_TILE, 256:256 + KDA_DIM], pl.FP32)
            estep_gm_value_sub_3 = pl.sub(e_wide, two_eye)
            estep_gm_value_matmul_3 = pl.matmul(e16, estep_gm_value_sub_3, pl.FP32)
            estep_gm[rb:rb + CHUNK_TILE, 384:384 + KDA_DIM] = pl.add(estep_gm_value_matmul_3, two_eye)

    with pl.spmd(LOCAL_KDA_H * requests, name_hint="prefill_kda_inv3", deps=[inv2_tid]) as inv3_tid:
        task = pl.tile.get_block_idx()
        h = task // requests
        n = task % requests
        bos = pl.cast(pl.read(query_start_loc, [n]), pl.INDEX)
        eos = pl.cast(pl.read(query_start_loc, [n + 1]), pl.INDEX)
        length = eos - bos
        for s_idx in pl.range(0, length, CHUNK_TILE):
            off = bos + s_idx
            base = off + n * CHUNK_TILE
            rb = h * scratch_rows + base
            e16 = estep_gm[rb : rb + CHUNK_TILE, 384 : 384 + CHUNK_TILE]
            e_wide = estep_gm[rb : rb + CHUNK_TILE, 384 : 384 + KDA_DIM]
            two_eye = pl.mul(eye_wide[0:CHUNK_TILE, 0:KDA_DIM], 2.0)
            prod_gm[rb:rb + CHUNK_TILE, 512:512 + KDA_DIM] = pl.matmul(e16, prod_gm[rb:rb + CHUNK_TILE, 384:384 + KDA_DIM], pl.FP32)
            estep_gm_value_sub_4 = pl.sub(e_wide, two_eye)
            estep_gm_value_matmul_4 = pl.matmul(e16, estep_gm_value_sub_4, pl.FP32)
            estep_gm[rb:rb + CHUNK_TILE, 512:512 + KDA_DIM] = pl.add(estep_gm_value_matmul_4, two_eye)

    u_gm = pl.create_tensor([LOCAL_KDA_H, scratch_rows, KDA_DIM], dtype=pl.FP32)
    w_gm = pl.create_tensor([LOCAL_KDA_H, scratch_rows, KDA_DIM], dtype=pl.BF16)
    with pl.spmd(LOCAL_KDA_H * requests, name_hint="prefill_kda_wu", deps=[inv3_tid]) as wu_tid:
        task = pl.tile.get_block_idx()
        h = task // requests
        n = task % requests
        bos = pl.cast(pl.read(query_start_loc, [n]), pl.INDEX)
        eos = pl.cast(pl.read(query_start_loc, [n + 1]), pl.INDEX)
        length = eos - bos
        for s_idx in pl.range(0, length, CHUNK_TILE):
            off = bos + s_idx
            base = off + n * CHUNK_TILE
            rb = h * scratch_rows + base
            actual = pl.min(length - s_idx, CHUNK_TILE)
            col0 = h * KDA_DIM
            # W = A^-1 (beta . kg) and U = A^-1 (beta . v). Scaling the inverse's
            # columns by beta is the same as scaling the rows of what it multiplies.
            beta_block_slice_2 = pl.slice(beta, [CHUNK_TILE, BETA_PAD], [off, 0], valid_shape=[actual, BETA_PAD])
            beta_block_fillpad_2 = pl.fillpad(beta_block_slice_2, pad_value=pl.PadValue.zero)
            beta_block = pl.set_validshape(beta_block_fillpad_2, CHUNK_TILE, BETA_PAD)
            b_col = pl.row_sum(pl.col_expand_mul(beta_block, head_onehot[h : h + 1, 0:BETA_PAD]))
            a_inv_bf = pl.cast(prod_gm[rb:rb + CHUNK_TILE, 4 * KDA_DIM:4 * KDA_DIM + CHUNK_TILE], pl.BF16, mode="rint")
            kg_bf_slice = pl.slice(kg_gm, [1, CHUNK_TILE, KDA_DIM], [h, base, 0], drop_dims=[0])
            kg_bf_reshape = pl.reshape(kg_bf_slice, [CHUNK_TILE, KDA_DIM])
            kg_bf_row_expand_mul = pl.row_expand_mul(pl.cast(kg_bf_reshape, pl.FP32), b_col)
            kg_bf = pl.cast(kg_bf_row_expand_mul, pl.BF16, mode="rint")
            w_gm_value_matmul = pl.matmul(a_inv_bf, kg_bf, pl.FP32)
            w_gm_value_cast = pl.cast(w_gm_value_matmul, pl.BF16, mode="rint")
            w_gm[h:h + 1, base:base + CHUNK_TILE, 0:KDA_DIM] = pl.reshape(w_gm_value_cast, [1, CHUNK_TILE, KDA_DIM])
            v_c_slice_2 = pl.slice(v_rows, [CHUNK_TILE, KDA_DIM], [off, col0], valid_shape=[actual, KDA_DIM])
            v_c_fillpad_2 = pl.fillpad(v_c_slice_2, pad_value=pl.PadValue.zero)
            v_c = pl.set_validshape(v_c_fillpad_2, CHUNK_TILE, KDA_DIM)
            v_beta = pl.cast(pl.row_expand_mul(pl.cast(v_c, pl.FP32), b_col), pl.BF16, mode="rint")
            u_gm_value_matmul = pl.matmul(a_inv_bf, v_beta, pl.FP32)
            u_gm[h:h + 1, base:base + CHUNK_TILE, 0:KDA_DIM] = pl.reshape(u_gm_value_matmul, [1, CHUNK_TILE, KDA_DIM])
    # --- pass 2: the sequential scan over chunks, tiled over the value axis ---
    lanes = KDA_DIM // PREFILL_V_TILE
    # Two slots per task, alternating by chunk. Reading and writing one buffer inside
    # the loop would make each iteration alias its own input, and a read that wins
    # returns uninitialised memory -- which is exactly the NaN a multi-chunk sequence
    # produced while a single-chunk one passed.
    state_gm = pl.create_tensor([LOCAL_KDA_H * requests * lanes * 2 * PREFILL_V_TILE, KDA_DIM], dtype=pl.FP32)
    state_2d = pl.reshape(recurrent_state, [pl.tensor.dim(recurrent_state, 0) * LOCAL_KDA_H * KDA_DIM, KDA_DIM])
    out_rows = pl.reshape(output, [t_dim, head_width])
    with pl.spmd(LOCAL_KDA_H * requests * lanes, name_hint="prefill_kda_scan", deps=[wu_tid]):
        task = pl.tile.get_block_idx()
        h = task // (requests * lanes)
        n = (task // lanes) % requests
        v0 = (task % lanes) * PREFILL_V_TILE
        bos = pl.cast(pl.read(query_start_loc, [n]), pl.INDEX)
        eos = pl.cast(pl.read(query_start_loc, [n + 1]), pl.INDEX)
        length = eos - bos
        row = pl.cast(pl.read(state_rows, [n]), pl.INDEX)

        # The scan runs in the pool's own [V, K] orientation. The recurrence is natural
        # in [K, V], but every re-orientation it needs is a matmul operand flag, and
        # keeping [V, K] means the seed and the flush are plain copies. That matters
        # beyond tidiness: the transpose instruction constrains both of its row counts
        # to multiples of 16, which the simulator does not check and the device does.
        pool_row = (row * LOCAL_KDA_H + h) * KDA_DIM + v0
        seed = state_2d[pool_row:pool_row + PREFILL_V_TILE, 0:KDA_DIM]
        slot0 = task * 2 * PREFILL_V_TILE
        state_gm[slot0 : slot0 + PREFILL_V_TILE, 0:KDA_DIM] = seed

        for s_idx in pl.range(0, length, CHUNK_TILE):
            off = bos + s_idx
            actual = pl.min(length - s_idx, CHUNK_TILE)
            base = off + n * CHUNK_TILE
            chunk_idx = s_idx // CHUNK_TILE
            src = chunk_idx % 2
            dst = (chunk_idx + 1) % 2
            s_cur = state_gm[(task * 2 + src) * PREFILL_V_TILE:(task * 2 + src) * PREFILL_V_TILE + PREFILL_V_TILE, 0:KDA_DIM]
            s_bf = pl.cast(s_cur, pl.BF16, mode="rint")
            w_c_slice = pl.slice(w_gm, [1, CHUNK_TILE, KDA_DIM], [h, base, 0], drop_dims=[0])
            w_c = pl.reshape(w_c_slice, [CHUNK_TILE, KDA_DIM])
            u_c_slice = pl.slice(u_gm, [1, CHUNK_TILE, PREFILL_V_TILE], [h, base, v0], drop_dims=[0])
            u_c = pl.reshape(u_c_slice, [CHUNK_TILE, PREFILL_V_TILE])
            # v' = U - W S: the part of the chunk's values the carried state already
            # explains.
            vi = pl.sub(u_c, pl.matmul(w_c, s_bf, pl.FP32, b_trans=True))
            vi_bf = pl.cast(vi, pl.BF16, mode="rint")
            qg_c_slice = pl.slice(qg_gm, [1, CHUNK_TILE, KDA_DIM], [h, base, 0], drop_dims=[0])
            qg_c = pl.reshape(qg_c_slice, [CHUNK_TILE, KDA_DIM])
            a2_c_slice = pl.slice(a2_gm, [1, CHUNK_TILE, CHUNK_TILE], [h, base, 0], drop_dims=[0])
            a2_c = pl.reshape(a2_c_slice, [CHUNK_TILE, CHUNK_TILE])
            oc = pl.add(pl.matmul(qg_c, s_bf, pl.FP32, b_trans=True), pl.matmul(a2_c, vi_bf, pl.FP32))
            ocol = h * KDA_DIM + v0
            out_rows[off:off + CHUNK_TILE, ocol:ocol + PREFILL_V_TILE] = pl.set_validshape(pl.cast(oc, pl.BF16, mode="rint"), actual, PREFILL_V_TILE)

            kdec_c_slice = pl.slice(kdec_gm, [1, CHUNK_TILE, KDA_DIM], [h, base, 0], drop_dims=[0])
            kdec_c = pl.reshape(kdec_c_slice, [CHUNK_TILE, KDA_DIM])
            # In [V, K] the chunk-final decay is a per-column factor, so it applies as
            # the row it already is.
            eglast_row = eglast_gm[h * scratch_rows + base:h * scratch_rows + base + 1, 0:KDA_DIM]
            d_row = (task * 2 + dst) * PREFILL_V_TILE
            state_gm_value_col_expand_mul = pl.col_expand_mul(s_cur, eglast_row)
            state_gm_value_matmul = pl.matmul(vi_bf, kdec_c, pl.FP32, a_trans=True)
            state_gm[d_row:d_row + PREFILL_V_TILE, 0:KDA_DIM] = pl.add(state_gm_value_col_expand_mul, state_gm_value_matmul)

        # Flush. The last chunk wrote the slot with the parity of the chunk count.
        chunks = (length + CHUNK_TILE - 1) // CHUNK_TILE
        last = chunks % 2
        final = state_gm[(task * 2 + last) * PREFILL_V_TILE:(task * 2 + last) * PREFILL_V_TILE + PREFILL_V_TILE, 0:KDA_DIM]
        state_2d[pool_row : pool_row + PREFILL_V_TILE, 0:KDA_DIM] = final

    return output, recurrent_state


@pl.jit
def prefill_kda_test(
    query: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    key: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    value: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    decay: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.FP32],
    beta: pl.Tensor[[T_DYN, BETA_PAD], pl.FP32],
    recurrent_state: pl.InOut[pl.Tensor[[KDA_STATE_DYN, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_rows: pl.Tensor[[B_DYN], pl.INT32],
    tri_incl: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP32],
    tri_strict: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP32],
    head_onehot: pl.Tensor[[LOCAL_KDA_H, BETA_PAD], pl.FP32],
    eye_wide: pl.Tensor[[CHUNK_TILE, KDA_DIM], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
):
    (output, recurrent_state) = prefill_kda(
        query, key, value, decay, beta, recurrent_state, query_start_loc, state_rows, tri_incl, tri_strict,
        head_onehot, eye_wide, output,
    )
    return output, recurrent_state


@pl.jit.inline
def stack_conv_weight(
    conv_q: pl.Tensor[[LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    conv_k: pl.Tensor[[LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    conv_v: pl.Tensor[[LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    weight: pl.Tensor[[KDA_CONV_K, CONV_DIM], pl.BF16],
):
    """Stack Q/K/V convolution weights into tap-major channel rows.

    Each block owns a 32-column BF16 line; family boundaries remain 32-aligned."""
    with pl.spmd(CONV_DIM // 32, name_hint="stack_conv_weight"):
        c0 = pl.tile.get_block_idx() * 32
        for tap in pl.unroll(KDA_CONV_K):
            line = pl.tile.full([1, 32], dtype=pl.BF16, value=0.0)
            for col in pl.unroll(32):
                c = c0 + col
                if c < LOCAL_KDA_QKV_DIM:
                    value = pl.read(conv_q, [c, 0, tap])
                elif c < 2 * LOCAL_KDA_QKV_DIM:
                    value = pl.read(conv_k, [c - LOCAL_KDA_QKV_DIM, 0, tap])
                else:
                    value = pl.read(conv_v, [c - 2 * LOCAL_KDA_QKV_DIM, 0, tap])
                pl.tile.write(line, [0, col], value)
            pl.tile.store(line, [tap, c0], weight)


def _conv_reference(
    stream: torch.Tensor, state: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One request: convolve ``stream`` given ``state``, return output and new state.

    ``state`` is ``[STATE_LEN, CONV_DIM]`` of pre-conv rows, ``weight`` is
    ``[KDA_CONV_K, CONV_DIM]``.
    """
    full = torch.cat([state.float(), stream.float()], dim=0)
    length = stream.shape[0]
    out = torch.zeros(length, stream.shape[1], dtype=torch.float32)
    for j in range(KDA_CONV_K):
        out += full[j : j + length] * weight[j].float()
    # causal_conv1d applies SiLU as one FP32 division before the BF16 output.
    denominator = 1.0 + torch.exp(-out.double()).float()
    activated = (out.double() / denominator.double()).float()
    return activated, full[length : length + STATE_LEN]


def golden_kda_conv_prefill(
    mixed_qkv: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(query, key, value, conv_state)``; the state is returned updated.

    ``state_rows`` addresses the pool by request row rather than by batch position,
    so a reordered or paged batch is modelled rather than assumed away.
    """
    tokens, _ = mixed_qkv.shape
    out = torch.zeros(tokens, CONV_DIM, dtype=torch.float32)
    new_state = conv_state.clone()
    requests = query_start_loc.numel() - 1
    for b in range(requests):
        s = int(query_start_loc[b])
        e = int(query_start_loc[b + 1])
        if e <= s:
            continue
        row = int(state_rows[b])
        conv_out, tail = _conv_reference(mixed_qkv[s:e], conv_state[row], weight)
        out[s:e] = conv_out
        new_state[row] = tail.to(conv_state.dtype)
    shaped = out.to(mixed_qkv.dtype).view(tokens, 3, LOCAL_KDA_H, KDA_DIM)
    return shaped[:, 0], shaped[:, 1], shaped[:, 2], new_state


def golden_kda_conv_decode(
    mixed_qkv: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(query, key, value, conv_state)`` for one token per row."""
    tokens, _ = mixed_qkv.shape
    out = torch.zeros(tokens, CONV_DIM, dtype=torch.float32)
    new_state = conv_state.clone()
    for t in range(tokens):
        row = int(state_rows[t])
        conv_out, tail = _conv_reference(mixed_qkv[t : t + 1], new_state[row].float(), weight)
        out[t] = conv_out[0]
        new_state[row] = tail.to(conv_state.dtype)
    shaped = out.to(mixed_qkv.dtype).view(tokens, 3, LOCAL_KDA_H, KDA_DIM)
    return shaped[:, 0], shaped[:, 1], shaped[:, 2], new_state


def _golden_prefill_tensors(tensors) -> None:
    (q, k, v, st) = golden_kda_conv_prefill(
        tensors["mixed_qkv"], tensors["weight"], tensors["conv_state"], tensors["query_start_loc"],
        tensors["state_rows"],
    )
    tensors["query"][:] = q
    tensors["key"][:] = k
    tensors["value"][:] = v
    tensors["conv_state"][:] = st


def _golden_decode_tensors(tensors) -> None:
    (q, k, v, st) = golden_kda_conv_decode(
        tensors["mixed_qkv"], tensors["weight"], tensors["conv_state"], tensors["state_rows"],
    )
    tensors["query"][:] = q
    tensors["key"][:] = k
    tensors["value"][:] = v
    tensors["conv_state"][:] = st


def conv_build_tensor_specs(case: str = "ragged", decode: bool = False):
    from golden import TensorSpec

    bf, i32 = torch.bfloat16, torch.int32

    def common(tokens):
        return [
            TensorSpec("mixed_qkv", [tokens, CONV_DIM], bf, init_value=lambda : (torch.randn(tokens, CONV_DIM) * 0.8).to(bf)),
            TensorSpec("weight", [KDA_CONV_K, CONV_DIM], bf, init_value=lambda : (torch.randn(KDA_CONV_K, CONV_DIM) * 0.5).to(bf)),
            # Non-zero, so a dropped state read is visible rather than a no-op.
            TensorSpec("conv_state", [CONV_POOL_ROWS, STATE_LEN, CONV_DIM], bf, init_value=lambda : (torch.randn(CONV_POOL_ROWS, STATE_LEN, CONV_DIM) * 0.6).to(bf)),
        ]

    if decode:
        tokens = 3 if case == "single" else 5
        # Deliberately not the identity mapping: a kernel that ignores state_rows and
        # uses the token index would pass an identity fixture.
        rows = torch.tensor([4, 1, 5, 0, 3][:tokens], dtype=torch.int32)
        return common(tokens) + [
            TensorSpec("state_rows", [tokens], i32, init_value=lambda: rows),
            TensorSpec("query", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
            TensorSpec("key", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
            TensorSpec("value", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
        ]

    lengths = _PREFILL_LENGTHS[case]
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    tokens = starts[-1]
    rows = torch.tensor([4, 1, 5, 0, 2][: len(lengths)], dtype=torch.int32)
    return common(tokens) + [
        TensorSpec("query_start_loc", [len(starts)], i32, init_value=lambda : torch.tensor(starts, dtype=torch.int32)),
        TensorSpec("state_rows", [len(lengths)], i32, init_value=lambda: rows),
        TensorSpec("query", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
        TensorSpec("key", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
        TensorSpec("value", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
    ]


def build_boundary_tensor_specs(decode: bool = False):
    """Reproduce a BF16 midpoint reached by distinct FP32 addition orders.

    The key channel uses four exact BF16 inputs/taps. Sequential summation
    rounds SiLU to -0.01556396484375; pairwise summation rounds to
    -0.0155029296875. Prefill compares a bulk token with a continued head
    carrying the same history; decode exercises that head directly.
    """
    from golden import TensorSpec

    history = torch.tensor([0.003326416015625, 0.6796875, 0.625], dtype=torch.bfloat16)
    current = 0.212890625
    taps = torch.tensor([-0.0018157958984375, -0.011474609375, -0.03369140625, -0.0126953125], dtype=torch.bfloat16)
    tokens = 1 if decode else 5
    channel = LOCAL_KDA_QKV_DIM  # first key channel in q|k|v storage
    mixed = torch.zeros(tokens, CONV_DIM, dtype=torch.bfloat16)
    mixed[:, channel] = current
    if not decode:
        mixed[:3, channel] = history
    weight = torch.zeros(KDA_CONV_K, CONV_DIM, dtype=torch.bfloat16)
    weight[:, channel] = taps
    state = torch.zeros(2, STATE_LEN, CONV_DIM, dtype=torch.bfloat16)
    state[1, :, channel] = history
    specs = [
        TensorSpec("mixed_qkv", list(mixed.shape), mixed.dtype, init_value=mixed),
        TensorSpec("weight", list(weight.shape), weight.dtype, init_value=weight),
        TensorSpec("conv_state", list(state.shape), state.dtype, init_value=state),
    ]
    if not decode:
        specs.append(
            TensorSpec("query_start_loc", [3], torch.int32, init_value=torch.tensor([0, 4, 5], dtype=torch.int32)),
        )
    rows = torch.tensor([1] if decode else [0, 1], dtype=torch.int32)
    return specs + [
        TensorSpec("state_rows", list(rows.shape), rows.dtype, init_value=rows),
        TensorSpec("query", [tokens, LOCAL_KDA_H, KDA_DIM], torch.bfloat16),
        TensorSpec("key", [tokens, LOCAL_KDA_H, KDA_DIM], torch.bfloat16),
        TensorSpec("value", [tokens, LOCAL_KDA_H, KDA_DIM], torch.bfloat16),
    ]


def kda_host_constants() -> dict[str, torch.Tensor]:
    """Build the shared KDA tables uploaded once per rank."""
    return {
        "kda_tri_incl": torch.tril(torch.ones(CHUNK_TILE, CHUNK_TILE, dtype=torch.float32)),
        "kda_tri_strict": torch.tril(torch.ones(CHUNK_TILE, CHUNK_TILE, dtype=torch.float32), diagonal=-1),
        "kda_head_onehot": torch.eye(BETA_PAD, dtype=torch.float32)[:LOCAL_KDA_H].contiguous(),
        "kda_eye_wide": torch.nn.functional.pad(
            torch.eye(CHUNK_TILE, dtype=torch.float32), (0, KDA_DIM - CHUNK_TILE)
        ).contiguous(),
    }


def golden_kda_projection(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_f_a: torch.Tensor,
    w_f_b: torch.Tensor,
    dt_bias: torch.Tensor,
    a_log: torch.Tensor,
    w_b: torch.Tensor,
    w_g_a: torch.Tensor,
    w_g_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference for the four KDA front-end outputs.

    Returns ``(mixed_qkv, decay, beta, out_gate)``. ``decay`` is the **log** decay:
    the delta rule exponentiates it, this function does not.
    """
    tokens = x.shape[0]

    def linear_bf16(inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # FP32 dot products followed by the BF16 projection boundary.
        return golden_bf16_linear(inp, weight).to(torch.bfloat16)

    q = linear_bf16(x, w_q)
    k = linear_bf16(x, w_k)
    v = linear_bf16(x, w_v)
    mixed_qkv = torch.cat([q, k, v], dim=-1)

    f_a = linear_bf16(x, w_f_a)
    g_raw = linear_bf16(f_a, w_f_b).float().view(tokens, LOCAL_KDA_H, KDA_DIM)
    decay_rate = torch.exp(a_log.double()).float().view(1, LOCAL_KDA_H, 1)
    decay_logits = decay_rate * (g_raw + dt_bias.float().view(1, LOCAL_KDA_H, KDA_DIM))
    decay = KDA_GATE_LOWER_BOUND * torch.reciprocal(1.0 + torch.exp(-decay_logits.double()).float())

    # beta carries the cube's 16-column padding. The padded rows of b_proj are zero,
    # so their gate is sigmoid(0) = 0.5 rather than 0; the reference reproduces that
    # instead of leaving a mismatch the caller would have to know to ignore.
    beta = torch.full((tokens, BETA_PAD), 0.5, dtype=torch.float32)
    beta[:, :LOCAL_KDA_H] = torch.reciprocal(1.0 + torch.exp(-linear_bf16(x, w_b).double()).float())

    g_a = linear_bf16(x, w_g_a)
    out_gate = linear_bf16(g_a, w_g_b).view(tokens, LOCAL_KDA_H, KDA_DIM)

    return mixed_qkv, decay.float(), beta, out_gate


def golden_kda_projection_tensors(tensors) -> None:
    """Harness adapter: run the reference and write the four outputs in place."""
    (mixed_qkv, decay, beta, out_gate) = golden_kda_projection(
        tensors["x"], tensors["w_q"], tensors["w_k"], tensors["w_v"], tensors["w_f_a"], tensors["w_f_b"],
        tensors["dt_bias"], tensors["a_log"], tensors["w_b"], tensors["w_g_a"], tensors["w_g_b"],
    )
    tensors["mixed_qkv"][:] = mixed_qkv
    tensors["decay"][:] = decay
    tensors["beta"][:] = beta
    tensors["out_gate"][:] = out_gate


def projection_build_tensor_specs(tokens: int = 67):
    """A deliberately ragged token count, so the tail block is always exercised."""
    from golden import TensorSpec

    bf, f32 = torch.bfloat16, torch.float32

    def w(rows, cols, scale=0.02):
        return lambda: (torch.randn(rows, cols) * scale).to(bf)

    return [
        TensorSpec("x", [tokens, D], bf, init_value=lambda: (torch.randn(tokens, D) * 0.5).to(bf)),
        TensorSpec("w_q", [LOCAL_KDA_QKV_DIM, D], bf, init_value=w(LOCAL_KDA_QKV_DIM, D)),
        TensorSpec("w_k", [LOCAL_KDA_QKV_DIM, D], bf, init_value=w(LOCAL_KDA_QKV_DIM, D)),
        TensorSpec("w_v", [LOCAL_KDA_QKV_DIM, D], bf, init_value=w(LOCAL_KDA_QKV_DIM, D)),
        TensorSpec("w_f_a", [KDA_DIM, D], bf, init_value=w(KDA_DIM, D)),
        TensorSpec("w_f_b", [LOCAL_KDA_QKV_DIM, KDA_DIM], bf, init_value=w(LOCAL_KDA_QKV_DIM, KDA_DIM, 0.08)),
        # dt_bias spans a wide range in the real checkpoint; a narrow fixture would
        # hide a per-head / per-channel mix-up behind the sigmoid's flat tails.
        TensorSpec("dt_bias", [LOCAL_KDA_QKV_DIM], f32, init_value=lambda : torch.randn(LOCAL_KDA_QKV_DIM) * 2.0),
        # Distinct per-head values, so a head-axis broadcast error is visible. Spread
        # across however many heads the rank owns: four at TP16, sixteen at TP4.
        TensorSpec("a_log", [LOCAL_KDA_H], f32, init_value=lambda: torch.linspace(-0.7, 1.1, LOCAL_KDA_H)),
        TensorSpec("w_b", [LOCAL_KDA_H, D], bf, init_value=w(LOCAL_KDA_H, D, 0.05)),
        TensorSpec("w_g_a", [KDA_DIM, D], bf, init_value=w(KDA_DIM, D)),
        TensorSpec("w_g_b", [LOCAL_KDA_QKV_DIM, KDA_DIM], bf, init_value=w(LOCAL_KDA_QKV_DIM, KDA_DIM, 0.08)),
        TensorSpec("mixed_qkv", [tokens, 3 * LOCAL_KDA_QKV_DIM], bf),
        TensorSpec("decay", [tokens, LOCAL_KDA_H, KDA_DIM], f32),
        TensorSpec("beta", [tokens, BETA_PAD], f32),
        TensorSpec("out_gate", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
    ]


@dataclass
class KdaTensors:
    """Kimi-delta linear-attention weights of one KDA layer (FP32)."""

    w_q: torch.Tensor  # [qkv, D]
    w_k: torch.Tensor  # [qkv, D]
    w_v: torch.Tensor  # [qkv, D]
    conv_q: torch.Tensor  # [qkv, 1, conv_k]
    conv_k: torch.Tensor  # [qkv, 1, conv_k]
    conv_v: torch.Tensor  # [qkv, 1, conv_k]
    w_f_a: torch.Tensor  # [head_dim, D]
    w_f_b: torch.Tensor  # [qkv, head_dim]
    dt_bias: torch.Tensor  # [qkv] (FP32 in the checkpoint)
    a_log: torch.Tensor  # [H] (FP32 in the checkpoint)
    w_g_a: torch.Tensor  # [head_dim, D]
    w_g_b: torch.Tensor  # [qkv, head_dim]
    w_b: torch.Tensor  # [H, D]
    o_norm: torch.Tensor  # [head_dim]
    w_o: torch.Tensor  # [D, qkv]


def causal_conv1d_silu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Grouped causal conv + SiLU; ``x`` is ``[B, C, S]``, ``weight`` ``[C, 1, K]``.

    Port of ``causal_conv1d_fn`` with ``activation="silu"`` (hidden_act): pad
    ``K - 1`` on the left, conv1d with ``groups=C``, keep the first ``S`` outputs.
    """
    seq_len = x.shape[-1]
    out = F.conv1d(x, weight, bias=None, padding=weight.shape[-1] - 1, groups=x.shape[1])
    return F.silu(out[:, :, :seq_len])


def kda_chunk_core(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
    l2norm_eps: float = 1e-6,
) -> torch.Tensor:
    """Chunked Kimi-delta attention core (port of ``chunk_kimi_delta_attention``).

    Inputs ``[B, S, H, Dk]`` (q/k/v/g) and ``[B, S, H]`` (beta); output
    ``[B, S, H, Dv]``. The caller guarantees FP32. Follows the HF reference:
    in-kernel l2norm on q/k, ``1/sqrt(Dk)`` scale, log-space intra-chunk decay,
    inter-chunk state reuse, and the delta update rule.
    """
    query, key, value, beta, g = [t.transpose(1, 2).contiguous() for t in (query, key, value, beta, g)]
    query = reference_l2norm(query, dim=-1, eps=l2norm_eps)
    key = reference_l2norm(key, dim=-1, eps=l2norm_eps)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    total_sequence_length = sequence_length + pad_size

    query = F.pad(query, (0, 0, 0, pad_size)) * scale
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    g = F.pad(g, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, g, k_beta, v_beta = [
        t.reshape(t.shape[0], t.shape[1], -1, chunk_size, t.shape[-1])
        for t in (query, key, value, g, k_beta, v_beta)
    ]
    beta = beta.reshape(beta.shape[0], beta.shape[1], -1, chunk_size)

    # Intra-chunk attention in log space (per-head decay, unlike GDN's broadcast).
    g = g.cumsum(dim=-2)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=0)
    strict_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=1)
    decay_mask = (g.unsqueeze(-2) - g.unsqueeze(-3)).masked_fill(strict_mask[..., None], float("-inf"))
    decay_mask = decay_mask.exp()
    attn = -(k_beta.unsqueeze(-2) * key.unsqueeze(-3) * decay_mask).sum(dim=-1).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp())

    last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim)
    core_attn_out = torch.zeros_like(value)

    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=1)
    for i in range(total_sequence_length // chunk_size):
        q_i = query[:, :, i]
        k_i = key[:, :, i]
        v_i = value[:, :, i]
        g_i = g[:, :, i]

        attn_inter = (q_i * g_i.exp()) @ last_recurrent_state
        attn_intra = (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, i]).sum(dim=-1)
        attn_intra = attn_intra.masked_fill(mask, 0)
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime

        core_attn_out[:, :, i] = attn_inter + attn_intra @ v_new
        last_recurrent_state = (
            last_recurrent_state * g_i[:, :, -1].exp().unsqueeze(-1)
            + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    return core_attn_out[:, :, :sequence_length].transpose(1, 2).contiguous()


def kda_recurrent_core(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    l2norm_eps: float = 1e-6,
) -> torch.Tensor:
    """Step-by-step Kimi-delta recurrence (port of the HF recurrent fallback).

    O(S) loop over the reference recurrence itself — the ground truth the chunk
    core is checked against in the self-test. Same input contract as
    :func:`kda_chunk_core`.
    """
    query, key, value, g, beta = [t.transpose(1, 2).contiguous() for t in (query, key, value, g, beta)]
    query = reference_l2norm(query, dim=-1, eps=l2norm_eps)
    key = reference_l2norm(key, dim=-1, eps=l2norm_eps)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim)
    out = torch.zeros(batch_size, sequence_length, num_heads, v_head_dim)
    for i in range(sequence_length):
        q_i = query[:, :, i]
        k_i = key[:, :, i]
        v_i = value[:, :, i]
        g_i = g[:, :, i][..., None].exp()
        b_i = beta[:, :, i][..., None]

        state = state * g_i
        kv_mem = (state * k_i[..., None]).sum(dim=-2)
        delta = (v_i - kv_mem) * b_i
        state = state + k_i.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, i] = (state * q_i.unsqueeze(-1)).sum(dim=-2)
    return out.contiguous()  # [B, S, H, V], the layout HF's recurrent returns


def kda_forward(
    x: torch.Tensor, weights: KdaTensors, model: Glm53FlashConfig, attention_mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One KDA sub-layer: projections -> causal SiLU conv -> gates -> core -> o_proj.

    Port of ``Glm5NextTextLinearAttention.forward`` (full-prefill branch, no
    cache). Returns ``(output [B, S, D], internals)``.
    """
    if attention_mask is not None:
        # Tune out padding tokens before every projection (mamba-style).
        x = x * attention_mask[..., None].to(x.dtype)
    batch_size, seq_len, _ = x.shape
    heads, head_dim = model.linear_num_heads, model.linear_head_dim
    qkv = heads * head_dim
    hidden_shape = (batch_size, seq_len, heads, head_dim)

    mixed_qkv = torch.cat([x @ weights.w_q.t(), x @ weights.w_k.t(), x @ weights.w_v.t()], dim=-1)
    mixed_qkv = mixed_qkv.transpose(1, 2)
    conv_weight = torch.cat([weights.conv_q, weights.conv_k, weights.conv_v], dim=0)
    mixed_qkv = causal_conv1d_silu(mixed_qkv, conv_weight)
    mixed_qkv = mixed_qkv[:, :, -seq_len:]
    query, key, value = torch.split(mixed_qkv.transpose(1, 2), [qkv] * 3, dim=-1)
    query = query.reshape(hidden_shape)
    key = key.reshape(hidden_shape)
    value = value.reshape(hidden_shape)

    # Forget gate (safe lower-bound branch) and input gate.
    forget = (x @ weights.w_f_a.t()) @ weights.w_f_b.t() + weights.dt_bias
    g = model.linear_lower_bound * torch.sigmoid(
        weights.a_log.exp().view(1, 1, heads, 1) * forget.reshape(hidden_shape)
    )
    beta = torch.sigmoid(x @ weights.w_b.t())

    core = kda_chunk_core(query, key, value, g, beta)

    gate = ((x @ weights.w_g_a.t()) @ weights.w_g_b.t()).reshape(hidden_shape)
    # RMSNormGated with sigmoid activation, strict FP32 (weight applied in FP32).
    variance = core.pow(2).mean(-1, keepdim=True)
    normed = weights.o_norm * (core * torch.rsqrt(variance + model.rms_norm_eps))
    normed = normed * torch.sigmoid(gate)
    output = normed.reshape(batch_size, seq_len, qkv) @ weights.w_o.t()

    internals = {
        "kda_q": query,
        "kda_k": key,
        "kda_v": value,
        "kda_g": g,
        "kda_beta": beta,
        "kda_core": core,
        "kda_gate": gate,
        "kda_o_normed": normed.reshape(batch_size, seq_len, qkv),
    }
    return output, internals


def synthetic_kda(model: Glm53FlashConfig, generator: torch.Generator) -> KdaTensors:
    qkv = model.linear_num_heads * model.linear_head_dim
    return KdaTensors(
        w_q=_randn((qkv, model.hidden_size), generator), w_k=_randn((qkv, model.hidden_size), generator),
        w_v=_randn((qkv, model.hidden_size), generator),
        conv_q=torch.randn((qkv, 1, model.linear_conv_kernel_dim), generator=generator) * 0.2,
        conv_k=torch.randn((qkv, 1, model.linear_conv_kernel_dim), generator=generator) * 0.2,
        conv_v=torch.randn((qkv, 1, model.linear_conv_kernel_dim), generator=generator) * 0.2,
        w_f_a=_randn((model.linear_head_dim, model.hidden_size), generator),
        w_f_b=_randn((qkv, model.linear_head_dim), generator),
        dt_bias=torch.randn(qkv, generator=generator) - 4.0, a_log=torch.zeros(model.linear_num_heads),
        w_g_a=_randn((model.linear_head_dim, model.hidden_size), generator),
        w_g_b=_randn((qkv, model.linear_head_dim), generator),
        w_b=_randn((model.linear_num_heads, model.hidden_size), generator),
        o_norm=1.0 + torch.randn(model.linear_head_dim, generator=generator) * 0.05,
        w_o=_randn((model.hidden_size, qkv), generator),
    )


def _test_kda_cores(model: Glm53FlashConfig) -> None:
    torch.manual_seed(13)
    heads, head_dim = model.linear_num_heads, model.linear_head_dim
    b, s = 2, 37  # deliberately not chunk-aligned
    q = torch.randn(b, s, heads, head_dim)
    k = torch.randn(b, s, heads, head_dim) * 2
    v = torch.randn(b, s, heads, head_dim)
    g = -5.0 * torch.sigmoid(torch.randn(b, s, heads, head_dim))
    beta = torch.sigmoid(torch.randn(b, s, heads))

    chunked = kda_chunk_core(q, k, v, g, beta)
    recurrent = kda_recurrent_core(q, k, v, g, beta)
    assert chunked.shape == (b, s, heads, head_dim)
    core_diff = _assert_close(chunked, recurrent, "KDA chunk vs recurrent", atol=1e-3, rtol=1e-3)

    # Causal conv against an unfold-based reference (cross-correlation, left pad).
    channels = heads * head_dim
    kernel = model.linear_conv_kernel_dim
    x = torch.randn(1, channels, s)
    weight = torch.randn(channels, 1, kernel) * 0.2
    out = causal_conv1d_silu(x, weight)
    padded = F.pad(x, (kernel - 1, 0))
    expected = F.silu(torch.einsum("ctk,ck->ct", padded[0].unfold(-1, kernel, 1), weight[:, 0, :]))
    conv_diff = _assert_close(out[0], expected, "causal conv1d")
    print(f"  KDA cores: chunk==recurrent max|d|={core_diff:.2e} (S={s}); conv max|d|={conv_diff:.2e}")


def golden_kda_output(
    core_attn_out: torch.Tensor,
    norm_weight: torch.Tensor,
    out_gate: torch.Tensor,
    w_o: torch.Tensor,
) -> torch.Tensor:
    normalized = rms_norm_gated(core_attn_out, norm_weight, out_gate)
    flattened = normalized.reshape(*normalized.shape[:-2], -1)
    # Match the BF16 local linear output before TP reduction. FP32 is only
    # the exchange dtype; it carries the widened BF16 result.
    return golden_bf16_linear(flattened, w_o).to(torch.bfloat16).float()


def golden_kda_output_tensors(tensors) -> None:
    """Harness adapter: run the reference and write the output in place."""
    tensors["output"][:] = golden_kda_output(
        tensors["core_attn_out"], tensors["norm_weight"], tensors["out_gate"], tensors["w_o"],
    )


def compare_kda_output(actual, expected, **kwargs):
    """Require a BF16 local result carried through the FP32 exchange buffer."""
    from golden import ratio_allclose

    if not torch.equal(actual, actual.to(torch.bfloat16).float()):
        return False, "local projection was not rounded to BF16 before TP exchange"
    return ratio_allclose(max_error_ratio=0)(actual, expected, **kwargs)


def output_build_tensor_specs(tokens: int = 67):
    """A deliberately ragged token count, so the tail block is always exercised."""
    from golden import TensorSpec

    bf, f32 = torch.bfloat16, torch.float32
    del f32

    return [
        TensorSpec("core_attn_out", [tokens, LOCAL_KDA_H, KDA_DIM], bf, init_value=lambda : (torch.randn(tokens, LOCAL_KDA_H, KDA_DIM) * 0.7).to(bf)),
        # Not all-ones: a unit gamma hides a dropped or mis-broadcast norm weight.
        TensorSpec("norm_weight", [KDA_DIM], bf, init_value=lambda : (1.0 + torch.randn(KDA_DIM) * 0.3).to(bf)),
        # Wide enough to span both sigmoid tails.
        TensorSpec("out_gate", [tokens, LOCAL_KDA_H, KDA_DIM], bf, init_value=lambda : (torch.randn(tokens, LOCAL_KDA_H, KDA_DIM) * 3.0).to(bf)),
        TensorSpec("w_o", [D, LOCAL_KDA_QKV_DIM], bf, init_value=lambda : (torch.randn(D, LOCAL_KDA_QKV_DIM) * 0.02).to(bf)),
        TensorSpec("output", [tokens, D], torch.float32),
    ]


def golden_decode_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(output, recurrent_state)``; the state is returned updated.

    ``recurrent_state`` is ``[rows, LOCAL_KDA_H, V, K]``.
    """
    tokens = query.shape[0]
    state = recurrent_state.clone().float()
    out = torch.zeros(tokens, LOCAL_KDA_H, KDA_DIM, dtype=torch.float32)
    for t in range(tokens):
        row = int(state_rows[t])
        for h in range(LOCAL_KDA_H):
            q = l2norm(query[t, h].float(), eps=DECODE_L2_EPS) * DECODE_Q_SCALE  # [K]
            k = l2norm(key[t, h].float(), eps=DECODE_L2_EPS)  # [K]
            v = value[t, h].float()  # [V]
            g = decay[t, h].float()  # [K]
            b = float(beta[t, h])
            s = state[row, h] * torch.exp(g.double()).float().unsqueeze(0)  # [V, K]
            kv_mem = (s * k.unsqueeze(0)).double().sum(dim=-1).float()  # [V]
            delta = (v - kv_mem) * b  # [V]
            s = s + delta.unsqueeze(-1) * k.unsqueeze(0)
            state[row, h] = s
            out[t, h] = (s * q.unsqueeze(0)).double().sum(dim=-1).float()
    return out.to(query.dtype), state.to(recurrent_state.dtype)


def _decode_golden_tensors(tensors) -> None:
    (out, state) = golden_decode_kda(
        tensors["query"], tensors["key"], tensors["value"], tensors["decay"], tensors["beta"],
        tensors["recurrent_state"], tensors["state_rows"],
    )
    tensors["output"][:] = out
    tensors["recurrent_state"][:] = state


def decode_build_tensor_specs(tokens: int = 5):
    from golden import TensorSpec

    bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32
    # Deliberately not the identity mapping: a kernel that ignored state_rows and used
    # the token index would pass an identity fixture.
    rows = torch.tensor([4, 1, 5, 0, 3][:tokens], dtype=torch.int32)

    return [
        TensorSpec("query", [tokens, LOCAL_KDA_H, KDA_DIM], bf, init_value=lambda : torch.randn(tokens, LOCAL_KDA_H, KDA_DIM).to(bf)),
        TensorSpec("key", [tokens, LOCAL_KDA_H, KDA_DIM], bf, init_value=lambda : torch.randn(tokens, LOCAL_KDA_H, KDA_DIM).to(bf)),
        TensorSpec("value", [tokens, LOCAL_KDA_H, KDA_DIM], bf, init_value=lambda : torch.randn(tokens, LOCAL_KDA_H, KDA_DIM).to(bf)),
        # The real gate spans the whole (-5, 0) range. A near-zero fixture would make
        # exp(g) ~ 1 and hide both the decay and its overflow guards.
        TensorSpec("decay", [tokens, LOCAL_KDA_H, KDA_DIM], f32, init_value=lambda : -5.0 * torch.rand(tokens, LOCAL_KDA_H, KDA_DIM)),
        # beta carries kda_projection's cube-width padding; only the first
        # LOCAL_KDA_H columns are read.
        TensorSpec("beta", [tokens, BETA_PAD], f32, init_value=lambda: torch.rand(tokens, BETA_PAD)),
        # A non-zero initial state, so a kernel that silently starts from zero fails.
        TensorSpec("recurrent_state", [DECODE_POOL_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], f32, init_value=lambda : torch.randn(DECODE_POOL_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM) * 0.1),
        TensorSpec("state_rows", [tokens], i32, init_value=lambda: rows),
        TensorSpec("output", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
    ]


def golden_prefill_kda_fp32(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 mathematical reference for independent precision diagnostics.

    Algorithm: evaluate the delta-rule recurrence one token at a time.
    Precision: normalize Q/K, apply decay, update state and reduce outputs in FP32.
    Intermediate values retain FP32 precision throughout the recurrence.
    Returns: activations in ``query.dtype`` (BF16 for the kernel ABI) and the
    updated state in its original dtype (FP32), with ``[rows, H, V, K]`` layout.

    Use this reference to measure the mixed-precision approximation error.
    The default kernel comparison uses ``golden_prefill_kda_mixed_precision``.
    """
    tokens = query.shape[0]
    state = recurrent_state.clone().float()
    out = torch.zeros(tokens, LOCAL_KDA_H, KDA_DIM, dtype=torch.float32)
    requests = query_start_loc.numel() - 1
    for b in range(requests):
        s = int(query_start_loc[b])
        e = int(query_start_loc[b + 1])
        row = int(state_rows[b])
        for h in range(LOCAL_KDA_H):
            sh = state[row, h]  # [V, K]
            for t in range(s, e):
                q = l2norm(query[t, h].float(), eps=PREFILL_L2_EPS) * PREFILL_Q_SCALE
                k = l2norm(key[t, h].float(), eps=PREFILL_L2_EPS)
                v = value[t, h].float()
                g = decay[t, h].float()
                bta = float(beta[t, h])
                sh = sh * torch.exp(g).unsqueeze(0)
                kv_mem = (sh * k.unsqueeze(0)).sum(dim=-1)
                delta = (v - kv_mem) * bta
                sh = sh + delta.unsqueeze(-1) * k.unsqueeze(0)
                out[t, h] = (sh * q.unsqueeze(0)).sum(dim=-1)
            state[row, h] = sh
    return out.to(query.dtype), state.to(recurrent_state.dtype)


def golden_prefill_kda_mixed_precision(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixed-precision implementation reference for default PTO3 validation.

    Algorithm: process each request in ``CHUNK_TILE``-token blocks with zero-padded tails.
    Precision: compute normalization, decay, Gram matrices and the inverse in FP32;
    round the kernel's intermediate GEMM operands to BF16 and accumulate in FP32.
    The carried state remains FP32, with a BF16 view used by the chunk scan GEMMs.
    Returns: BF16 activations and the updated FP32 ``[rows, H, V, K]`` state.

    The ``_golden_tensors`` adapter selects this reference for kernel comparisons.
    Use ``golden_prefill_kda_fp32`` separately for mathematical precision diagnostics.
    """
    state = recurrent_state.clone().float()
    out = torch.zeros_like(query, dtype=torch.float32)
    heads = query.shape[1]
    eye = torch.eye(CHUNK_TILE, dtype=torch.float32, device=query.device)
    tri = torch.tril(torch.ones_like(eye))
    strict = torch.tril(tri, diagonal=-1)
    for request in range(query_start_loc.numel() - 1):
        begin = int(query_start_loc[request])
        end = int(query_start_loc[request + 1])
        row = int(state_rows[request])
        carried = state[row]
        for start in range(begin, end, CHUNK_TILE):
            stop = min(start + CHUNK_TILE, end)
            count = stop - start
            padding = (0, 0, 0, CHUNK_TILE - count)
            q = torch.nn.functional.pad(query[start:stop].float().transpose(0, 1), padding)
            k = torch.nn.functional.pad(key[start:stop].float().transpose(0, 1), padding)
            v = torch.nn.functional.pad(value[start:stop].float().transpose(0, 1), padding)
            g = torch.nn.functional.pad(decay[start:stop].float().transpose(0, 1), padding)
            b = beta[start:stop, :heads].float().T.unsqueeze(-1)
            b = torch.nn.functional.pad(b, padding)
            q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + PREFILL_L2_EPS) * PREFILL_Q_SCALE
            k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + PREFILL_L2_EPS)
            cumulative = tri @ g
            exp_decay = cumulative.exp()
            pivot = cumulative[:, :1]
            row_decay = (cumulative - pivot).exp()
            key_columns = k * (pivot - cumulative).clamp(max=DECAY_CAP).exp()
            gram = ((k * row_decay) @ key_columns.transpose(-1, -2)) * b * strict
            a2 = (((q * row_decay) @ key_columns.transpose(-1, -2)) * tri).bfloat16().float()
            qg = (q * exp_decay).bfloat16().float()
            kdec = (k * (cumulative[:, -1:] - cumulative).exp()).bfloat16().float()
            kg = (k * exp_decay).bfloat16().float()

            # Four FP32 doubling steps produce (I + gram)^-1 for CHUNK_TILE=16.
            inverse_step = eye - gram
            inverse = eye.expand(heads, -1, -1)
            for _ in range(4):
                inverse = inverse_step @ inverse
                inverse_step = inverse_step @ (inverse_step - 2 * eye) + 2 * eye
            inverse_bf16 = inverse.bfloat16().float()
            beta_kg = (kg * b).bfloat16().float()
            w = (inverse_bf16 @ beta_kg).bfloat16().float()
            beta_v = (v * b).bfloat16().float()
            u = inverse_bf16 @ beta_v

            # BF16 operands feed the scan; the persistent state update stays FP32.
            carried_bf16 = carried.bfloat16().float()
            vi = (u - w @ carried_bf16.transpose(-1, -2)).bfloat16().float()
            result = qg @ carried_bf16.transpose(-1, -2) + a2 @ vi
            out[start:stop] = result[:, :count].transpose(0, 1)
            carried = carried * cumulative[:, -1:].exp() + vi.transpose(-1, -2) @ kdec
        state[row] = carried
    return out.to(query.dtype), state.to(recurrent_state.dtype)


def _prefill_golden_tensors(tensors) -> None:
    (out, state) = golden_prefill_kda_mixed_precision(
        tensors["query"], tensors["key"], tensors["value"], tensors["decay"], tensors["beta"],
        tensors["recurrent_state"], tensors["query_start_loc"], tensors["state_rows"],
    )
    tensors["output"][:] = out
    tensors["recurrent_state"][:] = state


def prefill_build_tensor_specs(case: str = "ragged"):
    """``zerostate`` is a diagnostic: it isolates the carried state from everything else."""
    from golden import TensorSpec

    bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32
    lengths = _LENGTHS[case]
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    tokens = starts[-1]
    rows = torch.tensor([4, 1, 5][: len(lengths)], dtype=torch.int32)

    return [
        TensorSpec("query", [tokens, LOCAL_KDA_H, KDA_DIM], bf, init_value=lambda : torch.randn(tokens, LOCAL_KDA_H, KDA_DIM).to(bf)),
        # Keys share a per-head direction. Independent random keys are near-orthogonal
        # at width 128, which makes the Gram matrix ~0.09 and the triangular inverse
        # indistinguishable from the identity -- stubbing the inverse out entirely still
        # passed against the earlier fixture. A shared component puts the off-diagonal
        # terms at O(1), where the inverse is what the result depends on.
        TensorSpec("key", [tokens, LOCAL_KDA_H, KDA_DIM], bf, init_value=lambda : (torch.randn(tokens, LOCAL_KDA_H, KDA_DIM) * 0.4 + torch.randn(1, LOCAL_KDA_H, KDA_DIM)).to(bf)),
        TensorSpec("value", [tokens, LOCAL_KDA_H, KDA_DIM], bf, init_value=lambda : torch.randn(tokens, LOCAL_KDA_H, KDA_DIM).to(bf)),
        # The real gate spans the whole (-5, 0) range. A near-zero fixture leaves
        # exp(gcum) close to 1, which hides both the decay reconstruction and the
        # overflow clamp that keeps it finite.
        TensorSpec("decay", [tokens, LOCAL_KDA_H, KDA_DIM], f32, init_value=lambda : -5.0 * torch.rand(tokens, LOCAL_KDA_H, KDA_DIM)),
        # beta carries kda_projection's cube-width padding; only the first
        # LOCAL_KDA_H columns are read.
        TensorSpec("beta", [tokens, BETA_PAD], f32, init_value=(lambda : torch.zeros(tokens, BETA_PAD)) if case == "nobeta" else lambda : torch.rand(tokens, BETA_PAD)),
        # Non-zero, so a dropped carry or a transposed seed is visible.
        TensorSpec("recurrent_state", [PREFILL_POOL_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], f32, init_value=(lambda : torch.zeros(PREFILL_POOL_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM)) if case == "zerostate" else lambda : torch.randn(PREFILL_POOL_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM) * 0.1),
        TensorSpec("query_start_loc", [len(starts)], i32, init_value=lambda : torch.tensor(starts, dtype=torch.int32)),
        TensorSpec("state_rows", [len(lengths)], i32, init_value=lambda: rows),
        TensorSpec("tri_incl", [CHUNK_TILE, CHUNK_TILE], f32, init_value=lambda : torch.tril(torch.ones(CHUNK_TILE, CHUNK_TILE))),
        TensorSpec("tri_strict", [CHUNK_TILE, CHUNK_TILE], f32, init_value=lambda : torch.tril(torch.ones(CHUNK_TILE, CHUNK_TILE), diagonal=-1)),
        # Row h selects head h out of beta's padded 16-wide row.
        TensorSpec("head_onehot", [LOCAL_KDA_H, BETA_PAD], f32, init_value=lambda : torch.eye(BETA_PAD)[:LOCAL_KDA_H].contiguous()),
        # 2I in the left CHUNK_TILE columns, zero beside it: the padded width the triangular
        # inverse works at.
        # The identity at the padded width the triangular inverse works at.
        TensorSpec("eye_wide", [CHUNK_TILE, KDA_DIM], f32, init_value=lambda : torch.nn.functional.pad(torch.eye(CHUNK_TILE), (0, KDA_DIM - CHUNK_TILE))),
        TensorSpec("output", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
    ]


def run_convolution_case(argv=None):
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case", type=str, default="ragged", choices=["ragged", "single", "long", "decode", "decode-single"])
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)

    decode = args.case.startswith("decode")
    case = "single" if args.case == "decode-single" else ("ragged" if decode else args.case)
    result = run(
        fn=kda_conv_decode_test if decode else kda_conv_prefill_test,
        specs=conv_build_tensor_specs(case, decode),
        golden_fn=_golden_decode_tensors if decode else _golden_prefill_tensors,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=0.02,
        atol=0.02,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)

    def boundary_key_compare(actual, expected, *, rtol, atol, **_kwargs):
        from golden import ratio_allclose

        compare = ratio_allclose(rtol=rtol, atol=atol, max_error_ratio=0.0)
        passed, details = compare(actual, expected, rtol=rtol, atol=atol, **_kwargs)
        if not passed:
            return passed, details
        rows = [0] if decode else [3, 4]
        exact = torch.equal(actual[rows, 0, 0], expected[rows, 0, 0])
        return exact, "" if exact else "boundary key crossed a BF16 rounding tie"

    boundary = run(
        fn=kda_conv_decode_test if decode else kda_conv_prefill_test,
        specs=build_boundary_tensor_specs(decode),
        golden_fn=_golden_decode_tensors if decode else _golden_prefill_tensors,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=0.02,
        atol=0.02,
        compare_fn={"key": boundary_key_compare},
    )
    if not boundary.passed:
        if boundary.error:
            print(boundary.error)
        raise SystemExit(1)


def run_projection_case(argv=None):
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case", type=str, default="ragged", choices=["ragged", "aligned", "single"])
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)

    tokens = {"ragged": 67, "aligned": 64, "single": 1}[args.case]
    result = run(
        fn=kda_projection_test,
        specs=projection_build_tensor_specs(tokens),
        golden_fn=golden_kda_projection_tensors,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=0.02,
        atol=0.02,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


def run_output_case(argv=None):
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case", type=str, default="ragged", choices=["ragged", "aligned", "single"])
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)

    tokens = {"ragged": 67, "aligned": 64, "single": 1}[args.case]
    result = run(
        fn=kda_output_test,
        specs=output_build_tensor_specs(tokens),
        golden_fn=golden_kda_output_tensors,
        compare_fn={"output": compare_kda_output},
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=0.02,
        atol=0.02,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


def run_decode_case(argv=None):
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case", type=str, default="batch", choices=["batch", "single"])
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)

    result = run(
        fn=decode_kda_test,
        specs=decode_build_tensor_specs(1 if args.case == "single" else 5),
        golden_fn=_decode_golden_tensors,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=0.02,
        atol=0.02,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


def run_prefill_case(argv=None):
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case", type=str, default="ragged", choices=["ragged", "single", "aligned", "zerostate", "nobeta"])
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)

    result = run(
        fn=prefill_kda_test,
        specs=prefill_build_tensor_specs(args.case),
        golden_fn=_prefill_golden_tensors,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=0.02,
        atol=0.02,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    components = ["all", "projection", "convolution", "output", "decode", "prefill"]
    parser.add_argument("--component", choices=components, default="all")
    options, remaining = parser.parse_known_args(argv)
    if options.component == "all" and any(arg in ("-h", "--help") for arg in remaining):
        help_parser = argparse.ArgumentParser(description=__doc__, parents=[parser])
        help_parser.epilog = "Use --component NAME --help for stage options."
        help_parser.parse_args(argv)
    cases = {
        "projection": run_projection_case,
        "convolution": run_convolution_case,
        "output": run_output_case,
        "decode": run_decode_case,
        "prefill": run_prefill_case,
    }
    selected = cases if options.component == "all" else {options.component: cases[options.component]}
    for run_case in selected.values():
        run_case(remaining)


if __name__ == "__main__":
    main()
