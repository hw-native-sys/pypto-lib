# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GLM-5.3-Flash vision preprocessing, encoder and embedding fusion."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import math
import subprocess
from dataclasses import dataclass

import pypto.language as pl
import torch
import torch.nn.functional as F
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig

from models.glm5_3_flash.config import (
    HC_MULT, VISION_DEPTH, VISION_HEAD_DIM, VISION_HEADS, VISION_HIDDEN, VISION_INTER, VISION_MERGE,
    VISION_PATCH, VISION_PATCH_IN, VISION_PROJ_INTER, VISION_RMS_NORM_EPS, VISION_SWIGLU_LIMIT,
    VISION_TEMPORAL_PATCH, D,
)
from models.glm5_3_flash.elementwise import (
    accurate_divide, accurate_exp, accurate_reciprocal, accurate_row_sum_parts, accurate_rsqrt,
)
from models.glm5_3_flash.linear import bf16_linear, bf16_projection, golden_bf16_linear
from models.glm5_3_flash.rmsnorm import rms_norm, rmsnorm
from models.glm5_3_flash.utils import Checkpoint, component_hidden, component_options, vision_weight_layouts

T_DYN = pl.dynamic("GLM53_T_DYN")
VISION_ROWS_DYN = pl.dynamic("GLM53_VISION_ROWS_DYN")

VISION_STARTS_DYN = pl.dynamic("GLM53_VISION_STARTS")
VISION_POSITIONS_DYN = pl.dynamic("GLM53_VISION_POSITIONS")
ATTENTION_M_TILE = 16
ATTENTION_N_TILE = 64

VISION_MLP_WIDTH = pl.dynamic("GLM53_VISION_MLP_WIDTH")
ACT_M_TILE = 8
ACT_N_TILE = 128

FUSION_D_TILE = 128

RESIDUAL_M_TILE = 8
RESIDUAL_N_TILE = 128

MERGE_N_TILE = 128
MERGE_M_TILE = 8
LAYER_NORM_EPS = 1e-5


@pl.jit.inline
def vision_softmax_exp(
    x: pl.Tensor[[ATTENTION_M_TILE, ATTENTION_N_TILE], pl.FP32],
):
    """Use compensated exp in its supported range and retain underflow behavior."""
    bounded = pl.create_tensor([ATTENTION_M_TILE, ATTENTION_N_TILE], dtype=pl.FP32)
    bounded[:, :] = pl.maximum(x, -80.0)
    precise = accurate_exp(bounded)
    small = pl.exp(x)
    use_precise = pl.cast(pl.cast(pl.cmp(x, -80.0, cmp_type=5), pl.INT32), pl.FP32)
    use_small = pl.add(pl.neg(use_precise), 1.0)
    return pl.add(pl.mul(precise, use_precise), pl.mul(small, use_small))


@pl.jit.inline
def vision_axial_rope_table(
    position_ids: pl.Tensor[[T_DYN, 2], pl.INT32],
    cosine_table: pl.Tensor[[VISION_POSITIONS_DYN, VISION_HEAD_DIM // 4], pl.FP32],
    sine_table: pl.Tensor[[VISION_POSITIONS_DYN, VISION_HEAD_DIM // 4], pl.FP32],
    cos: pl.Tensor[[T_DYN, VISION_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[T_DYN, VISION_HEAD_DIM], pl.FP32],
):
    rows = pl.tensor.dim(position_ids, 0)
    for row in pl.spmd(rows, name_hint="vision_axial_rope"):
        height = pl.cast(pl.read(position_ids, [row, 0]), pl.INDEX)
        width = pl.cast(pl.read(position_ids, [row, 1]), pl.INDEX)
        height_cos = cosine_table[height : height + 1, :]
        width_cos = cosine_table[width : width + 1, :]
        height_sin = sine_table[height : height + 1, :]
        width_sin = sine_table[width : width + 1, :]
        for half in pl.unroll(2):
            col0 = half * (VISION_HEAD_DIM // 2)
            cos[row : row + 1, col0 : col0 + VISION_HEAD_DIM // 4] = height_cos
            cos[row : row + 1, col0 + VISION_HEAD_DIM // 4 : col0 + VISION_HEAD_DIM // 2] = width_cos
            sin[row : row + 1, col0 : col0 + VISION_HEAD_DIM // 4] = height_sin
            sin[row : row + 1, col0 + VISION_HEAD_DIM // 4 : col0 + VISION_HEAD_DIM // 2] = width_sin
    return cos, sin


@pl.jit.inline
def vision_attention(
    x: pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16],
    qkv_weight: pl.Tensor[[3 * VISION_HIDDEN, VISION_HIDDEN], pl.BF16],
    qkv_bias: pl.Tensor[[3 * VISION_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[VISION_HEAD_DIM], pl.BF16],
    k_norm_weight: pl.Tensor[[VISION_HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[T_DYN, VISION_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[T_DYN, VISION_HEAD_DIM], pl.FP32],
    proj_weight: pl.Tensor[[VISION_HIDDEN, VISION_HIDDEN], pl.BF16],
    proj_bias: pl.Tensor[[VISION_HIDDEN], pl.BF16],
    segment_start_loc: pl.Tensor[[VISION_STARTS_DYN], pl.INT32],
    output: pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16],
):
    rows = pl.tensor.dim(x, 0)
    qkv = pl.create_tensor([rows, 3 * VISION_HIDDEN], dtype=pl.BF16)
    bf16_projection(x, qkv_weight, qkv_bias, qkv, VISION_HIDDEN, 3 * VISION_HIDDEN, True)
    key_rows = pl.create_tensor([rows * VISION_HEADS, VISION_HEAD_DIM], dtype=pl.BF16)
    query_rows = pl.create_tensor([rows * VISION_HEADS, VISION_HEAD_DIM], dtype=pl.BF16)
    for block in pl.spmd(rows * VISION_HEADS, name_hint="vision_qk_rows"):
        split_row = block // VISION_HEADS
        split_head = block % VISION_HEADS
        split_col0 = split_head * VISION_HEAD_DIM
        query_rows[block:block + 1, :] = qkv[split_row:split_row + 1, split_col0:split_col0 + VISION_HEAD_DIM]
        key_rows[block : block + 1, :] = qkv[
            split_row : split_row + 1,
            VISION_HIDDEN + split_col0 : VISION_HIDDEN + split_col0 + VISION_HEAD_DIM,
        ]
    query_norm = pl.create_tensor([rows * VISION_HEADS, VISION_HEAD_DIM], dtype=pl.BF16)
    rmsnorm(query_rows, q_norm_weight, query_norm)
    key_norm = pl.create_tensor([rows * VISION_HEADS, VISION_HEAD_DIM], dtype=pl.BF16)
    rmsnorm(key_rows, k_norm_weight, key_norm)
    value = pl.create_tensor([VISION_HEADS, rows, VISION_HEAD_DIM], dtype=pl.BF16)
    key = pl.create_tensor([VISION_HEADS, rows, VISION_HEAD_DIM], dtype=pl.BF16)
    query = pl.create_tensor([VISION_HEADS, rows, VISION_HEAD_DIM], dtype=pl.BF16)
    for block in pl.spmd(rows * VISION_HEADS, name_hint="vision_qk_rope"):
        rope_row = block // VISION_HEADS
        rope_head = block % VISION_HEADS
        rope_col0 = rope_head * VISION_HEAD_DIM
        for half in pl.unroll(2):
            offset = half * (VISION_HEAD_DIM // 2)
            other_offset = (1 - half) * (VISION_HEAD_DIM // 2)
            q_part = pl.cast(query_norm[block : block + 1, offset : offset + VISION_HEAD_DIM // 2], pl.FP32)
            k_part = pl.cast(key_norm[block : block + 1, offset : offset + VISION_HEAD_DIM // 2], pl.FP32)
            q_other = pl.cast(query_norm[block:block + 1, other_offset:other_offset + VISION_HEAD_DIM // 2], pl.FP32)
            k_other = pl.cast(key_norm[block:block + 1, other_offset:other_offset + VISION_HEAD_DIM // 2], pl.FP32)
            cosine = cos[rope_row : rope_row + 1, offset : offset + VISION_HEAD_DIM // 2]
            sine = sin[rope_row : rope_row + 1, offset : offset + VISION_HEAD_DIM // 2]
            q_cos = pl.mul(q_part, cosine)
            k_cos = pl.mul(k_part, cosine)
            q_sin = pl.mul(q_other, sine)
            k_sin = pl.mul(k_other, sine)
            if half == 0:
                q_rotated = pl.sub(q_cos, q_sin)
                k_rotated = pl.sub(k_cos, k_sin)
            else:
                q_rotated = pl.add(q_cos, q_sin)
                k_rotated = pl.add(k_cos, k_sin)
            q_bf16 = pl.cast(q_rotated, pl.BF16, mode="rint")
            k_bf16 = pl.cast(k_rotated, pl.BF16, mode="rint")
            query[rope_head:rope_head + 1, rope_row:rope_row + 1, offset:offset + VISION_HEAD_DIM // 2] = pl.reshape(q_bf16, [1, 1, VISION_HEAD_DIM // 2])
            key[rope_head:rope_head + 1, rope_row:rope_row + 1, offset:offset + VISION_HEAD_DIM // 2] = pl.reshape(k_bf16, [1, 1, VISION_HEAD_DIM // 2])
        value_row = qkv[
            rope_row : rope_row + 1,
            2 * VISION_HIDDEN + rope_col0 : 2 * VISION_HIDDEN + rope_col0 + VISION_HEAD_DIM,
        ]
        value[rope_head:rope_head + 1, rope_row:rope_row + 1, :] = pl.reshape(value_row, [1, 1, VISION_HEAD_DIM])
    head_rows = rows * VISION_HEADS
    segments = pl.tensor.dim(segment_start_loc, 0) - 1
    value_flat = pl.reshape(value, [head_rows, VISION_HEAD_DIM])
    key_flat = pl.reshape(key, [head_rows, VISION_HEAD_DIM])
    query_flat = pl.reshape(query, [head_rows, VISION_HEAD_DIM])
    attended = pl.create_tensor([rows, VISION_HIDDEN], dtype=pl.BF16)
    for segment in pl.parallel(segments):
        begin = pl.cast(pl.read(segment_start_loc, [segment]), pl.INDEX)
        end = pl.cast(pl.read(segment_start_loc, [segment + 1]), pl.INDEX)
        query_tiles = (end - begin + ATTENTION_M_TILE - 1) // ATTENTION_M_TILE
        if end > begin:
            for block in pl.spmd(VISION_HEADS * query_tiles, name_hint="vision_noncausal_attention"):
                attention_head = block // query_tiles
                row0 = begin + block % query_tiles * ATTENTION_M_TILE
                valid_rows = pl.min(ATTENTION_M_TILE, end - row0)
                query_head_row0 = attention_head * rows + row0
                maximum = pl.full([1, ATTENTION_M_TILE], dtype=pl.FP32, value=-1e30)
                for key0 in pl.range(begin, end, ATTENTION_N_TILE):
                    valid_keys = pl.min(ATTENTION_N_TILE, end - key0)
                    key_head_row0 = attention_head * rows + key0
                    dot = bf16_linear(
                        query_flat, key_flat, query_head_row0, key_head_row0, valid_rows, valid_keys,
                        ATTENTION_M_TILE, ATTENTION_N_TILE, VISION_HEAD_DIM, True,
                    )
                    scores = pl.mul(dot, VISION_HEAD_DIM**-0.5)
                    scores_valid = pl.set_validshape(scores, ATTENTION_M_TILE, valid_keys)
                    scores_full_fillpad = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                    scores_full = pl.set_validshape(scores_full_fillpad, ATTENTION_M_TILE, ATTENTION_N_TILE)
                    block_maximum = pl.reshape(pl.row_max(scores_full), [1, ATTENTION_M_TILE])
                    updated_maximum = pl.maximum(maximum, block_maximum)
                    maximum = updated_maximum
                maximum_column = pl.reshape(maximum, [ATTENTION_M_TILE, 1])
                mass = pl.full([1, ATTENTION_M_TILE], dtype=pl.FP32, value=0.0)
                mass_error = pl.full([1, ATTENTION_M_TILE], dtype=pl.FP32, value=0.0)
                for mass_key0 in pl.range(begin, end, ATTENTION_N_TILE):
                    mass_valid_keys = pl.min(ATTENTION_N_TILE, end - mass_key0)
                    mass_key_row0 = attention_head * rows + mass_key0
                    mass_dot = bf16_linear(
                        query_flat, key_flat, query_head_row0, mass_key_row0, valid_rows, mass_valid_keys,
                        ATTENTION_M_TILE, ATTENTION_N_TILE, VISION_HEAD_DIM, True,
                    )
                    mass_scores = pl.mul(mass_dot, VISION_HEAD_DIM**-0.5)
                    mass_scores_valid = pl.set_validshape(mass_scores, ATTENTION_M_TILE, mass_valid_keys)
                    mass_padded = pl.fillpad(mass_scores_valid, pad_value=pl.PadValue.min)
                    mass_scores_full = pl.set_validshape(mass_padded, ATTENTION_M_TILE, ATTENTION_N_TILE)
                    mass_shifted = pl.row_expand_sub(mass_scores_full, maximum_column)
                    mass_argument = pl.create_tensor([ATTENTION_M_TILE, ATTENTION_N_TILE], dtype=pl.FP32)
                    mass_zero = pl.full([ATTENTION_M_TILE, ATTENTION_N_TILE], dtype=pl.FP32, value=0.0)
                    mass_argument[:, :] = pl.add(mass_zero, mass_shifted)
                    mass_exponentials = pl.create_tensor([ATTENTION_M_TILE, ATTENTION_N_TILE], dtype=pl.FP32)
                    mass_exponentials[:, :] = vision_softmax_exp(mass_argument)
                    mass_high, mass_low = accurate_row_sum_parts(mass_exponentials)
                    block_mass = pl.reshape(mass_high, [1, ATTENTION_M_TILE])
                    block_error = pl.reshape(mass_low, [1, ATTENTION_M_TILE])
                    updated_mass = pl.add(mass, block_mass)
                    recovered_mass = pl.sub(updated_mass, mass)
                    mass_left = pl.sub(mass, pl.sub(updated_mass, recovered_mass))
                    mass_right = pl.sub(block_mass, recovered_mass)
                    mass_error = pl.add(mass_error, pl.add(pl.add(mass_left, mass_right), block_error))
                    mass = updated_mass
                mass_buffer = pl.create_tensor([1, ATTENTION_M_TILE], dtype=pl.FP32)
                mass_buffer[:, :] = pl.add(mass, mass_error)
                inverse_mass_row = accurate_reciprocal(mass_buffer)
                inverse_mass = pl.reshape(inverse_mass_row, [ATTENTION_M_TILE, 1])
                context = pl.full([ATTENTION_M_TILE, VISION_HEAD_DIM], dtype=pl.FP32, value=0.0)
                context_error = pl.full([ATTENTION_M_TILE, VISION_HEAD_DIM], dtype=pl.FP32, value=0.0)
                for key0 in pl.range(begin, end, ATTENTION_N_TILE):
                    valid_keys = pl.min(ATTENTION_N_TILE, end - key0)
                    key_head_row0 = attention_head * rows + key0
                    dot = bf16_linear(
                        query_flat, key_flat, query_head_row0, key_head_row0, valid_rows, valid_keys,
                        ATTENTION_M_TILE, ATTENTION_N_TILE, VISION_HEAD_DIM, True,
                    )
                    scores = pl.mul(dot, VISION_HEAD_DIM**-0.5)
                    scores_valid = pl.set_validshape(scores, ATTENTION_M_TILE, valid_keys)
                    scores_full_fillpad_2 = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                    scores_full = pl.set_validshape(scores_full_fillpad_2, ATTENTION_M_TILE, ATTENTION_N_TILE)
                    probability_argument = pl.create_tensor([ATTENTION_M_TILE, ATTENTION_N_TILE], dtype=pl.FP32)
                    probability_shift = pl.row_expand_sub(scores_full, maximum_column)
                    probability_zero = pl.full([ATTENTION_M_TILE, ATTENTION_N_TILE], dtype=pl.FP32, value=0.0)
                    probability_argument[:, :] = pl.add(probability_zero, probability_shift)
                    exponential = vision_softmax_exp(probability_argument)
                    probabilities = pl.row_expand_mul(exponential, inverse_mass)
                    probabilities = pl.add(probability_zero, probabilities)
                    probabilities_bf16 = pl.cast(probabilities, pl.BF16, mode="rint")
                    probability_values = pl.cast(probabilities_bf16, pl.FP32)
                    for value_index in pl.range(valid_keys):
                        probability_index_zero = pl.full([ATTENTION_M_TILE, 8], dtype=pl.INT32, value=0)
                        probability_index = pl.add(probability_index_zero, pl.cast(value_index, pl.INT32))
                        probability_lanes = pl.gather(probability_values, dim=-1, index=probability_index)
                        probability_column = pl.row_max(probability_lanes)
                        value_part = value_flat[key_head_row0 + value_index:key_head_row0 + value_index + 1, :]
                        context_value_row = pl.cast(value_part, pl.FP32)
                        product_zero = pl.full([ATTENTION_M_TILE, VISION_HEAD_DIM], dtype=pl.FP32, value=0.0)
                        probability_rows = pl.row_expand(product_zero, probability_column)
                        partial_context = pl.col_expand_mul(probability_rows, context_value_row)
                        updated_context = pl.add(context, partial_context)
                        recovered_context = pl.sub(updated_context, context)
                        context_left = pl.sub(context, pl.sub(updated_context, recovered_context))
                        context_right = pl.sub(partial_context, recovered_context)
                        context_error = pl.add(context_error, pl.add(context_left, context_right))
                        context = updated_context
                context = pl.add(context, context_error)
                result = pl.cast(context, pl.BF16, mode="rint")
                attended[row0:row0 + ATTENTION_M_TILE, attention_head * VISION_HEAD_DIM:(attention_head + 1) * VISION_HEAD_DIM] = pl.set_validshape(result, valid_rows, VISION_HEAD_DIM)
    bf16_projection(attended, proj_weight, proj_bias, output, VISION_HIDDEN, VISION_HIDDEN, True)
    return output


@pl.jit
def vision_attention_test(
    x: pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16],
    qkv_weight: pl.Tensor[[3 * VISION_HIDDEN, VISION_HIDDEN], pl.BF16],
    qkv_bias: pl.Tensor[[3 * VISION_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[VISION_HEAD_DIM], pl.BF16],
    k_norm_weight: pl.Tensor[[VISION_HEAD_DIM], pl.BF16],
    proj_weight: pl.Tensor[[VISION_HIDDEN, VISION_HIDDEN], pl.BF16],
    proj_bias: pl.Tensor[[VISION_HIDDEN], pl.BF16],
    segment_start_loc: pl.Tensor[[VISION_STARTS_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN, 2], pl.INT32],
    cosine_table: pl.Tensor[[VISION_POSITIONS_DYN, VISION_HEAD_DIM // 4], pl.FP32],
    sine_table: pl.Tensor[[VISION_POSITIONS_DYN, VISION_HEAD_DIM // 4], pl.FP32],
    cos: pl.Out[pl.Tensor[[T_DYN, VISION_HEAD_DIM], pl.FP32]],
    sin: pl.Out[pl.Tensor[[T_DYN, VISION_HEAD_DIM], pl.FP32]],
    output: pl.Out[pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    cos.bind_dynamic(0, T_DYN)
    sin.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    segment_start_loc.bind_dynamic(0, VISION_STARTS_DYN)
    vision_axial_rope_table(position_ids, cosine_table, sine_table, cos, sin)
    vision_attention(
        x, qkv_weight, qkv_bias, q_norm_weight, k_norm_weight, cos, sin, proj_weight, proj_bias,
        segment_start_loc, output,
    )
    return cos, sin, output


@pl.jit.inline
def vision_swiglu(
    gate: pl.Tensor[[T_DYN, VISION_MLP_WIDTH], pl.BF16],
    up: pl.Tensor[[T_DYN, VISION_MLP_WIDTH], pl.BF16],
    output: pl.Tensor[[T_DYN, VISION_MLP_WIDTH], pl.BF16],
):
    rows = pl.tensor.dim(gate, 0)
    width = pl.tensor.dim(gate, 1)
    for block in pl.spmd((rows + ACT_M_TILE - 1) // ACT_M_TILE * (width // ACT_N_TILE), name_hint="vision_swiglu"):
        row0 = block // (width // ACT_N_TILE) * ACT_M_TILE
        col0 = block % (width // ACT_N_TILE) * ACT_N_TILE
        valid_rows = pl.min(ACT_M_TILE, rows - row0)
        gate_values = pl.slice(gate, [ACT_M_TILE, ACT_N_TILE], [row0, col0], valid_shape=[valid_rows, ACT_N_TILE])
        up_values = pl.slice(up, [ACT_M_TILE, ACT_N_TILE], [row0, col0], valid_shape=[valid_rows, ACT_N_TILE])
        gate_fp32 = pl.minimum(pl.cast(gate_values, pl.FP32), VISION_SWIGLU_LIMIT)
        up_fp32 = pl.minimum(pl.cast(up_values, pl.FP32), VISION_SWIGLU_LIMIT)
        up_fp32 = pl.maximum(up_fp32, -VISION_SWIGLU_LIMIT)
        gate_buffer = pl.create_tensor([ACT_M_TILE, ACT_N_TILE], dtype=pl.FP32)
        exponent = pl.create_tensor([ACT_M_TILE, ACT_N_TILE], dtype=pl.FP32)
        denominator = pl.create_tensor([ACT_M_TILE, ACT_N_TILE], dtype=pl.FP32)
        gate_buffer[:, :] = gate_fp32
        exponent[:, :] = pl.neg(gate_fp32)
        exponential = accurate_exp(exponent)
        denominator[:, :] = pl.add(exponential, 1.0)
        silu = accurate_divide(gate_buffer, denominator)
        silu_bf16 = pl.cast(silu, pl.BF16, mode="rint")
        product = pl.mul(pl.cast(silu_bf16, pl.FP32), up_fp32)
        result = pl.cast(product, pl.BF16, mode="rint")
        output[row0:row0 + ACT_M_TILE, col0:col0 + ACT_N_TILE] = pl.set_validshape(result, valid_rows, ACT_N_TILE)
    return output


@pl.jit.inline
def vision_mlp(
    x: pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16],
    w_gate: pl.Tensor[[VISION_INTER, VISION_HIDDEN], pl.BF16],
    b_gate: pl.Tensor[[VISION_INTER], pl.BF16],
    w_up: pl.Tensor[[VISION_INTER, VISION_HIDDEN], pl.BF16],
    b_up: pl.Tensor[[VISION_INTER], pl.BF16],
    w_down: pl.Tensor[[VISION_HIDDEN, VISION_INTER], pl.BF16],
    b_down: pl.Tensor[[VISION_HIDDEN], pl.BF16],
    output: pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16],
):
    rows = pl.tensor.dim(x, 0)
    gate = pl.create_tensor([rows, VISION_INTER], dtype=pl.BF16)
    bf16_projection(x, w_gate, b_gate, gate, VISION_HIDDEN, VISION_INTER, True)
    up = pl.create_tensor([rows, VISION_INTER], dtype=pl.BF16)
    bf16_projection(x, w_up, b_up, up, VISION_HIDDEN, VISION_INTER, True)
    activated = pl.create_tensor([rows, VISION_INTER], dtype=pl.BF16)
    vision_swiglu(gate, up, activated)
    bf16_projection(activated, w_down, b_down, output, VISION_INTER, VISION_HIDDEN, True)
    return output


@pl.jit
def vision_mlp_test(
    x: pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16],
    w_gate: pl.Tensor[[VISION_INTER, VISION_HIDDEN], pl.BF16],
    b_gate: pl.Tensor[[VISION_INTER], pl.BF16],
    w_up: pl.Tensor[[VISION_INTER, VISION_HIDDEN], pl.BF16],
    b_up: pl.Tensor[[VISION_INTER], pl.BF16],
    w_down: pl.Tensor[[VISION_HIDDEN, VISION_INTER], pl.BF16],
    b_down: pl.Tensor[[VISION_HIDDEN], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    vision_mlp(x, w_gate, b_gate, w_up, b_up, w_down, b_down, output)
    return output


@pl.jit.inline
def vision_fusion(
    vision_embeds: pl.Tensor[[VISION_ROWS_DYN, D], pl.BF16],
    placeholder_rows: pl.Tensor[[VISION_ROWS_DYN], pl.INT32],
    hidden_streams: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
):
    rows = pl.tensor.dim(vision_embeds, 0)
    for block in pl.spmd(rows * (D // FUSION_D_TILE), name_hint="vision_placeholder_scatter"):
        source_row = block // (D // FUSION_D_TILE)
        col0 = block % (D // FUSION_D_TILE) * FUSION_D_TILE
        target_row = pl.cast(pl.read(placeholder_rows, [source_row]), pl.INDEX)
        values = vision_embeds[source_row : source_row + 1, col0 : col0 + FUSION_D_TILE]
        stream_values = pl.reshape(values, [1, 1, FUSION_D_TILE])
        for lane in pl.unroll(HC_MULT):
            hidden_streams[target_row:target_row + 1, lane:lane + 1, col0:col0 + FUSION_D_TILE] = stream_values
    return hidden_streams


@pl.jit
def vision_fusion_test(
    vision_embeds: pl.Tensor[[VISION_ROWS_DYN, D], pl.BF16],
    placeholder_rows: pl.Tensor[[VISION_ROWS_DYN], pl.INT32],
    hidden_streams: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]],
):
    vision_embeds.bind_dynamic(0, VISION_ROWS_DYN)
    placeholder_rows.bind_dynamic(0, VISION_ROWS_DYN)
    hidden_streams.bind_dynamic(0, T_DYN)
    vision_fusion(vision_embeds, placeholder_rows, hidden_streams)
    return hidden_streams


@pl.jit.inline
def vision_patch_embed(
    patches: pl.Tensor[[T_DYN, VISION_PATCH_IN], pl.BF16],
    weight: pl.Tensor[[VISION_HIDDEN, VISION_PATCH_IN], pl.BF16],
    bias: pl.Tensor[[VISION_HIDDEN], pl.BF16],
    output: pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16],
):
    bf16_projection(patches, weight, bias, output, VISION_PATCH_IN, VISION_HIDDEN, True)
    return output


@pl.jit
def vision_patch_embed_test(
    patches: pl.Tensor[[T_DYN, VISION_PATCH_IN], pl.BF16],
    weight: pl.Tensor[[VISION_HIDDEN, VISION_PATCH_IN], pl.BF16],
    bias: pl.Tensor[[VISION_HIDDEN], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16]],
):
    patches.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    vision_patch_embed(patches, weight, bias, output)
    return output


@pl.jit
def vision_stem(
    patches: pl.Tensor[[T_DYN, VISION_PATCH_IN], pl.BF16],
    patch_weight: pl.Tensor[[VISION_HIDDEN, VISION_PATCH_IN], pl.BF16],
    patch_bias: pl.Tensor[[VISION_HIDDEN], pl.BF16],
    position_ids: pl.Tensor[[T_DYN, 2], pl.INT32],
    cosine_table: pl.Tensor[[VISION_POSITIONS_DYN, VISION_HEAD_DIM // 4], pl.FP32],
    sine_table: pl.Tensor[[VISION_POSITIONS_DYN, VISION_HEAD_DIM // 4], pl.FP32],
    hidden: pl.Out[pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16]],
    cosine: pl.Out[pl.Tensor[[T_DYN, VISION_HEAD_DIM], pl.FP32]],
    sine: pl.Out[pl.Tensor[[T_DYN, VISION_HEAD_DIM], pl.FP32]],
):
    vision_patch_embed(patches, patch_weight, patch_bias, hidden)
    vision_axial_rope_table(position_ids, cosine_table, sine_table, cosine, sine)
    return hidden, cosine, sine


@pl.jit
def vision_block(
    hidden: pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16],
    norm1_weight: pl.Tensor[[VISION_DEPTH, VISION_HIDDEN], pl.BF16],
    qkv_weight: pl.Tensor[[VISION_DEPTH, 3 * VISION_HIDDEN, VISION_HIDDEN], pl.BF16],
    qkv_bias: pl.Tensor[[VISION_DEPTH, 3 * VISION_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[VISION_DEPTH, VISION_HEAD_DIM], pl.BF16],
    k_norm_weight: pl.Tensor[[VISION_DEPTH, VISION_HEAD_DIM], pl.BF16],
    out_weight: pl.Tensor[[VISION_DEPTH, VISION_HIDDEN, VISION_HIDDEN], pl.BF16],
    out_bias: pl.Tensor[[VISION_DEPTH, VISION_HIDDEN], pl.BF16],
    norm2_weight: pl.Tensor[[VISION_DEPTH, VISION_HIDDEN], pl.BF16],
    gate_weight: pl.Tensor[[VISION_DEPTH, VISION_INTER, VISION_HIDDEN], pl.BF16],
    gate_bias: pl.Tensor[[VISION_DEPTH, VISION_INTER], pl.BF16],
    up_weight: pl.Tensor[[VISION_DEPTH, VISION_INTER, VISION_HIDDEN], pl.BF16],
    up_bias: pl.Tensor[[VISION_DEPTH, VISION_INTER], pl.BF16],
    down_weight: pl.Tensor[[VISION_DEPTH, VISION_HIDDEN, VISION_INTER], pl.BF16],
    down_bias: pl.Tensor[[VISION_DEPTH, VISION_HIDDEN], pl.BF16],
    cosine: pl.Tensor[[T_DYN, VISION_HEAD_DIM], pl.FP32],
    sine: pl.Tensor[[T_DYN, VISION_HEAD_DIM], pl.FP32],
    segment_start_loc: pl.Tensor[[VISION_STARTS_DYN], pl.INT32],
    output: pl.Out[pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16]],
    layer: pl.Scalar[pl.INT32],
):
    rows = pl.tensor.dim(hidden, 0)
    norm1 = norm1_weight[layer]
    normalized = pl.create_tensor([rows, VISION_HIDDEN], dtype=pl.BF16)
    rmsnorm(hidden, norm1, normalized)
    qkv_w = qkv_weight[layer]
    qkv_b = qkv_bias[layer]
    q_norm = q_norm_weight[layer]
    k_norm = k_norm_weight[layer]
    out_w = out_weight[layer]
    out_b = out_bias[layer]
    attended = pl.create_tensor([rows, VISION_HIDDEN], dtype=pl.BF16)
    vision_attention(normalized, qkv_w, qkv_b, q_norm, k_norm, cosine, sine, out_w, out_b, segment_start_loc, attended)
    residual = pl.create_tensor([rows, VISION_HIDDEN], dtype=pl.BF16)
    for attention_block in pl.spmd((rows + RESIDUAL_M_TILE - 1) // RESIDUAL_M_TILE * (VISION_HIDDEN // RESIDUAL_N_TILE), name_hint="vision_attention_residual"):
        attention_row0 = attention_block // (VISION_HIDDEN // RESIDUAL_N_TILE) * RESIDUAL_M_TILE
        attention_col0 = attention_block % (VISION_HIDDEN // RESIDUAL_N_TILE) * RESIDUAL_N_TILE
        attention_rows = pl.min(RESIDUAL_M_TILE, rows - attention_row0)
        previous = pl.slice(hidden, [RESIDUAL_M_TILE, RESIDUAL_N_TILE], [attention_row0, attention_col0], valid_shape=[attention_rows, RESIDUAL_N_TILE])
        update = pl.slice(attended, [RESIDUAL_M_TILE, RESIDUAL_N_TILE], [attention_row0, attention_col0], valid_shape=[attention_rows, RESIDUAL_N_TILE])
        combined = pl.add(pl.cast(previous, pl.FP32), pl.cast(update, pl.FP32))
        narrowed = pl.cast(combined, pl.BF16, mode="rint")
        residual[attention_row0:attention_row0 + RESIDUAL_M_TILE, attention_col0:attention_col0 + RESIDUAL_N_TILE] = pl.set_validshape(narrowed, attention_rows, RESIDUAL_N_TILE)
    norm2 = norm2_weight[layer]
    mlp_input = pl.create_tensor([rows, VISION_HIDDEN], dtype=pl.BF16)
    rmsnorm(residual, norm2, mlp_input)
    gate_w = gate_weight[layer]
    gate_b = gate_bias[layer]
    up_w = up_weight[layer]
    up_b = up_bias[layer]
    down_w = down_weight[layer]
    down_b = down_bias[layer]
    mlp_output = pl.create_tensor([rows, VISION_HIDDEN], dtype=pl.BF16)
    vision_mlp(mlp_input, gate_w, gate_b, up_w, up_b, down_w, down_b, mlp_output)
    for mlp_block in pl.spmd((rows + RESIDUAL_M_TILE - 1) // RESIDUAL_M_TILE * (VISION_HIDDEN // RESIDUAL_N_TILE), name_hint="vision_mlp_residual"):
        mlp_row0 = mlp_block // (VISION_HIDDEN // RESIDUAL_N_TILE) * RESIDUAL_M_TILE
        mlp_col0 = mlp_block % (VISION_HIDDEN // RESIDUAL_N_TILE) * RESIDUAL_N_TILE
        mlp_rows = pl.min(RESIDUAL_M_TILE, rows - mlp_row0)
        previous_mlp = pl.slice(residual, [RESIDUAL_M_TILE, RESIDUAL_N_TILE], [mlp_row0, mlp_col0], valid_shape=[mlp_rows, RESIDUAL_N_TILE])
        update_mlp = pl.slice(mlp_output, [RESIDUAL_M_TILE, RESIDUAL_N_TILE], [mlp_row0, mlp_col0], valid_shape=[mlp_rows, RESIDUAL_N_TILE])
        combined_mlp = pl.add(pl.cast(previous_mlp, pl.FP32), pl.cast(update_mlp, pl.FP32))
        narrowed_mlp = pl.cast(combined_mlp, pl.BF16, mode="rint")
        output[mlp_row0:mlp_row0 + RESIDUAL_M_TILE, mlp_col0:mlp_col0 + RESIDUAL_N_TILE] = pl.set_validshape(narrowed_mlp, mlp_rows, RESIDUAL_N_TILE)
    return output


@pl.jit
def vision_finish(
    hidden: pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16],
    post_norm_weight: pl.Tensor[[VISION_HIDDEN], pl.BF16],
    downsample_weight: pl.Tensor[[D, VISION_MERGE * VISION_MERGE * VISION_HIDDEN], pl.BF16],
    downsample_bias: pl.Tensor[[D], pl.BF16],
    merger_proj_weight: pl.Tensor[[D, D], pl.BF16],
    merger_norm_weight: pl.Tensor[[D], pl.BF16],
    merger_norm_bias: pl.Tensor[[D], pl.BF16],
    merger_gate_weight: pl.Tensor[[VISION_PROJ_INTER, D], pl.BF16],
    merger_up_weight: pl.Tensor[[VISION_PROJ_INTER, D], pl.BF16],
    merger_down_weight: pl.Tensor[[D, VISION_PROJ_INTER], pl.BF16],
    features: pl.Out[pl.Tensor[[VISION_ROWS_DYN, D], pl.BF16]],
):
    rows = pl.tensor.dim(hidden, 0)
    output_rows = pl.tensor.dim(features, 0)
    normalized = pl.create_tensor([rows, VISION_HIDDEN], dtype=pl.BF16)
    rmsnorm(hidden, post_norm_weight, normalized)
    grouped = pl.reshape(normalized, [output_rows, VISION_MERGE * VISION_MERGE, VISION_HIDDEN])
    vision_merger(
        grouped, downsample_weight, downsample_bias, merger_proj_weight, merger_norm_weight, merger_norm_bias,
        merger_gate_weight, merger_up_weight, merger_down_weight, features,
    )
    return features


@pl.jit.host
def vision_forward(
    patches: pl.Tensor[[1, T_DYN, VISION_PATCH_IN], pl.BF16],
    patch_weight: pl.Tensor[[1, VISION_HIDDEN, VISION_PATCH_IN], pl.BF16],
    patch_bias: pl.Tensor[[1, VISION_HIDDEN], pl.BF16],
    norm1_weight: pl.Tensor[[1, VISION_DEPTH, VISION_HIDDEN], pl.BF16],
    qkv_weight: pl.Tensor[[1, VISION_DEPTH, 3 * VISION_HIDDEN, VISION_HIDDEN], pl.BF16],
    qkv_bias: pl.Tensor[[1, VISION_DEPTH, 3 * VISION_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[1, VISION_DEPTH, VISION_HEAD_DIM], pl.BF16],
    k_norm_weight: pl.Tensor[[1, VISION_DEPTH, VISION_HEAD_DIM], pl.BF16],
    out_weight: pl.Tensor[[1, VISION_DEPTH, VISION_HIDDEN, VISION_HIDDEN], pl.BF16],
    out_bias: pl.Tensor[[1, VISION_DEPTH, VISION_HIDDEN], pl.BF16],
    norm2_weight: pl.Tensor[[1, VISION_DEPTH, VISION_HIDDEN], pl.BF16],
    gate_weight: pl.Tensor[[1, VISION_DEPTH, VISION_INTER, VISION_HIDDEN], pl.BF16],
    gate_bias: pl.Tensor[[1, VISION_DEPTH, VISION_INTER], pl.BF16],
    up_weight: pl.Tensor[[1, VISION_DEPTH, VISION_INTER, VISION_HIDDEN], pl.BF16],
    up_bias: pl.Tensor[[1, VISION_DEPTH, VISION_INTER], pl.BF16],
    down_weight: pl.Tensor[[1, VISION_DEPTH, VISION_HIDDEN, VISION_INTER], pl.BF16],
    down_bias: pl.Tensor[[1, VISION_DEPTH, VISION_HIDDEN], pl.BF16],
    post_norm_weight: pl.Tensor[[1, VISION_HIDDEN], pl.BF16],
    downsample_weight: pl.Tensor[[1, D, VISION_MERGE * VISION_MERGE * VISION_HIDDEN], pl.BF16],
    downsample_bias: pl.Tensor[[1, D], pl.BF16],
    merger_proj_weight: pl.Tensor[[1, D, D], pl.BF16],
    merger_norm_weight: pl.Tensor[[1, D], pl.BF16],
    merger_norm_bias: pl.Tensor[[1, D], pl.BF16],
    merger_gate_weight: pl.Tensor[[1, VISION_PROJ_INTER, D], pl.BF16],
    merger_up_weight: pl.Tensor[[1, VISION_PROJ_INTER, D], pl.BF16],
    merger_down_weight: pl.Tensor[[1, D, VISION_PROJ_INTER], pl.BF16],
    position_ids: pl.Tensor[[1, T_DYN, 2], pl.INT32],
    segment_start_loc: pl.Tensor[[1, VISION_STARTS_DYN], pl.INT32],
    cosine_table: pl.Tensor[[1, VISION_POSITIONS_DYN, VISION_HEAD_DIM // 4], pl.FP32],
    sine_table: pl.Tensor[[1, VISION_POSITIONS_DYN, VISION_HEAD_DIM // 4], pl.FP32],
    features: pl.Out[pl.Tensor[[1, VISION_ROWS_DYN, D], pl.BF16]],
    workspace_a: pl.InOut[pl.Tensor[[1, T_DYN, VISION_HIDDEN], pl.BF16]],
    workspace_b: pl.InOut[pl.Tensor[[1, T_DYN, VISION_HIDDEN], pl.BF16]],
    cosine: pl.Out[pl.Tensor[[1, T_DYN, VISION_HEAD_DIM], pl.FP32]],
    sine: pl.Out[pl.Tensor[[1, T_DYN, VISION_HEAD_DIM], pl.FP32]],
    num_layers: pl.Scalar[pl.INT32],
):
    patches.bind_dynamic(1, T_DYN)
    features.bind_dynamic(1, VISION_ROWS_DYN)
    workspace_a.bind_dynamic(1, T_DYN)
    workspace_b.bind_dynamic(1, T_DYN)
    position_ids.bind_dynamic(1, T_DYN)
    segment_start_loc.bind_dynamic(1, VISION_STARTS_DYN)
    cosine.bind_dynamic(1, T_DYN)
    sine.bind_dynamic(1, T_DYN)
    cosine_table.bind_dynamic(1, VISION_POSITIONS_DYN)
    sine_table.bind_dynamic(1, VISION_POSITIONS_DYN)
    vision_stem(
        patches[0], patch_weight[0], patch_bias[0], position_ids[0], cosine_table[0], sine_table[0],
        workspace_a[0], cosine[0], sine[0], device=0,
    )
    for layer in pl.range(num_layers):
        if layer % 2 == 0:
            vision_block(
                workspace_a[0], norm1_weight[0],
                qkv_weight[0], qkv_bias[0], q_norm_weight[0], k_norm_weight[0], out_weight[0], out_bias[0],
                norm2_weight[0], gate_weight[0], gate_bias[0], up_weight[0], up_bias[0], down_weight[0],
                down_bias[0], cosine[0], sine[0], segment_start_loc[0], workspace_b[0], layer, device=0,
            )
        else:
            vision_block(
                workspace_b[0], norm1_weight[0],
                qkv_weight[0], qkv_bias[0], q_norm_weight[0], k_norm_weight[0], out_weight[0], out_bias[0],
                norm2_weight[0], gate_weight[0], gate_bias[0], up_weight[0], up_bias[0], down_weight[0],
                down_bias[0], cosine[0], sine[0], segment_start_loc[0], workspace_a[0], layer, device=0,
            )
    if num_layers % 2 == 0:
        vision_finish(
            workspace_a[0], post_norm_weight[0], downsample_weight[0], downsample_bias[0],
            merger_proj_weight[0], merger_norm_weight[0], merger_norm_bias[0], merger_gate_weight[0],
            merger_up_weight[0], merger_down_weight[0], features[0], device=0,
        )
    else:
        vision_finish(
            workspace_b[0], post_norm_weight[0], downsample_weight[0], downsample_bias[0],
            merger_proj_weight[0], merger_norm_weight[0], merger_norm_bias[0], merger_gate_weight[0],
            merger_up_weight[0], merger_down_weight[0], features[0], device=0,
        )


@pl.jit.inline
def vision_merger_activation(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    norm_weight: pl.Tensor[[D], pl.BF16],
    norm_bias: pl.Tensor[[D], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
):
    rows = pl.tensor.dim(x, 0)
    for block in pl.spmd((rows + MERGE_M_TILE - 1) // MERGE_M_TILE, name_hint="vision_merger_layernorm_gelu"):
        row0 = block * MERGE_M_TILE
        valid_rows = pl.min(MERGE_M_TILE, rows - row0)
        total = pl.full([1, MERGE_M_TILE], dtype=pl.FP32, value=0.0)
        for col0 in pl.range(0, D, MERGE_N_TILE):
            source = pl.slice(x, [MERGE_M_TILE, MERGE_N_TILE], [row0, col0], valid_shape=[valid_rows, MERGE_N_TILE])
            value = pl.cast(source, pl.FP32)
            partial_sum = pl.reshape(pl.row_sum(value), [1, MERGE_M_TILE])
            total = pl.add(total, partial_sum)
        mean_row = pl.mul(total, 1.0 / D)
        mean = pl.reshape(mean_row, [MERGE_M_TILE, 1])
        squared_total = pl.full([1, MERGE_M_TILE], dtype=pl.FP32, value=0.0)
        for col0 in pl.range(0, D, MERGE_N_TILE):
            source = pl.slice(x, [MERGE_M_TILE, MERGE_N_TILE], [row0, col0], valid_shape=[valid_rows, MERGE_N_TILE])
            centered = pl.row_expand_sub(pl.cast(source, pl.FP32), mean)
            partial_square_sum = pl.reshape(pl.row_sum(pl.mul(centered, centered)), [1, MERGE_M_TILE])
            squared_total = pl.add(squared_total, partial_square_sum)
        variance = pl.add(pl.mul(squared_total, 1.0 / D), LAYER_NORM_EPS)
        variance_buffer = pl.create_tensor([1, MERGE_M_TILE], dtype=pl.FP32)
        variance_buffer[:, :] = variance
        inverse_row = accurate_rsqrt(variance_buffer)
        inverse = pl.reshape(inverse_row, [MERGE_M_TILE, 1])
        for col0 in pl.range(0, D, MERGE_N_TILE):
            source = pl.slice(x, [MERGE_M_TILE, MERGE_N_TILE], [row0, col0], valid_shape=[valid_rows, MERGE_N_TILE])
            centered = pl.row_expand_sub(pl.cast(source, pl.FP32), mean)
            unit = pl.row_expand_mul(centered, inverse)
            gamma = pl.reshape(pl.cast(norm_weight[col0 : col0 + MERGE_N_TILE], pl.FP32), [1, MERGE_N_TILE])
            beta = pl.reshape(pl.cast(norm_bias[col0 : col0 + MERGE_N_TILE], pl.FP32), [1, MERGE_N_TILE])
            affine = pl.col_expand_add(pl.col_expand_mul(unit, gamma), beta)
            normalized = pl.cast(affine, pl.BF16, mode="rint")
            normalized_fp32 = pl.cast(normalized, pl.FP32)
            z = pl.mul(normalized_fp32, 0.7071067811865476)
            gelu_denominator = pl.create_tensor([MERGE_M_TILE, MERGE_N_TILE], dtype=pl.FP32)
            gelu_denominator[:, :] = pl.add(pl.mul(pl.abs(z), 0.3275911), 1.0)
            t = accurate_reciprocal(gelu_denominator)
            polynomial = pl.add(pl.mul(t, 1.061405429), -1.453152027)
            polynomial = pl.add(pl.mul(t, polynomial), 1.421413741)
            polynomial = pl.add(pl.mul(t, polynomial), -0.284496736)
            polynomial = pl.add(pl.mul(t, polynomial), 0.254829592)
            gaussian = pl.exp(pl.neg(pl.mul(z, z)))
            erf_error = pl.mul(pl.mul(t, polynomial), gaussian)
            erf_absolute = pl.add(pl.neg(erf_error), 1.0)
            positive = pl.cast(pl.cmp(z, 0.0, cmp_type=5), pl.INT32)
            sign = pl.add(pl.mul(pl.cast(positive, pl.FP32), 2.0), -1.0)
            erf_value = pl.mul(sign, erf_absolute)
            gelu = pl.mul(pl.mul(normalized_fp32, 0.5), pl.add(erf_value, 1.0))
            result = pl.cast(gelu, pl.BF16, mode="rint")
            output[row0:row0 + MERGE_M_TILE, col0:col0 + MERGE_N_TILE] = pl.set_validshape(result, valid_rows, MERGE_N_TILE)
    return output


@pl.jit.inline
def vision_merger(
    x: pl.Tensor[[T_DYN, VISION_MERGE * VISION_MERGE, VISION_HIDDEN], pl.BF16],
    downsample_weight: pl.Tensor[[D, VISION_MERGE * VISION_MERGE * VISION_HIDDEN], pl.BF16],
    downsample_bias: pl.Tensor[[D], pl.BF16],
    proj_weight: pl.Tensor[[D, D], pl.BF16],
    post_norm_weight: pl.Tensor[[D], pl.BF16],
    post_norm_bias: pl.Tensor[[D], pl.BF16],
    w_gate: pl.Tensor[[VISION_PROJ_INTER, D], pl.BF16],
    w_up: pl.Tensor[[VISION_PROJ_INTER, D], pl.BF16],
    w_down: pl.Tensor[[D, VISION_PROJ_INTER], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
):
    rows = pl.tensor.dim(x, 0)
    packed_channels = pl.create_tensor([rows * VISION_HIDDEN, VISION_MERGE * VISION_MERGE], dtype=pl.BF16)
    for block in pl.spmd(rows * (VISION_HIDDEN // MERGE_N_TILE), name_hint="vision_merge_channel_order"):
        row = block // (VISION_HIDDEN // MERGE_N_TILE)
        col0 = block % (VISION_HIDDEN // MERGE_N_TILE) * MERGE_N_TILE
        spatial_rows = pl.full([16, MERGE_N_TILE], dtype=pl.BF16, value=0.0)
        for position in pl.unroll(VISION_MERGE * VISION_MERGE):
            spatial = x[row : row + 1, position : position + 1, col0 : col0 + MERGE_N_TILE]
            spatial_row = pl.reshape(spatial, [1, MERGE_N_TILE])
            spatial_rows[position : position + 1, :] = spatial_row
        channel_major = pl.transpose(spatial_rows, axis1=0, axis2=1)
        valid_channels = pl.set_validshape(channel_major, MERGE_N_TILE, VISION_MERGE * VISION_MERGE)
        packed_channels[row * VISION_HIDDEN + col0:row * VISION_HIDDEN + col0 + MERGE_N_TILE, :] = valid_channels
    downsampled = pl.create_tensor([rows, D], dtype=pl.BF16)
    packed = pl.reshape(packed_channels, [rows, VISION_MERGE * VISION_MERGE * VISION_HIDDEN])
    bf16_projection(
        packed, downsample_weight, downsample_bias, downsampled, VISION_HIDDEN * VISION_MERGE * VISION_MERGE,
        D, True,
    )
    projected = pl.create_tensor([rows, D], dtype=pl.BF16)
    bf16_projection(downsampled, proj_weight, post_norm_weight, projected, D, D, False)
    activated = pl.create_tensor([rows, D], dtype=pl.BF16)
    vision_merger_activation(projected, post_norm_weight, post_norm_bias, activated)
    gate = pl.create_tensor([rows, VISION_PROJ_INTER], dtype=pl.BF16)
    bf16_projection(activated, w_gate, post_norm_weight, gate, D, VISION_PROJ_INTER, False)
    up = pl.create_tensor([rows, VISION_PROJ_INTER], dtype=pl.BF16)
    bf16_projection(activated, w_up, post_norm_weight, up, D, VISION_PROJ_INTER, False)
    intermediate = pl.create_tensor([rows, VISION_PROJ_INTER], dtype=pl.BF16)
    vision_swiglu(gate, up, intermediate)
    bf16_projection(intermediate, w_down, post_norm_weight, output, VISION_PROJ_INTER, D, False)
    return output


@pl.jit
def vision_merger_test(
    x: pl.Tensor[[T_DYN, VISION_MERGE * VISION_MERGE, VISION_HIDDEN], pl.BF16],
    downsample_weight: pl.Tensor[[D, VISION_MERGE * VISION_MERGE * VISION_HIDDEN], pl.BF16],
    downsample_bias: pl.Tensor[[D], pl.BF16],
    proj_weight: pl.Tensor[[D, D], pl.BF16],
    post_norm_weight: pl.Tensor[[D], pl.BF16],
    post_norm_bias: pl.Tensor[[D], pl.BF16],
    w_gate: pl.Tensor[[VISION_PROJ_INTER, D], pl.BF16],
    w_up: pl.Tensor[[VISION_PROJ_INTER, D], pl.BF16],
    w_down: pl.Tensor[[D, VISION_PROJ_INTER], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    vision_merger(
        x, downsample_weight, downsample_bias, proj_weight, post_norm_weight, post_norm_bias,
        w_gate, w_up, w_down,
        output,
    )
    return output


def golden_vision_attention(
    x: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    segment_start_loc: torch.Tensor,
) -> torch.Tensor:
    projected = (golden_bf16_linear(x, qkv_weight) + qkv_bias.float()).bfloat16()
    query, key, value = projected.reshape(-1, 3, VISION_HEADS, VISION_HEAD_DIM).unbind(dim=1)
    query = rms_norm(query, q_norm_weight, VISION_RMS_NORM_EPS)
    key = rms_norm(key, k_norm_weight, VISION_RMS_NORM_EPS)
    half = VISION_HEAD_DIM // 2

    def rotate(values):
        value_fp32 = values.float()
        rotated = torch.cat((-value_fp32[..., half:], value_fp32[..., :half]), dim=-1)
        return (value_fp32 * cos[:, None] + rotated * sin[:, None]).bfloat16()

    query, key = rotate(query), rotate(key)
    attended = torch.empty_like(x)
    for begin, end in zip(segment_start_loc[:-1].tolist(), segment_start_loc[1:].tolist()):
        if begin == end:
            continue
        q = query[begin:end].transpose(0, 1).double()
        k = key[begin:end].transpose(0, 1).double()
        v = value[begin:end].transpose(0, 1).double()
        scores = (q @ k.transpose(-1, -2)).float() * VISION_HEAD_DIM**-0.5
        shifted = scores - scores.amax(dim=-1, keepdim=True)
        exponentials = shifted.double().exp().float()
        mass = exponentials.double().sum(dim=-1, keepdim=True).float()
        inverse = mass.double().reciprocal().float()
        probabilities = (exponentials * inverse).bfloat16()
        context = (probabilities.double() @ v).float().bfloat16()
        attended[begin:end] = context.transpose(0, 1).reshape(end - begin, VISION_HIDDEN)
    return (golden_bf16_linear(attended, proj_weight) + proj_bias.float()).bfloat16()


def vision_rope_tables(position_ids, inv_freq):
    """Prepare position-only axial RoPE constants in the checkpoint's FP32 arithmetic."""
    positions = torch.as_tensor(position_ids, device="cpu")
    frequencies = torch.as_tensor(inv_freq, dtype=torch.float32, device="cpu")
    if positions.ndim != 2 or positions.shape[1] != 2 or positions.numel() == 0:
        raise ValueError("vision positions must have shape [patches, 2]")
    if positions.is_floating_point() or (positions < 0).any():
        raise ValueError("vision positions must be nonnegative integers")
    if frequencies.shape != (VISION_HEAD_DIM // 4,) or not torch.isfinite(frequencies).all():
        raise ValueError("vision frequencies must be a finite head-quarter vector")
    extent = int(positions.max()) + 1
    angles = torch.arange(extent, dtype=torch.float32)[:, None] * frequencies[None, :]
    return angles.cos(), angles.sin()


@dataclass(frozen=True)
class VisualInput:
    pixel_values: torch.Tensor
    grid_thw: torch.Tensor
    marker: str
    prompt_fragment: str


def _canvas(frames, height, width, minimum, maximum):
    factor = VISION_PATCH * VISION_MERGE
    unit = VISION_TEMPORAL_PATCH * factor * factor
    aligned_frames = max(VISION_TEMPORAL_PATCH, math.ceil(frames / VISION_TEMPORAL_PATCH) * VISION_TEMPORAL_PATCH)

    def align(value):
        return math.ceil(value / factor) * factor

    h, w = align(height), align(width)
    if aligned_frames * h * w < minimum * unit:
        scale = math.sqrt(minimum * unit / (aligned_frames * height * width))
        h, w = align(math.ceil(height * scale)), align(math.ceil(width * scale))
    if aligned_frames * h * w > maximum * unit:
        if aligned_frames * factor * factor > maximum * unit:
            raise ValueError("visual token budget cannot fit one spatial merge group per temporal patch")
        low, high = 1, height
        h, w = factor, factor
        while low <= high:
            candidate = (low + high) // 2
            ch, cw = align(candidate), align(max(1, math.floor(width * candidate / height)))
            if aligned_frames * ch * cw <= maximum * unit:
                h, w, low = ch, cw, candidate + 1
            else:
                high = candidate - 1
    return h, w


def _rgb_tensor(image):
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
    else:
        import numpy as np
        from PIL import Image

        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                array = np.array(source.convert("RGB"), copy=True)
        else:
            array = np.array(image.convert("RGB"), copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1)
    if tensor.ndim != 3 or tensor.shape[0] != 3 or tensor.dtype != torch.uint8:
        raise ValueError("visual inputs must be RGB uint8 [3, height, width] tensors or Pillow images")
    if min(tensor.shape[1:]) <= 0:
        raise ValueError("visual dimensions must be positive")
    return tensor.contiguous()


class VisionProcessor:
    """Apply the checkpoint's fixed patch geometry, resize budget and CLIP normalization."""

    def __init__(self, checkpoint):
        root = Path(checkpoint.root if hasattr(checkpoint, "root") else checkpoint)
        config = json.loads((root / "processor_config.json").read_text())
        self.config = {}
        for kind in ("image", "video"):
            value = config[kind + "_processor"]
            for name, expected in (
                ("patch_size", VISION_PATCH),
                ("merge_size", VISION_MERGE),
                ("temporal_patch_size", VISION_TEMPORAL_PATCH),
                ("patch_expand_factor", 1),
            ):
                if value.get(name) != expected:
                    raise ValueError(f"{kind}_processor.{name} must be {expected}")
            if value.get("do_rescale", True) is not True or value.get("do_normalize", True) is not True:
                raise ValueError("vision preprocessing requires rescaling and CLIP normalization")
            if value.get("do_resize", True) is not True:
                raise ValueError("vision preprocessing requires the configured resize budget")
            self.config[kind] = value

    def _pixels(self, frames, kind):
        config = self.config[kind]
        count, _, height, width = frames.shape
        minimum, maximum = int(config["min_image_tokens"]), int(config["max_image_tokens"])
        if minimum <= 0 or maximum < minimum:
            raise ValueError("invalid visual token limits")
        budget_frames = math.ceil(count / VISION_TEMPORAL_PATCH) * VISION_TEMPORAL_PATCH
        target_h, target_w = _canvas(budget_frames, height, width, minimum, maximum)
        scale = min(target_h / height, target_w / width)
        if (
            budget_frames * height * width
            >= VISION_TEMPORAL_PATCH * (VISION_PATCH * VISION_MERGE) ** 2 * minimum
        ):
            scale = min(1.0, scale)
        content_h = max(1, min(target_h, math.floor(height * scale)))
        content_w = max(1, min(target_w, math.floor(width * scale)))
        if (content_h, content_w) != (height, width):
            frames = F.interpolate(
                frames, size=(content_h, content_w), mode="bicubic", align_corners=False, antialias=True,
            )
        frames = F.pad(frames, (0, target_w - content_w, 0, target_h - content_h))
        rescale_factor = float(config.get("rescale_factor", 1 / 255))
        if not math.isfinite(rescale_factor) or rescale_factor <= 0:
            raise ValueError("visual rescaling must be positive and finite")
        mean = torch.tensor(config["image_mean"], dtype=torch.float32).reshape(1, 3, 1, 1)
        std = torch.tensor(config["image_std"], dtype=torch.float32).reshape(1, 3, 1, 1)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("visual normalization must have finite means and positive finite standard deviations")
        pixel_scale = 1.0 / rescale_factor
        pixels = (frames.float() - mean * pixel_scale) / (std * pixel_scale)
        if kind == "image":
            pixels = pixels.expand(VISION_TEMPORAL_PATCH, -1, -1, -1)
        elif count % VISION_TEMPORAL_PATCH:
            pixels = torch.cat((pixels, pixels[-1:]), dim=0)
        grid_t = pixels.shape[0] // VISION_TEMPORAL_PATCH
        grid_h, grid_w = target_h // VISION_PATCH, target_w // VISION_PATCH
        grouped = pixels.reshape(
            grid_t, VISION_TEMPORAL_PATCH, 3, grid_h // VISION_MERGE, VISION_MERGE, VISION_PATCH,
            grid_w // VISION_MERGE, VISION_MERGE, VISION_PATCH,
        )
        ordered = grouped.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        patches = ordered.reshape(grid_t * grid_h * grid_w, -1).contiguous()
        return patches, torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.int64)

    def image(self, image):
        pixels, grid = self._pixels(_rgb_tensor(image).unsqueeze(0), "image")
        tokens = int(grid.prod()) // VISION_MERGE**2
        fragment = "<|begin_of_image|>" + "<|image|>" * tokens + "<|end_of_image|>"
        return VisualInput(pixels, grid, "<|image|>", fragment)

    def video(self, frames, timestamps):
        """Process already sampled RGB frames and their timestamps in seconds."""
        frames = torch.stack([_rgb_tensor(frame) for frame in frames])
        times = torch.as_tensor(timestamps, dtype=torch.float64)
        if times.shape != (frames.shape[0],) or not torch.isfinite(times).all():
            raise ValueError("provide one finite timestamp for every sampled video frame")
        if (times < 0).any() or (times[1:] < times[:-1]).any():
            raise ValueError("video timestamps must be nonnegative and monotone")
        pixels, grid = self._pixels(frames, "video")
        tokens = int(grid[0, 1:].prod()) // VISION_MERGE**2
        fragments = [
            f"<|begin_of_image|>{'<|image|>' * tokens}<|end_of_image|>{float(time):.1f} seconds"
            for time in times[::VISION_TEMPORAL_PATCH]
        ]
        fragment = "<|begin_of_video|>" + "".join(fragments) + "<|end_of_video|>"
        return VisualInput(pixels, grid, "<|video|>", fragment)

    def video_file(self, path):
        """Decode a video with FFmpeg using the reference processor's timestamp sampling."""
        import shutil
        import subprocess
        from fractions import Fraction

        import numpy as np

        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise RuntimeError("video files require ffmpeg and ffprobe; sampled RGB frames can use video()")
        path = Path(path).resolve(strict=True)
        command = [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration:format=duration",
            "-of", "json", str(path),
        ]
        metadata = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
        if not metadata.get("streams"):
            raise ValueError("video has no decodable visual stream")
        stream = metadata["streams"][0]
        source_fps = float(Fraction(stream["avg_frame_rate"]))
        if source_fps <= 0 or not math.isfinite(source_fps):
            raise ValueError("video needs a positive finite frame rate")
        count = stream.get("nb_frames")
        if count in (None, "N/A"):
            command = [
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                "-show_entries", "stream=nb_read_frames", "-of", "json", str(path),
            ]
            counted = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
            count = counted["streams"][0]["nb_read_frames"]
        count = int(count)
        if count <= 0:
            raise ValueError("video must contain at least one frame")
        duration = stream.get("duration", metadata.get("format", {}).get("duration"))
        duration = float(duration) if duration not in (None, "N/A") else round((count - 1) / source_fps) + 1
        target_fps = float(self.config["video"].get("fps", 2))
        if not math.isfinite(duration) or duration <= 0 or not math.isfinite(target_fps) or target_fps <= 0:
            raise ValueError("video duration and sampling rate must be positive and finite")
        requested = max(1, min(int(duration * target_fps), 2048))
        if count < requested:
            indices = np.linspace(0, count - 1, requested, dtype=int).tolist()
        else:
            indices, next_time = [], 0.0
            for index in range(count):
                if index / source_fps >= next_time:
                    indices.append(index)
                    next_time += 1 / target_fps
                    if next_time >= int(duration):
                        break
            if len(indices) < requested:
                begin, end = (indices[0], indices[-1]) if indices else (0, count - 1)
                indices = np.linspace(begin, end, requested, dtype=int).tolist()
            elif len(indices) > requested:
                indices = np.linspace(0, count - 1, requested, dtype=int).tolist()
        indices = list(dict.fromkeys(indices))
        if len(indices) % VISION_TEMPORAL_PATCH:
            indices.append(indices[-1])
        unique = sorted(set(indices))
        selection = "+".join(f"eq(n\\,{index})" for index in unique)
        command = [
            "ffmpeg", "-nostdin", "-v", "error", "-noautorotate", "-i", str(path),
            "-map", "0:v:0", "-vf", "select=" + selection,
            "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]
        decoded = subprocess.run(command, check=True, capture_output=True).stdout
        width, height = int(stream["width"]), int(stream["height"])
        frame_bytes = width * height * 3
        if len(decoded) != len(unique) * frame_bytes:
            raise ValueError("decoded video frame count does not match the sampled timeline")
        frames = torch.from_numpy(np.frombuffer(decoded, dtype=np.uint8).copy())
        frames = frames.reshape(len(unique), height, width, 3).permute(0, 3, 1, 2)
        lookup = dict(zip(unique, frames))
        return self.video([lookup[index] for index in indices], [index / source_fps for index in indices])


def golden_vision_mlp(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    b_gate: torch.Tensor,
    w_up: torch.Tensor,
    b_up: torch.Tensor,
    w_down: torch.Tensor,
    b_down: torch.Tensor,
) -> torch.Tensor:
    gate = (golden_bf16_linear(x, w_gate) + b_gate.float()).bfloat16().clamp(max=VISION_SWIGLU_LIMIT)
    up = (golden_bf16_linear(x, w_up) + b_up.float()).bfloat16()
    up = up.clamp(min=-VISION_SWIGLU_LIMIT, max=VISION_SWIGLU_LIMIT)
    activated = (F.silu(gate) * up).bfloat16()
    return (golden_bf16_linear(activated, w_down) + b_down.float()).bfloat16()


def golden_vision_fusion(
    inputs_embeds: torch.Tensor,
    vision_embeds: torch.Tensor,
    placeholder_mask: torch.Tensor,
) -> torch.Tensor:
    if placeholder_mask.dtype != torch.bool or placeholder_mask.shape != inputs_embeds.shape[:1]:
        raise ValueError("placeholder mask must be boolean with one entry per text row")
    if int(placeholder_mask.sum()) != vision_embeds.shape[0]:
        raise ValueError("placeholder count does not match the merged vision token count")
    fused = inputs_embeds.clone()
    fused[placeholder_mask] = vision_embeds.to(inputs_embeds.dtype)
    return fused


def golden_vision_patch_embed(
    patches: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return (golden_bf16_linear(patches, weight) + bias.float()).bfloat16()


def vision_metadata(grid_thw):
    """Build merge-group patch positions and one attention segment per temporal patch."""
    grid = torch.as_tensor(grid_thw, device="cpu")
    if grid.ndim != 2 or grid.shape[1] != 3 or grid.shape[0] == 0:
        raise ValueError("grid_thw must have shape [images_or_videos, 3]")
    if grid.is_floating_point() and not torch.equal(grid, grid.round()):
        raise ValueError("grid_thw must contain integers")
    grid = grid.long()
    if (grid <= 0).any() or (grid[:, 1:] % VISION_MERGE).any():
        raise ValueError("grid dimensions must be positive with spatial sizes divisible by the merge factor")
    positions, starts = [], [0]
    for frames, height, width in grid.tolist():
        h = torch.arange(height)[:, None].expand(height, width)
        w = torch.arange(width)[None, :].expand(height, width)
        shape = (height // VISION_MERGE, VISION_MERGE, width // VISION_MERGE, VISION_MERGE)
        h = h.reshape(shape).permute(0, 2, 1, 3).reshape(-1)
        w = w.reshape(shape).permute(0, 2, 1, 3).reshape(-1)
        frame_positions = torch.stack((h, w), dim=-1).int()
        for _ in range(frames):
            positions.append(frame_positions)
            starts.append(starts[-1] + height * width)
    return torch.cat(positions), torch.tensor(starts, dtype=torch.int32)


def golden_vision_single(values):
    hidden = golden_vision_patch_embed(values["patches"], values["patch_weight"], values["patch_bias"])
    values["workspace_a"].copy_(hidden)
    height, width = values["position_ids"].long().unbind(dim=-1)
    cosine_hw = torch.cat((values["cosine_table"][height], values["cosine_table"][width]), dim=-1)
    sine_hw = torch.cat((values["sine_table"][height], values["sine_table"][width]), dim=-1)
    cosine, sine = cosine_hw.repeat(1, 2), sine_hw.repeat(1, 2)
    values["cosine"].copy_(cosine)
    values["sine"].copy_(sine)
    for layer in range(int(values["num_layers"])):
        normalized = rms_norm(hidden, values["norm1_weight"][layer], VISION_RMS_NORM_EPS)
        attended = golden_vision_attention(
            normalized,
            values["qkv_weight"][layer], values["qkv_bias"][layer], values["q_norm_weight"][layer],
            values["k_norm_weight"][layer], cosine, sine, values["out_weight"][layer],
            values["out_bias"][layer], values["segment_start_loc"],
        )
        residual = (hidden.float() + attended.float()).bfloat16()
        normalized_mlp = rms_norm(residual, values["norm2_weight"][layer], VISION_RMS_NORM_EPS)
        mlp_output = golden_vision_mlp(
            normalized_mlp,
            values["gate_weight"][layer], values["gate_bias"][layer], values["up_weight"][layer],
            values["up_bias"][layer], values["down_weight"][layer], values["down_bias"][layer],
        )
        hidden = (residual.float() + mlp_output.float()).bfloat16()
        values["workspace_b" if layer % 2 == 0 else "workspace_a"].copy_(hidden)
    post_norm = rms_norm(hidden, values["post_norm_weight"], VISION_RMS_NORM_EPS)
    grouped = post_norm.reshape(-1, VISION_MERGE**2, VISION_HIDDEN)
    features = golden_vision_merger(
        grouped, values["downsample_weight"], values["downsample_bias"], values["merger_proj_weight"],
        values["merger_norm_weight"], values["merger_norm_bias"], values["merger_gate_weight"],
        values["merger_up_weight"], values["merger_down_weight"],
    )
    values["features"].copy_(features)


def golden_vision_forward(values):
    inner = {
        name: value[0] if isinstance(value, torch.Tensor) and value.ndim else value
        for name, value in values.items()
    }
    golden_vision_single(inner)


class VisionEncoder:
    """Execute the checkpoint's BF16 vision tower on one allocated device."""

    def __init__(self, checkpoint, device_id, *, platform="a2a3", compiled=None):
        self.checkpoint = checkpoint if isinstance(checkpoint, Checkpoint) else Checkpoint(checkpoint)
        self.weights = self.checkpoint.vision_weights()
        self.config = RunConfig(
            platform=platform,
            distributed_config=DistributedConfig(device_ids=[device_id], num_sub_workers=0),
        )
        self.program = vision_forward.compile(config=self.config) if compiled is None else compiled
        config = self.checkpoint._json("config.json")["vision_config"]
        rope = config.get("rope_parameters") or {"rope_type": "axial", "rope_theta": 10000.0}
        if rope.get("rope_type") != "axial" or float(rope.get("rope_theta", 0)) <= 0:
            raise ValueError("vision RoPE must use an axial positive-frequency base")
        powers = torch.arange(VISION_HEAD_DIM // 4).float() * 4 / VISION_HEAD_DIM
        self.inv_freq = float(rope["rope_theta"]) ** (-powers)

    def __call__(self, pixel_values, grid_thw):
        positions, starts = vision_metadata(grid_thw)
        if tuple(pixel_values.shape) != (positions.shape[0], VISION_PATCH_IN):
            raise ValueError("pixel_values rows must match grid_thw and the flattened patch width")
        if not torch.isfinite(pixel_values).all():
            raise ValueError("pixel_values must be finite")
        patches = pixel_values.cpu().bfloat16().contiguous()
        features = torch.empty((patches.shape[0] // VISION_MERGE**2, D), dtype=torch.bfloat16)
        rows = patches.shape[0]
        workspace_a = torch.zeros((1, rows, VISION_HIDDEN), dtype=torch.bfloat16)
        workspace_b = torch.zeros_like(workspace_a)
        cosine = torch.empty((1, rows, VISION_HEAD_DIM), dtype=torch.float32)
        sine = torch.empty_like(cosine)
        cosine_table, sine_table = vision_rope_tables(positions, self.inv_freq)
        arguments = [
            patches.unsqueeze(0),
            *(self.weights[name].unsqueeze(0) for name in vision_weight_layouts()),
            positions.unsqueeze(0),
            starts.unsqueeze(0),
            cosine_table.unsqueeze(0),
            sine_table.unsqueeze(0),
            features.unsqueeze(0),
            workspace_a,
            workspace_b,
            cosine,
            sine,
            torch.tensor(VISION_DEPTH, dtype=torch.int32),
        ]
        self.program(*arguments, config=self.config)
        if not torch.isfinite(features).all():
            raise FloatingPointError("vision encoder produced nonfinite features")
        return features


def golden_vision_merger(
    x: torch.Tensor,
    downsample_weight: torch.Tensor,
    downsample_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    post_norm_weight: torch.Tensor,
    post_norm_bias: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    packed = x.transpose(1, 2).reshape(-1, VISION_HIDDEN * VISION_MERGE**2)
    downsampled = (golden_bf16_linear(packed, downsample_weight) + downsample_bias.float()).bfloat16()
    projected = golden_bf16_linear(downsampled, proj_weight).bfloat16()
    normalized = F.layer_norm(projected.float(), (D,), post_norm_weight.float(), post_norm_bias.float(), LAYER_NORM_EPS)
    activated = F.gelu(normalized.bfloat16()).bfloat16()
    gate = golden_bf16_linear(activated, w_gate).bfloat16().clamp(max=VISION_SWIGLU_LIMIT)
    up = golden_bf16_linear(activated, w_up).bfloat16().clamp(-VISION_SWIGLU_LIMIT, VISION_SWIGLU_LIMIT)
    intermediate = (F.silu(gate) * up).bfloat16()
    return golden_bf16_linear(intermediate, w_down).bfloat16()


def run_attention_case(argv=None):
    from functools import lru_cache

    from golden import TensorSpec, ratio_allclose, run

    options = component_options(__doc__, layers=VISION_DEPTH, argv=argv)
    shapes = {
        "x": [options.tokens, VISION_HIDDEN],
        "qkv_weight": [3 * VISION_HIDDEN, VISION_HIDDEN],
        "qkv_bias": [3 * VISION_HIDDEN],
        "q_norm_weight": [VISION_HEAD_DIM],
        "k_norm_weight": [VISION_HEAD_DIM],
        "proj_weight": [VISION_HIDDEN, VISION_HIDDEN],
        "proj_bias": [VISION_HIDDEN],
    }
    names = {
        "qkv_weight": "qkv.weight",
        "qkv_bias": "qkv.bias",
        "q_norm_weight": "q_norm.weight",
        "k_norm_weight": "k_norm.weight",
        "proj_weight": "proj.weight",
        "proj_bias": "proj.bias",
    }

    @lru_cache(maxsize=1)
    def inputs():
        generator = torch.Generator().manual_seed(5352)
        checkpoint = Checkpoint(options.checkpoint) if options.checkpoint is not None else None
        values = {"x": component_hidden(options, VISION_HIDDEN, generator)}
        for name, suffix in names.items():
            shape = shapes[name]
            if checkpoint is not None:
                values[name] = checkpoint.vision_tensor(f"blocks.{options.layer}.attn.{suffix}", shape)
            elif "norm_weight" in name:
                values[name] = torch.ones(shape, dtype=torch.bfloat16)
            else:
                scale = VISION_HIDDEN**-0.5 if len(shape) == 2 else 0.1
                values[name] = (torch.randn(shape, generator=generator) * scale).bfloat16()
        return values

    specs = [
        TensorSpec(name, shape, torch.bfloat16, init_value=lambda key=name: inputs()[key])
        for name, shape in shapes.items()
    ]
    split = min(7, options.tokens)
    starts = torch.tensor([0, 0, split, options.tokens], dtype=torch.int32)
    positions = torch.stack((torch.arange(options.tokens) // 4, torch.arange(options.tokens) % 4), dim=-1).int()
    inverse = 10000.0 ** (-torch.arange(VISION_HEAD_DIM // 4).float() * 4 / VISION_HEAD_DIM)
    cosine_table, sine_table = vision_rope_tables(positions, inverse)
    specs.extend(
        [
            TensorSpec("segment_start_loc", [4], torch.int32, init_value=starts),
            TensorSpec("position_ids", [options.tokens, 2], torch.int32, init_value=positions),
            TensorSpec("cosine_table", cosine_table.shape, torch.float32, init_value=cosine_table),
            TensorSpec("sine_table", sine_table.shape, torch.float32, init_value=sine_table),
            TensorSpec("cos", [options.tokens, VISION_HEAD_DIM], torch.float32),
            TensorSpec("sin", [options.tokens, VISION_HEAD_DIM], torch.float32),
            TensorSpec("output", [options.tokens, VISION_HIDDEN], torch.bfloat16),
        ],
    )

    def golden(values):
        height, width = values["position_ids"].long().unbind(dim=-1)
        cosine_hw = torch.cat((values["cosine_table"][height], values["cosine_table"][width]), dim=-1)
        sine_hw = torch.cat((values["sine_table"][height], values["sine_table"][width]), dim=-1)
        cosine, sine = cosine_hw.repeat(1, 2), sine_hw.repeat(1, 2)
        values["cos"].copy_(cosine)
        values["sin"].copy_(sine)
        result = golden_vision_attention(
            values["x"], values["qkv_weight"], values["qkv_bias"], values["q_norm_weight"],
            values["k_norm_weight"], cosine, sine, values["proj_weight"], values["proj_bias"],
            values["segment_start_loc"],
        )
        values["output"].copy_(result)

    result = run(
        fn=vision_attention_test,
        specs=specs,
        golden_fn=golden,
        config=dict(backend_type=options.backend_type, execution_mode=options.execution_mode, device_id=options.device),
        compare_fn={"cos": ratio_allclose(atol=1e-05, rtol=1e-05, max_error_ratio=0), "sin": ratio_allclose(atol=1e-05, rtol=1e-05, max_error_ratio=0), "output": ratio_allclose(atol=0.01, rtol=0.01, max_error_ratio=0)},
        compile_only=options.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


def run_mlp_case(argv=None):
    from functools import lru_cache

    from golden import TensorSpec, ratio_allclose, run

    options = component_options(__doc__, layers=VISION_DEPTH, argv=argv)
    shapes = {
        "x": [options.tokens, VISION_HIDDEN],
        "w_gate": [VISION_INTER, VISION_HIDDEN],
        "b_gate": [VISION_INTER],
        "w_up": [VISION_INTER, VISION_HIDDEN],
        "b_up": [VISION_INTER],
        "w_down": [VISION_HIDDEN, VISION_INTER],
        "b_down": [VISION_HIDDEN],
    }
    names = {
        "w_gate": "gate_proj.weight",
        "b_gate": "gate_proj.bias",
        "w_up": "up_proj.weight",
        "b_up": "up_proj.bias",
        "w_down": "down_proj.weight",
        "b_down": "down_proj.bias",
    }

    @lru_cache(maxsize=1)
    def inputs():
        generator = torch.Generator().manual_seed(5351)
        checkpoint = Checkpoint(options.checkpoint) if options.checkpoint is not None else None
        values = {"x": component_hidden(options, VISION_HIDDEN, generator)}
        for name, suffix in names.items():
            shape = shapes[name]
            if checkpoint is not None:
                values[name] = checkpoint.vision_tensor(f"blocks.{options.layer}.mlp.{suffix}", shape)
            else:
                scale = shape[-1] ** -0.5 if len(shape) == 2 else 0.1
                values[name] = (torch.randn(shape, generator=generator) * scale).bfloat16()
        return values

    specs = [
        TensorSpec(name, shape, torch.bfloat16, init_value=lambda key=name: inputs()[key])
        for name, shape in shapes.items()
    ]
    specs.append(TensorSpec("output", [options.tokens, VISION_HIDDEN], torch.bfloat16))

    def golden(values):
        values["output"].copy_(golden_vision_mlp(*(values[name] for name in shapes)))

    result = run(
        fn=vision_mlp_test,
        specs=specs,
        golden_fn=golden,
        config=dict(backend_type=options.backend_type, execution_mode=options.execution_mode, device_id=options.device),
        compare_fn={"output": ratio_allclose(atol=0.01, rtol=0.01, max_error_ratio=0)},
        compile_only=options.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


def run_fusion_case(argv=None):
    import argparse
    from functools import lru_cache

    from golden import TensorSpec, ratio_allclose, run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", dest="backend_type", default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--vision-rows", type=int, default=5)
    parser.add_argument("--vision-input", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    options = parser.parse_args(argv)
    if not 1 <= options.vision_rows <= options.tokens:
        parser.error("require 1 <= --vision-rows <= --tokens")

    @lru_cache(maxsize=1)
    def inputs():
        generator = torch.Generator().manual_seed(5354)
        streams = torch.randn(options.tokens, HC_MULT, D, generator=generator).bfloat16()
        vision = torch.randn(options.vision_rows, D, generator=generator).bfloat16()
        if options.checkpoint is not None:
            checkpoint = Checkpoint(options.checkpoint)
            embedding = checkpoint.read("model.language_model.embed_tokens.weight", slice(2000, 2000 + options.tokens))
            streams = embedding[:, None, :].expand(-1, HC_MULT, -1).clone()
        if options.vision_input is not None:
            vision = torch.load(options.vision_input, map_location="cpu", weights_only=True)
            if tuple(vision.shape) != (options.vision_rows, D) or not torch.isfinite(vision).all():
                raise ValueError("--vision-input must contain finite [vision_rows, hidden_size] features")
            vision = vision.bfloat16()
        rows = torch.randperm(options.tokens, generator=generator)[: options.vision_rows].int()
        return {"vision_embeds": vision, "placeholder_rows": rows, "hidden_streams": streams}

    shapes = {
        "vision_embeds": [options.vision_rows, D],
        "placeholder_rows": [options.vision_rows],
        "hidden_streams": [options.tokens, HC_MULT, D],
    }
    specs = [
        TensorSpec(name, shape, torch.int32 if name == "placeholder_rows" else torch.bfloat16, init_value=lambda key=name: inputs()[key])
        for name, shape in shapes.items()
    ]

    def golden(values):
        rows = values["placeholder_rows"].long()
        mask = torch.zeros(options.tokens, dtype=torch.bool)
        mask[rows] = True
        ordered_features = values["vision_embeds"][rows.argsort()]
        for lane in range(HC_MULT):
            fused = golden_vision_fusion(values["hidden_streams"][:, lane], ordered_features, mask)
            values["hidden_streams"][:, lane].copy_(fused)

    result = run(
        fn=vision_fusion_test,
        specs=specs,
        golden_fn=golden,
        config=dict(backend_type=options.backend_type, execution_mode=options.execution_mode, device_id=options.device),
        compare_fn={"hidden_streams": ratio_allclose(atol=0, rtol=0, max_error_ratio=0)},
        compile_only=options.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


def run_patch_case(argv=None):
    from functools import lru_cache

    from golden import TensorSpec, ratio_allclose, run

    options = component_options(__doc__, argv=argv)

    @lru_cache(maxsize=1)
    def inputs():
        generator = torch.Generator().manual_seed(5350)
        values = {"patches": component_hidden(options, VISION_PATCH_IN, generator)}
        if options.checkpoint is None:
            values["weight"] = (torch.randn(VISION_HIDDEN, VISION_PATCH_IN, generator=generator) / VISION_PATCH_IN ** 0.5).bfloat16(
            )
            values["bias"] = (torch.randn(VISION_HIDDEN, generator=generator) * 0.1).bfloat16()
        else:
            checkpoint = Checkpoint(options.checkpoint)
            shape = (VISION_HIDDEN, 3, 2, 14, 14)
            values["weight"] = checkpoint.vision_tensor("patch_embed.proj.weight", shape).flatten(1)
            values["bias"] = checkpoint.vision_tensor("patch_embed.proj.bias", (VISION_HIDDEN,))
        return values

    shapes = {
        "patches": [options.tokens, VISION_PATCH_IN],
        "weight": [VISION_HIDDEN, VISION_PATCH_IN],
        "bias": [VISION_HIDDEN],
    }
    specs = [
        TensorSpec(name, shape, torch.bfloat16, init_value=lambda key=name: inputs()[key])
        for name, shape in shapes.items()
    ]
    specs.append(TensorSpec("output", [options.tokens, VISION_HIDDEN], torch.bfloat16))

    def golden(values):
        values["output"].copy_(golden_vision_patch_embed(*(values[name] for name in shapes)))

    result = run(
        fn=vision_patch_embed_test,
        specs=specs,
        golden_fn=golden,
        config=dict(backend_type=options.backend_type, execution_mode=options.execution_mode, device_id=options.device),
        compare_fn={"output": ratio_allclose(atol=0.01, rtol=0.01, max_error_ratio=0)},
        compile_only=options.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


def run_forward_case(argv=None):
    import argparse
    import json
    from functools import lru_cache

    from golden import ScalarSpec, TensorSpec, ratio_allclose, run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", dest="backend_type", default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pixels", type=Path)
    parser.add_argument("--grid", default="[[1, 2, 2], [1, 2, 6]]")
    parser.add_argument("--layers", type=int, default=VISION_DEPTH)
    parser.add_argument("--compile-only", action="store_true")
    options = parser.parse_args(argv)
    if not 0 <= options.layers <= VISION_DEPTH:
        parser.error(f"--layers must be in [0, {VISION_DEPTH}]")
    positions, starts = vision_metadata(json.loads(options.grid))
    rows = positions.shape[0]
    layouts = vision_weight_layouts()

    @lru_cache(maxsize=1)
    def inputs():
        generator = torch.Generator().manual_seed(5355)
        if options.checkpoint is not None:
            weights = Checkpoint(options.checkpoint).vision_weights()
        else:
            weights = {}
            for name, shape in layouts.items():
                if "norm" in name and "bias" not in name:
                    weights[name] = torch.ones(shape, dtype=torch.bfloat16)
                else:
                    scale = 0.1 if "bias" in name else shape[-1] ** -0.5
                    weights[name] = (torch.randn(shape, generator=generator) * scale).bfloat16()
        if options.pixels is None:
            patches = torch.randn(rows, VISION_PATCH_IN, generator=generator).bfloat16()
        else:
            patches = torch.load(options.pixels, map_location="cpu", weights_only=True)
            if tuple(patches.shape) != (rows, VISION_PATCH_IN) or not torch.isfinite(patches).all():
                raise ValueError("--pixels must contain finite patches matching --grid")
            patches = patches.bfloat16()
        return {"patches": patches, **weights}

    shapes = {"patches": (rows, VISION_PATCH_IN), **layouts}
    specs = [
        TensorSpec(name, (1, *shape), torch.bfloat16, init_value=lambda key=name: inputs()[key].unsqueeze(0))
        for name, shape in shapes.items()
    ]
    inverse = 10000.0 ** (-torch.arange(VISION_HEAD_DIM // 4).float() * 4 / VISION_HEAD_DIM)
    cosine_table, sine_table = vision_rope_tables(positions, inverse)
    specs.extend(
        [
            TensorSpec("position_ids", (1, *positions.shape), torch.int32, init_value=positions.unsqueeze(0)),
            TensorSpec("segment_start_loc", (1, *starts.shape), torch.int32, init_value=starts.unsqueeze(0)),
            TensorSpec("cosine_table", (1, *cosine_table.shape), torch.float32, init_value=cosine_table.unsqueeze(0)),
            TensorSpec("sine_table", (1, *sine_table.shape), torch.float32, init_value=sine_table.unsqueeze(0)),
            TensorSpec("features", [1, rows // VISION_MERGE ** 2, D], torch.bfloat16),
            TensorSpec("workspace_a", [1, rows, VISION_HIDDEN], torch.bfloat16, init_value=torch.zeros),
            TensorSpec("workspace_b", [1, rows, VISION_HIDDEN], torch.bfloat16, init_value=torch.zeros),
            TensorSpec("cosine", [1, rows, VISION_HEAD_DIM], torch.float32),
            TensorSpec("sine", [1, rows, VISION_HEAD_DIM], torch.float32),
            ScalarSpec("num_layers", torch.int32, options.layers),
        ],
    )
    result = run(
        fn=vision_forward,
        specs=specs,
        golden_fn=golden_vision_forward,
        config=dict(backend_type=options.backend_type, execution_mode=options.execution_mode, distributed_config=DistributedConfig(device_ids=[options.device], num_sub_workers=0)),
        compare_fn={
            "features": ratio_allclose(atol=0.02, rtol=0.02, max_error_ratio=0.05),
            "workspace_a": ratio_allclose(atol=0.02, rtol=0.02, max_error_ratio=0.05),
            "workspace_b": ratio_allclose(atol=0.02, rtol=0.02, max_error_ratio=0.05),
            "cosine": ratio_allclose(atol=1e-05, rtol=1e-05, max_error_ratio=0),
            "sine": ratio_allclose(atol=1e-05, rtol=1e-05, max_error_ratio=0),
        },
        compile_only=options.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


def run_merger_case(argv=None):
    from functools import lru_cache

    from golden import TensorSpec, ratio_allclose, run

    options = component_options(__doc__, tokens=3, argv=argv)
    input_width = VISION_MERGE**2 * VISION_HIDDEN
    shapes = {
        "x": [options.tokens, VISION_MERGE**2, VISION_HIDDEN],
        "downsample_weight": [D, input_width],
        "downsample_bias": [D],
        "proj_weight": [D, D],
        "post_norm_weight": [D],
        "post_norm_bias": [D],
        "w_gate": [VISION_PROJ_INTER, D],
        "w_up": [VISION_PROJ_INTER, D],
        "w_down": [D, VISION_PROJ_INTER],
    }
    names = {
        "downsample_weight": "downsample.weight",
        "downsample_bias": "downsample.bias",
        "proj_weight": "merger.proj.weight",
        "post_norm_weight": "merger.post_projection_norm.weight",
        "post_norm_bias": "merger.post_projection_norm.bias",
        "w_gate": "merger.gate_proj.weight",
        "w_up": "merger.up_proj.weight",
        "w_down": "merger.down_proj.weight",
    }

    @lru_cache(maxsize=1)
    def inputs():
        generator = torch.Generator().manual_seed(5353)
        checkpoint = Checkpoint(options.checkpoint) if options.checkpoint is not None else None
        values = {"x": component_hidden(options, input_width, generator).reshape(shapes["x"])}
        for name, suffix in names.items():
            shape = shapes[name]
            if checkpoint is not None:
                source_shape = [D, VISION_HIDDEN, VISION_MERGE, VISION_MERGE] if name == "downsample_weight" else shape
                values[name] = checkpoint.vision_tensor(suffix, source_shape).reshape(shape)
            elif name == "post_norm_weight":
                values[name] = torch.ones(shape, dtype=torch.bfloat16)
            else:
                scale = shape[-1] ** -0.5 if len(shape) == 2 else 0.1
                values[name] = (torch.randn(shape, generator=generator) * scale).bfloat16()
        return values

    specs = [
        TensorSpec(name, shape, torch.bfloat16, init_value=lambda key=name: inputs()[key])
        for name, shape in shapes.items()
    ]
    specs.append(TensorSpec("output", [options.tokens, D], torch.bfloat16))

    def golden(values):
        values["output"].copy_(golden_vision_merger(*(values[name] for name in shapes)))

    result = run(
        fn=vision_merger_test,
        specs=specs,
        golden_fn=golden,
        config=dict(backend_type=options.backend_type, execution_mode=options.execution_mode, device_id=options.device),
        compare_fn={"output": ratio_allclose(atol=0.01, rtol=0.01, max_error_ratio=0)},
        compile_only=options.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    components = ["all", "patch", "attention", "mlp", "merger", "fusion", "forward"]
    parser.add_argument("--component", choices=components, default="all")
    options, remaining = parser.parse_known_args(argv)
    if options.component == "all" and any(arg in ("-h", "--help") for arg in remaining):
        help_parser = argparse.ArgumentParser(description=__doc__, parents=[parser])
        help_parser.epilog = "Use --component NAME --help for stage options."
        help_parser.parse_args(argv)
    cases = {
        "patch": run_patch_case,
        "attention": run_attention_case,
        "mlp": run_mlp_case,
        "merger": run_merger_case,
        "fusion": run_fusion_case,
        "forward": run_forward_case,
    }
    if options.component == "all":
        # Single-device stages and the distributed forward own separate runtime lifetimes.
        for component in cases:
            command = [sys.executable, str(Path(__file__).resolve()), "--component", component, *remaining]
            result = subprocess.run(command, check=False)
            if result.returncode:
                raise SystemExit(result.returncode)
        return
    cases[options.component](remaining)


if __name__ == "__main__":
    main()
