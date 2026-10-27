# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed noncausal vision attention with head RMSNorm and axial two-dimensional RoPE."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import T_DYN
from models.glm5_3_flash.elementwise import accurate_exp, accurate_reciprocal, accurate_row_sum_parts
from models.glm5_3_flash.golden import rms_norm
from models.glm5_3_flash.linear import bf16_linear, bf16_projection, golden_bf16_linear
from models.glm5_3_flash.rmsnorm import rmsnorm
from models.glm5_3_flash.vision.config import VISION_DEPTH, VISION_HEAD_DIM, VISION_HEADS
from models.glm5_3_flash.vision.config import VISION_HIDDEN, VISION_RMS_NORM_EPS

VISION_STARTS_DYN = pl.dynamic("GLM53_VISION_STARTS")
VISION_POSITIONS_DYN = pl.dynamic("GLM53_VISION_POSITIONS")

ATTENTION_M_TILE = 16
ATTENTION_N_TILE = 64


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
        scores = (q @ k.transpose(-1, -2)).float() * VISION_HEAD_DIM ** -0.5
        shifted = scores - scores.amax(dim=-1, keepdim=True)
        exponentials = shifted.double().exp().float()
        mass = exponentials.double().sum(dim=-1, keepdim=True).float()
        inverse = mass.double().reciprocal().float()
        probabilities = (exponentials * inverse).bfloat16()
        context = (probabilities.double() @ v).float().bfloat16()
        attended[begin:end] = context.transpose(0, 1).reshape(end - begin, VISION_HIDDEN)
    return (golden_bf16_linear(attended, proj_weight) + proj_bias.float()).bfloat16()


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
        height_cos = cosine_table[height:height + 1, :]
        width_cos = cosine_table[width:width + 1, :]
        height_sin = sine_table[height:height + 1, :]
        width_sin = sine_table[width:width + 1, :]
        for half in pl.unroll(2):
            col0 = half * (VISION_HEAD_DIM // 2)
            cos[row:row + 1, col0:col0 + VISION_HEAD_DIM // 4] = height_cos
            cos[row:row + 1, col0 + VISION_HEAD_DIM // 4:col0 + VISION_HEAD_DIM // 2] = width_cos
            sin[row:row + 1, col0:col0 + VISION_HEAD_DIM // 4] = height_sin
            sin[row:row + 1, col0 + VISION_HEAD_DIM // 4:col0 + VISION_HEAD_DIM // 2] = width_sin
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
    query_rows = pl.create_tensor([rows * VISION_HEADS, VISION_HEAD_DIM], dtype=pl.BF16)
    key_rows = pl.create_tensor([rows * VISION_HEADS, VISION_HEAD_DIM], dtype=pl.BF16)
    for block in pl.spmd(rows * VISION_HEADS, name_hint="vision_qk_rows"):
        split_row = block // VISION_HEADS
        split_head = block % VISION_HEADS
        split_col0 = split_head * VISION_HEAD_DIM
        query_rows[block:block + 1, :] = qkv[split_row:split_row + 1, split_col0:split_col0 + VISION_HEAD_DIM]
        key_rows[block:block + 1, :] = qkv[split_row:split_row + 1, VISION_HIDDEN + split_col0:VISION_HIDDEN + split_col0 + VISION_HEAD_DIM]
    query_norm = pl.create_tensor([rows * VISION_HEADS, VISION_HEAD_DIM], dtype=pl.BF16)
    key_norm = pl.create_tensor([rows * VISION_HEADS, VISION_HEAD_DIM], dtype=pl.BF16)
    rmsnorm(query_rows, q_norm_weight, query_norm)
    rmsnorm(key_rows, k_norm_weight, key_norm)
    query = pl.create_tensor([VISION_HEADS, rows, VISION_HEAD_DIM], dtype=pl.BF16)
    key = pl.create_tensor([VISION_HEADS, rows, VISION_HEAD_DIM], dtype=pl.BF16)
    value = pl.create_tensor([VISION_HEADS, rows, VISION_HEAD_DIM], dtype=pl.BF16)
    for block in pl.spmd(rows * VISION_HEADS, name_hint="vision_qk_rope"):
        rope_row = block // VISION_HEADS
        rope_head = block % VISION_HEADS
        rope_col0 = rope_head * VISION_HEAD_DIM
        for half in pl.unroll(2):
            offset = half * (VISION_HEAD_DIM // 2)
            other_offset = (1 - half) * (VISION_HEAD_DIM // 2)
            q_part = pl.cast(query_norm[block:block + 1, offset:offset + VISION_HEAD_DIM // 2], pl.FP32)
            k_part = pl.cast(key_norm[block:block + 1, offset:offset + VISION_HEAD_DIM // 2], pl.FP32)
            q_other = pl.cast(query_norm[block:block + 1, other_offset:other_offset + VISION_HEAD_DIM // 2], pl.FP32)
            k_other = pl.cast(key_norm[block:block + 1, other_offset:other_offset + VISION_HEAD_DIM // 2], pl.FP32)
            cosine = cos[rope_row:rope_row + 1, offset:offset + VISION_HEAD_DIM // 2]
            sine = sin[rope_row:rope_row + 1, offset:offset + VISION_HEAD_DIM // 2]
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
        value_row = qkv[rope_row:rope_row + 1, 2 * VISION_HIDDEN + rope_col0:2 * VISION_HIDDEN + rope_col0 + VISION_HEAD_DIM]
        value[rope_head:rope_head + 1, rope_row:rope_row + 1, :] = pl.reshape(value_row, [1, 1, VISION_HEAD_DIM])
    attended = pl.create_tensor([rows, VISION_HIDDEN], dtype=pl.BF16)
    head_rows = rows * VISION_HEADS
    query_flat = pl.reshape(query, [head_rows, VISION_HEAD_DIM])
    key_flat = pl.reshape(key, [head_rows, VISION_HEAD_DIM])
    value_flat = pl.reshape(value, [head_rows, VISION_HEAD_DIM])
    segments = pl.tensor.dim(segment_start_loc, 0) - 1
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
                    dot = bf16_linear(query_flat, key_flat, query_head_row0, key_head_row0, valid_rows, valid_keys,
                                      ATTENTION_M_TILE, ATTENTION_N_TILE, VISION_HEAD_DIM, True)
                    scores = pl.mul(dot, VISION_HEAD_DIM ** -0.5)
                    scores_valid = pl.set_validshape(scores, ATTENTION_M_TILE, valid_keys)
                    scores_full = pl.set_validshape(pl.fillpad(scores_valid, pad_value=pl.PadValue.min), ATTENTION_M_TILE, ATTENTION_N_TILE)
                    block_maximum = pl.reshape(pl.row_max(scores_full), [1, ATTENTION_M_TILE])
                    updated_maximum = pl.maximum(maximum, block_maximum)
                    maximum = updated_maximum
                maximum_column = pl.reshape(maximum, [ATTENTION_M_TILE, 1])
                mass = pl.full([1, ATTENTION_M_TILE], dtype=pl.FP32, value=0.0)
                mass_error = pl.full([1, ATTENTION_M_TILE], dtype=pl.FP32, value=0.0)
                for mass_key0 in pl.range(begin, end, ATTENTION_N_TILE):
                    mass_valid_keys = pl.min(ATTENTION_N_TILE, end - mass_key0)
                    mass_key_row0 = attention_head * rows + mass_key0
                    mass_dot = bf16_linear(query_flat, key_flat, query_head_row0, mass_key_row0, valid_rows, mass_valid_keys,
                                           ATTENTION_M_TILE, ATTENTION_N_TILE, VISION_HEAD_DIM, True)
                    mass_scores = pl.mul(mass_dot, VISION_HEAD_DIM ** -0.5)
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
                    dot = bf16_linear(query_flat, key_flat, query_head_row0, key_head_row0, valid_rows, valid_keys,
                                      ATTENTION_M_TILE, ATTENTION_N_TILE, VISION_HEAD_DIM, True)
                    scores = pl.mul(dot, VISION_HEAD_DIM ** -0.5)
                    scores_valid = pl.set_validshape(scores, ATTENTION_M_TILE, valid_keys)
                    scores_full = pl.set_validshape(pl.fillpad(scores_valid, pad_value=pl.PadValue.min), ATTENTION_M_TILE, ATTENTION_N_TILE)
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
    vision_attention(x, qkv_weight, qkv_bias, q_norm_weight, k_norm_weight, cos, sin,
                     proj_weight, proj_bias, segment_start_loc, output)
    return cos, sin, output


def main():
    from functools import lru_cache

    from golden import TensorSpec, ratio_allclose, run
    from models.glm5_3_flash.utils import component_hidden, component_options
    from models.glm5_3_flash.weights import Checkpoint

    options = component_options(__doc__, layers=VISION_DEPTH)
    shapes = {
        "x": [options.tokens, VISION_HIDDEN],
        "qkv_weight": [3 * VISION_HIDDEN, VISION_HIDDEN], "qkv_bias": [3 * VISION_HIDDEN],
        "q_norm_weight": [VISION_HEAD_DIM], "k_norm_weight": [VISION_HEAD_DIM],
        "proj_weight": [VISION_HIDDEN, VISION_HIDDEN], "proj_bias": [VISION_HIDDEN],
    }
    names = {"qkv_weight": "qkv.weight", "qkv_bias": "qkv.bias", "q_norm_weight": "q_norm.weight",
             "k_norm_weight": "k_norm.weight", "proj_weight": "proj.weight", "proj_bias": "proj.bias"}

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
                scale = VISION_HIDDEN ** -0.5 if len(shape) == 2 else 0.1
                values[name] = (torch.randn(shape, generator=generator) * scale).bfloat16()
        return values

    specs = [TensorSpec(name, shape, torch.bfloat16, init_value=lambda key=name: inputs()[key])
             for name, shape in shapes.items()]
    split = min(7, options.tokens)
    starts = torch.tensor([0, 0, split, options.tokens], dtype=torch.int32)
    positions = torch.stack((torch.arange(options.tokens) // 4, torch.arange(options.tokens) % 4), dim=-1).int()
    inverse = 10000.0 ** (-torch.arange(VISION_HEAD_DIM // 4).float() * 4 / VISION_HEAD_DIM)
    cosine_table, sine_table = vision_rope_tables(positions, inverse)
    specs.extend([
        TensorSpec("segment_start_loc", [4], torch.int32, init_value=starts),
        TensorSpec("position_ids", [options.tokens, 2], torch.int32, init_value=positions),
        TensorSpec("cosine_table", cosine_table.shape, torch.float32, init_value=cosine_table),
        TensorSpec("sine_table", sine_table.shape, torch.float32, init_value=sine_table),
        TensorSpec("cos", [options.tokens, VISION_HEAD_DIM], torch.float32),
        TensorSpec("sin", [options.tokens, VISION_HEAD_DIM], torch.float32),
        TensorSpec("output", [options.tokens, VISION_HIDDEN], torch.bfloat16),
    ])

    def golden(values):
        height, width = values["position_ids"].long().unbind(dim=-1)
        cosine_hw = torch.cat((values["cosine_table"][height], values["cosine_table"][width]), dim=-1)
        sine_hw = torch.cat((values["sine_table"][height], values["sine_table"][width]), dim=-1)
        cosine, sine = cosine_hw.repeat(1, 2), sine_hw.repeat(1, 2)
        values["cos"].copy_(cosine)
        values["sin"].copy_(sine)
        result = golden_vision_attention(
            values["x"], values["qkv_weight"], values["qkv_bias"], values["q_norm_weight"], values["k_norm_weight"],
            cosine, sine, values["proj_weight"], values["proj_bias"], values["segment_start_loc"],
        )
        values["output"].copy_(result)

    result = run(
        fn=vision_attention_test,
        specs=specs,
        golden_fn=golden,
        config=dict(backend_type=options.backend_type, execution_mode=options.execution_mode, device_id=options.device),
        compare_fn={
            "cos": ratio_allclose(atol=1e-5, rtol=1e-5, max_error_ratio=0),
            "sin": ratio_allclose(atol=1e-5, rtol=1e-5, max_error_ratio=0),
            "output": ratio_allclose(atol=0.01, rtol=0.01, max_error_ratio=0),
        },
        compile_only=options.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()
