# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Inverse partial RoPE, grouped BF16 Wo-A and dynamic W8A8 Wo-B."""

import pypto.language as pl

from .ops import M_TILE, K_TILE, N_TILE, make_rope, make_quantize, make_pack_weight, make_dot, make_dequantize


def make_o_proj_inline(capacity, config):
    """Build the complete grouped attention output projection."""
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    rope_dim = config["qk_rope_head_dim"]
    pairs = rope_dim // 2
    width = heads * head_dim
    groups = config["o_groups"]
    group_input = width // groups
    rank = config["o_lora_rank"]
    group_width = groups * rank
    packed_rows = group_width * (hidden // N_TILE)
    rotate = make_rope(capacity, heads, head_dim, rope_dim)
    quantize = make_quantize(capacity, group_width)
    pack = make_pack_weight(group_width, hidden)
    dot = make_dot(capacity, group_width, hidden)
    dequantize = make_dequantize(capacity, hidden, combine_scales=True)

    @pl.jit.inline
    def project(
        attention: pl.Tensor[[capacity, width], pl.BF16],
        wo_a: pl.Tensor[[groups, group_input, rank], pl.BF16],
        wo_b: pl.Tensor[[group_width, hidden], pl.INT8],
        wo_b_scale: pl.Tensor[[hidden], pl.FP32],
        rope_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        output: pl.Tensor[[capacity, hidden], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        inverse_sine = pl.create_tensor([capacity, pairs], dtype=pl.FP32)
        for token in pl.spmd(active, name_hint="a2_inverse_sine"):
            inverse_sine[token:token + 1, :] = pl.mul(rope_sin[token:token + 1, :], -1.0)
        rotated = pl.create_tensor([capacity, width], dtype=pl.BF16)
        rotate(attention, rope_cos, inverse_sine, rotated, active)

        grouped = pl.create_tensor([capacity, group_width], dtype=pl.BF16)
        for block in pl.spmd((active + M_TILE - 1) // M_TILE * (group_width // N_TILE), name_hint="a2_wo_a"):
            token = block // (group_width // N_TILE) * M_TILE
            column = block % (group_width // N_TILE) * N_TILE
            group = column // rank
            group_column = column % rank
            rows = pl.min(M_TILE, active - token)
            accumulated = pl.create_tensor([M_TILE, N_TILE], dtype=pl.FP32)
            for part in pl.pipeline(group_input // K_TILE, stage=2):
                inner = part * K_TILE
                offset = group * group_input + inner
                x = pl.slice(rotated, [M_TILE, K_TILE], [token, offset], valid_shape=[rows, K_TILE])
                x = pl.fillpad(x, pad_value=pl.PadValue.zero)
                w = wo_a[group:group + 1, inner:inner + K_TILE, group_column:group_column + N_TILE]
                w = pl.reshape(w, [K_TILE, N_TILE])
                accumulated = pl.matmul_acc(accumulated, x, w, init_cond=(part == 0))
            narrowed = pl.cast(accumulated, pl.BF16, mode="rint")
            grouped[token:token + M_TILE, column:column + N_TILE] = pl.set_validshape(narrowed, rows, N_TILE)

        payload = pl.create_tensor([capacity, group_width], dtype=pl.INT8)
        token_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize(grouped, payload, token_scale, active)
        packed = pl.create_tensor([packed_rows, N_TILE], dtype=pl.INT8)
        pack(wo_b, packed)
        output_int32 = pl.create_tensor([capacity, hidden], dtype=pl.INT32)
        dot(payload, packed, output_int32, active)
        dequantize(output_int32, wo_b_scale, token_scale, output, active)
        return output

    return project


def make_o_proj_w8a8(capacity, config):
    """Build inverse RoPE and all grouped W8A8 Wo-A/Wo-B projections.

    Wo-A input quantization is independent for each native group. The package
    stores [group_input, groups*rank] with per-output-channel FP32 scales.
    """
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    rope_dim = config["qk_rope_head_dim"]
    pairs = rope_dim // 2
    width = heads * head_dim
    groups = config["o_groups"]
    group_input = width // groups
    rank = config["o_lora_rank"]
    group_width = groups * rank
    rotate = make_rope(capacity, heads, head_dim, rope_dim)
    quantize_a = make_quantize(capacity, group_input)
    pack_a = make_pack_weight(group_input, rank)
    dot_a = make_dot(capacity, group_input, rank)
    dequantize_a = make_dequantize(capacity, rank)
    quantize_b = make_quantize(capacity, group_width)
    pack_b = make_pack_weight(group_width, hidden)
    dot_b = make_dot(capacity, group_width, hidden)
    dequantize_b = make_dequantize(capacity, hidden, combine_scales=True)
    packed_a_rows = group_input * (rank // N_TILE)
    packed_b_rows = group_width * (hidden // N_TILE)

    @pl.jit
    def project(
        attention: pl.Tensor[[capacity, width], pl.FP32],
        wo_a: pl.Tensor[[group_input, group_width], pl.INT8],
        wo_a_scale: pl.Tensor[[group_width], pl.FP32],
        wo_b: pl.Tensor[[group_width, hidden], pl.INT8],
        wo_b_scale: pl.Tensor[[hidden], pl.FP32],
        rope_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        rotated: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        grouped: pl.InOut[pl.Tensor[[capacity, group_width], pl.BF16]],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        narrowed = pl.create_tensor([capacity, width], dtype=pl.BF16)
        for block in pl.spmd(active * (width // N_TILE), name_hint="a2_attention_bf16_boundary"):
            token = block // (width // N_TILE)
            col = block % (width // N_TILE) * N_TILE
            source = attention[token:token + 1, col:col + N_TILE]
            narrowed[token:token + 1, col:col + N_TILE] = pl.cast(source, pl.BF16, mode="rint")
        inverse_sine = pl.create_tensor([capacity, pairs], dtype=pl.FP32)
        for token in pl.spmd(active, name_hint="a2_w8a8_inverse_sine"):
            inverse_sine[token:token + 1, :] = pl.mul(rope_sin[token:token + 1, :], -1.0)
        rotate(narrowed, rope_cos, inverse_sine, rotated, active)

        for group in pl.range(groups):
            group_source = pl.create_tensor([capacity, group_input], dtype=pl.BF16)
            for block in pl.spmd(active * (group_input // N_TILE), name_hint="a2_wo_a_group_input"):
                token = block // (group_input // N_TILE)
                column = block % (group_input // N_TILE) * N_TILE
                offset = group * group_input + column
                group_source[token:token + 1, column:column + N_TILE] = rotated[token:token + 1, offset:offset + N_TILE]
            payload_a = pl.create_tensor([capacity, group_input], dtype=pl.INT8)
            scale_a = pl.create_tensor([capacity, 8], dtype=pl.FP32)
            quantize_a(group_source, payload_a, scale_a, active)
            group_weight = pl.create_tensor([group_input, rank], dtype=pl.INT8)
            for block in pl.spmd((group_input // K_TILE) * (rank // N_TILE), name_hint="a2_wo_a_group_weight"):
                inner = block // (rank // N_TILE) * K_TILE
                column = block % (rank // N_TILE) * N_TILE
                offset = group * rank + column
                group_weight[inner:inner + K_TILE, column:column + N_TILE] = wo_a[inner:inner + K_TILE, offset:offset + N_TILE]
            packed_a = pl.create_tensor([packed_a_rows, N_TILE], dtype=pl.INT8)
            pack_a(group_weight, packed_a)
            accumulator_a = pl.create_tensor([capacity, rank], dtype=pl.INT32)
            dot_a(payload_a, packed_a, accumulator_a, active)
            channel_scale = wo_a_scale[group * rank:(group + 1) * rank]
            group_output = pl.create_tensor([capacity, rank], dtype=pl.BF16)
            dequantize_a(accumulator_a, channel_scale, scale_a, group_output, active)
            for block in pl.spmd(active * (rank // N_TILE), name_hint="a2_wo_a_group_output"):
                token = block // (rank // N_TILE)
                column = block % (rank // N_TILE) * N_TILE
                offset = group * rank + column
                grouped[token:token + 1, offset:offset + N_TILE] = group_output[token:token + 1, column:column + N_TILE]

        payload_b = pl.create_tensor([capacity, group_width], dtype=pl.INT8)
        scale_b = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize_b(grouped, payload_b, scale_b, active)
        packed_b = pl.create_tensor([packed_b_rows, N_TILE], dtype=pl.INT8)
        pack_b(wo_b, packed_b)
        accumulator_b = pl.create_tensor([capacity, hidden], dtype=pl.INT32)
        dot_b(payload_b, packed_b, accumulator_b, active)
        dequantize_b(accumulator_b, wo_b_scale, scale_b, output, active)
        return rotated, grouped, output

    return project
