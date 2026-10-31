# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4.1 shared expert with A2 INT8 activation and weight matmuls."""

import pypto.language as pl

from .ops import N_TILE, VEC_M_TILE, make_dequantize, make_dot, make_pack_weight, make_quantize


def _make_shared_activation(capacity, intermediate, limit, fused_activation):
    """Select native fused FP32 or the preserved standalone BF16 rounding."""
    width = intermediate * 2
    tile = 256
    dequantize = make_dequantize(capacity, width)

    if not fused_activation:
        @pl.jit.inline
        def bf16_activation(
            accumulator: pl.Tensor[[capacity, width], pl.INT32],
            channels: pl.Tensor[[width], pl.FP32],
            scales: pl.Tensor[[capacity, 8], pl.FP32],
            output: pl.Tensor[[capacity, intermediate], pl.BF16],
            active: pl.Scalar[pl.INT32],
        ):
            gate_up = pl.create_tensor([capacity, width], dtype=pl.BF16)
            dequantize(accumulator, channels, scales, gate_up, active)
            row_blocks = (active + VEC_M_TILE - 1) // VEC_M_TILE
            for block in pl.spmd(row_blocks * (intermediate // tile), name_hint="a2_moe_swiglu"):
                token = block // (intermediate // tile) * VEC_M_TILE
                column = block % (intermediate // tile) * tile
                rows = pl.min(VEC_M_TILE, active - token)
                up_column = intermediate + column
                gate_bf16 = pl.slice(gate_up, [VEC_M_TILE, tile], [token, column], valid_shape=[rows, tile])
                up_bf16 = pl.slice(gate_up, [VEC_M_TILE, tile], [token, up_column], valid_shape=[rows, tile])
                gate_bf16 = pl.set_validshape(pl.fillpad(gate_bf16, pad_value=pl.PadValue.zero), VEC_M_TILE, tile)
                up_bf16 = pl.set_validshape(pl.fillpad(up_bf16, pad_value=pl.PadValue.zero), VEC_M_TILE, tile)
                gate = pl.minimum(pl.cast(gate_bf16, pl.FP32), limit)
                up = pl.maximum(pl.minimum(pl.cast(up_bf16, pl.FP32), limit), -limit)
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate)), 1.0))
                sigmoid_bf16 = pl.cast(sigmoid, pl.BF16, mode="rint")
                silu_bf16 = pl.cast(pl.mul(gate, pl.cast(sigmoid_bf16, pl.FP32)), pl.BF16, mode="rint")
                activated = pl.mul(pl.cast(silu_bf16, pl.FP32), up)
                narrowed = pl.cast(activated, pl.BF16, mode="rint")
                output[token:token + VEC_M_TILE, column:column + tile] = pl.set_validshape(narrowed, rows, tile)
            return output

        return bf16_activation

    @pl.jit.inline
    def fp32_activation(
        accumulator: pl.Tensor[[capacity, width], pl.INT32],
        channels: pl.Tensor[[width], pl.FP32],
        scales: pl.Tensor[[capacity, 8], pl.FP32],
        output: pl.Tensor[[capacity, intermediate], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        row_blocks = (active + VEC_M_TILE - 1) // VEC_M_TILE
        for block in pl.spmd(row_blocks * (intermediate // tile), name_hint="a2_moe_swiglu_fused"):
            token = block // (intermediate // tile) * VEC_M_TILE
            column = block % (intermediate // tile) * tile
            rows = pl.min(VEC_M_TILE, active - token)
            up_column = intermediate + column
            gate_i32 = pl.slice(accumulator, [VEC_M_TILE, tile], [token, column], valid_shape=[rows, tile])
            up_i32 = pl.slice(accumulator, [VEC_M_TILE, tile], [token, up_column], valid_shape=[rows, tile])
            gate_channels = pl.reshape(channels[column:column + tile], [1, tile])
            up_channels = pl.reshape(channels[up_column:up_column + tile], [1, tile])
            row_scale = pl.row_min(scales[token:token + VEC_M_TILE, :])
            gate = pl.row_expand_mul(pl.col_expand_mul(pl.cast(gate_i32, pl.FP32), gate_channels), row_scale)
            up = pl.row_expand_mul(pl.col_expand_mul(pl.cast(up_i32, pl.FP32), up_channels), row_scale)
            gate = pl.set_validshape(pl.fillpad(gate, pad_value=pl.PadValue.zero), VEC_M_TILE, tile)
            up = pl.set_validshape(pl.fillpad(up, pad_value=pl.PadValue.zero), VEC_M_TILE, tile)
            gate = pl.minimum(gate, limit)
            up = pl.maximum(pl.minimum(up, limit), -limit)
            denominator = pl.add(pl.exp(pl.neg(gate)), 1.0)
            silu = pl.div(gate, denominator)
            activated = pl.mul(silu, up)
            output[token:token + VEC_M_TILE, column:column + tile] = pl.set_validshape(activated, rows, tile)
        return output

    return fp32_activation


def make_shared_moe_w8a8_inline(capacity, config, *, fused_activation=False):
    """Build shared gate/up, clamped SwiGLU, and down projection."""
    hidden = config["hidden_size"]
    intermediate = config["moe_intermediate_size"]
    gate_up_width = intermediate * 2
    limit = config["swiglu_limit"]
    gate_up_packed_rows = hidden * (gate_up_width // N_TILE)
    down_packed_rows = intermediate * (hidden // N_TILE)
    activation_dtype = pl.FP32 if fused_activation else pl.BF16
    activate = _make_shared_activation(capacity, intermediate, limit, fused_activation)

    quant_x = make_quantize(capacity, hidden)
    quant_hidden = make_quantize(capacity, intermediate, activation_dtype)
    pack_gate_up = make_pack_weight(hidden, gate_up_width)
    pack_down = make_pack_weight(intermediate, hidden)
    dot_gate_up = make_dot(capacity, hidden, gate_up_width)
    dot_down = make_dot(capacity, intermediate, hidden)
    dequant_down = make_dequantize(capacity, hidden)

    @pl.jit.inline
    def shared_moe_w8a8_inline(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        gate_up_weight: pl.Tensor[[hidden, gate_up_width], pl.INT8],
        gate_up_scale: pl.Tensor[[gate_up_width], pl.FP32],
        down_weight: pl.Tensor[[intermediate, hidden], pl.INT8],
        down_scale: pl.Tensor[[hidden], pl.FP32],
        output: pl.Tensor[[capacity, hidden], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        x_payload = pl.create_tensor([capacity, hidden], dtype=pl.INT8)
        x_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quant_x(x, x_payload, x_scale, active)

        gate_up_packed = pl.create_tensor([gate_up_packed_rows, N_TILE], dtype=pl.INT8)
        pack_gate_up(gate_up_weight, gate_up_packed)
        gate_up_acc = pl.create_tensor([capacity, gate_up_width], dtype=pl.INT32)
        dot_gate_up(x_payload, gate_up_packed, gate_up_acc, active)
        activation = pl.create_tensor([capacity, intermediate], dtype=activation_dtype)
        activate(gate_up_acc, gate_up_scale, x_scale, activation, active)

        hidden_payload = pl.create_tensor([capacity, intermediate], dtype=pl.INT8)
        hidden_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quant_hidden(activation, hidden_payload, hidden_scale, active)
        down_packed = pl.create_tensor([down_packed_rows, N_TILE], dtype=pl.INT8)
        pack_down(down_weight, down_packed)
        down_acc = pl.create_tensor([capacity, hidden], dtype=pl.INT32)
        dot_down(hidden_payload, down_packed, down_acc, active)
        dequant_down(down_acc, down_scale, hidden_scale, output, active)
        return output

    return shared_moe_w8a8_inline


def make_shared_moe_w8a8(capacity, config, *, fused_activation=False):
    """Build a standalone JIT kernel for the shared expert."""
    hidden = config["hidden_size"]
    intermediate = config["moe_intermediate_size"]
    gate_up_width = intermediate * 2
    compute = make_shared_moe_w8a8_inline(capacity, config, fused_activation=fused_activation)

    @pl.jit
    def shared_moe_w8a8(
        x: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        gate_up_weight: pl.Tensor[[hidden, gate_up_width], pl.INT8],
        gate_up_scale: pl.Tensor[[gate_up_width], pl.FP32],
        down_weight: pl.Tensor[[intermediate, hidden], pl.INT8],
        down_scale: pl.Tensor[[hidden], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute(x, gate_up_weight, gate_up_scale, down_weight, down_scale, output, active)
        return x, output

    return shared_moe_w8a8
