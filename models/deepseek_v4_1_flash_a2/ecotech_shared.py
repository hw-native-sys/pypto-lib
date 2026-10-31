# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Dynamic W8A8 shared gate/up, clamped FP32 SwiGLU and INT8 down projection."""

import pypto.language as pl
import torch

from .ecotech_linear import prepare_ecotech_dynamic_weight
from .ecotech_linear_tail import prepare_ecotech_tail_weight
from .ops import make_dequantize, make_dot, make_pack_weight, make_quantize


def prepare_ecotech_shared_weights(gate, up, down):
    """Prepare the original CPU NK weight/F32 scale/offset triples."""
    if any(len(values) != 3 for values in (gate, up, down)):
        raise ValueError("Require weight, scale and offset for every projection")
    if gate[0].shape != up[0].shape or gate[0].ndim != 2:
        raise ValueError("Gate and up require the same local NK shape")
    intermediate, hidden = gate[0].shape
    if down[0].shape != (hidden, intermediate):
        raise ValueError("Down must consume the complete local intermediate")
    merged = tuple(torch.cat([left, right], dim=0) for left, right in zip(gate, up))
    gate_weight, gate_scale = prepare_ecotech_dynamic_weight(*merged)
    down_weight, down_scale = prepare_ecotech_tail_weight(*down)
    return gate_weight, gate_scale, down_weight, down_scale


def _make_shared_activation(capacity, intermediate, padded_intermediate, limit):
    width = intermediate * 2

    @pl.jit.inline
    def activate(
        accumulator: pl.Tensor[[capacity, width], pl.INT32],
        channels: pl.Tensor[[width], pl.FP32],
        scales: pl.Tensor[[capacity, 8], pl.FP32],
        output: pl.Tensor[[capacity, padded_intermediate], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(((active + 7) // 8) * (intermediate // 32), name_hint="ecotech_shared_swiglu"):
            token = block // (intermediate // 32) * 8
            column = block % (intermediate // 32) * 32
            up_column = intermediate + column
            rows = pl.min(8, active - token)
            gate_i32 = pl.slice(accumulator, [8, 32], [token, column], valid_shape=[rows, 32])
            up_i32 = pl.slice(accumulator, [8, 32], [token, up_column], valid_shape=[rows, 32])
            gate_channels = pl.reshape(channels[column:column + 32], [1, 32])
            up_channels = pl.reshape(channels[up_column:up_column + 32], [1, 32])
            row_scale = pl.row_min(scales[token:token + 8, :])
            gate_fp32 = pl.cast(gate_i32, pl.FP32, mode="rint")
            up_fp32 = pl.cast(up_i32, pl.FP32, mode="rint")
            gate_channel_scaled = pl.col_expand_mul(gate_fp32, gate_channels)
            up_channel_scaled = pl.col_expand_mul(up_fp32, up_channels)
            gate_scaled = pl.row_expand_mul(gate_channel_scaled, row_scale)
            up_scaled = pl.row_expand_mul(up_channel_scaled, row_scale)
            gate_filled = pl.fillpad(gate_scaled, pad_value=pl.PadValue.zero)
            up_filled = pl.fillpad(up_scaled, pad_value=pl.PadValue.zero)
            gate_full = pl.set_validshape(gate_filled, 8, 32)
            up_full = pl.set_validshape(up_filled, 8, 32)
            gate = pl.minimum(gate_full, limit)
            up_upper = pl.minimum(up_full, limit)
            up = pl.maximum(up_upper, -limit)
            negative_gate = pl.neg(gate)
            exponent = pl.exp(negative_gate)
            denominator = pl.add(exponent, 1.0)
            silu = pl.div(gate, denominator)
            activation = pl.mul(silu, up)
            output[token:token + 8, column:column + 32] = activation
        if intermediate < padded_intermediate:
            for block in pl.spmd((active + 7) // 8, name_hint="ecotech_shared_activation_pad"):
                token = block * 8
                for column in pl.range(intermediate, padded_intermediate, 32):
                    output[token:token + 8, column:column + 32] = pl.full([8, 32], dtype=pl.FP32, value=0.0)
        return output

    return activate


def make_ecotech_shared_inline(capacity, hidden, intermediate, *, limit=10.0):
    """Compose local shared computation with an explicit produced INT8 handoff."""
    if capacity <= 0 or capacity % 16 or hidden <= 0 or hidden % 256:
        raise ValueError("Require a positive capacity/16 and hidden/256")
    if intermediate <= 0 or intermediate % 32 or limit != 10.0:
        raise ValueError("Require local intermediate/32 and the source clamp10")
    width = intermediate * 2
    padded_intermediate = ((intermediate + 255) // 256) * 256
    gate_packed_rows = hidden * (width // 64)
    down_packed_rows = padded_intermediate * (hidden // 64)
    quantize_input = make_quantize(capacity, hidden)
    pack_gate = make_pack_weight(hidden, width)
    dot_gate = make_dot(capacity, hidden, width)
    activate = _make_shared_activation(capacity, intermediate, padded_intermediate, limit)
    quantize_down = make_quantize(capacity, padded_intermediate, pl.FP32)
    pack_down = make_pack_weight(padded_intermediate, hidden)
    dot_down = make_dot(capacity, padded_intermediate, hidden)
    dequantize_down = make_dequantize(capacity, hidden)

    @pl.jit.inline
    def shared(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        gate_weight: pl.Tensor[[hidden, width], pl.INT8],
        gate_scale: pl.Tensor[[width], pl.FP32],
        down_weight: pl.Tensor[[padded_intermediate, hidden], pl.INT8],
        down_scale: pl.Tensor[[hidden], pl.FP32],
        output: pl.Tensor[[capacity, hidden], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        input_payload = pl.create_tensor([capacity, hidden], dtype=pl.INT8)
        input_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize_input(x, input_payload, input_scale, active)
        gate_packed = pl.create_tensor([gate_packed_rows, 64], dtype=pl.INT8)
        pack_gate(gate_weight, gate_packed)
        gate_accumulator = pl.create_tensor([capacity, width], dtype=pl.INT32)
        dot_gate(input_payload, gate_packed, gate_accumulator, active)
        activation = pl.create_tensor([capacity, padded_intermediate], dtype=pl.FP32)
        activate(gate_accumulator, gate_scale, input_scale, activation, active)
        down_payload = pl.create_tensor([capacity, padded_intermediate], dtype=pl.INT8)
        down_token_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize_down(activation, down_payload, down_token_scale, active)
        down_packed = pl.create_tensor([down_packed_rows, 64], dtype=pl.INT8)
        pack_down(down_weight, down_packed)
        down_accumulator = pl.create_tensor([capacity, hidden], dtype=pl.INT32)
        dot_down(down_payload, down_packed, down_accumulator, active)
        dequantize_down(down_accumulator, down_scale, down_token_scale, output, active)
        return output, input_payload, input_scale, gate_accumulator, activation, down_payload, down_token_scale

    return shared


def make_ecotech_shared(capacity, hidden, intermediate):
    """Build the local shared-expert entry without outer TP reduction."""
    width = intermediate * 2
    padded_intermediate = ((intermediate + 255) // 256) * 256
    compute = make_ecotech_shared_inline(capacity, hidden, intermediate)

    @pl.jit
    def shared(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        gate_weight: pl.Tensor[[hidden, width], pl.INT8],
        gate_scale: pl.Tensor[[width], pl.FP32],
        down_weight: pl.Tensor[[padded_intermediate, hidden], pl.INT8],
        down_scale: pl.Tensor[[hidden], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute(x, gate_weight, gate_scale, down_weight, down_scale, output, active)
        return output

    return shared
