# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Eco-Tech dynamic INT8 linear with the native BF16 parameter scale boundary."""

import torch
import pypto.language as pl

from .ops import make_dequantize, make_dot, make_pack_weight, make_quantize


def prepare_ecotech_dynamic_weight(weight_nk, source_scale, source_offset):
    """Preprocess CPU checkpoint operands into KN INT8 and effective FP32 scales."""
    values = (weight_nk, source_scale, source_offset)
    if any(value.device.type != "cpu" for value in values):
        raise ValueError("Checkpoint preprocessing requires CPU source tensors")
    if weight_nk.ndim != 2 or weight_nk.dtype != torch.int8:
        raise ValueError("The source weight must be a complete NK INT8 tensor")
    output_width, input_width = weight_nk.shape
    if input_width % 256 or output_width % 64:
        raise ValueError("The current A2 projection requires K/256 and N/64 alignment")
    for value in (source_scale, source_offset):
        if value.shape != (output_width, 1) or value.dtype != torch.float32:
            raise ValueError("Source per-channel scale and offset require FP32 [N,1]")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Source scale and offset must be finite")
    if not bool((source_offset == 0).all()):
        raise ValueError("The symmetric native INT8 path requires zero checkpoint offsets")
    native_scale = source_scale.to(torch.bfloat16)
    if not bool(torch.isfinite(native_scale).all()) or not bool((native_scale > 0).all()):
        raise ValueError("Effective BF16 channel scales must remain finite and positive")
    weight_kn = weight_nk.T.contiguous()
    channel_scale = native_scale.to(torch.float32).flatten()
    return weight_kn, channel_scale


def make_ecotech_dynamic_linear(capacity, input_width, output_width, combine_scales=False):
    """Compose existing INT8 quantization, full matmul and FP32 scale operators."""
    if capacity <= 0 or capacity % 16 or input_width % 256 or output_width % 64:
        raise ValueError("Use a positive capacity/16, K/256 and N/64")
    packed_rows = input_width * (output_width // 64)
    pack = make_pack_weight(input_width, output_width)
    quantize = make_quantize(capacity, input_width)
    dot = make_dot(capacity, input_width, output_width)
    dequantize = make_dequantize(capacity, output_width, combine_scales=combine_scales)

    @pl.jit
    def linear(
        x: pl.Tensor[[capacity, input_width], pl.BF16],
        weight: pl.Tensor[[input_width, output_width], pl.INT8],
        channel_scale: pl.Tensor[[output_width], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, output_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        packed = pl.create_tensor([packed_rows, 64], dtype=pl.INT8)
        pack(weight, packed)
        payload = pl.create_tensor([capacity, input_width], dtype=pl.INT8)
        token_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize(x, payload, token_scale, active)
        accumulator = pl.create_tensor([capacity, output_width], dtype=pl.INT32)
        dot(payload, packed, accumulator, active)
        dequantize(accumulator, channel_scale, token_scale, output, active)
        return output

    return linear
