# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2 blocked INT8 projections, FP32 scaling, RMSNorm and adjacent-pair RoPE."""

import pypto.language as pl


# tiling
M_TILE = 16
VEC_M_TILE = 8
K_TILE = 256
N_TILE = 64


def make_cast_tile(rows, width, input_dtype, output_dtype):
    """Choose identity or a rounding cast before the DSL parses either body."""
    # PyPTO needs a concrete Tensor annotation while parsing an inline body.
    # A closure over ``input_dtype`` is a Python value, so the generic
    # annotation used here previously lost the tensor metadata during a full
    # layer compile.  Keep the small set of dtypes used by the model explicit.
    if input_dtype == pl.BF16 and output_dtype == pl.BF16:
        @pl.jit.inline
        def identity_bf16(value: pl.Tensor[[rows, width], pl.BF16]):
            return value
        return identity_bf16
    if input_dtype == pl.BF16 and output_dtype == pl.FP32:
        @pl.jit.inline
        def cast_bf16_fp32(value: pl.Tensor[[rows, width], pl.BF16]):
            return pl.cast(value, pl.FP32, mode="rint")
        return cast_bf16_fp32
    if input_dtype == pl.FP32 and output_dtype == pl.FP32:
        @pl.jit.inline
        def identity_fp32(value: pl.Tensor[[rows, width], pl.FP32]):
            return value
        return identity_fp32
    if input_dtype == pl.FP32 and output_dtype == pl.BF16:
        @pl.jit.inline
        def cast_fp32_bf16(value: pl.Tensor[[rows, width], pl.FP32]):
            return pl.cast(value, pl.BF16, mode="rint")
        return cast_fp32_bf16
    raise ValueError(f"unsupported cast tile dtypes: {input_dtype} -> {output_dtype}")


def make_quantize(capacity, width, input_dtype=pl.BF16):
    """Specialize symmetric per-token INT8 quantization."""
    if input_dtype == pl.FP32:
        @pl.jit.inline
        def quantize(
            x: pl.Tensor[[capacity, width], pl.FP32],
            payload: pl.Tensor[[capacity, width], pl.INT8],
            scales: pl.Tensor[[capacity, 8], pl.FP32],
            active: pl.Scalar[pl.INT32],
        ):
            for block in pl.spmd((active + VEC_M_TILE - 1) // VEC_M_TILE, name_hint="a2_token_quantize"):
                t0 = block * VEC_M_TILE
                rows = pl.min(VEC_M_TILE, active - t0)
                maximum_row = pl.full([1, VEC_M_TILE], dtype=pl.FP32, value=1e-30)
                for k0 in pl.range(0, width, K_TILE):
                    source = pl.slice(x, [VEC_M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                    source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), VEC_M_TILE, K_TILE)
                    chunk_maximum = pl.reshape(pl.row_max(pl.abs(source)), [1, VEC_M_TILE])
                    maximum_row = pl.maximum(maximum_row, chunk_maximum)
                maximum = pl.reshape(maximum_row, [VEC_M_TILE, 1])
                scale = pl.mul(maximum, 1.0 / 127.0)
                numerator = pl.full([1, VEC_M_TILE], dtype=pl.FP32, value=127.0)
                inverse_row = pl.div(numerator, maximum_row)
                inverse = pl.reshape(inverse_row, [VEC_M_TILE, 1])
                scale_rows = pl.full([VEC_M_TILE, 8], dtype=pl.FP32, value=1.0)
                scales[t0:t0 + VEC_M_TILE, :] = pl.row_expand_mul(scale_rows, scale)
                for k0 in pl.range(0, width, K_TILE):
                    source = pl.slice(x, [VEC_M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                    source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), VEC_M_TILE, K_TILE)
                    normalized = pl.row_expand_mul(source, inverse)
                    bounded = pl.minimum(pl.maximum(normalized, -127.0), 127.0)
                    rounded = pl.cast(bounded, pl.INT32, mode="rint")
                    exact_half = pl.cast(rounded, pl.FP16, mode="rint")
                    payload[t0:t0 + VEC_M_TILE, k0:k0 + K_TILE] = pl.cast(exact_half, pl.INT8, mode="trunc")
            return payload, scales
        return quantize

    @pl.jit.inline
    def quantize(
        x: pl.Tensor[[capacity, width], input_dtype],
        payload: pl.Tensor[[capacity, width], pl.INT8],
        scales: pl.Tensor[[capacity, 8], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd((active + VEC_M_TILE - 1) // VEC_M_TILE, name_hint="a2_token_quantize"):
            t0 = block * VEC_M_TILE
            rows = pl.min(VEC_M_TILE, active - t0)
            maximum_row = pl.full([1, VEC_M_TILE], dtype=pl.FP32, value=1e-30)
            for k0 in pl.range(0, width, K_TILE):
                source = pl.slice(x, [VEC_M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), VEC_M_TILE, K_TILE)
                value = pl.cast(source, pl.FP32, mode="rint")
                chunk_maximum = pl.reshape(pl.row_max(pl.abs(value)), [1, VEC_M_TILE])
                maximum_row = pl.maximum(maximum_row, chunk_maximum)
            maximum = pl.reshape(maximum_row, [VEC_M_TILE, 1])
            scale = pl.mul(maximum, 1.0 / 127.0)
            numerator = pl.full([1, VEC_M_TILE], dtype=pl.FP32, value=127.0)
            inverse_row = pl.div(numerator, maximum_row)
            inverse = pl.reshape(inverse_row, [VEC_M_TILE, 1])
            scale_rows = pl.full([VEC_M_TILE, 8], dtype=pl.FP32, value=1.0)
            scales[t0:t0 + VEC_M_TILE, :] = pl.row_expand_mul(scale_rows, scale)
            for k0 in pl.range(0, width, K_TILE):
                source = pl.slice(x, [VEC_M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), VEC_M_TILE, K_TILE)
                value = pl.cast(source, pl.FP32, mode="rint")
                normalized = pl.row_expand_mul(value, inverse)
                bounded = pl.minimum(pl.maximum(normalized, -127.0), 127.0)
                rounded = pl.cast(bounded, pl.INT32, mode="rint")
                exact_half = pl.cast(rounded, pl.FP16, mode="rint")
                payload[t0:t0 + VEC_M_TILE, k0:k0 + K_TILE] = pl.cast(exact_half, pl.INT8, mode="trunc")
        return payload, scales

    return quantize


def make_pack_weight(width, output_width):
    """Reorder canonical [K,N] INT8 bytes to [N/64,K,64]."""
    packed_rows = width * (output_width // N_TILE)

    @pl.jit.inline
    def pack_weight(
        weight: pl.Tensor[[width, output_width], pl.INT8],
        packed: pl.Tensor[[packed_rows, N_TILE], pl.INT8],
    ):
        for nb in pl.spmd(output_width // N_TILE, name_hint="a2_pack_weight"):
            n0 = nb * N_TILE
            for k0 in pl.range(0, width, K_TILE):
                block = weight[k0:k0 + K_TILE, n0:n0 + N_TILE]
                packed_k0 = nb * width + k0
                packed[packed_k0:packed_k0 + K_TILE, :] = block
        return packed

    return pack_weight


def make_dot(capacity, width, output_width):
    """Accumulate complete blocked INT8 products in INT32."""
    packed_rows = width * (output_width // N_TILE)

    @pl.jit.inline
    def dot(
        payload: pl.Tensor[[capacity, width], pl.INT8],
        packed: pl.Tensor[[packed_rows, N_TILE], pl.INT8],
        accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        row_blocks = (active + M_TILE - 1) // M_TILE
        for block in pl.spmd(row_blocks * (output_width // N_TILE), name_hint="a2_int8_dot"):
            t0 = (block // (output_width // N_TILE)) * M_TILE
            nb = block % (output_width // N_TILE)
            n0 = nb * N_TILE
            rows = pl.min(M_TILE, active - t0)
            acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, width, K_TILE, stage=2):
                a = pl.slice(payload, [M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                a = pl.set_validshape(pl.fillpad(a, pad_value=pl.PadValue.zero), M_TILE, K_TILE)
                packed_k0 = nb * width + k0
                b = packed[packed_k0:packed_k0 + K_TILE, :]
                acc = pl.matmul_acc(acc, a, b, init_cond=(k0 == 0))
            accumulator[t0:t0 + M_TILE, n0:n0 + N_TILE] = acc
        return accumulator

    return dot


def make_dequantize(capacity, output_width, combine_scales=False, output_dtype=pl.BF16):
    """Apply the selected native FP32 scale boundary to INT32 products.

    The native per-token optimized kernel combines channel/token scales before
    multiplying the accumulator. Other native kernels scale channel-first.
    Callers select their source boundary explicitly; QKV keeps its existing ABI.
    """
    if output_dtype == pl.FP32:
        @pl.jit.inline
        def dequantize(
            accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
            channel_scale: pl.Tensor[[output_width], pl.FP32],
            token_scale: pl.Tensor[[capacity, 8], pl.FP32],
            output: pl.Tensor[[capacity, output_width], pl.FP32],
            active: pl.Scalar[pl.INT32],
        ):
            row_blocks = (active + VEC_M_TILE - 1) // VEC_M_TILE
            for block in pl.spmd(row_blocks * (output_width // N_TILE), name_hint="a2_dequantize_fp32"):
                t0 = (block // (output_width // N_TILE)) * VEC_M_TILE
                n0 = (block % (output_width // N_TILE)) * N_TILE
                rows = pl.min(VEC_M_TILE, active - t0)
                integer = accumulator[t0:t0 + VEC_M_TILE, n0:n0 + N_TILE]
                value = pl.cast(integer, pl.FP32, mode="rint")
                channels = pl.reshape(channel_scale[n0:n0 + N_TILE], [1, N_TILE])
                scale = pl.row_min(token_scale[t0:t0 + VEC_M_TILE, :])
                if combine_scales:
                    ones = pl.full([VEC_M_TILE, N_TILE], dtype=pl.FP32, value=1.0)
                    channel_rows = pl.col_expand_mul(ones, channels)
                    combined = pl.row_expand_mul(channel_rows, scale)
                    scaled = pl.mul(value, combined)
                else:
                    scaled_channels = pl.col_expand_mul(value, channels)
                    scaled = pl.row_expand_mul(scaled_channels, scale)
                output[t0:t0 + VEC_M_TILE, n0:n0 + N_TILE] = pl.set_validshape(scaled, rows, N_TILE)
            return output

        return dequantize

    @pl.jit.inline
    def dequantize(
        accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
        channel_scale: pl.Tensor[[output_width], pl.FP32],
        token_scale: pl.Tensor[[capacity, 8], pl.FP32],
        output: pl.Tensor[[capacity, output_width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        row_blocks = (active + VEC_M_TILE - 1) // VEC_M_TILE
        for block in pl.spmd(row_blocks * (output_width // N_TILE), name_hint="a2_dequantize"):
            t0 = (block // (output_width // N_TILE)) * VEC_M_TILE
            n0 = (block % (output_width // N_TILE)) * N_TILE
            rows = pl.min(VEC_M_TILE, active - t0)
            integer = accumulator[t0:t0 + VEC_M_TILE, n0:n0 + N_TILE]
            value = pl.cast(integer, pl.FP32, mode="rint")
            channels = pl.reshape(channel_scale[n0:n0 + N_TILE], [1, N_TILE])
            scale = pl.row_min(token_scale[t0:t0 + VEC_M_TILE, :])
            if combine_scales:
                ones = pl.full([VEC_M_TILE, N_TILE], dtype=pl.FP32, value=1.0)
                channel_rows = pl.col_expand_mul(ones, channels)
                combined = pl.row_expand_mul(channel_rows, scale)
                scaled = pl.mul(value, combined)
            else:
                scaled_channels = pl.col_expand_mul(value, channels)
                scaled = pl.row_expand_mul(scaled_channels, scale)
            narrowed = pl.cast(scaled, pl.BF16, mode="rint")
            output[t0:t0 + VEC_M_TILE, n0:n0 + N_TILE] = pl.set_validshape(narrowed, rows, N_TILE)
        return output

    return dequantize


def make_norm(capacity, width, eps):
    """Specialize weighted FP32 RMSNorm with a BF16 output."""

    @pl.jit.inline
    def norm(
        x: pl.Tensor[[capacity, width], pl.BF16],
        weight: pl.Tensor[[width], pl.BF16],
        output: pl.Tensor[[capacity, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd((active + VEC_M_TILE - 1) // VEC_M_TILE, name_hint="a2_rms_norm"):
            t0 = block * VEC_M_TILE
            rows = pl.min(VEC_M_TILE, active - t0)
            source = pl.slice(x, [VEC_M_TILE, width], [t0, 0], valid_shape=[rows, width])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), VEC_M_TILE, width)
            value = pl.cast(source, pl.FP32)
            square = pl.mul(value, value)
            mean = pl.mul(pl.row_sum(square), 1.0 / width)
            inverse = pl.rsqrt(pl.add(mean, eps), high_precision=True)
            normalized = pl.row_expand_mul(value, inverse)
            gamma = pl.reshape(pl.cast(weight[:], pl.FP32), [1, width])
            weighted = pl.col_expand_mul(normalized, gamma)
            narrowed = pl.cast(weighted, pl.BF16, mode="rint")
            output[t0:t0 + VEC_M_TILE, :] = pl.set_validshape(narrowed, rows, width)
        return output

    return norm


def make_rope(capacity, heads, head_dim, rope_dim):
    """Rotate the final channels of each head in adjacent pairs."""
    output_width = heads * head_dim
    nope_dim = head_dim - rope_dim
    pair_width = rope_dim // 2

    @pl.jit.inline
    def rope(
        x: pl.Tensor[[capacity, output_width], pl.BF16],
        cosine: pl.Tensor[[capacity, pair_width], pl.FP32],
        sine: pl.Tensor[[capacity, pair_width], pl.FP32],
        output: pl.Tensor[[capacity, output_width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(active * heads, name_hint="a2_adjacent_rope"):
            t = block // heads
            base = (block % heads) * head_dim
            output[t:t + 1, base:base + nope_dim] = x[t:t + 1, base:base + nope_dim]
            tail = pl.cast(x[t:t + 1, base + nope_dim:base + head_dim], pl.FP32)
            even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            c = cosine[t:t + 1, :]
            s = sine[t:t + 1, :]
            even_c = pl.mul(even, c)
            odd_s = pl.mul(odd, s)
            odd_c = pl.mul(odd, c)
            even_s = pl.mul(even, s)
            real = pl.sub(even_c, odd_s)
            imag = pl.add(odd_c, even_s)
            rotated = pl.full([1, rope_dim], dtype=pl.FP32, value=0.0)
            rotated = pl.tensor.scatter(real, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
            rotated = pl.tensor.scatter(imag, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
            output[t:t + 1, base + nope_dim:base + head_dim] = pl.cast(rotated, pl.BF16, mode="rint")
        return output

    return rope
