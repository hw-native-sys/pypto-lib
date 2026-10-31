"""A2/A3 INT8 GEMM and FP32 dequantization kernels."""

from __future__ import annotations

import pypto.language as pl

M_TILE = 16
N_TILE = 64
K_TILE = 256


def make_pack_weight(width: int, output_width: int):
    """Pack canonical ``[K, N]`` INT8 weights into 64-column blocks."""

    if output_width % N_TILE or width % K_TILE:
        raise ValueError("A2/A3 weight packing requires K divisible by 256 and N by 64")
    packed_rows = width * (output_width // N_TILE)

    @pl.jit.inline
    def pack(
        weight: pl.Tensor[[width, output_width], pl.INT8],
        packed: pl.Tensor[[packed_rows, N_TILE], pl.INT8],
    ):
        for block in pl.spmd(output_width // N_TILE, name_hint="dsv41_a2a3_pack_weight"):
            n0 = block * N_TILE
            for k0 in pl.range(0, width, K_TILE):
                packed[block * width + k0:block * width + k0 + K_TILE, :] = weight[
                    k0:k0 + K_TILE, n0:n0 + N_TILE
                ]
        return packed

    return pack


def make_int8_gemm(capacity: int, width: int, output_width: int):
    """Accumulate INT8 activations and packed INT8 weights into INT32."""

    if capacity % M_TILE or width % K_TILE:
        raise ValueError("capacity must be divisible by 16 and K by 256")
    packed_rows = width * (output_width // N_TILE)

    @pl.jit.inline
    def gemm(
        payload: pl.Tensor[[capacity, width], pl.INT8],
        packed: pl.Tensor[[packed_rows, N_TILE], pl.INT8],
        accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        blocks = output_width // N_TILE
        for block in pl.spmd(((active + 15) // M_TILE) * blocks, name_hint="dsv41_a2a3_int8_gemm"):
            t0 = (block // blocks) * 16
            n0 = (block % blocks) * N_TILE
            rows = pl.min(M_TILE, active - t0)
            acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, width, K_TILE, stage=2):
                lhs = pl.slice(payload, [M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                lhs = pl.set_validshape(pl.fillpad(lhs, pad_value=pl.PadValue.zero), M_TILE, K_TILE)
                rhs = packed[(block % blocks) * width + k0:(block % blocks) * width + k0 + K_TILE, :]
                acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(k0 == 0))
            accumulator[t0:t0 + M_TILE, n0:n0 + N_TILE] = pl.set_validshape(acc, rows, N_TILE)
        return accumulator

    return gemm


def make_fp32_dequantize(capacity: int, output_width: int):
    """Apply token/channel FP32 scales and narrow the result to BF16."""

    if output_width % N_TILE:
        raise ValueError("output width must be divisible by 64")

    @pl.jit.inline
    def dequantize(
        accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
        token_scale: pl.Tensor[[capacity, 1], pl.FP32],
        channel_scale: pl.Tensor[[output_width], pl.FP32],
        output: pl.Tensor[[capacity, output_width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        blocks = output_width // N_TILE
        for block in pl.spmd(((active + 7) // 8) * blocks, name_hint="dsv41_a2a3_dequant"):
            t0 = (block // blocks) * 8
            n0 = (block % blocks) * N_TILE
            rows = pl.min(8, active - t0)
            value = pl.cast(accumulator[t0:t0 + 8, n0:n0 + N_TILE], pl.FP32)
            channels = pl.reshape(channel_scale[n0:n0 + N_TILE], [1, N_TILE])
            value = pl.col_expand_mul(value, channels)
            # Preserve the 2-D [rows, 1] view so the backend emits the legal
            # row-scale load used by the A2/A3 W8A8 kernels.
            scale = token_scale[t0:t0 + 8, 0:1]
            value = pl.row_expand_mul(value, scale)
            output[t0:t0 + 8, n0:n0 + N_TILE] = pl.set_validshape(
                pl.cast(value, pl.BF16, mode="rint"), rows, N_TILE
            )
        return output

    return dequantize


def make_int8_linear_direct(capacity: int, width: int, output_width: int):
    """Build a W8A8 linear that consumes canonical ``[K, N]`` weights.

    The packed-weight helper is useful for the leaf ABI, but materializing a
    full packed matrix is too large for the A2/A3 Mat buffer at V4.1 widths.
    This variant loads one ``[K_TILE, N_TILE]`` canonical tile per GEMM
    reduction and keeps only the accumulator tile on core.
    """

    if capacity % M_TILE or width % K_TILE or output_width % N_TILE:
        raise ValueError("direct INT8 GEMM requires aligned capacity, K and N")

    from .quant import make_dynamic_int8_quantize

    quantize = make_dynamic_int8_quantize(capacity, width)
    blocks = output_width // N_TILE

    @pl.jit
    def linear(
        value: pl.Tensor[[capacity, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.INT8],
        channel_scale: pl.Tensor[[output_width], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, output_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        payload = pl.create_tensor([capacity, width], dtype=pl.INT8)
        token_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quantize(value, payload, token_scale, active)
        for block in pl.spmd(((active + M_TILE - 1) // M_TILE) * blocks,
                             name_hint="dsv41_a2a3_direct_int8_linear"):
            t0 = (block // blocks) * M_TILE
            n0 = (block % blocks) * N_TILE
            rows = pl.min(M_TILE, active - t0)
            acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, width, K_TILE, stage=2):
                lhs = pl.slice(payload, [M_TILE, K_TILE], [t0, k0],
                               valid_shape=[rows, K_TILE])
                lhs = pl.set_validshape(pl.fillpad(lhs, pad_value=pl.PadValue.zero), M_TILE, K_TILE)
                rhs = weight[k0:k0 + K_TILE, n0:n0 + N_TILE]
                acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(k0 == 0))
            value_fp32 = pl.cast(acc, pl.FP32)
            channels = pl.reshape(channel_scale[n0:n0 + N_TILE], [1, N_TILE])
            value_fp32 = pl.col_expand_mul(value_fp32, channels)
            value_fp32 = pl.row_expand_mul(value_fp32, token_scale[t0:t0 + M_TILE, :])
            output[t0:t0 + M_TILE, n0:n0 + N_TILE] = pl.set_validshape(
                pl.cast(value_fp32, pl.BF16, mode="rint"), rows, N_TILE
            )
        return output

    return linear


__all__ = [
    "make_fp32_dequantize",
    "make_int8_gemm",
    "make_int8_linear_direct",
    "make_pack_weight",
]
