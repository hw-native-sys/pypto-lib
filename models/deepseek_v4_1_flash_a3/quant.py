# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 device and host quantization, checkpoint layouts, and weight fixtures."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

import torch
import pypto.language as pl

from .config import FLASH as M


# model config
MX_GROUP = M.weight_block_size[0]

# tiling
QUANT_K_TILE = 256
QUANT_SCALE_TILE = QUANT_K_TILE // 32
CODE_ROWS_DYN = pl.dynamic("A3_CODE_ROWS_DYN")
CODE_COLS_DYN = pl.dynamic("A3_CODE_COLS_DYN")


FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

# FP4 E2M1 nibble values encoded as FP8 E4M3FN bytes. The device kernel
# gathers two adjacent FP8 bytes from this table for every packed input byte.
FP4_E4M3_CODES = torch.tensor(
    [0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,
     0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC],
    dtype=torch.int16,
)


@pl.jit.inline(auto_scope=False)
def encode_fp8(
    value: pl.Tile[[CODE_ROWS_DYN, CODE_COLS_DYN], pl.FP32],
    rows: pl.constexpr,
    cols: pl.constexpr,
    integer_bits: pl.constexpr,
):
    """Encode finite FP32 values as saturated E4M3FN bytes with nearest-even rounding."""
    value = pl.reshape(value, [rows, cols])
    bits = pl.reinterpret_view(value, pl.INT32)
    if not integer_bits:
        sign_shifted = pl.cast(pl.shrs(bits, 24), pl.FP32)
        sign_bits = pl.ands(pl.cast(sign_shifted, pl.INT16, mode="rint"), 128)
    magnitude = pl.minimum(pl.abs(value), 448.0)
    absolute_bits = pl.reinterpret_view(magnitude, pl.INT32)
    if integer_bits:
        sign_integer = pl.mul(pl.shrs(bits, 31), -128)
        retained = pl.shrs(absolute_bits, 20)
        tie = pl.sub(retained, pl.shls(pl.shrs(retained, 1), 1))
    else:
        retained_fp32 = pl.cast(pl.shrs(absolute_bits, 20), pl.FP32)
        retained_i16 = pl.cast(retained_fp32, pl.INT16, mode="rint")
        tie_bits = pl.ands(retained_i16, 1)
        tie = pl.cast(pl.cast(tie_bits, pl.FP32), pl.INT32, mode="rint")
    rounded_bits = pl.add(pl.add(absolute_bits, 524287), tie)
    normal_codes = pl.sub(pl.shrs(rounded_bits, 20), 960)
    subnormal_value = pl.mul(magnitude, 512.0)
    subnormal_codes = pl.cast(subnormal_value, pl.INT32, mode="rint")
    predicate = pl.tile.cmps(magnitude, 0.015625, cmp_type=2)
    select_tmp = pl.create_tile([1, 16], dtype=pl.UINT32)
    codes = pl.tile.sel(predicate, subnormal_codes, normal_codes, select_tmp)
    saturated = pl.minimum(pl.maximum(codes, 0), 126)
    if integer_bits:
        signed_codes = pl.add(saturated, sign_integer)
    else:
        sign_fp32 = pl.cast(sign_bits, pl.FP32)
        sign_i32 = pl.cast(sign_fp32, pl.INT32, mode="rint")
        signed_codes = pl.add(saturated, sign_i32)
    signed = pl.sub(signed_codes, pl.mul(pl.shrs(signed_codes, 7), 256))
    return pl.reinterpret_view(pl.cast(signed, pl.INT8), pl.UINT8)


@pl.jit.inline(auto_scope=False)
def decode_fp8(payload: pl.Tile[[CODE_ROWS_DYN, CODE_COLS_DYN], pl.UINT8], rows: pl.constexpr, cols: pl.constexpr):
    """Decode E4M3FN bytes to exact BF16 values."""
    fixed = pl.reshape(payload, [rows, cols])
    signed_bytes = pl.reinterpret_view(fixed, pl.INT8)
    codes = pl.ands(pl.cast(pl.cast(signed_bytes, pl.FP16), pl.INT16, mode="rint"), 255)
    magnitude = pl.ands(codes, 127)
    exponent = pl.shrs(magnitude, 3)
    mantissa = pl.ands(magnitude, 7)
    factor_bits = pl.shls(pl.add(pl.cast(pl.cast(exponent, pl.FP32), pl.INT32, mode="rint"), 117), 23)
    factor = pl.reinterpret_view(factor_bits, pl.FP32)
    significand = pl.cast(pl.add(mantissa, 8), pl.FP32)
    normal = pl.mul(significand, factor)
    subnormal = pl.mul(pl.cast(mantissa, pl.FP32), 0.001953125)
    normal_mask = pl.cast(pl.minimum(exponent, 1), pl.FP32)
    normal = pl.mul(normal, normal_mask)
    subnormal_mask = pl.sub(pl.mul(normal_mask, -1.0), -1.0)
    subnormal = pl.mul(subnormal, subnormal_mask)
    value = pl.add(normal, subnormal)
    sign = pl.cast(pl.shrs(codes, 7), pl.FP32)
    signed_factor = pl.add(pl.mul(sign, -2.0), 1.0)
    return pl.cast(pl.mul(value, signed_factor), pl.BF16, mode="rint")


@pl.jit.inline(auto_scope=False)
def decode_scale(payload: pl.Tile[[CODE_ROWS_DYN, CODE_COLS_DYN], pl.UINT8], rows: pl.constexpr, cols: pl.constexpr):
    """Decode E8M0 bytes to FP32 powers of two."""
    fixed = pl.reshape(payload, [rows, cols])
    signed_bytes = pl.reinterpret_view(fixed, pl.INT8)
    widened = pl.ands(pl.cast(pl.cast(signed_bytes, pl.FP16), pl.INT16, mode="rint"), 255)
    codes = pl.cast(pl.cast(widened, pl.FP32), pl.INT32, mode="rint")
    normal_bits = pl.shls(pl.maximum(codes, 1), 23)
    normal = pl.reinterpret_view(normal_bits, pl.FP32, shape=[rows, cols])
    predicate = pl.tile.cmps(codes, 0, cmp_type=0)
    subnormal_bits = pl.tile.full([rows, cols], dtype=pl.INT32, value=4194304)
    subnormal = pl.reinterpret_view(subnormal_bits, pl.FP32, shape=[rows, cols])
    select_tmp = pl.create_tile([1, 16], dtype=pl.UINT32)
    return pl.tile.sel(predicate, subnormal, normal, select_tmp)


@pl.jit.inline(auto_scope=False)
def a3_a_scale(raw: pl.Tensor, rows: pl.constexpr, groups: pl.constexpr):
    """View logical A-side scale bytes in ND order."""
    return pl.reshape(raw, [rows, groups])


@pl.jit.inline(auto_scope=False)
def quantize_upward(source: pl.Tensor, rows: pl.constexpr, width: pl.constexpr):
    """Quantize SwiGLU rows to E4M3FN with upward group-32 E8M0 scales."""
    groups = pl.reshape(pl.cast(source, pl.FP32), [rows * (width // 32), 32])
    tmp = pl.create_tile([rows * (width // 32), 32], dtype=pl.FP32)
    maximum = pl.maximum(pl.row_max(pl.abs(groups), tmp_tile=tmp), 1e-4)
    bits = pl.reinterpret_view(pl.mul(maximum, 1.0 / 448.0), pl.INT32)
    exponent = pl.shrs(pl.add(bits, 8388607), 23)
    scale = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
    normalized = pl.row_expand_div(groups, scale)
    encoded = encode_fp8(normalized, rows * (width // 32), 32, integer_bits=False)
    payload = pl.reshape(encoded, [rows, width])
    signed = pl.sub(exponent, pl.mul(pl.shrs(exponent, 7), 256))
    codes = pl.reinterpret_view(pl.cast(signed, pl.INT8), pl.UINT8)
    return payload, pl.reshape(codes, [rows, width // 32])


@pl.jit.inline(auto_scope=False)
def encode_fp4(
    value: pl.Tile[[CODE_ROWS_DYN, CODE_COLS_DYN], pl.FP32],
    scale: pl.Tile[[CODE_ROWS_DYN, 1], pl.FP32],
    rows: pl.constexpr,
    cols: pl.constexpr,
):
    """Pack E2M1 pairs with nearest-even rounding against exact scaled midpoints."""
    value = pl.reshape(value, [rows, cols])
    scale = pl.reshape(scale, [rows, 1])
    magnitude = pl.abs(value)
    lanes = pl.tile.full([rows, cols], dtype=pl.FP32, value=1.0)
    delta0 = pl.sub(magnitude, pl.row_expand_mul(lanes, pl.mul(scale, 0.25)))
    delta1 = pl.sub(magnitude, pl.row_expand_mul(lanes, pl.mul(scale, 0.75)))
    delta2 = pl.sub(magnitude, pl.row_expand_mul(lanes, pl.mul(scale, 1.25)))
    delta3 = pl.sub(magnitude, pl.row_expand_mul(lanes, pl.mul(scale, 1.75)))
    delta4 = pl.sub(magnitude, pl.row_expand_mul(lanes, pl.mul(scale, 2.5)))
    delta5 = pl.sub(magnitude, pl.row_expand_mul(lanes, pl.mul(scale, 3.5)))
    delta6 = pl.sub(magnitude, pl.row_expand_mul(lanes, pl.mul(scale, 5.0)))
    step0 = pl.shrs(pl.sub(pl.reinterpret_view(delta0, pl.INT32), 1), 31)
    step1 = pl.shrs(pl.reinterpret_view(delta1, pl.INT32), 31)
    step2 = pl.shrs(pl.sub(pl.reinterpret_view(delta2, pl.INT32), 1), 31)
    step3 = pl.shrs(pl.reinterpret_view(delta3, pl.INT32), 31)
    step4 = pl.shrs(pl.sub(pl.reinterpret_view(delta4, pl.INT32), 1), 31)
    step5 = pl.shrs(pl.reinterpret_view(delta5, pl.INT32), 31)
    step6 = pl.shrs(pl.sub(pl.reinterpret_view(delta6, pl.INT32), 1), 31)
    lower = pl.add(pl.add(step0, step1), pl.add(step2, step3))
    upper = pl.add(pl.add(step4, step5), pl.add(step6, 7))
    index = pl.add(lower, upper)
    sign = pl.shrs(pl.reinterpret_view(value, pl.INT32), 31)
    codes = pl.reshape(pl.sub(index, pl.mul(sign, 8)), [1, rows * cols])
    low = pl.tile.gather_mask(codes, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.INT32)
    high = pl.tile.gather_mask(codes, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
    packed = pl.add(low, pl.mul(high, 16))
    signed = pl.sub(packed, pl.mul(pl.shrs(packed, 7), 256))
    return pl.reinterpret_view(pl.cast(signed, pl.INT8), pl.UINT8)


@pl.jit.inline(auto_scope=False)
def fp4_bf16_bits(code: pl.Tile[[CODE_ROWS_DYN, CODE_COLS_DYN], pl.INT16]):
    """Map an E2M1 nibble to the corresponding BF16 sign, exponent and mantissa."""
    magnitude = pl.ands(code, 7)
    normal = pl.add(pl.shls(magnitude, 6), 0x3F00)
    small = pl.mul(pl.ands(magnitude, 1), 0x3F00)
    is_small = pl.tile.cmps(pl.cast(magnitude, pl.FP16), pl.const(2.0, pl.FP16), cmp_type=2)
    bits = pl.tile.select(is_small, small, normal)
    return pl.or_(bits, pl.shls(pl.ands(code, 8), 12))


@pl.jit.inline(auto_scope=False)
def decode_fp4(payload: pl.Tile[[CODE_ROWS_DYN, CODE_COLS_DYN], pl.UINT8], rows: pl.constexpr, width: pl.constexpr):
    """Unpack E2M1 nibble pairs into BF16 rows."""
    fixed = pl.reshape(payload, [rows, width // 2])
    signed_bytes = pl.reinterpret_view(fixed, pl.INT8)
    raw = pl.cast(pl.cast(signed_bytes, pl.FP16), pl.INT16, mode="rint")
    low_word = fp4_bf16_bits(raw)
    high_word = fp4_bf16_bits(pl.shrs(raw, 4))
    low_lanes = pl.tile.full([rows, width], dtype=pl.INT16, value=0)
    low_lanes = pl.tile.scatter_mask(low_lanes, low_word, mask_pattern=pl.tile.MaskPattern.P0101)
    high_lanes = pl.tile.full([rows, width], dtype=pl.INT16, value=0)
    high_lanes = pl.tile.scatter_mask(high_lanes, high_word, mask_pattern=pl.tile.MaskPattern.P1010)
    return pl.reinterpret_view(pl.or_(low_lanes, high_lanes), pl.BF16)


@pl.jit.inline(auto_scope=False)
def make_fp8_cache_table():
    """Prepare the FP32 bits for each signed E4M3FN byte."""
    codes = pl.tile.arange(-128, [1, 256], dtype=pl.INT32)
    signed = pl.cast(codes, pl.INT8, mode="trunc")
    payload = pl.reinterpret_view(signed, pl.UINT8)
    value = pl.cast(decode_fp8(payload, 1, 256), pl.FP32)
    return pl.reinterpret_view(value, pl.INT32)


@pl.jit.inline(auto_scope=False)
def decode_fp8_cache_values(
    payload: pl.Tile[[CODE_ROWS_DYN, CODE_COLS_DYN], pl.UINT8],
    table: pl.Tile[[1, 256], pl.INT32],
    rows: pl.constexpr,
    width: pl.constexpr,
):
    """Look up the E4M3FN bytes of a Prefill cache tile."""
    bytes_flat = pl.reshape(payload, [1, rows * width])
    signed = pl.reinterpret_view(bytes_flat, pl.INT8)
    offset = pl.add(pl.cast(signed, pl.FP16), 128.0)
    index = pl.cast(pl.cast(offset, pl.FP32), pl.INT32, mode="rint")
    scratch = pl.create_tile([1, rows * width], dtype=pl.INT32, target_memory=pl.Mem.Vec)
    bits = pl.tile.gather(table, index, scratch)
    values = pl.reinterpret_view(bits, pl.FP32)
    return pl.reshape(values, [rows, width])


@pl.jit.inline(auto_scope=False)
def quantize_bf16_to_mxfp8(
    x: pl.Tensor,
    t0: pl.Scalar[pl.INT32],
    k0: pl.Scalar[pl.INT32],
    rows: pl.Scalar[pl.INT32],
    m_tile: pl.constexpr,
):
    """Quantize BF16 activation tiles to OCP E4M3FN and packed E8M0 scales."""
    source = pl.load(x, [t0, k0], [m_tile, QUANT_K_TILE], valid_shape=[rows, QUANT_K_TILE])
    source = pl.fillpad(source, pad_value=pl.PadValue.zero)
    source = pl.set_validshape(source, m_tile, QUANT_K_TILE)
    source_fp32 = pl.cast(source, pl.FP32)
    groups = pl.reshape(source_fp32, [m_tile * (QUANT_K_TILE // 32), 32])
    reduce_tmp = pl.create_tile([m_tile * (QUANT_K_TILE // 32), 32], dtype=pl.FP32)
    magnitude = pl.abs(groups)
    maximum = pl.row_max(magnitude, tmp_tile=reduce_tmp)
    bits = pl.reinterpret_view(maximum, pl.INT32)
    biased_exponent = pl.shrs(bits, 23)
    scale_exponent = pl.sub(biased_exponent, 8)
    exponent = pl.maximum(scale_exponent, 0)
    # Scale code zero has a zero reciprocal.
    negative_exponent = pl.mul(exponent, -1)
    reciprocal_exponent = pl.add(negative_exponent, 254)
    nonzero_exponent = pl.minimum(exponent, 1)
    reciprocal_code = pl.mul(reciprocal_exponent, nonzero_exponent)
    reciprocal_bits = pl.shls(reciprocal_code, 23)
    reciprocal = pl.reinterpret_view(reciprocal_bits, pl.FP32)
    normalized = pl.row_expand_mul(groups, reciprocal)
    normalized = pl.maximum(normalized, -448.0)
    normalized = pl.minimum(normalized, 448.0)
    payload_fp8 = encode_fp8(normalized, m_tile * (QUANT_K_TILE // 32), 32, integer_bits=False)
    payload = pl.reshape(payload_fp8, [m_tile, QUANT_K_TILE])
    sign_bit = pl.shrs(exponent, 7)
    sign_offset = pl.mul(sign_bit, 256)
    signed = pl.sub(exponent, sign_offset)
    signed_codes = pl.cast(signed, pl.INT8)
    codes = pl.reinterpret_view(signed_codes, pl.UINT8)
    flat = pl.reshape(codes, [1, m_tile * (QUANT_K_TILE // 32)])
    _pack_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
    packed = pl.reshape(flat, [m_tile, QUANT_SCALE_TILE])
    packed_scale = packed
    return payload, packed_scale


def decode_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Decode UE8M0 storage codes into FP32 powers of two."""
    codes = scale.contiguous().view(torch.uint8)
    return torch.exp2(codes.to(torch.float32) - 127.0)


def encode_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Encode positive FP32 powers of two as UE8M0 storage codes."""
    exponent = torch.round(torch.log2(scale.float())).clamp(-127, 128)
    return (exponent + 127).to(torch.uint8)


def pack_fp8_nz(weight: torch.Tensor) -> torch.Tensor:
    """Return the A3 logical ND carrier layout."""
    return weight.contiguous().view(torch.uint8)


def unpack_fp8_nz(weight: torch.Tensor) -> torch.Tensor:
    """Return the A3 logical ND carrier layout."""
    return weight.contiguous().view(torch.float8_e4m3fn)


def pack_mx_b_scale(scale: torch.Tensor) -> torch.Tensor:
    """Return the A3 logical ND carrier layout."""
    return scale.contiguous()


def unpack_mx_b_scale(scale: torch.Tensor) -> torch.Tensor:
    """Return the A3 logical ND carrier layout."""
    return scale.contiguous()


def _nearest_fp4_indices(values: torch.Tensor, *, round_to_even: bool = False) -> torch.Tensor:
    table = FP4_VALUES[:8].to(values.device)
    magnitude = values.abs().unsqueeze(-1)
    if round_to_even:
        order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=values.device)
        selected = (magnitude - table[order]).abs().argmin(dim=-1)
        index = order[selected].to(torch.uint8)
    else:
        boundaries = (table[:-1] + table[1:]) * 0.5
        absolute = magnitude.squeeze(-1)
        if torch.any(absolute > table[-1]):
            index = (magnitude - table).abs().argmin(dim=-1).to(torch.uint8)
        else:
            index = torch.bucketize(absolute, boundaries, out_int32=True).to(torch.uint8)
            index.masked_fill_(~torch.isfinite(absolute), 0)
    return index | (torch.signbit(values).to(torch.uint8) << 3)


def _pack_fp4(indices: torch.Tensor) -> torch.Tensor:
    if indices.shape[-1] % 2:
        raise ValueError("packed FP4 requires an even logical last dimension")
    pairs = indices.unflatten(-1, (-1, 2))
    return pairs[..., 0] | (pairs[..., 1] << 4)


def _unpack_fp4(payload: torch.Tensor) -> torch.Tensor:
    packed = payload.contiguous().view(torch.uint8)
    indices = torch.stack((packed & 0x0F, (packed >> 4) & 0x0F), dim=-1).flatten(-2)
    return FP4_VALUES.to(payload.device)[indices.to(torch.long)]


def quantize_mxfp4_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize output-major ``[..., N, K]`` weights to checkpoint MXFP4 carriers."""
    if weight.shape[-1] % MX_GROUP:
        raise ValueError("MXFP4 weights require K divisible by 32")
    grouped = weight.float().unflatten(-1, (-1, MX_GROUP))
    amax = grouped.abs().amax(dim=-1)
    exponent = torch.ceil(torch.log2((amax / 6.0).clamp_min(2.0**-127)))
    scale = torch.exp2(exponent.clamp(-127, 128))
    normalized = (grouped / scale.unsqueeze(-1)).clamp(-6.0, 6.0)
    payload = _pack_fp4(_nearest_fp4_indices(normalized).flatten(-2))
    return payload, encode_e8m0(scale)


def quantize_mxfp8_cache(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dimension to E4M3 payload and group-32 BF16 scales."""
    if value.shape[-1] % MX_GROUP:
        raise ValueError("MXFP8 cache width must be divisible by 32")
    grouped = value.float().unflatten(-1, (-1, MX_GROUP))
    amax = grouped.abs().amax(dim=-1).clamp_min(1e-4)
    scale = (amax / 448.0).to(torch.bfloat16)
    payload = (grouped / scale.float().unsqueeze(-1)).clamp(-448.0, 448.0)
    return payload.flatten(-2).to(torch.float8_e4m3fn), scale


def dequantize_mxfp8_cache(payload: torch.Tensor, scale_bf16: torch.Tensor) -> torch.Tensor:
    """Decode an MXFP8 cache tensor whose groups lie on the last dimension."""
    scales = scale_bf16.float().repeat_interleave(MX_GROUP, dim=-1)
    return payload.float() * scales


def quantize_mxfp4_cache(
    value: torch.Tensor, group_size: int, scale_format: str, *, round_to_even: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dimension to packed E2M1 with E4M3, E8M0 or BF16 scales."""
    if value.shape[-1] % group_size:
        raise ValueError(f"MXFP4 cache width must be divisible by {group_size}")
    grouped = value.float().unflatten(-1, (-1, group_size))
    amax = grouped.abs().amax(dim=-1)
    floor = 2.0**-126 if round_to_even and scale_format == "e8m0" else 2.0**-9
    raw_scale = (amax / 6.0).clamp_min(floor)
    if scale_format == "e8m0":
        exponent = torch.ceil(torch.log2(raw_scale)).clamp(-127, 128)
        scale_value = torch.exp2(exponent)
        stored_scale = encode_e8m0(scale_value)
    elif scale_format == "e4m3":
        stored_scale = raw_scale.clamp(max=448.0).to(torch.float8_e4m3fn)
        scale_value = stored_scale.float()
    elif scale_format == "bf16":
        stored_scale = raw_scale.to(torch.bfloat16)
        scale_value = stored_scale.float()
    else:
        raise ValueError(f"unsupported MXFP4 cache scale format {scale_format!r}")
    normalized = (grouped / scale_value.unsqueeze(-1)).clamp(-6.0, 6.0)
    payload = _pack_fp4(_nearest_fp4_indices(normalized, round_to_even=round_to_even).flatten(-2))
    return payload, stored_scale


def dequantize_mxfp4_cache(
    payload: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    scale_format: str,
) -> torch.Tensor:
    """Decode a packed E2M1 cache tensor with last-dimension MX groups."""
    if scale_format == "e8m0":
        scale_value = decode_e8m0(scale)
    elif scale_format in ("e4m3", "bf16"):
        scale_value = scale.float()
    else:
        raise ValueError(f"unsupported MXFP4 cache scale format {scale_format!r}")
    scales = scale_value.repeat_interleave(group_size, dim=-1)
    return _unpack_fp4(payload) * scales


def _quantize_mxfp8_activation(
    x: torch.Tensor, *, scale_alg: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    *leading, width = x.shape
    if width % MX_GROUP:
        raise ValueError("MXFP8 activations require the last dimension to be divisible by 32")
    grouped = x.float().reshape(*leading, width // MX_GROUP, MX_GROUP)
    amax = grouped.abs().amax(dim=-1)
    if scale_alg == 0:
        # npu_dynamic_mx_quant defaults to OCP: floor(log2(amax)) - 8.
        codes = _ocp_e8m0_codes_from_amax(amax)
        scale = decode_e8m0(codes)
        reciprocal = torch.where(codes > 0, torch.exp2(127.0 - codes.float()), 0.0)
        normalized = grouped * reciprocal.unsqueeze(-1)
    elif scale_alg == 1:
        scale = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-4) / 448.0)))
        normalized = grouped / scale.unsqueeze(-1)
    else:
        raise ValueError("MXFP8 scale_alg must be 0 (OCP) or 1 (upward)")
    quantized = normalized.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized, scale


def quantize_mxfp8_upward_reference(value):
    """Independent translation of official kernel.py act_quant_kernel, group size 32."""
    groups = value.float().unflatten(-1, (-1, MX_GROUP))
    maximum = groups.abs().amax(-1).clamp_min(1e-4)
    exponent = torch.ceil(torch.log2(maximum * (1.0 / 448.0)))
    scale = torch.pow(2.0, exponent)
    payload = (groups / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).flatten(-2)
    return payload, (exponent + 127).to(torch.uint8)


def quantize_grouped_mxfp8_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize grouped fixtures into NZ bytes with logical [..., G, K, N] shape."""
    payload, scale = _quantize_mxfp8_activation(weight)
    payload = payload.flatten(-2).transpose(-1, -2).contiguous()
    codes = encode_e8m0(scale).transpose(-1, -2).contiguous()
    return pack_fp8_nz(payload), pack_mx_b_scale(codes).view(getattr(torch, "float8_e8m0fnu", torch.uint8))


# E8M0 has no native dtype in some PyTorch versions, so it is always carried as uint8.
FP8_E4M3_MAX = 448.0

def _ocp_e8m0_codes_from_amax(amax: torch.Tensor) -> torch.Tensor:
    """Compute OCP shared-exponent codes for activations and random weight fixtures."""
    _, exponent = torch.frexp(amax.float())
    codes = exponent.to(torch.int32) - 1 - 8 + 127
    codes = codes.clamp(0, 255)
    return torch.where(amax == 0, torch.zeros_like(codes), codes).to(torch.uint8)


def pack_mx_a_scale(scale: torch.Tensor) -> torch.Tensor:
    """Return the A3 logical ND carrier layout."""
    return scale.contiguous()


def unpack_mx_a_scale(scale: torch.Tensor) -> torch.Tensor:
    """Return the A3 logical ND carrier layout."""
    return scale.contiguous()


def quantize_mxfp8_upward_e8m0(
    x: torch.Tensor
):
    """Match the released group-32 MXFP8 quantization and upward scale."""
    value = x.float()
    groups = value.reshape(*value.shape[:-1], -1, MX_GROUP)
    maximum = groups.abs().amax(dim=-1).clamp_min(1e-4)
    # Double precision keeps a one-ULP excess above a power of two from
    # rounding back onto the boundary before ceil.
    scale_exponent = torch.ceil(
        torch.log2((maximum * (1.0 / FP8_E4M3_MAX)).double())
    ).to(torch.int32)
    scale = torch.exp2(scale_exponent).float()
    payload = (
        (groups / scale.unsqueeze(-1))
        .clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .reshape_as(value)
    )
    codes = (scale_exponent + 127).to(torch.uint8)
    return payload, codes


def generate_mxfp8_weight_kn(out: int, inn: int, dequant_std: float, *, chan_cv: float = 0.5, seed: int = 0):
    """Generate MXFP8 weights in the device K-N layout with packed E8M0 scales."""
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(out, inn, generator=g) * torch.exp(chan_cv * torch.randn(out, 1, generator=g))
    groups = raw.reshape(out, inn // MX_GROUP, MX_GROUP)
    codes = _ocp_e8m0_codes_from_amax(groups.abs().amax(-1))
    scale = decode_e8m0(codes)
    quantized = (groups / scale.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    data_kn = quantized.reshape(out, inn).transpose(0, 1).contiguous()
    codes_kn = codes.transpose(0, 1).contiguous()
    decoded = data_kn.float() * decode_e8m0(codes_kn).repeat_interleave(MX_GROUP, dim=0)
    gain = dequant_std / decoded.float().std().clamp_min(1e-8)
    shift = int(torch.round(torch.log2(gain)).item())
    codes_kn = (codes_kn.to(torch.int32) + shift).clamp(0, 255).to(torch.uint8)
    scale_e8m0 = pack_mx_b_scale(codes_kn).contiguous()
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e8m0 is not None:
        scale_e8m0 = scale_e8m0.view(e8m0)
    return data_kn, scale_e8m0


def unpack_mx_scale_codes(scale: torch.Tensor, *, side: str = "a") -> torch.Tensor:
    """Recover logical E8M0 codes from an A-ZZ/B-NN physical scale backing."""
    codes = scale.contiguous().view(torch.uint8)
    if side == "a":
        return unpack_mx_a_scale(codes)
    if side == "b":
        return unpack_mx_b_scale(codes)
    raise ValueError(f"side must be 'a' or 'b', got {side!r}")


@pl.jit
def validate_codecs(
    source: pl.Tensor[[16, 256], pl.FP32],
    fp4_source: pl.Tensor[[16, 256], pl.FP32],
    fp4_factors: pl.Tensor[[16, 1], pl.FP32],
    fp8: pl.Tensor[[16, 256], pl.UINT8],
    fp4: pl.Tensor[[16, 128], pl.UINT8],
    scales: pl.Tensor[[16, 256], pl.UINT8],
    encoded: pl.Out[pl.Tensor[[16, 256], pl.UINT8]],
    encoded_fp4: pl.Out[pl.Tensor[[16, 128], pl.UINT8]],
    decoded_fp8: pl.Out[pl.Tensor[[16, 256], pl.BF16]],
    decoded_fp4: pl.Out[pl.Tensor[[16, 256], pl.BF16]],
    decoded_scales: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
):
    """Validate the finite cache code space and FP8 midpoint rounding."""
    with pl.spmd(1, name_hint="validate_cache_codecs"):
        _block = pl.tile.get_block_idx()
        value = pl.load(source, [0, 0], [16, 256])
        result = encode_fp8(value, 16, 256, integer_bits=False)
        encoded = pl.store(result, [0, 0], encoded)
        fp4_values = pl.load(fp4_source, [0, 0], [16, 256])
        fp4_scales = pl.load(fp4_factors, [0, 0], [16, 1])
        fp4_result = pl.reshape(encode_fp4(fp4_values, fp4_scales, 16, 256), [16, 128])
        encoded_fp4 = pl.store(fp4_result, [0, 0], encoded_fp4)
        raw_fp8 = pl.load(fp8, [0, 0], [16, 256])
        value_fp8 = decode_fp8(raw_fp8, 16, 256)
        decoded_fp8 = pl.store(value_fp8, [0, 0], decoded_fp8)
        raw_fp4 = pl.load(fp4, [0, 0], [16, 128])
        value_fp4 = decode_fp4(raw_fp4, 16, 256)
        decoded_fp4 = pl.store(value_fp4, [0, 0], decoded_fp4)
        raw_scales = pl.load(scales, [0, 0], [16, 256])
        value_scales = decode_scale(raw_scales, 16, 256)
        decoded_scales = pl.store(value_scales, [0, 0], decoded_scales)
    return encoded, encoded_fp4, decoded_fp8, decoded_fp4, decoded_scales


if __name__ == "__main__":
    import argparse
    from golden import TensorSpec, ratio_allclose, run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3"], default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    codes = torch.arange(256, dtype=torch.uint8)
    finite_codes = codes[(codes & 127) != 127]
    positive = codes[:127].view(torch.float8_e4m3fn).float()
    midpoints = (positive[1:] + positive[:-1]) * 0.5
    lower = torch.nextafter(midpoints, torch.full_like(midpoints, -float("inf")))
    upper = torch.nextafter(midpoints, torch.full_like(midpoints, float("inf")))
    boundaries = torch.cat((midpoints, lower, upper))
    values = torch.cat((finite_codes.view(torch.float8_e4m3fn).float(), boundaries, -boundaries))
    values = torch.cat((values, torch.tensor([-1000.0, 1000.0, -0.0, 0.0])))
    source = values.repeat((4096 + values.numel() - 1) // values.numel())[:4096].reshape(16, 256)
    fp8 = finite_codes.repeat((4096 + finite_codes.numel() - 1) // finite_codes.numel())[:4096].reshape(16, 256)
    fp4 = codes.repeat(8).reshape(16, 128)
    scales = codes[:255].repeat(17)[:4096].reshape(16, 256)
    fp4_factors = torch.tensor([0.001953125, 0.11328125, 0.12109375, 1.5]).repeat(4).reshape(16, 1)
    fp4_midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    fp4_midpoints = fp4_midpoints.repeat(37)[:256].unsqueeze(0) * fp4_factors
    fp4_below = torch.nextafter(fp4_midpoints, torch.full_like(fp4_midpoints, -float("inf")))
    fp4_above = torch.nextafter(fp4_midpoints, torch.full_like(fp4_midpoints, float("inf")))
    fp4_source = torch.stack((fp4_midpoints, fp4_below, fp4_above, -fp4_midpoints), dim=-1)
    fp4_source = fp4_source.flatten(1)[:, :256].contiguous()
    fp4_source[:, :2] = torch.tensor([0.0, -0.0])

    def golden(tensors):
        tensors["encoded"].copy_(tensors["source"].clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8))
        normalized = tensors["fp4_source"].double() / tensors["fp4_factors"].double()
        tensors["encoded_fp4"].copy_(_pack_fp4(_nearest_fp4_indices(normalized, round_to_even=True)))
        tensors["decoded_fp8"].copy_(tensors["fp8"].view(torch.float8_e4m3fn).bfloat16())
        tensors["decoded_fp4"].copy_(_unpack_fp4(tensors["fp4"]).bfloat16())
        tensors["decoded_scales"].copy_(torch.exp2(tensors["scales"].float() - 127.0))

    inputs = [("source", source), ("fp4_source", fp4_source), ("fp4_factors", fp4_factors),
              ("fp8", fp8), ("fp4", fp4), ("scales", scales)]
    specs = [TensorSpec(name, list(value.shape), value.dtype, init_value=value) for name, value in inputs]
    output_types = {"encoded": torch.uint8, "encoded_fp4": torch.uint8, "decoded_fp8": torch.bfloat16,
                    "decoded_fp4": torch.bfloat16, "decoded_scales": torch.float32}
    specs.extend(TensorSpec(name, [16, 128 if name == "encoded_fp4" else 256], dtype)
                 for name, dtype in output_types.items())
    result = run(
        fn=validate_codecs,
        specs=specs,
        golden_fn=golden,
        compare_fn={name: ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0) for name in output_types},
        config={"backend_type": args.platform, "execution_mode": args.sim, "device_id": args.device},
        compile_only=args.compile_only,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)


def mapped_fp4_cache_compare(
    mapping_name: str,
    *,
    mapping_shape: tuple[int, ...],
    block_size: int,
    pool_name: str,
    leading_rank_axis: bool = False,
    max_error_ratio: float = 0.005,
):
    """Bound sparse E2M1 changes after floating-point model computation.

    The existing mapped comparator keeps every unwritten byte exact and bounds
    differing written bytes by the declared ratio. A second, unconditional gate
    rejects sign changes (including signed zero) or a jump over more than one
    magnitude level in either nibble. This is an end-to-end numerical policy,
    not an exact quantizer test or a proof of a floating-point error bound.
    """
    from golden import mapped_pool_ratio_allclose

    mapped = mapped_pool_ratio_allclose(
        mapping_name, mapping_shape=mapping_shape, block_size=block_size,
        leading_rank_axis=leading_rank_axis, pool_name=pool_name,
        atol=0.0, rtol=0.0, max_error_ratio=max_error_ratio,
    )

    def compare(actual, expected, **kwargs):
        if actual.dtype != torch.uint8 or expected.dtype != torch.uint8:
            return False, f"{pool_name}: packed E2M1 payload must have UINT8 storage"
        ok, detail = mapped(actual, expected, **kwargs)
        if not ok:
            return False, detail
        a = actual.detach().cpu().to(torch.int16)
        e = expected.detach().cpu().to(torch.int16)
        sign_changed = torch.bitwise_and(torch.bitwise_xor(a, e), 0x88) != 0
        if bool(sign_changed.any()):
            return False, f"{pool_name}: E2M1 sign changed in {int(sign_changed.sum())} byte(s)"
        low_distance = ((a & 7) - (e & 7)).abs()
        high_distance = (((a >> 4) & 7) - ((e >> 4) & 7)).abs()
        invalid = (low_distance > 1) | (high_distance > 1)
        if bool(invalid.any()):
            return False, f"{pool_name}: E2M1 magnitude skips a level in {int(invalid.sum())} byte(s)"
        return True, ""

    return compare
