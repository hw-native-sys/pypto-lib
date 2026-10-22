# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 device and host quantization, checkpoint layouts, and weight fixtures."""

import torch
import pypto.language as pl

from config import FLASH as M


# model config
MX_GROUP = M.weight_block_size[0]

# tiling
QUANT_K_TILE = 256
QUANT_SCALE_TILE = QUANT_K_TILE // 32


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
    payload_fp8 = pl.cast(normalized, pl.FP8E4M3FN, mode="rint")
    payload = pl.reshape(payload_fp8, [m_tile, QUANT_K_TILE])
    sign_bit = pl.shrs(exponent, 7)
    sign_offset = pl.mul(sign_bit, 256)
    signed = pl.sub(exponent, sign_offset)
    signed_codes = pl.cast(signed, pl.INT8)
    codes = pl.reinterpret_view(signed_codes, pl.UINT8)
    flat = pl.reshape(codes, [1, m_tile * (QUANT_K_TILE // 32)])
    pack_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
    packed = pl.tmov_x2zz(flat, pack_tmp, group_axis=1, dst_rows=m_tile, dst_cols=QUANT_SCALE_TILE)
    packed_scale = pl.reinterpret_view(packed, pl.FP8E8M0)
    return payload, packed_scale


def decode_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Decode UE8M0 storage codes into FP32 powers of two."""
    codes = scale.contiguous().view(torch.uint8)
    return torch.exp2(codes.to(torch.float32) - 127.0)


def encode_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Encode positive FP32 powers of two as UE8M0 storage codes."""
    exponent = torch.round(torch.log2(scale.float())).clamp(-127, 128)
    return (exponent + 127).to(torch.uint8)


def pack_mx_b_scale(scale: torch.Tensor) -> torch.Tensor:
    """Pack logical ``[..., K/32, N]`` scales into the MX_B_NN physical order."""
    *leading, k_groups, output_dim = scale.shape
    if k_groups % 2 or output_dim % 16:
        raise ValueError("MX_B_NN requires an even K-group count and an output dimension divisible by 16")
    packed = scale.reshape(*leading, k_groups // 2, 2, output_dim // 16, 16)
    leading_axes = list(range(len(leading)))
    packed = packed.permute(*leading_axes, len(leading) + 2, len(leading), len(leading) + 3, len(leading) + 1)
    return packed.contiguous().reshape(*leading, k_groups, output_dim)


def unpack_mx_b_scale(scale: torch.Tensor) -> torch.Tensor:
    """Unpack the Cube MX_B_NN scale order into logical ``[..., K/32, N]`` rows."""
    *leading, k_groups, output_dim = scale.shape
    if k_groups % 2 or output_dim % 16:
        raise ValueError("MX_B_NN requires an even K-group count and an output dimension divisible by 16")
    logical = scale.reshape(*leading, output_dim // 16, k_groups // 2, 16, 2)
    leading_axes = list(range(len(leading)))
    logical = logical.permute(
        *leading_axes, len(leading) + 1, len(leading) + 3, len(leading), len(leading) + 2
    )
    return logical.contiguous().reshape(*leading, k_groups, output_dim)


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
    """Quantize output-major grouped fixtures into [G, K, N] MXFP8 storage."""
    payload, scale = _quantize_mxfp8_activation(weight)
    payload = payload.flatten(-2).transpose(-1, -2).contiguous()
    codes = encode_e8m0(scale).transpose(-1, -2).contiguous()
    return payload, pack_mx_b_scale(codes).view(torch.float8_e8m0fnu)


# E8M0 has no native dtype in some PyTorch versions, so it is always carried as uint8.
FP8_E4M3_MAX = 448.0

def _ocp_e8m0_codes_from_amax(amax: torch.Tensor) -> torch.Tensor:
    """Compute OCP shared-exponent codes for activations and random weight fixtures."""
    _, exponent = torch.frexp(amax.float())
    codes = exponent.to(torch.int32) - 1 - 8 + 127
    codes = codes.clamp(0, 255)
    return torch.where(amax == 0, torch.zeros_like(codes), codes).to(torch.uint8)


def pack_mx_a_scale(scale: torch.Tensor) -> torch.Tensor:
    m, groups = scale.shape
    if m % 16 or groups % 2:
        raise ValueError("MX_A_ZZ requires rows%16==0 and groups%2==0")
    return scale.reshape(m // 16, 16, groups // 2, 2).permute(0, 2, 1, 3).contiguous().reshape(m, groups)


def unpack_mx_a_scale(scale: torch.Tensor) -> torch.Tensor:
    m, groups = scale.shape
    return scale.reshape(m // 16, groups // 2, 16, 2).permute(0, 2, 1, 3).contiguous().reshape(m, groups)


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
