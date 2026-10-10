# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Checkpoint layouts and the released group-32 MX activation quantization."""

import math

import torch

from models.deepseek_v4_1_flash.config import MX_GROUP



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


def build_mxfp4_pair_lut() -> torch.Tensor:
    """Build the two-lane byte LUT consumed by the AIV FP4 decoder."""
    packed = torch.arange(256, dtype=torch.int64)
    codes = FP4_E4M3_CODES.to(torch.int64)
    low = codes[packed & 0x0F]
    high = codes[packed >> 4]
    pairs = low | (high << 8)
    pairs = torch.where(pairs < 0x8000, pairs, pairs - 0x10000)
    return pairs.to(torch.int16).reshape(1, 256).repeat(2, 1).contiguous()


def _pack_mxfp4_nibbles_kn_tiles(
    nibbles_kn: torch.Tensor, k_tile: int, n_tile: int
) -> torch.Tensor:
    """Pack a logical [K, N] nibble matrix into Cube lane rows."""
    k, n = nibbles_kn.shape
    if k % k_tile or n % n_tile or k_tile % 2 or n_tile % 2:
        raise ValueError("MXFP4 tile dimensions must divide K/N and be even")
    k_blocks = k // k_tile
    n_blocks = n // n_tile
    blocked = nibbles_kn.reshape(k_blocks, k_tile, n_blocks, n_tile)
    blocked = blocked.permute(2, 0, 1, 3)
    lanes = blocked.reshape(n_blocks, k_blocks, 2, k_tile // 2, n_tile)
    low = lanes[..., 0::2] & 0x0F
    high = lanes[..., 1::2] & 0x0F
    packed = low | (high << 4)
    lane_bytes = k_tile * n_tile // 4
    return packed.contiguous().reshape(n_blocks * k_blocks * 2, lane_bytes)


def pack_mxfp4_weight_tiles(
    packed_weight_fp4: torch.Tensor, k_tile: int, n_tile: int
) -> torch.Tensor:
    """Reorder checkpoint [N, K/2] bytes into Cube FP4 tiles."""
    payload = packed_weight_fp4.contiguous().view(torch.uint8)
    *lead, n, half_k = payload.shape
    k = 2 * half_k
    matrices = payload.reshape(-1, n, half_k)
    packed_rows = k * n // n_tile
    packed_cols = n_tile // 2
    result = torch.empty(
        [matrices.shape[0], packed_rows, packed_cols], dtype=torch.uint8
    )
    for index, matrix in enumerate(matrices):
        nibbles = torch.empty([n, k], dtype=torch.uint8)
        nibbles[:, 0::2] = matrix & 0x0F
        nibbles[:, 1::2] = matrix >> 4
        result[index] = _pack_mxfp4_nibbles_kn_tiles(
            nibbles.t(), k_tile, n_tile
        ).reshape(packed_rows, packed_cols)
    return result.reshape(*lead, packed_rows, packed_cols)


def unpack_mxfp4_weight_tiles(
    packed: torch.Tensor, k: int, n: int, k_tile: int, n_tile: int
) -> torch.Tensor:
    """Decode Cube FP4 tiles to an FP8 [K, N] reference matrix."""
    *lead, tile_rows, lane_bytes = packed.shape
    k_blocks = k // k_tile
    n_blocks = n // n_tile
    if tile_rows != n_blocks * k_blocks * 2 or lane_bytes != k_tile * n_tile // 4:
        raise ValueError("MXFP4 tile payload does not match requested matrix")
    payloads = packed.contiguous().view(torch.uint8).reshape(-1, tile_rows, lane_bytes)
    output = torch.empty(payloads.shape[0], k, n, dtype=torch.float8_e4m3fn)
    for batch in range(payloads.shape[0]):
        lane_shape = (n_blocks, k_blocks, 2, k_tile // 2, n_tile // 2)
        lane_bytes_view = payloads[batch].reshape(lane_shape)
        low = lane_bytes_view & 0x0F
        high = lane_bytes_view >> 4
        lanes = torch.stack((low, high), dim=-1)
        blocked = lanes.reshape(n_blocks, k_blocks, k_tile, n_tile)
        nibbles_kn = blocked.permute(1, 2, 0, 3).reshape(k, n)
        codes = FP4_E4M3_CODES.to(torch.uint8)[nibbles_kn.to(torch.long)]
        output[batch] = codes.view(torch.float8_e4m3fn)
    return output.reshape(*lead, k, n)


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


def dequantize_mxfp4(packed_weight: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Decode checkpoint MXFP4 directly into an FP32 output-major matrix."""
    packed = packed_weight.contiguous().view(torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    indices = torch.stack((low, high), dim=-1).flatten(-2)
    values = FP4_VALUES.to(indices.device)[indices.to(torch.long)]
    scales = decode_e8m0(scale_e8m0).repeat_interleave(MX_GROUP, dim=-1)
    return values * scales


def _nearest_fp4_indices(values: torch.Tensor) -> torch.Tensor:
    table = FP4_VALUES[:8].to(values.device)
    magnitude = values.abs().unsqueeze(-1)
    index = (magnitude - table).abs().argmin(dim=-1).to(torch.uint8)
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


def prepare_routed_weight_for_device(
    packed_weight_fp4: torch.Tensor,
    scale_e8m0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert checkpoint routed FP4 weights into the A5 NPU MXFP8 RHS ABI.

    The input comes from a packed MXFP4 checkpoint with logical shape
    ``[..., out, in]``, while ``pl.matmul_mx`` requires an ``[..., in, out]`` RHS.

    The PyPTO high-level API does not support ``FP8 activation x FP4 RHS``, so the
    FP4 decode, logical transpose, MX_B_NN scale reordering and FP8 quantisation
    all happen at checkpoint/weight-load time. This only prepares persistent device
    weights on the host; it is not a production kernel path that expands the full
    weight to FP32.

    Returns:
        ``device_weight``: ``[..., in, out]`` FP8E4M3FN weights;
        ``device_scale``: ``[..., in/32, out]`` E8M0 scales in MX_B_NN physical order.
    """
    logical_out_in = dequantize_mxfp4(packed_weight_fp4, scale_e8m0)
    if logical_out_in.shape[-1] % MX_GROUP:
        raise ValueError("routed FP4 input dimension must be divisible by MX group size")

    logical_in_out = logical_out_in.transpose(-2, -1).contiguous()
    k_dim, n_dim = logical_in_out.shape[-2:]
    if k_dim % MX_GROUP or n_dim % 16:
        raise ValueError("MX_B_NN requires K divisible by 32 and N divisible by 16")

    grouped = logical_in_out.float().reshape(*logical_in_out.shape[:-2], k_dim // MX_GROUP, MX_GROUP, n_dim)
    amax = grouped.abs().amax(dim=-2)
    exponent = torch.ceil(torch.log2((amax / 448.0).clamp_min(2.0 ** -127))).clamp(-127, 128)
    scale_value = torch.exp2(exponent)
    quantized = (grouped / scale_value.unsqueeze(-2)).clamp(-448.0, 448.0)
    quantized = quantized.to(torch.float8_e4m3fn).reshape(*logical_in_out.shape[:-2], k_dim, n_dim)
    scale_codes = encode_e8m0(scale_value)
    packed_scale = pack_mx_b_scale(scale_codes)
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e8m0 is not None:
        packed_scale = packed_scale.contiguous().view(e8m0)
    return quantized, packed_scale


def quantize_mxfp8_cache(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dimension to E4M3 payload and group-32 E8M0 scales."""
    if value.shape[-1] % MX_GROUP:
        raise ValueError("MXFP8 cache width must be divisible by 32")
    grouped = value.float().unflatten(-1, (-1, MX_GROUP))
    amax = grouped.abs().amax(dim=-1)
    exponent = torch.ceil(torch.log2((amax / 448.0).clamp_min(2.0**-127)))
    scale = torch.exp2(exponent.clamp(-127, 128))
    payload = (grouped / scale.unsqueeze(-1)).clamp(-448.0, 448.0)
    return payload.flatten(-2).to(torch.float8_e4m3fn), encode_e8m0(scale)


def dequantize_mxfp8_cache(payload: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Decode an MXFP8 cache tensor whose groups lie on the last dimension."""
    scales = decode_e8m0(scale_e8m0).repeat_interleave(MX_GROUP, dim=-1)
    return payload.float() * scales


def quantize_mxfp4_cache(
    value: torch.Tensor, group_size: int, scale_format: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dimension to packed E2M1 with E4M3 or E8M0 scales."""
    if value.shape[-1] % group_size:
        raise ValueError(f"MXFP4 cache width must be divisible by {group_size}")
    grouped = value.float().unflatten(-1, (-1, group_size))
    amax = grouped.abs().amax(dim=-1)
    raw_scale = (amax / 6.0).clamp_min(2.0**-9)
    if scale_format == "e8m0":
        exponent = torch.ceil(torch.log2(raw_scale)).clamp(-127, 128)
        scale_value = torch.exp2(exponent)
        stored_scale = encode_e8m0(scale_value)
    elif scale_format == "e4m3":
        stored_scale = raw_scale.clamp(max=448.0).to(torch.float8_e4m3fn)
        scale_value = stored_scale.float()
    else:
        raise ValueError(f"unsupported MXFP4 cache scale format {scale_format!r}")
    normalized = (grouped / scale_value.unsqueeze(-1)).clamp(-6.0, 6.0)
    payload = _pack_fp4(_nearest_fp4_indices(normalized).flatten(-2))
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
    elif scale_format == "e4m3":
        scale_value = scale.float()
    else:
        raise ValueError(f"unsupported MXFP4 cache scale format {scale_format!r}")
    scales = scale_value.repeat_interleave(group_size, dim=-1)
    return _unpack_fp4(payload) * scales


def dequantize_mxfp8(weight: torch.Tensor, logical_scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Decode an input-major MXFP8 matrix using logical ``[K/32, N]`` scales."""
    scales = decode_e8m0(logical_scale_e8m0).repeat_interleave(MX_GROUP, dim=-2)
    return weight.float() * scales


def _quantize_mxfp8_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    *leading, width = x.shape
    if width % MX_GROUP:
        raise ValueError("MXFP8 activations require the last dimension to be divisible by 32")
    grouped = x.float().reshape(*leading, width // MX_GROUP, MX_GROUP)
    amax = grouped.abs().amax(dim=-1).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    quantized = (grouped / scale.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized, scale


def quantize_mxfp8_activation(x: torch.Tensor) -> torch.Tensor:
    """Round one activation scale per row and group of 32, then dequantize to FP32."""
    quantized, scale = _quantize_mxfp8_activation(x)
    return (quantized.float() * scale.unsqueeze(-1)).reshape_as(x).float()


def mxfp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    packed_weight_scale: torch.Tensor | None,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Evaluate a linear projection with an optional explicit output dtype."""
    if packed_weight_scale is None:
        if output_dtype is not None:
            return torch.matmul(x.float(), weight.float()).to(output_dtype)
        return torch.matmul(x, weight)
    activation, activation_scale = _quantize_mxfp8_activation(x)
    logical_scale = unpack_mx_b_scale(packed_weight_scale)
    weight_scale = decode_e8m0(logical_scale)
    weight_groups = weight.float().unflatten(0, (-1, MX_GROUP))
    partials = torch.einsum("...gk,gkn->...gn", activation.float(), weight_groups)
    partials = partials * activation_scale.unsqueeze(-1) * weight_scale
    return partials.sum(dim=-2).to(output_dtype or x.dtype)


# E8M0 has no native dtype in some PyTorch versions, so it is always carried as uint8.
FP8_E4M3_MAX = 448.0

def _e8m0_codes_from_amax(amax: torch.Tensor, fp_max: float = FP8_E4M3_MAX) -> torch.Tensor:
    """OCP shared-exponent scales used by random weight fixtures.

    Released activation quantization uses the upward E8M0 scale in
    ``host_quant_mxfp8_v41`` instead.
    """
    format_emax = int(math.floor(math.log2(fp_max)))
    _, exponent = torch.frexp(amax.float())
    codes = exponent.to(torch.int32) - 1 - format_emax + 127
    codes = codes.clamp(0, 255)
    return torch.where(amax == 0, torch.zeros_like(codes), codes).to(torch.uint8)


def _e8m0_to_fp32(codes: torch.Tensor) -> torch.Tensor:
    return torch.exp2(codes.to(torch.float32) - 127.0)


def pack_mx_a_scale(scale: torch.Tensor) -> torch.Tensor:
    m, groups = scale.shape
    if m % 16 or groups % 2:
        raise ValueError("MX_A_ZZ requires rows%16==0 and groups%2==0")
    return scale.reshape(m // 16, 16, groups // 2, 2).permute(0, 2, 1, 3).contiguous().reshape(m, groups)


def unpack_mx_a_scale(scale: torch.Tensor) -> torch.Tensor:
    m, groups = scale.shape
    return scale.reshape(m // 16, groups // 2, 16, 2).permute(0, 2, 1, 3).contiguous().reshape(m, groups)


def host_quant_mxfp8_v41(x: torch.Tensor, *, return_e8m0: bool = False):
    """Quantize with the released group-32 upward E8M0 activation scale."""
    return host_quant_swiglu_mxfp8_v41(x, return_e8m0=return_e8m0)


def host_quant_swiglu_mxfp8_v41(
    x: torch.Tensor, *, return_e8m0: bool = False
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
    if not return_e8m0:
        return payload, scale
    codes = (scale_exponent + 127).to(torch.uint8)
    return payload, codes


def gen_mxfp8_weight_kn_v41(out: int, inn: int, dequant_std: float, *, chan_cv: float = 0.5, seed: int = 0):
    """Generate MXFP8 weights in the device K-N layout with packed E8M0 scales."""
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(out, inn, generator=g) * torch.exp(chan_cv * torch.randn(out, 1, generator=g))
    groups = raw.reshape(out, inn // MX_GROUP, MX_GROUP)
    codes = _e8m0_codes_from_amax(groups.abs().amax(-1))
    scale = _e8m0_to_fp32(codes)
    quantized = (groups / scale.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    data_kn = quantized.reshape(out, inn).transpose(0, 1).contiguous()
    codes_kn = codes.transpose(0, 1).contiguous()
    decoded = data_kn.float() * _e8m0_to_fp32(codes_kn).repeat_interleave(MX_GROUP, dim=0)
    gain = dequant_std / decoded.float().std().clamp_min(1e-8)
    shift = int(torch.round(torch.log2(gain)).item())
    codes_kn = (codes_kn.to(torch.int32) + shift).clamp(0, 255).to(torch.uint8)
    scale_e8m0 = pack_mx_b_scale(codes_kn).contiguous()
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e8m0 is not None:
        scale_e8m0 = scale_e8m0.view(e8m0)
    return data_kn, scale_e8m0


def decode_e8m0_codes_v41(scale: torch.Tensor, *, side: str = "a") -> torch.Tensor:
    """Recover logical E8M0 codes from an A-ZZ/B-NN physical scale backing."""
    codes = scale.contiguous().view(torch.uint8)
    if side == "a":
        return unpack_mx_a_scale(codes)
    if side == "b":
        return unpack_mx_b_scale(codes)
    raise ValueError(f"side must be 'a' or 'b', got {side!r}")


def matmul_mx_golden_v41(a, a_scale, b, b_scale):
    """Evaluate the FP32 MX matmul golden using logical E8M0 scales."""
    a_codes = a_scale.contiguous().view(torch.uint8)
    b_codes = b_scale.contiguous().view(torch.uint8)
    a_f = a.float() * _e8m0_to_fp32(a_codes).repeat_interleave(MX_GROUP, -1)
    b_f = b.float() * _e8m0_to_fp32(b_codes).repeat_interleave(MX_GROUP, -2)
    return torch.matmul(a_f.to(torch.float32), b_f.to(torch.float32))
