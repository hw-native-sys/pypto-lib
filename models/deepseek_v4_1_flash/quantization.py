# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Torch references for the checkpoint and persistent-cache MX formats."""

import math

import torch

from models.deepseek_v4_1_flash.config import MX_GROUP

import pypto.language as pl


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
    """Ascend OCP shared-exponent E8M0 codes for each group maximum.

    ``pl.quant_mx`` implements the OCP MX shared exponent
    ``X = 2 ** (floor(log2(amax)) - emax)``, i.e. it rounds the amax exponent down
    (``emax = floor(log2(448)) = 8``). Whenever the log2 fraction of amax lands in
    ``[log2(448/256), 1)`` - about 19.3% of groups - the payload ``amax / X`` falls
    in ``(448, 512)`` and saturates at the E4M3FN maximum of 448.

    Rounding up with ``ceil(log2(amax / 448))``, as this helper did before, made the
    golden scale twice the device value and the payload half the size for those same
    ~19.3% of groups, which showed up as a systematic 1%-5% relative error on the w2
    output (measured match rate 19.35% vs the theoretical 19.26%). The rounding
    direction must match the device.
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
    value = x.float()
    groups = value.reshape(*value.shape[:-1], -1, MX_GROUP)
    amax = groups.abs().amax(dim=-1)
    codes = _e8m0_codes_from_amax(amax)
    scale = _e8m0_to_fp32(codes)
    payload = (groups / scale.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).reshape_as(value)
    if not return_e8m0:
        return payload, scale
    return payload, codes.to(torch.uint8)


def host_quant_swiglu_mxfp8_v41(
    x: torch.Tensor, *, return_e8m0: bool = False
):
    """Match CANN SwiGLU's group-32 MXFP8 quantization and upward scale."""
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


# Resident CED prefill precision variants.
_PREFILL_OPS_ROWS = pl.dynamic("V41_FP32_ROWS")


_PREFILL_OPS_INPUT_WIDTH = pl.dynamic("V41_FP32_INPUT_WIDTH")


@pl.jit.inline(auto_scope=False)
def prefill_round_bf16_inline(
    x: pl.Tensor[[_PREFILL_OPS_ROWS, _PREFILL_OPS_INPUT_WIDTH], pl.FP32],
    output: pl.Tensor[[_PREFILL_OPS_ROWS, _PREFILL_OPS_INPUT_WIDTH], pl.FP32],
):
    """Round FP32 transport values to BF16 with nearest-even ties."""
    rows = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(x, 1)
    column_tiles = (width + 1023) // 1024
    for worker in pl.spmd(32, name_hint="prefill_round_bf16"):
        for block in pl.range(worker, rows * column_tiles, 32):
            row = block // column_tiles
            column = block % column_tiles * 1024
            active = pl.min(1024, width - column)
            values = pl.slice(x, [1, 1024], [row, column], valid_shape=[1, active])
            rounded = pl.cast(pl.cast(values, pl.BF16, mode="rint"), pl.FP32)
            output[row : row + 1, column : column + 1024] = rounded
    return output


def _prefill_make_quantization(group, fp8, e8m0, name):
    @pl.jit.inline(auto_scope=False)
    def quantize(
        x: pl.Tensor[[_PREFILL_OPS_ROWS, _PREFILL_OPS_INPUT_WIDTH], pl.FP32],
        output: pl.Tensor[[_PREFILL_OPS_ROWS, _PREFILL_OPS_INPUT_WIDTH], pl.FP32],
    ):
        rows = pl.tensor.dim(x, 0)
        width = pl.tensor.dim(x, 1)
        column_tiles = (width + 1023) // 1024
        for worker in pl.spmd(32, name_hint=name):
            for block in pl.range(worker, rows * column_tiles, 32):
                row = block // column_tiles
                column = block % column_tiles * 1024
                active = pl.min(1024, width - column)
                source = pl.slice(x, [1, 1024], [row, column], valid_shape=[1, active])
                source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 1, 1024)
                narrowed = pl.cast(pl.cast(source, pl.BF16, mode="rint"), pl.FP32)
                values = pl.reshape(narrowed, [1024 // group, group])
                amax = pl.reshape(pl.row_max(pl.abs(values)), [1, 1024 // group])
                if fp8:
                    scaled = pl.mul(pl.maximum(amax, 1e-4), 1.0 / 448.0)
                elif e8m0:
                    scaled = pl.mul(pl.maximum(amax, 7.052966104933725e-38), 1.0 / 6.0)
                else:
                    bounded = pl.maximum(amax, 0.01171875)
                    six = pl.full([1, 1024 // group], dtype=pl.FP32, value=6.0)
                    scaled = pl.div(bounded, six, high_precision=True)
                if e8m0:
                    bits = pl.reinterpret_view(scaled, pl.INT32)
                    exponent = pl.shrs(pl.add(bits, 8388607), 23)
                    scale = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
                else:
                    scale = pl.cast(pl.cast(scaled, pl.FP8E4M3FN, mode="rint"), pl.FP32)
                group_scale = pl.reshape(scale, [1024 // group, 1])
                ones = pl.full([1024 // group, group], dtype=pl.FP32, value=1.0)
                divisor = pl.row_expand_mul(ones, group_scale)
                normalized = pl.div(values, divisor, high_precision=True)
                if fp8:
                    normalized = pl.minimum(pl.maximum(normalized, -448.0), 448.0)
                    payload = pl.cast(pl.cast(normalized, pl.FP8E4M3FN, mode="rint"), pl.FP32)
                else:
                    magnitude = pl.minimum(pl.abs(normalized), 6.0)
                    # Even E2M1 codes win exact midpoint ties.
                    step0 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 0.25), pl.INT32), 1), 31)
                    step1 = pl.shrs(pl.reinterpret_view(pl.sub(magnitude, 0.75), pl.INT32), 31)
                    step2 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 1.25), pl.INT32), 1), 31)
                    step3 = pl.shrs(pl.reinterpret_view(pl.sub(magnitude, 1.75), pl.INT32), 31)
                    step4 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 2.5), pl.INT32), 1), 31)
                    step5 = pl.shrs(pl.reinterpret_view(pl.sub(magnitude, 3.5), pl.INT32), 31)
                    step6 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 5.0), pl.INT32), 1), 31)
                    first = pl.add(pl.add(step0, step1), pl.add(step2, step3))
                    second = pl.add(pl.add(step4, step5), pl.add(step6, 7))
                    code = pl.add(first, second)
                    base = pl.mul(pl.cast(code, pl.FP32), 0.5)
                    extra4 = pl.cast(pl.minimum(pl.maximum(pl.sub(code, 4), 0), 1), pl.FP32)
                    extra5 = pl.cast(pl.minimum(pl.maximum(pl.sub(code, 5), 0), 1), pl.FP32)
                    extra6 = pl.cast(pl.minimum(pl.maximum(pl.sub(code, 6), 0), 1), pl.FP32)
                    magnitude_payload = pl.add(
                        pl.add(base, pl.mul(pl.add(extra4, extra5), 0.5)), pl.mul(extra6, 1.5)
                    )
                    sign = pl.ands(pl.reinterpret_view(values, pl.INT32), -2147483648)
                    payload = pl.reinterpret_view(
                        pl.or_(pl.reinterpret_view(magnitude_payload, pl.INT32), sign), pl.FP32
                    )
                decoded = pl.row_expand_mul(payload, group_scale)
                rounded = pl.cast(pl.cast(decoded, pl.BF16, mode="rint"), pl.FP32)
                result = pl.set_validshape(pl.reshape(rounded, [1, 1024]), 1, active)
                output[row : row + 1, column : column + 1024] = result
        return output

    return quantize


prefill_quantize_fp8_inline = _prefill_make_quantization(32, True, True, "prefill_quantize_fp8")


prefill_quantize_index_fp4_inline = _prefill_make_quantization(32, False, True, "prefill_quantize_index_fp4")


prefill_quantize_kv_fp4_inline = _prefill_make_quantization(16, False, False, "prefill_quantize_kv_fp4")


_PREFILL_WEIGHT_OPS_BLOCKS4 = pl.dynamic("V41_WEIGHT_BLOCKS4")


_PREFILL_WEIGHT_OPS_INPUT_WIDTH = pl.dynamic("V41_WEIGHT_INPUT_WIDTH")


_PREFILL_WEIGHT_OPS_OUTPUT_WIDTH = pl.dynamic("V41_WEIGHT_OUTPUT_WIDTH")


@pl.jit.inline(auto_scope=False)
def prefill_decode_fp4_inline(
    bank: pl.Tensor[[_PREFILL_WEIGHT_OPS_BLOCKS4, 512], pl.INT8],
    scales: pl.Tensor[[_PREFILL_WEIGHT_OPS_BLOCKS4, 32], pl.UINT8],
    block_offset: pl.Scalar[pl.INDEX],
    output: pl.Tensor[[_PREFILL_WEIGHT_OPS_INPUT_WIDTH, _PREFILL_WEIGHT_OPS_OUTPUT_WIDTH], pl.FP32],
):
    """Decode output-major E2M1 blocks to input-major FP32 without requantization.

    Each bank row packs a 32x32 weight block, with even input columns in low
    nibbles. Its 32 scale bytes are one E8M0 factor per output row. Matrix blocks
    are ordered by output block, then input block; both matrix widths are
    multiples of 32. Callers validate finite checkpoint scales before upload.
    """
    bank.bind_dynamic(0, _PREFILL_WEIGHT_OPS_BLOCKS4)
    output.bind_dynamic(0, _PREFILL_WEIGHT_OPS_INPUT_WIDTH)
    output.bind_dynamic(1, _PREFILL_WEIGHT_OPS_OUTPUT_WIDTH)
    width = pl.tensor.dim(output, 0)
    columns = pl.tensor.dim(output, 1)
    input_blocks = width // 32
    matrix_blocks = (columns // 32) * input_blocks
    # Resolve the large bank address in orchestration's 64-bit Tensor view.
    # InCore loads use only matrix-local offsets, bounded by one projection.
    matrix = pl.slice(bank, [matrix_blocks, 512], [block_offset, 0])
    matrix_scales = pl.slice(scales, [matrix_blocks, 32], [block_offset, 0])
    for worker in pl.spmd(32, name_hint="prefill_decode_fp4"):
        for block in pl.range(worker, matrix_blocks, 32):
            row = block
            packed = pl.load(matrix, [row, 0], [1, 512])
            packed_i32 = pl.reshape(pl.cast(packed, pl.INT32), [32, 16])
            low = pl.ands(packed_i32, 15)
            high = pl.ands(pl.shrs(packed_i32, 4), 15)
            low_m = pl.cast(pl.ands(low, 7), pl.FP32)
            high_m = pl.cast(pl.ands(high, 7), pl.FP32)
            low_value = pl.add(
                pl.add(pl.mul(low_m, 0.5), pl.mul(pl.maximum(pl.sub(low_m, 4.0), 0.0), 0.5)),
                pl.maximum(pl.sub(low_m, 6.0), 0.0),
            )
            high_value = pl.add(
                pl.add(pl.mul(high_m, 0.5), pl.mul(pl.maximum(pl.sub(high_m, 4.0), 0.0), 0.5)),
                pl.maximum(pl.sub(high_m, 6.0), 0.0),
            )
            low_signed = pl.reinterpret_view(
                pl.or_(pl.reinterpret_view(low_value, pl.INT32), pl.shls(pl.ands(low, 8), 28)), pl.FP32
            )
            high_signed = pl.reinterpret_view(
                pl.or_(pl.reinterpret_view(high_value, pl.INT32), pl.shls(pl.ands(high, 8), 28)), pl.FP32
            )
            # Mask scatter zero-fills unselected lanes on A5. Merge separate
            # halves by bits so the second scatter cannot erase the first.
            low_lanes = pl.tile.full([32, 32], dtype=pl.FP32, value=0.0)
            high_lanes = pl.tile.full([32, 32], dtype=pl.FP32, value=0.0)
            low_lanes = pl.tile.scatter_mask(low_lanes, low_signed, pl.tile.MaskPattern.P0101)
            high_lanes = pl.tile.scatter_mask(high_lanes, high_signed, pl.tile.MaskPattern.P1010)
            values = pl.reinterpret_view(
                pl.or_(pl.reinterpret_view(low_lanes, pl.INT32), pl.reinterpret_view(high_lanes, pl.INT32)),
                pl.FP32,
            )
            scale_bytes = pl.reinterpret_view(pl.load(matrix_scales, [row, 0], [1, 32]), pl.INT8)
            code = pl.ands(pl.cast(scale_bytes, pl.INT32), 255)
            # E8M0 code zero is 2^-127, a valid FP32 subnormal; other finite
            # codes directly supply the FP32 biased exponent bits.
            zero_code = pl.shrs(pl.sub(code, 1), 31)
            bits = pl.or_(pl.shls(code, 23), pl.ands(zero_code, 4194304))
            factor = pl.reinterpret_view(bits, pl.FP32)
            decoded = pl.row_expand_mul(values, pl.reshape(factor, [32, 1]))
            transposed = pl.transpose(decoded, 0, 1)
            pl.store(transposed, [(block % input_blocks) * 32, (block // input_blocks) * 32], output)
    return output


_PREFILL_WEIGHT_OPS_BLOCKS8 = pl.dynamic("V41_WEIGHT_BLOCKS8")


@pl.jit.inline(auto_scope=False)
def prefill_decode_fp8_inline(
    bank: pl.Tensor[[_PREFILL_WEIGHT_OPS_BLOCKS8, 1024], pl.INT8],
    scales: pl.Tensor[[_PREFILL_WEIGHT_OPS_BLOCKS8], pl.UINT8],
    block_offset: pl.Scalar[pl.INDEX],
    output: pl.Tensor[[_PREFILL_WEIGHT_OPS_INPUT_WIDTH, _PREFILL_WEIGHT_OPS_OUTPUT_WIDTH], pl.FP32],
):
    """Exactly decode E4M3/E8M0 32x32 blocks into input-major FP32 weights."""
    bank.bind_dynamic(0, _PREFILL_WEIGHT_OPS_BLOCKS8)
    output.bind_dynamic(0, _PREFILL_WEIGHT_OPS_INPUT_WIDTH)
    output.bind_dynamic(1, _PREFILL_WEIGHT_OPS_OUTPUT_WIDTH)
    width = pl.tensor.dim(output, 0)
    columns = pl.tensor.dim(output, 1)
    input_blocks = width // 32
    matrix_blocks = (columns // 32) * input_blocks
    matrix = pl.slice(bank, [matrix_blocks, 1024], [block_offset, 0])
    matrix_scales = pl.slice(scales, [matrix_blocks], [block_offset])
    scale_rows = pl.reshape(matrix_scales, [1, matrix_blocks])
    for worker in pl.spmd(32, name_hint="prefill_decode_fp8"):
        for block in pl.range(worker, matrix_blocks, 32):
            source = block
            packed = pl.reshape(pl.load(matrix, [source, 0], [1, 1024]), [32, 32])
            values = pl.cast(pl.reinterpret_view(packed, pl.FP8E4M3FN), pl.FP32)
            scale_byte = pl.load(scale_rows, [0, source], [1, 32], valid_shape=[1, 1])
            code = pl.ands(pl.cast(pl.reinterpret_view(scale_byte, pl.INT8), pl.INT32), 255)
            zero_code = pl.shrs(pl.sub(code, 1), 31)
            bits = pl.or_(pl.shls(code, 23), pl.ands(zero_code, 4194304))
            factors = pl.reinterpret_view(bits, pl.FP32)
            decoded = pl.mul(values, pl.tile.read(factors, [0, 0]))
            transposed = pl.transpose(decoded, 0, 1)
            output = pl.store(transposed, [(block % input_blocks) * 32, (block // input_blocks) * 32], output)
    return output


_PREFILL_COMPUTE_T = pl.dynamic("V41_COMPUTE_T")


_PREFILL_COMPUTE_K = pl.dynamic("V41_COMPUTE_K")


@pl.jit.inline(auto_scope=False)
def prefill_round_inline(
    x: pl.Tensor[[_PREFILL_COMPUTE_T, _PREFILL_COMPUTE_K], pl.FP32],
    output: pl.Tensor[[_PREFILL_COMPUTE_T, _PREFILL_COMPUTE_K], pl.FP32],
    official: pl.Scalar[pl.INT32],
):
    rows = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(x, 1)
    for worker in pl.spmd(32, name_hint="prefill_precision_copy"):
        for job in pl.range(worker, ((rows + 7) // 8) * ((width + 511) // 512), 32):
            row = job // ((width + 511) // 512) * 8
            col = job % ((width + 511) // 512) * 512
            active_rows = pl.min(8, rows - row)
            active_width = pl.min(512, width - col)
            value = pl.slice(x, [8, 512], [row, col], valid_shape=[active_rows, active_width])
            if official == 1:
                value = pl.cast(pl.cast(value, pl.BF16, mode="rint"), pl.FP32)
            output[row : row + 8, col : col + 512] = value
    return output
