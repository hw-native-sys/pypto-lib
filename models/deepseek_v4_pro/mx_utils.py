# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side MXFP4/MXFP8 helpers for DeepSeek-V4-Pro MoE experts.

Native MX payloads (FP8E4M3FN + per-32 group scales) are produced on the host.
Device expert kernels use ``pl.matmul_mx`` for W1/W3 (see ``expert_shared``)
and still use INT8 ``pl.matmul`` for W2 until the down-projection migrates.
**Never** use ``pl.quant_mx``.
"""

from __future__ import annotations

import math

MX_GROUP = 32
SCALE_BLOCK_SIZE = 16
SCALE_C0_SIZE = 2
FP8_E4M3_MAX = 448.0
FP4_MAX = 6.0
TINY = 1e-20

# Precomputed FP4 nibble -> FP8 E4M3 codes (Issue #238 MXFP4 magnitude table).
NIBBLE_LUT = [
    0x00,
    0x30,
    0x38,
    0x3C,
    0x40,
    0x44,
    0x48,
    0x4C,
    0x80,
    0xB0,
    0xB8,
    0xBC,
    0xC0,
    0xC4,
    0xC8,
    0xCC,
]


def pack_a_scale(scale_codes):
    """Pack logical A scales ``[M, K/32]`` into the MX_A_ZZ physical layout."""
    m, k_groups = scale_codes.shape
    assert m % SCALE_BLOCK_SIZE == 0
    assert k_groups % SCALE_C0_SIZE == 0
    return (
        scale_codes.reshape(
            m // SCALE_BLOCK_SIZE,
            SCALE_BLOCK_SIZE,
            k_groups // SCALE_C0_SIZE,
            SCALE_C0_SIZE,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(m, k_groups)
    )


def unpack_a_scale(packed_codes):
    """Restore MX_A_ZZ physical scale bytes to logical ``[M, K/32]``."""
    m, k_groups = packed_codes.shape
    return (
        packed_codes.reshape(
            m // SCALE_BLOCK_SIZE,
            k_groups // SCALE_C0_SIZE,
            SCALE_BLOCK_SIZE,
            SCALE_C0_SIZE,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(m, k_groups)
    )


def pack_b_scale(scale_codes):
    """Pack logical B scales ``[K/32, N]`` into the MX_B_NN physical layout."""
    k_groups, n = scale_codes.shape
    assert k_groups % SCALE_C0_SIZE == 0
    assert n % SCALE_BLOCK_SIZE == 0
    return (
        scale_codes.reshape(
            k_groups // SCALE_C0_SIZE,
            SCALE_C0_SIZE,
            n // SCALE_BLOCK_SIZE,
            SCALE_BLOCK_SIZE,
        )
        .permute(2, 0, 3, 1)
        .contiguous()
        .reshape(k_groups, n)
    )


def unpack_b_scale(packed_codes):
    """Restore MX_B_NN physical scale bytes to logical ``[K/32, N]``."""
    k_groups, n = packed_codes.shape
    return (
        packed_codes.reshape(
            n // SCALE_BLOCK_SIZE,
            k_groups // SCALE_C0_SIZE,
            SCALE_BLOCK_SIZE,
            SCALE_C0_SIZE,
        )
        .permute(1, 3, 0, 2)
        .contiguous()
        .reshape(k_groups, n)
    )


def _e8m0_codes_from_amax(amax, fp_max: float):
    """Round-up power-of-two E8M0 codes for per-group amax / fp_max."""
    import torch

    scale = (amax / fp_max).clamp_min(TINY)
    exp = torch.ceil(torch.log2(scale)).to(torch.int32) + 127
    return exp.clamp(0, 255).to(torch.uint8)


def e8m0_codes_to_fp32(codes):
    """Decode logical E8M0 uint8 codes to FP32 powers of two."""
    import torch

    return torch.exp2(codes.to(torch.float32) - 127.0)


def host_quant_mxfp8(x_bf16_or_fp32, *, pack_zz: bool = False, return_e8m0: bool = False):
    """Per-row group-32 MXFP8 quant along the last dim.

    Returns ``(data_fp8, scale)`` with logical shapes ``[..., K]`` /
    ``[..., K/32]``. By default ``scale`` is **decoded FP32** for the kernel
    ABI. Pass ``return_e8m0=True`` for packed/logical E8M0 codes (docs /
    fixtures). ``pack_zz`` only applies when returning E8M0 and the leading
    row dim is a multiple of 16 with even K/32.
    """
    import torch

    x = x_bf16_or_fp32.float()
    *lead, k = x.shape
    assert k % MX_GROUP == 0
    groups = k // MX_GROUP
    xg = x.reshape(*lead, groups, MX_GROUP)
    amax = xg.abs().amax(dim=-1)
    codes = _e8m0_codes_from_amax(amax, FP8_E4M3_MAX)
    scale_f = e8m0_codes_to_fp32(codes)
    q = (xg / scale_f.unsqueeze(-1)).to(torch.float8_e4m3fn)
    data = q.reshape(*lead, k)
    if not return_e8m0:
        return data, scale_f.contiguous()

    scale = codes
    if pack_zz and len(lead) == 1:
        m = lead[0]
        if m % SCALE_BLOCK_SIZE == 0 and groups % SCALE_C0_SIZE == 0:
            scale = pack_a_scale(scale.reshape(m, groups)).reshape(m, groups)
    scale_e8m0 = scale.contiguous().view(torch.float8_e8m0fnu)
    return data, scale_e8m0


def gen_mxfp8_weight_kn(out: int, inn: int, dequant_std: float, *, chan_cv: float = 0.5, seed: int = 0):
    """Simulate an MXFP8 weight grid in Cube ``[K, N] = [inn, out]`` layout.

    Returns FP8 data ``[inn, out]`` and **decoded FP32** logical scales
    ``[inn/32, out]`` (kernel ABI). Does **not** requantize to INT8.
    """
    import torch

    g = torch.Generator().manual_seed(seed)
    weight_base = torch.randn(out, inn, generator=g)
    channel_noise = torch.randn(out, 1, generator=g)
    channel_gain = torch.exp(chan_cv * channel_noise)
    w = weight_base * channel_gain  # [out, inn]
    assert inn % MX_GROUP == 0
    wg = w.reshape(out, inn // MX_GROUP, MX_GROUP)
    amax = wg.abs().amax(dim=-1)
    codes_on = _e8m0_codes_from_amax(amax, FP8_E4M3_MAX)  # [out, inn/32]
    scale_f = e8m0_codes_to_fp32(codes_on)
    q = (wg / scale_f.unsqueeze(-1)).to(torch.float8_e4m3fn)
    data_on = q.reshape(out, inn)
    data_kn = data_on.transpose(0, 1).contiguous()  # [inn, out]
    codes_kn = codes_on.transpose(0, 1).contiguous()
    decoded = data_kn.float() * e8m0_codes_to_fp32(codes_kn).repeat_interleave(MX_GROUP, dim=0)
    cur_std = decoded.std().clamp_min(TINY)
    gain = dequant_std / cur_std
    exp_shift = int(round(math.log2(float(gain))))
    codes_kn = (codes_kn.to(torch.int32) + exp_shift).clamp(0, 255).to(torch.uint8)
    scale_fp32 = e8m0_codes_to_fp32(codes_kn).contiguous()
    return data_kn.to(torch.float8_e4m3fn), scale_fp32


def gen_mxfp8_weight_kn_device(out, inn, dequant_std, *, chan_cv=0.5, seed=0):
    """Device payloads for Cube ``[K, N] = [inn, out]`` MX matmul rhs.

    Returns FP8 data ``[inn, out]`` and MX_B_NN-packed E8M0 scale
    ``[inn // 32, out]``.
    """
    import torch

    data_kn, scale_fp32 = gen_mxfp8_weight_kn(out, inn, dequant_std, chan_cv=chan_cv, seed=seed)
    codes = (
        (torch.log2(scale_fp32.clamp_min(TINY)) + 127.0)
        .round()
        .to(torch.int32)
        .clamp(0, 255)
        .to(torch.uint8)
    )
    scale_e8m0 = pack_b_scale(codes).contiguous().view(torch.float8_e8m0fnu)
    return data_kn, scale_e8m0


def host_mxfp8_activation(x_bf16_or_fp32):
    """MXFP8 activation + MX_A_ZZ E8M0 scales for device ``matmul_mx`` lhs."""
    return host_quant_mxfp8(x_bf16_or_fp32, pack_zz=True, return_e8m0=True)


def gen_mxfp4_weight_kn(out: int, inn: int, dequant_std: float, *, seed: int = 0):
    """Simulate MXFP4 (e2m1 + per-32 E8M0) in Cube ``[inn, out]`` layout.

    Returns packed FP4 bytes ``[inn/2, out]``, **decoded FP32** scales
    ``[inn/32, out]``, and INT16 nibble-index tensor ``[inn, out]`` ready for
    host/device LUT gather → FP8E4M3FN.
    """
    import torch

    FP4_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    FP4_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])

    g = torch.Generator().manual_seed(seed)
    w = torch.randn(out, inn, generator=g)  # [out, inn]
    assert inn % MX_GROUP == 0
    wg = w.reshape(out, inn // MX_GROUP, MX_GROUP)
    absw = wg.abs()
    codes_on = _e8m0_codes_from_amax(absw.amax(dim=-1), FP4_MAX)  # [out, inn/32]
    scale_f = e8m0_codes_to_fp32(codes_on)
    idx = torch.bucketize(absw / scale_f.unsqueeze(-1), FP4_MID).clamp_max(7)
    sign = (wg < 0).to(torch.int64)
    nibble = idx + sign * 8  # 0..15
    nibble_flat = nibble.reshape(out, inn)

    values = torch.sign(wg) * FP4_MAG[idx]
    decoded_on = (values * scale_f.unsqueeze(-1)).reshape(out, inn)
    cur_std = decoded_on.std().clamp_min(TINY)
    gain = dequant_std / cur_std
    exp_shift = int(round(math.log2(float(gain))))
    codes_on = (codes_on.to(torch.int32) + exp_shift).clamp(0, 255).to(torch.uint8)

    nibble_kn = nibble_flat.transpose(0, 1).contiguous()  # [inn, out]
    codes_kn = codes_on.transpose(0, 1).contiguous()  # [inn/32, out]

    assert inn % 2 == 0
    lo = nibble_kn[0::2, :] & 0x0F
    hi = nibble_kn[1::2, :] & 0x0F
    packed = (lo | (hi << 4)).to(torch.uint8).contiguous()  # [inn/2, out]

    indices = nibble_kn.to(torch.int16)
    scale_fp32 = e8m0_codes_to_fp32(codes_kn).contiguous()
    return packed, scale_fp32, indices


def fp4_packed_to_nibble_indices(packed):
    """Unpack FP4 bytes ``[inn/2, out]`` to INT16 nibble indices ``[inn, out]``.

    ``gen_mxfp4_weight_kn`` packs adjacent K rows on axis 0 (lo then hi nibble).
    """
    import torch

    packed_u8 = packed.contiguous().view(torch.uint8)
    half, *rest = packed_u8.shape
    lo = (packed_u8 & 0x0F).to(torch.int16)
    hi = ((packed_u8 >> 4) & 0x0F).to(torch.int16)
    out = torch.empty(half * 2, *rest, dtype=torch.int16)
    out[0::2, ...] = lo
    out[1::2, ...] = hi
    return out


def nibble_indices_to_fp8(indices):
    """Host LUT: INT16 nibble indices → FP8E4M3FN payload (same codes as device gather)."""
    import torch

    lut = torch.tensor(NIBBLE_LUT, dtype=torch.int16)
    codes = lut[indices.to(torch.int64).clamp(0, 15)]
    return (codes & 0xFF).to(torch.uint8).view(torch.float8_e4m3fn)


def matmul_mx_golden(a, a_scale, b, b_scale):
    """FP32 golden for MX-style matmul from data + per-group scales.

    ``a``/``b`` are ``[M,K]`` / ``[K,N]``. Scales are either decoded FP32
    ``[M, K/32]`` / ``[K/32, N]`` or logical E8M0 uint8 codes (auto-detected).
    """
    import torch

    m, k = a.shape
    k2, n = b.shape
    assert k == k2
    a_s = a_scale
    b_s = b_scale
    if a_s.dtype != torch.float32 and a_s.dtype != torch.float64:
        a_s = e8m0_codes_to_fp32(a_s.contiguous().view(torch.uint8))
    if b_s.dtype != torch.float32 and b_s.dtype != torch.float64:
        b_s = e8m0_codes_to_fp32(b_s.contiguous().view(torch.uint8))
    a_s = a_s.to(torch.float64)
    b_s = b_s.to(torch.float64)
    k_group = torch.arange(k) // MX_GROUP
    a_scaled = a.to(torch.float64) * a_s[:, k_group]
    b_scaled = b.to(torch.float64) * b_s[k_group, :]
    return torch.matmul(a_scaled, b_scaled).to(torch.float32)


def decode_e8m0_codes(scale_e8m0, *, side: str = "a"):
    """Unpack ZZ/NN-packed E8M0 tensor to logical uint8 codes."""
    import torch

    codes = scale_e8m0.contiguous().view(torch.uint8)
    if side == "a":
        return unpack_a_scale(codes)
    if side == "b":
        return unpack_b_scale(codes)
    raise ValueError(f"side must be 'a' or 'b', got {side!r}")


def make_nibble_lut_tensor():
    """``[1, 16]`` INT16 LUT tensor for device gather."""
    import torch

    return torch.tensor(NIBBLE_LUT, dtype=torch.int16).view(1, 16).contiguous()
