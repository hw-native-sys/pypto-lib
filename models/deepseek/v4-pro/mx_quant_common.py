# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side MXFP8 / MXFP4 helpers aligned with AscendC Hybrid + hardware tquant.

E8M0 shared exponent follows pto-isa OCP ``fp32_to_fp8_element`` (same as
``pto.tquant.mx`` on device): ``e8m0 = max(fp32_biased_exp(amax) - emax, 0)``,
quant multiplies by ``2**(127 - e8m0)``. Used for v4-pro golden / weight
synthesis; device kernels use ``pl.mx_quant`` / ``pl.matmul_mx``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

# AscendC Linear / MoE / FIA
MX_BLOCK_K = 32
# AscendC KV Cache C8 (shared-KV path)
MX_KV_GROUP = 64

FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0
E8M0_BIAS = 127
FP4_E2M1_MAX = 6.0
# OCP / pto-isa TQUANT shared-exponent emax (fp32_to_fp8_element).
E4M3_EMAX = 8
E2M1_EMAX = 2

# atol / rtol tables ported from AscendC operator golden conventions
ATOL_RTOL: Dict[str, Dict[str, float]] = {
    "li_bf16": {"atol": 1e-4, "rtol": 0.0, "eps": 1e-9, "pct": 0.005},
    "li_fp16": {"atol": 2.5e-5, "rtol": 5e-3, "eps": 1e-9, "pct": 0.005},
    "sas_bf16": {"atol": 1e-4, "rtol": 7.8125e-3, "eps": 1e-9, "pct": 0.005},
    "fia_mxfp8": {"atol": 1e-4, "rtol": 7.8125e-3, "eps": 1e-9, "pct": 0.005},
    "moe_mx": {"atol": 1e-4, "rtol": 7.8125e-3, "eps": 1e-9, "pct": 0.01},
    "qkv_mxfp8": {"atol": 1e-4, "rtol": 7.8125e-3, "eps": 1e-9, "pct": 0.05},
    "mtp_mxfp8": {"atol": 1e-4, "rtol": 7.8125e-3, "eps": 1e-9, "pct": 0.05},
    "oproj_mxfp8": {"atol": 1e-4, "rtol": 7.8125e-3, "eps": 1e-9, "pct": 0.005},
    "indexer_fp8": {"atol": 1e-4, "rtol": 5e-3, "eps": 1e-9, "pct": 0.005},
    "kv_c8": {"atol": 1e-4, "rtol": 5e-3, "eps": 1e-9, "pct": 0.005},
}


def e8m0_uint8_to_float(scale_u8: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 bytes (bias 127) to positive power-of-two float32 scales."""
    return torch.exp2(scale_u8.to(torch.float32) - float(E8M0_BIAS))


def e8m0_float_to_uint8(scale_f: torch.Tensor) -> torch.Tensor:
    """Encode positive float scales to E8M0 uint8 (round via ceil-log2 path upstream)."""
    # Caller should already be on exponent grid; clamp to representable range.
    exp = torch.round(torch.log2(scale_f.clamp_min(2.0 ** -127)))
    return (exp + E8M0_BIAS).clamp(0, 255).to(torch.uint8)


def e8m0_torch(scale_u8: torch.Tensor) -> torch.Tensor:
    """View uint8 E8M0 payload as ``torch.float8_e8m0fnu`` (same bytes)."""
    return scale_u8.contiguous().view(torch.float8_e8m0fnu)


def float8_e8m0_to_uint8(scale: torch.Tensor) -> torch.Tensor:
    return scale.view(torch.uint8)


def _ocp_e8m0_and_inv_scale(amax: torch.Tensor, emax: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match pto-isa ``fp32_to_fp8_element`` / hardware ``pto.tquant.mx``.

    Shared E8M0 byte: ``0`` if ``biased_exp(amax) <= emax``, else
    ``biased_exp - emax`` (NaN → ``0xFF``). Quant multiply factor is
    ``2**(127 - e8m0)`` assembled from exact FP32 exponent bits (same as ISA
    ``scale_exp = 254 - e8m0``), not ``ceil(log2(amax / dtype_max))``.
    """
    a = amax.to(torch.float32).contiguous()
    bits = a.view(torch.int32)
    exp_b = ((bits >> 23) & 0xFF).to(torch.int32)
    mant = bits & 0x007FFFFF
    is_nan = (exp_b == 0xFF) & (mant != 0)
    e8m0_i = torch.where(exp_b <= emax, torch.zeros_like(exp_b), exp_b - int(emax))
    e8m0_i = torch.where(is_nan, torch.full_like(e8m0_i, 0xFF), e8m0_i)
    # Reciprocal scale: biased exp 254 - e8m0 → float 2**(127 - e8m0).
    # Underflow clamp matches ISA (scaling==0 → 2**-127).
    scale_exp = torch.where(
        exp_b <= emax,
        torch.full_like(exp_b, 254),
        254 - (exp_b - int(emax)),
    )
    scale_exp = torch.where(is_nan, torch.full_like(scale_exp, 0xFF), scale_exp).clamp(0, 255)
    inv_bits = (scale_exp.to(torch.int32) << 23).view(torch.float32)
    inv_bits = torch.where(inv_bits == 0.0, torch.full_like(inv_bits, 2.0**-127), inv_bits)
    inv_bits = torch.where(is_nan, torch.full_like(inv_bits, float("nan")), inv_bits)
    return e8m0_i.to(torch.uint8), inv_bits


def dynamic_kv_c8_quant(
    x: torch.Tensor,
    block_k: int = MX_KV_GROUP,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """KV Cache C8 dynamic quant: ``float8_e4m3fn`` + ``float8_e8m0fnu`` per group-64."""
    return dynamic_mx_quant_e4m3(x, block_k=block_k)


def dequant_kv_c8(
    weight_fp8: torch.Tensor,
    scale_e8m0: torch.Tensor,
    block_k: int = MX_KV_GROUP,
) -> torch.Tensor:
    """Dequant KV C8 tensor with E8M0 scales (group=64 along last dim)."""
    return dequant_mxfp8(weight_fp8, scale_e8m0, block_k=block_k)


def dequant_kv_c8_fp32_scale(
    weight_fp8: torch.Tensor,
    scale_fp32: torch.Tensor,
    block_k: int = MX_KV_GROUP,
) -> torch.Tensor:
    """Dequant KV C8 with **FP32** per-group scales ``2**exp`` (device interim layout)."""
    w = weight_fp8.to(torch.float32)
    scale = scale_fp32.to(torch.float32)
    lead = w.shape[:-1]
    k = w.shape[-1]
    if k % block_k != 0:
        raise ValueError(f"K={k} must be divisible by block_k={block_k}")
    if scale.shape[-1] != k // block_k:
        raise ValueError(
            f"scale last dim {scale.shape[-1]} != K/block ({k // block_k})"
        )
    wb = w.reshape(*lead, k // block_k, block_k)
    return (wb * scale.unsqueeze(-1)).reshape_as(w)


def golden_kv_c8_quant_row(row_bf16: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize one compressed KV row after BF16 round (host golden path)."""
    return dynamic_kv_c8_quant(row_bf16.float())


def golden_kv_c8_quant_row_fp32_scale(
    row_bf16: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Golden KV C8 row + **FP32** per-group scale ``2**exp`` (device interim layout)."""
    q, s_e8m0 = dynamic_kv_c8_quant(row_bf16.float())
    s_u8 = float8_e8m0_to_uint8(s_e8m0)
    s_fp32 = e8m0_uint8_to_float(s_u8)
    return q, s_fp32


def dynamic_mx_quant_e4m3(
    x: torch.Tensor,
    block_k: int = MX_BLOCK_K,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Host dynamic MX quant → ``float8_e4m3fn`` + ``float8_e8m0fnu`` scale.

    Matches hardware ``pto.tquant.mx`` / pto-isa OCP golden: per-block-32 abs-max
    → E8M0 ``max(biased_exp(amax) - 8, 0)``, data ``clip(x * 2**(127-e8m0), ±448)``
    cast to e4m3 (multiply by inv-scale, not divide by ``2**ceil(log2(amax/448))``).

    ``x``: [..., K] with ``K % block_k == 0``.
    Returns ``(q, scale)`` with ``q`` same shape as ``x``, ``scale`` [..., K/block_k].
    """
    if x.shape[-1] % block_k != 0:
        raise ValueError(f"K={x.shape[-1]} must be divisible by block_k={block_k}")
    xf = x.to(torch.float32)
    lead = xf.shape[:-1]
    k = xf.shape[-1]
    xb = xf.reshape(*lead, k // block_k, block_k)
    amax = xb.abs().amax(dim=-1)
    scale_u8, inv = _ocp_e8m0_and_inv_scale(amax, E4M3_EMAX)
    q = (xb * inv.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return q.reshape_as(x), e8m0_torch(scale_u8)


def quantize_weight_mxfp8(
    weight: torch.Tensor,
    block_k: int = MX_BLOCK_K,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Static MXFP8 quant along last dim (in / K).

    ``weight``: [..., out, in] or [out, in]; returns e4m3 weight + e8m0 scale [..., out, in/block].
    """
    return dynamic_mx_quant_e4m3(weight, block_k=block_k)


def dequant_mxfp8(
    weight_fp8: torch.Tensor,
    scale_e8m0: torch.Tensor,
    block_k: int = MX_BLOCK_K,
) -> torch.Tensor:
    """Dequant MXFP8 tensor with E8M0 scales along the last dim."""
    w = weight_fp8.to(torch.float32)
    if scale_e8m0.dtype == torch.float8_e8m0fnu:
        scale_u8 = float8_e8m0_to_uint8(scale_e8m0)
    else:
        scale_u8 = scale_e8m0.to(torch.uint8)
    scale = e8m0_uint8_to_float(scale_u8)
    lead = w.shape[:-1]
    k = w.shape[-1]
    if k % block_k != 0:
        raise ValueError(f"K={k} must be divisible by block_k={block_k}")
    if scale.shape[-1] != k // block_k:
        raise ValueError(
            f"scale last dim {scale.shape[-1]} != K/block ({k // block_k})"
        )
    wb = w.reshape(*lead, k // block_k, block_k)
    return (wb * scale.unsqueeze(-1)).reshape_as(w)


def pack_fp4_e2m1(weight_fp4_even_odd: torch.Tensor) -> torch.Tensor:
    """Pack two FP4 e2m1 values per uint8 (low nibble = even index).

    ``weight_fp4_even_odd``: float tensor of shape [..., N] with N even; values in FP4 grid.
    Returns uint8 [..., N/2].
    """
    # Encode via float→uint8 nibbles using torch float4 if available; else manual.
    w = weight_fp4_even_odd.to(torch.float32)
    if w.shape[-1] % 2 != 0:
        raise ValueError("FP4 pack requires even last dim")
    # Map to e2m1 code via round-to-nearest on allowed levels.
    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=w.device
    )
    codes_pos = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32, device=w.device)

    def _encode(vals: torch.Tensor) -> torch.Tensor:
        sign = (vals < 0).to(torch.int32) << 3
        abs_v = vals.abs()
        # nearest level index
        idx = (abs_v.unsqueeze(-1) - levels).abs().argmin(dim=-1)
        return (codes_pos[idx] | sign).to(torch.uint8)

    lo = _encode(w[..., 0::2])
    hi = _encode(w[..., 1::2])
    return (lo | (hi << 4)).to(torch.uint8)


def unpack_fp4_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 FP4 pairs → float32 [..., 2*N]."""
    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=packed.device
    )
    p = packed.to(torch.uint8)
    lo = p & 0xF
    hi = (p >> 4) & 0xF

    def _decode(code: torch.Tensor) -> torch.Tensor:
        sign = torch.where((code & 0x8) != 0, -1.0, 1.0)
        mag = levels[(code & 0x7).long()]
        return sign * mag

    out = torch.stack((_decode(lo), _decode(hi)), dim=-1)
    return out.reshape(*p.shape[:-1], p.shape[-1] * 2)


def quantize_weight_mxfp4(
    weight: torch.Tensor,
    block_k: int = MX_BLOCK_K,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Static MXFP4 along last dim → packed uint8 [..., in/2] + e8m0 [..., in/block]."""
    if weight.shape[-1] % block_k != 0:
        raise ValueError("in dim must be divisible by block_k")
    if weight.shape[-1] % 2 != 0:
        raise ValueError("in dim must be even for FP4 pack")
    wf = weight.to(torch.float32)
    lead = wf.shape[:-1]
    k = wf.shape[-1]
    xb = wf.reshape(*lead, k // block_k, block_k)
    amax = xb.abs().amax(dim=-1)
    scale_u8, inv = _ocp_e8m0_and_inv_scale(amax, E2M1_EMAX)
    q = xb * inv.unsqueeze(-1)
    packed = pack_fp4_e2m1(q.reshape(*lead, k))
    return packed, e8m0_torch(scale_u8)


def dequant_mxfp4(
    packed: torch.Tensor,
    scale_e8m0: torch.Tensor,
    block_k: int = MX_BLOCK_K,
) -> torch.Tensor:
    """Dequant packed MXFP4 + E8M0 → float32 [..., in]."""
    w_fp4 = unpack_fp4_e2m1(packed)
    if scale_e8m0.dtype == torch.float8_e8m0fnu:
        scale_u8 = float8_e8m0_to_uint8(scale_e8m0)
    else:
        scale_u8 = scale_e8m0.to(torch.uint8)
    scale = e8m0_uint8_to_float(scale_u8)
    lead = w_fp4.shape[:-1]
    k = w_fp4.shape[-1]
    if k % block_k != 0:
        raise ValueError(f"K={k} must be divisible by block_k={block_k}")
    wb = w_fp4.reshape(*lead, k // block_k, block_k)
    return (wb * scale.unsqueeze(-1)).reshape_as(w_fp4)


# --- ISA ZZ / NN scale packing (pto-isa tmatmul_mx gen_data) ---


def convert_x1_scale_format(
    x1_mx_gm: np.ndarray, block_size: int = 16, c0_size_mx: int = 2
) -> np.ndarray:
    """Pack A-side E8M0 scales to MX_A_ZZ GM layout."""
    m, k = x1_mx_gm.shape
    pad_m = (block_size - m % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    if pad_m > 0 or pad_k > 0:
        padded = np.pad(x1_mx_gm, ((0, pad_m), (0, pad_k)), mode="constant", constant_values=0)
    else:
        padded = x1_mx_gm
    m_padded, k_padded = padded.shape
    x1 = padded.reshape(
        (int(m_padded / block_size), block_size, int(k_padded / c0_size_mx), c0_size_mx)
    )
    x1 = x1.transpose(0, 2, 1, 3)
    return x1.reshape(x1.shape[0] * x1.shape[1], x1.shape[2] * x1.shape[3])


def zz_fp16_gather_indices(m: int, kmx: int, block_size: int = 16, c0_size_mx: int = 2):
    """INT32 gather indices for device-side ND→ZZ pack via FP16 reinterpret.

    ``tile.gather`` rejects UINT8/FP8E8M0; ZZ reorders at ``c0=2`` bytes, so
    reinterpret the flat scale as FP16 and gather with these indices::

        xs_f16 = pl.tile.reinterpret_view(xs_u8, pl.FP16)
        packed_f16 = pl.tile.gather(xs_f16, idx_tile, tmp)
    """
    import torch

    assert kmx % c0_size_mx == 0 and m % block_size == 0
    nd = np.arange(m * kmx, dtype=np.float64).reshape(m, kmx)
    zz_bytes = convert_x1_scale_format(nd, block_size, c0_size_mx).astype(np.int32).reshape(-1)
    idx = (zz_bytes[::2] // 2).astype(np.int32)
    assert idx.size == m * kmx // 2
    return torch.from_numpy(idx.copy()).reshape(1, m * kmx // 2)

def convert_x2_scale_format(
    x2_mx_gm: np.ndarray, block_size: int = 16, c0_size_mx: int = 2
) -> np.ndarray:
    """Pack B-side E8M0 scales to MX_B_NN GM layout."""
    k, n = x2_mx_gm.shape
    pad_n = (block_size - n % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    if pad_n > 0 or pad_k > 0:
        padded = np.pad(x2_mx_gm, ((0, pad_k), (0, pad_n)), mode="constant", constant_values=0)
    else:
        padded = x2_mx_gm
    k_padded, n_padded = padded.shape
    x2 = padded.reshape(
        (int(k_padded / c0_size_mx), c0_size_mx, int(n_padded / 16), 16)
    ).transpose(2, 0, 3, 1)
    return x2.reshape(x2.shape[1] * x2.shape[3], x2.shape[0] * x2.shape[2])


def unpack_x1_scale_format(
    packed: np.ndarray, m: int, k: int, block_size: int = 16, c0_size_mx: int = 2
) -> np.ndarray:
    """Inverse of :func:`convert_x1_scale_format` (trim padding)."""
    pad_m = (block_size - m % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    m_p, k_p = m + pad_m, k + pad_k
    x = packed.reshape(
        int(m_p / block_size), int(k_p / c0_size_mx), block_size, c0_size_mx
    )
    x = x.transpose(0, 2, 1, 3).reshape(m_p, k_p)
    return x[:m, :k]


def unpack_x2_scale_format(
    packed: np.ndarray, k: int, n: int, block_size: int = 16, c0_size_mx: int = 2
) -> np.ndarray:
    """Inverse of :func:`convert_x2_scale_format` (trim padding)."""
    pad_n = (block_size - n % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    k_p, n_p = k + pad_k, n + pad_n
    # packed reshape matches convert output layout
    x = packed.reshape(k_p // c0_size_mx, n_p // 16, c0_size_mx, 16)
    # convert did transpose(2,0,3,1) on (k/c0, c0, n/16, 16) → (n/16, k/c0, 16, c0)
    # packed stored as (k_p, n_p) with that order flattened as
    # (k/c0 * c0, n/16 * 16) = (k_p, n_p) after reshape(shape[1]*shape[3], shape[0]*shape[2])
    # Recover via inverse of that flatten:
    tmp = packed.reshape(c0_size_mx, n_p // 16, k_p // c0_size_mx, 16)
    # Actually use the known forward path inverse carefully:
    fwd_in = np.zeros((k_p, n_p), dtype=packed.dtype)
    # Brute-force safe inverse: scatter by replaying indices
    src = np.arange(k_p * n_p, dtype=np.int64).reshape(k_p, n_p)
    packed_idx = convert_x2_scale_format(src.astype(np.float64)).astype(np.int64)
    flat_packed = packed.reshape(-1)
    flat_out = np.empty(k_p * n_p, dtype=packed.dtype)
    flat_out[packed_idx.reshape(-1)] = flat_packed
    return flat_out.reshape(k_p, n_p)[:k, :n]


def pack_scale_b_nn(
    scale_e8m0: torch.Tensor, n_tile: int | None = None
) -> torch.Tensor:
    """Pack logical B-scale [K/32, N] to MX_B_NN bytes (single full-matrix convert_x2).

    For device loads that tile along K and/or N with BaseShape==TileShape (ptoas
    EmitC), use :func:`pack_scale_b_nn_tiled` instead.
    """
    del n_tile
    u8 = float8_e8m0_to_uint8(scale_e8m0).cpu().numpy()
    packed = convert_x2_scale_format(u8)
    assert packed.size == u8.size
    return (
        torch.from_numpy(np.ascontiguousarray(packed).reshape(-1).copy())
        .view(torch.float8_e8m0fnu)
        .reshape(scale_e8m0.shape)
    )


def unpack_scale_b_nn(
    packed_e8m0: torch.Tensor, n_tile: int | None = None
) -> torch.Tensor:
    """Unpack a single full-matrix MX_B_NN buffer to logical [K/32, N]."""
    del n_tile
    k, n = packed_e8m0.shape
    u8 = float8_e8m0_to_uint8(packed_e8m0).cpu().numpy().reshape(k, n)
    logical = unpack_x2_scale_format(u8, k, n)
    return (
        torch.from_numpy(np.ascontiguousarray(logical).reshape(-1).copy())
        .view(torch.float8_e8m0fnu)
        .reshape(k, n)
    )


def pack_scale_b_nn_tiled(
    scale_e8m0: torch.Tensor,
    k_tile_rows: int,
    n_tile: int,
) -> torch.Tensor:
    """Independently pack each ``[k_tile_rows, n_tile]`` logical block.

    Returns shape ``[num_n * num_k * k_tile_rows, n_tile]`` with tiles ordered
    ``for nb in num_n: for kb in num_k: pack(...)``. Device loads use offset
    ``[(nb * num_k + kb) * k_tile_rows, 0]`` so ptoas BaseShape==TileShape is valid.
    ``convert_x2`` is not windowable along K or N — full-matrix pack + col/row
    offset loads read the wrong bytes.
    """
    u8 = float8_e8m0_to_uint8(scale_e8m0).cpu().numpy()
    k, n = u8.shape
    if k % k_tile_rows != 0 or n % n_tile != 0:
        raise ValueError(
            f"scale shape {(k, n)} must be divisible by tile {(k_tile_rows, n_tile)}"
        )
    num_k = k // k_tile_rows
    num_n = n // n_tile
    parts: list[np.ndarray] = []
    for nb in range(num_n):
        for kb in range(num_k):
            block = u8[
                kb * k_tile_rows : (kb + 1) * k_tile_rows,
                nb * n_tile : (nb + 1) * n_tile,
            ]
            parts.append(convert_x2_scale_format(block))
    packed = np.concatenate(parts, axis=0)
    assert packed.shape == (num_n * num_k * k_tile_rows, n_tile)
    return (
        torch.from_numpy(np.ascontiguousarray(packed).reshape(-1).copy())
        .view(torch.float8_e8m0fnu)
        .reshape(packed.shape)
    )


def unpack_scale_b_nn_tiled(
    packed_e8m0: torch.Tensor,
    k_tile_rows: int,
    n_tile: int,
    logical_k: int,
    logical_n: int,
) -> torch.Tensor:
    """Inverse of :func:`pack_scale_b_nn_tiled` → logical ``[logical_k, logical_n]``."""
    if logical_k % k_tile_rows != 0 or logical_n % n_tile != 0:
        raise ValueError("logical shape must be divisible by tile")
    num_k = logical_k // k_tile_rows
    num_n = logical_n // n_tile
    u8 = float8_e8m0_to_uint8(packed_e8m0).cpu().numpy()
    if u8.shape != (num_n * num_k * k_tile_rows, n_tile):
        raise ValueError(
            f"packed shape {u8.shape} != expected {(num_n * num_k * k_tile_rows, n_tile)}"
        )
    out = np.empty((logical_k, logical_n), dtype=np.uint8)
    idx = 0
    for nb in range(num_n):
        for kb in range(num_k):
            block = u8[idx * k_tile_rows : (idx + 1) * k_tile_rows]
            idx += 1
            out[
                kb * k_tile_rows : (kb + 1) * k_tile_rows,
                nb * n_tile : (nb + 1) * n_tile,
            ] = unpack_x2_scale_format(block, k_tile_rows, n_tile)
    return (
        torch.from_numpy(np.ascontiguousarray(out).reshape(-1).copy())
        .view(torch.float8_e8m0fnu)
        .reshape(logical_k, logical_n)
    )


def mx_b_nn_tile_row0(nb: int, kb: int, num_k: int, k_tile_rows: int) -> int:
    """Row offset into a :func:`pack_scale_b_nn_tiled` buffer for tile (nb, kb)."""
    return (nb * num_k + kb) * k_tile_rows



def dequant_mxfp8_b(
    weight_fp8: torch.Tensor,
    scale_e8m0: torch.Tensor,
    block_k: int = MX_BLOCK_K,
) -> torch.Tensor:
    """Dequant Right-matrix MXFP8 ``[K, N]`` with scale ``[K/block, N]``."""
    bf = weight_fp8.to(torch.float32)
    if scale_e8m0.dtype == torch.float8_e8m0fnu:
        b_u8 = float8_e8m0_to_uint8(scale_e8m0)
    else:
        b_u8 = scale_e8m0.to(torch.uint8)
    b_s = e8m0_uint8_to_float(b_u8)
    k, n = bf.shape
    if k % block_k != 0:
        raise ValueError(f"K={k} must be divisible by block_k={block_k}")
    if b_s.shape != (k // block_k, n):
        raise ValueError(f"B scale shape {tuple(b_s.shape)} != ({k // block_k}, {n})")
    bb = bf.reshape(k // block_k, block_k, n)
    return (bb * b_s.unsqueeze(1)).reshape(k, n)


def quantize_weight_mxfp8_kn(
    weight_kn: torch.Tensor,
    block_k: int = MX_BLOCK_K,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Static MXFP8 quant for Right-matrix ``[K, N]`` → e4m3 + e8m0 ``[K/block, N]``."""
    if weight_kn.shape[0] % block_k != 0:
        raise ValueError(f"K={weight_kn.shape[0]} must be divisible by {block_k}")
    wf = weight_kn.to(torch.float32)
    k, n = wf.shape
    xb = wf.reshape(k // block_k, block_k, n)
    amax = xb.abs().amax(dim=1)  # [K/block, N]
    scale_u8, inv = _ocp_e8m0_and_inv_scale(amax, E4M3_EMAX)
    q = (xb * inv.unsqueeze(1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return q.reshape(k, n), e8m0_torch(scale_u8)


def mx_matmul_fp8(
    a_fp8: torch.Tensor,
    a_scale: torch.Tensor,
    b_fp8: torch.Tensor,
    b_scale: torch.Tensor,
    block_k: int = MX_BLOCK_K,
) -> torch.Tensor:
    """Host MX matmul: dequant A[M,K], B[K,N] then FP32 matmul → FP32 [M,N]."""
    a = dequant_mxfp8(a_fp8, a_scale, block_k=block_k)
    b = dequant_mxfp8_b(b_fp8, b_scale, block_k=block_k)
    return a @ b


def gen_mxfp8_weight_kn(
    shape_kn: Tuple[int, int],
    dequant_std: float,
    chan_cv: float,
    block_k: int = MX_BLOCK_K,
    pack_nn: bool = True,
    n_tile: int | None = None,
    k_tile: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Synthesize MXFP8 Right-weight ``[K, N]`` + E8M0 scale.

    When both ``n_tile`` and ``k_tile`` are set, scale is
    :func:`pack_scale_b_nn_tiled` with shape
    ``[(N/n_tile)*(K/k_tile)*(k_tile/block_k), n_tile]``.
    Otherwise scale stays ``[K/block, N]`` via :func:`pack_scale_b_nn`.
    """
    k, n = shape_kn
    if k % block_k != 0:
        raise ValueError(f"K={k} must be divisible by {block_k}")
    W = torch.randn(k, n) * torch.exp(chan_cv * torch.randn(1, n))
    w_kn, scale_kn = quantize_weight_mxfp8_kn(W, block_k=block_k)
    w_dq = dequant_mxfp8_b(w_kn, scale_kn, block_k=block_k)
    std = w_dq.std().clamp_min(1e-12)
    W2 = w_dq * (dequant_std / std)
    w_kn, scale_kn = quantize_weight_mxfp8_kn(W2, block_k=block_k)
    if pack_nn:
        if n_tile is not None and k_tile is not None:
            if k % k_tile != 0 or n % n_tile != 0:
                raise ValueError(f"K,N={(k, n)} must be divisible by k_tile,n_tile={(k_tile, n_tile)}")
            if k_tile % block_k != 0:
                raise ValueError(f"k_tile={k_tile} must be divisible by block_k={block_k}")
            scale_kn = pack_scale_b_nn_tiled(
                scale_kn, k_tile_rows=k_tile // block_k, n_tile=n_tile
            )
        else:
            scale_kn = pack_scale_b_nn(scale_kn)
    return w_kn, scale_kn


def shared_expert_mxfp8_golden(
    x_bf16: torch.Tensor,
    w1_fp8: torch.Tensor,
    w1_scale: torch.Tensor,
    w3_fp8: torch.Tensor,
    w3_scale: torch.Tensor,
    w2_fp8: torch.Tensor,
    w2_scale: torch.Tensor,
    swiglu_limit: float = 0.0,
    block_k: int = MX_BLOCK_K,
    scales_packed_nn: bool = True,
) -> torch.Tensor:
    """AscendC-style shared expert: dyn MX → GEMM → SwiGLU → dyn MX → down GEMM.

    Weights are stored as Right matrices: w1/w3 ``[D, MOE_INTER]``, w2 ``[MOE_INTER, D]``,
    with B-side scales ``[K/32, N]`` (optionally MX_B_NN packed).
    """
    import torch.nn.functional as F

    def _b_scale(s: torch.Tensor) -> torch.Tensor:
        return unpack_scale_b_nn(s) if scales_packed_nn else s

    x_q, x_s = dynamic_mx_quant_e4m3(x_bf16, block_k=block_k)
    gate = mx_matmul_fp8(x_q, x_s, w1_fp8, _b_scale(w1_scale), block_k=block_k)
    up = mx_matmul_fp8(x_q, x_s, w3_fp8, _b_scale(w3_scale), block_k=block_k)
    if swiglu_limit and swiglu_limit > 0:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(-swiglu_limit, swiglu_limit)
    h = F.silu(gate) * up
    h_q, h_s = dynamic_mx_quant_e4m3(h, block_k=block_k)
    out = mx_matmul_fp8(h_q, h_s, w2_fp8, _b_scale(w2_scale), block_k=block_k)
    return out


def fp4_floats_to_torch_x2(w_fp4: torch.Tensor) -> torch.Tensor:
    """Encode logical FP4 float values as ``torch.float4_e2m1fn_x2`` (same shape).

    Each element stores one FP4 value in the low nibble (high nibble 0). This keeps
    ``pl.Tensor[..., pl.FP4]`` shapes identical to the FP8 case for ``matmul_mx``.
    """
    packed = pack_fp4_e2m1(
        torch.stack((w_fp4, torch.zeros_like(w_fp4)), dim=-1).reshape(*w_fp4.shape[:-1], w_fp4.shape[-1] * 2)
    )
    # pack_fp4 on [..., 2N] → [..., N]; that is our same-shape encoding.
    return packed.contiguous().view(torch.float4_e2m1fn_x2)


def torch_x2_to_fp4_floats(w_x2: torch.Tensor) -> torch.Tensor:
    """Decode ``float4_e2m1fn_x2`` (low-nibble encoding) → float32 FP4 values."""
    u8 = w_x2.view(torch.uint8)
    # Re-pack as (low, 0) pairs then unpack via shared helper by treating each
    # byte as a packed pair with high nibble zero.
    return unpack_fp4_e2m1(u8)[..., 0::2]


def quantize_weight_mxfp4_kn(
    weight_kn: torch.Tensor,
    block_k: int = MX_BLOCK_K,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Static MXFP4 for Right-matrix ``[K, N]`` → float4_e2m1fn_x2 + e8m0 ``[K/block, N]``."""
    if weight_kn.shape[0] % block_k != 0:
        raise ValueError(f"K={weight_kn.shape[0]} must be divisible by {block_k}")
    wf = weight_kn.to(torch.float32)
    k, n = wf.shape
    xb = wf.reshape(k // block_k, block_k, n)
    amax = xb.abs().amax(dim=1)
    scale_u8, inv = _ocp_e8m0_and_inv_scale(amax, E2M1_EMAX)
    q = xb * inv.unsqueeze(1)
    # Snap to FP4 grid
    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=q.device
    )
    sign = torch.where(q < 0, -1.0, 1.0)
    abs_q = q.abs()
    idx = (abs_q.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    q_grid = sign * levels[idx]
    w_x2 = fp4_floats_to_torch_x2(q_grid.reshape(k, n))
    return w_x2, e8m0_torch(scale_u8)


def dequant_mxfp4_b(
    weight_x2: torch.Tensor,
    scale_e8m0: torch.Tensor,
    block_k: int = MX_BLOCK_K,
) -> torch.Tensor:
    """Dequant Right-matrix MXFP4 ``[K, N]`` float4_x2 + scale ``[K/block, N]``."""
    w = torch_x2_to_fp4_floats(weight_x2)
    if scale_e8m0.dtype == torch.float8_e8m0fnu:
        b_u8 = float8_e8m0_to_uint8(scale_e8m0)
    else:
        b_u8 = scale_e8m0.to(torch.uint8)
    b_s = e8m0_uint8_to_float(b_u8)
    k, n = w.shape
    bb = w.reshape(k // block_k, block_k, n)
    return (bb * b_s.unsqueeze(1)).reshape(k, n)


def mx_matmul_fp8_fp4(
    a_fp8: torch.Tensor,
    a_scale: torch.Tensor,
    b_fp4_x2: torch.Tensor,
    b_scale: torch.Tensor,
    block_k: int = MX_BLOCK_K,
) -> torch.Tensor:
    """Host W4A8 MX matmul: FP8 A × FP4 B."""
    a = dequant_mxfp8(a_fp8, a_scale, block_k=block_k)
    b = dequant_mxfp4_b(b_fp4_x2, b_scale, block_k=block_k)
    return a @ b


def dynamic_mx_quant_e2m1(
    x: torch.Tensor,
    block_k: int = MX_BLOCK_K,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Host dynamic MXFP4 act quant → float4_x2 (low-nibble) + E8M0 scale.

    Matches device ``pl.mx_quant(..., mode=\"mxfp4\")`` along the last dim:
    per-block abs-max → E8M0 with ``E2M1_EMAX``, snap to FP4 grid, pack as
    ``float4_e2m1fn_x2`` with the same element shape as ``x``.
    """
    if x.shape[-1] % block_k != 0:
        raise ValueError(f"K={x.shape[-1]} must be divisible by block_k={block_k}")
    xf = x.to(torch.float32)
    lead = xf.shape[:-1]
    k = xf.shape[-1]
    xb = xf.reshape(*lead, k // block_k, block_k)
    amax = xb.abs().amax(dim=-1)
    scale_u8, inv = _ocp_e8m0_and_inv_scale(amax, E2M1_EMAX)
    q = xb * inv.unsqueeze(-1)
    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=q.device
    )
    sign = torch.where(q < 0, -1.0, 1.0)
    abs_q = q.abs()
    idx = (abs_q.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    q_grid = (sign * levels[idx]).reshape_as(x)
    return fp4_floats_to_torch_x2(q_grid), e8m0_torch(scale_u8)


def dequant_mxfp4_a(
    a_x2: torch.Tensor,
    scale_e8m0: torch.Tensor,
    block_k: int = MX_BLOCK_K,
) -> torch.Tensor:
    """Dequant Left-matrix MXFP4 ``[..., K]`` float4_x2 + scale ``[..., K/block]``."""
    a = torch_x2_to_fp4_floats(a_x2)
    if scale_e8m0.dtype == torch.float8_e8m0fnu:
        s_u8 = float8_e8m0_to_uint8(scale_e8m0)
    else:
        s_u8 = scale_e8m0.to(torch.uint8)
    scale = e8m0_uint8_to_float(s_u8)
    lead = a.shape[:-1]
    k = a.shape[-1]
    if k % block_k != 0:
        raise ValueError(f"K={k} must be divisible by block_k={block_k}")
    if scale.shape[-1] != k // block_k:
        raise ValueError(
            f"scale last dim {scale.shape[-1]} != K/block ({k // block_k})"
        )
    ab = a.reshape(*lead, k // block_k, block_k)
    return (ab * scale.unsqueeze(-1)).reshape_as(a)


def mx_matmul_fp4(
    a_fp4_x2: torch.Tensor,
    a_scale: torch.Tensor,
    b_fp4_x2: torch.Tensor,
    b_scale: torch.Tensor,
    block_k: int = MX_BLOCK_K,
) -> torch.Tensor:
    """Host W4A4 MX matmul: FP4 A × FP4 B (temporary until PTOAS mixed W4A8)."""
    a = dequant_mxfp4_a(a_fp4_x2, a_scale, block_k=block_k)
    b = dequant_mxfp4_b(b_fp4_x2, b_scale, block_k=block_k)
    return a @ b


def gen_mxfp4_weight_kn(
    shape_kn: Tuple[int, int],
    dequant_std: float,
    block_k: int = MX_BLOCK_K,
    pack_nn: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Synthesize MXFP4 Right-weight ``[K, N]`` float4_x2 + E8M0 ``[K/block, N]``."""
    k, n = shape_kn
    if k % block_k != 0:
        raise ValueError(f"K={k} must be divisible by {block_k}")
    W = torch.randn(k, n)
    w_x2, scale_kn = quantize_weight_mxfp4_kn(W, block_k=block_k)
    w_dq = dequant_mxfp4_b(w_x2, scale_kn, block_k=block_k)
    std = w_dq.std().clamp_min(1e-12)
    W2 = w_dq * (dequant_std / std)
    w_x2, scale_kn = quantize_weight_mxfp4_kn(W2, block_k=block_k)
    if pack_nn:
        scale_kn = pack_scale_b_nn(scale_kn)
    return w_x2, scale_kn


def routed_expert_mx_golden(
    recv_x_bf16: torch.Tensor,
    recv_weights: torch.Tensor,
    recv_expert_count: torch.Tensor,
    w1_x2: torch.Tensor,
    w1_scale: torch.Tensor,
    w3_x2: torch.Tensor,
    w3_scale: torch.Tensor,
    w2_x2: torch.Tensor,
    w2_scale: torch.Tensor,
    swiglu_limit: float = 0.0,
    block_k: int = MX_BLOCK_K,
    scales_packed_nn: bool = True,
) -> torch.Tensor:
    """Routed expert golden for legacy W4A4 (MXFP4×MXFP4) fixtures.

    Prefer :func:`routed_expert_mxfp8_golden` for the current A5 device path
    (temporary MXFP8 until FP4 Mat→Right EmitC is fixed).
    """
    import torch.nn.functional as F

    def _b_scale(s: torch.Tensor) -> torch.Tensor:
        return unpack_scale_b_nn(s) if scales_packed_nn else s

    e, recv_max, d = recv_x_bf16.shape
    out = torch.zeros(e, recv_max, d, dtype=torch.float32)
    for ei in range(e):
        n_rows = int(recv_expert_count[ei, 0].item())
        if n_rows <= 0:
            continue
        x = recv_x_bf16[ei, :n_rows, :]
        x_q, x_s = dynamic_mx_quant_e2m1(x, block_k=block_k)
        gate = mx_matmul_fp4(x_q, x_s, w1_x2[ei], _b_scale(w1_scale[ei]), block_k)
        up = mx_matmul_fp4(x_q, x_s, w3_x2[ei], _b_scale(w3_scale[ei]), block_k)
        if swiglu_limit and swiglu_limit > 0:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(-swiglu_limit, swiglu_limit)
        h = F.silu(gate) * up
        h_q, h_s = dynamic_mx_quant_e2m1(h, block_k=block_k)
        y = mx_matmul_fp4(h_q, h_s, w2_x2[ei], _b_scale(w2_scale[ei]), block_k)
        y = y * recv_weights[ei, :n_rows].reshape(-1, 1).float()
        out[ei, :n_rows, :] = y
    return out


def routed_expert_mxfp8_golden(
    recv_x_bf16: torch.Tensor,
    recv_weights: torch.Tensor,
    recv_expert_count: torch.Tensor,
    w1_fp8: torch.Tensor,
    w1_scale: torch.Tensor,
    w3_fp8: torch.Tensor,
    w3_scale: torch.Tensor,
    w2_fp8: torch.Tensor,
    w2_scale: torch.Tensor,
    swiglu_limit: float = 0.0,
    block_k: int = MX_BLOCK_K,
    k_tile: int | None = None,
    n_tile_w13: int | None = None,
    n_tile_w2: int | None = None,
    scales_packed_nn: bool = True,
) -> torch.Tensor:
    """Routed expert golden for temporary W8A8 (MXFP8×MXFP8) device path.

    When ``k_tile`` / ``n_tile_*`` are set, B-scales are tiled MX_B_NN
    (``pack_scale_b_nn_tiled``) and activations are re-quantized per K-tile
    (matches ``expert_routed`` / ``expert_shared``). Otherwise full-matrix
    ``pack_scale_b_nn`` + single full-K quant is used.
    ``recv_weights`` scales the down-proj output (combine-ready).
    """
    import torch.nn.functional as F

    def _b_scale(s: torch.Tensor, n_tile: int | None, logical_k: int, logical_n: int) -> torch.Tensor:
        if not scales_packed_nn:
            return s
        if k_tile is not None and n_tile is not None:
            return unpack_scale_b_nn_tiled(
                s,
                k_tile_rows=k_tile // block_k,
                n_tile=n_tile,
                logical_k=logical_k // block_k,
                logical_n=logical_n,
            )
        return unpack_scale_b_nn(s)

    def _mx_mm(x_f: torch.Tensor, w: torch.Tensor, w_s: torch.Tensor) -> torch.Tensor:
        if k_tile is None:
            xq, xs = dynamic_mx_quant_e4m3(x_f, block_k=block_k)
            return mx_matmul_fp8(xq, xs, w, w_s, block_k=block_k)
        acc = None
        for k0 in range(0, x_f.shape[-1], k_tile):
            xq, xs = dynamic_mx_quant_e4m3(x_f[..., k0 : k0 + k_tile], block_k=block_k)
            part = mx_matmul_fp8(
                xq,
                xs,
                w[k0 : k0 + k_tile],
                w_s[k0 // block_k : (k0 + k_tile) // block_k],
                block_k=block_k,
            )
            acc = part if acc is None else acc + part
        return acc

    e, recv_max, d = recv_x_bf16.shape
    moe_inter = w1_fp8.shape[-1]
    out = torch.zeros(e, recv_max, d, dtype=torch.float32)
    for ei in range(e):
        n_rows = int(recv_expert_count[ei, 0].item())
        if n_rows <= 0:
            continue
        x = recv_x_bf16[ei, :n_rows, :].float()
        w1_s = _b_scale(w1_scale[ei], n_tile_w13, d, moe_inter)
        w3_s = _b_scale(w3_scale[ei], n_tile_w13, d, moe_inter)
        w2_s = _b_scale(w2_scale[ei], n_tile_w2, moe_inter, d)
        gate = _mx_mm(x, w1_fp8[ei], w1_s)
        up = _mx_mm(x, w3_fp8[ei], w3_s)
        if swiglu_limit and swiglu_limit > 0:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(-swiglu_limit, swiglu_limit)
        h = F.silu(gate) * up
        y = _mx_mm(h, w2_fp8[ei], w2_s)
        y = y * recv_weights[ei, :n_rows].reshape(-1, 1).float()
        out[ei, :n_rows, :] = y
    return out

