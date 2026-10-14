# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared helpers for the DeepSeek-V4-Pro drivers.

Previously ``utils/golden_fwd.py`` and ``utils/weights_flash.py``. Flattened
into one module so this directory matches the other model dirs
(``deepseek_v4_flash_mtp/utils.py``, ``deepseek_v4_flash_dspark/utils.py``),
and so the daily sweeps skip it. Those sweeps decide what is a case by
grepping each file for a script-entry guard, and the converter's guard made it
one; the converter now runs through :func:`main` instead, reading ``sys.argv``
unchanged. That grep is a plain substring match over the whole file, so do not
spell the guard's dunder name anywhere in this module, comments included.

The MXFP4/MXFP8 helpers in this module are intentionally lightweight to
import. Full-network golden dependencies are loaded only by the functions
that use them, so leaf MoE tests can reuse the helpers without freezing the
prefill configuration or importing the distributed kernels.

Full-network torch golden (packed-prefill forward)
--------------------------------------------------

``golden_prefill_fwd(tensors)`` replays ``prefill_fwd.l3_prefill_fwd`` on host
by chaining the leaf goldens exactly the way the device program composes the
leaf kernels:

- ``input_pack.golden_pack_x_hc`` builds the layer-0 hidden state per rank.
- Per layer, the preset's attention kind runs per rank
  (``golden_prefill_attention_{swa,hca,csa}``) followed by the cross-rank
  ``moe.golden_moe`` whose ``x_next`` becomes the next layer's ``x_hc``.
- The trailing layer's MoE output is written straight into
  ``pre_hc_hidden_out``; ``hc_head`` + the final ``rms_norm`` produce
  ``hidden_out``, and a distributed-LM-head replay fills ``logits`` and
  ``sampled_ids``.

The *tensors* dict is the golden-harness scratch dict for
``prefill_fwd.build_tensor_specs()``: every tensor is ``[N_RANKS, ...]``
rank-stacked, per-FWD-layer weights and caches are packed along dim 1 by
FWD-layer id, and the CSA/HCA-compact stacks are packed by per-kind order
(ascending layer id of that kind). All cache slices handed to the leaf goldens
are torch views of the stacked tensors, so their in-place slot updates land in
the output cache tensors that validation reads back.

Real DeepSeek-V4-Flash checkpoint loader
----------------------------------------

Converts the HuggingFace-style hybrid MXFP4-MXFP8 checkpoint (43 layers,
256 routed experts, ``expert_dtype=fp4`` + block-FP8 attention linears) into
the exact host-tensor ABI ``prefill_fwd.py`` / ``decode_fwd.py`` consume:

- FP8 e4m3 weights (128x128-block UE8M0 scales) are dequantized and converted
  to group-32 MXFP8 with the pinned CANN weight converter's scale and rounding
  rules. Every attention and shared-expert projection uses FP8 KN payloads
  and E8M0 scales packed for ``MX_B_NN``.
- FP4 e2m1 routed-expert weights are dequantized and requantized using the
  same pinned CANN rules before packing the native FP4 tile layout.
- Per-layer tensors are stacked along dim 1 exactly like
  ``_make_stacked_spec`` (FWD stacks by model layer id 0..42; CSA/HCA stacks
  by kind order = ascending layer id of that compress-ratio kind), sharded
  per EP rank for the routed experts (rank ``r`` owns global experts
  ``[r*N_LOCAL, (r+1)*N_LOCAL)``) and per TP rank for ``lm_head_weight``
  (rank ``r`` reads vocab shard ``r % TP``), and replicated across ranks
  otherwise.

Synthesized inputs (RoPE tables, ``csa_hadamard_idx``, caches, per-step
metadata) keep their fixture initializers and are not touched here.

Usage — one-time offline conversion, then run the drivers against the cache::

    PYTHONPATH=.:models/deepseek_v4_pro python -c 'import utils; utils.main()' \\
        --variant flash --ep 8 --tp 2 \\
        --ckpt /path/to/DeepSeek-V4-Flash --out build_output/flash_weights_ep8_tp2
    python models/deepseek_v4_pro/prefill_fwd.py --variant flash --ep 8 --tp 2 \\
        -p a5 -d 0,1,2,3,4,5,6,7 --weights build_output/flash_weights_ep8_tp2

``--weights`` also accepts the raw checkpoint directory directly (detected by
``model.safetensors.index.json``); every weight is then converted on the fly
while the harness builds its inputs.
"""

import config

import argparse
import hashlib
import json
import math
import mmap
import os
import struct
import warnings
from pathlib import Path
from typing import Callable

import torch

M = config.ACTIVE
MODEL_CONFIG = config.ACTIVE
ACTIVE_BASE = config.ACTIVE_BASE
INT8_AMAX_EPS = config.INT8_AMAX_EPS
INT8_SCALE_MAX = config.INT8_SCALE_MAX
D = M.hidden_size

# ===========================================================================
# Host-side MXFP4/MXFP8 helpers
# ===========================================================================
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


def build_mxfp4_pair_lut():
    """Build a packed-byte LUT whose INT16 entries contain two FP8 payloads."""
    import torch

    packed = torch.arange(256, dtype=torch.int64)
    fp8_codes = torch.tensor(NIBBLE_LUT, dtype=torch.int64)
    low_codes = fp8_codes[packed & 0x0F]
    high_codes = fp8_codes[packed >> 4]
    pairs_u16 = low_codes | (high_codes << 8)
    pairs_i16 = torch.where(pairs_u16 < 0x8000, pairs_u16, pairs_u16 - 0x10000)
    return pairs_i16.to(torch.int16).reshape(1, 256).repeat(2, 1).contiguous()


def _pack_mxfp4_nibbles_kn_tiles(nibbles_kn, k_tile, n_tile, split_mode):
    """Pack one ``[K, N]`` nibble grid into tile-major AIV lane rows."""
    k, n = nibbles_kn.shape
    if k % k_tile != 0 or n % n_tile != 0:
        raise ValueError("MXFP4 tile packing requires K and N to divide their tiles")
    if k_tile % 2 != 0 or n_tile % 2 != 0:
        raise ValueError("MXFP4 tile packing requires even K and N tiles")

    k_blocks = k // k_tile
    n_blocks = n // n_tile
    blocked = nibbles_kn.reshape(k_blocks, k_tile, n_blocks, n_tile)
    blocked = blocked.permute(2, 0, 1, 3)
    if split_mode == "up_down":
        lanes = blocked.reshape(n_blocks, k_blocks, 2, k_tile // 2, n_tile)
    elif split_mode == "left_right":
        if n_tile % 4 != 0:
            raise ValueError("left-right MXFP4 packing requires N tile divisible by four")
        lanes = blocked.reshape(n_blocks, k_blocks, k_tile, 2, n_tile // 2)
        lanes = lanes.permute(0, 1, 3, 2, 4)
    else:
        raise ValueError(f"unsupported MXFP4 split mode: {split_mode!r}")

    low = lanes[..., 0::2] & 0x0F
    high = lanes[..., 1::2] & 0x0F
    packed = low | (high << 4)
    lane_bytes = k_tile * n_tile // 4
    return packed.contiguous().reshape(n_blocks * k_blocks * 2, lane_bytes)


def pack_checkpoint_mxfp4_weight_tiles(weight_packed, k_tile, n_tile):
    """Reorder checkpoint E2M1 nibbles into the routed Cube tile layout exactly."""
    payload = weight_packed.contiguous().view(torch.uint8)
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
            nibbles.t(), k_tile, n_tile, "up_down"
        ).reshape(packed_rows, packed_cols)
    return result.reshape(*lead, packed_rows, packed_cols)


def _mxfp8_grid_to_mxfp4_nibbles(weight):
    """Map an FP8 grid restricted to the MXFP4 value set back to nibble codes."""
    import torch

    codes = weight.contiguous().view(torch.uint8)
    inverse = torch.zeros(256, dtype=torch.uint8)
    valid = torch.zeros(256, dtype=torch.bool)
    lut_codes = torch.tensor(NIBBLE_LUT, dtype=torch.uint8)
    lut_indices = lut_codes.to(torch.int64)
    inverse[lut_indices] = torch.arange(16, dtype=torch.uint8)
    valid[lut_indices] = True
    code_indices = codes.to(torch.int64)
    if not bool(valid[code_indices].all()):
        raise ValueError("MXFP8 fixture contains a code outside the MXFP4 LUT")
    return inverse[code_indices]


def pack_mxfp4_weight_tiles(weight, k_tile, n_tile, split_mode="up_down"):
    """Pack ``[..., K, N]`` MXFP4-grid FP8 weights into tile-major lane rows."""
    import torch

    *leading, k, n = weight.shape
    lane_bytes = k_tile * n_tile // 4
    tile_rows = (k // k_tile) * (n // n_tile) * 2
    weights = weight.reshape(-1, k, n)
    packed = torch.empty(weights.shape[0], tile_rows, lane_bytes, dtype=torch.uint8)
    for batch in range(weights.shape[0]):
        nibbles_kn = _mxfp8_grid_to_mxfp4_nibbles(weights[batch])
        packed[batch] = _pack_mxfp4_nibbles_kn_tiles(
            nibbles_kn,
            k_tile,
            n_tile,
            split_mode,
        )
    return packed.reshape(*leading, tile_rows, lane_bytes)


def unpack_mxfp4_weight_tiles(
    packed,
    k,
    n,
    k_tile,
    n_tile,
    split_mode="up_down",
):
    """Expand tile-major lane rows into FP8 ``[..., K, N]`` weights."""
    import torch

    *leading, tile_rows, lane_bytes = packed.shape
    k_blocks = k // k_tile
    n_blocks = n // n_tile
    expected_rows = n_blocks * k_blocks * 2
    expected_bytes = k_tile * n_tile // 4
    if tile_rows != expected_rows or lane_bytes != expected_bytes:
        raise ValueError("MXFP4 tile-major payload does not match the requested matrix")

    payloads = packed.contiguous().view(torch.uint8).reshape(-1, tile_rows, lane_bytes)
    output = torch.empty(payloads.shape[0], k, n, dtype=torch.float8_e4m3fn)
    for batch in range(payloads.shape[0]):
        if split_mode == "up_down":
            lane_shape = (n_blocks, k_blocks, 2, k_tile // 2, n_tile // 2)
            lane_bytes_view = payloads[batch].reshape(lane_shape)
            low = lane_bytes_view & 0x0F
            high = lane_bytes_view >> 4
            lanes = torch.stack((low, high), dim=-1)
            blocked = lanes.reshape(n_blocks, k_blocks, k_tile, n_tile)
        elif split_mode == "left_right":
            lane_shape = (n_blocks, k_blocks, 2, k_tile, n_tile // 4)
            lane_bytes_view = payloads[batch].reshape(lane_shape)
            low = lane_bytes_view & 0x0F
            high = lane_bytes_view >> 4
            lanes = torch.stack((low, high), dim=-1)
            lanes = lanes.reshape(n_blocks, k_blocks, 2, k_tile, n_tile // 2)
            blocked = lanes.permute(0, 1, 3, 2, 4).reshape(
                n_blocks,
                k_blocks,
                k_tile,
                n_tile,
            )
        else:
            raise ValueError(f"unsupported MXFP4 split mode: {split_mode!r}")
        nibbles_kn = blocked.permute(1, 2, 0, 3).reshape(k, n)
        output[batch] = nibble_indices_to_fp8(nibbles_kn)
    return output.reshape(*leading, k, n)


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


def pack_b_scale_batched(scale_codes):
    """Pack ``[..., K/32, N]`` E8M0 codes as independent MX_B_NN matrices."""
    *lead, k_groups, n = scale_codes.shape
    assert k_groups % SCALE_C0_SIZE == 0
    assert n % SCALE_BLOCK_SIZE == 0
    lead_axes = list(range(len(lead)))
    base = len(lead)
    return (
        scale_codes.reshape(
            *lead,
            k_groups // SCALE_C0_SIZE,
            SCALE_C0_SIZE,
            n // SCALE_BLOCK_SIZE,
            SCALE_BLOCK_SIZE,
        )
        .permute(*lead_axes, base + 2, base, base + 3, base + 1)
        .contiguous()
        .reshape(*lead, k_groups, n)
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


def unpack_b_scale_batched(packed_codes):
    """Restore independently packed ``[..., K/32, N]`` MX_B_NN matrices."""
    *lead, k_groups, n = packed_codes.shape
    lead_axes = list(range(len(lead)))
    base = len(lead)
    return (
        packed_codes.reshape(
            *lead,
            n // SCALE_BLOCK_SIZE,
            k_groups // SCALE_C0_SIZE,
            SCALE_BLOCK_SIZE,
            SCALE_C0_SIZE,
        )
        .permute(*lead_axes, base + 1, base + 3, base, base + 2)
        .contiguous()
        .reshape(*lead, k_groups, n)
    )


def _e8m0_codes_from_amax(amax, fp_max: float):
    """Ascend OCP shared-exponent E8M0 codes for each group maximum."""
    format_emax = int(math.floor(math.log2(fp_max)))
    _, exponent = torch.frexp(amax)
    codes = exponent.to(torch.int32) - 1 - format_emax + 127
    codes = codes.clamp(0, 255)
    codes = torch.where(amax == 0, torch.zeros_like(codes), codes)
    return codes.to(torch.uint8)


def e8m0_codes_to_fp32(codes):
    """Decode logical E8M0 uint8 codes to FP32 powers of two."""
    return torch.exp2(codes.to(torch.float32) - 127.0)


def host_quant_mxfp8(
    x_bf16_or_fp32,
    *,
    pack_zz: bool = False,
    return_e8m0: bool = False,
):
    """Per-row group-32 MXFP8 quant along the last dim.

    Returns ``(data_fp8, scale)`` with logical shapes ``[..., K]`` /
    ``[..., K/32]``. By default ``scale`` is decoded FP32 for the kernel ABI.
    Pass ``return_e8m0=True`` for packed/logical E8M0 codes. ``pack_zz`` only
    applies when returning E8M0 and the leading row dim is a multiple of 16
    with even K/32.
    """
    x = x_bf16_or_fp32.float()
    *lead, k = x.shape
    assert k % MX_GROUP == 0
    groups = k // MX_GROUP
    xg = x.reshape(*lead, groups, MX_GROUP)
    amax = xg.abs().amax(dim=-1)
    codes = _e8m0_codes_from_amax(amax, FP8_E4M3_MAX)
    scale_f = e8m0_codes_to_fp32(codes)
    # Ascend TQUANT saturates finite overflow; the CPU cast otherwise emits NaN.
    q = (xg / scale_f.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
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


def gen_mxfp8_weight_kn(
    out: int,
    inn: int,
    dequant_std: float,
    *,
    chan_cv: float = 0.5,
    seed: int = 0,
):
    """Simulate an MXFP8 weight grid in Cube ``[K, N]`` layout.

    Returns FP8 data ``[inn, out]`` and decoded FP32 logical scales
    ``[inn/32, out]``.
    """
    generator = torch.Generator().manual_seed(seed)
    weight_base = torch.randn(out, inn, generator=generator)
    channel_noise = torch.randn(out, 1, generator=generator)
    channel_gain = torch.exp(chan_cv * channel_noise)
    weight = weight_base * channel_gain
    assert inn % MX_GROUP == 0
    weight_groups = weight.reshape(out, inn // MX_GROUP, MX_GROUP)
    amax = weight_groups.abs().amax(dim=-1)
    codes_on = _e8m0_codes_from_amax(amax, FP8_E4M3_MAX)
    scale_f = e8m0_codes_to_fp32(codes_on)
    quantized = (weight_groups / scale_f.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    data_on = quantized.reshape(out, inn)
    data_kn = data_on.transpose(0, 1).contiguous()
    codes_kn = codes_on.transpose(0, 1).contiguous()
    decoded = data_kn.float() * e8m0_codes_to_fp32(codes_kn).repeat_interleave(
        MX_GROUP, dim=0
    )
    current_std = decoded.std().clamp_min(TINY)
    gain = dequant_std / current_std
    exponent_shift = int(round(math.log2(float(gain))))
    codes_kn = (
        (codes_kn.to(torch.int32) + exponent_shift)
        .clamp(0, 255)
        .to(torch.uint8)
    )
    scale_fp32 = e8m0_codes_to_fp32(codes_kn).contiguous()
    return data_kn.to(torch.float8_e4m3fn), scale_fp32


def gen_mxfp8_weight_kn_device(
    out,
    inn,
    dequant_std,
    *,
    chan_cv=0.5,
    seed=0,
):
    """Return Cube-layout MXFP8 weights with MX_B_NN-packed E8M0 scales."""
    data_kn, scale_fp32 = gen_mxfp8_weight_kn(
        out,
        inn,
        dequant_std,
        chan_cv=chan_cv,
        seed=seed,
    )
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
    return host_quant_mxfp8(
        x_bf16_or_fp32,
        pack_zz=True,
        return_e8m0=True,
    )


def host_quant_mxfp8_weight_kn(weight_nk):
    """Quantize ``[..., N, K]`` weights for the Cube MXFP8 rhs ABI."""
    data_nk, scale_ng = host_quant_mxfp8(weight_nk, return_e8m0=True)
    data_kn = data_nk.transpose(-2, -1).contiguous()
    codes_kn = (
        scale_ng.contiguous()
        .view(torch.uint8)
        .transpose(-2, -1)
        .contiguous()
    )
    scale_nn = pack_b_scale_batched(codes_kn).view(torch.float8_e8m0fnu)
    return data_kn, scale_nn


def cann_quant_mxfp8_weight_kn(weight_nk):
    """CANN ba83ab4 weight conversion: group-32 E4M3 with packed E8M0.

    This follows ``models/deepseek_v4/utils/mx_quantize.py:quantize_mx``:
    round2decimal shared exponents, nearest element rounding (ties away
    from zero), and finite saturation. It differs from the OCP activation
    scale used by :func:`host_quant_mxfp8`.
    """
    weight = weight_nk.float()
    *lead, n, k = weight.shape
    if k % MX_GROUP or n % SCALE_BLOCK_SIZE:
        raise ValueError("MXFP8 KN weights require K divisible by 32 and N by 16")
    grouped = weight.reshape(*lead, n, k // MX_GROUP, MX_GROUP)
    amax = grouped.abs().amax(dim=-1)
    # The official converter substitutes FP32_MIN_NORMAL for an all-zero block.
    safe_amax = torch.where(amax == 0, torch.full_like(amax, 2.0**-126), amax)
    exponent = torch.floor(torch.log2(safe_amax))
    mantissa = safe_amax / torch.exp2(exponent)
    shared = (exponent + (mantissa > 1.75) - 8).clamp(-127, 127)
    scaled = grouped / torch.exp2(shared).unsqueeze(-1)
    absolute = scaled.abs()
    private_exp = torch.floor(torch.log2(absolute + (absolute == 0))).clamp_min(-6)
    step = torch.exp2(private_exp - 3)
    rounded = torch.sign(scaled) * torch.floor(absolute / step + 0.5) * step
    payload = rounded.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    data_kn = payload.reshape(*lead, n, k).transpose(-2, -1).contiguous()
    codes_kn = (shared + 127).to(torch.uint8).transpose(-2, -1).contiguous()
    return data_kn, pack_b_scale_batched(codes_kn).view(torch.float8_e8m0fnu)


def cann_quant_mxfp4_weight_nk(weight_nk):
    """Pinned CANN group-32 E2M1 weights, low nibble first and logical E8M0.

    CANN requantizes the dequantized checkpoint even when it already contains
    MXFP4. A source block whose largest magnitude is 3, for example, receives
    half the original scale and twice the original FP4 payload values.
    """
    weight = weight_nk.float()
    if weight.shape[-1] % MX_GROUP:
        raise ValueError("MXFP4 weights require K divisible by 32")
    grouped = weight.reshape(*weight.shape[:-1], -1, MX_GROUP)
    amax = grouped.abs().amax(dim=-1)
    safe_amax = torch.where(amax == 0, torch.full_like(amax, 2.0**-126), amax)
    exponent = torch.floor(torch.log2(safe_amax))
    mantissa = safe_amax / torch.exp2(exponent)
    shared = (exponent + (mantissa > 1.75) - 2).clamp(-127, 127)
    scaled = grouped / torch.exp2(shared).unsqueeze(-1)
    # E2M1 nearest-away ties occur at these exact midpoints. right=True
    # selects the larger magnitude at a tie, unlike ordinary OCP rint.
    midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    magnitude = torch.bucketize(scaled.abs(), midpoints, right=True).to(torch.uint8)
    # A negative nonzero value can round to negative zero in CANN.
    sign = (scaled < 0).to(torch.uint8) * 8
    nibbles = (magnitude + sign).reshape(weight.shape)
    packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)
    return packed.contiguous(), (shared + 127).to(torch.uint8).contiguous()


def cann_requant_checkpoint_mxfp4(packed, scale_codes):
    """Convert official packed weights, avoiding dense expansion when exact.

    Blocks with maximum E2M1 magnitude 4 or 6 already have the CANN scale.
    Their only possible change is canonicalizing source negative-zero nibbles.
    Other blocks use the complete CANN quantizer, so this optimization does not
    assume that every source expert already satisfies the scale invariant.
    """
    raw = packed.contiguous().view(torch.uint8)
    codes = scale_codes.contiguous().view(torch.uint8)
    low, high = raw & 15, raw >> 4
    group_max = torch.maximum(low & 7, high & 7).reshape(*raw.shape[:-1], -1, 16).amax(-1)
    if bool((group_max >= 6).all()):
        low = torch.where(low == 8, 0, low)
        high = torch.where(high == 8, 0, high)
        return (low | (high << 4)).contiguous(), codes
    return cann_quant_mxfp4_weight_nk(dequant_fp4(raw, codes))


def gen_mxfp4_weight_kn(
    out: int,
    inn: int,
    dequant_std: float,
    *,
    seed: int = 0,
):
    """Simulate MXFP4 in Cube ``[inn, out]`` layout."""
    fp4_magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    fp4_midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])

    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(out, inn, generator=generator)
    assert inn % MX_GROUP == 0
    weight_groups = weight.reshape(out, inn // MX_GROUP, MX_GROUP)
    absolute_weight = weight_groups.abs()
    codes_on = _e8m0_codes_from_amax(
        absolute_weight.amax(dim=-1),
        FP4_MAX,
    )
    scale_f = e8m0_codes_to_fp32(codes_on)
    indices = torch.bucketize(
        absolute_weight / scale_f.unsqueeze(-1),
        fp4_midpoints,
    ).clamp_max(7)
    sign = (weight_groups < 0).to(torch.int64)
    nibble = indices + sign * 8
    nibble_flat = nibble.reshape(out, inn)

    values = torch.sign(weight_groups) * fp4_magnitudes[indices]
    decoded_on = (values * scale_f.unsqueeze(-1)).reshape(out, inn)
    current_std = decoded_on.std().clamp_min(TINY)
    gain = dequant_std / current_std
    exponent_shift = int(round(math.log2(float(gain))))
    codes_on = (
        (codes_on.to(torch.int32) + exponent_shift)
        .clamp(0, 255)
        .to(torch.uint8)
    )

    nibble_kn = nibble_flat.transpose(0, 1).contiguous()
    codes_kn = codes_on.transpose(0, 1).contiguous()

    assert inn % 2 == 0
    low = nibble_kn[0::2, :] & 0x0F
    high = nibble_kn[1::2, :] & 0x0F
    packed = (low | (high << 4)).to(torch.uint8).contiguous()

    nibble_indices = nibble_kn.to(torch.int16)
    scale_fp32 = e8m0_codes_to_fp32(codes_kn).contiguous()
    return packed, scale_fp32, nibble_indices


def nibble_indices_to_fp8(indices):
    """Convert INT16 FP4 nibble indices to FP8E4M3FN payloads."""
    lut = torch.tensor(NIBBLE_LUT, dtype=torch.int16)
    codes = lut[indices.to(torch.int64).clamp(0, 15)]
    return (codes & 0xFF).to(torch.uint8).view(torch.float8_e4m3fn)


def mxfp4_to_mxfp8_weight_kn(weight_packed, scale_e8m0):
    """Expand a checkpoint MXFP4 weight into the current Cube MXFP8 ABI.

    ``weight_packed`` is ``[..., N, K/2]`` with adjacent K values stored low
    nibble first. ``scale_e8m0`` is ``[..., N, K/32]``. The result is exact
    FP8E4M3 data ``[..., K, N]`` plus the unchanged E8M0 codes transposed and
    packed for ``MX_B_NN`` as ``[..., K/32, N]``.
    """
    packed_u8 = weight_packed.contiguous().view(torch.uint8)
    scale_codes = scale_e8m0.contiguous().view(torch.uint8)
    *weight_lead, n, half_k = packed_u8.shape
    *scale_lead, scale_n, k_groups = scale_codes.shape
    if (
        weight_lead != scale_lead
        or n != scale_n
        or half_k * 2 != k_groups * MX_GROUP
    ):
        raise ValueError(
            "MXFP4 weight/scale shapes must be [..., N, K/2] and "
            "[..., N, K/32], "
            f"got {tuple(weight_packed.shape)} and {tuple(scale_e8m0.shape)}"
        )

    low = packed_u8 & 0x0F
    high = (packed_u8 >> 4) & 0x0F
    indices_nk = torch.stack((low, high), dim=-1).reshape(
        *weight_lead,
        n,
        half_k * 2,
    )
    data_kn = nibble_indices_to_fp8(indices_nk).transpose(-2, -1).contiguous()
    codes_kn = scale_codes.transpose(-2, -1).contiguous()
    packed_codes = pack_b_scale_batched(codes_kn)
    return data_kn, packed_codes.view(torch.float8_e8m0fnu)


def gen_mxfp4_weight_kn_device(
    out: int,
    inn: int,
    dequant_std: float,
    *,
    seed: int = 0,
):
    """Generate checkpoint-shaped MXFP4 and expand it through the real bridge."""
    packed_kn, scale_fp32, _ = gen_mxfp4_weight_kn(
        out,
        inn,
        dequant_std,
        seed=seed,
    )
    codes_kn = (
        (torch.log2(scale_fp32.clamp_min(TINY)) + 127.0)
        .round()
        .to(torch.int32)
        .clamp(0, 255)
        .to(torch.uint8)
    )
    packed_nk = packed_kn.transpose(0, 1).contiguous()
    codes_nk = (
        codes_kn.transpose(0, 1)
        .contiguous()
        .view(torch.float8_e8m0fnu)
    )
    return mxfp4_to_mxfp8_weight_kn(packed_nk, codes_nk)


def matmul_mx_golden(a, a_scale, b, b_scale):
    """Compute an FP32 golden for MX-style matmul from data and scales."""
    m, k = a.shape
    k2, n = b.shape
    assert k == k2
    a_s = a_scale
    b_s = b_scale
    if a_s.dtype not in (torch.float32, torch.float64):
        a_s = e8m0_codes_to_fp32(a_s.contiguous().view(torch.uint8))
    if b_s.dtype not in (torch.float32, torch.float64):
        b_s = e8m0_codes_to_fp32(b_s.contiguous().view(torch.uint8))
    a_s = a_s.to(torch.float64)
    b_s = b_s.to(torch.float64)
    k_group = torch.arange(k) // MX_GROUP
    a_scaled = a.to(torch.float64) * a_s[:, k_group]
    b_scaled = b.to(torch.float64) * b_s[k_group, :]
    return torch.matmul(a_scaled, b_scaled).to(torch.float32)


def decode_e8m0_codes(scale_e8m0, *, side: str = "a"):
    """Unpack ZZ/NN-packed E8M0 tensor to logical uint8 codes."""
    codes = scale_e8m0.contiguous().view(torch.uint8)
    if side == "a":
        return unpack_a_scale(codes)
    if side == "b":
        return unpack_b_scale(codes)
    raise ValueError(f"side must be 'a' or 'b', got {side!r}")


# ===========================================================================
# Real DeepSeek-V4-Flash checkpoint loader
# ===========================================================================
# ---------------------------------------------------------------------------
# Model-layer geometry (mirrors prefill_fwd/decode_fwd stacking rules).
# ---------------------------------------------------------------------------
NUM_LAYERS = M.num_hidden_layers
FWD_RATIOS = M.compress_ratios[:NUM_LAYERS]
# Kind-order slot k of the CSA/HCA stacks maps to the k-th layer of that
# compress-ratio kind in ascending layer-id order (lead layers first, loop
# pairs next, trailing CSA last — ascending id gives exactly that order).
CSA_LAYERS = [i for i, r in enumerate(FWD_RATIOS) if r == 4]
HCA_LAYERS = [i for i, r in enumerate(FWD_RATIOS) if r == 128]

O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
MOE_INTER = M.moe_intermediate_size
# The FULL deployment expert count. Read from the immutable ACTIVE_BASE, never
# from ACTIVE: moe.py shrinks config.ACTIVE.n_routed_experts to 32*EP at import,
# so in driver context ACTIVE already carries the reduced routing space.
N_EXPERTS_FULL = ACTIVE_BASE.n_routed_experts
VOCAB = M.vocab_size

FP8_BLOCK = 128   # weight_block_size for e4m3 weights
FP4_GROUP = 32    # scale group along the input dim for e2m1 experts

# fp4 e2m1 value table for nibble indices 0..15 (bit 3 = sign).
_FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

_SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.uint8,  # decoded via _e8m0_to_fp32
    "I8": torch.int8,
    "I64": torch.int64,
}

FLASH_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
FLASH_CHECKPOINT_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"
CANN_WEIGHT_REFERENCE_REVISION = "ba83ab4aecb8969c831d28abd02de159d279917b"
FLASH_CONFIG_SHA256 = "b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818"
FLASH_INDEX_SHA256 = "7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a"
FLASH_CACHE_SCHEMA = 3
FLASH_SHARDS = {
    "model-00001-of-00046.safetensors": (1059061856, "5176586613905d4beaadbaea1cfecd1693e17f9da4a0ea09d99aab7d3f2f5b7c"),
    "model-00002-of-00046.safetensors": (3566321192, "f04048189d3b472b26d7d02331edd0412e85b184725041c978fdce793a286a0d"),
    "model-00003-of-00046.safetensors": (3566321192, "df5f80b9b4ca54edf7bda4e640a1b0d1bb95d5995843198dfcce2af0bfc4cb35"),
    "model-00004-of-00046.safetensors": (3596229272, "948250b46a6f92df92ef093ab2d0023c31f924232846a84affe84fa3c7794f5c"),
    "model-00005-of-00046.safetensors": (3568768976, "9fda158bc636215aea4f6834821c81f59eea3733223c874ab66b9f3d6740c4c1"),
    "model-00006-of-00046.safetensors": (3590024776, "51a65e6d9d0ccb70013e25ae70a50b177af8f97e59ac798c2d0ed5ebb169fe7a"),
    "model-00007-of-00046.safetensors": (3568768976, "2d782c46d6d293189e01ed13ea108de355fd32a9902cea77b264dfc9c4f10c42"),
    "model-00008-of-00046.safetensors": (3590024776, "b7d9d8d8932e12ea113f2e83ff412fc1d6b663acedf7f34bbe2bfff41a71c595"),
    "model-00009-of-00046.safetensors": (3568768976, "3197a42d282a8368d4286d11aed52d7fee7928d7950fee4dc18d5154b8111060"),
    "model-00010-of-00046.safetensors": (3590024776, "c9cef4200444326f8d802bc1bb5be7c1d15af4bfdec0426db7e841a7bd516a28"),
    "model-00011-of-00046.safetensors": (3568768976, "916b0b34b713c51dc7a83935ccd9e07c4af017ddfcc5feaeec8faa26f1263f89"),
    "model-00012-of-00046.safetensors": (3590026352, "05739c7d91a302f41a4627587982016a6cc874f875a3ea299d1f2e1dcea5cbb6"),
    "model-00013-of-00046.safetensors": (3568770544, "47c5e416b60b9bef9e9005cdad9c991a306ab2dd25a95e1994dda30bd4011905"),
    "model-00014-of-00046.safetensors": (3590026352, "c881e2671ab45428d824197f174d64fb4403b9530451cb1b0b3899881ab34b7e"),
    "model-00015-of-00046.safetensors": (3568770544, "cb3daae8e465c5b49f2b439c045971e96cd658fda152882605e9717573d4c8f5"),
    "model-00016-of-00046.safetensors": (3590026352, "9eba661fba3162a8a051b0283ad3c11c7c17b33d99684e7029ddf732494bd069"),
    "model-00017-of-00046.safetensors": (3568770544, "0c36cbc026c5067164cb506ac6522078478feb539fef264289ef70f1d1688f54"),
    "model-00018-of-00046.safetensors": (3590026352, "1298a07452a409ddbcc5172a4f57a011193bea4f16a6a964c3dd8531cbc06b49"),
    "model-00019-of-00046.safetensors": (3568770544, "d3687748aff78adf42b2bbf7469b8df92e1e3e5dcd90c1c050c8fac62d1fe77a"),
    "model-00020-of-00046.safetensors": (3590026352, "906c652f3c36b510689c2637ebe5172865cc6ccc17515b9fc70ee9e048e7c5af"),
    "model-00021-of-00046.safetensors": (3568770544, "f270bf4d0f0067165020baf3c11264a177182918c1ebeec21d2bf33166b44592"),
    "model-00022-of-00046.safetensors": (3590026352, "d02261b8f1c8d697bac4b23ec9a8423096fb1a1edf923ed1c2faea6cebdb05e8"),
    "model-00023-of-00046.safetensors": (3568770544, "69fab8bfa1cdfd819382cdef5923dcf10d81879aceb0dc0950be45a6d45771f2"),
    "model-00024-of-00046.safetensors": (3590026352, "baba23c06a7b80e108334eb9fe30349de851e822e9b72df756614ae6b5088dbf"),
    "model-00025-of-00046.safetensors": (3568770544, "085b7736ebe3d69574930187f8b213f84b35b36852f0a505067f8755e8b11c89"),
    "model-00026-of-00046.safetensors": (3590026352, "fdde6791ab713c93cb256d05d9560d5ecfd2b913563658b62d32af9a17a91546"),
    "model-00027-of-00046.safetensors": (3568770544, "2f207b9aef9c56e38f73234e9c5c118c1898676c8ab83d2be0dd410fb196bd01"),
    "model-00028-of-00046.safetensors": (3590026352, "2cc519b5a03a30d45717ffb8408a4f833f3a94f70e35a1de38e95a0ffcdc152e"),
    "model-00029-of-00046.safetensors": (3568770544, "d10bf34c789f9294d2cc50b695d259dc1d0d5b2303105329be370eb55f0fd882"),
    "model-00030-of-00046.safetensors": (3590026352, "0f1c471fa9d9d3612c94039e9efb2a60f367782918f236e95c45ab3fe36c166c"),
    "model-00031-of-00046.safetensors": (3568770544, "beafa59d64fae3a5c636be4b2a08956d8d5abaf70856a4a5adb70a200011e43e"),
    "model-00032-of-00046.safetensors": (3590026352, "5c6b2934d87ada60493e201652d72075c56ce608091a3d395f3d8e31b6ce036a"),
    "model-00033-of-00046.safetensors": (3568770544, "c05f917e873d2da0513385c3c30758d29b4b9d244ffde2a5258ae2e6745f23da"),
    "model-00034-of-00046.safetensors": (3590026352, "666b77201ec6946fb1f299f2ebcb462bce53575a97a91fb30db806bbcd099100"),
    "model-00035-of-00046.safetensors": (3568770544, "5e6b9a54fd149ea4908f81cdffd6b09701d9c98e5e7041bb5bd00c6d5c85623f"),
    "model-00036-of-00046.safetensors": (3590026352, "ab72ad9d171fc0867350948e5091878b3c2445a5cfb8a83dd8c25d4272628107"),
    "model-00037-of-00046.safetensors": (3568770544, "93d68bcfc36fdf239f901653c0e96c5d45d8fce4f5be633bbbf93cc75067ec5d"),
    "model-00038-of-00046.safetensors": (3590026352, "809fb799edcf1d9b4511dea68a1ea35c5fc03858e0611a0010913f1c84e66efe"),
    "model-00039-of-00046.safetensors": (3568770544, "49dba248917454c0a8ac90cc5a012fee625fea6672ae55c40d32a5922a29dd91"),
    "model-00040-of-00046.safetensors": (3590026352, "09a7b8b6957ff3426d7461dd49a158f4576c0277ab18b78a881094aef29ba84b"),
    "model-00041-of-00046.safetensors": (3568770544, "a564ac6c6cc7514beadb2e1d9d1fc2baeed5e680ba1b75cd9afc7a50fb90ee85"),
    "model-00042-of-00046.safetensors": (3590026352, "bd3f5b898b041559a534c81a2f9bb53a8f9744ee8db37ed11b3a5c9fcf848882"),
    "model-00043-of-00046.safetensors": (3568770544, "85a414c7991c1276e8db780f6a4390ac25b8f2c7fdb3551831993adb9db69430"),
    "model-00044-of-00046.safetensors": (3590026352, "438b052b8a2d650939e63704f55f1352b946152ba6633cb256c3864ef21d2f62"),
    "model-00045-of-00046.safetensors": (1059332516, "9a0fd242134e9ebe4e6993a7631692944838e4fdf20067b3219caa48eab68045"),
    "model-00046-of-00046.safetensors": (3593956092, "f58f722893a6148216a2155cee4a57fe691cea4d3b323135c433a936b932055d"),
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flash_source_identity() -> dict:
    return {
        "model_id": FLASH_MODEL_ID,
        "revision": FLASH_CHECKPOINT_REVISION,
        "config_sha256": FLASH_CONFIG_SHA256,
        "index_sha256": FLASH_INDEX_SHA256,
    }


class FlashCheckpoint:
    """Zero-copy reader for the sharded DeepSeek-V4-Flash safetensors checkpoint."""

    def __init__(self, ckpt_dir: str | Path) -> None:
        self.dir = Path(ckpt_dir)
        index_path = self.dir / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"not a checkpoint dir (missing {index_path})")
        config_path = self.dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"checkpoint config missing: {config_path}")
        for path, expected in ((config_path, FLASH_CONFIG_SHA256), (index_path, FLASH_INDEX_SHA256)):
            if _file_sha256(path) != expected:
                raise ValueError(
                    f"{path.name} does not match {FLASH_MODEL_ID}@{FLASH_CHECKPOINT_REVISION}; "
                    "use the pinned original Flash checkpoint"
                )
        source_config = json.loads(config_path.read_text(encoding="utf-8"))
        for field in (
            "hidden_size", "num_attention_heads", "head_dim", "q_lora_rank", "o_lora_rank",
            "o_groups", "vocab_size", "moe_intermediate_size", "n_routed_experts",
            "num_experts_per_tok", "num_hidden_layers", "num_hash_layers", "hc_mult",
            "index_n_heads", "index_head_dim", "index_topk", "expert_dtype",
        ):
            if source_config[field] != getattr(ACTIVE_BASE, field):
                raise ValueError(f"checkpoint {field} does not match the active base model")
        self.source_identity = _flash_source_identity()
        with open(index_path, encoding="utf-8") as f:
            self._shard_of = json.load(f)["weight_map"]
        if any(Path(name).name != name for name in self._shard_of.values()):
            raise ValueError("checkpoint index must reference files inside the checkpoint directory")
        # shard name -> (mmap, {tensor: (dtype_str, shape, start, end)})
        self._shards: dict[str, tuple[mmap.mmap, dict]] = {}

    def _shard(self, shard_name: str) -> tuple[mmap.mmap, dict]:
        cached = self._shards.get(shard_name)
        if cached is not None:
            return cached
        path = self.dir / shard_name
        expected_size, expected_sha256 = FLASH_SHARDS[shard_name]
        if path.stat().st_size != expected_size or _file_sha256(path) != expected_sha256:
            raise ValueError(f"checkpoint shard does not match the pinned official snapshot: {shard_name}")
        with open(path, "rb") as f:
            file_size = os.fstat(f.fileno()).st_size
            header_len = struct.unpack("<Q", f.read(8))[0]
            if not 0 < header_len <= min(file_size - 8, 16 * 1024 * 1024):
                raise ValueError(f"invalid safetensors header size: {shard_name}")
            header = json.loads(f.read(header_len))
            header.pop("__metadata__", None)
            base = 8 + header_len
            for name, info in header.items():
                if self._shard_of.get(name) != shard_name or info["dtype"] not in _SAFETENSORS_DTYPES:
                    raise ValueError(f"invalid tensor entry {name!r} in {shard_name}")
                start, end = info["data_offsets"]
                shape = info["shape"]
                width = torch.empty((), dtype=_SAFETENSORS_DTYPES[info["dtype"]]).element_size()
                if (
                    any(not isinstance(dim, int) or dim < 0 for dim in shape)
                    or not 0 <= start <= end <= file_size - base
                    or end - start != math.prod(shape) * width
                ):
                    raise ValueError(f"invalid tensor shape/offsets for {name!r} in {shard_name}")
            entries = {
                name: (info["dtype"], info["shape"], base + info["data_offsets"][0], base + info["data_offsets"][1])
                for name, info in header.items()
            }
            # The mapping stays valid after the fd is closed.
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self._shards[shard_name] = (mm, entries)
        return mm, entries

    def __contains__(self, name: str) -> bool:
        return name in self._shard_of

    def get(self, name: str) -> torch.Tensor:
        """Return tensor ``name`` as a read-only view into the shard mmap."""
        shard_name = self._shard_of.get(name)
        if shard_name is None:
            raise KeyError(f"tensor {name!r} not in checkpoint index")
        mm, entries = self._shard(shard_name)
        dtype_str, shape, start, end = entries[name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # non-writable mmap buffer
            flat = torch.frombuffer(memoryview(mm)[start:end], dtype=torch.uint8)
        return flat.view(_SAFETENSORS_DTYPES[dtype_str]).reshape(shape)


# ---------------------------------------------------------------------------
# Dequantization (checkpoint grids) and conversion to kernel weight ABIs.
# ---------------------------------------------------------------------------
def _e8m0_to_fp32(scale_u8: torch.Tensor) -> torch.Tensor:
    """Decode UE8M0 bytes (unsigned power-of-two exponents) to fp32: 2^(x-127)."""
    return torch.exp2(scale_u8.to(torch.float32) - 127.0)


def dequant_fp8_block(weight: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    """Dequantize an e4m3 ``[out, in]`` weight with a 128x128-block UE8M0 scale."""
    out_dim, in_dim = weight.shape
    expected = ((out_dim + FP8_BLOCK - 1) // FP8_BLOCK, (in_dim + FP8_BLOCK - 1) // FP8_BLOCK)
    if weight.dtype != torch.float8_e4m3fn or tuple(scale_u8.shape) != expected:
        raise ValueError(f"invalid official FP8 weight/scale: {weight.shape}/{scale_u8.shape}")
    scale = _e8m0_to_fp32(scale_u8)
    scale = scale.repeat_interleave(FP8_BLOCK, dim=0)[:out_dim]
    scale = scale.repeat_interleave(FP8_BLOCK, dim=1)[:, :in_dim]
    return weight.to(torch.float32) * scale


def dequant_fp4(weight_packed: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    """Dequantize a packed e2m1 ``[..., out, in/2]`` weight to fp32 ``[..., out, in]``.

    Two fp4 values per byte along the input dim, low nibble first; one UE8M0
    scale per output row per group of 32 unpacked input elements.
    """
    bytes_u8 = weight_packed.view(torch.uint8)
    low = bytes_u8 & 0x0F
    high = (bytes_u8 >> 4) & 0x0F
    nibbles = torch.stack([low, high], dim=-1).reshape(*bytes_u8.shape[:-1], -1)
    values = _FP4_TABLE[nibbles.to(torch.int64)]
    scale = _e8m0_to_fp32(scale_u8).repeat_interleave(FP4_GROUP, dim=-1)
    return values * scale


# ---------------------------------------------------------------------------
# Spec-name converters: checkpoint tensors -> per-rank-stacked host tensors.
# ---------------------------------------------------------------------------
def _replicate(x: torch.Tensor, n_ranks: int) -> torch.Tensor:
    return x.unsqueeze(0).expand(n_ranks, *x.shape).contiguous()


class FlashWeightConverter:
    """Converts spec-named host tensors from a :class:`FlashCheckpoint`.

    Weight/scale pairs (``wq_b``/``wq_b_scale``, routed and shared experts,
    ...) are produced by one dequant+requant pass: converting the weight
    stashes its scale, so requesting the scale right after (the spec order of
    the drivers and of :data:`REAL_WEIGHT_NAMES`) is free.
    """

    def __init__(self, ckpt: FlashCheckpoint, *, ep: int, tp: int) -> None:
        if VOCAB % tp != 0:
            raise ValueError(f"TP {tp} does not divide vocab_size {VOCAB}")
        self.ckpt = ckpt
        self.ep = ep
        self.tp = tp
        # Mirror moe.py: the kernel programs keep 32 local experts per rank and
        # shrink the GLOBAL routing space to 32*EP, so only EP8 deploys the full
        # 256-expert model. For EP<8 this loader takes the FIRST 32*EP checkpoint
        # experts and reduces the router tables to match — a reduced-expert
        # smoke configuration, not the true model output.
        self.n_experts = N_EXPERTS_FULL // 8 * ep
        self.n_local = self.n_experts // ep
        self._stash: dict[str, torch.Tensor] = {}

    # ---- generic helpers -------------------------------------------------
    def _fwd_stack(self, per_layer: Callable[[int], torch.Tensor]) -> torch.Tensor:
        return torch.cat([per_layer(layer) for layer in range(NUM_LAYERS)], dim=0)

    def _kind_stack(self, layers: list[int], per_layer: Callable[[int], torch.Tensor]) -> torch.Tensor:
        return torch.cat([per_layer(layer) for layer in layers], dim=0)

    def _deq_fp8(self, prefix: str) -> torch.Tensor:
        return dequant_fp8_block(self.ckpt.get(f"{prefix}.weight"), self.ckpt.get(f"{prefix}.scale"))

    def _raw(self, name: str) -> torch.Tensor:
        return self.ckpt.get(name).clone()

    def _stacked_pair(
        self, scale_name: str, layers: list[int],
        pair_of_layer: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Convert a per-layer (weight, scale) pair; stash the stacked scale."""
        weights, scales = zip(*[pair_of_layer(layer) for layer in layers])
        self._stash[scale_name] = _replicate(torch.cat(scales, dim=0), self.ep)
        return _replicate(torch.cat(weights, dim=0), self.ep)

    def _stacked_mx_pair(
        self,
        scale_name: str,
        layers: list[int],
        pair_of_layer: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Stack independently packed MX_B_NN matrices as contiguous layer blocks."""
        weights, packed_scales = zip(*[pair_of_layer(layer) for layer in layers])
        self._stash[scale_name] = _replicate(torch.cat(packed_scales, dim=0), self.ep)
        return _replicate(torch.cat(weights, dim=0), self.ep)

    # ---- attention linears ----------------------------------------------
    def _attention_linear(self, layer: int, which: str) -> tuple[torch.Tensor, torch.Tensor]:
        return cann_quant_mxfp8_weight_kn(self._deq_fp8(f"layers.{layer}.attn.{which}"))

    def _wq_b(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._attention_linear(layer, "wq_b")

    def _wo_a(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self._deq_fp8(f"layers.{layer}.attn.wo_a").view(O_GROUPS, O_LORA, -1)
        payload, scale = cann_quant_mxfp8_weight_kn(weight)
        # Every group is independently packed, then concatenated along scale rows.
        return payload, scale.flatten(0, 1)

    def _wo_b(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._attention_linear(layer, "wo_b")

    def _idx_wq_b(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._attention_linear(layer, "indexer.wq_b")

    def _shared(self, layer: int, which: str) -> tuple[torch.Tensor, torch.Tensor]:
        weight_nk = self._deq_fp8(f"layers.{layer}.ffn.shared_experts.{which}")
        return cann_quant_mxfp8_weight_kn(weight_nk)

    # ---- routed experts (EP-sharded) ------------------------------------
    def _routed_layer(self, layer: int, which: str) -> tuple[torch.Tensor, torch.Tensor]:
        """One layer's EP-sharded CANN-requantized MXFP4 weights and scales."""
        from expert_routed import MX_K_TILE, MX_MM_INTER_TILE, MX_W2_D_OUT_TILE

        n_tile = MX_W2_D_OUT_TILE if which == "w2" else MX_MM_INTER_TILE
        weights, scales = [], []
        for rank in range(self.ep):
            experts = range(rank * self.n_local, (rank + 1) * self.n_local)
            rank_weights, rank_scales = [], []
            for expert in experts:
                key = f"layers.{layer}.ffn.experts.{expert}.{which}"
                packed, scale = cann_requant_checkpoint_mxfp4(
                    self.ckpt.get(f"{key}.weight"), self.ckpt.get(f"{key}.scale"))
                rank_weights.append(packed)
                rank_scales.append(scale)
            packed = torch.stack(rank_weights)
            scales_u8 = torch.stack(rank_scales)
            weight_tiles = pack_checkpoint_mxfp4_weight_tiles(packed, MX_K_TILE, n_tile)
            scale_kn = scales_u8.transpose(-2, -1).contiguous()
            weights.append(weight_tiles)
            scales.append(pack_b_scale_batched(scale_kn).view(torch.float8_e8m0fnu))
        return torch.stack(weights), torch.stack(scales)

    def _routed(self, scale_name: str, which: str) -> torch.Tensor:
        from expert_routed import MX_MM_INTER_TILE, MX_W2_D_OUT_TILE

        k_dim, n_dim = (D, MOE_INTER) if which in ("w1", "w3") else (MOE_INTER, D)
        n_tile = MX_W2_D_OUT_TILE if which == "w2" else MX_MM_INTER_TILE
        groups = k_dim // FP4_GROUP
        weight = torch.empty(
            [self.ep, NUM_LAYERS * self.n_local, k_dim * n_dim // n_tile, n_tile // 2],
            dtype=torch.uint8,
        )
        scale = torch.empty(
            [self.ep, NUM_LAYERS * self.n_local * groups, n_dim], dtype=torch.float8_e8m0fnu
        )
        for layer in range(NUM_LAYERS):
            block = slice(layer * self.n_local, (layer + 1) * self.n_local)
            scale_block = slice(layer * self.n_local * groups, (layer + 1) * self.n_local * groups)
            weight_tiles, w_scale = self._routed_layer(layer, which)
            weight[:, block] = weight_tiles
            logical_scale = unpack_b_scale_batched(w_scale.contiguous().view(torch.uint8))
            layer_scale = pack_b_scale_batched(
                logical_scale.reshape(self.ep, self.n_local * groups, n_dim)
            ).view(torch.float8_e8m0fnu)
            scale[:, scale_block] = layer_scale
        self._stash[scale_name] = scale
        return weight

    # ---- gate / routing --------------------------------------------------
    def _gate_bias(self, layer: int) -> torch.Tensor:
        # Decide by layer id, not tensor presence: a checkpoint missing an
        # expected tensor must fail loudly instead of silently zero-filling.
        if layer < M.num_hash_layers:
            return torch.zeros(self.n_experts, dtype=torch.float32)  # hash layers carry no bias
        return self._raw(f"layers.{layer}.ffn.gate.bias")[: self.n_experts]

    def _gate_w(self, layer: int) -> torch.Tensor:
        return self.ckpt.get(f"layers.{layer}.ffn.gate.weight")[: self.n_experts].to(torch.float32)

    def _tid2eid(self, layer: int) -> torch.Tensor:
        if layer < M.num_hash_layers:
            table = self.ckpt.get(f"layers.{layer}.ffn.gate.tid2eid").to(torch.int32)
            if self.n_experts < N_EXPERTS_FULL:
                table = table % self.n_experts  # remap into the reduced expert space
            return table
        return torch.zeros(VOCAB, M.num_experts_per_tok, dtype=torch.int32)  # unused slots

    # ---- heads -----------------------------------------------------------
    def _lm_head(self) -> torch.Tensor:
        head = self.ckpt.get("head.weight")
        vocab_per_tp = VOCAB // self.tp
        shards = [head[s * vocab_per_tp:(s + 1) * vocab_per_tp].clone() for s in range(self.tp)]
        return torch.stack([shards[r % self.tp] for r in range(self.ep)], dim=0)

    # ---- dispatch --------------------------------------------------------
    def convert(self, name: str) -> torch.Tensor:
        """Return the full ``[n_ranks, ...]`` host tensor for spec ``name``."""
        stashed = self._stash.pop(name, None)
        if stashed is not None:
            return stashed
        ckpt, rep, fwd = self.ckpt, _replicate, self._fwd_stack
        csa = lambda fn: self._kind_stack(CSA_LAYERS, fn)  # noqa: E731 — local dispatch shorthand
        hca = lambda fn: self._kind_stack(HCA_LAYERS, fn)  # noqa: E731
        match name:
            # ---- per-FWD-layer stacked attention weights ----
            case "hc_attn_fn" | "hc_attn_scale" | "hc_attn_base" | "hc_ffn_fn" | "hc_ffn_scale" | "hc_ffn_base":
                return rep(fwd(lambda l: self._raw(f"layers.{l}.{name}")), self.ep)
            case "attn_norm_w":
                return rep(fwd(lambda l: self._raw(f"layers.{l}.attn_norm.weight")), self.ep)
            case "wq_a" | "wkv":
                return self._stacked_mx_pair(
                    f"{name}_scale", list(range(NUM_LAYERS)), lambda l: self._attention_linear(l, name))
            case "wq_b":
                return self._stacked_mx_pair("wq_b_scale", list(range(NUM_LAYERS)), self._wq_b)
            case "wo_b":
                return self._stacked_mx_pair("wo_b_scale", list(range(NUM_LAYERS)), self._wo_b)
            case "wq_a_scale" | "wq_b_scale" | "wkv_scale" | "wo_a_scale" | "wo_b_scale":
                self.convert(name.removesuffix("_scale"))  # populates the stash
                return self._stash.pop(name)
            case "gamma_cq":
                return rep(fwd(lambda l: self._raw(f"layers.{l}.attn.q_norm.weight")), self.ep)
            case "gamma_ckv":
                return rep(fwd(lambda l: self._raw(f"layers.{l}.attn.kv_norm.weight")), self.ep)
            case "attn_sink":
                return rep(fwd(lambda l: self._raw(f"layers.{l}.attn.attn_sink")), self.ep)
            case "wo_a":
                return self._stacked_mx_pair("wo_a_scale", list(range(NUM_LAYERS)), self._wo_a)
            # ---- per-FWD-layer stacked MoE weights ----
            case "norm_w":
                return rep(fwd(lambda l: self._raw(f"layers.{l}.ffn_norm.weight")), self.ep)
            case "gate_w":
                return rep(fwd(self._gate_w), self.ep)
            case "gate_bias":
                return rep(fwd(self._gate_bias), self.ep)
            case "tid2eid":
                return rep(fwd(self._tid2eid), self.ep)
            case "routed_w1" | "routed_w2" | "routed_w3":
                return self._routed(f"{name}_scale", name.removeprefix("routed_"))
            case "routed_w1_scale" | "routed_w2_scale" | "routed_w3_scale":
                self.convert(name.removesuffix("_scale"))
                return self._stash.pop(name)
            case "shared_w1" | "shared_w2" | "shared_w3":
                which = name.removeprefix("shared_")
                return self._stacked_mx_pair(
                    f"{name}_scale", list(range(NUM_LAYERS)), lambda l: self._shared(l, which))
            case "shared_w1_scale" | "shared_w2_scale" | "shared_w3_scale":
                self.convert(name.removesuffix("_scale"))
                return self._stash.pop(name)
            # ---- CSA-compact stacks (slot order = ascending ratio-4 layer id) ----
            case "csa_cmp_wkv" | "csa_cmp_wgate":
                part = name.removeprefix("csa_cmp_")
                return rep(csa(lambda l: self._raw(f"layers.{l}.attn.compressor.{part}.weight")), self.ep)
            case "csa_cmp_ape":
                return rep(csa(lambda l: self._raw(f"layers.{l}.attn.compressor.ape")), self.ep)
            case "csa_cmp_norm_w":
                return rep(csa(lambda l: self._raw(f"layers.{l}.attn.compressor.norm.weight")), self.ep)
            case "csa_idx_wq_b":
                return self._stacked_mx_pair("csa_idx_wq_b_scale", CSA_LAYERS, self._idx_wq_b)
            case "csa_idx_wq_b_scale":
                self.convert("csa_idx_wq_b")
                return self._stash.pop(name)
            case "csa_weights_proj":
                return rep(csa(
                    lambda l: ckpt.get(f"layers.{l}.attn.indexer.weights_proj.weight").t().contiguous()), self.ep)
            case "csa_inner_wkv" | "csa_inner_wgate":
                part = name.removeprefix("csa_inner_")
                return rep(csa(lambda l: self._raw(f"layers.{l}.attn.indexer.compressor.{part}.weight")), self.ep)
            case "csa_inner_ape":
                return rep(csa(lambda l: self._raw(f"layers.{l}.attn.indexer.compressor.ape")), self.ep)
            case "csa_inner_norm_w":
                return rep(csa(lambda l: self._raw(f"layers.{l}.attn.indexer.compressor.norm.weight")), self.ep)
            # ---- HCA-compact stacks (slot order = ascending ratio-128 layer id) ----
            case "hca_cmp_wkv" | "hca_cmp_wgate":
                part = name.removeprefix("hca_cmp_")
                return rep(hca(lambda l: self._raw(f"layers.{l}.attn.compressor.{part}.weight")), self.ep)
            case "hca_cmp_ape":
                return rep(hca(lambda l: self._raw(f"layers.{l}.attn.compressor.ape")), self.ep)
            case "hca_cmp_norm_w":
                return rep(hca(lambda l: self._raw(f"layers.{l}.attn.compressor.norm.weight")), self.ep)
            # ---- replicated head / embedding weights ----
            case "hc_head_fn" | "hc_head_scale" | "hc_head_base":
                return rep(self._raw(name), self.ep)
            case "final_norm_w":
                return rep(self._raw("norm.weight"), self.ep)
            case "embed_weight":
                return rep(self._raw("embed.weight"), self.ep)
            case "lm_head_weight":
                return self._lm_head()
        raise KeyError(f"{name!r} is not a real-weight spec name; expected one of {sorted(REAL_WEIGHT_NAMES)}")

    def convert_layer(self, layer_id: int) -> dict[str, torch.Tensor]:
        """Single-layer real weights keyed by the layer-driver spec names.

        Shapes carry the ``[ep, ...]`` rank dim but no layer stacking — the
        layout ``decode_layer.py`` / ``prefill_layer.py`` consume. Only the
        layer's own kind contributes CSA/HCA entries; the inactive kind keeps
        its fixture. Synthesized inputs (``csa_hadamard_idx``, RoPE tables,
        caches, metadata) are never included.
        """
        if not 0 <= layer_id < NUM_LAYERS:
            raise ValueError(f"layer_id must be in [0, {NUM_LAYERS}), got {layer_id}")
        lyr = layer_id

        def rep(t: torch.Tensor) -> torch.Tensor:
            return _replicate(t, self.ep)

        out: dict[str, torch.Tensor] = {}
        for name in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base",
                     "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base"):
            out[name] = rep(self._raw(f"layers.{lyr}.{name}"))
        out["attn_norm_w"] = rep(self._raw(f"layers.{lyr}.attn_norm.weight"))
        out["norm_w"] = rep(self._raw(f"layers.{lyr}.ffn_norm.weight"))
        out["gamma_cq"] = rep(self._raw(f"layers.{lyr}.attn.q_norm.weight"))
        out["gamma_ckv"] = rep(self._raw(f"layers.{lyr}.attn.kv_norm.weight"))
        out["attn_sink"] = rep(self._raw(f"layers.{lyr}.attn.attn_sink"))
        for name in ("wq_a", "wkv"):
            w, s = self._attention_linear(lyr, name)
            out[name], out[f"{name}_scale"] = rep(w), rep(s)
        w, s = self._wo_a(lyr)
        out["wo_a"], out["wo_a_scale"] = rep(w), rep(s)
        w, s = self._wq_b(lyr)
        out["wq_b"], out["wq_b_scale"] = rep(w), rep(s)
        w, s = self._wo_b(lyr)
        out["wo_b"], out["wo_b_scale"] = rep(w), rep(s)
        out["gate_w"] = rep(self._gate_w(lyr))
        out["gate_bias"] = rep(self._gate_bias(lyr))
        out["tid2eid"] = rep(self._tid2eid(lyr))
        for which in ("w1", "w2", "w3"):
            weight_tiles, w_scale = self._routed_layer(lyr, which)
            out[f"routed_{which}"] = weight_tiles
            logical_scale = unpack_b_scale_batched(w_scale.contiguous().view(torch.uint8))
            flat_scale = logical_scale.flatten(1, 2)
            out[f"routed_{which}_scale"] = pack_b_scale_batched(flat_scale).view(
                torch.float8_e8m0fnu
            )
            w, s = self._shared(lyr, which)
            out[f"shared_{which}"], out[f"shared_{which}_scale"] = rep(w), rep(s)
        if lyr in CSA_LAYERS:
            out["csa_cmp_wkv"] = rep(self._raw(f"layers.{lyr}.attn.compressor.wkv.weight"))
            out["csa_cmp_wgate"] = rep(self._raw(f"layers.{lyr}.attn.compressor.wgate.weight"))
            out["csa_cmp_ape"] = rep(self._raw(f"layers.{lyr}.attn.compressor.ape"))
            out["csa_cmp_norm_w"] = rep(self._raw(f"layers.{lyr}.attn.compressor.norm.weight"))
            w, s = self._idx_wq_b(lyr)
            out["csa_idx_wq_b"], out["csa_idx_wq_b_scale"] = rep(w), rep(s)
            out["csa_weights_proj"] = rep(
                self.ckpt.get(f"layers.{lyr}.attn.indexer.weights_proj.weight").t().contiguous())
            out["csa_inner_wkv"] = rep(self._raw(f"layers.{lyr}.attn.indexer.compressor.wkv.weight"))
            out["csa_inner_wgate"] = rep(self._raw(f"layers.{lyr}.attn.indexer.compressor.wgate.weight"))
            out["csa_inner_ape"] = rep(self._raw(f"layers.{lyr}.attn.indexer.compressor.ape"))
            out["csa_inner_norm_w"] = rep(self._raw(f"layers.{lyr}.attn.indexer.compressor.norm.weight"))
        if lyr in HCA_LAYERS:
            out["hca_cmp_wkv"] = rep(self._raw(f"layers.{lyr}.attn.compressor.wkv.weight"))
            out["hca_cmp_wgate"] = rep(self._raw(f"layers.{lyr}.attn.compressor.wgate.weight"))
            out["hca_cmp_ape"] = rep(self._raw(f"layers.{lyr}.attn.compressor.ape"))
            out["hca_cmp_norm_w"] = rep(self._raw(f"layers.{lyr}.attn.compressor.norm.weight"))
        return out


def apply_real_layer_weights(specs: list, ckpt_dir: str | Path, *, layer_id: int, ep: int) -> int:
    """Point a layer driver's weight specs at one layer of the real checkpoint.

    For ``decode_layer.py`` / ``prefill_layer.py``: single-layer shapes, no
    stacking. ``ckpt_dir`` must be the HF checkpoint directory (per-layer
    conversion is cheap, no cache needed). Returns the number of specs rewired.
    """
    if M.name != "flash":
        raise ValueError(f"real-weight loading supports the flash variant only, got {M.name!r}")
    converter = FlashWeightConverter(FlashCheckpoint(ckpt_dir), ep=ep, tp=1)
    tensors = converter.convert_layer(layer_id)
    by_name = {getattr(s, "name", None): s for s in specs}
    missing = set(tensors) - set(by_name)
    if missing:
        raise ValueError(f"layer specs are missing expected real-weight names: {sorted(missing)}")
    count = 0
    for name, value in tensors.items():
        spec = by_name[name]
        if list(value.shape) != list(spec.shape) or value.dtype != spec.dtype:
            raise ValueError(
                f"{name}: converted layer weight {tuple(value.shape)}/{value.dtype} does not match "
                f"spec {tuple(spec.shape)}/{spec.dtype} (check --ep)"
            )
        spec.init_value = value
        count += 1
    return count


# Every spec name loaded from the checkpoint, weight before its scale so the
# scale conversion hits the converter's stash.
REAL_WEIGHT_NAMES = (
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_a_scale", "wq_b", "wq_b_scale", "wkv", "wkv_scale", "gamma_cq", "gamma_ckv",
    "attn_sink", "wo_a", "wo_a_scale", "wo_b", "wo_b_scale",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
    "shared_w2", "shared_w2_scale",
    "csa_cmp_wkv", "csa_cmp_wgate", "csa_cmp_ape", "csa_cmp_norm_w",
    "csa_idx_wq_b", "csa_idx_wq_b_scale", "csa_weights_proj",
    "csa_inner_wkv", "csa_inner_wgate", "csa_inner_ape", "csa_inner_norm_w",
    "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape", "hca_cmp_norm_w",
    "hc_head_fn", "hc_head_scale", "hc_head_base", "final_norm_w",
    "embed_weight", "lm_head_weight",
)


def _flash_cache_identity(*, ep: int, tp: int) -> dict:
    return {
        "schema": FLASH_CACHE_SCHEMA,
        "source": _flash_source_identity(),
        "reference": {
            "repository": "https://github.com/Ascend/cann-recipes-infer",
            "revision": CANN_WEIGHT_REFERENCE_REVISION,
            "converter": "models/deepseek_v4/utils/mx_quantize.py",
            "dense_weight_quantization": "group32_e4m3_round2decimal_nearest_satfinite",
            "scale_layout": "MX_B_NN_independent_layer_and_output_group",
            "routed_weight_quantization": "group32_e2m1_round2decimal_nearest_satfinite",
        },
        "variant": M.name,
        "ep": ep,
        "tp": tp,
        "num_layers": NUM_LAYERS,
        "csa_layers": CSA_LAYERS,
        "hca_layers": HCA_LAYERS,
        "n_routed_experts": N_EXPERTS_FULL // 8 * ep,
        "weight_names": list(REAL_WEIGHT_NAMES),
    }


def _write_json_atomic(path: Path, content: dict) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(content, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _check_cache_identity(manifest: dict, *, ep: int, tp: int) -> None:
    expected = _flash_cache_identity(ep=ep, tp=tp)
    if manifest.get("identity") != expected:
        raise ValueError("weight cache has a stale schema, source, quantization, or EP/TP configuration; reconvert it")


def _check_cache_file(directory: Path, name: str, entry: dict, *, digest: bool = True) -> Path:
    path = directory / f"{name}.pt"
    if not path.is_file() or path.stat().st_size != entry.get("file_bytes"):
        raise ValueError(f"weight cache is incomplete or truncated: {path}")
    if digest and _file_sha256(path) != entry.get("sha256"):
        raise ValueError(f"weight cache checksum failed: {path}")
    return path


def _load_complete_cache(directory: Path, *, ep: int, tp: int) -> dict:
    path = directory / "manifest.json"
    if not path.is_file():
        raise ValueError(f"weight cache has no completed manifest: {directory}; finish or rerun conversion")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _check_cache_identity(manifest, ep=ep, tp=tp)
    if manifest.get("complete") is not True or set(manifest.get("tensors", {})) != set(REAL_WEIGHT_NAMES):
        raise ValueError(f"weight cache is partial: {directory}; finish conversion before inference")
    for name, entry in manifest["tensors"].items():
        _check_cache_file(directory, name, entry, digest=False)
    return manifest


# ---------------------------------------------------------------------------
# Driver integration: swap the fixture init_value of every real-weight spec.
# ---------------------------------------------------------------------------
def apply_real_weights(specs: list, weights_dir: str | Path, *, ep: int, tp: int) -> int:
    """Point every real-weight ``TensorSpec`` in ``specs`` at the checkpoint.

    ``weights_dir`` is either the HF checkpoint directory (detected by
    ``model.safetensors.index.json``; converts on the fly) or a cache
    directory of per-name ``.pt`` files produced by this module's CLI.
    Returns the number of specs rewired.  Raises on a shape/dtype mismatch
    between the converted tensor and the spec, and on a missing cache file.
    """
    if M.name != "flash":
        raise ValueError(f"real-weight loading supports the flash variant only, got {M.name!r}")
    weights_dir = Path(weights_dir)
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"--weights dir not found: {weights_dir}")
    converter = None
    cache_manifest = None
    if (weights_dir / "model.safetensors.index.json").is_file():
        converter = FlashWeightConverter(FlashCheckpoint(weights_dir), ep=ep, tp=tp)
    else:
        cache_manifest = _load_complete_cache(weights_dir, ep=ep, tp=tp)

    def make_init(spec):
        def init() -> torch.Tensor:
            if converter is not None:
                value = converter.convert(spec.name)
            else:
                entry = cache_manifest["tensors"][spec.name]
                path = _check_cache_file(weights_dir, spec.name, entry)
                value = torch.load(path, weights_only=True, mmap=True)
                if list(value.shape) != entry.get("shape") or str(value.dtype) != entry.get("dtype"):
                    raise ValueError(f"weight cache tensor metadata mismatch: {path}")
            if list(value.shape) != list(spec.shape) or value.dtype != spec.dtype:
                raise ValueError(
                    f"{spec.name}: converted weight {tuple(value.shape)}/{value.dtype} does not match "
                    f"spec {tuple(spec.shape)}/{spec.dtype} (check --ep/--tp used for conversion)"
                )
            return value

        return init

    count = 0
    for spec in specs:
        if getattr(spec, "name", None) in REAL_WEIGHT_NAMES:
            spec.init_value = make_init(spec)
            count += 1
    missing = set(REAL_WEIGHT_NAMES) - {getattr(s, "name", None) for s in specs}
    if missing:
        raise ValueError(f"specs are missing expected real-weight names: {sorted(missing)}")
    return count


# ---------------------------------------------------------------------------
# CLI: offline conversion into a per-name .pt cache.
# ---------------------------------------------------------------------------
def main() -> None:
    # Import the shape-freezing modules first: they consume --variant / --ep /
    # --tp from sys.argv exactly like the drivers, keeping one source of truth
    # for the EP/TP world shape.
    import lm_head
    import moe

    parser = argparse.ArgumentParser(description="Convert DeepSeek-V4-Flash weights to the kernel ABI.")
    parser.add_argument("--ckpt", type=str, required=True, help="HF checkpoint dir (with model.safetensors.index.json)")
    parser.add_argument("--out", type=str, required=True, help="output cache dir for per-name .pt files")
    parser.add_argument("--only", type=str, nargs="*", default=None, help="convert only these spec names")
    parser.add_argument("--force", action="store_true", help="overwrite existing .pt files")
    # Consumed at import time (config strips --variant; moe/lm_head peek --ep/--tp);
    # declared here like the drivers so argparse accepts and documents them.
    parser.add_argument("--variant", choices=("pro", "flash"), default=M.name,
                        help="Architecture preset selected before module import.")
    parser.add_argument("--ep", type=int, default=moe.N_RANKS, choices=[2, 4, 8],
                        help="EP world size (parsed at import by moe).")
    parser.add_argument("--tp", type=int, default=lm_head.TP_SIZE, choices=[2, 4, 8, 16],
                        help="LM-head TP group size (parsed at import by lm_head).")
    args = parser.parse_args()

    if M.name != "flash":
        raise SystemExit(f"pass --variant flash (or DEEPSEEK_V4_VARIANT=flash); active variant is {M.name!r}")
    ep, tp = moe.N_RANKS, lm_head.TP_SIZE

    converter = FlashWeightConverter(FlashCheckpoint(args.ckpt), ep=ep, tp=tp)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(args.only) if args.only else list(REAL_WEIGHT_NAMES)
    unknown = [n for n in names if n not in REAL_WEIGHT_NAMES]
    if unknown:
        raise SystemExit(f"unknown spec names {unknown}; expected among {sorted(REAL_WEIGHT_NAMES)}")

    state_path = out_dir / "conversion-state.json"
    complete_path = out_dir / "manifest.json"
    state = {"identity": _flash_cache_identity(ep=ep, tp=tp), "complete": False, "tensors": {}}
    if state_path.is_file() and not args.force:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        _check_cache_identity(state, ep=ep, tp=tp)
    elif not args.force and any(out_dir.glob("*.pt")):
        raise SystemExit("existing weight files have no matching conversion state; use --force to replace them")
    # A reader must never observe a completed manifest while files are being replaced.
    complete_path.unlink(missing_ok=True)
    state["complete"] = False
    _write_json_atomic(state_path, state)

    print(f"[CONVERT] variant={M.name} ep={ep} tp={tp} ckpt={args.ckpt} out={out_dir}", flush=True)
    for i, name in enumerate(names):
        path = out_dir / f"{name}.pt"
        entry = state["tensors"].get(name)
        if entry is not None and not args.force:
            _check_cache_file(out_dir, name, entry)
            print(f"[CONVERT] ({i + 1}/{len(names)}) {name}: verified, skipped", flush=True)
            continue
        value = converter.convert(name)
        temporary = path.with_name(path.name + ".incomplete")
        with temporary.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        entry = {
            "shape": list(value.shape), "dtype": str(value.dtype),
            "file_bytes": temporary.stat().st_size, "sha256": _file_sha256(temporary),
        }
        os.replace(temporary, path)
        state["tensors"][name] = entry
        _write_json_atomic(state_path, state)
        size_gb = value.numel() * value.element_size() / 1024**3
        print(f"[CONVERT] ({i + 1}/{len(names)}) {name}: {tuple(value.shape)} {value.dtype} {size_gb:.2f} GiB",
              flush=True)
        del value
    missing = set(REAL_WEIGHT_NAMES) - set(state["tensors"])
    if missing:
        print(f"[CONVERT] partial cache: {len(missing)} tensors remain; rerun without --only", flush=True)
    else:
        for name, entry in state["tensors"].items():
            _check_cache_file(out_dir, name, entry, digest=False)
        state["complete"] = True
        _write_json_atomic(state_path, state)
        _write_json_atomic(complete_path, state)
        print(f"[CONVERT] complete manifest published: {out_dir}", flush=True)


# ===========================================================================
# Full-network torch golden (packed-prefill forward)
# ===========================================================================
# ---------------------------------------------------------------------------
# Layer schedule, mirrored from prefill_fwd: per-layer attention kind from the
# preset's compress ratios, plus each kind-compact stack's slot order
# (ascending layer id within the kind).
# ---------------------------------------------------------------------------
_KIND_BY_RATIO = {0: "swa", 4: "csa", 128: "hca"}
_ROPE_PROFILE_BY_KIND = {"swa": 0, "csa": 1, "hca": 1}
FWD_NUM_LAYERS = MODEL_CONFIG.num_hidden_layers
FWD_COMPRESS_RATIOS = MODEL_CONFIG.compress_ratios[:FWD_NUM_LAYERS]
LAYER_KINDS = tuple(_KIND_BY_RATIO[ratio] for ratio in FWD_COMPRESS_RATIOS)
CSA_NUM_LAYERS = LAYER_KINDS.count("csa")
HCA_NUM_LAYERS = LAYER_KINDS.count("hca")


def _kind_orders():
    orders = {}
    counters = {"swa": 0, "csa": 0, "hca": 0}
    for layer, kind in enumerate(LAYER_KINDS):
        orders[layer] = counters[kind]
        counters[kind] += 1
    return orders


KIND_ORDER = _kind_orders()


def _rope_profile_for_kind(stacked, kind):
    try:
        profile = _ROPE_PROFILE_BY_KIND[kind]
    except KeyError as exc:
        raise ValueError(f"unsupported DeepSeek V4 attention kind {kind!r}") from exc
    return stacked[profile]


def _layer_rows(stacked, count, index):
    """Packed-layer slice *index* along dim 1 of a rank-stacked tensor.

    Returns a torch view (never a copy) so leaf-golden in-place cache writes
    propagate into the stacked fwd tensor.
    """
    unit = stacked.shape[1] // count
    return stacked[:, index * unit:(index + 1) * unit]


# ---------------------------------------------------------------------------
# Per-layer view dicts: invert build_single_layer_tensor_specs' name mapping
# between the fwd namespace (csa_cmp_wkv, hca_cmp_ape, ...) and each leaf
# golden's own namespace (cmp_wkv, cmp_ape, ...).
# ---------------------------------------------------------------------------
def _base_attention_views(tensors, rank, layer, x_hc, x_out, num_tokens):
    """Keys shared by all three attention-kind goldens, for one rank/layer."""

    def fwd(name):
        return _layer_rows(tensors[name], FWD_NUM_LAYERS, layer)[rank]

    kind = LAYER_KINDS[layer]
    return {
        "num_tokens": num_tokens,
        "x_hc": x_hc,
        "hc_attn_fn": fwd("hc_attn_fn"),
        "hc_attn_scale": fwd("hc_attn_scale"),
        "hc_attn_base": fwd("hc_attn_base"),
        "attn_norm_w": fwd("attn_norm_w"),
        "wq_a": fwd("wq_a"),
        "wq_a_scale": fwd("wq_a_scale"),
        "wq_b": fwd("wq_b"),
        "wq_b_scale": fwd("wq_b_scale"),
        "wkv": fwd("wkv"),
        "wkv_scale": fwd("wkv_scale"),
        "gamma_cq": fwd("gamma_cq"),
        "gamma_ckv": fwd("gamma_ckv"),
        "freqs_cos": _rope_profile_for_kind(tensors["freqs_cos"][rank], kind),
        "freqs_sin": _rope_profile_for_kind(tensors["freqs_sin"][rank], kind),
        "position_ids": tensors["position_ids"][rank],
        "kv_cache": fwd("kv_cache"),
        "ori_slot_mapping": tensors["ori_slot_mapping"][rank],
        "attn_sink": fwd("attn_sink"),
        "wo_a": fwd("wo_a"),
        "wo_a_scale": fwd("wo_a_scale"),
        "wo_b": fwd("wo_b"),
        "wo_b_scale": fwd("wo_b_scale"),
        "x_out": x_out,
    }


def _swa_attention_views(tensors, rank, layer, x_hc, x_out, num_tokens):
    views = _base_attention_views(tensors, rank, layer, x_hc, x_out, num_tokens)
    # The SWA golden names the original-KV table plainly "block_table".
    views["block_table"] = tensors["ori_block_table"][rank]
    return views


def _hca_attention_views(tensors, rank, layer, x_hc, x_out, num_tokens):
    views = _base_attention_views(tensors, rank, layer, x_hc, x_out, num_tokens)
    order = KIND_ORDER[layer]

    def hca(name):
        return _layer_rows(tensors[name], HCA_NUM_LAYERS, order)[rank]

    views.update({
        "ori_block_table": tensors["ori_block_table"][rank],
        "cmp_kv": _layer_rows(tensors["cmp_kv"], FWD_NUM_LAYERS, layer)[rank],
        "cmp_block_table": tensors["cmp_block_table"][rank],
        "cmp_wkv": hca("hca_cmp_wkv"),
        "cmp_wgate": hca("hca_cmp_wgate"),
        "cmp_ape": hca("hca_cmp_ape"),
        "cmp_norm_w": hca("hca_cmp_norm_w"),
        "compress_state": hca("hca_compress_state"),
        "compress_state_block_table": tensors["hca_compress_state_block_table"][rank],
        "cmp_slot_mapping": tensors["hca_cmp_slot_mapping"][rank],
        "state_slot_mapping": tensors["hca_state_slot_mapping"][rank],
    })
    return views


def _csa_attention_views(tensors, rank, layer, x_hc, x_out, num_tokens):
    views = _base_attention_views(tensors, rank, layer, x_hc, x_out, num_tokens)
    order = KIND_ORDER[layer]

    def csa(name):
        return _layer_rows(tensors[name], CSA_NUM_LAYERS, order)[rank]

    views.update({
        "ori_block_table": tensors["ori_block_table"][rank],
        "cmp_kv": _layer_rows(tensors["cmp_kv"], FWD_NUM_LAYERS, layer)[rank],
        "cmp_block_table": tensors["cmp_block_table"][rank],
        "cmp_wkv": csa("csa_cmp_wkv"),
        "cmp_wgate": csa("csa_cmp_wgate"),
        "cmp_ape": csa("csa_cmp_ape"),
        "cmp_norm_w": csa("csa_cmp_norm_w"),
        "compress_state": csa("csa_compress_state"),
        "compress_state_block_table": tensors["csa_compress_state_block_table"][rank],
        "hadamard_idx": csa("csa_hadamard_idx"),
        "idx_wq_b": csa("csa_idx_wq_b"),
        "idx_wq_b_scale": csa("csa_idx_wq_b_scale"),
        "idx_weights_proj": csa("csa_weights_proj"),
        "inner_wkv": csa("csa_inner_wkv"),
        "inner_wgate": csa("csa_inner_wgate"),
        "inner_ape": csa("csa_inner_ape"),
        "inner_norm_w": csa("csa_inner_norm_w"),
        "inner_compress_state": csa("csa_inner_compress_state"),
        "inner_compress_state_block_table": tensors["csa_inner_compress_state_block_table"][rank],
        "idx_kv_cache": csa("idx_kv_cache"),
        "idx_kv_scale": csa("idx_kv_scale"),
        "idx_block_table": tensors["idx_block_table"][rank],
        "cmp_slot_mapping": tensors["csa_cmp_slot_mapping"][rank],
        "idx_slot_mapping": tensors["csa_idx_slot_mapping"][rank],
        "state_slot_mapping": tensors["csa_state_slot_mapping"][rank],
        "inner_state_slot_mapping": tensors["csa_inner_state_slot_mapping"][rank],
    })
    return views


_ATTENTION_VIEWS = {
    "swa": _swa_attention_views,
    "hca": _hca_attention_views,
    "csa": _csa_attention_views,
}

_MOE_LAYER_STACKED = (
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
    "shared_w2", "shared_w2_scale",
)


def _moe_views(tensors, layer, x_hc, x_next, num_tokens):
    """golden_moe consumes the full rank stack (dispatch/combine cross ranks)."""
    views = {
        name: _layer_rows(tensors[name], FWD_NUM_LAYERS, layer)
        for name in _MOE_LAYER_STACKED
    }
    views.update({
        "x_hc": x_hc,
        "mxfp4_pair_lut": tensors["mxfp4_pair_lut"],
        "input_ids": tensors["input_ids"],
        "layer_id": layer,
        "num_tokens": num_tokens,
        "x_next": x_next,
    })
    return views




def golden_prefill_fwd(tensors):
    """Fill every output tensor of prefill_fwd's spec list in place."""
    import sys

    if "moe" not in sys.modules:
        config.MOE_TOKENS = config.PREFILL_TOKENS
    from moe import D, HC_MULT, N_RANKS, T, golden_moe

    from hc_head import golden_hc_head_rows
    from input_pack import golden_pack_x_hc
    from lm_head import golden_lm_head_all_ranks
    from prefill_attention_csa import golden_prefill_attention_csa
    from prefill_attention_hca import golden_prefill_attention_hca
    from prefill_attention_swa import golden_prefill_attention_swa
    from rmsnorm import golden_rms_norm

    attention_golden = {
        "swa": golden_prefill_attention_swa,
        "hca": golden_prefill_attention_hca,
        "csa": golden_prefill_attention_csa,
    }

    num_tokens = int(tensors["num_tokens"])

    # Layer-0 hidden state: embedding lookup replicated across the HC lanes.
    x_hc = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.float32)
    for rank in range(N_RANKS):
        golden_pack_x_hc({
            "input_ids": tensors["input_ids"][rank],
            "embed_weight": tensors["embed_weight"][rank],
            "x_hc": x_hc[rank],
        })

    for layer, kind in enumerate(LAYER_KINDS):
        attn_out = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.float32)
        for rank in range(N_RANKS):
            views = _ATTENTION_VIEWS[kind](
                tensors, rank, layer, x_hc[rank], attn_out[rank], num_tokens,
            )
            attention_golden[kind](views)
        # The trailing layer's MoE writes the fwd's pre-hc hidden output.
        if layer == FWD_NUM_LAYERS - 1:
            x_next = tensors["pre_hc_hidden_out"]
        else:
            x_next = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.float32)
        golden_moe(_moe_views(tensors, layer, attn_out, x_next, num_tokens))
        x_hc = x_next

    # hc_head -> final rms_norm per rank, matching the tail of prefill_fwd.
    x_head = torch.zeros(N_RANKS, T, D, dtype=torch.bfloat16)
    for rank in range(N_RANKS):
        golden_hc_head_rows({
            "x_hc": tensors["pre_hc_hidden_out"][rank],
            "hc_head_fn": tensors["hc_head_fn"][rank],
            "hc_head_scale": tensors["hc_head_scale"][rank],
            "hc_head_base": tensors["hc_head_base"][rank],
            "y": x_head[rank],
        })
        tensors["hidden_out"][rank].copy_(
            golden_rms_norm(x_head[rank], tensors["final_norm_w"][rank])
        )

    golden_lm_head_all_ranks(tensors, n_ranks=N_RANKS)


# ---------------------------------------------------------------------------
# --validate comparators. 43 chained layers accumulate error well past the
# leaf kernels' per-point bars, so the hidden/logits comparisons accept
# direction/magnitude agreement (bounded outlier ratio, cosine + relative-L2)
# instead of strict per-element closeness.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Full-net comparators. Model-specific by design (stacked pool layout, prompt
# prefix semantics, lm_head sampling): they live with this model rather than in
# the shared golden package, composing the public ratio_allclose primitive.
# ---------------------------------------------------------------------------
def _stacked_mapped_nonfinite_coordinates(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    valid_mapping: torch.Tensor,
    valid_mapping_items: torch.Tensor,
    feature_shape: tuple[int, ...],
    rank: int,
    layer_index: int,
    blocks_per_layer: int,
    block_size: int,
    max_show: int,
) -> str:
    """Format logical and physical coordinates for mapped non-finite values."""
    nonfinite = ~torch.isfinite(actual) | ~torch.isfinite(expected)
    coordinates = nonfinite.nonzero(as_tuple=False)
    total = int(coordinates.shape[0])
    if total == 0:
        return ""

    show_count = min(max(max_show, 0), total)
    lines = [f"    non-finite mapped coordinate(s): showing {show_count}/{total}"]
    for coordinate in coordinates[:show_count]:
        mapped_offset = int(coordinate[0].item())
        flat_feature = int(coordinate[1].item())
        remaining = flat_feature
        feature_coordinates = []
        for size in reversed(feature_shape):
            feature_coordinates.append(remaining % size)
            remaining //= size
        feature_coordinates.reverse()

        mapped_row = int(valid_mapping_items[mapped_offset].item())
        physical_row = int(valid_mapping[mapped_offset].item())
        pool_block = layer_index * blocks_per_layer + physical_row // block_size
        block_row = physical_row % block_size
        logical_coordinate = (mapped_row, *feature_coordinates)
        physical_coordinate = (rank, pool_block, block_row, *feature_coordinates)
        actual_value = actual[mapped_offset, flat_feature].item()
        expected_value = expected[mapped_offset, flat_feature].item()
        lines.append(
            "      "
            f"logical(mapped_row, feature...)={logical_coordinate} "
            "physical_pool(rank, block, block_row, feature...)="
            f"{physical_coordinate} layer_physical_row={physical_row} "
            f"actual={actual_value!r} expected={expected_value!r}"
        )
    if total > show_count:
        lines.append(
            f"      ... and {total - show_count} more non-finite coordinate(s)"
        )
    return "\n".join(lines)


def sampled_ids_golden_compare(
    *,
    logits_name: str = "logits",
    row_indices_name: str = "logit_row_indices",
    sampled_id_column: int = 0,
    max_show: int = 20,
) -> Callable:
    """Validate sampled IDs against golden and the device's own logits.

    This combines two independent contracts on every selected logit row:

    * semantic correctness — the complete sampled-ID tensor equals golden;
    * sampler self-consistency — the sampled token equals ``argmax`` of the
      actual device logits.

    Checking only the second condition can report PASS for a confidently
    wrong model token, so full-model validation should use both.
    """

    def compare(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        actual_outputs: dict[str, torch.Tensor],
        expected_outputs: dict[str, torch.Tensor],
        inputs: dict[str, torch.Tensor],
        rtol: float,
        atol: float,
    ) -> tuple[bool, str]:
        del expected_outputs, rtol, atol
        if actual.shape != expected.shape:
            return False, (
                f"    sampled_ids shape mismatch: actual={tuple(actual.shape)} "
                f"expected={tuple(expected.shape)}"
            )
        if actual.ndim < 1 or not 0 <= sampled_id_column < actual.shape[-1]:
            return False, (
                f"    sampled_id_column={sampled_id_column} out of range for "
                f"shape {tuple(actual.shape)}"
            )
        if logits_name not in actual_outputs:
            return False, (
                f"    compare_fn misconfigured: missing actual output '{logits_name}'"
            )
        if row_indices_name not in inputs:
            return False, (
                f"    compare_fn misconfigured: missing input '{row_indices_name}'"
            )

        logits = actual_outputs[logits_name].cpu()
        row_indices = inputs[row_indices_name].cpu()
        leading_shape = actual.shape[:-1]
        if tuple(logits.shape[:-1]) != tuple(leading_shape):
            return False, (
                f"    '{logits_name}' leading shape must be {tuple(leading_shape)}, "
                f"got {tuple(logits.shape)}"
            )
        if tuple(row_indices.shape) != tuple(leading_shape):
            return False, (
                f"    '{row_indices_name}' shape must be {tuple(leading_shape)}, "
                f"got {tuple(row_indices.shape)}"
            )

        actual_cpu = actual.cpu()
        expected_cpu = expected.cpu()
        valid = row_indices >= 0
        device_argmax = torch.argmax(logits, dim=-1).to(actual_cpu.dtype)
        actual_ids = actual_cpu[..., sampled_id_column]
        expected_ids = expected_cpu[..., sampled_id_column]
        semantic_bad = actual_cpu != expected_cpu
        self_bad = valid & (actual_ids != device_argmax)
        failures: list[str] = []

        if semantic_bad.any().item():
            failures.append(
                f"    sampled_ids differ from golden: "
                f"bad={int(semantic_bad.count_nonzero().item())}/{actual_cpu.numel()}"
            )
        if self_bad.any().item():
            failures.append(
                f"    sampled_ids disagree with argmax(actual {logits_name}): "
                f"bad={int(self_bad.count_nonzero().item())}/{int(valid.count_nonzero().item())}"
            )

        interesting = valid & (
            (actual_ids != expected_ids) | (actual_ids != device_argmax)
        )
        coords = interesting.nonzero(as_tuple=False)
        for coord in coords[:max_show]:
            index = tuple(int(value.item()) for value in coord)
            failures.append(
                f"      row={index} actual={int(actual_ids[index].item())} "
                f"golden={int(expected_ids[index].item())} "
                f"device_argmax={int(device_argmax[index].item())}"
            )
        if coords.shape[0] > max_show:
            failures.append(f"      ... and {coords.shape[0] - max_show} more")
        return (not failures), "\n".join(failures)

    compare.__name__ = "sampled_ids_golden_compare"
    return compare


def input_prefix_ratio_allclose(
    valid_rows_name: str,
    *,
    valid_axis: int = 0,
    exact_tail: bool = True,
    atol: float | None = None,
    rtol: float | None = None,
    max_error_ratio: float = 0.005,
    max_show: int = 10,
) -> Callable:
    """Compare an active prefix whose row count comes from a scalar input.

    Unlike :func:`ratio_allclose`'s static ``valid_rows`` option, this helper
    resolves the prefix length from ``inputs[valid_rows_name]`` for every
    validation call.  This is important for golden-data replay: cached scalar
    inputs override the initializer used to rebuild the specs.

    When ``exact_tail`` is true, rows outside the active prefix must remain
    exactly equal to the golden output.  The inactive rows therefore cannot
    dilute the active-region error ratio or hide a stray write.
    """
    from golden import ratio_allclose

    prefix_compare = ratio_allclose(
        atol=atol,
        rtol=rtol,
        max_error_ratio=max_error_ratio,
        max_show=max_show,
    )

    def compare(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        actual_outputs: dict[str, torch.Tensor],
        expected_outputs: dict[str, torch.Tensor],
        inputs: dict[str, torch.Tensor],
        rtol: float,
        atol: float,
    ) -> tuple[bool, str]:
        if actual.shape != expected.shape:
            return False, (
                f"    shape mismatch: actual={tuple(actual.shape)} "
                f"expected={tuple(expected.shape)}"
            )
        if valid_rows_name not in inputs:
            return False, (
                f"    compare_fn misconfigured: missing input '{valid_rows_name}'"
            )

        ndim = actual.ndim
        axis = valid_axis if valid_axis >= 0 else valid_axis + ndim
        if not 0 <= axis < ndim:
            return False, (
                f"    valid_axis={valid_axis} out of range for shape {tuple(actual.shape)}"
            )

        valid_rows_value = inputs[valid_rows_name].cpu()
        if valid_rows_value.numel() != 1:
            return False, (
                f"    '{valid_rows_name}' must be a scalar, "
                f"got shape {tuple(valid_rows_value.shape)}"
            )
        valid_rows = int(valid_rows_value.item())
        total_rows = actual.shape[axis]
        if not 0 <= valid_rows <= total_rows:
            return False, (
                f"    {valid_rows_name}={valid_rows} out of range for axis "
                f"{axis} of length {total_rows}"
            )

        if exact_tail and valid_rows < total_rows:
            actual_tail = actual.narrow(axis, valid_rows, total_rows - valid_rows)
            expected_tail = expected.narrow(axis, valid_rows, total_rows - valid_rows)
            unequal_tail = actual_tail != expected_tail
            if unequal_tail.any().item():
                return False, (
                    f"    inactive tail differs from golden: "
                    f"changed_values={int(unequal_tail.count_nonzero().item())} "
                    f"axis={axis} active_rows={valid_rows}"
                )

        actual_prefix = actual.narrow(axis, 0, valid_rows)
        expected_prefix = expected.narrow(axis, 0, valid_rows)
        return prefix_compare(
            actual_prefix,
            expected_prefix,
            actual_outputs=actual_outputs,
            expected_outputs=expected_outputs,
            inputs=inputs,
            rtol=rtol,
            atol=atol,
        )

    compare.__name__ = (
        f"input_prefix_ratio_allclose(valid_rows_name={valid_rows_name}, "
        f"valid_axis={valid_axis}, exact_tail={exact_tail}, atol={atol}, "
        f"rtol={rtol}, max_error_ratio={max_error_ratio})"
    )
    return compare


def stacked_mapped_pool_ratio_allclose(
    layer_mapping_names: tuple[str | None, ...],
    *,
    mapping_shape: tuple[int, int],
    block_size: int,
    active_rows_name: str,
    layer_labels: tuple[int, ...] | None = None,
    pool_name: str = "pool",
    atol: float | None = None,
    rtol: float | None = None,
    max_error_ratio: float = 0.005,
    max_show: int = 3,
) -> Callable:
    """Compare active mapped rows in a rank- and layer-stacked pool.

    The pool layout is ``[ranks, layers * blocks, block_size, ...]``.  Each
    entry in ``layer_mapping_names`` selects the rank-local mapping input for
    that layer; ``None`` means the layer must not write this pool.  Only the
    leading ``inputs[active_rows_name]`` mapping entries participate, which
    makes replay honor the cached active-token scalar.

    Mapped values use a ratio-based numerical comparison independently for
    every layer and rank.  All other physical rows must remain exactly equal
    to golden.  Failure diagnostics therefore identify the first bad logical
    layer/rank instead of reporting a ratio diluted by the unused pool.
    """
    from golden import ratio_allclose

    if not layer_mapping_names:
        raise ValueError("layer_mapping_names must not be empty")
    if len(mapping_shape) != 2 or any(dim <= 0 for dim in mapping_shape):
        raise ValueError(
            f"mapping_shape must be (ranks, mapped_items), got {mapping_shape}"
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if layer_labels is None:
        layer_labels = tuple(range(len(layer_mapping_names)))
    if len(layer_labels) != len(layer_mapping_names):
        raise ValueError(
            "layer_labels and layer_mapping_names must have the same length, "
            f"got {len(layer_labels)} and {len(layer_mapping_names)}"
        )

    mapped_compare = ratio_allclose(
        atol=atol,
        rtol=rtol,
        max_error_ratio=max_error_ratio,
        max_show=max_show,
    )
    integer_dtypes = (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    )

    def compare(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        actual_outputs: dict[str, torch.Tensor],
        expected_outputs: dict[str, torch.Tensor],
        inputs: dict[str, torch.Tensor],
        rtol: float,
        atol: float,
    ) -> tuple[bool, str]:
        if actual.shape != expected.shape:
            return False, (
                f"    {pool_name} shape mismatch: actual={tuple(actual.shape)} "
                f"expected={tuple(expected.shape)}"
            )
        rank_count, mapped_items = mapping_shape
        layer_count = len(layer_mapping_names)
        if actual.ndim < 4 or actual.shape[0] != rank_count:
            return False, (
                f"    expected {pool_name} layout "
                "[ranks, layers * blocks, block_size, ...] with "
                f"ranks={rank_count}, got {tuple(actual.shape)}"
            )
        if actual.shape[1] % layer_count != 0 or actual.shape[2] != block_size:
            return False, (
                f"    expected {pool_name} layout "
                "[ranks, layers * blocks, block_size, ...] with "
                f"layers={layer_count} block_size={block_size}, "
                f"got {tuple(actual.shape)}"
            )
        if active_rows_name not in inputs:
            return False, (
                f"    compare_fn misconfigured: missing input '{active_rows_name}'"
            )
        active_value = inputs[active_rows_name].cpu()
        if active_value.numel() != 1:
            return False, (
                f"    '{active_rows_name}' must be a scalar, "
                f"got shape {tuple(active_value.shape)}"
            )
        active_rows = int(active_value.item())
        if not 0 <= active_rows <= mapped_items:
            return False, (
                f"    {active_rows_name}={active_rows} out of range for "
                f"mapping length {mapped_items}"
            )

        mapping_cache: dict[str, torch.Tensor] = {}
        for mapping_name in set(layer_mapping_names) - {None}:
            mapping = inputs.get(mapping_name)
            if mapping is None:
                return False, (
                    f"    compare_fn misconfigured: missing input '{mapping_name}'"
                )
            if mapping.dtype not in integer_dtypes:
                return False, (
                    f"    '{mapping_name}' must have an integer dtype, "
                    f"got {mapping.dtype}"
                )
            if tuple(mapping.shape) != mapping_shape:
                return False, (
                    f"    '{mapping_name}' must have shape {mapping_shape}, "
                    f"got {tuple(mapping.shape)}"
                )
            mapping_cache[mapping_name] = mapping.cpu().to(torch.int64)

        blocks_per_layer = actual.shape[1] // layer_count
        rows_per_layer = blocks_per_layer * block_size
        actual_rows = actual.cpu().reshape(
            rank_count, layer_count, rows_per_layer, -1
        )
        expected_rows = expected.cpu().reshape(
            rank_count, layer_count, rows_per_layer, -1
        )
        if actual_rows.dtype == torch.float8_e4m3fn:
            # CPU float8 does not implement the indexing/isfinite primitives
            # below. FP32 represents each finite FP8 value exactly, including
            # the values checked outside the active mapped rows.
            actual_rows, expected_rows = actual_rows.float(), expected_rows.float()
        failures: list[str] = []

        for layer_index, (layer_label, mapping_name) in enumerate(
            zip(layer_labels, layer_mapping_names)
        ):
            for rank in range(rank_count):
                if mapping_name is None:
                    mapping = torch.empty(0, dtype=torch.int64)
                else:
                    mapping = mapping_cache[mapping_name][rank, :active_rows]

                invalid_negative = mapping < -1
                if invalid_negative.any().item():
                    item = int(invalid_negative.nonzero(as_tuple=False)[0, 0].item())
                    failures.append(
                        f"    layer={layer_label} rank={rank} mapping='{mapping_name}' "
                        f"item={item} value={int(mapping[item].item())}: "
                        "only -1 is a negative sentinel"
                    )
                    continue
                valid_mapping_items = (mapping >= 0).nonzero(as_tuple=False).flatten()
                valid_mapping = mapping[valid_mapping_items]
                if (valid_mapping >= rows_per_layer).any().item():
                    value = int(valid_mapping[valid_mapping >= rows_per_layer][0].item())
                    failures.append(
                        f"    layer={layer_label} rank={rank} mapping='{mapping_name}' "
                        f"row={value} outside physical row range "
                        f"[0, {rows_per_layer})"
                    )
                    continue
                if valid_mapping.numel() > 1:
                    unique_rows, counts = torch.unique(
                        valid_mapping, return_counts=True
                    )
                    duplicates = counts > 1
                    if duplicates.any().item():
                        duplicate_row = int(unique_rows[duplicates][0].item())
                        failures.append(
                            f"    layer={layer_label} rank={rank} "
                            f"mapping='{mapping_name}' contains duplicate "
                            f"physical row {duplicate_row}"
                        )
                        continue

                written_rows = torch.zeros(rows_per_layer, dtype=torch.bool)
                written_rows[valid_mapping] = True
                actual_layer = actual_rows[rank, layer_index]
                expected_layer = expected_rows[rank, layer_index]
                stray_values = actual_layer[~written_rows] != expected_layer[~written_rows]
                if stray_values.any().item():
                    changed_rows = (
                        (actual_layer[~written_rows] != expected_layer[~written_rows])
                        .any(dim=-1)
                        .count_nonzero()
                        .item()
                    )
                    failures.append(
                        f"    layer={layer_label} rank={rank} mapping='{mapping_name}' "
                        f"has writes outside active mapped rows: "
                        f"changed_rows={int(changed_rows)} "
                        f"changed_values={int(stray_values.count_nonzero().item())}"
                    )
                    continue
                if valid_mapping.numel() == 0:
                    continue

                mapped_actual = actual_layer[valid_mapping]
                mapped_expected = expected_layer[valid_mapping]
                ok, detail = mapped_compare(
                    mapped_actual,
                    mapped_expected,
                    actual_outputs=actual_outputs,
                    expected_outputs=expected_outputs,
                    inputs=inputs,
                    rtol=rtol,
                    atol=atol,
                )
                if not ok:
                    nonfinite_coordinates = _stacked_mapped_nonfinite_coordinates(
                        mapped_actual,
                        mapped_expected,
                        valid_mapping=valid_mapping,
                        valid_mapping_items=valid_mapping_items,
                        feature_shape=tuple(actual.shape[3:]),
                        rank=rank,
                        layer_index=layer_index,
                        blocks_per_layer=blocks_per_layer,
                        block_size=block_size,
                        max_show=max_show,
                    )
                    coordinate_detail = (
                        f"\n{nonfinite_coordinates}" if nonfinite_coordinates else ""
                    )
                    failures.append(
                        f"    layer={layer_label} rank={rank} "
                        f"mapping='{mapping_name}' mapped_rows={valid_mapping.numel()}\n"
                        f"{detail}{coordinate_detail}"
                    )

        return (not failures), "\n".join(failures)

    compare.__name__ = (
        f"stacked_mapped_pool_ratio_allclose(pool={pool_name}, "
        f"layers={len(layer_mapping_names)}, block_size={block_size}, "
        f"active_rows_name={active_rows_name}, atol={atol}, rtol={rtol}, "
        f"max_error_ratio={max_error_ratio})"
    )
    return compare


def _logits_cosine_compare(min_cosine=0.99, max_rel_l2=0.10):
    """Cosine + relative-L2 acceptance on the rows logit_row_indices selects."""

    def cmp(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        import torch

        del actual_outputs, expected_outputs, rtol, atol
        if actual.shape != expected.shape:
            return False, (
                f"    logits shape mismatch: actual={tuple(actual.shape)} "
                f"expected={tuple(expected.shape)}"
            )
        row_indices = inputs["logit_row_indices"].cpu()
        a = actual.cpu().float()
        e = expected.cpu().float()
        failures = []
        for rank in range(a.shape[0]):
            for row in range(a.shape[1]):
                if int(row_indices[rank, row]) < 0:
                    continue
                a_row = a[rank, row]
                e_row = e[rank, row]
                if not bool(torch.isfinite(a_row).all()):
                    failures.append(f"    logits[{rank},{row}]: non-finite actual values")
                    continue
                if not bool(torch.isfinite(e_row).all()):
                    # A NaN golden row would turn cosine and rel_l2 into NaN,
                    # and every NaN comparison below reads as a pass.
                    failures.append(f"    logits[{rank},{row}]: non-finite expected values")
                    continue
                denom = float(a_row.norm() * e_row.norm())
                cosine = float(a_row @ e_row) / denom if denom > 0.0 else 0.0
                rel_l2 = float((a_row - e_row).norm() / e_row.norm().clamp_min(1e-12))
                if cosine < min_cosine or rel_l2 > max_rel_l2:
                    failures.append(
                        f"    logits[{rank},{row}]: cosine={cosine:.6f} "
                        f"(min {min_cosine}) rel_l2={rel_l2:.6f} (max {max_rel_l2})"
                    )
        return (not failures), "\n".join(failures)

    cmp.__name__ = "logits_cosine_compare"
    return cmp


def build_validate_compare_fn(num_tokens):
    """Per-output comparators for prefill_fwd --validate.

    Hidden states resolve their active prefix from the run input, so replayed
    scalar data cannot be diluted by the spec initializer. Logits use
    cosine/relative-L2 acceptance on the selected rows, while sampled IDs must
    match both golden and the device-logit argmax. Layer-stacked pools compare
    only active mapped rows, independently per layer/rank, and require every
    other physical row to remain exactly equal to golden.
    """
    import sys

    if "moe" not in sys.modules:
        config.MOE_TOKENS = config.PREFILL_TOKENS
    from moe import N_RANKS, T

    from prefill_attention_csa import CSA_STATE_BLOCK_SIZE, INNER_STATE_BLOCK_SIZE
    from prefill_attention_hca import HCA_STATE_BLOCK_SIZE

    # Keep the public builder signature used by prefill_fwd. The comparator
    # intentionally reads the effective value from inputs["num_tokens"] at
    # validation time because golden-data replay overrides this initializer.
    del num_tokens
    hidden_cmp = input_prefix_ratio_allclose(
        "num_tokens",
        atol=1e-2, rtol=5e-2, max_error_ratio=0.02,
        valid_axis=1, exact_tail=True,
    )
    mapping_shape = (N_RANKS, T)
    all_layer_labels = tuple(range(FWD_NUM_LAYERS))
    csa_layer_labels = tuple(
        layer for layer, kind in enumerate(LAYER_KINDS) if kind == "csa"
    )
    hca_layer_labels = tuple(
        layer for layer, kind in enumerate(LAYER_KINDS) if kind == "hca"
    )

    def stacked_pool(
        mapping_names,
        *,
        layer_labels,
        block_size,
        pool_name,
        atol,
        rtol,
        max_error_ratio,
    ):
        return stacked_mapped_pool_ratio_allclose(
            tuple(mapping_names),
            mapping_shape=mapping_shape,
            block_size=block_size,
            active_rows_name="num_tokens",
            layer_labels=tuple(layer_labels),
            pool_name=pool_name,
            atol=atol,
            rtol=rtol,
            max_error_ratio=max_error_ratio,
        )

    return {
        "pre_hc_hidden_out": hidden_cmp,
        "hidden_out": hidden_cmp,
        "logits": _logits_cosine_compare(),
        "sampled_ids": sampled_ids_golden_compare(),
        "kv_cache": stacked_pool(
            ("ori_slot_mapping",) * FWD_NUM_LAYERS,
            layer_labels=all_layer_labels,
            block_size=config.BLOCK_SIZE,
            pool_name="kv_cache",
            atol=1e-3,
            rtol=3e-2,
            max_error_ratio=0.01,
        ),
        "cmp_kv": stacked_pool(
            (
                None if kind == "swa" else f"{kind}_cmp_slot_mapping"
                for kind in LAYER_KINDS
            ),
            layer_labels=all_layer_labels,
            block_size=config.BLOCK_SIZE,
            pool_name="cmp_kv",
            atol=1e-3,
            rtol=3e-2,
            max_error_ratio=0.01,
        ),
        "hca_compress_state": stacked_pool(
            ("hca_state_slot_mapping",) * HCA_NUM_LAYERS,
            layer_labels=hca_layer_labels,
            block_size=HCA_STATE_BLOCK_SIZE,
            pool_name="hca_compress_state",
            atol=5e-3,
            rtol=2e-2,
            max_error_ratio=0.01,
        ),
        "csa_compress_state": stacked_pool(
            ("csa_state_slot_mapping",) * CSA_NUM_LAYERS,
            layer_labels=csa_layer_labels,
            block_size=CSA_STATE_BLOCK_SIZE,
            pool_name="csa_compress_state",
            atol=5e-3,
            rtol=2e-2,
            max_error_ratio=0.01,
        ),
        "csa_inner_compress_state": stacked_pool(
            ("csa_inner_state_slot_mapping",) * CSA_NUM_LAYERS,
            layer_labels=csa_layer_labels,
            block_size=INNER_STATE_BLOCK_SIZE,
            pool_name="csa_inner_compress_state",
            atol=5e-3,
            rtol=2e-2,
            max_error_ratio=0.01,
        ),
        # FP8 index cache: allow one mantissa step on a bounded fraction.
        "idx_kv_cache": stacked_pool(
            ("csa_idx_slot_mapping",) * CSA_NUM_LAYERS,
            layer_labels=csa_layer_labels,
            block_size=config.BLOCK_SIZE,
            pool_name="idx_kv_cache",
            atol=0.001953125,
            rtol=0.125,
            max_error_ratio=0.02,
        ),
        "idx_kv_scale": stacked_pool(
            ("csa_idx_slot_mapping",) * CSA_NUM_LAYERS,
            layer_labels=csa_layer_labels,
            block_size=config.BLOCK_SIZE,
            pool_name="idx_kv_scale",
            atol=1e-3,
            rtol=1e-2,
            max_error_ratio=0.01,
        ),
    }
