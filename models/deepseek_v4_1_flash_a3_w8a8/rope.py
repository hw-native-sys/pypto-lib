# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Interleaved rotary embedding of the trailing adjacent pairs of each head."""

import pypto.language as pl
import torch

from config import FLASH as M


# Dynamic shape variable.

# model config
ROPE_DIM = M.qk_rope_head_dim
ROPE_HALF = ROPE_DIM // 2
# tiling
T_TILE = 8


@pl.jit.inline(auto_scope=False)
def rope_tail_fp32(
    source: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    width: pl.constexpr,
    heads: pl.constexpr,
):
    """Copy FP32 heads into BF16 and rotate their trailing adjacent RoPE pairs."""
    t_dim = pl.tensor.dim(source, 0)
    blocks = (t_dim + T_TILE - 1) // T_TILE
    for block in pl.spmd(blocks, name_hint="rope_tail_fp32"):
        t0 = block * T_TILE
        rows = pl.min(T_TILE, t_dim - t0)
        cosine = pl.load(cos, [t0, 0], [T_TILE, ROPE_HALF], valid_shape=[rows, ROPE_HALF])
        sine = pl.load(sin, [t0, 0], [T_TILE, ROPE_HALF], valid_shape=[rows, ROPE_HALF])
        cosine = pl.set_validshape(pl.fillpad(cosine, pad_value=pl.PadValue.zero), T_TILE, ROPE_HALF)
        sine = pl.set_validshape(pl.fillpad(sine, pad_value=pl.PadValue.zero), T_TILE, ROPE_HALF)
        for head in pl.range(heads):
            head0 = head * width
            nope_dim = width - ROPE_DIM
            nope = pl.load(source, [t0, head0], [T_TILE, nope_dim], valid_shape=[rows, nope_dim])
            nope_out = pl.cast(nope, target_type=pl.BF16, mode="rint")
            nope_valid = pl.set_validshape(nope_out, rows, nope_dim)
            output = pl.store(nope_valid, [t0, head0], output)
            tail = pl.load(source, [t0, head0 + nope_dim], [T_TILE, ROPE_DIM], valid_shape=[rows, ROPE_DIM])
            even = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            real = pl.sub(pl.mul(even, cosine), pl.mul(odd, sine))
            imag = pl.add(pl.mul(even, sine), pl.mul(odd, cosine))
            real_lanes = pl.tile.full([T_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
            real_lanes = pl.tile.scatter_mask(real_lanes, real, mask_pattern=pl.tile.MaskPattern.P0101)
            imag_lanes = pl.tile.full([T_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
            imag_lanes = pl.tile.scatter_mask(imag_lanes, imag, mask_pattern=pl.tile.MaskPattern.P1010)
            real_bits = pl.reinterpret_view(real_lanes, pl.UINT32)
            imag_bits = pl.reinterpret_view(imag_lanes, pl.UINT32)
            rotated_bits = pl.or_(real_bits, imag_bits)
            lanes = pl.reinterpret_view(rotated_bits, pl.FP32)
            rotated = pl.cast(lanes, target_type=pl.BF16, mode="rint")
            tail0 = head0 + nope_dim
            rotated_valid = pl.set_validshape(rotated, rows, ROPE_DIM)
            output = pl.store(rotated_valid, [t0, tail0], output)
    return output


@pl.jit.inline(auto_scope=False)
def rope_tail_bf16(
    source: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    width: pl.constexpr,
    heads: pl.constexpr,
):
    """Copy BF16 heads and rotate their trailing adjacent RoPE pairs."""
    t_dim = pl.tensor.dim(source, 0)
    blocks = (t_dim + T_TILE - 1) // T_TILE
    for block in pl.spmd(blocks, name_hint="rope_tail_bf16"):
        t0 = block * T_TILE
        rows = pl.min(T_TILE, t_dim - t0)
        cosine = pl.load(cos, [t0, 0], [T_TILE, ROPE_HALF], valid_shape=[rows, ROPE_HALF])
        sine = pl.load(sin, [t0, 0], [T_TILE, ROPE_HALF], valid_shape=[rows, ROPE_HALF])
        cosine = pl.set_validshape(pl.fillpad(cosine, pad_value=pl.PadValue.zero), T_TILE, ROPE_HALF)
        sine = pl.set_validshape(pl.fillpad(sine, pad_value=pl.PadValue.zero), T_TILE, ROPE_HALF)
        for head in pl.range(heads):
            head0 = head * width
            nope_dim = width - ROPE_DIM
            nope = pl.load(source, [t0, head0], [T_TILE, nope_dim], valid_shape=[rows, nope_dim])
            nope_valid = pl.set_validshape(nope, rows, nope_dim)
            output = pl.store(nope_valid, [t0, head0], output)
            tail = pl.load(source, [t0, head0 + nope_dim], [T_TILE, ROPE_DIM], valid_shape=[rows, ROPE_DIM])
            tail_fp32 = pl.cast(tail, target_type=pl.FP32)
            even = pl.tile.gather_mask(tail_fp32, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.tile.gather_mask(tail_fp32, mask_pattern=pl.tile.MaskPattern.P1010)
            real = pl.sub(pl.mul(even, cosine), pl.mul(odd, sine))
            imag = pl.add(pl.mul(even, sine), pl.mul(odd, cosine))
            real_lanes = pl.tile.full([T_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
            real_lanes = pl.tile.scatter_mask(real_lanes, real, mask_pattern=pl.tile.MaskPattern.P0101)
            imag_lanes = pl.tile.full([T_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
            imag_lanes = pl.tile.scatter_mask(imag_lanes, imag, mask_pattern=pl.tile.MaskPattern.P1010)
            real_bits = pl.reinterpret_view(real_lanes, pl.UINT32)
            imag_bits = pl.reinterpret_view(imag_lanes, pl.UINT32)
            rotated_bits = pl.or_(real_bits, imag_bits)
            lanes = pl.reinterpret_view(rotated_bits, pl.FP32)
            rotated = pl.cast(lanes, target_type=pl.BF16, mode="rint")
            tail0 = head0 + nope_dim
            rotated_valid = pl.set_validshape(rotated, rows, ROPE_DIM)
            output = pl.store(rotated_valid, [t0, tail0], output)
    return output


def golden_rope_tail_bf16(source, cos, sin):
    """Rotate trailing adjacent pairs into a new BF16 tensor."""
    value = source.float()
    pairs = value[..., -ROPE_DIM:].unflatten(-1, (-1, 2))
    cosine = cos.float()
    sine = sin.float()
    while cosine.ndim < pairs[..., 0].ndim:
        cosine = cosine.unsqueeze(-2)
        sine = sine.unsqueeze(-2)
    real = pairs[..., 0] * cosine - pairs[..., 1] * sine
    imag = pairs[..., 0] * sine + pairs[..., 1] * cosine
    rotated = torch.stack((real, imag), dim=-1).flatten(-2)
    return torch.cat((value[..., :-ROPE_DIM], rotated), dim=-1).to(torch.bfloat16)
