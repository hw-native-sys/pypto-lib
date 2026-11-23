# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""HC post vector kernel and ordered Torch reference."""

from math import gcd

import pypto.language as pl

from .config import FLASH as M

T_DYN = pl.dynamic("HC_POST_T_DYN")

D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim

TOKEN_TILE = 4
HC_RESIDENT_F32_ELEMENTS = 8192
HC_MAX_D_TILE = 4096


def _hc_d_tile(hidden_size, hc_mult):
    """Bound each residual bank to 32 KiB and align disjoint output writes."""
    if hidden_size <= 0 or hc_mult <= 0:
        raise ValueError("hidden size and HC multiplier must be positive")
    if hidden_size % 16 != 0:
        raise ValueError("HC hidden size must align FP32 rows to 64-byte cache lines")
    column_bound = min(HC_MAX_D_TILE, HC_RESIDENT_F32_ELEMENTS // hc_mult)
    if column_bound < 16:
        raise ValueError("HC residual budget cannot fit one cache line per input row")
    power_two_bound = 1 << (column_bound.bit_length() - 1)
    return gcd(hidden_size, power_two_bound)


HC_D_TILE = _hc_d_tile(D, HC_MULT)
HC_D_BLOCKS = D // HC_D_TILE


def _hc_post(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    x.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    comb.bind_dynamic(0, T_DYN)
    y.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(x, 0)

    y_flat = pl.reshape(y, [t_dim, HC_DIM])

    token_tiles = (t_dim + TOKEN_TILE - 1) // TOKEN_TILE
    for block in pl.spmd(token_tiles * HC_D_BLOCKS, name_hint="hc_post"):
        token_block = block // HC_D_BLOCKS
        d0 = (block % HC_D_BLOCKS) * HC_D_TILE
        t0 = token_block * TOKEN_TILE
        for t in pl.pipeline(t0, t0 + TOKEN_TILE, stage=2):
            if t < t_dim:
                x_tile = pl.load(
                    x, [t, d0], [1, HC_D_TILE], target_memory=pl.MemorySpace.Vec,
                )
                x_row = pl.cast(x_tile, target_type=pl.FP32)
                # One GM load supplies every output's ordered residual sum.
                residual_tile = pl.load(
                    residual, [t, 0, d0], [1, HC_MULT, HC_D_TILE],
                    target_memory=pl.MemorySpace.Vec,
                )
                residual_bank = pl.reshape(residual_tile, [HC_MULT, HC_D_TILE])
                for out_h in pl.unroll(HC_MULT):
                    post_w = pl.read(post, [t, out_h])
                    y_row = pl.mul(x_row, post_w)
                    for in_h in pl.unroll(HC_MULT):
                        comb_w = pl.read(comb, [t, in_h * HC_MULT + out_h])
                        res_row = pl.tile.slice(residual_bank, [1, HC_D_TILE], [in_h, 0])
                        y_row = pl.add(y_row, pl.mul(res_row, comb_w))
                    out_d0 = out_h * D + d0
                    y_flat = pl.store(y_row, [t, out_d0], y_flat)
    return y


hc_post = pl.jit.inline(_hc_post)


def golden_hc_post(tensors):
    """Torch reference, direct port of model.py Block.hc_post 684-687."""
    import torch

    x        = tensors["x"].float()
    residual = tensors["residual"].float()
    post     = tensors["post"].float()
    comb     = tensors["comb"].float().reshape(-1, HC_MULT, HC_MULT)  # [B, S, HC, HC]

    T = x.shape[0]
    y_fp32 = torch.zeros(T, HC_MULT, D, dtype=torch.float32)
    for out_h in range(HC_MULT):
        y_row = x * post[:, out_h:out_h + 1]
        for in_h in range(HC_MULT):
            y_row = y_row + residual[:, in_h, :] * comb[:, in_h, out_h:out_h + 1]
        y_fp32[:, out_h, :] = y_row
    y = y_fp32  # hc residual stream is FP32 end-to-end: no BF16 rounding

    tensors["y"][:] = y
