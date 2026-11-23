# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl

from .config import FLASH as M, DECODE_BATCH, DECODE_SEQ, TP, PREFILL_BATCH, PREFILL_SEQ

T_DYN = pl.dynamic("NATIVE_D_T_DYN")  # T = B * S

D = M.hidden_size

HC_MULT = M.hc_mult

HC_DIM = M.hc_dim

T_TILE = 4

INACTIVE_FILL_T_TILE = 16

INACTIVE_FILL_D_TILE = 256

assert (DECODE_BATCH // TP * DECODE_SEQ) % T_TILE == 0

assert (PREFILL_BATCH * PREFILL_SEQ) % T_TILE == 0


@pl.jit.inline
def hc_post_prefill(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Compute prefill active rows and deterministically pad the static tail."""
    t_dim = pl.tensor.dim(x, 0)
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > t_dim:
        active_tokens = t_dim

    residual_flat = pl.reshape(residual, [t_dim, HC_DIM])
    y_flat = pl.reshape(y, [t_dim, HC_DIM])
    active_tiles = (active_tokens + T_TILE - 1) // T_TILE

    if active_tokens > 0:
        for block in pl.spmd(active_tiles * HC_MULT, name_hint="hc_post_prefill"):
            token_block = block // HC_MULT
            out_h = block % HC_MULT
            t0 = token_block * T_TILE
            for t in pl.pipeline(t0, t0 + T_TILE, stage=2):
                if t < active_tokens:
                    post_w = pl.read(post, [t, out_h])
                    x_row = pl.cast(x[t : t + 1, 0:D], target_type=pl.FP32)
                    y_row = pl.mul(x_row, post_w)
                    for in_h in pl.pipeline(HC_MULT, stage=4):
                        comb_w = pl.read(comb, [t, in_h * HC_MULT + out_h])
                        res_d = in_h * D
                        res_row = residual_flat[t : t + 1, res_d : res_d + D]
                        y_row = pl.add(y_row, pl.mul(res_row, comb_w))
                    y_flat[t : t + 1, out_h * D : out_h * D + D] = y_row

    inactive_tokens = t_dim - active_tokens
    if inactive_tokens > 0:
        inactive_tiles = (inactive_tokens + INACTIVE_FILL_T_TILE - 1) // INACTIVE_FILL_T_TILE
        for fill_block in pl.spmd(inactive_tiles * HC_MULT, name_hint="hc_post_inactive_pad"):
            fill_tile = fill_block // HC_MULT
            out_h = fill_block % HC_MULT
            fill_t0 = active_tokens + fill_tile * INACTIVE_FILL_T_TILE
            zero = pl.full([1, INACTIVE_FILL_D_TILE], dtype=pl.FP32, value=0.0)
            for fill_dt in pl.range(INACTIVE_FILL_T_TILE):
                fill_t = fill_t0 + fill_dt
                if fill_t < t_dim:
                    for fill_d0 in pl.range(0, D, INACTIVE_FILL_D_TILE):
                        out_d0 = out_h * D + fill_d0
                        y_flat[fill_t : fill_t + 1, out_d0 : out_d0 + INACTIVE_FILL_D_TILE] = zero
    return y
