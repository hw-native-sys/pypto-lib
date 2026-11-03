# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""从上游 pypto-lib `models/deepseek_v4_flash_dspark/hc_post.py` 移植。

mHC 的收尾：y[t, out_h] = post[t, out_h] * x[t] + sum_in_h comb[t, in_h, out_h] * residual[t, in_h]。
其中只有 `post * x` 这一项依赖 attention 的输出，comb 与 residual 的组合
完全可以和 attention 并行——这正是把它搬进 PTO kernel 的理由。

与上游一样按 dtype 说明：上游 hc 残差流是 FP32 端到端，这里改成与
vllm-ascend Native 一致的 BF16 收发，在行一级 cast 成 FP32 参与计算。
prefill 入口与 standalone 用例不搬，本包只做 decode。
"""

import pypto.language as pl

from ..common.config import DECODE_BATCH, DECODE_SEQ, TP
from ..common.config import FLASH as M

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S

# model config
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
assert HC_MULT == 4

# tiling
T_TILE = 4
INACTIVE_FILL_T_TILE = 16
INACTIVE_FILL_D_TILE = 256
assert (DECODE_BATCH // TP * DECODE_SEQ) % T_TILE == 0


@pl.jit.inline
def hc_post_block(
    x_block: pl.Tensor,
    residual_flat: pl.Tensor,
    post: pl.Tensor,
    comb: pl.Tensor,
    y_flat: pl.Out[pl.Tensor],
    t0: pl.Scalar[pl.INDEX],
    n0: pl.Scalar[pl.INDEX],
    valid_rows: pl.Scalar[pl.INDEX],
    ROWS: pl.constexpr,
    COLS: pl.constexpr,
):
    # Both callers use ROWS=1: scalar gates apply to this single token row.
    # x_block is already BF16: retain the rounding boundary before HC math.
    x_f32 = pl.cast(x_block, pl.FP32)
    res_0 = pl.cast(pl.slice(residual_flat, [ROWS, COLS], [t0, n0],
                            valid_shape=[valid_rows, COLS], clamp=True), pl.FP32)
    res_1 = pl.cast(pl.slice(residual_flat, [ROWS, COLS], [t0, D + n0],
                            valid_shape=[valid_rows, COLS], clamp=True), pl.FP32)
    res_2 = pl.cast(pl.slice(residual_flat, [ROWS, COLS], [t0, 2 * D + n0],
                            valid_shape=[valid_rows, COLS], clamp=True), pl.FP32)
    res_3 = pl.cast(pl.slice(residual_flat, [ROWS, COLS], [t0, 3 * D + n0],
                            valid_shape=[valid_rows, COLS], clamp=True), pl.FP32)
    for out_h in pl.unroll(HC_MULT):
        post_scalar = pl.read(post, [t0, out_h])
        single_y = pl.mul(x_f32, post_scalar)
        comb_scalar_0 = pl.read(comb, [t0, out_h])
        single_weighted_0 = pl.mul(res_0, comb_scalar_0)
        single_y = pl.add(single_y, single_weighted_0)
        comb_scalar_1 = pl.read(comb, [t0, HC_MULT + out_h])
        single_weighted_1 = pl.mul(res_1, comb_scalar_1)
        single_y = pl.add(single_y, single_weighted_1)
        comb_scalar_2 = pl.read(comb, [t0, 2 * HC_MULT + out_h])
        single_weighted_2 = pl.mul(res_2, comb_scalar_2)
        single_y = pl.add(single_y, single_weighted_2)
        comb_scalar_3 = pl.read(comb, [t0, 3 * HC_MULT + out_h])
        single_weighted_3 = pl.mul(res_3, comb_scalar_3)
        single_y = pl.add(single_y, single_weighted_3)
        single_result = pl.cast(single_y, pl.BF16, mode="rint")
        y_flat[t0 : t0 + ROWS, out_h * D + n0 : out_h * D + n0 + COLS] = pl.set_validshape(
            single_result, valid_rows, COLS)
    return y_flat


def _hc_post(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    comb.bind_dynamic(0, T_DYN)
    y.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(x, 0)

    residual_flat = pl.reshape(residual, [t_dim, HC_DIM])
    y_flat = pl.reshape(y, [t_dim, HC_DIM])

    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    for token_block in pl.spmd(token_tiles, name_hint="hc_post"):
        t0 = token_block * T_TILE
        for t in pl.pipeline(t0, t0 + T_TILE, stage=2):
            if t < t_dim:
                x_block = pl.slice(x, [1, D], [t, 0])
                y_flat = hc_post_block(x_block, residual_flat, post, comb, y_flat, t, 0, 1, 1, D)
    return y


hc_post = pl.jit.inline(_hc_post)
hc_post_test = pl.jit(_hc_post)
