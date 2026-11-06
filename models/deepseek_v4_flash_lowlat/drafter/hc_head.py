# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl

from .config import FLASH as M

T_DYN = pl.dynamic("NATIVE_D_T_DYN")  # T = B * S

D = M.hidden_size

HC_MULT = M.hc_mult

HC_DIM = M.hc_dim

EPS = M.rms_norm_eps

HC_EPS = M.hc_eps

HC_DIM_INV = 1.0 / HC_DIM

HC_PAD = 16

HC_VPAD = 8

T_TILE = 16

LINEAR_T_TILE = 16

RMS_T_TILE = 64

RMS_K_TILE = 256

LINEAR_K_TILE = 256

D_TILE = 256

D_SPMD = 2048

RMS_OK = 8

ROW_ALIGN = max(T_TILE, LINEAR_T_TILE, RMS_T_TILE)


def _hc_head(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    x_hc.bind_dynamic(0, T_DYN)
    y.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(x_hc, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    rms_tiles = (t_dim + RMS_T_TILE - 1) // RMS_T_TILE
    linear_full_rows = (t_dim // LINEAR_T_TILE) * LINEAR_T_TILE
    t_linear = ((t_dim + ROW_ALIGN - 1) // ROW_ALIGN) * ROW_ALIGN
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    y_flat = pl.reshape(y, [t_dim, D])
    # rms: split-K sum-of-squares, fanned over (rms-tile x K-slice). RMS_T_TILE is
    # taller than the gate's T_TILE so the fan-out stays inside the vector array
    # instead of queueing several waves of short tasks.
    sq_part = pl.create_tensor([RMS_OK, t_linear], dtype=pl.FP32)
    for task in pl.spmd(rms_tiles * RMS_OK, name_hint="hc_head_rms"):
        t0 = (task // RMS_OK) * RMS_T_TILE
        ok = task % RMS_OK
        k_base = ok * (HC_DIM // RMS_OK)
        valid_rows = pl.min(RMS_T_TILE, t_dim - t0)
        sq_sum = pl.full([1, RMS_T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HC_DIM // RMS_OK // RMS_K_TILE, stage=4):
            k0 = k_base + kb * RMS_K_TILE
            if valid_rows == RMS_T_TILE:
                x_rms_full = x_flat[t0 : t0 + RMS_T_TILE, k0 : k0 + RMS_K_TILE]
                sq_col_full = pl.row_sum(pl.mul(x_rms_full, x_rms_full))
                sq_sum = pl.add(sq_sum, pl.reshape(sq_col_full, [1, RMS_T_TILE]))
            else:
                x_rms_tail = pl.slice(
                    x_flat,
                    [RMS_T_TILE, RMS_K_TILE],
                    [t0, k0],
                    valid_shape=[valid_rows, RMS_K_TILE],
                )
                sq_col_tail = pl.row_sum(pl.mul(x_rms_tail, x_rms_tail))
                sq_sum = pl.add(sq_sum, pl.reshape(sq_col_tail, [1, RMS_T_TILE]))
        sq_part = pl.assemble(sq_part, sq_sum, [ok, t0])

    # One writer per row block computes the complete head projection.
    mixes_raw = pl.create_tensor([t_linear, HC_PAD], dtype=pl.FP32)
    if linear_full_rows > 0:
        with pl.spmd(
            linear_full_rows // LINEAR_T_TILE,
            name_hint="hc_head_linear",
        ) as _linear_tid:
            # Weight reads bypass L2.
            pl.set_cache_policy(hc_head_fn, pl.CachePolicy.BYPASS)
            task = pl.tile.get_block_idx()
            t0 = task * LINEAR_T_TILE
            acc_full = pl.create_tensor([LINEAR_T_TILE, HC_PAD], dtype=pl.FP32)
            for kb in pl.pipeline(0, HC_DIM // LINEAR_K_TILE, stage=2):
                k0 = kb * LINEAR_K_TILE
                x_lin_full = x_flat[t0 : t0 + LINEAR_T_TILE, k0 : k0 + LINEAR_K_TILE]
                w_full = pl.slice(
                    hc_head_fn,
                    [HC_PAD, LINEAR_K_TILE],
                    [0, k0],
                    valid_shape=[HC_MULT, LINEAR_K_TILE],
                )
                acc_full = pl.matmul_acc(acc_full, x_lin_full, w_full, b_trans=True, init_cond=(kb == 0))
            mixes_raw = pl.assemble(mixes_raw, acc_full, [t0, 0])

    # At most one incomplete M=16 block exists. Keep it in a separate conditional
    # task so every aligned block above retains the original static-M Cube path.
    if linear_full_rows < t_dim:
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="hc_head_linear_tail",
        ):
            pl.set_cache_policy(hc_head_fn, pl.CachePolicy.BYPASS)
            tail_rows = t_dim - linear_full_rows
            acc_tail = pl.create_tensor([LINEAR_T_TILE, HC_PAD], dtype=pl.FP32)
            for kb in pl.pipeline(0, HC_DIM // LINEAR_K_TILE, stage=2):
                k0 = kb * LINEAR_K_TILE
                x_lin_tail = pl.slice(
                    x_flat,
                    [LINEAR_T_TILE, LINEAR_K_TILE],
                    [linear_full_rows, k0],
                    valid_shape=[tail_rows, LINEAR_K_TILE],
                )
                w_tail = pl.slice(
                    hc_head_fn,
                    [HC_PAD, LINEAR_K_TILE],
                    [0, k0],
                    valid_shape=[HC_MULT, LINEAR_K_TILE],
                )
                acc_tail = pl.matmul_acc(acc_tail, x_lin_tail, w_tail, b_trans=True, init_cond=(kb == 0))
            mixes_raw = pl.assemble(
                mixes_raw,
                acc_tail,
                [linear_full_rows, 0],
            )

    # reduce: gate + hc mix, fanned over (token-tile x D-slice). The rsqrt/sigmoid gate is
    # recomputed per task instead of being published by its own scope.
    for blk in pl.spmd(token_tiles * (D // D_SPMD), name_hint="hc_head_reduce"):
        t0 = (blk // (D // D_SPMD)) * T_TILE
        d_base = (blk % (D // D_SPMD)) * D_SPMD
        valid_rows = pl.min(T_TILE, t_dim - t0)
        scale = pl.read(hc_head_scale, [0])
        base = pl.reshape(pl.slice(hc_head_base, [HC_VPAD], [0], valid_shape=[HC_MULT]), [1, HC_VPAD])
        # sq_part is [RMS_OK, T], so the per-token total is a column sum; transposing the
        # slab makes it one row_sum landing in the [T_TILE, 1] shape row_expand_mul wants
        ssq_slab = pl.set_validshape(sq_part[0:RMS_OK, t0 : t0 + T_TILE], RMS_OK, T_TILE)
        ssq = pl.row_sum(pl.transpose(ssq_slab, axis1=0, axis2=1))
        inv_row = pl.rsqrt(pl.add(pl.mul(ssq, HC_DIM_INV), EPS), high_precision=True)
        inv_col = pl.reshape(inv_row, [T_TILE, 1])
        mix = mixes_raw[t0 : t0 + T_TILE, 0:HC_VPAD]
        scaled = pl.mul(pl.row_expand_mul(mix, inv_col), scale)
        logits = pl.add(scaled, pl.col_expand(scaled, base))
        pre_val = pl.add(pl.recip(pl.add(pl.exp(pl.neg(logits)), 1.0)), HC_EPS)
        # pin the valid shape: the gate carries valid=?x? from the dynamic-offset slice
        pre_val = pl.set_validshape(pre_val, T_TILE, HC_VPAD)
        pre_tile_t = pl.transpose(pre_val, axis1=0, axis2=1)
        pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
        for db in pl.pipeline(D_SPMD // D_TILE, stage=2):
            d0 = d_base + db * D_TILE
            if valid_rows == T_TILE:
                x_h0_full = x_flat[t0 : t0 + T_TILE, 0 * D + d0 : 0 * D + d0 + D_TILE]
                x_h1_full = x_flat[t0 : t0 + T_TILE, 1 * D + d0 : 1 * D + d0 + D_TILE]
                x_h2_full = x_flat[t0 : t0 + T_TILE, 2 * D + d0 : 2 * D + d0 + D_TILE]
                x_h3_full = x_flat[t0 : t0 + T_TILE, 3 * D + d0 : 3 * D + d0 + D_TILE]
                y01_full = pl.add(
                    pl.row_expand_mul(x_h0_full, pre0),
                    pl.row_expand_mul(x_h1_full, pre1),
                )
                y23_full = pl.add(
                    pl.row_expand_mul(x_h2_full, pre2),
                    pl.row_expand_mul(x_h3_full, pre3),
                )
                y_tile_full = pl.add(y01_full, y23_full)
                y_flat[t0 : t0 + T_TILE, d0 : d0 + D_TILE] = pl.cast(
                    y_tile_full,
                    target_type=pl.BF16,
                    mode="rint",
                )
            else:
                x_h0_tail = pl.slice(
                    x_flat,
                    [T_TILE, D_TILE],
                    [t0, 0 * D + d0],
                    valid_shape=[valid_rows, D_TILE],
                )
                x_h1_tail = pl.slice(
                    x_flat,
                    [T_TILE, D_TILE],
                    [t0, 1 * D + d0],
                    valid_shape=[valid_rows, D_TILE],
                )
                x_h2_tail = pl.slice(
                    x_flat,
                    [T_TILE, D_TILE],
                    [t0, 2 * D + d0],
                    valid_shape=[valid_rows, D_TILE],
                )
                x_h3_tail = pl.slice(
                    x_flat,
                    [T_TILE, D_TILE],
                    [t0, 3 * D + d0],
                    valid_shape=[valid_rows, D_TILE],
                )
                y01_tail = pl.add(
                    pl.row_expand_mul(x_h0_tail, pre0),
                    pl.row_expand_mul(x_h1_tail, pre1),
                )
                y23_tail = pl.add(
                    pl.row_expand_mul(x_h2_tail, pre2),
                    pl.row_expand_mul(x_h3_tail, pre3),
                )
                y_tile_tail = pl.add(y01_tail, y23_tail)
                y_bf16_tail = pl.cast(y_tile_tail, target_type=pl.BF16, mode="rint")
                y_valid_tail = pl.set_validshape(y_bf16_tail, valid_rows, D_TILE)
                y_flat[t0 : t0 + T_TILE, d0 : d0 + D_TILE] = y_valid_tail

    y = pl.reshape(y_flat, [t_dim, D])
    return y


hc_head = pl.jit.inline(_hc_head)
