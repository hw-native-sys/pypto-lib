# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-dep-gen
"""DeepSeek-V4 Hyper-Connections pre-mix with syncall-fused and separate-task implementations."""


import os

import pypto.language as pl

from config import ACTIVE as M, DECODE_BATCH, DECODE_SEQ, PREFILL_BATCH, PREFILL_SEQ


# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S

# model config
D = M.hidden_size
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
NORM_EPS = M.rms_norm_eps

# tiling
T_TILE = 8
LINEAR_T_TILE = 16
COMB_T_TILE = 8
RMS_K_TILE = 256
LINEAR_K_TILE = 256
D_TILE = 256
MIX_D_TILE = 1024
LINEAR_K_SPLIT_TILE = HC_DIM // 4
RMS_K_SPLIT_TILE = HC_DIM // 16

# implementation config
HC_PRE_IMPL = os.environ.get("DSV4_HC_PRE_IMPL", "separate").lower()

# layout
MIX_PAD = 32
HC_PAD = 8

# hardware config
# Full-chip syncall requires one persistent block per Ascend 910B AIC.
NUM_CORES = 24

# The e2e nightly builds a pinned pypto whose language surface predates
# pl.KernelType; the string spelling selects the same participant set there
# and stays accepted on current pypto.
SYNCALL_MIX = pl.KernelType.MIX if hasattr(pl, "KernelType") else "mix"


@pl.jit.inline
def _hc_pre_syncall(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Tensor[[T_DYN, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
):
    t_dim = pl.tensor.dim(x, 0)
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    scale0 = pl.read(hc_scale, [0])
    scale1 = pl.read(hc_scale, [1])
    scale2 = pl.read(hc_scale, [2])

    tt_n = t_dim // T_TILE
    lin_n = (t_linear // LINEAR_T_TILE) * (HC_DIM // LINEAR_K_SPLIT_TILE)
    rms_n = tt_n * (HC_DIM // RMS_K_SPLIT_TILE)
    linear_reduce_n = t_linear // LINEAR_T_TILE
    reduce_n = linear_reduce_n + tt_n
    mixx_n = tt_n * (D // MIX_D_TILE)
    pool_d = 2 * tt_n + mixx_n

    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    hc_reshaped = pl.reshape(hc_base, [1, MIX_HC])
    mixes_raw = pl.create_tensor([t_linear, MIX_PAD], dtype=pl.FP32)
    linear_partial_rows = (HC_DIM // LINEAR_K_SPLIT_TILE) * t_linear
    mixes_partials = pl.create_tensor([linear_partial_rows, MIX_PAD], dtype=pl.FP32)
    # HC_PAD provides 32-byte rows for the post tile.
    post_pad_store = pl.create_tensor([t_linear, HC_PAD], dtype=pl.FP32)
    # The comb gate uses a tile view of the inverse RMS column.
    inv_gm = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    sq_sum_acc = pl.create_tensor([1, t_linear], dtype=pl.FP32)
    sq_partials = pl.create_tensor([HC_DIM // RMS_K_SPLIT_TILE, t_linear], dtype=pl.FP32)
    # Capture the TaskId required by the inline spmd form.
    with pl.spmd(NUM_CORES, name_hint="hc_pre_fused", sync_start=True) as _hc_tid:
        core = pl.tile.get_block_idx()

        for task in pl.range(core, lin_n, NUM_CORES):
            t0 = (task // (HC_DIM // LINEAR_K_SPLIT_TILE)) * LINEAR_T_TILE
            linear_split = task % (HC_DIM // LINEAR_K_SPLIT_TILE)
            k_base = linear_split * LINEAR_K_SPLIT_TILE
            t_rows = pl.min(LINEAR_T_TILE, t_dim - t0)
            acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
            for kb in pl.pipeline(0, LINEAR_K_SPLIT_TILE // LINEAR_K_TILE, stage=2):
                k0 = k_base + kb * LINEAR_K_TILE
                x_linear_chunk = pl.slice(
                    x_flat, [LINEAR_T_TILE, LINEAR_K_TILE], [t0, k0],
                    valid_shape=[t_rows, LINEAR_K_TILE],
                )
                w_chunk = pl.slice(hc_fn, [MIX_PAD, LINEAR_K_TILE], [0, k0], valid_shape=[MIX_HC, LINEAR_K_TILE])
                if kb == 0:
                    acc = pl.matmul(x_linear_chunk, w_chunk, b_trans=True, out_dtype=pl.FP32)
                else:
                    acc = pl.matmul_acc(acc, x_linear_chunk, w_chunk, b_trans=True)
            partial_t0 = linear_split * t_linear + t0
            mixes_partials[partial_t0 : partial_t0 + LINEAR_T_TILE, 0:MIX_PAD] = acc
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            lane = core * 2 + aiv_id
            for task in pl.range(lane, rms_n, NUM_CORES * 2):
                t0 = (task // (HC_DIM // RMS_K_SPLIT_TILE)) * T_TILE
                rms_split = task % (HC_DIM // RMS_K_SPLIT_TILE)
                k_base = rms_split * RMS_K_SPLIT_TILE
                sq_part = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
                for k0 in pl.pipeline(k_base, k_base + RMS_K_SPLIT_TILE, RMS_K_TILE, stage=4):
                    x_chunk = x_flat[t0:t0 + T_TILE, k0:k0 + RMS_K_TILE]
                    x_sq = pl.mul(x_chunk, x_chunk)
                    x_sq_sum = pl.row_sum(x_sq)
                    x_sq_row = pl.reshape(x_sq_sum, [1, T_TILE])
                    sq_part = pl.add(sq_part, x_sq_row)
                sq_partials[rms_split : rms_split + 1, t0 : t0 + T_TILE] = sq_part
        pl.system.syncall(core_type=SYNCALL_MIX)

        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            lane = core * 2 + aiv_id
            for reduce_task in pl.range(lane, reduce_n, NUM_CORES * 2):
                if reduce_task < linear_reduce_n:
                    linear_t0 = reduce_task * LINEAR_T_TILE
                    mixes_total = mixes_partials[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD]
                    for linear_split in pl.range(1, HC_DIM // LINEAR_K_SPLIT_TILE):
                        partial_t0 = linear_split * t_linear + linear_t0
                        partial_mix = mixes_partials[partial_t0 : partial_t0 + LINEAR_T_TILE, 0:MIX_PAD]
                        mixes_total = pl.add(mixes_total, partial_mix)
                    mixes_raw[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD] = mixes_total
                else:
                    rms_t0 = (reduce_task - linear_reduce_n) * T_TILE
                    sq_total = sq_partials[0:1, rms_t0 : rms_t0 + T_TILE]
                    for rms_split in pl.range(1, HC_DIM // RMS_K_SPLIT_TILE):
                        sq_partial = sq_partials[rms_split : rms_split + 1, rms_t0 : rms_t0 + T_TILE]
                        sq_total = pl.add(sq_total, sq_partial)
                    sq_sum_acc[0:1, rms_t0 : rms_t0 + T_TILE] = sq_total
        pl.system.syncall(core_type=SYNCALL_MIX)

        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            lane = core * 2 + aiv_id
            for gw in pl.range(lane, pool_d, NUM_CORES * 2):
                if gw < tt_n:
                    t0 = gw * COMB_T_TILE
                    ssq_row = sq_sum_acc[0:1, t0:t0 + COMB_T_TILE]
                    ssq_mean = pl.mul(ssq_row, HC_DIM_INV)
                    ssq_eps = pl.add(ssq_mean, NORM_EPS)
                    inv_row = pl.rsqrt(ssq_eps, high_precision=True)
                    inv_col_tensor = pl.reshape(inv_row, [COMB_T_TILE, 1])
                    inv_gm[t0 : t0 + COMB_T_TILE, 0:1] = inv_col_tensor
                    inv_col_t = pl.load(inv_gm, [t0, 0], [COMB_T_TILE, 1], target_memory=pl.MemorySpace.Vec)
                    comb_off = HC_MULT * 2
                    mix_g0 = pl.load(
                        mixes_raw, [t0, comb_off + 0 * HC_MULT], [COMB_T_TILE, HC_PAD],
                        valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec,
                    )
                    mix_g1 = pl.load(
                        mixes_raw, [t0, comb_off + 1 * HC_MULT], [COMB_T_TILE, HC_PAD],
                        valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec,
                    )
                    mix_g2 = pl.load(
                        mixes_raw, [t0, comb_off + 2 * HC_MULT], [COMB_T_TILE, HC_PAD],
                        valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec,
                    )
                    mix_g3 = pl.load(
                        mixes_raw, [t0, comb_off + 3 * HC_MULT], [COMB_T_TILE, HC_PAD],
                        valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec,
                    )
                    cb0 = pl.load(
                        hc_reshaped, [0, comb_off + 0 * HC_MULT], [1, HC_PAD],
                        valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec,
                    )
                    cb1 = pl.load(
                        hc_reshaped, [0, comb_off + 1 * HC_MULT], [1, HC_PAD],
                        valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec,
                    )
                    cb2 = pl.load(
                        hc_reshaped, [0, comb_off + 2 * HC_MULT], [1, HC_PAD],
                        valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec,
                    )
                    cb3 = pl.load(
                        hc_reshaped, [0, comb_off + 3 * HC_MULT], [1, HC_PAD],
                        valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec,
                    )
                    row0_normed = pl.row_expand_mul(mix_g0, inv_col_t)
                    row0_scaled = pl.mul(row0_normed, scale2)
                    row0_base = pl.col_expand(mix_g0, cb0)
                    row0 = pl.add(row0_scaled, row0_base)
                    row1_normed = pl.row_expand_mul(mix_g1, inv_col_t)
                    row1_scaled = pl.mul(row1_normed, scale2)
                    row1_base = pl.col_expand(mix_g1, cb1)
                    row1 = pl.add(row1_scaled, row1_base)
                    row2_normed = pl.row_expand_mul(mix_g2, inv_col_t)
                    row2_scaled = pl.mul(row2_normed, scale2)
                    row2_base = pl.col_expand(mix_g2, cb2)
                    row2 = pl.add(row2_scaled, row2_base)
                    row3_normed = pl.row_expand_mul(mix_g3, inv_col_t)
                    row3_scaled = pl.mul(row3_normed, scale2)
                    row3_base = pl.col_expand(mix_g3, cb3)
                    row3 = pl.add(row3_scaled, row3_base)
                    row0_p = pl.fillpad(row0, pad_value=pl.PadValue.min)
                    row1_p = pl.fillpad(row1, pad_value=pl.PadValue.min)
                    row2_p = pl.fillpad(row2, pad_value=pl.PadValue.min)
                    row3_p = pl.fillpad(row3, pad_value=pl.PadValue.min)

                    row_max_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    row_sum_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    row0_max = pl.row_max(row0_p, row_max_tmp)
                    row1_max = pl.row_max(row1_p, row_max_tmp)
                    row2_max = pl.row_max(row2_p, row_max_tmp)
                    row3_max = pl.row_max(row3_p, row_max_tmp)
                    row0_centered = pl.row_expand_sub(row0_p, row0_max)
                    row0_exp = pl.exp(row0_centered)
                    row1_centered = pl.row_expand_sub(row1_p, row1_max)
                    row1_exp = pl.exp(row1_centered)
                    row2_centered = pl.row_expand_sub(row2_p, row2_max)
                    row2_exp = pl.exp(row2_centered)
                    row3_centered = pl.row_expand_sub(row3_p, row3_max)
                    row3_exp = pl.exp(row3_centered)
                    row0_sum = pl.row_sum(row0_exp, row_sum_tmp)
                    row1_sum = pl.row_sum(row1_exp, row_sum_tmp)
                    row2_sum = pl.row_sum(row2_exp, row_sum_tmp)
                    row3_sum = pl.row_sum(row3_exp, row_sum_tmp)
                    row0_prob = pl.row_expand_div(row0_exp, row0_sum)
                    row0_soft = pl.add(row0_prob, HC_EPS)
                    row1_prob = pl.row_expand_div(row1_exp, row1_sum)
                    row1_soft = pl.add(row1_prob, HC_EPS)
                    row2_prob = pl.row_expand_div(row2_exp, row2_sum)
                    row2_soft = pl.add(row2_prob, HC_EPS)
                    row3_prob = pl.row_expand_div(row3_exp, row3_sum)
                    row3_soft = pl.add(row3_prob, HC_EPS)

                    row0_valid = pl.set_validshape(row0_soft, COMB_T_TILE, HC_MULT)
                    row1_valid = pl.set_validshape(row1_soft, COMB_T_TILE, HC_MULT)
                    row2_valid = pl.set_validshape(row2_soft, COMB_T_TILE, HC_MULT)
                    row3_valid = pl.set_validshape(row3_soft, COMB_T_TILE, HC_MULT)
                    row0_eff = pl.fillpad(row0_valid, pad_value=pl.PadValue.zero)
                    row1_eff = pl.fillpad(row1_valid, pad_value=pl.PadValue.zero)
                    row2_eff = pl.fillpad(row2_valid, pad_value=pl.PadValue.zero)
                    row3_eff = pl.fillpad(row3_valid, pad_value=pl.PadValue.zero)

                    row_sum_tmp_iter = pl.create_tile(
                        [COMB_T_TILE, HC_PAD], dtype=pl.FP32,
                        target_memory=pl.MemorySpace.Vec,
                    )
                    row01_eff = pl.add(row0_eff, row1_eff)
                    row23_eff = pl.add(row2_eff, row3_eff)
                    col_sum_raw = pl.add(row01_eff, row23_eff)
                    col_sum_eps = pl.add(col_sum_raw, HC_EPS)
                    row0_cur = pl.div(row0_eff, col_sum_eps)
                    row1_cur = pl.div(row1_eff, col_sum_eps)
                    row2_cur = pl.div(row2_eff, col_sum_eps)
                    row3_cur = pl.div(row3_eff, col_sum_eps)

                    for _sk_it in pl.pipeline(HC_SINKHORN_ITER - 1, stage=2):
                        row0_sum_raw = pl.row_sum(row0_cur, row_sum_tmp_iter)
                        row0_rowsum = pl.add(row0_sum_raw, HC_EPS)
                        row1_sum_raw = pl.row_sum(row1_cur, row_sum_tmp_iter)
                        row1_rowsum = pl.add(row1_sum_raw, HC_EPS)
                        row2_sum_raw = pl.row_sum(row2_cur, row_sum_tmp_iter)
                        row2_rowsum = pl.add(row2_sum_raw, HC_EPS)
                        row3_sum_raw = pl.row_sum(row3_cur, row_sum_tmp_iter)
                        row3_rowsum = pl.add(row3_sum_raw, HC_EPS)
                        row0_norm = pl.row_expand_div(row0_cur, row0_rowsum)
                        row1_norm = pl.row_expand_div(row1_cur, row1_rowsum)
                        row2_norm = pl.row_expand_div(row2_cur, row2_rowsum)
                        row3_norm = pl.row_expand_div(row3_cur, row3_rowsum)
                        row01_norm = pl.add(row0_norm, row1_norm)
                        row23_norm = pl.add(row2_norm, row3_norm)
                        col_sum_iter_raw = pl.add(row01_norm, row23_norm)
                        col_sum_iter_eps = pl.add(col_sum_iter_raw, HC_EPS)
                        row0_cur = pl.div(row0_norm, col_sum_iter_eps)
                        row1_cur = pl.div(row1_norm, col_sum_iter_eps)
                        row2_cur = pl.div(row2_norm, col_sum_iter_eps)
                        row3_cur = pl.div(row3_norm, col_sum_iter_eps)

                    row0_out = pl.set_validshape(row0_cur, COMB_T_TILE, HC_MULT)
                    row1_out = pl.set_validshape(row1_cur, COMB_T_TILE, HC_MULT)
                    row2_out = pl.set_validshape(row2_cur, COMB_T_TILE, HC_MULT)
                    row3_out = pl.set_validshape(row3_cur, COMB_T_TILE, HC_MULT)
                    pl.store(row0_out, [t0, 0 * HC_MULT], comb)
                    pl.store(row1_out, [t0, 1 * HC_MULT], comb)
                    pl.store(row2_out, [t0, 2 * HC_MULT], comb)
                    pl.store(row3_out, [t0, 3 * HC_MULT], comb)

                elif gw < tt_n + mixx_n:
                    blk = gw - tt_n
                    t0 = (blk // (D // MIX_D_TILE)) * T_TILE
                    d_base = (blk % (D // MIX_D_TILE)) * MIX_D_TILE
                    ssq_row = sq_sum_acc[0:1, t0:t0 + T_TILE]
                    ssq_mean = pl.mul(ssq_row, HC_DIM_INV)
                    ssq_eps = pl.add(ssq_mean, NORM_EPS)
                    inv_row = pl.rsqrt(ssq_eps, high_precision=True)
                    inv_col = pl.reshape(inv_row, [T_TILE, 1])
                    pre_base = hc_reshaped[0:1, 0:HC_PAD]
                    pre_normed = pl.row_expand_mul(mixes_raw[t0:t0 + T_TILE, 0:HC_PAD], inv_col)
                    pre_scaled = pl.mul(pre_normed, scale0)
                    pre_base_tile = pl.col_expand(pre_scaled, pre_base)
                    pre_logits = pl.add(pre_scaled, pre_base_tile)
                    pre_neg = pl.neg(pre_logits)
                    pre_exp = pl.exp(pre_neg)
                    pre_denom = pl.add(pre_exp, 1.0)
                    pre_sigmoid = pl.recip(pre_denom)
                    pre_val_raw = pl.add(pre_sigmoid, HC_EPS)
                    # Fix the dynamic-offset gate to a static tile shape.
                    pre_val = pl.set_validshape(pre_val_raw, T_TILE, HC_PAD)
                    pre_tile_t = pl.transpose(pre_val, axis1=0, axis2=1)
                    pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
                    pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
                    pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
                    pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
                    for d0 in pl.pipeline(d_base, d_base + MIX_D_TILE, D_TILE, stage=2):
                        x0 = x_flat[t0:t0 + T_TILE, 0 * D + d0:0 * D + d0 + D_TILE]
                        x1 = x_flat[t0:t0 + T_TILE, 1 * D + d0:1 * D + d0 + D_TILE]
                        x2 = x_flat[t0:t0 + T_TILE, 2 * D + d0:2 * D + d0 + D_TILE]
                        x3 = x_flat[t0:t0 + T_TILE, 3 * D + d0:3 * D + d0 + D_TILE]
                        y0 = pl.row_expand_mul(x0, pre0)
                        y1 = pl.row_expand_mul(x1, pre1)
                        y2 = pl.row_expand_mul(x2, pre2)
                        y3 = pl.row_expand_mul(x3, pre3)
                        y01 = pl.add(y0, y1)
                        y23 = pl.add(y2, y3)
                        y_tile = pl.add(y01, y23)
                        y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
                        x_mixed[t0 : t0 + T_TILE, d0 : d0 + D_TILE] = y_bf16

                else:
                    ob = gw - tt_n - mixx_n
                    t0 = ob * COMB_T_TILE
                    ssq_row = sq_sum_acc[0:1, t0:t0 + COMB_T_TILE]
                    ssq_mean = pl.mul(ssq_row, HC_DIM_INV)
                    ssq_eps = pl.add(ssq_mean, NORM_EPS)
                    inv_row = pl.rsqrt(ssq_eps, high_precision=True)
                    inv_col = pl.reshape(inv_row, [COMB_T_TILE, 1])
                    post_base = hc_reshaped[0:1, HC_MULT:HC_MULT + HC_PAD]
                    post_normed = pl.row_expand_mul(mixes_raw[t0:t0 + COMB_T_TILE, HC_MULT:HC_MULT + HC_PAD], inv_col)
                    post_scaled = pl.mul(post_normed, scale1)
                    post_base_tile = pl.col_expand(post_scaled, post_base)
                    post_logits = pl.add(post_scaled, post_base_tile)
                    post_neg = pl.neg(post_logits)
                    post_exp = pl.exp(post_neg)
                    post_denom = pl.add(post_exp, 1.0)
                    post_sigmoid = pl.recip(post_denom)
                    post_pad = pl.mul(post_sigmoid, 2.0)
                    post_pad_store[t0 : t0 + COMB_T_TILE, 0:HC_PAD] = post_pad
                    post_tile = pl.load(
                        post_pad_store, [t0, 0], [COMB_T_TILE, HC_PAD],
                        valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec,
                    )
                    pl.store(post_tile, [t0, 0], post)
    return x_mixed


@pl.jit.inline
def _hc_pre_separate(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Tensor[[T_DYN, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
):
    """Compute the Hyper-Connections pre-mix with separate tasks."""
    t_dim = pl.tensor.dim(x, 0)
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    scale0 = pl.read(hc_scale, [0])
    scale1 = pl.read(hc_scale, [1])
    scale2 = pl.read(hc_scale, [2])

    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    for t in pl.spmd(t_dim // T_TILE, name_hint="hc_pre_rms"):
        t0 = t * T_TILE
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for k0 in pl.pipeline(0, HC_DIM, RMS_K_TILE, stage=4):
            x_chunk = x_flat[t0:t0 + T_TILE, k0:k0 + RMS_K_TILE]
            x_sq = pl.mul(x_chunk, x_chunk)
            x_sq_sum = pl.row_sum(x_sq)
            x_sq_row = pl.reshape(x_sq_sum, [1, T_TILE])
            sq_sum = pl.add(sq_sum, x_sq_row)
        sq_mean = pl.mul(sq_sum, HC_DIM_INV)
        sq_eps = pl.add(sq_mean, NORM_EPS)
        inv_row = pl.rsqrt(sq_eps, high_precision=True)
        inv = pl.reshape(inv_row, [T_TILE, 1])
        inv_rms[t0 : t0 + T_TILE, 0:1] = inv

    linear_partial_rows = (HC_DIM // LINEAR_K_SPLIT_TILE) * t_linear
    mixes_partials = pl.create_tensor([linear_partial_rows, MIX_PAD], dtype=pl.FP32)
    for task in pl.spmd(
        (t_linear // LINEAR_T_TILE) * (HC_DIM // LINEAR_K_SPLIT_TILE),
        name_hint="hc_pre_linear",
    ):
        t0 = (task // (HC_DIM // LINEAR_K_SPLIT_TILE)) * LINEAR_T_TILE
        linear_split = task % (HC_DIM // LINEAR_K_SPLIT_TILE)
        k_base = linear_split * LINEAR_K_SPLIT_TILE
        t_rows = pl.min(LINEAR_T_TILE, t_dim - t0)
        acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
        for kb in pl.pipeline(0, LINEAR_K_SPLIT_TILE // LINEAR_K_TILE, stage=2):
            k0 = k_base + kb * LINEAR_K_TILE
            x_linear_chunk = pl.slice(
                x_flat, [LINEAR_T_TILE, LINEAR_K_TILE], [t0, k0],
                valid_shape=[t_rows, LINEAR_K_TILE],
            )
            w_chunk = pl.slice(hc_fn, [MIX_PAD, LINEAR_K_TILE], [0, k0], valid_shape=[MIX_HC, LINEAR_K_TILE])
            if kb == 0:
                acc = pl.matmul(x_linear_chunk, w_chunk, b_trans=True, out_dtype=pl.FP32)
            else:
                acc = pl.matmul_acc(acc, x_linear_chunk, w_chunk, b_trans=True)
        partial_t0 = linear_split * t_linear + t0
        mixes_partials[partial_t0 : partial_t0 + LINEAR_T_TILE, 0:MIX_PAD] = acc

    pre_val_store = pl.create_tensor([t_linear, HC_PAD], dtype=pl.FP32)
    for ob in pl.spmd(t_dim // T_TILE, name_hint="split_pre_post"):
        t0 = ob * T_TILE
        inv_col = inv_rms[t0:t0 + T_TILE, 0:1]
        pre_mixes = mixes_partials[t0:t0 + T_TILE, 0:HC_PAD]
        post_mixes = mixes_partials[t0:t0 + T_TILE, HC_MULT:HC_MULT + HC_PAD]
        for linear_split in pl.unroll(1, HC_DIM // LINEAR_K_SPLIT_TILE):
            p0 = linear_split * t_linear + t0
            pre_mixes = pl.add(pre_mixes, mixes_partials[p0:p0 + T_TILE, 0:HC_PAD])
            post_mixes = pl.add(post_mixes, mixes_partials[p0:p0 + T_TILE, HC_MULT:HC_MULT + HC_PAD])

        pre_base = pl.reshape(hc_base[0:HC_PAD], [1, HC_PAD])
        pre_normed = pl.row_expand_mul(pre_mixes, inv_col)
        pre_scaled = pl.mul(pre_normed, scale0)
        pre_base_tile = pl.col_expand(pre_scaled, pre_base)
        pre_logits = pl.add(pre_scaled, pre_base_tile)
        pre_neg = pl.neg(pre_logits)
        pre_exp = pl.exp(pre_neg)
        pre_denom = pl.add(pre_exp, 1.0)
        pre_sig = pl.recip(pre_denom)
        pre_val = pl.add(pre_sig, HC_EPS)
        pre_val_store[t0 : t0 + T_TILE, 0:HC_PAD] = pre_val

        post_base = pl.reshape(hc_base[HC_MULT:HC_MULT + HC_PAD], [1, HC_PAD])
        post_normed = pl.row_expand_mul(post_mixes, inv_col)
        post_scaled = pl.mul(post_normed, scale1)
        post_base_tile = pl.col_expand(post_scaled, post_base)
        post_logits = pl.add(post_scaled, post_base_tile)
        post_neg = pl.neg(post_logits)
        post_exp = pl.exp(post_neg)
        post_denom = pl.add(post_exp, 1.0)
        post_sig = pl.recip(post_denom)
        post_pad = pl.mul(post_sig, 2.0)
        # HC_PAD supplies a 32-byte row while HC_MULT marks the valid output width.
        post[t0:t0 + T_TILE, 0:HC_MULT] = pl.slice(post_pad, [T_TILE, HC_PAD], [0, 0], valid_shape=[T_TILE, HC_MULT])

    hc_base_2d = pl.reshape(hc_base, [1, MIX_HC])
    for ob in pl.spmd(t_dim // COMB_T_TILE, name_hint="comb_sinkhorn"):
        t0 = ob * COMB_T_TILE
        inv_col_t = pl.load(inv_rms, [t0, 0], [COMB_T_TILE, 1], target_memory=pl.MemorySpace.Vec)
        comb_off = HC_MULT * 2
        mix_g0 = pl.load(
            mixes_partials, [t0, comb_off + 0 * HC_MULT], [COMB_T_TILE, HC_PAD],
            valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec,
        )
        mix_g1 = pl.load(
            mixes_partials, [t0, comb_off + 1 * HC_MULT], [COMB_T_TILE, HC_PAD],
            valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec,
        )
        mix_g2 = pl.load(
            mixes_partials, [t0, comb_off + 2 * HC_MULT], [COMB_T_TILE, HC_PAD],
            valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec,
        )
        mix_g3 = pl.load(
            mixes_partials, [t0, comb_off + 3 * HC_MULT], [COMB_T_TILE, HC_PAD],
            valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec,
        )
        for linear_split in pl.unroll(1, HC_DIM // LINEAR_K_SPLIT_TILE):
            p0 = linear_split * t_linear + t0
            partial_g0 = pl.load(
                mixes_partials, [p0, comb_off + 0 * HC_MULT], [COMB_T_TILE, HC_PAD],
                valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec,
            )
            mix_g0 = pl.add(mix_g0, partial_g0)
            partial_g1 = pl.load(
                mixes_partials, [p0, comb_off + 1 * HC_MULT], [COMB_T_TILE, HC_PAD],
                valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec,
            )
            mix_g1 = pl.add(mix_g1, partial_g1)
            partial_g2 = pl.load(
                mixes_partials, [p0, comb_off + 2 * HC_MULT], [COMB_T_TILE, HC_PAD],
                valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec,
            )
            mix_g2 = pl.add(mix_g2, partial_g2)
            partial_g3 = pl.load(
                mixes_partials, [p0, comb_off + 3 * HC_MULT], [COMB_T_TILE, HC_PAD],
                valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec,
            )
            mix_g3 = pl.add(mix_g3, partial_g3)
        cb0 = pl.load(
            hc_base_2d, [0, comb_off + 0 * HC_MULT], [1, HC_PAD],
            valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec,
        )
        cb1 = pl.load(
            hc_base_2d, [0, comb_off + 1 * HC_MULT], [1, HC_PAD],
            valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec,
        )
        cb2 = pl.load(
            hc_base_2d, [0, comb_off + 2 * HC_MULT], [1, HC_PAD],
            valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec,
        )
        cb3 = pl.load(
            hc_base_2d, [0, comb_off + 3 * HC_MULT], [1, HC_PAD],
            valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec,
        )
        row0_normed = pl.row_expand_mul(mix_g0, inv_col_t)
        row0_scaled = pl.mul(row0_normed, scale2)
        row0_base = pl.col_expand(mix_g0, cb0)
        row0 = pl.add(row0_scaled, row0_base)
        row1_normed = pl.row_expand_mul(mix_g1, inv_col_t)
        row1_scaled = pl.mul(row1_normed, scale2)
        row1_base = pl.col_expand(mix_g1, cb1)
        row1 = pl.add(row1_scaled, row1_base)
        row2_normed = pl.row_expand_mul(mix_g2, inv_col_t)
        row2_scaled = pl.mul(row2_normed, scale2)
        row2_base = pl.col_expand(mix_g2, cb2)
        row2 = pl.add(row2_scaled, row2_base)
        row3_normed = pl.row_expand_mul(mix_g3, inv_col_t)
        row3_scaled = pl.mul(row3_normed, scale2)
        row3_base = pl.col_expand(mix_g3, cb3)
        row3 = pl.add(row3_scaled, row3_base)
        row0_p = pl.fillpad(row0, pad_value=pl.PadValue.min)
        row1_p = pl.fillpad(row1, pad_value=pl.PadValue.min)
        row2_p = pl.fillpad(row2, pad_value=pl.PadValue.min)
        row3_p = pl.fillpad(row3, pad_value=pl.PadValue.min)

        row_max_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row_sum_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row0_max = pl.row_max(row0_p, row_max_tmp)
        row1_max = pl.row_max(row1_p, row_max_tmp)
        row2_max = pl.row_max(row2_p, row_max_tmp)
        row3_max = pl.row_max(row3_p, row_max_tmp)
        row0_centered = pl.row_expand_sub(row0_p, row0_max)
        row0_exp = pl.exp(row0_centered)
        row1_centered = pl.row_expand_sub(row1_p, row1_max)
        row1_exp = pl.exp(row1_centered)
        row2_centered = pl.row_expand_sub(row2_p, row2_max)
        row2_exp = pl.exp(row2_centered)
        row3_centered = pl.row_expand_sub(row3_p, row3_max)
        row3_exp = pl.exp(row3_centered)
        row0_sum = pl.row_sum(row0_exp, row_sum_tmp)
        row1_sum = pl.row_sum(row1_exp, row_sum_tmp)
        row2_sum = pl.row_sum(row2_exp, row_sum_tmp)
        row3_sum = pl.row_sum(row3_exp, row_sum_tmp)
        row0_prob = pl.row_expand_div(row0_exp, row0_sum)
        row0_soft = pl.add(row0_prob, HC_EPS)
        row1_prob = pl.row_expand_div(row1_exp, row1_sum)
        row1_soft = pl.add(row1_prob, HC_EPS)
        row2_prob = pl.row_expand_div(row2_exp, row2_sum)
        row2_soft = pl.add(row2_prob, HC_EPS)
        row3_prob = pl.row_expand_div(row3_exp, row3_sum)
        row3_soft = pl.add(row3_prob, HC_EPS)

        row0_valid = pl.set_validshape(row0_soft, COMB_T_TILE, HC_MULT)
        row1_valid = pl.set_validshape(row1_soft, COMB_T_TILE, HC_MULT)
        row2_valid = pl.set_validshape(row2_soft, COMB_T_TILE, HC_MULT)
        row3_valid = pl.set_validshape(row3_soft, COMB_T_TILE, HC_MULT)
        row0_eff = pl.fillpad(row0_valid, pad_value=pl.PadValue.zero)
        row1_eff = pl.fillpad(row1_valid, pad_value=pl.PadValue.zero)
        row2_eff = pl.fillpad(row2_valid, pad_value=pl.PadValue.zero)
        row3_eff = pl.fillpad(row3_valid, pad_value=pl.PadValue.zero)

        row_sum_tmp_iter = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row01_eff = pl.add(row0_eff, row1_eff)
        row23_eff = pl.add(row2_eff, row3_eff)
        col_sum_raw = pl.add(row01_eff, row23_eff)
        col_sum_eps = pl.add(col_sum_raw, HC_EPS)
        row0_cur = pl.div(row0_eff, col_sum_eps)
        row1_cur = pl.div(row1_eff, col_sum_eps)
        row2_cur = pl.div(row2_eff, col_sum_eps)
        row3_cur = pl.div(row3_eff, col_sum_eps)

        for _sk_it in pl.pipeline(HC_SINKHORN_ITER - 1, stage=2):
            row0_sum_raw = pl.row_sum(row0_cur, row_sum_tmp_iter)
            row0_rowsum = pl.add(row0_sum_raw, HC_EPS)
            row1_sum_raw = pl.row_sum(row1_cur, row_sum_tmp_iter)
            row1_rowsum = pl.add(row1_sum_raw, HC_EPS)
            row2_sum_raw = pl.row_sum(row2_cur, row_sum_tmp_iter)
            row2_rowsum = pl.add(row2_sum_raw, HC_EPS)
            row3_sum_raw = pl.row_sum(row3_cur, row_sum_tmp_iter)
            row3_rowsum = pl.add(row3_sum_raw, HC_EPS)
            row0_norm = pl.row_expand_div(row0_cur, row0_rowsum)
            row1_norm = pl.row_expand_div(row1_cur, row1_rowsum)
            row2_norm = pl.row_expand_div(row2_cur, row2_rowsum)
            row3_norm = pl.row_expand_div(row3_cur, row3_rowsum)
            row01_norm = pl.add(row0_norm, row1_norm)
            row23_norm = pl.add(row2_norm, row3_norm)
            col_sum_iter_raw = pl.add(row01_norm, row23_norm)
            col_sum_iter_eps = pl.add(col_sum_iter_raw, HC_EPS)
            row0_cur = pl.div(row0_norm, col_sum_iter_eps)
            row1_cur = pl.div(row1_norm, col_sum_iter_eps)
            row2_cur = pl.div(row2_norm, col_sum_iter_eps)
            row3_cur = pl.div(row3_norm, col_sum_iter_eps)

        row0_out = pl.set_validshape(row0_cur, COMB_T_TILE, HC_MULT)
        row1_out = pl.set_validshape(row1_cur, COMB_T_TILE, HC_MULT)
        row2_out = pl.set_validshape(row2_cur, COMB_T_TILE, HC_MULT)
        row3_out = pl.set_validshape(row3_cur, COMB_T_TILE, HC_MULT)
        pl.store(row0_out, [t0, 0 * HC_MULT], comb)
        pl.store(row1_out, [t0, 1 * HC_MULT], comb)
        pl.store(row2_out, [t0, 2 * HC_MULT], comb)
        pl.store(row3_out, [t0, 3 * HC_MULT], comb)

    for blk in pl.spmd((t_dim // T_TILE) * (D // MIX_D_TILE), name_hint="mix_x"):
        t0 = (blk // (D // MIX_D_TILE)) * T_TILE
        d_base = (blk % (D // MIX_D_TILE)) * MIX_D_TILE
        pre_tile_t = pl.transpose(pre_val_store[t0:t0 + T_TILE, 0:HC_PAD], axis1=0, axis2=1)
        pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
        for d0 in pl.pipeline(d_base, d_base + MIX_D_TILE, D_TILE, stage=2):
            x0 = x_flat[t0:t0 + T_TILE, 0 * D + d0:0 * D + d0 + D_TILE]
            x1 = x_flat[t0:t0 + T_TILE, 1 * D + d0:1 * D + d0 + D_TILE]
            x2 = x_flat[t0:t0 + T_TILE, 2 * D + d0:2 * D + d0 + D_TILE]
            x3 = x_flat[t0:t0 + T_TILE, 3 * D + d0:3 * D + d0 + D_TILE]
            y0 = pl.row_expand_mul(x0, pre0)
            y1 = pl.row_expand_mul(x1, pre1)
            y2 = pl.row_expand_mul(x2, pre2)
            y3 = pl.row_expand_mul(x3, pre3)
            y01 = pl.add(y0, y1)
            y23 = pl.add(y2, y3)
            y_tile = pl.add(y01, y23)
            y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
            x_mixed[t0:t0 + T_TILE, d0:d0 + D_TILE] = y_bf16
    return x_mixed


def _bind_hc_pre():
    """Bind the public inline kernel for HC_PRE_IMPL."""
    if HC_PRE_IMPL == "separate":
        @pl.jit.inline
        def hc_pre(
            x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
            hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
            hc_scale: pl.Tensor[[3], pl.FP32],
            hc_base: pl.Tensor[[MIX_HC], pl.FP32],
            x_mixed: pl.Tensor[[T_DYN, D], pl.BF16],
            post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
            comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
        ):
            _hc_pre_separate(x, hc_fn, hc_scale, hc_base, x_mixed, post, comb)
            return x_mixed
    else:
        @pl.jit.inline
        def hc_pre(
            x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
            hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
            hc_scale: pl.Tensor[[3], pl.FP32],
            hc_base: pl.Tensor[[MIX_HC], pl.FP32],
            x_mixed: pl.Tensor[[T_DYN, D], pl.BF16],
            post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
            comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
        ):
            _hc_pre_syncall(x, hc_fn, hc_scale, hc_base, x_mixed, post, comb)
            return x_mixed
    return hc_pre


hc_pre = _bind_hc_pre()


@pl.jit
def hc_pre_test(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    post: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    comb: pl.Out[pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32]],
):
    x.bind_dynamic(0, T_DYN)
    x_mixed.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    comb.bind_dynamic(0, T_DYN)

    hc_pre(x, hc_fn, hc_scale, hc_base, x_mixed, post, comb)
    return x_mixed


_A5_FP32_VECTOR_LANES = 64


def _golden_a5_trowsum_fp32(values):
    """Mirror A5 TROWSUM's FP32 reduction order along the last dimension."""
    import torch

    if values.dtype != torch.float32:
        raise ValueError(f"A5 FP32 TROWSUM golden requires float32, got {values.dtype}")
    if values.shape[-1] % _A5_FP32_VECTOR_LANES != 0:
        message = f"A5 FP32 TROWSUM width must be divisible by {_A5_FP32_VECTOR_LANES}, got {values.shape[-1]}"
        raise ValueError(message)

    # Reduce 64-lane groups pairwise, then accumulate the group sums in address order.
    groups = values.reshape(*values.shape[:-1], -1, _A5_FP32_VECTOR_LANES)
    while groups.shape[-1] > 1:
        pairs = groups.reshape(*groups.shape[:-1], -1, 2)
        groups = pairs[..., 0] + pairs[..., 1]

    group_sums = groups[..., 0]
    total = torch.zeros_like(group_sums[..., :1])
    for group in range(group_sums.shape[-1]):
        total += group_sums[..., group:group + 1]
    return total


def _golden_a5_high_precision_rsqrt(value):
    """Compute the A5 high-precision FP32 reciprocal square root."""
    import torch

    return torch.rsqrt(value.to(torch.float64)).to(torch.float32)


def _golden_a5_rms_inv(x_flat_2d):
    """Compute the hc_pre inverse RMS with A5 reduction order."""
    import torch

    if x_flat_2d.dtype != torch.float32:
        raise ValueError(f"hc_pre RMS golden requires float32, got {x_flat_2d.dtype}")
    if x_flat_2d.shape[-1] != HC_DIM:
        raise ValueError(f"hc_pre RMS width must be HC_DIM={HC_DIM}, got {x_flat_2d.shape[-1]}")

    sq_sum = torch.zeros(x_flat_2d.shape[0], 1, dtype=torch.float32, device=x_flat_2d.device)
    for k0 in range(0, HC_DIM, RMS_K_TILE):
        x_chunk = x_flat_2d[:, k0:k0 + RMS_K_TILE]
        sq_sum += _golden_a5_trowsum_fp32(x_chunk * x_chunk)

    rms_arg = sq_sum * HC_DIM_INV + NORM_EPS
    return _golden_a5_high_precision_rsqrt(rms_arg)


def golden_hc_pre(tensors):
    """Compute the Hyper-Connections pre-mix reference."""
    import torch

    x = tensors["x"].float()  # [T, hc, D]
    hc_fn = tensors["hc_fn"].float()  # [mix_hc, hc*D]
    hc_scale = tensors["hc_scale"].float()  # [3]
    hc_base = tensors["hc_base"].float()  # [mix_hc]

    t_dim = x.shape[0]
    x_flat_2d = x.reshape(t_dim, HC_DIM)

    rsqrt = _golden_a5_rms_inv(x_flat_2d)

    # Mirror the split-K cube accumulation and consumer reduction order.
    split_count = HC_DIM // LINEAR_K_SPLIT_TILE
    chunks_per_split = LINEAR_K_SPLIT_TILE // LINEAR_K_TILE
    x_k = x_flat_2d.reshape(t_dim, 1, split_count, chunks_per_split, LINEAR_K_TILE)
    w_k = hc_fn.reshape(1, MIX_HC, split_count, chunks_per_split, LINEAR_K_TILE)
    per_chunk = (x_k * w_k).sum(dim=4)
    split_partials = []
    for split in range(split_count):
        partial = per_chunk[:, :, split, 0]
        for chunk in range(1, chunks_per_split):
            partial = partial + per_chunk[:, :, split, chunk]
        split_partials.append(partial)
    mixes = split_partials[0]
    for split in range(1, split_count):
        mixes = mixes + split_partials[split]
    mixes *= rsqrt

    pre = torch.sigmoid(mixes[..., :HC_MULT] * hc_scale[0] + hc_base[:HC_MULT]) + HC_EPS
    post_logits = mixes[..., HC_MULT:HC_MULT * 2] * hc_scale[1] + hc_base[HC_MULT:HC_MULT * 2]
    post_t = 2 * torch.sigmoid(post_logits)
    comb_logits = mixes[..., HC_MULT * 2:] * hc_scale[2] + hc_base[HC_MULT * 2:]
    comb_t = comb_logits.view(t_dim, HC_MULT, HC_MULT)

    comb_t = torch.softmax(comb_t, dim=-1) + HC_EPS
    comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITER - 1):
        comb_t = comb_t / (comb_t.sum(-1, keepdim=True) + HC_EPS)
        comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)

    y0 = x[:, 0, :] * pre[:, 0:1]
    y1 = x[:, 1, :] * pre[:, 1:2]
    y2 = x[:, 2, :] * pre[:, 2:3]
    y3 = x[:, 3, :] * pre[:, 3:4]
    y = (y0 + y1) + (y2 + y3)

    tensors["x_mixed"][:] = y.to(torch.bfloat16).reshape(t_dim, D)
    tensors["post"][:] = post_t.reshape(t_dim, HC_MULT)
    tensors["comb"][:] = comb_t.reshape(t_dim, HC_MULT * HC_MULT)


def build_tensor_specs(B, S):
    import torch
    from golden import TensorSpec

    T = B * S

    def init_x():
        return torch.randn(T, HC_MULT, D) * 0.05

    def init_hc_fn():
        return torch.randn(MIX_HC, HC_DIM) * 0.0519

    def init_hc_scale():
        return torch.tensor([0.076099, 0.032597, 0.226994])

    def init_hc_base():
        return torch.tensor(
            [
                5.9166, -3.6223, -2.9324, -3.3124,
                -3.9100, -0.9384, -3.3256, -2.5240,
                2.0706, -2.5728, 0.1424, -3.9453,
                -3.8859, 3.4634, -3.3799, -2.6077,
                -2.7191, -2.4846, 2.0395, -0.5010,
                -3.5992, -2.7520, -3.3493, 3.1587,
            ]
        )

    return [
        TensorSpec("x", [T, HC_MULT, D], torch.float32, init_value=init_x),
        TensorSpec("hc_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_fn),
        TensorSpec("hc_scale", [3], torch.float32, init_value=init_hc_scale),
        TensorSpec("hc_base", [MIX_HC], torch.float32, init_value=init_hc_base),
        TensorSpec("x_mixed", [T, D], torch.bfloat16),
        TensorSpec("post", [T, HC_MULT], torch.float32),
        TensorSpec("comb", [T, HC_MULT * HC_MULT], torch.float32),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    MODES = {
        "decode": (DECODE_BATCH, DECODE_SEQ),
        "prefill": (PREFILL_BATCH, PREFILL_SEQ),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    mode_help = "Use decode or prefill batch sizes, or 'all' to test both."
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="all", help=mode_help)
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    no_dep_gen_help = "Deprecated no-op retained for CLI compatibility; dep_gen follows --impl."
    impl_help = "Select the syncall-fused or separate-task implementation."
    parser.add_argument("--no-dep-gen", action="store_true", default=False, help=no_dep_gen_help)
    parser.add_argument("--impl", choices=["syncall", "separate"], default=HC_PRE_IMPL, help=impl_help)
    args = parser.parse_args()

    HC_PRE_IMPL = args.impl
    hc_pre = _bind_hc_pre()
    print(f"hc_pre implementation: {HC_PRE_IMPL}")

    # Full-chip syncall is limited to Ascend 910B.
    if args.platform in ("a5", "a5sim") and HC_PRE_IMPL == "syncall":
        raise SystemExit(
            f"hc_pre 'syncall' impl is specialized to Ascend 910B (NUM_CORES={NUM_CORES} == physical "
            f"AIC count); its full-occupancy mix-syncall would hang (AICore timeout 507018) on "
            f"{args.platform!r}. Re-run with --impl separate on {args.platform!r}, or -p a2a3."
        )

    modes_to_run = list(MODES.keys()) if args.mode == "all" else [args.mode]

    for mode_name in modes_to_run:
        B, S = MODES[mode_name]
        print(f"--- hc_pre {mode_name}: B={B}, S={S} ---")
        result = run_jit(
            fn=hc_pre_test,
            specs=build_tensor_specs(B, S),
            golden_fn=golden_hc_pre,
            runtime_dir=args.runtime_dir,
            golden_data=args.golden_data,
            compile_cfg=dict(dump_passes=args.dump_passes),
            runtime_cfg=dict(
                platform=args.platform,
                device_id=args.device,
                enable_chip_swimlane=args.enable_chip_swimlane,
                # Full-chip syncall runs with dependency generation disabled.
                enable_dep_gen=(HC_PRE_IMPL == "separate"),
            ),
            rtol=1e-3,
            atol=1e-3,
            compare_fn={
                "x_mixed": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "post": ratio_allclose(atol=2.5e-5, rtol=5e-3),
                "comb": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            },
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
