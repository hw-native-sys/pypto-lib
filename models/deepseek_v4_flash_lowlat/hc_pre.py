# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 hc_pre -- hyper-connection pre-mix.

x[T, hc, D] holds hc streams of the hidden state (x_flat = x.reshape(T, hc*D)). With the
projection hc_fn[mix_hc, hc*D], the per-group scales hc_scale[3] and biases hc_base[mix_hc]:

    inv_rms[T, 1]   = rsqrt(mean_k(x_flat^2) + eps)
    mixes[T, mix_hc]= inv_rms * (x_flat @ hc_fn^T)              # 1/rms deferred past the matmul
    pre [T, hc]     = sigmoid(mixes[:, :hc]   * s0 + base) + eps
    post[T, hc]     = 2 * sigmoid(mixes[:, hc:2hc] * s1 + base)
    comb[T, hc, hc] = sinkhorn(softmax(reshape(mixes[:, 2hc:] * s2 + base, hc, hc)))
    x_mixed[T, D]   = sum_h pre[:, h] * x[:, h, :]

Each work-type is its own pl.spmd task (rms / linear / linear_reduce / split_pre_post /
comb_sinkhorn / mix_x), ordered by their GM read/write dependencies. mix_hc(24) pads to a
32-wide cube N (MIX_PAD) and hc(4) to an 8-wide vector row (HC_PAD).
"""

import pypto.language as pl
from config import FLASH as M
from hc_pre_numerics import hc_sigmoid_math, hc_softmax_exp_math
from decode_shapes import DECODE_BATCH, DECODE_SEQ

T_DYN = pl.dynamic("T_DYN")
D = M.hidden_size
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
NORM_EPS = M.rms_norm_eps
MIX_PAD = 32
HC_PAD = 8
T_TILE = 8
LINEAR_T_TILE = 16
COMB_T_TILE = 8
RMS_K_TILE = 512
LINEAR_K_TILE = 256
D_TILE = 256
D_SPMD = 1024
LINEAR_OK = 8
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK
assert HC_MULT == 4, f"hc_pre is specialized to HC_MULT == 4, got {HC_MULT}"


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
    """One pl.spmd task per work-type, ordered by their GM read/write dependencies.

    rms -> linear -> linear_reduce -> split_pre_post / comb_sinkhorn / mix_x. Cross-scope
    buffers are sized to t_linear, the token count padded up to whole 16-row cube tiles.
    """
    t_dim = pl.tensor.dim(x, 0)
    t_linear = (t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE * LINEAR_T_TILE
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    scale0 = pl.read(hc_scale, [0])
    scale1 = pl.read(hc_scale, [1])
    scale2 = pl.read(hc_scale, [2])
    hc_base_2d = pl.reshape(hc_base, [1, MIX_HC])
    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    for t in pl.spmd((t_dim + T_TILE - 1) // T_TILE, name_hint="hc_pre_rms", allow_early_resolve=True):
        t0 = t * T_TILE
        rms_rows = pl.min(T_TILE, t_dim - t0)
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HC_DIM // RMS_K_TILE, stage=4):
            k0 = kb * RMS_K_TILE
            x_chunk = pl.slice(x_flat, [T_TILE, RMS_K_TILE], [t0, k0], valid_shape=[rms_rows, RMS_K_TILE])
            x_sq = pl.mul(x_chunk, x_chunk)
            # Preserve each FP32 addition of the reference's adjacent-pair tree.
            hc_rms_sq256_even = pl.gather(x_sq, mask_pattern=1)
            hc_rms_sq256_odd = pl.gather(x_sq, mask_pattern=2)
            hc_rms_sq256 = pl.add(hc_rms_sq256_even, hc_rms_sq256_odd)
            hc_rms_sq128_even = pl.gather(hc_rms_sq256, mask_pattern=1)
            hc_rms_sq128_odd = pl.gather(hc_rms_sq256, mask_pattern=2)
            hc_rms_sq128 = pl.add(hc_rms_sq128_even, hc_rms_sq128_odd)
            hc_rms_sq64_even = pl.gather(hc_rms_sq128, mask_pattern=1)
            hc_rms_sq64_odd = pl.gather(hc_rms_sq128, mask_pattern=2)
            hc_rms_sq64 = pl.add(hc_rms_sq64_even, hc_rms_sq64_odd)
            hc_rms_sq32_even = pl.gather(hc_rms_sq64, mask_pattern=1)
            hc_rms_sq32_odd = pl.gather(hc_rms_sq64, mask_pattern=2)
            hc_rms_sq32 = pl.add(hc_rms_sq32_even, hc_rms_sq32_odd)
            hc_rms_sq16_even = pl.gather(hc_rms_sq32, mask_pattern=1)
            hc_rms_sq16_odd = pl.gather(hc_rms_sq32, mask_pattern=2)
            hc_rms_sq16 = pl.add(hc_rms_sq16_even, hc_rms_sq16_odd)
            hc_rms_sq8_even = pl.gather(hc_rms_sq16, mask_pattern=1)
            hc_rms_sq8_odd = pl.gather(hc_rms_sq16, mask_pattern=2)
            hc_rms_sq8 = pl.add(hc_rms_sq8_even, hc_rms_sq8_odd)
            # Keep eight FP32 columns (32 bytes) for the last three levels.
            hc_rms_sq8_t = pl.transpose(hc_rms_sq8, 0, 1)
            hc_rms_pair0 = pl.add(hc_rms_sq8_t[0:1, :], hc_rms_sq8_t[1:2, :])
            hc_rms_pair1 = pl.add(hc_rms_sq8_t[2:3, :], hc_rms_sq8_t[3:4, :])
            hc_rms_pair2 = pl.add(hc_rms_sq8_t[4:5, :], hc_rms_sq8_t[5:6, :])
            hc_rms_pair3 = pl.add(hc_rms_sq8_t[6:7, :], hc_rms_sq8_t[7:8, :])
            hc_rms_sum_first = pl.add(hc_rms_pair0, hc_rms_pair1)
            hc_rms_sum_last = pl.add(hc_rms_pair2, hc_rms_pair3)
            x_sq_row = pl.add(hc_rms_sum_first, hc_rms_sum_last)
            sq_sum = pl.add(sq_sum, x_sq_row)
        sq_mean = sq_sum
        sq_mean = pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS)
        inv_seed = sq_mean
        inv_seed = pl.rsqrt(sq_mean, high_precision=True)
        rms_work = pl.minimum(sq_mean, 1099511627776.0)
        rms_seed = pl.rsqrt(rms_work, high_precision=True)
        rms_y2_as = pl.mul(rms_seed, 4097.0)
        rms_y2_ad = pl.sub(rms_y2_as, rms_seed)
        rms_y2_ah = pl.sub(rms_y2_as, rms_y2_ad)
        rms_y2_al = pl.sub(rms_seed, rms_y2_ah)
        rms_y2_bs = pl.mul(rms_seed, 4097.0)
        rms_y2_bd = pl.sub(rms_y2_bs, rms_seed)
        rms_y2_bh = pl.sub(rms_y2_bs, rms_y2_bd)
        rms_y2_bl = pl.sub(rms_seed, rms_y2_bh)
        rms_y2_p = pl.mul(rms_seed, rms_seed)
        rms_y2_hh = pl.mul(rms_y2_ah, rms_y2_bh)
        rms_y2_e0 = pl.sub(rms_y2_hh, rms_y2_p)
        rms_y2_hl = pl.mul(rms_y2_ah, rms_y2_bl)
        rms_y2_e1 = pl.add(rms_y2_e0, rms_y2_hl)
        rms_y2_lh = pl.mul(rms_y2_al, rms_y2_bh)
        rms_y2_e2 = pl.add(rms_y2_e1, rms_y2_lh)
        rms_y2_ll = pl.mul(rms_y2_al, rms_y2_bl)
        rms_y2_e = pl.add(rms_y2_e2, rms_y2_ll)
        rms_xy2_as = pl.mul(rms_work, 4097.0)
        rms_xy2_ad = pl.sub(rms_xy2_as, rms_work)
        rms_xy2_ah = pl.sub(rms_xy2_as, rms_xy2_ad)
        rms_xy2_al = pl.sub(rms_work, rms_xy2_ah)
        rms_xy2_bs = pl.mul(rms_y2_p, 4097.0)
        rms_xy2_bd = pl.sub(rms_xy2_bs, rms_y2_p)
        rms_xy2_bh = pl.sub(rms_xy2_bs, rms_xy2_bd)
        rms_xy2_bl = pl.sub(rms_y2_p, rms_xy2_bh)
        rms_xy2_p = pl.mul(rms_work, rms_y2_p)
        rms_xy2_hh = pl.mul(rms_xy2_ah, rms_xy2_bh)
        rms_xy2_e0 = pl.sub(rms_xy2_hh, rms_xy2_p)
        rms_xy2_hl = pl.mul(rms_xy2_ah, rms_xy2_bl)
        rms_xy2_e1 = pl.add(rms_xy2_e0, rms_xy2_hl)
        rms_xy2_lh = pl.mul(rms_xy2_al, rms_xy2_bh)
        rms_xy2_e2 = pl.add(rms_xy2_e1, rms_xy2_lh)
        rms_xy2_ll = pl.mul(rms_xy2_al, rms_xy2_bl)
        rms_xy2_e = pl.add(rms_xy2_e2, rms_xy2_ll)
        rms_one = pl.full([1, T_TILE], dtype=pl.FP32, value=1.0)
        rms_r0 = pl.sub(rms_one, rms_xy2_p)
        rms_r1 = pl.sub(rms_r0, rms_xy2_e)
        rms_low = pl.mul(rms_work, rms_y2_e)
        rms_residual = pl.sub(rms_r1, rms_low)
        rms_half_seed = pl.mul(rms_seed, 0.5)
        rms_step = pl.mul(rms_half_seed, rms_residual)
        rms_corrected = pl.add(rms_seed, rms_step)
        rms_overflow = pl.sub(sq_mean, 1099511627776.0)
        rms_gate = pl.minimum(pl.maximum(pl.mul(rms_overflow, 7.62939453125e-06), 0.0), 1.0)
        rms_inside = pl.sub(rms_one, rms_gate)
        rms_value = pl.add(pl.mul(rms_corrected, rms_inside), pl.mul(inv_seed, rms_gate))
        inv = pl.reshape(rms_value, [T_TILE, 1])
        inv_valid = pl.set_validshape(inv, rms_rows, 1)
        inv_rms[t0 : t0 + T_TILE, 0:1] = inv_valid
    mixes_partials = pl.create_tensor([LINEAR_OK * t_linear, MIX_PAD], dtype=pl.FP32)
    for task in pl.spmd(
        t_linear // LINEAR_T_TILE * LINEAR_OK, name_hint="hc_pre_linear", allow_early_resolve=True
    ):
        t0 = task // LINEAR_OK * LINEAR_T_TILE
        linear_split = task % LINEAR_OK
        k_base = linear_split * LINEAR_K_PER_SPLIT
        t_rows = pl.min(LINEAR_T_TILE, t_dim - t0)
        acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
        for kb in pl.pipeline(0, LINEAR_K_PER_SPLIT // LINEAR_K_TILE, stage=2):
            k0 = k_base + kb * LINEAR_K_TILE
            x_linear_chunk = pl.slice(
                x_flat, [LINEAR_T_TILE, LINEAR_K_TILE], [t0, k0], valid_shape=[t_rows, LINEAR_K_TILE]
            )
            w_chunk = pl.slice(hc_fn, [MIX_PAD, LINEAR_K_TILE], [0, k0], valid_shape=[MIX_HC, LINEAR_K_TILE])
            acc = pl.matmul_acc(acc, x_linear_chunk, w_chunk, b_trans=True, init_cond=kb == 0)
        partial_row0 = linear_split * t_linear + t0
        mixes_partials[partial_row0 : partial_row0 + LINEAR_T_TILE, 0:MIX_PAD] = acc
    mixes_raw = pl.create_tensor([t_linear, MIX_PAD], dtype=pl.FP32)
    for linear_block in pl.spmd(
        t_linear // LINEAR_T_TILE, name_hint="hc_pre_linear_reduce", allow_early_resolve=True
    ):
        linear_t0 = linear_block * LINEAR_T_TILE
        reduce_rows = pl.min(LINEAR_T_TILE, t_dim - linear_t0)
        mixes_total = pl.slice(
            mixes_partials, [LINEAR_T_TILE, MIX_PAD], [linear_t0, 0], valid_shape=[reduce_rows, MIX_HC]
        )
        for linear_split in pl.range(1, LINEAR_OK):
            partial_t0 = linear_split * t_linear + linear_t0
            partial_tile = pl.slice(
                mixes_partials, [LINEAR_T_TILE, MIX_PAD], [partial_t0, 0], valid_shape=[reduce_rows, MIX_HC]
            )
            mixes_total = pl.add(mixes_total, partial_tile)
        reduced_valid = pl.set_validshape(mixes_total, reduce_rows, MIX_HC)
        mixes_raw[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD] = reduced_valid
    pre_val_store = pl.create_tensor([t_linear, HC_PAD], dtype=pl.FP32)
    for ob in pl.spmd((t_dim + T_TILE - 1) // T_TILE, name_hint="split_pre_post", allow_early_resolve=True):
        t0 = ob * T_TILE
        split_rows = pl.min(T_TILE, t_dim - t0)
        inv_col = pl.slice(inv_rms, [T_TILE, 1], [t0, 0], valid_shape=[split_rows, 1])
        pre_mix = pl.slice(mixes_raw, [T_TILE, HC_PAD], [t0, 0], valid_shape=[split_rows, HC_PAD])
        pre_sig = hc_sigmoid_math(pre_mix, inv_col, hc_scale, hc_base, group=0, offset=0)
        pre_val = pl.add(pre_sig, HC_EPS)
        pre_out = pl.set_validshape(pre_val, split_rows, HC_PAD)
        pre_val_store[t0 : t0 + T_TILE, 0:HC_PAD] = pre_out
        post_mix = pl.slice(mixes_raw, [T_TILE, HC_PAD], [t0, HC_MULT], valid_shape=[split_rows, HC_PAD])
        post_sig = hc_sigmoid_math(post_mix, inv_col, hc_scale, hc_base, group=1, offset=4)
        post_pad = pl.mul(post_sig, 2.0)
        post_out = pl.slice(post_pad, [T_TILE, HC_PAD], [0, 0], valid_shape=[split_rows, HC_MULT])
        post[t0 : t0 + T_TILE, 0:HC_MULT] = post_out
    for ob in pl.spmd(
        (t_dim + COMB_T_TILE - 1) // COMB_T_TILE, name_hint="comb_sinkhorn", allow_early_resolve=True
    ):
        t0 = ob * COMB_T_TILE
        comb_rows = pl.min(COMB_T_TILE, t_dim - t0)
        inv_col_t = pl.slice(inv_rms, [COMB_T_TILE, 1], [t0, 0], valid_shape=[comb_rows, 1])
        exp_rows = pl.full([HC_MULT * COMB_T_TILE, HC_PAD], dtype=pl.FP32, value=0.0)
        for hc_row in pl.range(HC_MULT):
            mix_offset = 2 * HC_MULT + hc_row * HC_MULT
            mix_g = pl.slice(mixes_raw, [COMB_T_TILE, HC_PAD], [t0, mix_offset], valid_shape=[comb_rows, HC_MULT])
            row_exp = hc_softmax_exp_math(mix_g, inv_col_t, hc_scale, hc_base, group=2, offset=mix_offset)
            row_exp = pl.set_validshape(row_exp, COMB_T_TILE, HC_MULT)
            row_exp = pl.fillpad(row_exp, pad_value=pl.PadValue.zero)
            row_exp = pl.set_validshape(row_exp, COMB_T_TILE, HC_PAD)
            # Materialize a dense tile before local assembly.
            row_zero = pl.full([COMB_T_TILE, HC_PAD], dtype=pl.FP32, value=0.0)
            row_exp = pl.add(row_zero, row_exp)
            row_start = hc_row * COMB_T_TILE
            exp_rows[row_start : row_start + COMB_T_TILE, :] = row_exp
        row0_exp = pl.slice(exp_rows, [COMB_T_TILE, HC_PAD], [0, 0])
        row1_exp = pl.slice(exp_rows, [COMB_T_TILE, HC_PAD], [COMB_T_TILE, 0])
        row2_exp = pl.slice(exp_rows, [COMB_T_TILE, HC_PAD], [2 * COMB_T_TILE, 0])
        row3_exp = pl.slice(exp_rows, [COMB_T_TILE, HC_PAD], [3 * COMB_T_TILE, 0])
        row0_sum = pl.row_sum(row0_exp)
        row1_sum = pl.row_sum(row1_exp)
        row2_sum = pl.row_sum(row2_exp)
        row3_sum = pl.row_sum(row3_exp)
        hc_div1_zero = pl.sub(row0_exp, row0_exp)
        hc_div1_den = pl.row_expand_add(hc_div1_zero, row0_sum)
        hc_div1_q = pl.div(row0_exp, hc_div1_den)
        hc_div1_as = pl.mul(hc_div1_q, 4097.0)
        hc_div1_ad = pl.sub(hc_div1_as, hc_div1_q)
        hc_div1_ah = pl.sub(hc_div1_as, hc_div1_ad)
        hc_div1_al = pl.sub(hc_div1_q, hc_div1_ah)
        hc_div1_bs = pl.mul(hc_div1_den, 4097.0)
        hc_div1_bd = pl.sub(hc_div1_bs, hc_div1_den)
        hc_div1_bh = pl.sub(hc_div1_bs, hc_div1_bd)
        hc_div1_bl = pl.sub(hc_div1_den, hc_div1_bh)
        hc_div1_prod = pl.mul(hc_div1_q, hc_div1_den)
        hc_div1_hh = pl.mul(hc_div1_ah, hc_div1_bh)
        hc_div1_e0 = pl.sub(hc_div1_hh, hc_div1_prod)
        hc_div1_hl = pl.mul(hc_div1_ah, hc_div1_bl)
        hc_div1_e1 = pl.add(hc_div1_e0, hc_div1_hl)
        hc_div1_lh = pl.mul(hc_div1_al, hc_div1_bh)
        hc_div1_e2 = pl.add(hc_div1_e1, hc_div1_lh)
        hc_div1_ll = pl.mul(hc_div1_al, hc_div1_bl)
        hc_div1_error = pl.add(hc_div1_e2, hc_div1_ll)
        hc_div1_r0 = pl.sub(row0_exp, hc_div1_prod)
        hc_div1_residual = pl.sub(hc_div1_r0, hc_div1_error)
        hc_div1_correction = pl.div(hc_div1_residual, hc_div1_den)
        hc_div1_fixed = pl.add(hc_div1_q, hc_div1_correction)
        row0_soft = pl.add(hc_div1_fixed, HC_EPS)
        hc_div2_zero = pl.sub(row1_exp, row1_exp)
        hc_div2_den = pl.row_expand_add(hc_div2_zero, row1_sum)
        hc_div2_q = pl.div(row1_exp, hc_div2_den)
        hc_div2_as = pl.mul(hc_div2_q, 4097.0)
        hc_div2_ad = pl.sub(hc_div2_as, hc_div2_q)
        hc_div2_ah = pl.sub(hc_div2_as, hc_div2_ad)
        hc_div2_al = pl.sub(hc_div2_q, hc_div2_ah)
        hc_div2_bs = pl.mul(hc_div2_den, 4097.0)
        hc_div2_bd = pl.sub(hc_div2_bs, hc_div2_den)
        hc_div2_bh = pl.sub(hc_div2_bs, hc_div2_bd)
        hc_div2_bl = pl.sub(hc_div2_den, hc_div2_bh)
        hc_div2_prod = pl.mul(hc_div2_q, hc_div2_den)
        hc_div2_hh = pl.mul(hc_div2_ah, hc_div2_bh)
        hc_div2_e0 = pl.sub(hc_div2_hh, hc_div2_prod)
        hc_div2_hl = pl.mul(hc_div2_ah, hc_div2_bl)
        hc_div2_e1 = pl.add(hc_div2_e0, hc_div2_hl)
        hc_div2_lh = pl.mul(hc_div2_al, hc_div2_bh)
        hc_div2_e2 = pl.add(hc_div2_e1, hc_div2_lh)
        hc_div2_ll = pl.mul(hc_div2_al, hc_div2_bl)
        hc_div2_error = pl.add(hc_div2_e2, hc_div2_ll)
        hc_div2_r0 = pl.sub(row1_exp, hc_div2_prod)
        hc_div2_residual = pl.sub(hc_div2_r0, hc_div2_error)
        hc_div2_correction = pl.div(hc_div2_residual, hc_div2_den)
        hc_div2_fixed = pl.add(hc_div2_q, hc_div2_correction)
        row1_soft = pl.add(hc_div2_fixed, HC_EPS)
        hc_div3_zero = pl.sub(row2_exp, row2_exp)
        hc_div3_den = pl.row_expand_add(hc_div3_zero, row2_sum)
        hc_div3_q = pl.div(row2_exp, hc_div3_den)
        hc_div3_as = pl.mul(hc_div3_q, 4097.0)
        hc_div3_ad = pl.sub(hc_div3_as, hc_div3_q)
        hc_div3_ah = pl.sub(hc_div3_as, hc_div3_ad)
        hc_div3_al = pl.sub(hc_div3_q, hc_div3_ah)
        hc_div3_bs = pl.mul(hc_div3_den, 4097.0)
        hc_div3_bd = pl.sub(hc_div3_bs, hc_div3_den)
        hc_div3_bh = pl.sub(hc_div3_bs, hc_div3_bd)
        hc_div3_bl = pl.sub(hc_div3_den, hc_div3_bh)
        hc_div3_prod = pl.mul(hc_div3_q, hc_div3_den)
        hc_div3_hh = pl.mul(hc_div3_ah, hc_div3_bh)
        hc_div3_e0 = pl.sub(hc_div3_hh, hc_div3_prod)
        hc_div3_hl = pl.mul(hc_div3_ah, hc_div3_bl)
        hc_div3_e1 = pl.add(hc_div3_e0, hc_div3_hl)
        hc_div3_lh = pl.mul(hc_div3_al, hc_div3_bh)
        hc_div3_e2 = pl.add(hc_div3_e1, hc_div3_lh)
        hc_div3_ll = pl.mul(hc_div3_al, hc_div3_bl)
        hc_div3_error = pl.add(hc_div3_e2, hc_div3_ll)
        hc_div3_r0 = pl.sub(row2_exp, hc_div3_prod)
        hc_div3_residual = pl.sub(hc_div3_r0, hc_div3_error)
        hc_div3_correction = pl.div(hc_div3_residual, hc_div3_den)
        hc_div3_fixed = pl.add(hc_div3_q, hc_div3_correction)
        row2_soft = pl.add(hc_div3_fixed, HC_EPS)
        hc_div4_zero = pl.sub(row3_exp, row3_exp)
        hc_div4_den = pl.row_expand_add(hc_div4_zero, row3_sum)
        hc_div4_q = pl.div(row3_exp, hc_div4_den)
        hc_div4_as = pl.mul(hc_div4_q, 4097.0)
        hc_div4_ad = pl.sub(hc_div4_as, hc_div4_q)
        hc_div4_ah = pl.sub(hc_div4_as, hc_div4_ad)
        hc_div4_al = pl.sub(hc_div4_q, hc_div4_ah)
        hc_div4_bs = pl.mul(hc_div4_den, 4097.0)
        hc_div4_bd = pl.sub(hc_div4_bs, hc_div4_den)
        hc_div4_bh = pl.sub(hc_div4_bs, hc_div4_bd)
        hc_div4_bl = pl.sub(hc_div4_den, hc_div4_bh)
        hc_div4_prod = pl.mul(hc_div4_q, hc_div4_den)
        hc_div4_hh = pl.mul(hc_div4_ah, hc_div4_bh)
        hc_div4_e0 = pl.sub(hc_div4_hh, hc_div4_prod)
        hc_div4_hl = pl.mul(hc_div4_ah, hc_div4_bl)
        hc_div4_e1 = pl.add(hc_div4_e0, hc_div4_hl)
        hc_div4_lh = pl.mul(hc_div4_al, hc_div4_bh)
        hc_div4_e2 = pl.add(hc_div4_e1, hc_div4_lh)
        hc_div4_ll = pl.mul(hc_div4_al, hc_div4_bl)
        hc_div4_error = pl.add(hc_div4_e2, hc_div4_ll)
        hc_div4_r0 = pl.sub(row3_exp, hc_div4_prod)
        hc_div4_residual = pl.sub(hc_div4_r0, hc_div4_error)
        hc_div4_correction = pl.div(hc_div4_residual, hc_div4_den)
        hc_div4_fixed = pl.add(hc_div4_q, hc_div4_correction)
        row3_soft = pl.add(hc_div4_fixed, HC_EPS)
        row0_valid = pl.set_validshape(row0_soft, comb_rows, HC_MULT)
        row1_valid = pl.set_validshape(row1_soft, comb_rows, HC_MULT)
        row2_valid = pl.set_validshape(row2_soft, comb_rows, HC_MULT)
        row3_valid = pl.set_validshape(row3_soft, comb_rows, HC_MULT)
        row0_eff = pl.fillpad(row0_valid, pad_value=pl.PadValue.zero)
        row1_eff = pl.fillpad(row1_valid, pad_value=pl.PadValue.zero)
        row2_eff = pl.fillpad(row2_valid, pad_value=pl.PadValue.zero)
        row3_eff = pl.fillpad(row3_valid, pad_value=pl.PadValue.zero)
        col_sum = pl.add(pl.add(row0_eff, row1_eff), pl.add(row2_eff, row3_eff))
        col_sum = pl.add(col_sum, HC_EPS)
        hc_div5_q = pl.div(row0_eff, col_sum)
        hc_div5_as = pl.mul(hc_div5_q, 4097.0)
        hc_div5_ad = pl.sub(hc_div5_as, hc_div5_q)
        hc_div5_ah = pl.sub(hc_div5_as, hc_div5_ad)
        hc_div5_al = pl.sub(hc_div5_q, hc_div5_ah)
        hc_div5_bs = pl.mul(col_sum, 4097.0)
        hc_div5_bd = pl.sub(hc_div5_bs, col_sum)
        hc_div5_bh = pl.sub(hc_div5_bs, hc_div5_bd)
        hc_div5_bl = pl.sub(col_sum, hc_div5_bh)
        hc_div5_prod = pl.mul(hc_div5_q, col_sum)
        hc_div5_hh = pl.mul(hc_div5_ah, hc_div5_bh)
        hc_div5_e0 = pl.sub(hc_div5_hh, hc_div5_prod)
        hc_div5_hl = pl.mul(hc_div5_ah, hc_div5_bl)
        hc_div5_e1 = pl.add(hc_div5_e0, hc_div5_hl)
        hc_div5_lh = pl.mul(hc_div5_al, hc_div5_bh)
        hc_div5_e2 = pl.add(hc_div5_e1, hc_div5_lh)
        hc_div5_ll = pl.mul(hc_div5_al, hc_div5_bl)
        hc_div5_error = pl.add(hc_div5_e2, hc_div5_ll)
        hc_div5_r0 = pl.sub(row0_eff, hc_div5_prod)
        hc_div5_residual = pl.sub(hc_div5_r0, hc_div5_error)
        hc_div5_correction = pl.div(hc_div5_residual, col_sum)
        hc_div5_fixed = pl.add(hc_div5_q, hc_div5_correction)
        row0_cur = hc_div5_fixed
        hc_div6_q = pl.div(row1_eff, col_sum)
        hc_div6_as = pl.mul(hc_div6_q, 4097.0)
        hc_div6_ad = pl.sub(hc_div6_as, hc_div6_q)
        hc_div6_ah = pl.sub(hc_div6_as, hc_div6_ad)
        hc_div6_al = pl.sub(hc_div6_q, hc_div6_ah)
        hc_div6_bs = pl.mul(col_sum, 4097.0)
        hc_div6_bd = pl.sub(hc_div6_bs, col_sum)
        hc_div6_bh = pl.sub(hc_div6_bs, hc_div6_bd)
        hc_div6_bl = pl.sub(col_sum, hc_div6_bh)
        hc_div6_prod = pl.mul(hc_div6_q, col_sum)
        hc_div6_hh = pl.mul(hc_div6_ah, hc_div6_bh)
        hc_div6_e0 = pl.sub(hc_div6_hh, hc_div6_prod)
        hc_div6_hl = pl.mul(hc_div6_ah, hc_div6_bl)
        hc_div6_e1 = pl.add(hc_div6_e0, hc_div6_hl)
        hc_div6_lh = pl.mul(hc_div6_al, hc_div6_bh)
        hc_div6_e2 = pl.add(hc_div6_e1, hc_div6_lh)
        hc_div6_ll = pl.mul(hc_div6_al, hc_div6_bl)
        hc_div6_error = pl.add(hc_div6_e2, hc_div6_ll)
        hc_div6_r0 = pl.sub(row1_eff, hc_div6_prod)
        hc_div6_residual = pl.sub(hc_div6_r0, hc_div6_error)
        hc_div6_correction = pl.div(hc_div6_residual, col_sum)
        hc_div6_fixed = pl.add(hc_div6_q, hc_div6_correction)
        row1_cur = hc_div6_fixed
        hc_div7_q = pl.div(row2_eff, col_sum)
        hc_div7_as = pl.mul(hc_div7_q, 4097.0)
        hc_div7_ad = pl.sub(hc_div7_as, hc_div7_q)
        hc_div7_ah = pl.sub(hc_div7_as, hc_div7_ad)
        hc_div7_al = pl.sub(hc_div7_q, hc_div7_ah)
        hc_div7_bs = pl.mul(col_sum, 4097.0)
        hc_div7_bd = pl.sub(hc_div7_bs, col_sum)
        hc_div7_bh = pl.sub(hc_div7_bs, hc_div7_bd)
        hc_div7_bl = pl.sub(col_sum, hc_div7_bh)
        hc_div7_prod = pl.mul(hc_div7_q, col_sum)
        hc_div7_hh = pl.mul(hc_div7_ah, hc_div7_bh)
        hc_div7_e0 = pl.sub(hc_div7_hh, hc_div7_prod)
        hc_div7_hl = pl.mul(hc_div7_ah, hc_div7_bl)
        hc_div7_e1 = pl.add(hc_div7_e0, hc_div7_hl)
        hc_div7_lh = pl.mul(hc_div7_al, hc_div7_bh)
        hc_div7_e2 = pl.add(hc_div7_e1, hc_div7_lh)
        hc_div7_ll = pl.mul(hc_div7_al, hc_div7_bl)
        hc_div7_error = pl.add(hc_div7_e2, hc_div7_ll)
        hc_div7_r0 = pl.sub(row2_eff, hc_div7_prod)
        hc_div7_residual = pl.sub(hc_div7_r0, hc_div7_error)
        hc_div7_correction = pl.div(hc_div7_residual, col_sum)
        hc_div7_fixed = pl.add(hc_div7_q, hc_div7_correction)
        row2_cur = hc_div7_fixed
        hc_div8_q = pl.div(row3_eff, col_sum)
        hc_div8_as = pl.mul(hc_div8_q, 4097.0)
        hc_div8_ad = pl.sub(hc_div8_as, hc_div8_q)
        hc_div8_ah = pl.sub(hc_div8_as, hc_div8_ad)
        hc_div8_al = pl.sub(hc_div8_q, hc_div8_ah)
        hc_div8_bs = pl.mul(col_sum, 4097.0)
        hc_div8_bd = pl.sub(hc_div8_bs, col_sum)
        hc_div8_bh = pl.sub(hc_div8_bs, hc_div8_bd)
        hc_div8_bl = pl.sub(col_sum, hc_div8_bh)
        hc_div8_prod = pl.mul(hc_div8_q, col_sum)
        hc_div8_hh = pl.mul(hc_div8_ah, hc_div8_bh)
        hc_div8_e0 = pl.sub(hc_div8_hh, hc_div8_prod)
        hc_div8_hl = pl.mul(hc_div8_ah, hc_div8_bl)
        hc_div8_e1 = pl.add(hc_div8_e0, hc_div8_hl)
        hc_div8_lh = pl.mul(hc_div8_al, hc_div8_bh)
        hc_div8_e2 = pl.add(hc_div8_e1, hc_div8_lh)
        hc_div8_ll = pl.mul(hc_div8_al, hc_div8_bl)
        hc_div8_error = pl.add(hc_div8_e2, hc_div8_ll)
        hc_div8_r0 = pl.sub(row3_eff, hc_div8_prod)
        hc_div8_residual = pl.sub(hc_div8_r0, hc_div8_error)
        hc_div8_correction = pl.div(hc_div8_residual, col_sum)
        hc_div8_fixed = pl.add(hc_div8_q, hc_div8_correction)
        row3_cur = hc_div8_fixed
        for _sk_it in pl.pipeline(HC_SINKHORN_ITER - 1, stage=2):
            row0_rowsum = pl.add(pl.row_sum(row0_cur), HC_EPS)
            row1_rowsum = pl.add(pl.row_sum(row1_cur), HC_EPS)
            row2_rowsum = pl.add(pl.row_sum(row2_cur), HC_EPS)
            row3_rowsum = pl.add(pl.row_sum(row3_cur), HC_EPS)
            hc_div9_zero = pl.sub(row0_cur, row0_cur)
            hc_div9_den = pl.row_expand_add(hc_div9_zero, row0_rowsum)
            hc_div9_q = pl.div(row0_cur, hc_div9_den)
            hc_div9_as = pl.mul(hc_div9_q, 4097.0)
            hc_div9_ad = pl.sub(hc_div9_as, hc_div9_q)
            hc_div9_ah = pl.sub(hc_div9_as, hc_div9_ad)
            hc_div9_al = pl.sub(hc_div9_q, hc_div9_ah)
            hc_div9_bs = pl.mul(hc_div9_den, 4097.0)
            hc_div9_bd = pl.sub(hc_div9_bs, hc_div9_den)
            hc_div9_bh = pl.sub(hc_div9_bs, hc_div9_bd)
            hc_div9_bl = pl.sub(hc_div9_den, hc_div9_bh)
            hc_div9_prod = pl.mul(hc_div9_q, hc_div9_den)
            hc_div9_hh = pl.mul(hc_div9_ah, hc_div9_bh)
            hc_div9_e0 = pl.sub(hc_div9_hh, hc_div9_prod)
            hc_div9_hl = pl.mul(hc_div9_ah, hc_div9_bl)
            hc_div9_e1 = pl.add(hc_div9_e0, hc_div9_hl)
            hc_div9_lh = pl.mul(hc_div9_al, hc_div9_bh)
            hc_div9_e2 = pl.add(hc_div9_e1, hc_div9_lh)
            hc_div9_ll = pl.mul(hc_div9_al, hc_div9_bl)
            hc_div9_error = pl.add(hc_div9_e2, hc_div9_ll)
            hc_div9_r0 = pl.sub(row0_cur, hc_div9_prod)
            hc_div9_residual = pl.sub(hc_div9_r0, hc_div9_error)
            hc_div9_correction = pl.div(hc_div9_residual, hc_div9_den)
            hc_div9_fixed = pl.add(hc_div9_q, hc_div9_correction)
            row0_norm = hc_div9_fixed
            hc_div10_zero = pl.sub(row1_cur, row1_cur)
            hc_div10_den = pl.row_expand_add(hc_div10_zero, row1_rowsum)
            hc_div10_q = pl.div(row1_cur, hc_div10_den)
            hc_div10_as = pl.mul(hc_div10_q, 4097.0)
            hc_div10_ad = pl.sub(hc_div10_as, hc_div10_q)
            hc_div10_ah = pl.sub(hc_div10_as, hc_div10_ad)
            hc_div10_al = pl.sub(hc_div10_q, hc_div10_ah)
            hc_div10_bs = pl.mul(hc_div10_den, 4097.0)
            hc_div10_bd = pl.sub(hc_div10_bs, hc_div10_den)
            hc_div10_bh = pl.sub(hc_div10_bs, hc_div10_bd)
            hc_div10_bl = pl.sub(hc_div10_den, hc_div10_bh)
            hc_div10_prod = pl.mul(hc_div10_q, hc_div10_den)
            hc_div10_hh = pl.mul(hc_div10_ah, hc_div10_bh)
            hc_div10_e0 = pl.sub(hc_div10_hh, hc_div10_prod)
            hc_div10_hl = pl.mul(hc_div10_ah, hc_div10_bl)
            hc_div10_e1 = pl.add(hc_div10_e0, hc_div10_hl)
            hc_div10_lh = pl.mul(hc_div10_al, hc_div10_bh)
            hc_div10_e2 = pl.add(hc_div10_e1, hc_div10_lh)
            hc_div10_ll = pl.mul(hc_div10_al, hc_div10_bl)
            hc_div10_error = pl.add(hc_div10_e2, hc_div10_ll)
            hc_div10_r0 = pl.sub(row1_cur, hc_div10_prod)
            hc_div10_residual = pl.sub(hc_div10_r0, hc_div10_error)
            hc_div10_correction = pl.div(hc_div10_residual, hc_div10_den)
            hc_div10_fixed = pl.add(hc_div10_q, hc_div10_correction)
            row1_norm = hc_div10_fixed
            hc_div11_zero = pl.sub(row2_cur, row2_cur)
            hc_div11_den = pl.row_expand_add(hc_div11_zero, row2_rowsum)
            hc_div11_q = pl.div(row2_cur, hc_div11_den)
            hc_div11_as = pl.mul(hc_div11_q, 4097.0)
            hc_div11_ad = pl.sub(hc_div11_as, hc_div11_q)
            hc_div11_ah = pl.sub(hc_div11_as, hc_div11_ad)
            hc_div11_al = pl.sub(hc_div11_q, hc_div11_ah)
            hc_div11_bs = pl.mul(hc_div11_den, 4097.0)
            hc_div11_bd = pl.sub(hc_div11_bs, hc_div11_den)
            hc_div11_bh = pl.sub(hc_div11_bs, hc_div11_bd)
            hc_div11_bl = pl.sub(hc_div11_den, hc_div11_bh)
            hc_div11_prod = pl.mul(hc_div11_q, hc_div11_den)
            hc_div11_hh = pl.mul(hc_div11_ah, hc_div11_bh)
            hc_div11_e0 = pl.sub(hc_div11_hh, hc_div11_prod)
            hc_div11_hl = pl.mul(hc_div11_ah, hc_div11_bl)
            hc_div11_e1 = pl.add(hc_div11_e0, hc_div11_hl)
            hc_div11_lh = pl.mul(hc_div11_al, hc_div11_bh)
            hc_div11_e2 = pl.add(hc_div11_e1, hc_div11_lh)
            hc_div11_ll = pl.mul(hc_div11_al, hc_div11_bl)
            hc_div11_error = pl.add(hc_div11_e2, hc_div11_ll)
            hc_div11_r0 = pl.sub(row2_cur, hc_div11_prod)
            hc_div11_residual = pl.sub(hc_div11_r0, hc_div11_error)
            hc_div11_correction = pl.div(hc_div11_residual, hc_div11_den)
            hc_div11_fixed = pl.add(hc_div11_q, hc_div11_correction)
            row2_norm = hc_div11_fixed
            hc_div12_zero = pl.sub(row3_cur, row3_cur)
            hc_div12_den = pl.row_expand_add(hc_div12_zero, row3_rowsum)
            hc_div12_q = pl.div(row3_cur, hc_div12_den)
            hc_div12_as = pl.mul(hc_div12_q, 4097.0)
            hc_div12_ad = pl.sub(hc_div12_as, hc_div12_q)
            hc_div12_ah = pl.sub(hc_div12_as, hc_div12_ad)
            hc_div12_al = pl.sub(hc_div12_q, hc_div12_ah)
            hc_div12_bs = pl.mul(hc_div12_den, 4097.0)
            hc_div12_bd = pl.sub(hc_div12_bs, hc_div12_den)
            hc_div12_bh = pl.sub(hc_div12_bs, hc_div12_bd)
            hc_div12_bl = pl.sub(hc_div12_den, hc_div12_bh)
            hc_div12_prod = pl.mul(hc_div12_q, hc_div12_den)
            hc_div12_hh = pl.mul(hc_div12_ah, hc_div12_bh)
            hc_div12_e0 = pl.sub(hc_div12_hh, hc_div12_prod)
            hc_div12_hl = pl.mul(hc_div12_ah, hc_div12_bl)
            hc_div12_e1 = pl.add(hc_div12_e0, hc_div12_hl)
            hc_div12_lh = pl.mul(hc_div12_al, hc_div12_bh)
            hc_div12_e2 = pl.add(hc_div12_e1, hc_div12_lh)
            hc_div12_ll = pl.mul(hc_div12_al, hc_div12_bl)
            hc_div12_error = pl.add(hc_div12_e2, hc_div12_ll)
            hc_div12_r0 = pl.sub(row3_cur, hc_div12_prod)
            hc_div12_residual = pl.sub(hc_div12_r0, hc_div12_error)
            hc_div12_correction = pl.div(hc_div12_residual, hc_div12_den)
            hc_div12_fixed = pl.add(hc_div12_q, hc_div12_correction)
            row3_norm = hc_div12_fixed
            col_sum = pl.add(pl.add(row0_norm, row1_norm), pl.add(row2_norm, row3_norm))
            col_sum = pl.add(col_sum, HC_EPS)
            hc_div13_q = pl.div(row0_norm, col_sum)
            hc_div13_as = pl.mul(hc_div13_q, 4097.0)
            hc_div13_ad = pl.sub(hc_div13_as, hc_div13_q)
            hc_div13_ah = pl.sub(hc_div13_as, hc_div13_ad)
            hc_div13_al = pl.sub(hc_div13_q, hc_div13_ah)
            hc_div13_bs = pl.mul(col_sum, 4097.0)
            hc_div13_bd = pl.sub(hc_div13_bs, col_sum)
            hc_div13_bh = pl.sub(hc_div13_bs, hc_div13_bd)
            hc_div13_bl = pl.sub(col_sum, hc_div13_bh)
            hc_div13_prod = pl.mul(hc_div13_q, col_sum)
            hc_div13_hh = pl.mul(hc_div13_ah, hc_div13_bh)
            hc_div13_e0 = pl.sub(hc_div13_hh, hc_div13_prod)
            hc_div13_hl = pl.mul(hc_div13_ah, hc_div13_bl)
            hc_div13_e1 = pl.add(hc_div13_e0, hc_div13_hl)
            hc_div13_lh = pl.mul(hc_div13_al, hc_div13_bh)
            hc_div13_e2 = pl.add(hc_div13_e1, hc_div13_lh)
            hc_div13_ll = pl.mul(hc_div13_al, hc_div13_bl)
            hc_div13_error = pl.add(hc_div13_e2, hc_div13_ll)
            hc_div13_r0 = pl.sub(row0_norm, hc_div13_prod)
            hc_div13_residual = pl.sub(hc_div13_r0, hc_div13_error)
            hc_div13_correction = pl.div(hc_div13_residual, col_sum)
            hc_div13_fixed = pl.add(hc_div13_q, hc_div13_correction)
            row0_cur = hc_div13_fixed
            hc_div14_q = pl.div(row1_norm, col_sum)
            hc_div14_as = pl.mul(hc_div14_q, 4097.0)
            hc_div14_ad = pl.sub(hc_div14_as, hc_div14_q)
            hc_div14_ah = pl.sub(hc_div14_as, hc_div14_ad)
            hc_div14_al = pl.sub(hc_div14_q, hc_div14_ah)
            hc_div14_bs = pl.mul(col_sum, 4097.0)
            hc_div14_bd = pl.sub(hc_div14_bs, col_sum)
            hc_div14_bh = pl.sub(hc_div14_bs, hc_div14_bd)
            hc_div14_bl = pl.sub(col_sum, hc_div14_bh)
            hc_div14_prod = pl.mul(hc_div14_q, col_sum)
            hc_div14_hh = pl.mul(hc_div14_ah, hc_div14_bh)
            hc_div14_e0 = pl.sub(hc_div14_hh, hc_div14_prod)
            hc_div14_hl = pl.mul(hc_div14_ah, hc_div14_bl)
            hc_div14_e1 = pl.add(hc_div14_e0, hc_div14_hl)
            hc_div14_lh = pl.mul(hc_div14_al, hc_div14_bh)
            hc_div14_e2 = pl.add(hc_div14_e1, hc_div14_lh)
            hc_div14_ll = pl.mul(hc_div14_al, hc_div14_bl)
            hc_div14_error = pl.add(hc_div14_e2, hc_div14_ll)
            hc_div14_r0 = pl.sub(row1_norm, hc_div14_prod)
            hc_div14_residual = pl.sub(hc_div14_r0, hc_div14_error)
            hc_div14_correction = pl.div(hc_div14_residual, col_sum)
            hc_div14_fixed = pl.add(hc_div14_q, hc_div14_correction)
            row1_cur = hc_div14_fixed
            hc_div15_q = pl.div(row2_norm, col_sum)
            hc_div15_as = pl.mul(hc_div15_q, 4097.0)
            hc_div15_ad = pl.sub(hc_div15_as, hc_div15_q)
            hc_div15_ah = pl.sub(hc_div15_as, hc_div15_ad)
            hc_div15_al = pl.sub(hc_div15_q, hc_div15_ah)
            hc_div15_bs = pl.mul(col_sum, 4097.0)
            hc_div15_bd = pl.sub(hc_div15_bs, col_sum)
            hc_div15_bh = pl.sub(hc_div15_bs, hc_div15_bd)
            hc_div15_bl = pl.sub(col_sum, hc_div15_bh)
            hc_div15_prod = pl.mul(hc_div15_q, col_sum)
            hc_div15_hh = pl.mul(hc_div15_ah, hc_div15_bh)
            hc_div15_e0 = pl.sub(hc_div15_hh, hc_div15_prod)
            hc_div15_hl = pl.mul(hc_div15_ah, hc_div15_bl)
            hc_div15_e1 = pl.add(hc_div15_e0, hc_div15_hl)
            hc_div15_lh = pl.mul(hc_div15_al, hc_div15_bh)
            hc_div15_e2 = pl.add(hc_div15_e1, hc_div15_lh)
            hc_div15_ll = pl.mul(hc_div15_al, hc_div15_bl)
            hc_div15_error = pl.add(hc_div15_e2, hc_div15_ll)
            hc_div15_r0 = pl.sub(row2_norm, hc_div15_prod)
            hc_div15_residual = pl.sub(hc_div15_r0, hc_div15_error)
            hc_div15_correction = pl.div(hc_div15_residual, col_sum)
            hc_div15_fixed = pl.add(hc_div15_q, hc_div15_correction)
            row2_cur = hc_div15_fixed
            hc_div16_q = pl.div(row3_norm, col_sum)
            hc_div16_as = pl.mul(hc_div16_q, 4097.0)
            hc_div16_ad = pl.sub(hc_div16_as, hc_div16_q)
            hc_div16_ah = pl.sub(hc_div16_as, hc_div16_ad)
            hc_div16_al = pl.sub(hc_div16_q, hc_div16_ah)
            hc_div16_bs = pl.mul(col_sum, 4097.0)
            hc_div16_bd = pl.sub(hc_div16_bs, col_sum)
            hc_div16_bh = pl.sub(hc_div16_bs, hc_div16_bd)
            hc_div16_bl = pl.sub(col_sum, hc_div16_bh)
            hc_div16_prod = pl.mul(hc_div16_q, col_sum)
            hc_div16_hh = pl.mul(hc_div16_ah, hc_div16_bh)
            hc_div16_e0 = pl.sub(hc_div16_hh, hc_div16_prod)
            hc_div16_hl = pl.mul(hc_div16_ah, hc_div16_bl)
            hc_div16_e1 = pl.add(hc_div16_e0, hc_div16_hl)
            hc_div16_lh = pl.mul(hc_div16_al, hc_div16_bh)
            hc_div16_e2 = pl.add(hc_div16_e1, hc_div16_lh)
            hc_div16_ll = pl.mul(hc_div16_al, hc_div16_bl)
            hc_div16_error = pl.add(hc_div16_e2, hc_div16_ll)
            hc_div16_r0 = pl.sub(row3_norm, hc_div16_prod)
            hc_div16_residual = pl.sub(hc_div16_r0, hc_div16_error)
            hc_div16_correction = pl.div(hc_div16_residual, col_sum)
            hc_div16_fixed = pl.add(hc_div16_q, hc_div16_correction)
            row3_cur = hc_div16_fixed
        row0_out = pl.set_validshape(row0_cur, comb_rows, HC_MULT)
        row1_out = pl.set_validshape(row1_cur, comb_rows, HC_MULT)
        row2_out = pl.set_validshape(row2_cur, comb_rows, HC_MULT)
        row3_out = pl.set_validshape(row3_cur, comb_rows, HC_MULT)
        row0_write = pl.slice(row0_out, [COMB_T_TILE, HC_PAD], [0, 0], valid_shape=[comb_rows, HC_MULT])
        comb[t0 : t0 + COMB_T_TILE, 0 * HC_MULT : (0 + 1) * HC_MULT] = row0_write
        row1_write = pl.slice(row1_out, [COMB_T_TILE, HC_PAD], [0, 0], valid_shape=[comb_rows, HC_MULT])
        comb[t0 : t0 + COMB_T_TILE, 1 * HC_MULT : (1 + 1) * HC_MULT] = row1_write
        row2_write = pl.slice(row2_out, [COMB_T_TILE, HC_PAD], [0, 0], valid_shape=[comb_rows, HC_MULT])
        comb[t0 : t0 + COMB_T_TILE, 2 * HC_MULT : (2 + 1) * HC_MULT] = row2_write
        row3_write = pl.slice(row3_out, [COMB_T_TILE, HC_PAD], [0, 0], valid_shape=[comb_rows, HC_MULT])
        comb[t0 : t0 + COMB_T_TILE, 3 * HC_MULT : (3 + 1) * HC_MULT] = row3_write
    for blk in pl.spmd(
        (t_dim + T_TILE - 1) // T_TILE * (D // D_SPMD), name_hint="mix_x", allow_early_resolve=True
    ):
        t0 = blk // (D // D_SPMD) * T_TILE
        mix_rows = pl.min(T_TILE, t_dim - t0)
        d_base = blk % (D // D_SPMD) * D_SPMD
        pre_tile = pl.slice(pre_val_store, [T_TILE, HC_PAD], [t0, 0], valid_shape=[mix_rows, HC_PAD])
        pre_tile_t = pl.transpose(pre_tile, axis1=0, axis2=1)
        pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
        pre0_valid = pl.set_validshape(pre0, mix_rows, 1)
        pre1_valid = pl.set_validshape(pre1, mix_rows, 1)
        pre2_valid = pl.set_validshape(pre2, mix_rows, 1)
        pre3_valid = pl.set_validshape(pre3, mix_rows, 1)
        for db in pl.pipeline(D_SPMD // D_TILE, stage=2):
            d0 = d_base + db * D_TILE
            x0 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 0 * D + d0], valid_shape=[mix_rows, D_TILE])
            x1 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 1 * D + d0], valid_shape=[mix_rows, D_TILE])
            x2 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 2 * D + d0], valid_shape=[mix_rows, D_TILE])
            x3 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 3 * D + d0], valid_shape=[mix_rows, D_TILE])
            y0 = pl.row_expand_mul(x0, pre0_valid)
            y1 = pl.row_expand_mul(x1, pre1_valid)
            y2 = pl.row_expand_mul(x2, pre2_valid)
            y3 = pl.row_expand_mul(x3, pre3_valid)
            y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
            y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
            mixed_valid = pl.set_validshape(y_bf16, mix_rows, D_TILE)
            x_mixed[t0 : t0 + T_TILE, d0 : d0 + D_TILE] = mixed_valid
    return x_mixed


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


def _golden_a2a3_cube_linear(x_flat_2d, hc_fn):
    """Emulate the A2/A3 f32 Cube MAD reduction used by the HC projection."""
    import torch

    cube_acc_k = 4
    group_block = 64
    assert LINEAR_K_PER_SPLIT % cube_acc_k == 0
    groups_per_split = LINEAR_K_PER_SPLIT // cube_acc_k
    t_dim = x_flat_2d.shape[0]
    x_groups = x_flat_2d.reshape(t_dim, 1, LINEAR_OK, groups_per_split, cube_acc_k)
    w_groups = hc_fn.reshape(1, MIX_HC, LINEAR_OK, groups_per_split, cube_acc_k)
    split_mixes = torch.zeros(t_dim, MIX_HC, LINEAR_OK, dtype=torch.float32)
    for group0 in range(0, groups_per_split, group_block):
        x_block = x_groups[..., group0 : group0 + group_block, :].double()
        w_block = w_groups[..., group0 : group0 + group_block, :].double()
        group_dots = (x_block * w_block).sum(dim=-1)
        for group in range(group_dots.shape[-1]):
            split_mixes = (split_mixes.double() + group_dots[..., group]).float()
    mixes = torch.zeros(t_dim, MIX_HC, dtype=torch.float32)
    for split in range(LINEAR_OK):
        mixes += split_mixes[..., split]
    return mixes


def golden_hc_pre(tensors):
    """Torch reference matching the target HC-pre reduction and nonlinearities."""
    import torch
    from utils import fp32_matmul_fp32

    x = tensors["x"].float()
    hc_fn = tensors["hc_fn"].float()
    hc_scale = tensors["hc_scale"].float()
    hc_base = tensors["hc_base"].float()
    t_dim = x.shape[0]
    x_flat_2d = x.reshape(t_dim, HC_DIM)
    sq_sum = torch.zeros(t_dim, 1, dtype=torch.float32)
    for k0 in range(0, HC_DIM, RMS_K_TILE):
        x_chunk = x_flat_2d[:, k0 : k0 + RMS_K_TILE]
        square_row = x_chunk * x_chunk
        while square_row.shape[-1] > 1:
            square_row = square_row[:, 0::2] + square_row[:, 1::2]
        sq_sum += square_row
    variance = sq_sum * HC_DIM_INV + NORM_EPS
    rsqrt = torch.rsqrt(variance.double()).float()
    mixes = fp32_matmul_fp32(x_flat_2d, hc_fn.t(), split_k=LINEAR_OK)
    mixes *= rsqrt
    pre_logits = mixes[..., :HC_MULT] * hc_scale[0] + hc_base[:HC_MULT]
    pre_exp = torch.exp(-pre_logits.double()).float()
    pre = torch.reciprocal((pre_exp + 1.0).double()).float() + HC_EPS
    post_logits = mixes[..., HC_MULT : HC_MULT * 2] * hc_scale[1] + hc_base[HC_MULT : HC_MULT * 2]
    post_exp = torch.exp(-post_logits.double()).float()
    post_t = 2 * torch.reciprocal((post_exp + 1.0).double()).float()
    comb_logits = mixes[..., HC_MULT * 2 :] * hc_scale[2] + hc_base[HC_MULT * 2 :]
    comb_t = comb_logits.view(t_dim, HC_MULT, HC_MULT)
    comb_max = comb_t.max(dim=-1, keepdim=True).values
    comb_exp = torch.exp((comb_t - comb_max).double()).float()

    def sum_four(value):
        return value[..., 0:1] + value[..., 1:2] + (value[..., 2:3] + value[..., 3:4])

    def column_sum(value):
        return value[:, 0:1, :] + value[:, 1:2, :] + (value[:, 2:3, :] + value[:, 3:4, :])

    comb_t = (comb_exp.double() / sum_four(comb_exp).double()).float() + HC_EPS
    comb_t = (comb_t.double() / (column_sum(comb_t) + HC_EPS).double()).float()
    for _ in range(HC_SINKHORN_ITER - 1):
        comb_t = (comb_t.double() / (sum_four(comb_t) + HC_EPS).double()).float()
        comb_t = (comb_t.double() / (column_sum(comb_t) + HC_EPS).double()).float()
    y0 = x[:, 0, :] * pre[:, 0:1]
    y1 = x[:, 1, :] * pre[:, 1:2]
    y2 = x[:, 2, :] * pre[:, 2:3]
    y3 = x[:, 3, :] * pre[:, 3:4]
    y = y0 + y1 + (y2 + y3)
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
                5.9166,
                -3.6223,
                -2.9324,
                -3.3124,
                -3.91,
                -0.9384,
                -3.3256,
                -2.524,
                2.0706,
                -2.5728,
                0.1424,
                -3.9453,
                -3.8859,
                3.4634,
                -3.3799,
                -2.6077,
                -2.7191,
                -2.4846,
                2.0395,
                -0.501,
                -3.5992,
                -2.752,
                -3.3493,
                3.1587,
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
    from golden import ratio_allclose, run

    MODES = {"decode": (DECODE_BATCH, DECODE_SEQ)}
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"]
    )
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode"], default="decode")
    parser.add_argument(
        "--tokens",
        type=int,
        default=None,
        help="Validate one token count; default covers configured, single, short, and multi-tile tails.",
    )
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    if args.tokens is not None and args.tokens <= 0:
        parser.error("--tokens must be positive")
    modes_to_run = [args.mode]
    for mode_name in modes_to_run:
        B, S = MODES[mode_name]
        if args.tokens is not None:
            token_counts = [args.tokens]
        elif args.golden_data is not None:
            token_counts = [B * S]
        else:
            token_counts = list(dict.fromkeys([B * S, 1, 6, 9, 17]))
        for tokens in token_counts:
            print(f"--- hc_pre {mode_name}: T={tokens} ---")
            result = run(
                fn=hc_pre_test,
                specs=build_tensor_specs(1, tokens),
                golden_fn=golden_hc_pre,
                runtime_dir=args.runtime_dir,
                golden_data=args.golden_data,
                config=dict(
                    dump_passes=args.dump_passes,
                    backend_type=args.platform.removesuffix("sim"),
                    execution_mode=args.sim or args.platform.endswith("sim"),
                    device_id=args.device,
                    enable_chip_swimlane=args.enable_chip_swimlane,
                ),
                rtol=0.001,
                atol=0.001,
                compare_fn={
                    "x_mixed": ratio_allclose(atol=0.0001, rtol=1.0 / 128),
                    "post": ratio_allclose(atol=2.5e-05, rtol=0.005),
                    "comb": ratio_allclose(atol=2.5e-05, rtol=0.005),
                },
                compile_only=args.compile_only,
            )
            if not result.passed:
                if result.error:
                    print(result.error)
                raise SystemExit(1)
