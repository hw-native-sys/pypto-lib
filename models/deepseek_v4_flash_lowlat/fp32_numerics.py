# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim  # A2/A3 device validation; simulator CI is deferred for this model.
"""Scope-local FP32 error-free transforms and bounded unary corrections.

The caller retains scope, shape and valid extents. Exp round-before-scale is
explicit; log consumes an unrounded pair. No helper creates a task or GM buffer.
"""
from __future__ import annotations
import pypto.language as pl

@pl.jit.inline(auto_scope=False)
def two_sum(a: pl.Tensor, b: pl.Tensor, rows: pl.constexpr, cols: pl.constexpr) -> tuple[pl.Tensor, pl.Tensor]:
    total = a
    error = a
    total = pl.add(a, b)
    z = pl.sub(total, a)
    total_minus_z = pl.sub(total, z)
    a_error = pl.sub(a, total_minus_z)
    b_error = pl.sub(b, z)
    error = pl.add(a_error, b_error)
    return (total, error)

@pl.jit.inline(auto_scope=False)
def two_product(a: pl.Tensor, b: pl.Tensor, rows: pl.constexpr, cols: pl.constexpr) -> tuple[pl.Tensor, pl.Tensor]:
    product = a
    error = a
    a_split = pl.mul(a, 4097.0)
    a_delta = pl.sub(a_split, a)
    a_hi = pl.sub(a_split, a_delta)
    a_lo = pl.sub(a, a_hi)
    b_split = pl.mul(b, 4097.0)
    b_delta = pl.sub(b_split, b)
    b_hi = pl.sub(b_split, b_delta)
    b_lo = pl.sub(b, b_hi)
    product = pl.mul(a, b)
    high_product = pl.mul(a_hi, b_hi)
    error = pl.sub(high_product, product)
    high_low = pl.mul(a_hi, b_lo)
    error = pl.add(error, high_low)
    low_high = pl.mul(a_lo, b_hi)
    error = pl.add(error, low_high)
    low_product = pl.mul(a_lo, b_lo)
    error = pl.add(error, low_product)
    return (product, error)

@pl.jit.inline(auto_scope=False)
def ds_add(ah: pl.Tensor, al: pl.Tensor, bh: pl.Tensor, bl: pl.Tensor, rows: pl.constexpr, cols: pl.constexpr) -> tuple[pl.Tensor, pl.Tensor]:
    ssa1_total, ssa1_error = two_sum(ah, bh, rows, cols)
    low = al
    low = pl.add(al, bl)
    tail = ssa1_error
    tail = pl.add(ssa1_error, low)
    ssa2_high, ssa2_low = two_sum(ssa1_total, tail, rows, cols)
    return (ssa2_high, ssa2_low)

@pl.jit.inline(auto_scope=False)
def ds_multiply(ah: pl.Tensor, al: pl.Tensor, bh: pl.Tensor, bl: pl.Tensor, rows: pl.constexpr, cols: pl.constexpr) -> tuple[pl.Tensor, pl.Tensor]:
    ssa1_high, ssa1_error = two_product(ah, bh, rows, cols)
    cross1 = pl.mul(ah, bl)
    cross2 = pl.mul(al, bh)
    cross3 = pl.mul(al, bl)
    low = ssa1_error
    low = pl.add(ssa1_error, cross1)
    low = pl.add(low, cross2)
    low = pl.add(low, cross3)
    ssa2_high, ssa2_low = two_sum(ssa1_high, low, rows, cols)
    return (ssa2_high, ssa2_low)

@pl.jit.inline(auto_scope=False)
def exp_reduced(x: pl.Tensor, zero: pl.Tensor, one: pl.Tensor, rows: pl.constexpr, cols: pl.constexpr, round_first: pl.constexpr) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
    safe = x
    safe = pl.maximum(x, -80.0)
    scaled = pl.mul(safe, 1.4426950216293335)
    n_i32 = pl.cast(scaled, target_type=pl.INT32, mode='rint')
    n = x
    n = pl.cast(n_i32, target_type=pl.FP32)
    ln_hi = zero
    ln_lo = zero
    ln_hi = pl.add(zero, 0.6931471824645996)
    ln_lo = pl.add(zero, -1.9046542121259336e-09)
    ssa1_ph, ssa1_pe = two_product(n, ln_hi, rows, cols)
    ssa2_qh, ssa2_qe = two_product(n, ln_lo, rows, cols)
    negative_ph = x
    negative_ph = pl.neg(ssa1_ph)
    ssa3_rh, ssa3_rl = two_sum(safe, negative_ph, rows, cols)
    negative_pe = x
    negative_pe = pl.neg(ssa1_pe)
    ssa4_rh, ssa4_rl = ds_add(ssa3_rh, ssa3_rl, negative_pe, zero, rows, cols)
    negative_qh = x
    negative_qe = x
    negative_qh = pl.neg(ssa2_qh)
    negative_qe = pl.neg(ssa2_qe)
    ssa5_rh, ssa5_rl = ds_add(ssa4_rh, ssa4_rl, negative_qh, negative_qe, rows, cols)
    ch = zero
    cl = zero
    ch = pl.add(zero, 4.7794772561329454e-14)
    cl = pl.add(zero, 7.62544404448643e-22)
    ssa6_ch, ssa6_cl = ds_multiply(ch, cl, ssa5_rh, ssa5_rl, rows, cols)
    coefficient_hi15 = zero
    coefficient_lo15 = zero
    coefficient_hi15 = pl.add(zero, 7.647163609812713e-13)
    coefficient_lo15 = pl.add(zero, 1.2200710471178288e-20)
    ssa7_ch, ssa7_cl = ds_add(ssa6_ch, ssa6_cl, coefficient_hi15, coefficient_lo15, rows, cols)
    ssa8_ch, ssa8_cl = ds_multiply(ssa7_ch, ssa7_cl, ssa5_rh, ssa5_rl, rows, cols)
    coefficient_hi14 = zero
    coefficient_lo14 = zero
    coefficient_hi14 = pl.add(zero, 1.147074536050896e-11)
    coefficient_lo14 = pl.add(zero, 2.372207689231238e-19)
    ssa9_ch, ssa9_cl = ds_add(ssa8_ch, ssa8_cl, coefficient_hi14, coefficient_lo14, rows, cols)
    ssa10_ch, ssa10_cl = ds_multiply(ssa9_ch, ssa9_cl, ssa5_rh, ssa5_rl, rows, cols)
    coefficient_hi13 = zero
    coefficient_lo13 = zero
    coefficient_hi13 = pl.add(zero, 1.6059044372074283e-10)
    coefficient_lo13 = pl.add(zero, -5.352526511562726e-18)
    ssa11_ch, ssa11_cl = ds_add(ssa10_ch, ssa10_cl, coefficient_hi13, coefficient_lo13, rows, cols)
    ssa12_ch, ssa12_cl = ds_multiply(ssa11_ch, ssa11_cl, ssa5_rh, ssa5_rl, rows, cols)
    coefficient_hi12 = zero
    coefficient_lo12 = zero
    coefficient_hi12 = pl.add(zero, 2.0876755879584152e-09)
    coefficient_lo12 = pl.add(zero, 1.1082839147459852e-16)
    ssa13_ch, ssa13_cl = ds_add(ssa12_ch, ssa12_cl, coefficient_hi12, coefficient_lo12, rows, cols)
    ssa14_ch, ssa14_cl = ds_multiply(ssa13_ch, ssa13_cl, ssa5_rh, ssa5_rl, rows, cols)
    coefficient_hi11 = zero
    coefficient_lo11 = zero
    coefficient_hi11 = pl.add(zero, 2.5052107943679403e-08)
    coefficient_lo11 = pl.add(zero, 4.4176230446483665e-16)
    ssa15_ch, ssa15_cl = ds_add(ssa14_ch, ssa14_cl, coefficient_hi11, coefficient_lo11, rows, cols)
    ssa16_ch, ssa16_cl = ds_multiply(ssa15_ch, ssa15_cl, ssa5_rh, ssa5_rl, rows, cols)
    coefficient_hi10 = zero
    coefficient_lo10 = zero
    coefficient_hi10 = pl.add(zero, 2.755731998149713e-07)
    coefficient_lo10 = pl.add(zero, -7.575112209051195e-15)
    ssa17_ch, ssa17_cl = ds_add(ssa16_ch, ssa16_cl, coefficient_hi10, coefficient_lo10, rows, cols)
    ssa18_ch, ssa18_cl = ds_multiply(ssa17_ch, ssa17_cl, ssa5_rh, ssa5_rl, rows, cols)
    coefficient_hi9 = zero
    coefficient_lo9 = zero
    coefficient_hi9 = pl.add(zero, 2.7557318844628753e-06)
    coefficient_lo9 = pl.add(zero, 3.793571224297229e-14)
    ssa19_ch, ssa19_cl = ds_add(ssa18_ch, ssa18_cl, coefficient_hi9, coefficient_lo9, rows, cols)
    ssa20_ch, ssa20_cl = ds_multiply(ssa19_ch, ssa19_cl, ssa5_rh, ssa5_rl, rows, cols)
    coefficient_hi8 = zero
    coefficient_lo8 = zero
    coefficient_hi8 = pl.add(zero, 2.4801587642286904e-05)
    coefficient_lo8 = pl.add(zero, -3.40699609366682e-13)
    ssa21_ch, ssa21_cl = ds_add(ssa20_ch, ssa20_cl, coefficient_hi8, coefficient_lo8, rows, cols)
    ssa22_ch, ssa22_cl = ds_multiply(ssa21_ch, ssa21_cl, ssa5_rh, ssa5_rl, rows, cols)
    coefficient_hi7 = zero
    coefficient_lo7 = zero
    coefficient_hi7 = pl.add(zero, 0.00019841270113829523)
    coefficient_lo7 = pl.add(zero, -2.725596874933456e-12)
    ssa23_ch, ssa23_cl = ds_add(ssa22_ch, ssa22_cl, coefficient_hi7, coefficient_lo7, rows, cols)
    ssa24_ch, ssa24_cl = ds_multiply(ssa23_ch, ssa23_cl, ssa5_rh, ssa5_rl, rows, cols)
    coefficient_hi6 = zero
    coefficient_lo6 = zero
    coefficient_hi6 = pl.add(zero, 0.0013888889225199819)
    coefficient_lo6 = pl.add(zero, -3.3631094437103215e-11)
    ssa25_ch, ssa25_cl = ds_add(ssa24_ch, ssa24_cl, coefficient_hi6, coefficient_lo6, rows, cols)
    ssa26_ch, ssa26_cl = ds_multiply(ssa25_ch, ssa25_cl, ssa5_rh, ssa5_rl, rows, cols)
    coefficient_hi5 = zero
    coefficient_lo5 = zero
    coefficient_hi5 = pl.add(zero, 0.008333333767950535)
    coefficient_lo5 = pl.add(zero, -4.34617203337595e-10)
    ssa27_ch, ssa27_cl = ds_add(ssa26_ch, ssa26_cl, coefficient_hi5, coefficient_lo5, rows, cols)
    ssa28_ch, ssa28_cl = ds_multiply(ssa27_ch, ssa27_cl, ssa5_rh, ssa5_rl, rows, cols)
    coefficient_hi4 = zero
    coefficient_lo4 = zero
    coefficient_hi4 = pl.add(zero, 0.0416666679084301)
    coefficient_lo4 = pl.add(zero, -1.2417634698280722e-09)
    ssa29_ch, ssa29_cl = ds_add(ssa28_ch, ssa28_cl, coefficient_hi4, coefficient_lo4, rows, cols)
    ssa30_ch, ssa30_cl = ds_multiply(ssa29_ch, ssa29_cl, ssa5_rh, ssa5_rl, rows, cols)
    coefficient_hi3 = zero
    coefficient_lo3 = zero
    coefficient_hi3 = pl.add(zero, 0.1666666716337204)
    coefficient_lo3 = pl.add(zero, -4.967053879312289e-09)
    ssa31_ch, ssa31_cl = ds_add(ssa30_ch, ssa30_cl, coefficient_hi3, coefficient_lo3, rows, cols)
    ssa32_ch, ssa32_cl = ds_multiply(ssa31_ch, ssa31_cl, ssa5_rh, ssa5_rl, rows, cols)
    coefficient_hi2 = zero
    coefficient_lo2 = zero
    coefficient_hi2 = pl.add(zero, 0.5)
    coefficient_lo2 = pl.add(zero, 0.0)
    ssa33_ch, ssa33_cl = ds_add(ssa32_ch, ssa32_cl, coefficient_hi2, coefficient_lo2, rows, cols)
    ssa34_ch, ssa34_cl = ds_multiply(ssa33_ch, ssa33_cl, ssa5_rh, ssa5_rl, rows, cols)
    coefficient_hi1 = zero
    coefficient_lo1 = zero
    coefficient_hi1 = pl.add(zero, 1.0)
    coefficient_lo1 = pl.add(zero, 0.0)
    ssa35_ch, ssa35_cl = ds_add(ssa34_ch, ssa34_cl, coefficient_hi1, coefficient_lo1, rows, cols)
    ssa36_ch, ssa36_cl = ds_multiply(ssa35_ch, ssa35_cl, ssa5_rh, ssa5_rl, rows, cols)
    coefficient_hi0 = zero
    coefficient_lo0 = zero
    coefficient_hi0 = pl.add(zero, 1.0)
    coefficient_lo0 = pl.add(zero, 0.0)
    ssa37_ch, ssa37_cl = ds_add(ssa36_ch, ssa36_cl, coefficient_hi0, coefficient_lo0, rows, cols)
    reduced = ssa37_ch
    if round_first:
        reduced = pl.add(ssa37_ch, ssa37_cl)
    remaining = pl.neg(n)
    power = one
    take64 = pl.sub(remaining, 63.0)
    take64 = pl.maximum(take64, 0.0)
    take64 = pl.minimum(take64, 1.0)
    used64 = pl.mul(take64, 64.0)
    remaining = pl.sub(remaining, used64)
    not64 = pl.sub(one, take64)
    small64 = pl.mul(take64, 5.421010862427522e-20)
    factor64 = pl.add(not64, small64)
    power = pl.mul(power, factor64)
    take32 = pl.sub(remaining, 31.0)
    take32 = pl.maximum(take32, 0.0)
    take32 = pl.minimum(take32, 1.0)
    used32 = pl.mul(take32, 32.0)
    remaining = pl.sub(remaining, used32)
    not32 = pl.sub(one, take32)
    small32 = pl.mul(take32, 2.3283064365386963e-10)
    factor32 = pl.add(not32, small32)
    power = pl.mul(power, factor32)
    take16 = pl.sub(remaining, 15.0)
    take16 = pl.maximum(take16, 0.0)
    take16 = pl.minimum(take16, 1.0)
    used16 = pl.mul(take16, 16.0)
    remaining = pl.sub(remaining, used16)
    not16 = pl.sub(one, take16)
    small16 = pl.mul(take16, 1.52587890625e-05)
    factor16 = pl.add(not16, small16)
    power = pl.mul(power, factor16)
    take8 = pl.sub(remaining, 7.0)
    take8 = pl.maximum(take8, 0.0)
    take8 = pl.minimum(take8, 1.0)
    used8 = pl.mul(take8, 8.0)
    remaining = pl.sub(remaining, used8)
    not8 = pl.sub(one, take8)
    small8 = pl.mul(take8, 0.00390625)
    factor8 = pl.add(not8, small8)
    power = pl.mul(power, factor8)
    take4 = pl.sub(remaining, 3.0)
    take4 = pl.maximum(take4, 0.0)
    take4 = pl.minimum(take4, 1.0)
    used4 = pl.mul(take4, 4.0)
    remaining = pl.sub(remaining, used4)
    not4 = pl.sub(one, take4)
    small4 = pl.mul(take4, 0.0625)
    factor4 = pl.add(not4, small4)
    power = pl.mul(power, factor4)
    take2 = pl.sub(remaining, 1.0)
    take2 = pl.maximum(take2, 0.0)
    take2 = pl.minimum(take2, 1.0)
    used2 = pl.mul(take2, 2.0)
    remaining = pl.sub(remaining, used2)
    not2 = pl.sub(one, take2)
    small2 = pl.mul(take2, 0.25)
    factor2 = pl.add(not2, small2)
    power = pl.mul(power, factor2)
    take1 = pl.sub(remaining, 0.0)
    take1 = pl.maximum(take1, 0.0)
    take1 = pl.minimum(take1, 1.0)
    used1 = pl.mul(take1, 1.0)
    remaining = pl.sub(remaining, used1)
    not1 = pl.sub(one, take1)
    small1 = pl.mul(take1, 0.5)
    factor1 = pl.add(not1, small1)
    power = pl.mul(power, factor1)
    return (reduced, ssa37_cl, power)

@pl.jit.inline(auto_scope=False)
def rne_exp_nonpositive(x: pl.Tensor, rows: pl.constexpr, cols: pl.constexpr) -> pl.Tensor:
    result = x
    zero = x
    fallback = pl.exp(x)
    zero = pl.sub(x, x)
    one = x
    one = pl.add(zero, 1.0)
    reduced, unused_low, power = exp_reduced(x, zero, one, rows, cols, round_first=True)
    value = pl.mul(reduced, power)
    shift = pl.sub(x, -80.00000762939453)
    gate = pl.mul(shift, 131072.0)
    gate = pl.maximum(gate, 0.0)
    gate = pl.minimum(gate, 1.0)
    one = pl.add(zero, 1.0)
    other = pl.sub(one, gate)
    corrected = pl.mul(value, gate)
    original = pl.mul(fallback, other)
    result = pl.add(corrected, original)
    return result

@pl.jit.inline(auto_scope=False)
def correct_log_seed(x: pl.Tensor, seed: pl.Tensor, rows: pl.constexpr, cols: pl.constexpr) -> pl.Tensor:
    result = x
    zero = x
    negative_seed = x
    zero = pl.mul(x, 0.0)
    negative_seed = pl.neg(seed)
    one = x
    one = pl.add(zero, 1.0)
    high, low, power = exp_reduced(negative_seed, zero, one, rows, cols, round_first=False)
    exp_hi = x
    exp_lo = x
    exp_hi = pl.mul(high, power)
    exp_lo = pl.mul(low, power)
    one = x
    one = pl.add(zero, 1.0)
    minus_one = x
    minus_one = pl.neg(one)
    ph, plow = ds_multiply(x, zero, exp_hi, exp_lo, rows, cols)
    rh, rl = ds_add(ph, plow, minus_one, zero, rows, cols)
    sh, sl = ds_multiply(rh, rl, rh, rl, rows, cols)
    nh = x
    nl = x
    nh = pl.mul(sh, -0.5)
    nl = pl.mul(sl, -0.5)
    ch, cl = ds_add(rh, rl, nh, nl, rows, cols)
    ah, al = ds_add(seed, zero, ch, cl, rows, cols)
    result = pl.add(ah, al)
    return result

@pl.jit.inline(auto_scope=False)
def newton_sqrt_seed(var: pl.Tensor, raw_sqrt: pl.Tensor, rows: pl.constexpr, cols: pl.constexpr) -> pl.Tensor:
    result = var
    work = pl.mul(var, 0.25)
    z = pl.mul(raw_sqrt, 0.5)
    split = pl.mul(z, 4097.0)
    delta = pl.sub(split, z)
    hi = pl.sub(split, delta)
    lo = pl.sub(z, hi)
    product = pl.mul(z, z)
    hi_product = pl.mul(hi, hi)
    error = pl.sub(hi_product, product)
    cross = pl.mul(hi, lo)
    error = pl.add(error, cross)
    error = pl.add(error, cross)
    lo_product = pl.mul(lo, lo)
    error = pl.add(error, lo_product)
    residual_high = pl.sub(work, product)
    residual = pl.sub(residual_high, error)
    denominator = pl.mul(z, 2.0)
    step = pl.div(residual, denominator)
    adjusted = pl.add(z, step)
    result = pl.mul(adjusted, 2.0)
    return result

@pl.jit.inline(auto_scope=False)
def midpoint_sqrt_seed(var: pl.Tensor, newton: pl.Tensor, rows: pl.constexpr, cols: pl.constexpr) -> pl.Tensor:
    result = newton
    neighbor_delta = pl.mul(newton, 2.0 ** (-24) + 2.0 ** (-47))
    lower = pl.sub(newton, neighbor_delta)
    upper = pl.add(newton, neighbor_delta)
    z = pl.mul(newton, 0.5)
    lo_gap_raw = pl.sub(newton, lower)
    lo_gap = pl.mul(lo_gap_raw, 0.5)
    hi_gap_raw = pl.sub(upper, newton)
    hi_gap = pl.mul(hi_gap_raw, 0.5)
    split = pl.mul(z, 4097.0)
    split_delta = pl.sub(split, z)
    hi = pl.sub(split, split_delta)
    lo = pl.sub(z, hi)
    product = pl.mul(z, z)
    cross = pl.mul(hi, lo)
    hi_product = pl.mul(hi, hi)
    error = pl.sub(hi_product, product)
    error = pl.add(error, cross)
    error = pl.add(error, cross)
    lo_product = pl.mul(lo, lo)
    error = pl.add(error, lo_product)
    quarter_var = pl.mul(var, 0.25)
    residual_input = pl.sub(quarter_var, product)
    negative_error = pl.neg(error)
    rh = pl.add(residual_input, negative_error)
    residual_z = pl.sub(rh, residual_input)
    residual_sum_minus_z = pl.sub(rh, residual_z)
    residual_a_error = pl.sub(residual_input, residual_sum_minus_z)
    residual_b_error = pl.sub(negative_error, residual_z)
    rl = pl.add(residual_a_error, residual_b_error)
    z_lo_gap = pl.mul(z, lo_gap)
    l1 = pl.add(rh, z_lo_gap)
    lower_first_z = pl.sub(l1, rh)
    lower_first_sum_minus_z = pl.sub(l1, lower_first_z)
    lower_first_a_error = pl.sub(rh, lower_first_sum_minus_z)
    lower_first_b_error = pl.sub(z_lo_gap, lower_first_z)
    e1 = pl.add(lower_first_a_error, lower_first_b_error)
    negative_lo_gap = pl.neg(lo_gap)
    negative_lo_square = pl.mul(negative_lo_gap, lo_gap)
    negative_lo_quarter_square = pl.mul(negative_lo_square, 0.25)
    l2 = pl.add(l1, negative_lo_quarter_square)
    lower_second_z = pl.sub(l2, l1)
    lower_second_sum_minus_z = pl.sub(l2, lower_second_z)
    lower_second_a_error = pl.sub(l1, lower_second_sum_minus_z)
    lower_second_b_error = pl.sub(negative_lo_quarter_square, lower_second_z)
    e2 = pl.add(lower_second_a_error, lower_second_b_error)
    lower_errors_first = pl.add(e1, rl)
    lower_errors = pl.add(lower_errors_first, e2)
    dlo = pl.add(l2, lower_errors)
    negative_z = pl.neg(z)
    negative_z_hi_gap = pl.mul(negative_z, hi_gap)
    h1 = pl.add(rh, negative_z_hi_gap)
    upper_first_z = pl.sub(h1, rh)
    upper_first_sum_minus_z = pl.sub(h1, upper_first_z)
    upper_first_a_error = pl.sub(rh, upper_first_sum_minus_z)
    upper_first_b_error = pl.sub(negative_z_hi_gap, upper_first_z)
    f1 = pl.add(upper_first_a_error, upper_first_b_error)
    negative_hi_gap = pl.neg(hi_gap)
    negative_hi_square = pl.mul(negative_hi_gap, hi_gap)
    negative_hi_quarter_square = pl.mul(negative_hi_square, 0.25)
    h2 = pl.add(h1, negative_hi_quarter_square)
    upper_second_z = pl.sub(h2, h1)
    upper_second_sum_minus_z = pl.sub(h2, upper_second_z)
    upper_second_a_error = pl.sub(h1, upper_second_sum_minus_z)
    upper_second_b_error = pl.sub(negative_hi_quarter_square, upper_second_z)
    f2 = pl.add(upper_second_a_error, upper_second_b_error)
    upper_errors_first = pl.add(f1, rl)
    upper_errors = pl.add(upper_errors_first, f2)
    dhi = pl.add(h2, upper_errors)
    negative_dlo = pl.neg(dlo)
    down_denominator = pl.mul(z, lo_gap)
    down_ratio = pl.div(negative_dlo, down_denominator)
    down_scaled = pl.mul(down_ratio, 2.0 ** 40)
    down_bounded = pl.minimum(down_scaled, 1.0)
    down_flag = pl.maximum(down_bounded, 0.0)
    up_denominator = pl.mul(z, hi_gap)
    up_ratio = pl.div(dhi, up_denominator)
    up_scaled = pl.mul(up_ratio, 2.0 ** 40)
    up_bounded = pl.minimum(up_scaled, 1.0)
    up_flag = pl.maximum(up_bounded, 0.0)
    lower_difference = pl.sub(lower, newton)
    down_adjustment = pl.mul(down_flag, lower_difference)
    first_result = pl.add(newton, down_adjustment)
    upper_difference = pl.sub(upper, newton)
    up_adjustment = pl.mul(up_flag, upper_difference)
    result = pl.add(first_result, up_adjustment)
    return result

@pl.jit.inline(auto_scope=False)
def correct_sqrt_bounded(x: pl.Tensor, raw: pl.Tensor, rows: pl.constexpr, cols: pl.constexpr) -> pl.Tensor:
    safe_x = x
    safe_seed = x
    result = x
    safe_x = pl.maximum(x, 2.0 ** (-40))
    safe_x = pl.minimum(safe_x, 2.0 ** 40)
    safe_seed = pl.maximum(raw, 2.0 ** (-20))
    safe_seed = pl.minimum(safe_seed, 2.0 ** 20)
    newton = newton_sqrt_seed(safe_x, safe_seed, rows=rows, cols=cols)
    fixed = midpoint_sqrt_seed(safe_x, newton, rows=rows, cols=cols)
    zero = x
    zero = pl.sub(safe_x, safe_x)
    one = pl.add(zero, 1.0)
    upper_boundary = x
    upper_boundary = pl.add(zero, 1099511758848.0)
    lower_delta = pl.sub(x, 9.094946475628196e-13)
    lower_scaled = pl.mul(lower_delta, 2.0 ** 64)
    lower_gate = pl.maximum(lower_scaled, 0.0)
    lower_gate = pl.minimum(lower_gate, 1.0)
    upper_delta = pl.sub(upper_boundary, x)
    upper_scaled = pl.mul(upper_delta, 2.0 ** (-17))
    upper_gate = pl.maximum(upper_scaled, 0.0)
    upper_gate = pl.minimum(upper_gate, 1.0)
    gate = pl.mul(lower_gate, upper_gate)
    corrected = pl.mul(fixed, gate)
    other = pl.sub(one, gate)
    original = pl.mul(raw, other)
    result = pl.add(corrected, original)
    return result

@pl.jit.inline(auto_scope=False)
def normalize_topk(selected: pl.Tensor, route_scale: pl.constexpr) -> pl.Tensor:
    weights = selected
    denom = pl.row_sum(selected)
    hd_estimate = pl.row_expand_div(selected, denom)
    hd_left_split = pl.mul(hd_estimate, 4097.0)
    hd_left_delta = pl.sub(hd_left_split, hd_estimate)
    hd_left_hi = pl.sub(hd_left_split, hd_left_delta)
    hd_left_lo = pl.sub(hd_estimate, hd_left_hi)
    hd_right_split = pl.mul(denom, 4097.0)
    hd_right_delta = pl.sub(hd_right_split, denom)
    hd_right_hi = pl.sub(hd_right_split, hd_right_delta)
    hd_right_lo = pl.sub(denom, hd_right_hi)
    hd_product = pl.row_expand_mul(hd_estimate, denom)
    hd_hi_product = pl.row_expand_mul(hd_left_hi, hd_right_hi)
    hd_error = pl.sub(hd_hi_product, hd_product)
    hd_hi_lo = pl.row_expand_mul(hd_left_hi, hd_right_lo)
    hd_error = pl.add(hd_error, hd_hi_lo)
    hd_lo_hi = pl.row_expand_mul(hd_left_lo, hd_right_hi)
    hd_error = pl.add(hd_error, hd_lo_hi)
    hd_lo_product = pl.row_expand_mul(hd_left_lo, hd_right_lo)
    hd_error = pl.add(hd_error, hd_lo_product)
    hd_residual_hi = pl.sub(selected, hd_product)
    hd_residual = pl.sub(hd_residual_hi, hd_error)
    hd_correction = pl.row_expand_div(hd_residual, denom)
    hd_corrected = pl.add(hd_estimate, hd_correction)
    weights = pl.mul(hd_corrected, route_scale)
    return weights
