# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim  # A2/A3 device validation; simulator CI is deferred for this model.
"""Compensated HC-pre maps inside the caller's existing SPMD scope.

These helpers keep FP32 operands and outputs on physical 8x8 tiles. The
exponential correction is bounded to [-80, 80], with the original exponential
retained outside that interval. No task or GM tensor is allocated here.
"""

import pypto.language as pl
if __package__:
    from .fp32_numerics import rne_exp_signed
else:
    from fp32_numerics import rne_exp_signed


@pl.jit.inline(auto_scope=False)
def hc_sigmoid_math(
    mix: pl.Tensor[[8, 8], pl.FP32],
    inv: pl.Tensor[[8, 1], pl.FP32],
    scales: pl.Tensor[[3], pl.FP32],
    bases: pl.Tensor[[24], pl.FP32],
    group: pl.constexpr,
    offset: pl.constexpr,
) -> pl.Tensor:
    hc_scale = pl.read(scales, [group])
    hc_base_raw = pl.slice(bases, [8], [offset], clamp=True)
    hc_base = pl.reshape(hc_base_raw, [1, 8])
    hc_norm = pl.row_expand_mul(mix, inv)
    hc_scaled = pl.mul(hc_norm, hc_scale)
    hc_logits = pl.add(hc_scaled, pl.col_expand(mix, hc_base))
    hc_exp_argument = pl.create_tensor([8, 8], dtype=pl.FP32)
    hc_exp_argument = pl.neg(hc_logits)
    hc_exp_value = rne_exp_signed(hc_exp_argument, rows=8, cols=8, safe_zero=True, negate=False)
    hc_rcp_denominator = pl.add(hc_exp_value, 1.0)
    hc_rcp_original = pl.recip(hc_rcp_denominator)
    hc_rcp_bounded: pl.Tensor[[8, 8], pl.FP32] = pl.minimum(hc_rcp_denominator, 4294967296.0)
    hc_rcp_estimate: pl.Tensor[[8, 8], pl.FP32] = pl.recip(hc_rcp_bounded)
    hc_rcp_product = hc_rcp_estimate
    hc_rcp_error = hc_rcp_estimate
    hc_rcp_a_split = pl.mul(hc_rcp_estimate, 4097.0)
    hc_rcp_a_delta = pl.sub(hc_rcp_a_split, hc_rcp_estimate)
    hc_rcp_a_hi = pl.sub(hc_rcp_a_split, hc_rcp_a_delta)
    hc_rcp_a_lo = pl.sub(hc_rcp_estimate, hc_rcp_a_hi)
    hc_rcp_b_split = pl.mul(hc_rcp_bounded, 4097.0)
    hc_rcp_b_delta = pl.sub(hc_rcp_b_split, hc_rcp_bounded)
    hc_rcp_b_hi = pl.sub(hc_rcp_b_split, hc_rcp_b_delta)
    hc_rcp_b_lo = pl.sub(hc_rcp_bounded, hc_rcp_b_hi)
    hc_rcp_product = pl.mul(hc_rcp_estimate, hc_rcp_bounded)
    hc_rcp_high_product = pl.mul(hc_rcp_a_hi, hc_rcp_b_hi)
    hc_rcp_error = pl.sub(hc_rcp_high_product, hc_rcp_product)
    hc_rcp_high_low = pl.mul(hc_rcp_a_hi, hc_rcp_b_lo)
    hc_rcp_error = pl.add(hc_rcp_error, hc_rcp_high_low)
    hc_rcp_low_high = pl.mul(hc_rcp_a_lo, hc_rcp_b_hi)
    hc_rcp_error = pl.add(hc_rcp_error, hc_rcp_low_high)
    hc_rcp_low_product = pl.mul(hc_rcp_a_lo, hc_rcp_b_lo)
    hc_rcp_error = pl.add(hc_rcp_error, hc_rcp_low_product)
    hc_rcp_one = pl.add(pl.sub(hc_rcp_bounded, hc_rcp_bounded), 1.0)
    hc_rcp_residual_hi = pl.sub(hc_rcp_one, hc_rcp_product)
    hc_rcp_residual = pl.sub(hc_rcp_residual_hi, hc_rcp_error)
    hc_rcp_correction = pl.div(hc_rcp_residual, hc_rcp_bounded)
    hc_rcp_corrected = pl.add(hc_rcp_estimate, hc_rcp_correction)
    hc_rcp_overflow = pl.sub(hc_rcp_denominator, 4294967296.0)
    hc_rcp_overflow = pl.mul(hc_rcp_overflow, 0.001953125)
    hc_rcp_overflow = pl.minimum(pl.maximum(hc_rcp_overflow, 0.0), 1.0)
    hc_rcp_inside = pl.sub(hc_rcp_one, hc_rcp_overflow)
    hc_rcp_fixed_inside = pl.mul(hc_rcp_corrected, hc_rcp_inside)
    hc_rcp_retained = pl.mul(hc_rcp_original, hc_rcp_overflow)
    hc_rcp_fixed = pl.add(hc_rcp_fixed_inside, hc_rcp_retained)
    return hc_rcp_fixed


@pl.jit.inline(auto_scope=False)
def hc_softmax_exp_math(
    mix: pl.Tensor[[8, 8], pl.FP32],
    inv: pl.Tensor[[8, 1], pl.FP32],
    scales: pl.Tensor[[3], pl.FP32],
    bases: pl.Tensor[[24], pl.FP32],
    group: pl.constexpr,
    offset: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    hc_scale = pl.read(scales, [group])
    hc_base_raw = pl.slice(bases, [8], [offset], valid_shape=[4], clamp=True)
    hc_base = pl.reshape(hc_base_raw, [1, 8])
    hc_norm = pl.row_expand_mul(mix, inv)
    hc_scaled = pl.mul(hc_norm, hc_scale)
    hc_logits = pl.add(hc_scaled, pl.col_expand(mix, hc_base))
    hc_valid = pl.set_validshape(hc_logits, 8, 4)
    hc_padded = pl.fillpad(hc_valid, pad_value=pl.PadValue.min)
    hc_max = pl.row_max(hc_padded)
    hc_exp_argument = pl.create_tensor([8, 8], dtype=pl.FP32)
    hc_exp_argument = pl.row_expand_sub(hc_padded, hc_max)
    hc_exp_value = rne_exp_signed(hc_exp_argument, rows=8, cols=8, safe_zero=True, negate=False)
    return hc_exp_value
