# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""FP32 exponentials and division for the 16-head, 128-key SWA softmax."""

import pypto.language as pl


def _make_centered_exp(rows, columns):
    """Specialize range reduction for the sink row or the full score tile."""

    @pl.jit.inline(auto_scope=False)
    def compensated_add_product(
        a: pl.Tile[[rows, columns], pl.FP32],
        b: pl.Tile[[rows, columns], pl.FP32],
        c: pl.Tile[[rows, columns], pl.FP32],
    ):
        """Compensate the product before rounding the addition when abs(c) >= abs(a*b)."""
        product = pl.mul(a, b)
        total = pl.add(c, product)
        left = pl.mul(a, 4097.0)
        right = pl.mul(b, 4097.0)
        a_high = pl.sub(left, pl.sub(left, a))
        b_high = pl.sub(right, pl.sub(right, b))
        a_low = pl.sub(a, a_high)
        b_low = pl.sub(b, b_high)
        low = pl.sub(pl.mul(a_high, b_high), product)
        low = pl.add(low, pl.mul(a_high, b_low))
        low = pl.add(low, pl.mul(a_low, b_high))
        low = pl.add(low, pl.mul(a_low, b_low))
        correction = pl.add(pl.sub(c, total), product)
        return pl.add(total, pl.add(correction, low))

    @pl.jit.inline(auto_scope=False)
    def centered_exp(x: pl.Tile[[rows, columns], pl.FP32]):
        """Reduce the argument by a split log(2), then evaluate the FP32 exponential polynomial."""
        bounded = pl.maximum(x, -104.0)
        power = pl.cast(pl.mul(bounded, 1.4426950216293335), pl.INT32, mode="rint")
        power_fp32 = pl.cast(power, pl.FP32)
        reduced = pl.add(bounded, pl.mul(power_fp32, -0.693145751953125))
        reduced = pl.add(reduced, pl.mul(power_fp32, -1.428606765330187e-06))
        polynomial = pl.add(pl.mul(reduced, 0.00019852761761285365), 0.0013930435525253415)
        polynomial = pl.add(pl.mul(polynomial, reduced), 0.008333360776305199)
        coefficient = pl.add(pl.mul(reduced, 0.0), 0.041666485369205475)
        polynomial = compensated_add_product(polynomial, reduced, coefficient)
        coefficient = pl.add(pl.mul(reduced, 0.0), 0.1666666716337204)
        polynomial = compensated_add_product(polynomial, reduced, coefficient)
        coefficient = pl.add(pl.mul(reduced, 0.0), 0.5)
        polynomial = compensated_add_product(polynomial, reduced, coefficient)
        polynomial = pl.add(compensated_add_product(polynomial, pl.mul(reduced, reduced), reduced), 1.0)
        half = pl.shrs(power, 1)
        first_scale = pl.reinterpret_view(pl.shls(pl.add(half, 127), 23), pl.FP32)
        second_scale = pl.reinterpret_view(pl.shls(pl.add(pl.sub(power, half), 127), 23), pl.FP32)
        return pl.mul(pl.mul(polynomial, first_scale), second_scale)

    return centered_exp


centered_exp_line = _make_centered_exp(1, 16)
centered_exp_scores = _make_centered_exp(16, 128)


@pl.jit.inline(auto_scope=False)
def normalized_quotient(numerator: pl.Tile[[8, 128], pl.FP32], denominator: pl.Tile[[8, 128], pl.FP32]):
    """Round the normalized SWA numerator using the remainder of its significand division."""
    numerator_bits = pl.reinterpret_view(pl.abs(numerator), pl.INT32)
    denominator_bits = pl.reinterpret_view(denominator, pl.INT32)
    numerator_mantissa = pl.add(pl.sub(numerator_bits, pl.shls(pl.shrs(numerator_bits, 23), 23)), 8388608)
    denominator_mantissa = pl.add(
        pl.sub(denominator_bits, pl.shls(pl.shrs(denominator_bits, 23), 23)), 8388608
    )
    less = pl.shrs(pl.sub(numerator_mantissa, denominator_mantissa), 31)
    shift = pl.mul(less, -1)
    scale = pl.add(numerator_mantissa, pl.mul(numerator_mantissa, shift))
    estimate = pl.div(pl.cast(scale, pl.FP32), pl.cast(denominator_mantissa, pl.FP32))
    quotient_mantissa = pl.cast(pl.mul(estimate, 8388608.0), pl.INT32, mode="rint")
    for _ in pl.unroll(2):
        residual = pl.sub(pl.shls(scale, 23), pl.mul(quotient_mantissa, denominator_mantissa))
        twice = pl.shls(residual, 1)
        limit = pl.sub(
            denominator_mantissa, pl.sub(quotient_mantissa, pl.shls(pl.shrs(quotient_mantissa, 1), 1))
        )
        high = pl.mul(pl.shrs(pl.sub(limit, twice), 31), -1)
        low = pl.mul(pl.shrs(pl.add(limit, twice), 31), -1)
        quotient_mantissa = pl.add(quotient_mantissa, pl.sub(high, low))
    exponent = pl.sub(pl.add(pl.sub(pl.shrs(numerator_bits, 23), pl.shrs(denominator_bits, 23)), 126), shift)
    bits = pl.add(pl.shls(exponent, 23), quotient_mantissa)
    nonzero = pl.mul(pl.shrs(pl.mul(numerator_bits, -1), 31), -1)
    magnitude = pl.reinterpret_view(pl.mul(bits, nonzero), pl.FP32)
    opposite = pl.mul(numerator, -1.0)
    sign = pl.shrs(pl.reinterpret_view(opposite, pl.INT32), 31)
    signed_quotient = pl.mul(magnitude, pl.cast(pl.sub(pl.mul(sign, -2), 1), pl.FP32))
    normal = pl.tile.cmps(pl.cast(exponent, pl.FP32), -1.0, cmp_type=4)
    fallback = pl.div(numerator, denominator)
    return pl.tile.select(normal, signed_quotient, fallback)
