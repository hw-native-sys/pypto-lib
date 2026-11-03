# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compensated FP32 arithmetic for GLM decode gates and reductions.

These model primitives retain extra product/reduction bits before the FP32
rounding boundary. Inputs must be finite and keep splitter products normal;
normalization denominators must be positive. Row reductions accept at most
128 columns so their fixed-point accumulator cannot overflow."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

FP32_NEG_INF = -3.4028234663852886e38  # Finite mask floor for sort and softmax.
M = pl.dynamic("ELEMENTWISE_M")
N = pl.dynamic("ELEMENTWISE_N")


@pl.jit.inline
def accurate_exp(x: pl.Tensor[[M, N], pl.FP32]):
    """Evaluate exp on [-80, 80], saturating larger sigmoid arguments."""
    limited = pl.maximum(pl.minimum(x, 80.0), -80.0)
    n = pl.cast(pl.mul(limited, 1.4426950408889634), pl.INT32, mode="rint")
    nf = pl.cast(n, pl.FP32, mode="rint")
    rh = pl.sub(limited, pl.mul(nf, 0.693145751953125))
    rl = pl.mul(nf, -1.428606765330187e-06)
    r = pl.add(rh, rl)
    rt = pl.add(pl.sub(rh, r), rl)
    high = pl.add(pl.mul(x, 0.0), 2.755731998149713e-07)
    low = pl.add(pl.mul(x, 0.0), -7.575112209051195e-15)
    # Keep coefficients in compile-time scalars: parallel activation blocks
    # must not initialize and read a shared mutable GM lookup table.
    for degree in pl.unroll(10):
        ch = pl.cast(2.7557318844628753e-06, pl.FP32)
        cl = pl.cast(3.793571224297229e-14, pl.FP32)
        if degree == 1:
            ch = pl.cast(2.4801587642286904e-05, pl.FP32)
            cl = pl.cast(-3.40699609366682e-13, pl.FP32)
        if degree == 2:
            ch = pl.cast(0.00019841270113829523, pl.FP32)
            cl = pl.cast(-2.725596874933456e-12, pl.FP32)
        if degree == 3:
            ch = pl.cast(0.0013888889225199819, pl.FP32)
            cl = pl.cast(-3.3631094437103215e-11, pl.FP32)
        if degree == 4:
            ch = pl.cast(0.008333333767950535, pl.FP32)
            cl = pl.cast(-4.34617203337595e-10, pl.FP32)
        if degree == 5:
            ch = pl.cast(0.0416666679084301, pl.FP32)
            cl = pl.cast(-1.2417634698280722e-09, pl.FP32)
        if degree == 6:
            ch = pl.cast(0.1666666716337204, pl.FP32)
            cl = pl.cast(-4.967053879312289e-09, pl.FP32)
        if degree == 7:
            ch = pl.cast(0.5, pl.FP32)
            cl = pl.cast(0.0, pl.FP32)
        if degree == 8:
            ch = pl.cast(1.0, pl.FP32)
            cl = pl.cast(0.0, pl.FP32)
        if degree == 9:
            ch = pl.cast(1.0, pl.FP32)
            cl = pl.cast(0.0, pl.FP32)
        product = pl.mul(high, r)
        ah = pl.mul(high, 4097.0)
        ah = pl.sub(ah, pl.sub(ah, high))
        al = pl.sub(high, ah)
        bh = pl.mul(r, 4097.0)
        bh = pl.sub(bh, pl.sub(bh, r))
        bl = pl.sub(r, bh)
        error = pl.add(pl.add(pl.sub(pl.mul(ah, bh), product), pl.mul(ah, bl)), pl.mul(al, bh))
        error = pl.add(error, pl.mul(al, bl))
        error = pl.add(error, pl.add(pl.mul(high, rt), pl.mul(low, r)))
        summed = pl.add(product, ch)
        recovered = pl.sub(summed, product)
        tail = pl.add(pl.sub(product, pl.sub(summed, recovered)), pl.add(pl.neg(recovered), ch))
        tail = pl.add(pl.add(error, cl), tail)
        high = pl.add(summed, tail)
        low = pl.add(pl.sub(summed, high), tail)
    bits = pl.shl(pl.add(n, 127), 23)
    scale = pl.reinterpret_view(bits, pl.FP32)
    return pl.mul(pl.add(high, low), scale)


@pl.jit.inline
def accurate_reciprocal(x: pl.Tensor[[M, N], pl.FP32]):
    approximate = pl.recip(x)
    product = pl.mul(x, approximate)
    ah = pl.mul(x, 4097.0)
    ah = pl.sub(ah, pl.sub(ah, x))
    al = pl.sub(x, ah)
    bh = pl.mul(approximate, 4097.0)
    bh = pl.sub(bh, pl.sub(bh, approximate))
    bl = pl.sub(approximate, bh)
    error = pl.add(pl.add(pl.sub(pl.mul(ah, bh), product), pl.mul(ah, bl)), pl.mul(al, bh))
    error = pl.add(error, pl.mul(al, bl))
    residual = pl.sub(pl.add(pl.neg(product), 1.0), error)
    return pl.add(approximate, pl.mul(residual, approximate))


@pl.jit.inline
def accurate_sigmoid(x: pl.Tensor[[M, N], pl.FP32], ROWS: pl.constexpr, COLS: pl.constexpr):
    argument = pl.create_tensor([ROWS, COLS], dtype=pl.FP32)
    argument[:, :] = pl.neg(x)
    exponential = accurate_exp(argument)
    denominator = pl.create_tensor([ROWS, COLS], dtype=pl.FP32)
    denominator[:, :] = pl.add(exponential, 1.0)
    return accurate_reciprocal(denominator)


@pl.jit.inline
def accurate_silu(x: pl.Tensor[[M, N], pl.FP32], ROWS: pl.constexpr, COLS: pl.constexpr):
    """Evaluate SiLU with one FP32 division before the activation is rounded."""
    value = pl.create_tensor([ROWS, COLS], dtype=pl.FP32)
    value[:, :] = x
    argument = pl.create_tensor([ROWS, COLS], dtype=pl.FP32)
    argument[:, :] = pl.neg(value)
    exponential = accurate_exp(argument)
    denominator = pl.create_tensor([ROWS, COLS], dtype=pl.FP32)
    denominator[:, :] = pl.add(exponential, 1.0)
    return accurate_divide(value, denominator)


@pl.jit.inline
def accurate_rsqrt(x: pl.Tensor[[M, N], pl.FP32]):
    approximate = pl.rsqrt(x, high_precision=True)
    squared = pl.mul(approximate, approximate)
    ah = pl.mul(approximate, 4097.0)
    ah = pl.sub(ah, pl.sub(ah, approximate))
    al = pl.sub(approximate, ah)
    bh = pl.mul(approximate, 4097.0)
    bh = pl.sub(bh, pl.sub(bh, approximate))
    bl = pl.sub(approximate, bh)
    pe = pl.add(pl.add(pl.sub(pl.mul(ah, bh), squared), pl.mul(ah, bl)), pl.mul(al, bh))
    pe = pl.add(pe, pl.mul(al, bl))
    square_tail = pe
    product = pl.mul(squared, x)
    ah = pl.mul(squared, 4097.0)
    ah = pl.sub(ah, pl.sub(ah, squared))
    al = pl.sub(squared, ah)
    bh = pl.mul(x, 4097.0)
    bh = pl.sub(bh, pl.sub(bh, x))
    bl = pl.sub(x, bh)
    pe = pl.add(pl.add(pl.sub(pl.mul(ah, bh), product), pl.mul(ah, bl)), pl.mul(al, bh))
    pe = pl.add(pe, pl.mul(al, bl))
    product_tail = pl.add(pe, pl.mul(square_tail, x))
    residual = pl.sub(pl.add(pl.neg(product), 1.0), product_tail)
    correction = pl.mul(pl.mul(approximate, 0.5), residual)
    return pl.add(approximate, correction)


@pl.jit.inline
def accurate_sqrt(x: pl.Tensor[[M, N], pl.FP32]):
    approximate = pl.sqrt(x)
    squared = pl.mul(approximate, approximate)
    ah = pl.mul(approximate, 4097.0)
    ah = pl.sub(ah, pl.sub(ah, approximate))
    al = pl.sub(approximate, ah)
    bh = pl.mul(approximate, 4097.0)
    bh = pl.sub(bh, pl.sub(bh, approximate))
    bl = pl.sub(approximate, bh)
    pe = pl.add(pl.add(pl.sub(pl.mul(ah, bh), squared), pl.mul(ah, bl)), pl.mul(al, bh))
    pe = pl.add(pe, pl.mul(al, bl))
    residual = pl.sub(pl.sub(x, squared), pe)
    correction = pl.div(pl.mul(residual, 0.5), approximate)
    return pl.add(approximate, correction)


@pl.jit.inline
def accurate_divide(a: pl.Tensor[[M, N], pl.FP32], b: pl.Tensor[[M, N], pl.FP32]):
    approximate = pl.div(a, b)
    product = pl.mul(approximate, b)
    ah = pl.mul(approximate, 4097.0)
    ah = pl.sub(ah, pl.sub(ah, approximate))
    al = pl.sub(approximate, ah)
    bh = pl.mul(b, 4097.0)
    bh = pl.sub(bh, pl.sub(bh, b))
    bl = pl.sub(b, bh)
    pe = pl.add(pl.add(pl.sub(pl.mul(ah, bh), product), pl.mul(ah, bl)), pl.mul(al, bh))
    pe = pl.add(pe, pl.mul(al, bl))
    residual = pl.sub(pl.sub(a, product), pe)
    return pl.add(approximate, pl.div(residual, b))


@pl.jit.inline
def accurate_product_error(
    a: pl.Tensor[[M, N], pl.FP32],
    b: pl.Tensor[[M, N], pl.FP32],
    product: pl.Tensor[[M, N], pl.FP32],
):
    """Recover the low product bits under the module's splitter-domain contract."""
    ah = pl.mul(a, 4097.0)
    ah = pl.sub(ah, pl.sub(ah, a))
    al = pl.sub(a, ah)
    bh = pl.mul(b, 4097.0)
    bh = pl.sub(bh, pl.sub(bh, b))
    bl = pl.sub(b, bh)
    error = pl.add(pl.add(pl.sub(pl.mul(ah, bh), product), pl.mul(ah, bl)), pl.mul(al, bh))
    return pl.add(error, pl.mul(al, bl))


@pl.jit.inline
def accurate_divide_parts(
    high: pl.Tensor[[M, N], pl.FP32],
    low: pl.Tensor[[M, N], pl.FP32],
    denominator: pl.Tensor[[M, N], pl.FP32],
    ROWS: pl.constexpr,
    COLS: pl.constexpr,
):
    """Divide a high/low numerator without dropping its reduction tail first."""
    high_buffer = pl.create_tensor([ROWS, COLS], dtype=pl.FP32)
    high_buffer[:, :] = high
    low_buffer = pl.create_tensor([ROWS, COLS], dtype=pl.FP32)
    low_buffer[:, :] = low
    divisor = pl.create_tensor([ROWS, COLS], dtype=pl.FP32)
    divisor[:, :] = denominator
    estimate = pl.create_tensor([ROWS, COLS], dtype=pl.FP32)
    estimate[:, :] = accurate_divide(high_buffer, divisor)
    product = pl.create_tensor([ROWS, COLS], dtype=pl.FP32)
    product[:, :] = pl.mul(estimate, divisor)
    product_error = pl.create_tensor([ROWS, COLS], dtype=pl.FP32)
    product_error[:, :] = accurate_product_error(estimate, divisor, product)
    residual = pl.create_tensor([ROWS, COLS], dtype=pl.FP32)
    residual[:, :] = pl.add(pl.sub(pl.sub(high_buffer, product), product_error), low_buffer)
    correction = pl.create_tensor([ROWS, COLS], dtype=pl.FP32)
    correction[:, :] = accurate_divide(residual, divisor)
    return pl.add(estimate, correction)


@pl.jit.inline
def accurate_row_sum_parts(x: pl.Tensor[[M, N], pl.FP32]):
    """Return high/low row sums for blocks with at most 128 finite columns."""
    magnitude = pl.maximum(pl.row_max(pl.abs(x)), 1e-30)
    magnitude_bits = pl.reinterpret_view(magnitude, pl.INT32)
    exponent = pl.shr(magnitude_bits, 23)
    # Each rounded lane is at most 2**23; 128 lanes fit signed INT32.
    scale_exponent = pl.add(pl.neg(exponent), 276)
    scale_bits = pl.shl(scale_exponent, 23)
    scale = pl.reinterpret_view(scale_bits, pl.FP32)
    scaled = pl.row_expand_mul(x, scale)
    integral = pl.cast(scaled, pl.INT32, mode="rint")
    integral_sum = pl.row_sum(integral)
    high = pl.cast(integral_sum, pl.FP32, mode="rint")
    high_integral = pl.cast(high, pl.INT32, mode="rint")
    lost = pl.cast(pl.sub(integral_sum, high_integral), pl.FP32, mode="rint")
    remainder = pl.sub(scaled, pl.cast(integral, pl.FP32, mode="rint"))
    low = pl.add(pl.row_sum(remainder), lost)
    inverse_bits = pl.shl(pl.sub(exponent, 22), 23)
    inverse_scale = pl.reinterpret_view(inverse_bits, pl.FP32)
    return pl.mul(high, inverse_scale), pl.mul(low, inverse_scale)


@pl.jit.inline
def accurate_row_sum(x: pl.Tensor[[M, N], pl.FP32]):
    high, low = accurate_row_sum_parts(x)
    return pl.add(high, low)


@pl.jit
def elementwise_test(
    x: pl.Tensor[[16, 128], pl.FP32],
    positive: pl.Tensor[[16, 128], pl.FP32],
    summands: pl.Tensor[[16, 128], pl.FP32],
    exponential: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
    sigmoid: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
    root: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
    inverse_root: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
    quotient: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
    total: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
):
    with pl.spmd(1):
        row = pl.tile.get_block_idx() * 16
        argument = x[row:row + 16, 0:128]
        positive_part = positive[row:row + 16, 0:128]
        sum_part = summands[row:row + 16, 0:128]
        exponential[:, :] = accurate_exp(argument)
        sigmoid[:, :] = accurate_sigmoid(argument, 16, 128)
        root[:, :] = accurate_sqrt(positive_part)
        inverse_root[:, :] = accurate_rsqrt(positive_part)
        denominator = pl.create_tensor([16, 128], dtype=pl.FP32)
        denominator[:, :] = pl.add(positive_part, 1.0)
        quotient[:, :] = accurate_divide(argument, denominator)
        total[:, :] = accurate_row_sum(sum_part)
    return exponential, sigmoid, root, inverse_root, quotient, total


def _sum_fp32_reference(
    value: torch.Tensor,
    dim: int = -1,
    keepdim: bool = True,
    tile_size: int | None = None,
) -> torch.Tensor:
    """Round an independent wide reduction to FP32, preserving requested tile boundaries."""
    if tile_size is None:
        return value.double().sum(dim=dim, keepdim=keepdim).float()
    total = None
    for tile in value.split(tile_size, dim=dim):
        partial = tile.float().sum(dim=dim, keepdim=True)
        total = partial if total is None else total + partial
    return total if keepdim else total.squeeze(dim)


def golden_elementwise(tensors):
    """Round independent FP64 unaries/reductions to each FP32 stage."""
    x, positive = tensors["x"], tensors["positive"]
    tensors["exponential"].copy_(torch.exp(x.double()).float())
    denominator = 1.0 + torch.exp(-x.double()).float()
    tensors["sigmoid"].copy_((1.0 / denominator.double()).float())
    tensors["root"].copy_(torch.sqrt(positive.double()).float())
    tensors["inverse_root"].copy_(torch.rsqrt(positive.double()).float())
    tensors["quotient"].copy_((x.double() / (positive + 1.0).double()).float())
    tensors["total"].copy_(tensors["summands"].double().sum(-1, keepdim=True).float())


def main():
    import argparse

    from golden import TensorSpec, ratio_allclose, run

    parser = argparse.ArgumentParser(description="GLM compensated FP32 arithmetic validation")
    parser.add_argument("-p", "--platform", dest="backend_type", default="a2a3", choices=["a2a3"])
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--seed", type=int, default=533)
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(args.seed)
    shape = [16, 128]
    x = torch.rand(shape, generator=generator) * 32.0 - 16.0
    positive = torch.exp(torch.rand(shape, generator=generator) * 32.0 - 16.0)
    summands = torch.randn(shape, generator=generator) * torch.exp(torch.rand(shape, generator=generator) * 20.0 - 10.0)
    names = ("exponential", "sigmoid", "root", "inverse_root", "quotient", "total")
    specs = [
        TensorSpec(name, shape, torch.float32, init_value=value)
        for name, value in (("x", x), ("positive", positive), ("summands", summands))
    ]
    specs += [TensorSpec(name, [16, 1] if name == "total" else shape, torch.float32) for name in names]
    result = run(
        fn=elementwise_test,
        specs=specs,
        golden_fn=golden_elementwise,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        compile_only=args.compile_only,
        compare_fn={name: ratio_allclose(atol=0, rtol=1.2e-07, max_error_ratio=0) for name in names},
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()
