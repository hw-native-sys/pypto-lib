# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""BF16 dot products with FP32 accumulation."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

DOT_M = pl.dynamic("DOT_M")
DOT_K = pl.dynamic("DOT_K")
DOT_W0 = pl.dynamic("DOT_W0")
DOT_W1 = pl.dynamic("DOT_W1")
DOT_K_TILE = 16


def golden_bf16_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Reference BF16 products without CPU GEMM accumulation error at BF16 ties.

    Round the independent FP64 dot product to the FP32 accumulator domain;
    callers retain ownership of their BF16 projection boundaries.
    """
    return (x.double() @ weight.double().t()).float()


@pl.jit.inline
def bf16_linear(
    x: pl.Tensor[[DOT_M, DOT_K], pl.BF16],
    weight: pl.Tensor[[DOT_W0, DOT_W1], pl.BF16],
    x_start: pl.Scalar[pl.INDEX],
    w_start: pl.Scalar[pl.INDEX],
    valid_rows: pl.Scalar[pl.INDEX],
    weight_rows: pl.Scalar[pl.INDEX],
    M_TILE: pl.constexpr,
    N_TILE: pl.constexpr,
    K_DIM: pl.constexpr,
    B_TRANS: pl.constexpr,
):
    """Accumulate BF16 dot-product blocks with compensated FP32 sums."""
    total = pl.full([M_TILE, N_TILE], dtype=pl.FP32, value=0.0)
    correction = pl.full([M_TILE, N_TILE], dtype=pl.FP32, value=0.0)
    for k0 in pl.pipeline(0, K_DIM, 1, stage=1):
        x_part = pl.slice(x, [M_TILE, DOT_K_TILE], [x_start, k0], valid_shape=[valid_rows, 1])
        x_full = pl.set_validshape(pl.fillpad(x_part, pad_value=pl.PadValue.zero), M_TILE, DOT_K_TILE)
        if B_TRANS:
            w_part = pl.slice(weight, [N_TILE, DOT_K_TILE], [w_start, k0], valid_shape=[weight_rows, 1])
            w_full = pl.set_validshape(pl.fillpad(w_part, pad_value=pl.PadValue.zero), N_TILE, DOT_K_TILE)
            partial = pl.matmul(x_full, w_full, out_dtype=pl.FP32, b_trans=True)
        else:
            w_forward = pl.slice(weight, [DOT_K_TILE, N_TILE], [k0, w_start], valid_shape=[1, weight_rows])
            w_forward_full = pl.set_validshape(pl.fillpad(w_forward, pad_value=pl.PadValue.zero), DOT_K_TILE, N_TILE)
            partial = pl.matmul(x_full, w_forward_full, out_dtype=pl.FP32)
        updated = pl.add(total, partial)
        recovered = pl.sub(updated, total)
        left_error = pl.sub(total, pl.sub(updated, recovered))
        right_error = pl.sub(partial, recovered)
        correction = pl.add(correction, pl.add(left_error, right_error))
        total = updated
    return pl.add(total, correction)


@pl.jit(auto_scope=False)
def linear_test(
    x: pl.Tensor[[16, 4096], pl.BF16],
    weight: pl.Tensor[[128, 4096], pl.BF16],
    output: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
    narrowed: pl.Out[pl.Tensor[[16, 128], pl.BF16]],
):
    for block in pl.spmd(1, name_hint="bf16_linear"):
        zero = pl.cast(0, pl.INDEX)
        rows = pl.cast(16, pl.INDEX)
        cols = pl.cast(128, pl.INDEX)
        result = bf16_linear(x, weight, zero, zero, rows, cols, 16, 128, 4096, True)
        output[:, :] = result
        narrowed[:, :] = pl.cast(result, pl.BF16, mode="rint")
    return (output, narrowed)


def golden_linear(
    tensors,
):
    """Reference BF16 operands against an independent high-precision dot product."""
    result = golden_bf16_linear(tensors["x"], tensors["weight"])
    tensors["output"].copy_(result)
    tensors["narrowed"].copy_(result.bfloat16())


def main():
    import argparse

    from golden import TensorSpec, ratio_allclose, run

    parser = argparse.ArgumentParser(description="GLM-5.3-Flash BF16 linear validation")
    parser.add_argument("-p", "--platform", dest="backend_type", default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(533)
    value = torch.randn(16, 4096, generator=generator).bfloat16()
    weight = (torch.randn(128, 4096, generator=generator) / 64).bfloat16()
    result = run(
        fn=linear_test,
        specs=[
            TensorSpec("x", [16, 4096], torch.bfloat16, init_value=value),
            TensorSpec("weight", [128, 4096], torch.bfloat16, init_value=weight),
            TensorSpec("output", [16, 128], torch.float32),
            TensorSpec("narrowed", [16, 128], torch.bfloat16),
        ],
        golden_fn=golden_linear,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        compile_only=args.compile_only,
        compare_fn={
            "output": ratio_allclose(atol=1e-5, rtol=1e-6, max_error_ratio=0),
            "narrowed": ratio_allclose(atol=0, rtol=0, max_error_ratio=0),
        },
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()
