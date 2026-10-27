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

from models.glm5_3_flash.quantization import quantize_bf16_rows, quantize_per_token_int8

DOT_M = pl.dynamic("DOT_M")
DOT_K = pl.dynamic("DOT_K")
DOT_W0 = pl.dynamic("DOT_W0")
DOT_W1 = pl.dynamic("DOT_W1")
DOT_BIAS = pl.dynamic("DOT_BIAS")
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


@pl.jit.inline
def bf16_projection(
    x: pl.Tensor[[DOT_M, DOT_K], pl.BF16],
    weight: pl.Tensor[[DOT_W0, DOT_K], pl.BF16],
    bias: pl.Tensor[[DOT_BIAS], pl.BF16],
    output: pl.Tensor[[DOT_M, DOT_W0], pl.BF16],
    K_DIM: pl.constexpr,
    N_DIM: pl.constexpr,
    HAS_BIAS: pl.constexpr = True,
):
    """Project rows with compensated dot products and an optional bias."""
    rows = pl.tensor.dim(x, 0)
    for block in pl.spmd(((rows + 15) // 16) * ((N_DIM + 63) // 64), name_hint="bf16_projection"):
        row0 = block // ((N_DIM + 63) // 64) * 16
        col0 = block % ((N_DIM + 63) // 64) * 64
        valid_rows = pl.min(16, rows - row0)
        valid_columns = pl.min(64, N_DIM - col0)
        projected = bf16_linear(x, weight, row0, col0, valid_rows, valid_columns, 16, 64, K_DIM, True)
        if HAS_BIAS:
            bias_values = pl.slice(bias, [64], [col0], valid_shape=[valid_columns])
            bias_row = pl.reshape(pl.cast(bias_values, pl.FP32), [1, 64])
            projected = pl.col_expand_add(projected, bias_row)
        narrowed = pl.cast(projected, pl.BF16, mode="rint")
        output[row0:row0 + 16, col0:col0 + 64] = pl.set_validshape(narrowed, valid_rows, valid_columns)
    return output


@pl.jit.inline
def int8_linear_tile(
    x: pl.Tensor[[DOT_M, DOT_K], pl.INT8],
    weight: pl.Tensor[[DOT_W0, DOT_K], pl.INT8],
    token_scale: pl.Tensor[[DOT_M, 8], pl.FP32],
    weight_scale: pl.Tensor[[DOT_W0], pl.FP32],
    x_start: pl.Scalar[pl.INDEX],
    w_start: pl.Scalar[pl.INDEX],
    valid_rows: pl.Scalar[pl.INDEX],
    valid_columns: pl.Scalar[pl.INDEX],
    M_TILE: pl.constexpr,
    N_TILE: pl.constexpr,
    K_DIM: pl.constexpr,
):
    """Return a dynamically quantized INT8 dot product with FP32 per-row/channel dequantization."""
    accum = pl.create_tensor([M_TILE, N_TILE], dtype=pl.INT32)
    for k0 in pl.pipeline(0, K_DIM, 128, stage=2):
        valid_k = pl.min(128, K_DIM - k0)
        source = pl.slice(x, [M_TILE, 128], [x_start, k0], valid_shape=[valid_rows, valid_k])
        source_full = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), M_TILE, 128)
        projection = pl.slice(weight, [N_TILE, 128], [w_start, k0], valid_shape=[valid_columns, valid_k])
        projection_full = pl.set_validshape(pl.fillpad(projection, pad_value=pl.PadValue.zero), N_TILE, 128)
        accum = pl.matmul_acc(accum, source_full, projection_full, b_trans=True, init_cond=(k0 == 0))
    row_scale = pl.slice(token_scale, [M_TILE, 8], [x_start, 0], valid_shape=[valid_rows, 8])
    row_scale_column = pl.row_max(row_scale)
    channel_scale = pl.slice(weight_scale, [N_TILE], [w_start], valid_shape=[valid_columns])
    channel_scale_row = pl.reshape(channel_scale, [1, N_TILE])
    products = pl.cast(accum, pl.FP32, mode="rint")
    row_dequantized = pl.row_expand_mul(products, row_scale_column)
    return pl.col_expand_mul(row_dequantized, channel_scale_row)


@pl.jit.inline
def w8a8_projection_fp32(
    x: pl.Tensor[[DOT_M, DOT_K], pl.BF16],
    weight: pl.Tensor[[DOT_W0, DOT_K], pl.INT8],
    weight_scale: pl.Tensor[[DOT_W0], pl.FP32],
    output: pl.Tensor[[DOT_M, DOT_W0], pl.FP32],
    K_DIM: pl.constexpr,
    N_DIM: pl.constexpr,
):
    """Quantize each activation row and publish dequantized INT32 cube sums."""
    rows = pl.tensor.dim(x, 0)
    quantized = pl.create_tensor([rows, K_DIM], dtype=pl.INT8)
    token_scale = pl.create_tensor([rows, 8], dtype=pl.FP32)
    quantize_bf16_rows(x, quantized, token_scale)
    for block in pl.spmd(((rows + 15) // 16) * ((N_DIM + 127) // 128), name_hint="w8a8_projection"):
        row0 = block // ((N_DIM + 127) // 128) * 16
        col0 = block % ((N_DIM + 127) // 128) * 128
        valid_rows = pl.min(16, rows - row0)
        valid_columns = pl.min(128, N_DIM - col0)
        result = int8_linear_tile(quantized, weight, token_scale, weight_scale,
                                  row0, col0, valid_rows, valid_columns, 16, 128, K_DIM)
        output[row0:row0 + 16, col0:col0 + 128] = pl.set_validshape(result, valid_rows, valid_columns)
    return output


@pl.jit.inline
def w8a8_projection_bf16(
    x: pl.Tensor[[DOT_M, DOT_K], pl.BF16],
    weight: pl.Tensor[[DOT_W0, DOT_K], pl.INT8],
    weight_scale: pl.Tensor[[DOT_W0], pl.FP32],
    output: pl.Tensor[[DOT_M, DOT_W0], pl.BF16],
    K_DIM: pl.constexpr,
    N_DIM: pl.constexpr,
):
    """Preserve the BF16 boundary after native W8A8 projection."""
    rows = pl.tensor.dim(x, 0)
    projected = pl.create_tensor([rows, N_DIM], dtype=pl.FP32)
    w8a8_projection_fp32(x, weight, weight_scale, projected, K_DIM, N_DIM)
    for block in pl.spmd(((rows + 7) // 8) * ((N_DIM + 127) // 128), name_hint="w8a8_projection_bf16"):
        row0 = block // ((N_DIM + 127) // 128) * 8
        col0 = block % ((N_DIM + 127) // 128) * 128
        valid_rows = pl.min(8, rows - row0)
        valid_columns = pl.min(128, N_DIM - col0)
        source = pl.slice(projected, [8, 128], [row0, col0], valid_shape=[valid_rows, valid_columns])
        output[row0:row0 + 8, col0:col0 + 128] = pl.cast(source, pl.BF16, mode="rint")
    return output


@pl.jit
def int8_linear_test(
    x: pl.Tensor[[DOT_M, 4096], pl.BF16],
    weight: pl.Tensor[[128, 4096], pl.INT8],
    weight_scale: pl.Tensor[[128], pl.FP32],
    quantized: pl.Out[pl.Tensor[[DOT_M, 4096], pl.INT8]],
    token_scale: pl.Out[pl.Tensor[[DOT_M, 8], pl.FP32]],
    output: pl.Out[pl.Tensor[[DOT_M, 128], pl.FP32]],
):
    x.bind_dynamic(0, DOT_M)
    quantized.bind_dynamic(0, DOT_M)
    token_scale.bind_dynamic(0, DOT_M)
    output.bind_dynamic(0, DOT_M)
    quantize_bf16_rows(x, quantized, token_scale)
    rows = pl.tensor.dim(x, 0)
    for block in pl.spmd((rows + 15) // 16, name_hint="int8_linear"):
        row0 = block * 16
        valid_rows = pl.min(16, rows - row0)
        col0 = pl.cast(0, pl.INDEX)
        valid_columns = pl.cast(128, pl.INDEX)
        result = int8_linear_tile(quantized, weight, token_scale, weight_scale,
                                  row0, col0, valid_rows, valid_columns, 16, 128, 4096)
        output[row0:row0 + 16, :] = pl.set_validshape(result, valid_rows, 128)
    return quantized, token_scale, output


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
    from functools import lru_cache

    from golden import TensorSpec, ratio_allclose, run
    from models.glm5_3_flash.quantization import quantize_per_channel_int8
    from models.glm5_3_flash.weights import Checkpoint

    parser = argparse.ArgumentParser(description="GLM-5.3-Flash BF16 and dynamic INT8 linear validation")
    parser.add_argument("-p", "--platform", dest="backend_type", default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--case", choices=["all", "bf16", "int8"], default="all")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokens", type=int, default=17, help="rows in the INT8 validation case")
    args = parser.parse_args()
    if args.checkpoint is not None and args.case != "int8":
        parser.error("--checkpoint requires --case int8")
    if args.case in ("all", "int8"):
        if args.tokens < 1:
            parser.error("--tokens must be positive")

        @lru_cache(maxsize=1)
        def native_inputs():
            generator = torch.Generator().manual_seed(5356)
            if args.checkpoint is None:
                x = torch.randn(args.tokens, 4096, generator=generator).bfloat16()
                weight = torch.randn(128, 4096, generator=generator) / 64
                quantized_weight, scale = quantize_per_channel_int8(weight)
                scale = scale.flatten()
            else:
                checkpoint = Checkpoint(args.checkpoint)
                x = checkpoint.embedding(torch.arange(2000, 2000 + args.tokens))
                key = checkpoint.layer_key(3, "self_attn.q_a_proj.weight")
                pair = checkpoint.matrix(key, slice(0, 128), slice(None), mode="native")
                quantized_weight, scale = pair["weight"], pair["scale"].flatten()
            x[0].zero_()
            return {"x": x, "weight": quantized_weight, "weight_scale": scale}

        def native_golden(values):
            quantized, scales = quantize_per_token_int8(values["x"])
            values["quantized"].copy_(quantized)
            values["token_scale"].copy_(scales.expand(-1, 8))
            products = (quantized.int() @ values["weight"].int().t()).float()
            values["output"].copy_(products * scales * values["weight_scale"].float())

        result = run(
            fn=int8_linear_test,
            specs=[
                TensorSpec("x", [args.tokens, 4096], torch.bfloat16, init_value=lambda: native_inputs()["x"]),
                TensorSpec("weight", [128, 4096], torch.int8, init_value=lambda: native_inputs()["weight"]),
                TensorSpec("weight_scale", [128], torch.float32, init_value=lambda: native_inputs()["weight_scale"]),
                TensorSpec("quantized", [args.tokens, 4096], torch.int8),
                TensorSpec("token_scale", [args.tokens, 8], torch.float32),
                TensorSpec("output", [args.tokens, 128], torch.float32),
            ],
            golden_fn=native_golden,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            compile_only=args.compile_only,
            compare_fn={
                "quantized": ratio_allclose(atol=0, rtol=0, max_error_ratio=0),
                "token_scale": ratio_allclose(atol=1e-8, rtol=1e-6, max_error_ratio=0),
                "output": ratio_allclose(atol=1e-5, rtol=1e-5, max_error_ratio=0),
            },
        )
        print(result)
        if not result.passed:
            raise SystemExit(result.error or 1)
        if args.case == "int8":
            return
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
