# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Per-token symmetric int8 quantization with per-token fp32 scales; bf16 or fp32 input (N must be divisible by 128)."""
import pypto.language as pl

DEFAULT_SHAPE = {"M": 512, "N": 1024, "dtype": "bf16"}


ROWS = 512
COLS = 1024
ROW_TILE = 16
REDUCTION_TILE = 256
QUANT_TILE = 256
FULL_COLS = 1024
FULL_CHUNKS = 4
TAIL_COLS = 0
ROW_BLOCKS = 32
COLUMN_PARTS = 1


@pl.jit
def per_token_quant_bf16(
    x: pl.Tensor[[ROWS, COLS], pl.BF16],
    q: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT8]],
    scale: pl.Out[pl.Tensor[[ROWS, 1], pl.FP32]],
):
    for row_block in pl.spmd(ROW_BLOCKS, name_hint="fused_resident_rows"):
        row0 = row_block * ROW_TILE
        valid_rows = pl.min(ROW_TILE, ROWS - row0)
        values = pl.load(x, [row0, 0], [ROW_TILE, COLS], valid_shape=[valid_rows, COLS])
        maximum = pl.tile.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
        for reduce0 in pl.range(0, FULL_COLS, REDUCTION_TILE):
            reduce_chunk = pl.tile.slice(values, [ROW_TILE, REDUCTION_TILE], [0, reduce0])
            reduce_chunk = pl.cast(reduce_chunk, pl.FP32)
            reduce_chunk = pl.fillpad(pl.set_validshape(reduce_chunk, valid_rows, REDUCTION_TILE), pad_value=pl.PadValue.zero)
            reduction_tmp = pl.create_tile([ROW_TILE, REDUCTION_TILE], dtype=pl.FP32)
            partial_maximum = pl.reshape(pl.row_max(pl.abs(reduce_chunk), tmp_tile=reduction_tmp), [1, ROW_TILE])
            maximum = pl.maximum(maximum, partial_maximum)
        if TAIL_COLS == 128:
            reduce_tail = pl.load(x, [row0, COLS - 128], [ROW_TILE, 128], valid_shape=[valid_rows, 128])
            reduce_tail = pl.cast(reduce_tail, pl.FP32)
            reduce_tail = pl.fillpad(pl.set_validshape(reduce_tail, valid_rows, 128), pad_value=pl.PadValue.zero)
            tail_reduction_tmp = pl.create_tile([ROW_TILE, 128], dtype=pl.FP32)
            tail_maximum = pl.reshape(pl.row_max(pl.abs(reduce_tail), tmp_tile=tail_reduction_tmp), [1, ROW_TILE])
            maximum = pl.maximum(maximum, tail_maximum)
        row_scale = pl.div(maximum, 127.0)
        scale_product = pl.mul(row_scale, 127.0)
        scale_split = pl.mul(row_scale, 4097.0)
        scale_high = pl.sub(scale_split, pl.sub(scale_split, row_scale))
        scale_low = pl.sub(row_scale, scale_high)
        scale_error = pl.add(
            pl.sub(pl.mul(scale_high, 127.0), scale_product),
            pl.mul(scale_low, 127.0),
        )
        scale_residual = pl.sub(pl.sub(maximum, scale_product), scale_error)
        row_scale = pl.add(row_scale, pl.div(scale_residual, 127.0))
        safe_scale_row = pl.tile.select(
            pl.tile.cmps(maximum, 0.0, cmp_type=0), 1.0, row_scale
        )
        safe_scale = pl.reshape(safe_scale_row, [ROW_TILE, 1])
        for col0 in pl.range(0, FULL_COLS, QUANT_TILE):
            chunk = pl.tile.slice(values, [ROW_TILE, QUANT_TILE], [0, col0])
            chunk = pl.cast(chunk, pl.FP32)
            scaled = pl.row_expand_div(chunk, safe_scale)
            product = pl.row_expand_mul(scaled, safe_scale)
            quotient_split = pl.mul(scaled, 4097.0)
            quotient_high = pl.sub(
                quotient_split, pl.sub(quotient_split, scaled)
            )
            quotient_low = pl.sub(scaled, quotient_high)
            denominator_split = pl.mul(safe_scale, 4097.0)
            denominator_high = pl.sub(
                denominator_split, pl.sub(denominator_split, safe_scale)
            )
            denominator_low = pl.sub(safe_scale, denominator_high)
            product_error = pl.sub(
                pl.row_expand_mul(quotient_high, denominator_high), product
            )
            product_error = pl.add(
                product_error,
                pl.row_expand_mul(quotient_high, denominator_low),
            )
            product_error = pl.add(
                product_error,
                pl.row_expand_mul(quotient_low, denominator_high),
            )
            product_error = pl.add(
                product_error,
                pl.row_expand_mul(quotient_low, denominator_low),
            )
            residual = pl.sub(pl.sub(chunk, product), product_error)
            scaled = pl.add(
                scaled, pl.row_expand_div(residual, safe_scale)
            )
            rounded = pl.cast(scaled, pl.INT32, mode="rint")
            bounded = pl.minimum(pl.maximum(rounded, -127), 127)
            half = pl.cast(bounded, pl.FP16, mode="rint")
            quantized = pl.cast(half, pl.INT8, mode="trunc")
            q = pl.store(pl.set_validshape(quantized, valid_rows, QUANT_TILE), [row0, col0], q)
        if TAIL_COLS == 128:
            chunk_tail = pl.load(x, [row0, COLS - 128], [ROW_TILE, 128], valid_shape=[valid_rows, 128])
            chunk_tail = pl.cast(chunk_tail, pl.FP32)
            scaled_tail = pl.row_expand_div(chunk_tail, safe_scale)
            product_tail = pl.row_expand_mul(scaled_tail, safe_scale)
            quotient_split_tail = pl.mul(scaled_tail, 4097.0)
            quotient_high_tail = pl.sub(
                quotient_split_tail, pl.sub(quotient_split_tail, scaled_tail)
            )
            quotient_low_tail = pl.sub(scaled_tail, quotient_high_tail)
            denominator_split_tail = pl.mul(safe_scale, 4097.0)
            denominator_high_tail = pl.sub(
                denominator_split_tail, pl.sub(denominator_split_tail, safe_scale)
            )
            denominator_low_tail = pl.sub(safe_scale, denominator_high_tail)
            product_error_tail = pl.sub(
                pl.row_expand_mul(quotient_high_tail, denominator_high_tail), product_tail
            )
            product_error_tail = pl.add(
                product_error_tail,
                pl.row_expand_mul(quotient_high_tail, denominator_low_tail),
            )
            product_error_tail = pl.add(
                product_error_tail,
                pl.row_expand_mul(quotient_low_tail, denominator_high_tail),
            )
            product_error_tail = pl.add(
                product_error_tail,
                pl.row_expand_mul(quotient_low_tail, denominator_low_tail),
            )
            residual_tail = pl.sub(pl.sub(chunk_tail, product_tail), product_error_tail)
            scaled_tail = pl.add(
                scaled_tail, pl.row_expand_div(residual_tail, safe_scale)
            )
            rounded_tail = pl.cast(scaled_tail, pl.INT32, mode="rint")
            bounded_tail = pl.minimum(pl.maximum(rounded_tail, -127), 127)
            half_tail = pl.cast(bounded_tail, pl.FP16, mode="rint")
            quantized_tail = pl.cast(half_tail, pl.INT8, mode="trunc")
            q = pl.store(pl.set_validshape(quantized_tail, valid_rows, 128), [row0, COLS - 128], q)
        scale = pl.store(pl.set_validshape(safe_scale, valid_rows, 1), [row0, 0], scale)
    return q, scale


@pl.jit
def per_token_quant_fp32(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    q: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT8]],
    scale: pl.Out[pl.Tensor[[ROWS, 1], pl.FP32]],
):
    for row0 in pl.parallel(0, ROWS, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="fp32_streamed_rows"):
            valid_rows = pl.min(ROW_TILE, ROWS - row0)
            maximum = pl.tile.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
            for reduce0 in pl.range(0, FULL_COLS, REDUCTION_TILE):
                reduce_chunk = pl.load(x, [row0, reduce0], [ROW_TILE, REDUCTION_TILE], valid_shape=[valid_rows, REDUCTION_TILE])
                reduce_chunk = pl.fillpad(pl.set_validshape(reduce_chunk, valid_rows, REDUCTION_TILE), pad_value=pl.PadValue.zero)
                reduction_tmp = pl.create_tile([ROW_TILE, REDUCTION_TILE], dtype=pl.FP32)
                partial_maximum = pl.reshape(pl.row_max(pl.abs(reduce_chunk), tmp_tile=reduction_tmp), [1, ROW_TILE])
                maximum = pl.maximum(maximum, partial_maximum)
            if TAIL_COLS == 128:
                reduce_tail = pl.load(x, [row0, COLS - 128], [ROW_TILE, 128], valid_shape=[valid_rows, 128])
                reduce_tail = pl.fillpad(pl.set_validshape(reduce_tail, valid_rows, 128), pad_value=pl.PadValue.zero)
                tail_reduction_tmp = pl.create_tile([ROW_TILE, 128], dtype=pl.FP32)
                tail_maximum = pl.reshape(pl.row_max(pl.abs(reduce_tail), tmp_tile=tail_reduction_tmp), [1, ROW_TILE])
                maximum = pl.maximum(maximum, tail_maximum)
            row_scale = pl.div(maximum, 127.0)
            scale_product = pl.mul(row_scale, 127.0)
            scale_split = pl.mul(row_scale, 4097.0)
            scale_high = pl.sub(scale_split, pl.sub(scale_split, row_scale))
            scale_low = pl.sub(row_scale, scale_high)
            scale_error = pl.add(
                pl.sub(pl.mul(scale_high, 127.0), scale_product),
                pl.mul(scale_low, 127.0),
            )
            scale_residual = pl.sub(pl.sub(maximum, scale_product), scale_error)
            row_scale = pl.add(row_scale, pl.div(scale_residual, 127.0))
            safe_scale_row = pl.tile.select(
                pl.tile.cmps(maximum, 0.0, cmp_type=0), 1.0, row_scale
            )
            safe_scale = pl.reshape(safe_scale_row, [ROW_TILE, 1])
            for col0 in pl.range(0, FULL_COLS, QUANT_TILE):
                chunk = pl.load(x, [row0, col0], [ROW_TILE, QUANT_TILE], valid_shape=[valid_rows, QUANT_TILE])
                scaled = pl.row_expand_div(chunk, safe_scale)
                product = pl.row_expand_mul(scaled, safe_scale)
                quotient_split = pl.mul(scaled, 4097.0)
                quotient_high = pl.sub(
                    quotient_split, pl.sub(quotient_split, scaled)
                )
                quotient_low = pl.sub(scaled, quotient_high)
                denominator_split = pl.mul(safe_scale, 4097.0)
                denominator_high = pl.sub(
                    denominator_split, pl.sub(denominator_split, safe_scale)
                )
                denominator_low = pl.sub(safe_scale, denominator_high)
                product_error = pl.sub(
                    pl.row_expand_mul(quotient_high, denominator_high), product
                )
                product_error = pl.add(
                    product_error,
                    pl.row_expand_mul(quotient_high, denominator_low),
                )
                product_error = pl.add(
                    product_error,
                    pl.row_expand_mul(quotient_low, denominator_high),
                )
                product_error = pl.add(
                    product_error,
                    pl.row_expand_mul(quotient_low, denominator_low),
                )
                residual = pl.sub(pl.sub(chunk, product), product_error)
                scaled = pl.add(
                    scaled, pl.row_expand_div(residual, safe_scale)
                )
                rounded = pl.cast(scaled, pl.INT32, mode="rint")
                bounded = pl.minimum(pl.maximum(rounded, -127), 127)
                half = pl.cast(bounded, pl.FP16, mode="rint")
                quantized = pl.cast(half, pl.INT8, mode="trunc")
                q = pl.store(pl.set_validshape(quantized, valid_rows, QUANT_TILE), [row0, col0], q)
            if TAIL_COLS == 128:
                chunk_tail = pl.load(x, [row0, COLS - 128], [ROW_TILE, 128], valid_shape=[valid_rows, 128])
                scaled_tail = pl.row_expand_div(chunk_tail, safe_scale)
                product_tail = pl.row_expand_mul(scaled_tail, safe_scale)
                quotient_split_tail = pl.mul(scaled_tail, 4097.0)
                quotient_high_tail = pl.sub(
                    quotient_split_tail, pl.sub(quotient_split_tail, scaled_tail)
                )
                quotient_low_tail = pl.sub(scaled_tail, quotient_high_tail)
                denominator_split_tail = pl.mul(safe_scale, 4097.0)
                denominator_high_tail = pl.sub(
                    denominator_split_tail, pl.sub(denominator_split_tail, safe_scale)
                )
                denominator_low_tail = pl.sub(safe_scale, denominator_high_tail)
                product_error_tail = pl.sub(
                    pl.row_expand_mul(quotient_high_tail, denominator_high_tail), product_tail
                )
                product_error_tail = pl.add(
                    product_error_tail,
                    pl.row_expand_mul(quotient_high_tail, denominator_low_tail),
                )
                product_error_tail = pl.add(
                    product_error_tail,
                    pl.row_expand_mul(quotient_low_tail, denominator_high_tail),
                )
                product_error_tail = pl.add(
                    product_error_tail,
                    pl.row_expand_mul(quotient_low_tail, denominator_low_tail),
                )
                residual_tail = pl.sub(pl.sub(chunk_tail, product_tail), product_error_tail)
                scaled_tail = pl.add(
                    scaled_tail, pl.row_expand_div(residual_tail, safe_scale)
                )
                rounded_tail = pl.cast(scaled_tail, pl.INT32, mode="rint")
                bounded_tail = pl.minimum(pl.maximum(rounded_tail, -127), 127)
                half_tail = pl.cast(bounded_tail, pl.FP16, mode="rint")
                quantized_tail = pl.cast(half_tail, pl.INT8, mode="trunc")
                q = pl.store(pl.set_validshape(quantized_tail, valid_rows, 128), [row0, COLS - 128], q)
            scale = pl.store(pl.set_validshape(safe_scale, valid_rows, 1), [row0, 0], scale)
    return q, scale


def golden_quant(tensors):
    import torch

    values = tensors["x"].float()
    maximum = values.abs().amax(dim=-1, keepdim=True)
    row_scale = torch.where(maximum == 0.0, torch.ones_like(maximum), maximum / 127.0)
    tensors["scale"][:] = row_scale
    tensors["q"][:] = torch.round(values / row_scale).clamp(-127, 127).to(torch.int8)


def build(shape):
    import torch
    from golden import TensorSpec

    global ROWS, COLS, FULL_COLS, FULL_CHUNKS, TAIL_COLS, ROW_BLOCKS
    fn = per_token_quant_bf16 if shape["dtype"] == "bf16" else per_token_quant_fp32
    torch_dtype = torch.bfloat16 if shape["dtype"] == "bf16" else torch.float32
    ROWS = int(shape["M"])
    COLS = int(shape["N"])
    if ROWS < 0 or COLS <= 0 or COLS % 128 or shape["dtype"] not in {"bf16", "fp32"}:
        raise ValueError("Expected nonnegative M, positive N divisible by 128, and bf16/fp32")
    FULL_COLS = (COLS // 256) * 256
    FULL_CHUNKS = COLS // 256
    TAIL_COLS = COLS % 256
    ROW_BLOCKS = (ROWS + ROW_TILE - 1) // ROW_TILE
    specs = [
        TensorSpec("x", [ROWS, COLS], torch_dtype, init_value=torch.randn),
        TensorSpec("q", [ROWS, COLS], torch.int8),
        TensorSpec("scale", [ROWS, 1], torch.float32),
    ]
    return fn, specs, golden_quant, 1e-4, 1e-4

if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    args = parser.parse_args()

    fn, specs, golden_fn, rtol, atol = build(DEFAULT_SHAPE)
    result = run(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn,
        config=dict(
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=rtol,
        atol=atol,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
