# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""BF16-input SwiGLU with row-owned sequential or pipelined chunks."""

import pypto.language as pl

DEFAULT_SHAPE = {"M": 512, "N": 1024, "dtype": "bf16"}


ROW_TILE = 32
COL_TILE = 256
PACKET_ELEMENTS = 4096
INTERIOR_PACKET_ELEMENTS = 7168
INTERIOR_TAIL_ELEMENTS = 2048
ROW_ALIGNED_PACKET_ELEMENTS = 7168
PIPELINE_DEPTH = 2


@pl.jit
def swiglu(g: pl.Tensor, u: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = g.shape
    for row_offset in pl.parallel(0, rows, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swiglu_bf16_row_pipeline"):
            column_tile_count = (cols + COL_TILE - 1) // COL_TILE
            for tile_index in pl.pipeline(0, column_tile_count, stage=2):
                col_offset = tile_index * COL_TILE
                valid_rows = pl.min(ROW_TILE, rows - row_offset)
                valid_cols = pl.min(COL_TILE, cols - col_offset)
                gate_tile_bf16 = pl.slice(
                    g,
                    [ROW_TILE, COL_TILE],
                    [row_offset, col_offset],
                    valid_shape=[valid_rows, valid_cols],
                )
                up_tile_bf16 = pl.slice(
                    u,
                    [ROW_TILE, COL_TILE],
                    [row_offset, col_offset],
                    valid_shape=[valid_rows, valid_cols],
                )
                gate_fp32 = pl.cast(gate_tile_bf16, target_type=pl.FP32)
                up_fp32 = pl.cast(up_tile_bf16, target_type=pl.FP32)
                denominator = pl.add(pl.exp(pl.neg(gate_fp32)), 1.0)
                activation = pl.div(gate_fp32, denominator)
                result = pl.mul(activation, up_fp32)
                output_tile_bf16 = pl.cast(result, target_type=pl.BF16)
                y = pl.assemble(y, output_tile_bf16, [row_offset, col_offset])
    return y


@pl.jit
def swiglu_experiment(g: pl.Tensor, u: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = g.shape
    for row_offset in pl.parallel(0, rows, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swiglu_bf16_row_pipeline"):
            column_tile_count = cols // COL_TILE
            for tile_index in pl.pipeline(0, column_tile_count, stage=2):
                col_offset = tile_index * COL_TILE
                gate_tile_bf16 = pl.slice(
                    g,
                    [ROW_TILE, COL_TILE],
                    [row_offset, col_offset],
                )
                up_tile_bf16 = pl.slice(
                    u,
                    [ROW_TILE, COL_TILE],
                    [row_offset, col_offset],
                )
                gate_fp32 = pl.cast(gate_tile_bf16, target_type=pl.FP32)
                up_fp32 = pl.cast(up_tile_bf16, target_type=pl.FP32)
                denominator = pl.add(pl.exp(pl.neg(gate_fp32)), 1.0)
                activation = pl.div(gate_fp32, denominator)
                result = pl.mul(activation, up_fp32)
                output_tile_bf16 = pl.cast(result, target_type=pl.BF16)
                y = pl.assemble(y, output_tile_bf16, [row_offset, col_offset])
    return y


@pl.jit
def swiglu_packets(g: pl.Tensor, u: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = g.shape
    total_elements = rows * cols
    gate_flat = pl.reshape(g, [1, total_elements])
    up_flat = pl.reshape(u, [1, total_elements])
    output_flat = pl.reshape(y, [1, total_elements])
    for block_index in pl.spmd(rows // ROW_TILE, name_hint="swiglu_bf16_slab_packets"):
        row_offset = block_index * ROW_TILE
        pl.set_cache_policy(gate_flat, pl.CachePolicy.BYPASS)
        pl.set_cache_policy(up_flat, pl.CachePolicy.BYPASS)
        packet_count = ROW_TILE * cols // PACKET_ELEMENTS
        for packet_index in pl.pipeline(0, packet_count, stage=2):
            element_offset = row_offset * cols + packet_index * PACKET_ELEMENTS
            gate_tile_bf16 = pl.slice(
                gate_flat,
                [1, PACKET_ELEMENTS],
                [0, element_offset],
            )
            up_tile_bf16 = pl.slice(
                up_flat,
                [1, PACKET_ELEMENTS],
                [0, element_offset],
            )
            gate_fp32 = pl.cast(gate_tile_bf16, target_type=pl.FP32)
            up_fp32 = pl.cast(up_tile_bf16, target_type=pl.FP32)
            denominator = pl.add(pl.exp(pl.neg(gate_fp32)), 1.0)
            activation = pl.div(gate_fp32, denominator)
            result = pl.mul(activation, up_fp32)
            output_tile_bf16 = pl.cast(result, target_type=pl.BF16)
            output_flat = pl.assemble(output_flat, output_tile_bf16, [0, element_offset])
    return pl.reshape(output_flat, [rows, cols])


@pl.jit
def swiglu_topology(g: pl.Tensor, u: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = g.shape
    total_elements = rows * cols
    gate_flat = pl.reshape(g, [1, total_elements])
    up_flat = pl.reshape(u, [1, total_elements])
    output_flat = pl.reshape(y, [1, total_elements])
    for stream_index in pl.spmd(128, name_hint="swiglu_independent_streams"):
        pl.set_cache_policy(gate_flat, pl.CachePolicy.BYPASS)
        pl.set_cache_policy(up_flat, pl.CachePolicy.BYPASS)
        packet_count = total_elements // PACKET_ELEMENTS // 128
        for packet_index in pl.pipeline(0, packet_count, stage=2):
            element_offset = (stream_index * packet_count + packet_index) * PACKET_ELEMENTS
            gate_tile_bf16 = pl.slice(
                gate_flat,
                [1, PACKET_ELEMENTS],
                [0, element_offset],
            )
            up_tile_bf16 = pl.slice(
                up_flat,
                [1, PACKET_ELEMENTS],
                [0, element_offset],
            )
            gate_fp32 = pl.cast(gate_tile_bf16, target_type=pl.FP32)
            up_fp32 = pl.cast(up_tile_bf16, target_type=pl.FP32)
            denominator = pl.add(pl.exp(pl.neg(gate_fp32)), 1.0)
            activation = pl.div(gate_fp32, denominator)
            result = pl.mul(activation, up_fp32)
            output_tile_bf16 = pl.cast(result, target_type=pl.BF16)
            output_flat = pl.assemble(output_flat, output_tile_bf16, [0, element_offset])
    return pl.reshape(output_flat, [rows, cols])


@pl.jit
def swiglu_row_aligned(g: pl.Tensor, u: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = g.shape
    total_elements = rows * cols
    gate_flat = pl.reshape(g, [1, total_elements])
    up_flat = pl.reshape(u, [1, total_elements])
    output_flat = pl.reshape(y, [1, total_elements])
    for stream_index in pl.spmd(128, name_hint="swiglu_independent_streams"):
        pl.set_cache_policy(gate_flat, pl.CachePolicy.BYPASS)
        pl.set_cache_policy(up_flat, pl.CachePolicy.BYPASS)
        packet_count = total_elements // ROW_ALIGNED_PACKET_ELEMENTS // 128
        for packet_index in pl.pipeline(0, packet_count, stage=2):
            element_offset = (stream_index * packet_count + packet_index) * ROW_ALIGNED_PACKET_ELEMENTS
            gate_tile_bf16 = pl.slice(
                gate_flat,
                [1, ROW_ALIGNED_PACKET_ELEMENTS],
                [0, element_offset],
            )
            up_tile_bf16 = pl.slice(
                up_flat,
                [1, ROW_ALIGNED_PACKET_ELEMENTS],
                [0, element_offset],
            )
            gate_fp32 = pl.cast(gate_tile_bf16, target_type=pl.FP32)
            up_fp32 = pl.cast(up_tile_bf16, target_type=pl.FP32)
            denominator = pl.add(pl.exp(pl.neg(gate_fp32)), 1.0)
            activation = pl.div(gate_fp32, denominator)
            result = pl.mul(activation, up_fp32)
            output_tile_bf16 = pl.cast(result, target_type=pl.BF16)
            output_flat = pl.assemble(output_flat, output_tile_bf16, [0, element_offset])
    return pl.reshape(output_flat, [rows, cols])


@pl.jit
def swiglu_interior(g: pl.Tensor, u: pl.Tensor, y: pl.Out[pl.Tensor]):
    rows, cols = g.shape
    total_elements = rows * cols
    gate_flat = pl.reshape(g, [1, total_elements])
    up_flat = pl.reshape(u, [1, total_elements])
    output_flat = pl.reshape(y, [1, total_elements])
    for stream_index in pl.spmd(128, name_hint="swiglu_independent_streams"):
        pl.set_cache_policy(gate_flat, pl.CachePolicy.BYPASS)
        pl.set_cache_policy(up_flat, pl.CachePolicy.BYPASS)
        stream_elements = total_elements // 128
        full_packet_count = stream_elements // INTERIOR_PACKET_ELEMENTS
        for packet_index in pl.pipeline(0, full_packet_count, stage=2):
            element_offset = stream_index * stream_elements + packet_index * INTERIOR_PACKET_ELEMENTS
            gate_tile_bf16 = pl.slice(
                gate_flat,
                [1, INTERIOR_PACKET_ELEMENTS],
                [0, element_offset],
            )
            up_tile_bf16 = pl.slice(
                up_flat,
                [1, INTERIOR_PACKET_ELEMENTS],
                [0, element_offset],
            )
            gate_fp32 = pl.cast(gate_tile_bf16, target_type=pl.FP32)
            up_fp32 = pl.cast(up_tile_bf16, target_type=pl.FP32)
            denominator = pl.add(pl.exp(pl.neg(gate_fp32)), 1.0)
            activation = pl.div(gate_fp32, denominator)
            result = pl.mul(activation, up_fp32)
            output_tile_bf16 = pl.cast(result, target_type=pl.BF16)
            output_flat = pl.assemble(output_flat, output_tile_bf16, [0, element_offset])
        tail_element_offset = stream_index * stream_elements + full_packet_count * INTERIOR_PACKET_ELEMENTS
        tail_gate_tile_bf16 = pl.slice(
            gate_flat,
            [1, INTERIOR_TAIL_ELEMENTS],
            [0, tail_element_offset],
        )
        tail_up_tile_bf16 = pl.slice(
            up_flat,
            [1, INTERIOR_TAIL_ELEMENTS],
            [0, tail_element_offset],
        )
        tail_gate_fp32 = pl.cast(tail_gate_tile_bf16, target_type=pl.FP32)
        tail_up_fp32 = pl.cast(tail_up_tile_bf16, target_type=pl.FP32)
        tail_denominator = pl.add(pl.exp(pl.neg(tail_gate_fp32)), 1.0)
        tail_activation = pl.div(tail_gate_fp32, tail_denominator)
        tail_result = pl.mul(tail_activation, tail_up_fp32)
        tail_output_tile_bf16 = pl.cast(tail_result, target_type=pl.BF16)
        output_flat = pl.assemble(output_flat, tail_output_tile_bf16, [0, tail_element_offset])
    return pl.reshape(output_flat, [rows, cols])


def golden_swiglu(t):
    import torch

    t["y"][:] = ((t["g"].float() * torch.sigmoid(t["g"].float())) * t["u"].float()).to(torch.bfloat16)


def build(shape: dict):
    import torch
    from golden import TensorSpec

    rows = int(shape["M"])
    cols = int(shape["N"])
    if shape["dtype"] != "bf16":
        raise ValueError("This campaign kernel requires bf16 inputs")
    if rows < 1 or cols < 1:
        raise ValueError("M and N must be positive")
    specs = [
        TensorSpec("g", [rows, cols], torch.bfloat16, init_value=torch.randn),
        TensorSpec("u", [rows, cols], torch.bfloat16, init_value=torch.randn),
        TensorSpec("y", [rows, cols], torch.bfloat16),
    ]
    kernel = swiglu
    if rows % ROW_TILE == 0 and cols % COL_TILE == 0:
        kernel = swiglu_experiment
    slab_elements = ROW_TILE * cols
    packet_count = slab_elements // PACKET_ELEMENTS
    if (
        rows % ROW_TILE == 0
        and slab_elements % PACKET_ELEMENTS == 0
        and packet_count >= PIPELINE_DEPTH
        and packet_count % PIPELINE_DEPTH == 0
    ):
        kernel = swiglu_packets
    total_packets = rows * cols // PACKET_ELEMENTS
    packets_per_stream = total_packets // 128
    if (
        kernel is swiglu_packets
        and total_packets % 128 == 0
        and packets_per_stream >= PIPELINE_DEPTH
        and packets_per_stream % PIPELINE_DEPTH == 0
    ):
        kernel = swiglu_topology
    if rows == 2048 and cols == 7168:
        aligned_packet_count = rows * cols // ROW_ALIGNED_PACKET_ELEMENTS // 128
        if (
            rows * cols % (128 * ROW_ALIGNED_PACKET_ELEMENTS) == 0
            and aligned_packet_count >= PIPELINE_DEPTH
            and aligned_packet_count % PIPELINE_DEPTH == 0
        ):
            kernel = swiglu_row_aligned
    if rows == 4096 and cols == 4096:
        stream_elements = rows * cols // 128
        full_packet_count = stream_elements // INTERIOR_PACKET_ELEMENTS
        tail_elements = stream_elements - full_packet_count * INTERIOR_PACKET_ELEMENTS
        if (
            stream_elements == full_packet_count * INTERIOR_PACKET_ELEMENTS + INTERIOR_TAIL_ELEMENTS
            and full_packet_count >= PIPELINE_DEPTH
            and full_packet_count % PIPELINE_DEPTH == 0
            and tail_elements == INTERIOR_TAIL_ELEMENTS
        ):
            kernel = swiglu_interior
    return kernel, specs, golden_swiglu, 1.6e-2, 1.6e-2

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
