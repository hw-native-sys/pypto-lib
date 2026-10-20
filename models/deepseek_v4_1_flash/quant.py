# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 quantized GEMMs, staged BF16 GEMMs, and checkpoint layouts."""

from collections.abc import Callable

import math

import torch
import pypto.language as pl

from config import FLASH as M


# model config
MX_GROUP = M.weight_block_size[0]

# tiling
MX_M_TILE = 16
MX_K_TILE = 256
MX_SCALE_TILE = MX_K_TILE // 32


FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

# FP4 E2M1 nibble values encoded as FP8 E4M3FN bytes. The device kernel
# gathers two adjacent FP8 bytes from this table for every packed input byte.
FP4_E4M3_CODES = torch.tensor(
    [0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,
     0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC],
    dtype=torch.int16,
)


@pl.jit.inline(auto_scope=False)
def mx_quantize_activation(
    x: pl.Tensor,
    payload: pl.Tensor,
    scale_backing: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    t0: pl.Scalar[pl.INDEX],
    input_n: pl.Scalar[pl.INDEX],
    width: pl.constexpr,
    quant_name: pl.constexpr,
    floor_scale: pl.constexpr,
    group_count: pl.constexpr,
):
    """Quantize one M16 activation row tile into MX payload and packed scales."""
    quant_chunks = 1
    if group_count > 1 and width % (2 * MX_K_TILE) == 0:
        quant_chunks = 2
    for block in pl.spmd(width // (MX_K_TILE * quant_chunks), name_hint=quant_name):
        rows = pl.min(MX_M_TILE, num_tokens - t0)
        for chunk in pl.unroll(2):
            if chunk == 0 or (group_count > 1 and width % (2 * MX_K_TILE) == 0):
                k0 = (chunk * (width // (MX_K_TILE * quant_chunks)) + block) * MX_K_TILE
                source = pl.load(x, [t0, input_n + k0], [MX_M_TILE, MX_K_TILE], valid_shape=[rows, MX_K_TILE])
                source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), MX_M_TILE, MX_K_TILE)
                groups = pl.reshape(pl.cast(source, pl.FP32), [MX_M_TILE * (MX_K_TILE // 32), 32])
                reduce_tmp = pl.create_tile([MX_M_TILE * (MX_K_TILE // 32), 32], dtype=pl.FP32)
                if floor_scale:
                    maximum_floor = pl.row_max(pl.abs(groups), tmp_tile=reduce_tmp)
                    bits_floor = pl.reinterpret_view(maximum_floor, pl.INT32)
                    exponent = pl.maximum(pl.sub(pl.shrs(bits_floor, 23), 8), 0)
                    divisor_floor = pl.reinterpret_view(pl.shls(pl.maximum(exponent, 1), 23), pl.FP32)
                    normalized_floor = pl.row_expand_div(groups, divisor_floor)
                    floor_mask = pl.cast(pl.minimum(exponent, 1), pl.FP32)
                    normalized_floor = pl.row_expand_mul(normalized_floor, floor_mask)
                    normalized = pl.minimum(pl.maximum(normalized_floor, -448.0), 448.0)
                else:
                    maximum = pl.maximum(pl.row_max(pl.abs(groups), tmp_tile=reduce_tmp), 1e-4)
                    bits = pl.reinterpret_view(pl.mul(maximum, 1.0 / 448.0), pl.INT32)
                    exponent = pl.shrs(pl.add(bits, 8388607), 23)
                    group_scale = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
                    normalized = pl.row_expand_div(groups, group_scale)
                quantized = pl.cast(normalized, pl.FP8E4M3FN, mode="rint")
                quantized_tile = pl.reshape(quantized, [MX_M_TILE, MX_K_TILE])
                payload = pl.store(quantized_tile, [0, k0], payload)
                signed = pl.sub(exponent, pl.mul(pl.shrs(exponent, 7), 256))
                codes = pl.reinterpret_view(pl.cast(signed, pl.INT8), pl.UINT8)
                flat = pl.reshape(codes, [1, MX_M_TILE * (MX_K_TILE // 32)])
                pack_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                packed = pl.tmov_x2zz(flat, pack_tmp, group_axis=1, dst_rows=MX_M_TILE, dst_cols=8)
                packed_scale = pl.reinterpret_view(packed, pl.FP8E8M0)
                scale_offset = k0 // MX_K_TILE * MX_M_TILE * (MX_K_TILE // 32)
                packed_flat = pl.reshape(packed_scale, [1, MX_M_TILE * (MX_K_TILE // 32)])
                scale_backing = pl.store(packed_flat, [0, scale_offset], scale_backing)
    return payload, scale_backing


@pl.jit.inline(auto_scope=False)
def mx_projection_from_mx(
    payload: pl.Tensor,
    activation_scale_backing: pl.Tensor,
    weight: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    t0: pl.Scalar[pl.INDEX],
    weight_row: pl.Scalar[pl.INDEX],
    output_col: pl.Scalar[pl.INDEX],
    width: pl.constexpr,
    output_width: pl.constexpr,
    name_hint: pl.constexpr,
    n_tile: pl.constexpr,
    k_tile: pl.constexpr,
    fp32_output: pl.constexpr,
    group_count: pl.constexpr,
):
    """Project one prequantized M16 row tile into BF16 or FP32 output."""
    activation_scale = pl.tensor.view(activation_scale_backing, [MX_M_TILE, width // 32], layout=pl.MX_A_ZZ)
    scale_group_size = (width // 32) * output_width
    scale_backing_flat = pl.tensor.view(weight_scale, [1, group_count * scale_group_size], layout=pl.ND)
    scale_group_offset = weight_row // 32 * output_width
    scale_group_backing = pl.tensor.slice(scale_backing_flat, [1, scale_group_size], [0, scale_group_offset])
    scale_group = pl.tensor.view(scale_group_backing, [width // 32, output_width], layout=pl.MX_B_NN)
    workers = pl.min(output_width // n_tile, pl.system.available_cluster_count())
    for worker in pl.spmd(workers, name_hint=name_hint, sync_start=True):
        block_num = pl.tile.get_block_num()
        for wave in pl.range((output_width // n_tile + block_num - 1) // block_num):
            block = wave * block_num + worker
            if block < output_width // n_tile:
                n0 = block * n_tile
                rows = pl.min(MX_M_TILE, num_tokens - t0)
                a0 = pl.load(payload, [0, 0], [MX_M_TILE, k_tile])
                sa0 = pl.load(activation_scale, [0, 0], [MX_M_TILE, k_tile // 32])
                b0 = pl.load(weight, [weight_row, n0], [k_tile, n_tile])
                sb0 = pl.load(scale_group, [0, n0], [k_tile // 32, n_tile])
                acc = pl.matmul_mx(a0, sa0, b0, sb0)
                for k0 in pl.pipeline(k_tile, width, k_tile, stage=2):
                    a = pl.load(payload, [0, k0], [MX_M_TILE, k_tile])
                    sa = pl.load(activation_scale, [0, k0 // 32], [MX_M_TILE, k_tile // 32])
                    b = pl.load(weight, [weight_row + k0, n0], [k_tile, n_tile])
                    sb = pl.load(scale_group, [k0 // 32, n0], [k_tile // 32, n_tile])
                    acc = pl.matmul_mx_acc(acc, a, sa, b, sb)
                output_n = output_col + n0
                if fp32_output:
                    value_fp32 = pl.mul(acc, 1.0)
                    output = pl.store(pl.set_validshape(value_fp32, rows, n_tile), [t0, output_n], output)
                else:
                    value = pl.cast(acc, pl.BF16, mode="rint")
                    output = pl.store(pl.set_validshape(value, rows, n_tile), [t0, output_n], output)
    return output


@pl.jit.inline(auto_scope=False)
def mx_projection(
    x: pl.Tensor,
    weight: pl.Tensor,
    scale: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.constexpr,
    output_width: pl.constexpr,
    name_hint: pl.constexpr,
    quant_name: pl.constexpr,
    n_tile: pl.constexpr,
    k_tile: pl.constexpr,
    fp32_output: pl.constexpr = False,
    floor_scale: pl.constexpr = False,
    group_count: pl.constexpr = 1,
):
    """Quantize activation tiles once, then run separate MX projection tasks."""
    weight_flat = pl.reshape(weight, [group_count * width, output_width])
    for group in pl.parallel(group_count):
        for mt in pl.parallel((num_tokens + MX_M_TILE - 1) // MX_M_TILE):
            t0 = mt * MX_M_TILE
            payload = pl.create_tensor([MX_M_TILE, width], dtype=pl.FP8E4M3FN)
            scale_backing = pl.create_tensor([1, MX_M_TILE * (width // 32)], dtype=pl.FP8E8M0)
            payload, scale_backing = mx_quantize_activation(
                x, payload, scale_backing, num_tokens, t0, group * width,
                width, quant_name, floor_scale, group_count,
            )
            mx_projection_from_mx(
                payload, scale_backing, weight_flat, scale, output,
                num_tokens, t0, group * width, group * output_width,
                width, output_width, name_hint, n_tile, k_tile, fp32_output, group_count,
            )
    return output


@pl.jit.inline(auto_scope=False)
def mx_projection_from_mx_to_mx(
    payload: pl.Tensor,
    scale_backing: pl.Tensor,
    weight_flat: pl.Tensor,
    scale: pl.Tensor,
    output: pl.Tensor,
    output_scale: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    group: pl.Scalar[pl.INDEX],
    width: pl.constexpr,
    output_width: pl.constexpr,
    name_hint: pl.constexpr,
    n_tile: pl.constexpr,
    k_tile: pl.constexpr,
    group_count: pl.constexpr,
    pack_cols: pl.constexpr,
):
    """Project a prequantized M16 group, round to BF16, and quantize its MX output."""
    activation_scale = pl.tensor.view(scale_backing, [MX_M_TILE, width // 32], layout=pl.MX_A_ZZ)
    scale_group_size = (width // 32) * output_width
    scale_backing_flat = pl.tensor.view(scale, [1, group_count * scale_group_size], layout=pl.ND)
    scale_group_offset = (group * width) // 32 * output_width
    scale_group_backing = pl.tensor.slice(scale_backing_flat, [1, scale_group_size], [0, scale_group_offset])
    scale_group = pl.tensor.view(scale_group_backing, [width // 32, output_width], layout=pl.MX_B_NN)
    workers = pl.min(output_width // n_tile, pl.system.available_cluster_count())
    for worker in pl.spmd(workers, name_hint=name_hint, sync_start=True):
        block_num = pl.tile.get_block_num()
        for wave in pl.range((output_width // n_tile + block_num - 1) // block_num):
            block = wave * block_num + worker
            if block < output_width // n_tile:
                n0 = block * n_tile
                rows = pl.min(MX_M_TILE, num_tokens)
                a0 = pl.load(payload, [0, 0], [MX_M_TILE, k_tile])
                sa0 = pl.load(activation_scale, [0, 0], [MX_M_TILE, k_tile // 32])
                b0 = pl.load(weight_flat, [(group * width), n0], [k_tile, n_tile])
                sb0 = pl.load(scale_group, [0, n0], [k_tile // 32, n_tile])
                acc = pl.matmul_mx(a0, sa0, b0, sb0)
                for k0 in pl.pipeline(k_tile, width, k_tile, stage=2):
                    a = pl.load(payload, [0, k0], [MX_M_TILE, k_tile])
                    sa = pl.load(activation_scale, [0, k0 // 32], [MX_M_TILE, k_tile // 32])
                    b = pl.load(weight_flat, [(group * width) + k0, n0], [k_tile, n_tile])
                    sb = pl.load(scale_group, [k0 // 32, n0], [k_tile // 32, n_tile])
                    acc = pl.matmul_mx_acc(acc, a, sa, b, sb)
                rounded = pl.set_validshape(pl.cast(acc, pl.BF16, mode="rint"), rows, n_tile)
                source = pl.set_validshape(pl.fillpad(rounded, pad_value=pl.PadValue.zero), MX_M_TILE, n_tile)
                groups = pl.reshape(pl.cast(source, pl.FP32), [MX_M_TILE * (n_tile // 32), 32])
                reduce_tmp = pl.create_tile([MX_M_TILE * (n_tile // 32), 32], dtype=pl.FP32)
                maximum = pl.maximum(pl.row_max(pl.abs(groups), tmp_tile=reduce_tmp), 1e-4)
                bits = pl.reinterpret_view(pl.mul(maximum, 1.0 / 448.0), pl.INT32)
                exponent = pl.shrs(pl.add(bits, 8388607), 23)
                output_divisor = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
                normalized = pl.row_expand_div(groups, output_divisor)
                quantized = pl.cast(normalized, pl.FP8E4M3FN, mode="rint")
                quantized_tile = pl.reshape(quantized, [MX_M_TILE, n_tile])
                output_n = group * output_width + n0
                output = pl.store(quantized_tile, [0, output_n], output)
                signed = pl.sub(exponent, pl.mul(pl.shrs(exponent, 7), 256))
                codes = pl.reinterpret_view(pl.cast(signed, pl.INT8), pl.UINT8)
                flat = pl.reshape(codes, [1, MX_M_TILE * (n_tile // 32)])
                pack_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                packed = pl.tmov_x2zz(flat, pack_tmp, group_axis=1, dst_rows=MX_M_TILE, dst_cols=pack_cols)
                packed_scale = pl.reinterpret_view(packed, pl.FP8E8M0)
                packed_flat = pl.reshape(packed_scale, [1, MX_M_TILE * (n_tile // 32)])
                scale_offset = output_n // 32 * MX_M_TILE
                output_scale = pl.store(packed_flat, [0, scale_offset], output_scale)
    return output, output_scale



@pl.jit.inline(auto_scope=False)
def mxfp8_gemm(
    x: pl.Tensor,
    weight: pl.Tensor,
    scale: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.constexpr,
    output_width: pl.constexpr,
    fp32_output: pl.constexpr,
    n_tile: pl.constexpr,
    name_hint: pl.constexpr,
    single_task: pl.constexpr,
):
    """Compute native M32 MXFP8 products with in-task activation quantization."""
    task_groups = (num_tokens + 31) // 32
    row_iterations = 1
    if single_task:
        row_iterations = task_groups
        task_groups = pl.min(task_groups, 1)
    for group in pl.parallel(task_groups):
        for block in pl.spmd(output_width // n_tile, name_hint=name_hint):
            for row in pl.range(row_iterations):
                t0 = group * 32
                if single_task:
                    t0 = row * 32
                n0 = block * n_tile
                rows = pl.min(32, num_tokens - t0)
                first = pl.load(x, [t0, 0], [32, MX_K_TILE], valid_shape=[rows, MX_K_TILE])
                first = pl.set_validshape(pl.fillpad(first, pad_value=pl.PadValue.zero), 32, MX_K_TILE)
                firstq_values = pl.reshape(pl.cast(first, pl.FP32), [32 * (MX_K_TILE // 32), 32])
                firstq_reduce_tmp = pl.create_tile([32 * (MX_K_TILE // 32), 32], dtype=pl.FP32)
                firstq_maximum = pl.maximum(pl.row_max(pl.abs(firstq_values), tmp_tile=firstq_reduce_tmp), 1e-4)
                firstq_bits = pl.reinterpret_view(pl.mul(firstq_maximum, 1.0 / 448.0), pl.INT32)
                firstq_exponent = pl.shrs(pl.add(firstq_bits, 8388607), 23)
                firstq_scale = pl.reinterpret_view(pl.shls(firstq_exponent, 23), pl.FP32)
                firstq_quantized = pl.cast(pl.row_expand_div(firstq_values, firstq_scale), pl.FP8E4M3FN, mode="rint")
                firstq_payload = pl.reshape(firstq_quantized, [32, MX_K_TILE])
                firstq_signed_exponent = pl.sub(firstq_exponent, pl.mul(pl.shrs(firstq_exponent, 7), 256))
                firstq_codes = pl.reinterpret_view(pl.cast(firstq_signed_exponent, pl.INT8), pl.UINT8)
                firstq_flat = pl.reshape(firstq_codes, [1, 32 * (MX_K_TILE // 32)])
                firstq_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                firstq_packed = pl.tmov_x2zz(firstq_flat, firstq_tmp, group_axis=1, dst_rows=32, dst_cols=MX_SCALE_TILE)
                a0 = firstq_payload
                sa0 = pl.reinterpret_view(firstq_packed, pl.FP8E8M0)
                b0 = pl.load(weight, [0, n0], [MX_K_TILE, n_tile])
                sb0 = pl.load(scale, [0, n0], [MX_K_TILE // 32, n_tile])
                acc = pl.matmul_mx(a0, sa0, b0, sb0)
                for kb in pl.range(1, width // MX_K_TILE):
                    k0 = kb * MX_K_TILE
                    values = pl.load(x, [t0, k0], [32, MX_K_TILE], valid_shape=[rows, MX_K_TILE])
                    values = pl.set_validshape(pl.fillpad(values, pad_value=pl.PadValue.zero), 32, MX_K_TILE)
                    nextq_values = pl.reshape(pl.cast(values, pl.FP32), [32 * (MX_K_TILE // 32), 32])
                    nextq_reduce_tmp = pl.create_tile([32 * (MX_K_TILE // 32), 32], dtype=pl.FP32)
                    nextq_maximum = pl.maximum(pl.row_max(pl.abs(nextq_values), tmp_tile=nextq_reduce_tmp), 1e-4)
                    nextq_bits = pl.reinterpret_view(pl.mul(nextq_maximum, 1.0 / 448.0), pl.INT32)
                    nextq_exponent = pl.shrs(pl.add(nextq_bits, 8388607), 23)
                    nextq_scale = pl.reinterpret_view(pl.shls(nextq_exponent, 23), pl.FP32)
                    nextq_quantized = pl.cast(pl.row_expand_div(nextq_values, nextq_scale), pl.FP8E4M3FN, mode="rint")
                    nextq_payload = pl.reshape(nextq_quantized, [32, MX_K_TILE])
                    nextq_signed_exponent = pl.sub(nextq_exponent, pl.mul(pl.shrs(nextq_exponent, 7), 256))
                    nextq_codes = pl.reinterpret_view(pl.cast(nextq_signed_exponent, pl.INT8), pl.UINT8)
                    nextq_flat = pl.reshape(nextq_codes, [1, 32 * (MX_K_TILE // 32)])
                    nextq_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                    nextq_packed = pl.tmov_x2zz(nextq_flat, nextq_tmp, group_axis=1, dst_rows=32, dst_cols=MX_SCALE_TILE)
                    a = nextq_payload
                    sa = pl.reinterpret_view(nextq_packed, pl.FP8E8M0)
                    b = pl.load(weight, [k0, n0], [MX_K_TILE, n_tile])
                    sb = pl.load(scale, [k0 // 32, n0], [MX_K_TILE // 32, n_tile])
                    acc = pl.matmul_mx_acc(acc, a, sa, b, sb)
                if fp32_output:
                    output = pl.store(pl.set_validshape(pl.mul(acc, 1.0), rows, n_tile), [t0, n0], output)
                else:
                    value = pl.cast(acc, target_type=pl.BF16, mode="rint")
                    output = pl.store(pl.set_validshape(value, rows, n_tile), [t0, n0], output)
    return output


@pl.jit.inline(auto_scope=False)
def bf16_gemm_staged(
    source: pl.Tensor,
    weight: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
    width: pl.constexpr,
    output_width: pl.constexpr,
    n_tile: pl.constexpr,
    k_tile: pl.constexpr,
    workers: pl.constexpr,
    name_hint: pl.constexpr,
):
    """Stage FP32 GEMM results before the separate BF16 rounding task."""
    tokens = pl.tensor.dim(source, 0)
    accumulated = pl.create_tensor([tokens, output_width], dtype=pl.FP32)
    blocks = (num_tokens + 16 - 1) // 16 * (output_width // n_tile)
    task_workers = pl.max(pl.min(blocks, workers), 1)
    entry_deps = pl.array.create(1, pl.TASK_ID)
    entry_deps[0] = ready
    with pl.spmd(task_workers, name_hint=name_hint, deps=[entry_deps[0]]) as cube_tid:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, blocks, task_workers):
            row = block // (output_width // n_tile) * 16
            col = block % (output_width // n_tile) * n_tile
            rows = pl.min(16, num_tokens - row)
            acc = pl.create_tensor([16, n_tile], dtype=pl.FP32)
            for k in pl.range(0, width, k_tile):
                a = pl.slice(source, [16, k_tile], [row, k], valid_shape=[rows, k_tile])
                # Establish Vector startup before the local Cube result transfer (#2829).
                a = pl.cast(pl.cast(a, pl.FP32), pl.BF16, mode="rint")
                w = pl.slice(weight, [k_tile, n_tile], [k, col])
                acc = pl.matmul_acc(acc, a, w, init_cond=(k == 0))
            value = pl.set_validshape(pl.mul(acc, 1.0), rows, n_tile)
            accumulated[row:row + 16, col:col + n_tile] = value
    with pl.spmd(task_workers, name_hint="bf16_projection_cast", deps=[cube_tid]) as cast_tid:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, blocks, task_workers):
            row = block // (output_width // n_tile) * 16
            col = block % (output_width // n_tile) * n_tile
            rows = pl.min(16, num_tokens - row)
            value = pl.slice(accumulated, [16, n_tile], [row, col], valid_shape=[rows, n_tile])
            output[row:row + 16, col:col + n_tile] = pl.cast(value, pl.BF16, mode="rint")
    return output, cast_tid


def build_mxfp4_pair_lut() -> torch.Tensor:
    """Build the two-lane byte LUT consumed by the AIV FP4 decoder."""
    packed = torch.arange(256, dtype=torch.int64)
    codes = FP4_E4M3_CODES.to(torch.int64)
    low = codes[packed & 0x0F]
    high = codes[packed >> 4]
    pairs = low | (high << 8)
    pairs = torch.where(pairs < 0x8000, pairs, pairs - 0x10000)
    return pairs.to(torch.int16).reshape(1, 256).repeat(2, 1).contiguous()


def _pack_mxfp4_nibbles_kn_tiles(
    nibbles_kn: torch.Tensor, k_tile: int, n_tile: int
) -> torch.Tensor:
    """Pack a logical [K, N] nibble matrix into Cube lane rows."""
    k, n = nibbles_kn.shape
    if k % k_tile or n % n_tile or k_tile % 2 or n_tile % 2:
        raise ValueError("MXFP4 tile dimensions must divide K/N and be even")
    k_blocks = k // k_tile
    n_blocks = n // n_tile
    blocked = nibbles_kn.reshape(k_blocks, k_tile, n_blocks, n_tile)
    blocked = blocked.permute(2, 0, 1, 3)
    lanes = blocked.reshape(n_blocks, k_blocks, 2, k_tile // 2, n_tile)
    low = lanes[..., 0::2] & 0x0F
    high = lanes[..., 1::2] & 0x0F
    packed = low | (high << 4)
    lane_bytes = k_tile * n_tile // 4
    return packed.contiguous().reshape(n_blocks * k_blocks * 2, lane_bytes)


def pack_mxfp4_weight_tiles(
    packed_weight_fp4: torch.Tensor, k_tile: int, n_tile: int
) -> torch.Tensor:
    """Reorder checkpoint [N, K/2] bytes into Cube FP4 tiles."""
    payload = packed_weight_fp4.contiguous().view(torch.uint8)
    *lead, n, half_k = payload.shape
    k = 2 * half_k
    matrices = payload.reshape(-1, n, half_k)
    packed_rows = k * n // n_tile
    packed_cols = n_tile // 2
    result = torch.empty(
        [matrices.shape[0], packed_rows, packed_cols], dtype=torch.uint8
    )
    for index, matrix in enumerate(matrices):
        nibbles = torch.empty([n, k], dtype=torch.uint8)
        nibbles[:, 0::2] = matrix & 0x0F
        nibbles[:, 1::2] = matrix >> 4
        result[index] = _pack_mxfp4_nibbles_kn_tiles(
            nibbles.t(), k_tile, n_tile
        ).reshape(packed_rows, packed_cols)
    return result.reshape(*lead, packed_rows, packed_cols)


def unpack_mxfp4_weight_tiles(
    packed: torch.Tensor, k: int, n: int, k_tile: int, n_tile: int
) -> torch.Tensor:
    """Decode Cube FP4 tiles to an FP8 [K, N] reference matrix."""
    *lead, tile_rows, lane_bytes = packed.shape
    k_blocks = k // k_tile
    n_blocks = n // n_tile
    if tile_rows != n_blocks * k_blocks * 2 or lane_bytes != k_tile * n_tile // 4:
        raise ValueError("MXFP4 tile payload does not match requested matrix")
    payloads = packed.contiguous().view(torch.uint8).reshape(-1, tile_rows, lane_bytes)
    output = torch.empty(payloads.shape[0], k, n, dtype=torch.float8_e4m3fn)
    for batch in range(payloads.shape[0]):
        lane_shape = (n_blocks, k_blocks, 2, k_tile // 2, n_tile // 2)
        lane_bytes_view = payloads[batch].reshape(lane_shape)
        low = lane_bytes_view & 0x0F
        high = lane_bytes_view >> 4
        lanes = torch.stack((low, high), dim=-1)
        blocked = lanes.reshape(n_blocks, k_blocks, k_tile, n_tile)
        nibbles_kn = blocked.permute(1, 2, 0, 3).reshape(k, n)
        codes = FP4_E4M3_CODES.to(torch.uint8)[nibbles_kn.to(torch.long)]
        output[batch] = codes.view(torch.float8_e4m3fn)
    return output.reshape(*lead, k, n)


def decode_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Decode UE8M0 storage codes into FP32 powers of two."""
    codes = scale.contiguous().view(torch.uint8)
    return torch.exp2(codes.to(torch.float32) - 127.0)


def encode_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Encode positive FP32 powers of two as UE8M0 storage codes."""
    exponent = torch.round(torch.log2(scale.float())).clamp(-127, 128)
    return (exponent + 127).to(torch.uint8)


def pack_mx_b_scale(scale: torch.Tensor) -> torch.Tensor:
    """Pack logical ``[..., K/32, N]`` scales into the MX_B_NN physical order."""
    *leading, k_groups, output_dim = scale.shape
    if k_groups % 2 or output_dim % 16:
        raise ValueError("MX_B_NN requires an even K-group count and an output dimension divisible by 16")
    packed = scale.reshape(*leading, k_groups // 2, 2, output_dim // 16, 16)
    leading_axes = list(range(len(leading)))
    packed = packed.permute(*leading_axes, len(leading) + 2, len(leading), len(leading) + 3, len(leading) + 1)
    return packed.contiguous().reshape(*leading, k_groups, output_dim)


def unpack_mx_b_scale(scale: torch.Tensor) -> torch.Tensor:
    """Unpack the Cube MX_B_NN scale order into logical ``[..., K/32, N]`` rows."""
    *leading, k_groups, output_dim = scale.shape
    if k_groups % 2 or output_dim % 16:
        raise ValueError("MX_B_NN requires an even K-group count and an output dimension divisible by 16")
    logical = scale.reshape(*leading, output_dim // 16, k_groups // 2, 16, 2)
    leading_axes = list(range(len(leading)))
    logical = logical.permute(
        *leading_axes, len(leading) + 1, len(leading) + 3, len(leading), len(leading) + 2
    )
    return logical.contiguous().reshape(*leading, k_groups, output_dim)


def dequantize_mxfp4(packed_weight: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Decode checkpoint MXFP4 directly into an FP32 output-major matrix."""
    packed = packed_weight.contiguous().view(torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    indices = torch.stack((low, high), dim=-1).flatten(-2)
    values = FP4_VALUES.to(indices.device)[indices.to(torch.long)]
    scales = decode_e8m0(scale_e8m0).repeat_interleave(MX_GROUP, dim=-1)
    return values * scales


def _nearest_fp4_indices(values: torch.Tensor, *, round_to_even: bool = False) -> torch.Tensor:
    table = FP4_VALUES[:8].to(values.device)
    magnitude = values.abs().unsqueeze(-1)
    if round_to_even:
        order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=values.device)
        selected = (magnitude - table[order]).abs().argmin(dim=-1)
        index = order[selected].to(torch.uint8)
    else:
        boundaries = (table[:-1] + table[1:]) * 0.5
        absolute = magnitude.squeeze(-1)
        if torch.any(absolute > table[-1]):
            index = (magnitude - table).abs().argmin(dim=-1).to(torch.uint8)
        else:
            index = torch.bucketize(absolute, boundaries, out_int32=True).to(torch.uint8)
            index.masked_fill_(~torch.isfinite(absolute), 0)
    return index | (torch.signbit(values).to(torch.uint8) << 3)


def _pack_fp4(indices: torch.Tensor) -> torch.Tensor:
    if indices.shape[-1] % 2:
        raise ValueError("packed FP4 requires an even logical last dimension")
    pairs = indices.unflatten(-1, (-1, 2))
    return pairs[..., 0] | (pairs[..., 1] << 4)


def _unpack_fp4(payload: torch.Tensor) -> torch.Tensor:
    packed = payload.contiguous().view(torch.uint8)
    indices = torch.stack((packed & 0x0F, (packed >> 4) & 0x0F), dim=-1).flatten(-2)
    return FP4_VALUES.to(payload.device)[indices.to(torch.long)]


def quantize_mxfp4_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize output-major ``[..., N, K]`` weights to checkpoint MXFP4 carriers."""
    if weight.shape[-1] % MX_GROUP:
        raise ValueError("MXFP4 weights require K divisible by 32")
    grouped = weight.float().unflatten(-1, (-1, MX_GROUP))
    amax = grouped.abs().amax(dim=-1)
    exponent = torch.ceil(torch.log2((amax / 6.0).clamp_min(2.0**-127)))
    scale = torch.exp2(exponent.clamp(-127, 128))
    normalized = (grouped / scale.unsqueeze(-1)).clamp(-6.0, 6.0)
    payload = _pack_fp4(_nearest_fp4_indices(normalized).flatten(-2))
    return payload, encode_e8m0(scale)


def quantize_mxfp8_cache(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dimension to E4M3 payload and group-32 BF16 scales."""
    if value.shape[-1] % MX_GROUP:
        raise ValueError("MXFP8 cache width must be divisible by 32")
    grouped = value.float().unflatten(-1, (-1, MX_GROUP))
    amax = grouped.abs().amax(dim=-1).clamp_min(1e-4)
    scale = (amax / 448.0).to(torch.bfloat16)
    payload = (grouped / scale.float().unsqueeze(-1)).clamp(-448.0, 448.0)
    return payload.flatten(-2).to(torch.float8_e4m3fn), scale


def dequantize_mxfp8_cache(payload: torch.Tensor, scale_bf16: torch.Tensor) -> torch.Tensor:
    """Decode an MXFP8 cache tensor whose groups lie on the last dimension."""
    scales = scale_bf16.float().repeat_interleave(MX_GROUP, dim=-1)
    return payload.float() * scales


def quantize_mxfp4_cache(
    value: torch.Tensor, group_size: int, scale_format: str, *, round_to_even: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dimension to packed E2M1 with E4M3, E8M0 or BF16 scales."""
    if value.shape[-1] % group_size:
        raise ValueError(f"MXFP4 cache width must be divisible by {group_size}")
    grouped = value.float().unflatten(-1, (-1, group_size))
    amax = grouped.abs().amax(dim=-1)
    floor = 2.0**-126 if round_to_even and scale_format == "e8m0" else 2.0**-9
    raw_scale = (amax / 6.0).clamp_min(floor)
    if scale_format == "e8m0":
        exponent = torch.ceil(torch.log2(raw_scale)).clamp(-127, 128)
        scale_value = torch.exp2(exponent)
        stored_scale = encode_e8m0(scale_value)
    elif scale_format == "e4m3":
        stored_scale = raw_scale.clamp(max=448.0).to(torch.float8_e4m3fn)
        scale_value = stored_scale.float()
    elif scale_format == "bf16":
        stored_scale = raw_scale.to(torch.bfloat16)
        scale_value = stored_scale.float()
    else:
        raise ValueError(f"unsupported MXFP4 cache scale format {scale_format!r}")
    normalized = (grouped / scale_value.unsqueeze(-1)).clamp(-6.0, 6.0)
    payload = _pack_fp4(_nearest_fp4_indices(normalized, round_to_even=round_to_even).flatten(-2))
    return payload, stored_scale


def dequantize_mxfp4_cache(
    payload: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    scale_format: str,
) -> torch.Tensor:
    """Decode a packed E2M1 cache tensor with last-dimension MX groups."""
    if scale_format == "e8m0":
        scale_value = decode_e8m0(scale)
    elif scale_format in ("e4m3", "bf16"):
        scale_value = scale.float()
    else:
        raise ValueError(f"unsupported MXFP4 cache scale format {scale_format!r}")
    scales = scale_value.repeat_interleave(group_size, dim=-1)
    return _unpack_fp4(payload) * scales


def dequantize_mxfp8(weight: torch.Tensor, logical_scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Decode an input-major MXFP8 matrix using logical ``[K/32, N]`` scales."""
    scales = decode_e8m0(logical_scale_e8m0).repeat_interleave(MX_GROUP, dim=-2)
    return weight.float() * scales


def _quantize_mxfp8_activation(
    x: torch.Tensor, *, scale_alg: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    *leading, width = x.shape
    if width % MX_GROUP:
        raise ValueError("MXFP8 activations require the last dimension to be divisible by 32")
    grouped = x.float().reshape(*leading, width // MX_GROUP, MX_GROUP)
    amax = grouped.abs().amax(dim=-1)
    if scale_alg == 0:
        # npu_dynamic_mx_quant defaults to OCP: floor(log2(amax)) - 8.
        codes = _e8m0_codes_from_amax(amax)
        scale = decode_e8m0(codes)
        normalized = torch.where(codes.unsqueeze(-1) > 0, grouped / scale.unsqueeze(-1), 0.0)
    elif scale_alg == 1:
        scale = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-4) / 448.0)))
        normalized = grouped / scale.unsqueeze(-1)
    else:
        raise ValueError("MXFP8 scale_alg must be 0 (OCP) or 1 (upward)")
    quantized = normalized.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized, scale


def mxfp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    packed_weight_scale: torch.Tensor | None,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Evaluate a linear projection with an optional explicit output dtype."""
    if packed_weight_scale is None:
        if output_dtype is not None:
            return torch.matmul(x.float(), weight.float()).to(output_dtype)
        return torch.matmul(x, weight)
    activation, activation_scale = _quantize_mxfp8_activation(x)
    logical_scale = unpack_mx_b_scale(packed_weight_scale)
    weight_scale = decode_e8m0(logical_scale)
    weight_groups = weight.float().unflatten(0, (-1, MX_GROUP))
    partials = torch.einsum("...gk,gkn->...gn", activation.float(), weight_groups)
    partials = partials * activation_scale.unsqueeze(-1) * weight_scale
    return partials.sum(dim=-2).to(output_dtype or x.dtype)


def golden_bf16_gemm(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """BF16 GEMM with the A5 Cube's arithmetic: FP32 products, FP32 accumulator, one BF16 cast.

    Torch's own BF16 GEMM rounds through its blocking; one ULP of difference is enough to move a
    published E2M1/E8M0 row one code step, or to reorder a near-tied indexer cutoff.
    """
    return (lhs.float() @ rhs.float()).to(torch.bfloat16)


def golden_mxfp8_quantize(value):
    """Independent translation of official kernel.py act_quant_kernel, group size 32."""
    groups = value.float().unflatten(-1, (-1, MX_GROUP))
    maximum = groups.abs().amax(-1).clamp_min(1e-4)
    exponent = torch.ceil(torch.log2(maximum * (1.0 / 448.0)))
    scale = torch.pow(2.0, exponent)
    payload = (groups / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).flatten(-2)
    return payload, (exponent + 127).to(torch.uint8)


def golden_mxfp8_linear(x, weight, packed_scale, fp32=False):
    """Match official FP8 GEMM's group-32 product, A scale, B scale, FP32 sum."""
    payload, codes = golden_mxfp8_quantize(x)
    activation, weights = payload.float(), weight.float()
    scale_a = decode_e8m0(codes)
    scale_b = decode_e8m0(unpack_mx_b_scale(packed_scale))
    result = torch.zeros(x.shape[0], weight.shape[1], dtype=torch.float32)
    for group in range(weight.shape[0] // MX_GROUP):
        start = group * MX_GROUP
        product = activation[:, start : start + MX_GROUP] @ weights[start : start + MX_GROUP]
        scaled = product * scale_a[:, group : group + 1]
        scaled = scaled * scale_b[group : group + 1]
        result = result + scaled
    return result if fp32 else result.bfloat16()

def quantize_grouped_mxfp8_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize output-major grouped fixtures into [G, K, N] MXFP8 storage."""
    payload, scale = _quantize_mxfp8_activation(weight)
    payload = payload.flatten(-2).transpose(-1, -2).contiguous()
    codes = encode_e8m0(scale).transpose(-1, -2).contiguous()
    return payload, pack_mx_b_scale(codes).view(torch.float8_e8m0fnu)


def grouped_mxfp8_linear(
    grouped: torch.Tensor,
    wo_a: torch.Tensor,
    wo_a_scale: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Evaluate grouped W8A8 MXFP8 with [G, K, N] weights and MX_B_NN scales."""
    activation, activation_scale = _quantize_mxfp8_activation(grouped, scale_alg=0)
    weight_scale = decode_e8m0(unpack_mx_b_scale(wo_a_scale))
    weight_groups = wo_a.float().unflatten(-2, (-1, MX_GROUP))
    partials = torch.einsum("tgmk,gmkr->tgmr", activation.float(), weight_groups)
    partials = partials * activation_scale.unsqueeze(-1) * weight_scale.unsqueeze(0)
    return partials.sum(dim=2).to(output_dtype or grouped.dtype)


# E8M0 has no native dtype in some PyTorch versions, so it is always carried as uint8.
FP8_E4M3_MAX = 448.0

def _e8m0_codes_from_amax(amax: torch.Tensor, fp_max: float = FP8_E4M3_MAX) -> torch.Tensor:
    """OCP shared-exponent scales used by random weight fixtures.

    Grouped Wo-A uses this default OCP policy. The other projection references
    retain the upward E8M0 policy in ``host_quant_mxfp8_v41``.
    """
    format_emax = int(math.floor(math.log2(fp_max)))
    _, exponent = torch.frexp(amax.float())
    codes = exponent.to(torch.int32) - 1 - format_emax + 127
    codes = codes.clamp(0, 255)
    return torch.where(amax == 0, torch.zeros_like(codes), codes).to(torch.uint8)


def _e8m0_to_fp32(codes: torch.Tensor) -> torch.Tensor:
    return torch.exp2(codes.to(torch.float32) - 127.0)


def pack_mx_a_scale(scale: torch.Tensor) -> torch.Tensor:
    m, groups = scale.shape
    if m % 16 or groups % 2:
        raise ValueError("MX_A_ZZ requires rows%16==0 and groups%2==0")
    return scale.reshape(m // 16, 16, groups // 2, 2).permute(0, 2, 1, 3).contiguous().reshape(m, groups)


def unpack_mx_a_scale(scale: torch.Tensor) -> torch.Tensor:
    m, groups = scale.shape
    return scale.reshape(m // 16, groups // 2, 16, 2).permute(0, 2, 1, 3).contiguous().reshape(m, groups)


def host_quant_mxfp8_v41(x: torch.Tensor, *, return_e8m0: bool = False):
    """Quantize with the released group-32 upward E8M0 activation scale."""
    return host_quant_swiglu_mxfp8_v41(x, return_e8m0=return_e8m0)


def host_quant_swiglu_mxfp8_v41(
    x: torch.Tensor, *, return_e8m0: bool = False
):
    """Match the released group-32 MXFP8 quantization and upward scale."""
    value = x.float()
    groups = value.reshape(*value.shape[:-1], -1, MX_GROUP)
    maximum = groups.abs().amax(dim=-1).clamp_min(1e-4)
    # Double precision keeps a one-ULP excess above a power of two from
    # rounding back onto the boundary before ceil.
    scale_exponent = torch.ceil(
        torch.log2((maximum * (1.0 / FP8_E4M3_MAX)).double())
    ).to(torch.int32)
    scale = torch.exp2(scale_exponent).float()
    payload = (
        (groups / scale.unsqueeze(-1))
        .clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .reshape_as(value)
    )
    if not return_e8m0:
        return payload, scale
    codes = (scale_exponent + 127).to(torch.uint8)
    return payload, codes


def gen_mxfp8_weight_kn_v41(out: int, inn: int, dequant_std: float, *, chan_cv: float = 0.5, seed: int = 0):
    """Generate MXFP8 weights in the device K-N layout with packed E8M0 scales."""
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(out, inn, generator=g) * torch.exp(chan_cv * torch.randn(out, 1, generator=g))
    groups = raw.reshape(out, inn // MX_GROUP, MX_GROUP)
    codes = _e8m0_codes_from_amax(groups.abs().amax(-1))
    scale = _e8m0_to_fp32(codes)
    quantized = (groups / scale.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    data_kn = quantized.reshape(out, inn).transpose(0, 1).contiguous()
    codes_kn = codes.transpose(0, 1).contiguous()
    decoded = data_kn.float() * _e8m0_to_fp32(codes_kn).repeat_interleave(MX_GROUP, dim=0)
    gain = dequant_std / decoded.float().std().clamp_min(1e-8)
    shift = int(torch.round(torch.log2(gain)).item())
    codes_kn = (codes_kn.to(torch.int32) + shift).clamp(0, 255).to(torch.uint8)
    scale_e8m0 = pack_mx_b_scale(codes_kn).contiguous()
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e8m0 is not None:
        scale_e8m0 = scale_e8m0.view(e8m0)
    return data_kn, scale_e8m0


def decode_e8m0_codes_v41(scale: torch.Tensor, *, side: str = "a") -> torch.Tensor:
    """Recover logical E8M0 codes from an A-ZZ/B-NN physical scale backing."""
    codes = scale.contiguous().view(torch.uint8)
    if side == "a":
        return unpack_mx_a_scale(codes)
    if side == "b":
        return unpack_mx_b_scale(codes)
    raise ValueError(f"side must be 'a' or 'b', got {side!r}")


def matmul_mx_golden_v41(a, a_scale, b, b_scale):
    """Evaluate the FP32 MX matmul golden using logical E8M0 scales."""
    a_codes = a_scale.contiguous().view(torch.uint8)
    b_codes = b_scale.contiguous().view(torch.uint8)
    a_f = a.float() * _e8m0_to_fp32(a_codes).repeat_interleave(MX_GROUP, -1)
    b_f = b.float() * _e8m0_to_fp32(b_codes).repeat_interleave(MX_GROUP, -2)
    return torch.matmul(a_f.to(torch.float32), b_f.to(torch.float32))


def _copy_cache_rows(destination: torch.Tensor, source: torch.Tensor, rows: torch.Tensor) -> None:
    destination_rows = destination.flatten(0, 1).view(torch.uint8)
    source_rows = source.contiguous().view(torch.uint8).reshape_as(destination_rows[rows])
    destination_rows[rows] = source_rows


def _publish_quantized_cache(
    payload: torch.Tensor,
    scale: torch.Tensor,
    values: torch.Tensor,
    slots: torch.Tensor,
    quantize: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize and replace only mapped rows in a packed cache pair."""
    updated_payload = payload.clone()
    updated_scale = scale.clone()
    valid = slots >= 0
    rows = slots[valid].to(torch.int64)
    if rows.numel():
        row_payload, row_scale = quantize(values[valid])
        _copy_cache_rows(updated_payload, row_payload, rows)
        _copy_cache_rows(updated_scale, row_scale, rows)
    return updated_payload, updated_scale
