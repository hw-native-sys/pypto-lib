# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A3 dynamic INT8 Cube projections and ordered BF16 products."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

import torch
import pypto.language as pl
from .activation_quant import quantize_activation, quantize_projection_activation
from .quant import decode_fp8, decode_scale


MM_M_DYN = pl.dynamic("A3_MM_M_DYN")
MM_K_DYN = pl.dynamic("A3_MM_K_DYN")
MM_N_DYN = pl.dynamic("A3_MM_N_DYN")
MM_S_DYN = pl.dynamic("A3_MM_S_DYN")

DOT_K_TILE = 16
EXPERT_N_TILE = 128
EXPERT_K_TILE = 512
PREFILL_EXPERT_LARGE_M_TILE = 64
PREFILL_EXPERT_DOWN_N_TILE = 256
PREFILL_BF16_K_SUBTILE = 64


@pl.jit.inline(auto_scope=False)
def bf16_ordered_accumulate(
    total: pl.Tile[[MM_M_DYN, MM_N_DYN], pl.FP32],
    left: pl.Tile[[MM_M_DYN, MM_K_DYN], pl.BF16],
    right: pl.Tile[[MM_K_DYN, MM_N_DYN], pl.BF16],
    rows: pl.constexpr,
    columns: pl.constexpr,
    width: pl.constexpr,
):
    """Add BF16 outer products to FP32 in increasing reduction order."""
    total = pl.reshape(total, [rows, columns])
    left = pl.reshape(left, [rows, width])
    right = pl.reshape(right, [width, columns])
    left_columns = pl.tile.transpose(left, 0, 1)
    ones = pl.tile.full([rows, columns], dtype=pl.FP32, value=1.0)
    for inner in pl.range(width):
        left_row = pl.tile.slice(left_columns, [1, rows], [inner, 0])
        right_row = pl.tile.slice(right, [1, columns], [inner, 0])
        left_column = pl.reshape(pl.cast(left_row, pl.FP32), [rows, 1])
        right_row = pl.cast(right_row, pl.FP32)
        expanded = pl.row_expand_mul(ones, left_column)
        product = pl.col_expand_mul(expanded, right_row)
        total = pl.add(total, product)
    return total


@pl.jit.inline(auto_scope=False)
def bf16_dot_sequential(
    query: pl.Tensor,
    keys: pl.Tensor,
    query_offset: pl.Scalar[pl.INDEX],
    key_offset: pl.Scalar[pl.INDEX],
    heads: pl.constexpr,
    key_rows: pl.constexpr,
    width: pl.constexpr,
):
    """Accumulate BF16 outer products in increasing K order in FP32."""
    total = pl.tile.full([heads, key_rows], dtype=pl.FP32, value=0.0)
    for first in pl.range(0, width, DOT_K_TILE):
        left = pl.load(query, [query_offset, first], [heads, DOT_K_TILE])
        right = pl.load(keys, [key_offset, first], [key_rows, DOT_K_TILE])
        right_columns = pl.tile.transpose(right, 0, 1)
        total = bf16_ordered_accumulate(total, left, right_columns, heads, key_rows, DOT_K_TILE)
    return total


@pl.jit.inline(auto_scope=False)
def mx_mm_acc(
    acc: pl.Tile[[MM_M_DYN, MM_N_DYN], pl.FP32],
    a: pl.Tile[[MM_M_DYN, MM_K_DYN], pl.UINT8],
    sa: pl.Tile[[MM_M_DYN, MM_S_DYN], pl.UINT8],
    b: pl.Tile[[MM_K_DYN, MM_N_DYN], pl.UINT8],
    sb: pl.Tile[[MM_S_DYN, MM_N_DYN], pl.UINT8],
    m: pl.constexpr,
    k: pl.constexpr,
    n: pl.constexpr,
):
    """Accumulate group-32 products through BF16 Cube and FP32 scale application."""
    acc = pl.reshape(acc, [m, n])
    group_count = k // 32
    left_codes = pl.reshape(sa, [1, m * group_count])
    right_codes = pl.reshape(sb, [1, n * group_count])
    left_factors = decode_scale(left_codes, 1, m * (k // 32))
    right_factors = decode_scale(right_codes, 1, n * (k // 32))
    left_lanes = pl.tile.arange(0, [1, m], dtype=pl.INT32)
    right_lanes = pl.tile.arange(0, [1, n], dtype=pl.INT32)
    left_tmp = pl.create_tile([1, m], dtype=pl.INT32)
    right_tmp = pl.create_tile([1, n], dtype=pl.INT32)
    decoded_a = decode_fp8(a, m, k)
    decoded_b = decode_fp8(b, k, n)
    for group in pl.range(group_count):
        a_group = pl.tile.slice(decoded_a, [m, 32], [0, group * 32])
        b_group = pl.tile.slice(decoded_b, [32, n], [group * 32, 0])
        left = pl.cast(pl.cast(a_group, pl.FP32), pl.BF16, mode="rint")
        right = pl.cast(pl.cast(b_group, pl.FP32), pl.BF16, mode="rint")
        product = pl.matmul(left, right)
        left_indices = pl.add(pl.mul(left_lanes, pl.cast(group_count, pl.INT32)), pl.cast(group, pl.INT32))
        right_indices = pl.add(pl.mul(right_lanes, pl.cast(group_count, pl.INT32)), pl.cast(group, pl.INT32))
        left_scale = pl.tile.gather(left_factors, left_indices, left_tmp)
        left_scale = pl.reshape(left_scale, [m, 1])
        right_scale = pl.tile.gather(right_factors, right_indices, right_tmp)
        scaled_left = pl.row_expand_mul(product, left_scale)
        scaled = pl.col_expand_mul(scaled_left, right_scale)
        acc = pl.add(acc, scaled)
    return acc


@pl.jit.inline(auto_scope=False)
def mx_mm(
    a: pl.Tile[[MM_M_DYN, MM_K_DYN], pl.UINT8],
    sa: pl.Tile[[MM_M_DYN, MM_S_DYN], pl.UINT8],
    b: pl.Tile[[MM_K_DYN, MM_N_DYN], pl.UINT8],
    sb: pl.Tile[[MM_S_DYN, MM_N_DYN], pl.UINT8],
    m: pl.constexpr,
    k: pl.constexpr,
    n: pl.constexpr,
):
    """Compute group-32 products with a zero FP32 accumulator."""
    acc = pl.tile.full([m, n], dtype=pl.FP32, value=0.0)
    return mx_mm_acc(acc, a, sa, b, sb, m, k, n)


def make_gemm_int8_tiled(*, k_stages, pipe_slots):
    @pl.jit.inline(auto_scope=False)
    def gemm_int8_tiled(
        payload: pl.Tensor,
        activation_scale: pl.Tensor,
        weight: pl.Tensor,
        weight_scale: pl.Tensor,
        output: pl.Tensor,
        num_tokens: pl.Scalar[pl.INT32],
        width: pl.constexpr,
        output_width: pl.constexpr,
        name_hint: pl.constexpr,
        n_tile: pl.constexpr,
        k_tile: pl.constexpr,
        output_major: pl.constexpr,
        row_tile: pl.constexpr,
    ):
        """Apply per-token and per-channel scales to an INT8 Cube product."""
        blocks = (num_tokens + row_tile - 1) // row_tile * (output_width // n_tile)
        workers = pl.max(1, pl.min(blocks, pl.system.available_cluster_count()))
        for worker in pl.spmd(workers, name_hint=name_hint, optimizations=[pl.cross_core_slot(slot_num=pipe_slots)]):
            for block in pl.range(worker, blocks, workers):
                row = block // (output_width // n_tile) * row_tile
                col = block % (output_width // n_tile) * n_tile
                valid_rows = pl.max(0, pl.min(row_tile, num_tokens - row))
                seed = pl.tile.create([row_tile, n_tile], dtype=pl.INT32, target_memory=pl.MemorySpace.Acc, compact=True)
                acc = pl.set_validshape(seed, valid_rows, n_tile)
                for k0 in pl.pipeline(0, width, k_tile, stage=k_stages):
                    a = pl.load(payload, [row, k0], [row_tile, k_tile], valid_shape=[valid_rows, k_tile])
                    if output_major:
                        natural = pl.load(weight, [col, k0], [n_tile, k_tile], target_memory=pl.MemorySpace.Mat)
                        transposed = pl.tile.transpose_view(natural)
                        acc = pl.matmul_acc(acc, a, transposed, init_cond=(k0 == 0))
                    else:
                        b = pl.load(weight, [k0, col], [k_tile, n_tile])
                        acc = pl.matmul_acc(acc, a, b, init_cond=(k0 == 0))
                value = pl.cast(acc, pl.FP32)
                a_scale = pl.load(activation_scale, [row, 0], [row_tile, 1], valid_shape=[valid_rows, 1])
                b_scale = pl.load(weight_scale, [0, col], [1, n_tile])
                scaled = pl.row_expand_mul(value, a_scale)
                scaled = pl.col_expand_mul(scaled, b_scale)
                rounded = pl.cast(scaled, pl.BF16, mode="rint")
                output = pl.store(pl.set_validshape(rounded, valid_rows, n_tile), [row, col], output)
        return output

    return gemm_int8_tiled


gemm_int8_tiled = make_gemm_int8_tiled(k_stages=1, pipe_slots=2)
prefill_gemm_int8_tiled = make_gemm_int8_tiled(k_stages=2, pipe_slots=1)


def make_gemm_int8_tiled_nz(*, k_stages, pipe_slots):
    @pl.jit.inline(auto_scope=False)
    def gemm_int8_tiled_nz(
        payload: pl.Tensor,
        activation_scale: pl.Tensor,
        weight: pl.Tensor,
        weight_scale: pl.Tensor,
        output: pl.Tensor,
        num_tokens: pl.Scalar[pl.INT32],
        width: pl.constexpr,
        output_width: pl.constexpr,
        name_hint: pl.constexpr,
        n_tile: pl.constexpr,
        k_tile: pl.constexpr,
        row_tile: pl.constexpr,
    ):
        """Apply per-token and per-channel scales to an NZ weight projection."""
        row_blocks = (num_tokens + row_tile - 1) // row_tile
        col_blocks = output_width // n_tile
        workers = pl.max(1, pl.min(row_blocks * col_blocks, pl.system.available_cluster_count()))
        for worker in pl.spmd(workers, name_hint=name_hint, optimizations=[pl.cross_core_slot(slot_num=pipe_slots)]):
            for panel in pl.range(0, col_blocks):
                col = panel * n_tile
                # Stripe the row tiles and start the NZ column offset at a nonnegative panel.
                first_row = (worker + workers - (panel * row_blocks) % workers) % workers
                for block in pl.range(first_row, row_blocks, workers):
                    row = block * row_tile
                    valid_rows = pl.max(0, pl.min(row_tile, num_tokens - row))
                    seed = pl.tile.create([row_tile, n_tile], dtype=pl.INT32, target_memory=pl.MemorySpace.Acc, compact=True)
                    acc = pl.set_validshape(seed, valid_rows, n_tile)
                    for k0 in pl.pipeline(0, width, k_tile, stage=k_stages):
                        a = pl.load(payload, [row, k0], [row_tile, k_tile], valid_shape=[valid_rows, k_tile])
                        b = pl.load(weight, [k0, col], [k_tile, n_tile], target_memory=pl.Mem.Mat)
                        acc = pl.matmul_acc(acc, a, b, init_cond=(k0 == 0))
                    value = pl.cast(acc, pl.FP32)
                    a_scale = pl.load(activation_scale, [row, 0], [row_tile, 1], valid_shape=[valid_rows, 1])
                    b_scale = pl.load(weight_scale, [0, col], [1, n_tile])
                    scaled = pl.row_expand_mul(value, a_scale)
                    scaled = pl.col_expand_mul(scaled, b_scale)
                    rounded = pl.cast(scaled, pl.BF16, mode="rint")
                    output = pl.store(pl.set_validshape(rounded, valid_rows, n_tile), [row, col], output)
        return output

    return gemm_int8_tiled_nz


gemm_int8_tiled_nz = make_gemm_int8_tiled_nz(k_stages=2, pipe_slots=1)


@pl.jit.inline(auto_scope=False)
def prefill_gemm_int8_reuse_weight(
    payload: pl.Tensor,
    activation_scale: pl.Tensor,
    weight: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.constexpr,
    output_width: pl.constexpr,
    name_hint: pl.constexpr,
    n_tile: pl.constexpr,
    k_tile: pl.constexpr,
    row_tile: pl.constexpr,
):
    """Keep a complete [K, N tile] weight panel in Mat across Prefill row tiles."""
    # Bound the reuse window to spread the rows across workers.
    row_tiles_per_group = pl.min(24, pl.max(8, num_tokens // (2 * row_tile)))
    group_rows = row_tile * row_tiles_per_group
    col_blocks = output_width // n_tile
    groups = (num_tokens + group_rows - 1) // group_rows
    workers = pl.max(1, pl.min(groups * col_blocks, pl.system.available_cluster_count()))
    with pl.spmd(workers, name_hint=name_hint, optimizations=[pl.cross_core_slot(slot_num=1)]):
        worker = pl.tile.get_block_idx()
        for panel in pl.range(0, col_blocks):
            col = panel * n_tile
            for group in pl.range(0, groups):
                if (group * col_blocks + panel) % workers == worker:
                    first_row = group * group_rows
                    last_row = pl.min(first_row + group_rows, num_tokens)
                    weight_panel = pl.load(weight, [0, col], [width, n_tile], target_memory=pl.MemorySpace.Mat)
                    for row in pl.range(first_row, last_row, row_tile):
                        rows = pl.min(row_tile, num_tokens - row)
                        seed = pl.tile.create([row_tile, n_tile], dtype=pl.INT32, target_memory=pl.MemorySpace.Acc, compact=True)
                        acc = pl.set_validshape(seed, rows, n_tile)
                        for k in pl.pipeline(0, width, k_tile, stage=2):
                            a = pl.load(payload, [row, k], [row_tile, k_tile], valid_shape=[rows, k_tile])
                            b = pl.tile.slice(weight_panel, [k_tile, n_tile], [k, 0])
                            acc = pl.matmul_acc(acc, a, b, init_cond=(k == 0))
                        value = pl.cast(acc, pl.FP32)
                        a_scale = pl.load(activation_scale, [row, 0], [row_tile, 1], valid_shape=[rows, 1])
                        b_scale = pl.load(weight_scale, [0, col], [1, n_tile])
                        scaled = pl.row_expand_mul(value, a_scale)
                        scaled = pl.col_expand_mul(scaled, b_scale)
                        rounded = pl.cast(scaled, pl.BF16, mode="rint")
                        output = pl.store(pl.set_validshape(rounded, rows, n_tile), [row, col], output)
    return output


@pl.jit.inline(auto_scope=False)
def gemm_int8_pair_columns(
    payload: pl.Tensor,
    activation_scale: pl.Tensor,
    weight: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.constexpr,
    output_width: pl.constexpr,
    name_hint: pl.constexpr,
    n_tile: pl.constexpr,
    k_tile: pl.constexpr,
    output_major: pl.constexpr,
    row_tile: pl.constexpr,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    fuse_rope: pl.constexpr,
    rope_dim: pl.constexpr,
    rope_offset: pl.constexpr,
    rope_rows: pl.constexpr,
):
    """Compute two adjacent output panels while reusing each activation tile.

    Decode Q-A and Q-B use an even number of output panels. The paired
    schedule keeps one activation tile live while it feeds two Cube products,
    reducing repeated activation loads and the number of SPMD blocks. The
    weight can be packed for direct Cube loads. Q-B can apply RoPE to its second output panel
    before the result is stored. A single-token caller can use one
    vector row for RoPE while retaining the Cube tile dimensions.
    """
    panels = output_width // n_tile
    pairs = panels // 2
    row_blocks = (num_tokens + row_tile - 1) // row_tile
    blocks = row_blocks * pairs
    pair_tiles = blocks
    pair_workers = pl.max(1, pl.min(pair_tiles, pl.system.available_cluster_count()))
    with pl.spmd(pair_workers, name_hint=name_hint):
        pair_worker = pl.tile.get_block_idx()
        for block in pl.range(pair_worker, pair_tiles, pl.tile.get_block_num()):
            row = block // pairs * row_tile
            first_col = (block % pairs) * (2 * n_tile)
            second_col = first_col + n_tile
            valid_rows = pl.max(0, pl.min(row_tile, num_tokens - row))
            first_seed = pl.tile.create(
                [row_tile, n_tile], dtype=pl.INT32,
                target_memory=pl.MemorySpace.Acc, compact=True,
            )
            second_seed = pl.tile.create(
                [row_tile, n_tile], dtype=pl.INT32,
                target_memory=pl.MemorySpace.Acc, compact=True,
            )
            first_acc = pl.set_validshape(first_seed, valid_rows, n_tile)
            second_acc = pl.set_validshape(second_seed, valid_rows, n_tile)
            for k0 in pl.range(0, width, k_tile):
                source = pl.load(
                    payload, [row, k0], [row_tile, k_tile],
                    valid_shape=[valid_rows, k_tile],
                )
                if output_major:
                    first_nk = pl.load(
                        weight, [first_col, k0], [n_tile, k_tile],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    second_nk = pl.load(
                        weight, [second_col, k0], [n_tile, k_tile],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    first_weight_mat = pl.tile.transpose_view(first_nk)
                    second_weight_mat = pl.tile.transpose_view(second_nk)
                    first_acc = pl.matmul_acc(
                        first_acc, source, first_weight_mat, init_cond=(k0 == 0),
                    )
                    second_acc = pl.matmul_acc(
                        second_acc, source, second_weight_mat, init_cond=(k0 == 0),
                    )
                else:
                    first_weight_mat = pl.load(weight, [k0, first_col], [k_tile, n_tile], target_memory=pl.Mem.Mat)
                    second_weight_mat = pl.load(weight, [k0, second_col], [k_tile, n_tile], target_memory=pl.Mem.Mat)
                    first_acc = pl.matmul_acc(
                        first_acc, source, first_weight_mat, init_cond=(k0 == 0),
                    )
                    second_acc = pl.matmul_acc(
                        second_acc, source, second_weight_mat, init_cond=(k0 == 0),
                    )

            a_scale = pl.load(
                activation_scale, [row, 0], [row_tile, 1],
                valid_shape=[valid_rows, 1],
            )
            first_scale = pl.load(weight_scale, [0, first_col], [1, n_tile])
            second_scale = pl.load(weight_scale, [0, second_col], [1, n_tile])

            first_value = pl.row_expand_mul(pl.cast(first_acc, pl.FP32), a_scale)
            first_value = pl.col_expand_mul(first_value, first_scale)
            first_value = pl.cast(first_value, pl.BF16, mode="rint")
            output = pl.store(
                pl.set_validshape(first_value, valid_rows, n_tile),
                [row, first_col], output,
            )

            second_value = pl.row_expand_mul(pl.cast(second_acc, pl.FP32), a_scale)
            second_value = pl.col_expand_mul(second_value, second_scale)
            second_value = pl.cast(second_value, pl.BF16, mode="rint")
            if fuse_rope:
                rope_value = pl.set_validshape(second_value, rope_rows, n_tile)
                tail_bf16 = pl.tile.slice(
                    rope_value, [rope_rows, rope_dim], [0, rope_offset],
                )
                tail = pl.cast(tail_bf16, pl.FP32)
                even = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P0101)
                odd = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P1010)
                cosine = pl.load(
                    rope_cos, [row, 0], [rope_rows, rope_dim // 2],
                    valid_shape=[valid_rows, rope_dim // 2],
                )
                sine = pl.load(
                    rope_sin, [row, 0], [rope_rows, rope_dim // 2],
                    valid_shape=[valid_rows, rope_dim // 2],
                )
                cosine = pl.set_validshape(
                    pl.fillpad(cosine, pad_value=pl.PadValue.zero),
                    rope_rows, rope_dim // 2,
                )
                sine = pl.set_validshape(
                    pl.fillpad(sine, pad_value=pl.PadValue.zero),
                    rope_rows, rope_dim // 2,
                )
                real = pl.sub(pl.mul(even, cosine), pl.mul(odd, sine))
                imag = pl.add(pl.mul(even, sine), pl.mul(odd, cosine))
                real_lanes = pl.tile.full([rope_rows, rope_dim], dtype=pl.FP32, value=0.0)
                real_lanes = pl.tile.scatter_mask(
                    real_lanes, real, mask_pattern=pl.tile.MaskPattern.P0101,
                )
                imag_lanes = pl.tile.full([rope_rows, rope_dim], dtype=pl.FP32, value=0.0)
                imag_lanes = pl.tile.scatter_mask(
                    imag_lanes, imag, mask_pattern=pl.tile.MaskPattern.P1010,
                )
                real_bits = pl.reinterpret_view(real_lanes, pl.INT16)
                imag_bits = pl.reinterpret_view(imag_lanes, pl.INT16)
                rotated = pl.reinterpret_view(pl.or_(real_bits, imag_bits), pl.FP32)
                rotated = pl.cast(rotated, pl.BF16, mode="rint")
                prefix = pl.tile.slice(
                    rope_value, [rope_rows, rope_offset], [0, 0],
                    valid_shape=[valid_rows, rope_offset],
                )
                rotated = pl.tile.slice(
                    rotated, [rope_rows, rope_dim], [0, 0],
                    valid_shape=[valid_rows, rope_dim],
                )
                output = pl.store(
                    prefix, [row, second_col], output,
                )
                output = pl.store(
                    rotated, [row, second_col + rope_offset], output,
                )
            else:
                output = pl.store(
                    pl.set_validshape(second_value, valid_rows, n_tile),
                    [row, second_col], output,
                )
    return output


@pl.jit.inline(auto_scope=False)
def gemm_int8(
    payload: pl.Tensor,
    activation_scale: pl.Tensor,
    weight: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.constexpr,
    output_width: pl.constexpr,
    name_hint: pl.constexpr,
    n_tile: pl.constexpr,
    k_tile: pl.constexpr,
    output_major: pl.constexpr,
):
    """Use larger Cube panels for densely populated projections."""
    if num_tokens >= 32:
        gemm_int8_tiled(payload, activation_scale, weight, weight_scale, output, num_tokens, width, output_width, name_hint, n_tile, k_tile, output_major, 32)
    else:
        gemm_int8_tiled(payload, activation_scale, weight, weight_scale, output, num_tokens, width, output_width, name_hint, n_tile, k_tile, output_major, 16)
    return output


@pl.jit.inline(auto_scope=False)
def expert_swiglu_tiled(
    x: pl.Tensor, factor: pl.Tensor, w1: pl.Tensor, s1: pl.Tensor,
    w3: pl.Tensor, s3: pl.Tensor, route: pl.Tensor, hidden: pl.Tensor,
    count: pl.Scalar[pl.INT32], width: pl.constexpr, intermediate: pl.constexpr,
    n_tile: pl.constexpr, k_tile: pl.constexpr, limit: pl.constexpr,
    weighted: pl.constexpr, hint: pl.constexpr, output_major: pl.constexpr, row_tile: pl.constexpr,
    k_stages: pl.constexpr,
):
    """Share the activation panel between the two INT8 expert projections."""
    panels = intermediate // n_tile
    blocks = (count + row_tile - 1) // row_tile * panels
    workers = pl.max(1, pl.min(blocks, pl.system.available_cluster_count()))
    for worker in pl.spmd(workers, name_hint=hint, optimizations=[pl.cross_core_slot(slot_num=1)]):
        for tile in pl.range(worker, blocks, workers):
            row = tile // panels * row_tile
            col = tile % panels * n_tile
            rows = pl.min(row_tile, count - row)
            gate_seed = pl.tile.create([row_tile, n_tile], dtype=pl.INT32, target_memory=pl.MemorySpace.Acc, compact=True)
            gate_acc = pl.set_validshape(gate_seed, rows, n_tile)
            up_seed = pl.tile.create([row_tile, n_tile], dtype=pl.INT32, target_memory=pl.MemorySpace.Acc, compact=True)
            up_acc = pl.set_validshape(up_seed, rows, n_tile)
            if k_stages > 1:
                for k in pl.pipeline(0, width, k_tile, stage=k_stages):
                    source = pl.load(x, [row, k], [row_tile, k_tile], valid_shape=[rows, k_tile])
                    if output_major:
                        gate_nk = pl.load(w1, [col, k], [n_tile, k_tile], target_memory=pl.MemorySpace.Mat)
                        gate_t = pl.tile.transpose_view(gate_nk)
                        up_nk = pl.load(w3, [col, k], [n_tile, k_tile], target_memory=pl.MemorySpace.Mat)
                        up_t = pl.tile.transpose_view(up_nk)
                        gate_acc = pl.matmul_acc(gate_acc, source, gate_t, init_cond=(k == 0))
                        up_acc = pl.matmul_acc(up_acc, source, up_t, init_cond=(k == 0))
                    else:
                        gate_weights = pl.load(w1, [k, col], [k_tile, n_tile])
                        up_weights = pl.load(w3, [k, col], [k_tile, n_tile])
                        gate_acc = pl.matmul_acc(gate_acc, source, gate_weights, init_cond=(k == 0))
                        up_acc = pl.matmul_acc(up_acc, source, up_weights, init_cond=(k == 0))
            else:
                for k in pl.range(0, width, k_tile):
                    source = pl.load(x, [row, k], [row_tile, k_tile], valid_shape=[rows, k_tile])
                    if output_major:
                        gate_nk = pl.load(w1, [col, k], [n_tile, k_tile], target_memory=pl.MemorySpace.Mat)
                        gate_t = pl.tile.transpose_view(gate_nk)
                        up_nk = pl.load(w3, [col, k], [n_tile, k_tile], target_memory=pl.MemorySpace.Mat)
                        up_t = pl.tile.transpose_view(up_nk)
                        gate_acc = pl.matmul_acc(gate_acc, source, gate_t, init_cond=(k == 0))
                        up_acc = pl.matmul_acc(up_acc, source, up_t, init_cond=(k == 0))
                    else:
                        gate_weights = pl.load(w1, [k, col], [k_tile, n_tile])
                        up_weights = pl.load(w3, [k, col], [k_tile, n_tile])
                        gate_acc = pl.matmul_acc(gate_acc, source, gate_weights, init_cond=(k == 0))
                        up_acc = pl.matmul_acc(up_acc, source, up_weights, init_cond=(k == 0))
            scale = pl.load(factor, [row, 0], [row_tile, 1], valid_shape=[rows, 1])
            gate_scale = pl.load(s1, [0, col], [1, n_tile])
            up_scale = pl.load(s3, [0, col], [1, n_tile])
            gate_fp = pl.col_expand_mul(pl.row_expand_mul(pl.cast(gate_acc, pl.FP32), scale), gate_scale)
            up_fp = pl.col_expand_mul(pl.row_expand_mul(pl.cast(up_acc, pl.FP32), scale), up_scale)
            gate = pl.minimum(pl.cast(pl.cast(gate_fp, pl.BF16, mode="rint"), pl.FP32), limit)
            up = pl.cast(pl.cast(up_fp, pl.BF16, mode="rint"), pl.FP32)
            up = pl.minimum(pl.maximum(up, -limit), limit)
            sigma = pl.recip(pl.add(pl.exp(pl.neg(gate)), 1.0))
            value = pl.mul(pl.mul(gate, sigma), up)
            if weighted:
                coefficients = pl.load(route, [0, row], [1, row_tile], valid_shape=[1, rows])
                value = pl.row_expand_mul(value, pl.reshape(coefficients, [row_tile, 1]))
            result = pl.cast(value, pl.BF16, mode="rint")
            pl.store(pl.set_validshape(result, rows, n_tile), [row, col], hidden)
    return hidden


@pl.jit.inline(auto_scope=False)
def expert_swiglu(
    x: pl.Tensor, factor: pl.Tensor, w1: pl.Tensor, s1: pl.Tensor,
    w3: pl.Tensor, s3: pl.Tensor, route: pl.Tensor, hidden: pl.Tensor,
    count: pl.Scalar[pl.INT32], width: pl.constexpr, intermediate: pl.constexpr,
    n_tile: pl.constexpr, k_tile: pl.constexpr, limit: pl.constexpr,
    weighted: pl.constexpr, hint: pl.constexpr, output_major: pl.constexpr, prefill: pl.constexpr,
    large_row_tile: pl.constexpr, large_n_tile: pl.constexpr,
):
    """Select a row tile for the number of tokens assigned to this expert."""
    if prefill:
        if count > PREFILL_EXPERT_LARGE_M_TILE:
            expert_swiglu_tiled(
                x, factor, w1, s1, w3, s3, route, hidden, count, width, intermediate,
                large_n_tile, k_tile, limit, weighted, hint, output_major, large_row_tile, 2,
            )
        elif count >= 17:
            expert_swiglu_tiled(
                x, factor, w1, s1, w3, s3, route, hidden, count, width, intermediate,
                128, k_tile, limit, weighted, hint, output_major, 32, 2,
            )
        else:
            expert_swiglu_tiled(
                x, factor, w1, s1, w3, s3, route, hidden, count, width, intermediate,
                n_tile, k_tile, limit, weighted, hint, output_major, 16, 2,
            )
    elif count >= 32:
        expert_swiglu_tiled(
            x, factor, w1, s1, w3, s3, route, hidden, count, width, intermediate,
            128, k_tile, limit, weighted, hint, output_major, 32, 1,
        )
    else:
        expert_swiglu_tiled(
            x, factor, w1, s1, w3, s3, route, hidden, count, width, intermediate,
            n_tile, k_tile, limit, weighted, hint, output_major, 16, 1,
        )
    return hidden


@pl.jit.inline(auto_scope=False)
def expert_down(
    hidden: pl.Tensor,
    weight: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.Tensor,
    count: pl.Scalar[pl.INT32],
    width: pl.constexpr,
    output_width: pl.constexpr,
    hint: pl.constexpr,
    output_major: pl.constexpr,
    prefill: pl.constexpr,
    large_row_tile: pl.constexpr,
):
    """Quantize the expert activation and project it back to the model width."""
    payload, scale = quantize_projection_activation(hidden, count, width, True, prefill)
    if prefill:
        if count > PREFILL_EXPERT_LARGE_M_TILE:
            prefill_gemm_int8_tiled(
                payload, scale, weight, weight_scale, output, count, width, output_width,
                hint, 128, 256, output_major, large_row_tile,
            )
        elif count > 16:
            prefill_gemm_int8_tiled(
                payload, scale, weight, weight_scale, output, count, width, output_width,
                hint, PREFILL_EXPERT_DOWN_N_TILE, 256, output_major, 32,
            )
        else:
            prefill_gemm_int8_tiled(
                payload, scale, weight, weight_scale, output, count, width, output_width,
                hint, PREFILL_EXPERT_DOWN_N_TILE, 256, output_major, 16,
            )
    elif width == 2304 and output_width == 5120 and count == 1:
        prefill_gemm_int8_tiled(
            payload, scale, weight, weight_scale, output, count, width, output_width,
            hint, 256, 256, output_major, 16,
        )
    else:
        gemm_int8(payload, scale, weight, weight_scale, output, count, width, output_width, hint, 128, 256, output_major)
    return output


@pl.jit.inline(auto_scope=False)
def gemm_dynamic_int8(
    source: pl.Tensor,
    weight: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.constexpr,
    output_width: pl.constexpr,
    name_hint: pl.constexpr,
    output_major: pl.constexpr,
    prefill: pl.constexpr,
):
    """Quantize BF16 token rows and project an INT8 weight matrix."""
    payload, scale = quantize_projection_activation(source, num_tokens, width, True, prefill)
    return gemm_int8(payload, scale, weight, weight_scale, output, num_tokens, width, output_width, name_hint, 128, 256, output_major)


def golden_activation_int8(source):
    """Round the exact BF16 token/maximum ratio to even, independent of reciprocal approximations."""
    value = source.double()
    maximum = value.abs().amax(-1, keepdim=True).clamp_min(float(torch.tensor(127e-30)))
    return torch.round(value * 127.0 / maximum).clamp(-127, 127).to(torch.int8)


def golden_dynamic_int8(source, weight, scale):
    """Evaluate symmetric per-token INT8 quantization and per-channel dequantization."""
    value = source.float()
    activation_scale = (value.abs().amax(-1, keepdim=True) * (1.0 / 127.0)).clamp_min(1e-30)
    quantized = golden_activation_int8(source).long()
    product = quantized @ weight.long()
    return (product.float() * activation_scale * scale.float().reshape(1, -1)).bfloat16()


def make_gemm_bf16_to_bf16(*, columns_first, k_subtile=0):
    @pl.jit.inline(auto_scope=False)
    def gemm_bf16_to_bf16(
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
        source_col: pl.Scalar[pl.INT32],
        output_col: pl.Scalar[pl.INT32],
        ordered: pl.constexpr,
        row_tile: pl.constexpr,
        stages: pl.constexpr,
    ):
        """Project BF16 ND buffers with explicit FP32 accumulation and column offsets."""
        col_blocks = output_width // n_tile
        row_blocks = (num_tokens + row_tile - 1) // row_tile
        if columns_first:
            blocks = row_blocks * col_blocks
        else:
            blocks = (num_tokens + row_tile - 1) // row_tile * col_blocks
        task_workers = pl.max(1, pl.min(pl.min(blocks, workers), pl.system.available_cluster_count()))
        entry_deps = pl.array.create(1, pl.TASK_ID)
        entry_deps[0] = ready
        with pl.spmd(task_workers, name_hint=name_hint, deps=[entry_deps[0]]) as projection_tid:
            worker = pl.tile.get_block_idx()
            for block in pl.range(worker, blocks, task_workers):
                if columns_first:
                    row = block % row_blocks * row_tile
                    col = block // row_blocks * n_tile
                else:
                    row = block // col_blocks * row_tile
                    col = block % col_blocks * n_tile
                rows = pl.max(0, pl.min(row_tile, num_tokens - row))
                if ordered:
                    ordered_acc = pl.tile.full([row_tile, n_tile], dtype=pl.FP32, value=0.0)
                    for k in pl.range(0, width, k_tile):
                        ordered_a = pl.load(source, [row, source_col + k], [row_tile, k_tile], valid_shape=[rows, k_tile])
                        ordered_a = pl.set_validshape(ordered_a, row_tile, k_tile)
                        ordered_w = pl.load(weight, [k, col], [k_tile, n_tile])
                        ordered_acc = bf16_ordered_accumulate(ordered_acc, ordered_a, ordered_w, row_tile, n_tile, k_tile)
                    rounded = pl.cast(ordered_acc, pl.BF16, mode="rint")
                    ordered_value = pl.set_validshape(rounded, rows, n_tile)
                    output = pl.store(ordered_value, [row, output_col + col], output)
                else:
                    cube_seed = pl.tile.create([row_tile, n_tile], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc, compact=True)
                    cube_acc = pl.set_validshape(cube_seed, rows, n_tile)
                    for k in pl.pipeline(0, width, k_tile, stage=stages):
                        if k_subtile > 0:
                            a_panel = pl.load(
                                source, [row, source_col + k], [row_tile, k_tile],
                                valid_shape=[rows, k_tile], target_memory=pl.MemorySpace.Mat,
                            )
                            w_panel = pl.load(weight, [k, col], [k_tile, n_tile], target_memory=pl.MemorySpace.Mat)
                            for inner_k in pl.pipeline(0, k_tile, k_subtile, stage=2):
                                a_fragment = pl.tile.slice(a_panel, [row_tile, k_subtile], [0, inner_k])
                                w_fragment = pl.tile.slice(w_panel, [k_subtile, n_tile], [inner_k, 0])
                                cube_acc = pl.matmul_acc(cube_acc, a_fragment, w_fragment, init_cond=(k + inner_k == 0))
                        else:
                            cube_a = pl.load(source, [row, source_col + k], [row_tile, k_tile], valid_shape=[rows, k_tile])
                            cube_w = pl.load(weight, [k, col], [k_tile, n_tile], target_memory=pl.MemorySpace.Mat)
                            cube_acc = pl.matmul_acc(cube_acc, cube_a, cube_w, init_cond=(k == 0))
                    cube_value = pl.set_validshape(cube_acc, rows, n_tile)
                    output = pl.store(cube_value, [row, output_col + col], output)
        return output, projection_tid

    return gemm_bf16_to_bf16


gemm_bf16_to_bf16 = make_gemm_bf16_to_bf16(columns_first=False)
prefill_gemm_bf16_to_bf16 = make_gemm_bf16_to_bf16(columns_first=False, k_subtile=PREFILL_BF16_K_SUBTILE)
prefill_gemm_bf16_to_bf16_columns_first = make_gemm_bf16_to_bf16(
    columns_first=True, k_subtile=PREFILL_BF16_K_SUBTILE,
)


def make_projection_validation(width):
    """Specialize projection validation to a checkpoint's complete reduction dimension."""
    @pl.jit
    def validate_projection(
        source: pl.Tensor[[32, width], pl.BF16],
        weight: pl.Tensor[[width, 128], pl.INT8],
        scale: pl.Tensor[[1, 128], pl.FP32],
        output: pl.Out[pl.Tensor[[32, 128], pl.BF16]],
        quantized: pl.Out[pl.Tensor[[32, width], pl.INT8]],
        activation_scale: pl.Out[pl.Tensor[[32, 1], pl.FP32]],
    ):
        payload, factor = quantize_activation(source, 32, width)
        validate_tiles = 2
        validate_workers = pl.max(1, pl.min(validate_tiles, pl.system.available_aiv_count()))
        for validate_worker in pl.spmd(validate_workers, name_hint="validate_quantization"):
            for block in pl.range(validate_worker, validate_tiles, validate_workers):
                row = block * 16
                for col in pl.range(0, width, 256):
                    value = pl.load(payload, [row, col], [16, 256])
                    pl.store(value, [row, col], quantized)
                pl.store(pl.load(factor, [row, 0], [16, 1]), [row, 0], activation_scale)
        gemm_int8(payload, factor, weight, scale, output, 32, width, 128, "validate_int8", 64, 256, False)
        return output, quantized, activation_scale
    return validate_projection


def golden_projection(tensors):
    """Validate activation codes, FP32 factors and independently accumulated INT64 products."""
    value = tensors["source"].float()
    factor = (value.abs().amax(-1, keepdim=True) * (1.0 / 127.0)).clamp_min(1e-30)
    tensors["activation_scale"].copy_(factor)
    tensors["quantized"].copy_(golden_activation_int8(tensors["source"]))
    tensors["output"].copy_(golden_dynamic_int8(tensors["source"], tensors["weight"], tensors["scale"]))


@pl.jit
def validate_bf16_dot(
    query: pl.Tensor[[16, 512], pl.BF16],
    keys: pl.Tensor[[128, 512], pl.BF16],
    output: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
):
    """Validate ordered BF16 score accumulation on one attention tile."""
    for block in pl.spmd(1, name_hint="bf16_dot"):
        row = block * 16
        product = bf16_dot_sequential(query, keys, row, 0, 16, 128, 512)
        pl.store(product, [row, 0], output)
    return output


def golden_bf16_dot(tensors):
    """Evaluate the independent CPU FP32 attention score matrix."""
    tensors["output"].copy_(tensors["query"].float() @ tensors["keys"].float().t())


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    from golden import TensorSpec, ratio_allclose, run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3"], default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--dot-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(41)
    stems = ["synthetic"]
    weights = None
    if args.checkpoint is not None:
        from .weights import Checkpoint, ResidentWeights

        weights = ResidentWeights(Checkpoint(args.checkpoint))
        stems = ["layers.0.attn.wq_a", "layers.0.attn.wq_b", "layers.0.attn.wkv",
                 "layers.2.attn.indexer.wq_b", "layers.0.ffn.shared_experts.w1",
                 "layers.0.ffn.shared_experts.w3", "layers.0.ffn.shared_experts.w2",
                 "layers.0.ffn.experts.0.w1", "layers.0.ffn.experts.383.w2"]
    report = {"platform": args.platform, "device": args.device, "compile_only": args.compile_only, "cases": []}
    if weights is not None:
        report["checkpoint_identity"] = weights.checkpoint.identity
    for stem in ([] if args.dot_only else stems):
        if weights is None:
            weight = torch.randint(-127, 128, (256, 128), generator=generator, dtype=torch.int8)
            scales = torch.rand(1, 128, generator=generator) * 0.001
        else:
            weight, scales = weights.projection(stem)
            columns = torch.cat((torch.arange(64), torch.arange(weight.shape[1] - 64, weight.shape[1])))
            weight, scales = weight[:, columns].contiguous(), scales[:, columns].contiguous()
        width = weight.shape[0]
        source = torch.randn(32, width, generator=generator).bfloat16()
        source[0].zero_()
        source[1].zero_()
        source[1, :6] = torch.tensor([254., 127., 125., -127., -125., -254.]).bfloat16()
        print(f"A3_PROJECTION {stem} K={width}", flush=True)
        specs = [
            TensorSpec("source", [32, width], torch.bfloat16, init_value=source),
            TensorSpec("weight", [width, 128], torch.int8, init_value=weight),
            TensorSpec("scale", [1, 128], torch.float32, init_value=scales),
            TensorSpec("output", [32, 128], torch.bfloat16),
            TensorSpec("quantized", [32, width], torch.int8),
            TensorSpec("activation_scale", [32, 1], torch.float32),
        ]
        result = run(
            fn=make_projection_validation(width),
            specs=specs,
            golden_fn=golden_projection,
            compare_fn={name: ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
                        for name in ("output", "quantized", "activation_scale")},
            config={"backend_type": args.platform, "execution_mode": args.sim, "device_id": args.device},
            compile_only=args.compile_only,
        )
        report["cases"].append({"projection": stem, "passed": result.passed, "build": str(result.work_dir)})
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        if not result.passed:
            raise SystemExit(result.error or 1)
    query = torch.randn(16, 512, generator=generator).bfloat16()
    keys = torch.randn(128, 512, generator=generator).bfloat16()
    query[0].zero_()
    keys[0].zero_()
    result = run(
        fn=validate_bf16_dot,
        specs=[TensorSpec("query", [16, 512], torch.bfloat16, init_value=query),
               TensorSpec("keys", [128, 512], torch.bfloat16, init_value=keys),
               TensorSpec("output", [16, 128], torch.float32)],
        golden_fn=golden_bf16_dot,
        compare_fn={"output": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)},
        config={"backend_type": args.platform, "execution_mode": args.sim, "device_id": args.device},
        compile_only=args.compile_only,
    )
    report["cases"].append({"projection": "bf16_attention_dot", "passed": result.passed, "build": str(result.work_dir)})
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    if not result.passed:
        raise SystemExit(result.error or 1)
