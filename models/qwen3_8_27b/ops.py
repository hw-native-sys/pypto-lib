# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shape-specialized PyPTO primitives for the fixed Qwen3.8 layer contract."""

import pypto.language as pl

ROWS_DYN = pl.dynamic("ROWS_DYN")
K_DYN = pl.dynamic("K_DYN")
N_DYN = pl.dynamic("N_DYN")
ROW_TILE = 16
K_TILE = 256
N_TILE = 128
EPS = 1e-6
HIDDEN = 5120
MLP_WIDTH = 17408
PADDED_TOKENS = 16


@pl.jit.inline
def linear_into(
    x: pl.Tensor[[PADDED_TOKENS, K_DYN], pl.BF16],
    weight: pl.Tensor[[N_DYN, K_DYN], pl.BF16],
    out: pl.Tensor[[PADDED_TOKENS, N_DYN], pl.BF16],
):
    rows = pl.tensor.dim(x, 0)
    inner = pl.tensor.dim(x, 1)
    columns = pl.tensor.dim(weight, 0)
    for block in pl.spmd((rows // ROW_TILE) * (columns // N_TILE), name_hint="linear_bf16"):
        row = (block // (columns // N_TILE)) * ROW_TILE
        column = (block % (columns // N_TILE)) * N_TILE
        acc = pl.create_tensor([ROW_TILE, N_TILE], dtype=pl.FP32)
        for kb in pl.range(inner // K_TILE):
            k0 = kb * K_TILE
            lhs = x[row : row + ROW_TILE, k0 : k0 + K_TILE]
            rhs = weight[column : column + N_TILE, k0 : k0 + K_TILE]
            acc = pl.matmul_acc(acc, lhs, rhs, b_trans=True, init_cond=(kb == 0))
        out[row : row + ROW_TILE, column : column + N_TILE] = pl.cast(acc, target_type=pl.BF16, mode="rint")
    return out


@pl.jit
def linear_bf16(
    x: pl.Tensor[[PADDED_TOKENS, K_DYN], pl.BF16],
    weight: pl.Tensor[[N_DYN, K_DYN], pl.BF16],
    out: pl.Out[pl.Tensor[[PADDED_TOKENS, N_DYN], pl.BF16]],
):
    return linear_into(x, weight, out)


@pl.jit.inline
def rmsnorm_into(
    x: pl.Tensor[[PADDED_TOKENS, K_DYN], pl.BF16],
    weight: pl.Tensor[[1, K_DYN], pl.BF16],
    inv_width: pl.Scalar[pl.FP32],
    out: pl.Tensor[[PADDED_TOKENS, K_DYN], pl.BF16],
):
    rows = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(x, 1)
    for block in pl.spmd(rows // ROW_TILE, name_hint="rmsnorm_bf16"):
        row = block * ROW_TILE
        square_sum = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.range(width // K_TILE):
            k0 = kb * K_TILE
            x_float = pl.cast(x[row : row + ROW_TILE, k0 : k0 + K_TILE], target_type=pl.FP32)
            partial = pl.reshape(pl.row_sum(pl.mul(x_float, x_float)), [1, ROW_TILE])
            square_sum = pl.add(square_sum, partial)
        inv_rms = pl.reshape(
            pl.rsqrt(pl.add(pl.mul(square_sum, inv_width), EPS)),
            [ROW_TILE, 1],
        )
        for kb in pl.range(width // K_TILE):
            k0 = kb * K_TILE
            x_float = pl.cast(x[row : row + ROW_TILE, k0 : k0 + K_TILE], target_type=pl.FP32)
            norm_chunk = pl.cast(weight[:, k0 : k0 + K_TILE], target_type=pl.FP32)
            gamma = pl.add(norm_chunk, 1.0)
            scaled = pl.row_expand_mul(x_float, inv_rms)
            out[row : row + ROW_TILE, k0 : k0 + K_TILE] = pl.cast(
                pl.col_expand_mul(scaled, gamma), target_type=pl.BF16, mode="rint"
            )
    return out


@pl.jit
def rmsnorm_bf16(
    x: pl.Tensor[[ROWS_DYN, K_DYN], pl.BF16],
    weight: pl.Tensor[[1, K_DYN], pl.BF16],
    inv_width: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS_DYN, K_DYN], pl.BF16]],
):
    rows = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(x, 1)
    for block in pl.spmd(rows // ROW_TILE, name_hint="rmsnorm_bf16"):
        row = block * ROW_TILE
        square_sum = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.range(width // K_TILE):
            k0 = kb * K_TILE
            x_float = pl.cast(x[row : row + ROW_TILE, k0 : k0 + K_TILE], target_type=pl.FP32)
            partial = pl.reshape(pl.row_sum(pl.mul(x_float, x_float)), [1, ROW_TILE])
            square_sum = pl.add(square_sum, partial)
        inv_rms = pl.reshape(
            pl.rsqrt(pl.add(pl.mul(square_sum, inv_width), EPS)),
            [ROW_TILE, 1],
        )
        for kb in pl.range(width // K_TILE):
            k0 = kb * K_TILE
            x_float = pl.cast(x[row : row + ROW_TILE, k0 : k0 + K_TILE], target_type=pl.FP32)
            norm_chunk = pl.cast(weight[:, k0 : k0 + K_TILE], target_type=pl.FP32)
            gamma = pl.add(norm_chunk, 1.0)
            scaled = pl.row_expand_mul(x_float, inv_rms)
            out[row : row + ROW_TILE, k0 : k0 + K_TILE] = pl.cast(
                pl.col_expand_mul(scaled, gamma), target_type=pl.BF16, mode="rint"
            )
    return out


@pl.jit.inline
def residual_into(
    x: pl.Tensor[[PADDED_TOKENS, HIDDEN], pl.BF16],
    branch: pl.Tensor[[PADDED_TOKENS, HIDDEN], pl.BF16],
    out: pl.Tensor[[PADDED_TOKENS, HIDDEN], pl.BF16],
):
    for block in pl.spmd(HIDDEN // N_TILE, name_hint="qwen_residual"):
        col = block * N_TILE
        x_tile = pl.cast(x[:, col : col + N_TILE], target_type=pl.FP32)
        branch_tile = pl.cast(branch[:, col : col + N_TILE], target_type=pl.FP32)
        out[:, col : col + N_TILE] = pl.cast(pl.add(x_tile, branch_tile), target_type=pl.BF16, mode="rint")
    return out


@pl.jit.inline
def swiglu_into(
    gate: pl.Tensor[[PADDED_TOKENS, MLP_WIDTH], pl.BF16],
    up: pl.Tensor[[PADDED_TOKENS, MLP_WIDTH], pl.BF16],
    out: pl.Tensor[[PADDED_TOKENS, MLP_WIDTH], pl.BF16],
):
    for block in pl.spmd(MLP_WIDTH // N_TILE, name_hint="qwen_swiglu"):
        col = block * N_TILE
        gate_f32 = pl.cast(gate[:, col : col + N_TILE], target_type=pl.FP32)
        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_f32)), 1.0))
        activated = pl.cast(pl.mul(gate_f32, sigmoid), target_type=pl.BF16, mode="rint")
        up_f32 = pl.cast(up[:, col : col + N_TILE], target_type=pl.FP32)
        out[:, col : col + N_TILE] = pl.cast(
            pl.mul(pl.cast(activated, target_type=pl.FP32), up_f32),
            target_type=pl.BF16,
            mode="rint",
        )
    return out


@pl.jit.inline
def mlp_into(
    x: pl.Tensor[[PADDED_TOKENS, HIDDEN], pl.BF16],
    norm_weight: pl.Tensor[[1, HIDDEN], pl.BF16],
    gate_weight: pl.Tensor[[MLP_WIDTH, HIDDEN], pl.BF16],
    up_weight: pl.Tensor[[MLP_WIDTH, HIDDEN], pl.BF16],
    down_weight: pl.Tensor[[HIDDEN, MLP_WIDTH], pl.BF16],
    out: pl.Tensor[[PADDED_TOKENS, HIDDEN], pl.BF16],
):
    normalized = pl.create_tensor([PADDED_TOKENS, HIDDEN], dtype=pl.BF16)
    gate = pl.create_tensor([PADDED_TOKENS, MLP_WIDTH], dtype=pl.BF16)
    up = pl.create_tensor([PADDED_TOKENS, MLP_WIDTH], dtype=pl.BF16)
    product = pl.create_tensor([PADDED_TOKENS, MLP_WIDTH], dtype=pl.BF16)
    projected = pl.create_tensor([PADDED_TOKENS, HIDDEN], dtype=pl.BF16)
    rmsnorm_into(x, norm_weight, 1.0 / HIDDEN, normalized)
    linear_into(normalized, gate_weight, gate)
    linear_into(normalized, up_weight, up)
    swiglu_into(gate, up, product)
    linear_into(product, down_weight, projected)
    residual_into(x, projected, out)
    return out
