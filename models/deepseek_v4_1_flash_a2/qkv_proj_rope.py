# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2 single-layer dependencies extracted from the shared model operators."""

import pypto.language as pl
from .config import FLASH as M
from .operator_config import T_DYN


HEAD_DIM = M.head_dim


ROPE_DIM = M.qk_rope_head_dim


M_TILE = 16


N_TILE = 128


K_TILE = 256


def _make_rope_block(heads, inverse, head_dim, rope_dim):
    sign = -1.0 if inverse else 1.0
    nope_dim = head_dim - rope_dim

    @pl.jit.inline
    def rotate_block(
        x: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        cos: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        output: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        block: pl.Scalar[pl.INT32],
    ):
        t = block // heads
        h = block % heads
        base = h * head_dim
        output[t : t + 1, base : base + nope_dim] = x[t : t + 1, base : base + nope_dim]
        tail = pl.cast(x[t : t + 1, base + nope_dim : base + head_dim], pl.FP32)
        even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
        odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
        c = cos[t : t + 1, :]
        s = pl.mul(sin[t : t + 1, :], sign)
        re = pl.sub(pl.mul(even, c), pl.mul(odd, s))
        im = pl.add(pl.mul(even, s), pl.mul(odd, c))
        rotated = pl.full([1, rope_dim], dtype=pl.FP32, value=0.0)
        rotated = pl.tensor.scatter(re, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
        rotated = pl.tensor.scatter(im, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
        output[t : t + 1, base + nope_dim : base + head_dim] = pl.cast(
            rotated, pl.BF16, mode="rint"
        )
        return output

    return rotate_block


def make_rope(
    heads,
    inverse=False,
    *,
    head_dim=HEAD_DIM,
    rope_dim=ROPE_DIM,
    max_workers=None,
    name_hint="attention_rope",
):
    """Specialize adjacent-pair RoPE with direct or capped strided scheduling."""
    rotate_block = _make_rope_block(heads, inverse, head_dim, rope_dim)
    worker_limit = 2**31 - 1 if max_workers is None else max_workers

    @pl.jit.inline
    def rotate(
        x: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        cos: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        output: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        blocks = num_tokens * heads
        workers = pl.min(blocks, worker_limit)
        for worker in pl.spmd(workers, name_hint=name_hint):
            for block in pl.range(worker, blocks, workers):
                rotate_block(x, cos, sin, output, block)
        return output

    return rotate


def make_bf16_projection_staged(width, output_width):
    """Stage FP32 projection values before a separate BF16 narrowing task.

    The BF16 operand passes through Vector before Cube starts its result transfer
    (hw-native-sys/pypto#2829). Store FP32 values through Vector as well: direct
    Acc-to-GM stores were unstable in the concurrent A5 prefill workload.
    Accumulation and RNE rounding are unchanged.
    """
    n_tile = min(N_TILE, output_width)
    k_tile = min(K_TILE, width)

    @pl.jit.inline
    def project(
        source: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.BF16],
        output: pl.Tensor[[T_DYN, output_width], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        tokens = pl.tensor.dim(source, 0)
        accumulated = pl.create_tensor([tokens, output_width], dtype=pl.FP32)
        blocks = (num_tokens + M_TILE - 1) // M_TILE * (output_width // n_tile)
        with pl.spmd(blocks, name_hint="prefill_bf16_projection_cube") as cube_tid:
            block = pl.tile.get_block_idx()
            row = block // (output_width // n_tile) * M_TILE
            col = block % (output_width // n_tile) * n_tile
            rows = pl.min(M_TILE, num_tokens - row)
            acc = pl.create_tensor([M_TILE, n_tile], dtype=pl.FP32)
            for k in pl.range(0, width, k_tile):
                a = pl.slice(source, [M_TILE, k_tile], [row, k], valid_shape=[rows, k_tile])
                # Preserve BF16 values while establishing Vector-to-Cube startup.
                a = pl.cast(pl.cast(a, pl.FP32), pl.BF16, mode="rint")
                b = pl.slice(weight, [k_tile, n_tile], [k, col])
                acc = pl.matmul_acc(acc, a, b, init_cond=(k == 0))
            accumulated[row : row + M_TILE, col : col + n_tile] = pl.set_validshape(pl.mul(acc, 1.0), rows, n_tile)
        with pl.spmd(blocks, name_hint="prefill_bf16_projection_cast", deps=[cube_tid]):
            block = pl.tile.get_block_idx()
            row = block // (output_width // n_tile) * M_TILE
            col = block % (output_width // n_tile) * n_tile
            rows = pl.min(M_TILE, num_tokens - row)
            value = pl.slice(accumulated, [M_TILE, n_tile], [row, col], valid_shape=[rows, n_tile])
            output[row : row + M_TILE, col : col + n_tile] = pl.cast(value, pl.BF16, mode="rint")
        return output

    return project
