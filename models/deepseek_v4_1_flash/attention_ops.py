# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared V4.1 Attention projection, RMSNorm, and RoPE kernel factories."""

import pypto.language as pl

from models.deepseek_v4_1_flash.config import D, FLASH, HEAD_DIM, ROPE_DIM, T_DYN


EPS = FLASH.rms_norm_eps
M_TILE = 16
MX_M_TILE = 32
N_TILE = 128
K_TILE = 256


def make_mx_projection(width, output_width, output_dtype=pl.BF16, *, name_hint="attention_mx_projection"):
    """Specialize an MXFP8 projection without expanding weights in HBM."""
    fp32_output = output_dtype == pl.FP32

    @pl.jit.inline
    def project(
        x: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.FP8E4M3FN],
        scale: pl.Tensor[[width // 32, output_width], pl.FP8E8M0, pl.MX_B_NN],
        output: pl.Tensor[[T_DYN, output_width], output_dtype],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        for mt in pl.parallel((num_tokens + MX_M_TILE - 1) // MX_M_TILE):
            t0 = mt * MX_M_TILE
            for block in pl.spmd(output_width // N_TILE, name_hint=name_hint):
                n0 = block * N_TILE
                rows = pl.min(MX_M_TILE, num_tokens - t0)
                first = pl.load(x, [t0, 0], [MX_M_TILE, K_TILE], valid_shape=[rows, K_TILE])
                first = pl.set_validshape(pl.fillpad(first, pad_value=pl.PadValue.zero), MX_M_TILE, K_TILE)
                firstq_values = pl.reshape(pl.cast(first, pl.FP32), [MX_M_TILE * (K_TILE // 32), 32])
                firstq_reduce_tmp = pl.create_tile([MX_M_TILE * (K_TILE // 32), 32], dtype=pl.FP32)
                firstq_maximum = pl.maximum(pl.row_max(pl.abs(firstq_values), tmp_tile=firstq_reduce_tmp), 1e-4)
                firstq_bits = pl.reinterpret_view(pl.mul(firstq_maximum, 1.0 / 448.0), pl.INT32)
                firstq_exponent = pl.shrs(pl.add(firstq_bits, 8388607), 23)
                firstq_scale = pl.reinterpret_view(pl.shls(firstq_exponent, 23), pl.FP32)
                firstq_quantized = pl.cast(
                    pl.row_expand_div(firstq_values, firstq_scale), pl.FP8E4M3FN, mode="rint"
                )
                firstq_payload = pl.reshape(firstq_quantized, [MX_M_TILE, K_TILE])
                firstq_signed_exponent = pl.sub(firstq_exponent, pl.mul(pl.shrs(firstq_exponent, 7), 256))
                firstq_codes = pl.reinterpret_view(pl.cast(firstq_signed_exponent, pl.INT8), pl.UINT8)
                firstq_flat = pl.reshape(firstq_codes, [1, MX_M_TILE * (K_TILE // 32)])
                firstq_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                firstq_packed = pl.tmov_x2zz(
                    firstq_flat, firstq_tmp, group_axis=1, dst_rows=MX_M_TILE, dst_cols=8
                )
                a0 = firstq_payload
                sa0 = pl.reinterpret_view(firstq_packed, pl.FP8E8M0)
                b0 = pl.load(weight, [0, n0], [K_TILE, N_TILE])
                sb0 = pl.load(scale, [0, n0], [K_TILE // 32, N_TILE])
                acc = pl.matmul_mx(a0, sa0, b0, sb0)
                for kb in pl.range(1, width // K_TILE):
                    k0 = kb * K_TILE
                    values = pl.load(x, [t0, k0], [MX_M_TILE, K_TILE], valid_shape=[rows, K_TILE])
                    values = pl.set_validshape(pl.fillpad(values, pad_value=pl.PadValue.zero), MX_M_TILE, K_TILE)
                    nextq_values = pl.reshape(pl.cast(values, pl.FP32), [MX_M_TILE * (K_TILE // 32), 32])
                    nextq_reduce_tmp = pl.create_tile([MX_M_TILE * (K_TILE // 32), 32], dtype=pl.FP32)
                    nextq_maximum = pl.maximum(
                        pl.row_max(pl.abs(nextq_values), tmp_tile=nextq_reduce_tmp), 1e-4
                    )
                    nextq_bits = pl.reinterpret_view(pl.mul(nextq_maximum, 1.0 / 448.0), pl.INT32)
                    nextq_exponent = pl.shrs(pl.add(nextq_bits, 8388607), 23)
                    nextq_scale = pl.reinterpret_view(pl.shls(nextq_exponent, 23), pl.FP32)
                    nextq_quantized = pl.cast(
                        pl.row_expand_div(nextq_values, nextq_scale), pl.FP8E4M3FN, mode="rint"
                    )
                    nextq_payload = pl.reshape(nextq_quantized, [MX_M_TILE, K_TILE])
                    nextq_signed_exponent = pl.sub(
                        nextq_exponent, pl.mul(pl.shrs(nextq_exponent, 7), 256)
                    )
                    nextq_codes = pl.reinterpret_view(pl.cast(nextq_signed_exponent, pl.INT8), pl.UINT8)
                    nextq_flat = pl.reshape(nextq_codes, [1, MX_M_TILE * (K_TILE // 32)])
                    nextq_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                    nextq_packed = pl.tmov_x2zz(
                        nextq_flat, nextq_tmp, group_axis=1, dst_rows=MX_M_TILE, dst_cols=8
                    )
                    a = nextq_payload
                    sa = pl.reinterpret_view(nextq_packed, pl.FP8E8M0)
                    b = pl.load(weight, [k0, n0], [K_TILE, N_TILE])
                    sb = pl.load(scale, [k0 // 32, n0], [K_TILE // 32, N_TILE])
                    acc = pl.matmul_mx_acc(acc, a, sa, b, sb)
                if fp32_output:
                    output = pl.store(pl.set_validshape(pl.mul(acc, 1.0), rows, N_TILE), [t0, n0], output)
                else:
                    value = pl.cast(acc, target_type=pl.BF16, mode="rint")
                    output = pl.store(pl.set_validshape(value, rows, N_TILE), [t0, n0], output)
        return output

    return project


def _make_norm_block(width):
    @pl.jit.inline
    def normalize_block(
        x: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width], pl.BF16],
        output: pl.Tensor[[T_DYN, width], pl.BF16],
        block: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        t = block * 8
        rows = pl.min(8, num_tokens - t)
        if width == D:
            square_sum = pl.full([1, 8], dtype=pl.FP32, value=0.0)
            for chunk in pl.pipeline(width // 128, stage=2):
                d0 = chunk * 128
                source_chunk = pl.slice(x, [8, 128], [t, d0], valid_shape=[rows, 128])
                source_chunk = pl.set_validshape(
                    pl.fillpad(source_chunk, pad_value=pl.PadValue.zero), 8, 128
                )
                value_chunk = pl.cast(source_chunk, pl.FP32)
                chunk_sum = pl.reshape(pl.row_sum(pl.mul(value_chunk, value_chunk)), [1, 8])
                square_sum = pl.add(square_sum, chunk_sum)
            inverse = pl.rsqrt(pl.add(pl.mul(square_sum, 1.0 / width), EPS), high_precision=True)
            inverse_col = pl.reshape(inverse, [8, 1])
            for chunk in pl.pipeline(width // 128, stage=2):
                d0 = chunk * 128
                source_chunk = pl.slice(x, [8, 128], [t, d0], valid_shape=[rows, 128])
                source_chunk = pl.set_validshape(
                    pl.fillpad(source_chunk, pad_value=pl.PadValue.zero), 8, 128
                )
                value_chunk = pl.cast(source_chunk, pl.FP32)
                gamma_chunk = pl.reshape(pl.cast(weight[d0 : d0 + 128], pl.FP32), [1, 128])
                result_chunk = pl.col_expand_mul(
                    pl.row_expand_mul(value_chunk, inverse_col), gamma_chunk
                )
                output[t : t + 8, d0 : d0 + 128] = pl.set_validshape(
                    pl.cast(result_chunk, pl.BF16, mode="rint"), rows, 128
                )
        else:
            source = pl.slice(x, [8, width], [t, 0], valid_shape=[rows, width])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 8, width)
            value = pl.cast(source, pl.FP32)
            row_inverse = pl.rsqrt(
                pl.add(pl.mul(pl.row_sum(pl.mul(value, value)), 1.0 / width), EPS),
                high_precision=True,
            )
            gamma = pl.reshape(pl.cast(weight[:], pl.FP32), [1, width])
            normalized = pl.col_expand_mul(pl.row_expand_mul(value, row_inverse), gamma)
            output[t : t + 8, :] = pl.set_validshape(
                pl.cast(normalized, pl.BF16, mode="rint"), rows, width
            )
        return output

    return normalize_block


def make_norm(width, *, max_workers=None, name_hint="attention_rmsnorm"):
    """Specialize RMSNorm with direct or capped strided SPMD scheduling."""
    normalize_block = _make_norm_block(width)
    worker_limit = 2**31 - 1 if max_workers is None else max_workers

    @pl.jit.inline
    def normalize(
        x: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width], pl.BF16],
        output: pl.Tensor[[T_DYN, width], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        blocks = (num_tokens + 7) // 8
        workers = pl.min(blocks, worker_limit)
        for worker in pl.spmd(workers, name_hint=name_hint):
            for block in pl.range(worker, blocks, workers):
                normalize_block(x, weight, output, block, num_tokens)
        return output

    return normalize


def make_norm_with_deps(width, *, max_workers=None, name_hint="attention_rmsnorm"):
    """Specialize RMSNorm with an explicit producer dependency."""
    normalize_block = _make_norm_block(width)
    worker_limit = 2**31 - 1 if max_workers is None else max_workers

    @pl.jit.inline
    def normalize(
        x: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width], pl.BF16],
        output: pl.Tensor[[T_DYN, width], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
        ready: pl.Scalar[pl.TASK_ID],
    ):
        blocks = (num_tokens + 7) // 8
        workers = pl.min(blocks, worker_limit)
        with pl.spmd(workers, name_hint=name_hint, deps=[ready]) as norm_tid:
            worker = pl.tile.get_block_idx()
            for block in pl.range(worker, blocks, workers):
                normalize_block(x, weight, output, block, num_tokens)
        return norm_tid

    return normalize


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


def make_rope_with_deps(
    heads,
    inverse=False,
    *,
    head_dim=HEAD_DIM,
    rope_dim=ROPE_DIM,
    max_workers=None,
    name_hint="attention_rope",
):
    """Specialize adjacent-pair RoPE with an explicit producer dependency."""
    rotate_block = _make_rope_block(heads, inverse, head_dim, rope_dim)
    worker_limit = 2**31 - 1 if max_workers is None else max_workers

    @pl.jit.inline
    def rotate(
        x: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        cos: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        output: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
        ready: pl.Scalar[pl.TASK_ID],
    ):
        blocks = num_tokens * heads
        workers = pl.min(blocks, worker_limit)
        with pl.spmd(workers, name_hint=name_hint, deps=[ready]) as rope_tid:
            worker = pl.tile.get_block_idx()
            for block in pl.range(worker, blocks, workers):
                rotate_block(x, cos, sin, output, block)
        return rope_tid

    return rotate


def _make_bf16_projection_block(width, output_width):
    n_tile = min(N_TILE, output_width)
    k_tile = min(K_TILE, width)

    @pl.jit.inline
    def project_block(
        source: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.BF16],
        output: pl.Tensor[[T_DYN, output_width], pl.BF16],
        task: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        output_tiles = output_width // n_tile
        row = task // output_tiles * M_TILE
        col = task % output_tiles * n_tile
        rows = pl.min(M_TILE, num_tokens - row)
        acc = pl.create_tensor([M_TILE, n_tile], dtype=pl.FP32)
        for k in pl.range(0, width, k_tile):
            a = pl.slice(source, [M_TILE, k_tile], [row, k], valid_shape=[rows, k_tile])
            b = pl.slice(weight, [k_tile, n_tile], [k, col])
            acc = pl.matmul_acc(acc, a, b, init_cond=(k == 0))
        output[row : row + M_TILE, col : col + n_tile] = pl.set_validshape(
            pl.cast(acc, pl.BF16, mode="rint"), rows, n_tile
        )
        return output

    return project_block


def make_bf16_projection(
    width,
    output_width,
    *,
    max_workers=None,
    name_hint="attention_bf16_projection",
):
    """Specialize a BF16 projection with an FP32 accumulator."""
    project_block = _make_bf16_projection_block(width, output_width)
    n_tile = min(N_TILE, output_width)
    worker_limit = 2**31 - 1 if max_workers is None else max_workers

    @pl.jit.inline
    def project(
        source: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.BF16],
        output: pl.Tensor[[T_DYN, output_width], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        blocks = (num_tokens + M_TILE - 1) // M_TILE * (output_width // n_tile)
        workers = pl.min(blocks, worker_limit)
        for worker in pl.spmd(workers, name_hint=name_hint):
            for task in pl.range(worker, blocks, workers):
                project_block(source, weight, output, task, num_tokens)
        return output

    return project


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


def make_bf16_projection_with_deps(
    width,
    output_width,
    *,
    workers=32,
    name_hint="attention_bf16_projection",
):
    """Specialize a BF16 projection with an explicit producer dependency."""
    project_block = _make_bf16_projection_block(width, output_width)
    n_tile = min(N_TILE, output_width)

    @pl.jit.inline
    def project(
        source: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.BF16],
        output: pl.Tensor[[T_DYN, output_width], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
        ready: pl.Scalar[pl.TASK_ID],
    ):
        blocks = (num_tokens + M_TILE - 1) // M_TILE * (output_width // n_tile)
        with pl.spmd(workers, name_hint=name_hint, deps=[ready]) as projection_tid:
            worker = pl.tile.get_block_idx()
            for task in pl.range(worker, blocks, workers):
                project_block(source, weight, output, task, num_tokens)
        return projection_tid

    return project


__all__ = [
    "EPS",
    "K_TILE",
    "M_TILE",
    "MX_M_TILE",
    "N_TILE",
    "make_bf16_projection",
    "make_bf16_projection_staged",
    "make_bf16_projection_with_deps",
    "make_norm",
    "make_norm_with_deps",
    "make_mx_projection",
    "make_rope",
    "make_rope_with_deps",
]
