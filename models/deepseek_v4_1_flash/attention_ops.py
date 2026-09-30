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

from models.deepseek_v4_1_flash.config import INDEX_TOPK
from models.deepseek_v4_1_flash.quantization import prefill_decode_fp8_inline
from models.deepseek_v4_1_flash.quantization import prefill_quantize_fp8_inline
from models.deepseek_v4_1_flash.quantization import prefill_round_bf16_inline
from models.deepseek_v4_1_flash.quantization import prefill_round_inline
from models.deepseek_v4_1_flash.rmsnorm import prefill_rmsnorm_inline


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


# Resident CED prefill precision variants.
_PREFILL_OPS_INPUT_WIDTH = pl.dynamic("V41_FP32_INPUT_WIDTH")


_PREFILL_OPS_OUTPUT_WIDTH = pl.dynamic("V41_FP32_OUTPUT_WIDTH")


_PREFILL_OPS_ROWS = pl.dynamic("V41_FP32_ROWS")


@pl.jit.inline(auto_scope=False)
def prefill_linear_inline(
    x: pl.Tensor[[_PREFILL_OPS_ROWS, _PREFILL_OPS_INPUT_WIDTH], pl.FP32],
    weight: pl.Tensor[[_PREFILL_OPS_INPUT_WIDTH, _PREFILL_OPS_OUTPUT_WIDTH], pl.FP32],
    output: pl.Tensor[[_PREFILL_OPS_ROWS, _PREFILL_OPS_OUTPUT_WIDTH], pl.FP32],
):
    """FP32 linear projection with input-major weights and compensated K sums."""
    rows = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(x, 1)
    columns = pl.tensor.dim(weight, 1)
    column_tiles = (columns + 63) // 64
    for mt in pl.parallel((rows + 15) // 16):
        r0 = mt * 16
        for block in pl.spmd(column_tiles, name_hint="prefill_linear"):
            c0 = block * 64
            active_rows = pl.min(16, rows - r0)
            active_columns = pl.min(64, columns - c0)
            acc = pl.tile.full([16, 64], dtype=pl.FP32, value=0.0)
            error = pl.tile.full([16, 64], dtype=pl.FP32, value=0.0)
            for k0 in pl.range(0, width, 64):
                active_k = pl.min(64, width - k0)
                a = pl.load(x, [r0, k0], [16, 64], valid_shape=[active_rows, active_k])
                a = pl.set_validshape(pl.fillpad(a, pad_value=pl.PadValue.zero), 16, 64)
                b = pl.load(weight, [k0, c0], [64, 64], valid_shape=[active_k, active_columns])
                b = pl.set_validshape(pl.fillpad(b, pad_value=pl.PadValue.zero), 64, 64)
                part = pl.matmul(a, b)
                corrected = pl.sub(part, error)
                updated = pl.add(acc, corrected)
                error = pl.sub(pl.sub(updated, acc), corrected)
                acc = updated
            output = pl.store(pl.set_validshape(acc, active_rows, active_columns), [r0, c0], output)
    return output


_PREFILL_OPS_HEADS = pl.dynamic("V41_FP32_HEADS")


_PREFILL_OPS_CHANNELS = pl.dynamic("V41_FP32_CHANNELS")


@pl.jit.inline(auto_scope=False)
def prefill_rope_values_inline(
    x: pl.Tensor[[_PREFILL_OPS_ROWS, _PREFILL_OPS_HEADS, _PREFILL_OPS_CHANNELS], pl.FP32],
    cos: pl.Tensor[[_PREFILL_OPS_ROWS, ROPE_DIM // 2], pl.FP32],
    sin: pl.Tensor[[_PREFILL_OPS_ROWS, ROPE_DIM // 2], pl.FP32],
    output: pl.Tensor[[_PREFILL_OPS_ROWS, _PREFILL_OPS_HEADS, _PREFILL_OPS_CHANNELS], pl.FP32],
    direction: pl.Scalar[pl.FP32],
):
    """Adjacent-pair RoPE; direction is +1 for Q/K and -1 for attention output."""
    rows = pl.tensor.dim(x, 0)
    heads = pl.tensor.dim(x, 1)
    channels = pl.tensor.dim(x, 2)
    for worker in pl.spmd(32, name_hint="prefill_rope"):
        for flat_row in pl.range(worker, rows * heads, 32):
            row = flat_row // heads
            head = flat_row % heads
            for c0 in pl.range(0, channels - ROPE_DIM, 64):
                active = pl.min(64, channels - ROPE_DIM - c0)
                part = pl.slice(x, [1, 1, 64], [row, head, c0], valid_shape=[1, 1, active])
                output[row : row + 1, head : head + 1, c0 : c0 + 64] = part
            tail = pl.reshape(pl.slice(x, [1, 1, ROPE_DIM], [row, head, channels - ROPE_DIM]), [1, ROPE_DIM])
            even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            cosine = cos[row : row + 1, :]
            sine = pl.mul(sin[row : row + 1, :], direction)
            re = pl.sub(pl.mul(even, cosine), pl.mul(odd, sine))
            im = pl.add(pl.mul(even, sine), pl.mul(odd, cosine))
            rotated = pl.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
            rotated = pl.tensor.scatter(re, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
            rotated = pl.tensor.scatter(im, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
            output[row : row + 1, head : head + 1, channels - ROPE_DIM : channels] = pl.reshape(
                rotated, [1, 1, ROPE_DIM]
            )
    return output


_PREFILL_OPS_GLOBAL_BLOCKS = pl.dynamic("V41_FP32_GLOBAL_BLOCKS")


_PREFILL_OPS_SELECTED_KEYS = 128 + INDEX_TOPK


_PREFILL_OPS_WINDOW_BLOCKS = pl.dynamic("V41_FP32_WINDOW_BLOCKS")


@pl.jit.inline(auto_scope=False)
def prefill_sparse_attention_inline(
    query: pl.Tensor[[_PREFILL_OPS_ROWS, _PREFILL_OPS_HEADS, HEAD_DIM], pl.FP32],
    window_cache: pl.Tensor[[_PREFILL_OPS_WINDOW_BLOCKS, 128, HEAD_DIM], pl.FP32],
    window_indices: pl.Tensor[[_PREFILL_OPS_ROWS, 128], pl.INT32],
    global_cache: pl.Tensor[[_PREFILL_OPS_GLOBAL_BLOCKS, 128, HEAD_DIM], pl.FP32],
    global_indices: pl.Tensor[[_PREFILL_OPS_ROWS, INDEX_TOPK], pl.INT32],
    sink: pl.Tensor[[_PREFILL_OPS_HEADS], pl.FP32],
    output: pl.Tensor[[_PREFILL_OPS_ROWS, _PREFILL_OPS_HEADS, HEAD_DIM], pl.FP32],
):
    """FP32 sparse attention over explicit window/global rows and learned sinks.

    Negative indices are masked. Causal ownership and selection remain the
    caller's metadata contract. SWA passes all-negative global indices.
    """
    rows = pl.tensor.dim(query, 0)
    heads = pl.tensor.dim(query, 1)
    sink_rows = pl.reshape(sink, [1, heads])
    windows = pl.reshape(window_cache, [pl.tensor.dim(window_cache, 0) * 128, HEAD_DIM])
    globals_ = pl.reshape(global_cache, [pl.tensor.dim(global_cache, 0) * 128, HEAD_DIM])
    selected = pl.create_tensor([rows * _PREFILL_OPS_SELECTED_KEYS, HEAD_DIM], dtype=pl.FP32)
    valid = pl.create_tensor([rows, _PREFILL_OPS_SELECTED_KEYS], dtype=pl.FP32)
    logits = pl.create_tensor([rows, heads, _PREFILL_OPS_SELECTED_KEYS], dtype=pl.FP32)
    probabilities = pl.create_tensor([rows, heads, _PREFILL_OPS_SELECTED_KEYS], dtype=pl.FP32)

    for worker in pl.spmd(32, name_hint="prefill_sparse_gather"):
        for row in pl.range(worker, rows, 32):
            window_ids = pl.cast(window_indices[row : row + 1, :], pl.FP32)
            valid[row : row + 1, :128] = pl.minimum(pl.maximum(pl.add(window_ids, 1.0), 0.0), 1.0)
            global_ids = pl.cast(global_indices[row : row + 1, :], pl.FP32)
            valid[row : row + 1, 128:_PREFILL_OPS_SELECTED_KEYS] = pl.minimum(
                pl.maximum(pl.add(global_ids, 1.0), 0.0), 1.0
            )
            for column in pl.range(128):
                slot = pl.cast(pl.read(window_indices, [row, column]), pl.INDEX)
                destination = row * _PREFILL_OPS_SELECTED_KEYS + column
                if slot >= 0:
                    selected[destination : destination + 1, :] = windows[slot : slot + 1, :]
                else:
                    selected[destination : destination + 1, :] = pl.full(
                        [1, HEAD_DIM], dtype=pl.FP32, value=0.0
                    )
            for column in pl.range(INDEX_TOPK):
                slot = pl.cast(pl.read(global_indices, [row, column]), pl.INDEX)
                destination = row * _PREFILL_OPS_SELECTED_KEYS + 128 + column
                if slot >= 0:
                    selected[destination : destination + 1, :] = globals_[slot : slot + 1, :]
                else:
                    selected[destination : destination + 1, :] = pl.full(
                        [1, HEAD_DIM], dtype=pl.FP32, value=0.0
                    )

    head_tiles = (heads + 15) // 16
    score_blocks = rows * heads * (_PREFILL_OPS_SELECTED_KEYS // 64)
    for worker in pl.spmd(32, name_hint="prefill_sparse_qk"):
        for block in pl.range(worker, score_blocks, 32):
            key_block = block % (_PREFILL_OPS_SELECTED_KEYS // 64)
            head = block // (_PREFILL_OPS_SELECTED_KEYS // 64) % heads
            row = block // ((_PREFILL_OPS_SELECTED_KEYS // 64) * heads)
            k0 = row * _PREFILL_OPS_SELECTED_KEYS + key_block * 64
            qk_acc = pl.full([1, 64], dtype=pl.FP32, value=0.0)
            qk_error = pl.full([1, 64], dtype=pl.FP32, value=0.0)
            for channel in pl.range(0, HEAD_DIM, 64):
                q = pl.reshape(pl.slice(query, [1, 1, 64], [row, head, channel]), [1, 64])
                k = pl.slice(selected, [64, 64], [k0, channel])
                qk_part = pl.reshape(pl.row_sum(pl.col_expand_mul(k, q)), [1, 64])
                qk_corrected = pl.sub(qk_part, qk_error)
                qk_updated = pl.add(qk_acc, qk_corrected)
                qk_error = pl.sub(pl.sub(qk_updated, qk_acc), qk_corrected)
                qk_acc = qk_updated
            logits[row : row + 1, head : head + 1, key_block * 64 : key_block * 64 + 64] = pl.reshape(
                qk_acc, [1, 1, 64]
            )

    for worker in pl.spmd(32, name_hint="prefill_sparse_softmax"):
        for block in pl.range(worker, rows * ((heads + 7) // 8), 32):
            row = block // ((heads + 7) // 8)
            head = block % ((heads + 7) // 8) * 8
            active_heads = pl.min(8, heads - head)
            mask = valid[row : row + 1, :]
            score_rows = pl.reshape(
                pl.slice(
                    logits,
                    [1, 8, _PREFILL_OPS_SELECTED_KEYS],
                    [row, head, 0],
                    valid_shape=[1, active_heads, _PREFILL_OPS_SELECTED_KEYS],
                ),
                [8, _PREFILL_OPS_SELECTED_KEYS],
            )
            score_rows = pl.set_validshape(
                pl.fillpad(score_rows, pad_value=pl.PadValue.zero), 8, _PREFILL_OPS_SELECTED_KEYS
            )
            scores = pl.col_expand_add(pl.mul(score_rows, HEAD_DIM**-0.5), pl.mul(pl.sub(mask, 1.0), 1e30))
            sink_value = pl.slice(sink_rows, [1, 8], [0, head], valid_shape=[1, active_heads])
            sink_value = pl.set_validshape(pl.fillpad(sink_value, pad_value=pl.PadValue.zero), 1, 8)
            maximum = pl.maximum(pl.reshape(pl.row_max(scores), [1, 8]), sink_value)
            mass = pl.col_expand_mul(pl.exp(pl.row_expand_sub(scores, pl.reshape(maximum, [8, 1]))), mask)
            denominator = pl.add(pl.reshape(pl.row_sum(mass), [1, 8]), pl.exp(pl.sub(sink_value, maximum)))
            inverse = pl.reshape(pl.recip(denominator, high_precision=True), [8, 1])
            result = pl.row_expand_mul(mass, inverse)
            probabilities[row : row + 1, head : head + 8, :] = pl.reshape(
                pl.set_validshape(result, active_heads, _PREFILL_OPS_SELECTED_KEYS),
                [1, 8, _PREFILL_OPS_SELECTED_KEYS],
            )

    for block in pl.parallel(rows * head_tiles):
        head_block = block % head_tiles
        row = block // head_tiles
        for channel_block in pl.spmd(HEAD_DIM // 64, name_hint="prefill_sparse_pv"):
            channel = channel_block * 64
            active_heads = pl.min(16, heads - head_block * 16)
            acc = pl.tile.full([16, 64], dtype=pl.FP32, value=0.0)
            error = pl.tile.full([16, 64], dtype=pl.FP32, value=0.0)
            for key in pl.range(0, _PREFILL_OPS_SELECTED_KEYS, 64):
                p = pl.reshape(
                    pl.load(
                        probabilities,
                        [row, head_block * 16, key],
                        [1, 16, 64],
                        valid_shape=[1, active_heads, 64],
                    ),
                    [16, 64],
                )
                p = pl.set_validshape(pl.fillpad(p, pad_value=pl.PadValue.zero), 16, 64)
                v = pl.load(selected, [row * _PREFILL_OPS_SELECTED_KEYS + key, channel], [64, 64])
                part = pl.matmul(p, v)
                corrected = pl.sub(part, error)
                updated = pl.add(acc, corrected)
                error = pl.sub(pl.sub(updated, acc), corrected)
                acc = updated
            pv_result = pl.reshape(pl.set_validshape(acc, active_heads, 64), [1, 16, 64])
            output = pl.store(pv_result, [row, head_block * 16, channel], output)
    return output


@pl.jit.inline(auto_scope=False)
def prefill_linear_quantized_inline(
    x: pl.Tensor[[_PREFILL_OPS_ROWS, _PREFILL_OPS_INPUT_WIDTH], pl.FP32],
    weight: pl.Tensor[[_PREFILL_OPS_INPUT_WIDTH, _PREFILL_OPS_OUTPUT_WIDTH], pl.FP32],
    output: pl.Tensor[[_PREFILL_OPS_ROWS, _PREFILL_OPS_OUTPUT_WIDTH], pl.FP32],
):
    """Accumulate decoded MX operands in group32 order and narrow to BF16."""
    rows = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(x, 1)
    columns = pl.tensor.dim(weight, 1)
    for mt in pl.parallel((rows + 15) // 16):
        r0 = mt * 16
        for block in pl.spmd((columns + 63) // 64, name_hint="prefill_linear_quantized"):
            c0 = block * 64
            active_rows = pl.min(16, rows - r0)
            active_columns = pl.min(64, columns - c0)
            acc = pl.tile.full([16, 64], dtype=pl.FP32, value=0.0)
            for k0 in pl.range(0, width, 32):
                a = pl.load(x, [r0, k0], [16, 32], valid_shape=[active_rows, 32])
                a = pl.set_validshape(pl.fillpad(a, pad_value=pl.PadValue.zero), 16, 32)
                b = pl.load(weight, [k0, c0], [32, 64], valid_shape=[32, active_columns])
                b = pl.set_validshape(pl.fillpad(b, pad_value=pl.PadValue.zero), 32, 64)
                acc = pl.add(acc, pl.matmul(a, b))
            rounded = pl.cast(pl.cast(acc, pl.BF16, mode="rint"), pl.FP32)
            output = pl.store(pl.set_validshape(rounded, active_rows, active_columns), [r0, c0], output)
    return output


@pl.jit.inline(auto_scope=False)
def prefill_sparse_attention_official_inline(
    query: pl.Tensor[[_PREFILL_OPS_ROWS, _PREFILL_OPS_HEADS, HEAD_DIM], pl.FP32],
    window_cache: pl.Tensor[[_PREFILL_OPS_WINDOW_BLOCKS, 128, HEAD_DIM], pl.FP32],
    window_indices: pl.Tensor[[_PREFILL_OPS_ROWS, 128], pl.INT32],
    global_cache: pl.Tensor[[_PREFILL_OPS_GLOBAL_BLOCKS, 128, HEAD_DIM], pl.FP32],
    global_indices: pl.Tensor[[_PREFILL_OPS_ROWS, INDEX_TOPK], pl.INT32],
    sink: pl.Tensor[[_PREFILL_OPS_HEADS], pl.FP32],
    extents: pl.Tensor[[_PREFILL_OPS_ROWS, 2], pl.INT32],
    output: pl.Tensor[[_PREFILL_OPS_ROWS, _PREFILL_OPS_HEADS, HEAD_DIM], pl.FP32],
):
    """Online64 sparse attention with BF16 Q/K/V, probabilities and output."""
    rows = pl.tensor.dim(query, 0)
    heads = pl.tensor.dim(query, 1)
    sink_rows = pl.reshape(sink, [1, heads])
    windows = pl.reshape(window_cache, [pl.tensor.dim(window_cache, 0) * 128, HEAD_DIM])
    globals_ = pl.reshape(global_cache, [pl.tensor.dim(global_cache, 0) * 128, HEAD_DIM])
    selected = pl.create_tensor([rows * _PREFILL_OPS_SELECTED_KEYS, HEAD_DIM], dtype=pl.FP32)
    valid = pl.create_tensor([rows, _PREFILL_OPS_SELECTED_KEYS], dtype=pl.FP32)
    logits = pl.create_tensor([rows, heads, _PREFILL_OPS_SELECTED_KEYS], dtype=pl.FP32)

    for worker in pl.spmd(32, name_hint="prefill_official_gather"):
        for row in pl.range(worker, rows, 32):
            window_width = pl.cast(pl.read(extents, [row, 0]), pl.INDEX)
            global_width = pl.cast(pl.read(extents, [row, 1]), pl.INDEX)
            validity = pl.full([1, _PREFILL_OPS_SELECTED_KEYS], dtype=pl.FP32, value=0.0)
            for column in pl.range(_PREFILL_OPS_SELECTED_KEYS):
                destination = row * _PREFILL_OPS_SELECTED_KEYS + column
                value = pl.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0)
                if column < window_width:
                    slot = pl.cast(pl.read(window_indices, [row, column]), pl.INDEX)
                    if slot >= 0:
                        value = windows[slot : slot + 1, :]
                        pl.write(validity, [0, column], 1.0)
                elif column < window_width + global_width:
                    slot = pl.cast(pl.read(global_indices, [row, column - window_width]), pl.INDEX)
                    if slot >= 0:
                        value = globals_[slot : slot + 1, :]
                        pl.write(validity, [0, column], 1.0)
                selected[destination : destination + 1, :] = pl.cast(
                    pl.cast(value, pl.BF16, mode="rint"), pl.FP32
                )
            valid[row : row + 1, :] = validity

    score_blocks = rows * heads * (_PREFILL_OPS_SELECTED_KEYS // 64)
    for worker in pl.spmd(32, name_hint="prefill_official_qk"):
        for block in pl.range(worker, score_blocks, 32):
            key_block = block % (_PREFILL_OPS_SELECTED_KEYS // 64)
            head = block // (_PREFILL_OPS_SELECTED_KEYS // 64) % heads
            row = block // ((_PREFILL_OPS_SELECTED_KEYS // 64) * heads)
            k0 = row * _PREFILL_OPS_SELECTED_KEYS + key_block * 64
            qk_acc = pl.full([1, 64], dtype=pl.FP32, value=0.0)
            error = pl.full([1, 64], dtype=pl.FP32, value=0.0)
            for channel in pl.range(0, HEAD_DIM, 64):
                q = pl.reshape(pl.slice(query, [1, 1, 64], [row, head, channel]), [1, 64])
                q = pl.cast(pl.cast(q, pl.BF16, mode="rint"), pl.FP32)
                k = pl.slice(selected, [64, 64], [k0, channel])
                part = pl.reshape(pl.row_sum(pl.col_expand_mul(k, q)), [1, 64])
                corrected = pl.sub(part, error)
                updated = pl.add(qk_acc, corrected)
                error = pl.sub(pl.sub(updated, qk_acc), corrected)
                qk_acc = updated
            logits[row : row + 1, head : head + 1, key_block * 64 : key_block * 64 + 64] = pl.reshape(
                qk_acc, [1, 1, 64]
            )

    head_tiles = (heads + 15) // 16
    for block in pl.parallel(rows * head_tiles):
        head_block = block % head_tiles
        row = block // head_tiles
        for channel_block in pl.spmd(HEAD_DIM // 64, name_hint="prefill_official_online64"):
            channel = channel_block * 64
            head = head_block * 16
            active_heads = pl.min(16, heads - head)
            total = pl.read(extents, [row, 0]) + pl.read(extents, [row, 1])
            maximum = pl.full([1, 16], dtype=pl.FP32, value=-1e30)
            denominator = pl.full([1, 16], dtype=pl.FP32, value=0.0)
            acc = pl.full([16, 64], dtype=pl.FP32, value=0.0)
            for key in pl.range(0, total, 64):
                raw = pl.reshape(
                    pl.slice(logits, [1, 16, 64], [row, head, key], valid_shape=[1, active_heads, 64]),
                    [16, 64],
                )
                raw = pl.set_validshape(pl.fillpad(raw, pad_value=pl.PadValue.zero), 16, 64)
                mask = valid[row : row + 1, key : key + 64]
                scores = pl.col_expand_add(pl.mul(raw, HEAD_DIM**-0.5), pl.mul(pl.sub(mask, 1.0), 1e30))
                next_maximum = pl.maximum(maximum, pl.reshape(pl.row_max(scores), [1, 16]))
                correction = pl.exp(pl.sub(maximum, next_maximum))
                probability = pl.exp(pl.row_expand_sub(scores, pl.reshape(next_maximum, [16, 1])))
                probability = pl.col_expand_mul(probability, mask)
                denominator = pl.add(
                    pl.mul(denominator, correction), pl.reshape(pl.row_sum(probability), [1, 16])
                )
                probability = pl.cast(pl.cast(probability, pl.BF16, mode="rint"), pl.FP32)
                value_block = selected[
                    row * _PREFILL_OPS_SELECTED_KEYS + key : row * _PREFILL_OPS_SELECTED_KEYS + key + 64,
                    channel : channel + 64,
                ]
                weighted = pl.matmul(probability, value_block)
                acc = pl.add(pl.row_expand_mul(acc, pl.reshape(correction, [16, 1])), weighted)
                maximum = next_maximum
            sink_value = pl.slice(sink_rows, [1, 16], [0, head], valid_shape=[1, active_heads])
            sink_value = pl.set_validshape(pl.fillpad(sink_value, pad_value=pl.PadValue.zero), 1, 16)
            denominator = pl.add(denominator, pl.exp(pl.sub(sink_value, maximum)))
            divisor = pl.row_expand_mul(
                pl.full([16, 64], dtype=pl.FP32, value=1.0), pl.reshape(denominator, [16, 1])
            )
            result = pl.div(acc, divisor, high_precision=True)
            result = pl.cast(pl.cast(result, pl.BF16, mode="rint"), pl.FP32)
            pv_result = pl.reshape(pl.set_validshape(result, active_heads, 64), [1, 16, 64])
            output[row : row + 1, head : head + 16, channel : channel + 64] = pv_result
    return output


_PREFILL_COMPUTE_T = pl.dynamic("V41_COMPUTE_T")


_PREFILL_COMPUTE_K = pl.dynamic("V41_COMPUTE_K")


_PREFILL_COMPUTE_N = pl.dynamic("V41_COMPUTE_N")


@pl.jit.inline(auto_scope=False)
def prefill_linear_policy_inline(
    x: pl.Tensor[[_PREFILL_COMPUTE_T, _PREFILL_COMPUTE_K], pl.FP32],
    weight: pl.Tensor[[_PREFILL_COMPUTE_K, _PREFILL_COMPUTE_N], pl.FP32],
    output: pl.Tensor[[_PREFILL_COMPUTE_T, _PREFILL_COMPUTE_N], pl.FP32],
    official: pl.Scalar[pl.INT32],
    format: pl.Scalar[pl.INT32],
):
    rows = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(x, 1)
    columns = pl.tensor.dim(weight, 1)
    source = pl.reshape(x, [rows, width])
    matrix = pl.reshape(weight, [width, columns])
    destination = pl.reshape(output, [rows, columns])
    selected_format = pl.cast(0, pl.INT32)
    if official == 1:
        selected_format = format
    if selected_format == 2:
        quantized = pl.create_tensor([rows, width], dtype=pl.FP32)
        prefill_quantize_fp8_inline(source, quantized)
        prefill_linear_quantized_inline(quantized, matrix, destination)
    elif selected_format == 1:
        rounded_x = pl.create_tensor([rows, width], dtype=pl.FP32)
        rounded_w = pl.create_tensor([width, columns], dtype=pl.FP32)
        projected = pl.create_tensor([rows, columns], dtype=pl.FP32)
        prefill_round_bf16_inline(source, rounded_x)
        prefill_round_bf16_inline(matrix, rounded_w)
        prefill_linear_inline(rounded_x, rounded_w, projected)
        prefill_round_bf16_inline(projected, destination)
    else:
        prefill_linear_inline(source, matrix, destination)
    return output


_PREFILL_COMPUTE_B8 = pl.dynamic("V41_COMPUTE_B8")


@pl.jit.inline(auto_scope=False)
def prefill_packed_linear_inline(
    x: pl.Tensor[[_PREFILL_COMPUTE_T, _PREFILL_COMPUTE_K], pl.FP32],
    bank: pl.Tensor[[_PREFILL_COMPUTE_B8, 1024], pl.INT8],
    scales: pl.Tensor[[_PREFILL_COMPUTE_B8], pl.UINT8],
    offset: pl.Scalar[pl.INDEX],
    output: pl.Tensor[[_PREFILL_COMPUTE_T, _PREFILL_COMPUTE_N], pl.FP32],
    official: pl.Scalar[pl.INT32],
    format: pl.Scalar[pl.INT32],
):
    rows = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(x, 1)
    columns = pl.tensor.dim(output, 1)
    source = pl.reshape(x, [rows, width])
    destination = pl.reshape(output, [rows, columns])
    with pl.scope():
        weight = pl.create_tensor([width, columns], dtype=pl.FP32)
        prefill_decode_fp8_inline(bank, scales, offset, weight)
        prefill_linear_policy_inline(source, weight, destination, official, format)
    return output


_PREFILL_COMPUTE_NORM_EPS = FLASH.rms_norm_eps


@pl.jit.inline(auto_scope=False)
def prefill_norm_inline(
    x: pl.Tensor[[_PREFILL_COMPUTE_T, _PREFILL_COMPUTE_K], pl.FP32],
    weight: pl.Tensor[[_PREFILL_COMPUTE_K], pl.FP32],
    output: pl.Tensor[[_PREFILL_COMPUTE_T, _PREFILL_COMPUTE_K], pl.FP32],
    official: pl.Scalar[pl.INT32],
):
    rows = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(x, 1)
    normalized = pl.create_tensor([rows, width], dtype=pl.FP32)
    source = pl.reshape(x, [rows, width])
    gamma = pl.reshape(weight, [width])
    destination = pl.reshape(output, [rows, width])
    prefill_rmsnorm_inline(source, gamma, normalized, _PREFILL_COMPUTE_NORM_EPS)
    prefill_round_inline(normalized, destination, official)
    return output


_PREFILL_COMPUTE_C = pl.dynamic("V41_COMPUTE_C")


_PREFILL_COMPUTE_H = pl.dynamic("V41_COMPUTE_H")


@pl.jit.inline(auto_scope=False)
def prefill_rope_inline(
    x: pl.Tensor[[_PREFILL_COMPUTE_T, _PREFILL_COMPUTE_H, _PREFILL_COMPUTE_C], pl.FP32],
    cos: pl.Tensor[[_PREFILL_COMPUTE_T, ROPE_DIM // 2], pl.FP32],
    sin: pl.Tensor[[_PREFILL_COMPUTE_T, ROPE_DIM // 2], pl.FP32],
    output: pl.Tensor[[_PREFILL_COMPUTE_T, _PREFILL_COMPUTE_H, _PREFILL_COMPUTE_C], pl.FP32],
    direction: pl.Scalar[pl.FP32],
    official: pl.Scalar[pl.INT32],
):
    rows = pl.tensor.dim(x, 0)
    heads = pl.tensor.dim(x, 1)
    channels = pl.tensor.dim(x, 2)
    source = pl.reshape(x, [rows, heads, channels])
    cosine = pl.reshape(cos, [rows, ROPE_DIM // 2])
    sine = pl.reshape(sin, [rows, ROPE_DIM // 2])
    destination = pl.reshape(output, [rows, heads, channels])
    prefill_rope_values_inline(source, cosine, sine, destination, direction)
    if official == 1:
        for worker in pl.spmd(32, name_hint="prefill_rope_bf16"):
            for flat_row in pl.range(worker, rows * heads, 32):
                row = flat_row // heads
                head = flat_row % heads
                for col in pl.range(0, channels, 128):
                    active = pl.min(128, channels - col)
                    value = pl.slice(output, [1, 1, 128], [row, head, col], valid_shape=[1, 1, active])
                    narrowed = pl.cast(pl.cast(value, pl.BF16, mode="rint"), pl.FP32)
                    output[row : row + 1, head : head + 1, col : col + 128] = narrowed
    return output
