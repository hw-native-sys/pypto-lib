# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4.1 Flash RMSNorm kernels and shared Torch references."""

import pypto.language as pl
import torch

from .activation_quant import _prefill_quant_params
from .config import FLASH as M, TP_SIZE
from .dsl_types import T_DYN


# model config
D = M.hidden_size
EPS = M.rms_norm_eps

# tiling
NORM_T_TILE = 8
NORM_D_TILE = 512
DECODE_NORM_REDUCE_ROWS = ((D // NORM_D_TILE + NORM_T_TILE - 1) // NORM_T_TILE) * NORM_T_TILE
DECODE_NORM_WIDTH_TILE = DECODE_NORM_REDUCE_ROWS * NORM_D_TILE
PREFILL_NORM_REDUCE_TILE = 2 * NORM_D_TILE
PREFILL_NORM_STORE_TILE = 1280
PREFILL_NORM_WORKERS = 48
PREFILL_FFN_ROW_TILE = NORM_T_TILE // 2


@pl.jit.inline
def _rms_norm_tiled(
    x: pl.Tensor,
    norm_w: pl.Tensor[[D], pl.BF16],
    x_normed: pl.Tensor,
    block: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    compensated: pl.constexpr,
    output_fp32: pl.constexpr,
):
    """Normalize eight rows with optional compensated 512-column accumulation."""
    t0 = block * NORM_T_TILE
    remaining_rows = pl.max(num_tokens - t0, 0)
    valid_rows = pl.min(NORM_T_TILE, remaining_rows)
    sq_sum = pl.full([1, NORM_T_TILE], dtype=pl.FP32, value=0.0)
    compensation = pl.full([1, NORM_T_TILE], dtype=pl.FP32, value=0.0)
    for kb in pl.pipeline(D // NORM_D_TILE, stage=2):
        k0 = kb * NORM_D_TILE
        source = pl.slice(x, [NORM_T_TILE, NORM_D_TILE], [t0, k0], valid_shape=[valid_rows, NORM_D_TILE])
        source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), NORM_T_TILE, NORM_D_TILE)
        value = pl.cast(source, target_type=pl.FP32)
        partial = pl.reshape(pl.row_sum(pl.mul(value, value)), [1, NORM_T_TILE])
        if compensated:
            delta = pl.sub(partial, compensation)
            next_sum = pl.add(sq_sum, delta)
            compensation = pl.sub(pl.sub(next_sum, sq_sum), delta)
            sq_sum = next_sum
        else:
            sq_sum = pl.add(sq_sum, partial)
    inv_rms = pl.reshape(
        pl.rsqrt(pl.add(pl.mul(sq_sum, 1.0 / D), EPS), high_precision=True), [NORM_T_TILE, 1]
    )
    for kb in pl.pipeline(D // NORM_D_TILE, stage=2):
        k0 = kb * NORM_D_TILE
        source = pl.slice(x, [NORM_T_TILE, NORM_D_TILE], [t0, k0], valid_shape=[valid_rows, NORM_D_TILE])
        source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), NORM_T_TILE, NORM_D_TILE)
        value = pl.cast(source, target_type=pl.FP32)
        gamma = pl.reshape(pl.cast(norm_w[k0 : k0 + NORM_D_TILE], target_type=pl.FP32), [1, NORM_D_TILE])
        normalized = pl.col_expand_mul(pl.row_expand_mul(value, inv_rms), gamma)
        rounded = pl.cast(normalized, target_type=pl.BF16, mode="rint")
        if output_fp32:
            rounded_fp32 = pl.cast(rounded, pl.FP32)
            x_normed[t0 : t0 + NORM_T_TILE, k0 : k0 + NORM_D_TILE] = rounded_fp32
        else:
            valid_output = pl.set_validshape(rounded, valid_rows, NORM_D_TILE)
            x_normed[t0 : t0 + NORM_T_TILE, k0 : k0 + NORM_D_TILE] = valid_output
    return x_normed


@pl.jit.inline(auto_scope=False)
def rms_norm_tiled(
    x: pl.Tensor,
    norm_w: pl.Tensor[[D], pl.BF16],
    x_normed: pl.Tensor,
    block: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    output_fp32: pl.constexpr,
    compensated: pl.constexpr,
):
    """Normalize eight rows with BF16 rounding and optional padded FP32 output."""
    return _rms_norm_tiled(x, norm_w, x_normed, block, num_tokens, compensated=compensated, output_fp32=output_fp32)


@pl.jit.inline
def _decode_rms_norm_owner_tile(
    source: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[1, D], pl.BF16],
    owner: pl.Scalar[pl.INT32],
):
    values = pl.load(source, [owner, 0], [1, DECODE_NORM_WIDTH_TILE], valid_shape=[1, D])
    padded = pl.fillpad(values, pad_value=pl.PadValue.zero)
    padded = pl.set_validshape(padded, 1, DECODE_NORM_WIDTH_TILE)
    value_fp32 = pl.cast(padded, pl.FP32)
    chunks = pl.reshape(value_fp32, [DECODE_NORM_REDUCE_ROWS, NORM_D_TILE])
    squares = pl.mul(chunks, chunks)
    reduce_tmp = pl.create_tile([DECODE_NORM_REDUCE_ROWS, NORM_D_TILE], dtype=pl.FP32)
    partials = pl.row_sum(squares, tmp_tile=reduce_tmp)
    sum_squares = pl.tile.read(partials, [0, 0])
    for kb in pl.unroll(1, D // NORM_D_TILE):
        sum_squares = sum_squares + pl.tile.read(partials, [kb, 0])
    sum_tile = pl.tile.full([1, NORM_T_TILE], dtype=pl.FP32, value=0.0)
    sum_tile = pl.add(sum_tile, sum_squares)
    mean_square = pl.mul(sum_tile, 1.0 / D)
    rms_arg = pl.add(mean_square, EPS)
    inverse_tmp = pl.create_tile([1, NORM_T_TILE], dtype=pl.FP32)
    inverse = pl.tile.rsqrt(rms_arg, inverse_tmp, high_precision=True)
    gamma = pl.load(weight, [0, 0], [1, D])
    gamma_fp32 = pl.cast(gamma, pl.FP32)
    real_values = pl.tile.slice(value_fp32, [1, D], [0, 0])
    scaled = pl.mul(real_values, pl.tile.read(inverse, [0, 0]))
    weighted = pl.mul(scaled, gamma_fp32)
    rounded = pl.cast(weighted, pl.BF16, mode="rint")
    return rounded


@pl.jit.inline
def decode_rms_norm_owner_row(
    source: pl.Tensor[[T_DYN, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    output: pl.Tensor,
    owner: pl.Scalar[pl.INT32],
):
    """Normalize one Decode owner row with the original 512-column reduction order."""
    weight = pl.reshape(norm_w, [1, D])
    for row in pl.spmd(1, name_hint="decode_moe_owner_norm"):
        rounded = _decode_rms_norm_owner_tile(source, weight, owner)
        pl.store(rounded, [row, 0], output)
    return output


@pl.jit.inline(auto_scope=False)
def decode_ffn_norm_gate_input(
    source: pl.Tensor,
    norm_w: pl.Tensor[[D], pl.BF16],
    output: pl.Tensor,
    gate_input: pl.Tensor,
    owner: pl.Scalar[pl.INT32],
):
    """Prepare both gate and quantization inputs from the same BF16 owner row."""
    weight = pl.reshape(norm_w, [1, D])
    gate_input_tiles = 1 + D // 512
    gate_input_workers = pl.max(1, pl.min(gate_input_tiles, pl.system.available_aiv_count()))
    for gate_input_worker in pl.spmd(gate_input_workers, name_hint="decode_moe_norm_gate_input"):
        for worker in pl.range(gate_input_worker, gate_input_tiles, gate_input_workers):
            if worker == 0:
                rounded = _decode_rms_norm_owner_tile(source, weight, owner)
                pl.store(rounded, [0, 0], output)
                pl.store(pl.cast(rounded, pl.FP32), [0, 0], gate_input)
            else:
                column = (worker - 1) * 512
                zeros = pl.tile.full([8, 512], dtype=pl.FP32, value=0.0)
                pl.store(pl.set_validshape(zeros, 7, 512), [1, column], gate_input)
                pl.store(zeros, [8, column], gate_input)
    return output, gate_input


@pl.jit.inline(auto_scope=False)
def prefill_rms_norm(
    x: pl.Tensor,
    weight: pl.Tensor[[D], pl.BF16],
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    """Normalize Prefill rows with a resident BF16 tile and paired 512-column reductions."""
    blocks = (num_tokens + NORM_T_TILE - 1) // NORM_T_TILE
    workers = pl.min(pl.min(blocks, PREFILL_NORM_WORKERS), pl.system.available_aiv_count())
    for worker in pl.spmd(workers, name_hint="rms_norm"):
        gamma = pl.reshape(pl.load(weight, [0], [D]), [1, D])
        begin = worker * blocks // workers
        end = (worker + 1) * blocks // workers
        for block in pl.pipeline(begin, end, stage=1):
            row = block * NORM_T_TILE
            rows = pl.min(NORM_T_TILE, num_tokens - row)
            raw = pl.load(x, [row, 0], [NORM_T_TILE, D], valid_shape=[rows, D])
            padded = pl.fillpad(raw, pad_value=pl.PadValue.zero)
            cached = pl.set_validshape(padded, NORM_T_TILE, D)
            sum_squares = pl.tile.full([1, NORM_T_TILE], dtype=pl.FP32, value=0.0)
            scratch = pl.create_tile([2 * NORM_T_TILE, NORM_D_TILE], dtype=pl.FP32)
            for col in pl.range(0, D, PREFILL_NORM_REDUCE_TILE):
                panel = pl.tile.slice(cached, [NORM_T_TILE, PREFILL_NORM_REDUCE_TILE], [0, col])
                value = pl.cast(panel, pl.FP32)
                flat = pl.reshape(value, [2 * NORM_T_TILE, NORM_D_TILE])
                squares = pl.mul(flat, flat)
                row_sums = pl.row_sum(squares, tmp_tile=scratch)
                partial = pl.reshape(row_sums, [1, 2 * NORM_T_TILE])
                first_sum = pl.tile.gather_mask(partial, mask_pattern=pl.tile.MaskPattern.P0101)
                second_sum = pl.tile.gather_mask(partial, mask_pattern=pl.tile.MaskPattern.P1010)
                sum_squares = pl.add(sum_squares, first_sum)
                sum_squares = pl.add(sum_squares, second_sum)
            rsqrt_scratch = pl.create_tile([1, NORM_T_TILE], dtype=pl.FP32)
            mean_square = pl.add(pl.mul(sum_squares, 1.0 / D), EPS)
            inv_rms = pl.tile.rsqrt(mean_square, rsqrt_scratch, high_precision=True)
            inverse_rows = pl.reshape(inv_rms, [NORM_T_TILE, 1])
            for col in pl.range(0, D, PREFILL_NORM_STORE_TILE):
                output_panel = pl.tile.slice(cached, [NORM_T_TILE, PREFILL_NORM_STORE_TILE], [0, col])
                output_value = pl.cast(output_panel, pl.FP32)
                weight_panel_bf16 = pl.tile.slice(gamma, [1, PREFILL_NORM_STORE_TILE], [0, col])
                weight_panel = pl.cast(weight_panel_bf16, pl.FP32)
                scaled = pl.row_expand_mul(output_value, inverse_rows)
                weighted = pl.col_expand_mul(scaled, weight_panel)
                rounded = pl.cast(weighted, pl.BF16, mode="rint")
                valid = pl.set_validshape(rounded, rows, PREFILL_NORM_STORE_TILE)
                pl.store(valid, [row, col], output)
    return output


@pl.jit.inline(auto_scope=False)
def prefill_ffn_norm_quant(
    x: pl.Tensor,
    weight: pl.Tensor[[D], pl.BF16],
    output: pl.Tensor,
    payload: pl.Tensor,
    scales: pl.Tensor,
    gate_input: pl.Tensor,
    local_tokens: pl.Scalar[pl.INDEX],
    tp_rank: pl.Scalar[pl.INT32],
    capacity: pl.constexpr,
):
    """Normalize TP owner rows and produce BF16, FP32 router and INT8 expert inputs."""
    blocks = capacity // PREFILL_FFN_ROW_TILE
    workers = pl.min(pl.min(blocks, PREFILL_NORM_WORKERS), pl.system.available_aiv_count())
    for worker in pl.spmd(workers, name_hint="prefill_ffn_norm_quant"):
        gamma = pl.reshape(pl.load(weight, [0], [D]), [1, D])
        # Each of the four rows occupies two consecutive vector lanes.
        lanes = pl.tile.arange(0, [1, NORM_T_TILE], dtype=pl.INT32)
        first_half = pl.mul(pl.shrs(lanes, 1), 2)
        second_half = pl.add(first_half, 1)
        scale_lanes = pl.mul(pl.sub(lanes, pl.mul(pl.shrs(lanes, 2), 4)), 2)
        gather_tmp = pl.create_tile([1, NORM_T_TILE], dtype=pl.INT32)
        begin = worker * blocks // workers
        end = (worker + 1) * blocks // workers
        for block in pl.range(begin, end):
            row = block * PREFILL_FFN_ROW_TILE
            cached = pl.tile.full([PREFILL_FFN_ROW_TILE, D], dtype=pl.BF16, value=0.0)
            for lane in pl.unroll(PREFILL_FFN_ROW_TILE):
                local_row = row + lane
                if local_row < local_tokens:
                    owner_row = local_row * TP_SIZE + tp_rank
                    owner_tile = pl.load(x, [owner_row, 0], [1, D])
                    cached[lane : lane + 1, :] = owner_tile
            sum_squares = pl.tile.full([1, NORM_T_TILE], dtype=pl.FP32, value=0.0)
            reduce_tmp = pl.create_tile([NORM_T_TILE, NORM_D_TILE], dtype=pl.FP32)
            for col in pl.range(0, D, PREFILL_NORM_REDUCE_TILE):
                reduce_panel = pl.tile.slice(cached, [PREFILL_FFN_ROW_TILE, PREFILL_NORM_REDUCE_TILE], [0, col])
                reduce_value = pl.cast(reduce_panel, pl.FP32)
                reduce_halves = pl.reshape(reduce_value, [NORM_T_TILE, NORM_D_TILE])
                squares = pl.mul(reduce_halves, reduce_halves)
                row_sums = pl.row_sum(squares, tmp_tile=reduce_tmp)
                partial = pl.reshape(row_sums, [1, NORM_T_TILE])
                # Accumulate the 512-column sums in the same order as the non-fused RMSNorm.
                first_sum = pl.tile.gather(partial, first_half, gather_tmp)
                sum_squares = pl.add(sum_squares, first_sum)
                second_sum = pl.tile.gather(partial, second_half, gather_tmp)
                sum_squares = pl.add(sum_squares, second_sum)
            rsqrt_tmp = pl.create_tile([1, NORM_T_TILE], dtype=pl.FP32)
            mean_square = pl.add(pl.mul(sum_squares, 1.0 / D), EPS)
            inv_rms = pl.tile.rsqrt(mean_square, rsqrt_tmp, high_precision=True)
            inverse_rows = pl.reshape(inv_rms, [NORM_T_TILE, 1])

            norm_cache = pl.create_tile([PREFILL_FFN_ROW_TILE, D], dtype=pl.BF16)
            peak_bits = pl.tile.full([NORM_T_TILE, PREFILL_NORM_STORE_TILE // 2], dtype=pl.INT16, value=0)
            for col in pl.range(0, D, PREFILL_NORM_STORE_TILE):
                norm_panel = pl.tile.slice(cached, [PREFILL_FFN_ROW_TILE, PREFILL_NORM_STORE_TILE], [0, col])
                norm_value = pl.cast(norm_panel, pl.FP32)
                weight_panel = pl.tile.slice(gamma, [1, PREFILL_NORM_STORE_TILE], [0, col])
                weight_fp32 = pl.cast(weight_panel, pl.FP32)
                norm_halves = pl.reshape(norm_value, [NORM_T_TILE, PREFILL_NORM_STORE_TILE // 2])
                scaled = pl.row_expand_mul(norm_halves, inverse_rows)
                scaled_rows = pl.reshape(scaled, [PREFILL_FFN_ROW_TILE, PREFILL_NORM_STORE_TILE])
                weighted = pl.col_expand_mul(scaled_rows, weight_fp32)
                rounded = pl.cast(weighted, pl.BF16, mode="rint")
                norm_cache[0:PREFILL_FFN_ROW_TILE, col:col + PREFILL_NORM_STORE_TILE] = rounded
                gate_panel = pl.cast(rounded, pl.FP32)
                pl.store(gate_panel, [row, col], gate_input)
                rounded_halves = pl.reshape(rounded, [NORM_T_TILE, PREFILL_NORM_STORE_TILE // 2])
                rounded_bits = pl.reinterpret_view(rounded_halves, pl.INT16)
                peak_bits = pl.maximum(peak_bits, pl.ands(rounded_bits, 32767))

            # Publish the complete diagnostic slice, including zero-padded inactive rows.
            pl.store(norm_cache, [row, 0], output)
            peak = pl.cast(pl.reinterpret_view(peak_bits, pl.BF16), pl.FP32)
            peak_tmp = pl.create_tile([NORM_T_TILE, PREFILL_NORM_STORE_TILE // 2], dtype=pl.FP32)
            half_peaks = pl.reshape(pl.row_max(peak, tmp_tile=peak_tmp), [1, NORM_T_TILE])
            first_peak = pl.tile.gather(half_peaks, first_half, gather_tmp)
            second_peak = pl.tile.gather(half_peaks, second_half, gather_tmp)
            maximum = pl.reshape(pl.maximum(first_peak, second_peak), [NORM_T_TILE, 1])
            scale, quant_inverse = _prefill_quant_params(maximum)
            gathered_scale = pl.tile.gather(pl.reshape(scale, [1, NORM_T_TILE]), scale_lanes, gather_tmp)
            scale_column = pl.reshape(gathered_scale, [NORM_T_TILE, 1])
            pl.store(pl.set_validshape(scale_column, PREFILL_FFN_ROW_TILE, 1), [row, 0], scales)
            for col in pl.range(0, D, PREFILL_NORM_STORE_TILE):
                quant_panel = pl.tile.slice(norm_cache, [PREFILL_FFN_ROW_TILE, PREFILL_NORM_STORE_TILE], [0, col])
                quant_value = pl.cast(quant_panel, pl.FP32)
                quant_halves = pl.reshape(quant_value, [NORM_T_TILE, PREFILL_NORM_STORE_TILE // 2])
                normalized = pl.row_expand_mul(quant_halves, quant_inverse)
                integer = pl.cast(normalized, pl.INT16, mode="rint")
                exact_fp16 = pl.cast(integer, pl.FP16, mode="rint")
                codes = pl.cast(exact_fp16, pl.INT8, mode="trunc")
                code_rows = pl.reshape(codes, [PREFILL_FFN_ROW_TILE, PREFILL_NORM_STORE_TILE])
                pl.store(code_rows, [row, col], payload)
    return output, payload, scales, gate_input


@pl.jit.inline
def rms_norm(
    x: pl.Tensor,
    weight: pl.Tensor,
    output: pl.Tensor,
    block: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.constexpr,
):
    """Normalize one eight-row block using whole-row reduction."""
    t = block * 8
    rows = pl.min(8, num_tokens - t)
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


def golden_rms_norm_tiled(x: torch.Tensor, norm_w: torch.Tensor) -> torch.Tensor:
    """Apply RMSNorm with FP32 reference accumulation over 512-column chunks."""
    value = x.float()
    sq_sum = torch.zeros(value.shape[0], 1, dtype=torch.float32)
    for d0 in range(0, D, NORM_D_TILE):
        chunk = value[:, d0 : d0 + NORM_D_TILE]
        sq_sum += (chunk * chunk).sum(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(sq_sum * (1.0 / D) + EPS)
    return (value * inv_rms * norm_w.float()).to(torch.bfloat16)


def golden_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = M.rms_norm_eps) -> torch.Tensor:
    """Apply RMSNorm in FP32 and restore the activation dtype."""
    dtype = x.dtype
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    return (value * weight.float()).to(dtype)


def make_norm_with_deps(width, *, max_workers=None, name_hint="attention_rmsnorm"):
    """Specialize whole-row RMSNorm with an explicit producer dependency."""
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
        workers = pl.min(pl.min(blocks, worker_limit), pl.system.available_aiv_count())
        with pl.spmd(workers, name_hint=name_hint, deps=[ready]) as norm_tid:
            worker = pl.tile.get_block_idx()
            for block in pl.range(worker, blocks, workers):
                rms_norm(x, weight, output, block, num_tokens, width=width)
        return norm_tid

    return normalize
