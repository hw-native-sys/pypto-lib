# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared V4.1 Q/KV projection, normalization, RoPE, and window cache publication."""


from __future__ import annotations

from pathlib import Path

import pypto.language as pl
from pypto.runtime import pto_isa_include_dir
import torch

from config import FLASH as M
from gemm import (
    MX_M_TILE, MX_K_TILE,
    golden_gemm_bf16_via_mxfp8_to_bf16_sequential, golden_mxfp8_linear_grouped_sum,
)
from rmsnorm import golden_rms_norm


# Dynamic shape variables.
QA_INPUT_ROWS = pl.dynamic("QA_INPUT_ROWS")
QA_SCALE_COLS = pl.dynamic("QA_SCALE_COLS")
QA_OUTPUT_ROWS = pl.dynamic("QA_OUTPUT_ROWS")
KV_INPUT_ROWS = pl.dynamic("KV_INPUT_ROWS")
KV_SCALE_COLS = pl.dynamic("KV_SCALE_COLS")
KV_OUTPUT_ROWS = pl.dynamic("KV_OUTPUT_ROWS")
QB_INPUT_ROWS = pl.dynamic("QB_INPUT_ROWS")
QB_SCALE_COLS = pl.dynamic("QB_SCALE_COLS")
QB_OUTPUT_ROWS = pl.dynamic("QB_OUTPUT_ROWS")

# model config
LOCAL_H = M.num_attention_heads
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank

# tiling
Q_A_N_TILE = 64
Q_B_N_TILE = 256
KV_N_TILE = 64
QKV_SCALE_TILE = MX_M_TILE * MX_K_TILE // 32
PROJ_M_TILE = 80
PROJ_WIDE_M_TILE = 128
WINDOW_PUBLISH_WORKER_TILE = 12
NATIVE_VALIDATION_WORKERS = 32
WINDOW_HEAD_DIM = 512
WINDOW_PAGE_SIZE = 128

# Window tiling.
WINDOW_ROW_TILE = 8
WINDOW_SLOT_TILE = 128


@pl.jit.extern(core_type="aic", source=Path(__file__).parent / "kernels" / "proj_q_a.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _proj_q_a(
    payload: pl.Tensor[[QA_INPUT_ROWS, D], pl.FP8E4M3FN],
    weight: pl.Tensor,
    activation_scale: pl.Tensor[[1, QA_SCALE_COLS], pl.FP8E8M0],
    weight_scale: pl.Tensor,
    output: pl.Out[pl.Tensor[[QA_OUTPUT_ROWS, Q_LORA], pl.BF16]],
    rows: pl.Scalar[pl.INDEX],
    row_tile: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    """Project ND activations and NZ weights with packed MX scales."""
    return output


@pl.jit.extern(core_type="mixed", aic_source=Path(__file__).parent / "kernels" / "proj_q_b.cpp",
               aiv_source=Path(__file__).parent / "kernels" / "proj_q_b.cpp", include_dirs=(pto_isa_include_dir(),))
def _proj_q_b(
    payload: pl.Tensor[[QB_INPUT_ROWS, Q_LORA], pl.FP8E4M3FN],
    weight: pl.Tensor,
    activation_scale: pl.Tensor[[1, QB_SCALE_COLS], pl.FP8E8M0],
    weight_scale: pl.Tensor,
    output: pl.Out[pl.Tensor[[QB_OUTPUT_ROWS, LOCAL_H * HEAD_DIM], pl.BF16]],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    rows: pl.Scalar[pl.INDEX],
    output_width: pl.Scalar[pl.INDEX],
    row_start: pl.Scalar[pl.INDEX],
    row_tile: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    """Project Q and rotate the BF16-rounded tail on the paired Vector core."""
    return output


@pl.jit.extern(core_type="aic", source=Path(__file__).parent / "kernels" / "proj_kv.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _proj_kv(
    payload: pl.Tensor[[KV_INPUT_ROWS, D], pl.FP8E4M3FN],
    weight: pl.Tensor,
    activation_scale: pl.Tensor[[1, KV_SCALE_COLS], pl.FP8E8M0],
    weight_scale: pl.Tensor,
    output: pl.Out[pl.Tensor[[KV_OUTPUT_ROWS, HEAD_DIM], pl.BF16]],
    rows: pl.Scalar[pl.INDEX],
    row_start: pl.Scalar[pl.INDEX],
    row_tile: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    """Project ND activations and NZ weights with packed MX scales."""
    return output


@pl.jit.inline(auto_scope=False)
def _load_qkv_work(
    x: pl.Tensor,
    inputs: pl.Tile[[2 * MX_M_TILE, MX_K_TILE], pl.BF16, pl.Mem.Vec],
    work: pl.Scalar[pl.INDEX],
    slot: pl.Scalar[pl.INDEX],
    num_tokens: pl.Scalar[pl.INT32],
):
    t0 = work // (D // MX_K_TILE) * MX_M_TILE
    k0 = work % (D // MX_K_TILE) * MX_K_TILE
    rows = pl.max(0, pl.min(MX_M_TILE, num_tokens - t0))
    if rows > 0:
        input_row = slot * MX_M_TILE
        inputs = pl.tile.gather_row(inputs, x, [input_row, 0], [t0, k0], [MX_M_TILE, MX_K_TILE],
                                    valid_shape=[rows, MX_K_TILE])
    return inputs


@pl.jit.incore
def qkv_quant(
    x: pl.Tensor,
    payload: pl.Out[pl.Tensor],
    scales: pl.Out[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
    padded_rows: pl.Scalar[pl.INDEX],
) -> tuple[pl.Tensor, pl.Tensor]:
    """Quantize row-major BF16 projection inputs with M16-packed E8M0 scales."""
    first = pl.tile.get_block_idx()
    workers = pl.tile.get_block_num()
    column_tiles = D // MX_K_TILE
    total = padded_rows // MX_M_TILE * column_tiles
    if first < total:
        inputs = pl.create_tile([2 * MX_M_TILE, MX_K_TILE], dtype=pl.BF16, target_memory=pl.Mem.Vec)
        output = pl.create_tile([2 * MX_M_TILE * MX_K_TILE], dtype=pl.INT8, target_memory=pl.Mem.Vec)
        # Reserve a full 256-byte register store per scale slot.
        codes = pl.create_tile([QKV_SCALE_TILE], dtype=pl.UINT32, target_memory=pl.Mem.Vec)
        inputs = _load_qkv_work(x, inputs, first, 0, num_tokens)
        pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.V, 0)
        pl.system.set_flag(pl.PipeType.MTE3, pl.PipeType.V, 0)
        pl.system.set_flag(pl.PipeType.MTE3, pl.PipeType.V, 1)
        rounds = (total - first + workers - 1) // workers
        for iteration in pl.range(rounds):
            work = first + iteration * workers
            slot = iteration % 2
            next_slot = 1 - slot
            next_work = work + workers
            # Prefetch the next tile while the vector pipe consumes the current slot.
            if next_work < total:
                if iteration > 0:
                    pl.system.wait_flag(pl.PipeType.V, pl.PipeType.MTE2, next_slot)
                inputs = _load_qkv_work(x, inputs, next_work, next_slot, num_tokens)
                pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.V, next_slot)
            t0 = work // column_tiles * MX_M_TILE
            k0 = work % column_tiles * MX_K_TILE
            rows = pl.max(0, pl.min(MX_M_TILE, num_tokens - t0))
            source = pl.reshape(inputs, [2 * MX_M_TILE * MX_K_TILE])
            pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.V, slot)
            pl.system.wait_flag(pl.PipeType.MTE3, pl.PipeType.V, slot)
            output_words = pl.reinterpret_view(output, pl.UINT32, shape=[2 * MX_M_TILE * MX_K_TILE // 4])
            with pl.reg_scope():
                # Keep aligned row and slot offsets visible to register lowering.
                slot_row = slot * MX_M_TILE
                code_offset = slot * (QKV_SCALE_TILE // 2)
                full = pl.reg.full_mask(lanes=MX_K_TILE)
                groups = pl.reg.full_mask(lanes=MX_K_TILE // 32)
                packed_mask = pl.reg.full_mask(lanes=64)
                # Eight words per group pair hold two rows of scale bytes each.
                lane = pl.reg.index_gen(0, lanes=64, dtype=pl.UINT32)
                pair_mask = pl.reg.broadcast(3, lanes=64, dtype=pl.UINT32)
                pair_index = pl.reg.shift_rights(lane, 3, packed_mask)
                pair_index = pl.reg.bitwise_and(pair_index, pair_mask, packed_mask)
                low_index = pl.reg.shift_lefts(pair_index, 4, packed_mask)
                high_index = pl.reg.adds(low_index, 8, packed_mask)
                row_mask = pl.reg.broadcast(7, lanes=64, dtype=pl.UINT32)
                row_indices = pl.reg.bitwise_and(lane, row_mask, packed_mask)
                zero = pl.reg.broadcast(0, lanes=64, dtype=pl.UINT32)
                for row, (packed_scales,) in pl.hardware_loop(rows, init_values=(zero,)):
                    tile_row = slot_row + row
                    offset = tile_row * MX_K_TILE
                    raw = pl.reg.load(source, offset, lanes=MX_K_TILE)
                    value = pl.reg.convert(raw, dtype=pl.FP32)
                    magnitude = pl.reg.abs(value, full)
                    maximum = pl.reg.reduce_max(magnitude, full, group_size=32)
                    maximum_bits = pl.reg.interpret_cast(maximum, dtype=pl.INT32)
                    exponent = pl.reg.shift_rights(maximum_bits, 23, groups)
                    exponent = pl.reg.adds(exponent, -8, groups)
                    exponent = pl.reg.maximums(exponent, 0, groups)
                    nonzero = pl.reg.minimums(exponent, 1, groups)
                    reciprocal_code = pl.reg.muls(exponent, -1, groups)
                    reciprocal_code = pl.reg.adds(reciprocal_code, 254, groups)
                    reciprocal_code = pl.reg.mul(reciprocal_code, nonzero, groups)
                    reciprocal_bits = pl.reg.shift_lefts(reciprocal_code, 23, groups)
                    reciprocal = pl.reg.interpret_cast(reciprocal_bits, dtype=pl.FP32)
                    reciprocal = pl.reg.group_broadcast(reciprocal, lanes=MX_K_TILE)
                    value = pl.reg.mul(value, reciprocal, full)
                    value = pl.reg.maximums(value, -448.0, full)
                    value = pl.reg.minimums(value, 448.0, full)
                    fp8 = pl.reg.convert(value, dtype=pl.FP8E4M3FN, rounding="R", saturate=False)
                    words = pl.reg.interpret_cast(fp8, dtype=pl.UINT32)
                    word_offset = tile_row * (MX_K_TILE // 4)
                    _payload = pl.reg.store(output_words, word_offset, words)

                    scale_words = pl.reg.interpret_cast(exponent, dtype=pl.UINT32)
                    scale_words = pl.reg.group_broadcast(scale_words, lanes=64)
                    low = pl.reg.gather_lanes(scale_words, low_index)
                    high = pl.reg.gather_lanes(scale_words, high_index)
                    high = pl.reg.shift_lefts(high, 8, packed_mask)
                    pair = pl.reg.bitwise_or(low, high, packed_mask)
                    bit = pl.cast((row % 2) * 16, pl.INT16)
                    pair = pl.reg.shift_lefts(pair, bit, packed_mask)
                    word = pl.cast(row // 2, pl.UINT32)
                    row_selected = pl.reg.compares(row_indices, word, packed_mask, predicate="eq")
                    pair = pl.reg.select(row_selected, pair, zero)
                    updated_scales = pl.reg.bitwise_or(packed_scales, pair, packed_mask)
                    packed_scales = pl.yield_(updated_scales)
                _scales = pl.reg.store(codes, code_offset, packed_scales)
                padding = pl.reg.broadcast(0, lanes=MX_K_TILE // 4, dtype=pl.UINT32)
                for inactive in pl.hardware_loop(MX_M_TILE - rows):
                    row = rows + inactive
                    tile_row = slot_row + row
                    word_offset = tile_row * (MX_K_TILE // 4)
                    _payload_pad = pl.reg.store(output_words, word_offset, padding)
            pl.system.set_flag(pl.PipeType.V, pl.PipeType.MTE2, slot)
            pl.system.set_flag(pl.PipeType.V, pl.PipeType.MTE3, slot)
            pl.system.wait_flag(pl.PipeType.V, pl.PipeType.MTE3, slot)
            output_rows = pl.reshape(output, [2 * MX_M_TILE, MX_K_TILE])
            output_tile = pl.tile.slice(output_rows, [MX_M_TILE, MX_K_TILE], [slot * MX_M_TILE, 0])
            quantized = pl.reinterpret_view(output_tile, pl.FP8E4M3FN)
            scale_bytes = pl.reinterpret_view(codes, pl.UINT8, shape=[2, 2 * QKV_SCALE_TILE])
            scale_tile = pl.tile.slice(scale_bytes, [1, QKV_SCALE_TILE], [slot, 0])
            scale_codes = pl.reinterpret_view(scale_tile, pl.FP8E8M0)
            scale_offset = t0 * (D // 32) + k0 // MX_K_TILE * QKV_SCALE_TILE
            pl.store(quantized, [t0, k0], payload)
            pl.store(scale_codes, [0, scale_offset], scales)
            pl.system.set_flag(pl.PipeType.MTE3, pl.PipeType.V, slot)
        pl.system.wait_flag(pl.PipeType.V, pl.PipeType.MTE2, 0)
        if rounds > 1:
            pl.system.wait_flag(pl.PipeType.V, pl.PipeType.MTE2, 1)
        pl.system.wait_flag(pl.PipeType.MTE3, pl.PipeType.V, 0)
        pl.system.wait_flag(pl.PipeType.MTE3, pl.PipeType.V, 1)
        pl.system.barrier(pipe=pl.PipeType.ALL)
    return payload, scales


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels" / "q_norm_quant.cpp",
               include_dirs=(pto_isa_include_dir(),))
def q_norm_quant(
    projected: pl.Tensor,
    weight: pl.Tensor,
    latent: pl.Out[pl.Tensor],
    payload: pl.Out[pl.Tensor],
    scales: pl.Out[pl.Tensor],
    rows: pl.Scalar[pl.INT32],
):
    """Normalize latent rows and write M16-packed MXFP8 outputs."""
    ...


@pl.jit.inline(auto_scope=False)
def quantize_projection_input(
    x: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    row_tile: pl.constexpr = 80,
):
    """Quantize shared Q/KV inputs with padded rows and M16-packed scales."""
    capacity = pl.tensor.dim(x, 0)
    projection_rows = ((capacity + row_tile - 1) // row_tile) * row_tile
    # Shared input covers both native projection row tiles.
    index_rows = ((capacity + PROJ_M_TILE - 1) // PROJ_M_TILE) * PROJ_M_TILE
    padded_rows = pl.max(projection_rows, index_rows)
    payload = pl.create_tensor([padded_rows, D], dtype=pl.FP8E4M3FN)
    scales = pl.create_tensor([1, padded_rows * (D // 32)], dtype=pl.FP8E8M0)
    quant_blocks = (padded_rows // MX_M_TILE) * (D // MX_K_TILE)
    workers = pl.min(quant_blocks, 2 * pl.system.available_cluster_count())
    with pl.spmd(workers, name_hint="qkv_quant"):
        payload, scales = qkv_quant(x, payload, scales, num_tokens, padded_rows)
    return payload, scales, padded_rows


@pl.jit.inline(auto_scope=False)
def q_proj_rope(
    payload: pl.Tensor,
    scales: pl.Tensor,
    wq_a: pl.Tensor,
    wq_a_scale: pl.Tensor,
    q_norm_weight: pl.Tensor,
    wq_b: pl.Tensor,
    wq_b_scale: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    query_latent: pl.Tensor,
    query: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    padded_rows: pl.Scalar[pl.INDEX],
    qa_worker_limit: pl.constexpr = 32,
    qb_worker_limit: pl.constexpr = 32,
    qa_row_tile: pl.constexpr = 80,
    qb_row_tile: pl.constexpr = 80,
):
    """Project NZ QA/QB weights, rotate Q, and retain normalized rows for Index-Q."""
    # Native QA tiles: M80/N64, M128/N128, or M256/N128; QB tiles: M80/N256 or M128/N256.
    qa_tile = qa_row_tile
    # Minimize padded row work; equal padding selects the wider tile.
    narrow_rows = ((num_tokens + PROJ_M_TILE - 1) // PROJ_M_TILE) * PROJ_M_TILE
    wide_rows = ((num_tokens + PROJ_WIDE_M_TILE - 1) // PROJ_WIDE_M_TILE) * PROJ_WIDE_M_TILE
    if qa_row_tile == PROJ_M_TILE:
        if wide_rows <= narrow_rows:
            if padded_rows >= wide_rows:
                qa_tile = PROJ_WIDE_M_TILE
    if qa_tile == 80:
        qa_n_tile = Q_A_N_TILE
    else:
        qa_n_tile = 128
    qb_tile = qb_row_tile
    if qb_row_tile == PROJ_M_TILE:
        if wide_rows <= narrow_rows:
            qb_tile = PROJ_WIDE_M_TILE
    qb_padded_rows = ((num_tokens + qb_tile - 1) // qb_tile) * qb_tile
    payload_rows = pl.max(padded_rows, qb_padded_rows)
    qb_payload = pl.create_tensor([payload_rows, Q_LORA], dtype=pl.FP8E4M3FN)
    qb_scales = pl.create_tensor([1, payload_rows * (Q_LORA // 32)], dtype=pl.FP8E8M0)
    projected = pl.create_tensor([payload_rows, Q_LORA], dtype=pl.BF16)
    qa_tasks = ((num_tokens + qa_tile - 1) // qa_tile) * (Q_LORA // qa_n_tile)
    qa_workers = pl.max(pl.min(qa_tasks, qa_worker_limit), 1)
    with pl.spmd(qa_workers, name_hint="proj_q_a"):
        projected = _proj_q_a(payload, wq_a, scales, wq_a_scale, projected, num_tokens, qa_tile)
    norm_workers = pl.max(pl.min(payload_rows // (MX_M_TILE // 2), 32), 1)
    latent_rows = pl.tensor.dim(query_latent, 0)
    norm_latent = pl.tensor.slice(query_latent, [latent_rows, Q_LORA], [0, 0])
    norm_payload = pl.tensor.slice(qb_payload, [payload_rows, Q_LORA], [0, 0])
    norm_scales = pl.tensor.slice(qb_scales, [1, payload_rows * (Q_LORA // 32)], [0, 0])
    with pl.spmd(norm_workers, name_hint="q_norm_quant"):
        q_norm_quant(projected, q_norm_weight, norm_latent, norm_payload, norm_scales, num_tokens)
    qb_tasks = ((num_tokens + qb_tile - 1) // qb_tile) * (LOCAL_H * HEAD_DIM // Q_B_N_TILE)
    qb_workers = pl.max(pl.min(qb_tasks, pl.min(qb_worker_limit, pl.system.available_cluster_count())), 1)
    with pl.spmd(qb_workers, name_hint="proj_q_b"):
        query = _proj_q_b(
            qb_payload, wq_b, qb_scales, wq_b_scale,
            query, rope_cos, rope_sin, num_tokens, LOCAL_H * HEAD_DIM, 0, qb_tile,
        )
    return query_latent, query, qb_payload, qb_scales


@pl.jit.inline(auto_scope=False)
def kv_proj(
    payload: pl.Tensor,
    scales: pl.Tensor,
    wkv: pl.Tensor,
    wkv_scale: pl.Tensor,
    kv_projection: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    worker_limit: pl.constexpr = 32,
    row_tile: pl.constexpr = 80,
):
    """Project NZ KV weights to BF16 with bounded active rows."""
    # Native KV tiles: M80/N64 or M256/N256.
    if row_tile == 256:
        n_tile = 256
    else:
        n_tile = KV_N_TILE
    tasks = ((num_tokens + row_tile - 1) // row_tile) * (HEAD_DIM // n_tile)
    workers = pl.max(pl.min(tasks, pl.min(worker_limit, pl.system.available_cluster_count())), 1)
    with pl.spmd(workers, name_hint="proj_kv"):
        kv_projection = _proj_kv(payload, wkv, scales, wkv_scale, kv_projection, num_tokens, 0, row_tile)
    return kv_projection


@pl.jit.inline(auto_scope=False)
def _window_postprocess(
    raw: pl.Tile[[WINDOW_ROW_TILE * WINDOW_HEAD_DIM], pl.BF16, pl.Mem.Vec],
    weight: pl.Tile[[WINDOW_HEAD_DIM], pl.BF16, pl.Mem.Vec],
    cosine: pl.Tile[[WINDOW_ROW_TILE * 32], pl.FP32, pl.Mem.Vec],
    sine: pl.Tile[[WINDOW_ROW_TILE * 32], pl.FP32, pl.Mem.Vec],
    rows: pl.Scalar[pl.INDEX],
):
    packed = pl.tile.create([WINDOW_ROW_TILE * WINDOW_HEAD_DIM], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
    factors = pl.tile.create([WINDOW_ROW_TILE * 8], dtype=pl.UINT32, target_memory=pl.Mem.Vec)
    with pl.reg_scope():
        full = pl.reg.full_mask(lanes=64)
        lower = pl.reg.active_mask(32, lanes=64)
        upper = pl.reg.mask_not(lower)
        first_index = pl.reg.broadcast(0, lanes=64, dtype=pl.UINT32)
        last_lane = pl.reg.broadcast(63, lanes=64, dtype=pl.UINT32)
        one = pl.reg.broadcast(1.0, lanes=64, dtype=pl.FP32)
        scale_words = pl.reg.broadcast(0, lanes=64, dtype=pl.UINT32)
        for row in pl.hardware_loop(rows):
            partials = pl.reg.broadcast(0.0, lanes=64, dtype=pl.FP32)
            lane = pl.reg.index_gen(0, lanes=64, dtype=pl.UINT32)
            for chunk in pl.hardware_loop(4):
                left_bf = pl.reg.load(raw, row * WINDOW_HEAD_DIM + chunk * 64, lanes=64)
                right_bf = pl.reg.load(raw, row * WINDOW_HEAD_DIM + 256 + chunk * 64, lanes=64)
                left = pl.reg.convert(left_bf, dtype=pl.FP32)
                right = pl.reg.convert(right_bf, dtype=pl.FP32)
                square = pl.reg.add(pl.reg.mul(left, left, full), pl.reg.mul(right, right, full), full)
                partial = pl.reg.reduce_sum(square, full, reassoc=True)
                partial = pl.reg.broadcast(partial, lanes=64, dtype=pl.FP32)
                slot_index = pl.cast(chunk, pl.UINT32)
                slot = pl.reg.compares(lane, slot_index, full, predicate="eq")
                partials = pl.reg.select(slot, partial, partials)
            partial_mask = pl.reg.active_mask(4, lanes=64)
            total = pl.reg.reduce_sum(partials, partial_mask, reassoc=True)
            total = pl.reg.broadcast(total, lanes=64, dtype=pl.FP32)
            variance = pl.reg.adds(pl.reg.muls(total, 1.0 / WINDOW_HEAD_DIM, full), 1.0e-6, full)
            reciprocal = pl.reg.div(one, variance, full)
            root = pl.reg.sqrt(reciprocal, full)
            product = pl.reg.mul(pl.reg.muls(variance, -0.5, full), root, full)
            three_halves = pl.reg.broadcast(1.5, lanes=64, dtype=pl.FP32)
            factor = pl.reg.mula(three_halves, product, root, full)
            inverse = pl.reg.mul(root, factor, full)
            negative_variance = pl.reg.muls(variance, -1.0, full)
            correction = pl.reg.mula(one, negative_variance, reciprocal, full)
            negative_inverse = pl.reg.muls(inverse, -1.0, full)
            error = pl.reg.mula(reciprocal, negative_inverse, inverse, full)
            correction = pl.reg.mula(correction, variance, error, full)
            correction = pl.reg.mul(correction, inverse, full)
            half = pl.reg.broadcast(0.5, lanes=64, dtype=pl.FP32)
            inverse = pl.reg.mula(inverse, correction, half, full)
            variance_bits = pl.reg.interpret_cast(variance, dtype=pl.UINT32)
            is_infinite = pl.reg.compares(variance_bits, 2139095040, full, predicate="eq")
            is_zero = pl.reg.compares(variance_bits, 0, full, predicate="eq")
            zero = pl.reg.broadcast(0.0, lanes=64, dtype=pl.FP32)
            infinity_bits = pl.reg.broadcast(2139095040, lanes=64, dtype=pl.UINT32)
            infinity = pl.reg.interpret_cast(infinity_bits, dtype=pl.FP32)
            inverse = pl.reg.select(is_infinite, zero, inverse)
            inverse = pl.reg.select(is_zero, infinity, inverse)
            cos = pl.reg.load(cosine, row * 32, lanes=64)
            sin = pl.reg.load(sine, row * 32, lanes=64)
            for segment in pl.hardware_loop(WINDOW_HEAD_DIM // 64 - 1):
                offset = row * WINDOW_HEAD_DIM + segment * 64
                value = pl.reg.convert(pl.reg.load(raw, offset, lanes=64), dtype=pl.FP32)
                gain = pl.reg.convert(pl.reg.load(weight, segment * 64, lanes=64), dtype=pl.FP32)
                normalized = pl.reg.mul(pl.reg.mul(value, inverse, full), gain, full)
                bf = pl.reg.convert(normalized, dtype=pl.BF16, rounding="R", saturate=False)
                value = pl.reg.convert(bf, dtype=pl.FP32)
                magnitude = pl.reg.abs(value, full)
                lo = pl.reg.broadcast(pl.reg.reduce_max(magnitude, lower), lanes=64, dtype=pl.FP32)
                hi = pl.reg.broadcast(pl.reg.reduce_max(magnitude, upper), lanes=64, dtype=pl.FP32)
                maximum = pl.reg.select(lower, lo, hi)
                scale = pl.reg.muls(pl.reg.maximums(maximum, 1.0e-4, full), 1.0 / 448.0, full)
                scale_exponent_bits = pl.reg.interpret_cast(scale, dtype=pl.INT32)
                rounded_bits = pl.reg.adds(scale_exponent_bits, 8388607, full)
                exponent = pl.reg.shift_rights(rounded_bits, 23, full)
                power_bits = pl.reg.shift_lefts(exponent, 23, full)
                scale = pl.reg.interpret_cast(power_bits, dtype=pl.FP32)
                quantized = pl.reg.div(value, scale, full)
                fp8 = pl.reg.convert(quantized, dtype=pl.FP8E4M3FN, rounding="R", saturate=False)
                bytes8 = pl.reg.interpret_cast(fp8, dtype=pl.UINT8)
                _packed = pl.reg.store(packed, offset, bytes8)
                # Pack two BF16 scale codes per segment.
                scale_bits = pl.reg.interpret_cast(scale, dtype=pl.UINT32)
                high_bits = pl.reg.gather_lanes(scale_bits, last_lane)
                low_bits = pl.reg.shift_rights(scale_bits, 16, full)
                codes = pl.reg.bitwise_or(low_bits, high_bits, full)
                code_index = row * 8 + segment
                before = pl.reg.active_mask(code_index, lanes=64)
                through = pl.reg.active_mask(code_index + 1, lanes=64)
                code_lane = pl.reg.mask_xor(before, through)
                code_word = pl.reg.gather_lanes(codes, first_index)
                scale_words = pl.reg.select(code_lane, code_word, scale_words)
            offset = row * WINDOW_HEAD_DIM + (WINDOW_HEAD_DIM // 64 - 1) * 64
            value = pl.reg.convert(pl.reg.load(raw, offset, lanes=64), dtype=pl.FP32)
            gain = pl.reg.convert(pl.reg.load(weight, (WINDOW_HEAD_DIM // 64 - 1) * 64, lanes=64), dtype=pl.FP32)
            normalized = pl.reg.mul(pl.reg.mul(value, inverse, full), gain, full)
            bf = pl.reg.convert(normalized, dtype=pl.BF16, rounding="R", saturate=False)
            value = pl.reg.convert(bf, dtype=pl.FP32)
            even, odd = pl.reg.dintlv(value, value, full)
            real = pl.reg.sub(pl.reg.mul(even, cos, lower), pl.reg.mul(odd, sin, lower), lower)
            imag = pl.reg.add(pl.reg.mul(even, sin, lower), pl.reg.mul(odd, cos, lower), lower)
            rotated, _unused = pl.reg.intlv(real, imag, full)
            bf = pl.reg.convert(rotated, dtype=pl.BF16, rounding="R", saturate=False)
            value = pl.reg.convert(bf, dtype=pl.FP32)
            magnitude = pl.reg.abs(value, full)
            lo = pl.reg.broadcast(pl.reg.reduce_max(magnitude, lower), lanes=64, dtype=pl.FP32)
            hi = pl.reg.broadcast(pl.reg.reduce_max(magnitude, upper), lanes=64, dtype=pl.FP32)
            maximum = pl.reg.select(lower, lo, hi)
            scale = pl.reg.muls(pl.reg.maximums(maximum, 1.0e-4, full), 1.0 / 448.0, full)
            scale_exponent_bits = pl.reg.interpret_cast(scale, dtype=pl.INT32)
            rounded_bits = pl.reg.adds(scale_exponent_bits, 8388607, full)
            exponent = pl.reg.shift_rights(rounded_bits, 23, full)
            power_bits = pl.reg.shift_lefts(exponent, 23, full)
            scale = pl.reg.interpret_cast(power_bits, dtype=pl.FP32)
            quantized = pl.reg.div(value, scale, full)
            fp8 = pl.reg.convert(quantized, dtype=pl.FP8E4M3FN, rounding="R", saturate=False)
            bytes8 = pl.reg.interpret_cast(fp8, dtype=pl.UINT8)
            _packed = pl.reg.store(packed, offset, bytes8)
            # Pack two BF16 scale codes per segment.
            scale_bits = pl.reg.interpret_cast(scale, dtype=pl.UINT32)
            high_bits = pl.reg.gather_lanes(scale_bits, last_lane)
            low_bits = pl.reg.shift_rights(scale_bits, 16, full)
            codes = pl.reg.bitwise_or(low_bits, high_bits, full)
            code_index = row * 8 + (WINDOW_HEAD_DIM // 64 - 1)
            before = pl.reg.active_mask(code_index, lanes=64)
            through = pl.reg.active_mask(code_index + 1, lanes=64)
            code_lane = pl.reg.mask_xor(before, through)
            code_word = pl.reg.gather_lanes(codes, first_index)
            scale_words = pl.reg.select(code_lane, code_word, scale_words)
        _factors = pl.reg.store(factors, 0, scale_words)
    return packed, factors


@pl.jit.inline(auto_scope=False)
def _flush_window(
    inputs: pl.Tile[[WINDOW_ROW_TILE, WINDOW_HEAD_DIM], pl.BF16, pl.Mem.Vec],
    weight: pl.Tile[[WINDOW_HEAD_DIM], pl.BF16, pl.Mem.Vec],
    cosine: pl.Tile[[WINDOW_ROW_TILE, 32], pl.FP32, pl.Mem.Vec],
    sine: pl.Tile[[WINDOW_ROW_TILE, 32], pl.FP32, pl.Mem.Vec],
    targets: pl.Tile[[WINDOW_ROW_TILE], pl.INT64, pl.Mem.Vec],
    rows: pl.Scalar[pl.INDEX],
    cache: pl.Tensor,
):
    raw = pl.reshape(inputs, [WINDOW_ROW_TILE * WINDOW_HEAD_DIM])
    cos = pl.reshape(cosine, [WINDOW_ROW_TILE * 32])
    sin = pl.reshape(sine, [WINDOW_ROW_TILE * 32])
    # Expose the populated prefix as valid columns for the specialized VF calls.
    raw = pl.reshape(raw, [1, WINDOW_ROW_TILE * WINDOW_HEAD_DIM])
    raw = pl.tile.slice(raw, [1, WINDOW_ROW_TILE * WINDOW_HEAD_DIM], [0, 0], valid_shape=[1, rows * WINDOW_HEAD_DIM])
    raw = pl.reshape(raw, [WINDOW_ROW_TILE * WINDOW_HEAD_DIM])
    cos = pl.reshape(cos, [1, WINDOW_ROW_TILE * 32])
    cos = pl.tile.slice(cos, [1, WINDOW_ROW_TILE * 32], [0, 0], valid_shape=[1, rows * 32])
    cos = pl.reshape(cos, [WINDOW_ROW_TILE * 32])
    sin = pl.reshape(sin, [1, WINDOW_ROW_TILE * 32])
    sin = pl.tile.slice(sin, [1, WINDOW_ROW_TILE * 32], [0, 0], valid_shape=[1, rows * 32])
    sin = pl.reshape(sin, [WINDOW_ROW_TILE * 32])
    # A constant row count lets PTOAS prove the VF buffers do not overlap.
    if rows == 2:
        payload, factors = _window_postprocess(raw, weight, cos, sin, 2)
    elif rows == 3:
        payload, factors = _window_postprocess(raw, weight, cos, sin, 3)
    elif rows == 4:
        payload, factors = _window_postprocess(raw, weight, cos, sin, 4)
    elif rows == 5:
        payload, factors = _window_postprocess(raw, weight, cos, sin, 5)
    elif rows == 6:
        payload, factors = _window_postprocess(raw, weight, cos, sin, 6)
    elif rows == 7:
        payload, factors = _window_postprocess(raw, weight, cos, sin, 7)
    else:
        # Keep the dynamic one-row tail: PTOAS 0.72 exhausts its layout search
        # when specializing it. A full batch reaches here with constant WINDOW_ROW_TILE.
        payload, factors = _window_postprocess(raw, weight, cos, sin, rows)
    payload_rows = pl.reshape(payload, [WINDOW_ROW_TILE, WINDOW_HEAD_DIM])
    factor_bytes = pl.reinterpret_view(factors, pl.UINT8)
    factor_rows = pl.reshape(factor_bytes, [WINDOW_ROW_TILE, WINDOW_HEAD_DIM // 16])
    for (row,) in pl.while_(init_values=(pl.cast(0, pl.INDEX),)):
        pl.cond(row < rows)
        slot = pl.tile.read(targets, [row])
        count = 1
        scanning = True
        while row + count < rows and scanning:
            next_slot = pl.tile.read(targets, [row + count])
            if next_slot == slot + count:
                count = count + 1
            else:
                scanning = False
        value = pl.tile.slice(payload_rows, [WINDOW_ROW_TILE, WINDOW_HEAD_DIM], [row, 0], valid_shape=[count, WINDOW_HEAD_DIM])
        factor = pl.tile.slice(factor_rows, [WINDOW_ROW_TILE, WINDOW_HEAD_DIM // 16], [row, 0], valid_shape=[count, WINDOW_HEAD_DIM // 16])
        _payload = pl.store(value, [slot, 0], cache)
        _scales = pl.store(factor, [slot, WINDOW_HEAD_DIM], cache)
        next_row = row + count
        final_row = pl.yield_(next_row)
    return cache


@pl.jit.incore
def norm_rope_publish(
    source: pl.Tensor,
    weight: pl.Tensor,
    cosine: pl.Tensor,
    sine: pl.Tensor,
    slots: pl.Tensor,
    cache: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    """Normalize, rotate, quantize, and publish rows owned by this cache page worker."""
    worker = pl.tile.get_block_idx()
    workers = pl.tile.get_block_num()
    gamma = pl.load(weight, [0], [WINDOW_HEAD_DIM], target_memory=pl.Mem.Vec)
    inputs = pl.tile.create([WINDOW_ROW_TILE, WINDOW_HEAD_DIM], dtype=pl.BF16, target_memory=pl.Mem.Vec)
    cosv = pl.tile.create([WINDOW_ROW_TILE, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec)
    sinv = pl.tile.create([WINDOW_ROW_TILE, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec)
    targets = pl.tile.create([WINDOW_ROW_TILE], dtype=pl.INT64, target_memory=pl.Mem.Vec)
    pending = 0
    for base in pl.range(0, num_tokens, WINDOW_SLOT_TILE):
        count = pl.min(WINDOW_SLOT_TILE, num_tokens - base)
        slot_tile = pl.load(slots, [base], [WINDOW_SLOT_TILE], valid_shape=[count], target_memory=pl.Mem.Vec)
        lane = 0
        while lane < count:
            slot = pl.tile.read(slot_tile, [lane])
            accepted = False
            if slot >= 0:
                unsigned_slot = pl.cast(slot, pl.UINT64)
                owner = (unsigned_slot // WINDOW_PAGE_SIZE) % pl.cast(workers, pl.UINT64)
                accepted = owner == pl.cast(worker, pl.UINT64)
            if accepted:
                run = 0
                scanning = True
                while lane + run < count and pending + run < WINDOW_ROW_TILE and scanning:
                    target = pl.tile.read(slot_tile, [lane + run])
                    owned = False
                    if target >= 0:
                        unsigned_target = pl.cast(target, pl.UINT64)
                        target_owner = (unsigned_target // WINDOW_PAGE_SIZE) % pl.cast(workers, pl.UINT64)
                        owned = target_owner == pl.cast(worker, pl.UINT64)
                    if owned:
                        _target = pl.tile.write(targets, [pending + run], target)
                        run = run + 1
                    else:
                        scanning = False
                token = base + lane
                inputs = pl.tile.gather_row(
                    inputs, source, [pending, 0], [token, 0], [WINDOW_ROW_TILE, WINDOW_HEAD_DIM], valid_shape=[run, WINDOW_HEAD_DIM]
                )
                cosv = pl.tile.gather_row(
                    cosv, cosine, [pending, 0], [token, 0], [WINDOW_ROW_TILE, 32], valid_shape=[run, 32]
                )
                sinv = pl.tile.gather_row(
                    sinv, sine, [pending, 0], [token, 0], [WINDOW_ROW_TILE, 32], valid_shape=[run, 32]
                )
                lane = lane + run
                pending = pending + run
                if pending == WINDOW_ROW_TILE:
                    cache = _flush_window(inputs, gamma, cosv, sinv, targets, WINDOW_ROW_TILE, cache)
                    pending = 0
            else:
                lane = lane + 1
    if pending > 0:
        cache = _flush_window(inputs, gamma, cosv, sinv, targets, pending, cache)
    return cache


@pl.jit.inline(auto_scope=False)
def window_publish(
    kv_projection: pl.Tensor,
    kv_norm_weight: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    slots: pl.Tensor,
    cache: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    """Fuse RMSNorm, RoPE, group-32 FP8 quantization, and window cache publication."""
    workers = pl.max(1, pl.min(num_tokens, WINDOW_PUBLISH_WORKER_TILE))
    cache_rows = pl.tensor.dim(cache, 0) * 128
    flat_cache = pl.reshape(cache, [cache_rows, 544])
    with pl.spmd(workers, name_hint="norm_rope_pub_win", deps=[cache_ready]) as published:
        _published = norm_rope_publish(kv_projection, kv_norm_weight, rope_cos, rope_sin, slots, flat_cache, num_tokens)
    return published



@pl.jit.inline(auto_scope=False)
def proj_q_a_pypto(
    payload: pl.Tensor,
    weight: pl.Tensor,
    activation_scale: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.Tensor,
    rows: pl.Scalar[pl.INDEX],
    worker_limit: pl.constexpr = 32,
):
    scale_width = weight.shape[0] // 32
    for mt in pl.parallel((rows + 79) // 80):
        scale_begin = mt * 80 * scale_width
        scale_end = scale_begin + 80 * scale_width
        sa = activation_scale[0, scale_begin:scale_end]
        sa = pl.reshape(sa, [80, scale_width])
        sa = pl.tensor.view(sa, [80, scale_width], layout=pl.MX_A_ZZ)
        for worker in pl.spmd(pl.min(worker_limit, 20)):
            for base in pl.range(0, 20, worker_limit):
                if base + worker < 20:
                    row = mt * 80
                    col = (base + worker) * 64
                    full_a = pl.load(payload, [row, 0], [80, 5120], target_memory=pl.Mem.Mat)
                    a0 = pl.tile.slice(full_a, [80, 256], [0, 0])
                    sa0 = pl.load(sa, [0, 0], [80, 8])
                    b0 = pl.load(weight, [0, col], [256, 64], target_memory=pl.Mem.Mat)
                    sb0 = pl.load(weight_scale, [0, col], [8, 64])
                    acc = pl.matmul_mx(a0, sa0, b0, sb0)
                    for k in pl.pipeline(256, 5120, 256, stage=2):
                        av = pl.tile.slice(full_a, [80, 256], [0, k])
                        sav = pl.load(sa, [0, k // 32], [80, 8])
                        bv = pl.load(weight, [k, col], [256, 64], target_memory=pl.Mem.Mat)
                        sbv = pl.load(weight_scale, [k // 32, col], [8, 64])
                        acc = pl.matmul_mx_acc(acc, av, sav, bv, sbv)
                    valid = pl.set_validshape(acc, pl.min(80, rows - row), 64)
                    output = pl.store(valid, [row, col], output)
    return output


@pl.jit.inline(auto_scope=False)
def _q_b_rope_chunk(
    source: pl.Tile[[SOURCE_TILE * 64], pl.BF16, pl.Mem.Vec],  # noqa: F821 - bound by the inline constexpr parameters
    cosine: pl.Tile[[SOURCE_TILE, 32], pl.FP32, pl.Mem.Vec, pl.TileView(stride=[32, 1])],  # noqa: F821 - bound by the inline constexpr parameters
    sine: pl.Tile[[SOURCE_TILE, 32], pl.FP32, pl.Mem.Vec, pl.TileView(stride=[32, 1])],  # noqa: F821 - bound by the inline constexpr parameters
    output: pl.Tile[[ROW_TILE * 64], pl.BF16, pl.Mem.Vec],  # noqa: F821 - bound by the inline constexpr parameters
    row_start: pl.constexpr,
    ROW_TILE: pl.constexpr,
    SOURCE_TILE: pl.constexpr,
):
    """Rotate adjacent BF16 lane pairs from a local UB row chunk."""
    with pl.reg_scope():
        full = pl.reg.full_mask(lanes=64)
        for row in pl.hardware_loop(ROW_TILE // 2):
            c = pl.reg.load(cosine, offset=row_start * 32 + row * 64, lanes=64)
            s = pl.reg.load(sine, offset=row_start * 32 + row * 64, lanes=64)
            raw0 = pl.reg.load(source, offset=row_start * 64 + row * 128, lanes=64)
            raw1 = pl.reg.load(source, offset=row_start * 64 + row * 128 + 64, lanes=64)
            x0 = pl.reg.convert(raw0, dtype=pl.FP32)
            x1 = pl.reg.convert(raw1, dtype=pl.FP32)
            even, odd = pl.reg.dintlv(x0, x1, full)
            ec = pl.reg.mul(even, c, full)
            os = pl.reg.mul(odd, s, full)
            es = pl.reg.mul(even, s, full)
            oc = pl.reg.mul(odd, c, full)
            real = pl.reg.sub(ec, os, full)
            imag = pl.reg.add(es, oc, full)
            low, high = pl.reg.intlv(real, imag, full)
            bf0 = pl.reg.convert(low, dtype=pl.BF16, rounding="R", saturate=False)
            bf1 = pl.reg.convert(high, dtype=pl.BF16, rounding="R", saturate=False)
            _low = pl.reg.store(output, offset=row * 128, value=bf0)
            _high = pl.reg.store(output, offset=row * 128 + 64, value=bf1)
    return output


@pl.jit.inline(auto_scope=False)
def _proj_q_b_tile_pypto(
    payload: pl.Tensor,
    weight: pl.Tensor,
    activation_scale: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.Tensor,
    cosine: pl.Tensor,
    sine: pl.Tensor,
    rows: pl.Scalar[pl.INDEX],
    width: pl.Scalar[pl.INDEX],
    row_start: pl.Scalar[pl.INDEX],
    worker_limit: pl.constexpr = 32,
    M_TILE: pl.constexpr = 80,
):
    """Project MXFP8 Q tiles and rotate BF16-rounded tails in a mixed group."""
    scale_rows = activation_scale.shape[1] // 40
    sa = pl.reshape(activation_scale, [scale_rows, 40])
    sa = pl.tensor.view(sa, [scale_rows, 40], layout=pl.MX_A_ZZ)
    row_tiles = (rows + M_TILE - 1) // M_TILE
    tasks = row_tiles * (width // 256)
    worker_cap = pl.min(worker_limit, pl.system.available_cluster_count())
    workers = pl.max(pl.min(tasks, worker_cap), 1)
    lanes = workers
    if row_tiles > 1:
        lanes = pl.max(workers // row_tiles // 2 * 2, 2)
    for worker in pl.spmd(workers, optimizations=[pl.cross_core_slot(slot_num=2)]):
        for mt in pl.range(row_tiles):
            for n_lane in pl.range(lanes):
                if (mt * lanes + n_lane) % workers == worker:
                    row = mt * M_TILE
                    load_rows = pl.min(M_TILE, payload.shape[0] - row)
                    full_a = pl.load(payload, [row, 0], [M_TILE, 1280], valid_shape=[load_rows, 1280], target_memory=pl.Mem.Mat)
                    # Preserve the physical M pitch through masked accumulator stores.
                    full_a = pl.set_validshape(full_a, M_TILE, 1280)
                    sa_cache_0 = pl.load(sa, [row, 0], [M_TILE, 4], valid_shape=[load_rows, 4], target_memory=pl.Mem.Mat)
                    sa_cache_0 = pl.set_validshape(sa_cache_0, M_TILE, 4)
                    sa_cache_1 = pl.load(sa, [row, 4], [M_TILE, 4], valid_shape=[load_rows, 4], target_memory=pl.Mem.Mat)
                    sa_cache_1 = pl.set_validshape(sa_cache_1, M_TILE, 4)
                    sa_cache_2 = pl.load(sa, [row, 8], [M_TILE, 4], valid_shape=[load_rows, 4], target_memory=pl.Mem.Mat)
                    sa_cache_2 = pl.set_validshape(sa_cache_2, M_TILE, 4)
                    sa_cache_3 = pl.load(sa, [row, 12], [M_TILE, 4], valid_shape=[load_rows, 4], target_memory=pl.Mem.Mat)
                    sa_cache_3 = pl.set_validshape(sa_cache_3, M_TILE, 4)
                    sa_cache_4 = pl.load(sa, [row, 16], [M_TILE, 4], valid_shape=[load_rows, 4], target_memory=pl.Mem.Mat)
                    sa_cache_4 = pl.set_validshape(sa_cache_4, M_TILE, 4)
                    sa_cache_5 = pl.load(sa, [row, 20], [M_TILE, 4], valid_shape=[load_rows, 4], target_memory=pl.Mem.Mat)
                    sa_cache_5 = pl.set_validshape(sa_cache_5, M_TILE, 4)
                    sa_cache_6 = pl.load(sa, [row, 24], [M_TILE, 4], valid_shape=[load_rows, 4], target_memory=pl.Mem.Mat)
                    sa_cache_6 = pl.set_validshape(sa_cache_6, M_TILE, 4)
                    sa_cache_7 = pl.load(sa, [row, 28], [M_TILE, 4], valid_shape=[load_rows, 4], target_memory=pl.Mem.Mat)
                    sa_cache_7 = pl.set_validshape(sa_cache_7, M_TILE, 4)
                    sa_cache_8 = pl.load(sa, [row, 32], [M_TILE, 4], valid_shape=[load_rows, 4], target_memory=pl.Mem.Mat)
                    sa_cache_8 = pl.set_validshape(sa_cache_8, M_TILE, 4)
                    sa_cache_9 = pl.load(sa, [row, 36], [M_TILE, 4], valid_shape=[load_rows, 4], target_memory=pl.Mem.Mat)
                    sa_cache_9 = pl.set_validshape(sa_cache_9, M_TILE, 4)
                    # M80 A slots 2/5 map to scale offsets 1280/3200.
                    # Two M80 accumulator slots; M256 occupies all L0C.
                    if M_TILE < 256:
                        # Select a positive stride for NZ offset range analysis.
                        angle_rows = pl.min(M_TILE, rows - row)
                        cos_resident = pl.load(cosine, [row_start + row, 0], [M_TILE, 32], valid_shape=[angle_rows, 32], target_memory=pl.Mem.Vec)
                        sin_resident = pl.load(sine, [row_start + row, 0], [M_TILE, 32], valid_shape=[angle_rows, 32], target_memory=pl.Mem.Vec)
                        for n_stride in pl.range(1, lanes + 1):
                            if n_stride == lanes:
                                waves = (width // 256 - n_lane + n_stride - 1) // n_stride
                                for wave in pl.range((waves + 3) // 4):
                                    if wave * 4 + 3 < waves:
                                        col_g0 = (n_lane + (wave * 4 + 0) * n_stride) * 256
                                        col_g1 = (n_lane + (wave * 4 + 1) * n_stride) * 256
                                        col_g2 = (n_lane + (wave * 4 + 2) * n_stride) * 256
                                        col_g3 = (n_lane + (wave * 4 + 3) * n_stride) * 256
                                        sb_cache_0_g0 = pl.load(weight_scale, [0, col_g0], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_1_g0 = pl.load(weight_scale, [4, col_g0], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_2_g0 = pl.load(weight_scale, [8, col_g0], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_3_g0 = pl.load(weight_scale, [12, col_g0], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_4_g0 = pl.load(weight_scale, [16, col_g0], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_5_g0 = pl.load(weight_scale, [20, col_g0], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_6_g0 = pl.load(weight_scale, [24, col_g0], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_7_g0 = pl.load(weight_scale, [28, col_g0], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_8_g0 = pl.load(weight_scale, [32, col_g0], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_9_g0 = pl.load(weight_scale, [36, col_g0], [4, 256], target_memory=pl.Mem.Mat)
                                        b_pre_0_g0: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[0], pl.Mem.Mat] = pl.load(weight, [0, col_g0], [256, 256], target_memory=pl.Mem.Mat)
                                        b_pre_1_g0: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[1], pl.Mem.Mat] = pl.load(weight, [256, col_g0], [256, 256], target_memory=pl.Mem.Mat)
                                        b_next_0_g0: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[2], pl.Mem.Mat] = pl.load(weight, [512, col_g0], [256, 256], target_memory=pl.Mem.Mat)
                                        b_next_1_g0: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[3], pl.Mem.Mat] = pl.load(weight, [768, col_g0], [256, 256], target_memory=pl.Mem.Mat)
                                        a_l0_0_g0: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 0, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_0_g0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_pre_0_g0, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_0_g0: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_0, target_memory=pl.Mem.LeftScale)
                                        sb_l0_0_g0: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_0_g0, target_memory=pl.Mem.RightScale)
                                        acc_g0: pl.Tile[[M_TILE, 256], pl.FP32, pl.MemRef("qb_acc", slots=2)[0], pl.Mem.Acc] = pl.matmul_mx(a_l0_0_g0, sa_l0_0_g0, b_l0_0_g0, sb_l0_0_g0)
                                        a_l0_1_g0: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 128, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_1_g0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_pre_0_g0, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_1_g0: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_1, target_memory=pl.Mem.LeftScale)
                                        sb_l0_1_g0: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_1_g0, target_memory=pl.Mem.RightScale)
                                        acc_g0 = pl.matmul_mx_acc(acc_g0, a_l0_1_g0, sa_l0_1_g0, b_l0_1_g0, sb_l0_1_g0)
                                        b_next_2_g0: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[0], pl.Mem.Mat] = pl.load(weight, [1024, col_g0], [256, 256], target_memory=pl.Mem.Mat)
                                        a_l0_2_g0: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 256, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_2_g0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_pre_1_g0, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_2_g0: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_2, target_memory=pl.Mem.LeftScale)
                                        sb_l0_2_g0: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_2_g0, target_memory=pl.Mem.RightScale)
                                        acc_g0 = pl.matmul_mx_acc(acc_g0, a_l0_2_g0, sa_l0_2_g0, b_l0_2_g0, sb_l0_2_g0)
                                        a_l0_3_g0: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 384, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_3_g0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_pre_1_g0, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_3_g0: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_3, target_memory=pl.Mem.LeftScale)
                                        sb_l0_3_g0: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_3_g0, target_memory=pl.Mem.RightScale)
                                        acc_g0 = pl.matmul_mx_acc(acc_g0, a_l0_3_g0, sa_l0_3_g0, b_l0_3_g0, sb_l0_3_g0)
                                        b_pre_0_g1: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[1], pl.Mem.Mat] = pl.load(weight, [0, col_g1], [256, 256], target_memory=pl.Mem.Mat)
                                        a_l0_4_g0: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 512, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_4_g0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_next_0_g0, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_4_g0: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_4, target_memory=pl.Mem.LeftScale)
                                        sb_l0_4_g0: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_4_g0, target_memory=pl.Mem.RightScale)
                                        acc_g0 = pl.matmul_mx_acc(acc_g0, a_l0_4_g0, sa_l0_4_g0, b_l0_4_g0, sb_l0_4_g0)
                                        a_l0_5_g0: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 640, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_5_g0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_next_0_g0, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_5_g0: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_5, target_memory=pl.Mem.LeftScale)
                                        sb_l0_5_g0: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_5_g0, target_memory=pl.Mem.RightScale)
                                        acc_g0 = pl.matmul_mx_acc(acc_g0, a_l0_5_g0, sa_l0_5_g0, b_l0_5_g0, sb_l0_5_g0)
                                        b_pre_1_g1: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[2], pl.Mem.Mat] = pl.load(weight, [256, col_g1], [256, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_0_g1 = pl.load(weight_scale, [0, col_g1], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_1_g1 = pl.load(weight_scale, [4, col_g1], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_2_g1 = pl.load(weight_scale, [8, col_g1], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_3_g1 = pl.load(weight_scale, [12, col_g1], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_4_g1 = pl.load(weight_scale, [16, col_g1], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_5_g1 = pl.load(weight_scale, [20, col_g1], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_6_g1 = pl.load(weight_scale, [24, col_g1], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_7_g1 = pl.load(weight_scale, [28, col_g1], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_8_g1 = pl.load(weight_scale, [32, col_g1], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_9_g1 = pl.load(weight_scale, [36, col_g1], [4, 256], target_memory=pl.Mem.Mat)
                                        a_l0_6_g0: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 768, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_6_g0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_next_1_g0, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_6_g0: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_6, target_memory=pl.Mem.LeftScale)
                                        sb_l0_6_g0: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_6_g0, target_memory=pl.Mem.RightScale)
                                        acc_g0 = pl.matmul_mx_acc(acc_g0, a_l0_6_g0, sa_l0_6_g0, b_l0_6_g0, sb_l0_6_g0)
                                        a_l0_7_g0: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 896, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_7_g0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_next_1_g0, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_7_g0: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_7, target_memory=pl.Mem.LeftScale)
                                        sb_l0_7_g0: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_7_g0, target_memory=pl.Mem.RightScale)
                                        acc_g0 = pl.matmul_mx_acc(acc_g0, a_l0_7_g0, sa_l0_7_g0, b_l0_7_g0, sb_l0_7_g0)
                                        b_next_0_g1: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[3], pl.Mem.Mat] = pl.load(weight, [512, col_g1], [256, 256], target_memory=pl.Mem.Mat)
                                        a_l0_8_g0: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 1024, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_8_g0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_next_2_g0, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_8_g0: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_8, target_memory=pl.Mem.LeftScale)
                                        sb_l0_8_g0: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_8_g0, target_memory=pl.Mem.RightScale)
                                        acc_g0 = pl.matmul_mx_acc(acc_g0, a_l0_8_g0, sa_l0_8_g0, b_l0_8_g0, sb_l0_8_g0)
                                        a_l0_9_g0: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 1152, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_9_g0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_next_2_g0, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_9_g0: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_9, target_memory=pl.Mem.LeftScale)
                                        sb_l0_9_g0: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_9_g0, target_memory=pl.Mem.RightScale)
                                        acc_g0 = pl.matmul_mx_acc(acc_g0, a_l0_9_g0, sa_l0_9_g0, b_l0_9_g0, sb_l0_9_g0)
                                        b_next_1_g1: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[0], pl.Mem.Mat] = pl.load(weight, [768, col_g1], [256, 256], target_memory=pl.Mem.Mat)
                                        valid_rows_g0 = pl.min(M_TILE, rows - row)
                                        if col_g0 % 512 == 256:
                                            tail_g0 = pl.tile.slice(acc_g0, [M_TILE, 64], [0, 192])
                                            prefix_g0 = pl.set_validshape(acc_g0, valid_rows_g0, 192)
                                            output = pl.store(prefix_g0, [row_start + row, col_g0], output)
                                            rounded_g0 = pl.cast(tail_g0, pl.BF16, mode="rint")
                                            source_g0 = rounded_g0
                                            flat_g0 = pl.reshape(source_g0, [M_TILE * 64])
                                            flat_out_decode_g0 = pl.tile.create([M_TILE * 64], dtype=pl.BF16, target_memory=pl.Mem.Vec)
                                            flat_out_decode_g0 = _q_b_rope_chunk(flat_g0, cos_resident, sin_resident, flat_out_decode_g0, 0, M_TILE, M_TILE)
                                            rotated_decode_g0 = pl.reshape(flat_out_decode_g0, [M_TILE, 64])
                                            valid_tail_g0 = pl.set_validshape(rotated_decode_g0, valid_rows_g0, 64)
                                            output = pl.store(valid_tail_g0, [row_start + row, col_g0 + 192], output)
                                        else:
                                            valid_acc_g0 = pl.set_validshape(acc_g0, valid_rows_g0, 256)
                                            output = pl.store(valid_acc_g0, [row_start + row, col_g0], output)
                                        a_l0_0_g1: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 0, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_0_g1: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_pre_0_g1, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_0_g1: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_0, target_memory=pl.Mem.LeftScale)
                                        sb_l0_0_g1: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_0_g1, target_memory=pl.Mem.RightScale)
                                        acc_g1: pl.Tile[[M_TILE, 256], pl.FP32, pl.MemRef("qb_acc", slots=2)[1], pl.Mem.Acc] = pl.matmul_mx(a_l0_0_g1, sa_l0_0_g1, b_l0_0_g1, sb_l0_0_g1)
                                        a_l0_1_g1: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 128, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_1_g1: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_pre_0_g1, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_1_g1: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_1, target_memory=pl.Mem.LeftScale)
                                        sb_l0_1_g1: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_1_g1, target_memory=pl.Mem.RightScale)
                                        acc_g1 = pl.matmul_mx_acc(acc_g1, a_l0_1_g1, sa_l0_1_g1, b_l0_1_g1, sb_l0_1_g1)
                                        b_next_2_g1: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[1], pl.Mem.Mat] = pl.load(weight, [1024, col_g1], [256, 256], target_memory=pl.Mem.Mat)
                                        a_l0_2_g1: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 256, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_2_g1: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_pre_1_g1, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_2_g1: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_2, target_memory=pl.Mem.LeftScale)
                                        sb_l0_2_g1: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_2_g1, target_memory=pl.Mem.RightScale)
                                        acc_g1 = pl.matmul_mx_acc(acc_g1, a_l0_2_g1, sa_l0_2_g1, b_l0_2_g1, sb_l0_2_g1)
                                        a_l0_3_g1: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 384, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_3_g1: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_pre_1_g1, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_3_g1: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_3, target_memory=pl.Mem.LeftScale)
                                        sb_l0_3_g1: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_3_g1, target_memory=pl.Mem.RightScale)
                                        acc_g1 = pl.matmul_mx_acc(acc_g1, a_l0_3_g1, sa_l0_3_g1, b_l0_3_g1, sb_l0_3_g1)
                                        b_pre_0_g2: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[2], pl.Mem.Mat] = pl.load(weight, [0, col_g2], [256, 256], target_memory=pl.Mem.Mat)
                                        a_l0_4_g1: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 512, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_4_g1: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_next_0_g1, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_4_g1: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_4, target_memory=pl.Mem.LeftScale)
                                        sb_l0_4_g1: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_4_g1, target_memory=pl.Mem.RightScale)
                                        acc_g1 = pl.matmul_mx_acc(acc_g1, a_l0_4_g1, sa_l0_4_g1, b_l0_4_g1, sb_l0_4_g1)
                                        a_l0_5_g1: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 640, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_5_g1: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_next_0_g1, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_5_g1: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_5, target_memory=pl.Mem.LeftScale)
                                        sb_l0_5_g1: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_5_g1, target_memory=pl.Mem.RightScale)
                                        acc_g1 = pl.matmul_mx_acc(acc_g1, a_l0_5_g1, sa_l0_5_g1, b_l0_5_g1, sb_l0_5_g1)
                                        b_pre_1_g2: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[3], pl.Mem.Mat] = pl.load(weight, [256, col_g2], [256, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_0_g2 = pl.load(weight_scale, [0, col_g2], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_1_g2 = pl.load(weight_scale, [4, col_g2], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_2_g2 = pl.load(weight_scale, [8, col_g2], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_3_g2 = pl.load(weight_scale, [12, col_g2], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_4_g2 = pl.load(weight_scale, [16, col_g2], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_5_g2 = pl.load(weight_scale, [20, col_g2], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_6_g2 = pl.load(weight_scale, [24, col_g2], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_7_g2 = pl.load(weight_scale, [28, col_g2], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_8_g2 = pl.load(weight_scale, [32, col_g2], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_9_g2 = pl.load(weight_scale, [36, col_g2], [4, 256], target_memory=pl.Mem.Mat)
                                        a_l0_6_g1: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 768, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_6_g1: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_next_1_g1, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_6_g1: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_6, target_memory=pl.Mem.LeftScale)
                                        sb_l0_6_g1: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_6_g1, target_memory=pl.Mem.RightScale)
                                        acc_g1 = pl.matmul_mx_acc(acc_g1, a_l0_6_g1, sa_l0_6_g1, b_l0_6_g1, sb_l0_6_g1)
                                        a_l0_7_g1: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 896, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_7_g1: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_next_1_g1, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_7_g1: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_7, target_memory=pl.Mem.LeftScale)
                                        sb_l0_7_g1: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_7_g1, target_memory=pl.Mem.RightScale)
                                        acc_g1 = pl.matmul_mx_acc(acc_g1, a_l0_7_g1, sa_l0_7_g1, b_l0_7_g1, sb_l0_7_g1)
                                        b_next_0_g2: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[0], pl.Mem.Mat] = pl.load(weight, [512, col_g2], [256, 256], target_memory=pl.Mem.Mat)
                                        a_l0_8_g1: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 1024, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_8_g1: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_next_2_g1, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_8_g1: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_8, target_memory=pl.Mem.LeftScale)
                                        sb_l0_8_g1: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_8_g1, target_memory=pl.Mem.RightScale)
                                        acc_g1 = pl.matmul_mx_acc(acc_g1, a_l0_8_g1, sa_l0_8_g1, b_l0_8_g1, sb_l0_8_g1)
                                        a_l0_9_g1: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 1152, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_9_g1: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_next_2_g1, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_9_g1: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_9, target_memory=pl.Mem.LeftScale)
                                        sb_l0_9_g1: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_9_g1, target_memory=pl.Mem.RightScale)
                                        acc_g1 = pl.matmul_mx_acc(acc_g1, a_l0_9_g1, sa_l0_9_g1, b_l0_9_g1, sb_l0_9_g1)
                                        b_next_1_g2: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[1], pl.Mem.Mat] = pl.load(weight, [768, col_g2], [256, 256], target_memory=pl.Mem.Mat)
                                        valid_rows_g1 = pl.min(M_TILE, rows - row)
                                        if col_g1 % 512 == 256:
                                            tail_g1 = pl.tile.slice(acc_g1, [M_TILE, 64], [0, 192])
                                            prefix_g1 = pl.set_validshape(acc_g1, valid_rows_g1, 192)
                                            output = pl.store(prefix_g1, [row_start + row, col_g1], output)
                                            rounded_g1 = pl.cast(tail_g1, pl.BF16, mode="rint")
                                            source_g1 = rounded_g1
                                            flat_g1 = pl.reshape(source_g1, [M_TILE * 64])
                                            flat_out_decode_g1 = pl.tile.create([M_TILE * 64], dtype=pl.BF16, target_memory=pl.Mem.Vec)
                                            flat_out_decode_g1 = _q_b_rope_chunk(flat_g1, cos_resident, sin_resident, flat_out_decode_g1, 0, M_TILE, M_TILE)
                                            rotated_decode_g1 = pl.reshape(flat_out_decode_g1, [M_TILE, 64])
                                            valid_tail_g1 = pl.set_validshape(rotated_decode_g1, valid_rows_g1, 64)
                                            output = pl.store(valid_tail_g1, [row_start + row, col_g1 + 192], output)
                                        else:
                                            valid_acc_g1 = pl.set_validshape(acc_g1, valid_rows_g1, 256)
                                            output = pl.store(valid_acc_g1, [row_start + row, col_g1], output)
                                        a_l0_0_g2: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 0, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_0_g2: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_pre_0_g2, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_0_g2: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_0, target_memory=pl.Mem.LeftScale)
                                        sb_l0_0_g2: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_0_g2, target_memory=pl.Mem.RightScale)
                                        acc_g2: pl.Tile[[M_TILE, 256], pl.FP32, pl.MemRef("qb_acc", slots=2)[0], pl.Mem.Acc] = pl.matmul_mx(a_l0_0_g2, sa_l0_0_g2, b_l0_0_g2, sb_l0_0_g2)
                                        a_l0_1_g2: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 128, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_1_g2: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_pre_0_g2, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_1_g2: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_1, target_memory=pl.Mem.LeftScale)
                                        sb_l0_1_g2: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_1_g2, target_memory=pl.Mem.RightScale)
                                        acc_g2 = pl.matmul_mx_acc(acc_g2, a_l0_1_g2, sa_l0_1_g2, b_l0_1_g2, sb_l0_1_g2)
                                        b_next_2_g2: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[2], pl.Mem.Mat] = pl.load(weight, [1024, col_g2], [256, 256], target_memory=pl.Mem.Mat)
                                        a_l0_2_g2: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 256, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_2_g2: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_pre_1_g2, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_2_g2: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_2, target_memory=pl.Mem.LeftScale)
                                        sb_l0_2_g2: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_2_g2, target_memory=pl.Mem.RightScale)
                                        acc_g2 = pl.matmul_mx_acc(acc_g2, a_l0_2_g2, sa_l0_2_g2, b_l0_2_g2, sb_l0_2_g2)
                                        a_l0_3_g2: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 384, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_3_g2: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_pre_1_g2, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_3_g2: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_3, target_memory=pl.Mem.LeftScale)
                                        sb_l0_3_g2: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_3_g2, target_memory=pl.Mem.RightScale)
                                        acc_g2 = pl.matmul_mx_acc(acc_g2, a_l0_3_g2, sa_l0_3_g2, b_l0_3_g2, sb_l0_3_g2)
                                        b_pre_0_g3: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[3], pl.Mem.Mat] = pl.load(weight, [0, col_g3], [256, 256], target_memory=pl.Mem.Mat)
                                        a_l0_4_g2: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 512, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_4_g2: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_next_0_g2, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_4_g2: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_4, target_memory=pl.Mem.LeftScale)
                                        sb_l0_4_g2: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_4_g2, target_memory=pl.Mem.RightScale)
                                        acc_g2 = pl.matmul_mx_acc(acc_g2, a_l0_4_g2, sa_l0_4_g2, b_l0_4_g2, sb_l0_4_g2)
                                        a_l0_5_g2: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 640, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_5_g2: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_next_0_g2, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_5_g2: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_5, target_memory=pl.Mem.LeftScale)
                                        sb_l0_5_g2: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_5_g2, target_memory=pl.Mem.RightScale)
                                        acc_g2 = pl.matmul_mx_acc(acc_g2, a_l0_5_g2, sa_l0_5_g2, b_l0_5_g2, sb_l0_5_g2)
                                        b_pre_1_g3: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[0], pl.Mem.Mat] = pl.load(weight, [256, col_g3], [256, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_0_g3 = pl.load(weight_scale, [0, col_g3], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_1_g3 = pl.load(weight_scale, [4, col_g3], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_2_g3 = pl.load(weight_scale, [8, col_g3], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_3_g3 = pl.load(weight_scale, [12, col_g3], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_4_g3 = pl.load(weight_scale, [16, col_g3], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_5_g3 = pl.load(weight_scale, [20, col_g3], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_6_g3 = pl.load(weight_scale, [24, col_g3], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_7_g3 = pl.load(weight_scale, [28, col_g3], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_8_g3 = pl.load(weight_scale, [32, col_g3], [4, 256], target_memory=pl.Mem.Mat)
                                        sb_cache_9_g3 = pl.load(weight_scale, [36, col_g3], [4, 256], target_memory=pl.Mem.Mat)
                                        a_l0_6_g2: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 768, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_6_g2: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_next_1_g2, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_6_g2: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_6, target_memory=pl.Mem.LeftScale)
                                        sb_l0_6_g2: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_6_g2, target_memory=pl.Mem.RightScale)
                                        acc_g2 = pl.matmul_mx_acc(acc_g2, a_l0_6_g2, sa_l0_6_g2, b_l0_6_g2, sb_l0_6_g2)
                                        a_l0_7_g2: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 896, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_7_g2: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_next_1_g2, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_7_g2: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_7, target_memory=pl.Mem.LeftScale)
                                        sb_l0_7_g2: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_7_g2, target_memory=pl.Mem.RightScale)
                                        acc_g2 = pl.matmul_mx_acc(acc_g2, a_l0_7_g2, sa_l0_7_g2, b_l0_7_g2, sb_l0_7_g2)
                                        b_next_0_g3: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[1], pl.Mem.Mat] = pl.load(weight, [512, col_g3], [256, 256], target_memory=pl.Mem.Mat)
                                        a_l0_8_g2: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 1024, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_8_g2: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_next_2_g2, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_8_g2: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_8, target_memory=pl.Mem.LeftScale)
                                        sb_l0_8_g2: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_8_g2, target_memory=pl.Mem.RightScale)
                                        acc_g2 = pl.matmul_mx_acc(acc_g2, a_l0_8_g2, sa_l0_8_g2, b_l0_8_g2, sb_l0_8_g2)
                                        a_l0_9_g2: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 1152, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_9_g2: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_next_2_g2, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_9_g2: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_9, target_memory=pl.Mem.LeftScale)
                                        sb_l0_9_g2: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_9_g2, target_memory=pl.Mem.RightScale)
                                        acc_g2 = pl.matmul_mx_acc(acc_g2, a_l0_9_g2, sa_l0_9_g2, b_l0_9_g2, sb_l0_9_g2)
                                        b_next_1_g3: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[2], pl.Mem.Mat] = pl.load(weight, [768, col_g3], [256, 256], target_memory=pl.Mem.Mat)
                                        valid_rows_g2 = pl.min(M_TILE, rows - row)
                                        if col_g2 % 512 == 256:
                                            tail_g2 = pl.tile.slice(acc_g2, [M_TILE, 64], [0, 192])
                                            prefix_g2 = pl.set_validshape(acc_g2, valid_rows_g2, 192)
                                            output = pl.store(prefix_g2, [row_start + row, col_g2], output)
                                            rounded_g2 = pl.cast(tail_g2, pl.BF16, mode="rint")
                                            source_g2 = rounded_g2
                                            flat_g2 = pl.reshape(source_g2, [M_TILE * 64])
                                            flat_out_decode_g2 = pl.tile.create([M_TILE * 64], dtype=pl.BF16, target_memory=pl.Mem.Vec)
                                            flat_out_decode_g2 = _q_b_rope_chunk(flat_g2, cos_resident, sin_resident, flat_out_decode_g2, 0, M_TILE, M_TILE)
                                            rotated_decode_g2 = pl.reshape(flat_out_decode_g2, [M_TILE, 64])
                                            valid_tail_g2 = pl.set_validshape(rotated_decode_g2, valid_rows_g2, 64)
                                            output = pl.store(valid_tail_g2, [row_start + row, col_g2 + 192], output)
                                        else:
                                            valid_acc_g2 = pl.set_validshape(acc_g2, valid_rows_g2, 256)
                                            output = pl.store(valid_acc_g2, [row_start + row, col_g2], output)
                                        a_l0_0_g3: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 0, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_0_g3: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_pre_0_g3, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_0_g3: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_0, target_memory=pl.Mem.LeftScale)
                                        sb_l0_0_g3: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_0_g3, target_memory=pl.Mem.RightScale)
                                        acc_g3: pl.Tile[[M_TILE, 256], pl.FP32, pl.MemRef("qb_acc", slots=2)[1], pl.Mem.Acc] = pl.matmul_mx(a_l0_0_g3, sa_l0_0_g3, b_l0_0_g3, sb_l0_0_g3)
                                        a_l0_1_g3: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 128, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_1_g3: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_pre_0_g3, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_1_g3: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_1, target_memory=pl.Mem.LeftScale)
                                        sb_l0_1_g3: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_1_g3, target_memory=pl.Mem.RightScale)
                                        acc_g3 = pl.matmul_mx_acc(acc_g3, a_l0_1_g3, sa_l0_1_g3, b_l0_1_g3, sb_l0_1_g3)
                                        b_next_2_g3: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[3], pl.Mem.Mat] = pl.load(weight, [1024, col_g3], [256, 256], target_memory=pl.Mem.Mat)
                                        a_l0_2_g3: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 256, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_2_g3: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_pre_1_g3, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_2_g3: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_2, target_memory=pl.Mem.LeftScale)
                                        sb_l0_2_g3: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_2_g3, target_memory=pl.Mem.RightScale)
                                        acc_g3 = pl.matmul_mx_acc(acc_g3, a_l0_2_g3, sa_l0_2_g3, b_l0_2_g3, sb_l0_2_g3)
                                        a_l0_3_g3: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 384, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_3_g3: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_pre_1_g3, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_3_g3: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_3, target_memory=pl.Mem.LeftScale)
                                        sb_l0_3_g3: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_3_g3, target_memory=pl.Mem.RightScale)
                                        acc_g3 = pl.matmul_mx_acc(acc_g3, a_l0_3_g3, sa_l0_3_g3, b_l0_3_g3, sb_l0_3_g3)
                                        a_l0_4_g3: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 512, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_4_g3: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_next_0_g3, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_4_g3: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_4, target_memory=pl.Mem.LeftScale)
                                        sb_l0_4_g3: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_4_g3, target_memory=pl.Mem.RightScale)
                                        acc_g3 = pl.matmul_mx_acc(acc_g3, a_l0_4_g3, sa_l0_4_g3, b_l0_4_g3, sb_l0_4_g3)
                                        a_l0_5_g3: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 640, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_5_g3: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_next_0_g3, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_5_g3: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_5, target_memory=pl.Mem.LeftScale)
                                        sb_l0_5_g3: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_5_g3, target_memory=pl.Mem.RightScale)
                                        acc_g3 = pl.matmul_mx_acc(acc_g3, a_l0_5_g3, sa_l0_5_g3, b_l0_5_g3, sb_l0_5_g3)
                                        a_l0_6_g3: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 768, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_6_g3: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_next_1_g3, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_6_g3: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_6, target_memory=pl.Mem.LeftScale)
                                        sb_l0_6_g3: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_6_g3, target_memory=pl.Mem.RightScale)
                                        acc_g3 = pl.matmul_mx_acc(acc_g3, a_l0_6_g3, sa_l0_6_g3, b_l0_6_g3, sb_l0_6_g3)
                                        a_l0_7_g3: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 896, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_7_g3: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_next_1_g3, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_7_g3: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_7, target_memory=pl.Mem.LeftScale)
                                        sb_l0_7_g3: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_7_g3, target_memory=pl.Mem.RightScale)
                                        acc_g3 = pl.matmul_mx_acc(acc_g3, a_l0_7_g3, sa_l0_7_g3, b_l0_7_g3, sb_l0_7_g3)
                                        a_l0_8_g3: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 1024, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_8_g3: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_next_2_g3, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_8_g3: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_8, target_memory=pl.Mem.LeftScale)
                                        sb_l0_8_g3: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_8_g3, target_memory=pl.Mem.RightScale)
                                        acc_g3 = pl.matmul_mx_acc(acc_g3, a_l0_8_g3, sa_l0_8_g3, b_l0_8_g3, sb_l0_8_g3)
                                        a_l0_9_g3: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 1152, [M_TILE, 128], target_memory=pl.Mem.Left)
                                        b_l0_9_g3: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_next_2_g3, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                        sa_l0_9_g3: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_9, target_memory=pl.Mem.LeftScale)
                                        sb_l0_9_g3: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_9_g3, target_memory=pl.Mem.RightScale)
                                        acc_g3 = pl.matmul_mx_acc(acc_g3, a_l0_9_g3, sa_l0_9_g3, b_l0_9_g3, sb_l0_9_g3)
                                        valid_rows_g3 = pl.min(M_TILE, rows - row)
                                        if col_g3 % 512 == 256:
                                            tail_g3 = pl.tile.slice(acc_g3, [M_TILE, 64], [0, 192])
                                            prefix_g3 = pl.set_validshape(acc_g3, valid_rows_g3, 192)
                                            output = pl.store(prefix_g3, [row_start + row, col_g3], output)
                                            rounded_g3 = pl.cast(tail_g3, pl.BF16, mode="rint")
                                            source_g3 = rounded_g3
                                            flat_g3 = pl.reshape(source_g3, [M_TILE * 64])
                                            flat_out_decode_g3 = pl.tile.create([M_TILE * 64], dtype=pl.BF16, target_memory=pl.Mem.Vec)
                                            flat_out_decode_g3 = _q_b_rope_chunk(flat_g3, cos_resident, sin_resident, flat_out_decode_g3, 0, M_TILE, M_TILE)
                                            rotated_decode_g3 = pl.reshape(flat_out_decode_g3, [M_TILE, 64])
                                            valid_tail_g3 = pl.set_validshape(rotated_decode_g3, valid_rows_g3, 64)
                                            output = pl.store(valid_tail_g3, [row_start + row, col_g3 + 192], output)
                                        else:
                                            valid_acc_g3 = pl.set_validshape(acc_g3, valid_rows_g3, 256)
                                            output = pl.store(valid_acc_g3, [row_start + row, col_g3], output)
                                    else:
                                        for remainder in pl.range(4):
                                            if wave * 4 + remainder < waves:
                                                col_tail4 = (n_lane + (wave * 4 + remainder) * n_stride) * 256
                                                sb_cache_0_tail4 = pl.load(weight_scale, [0, col_tail4], [4, 256], target_memory=pl.Mem.Mat)
                                                sb_cache_1_tail4 = pl.load(weight_scale, [4, col_tail4], [4, 256], target_memory=pl.Mem.Mat)
                                                sb_cache_2_tail4 = pl.load(weight_scale, [8, col_tail4], [4, 256], target_memory=pl.Mem.Mat)
                                                sb_cache_3_tail4 = pl.load(weight_scale, [12, col_tail4], [4, 256], target_memory=pl.Mem.Mat)
                                                sb_cache_4_tail4 = pl.load(weight_scale, [16, col_tail4], [4, 256], target_memory=pl.Mem.Mat)
                                                sb_cache_5_tail4 = pl.load(weight_scale, [20, col_tail4], [4, 256], target_memory=pl.Mem.Mat)
                                                sb_cache_6_tail4 = pl.load(weight_scale, [24, col_tail4], [4, 256], target_memory=pl.Mem.Mat)
                                                sb_cache_7_tail4 = pl.load(weight_scale, [28, col_tail4], [4, 256], target_memory=pl.Mem.Mat)
                                                sb_cache_8_tail4 = pl.load(weight_scale, [32, col_tail4], [4, 256], target_memory=pl.Mem.Mat)
                                                sb_cache_9_tail4 = pl.load(weight_scale, [36, col_tail4], [4, 256], target_memory=pl.Mem.Mat)
                                                b_pre_0_tail4: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[0], pl.Mem.Mat] = pl.load(weight, [0, col_tail4], [256, 256], target_memory=pl.Mem.Mat)
                                                b_pre_1_tail4: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[1], pl.Mem.Mat] = pl.load(weight, [256, col_tail4], [256, 256], target_memory=pl.Mem.Mat)
                                                a_l0_0_tail4: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 0, [M_TILE, 128], target_memory=pl.Mem.Left)
                                                b_l0_0_tail4: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_pre_0_tail4, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                                sa_l0_0_tail4: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_0, target_memory=pl.Mem.LeftScale)
                                                sb_l0_0_tail4: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_0_tail4, target_memory=pl.Mem.RightScale)
                                                acc_tail4 = pl.matmul_mx(a_l0_0_tail4, sa_l0_0_tail4, b_l0_0_tail4, sb_l0_0_tail4)
                                                a_l0_1_tail4: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 128, [M_TILE, 128], target_memory=pl.Mem.Left)
                                                b_l0_1_tail4: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_pre_0_tail4, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                                sa_l0_1_tail4: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_1, target_memory=pl.Mem.LeftScale)
                                                sb_l0_1_tail4: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_1_tail4, target_memory=pl.Mem.RightScale)
                                                acc_tail4 = pl.matmul_mx_acc(acc_tail4, a_l0_1_tail4, sa_l0_1_tail4, b_l0_1_tail4, sb_l0_1_tail4)
                                                b_next_0_tail4: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[0], pl.Mem.Mat] = pl.load(weight, [512, col_tail4], [256, 256], target_memory=pl.Mem.Mat)
                                                a_l0_2_tail4: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 256, [M_TILE, 128], target_memory=pl.Mem.Left)
                                                b_l0_2_tail4: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_pre_1_tail4, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                                sa_l0_2_tail4: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_2, target_memory=pl.Mem.LeftScale)
                                                sb_l0_2_tail4: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_2_tail4, target_memory=pl.Mem.RightScale)
                                                acc_tail4 = pl.matmul_mx_acc(acc_tail4, a_l0_2_tail4, sa_l0_2_tail4, b_l0_2_tail4, sb_l0_2_tail4)
                                                a_l0_3_tail4: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 384, [M_TILE, 128], target_memory=pl.Mem.Left)
                                                b_l0_3_tail4: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_pre_1_tail4, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                                sa_l0_3_tail4: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_3, target_memory=pl.Mem.LeftScale)
                                                sb_l0_3_tail4: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_3_tail4, target_memory=pl.Mem.RightScale)
                                                acc_tail4 = pl.matmul_mx_acc(acc_tail4, a_l0_3_tail4, sa_l0_3_tail4, b_l0_3_tail4, sb_l0_3_tail4)
                                                b_next_1_tail4: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[1], pl.Mem.Mat] = pl.load(weight, [768, col_tail4], [256, 256], target_memory=pl.Mem.Mat)
                                                a_l0_4_tail4: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 512, [M_TILE, 128], target_memory=pl.Mem.Left)
                                                b_l0_4_tail4: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_next_0_tail4, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                                sa_l0_4_tail4: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_4, target_memory=pl.Mem.LeftScale)
                                                sb_l0_4_tail4: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_4_tail4, target_memory=pl.Mem.RightScale)
                                                acc_tail4 = pl.matmul_mx_acc(acc_tail4, a_l0_4_tail4, sa_l0_4_tail4, b_l0_4_tail4, sb_l0_4_tail4)
                                                a_l0_5_tail4: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 640, [M_TILE, 128], target_memory=pl.Mem.Left)
                                                b_l0_5_tail4: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_next_0_tail4, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                                sa_l0_5_tail4: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_5, target_memory=pl.Mem.LeftScale)
                                                sb_l0_5_tail4: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_5_tail4, target_memory=pl.Mem.RightScale)
                                                acc_tail4 = pl.matmul_mx_acc(acc_tail4, a_l0_5_tail4, sa_l0_5_tail4, b_l0_5_tail4, sb_l0_5_tail4)
                                                b_next_2_tail4: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=4)[0], pl.Mem.Mat] = pl.load(weight, [1024, col_tail4], [256, 256], target_memory=pl.Mem.Mat)
                                                a_l0_6_tail4: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 768, [M_TILE, 128], target_memory=pl.Mem.Left)
                                                b_l0_6_tail4: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_next_1_tail4, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                                sa_l0_6_tail4: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_6, target_memory=pl.Mem.LeftScale)
                                                sb_l0_6_tail4: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_6_tail4, target_memory=pl.Mem.RightScale)
                                                acc_tail4 = pl.matmul_mx_acc(acc_tail4, a_l0_6_tail4, sa_l0_6_tail4, b_l0_6_tail4, sb_l0_6_tail4)
                                                a_l0_7_tail4: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 896, [M_TILE, 128], target_memory=pl.Mem.Left)
                                                b_l0_7_tail4: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_next_1_tail4, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                                sa_l0_7_tail4: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_7, target_memory=pl.Mem.LeftScale)
                                                sb_l0_7_tail4: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_7_tail4, target_memory=pl.Mem.RightScale)
                                                acc_tail4 = pl.matmul_mx_acc(acc_tail4, a_l0_7_tail4, sa_l0_7_tail4, b_l0_7_tail4, sb_l0_7_tail4)
                                                a_l0_8_tail4: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[2], pl.Mem.Left] = pl.tile.extract(full_a, 0, 1024, [M_TILE, 128], target_memory=pl.Mem.Left)
                                                b_l0_8_tail4: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_next_2_tail4, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                                sa_l0_8_tail4: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[4], pl.Mem.LeftScale] = pl.tile.move(sa_cache_8, target_memory=pl.Mem.LeftScale)
                                                sb_l0_8_tail4: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_8_tail4, target_memory=pl.Mem.RightScale)
                                                acc_tail4 = pl.matmul_mx_acc(acc_tail4, a_l0_8_tail4, sa_l0_8_tail4, b_l0_8_tail4, sb_l0_8_tail4)
                                                a_l0_9_tail4: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=6)[5], pl.Mem.Left] = pl.tile.extract(full_a, 0, 1152, [M_TILE, 128], target_memory=pl.Mem.Left)
                                                b_l0_9_tail4: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_next_2_tail4, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                                sa_l0_9_tail4: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=12)[10], pl.Mem.LeftScale] = pl.tile.move(sa_cache_9, target_memory=pl.Mem.LeftScale)
                                                sb_l0_9_tail4: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_9_tail4, target_memory=pl.Mem.RightScale)
                                                acc_tail4 = pl.matmul_mx_acc(acc_tail4, a_l0_9_tail4, sa_l0_9_tail4, b_l0_9_tail4, sb_l0_9_tail4)
                                                valid_rows_tail4 = pl.min(M_TILE, rows - row)
                                                if col_tail4 % 512 == 256:
                                                    tail_tail4 = pl.tile.slice(acc_tail4, [M_TILE, 64], [0, 192])
                                                    prefix_tail4 = pl.set_validshape(acc_tail4, valid_rows_tail4, 192)
                                                    output = pl.store(prefix_tail4, [row_start + row, col_tail4], output)
                                                    rounded_tail4 = pl.cast(tail_tail4, pl.BF16, mode="rint")
                                                    source_tail4 = rounded_tail4
                                                    flat_tail4 = pl.reshape(source_tail4, [M_TILE * 64])
                                                    flat_out_decode_tail4 = pl.tile.create([M_TILE * 64], dtype=pl.BF16, target_memory=pl.Mem.Vec)
                                                    flat_out_decode_tail4 = _q_b_rope_chunk(flat_tail4, cos_resident, sin_resident, flat_out_decode_tail4, 0, M_TILE, M_TILE)
                                                    rotated_decode_tail4 = pl.reshape(flat_out_decode_tail4, [M_TILE, 64])
                                                    valid_tail_tail4 = pl.set_validshape(rotated_decode_tail4, valid_rows_tail4, 64)
                                                    output = pl.store(valid_tail_tail4, [row_start + row, col_tail4 + 192], output)
                                                else:
                                                    valid_acc_tail4 = pl.set_validshape(acc_tail4, valid_rows_tail4, 256)
                                                    output = pl.store(valid_acc_tail4, [row_start + row, col_tail4], output)
                    else:
                        for nt in pl.range(width // 256):
                            if nt % lanes == n_lane:
                                col = nt * 256
                                sb_cache_0 = pl.load(weight_scale, [0, col], [4, 256], target_memory=pl.Mem.Mat)
                                sb_cache_1 = pl.load(weight_scale, [4, col], [4, 256], target_memory=pl.Mem.Mat)
                                sb_cache_2 = pl.load(weight_scale, [8, col], [4, 256], target_memory=pl.Mem.Mat)
                                sb_cache_3 = pl.load(weight_scale, [12, col], [4, 256], target_memory=pl.Mem.Mat)
                                sb_cache_4 = pl.load(weight_scale, [16, col], [4, 256], target_memory=pl.Mem.Mat)
                                sb_cache_5 = pl.load(weight_scale, [20, col], [4, 256], target_memory=pl.Mem.Mat)
                                sb_cache_6 = pl.load(weight_scale, [24, col], [4, 256], target_memory=pl.Mem.Mat)
                                sb_cache_7 = pl.load(weight_scale, [28, col], [4, 256], target_memory=pl.Mem.Mat)
                                sb_cache_8 = pl.load(weight_scale, [32, col], [4, 256], target_memory=pl.Mem.Mat)
                                sb_cache_9 = pl.load(weight_scale, [36, col], [4, 256], target_memory=pl.Mem.Mat)
                                b_pre_0: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=2)[0], pl.Mem.Mat] = pl.load(weight, [0, col], [256, 256], target_memory=pl.Mem.Mat)
                                b_pre_1: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=2)[1], pl.Mem.Mat] = pl.load(weight, [256, col], [256, 256], target_memory=pl.Mem.Mat)
                                a_l0_0: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=2)[0], pl.Mem.Left] = pl.tile.extract(full_a, 0, 0, [M_TILE, 128], target_memory=pl.Mem.Left)
                                b_l0_0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_pre_0, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                sa_l0_0: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=4)[0], pl.Mem.LeftScale] = pl.tile.move(sa_cache_0, target_memory=pl.Mem.LeftScale)
                                sb_l0_0: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_0, target_memory=pl.Mem.RightScale)
                                acc = pl.matmul_mx(a_l0_0, sa_l0_0, b_l0_0, sb_l0_0)
                                a_l0_1: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=2)[1], pl.Mem.Left] = pl.tile.extract(full_a, 0, 128, [M_TILE, 128], target_memory=pl.Mem.Left)
                                b_l0_1: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_pre_0, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                sa_l0_1: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=4)[2], pl.Mem.LeftScale] = pl.tile.move(sa_cache_1, target_memory=pl.Mem.LeftScale)
                                sb_l0_1: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_1, target_memory=pl.Mem.RightScale)
                                acc = pl.matmul_mx_acc(acc, a_l0_1, sa_l0_1, b_l0_1, sb_l0_1)
                                b_next_0: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=2)[0], pl.Mem.Mat] = pl.load(weight, [512, col], [256, 256], target_memory=pl.Mem.Mat)
                                a_l0_2: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=2)[0], pl.Mem.Left] = pl.tile.extract(full_a, 0, 256, [M_TILE, 128], target_memory=pl.Mem.Left)
                                b_l0_2: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_pre_1, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                sa_l0_2: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=4)[0], pl.Mem.LeftScale] = pl.tile.move(sa_cache_2, target_memory=pl.Mem.LeftScale)
                                sb_l0_2: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_2, target_memory=pl.Mem.RightScale)
                                acc = pl.matmul_mx_acc(acc, a_l0_2, sa_l0_2, b_l0_2, sb_l0_2)
                                a_l0_3: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=2)[1], pl.Mem.Left] = pl.tile.extract(full_a, 0, 384, [M_TILE, 128], target_memory=pl.Mem.Left)
                                b_l0_3: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_pre_1, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                sa_l0_3: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=4)[2], pl.Mem.LeftScale] = pl.tile.move(sa_cache_3, target_memory=pl.Mem.LeftScale)
                                sb_l0_3: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_3, target_memory=pl.Mem.RightScale)
                                acc = pl.matmul_mx_acc(acc, a_l0_3, sa_l0_3, b_l0_3, sb_l0_3)
                                b_next_1: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=2)[1], pl.Mem.Mat] = pl.load(weight, [768, col], [256, 256], target_memory=pl.Mem.Mat)
                                a_l0_4: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=2)[0], pl.Mem.Left] = pl.tile.extract(full_a, 0, 512, [M_TILE, 128], target_memory=pl.Mem.Left)
                                b_l0_4: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_next_0, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                sa_l0_4: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=4)[0], pl.Mem.LeftScale] = pl.tile.move(sa_cache_4, target_memory=pl.Mem.LeftScale)
                                sb_l0_4: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_4, target_memory=pl.Mem.RightScale)
                                acc = pl.matmul_mx_acc(acc, a_l0_4, sa_l0_4, b_l0_4, sb_l0_4)
                                a_l0_5: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=2)[1], pl.Mem.Left] = pl.tile.extract(full_a, 0, 640, [M_TILE, 128], target_memory=pl.Mem.Left)
                                b_l0_5: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_next_0, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                sa_l0_5: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=4)[2], pl.Mem.LeftScale] = pl.tile.move(sa_cache_5, target_memory=pl.Mem.LeftScale)
                                sb_l0_5: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_5, target_memory=pl.Mem.RightScale)
                                acc = pl.matmul_mx_acc(acc, a_l0_5, sa_l0_5, b_l0_5, sb_l0_5)
                                b_next_2: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef("b_l1", slots=2)[0], pl.Mem.Mat] = pl.load(weight, [1024, col], [256, 256], target_memory=pl.Mem.Mat)
                                a_l0_6: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=2)[0], pl.Mem.Left] = pl.tile.extract(full_a, 0, 768, [M_TILE, 128], target_memory=pl.Mem.Left)
                                b_l0_6: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_next_1, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                sa_l0_6: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=4)[0], pl.Mem.LeftScale] = pl.tile.move(sa_cache_6, target_memory=pl.Mem.LeftScale)
                                sb_l0_6: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_6, target_memory=pl.Mem.RightScale)
                                acc = pl.matmul_mx_acc(acc, a_l0_6, sa_l0_6, b_l0_6, sb_l0_6)
                                a_l0_7: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=2)[1], pl.Mem.Left] = pl.tile.extract(full_a, 0, 896, [M_TILE, 128], target_memory=pl.Mem.Left)
                                b_l0_7: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_next_1, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                sa_l0_7: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=4)[2], pl.Mem.LeftScale] = pl.tile.move(sa_cache_7, target_memory=pl.Mem.LeftScale)
                                sb_l0_7: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_7, target_memory=pl.Mem.RightScale)
                                acc = pl.matmul_mx_acc(acc, a_l0_7, sa_l0_7, b_l0_7, sb_l0_7)
                                a_l0_8: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=2)[0], pl.Mem.Left] = pl.tile.extract(full_a, 0, 1024, [M_TILE, 128], target_memory=pl.Mem.Left)
                                b_l0_8: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b_next_2, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                                sa_l0_8: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=4)[0], pl.Mem.LeftScale] = pl.tile.move(sa_cache_8, target_memory=pl.Mem.LeftScale)
                                sb_l0_8: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.move(sb_cache_8, target_memory=pl.Mem.RightScale)
                                acc = pl.matmul_mx_acc(acc, a_l0_8, sa_l0_8, b_l0_8, sb_l0_8)
                                a_l0_9: pl.Tile[[M_TILE, 128], pl.FP8E4M3FN, pl.MemRef("a_l0", slots=2)[1], pl.Mem.Left] = pl.tile.extract(full_a, 0, 1152, [M_TILE, 128], target_memory=pl.Mem.Left)
                                b_l0_9: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("b_l0", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b_next_2, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                                sa_l0_9: pl.Tile[[M_TILE, 4], pl.FP8E8M0, pl.MemRef("sa_l0", slots=4)[2], pl.Mem.LeftScale] = pl.tile.move(sa_cache_9, target_memory=pl.Mem.LeftScale)
                                sb_l0_9: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("sb_l0", slots=4)[2], pl.Mem.RightScale] = pl.tile.move(sb_cache_9, target_memory=pl.Mem.RightScale)
                                acc = pl.matmul_mx_acc(acc, a_l0_9, sa_l0_9, b_l0_9, sb_l0_9)
                                valid_rows = pl.min(M_TILE, rows - row)
                                if col % 512 == 256:
                                    tail = pl.tile.slice(acc, [M_TILE, 64], [0, 192])
                                    prefix = pl.set_validshape(acc, valid_rows, 192)
                                    output = pl.store(prefix, [row_start + row, col], output)
                                    rounded = pl.cast(tail, pl.BF16, mode="rint")
                                    source = pl.tile.move(rounded, target_memory=pl.Mem.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)
                                    flat = pl.reshape(source, [M_TILE * 64])
                                    cv = pl.load(cosine, [row_start + row, 0], [M_TILE, 32], valid_shape=[valid_rows, 32], target_memory=pl.Mem.Vec)
                                    sv = pl.load(sine, [row_start + row, 0], [M_TILE, 32], valid_shape=[valid_rows, 32], target_memory=pl.Mem.Vec)
                                    cv = pl.tile.fillpad_inplace(cv)
                                    sv = pl.tile.fillpad_inplace(sv)
                                    flat_out = pl.tile.create([8192], dtype=pl.BF16, target_memory=pl.Mem.Vec)
                                    flat_out = _q_b_rope_chunk(flat, cv, sv, flat_out, 0, 128, M_TILE)
                                    rotated_tile_h0 = pl.reshape(flat_out, [128, 64])
                                    part_rows_h0 = pl.min(128, pl.max(valid_rows, 0))
                                    valid_part_h0 = pl.set_validshape(rotated_tile_h0, part_rows_h0, 64)
                                    output = pl.store(valid_part_h0, [row_start + row, col + 192], output)
                                    flat_out = _q_b_rope_chunk(flat, cv, sv, flat_out, 128, 128, M_TILE)
                                    rotated_tile_h1 = pl.reshape(flat_out, [128, 64])
                                    part_rows_h1 = pl.min(128, pl.max(valid_rows - 128, 0))
                                    valid_part_h1 = pl.set_validshape(rotated_tile_h1, part_rows_h1, 64)
                                    output = pl.store(valid_part_h1, [row_start + row + 128, col + 192], output)
                                else:
                                    valid_acc = pl.set_validshape(acc, valid_rows, 256)
                                    output = pl.store(valid_acc, [row_start + row, col], output)
    return output


@pl.jit.inline(auto_scope=False)
def proj_q_b_pypto(
    payload: pl.Tensor,
    weight: pl.Tensor,
    activation_scale: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.Tensor,
    cosine: pl.Tensor,
    sine: pl.Tensor,
    rows: pl.Scalar[pl.INDEX],
    width: pl.Scalar[pl.INDEX],
    row_start: pl.Scalar[pl.INDEX],
    worker_limit: pl.constexpr = 32,
):
    """Select M256 or M80 Q projection from capacity and active rows."""
    if payload.shape[0] >= 256:
        if rows > 80 and ((rows + 255) // 256) * 256 <= ((rows + 79) // 80) * 80:
            output = _proj_q_b_tile_pypto(
                payload, weight, activation_scale, weight_scale, output, cosine, sine,
                rows, width, row_start, worker_limit, 256,
            )
        else:
            output = _proj_q_b_tile_pypto(
                payload, weight, activation_scale, weight_scale, output, cosine, sine,
                rows, width, row_start, worker_limit, 80,
            )
    else:
        output = _proj_q_b_tile_pypto(
            payload, weight, activation_scale, weight_scale, output, cosine, sine,
            rows, width, row_start, worker_limit, 80,
        )
    return output


@pl.jit.inline(auto_scope=False)
def proj_kv_pypto(
    payload: pl.Tensor,
    weight: pl.Tensor,
    activation_scale: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.Tensor,
    rows: pl.Scalar[pl.INDEX],
    row_start: pl.Scalar[pl.INDEX],
    worker_limit: pl.constexpr = 32,
):
    scale_width = weight.shape[0] // 32
    for mt in pl.parallel((rows + 79) // 80):
        scale_begin = mt * 80 * scale_width
        scale_end = scale_begin + 80 * scale_width
        sa = activation_scale[0, scale_begin:scale_end]
        sa = pl.reshape(sa, [80, scale_width])
        sa = pl.tensor.view(sa, [80, scale_width], layout=pl.MX_A_ZZ)
        for worker in pl.spmd(pl.min(worker_limit, 8)):
            for base in pl.range(0, 8, worker_limit):
                if base + worker < 8:
                    row = mt * 80
                    col = (base + worker) * 64
                    full_a = pl.load(payload, [row, 0], [80, 5120], target_memory=pl.Mem.Mat)
                    a0 = pl.tile.slice(full_a, [80, 256], [0, 0])
                    sa0 = pl.load(sa, [0, 0], [80, 8])
                    b0 = pl.load(weight, [0, col], [256, 64], target_memory=pl.Mem.Mat)
                    sb0 = pl.load(weight_scale, [0, col], [8, 64])
                    acc = pl.matmul_mx(a0, sa0, b0, sb0)
                    for k in pl.pipeline(256, 5120, 256, stage=2):
                        av = pl.tile.slice(full_a, [80, 256], [0, k])
                        sav = pl.load(sa, [0, k // 32], [80, 8])
                        bv = pl.load(weight, [k, col], [256, 64], target_memory=pl.Mem.Mat)
                        sbv = pl.load(weight_scale, [k // 32, col], [8, 64])
                        acc = pl.matmul_mx_acc(acc, av, sav, bv, sbv)
                    valid = pl.set_validshape(acc, pl.min(80, rows - row), 64)
                    output = pl.store(valid, [row_start + row, col], output)
    return output


@pl.jit.inline(auto_scope=False)
def q_proj_rope_pypto(
    payload: pl.Tensor,
    scales: pl.Tensor,
    wq_a: pl.Tensor,
    wq_a_scale: pl.Tensor,
    q_norm_weight: pl.Tensor,
    wq_b: pl.Tensor,
    wq_b_scale: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    query_latent: pl.Tensor,
    query: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    padded_rows: pl.Scalar[pl.INDEX],
    qa_worker_limit: pl.constexpr = 32,
    qb_worker_limit: pl.constexpr = 32,
):
    """Project NZ QA/QB weights, rotate Q, and retain normalized rows for Index-Q."""
    payload_rows = payload.shape[0]
    projected = pl.create_tensor([payload_rows, Q_LORA], dtype=pl.BF16)
    projected = proj_q_a_pypto(payload, wq_a, scales, wq_a_scale, projected, num_tokens, qa_worker_limit)
    qb_payload = pl.create_tensor([payload_rows, Q_LORA], dtype=pl.FP8E4M3FN)
    qb_scales = pl.create_tensor([1, payload_rows * (Q_LORA // 32)], dtype=pl.FP8E8M0)
    norm_workers = pl.max(pl.min(payload_rows // (MX_M_TILE // 2), 32), 1)
    latent_rows = pl.tensor.dim(query_latent, 0)
    norm_latent = query_latent[:latent_rows, :Q_LORA]
    norm_payload = qb_payload[:payload_rows, :Q_LORA]
    norm_scales = qb_scales[:, :payload_rows * (Q_LORA // 32)]
    with pl.spmd(norm_workers, name_hint="q_norm_quant"):
        q_norm_quant(projected, q_norm_weight, norm_latent, norm_payload, norm_scales, num_tokens)
    query = proj_q_b_pypto(
        qb_payload, wq_b, qb_scales, wq_b_scale,
        query, rope_cos, rope_sin, num_tokens, LOCAL_H * HEAD_DIM, 0, qb_worker_limit,
    )
    return (query_latent, query, qb_payload, qb_scales)


@pl.jit.inline(auto_scope=False)
def kv_proj_pypto(
    payload: pl.Tensor,
    scales: pl.Tensor,
    wkv: pl.Tensor,
    wkv_scale: pl.Tensor,
    kv_projection: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    worker_limit: pl.constexpr = 32,
):
    """Project NZ KV weights to BF16 rows in M80 groups."""
    kv_projection = proj_kv_pypto(payload, wkv, scales, wkv_scale, kv_projection, num_tokens, 0, worker_limit)
    return kv_projection


@pl.jit
def qkv_native(
    payload: pl.Tensor,
    scales: pl.Tensor,
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN, pl.NZ],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor,
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN, pl.NZ],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, 512], pl.FP8E4M3FN, pl.NZ],
    wkv_scale: pl.Tensor[[D // 32, 512], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    query_latent: pl.InOut[pl.Tensor],
    query: pl.InOut[pl.Tensor],
    kv_projection: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
    padded_rows: pl.Scalar[pl.INDEX],
):
    query_latent, query, q_payload, q_scales = q_proj_rope_pypto(
        payload, scales, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
        rope_cos, rope_sin, query_latent, query, num_tokens, padded_rows,
        qa_worker_limit=NATIVE_VALIDATION_WORKERS, qb_worker_limit=NATIVE_VALIDATION_WORKERS,
    )
    kv_projection = kv_proj_pypto(
        payload, scales, wkv, wkv_scale, kv_projection, num_tokens, worker_limit=NATIVE_VALIDATION_WORKERS,
    )
    return query_latent, query, kv_projection


@pl.jit
def qkv_cpp(
    payload: pl.Tensor,
    scales: pl.Tensor,
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor,
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, 512], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, 512], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    query_latent: pl.InOut[pl.Tensor],
    query: pl.InOut[pl.Tensor],
    kv_projection: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
    padded_rows: pl.Scalar[pl.INDEX],
):
    query_latent, query, q_payload, q_scales = q_proj_rope(
        payload, scales, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
        rope_cos, rope_sin, query_latent, query, num_tokens, padded_rows,
        qa_worker_limit=NATIVE_VALIDATION_WORKERS, qb_worker_limit=NATIVE_VALIDATION_WORKERS,
        qa_row_tile=80, qb_row_tile=80,
    )
    kv_projection = kv_proj(
        payload, scales, wkv, wkv_scale, kv_projection, num_tokens, worker_limit=NATIVE_VALIDATION_WORKERS, row_tile=80,
    )
    return query_latent, query, kv_projection



def golden_rope_interleave(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply adjacent-pair rotary embedding to the final dimension."""
    value = x.float().unflatten(-1, (-1, 2))
    real, imag = value.unbind(dim=-1)
    if inverse:
        sin = -sin
    while cos.ndim < real.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    rotated = torch.stack((real * cos - imag * sin, imag * cos + real * sin), dim=-1)
    return rotated.flatten(-2).to(x.dtype)


def golden_rope_tail(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Rotate the trailing 2*cos.shape[-1] lanes as adjacent (real, imag) pairs."""
    rope_dim = cos.shape[-1] * 2
    rotated = golden_rope_interleave(x[..., -rope_dim:], cos, sin, inverse)
    return torch.cat((x[..., :-rope_dim], rotated), dim=-1)


def golden_qkv_proj_rope(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor | None,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor | None,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor | None,
    kv_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    sequential_query: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project normalized query latent and shared KV, then rotate their RoPE tails."""
    from quant import unpack_fp8_nz

    if wq_a.dtype == torch.float8_e4m3fn:
        wq_a = unpack_fp8_nz(wq_a)
    if wq_b.dtype == torch.float8_e4m3fn:
        wq_b = unpack_fp8_nz(wq_b)
    if wkv.dtype == torch.float8_e4m3fn:
        wkv = unpack_fp8_nz(wkv)

    def project(value, weight, scale):
        if sequential_query and scale is not None:
            return golden_gemm_bf16_via_mxfp8_to_bf16_sequential(value, weight, scale)
        return golden_mxfp8_linear_grouped_sum(value, weight, scale)

    if wq_a_scale is not None:
        projected = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(x, wq_a, wq_a_scale)
    else:
        projected = golden_mxfp8_linear_grouped_sum(x, wq_a, wq_a_scale)
    qr = golden_rms_norm(projected, q_norm_weight)
    head_dim = wkv.shape[-1]
    num_heads = wq_b.shape[-1] // head_dim
    q = project(qr, wq_b, wq_b_scale).unflatten(-1, (num_heads, head_dim))
    kv = golden_rms_norm(project(x, wkv, wkv_scale), kv_norm_weight)
    q = golden_rope_tail(q, cos, sin)
    kv = golden_rope_tail(kv, cos, sin)
    return q, kv, qr


def build_qkv_validation_specs(active, capacity):
    """Build model-sized NZ weights, packed MX scales, and canary output buffers."""
    from golden import ScalarSpec, TensorSpec
    from quant import generate_mxfp8_weight_kn, pack_fp8_nz, pack_mx_a_scale
    from quant import _quantize_mxfp8_activation, encode_e8m0

    padded = (capacity + 79) // 80 * 80
    values = {}

    def inputs():
        if values:
            return values
        generator = torch.Generator().manual_seed(17)
        source = torch.randn(padded, D, generator=generator).bfloat16()
        source[active:] = 0
        payload, scale = _quantize_mxfp8_activation(source, scale_alg=0)
        values["payload"] = payload.flatten(-2)
        codes = pack_mx_a_scale(encode_e8m0(scale)).reshape(1, -1)
        values["scales"] = codes.view(torch.float8_e8m0fnu)
        for name, width, reduction, seed in (("wq_a", Q_LORA, D, 19),
                                             ("wq_b", LOCAL_H * HEAD_DIM, Q_LORA, 23),
                                             ("wkv", HEAD_DIM, D, 29)):
            weight, scales = generate_mxfp8_weight_kn(width, reduction, reduction**-0.5, seed=seed)
            values[name] = pack_fp8_nz(weight)
            values[name + "_scale"] = scales
        values["q_norm_weight"] = torch.ones(Q_LORA, dtype=torch.bfloat16)
        frequencies = 10000.0 ** (-torch.arange(0, M.qk_rope_head_dim, 2).float() / M.qk_rope_head_dim)
        angles = (8192 + torch.arange(capacity)).float()[:, None] * frequencies
        values.update(rope_cos=angles.cos(), rope_sin=angles.sin())
        return values

    fp8, scale_dtype = torch.float8_e4m3fn, torch.float8_e8m0fnu
    shapes = (
        ("payload", [padded, D], fp8), ("scales", [1, padded * (D // 32)], scale_dtype),
        ("wq_a", [D, Q_LORA], fp8), ("wq_a_scale", [D // 32, Q_LORA], scale_dtype),
        ("q_norm_weight", [Q_LORA], torch.bfloat16),
        ("wq_b", [Q_LORA, LOCAL_H * HEAD_DIM], fp8),
        ("wq_b_scale", [Q_LORA // 32, LOCAL_H * HEAD_DIM], scale_dtype),
        ("wkv", [D, HEAD_DIM], fp8), ("wkv_scale", [D // 32, HEAD_DIM], scale_dtype),
        ("rope_cos", [capacity, M.qk_rope_head_dim // 2], torch.float32),
        ("rope_sin", [capacity, M.qk_rope_head_dim // 2], torch.float32),
    )
    specs = [TensorSpec(name, shape, dtype, init_value=lambda name=name: inputs()[name])
             for name, shape, dtype in shapes]
    for name, width in (("query_latent", Q_LORA), ("query", LOCAL_H * HEAD_DIM), ("kv_projection", HEAD_DIM)):
        specs.append(TensorSpec(name, [capacity, width], torch.bfloat16, init_value=13.0))
    return specs + [ScalarSpec("num_tokens", torch.int32, active), ScalarSpec("padded_rows", torch.int64, padded)]


def golden_qkv_validation(values):
    """Compute Q normalization, Q RoPE, and raw KV projection with K32 accumulation."""
    from gemm import golden_gemm_bf16_via_mxfp8_to_bf16_sequential as project
    from quant import decode_e8m0, unpack_mx_a_scale, unpack_fp8_nz

    active = int(values["num_tokens"])
    if active == 0:
        return
    padded = values["payload"].shape[0]
    codes = values["scales"].view(torch.uint8).reshape(padded, D // 32)
    scales = decode_e8m0(unpack_mx_a_scale(codes))[:active].repeat_interleave(32, dim=-1)
    source = (values["payload"][:active].float() * scales).bfloat16()
    projected = project(source, unpack_fp8_nz(values["wq_a"]), values["wq_a_scale"])
    latent = golden_rms_norm(projected, values["q_norm_weight"])
    query = project(latent, unpack_fp8_nz(values["wq_b"]), values["wq_b_scale"])
    query = query.reshape(active, LOCAL_H, HEAD_DIM)
    query = golden_rope_tail(query, values["rope_cos"][:active], values["rope_sin"][:active])
    values["query_latent"][:active].copy_(latent)
    values["query"][:active].copy_(query.flatten(1).bfloat16())
    values["kv_projection"][:active].copy_(project(source, unpack_fp8_nz(values["wkv"]), values["wkv_scale"]))


def compare_qkv_output(actual, expected, *, inputs, **kwargs):
    """Check the active prefix and require exact preservation of inactive output rows."""
    from golden import ratio_allclose

    active = int(inputs["num_tokens"])
    compare = ratio_allclose(atol=1e-2, rtol=1.0 / 128, valid_rows=active)
    passed, detail = compare(actual, expected, inputs=inputs, **kwargs)
    tail_ok = torch.equal(actual[active:], expected[active:])
    return passed and tail_ok, f"{detail}; inactive canary preserved={tail_ok}"
