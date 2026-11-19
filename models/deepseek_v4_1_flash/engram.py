# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 Engram projection and residual gating from host rows or local lookup tables."""

import math
from pathlib import Path

import pypto.language as pl
from pypto.runtime import pto_isa_include_dir
import torch

from config import FLASH as M, T_DYN
from gemm import gemm_bf16_via_mxfp8_to_bf16_staged


# model config
D = M.hidden_size
HC_MULT = M.hc_mult


N_HASH_COLS = (M.engram_max_ngram_size - 1) * M.engram_n_heads  # 24
HEAD_DIM = M.engram_head_dim  # 256
ENGRAM_K = N_HASH_COLS * HEAD_DIM  # 6144 flattened lookup width
KV_OUT = (HC_MULT + 1) * D  # 5 * 5120 = 25600
CLAMP = 1e-6
NORM_EPS = M.rms_norm_eps
D_INV = 1.0 / D
DOT_SCALE = D**-0.5

# Quantization and projection tiling.
ENGRAM_QUANT_M_TILE = 16
ENGRAM_QUANT_K_TILE = 256
ENGRAM_QUANT_SCALE_TILE = 128
GATE_T_TILE = 8
GATE_D_TILE = 1280
PROJECTION_N_TILE = 256
PROJECTION_K_TILE = 256


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels/engram.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _engram_quantize(lookup: pl.Tensor, payload: pl.Out[pl.Tensor], scales: pl.Out[pl.Tensor],
                      tokens: pl.Scalar[pl.INT32], padded: pl.Scalar[pl.INDEX],
                      mode: pl.Scalar[pl.INDEX]) -> tuple[pl.Tensor, pl.Tensor]:
    """Write FP8 activation payload and packed E8M0 scales."""
    return payload, scales


@pl.jit.extern(core_type="aic", source=Path(__file__).parent / "kernels/engram.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _engram_project(payload: pl.Tensor, weight: pl.Tensor, scales: pl.Tensor, weight_scale: pl.Tensor,
                    kv: pl.Out[pl.Tensor], tokens: pl.Scalar[pl.INDEX], row_tile: pl.Scalar[pl.INDEX]) -> pl.Tensor:
    """Project quantized rows into BF16 keys and values."""
    return kv


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels/engram.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _engram_gate_native(kv: pl.Tensor, weight: pl.Tensor, x: pl.Tensor, out: pl.Out[pl.Tensor],
                        tokens: pl.Scalar[pl.INDEX], mode: pl.Scalar[pl.INDEX]) -> pl.Tensor:
    """Compute the normalized-dot gate and update the residual."""
    return out


@pl.jit.inline(auto_scope=False)
def engram_gate(
    kv: pl.Tensor, weight: pl.Tensor, x: pl.Tensor, out: pl.Tensor,
    lookup: pl.Tensor, mask_zero_rows: pl.constexpr,
):
    """Run native gating and preserve zero-lookup FP32 residual rows."""
    tokens = pl.tensor.dim(x, 0)
    if mask_zero_rows:
        original = pl.create_tensor([tokens, HC_MULT * D], dtype=pl.FP32)
        hidden = pl.reshape(x, [tokens, HC_MULT * D])
        with pl.spmd(tokens, name_hint="engram_save_residual") as saved_ready:
            row = pl.tile.get_block_idx()
            for column in pl.range(0, HC_MULT * D, 512):
                value = pl.load(hidden, [row, column], [1, 512])
                pl.store(value, [row, column], original)
        with pl.spmd(pl.system.available_aiv_count(), name_hint="engram_gate", deps=[saved_ready]):
            result = _engram_gate_native(kv, weight, x, out, tokens, 3)
        output = pl.reshape(result, [tokens, HC_MULT * D])
        for block in pl.spmd((tokens + 7) // 8, name_hint="engram_restore_masked"):
            row = block * 8
            valid_rows = pl.min(8, tokens - row)
            total = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
            reduction_tmp = pl.create_tile([8, 256], dtype=pl.FP32)
            for column in pl.range(0, ENGRAM_K, 256):
                source = pl.load(lookup, [row, column], [8, 256], valid_shape=[valid_rows, 256])
                padded = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 8, 256)
                source_fp32 = pl.cast(padded, pl.FP32)
                part = pl.reshape(pl.row_sum(pl.abs(source_fp32), reduction_tmp), [1, 8])
                total = pl.add(total, part)
            ones = pl.tile.full([8, 512], dtype=pl.FP32, value=1.0)
            active = pl.cmps(pl.row_expand_mul(ones, pl.reshape(total, [8, 1])), 0.0, cmp_type=4)
            for column in pl.range(0, HC_MULT * D, 512):
                saved = pl.load(original, [row, column], [8, 512], valid_shape=[valid_rows, 512])
                updated = pl.load(output, [row, column], [8, 512], valid_shape=[valid_rows, 512])
                restored = pl.tile.select(active, pl.set_validshape(updated, 8, 512), pl.set_validshape(saved, 8, 512))
                pl.store(pl.set_validshape(restored, valid_rows, 512), [row, column], output)
    else:
        with pl.spmd(pl.system.available_aiv_count(), name_hint="engram_gate"):
            result = _engram_gate_native(kv, weight, x, out, tokens, 3)
    return out


@pl.jit.inline(auto_scope=False)
def _load_engram_quant_work(
    x: pl.Tensor,
    inputs: pl.Tile[[2 * ENGRAM_QUANT_M_TILE, ENGRAM_QUANT_K_TILE], pl.BF16, pl.Mem.Vec],
    work: pl.Scalar[pl.INDEX],
    slot: pl.Scalar[pl.INDEX],
    num_tokens: pl.Scalar[pl.INT32],
):
    t0 = work // (ENGRAM_K // ENGRAM_QUANT_K_TILE) * ENGRAM_QUANT_M_TILE
    k0 = work % (ENGRAM_K // ENGRAM_QUANT_K_TILE) * ENGRAM_QUANT_K_TILE
    rows = pl.max(0, pl.min(ENGRAM_QUANT_M_TILE, num_tokens - t0))
    if rows > 0:
        input_row = slot * ENGRAM_QUANT_M_TILE
        inputs = pl.tile.gather_row(inputs, x, [input_row, 0], [t0, k0], [ENGRAM_QUANT_M_TILE, ENGRAM_QUANT_K_TILE],
                                    valid_shape=[rows, ENGRAM_QUANT_K_TILE])
    return inputs



@pl.jit.incore
def _engram_quantize_pypto(
    x: pl.Tensor,
    payload: pl.Out[pl.Tensor],
    scales: pl.Out[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
    padded_rows: pl.Scalar[pl.INDEX],
    auto_sync: pl.constexpr = True,
) -> tuple[pl.Tensor, pl.Tensor]:
    """Quantize row-major BF16 projection inputs with M16-packed E8M0 scales."""
    first = pl.tile.get_block_idx()
    workers = pl.tile.get_block_num()
    column_tiles = ENGRAM_K // ENGRAM_QUANT_K_TILE
    total = padded_rows // ENGRAM_QUANT_M_TILE * column_tiles
    if first < total:
        inputs = pl.create_tile([2 * ENGRAM_QUANT_M_TILE, ENGRAM_QUANT_K_TILE], dtype=pl.BF16, target_memory=pl.Mem.Vec)
        output = pl.create_tile([2 * ENGRAM_QUANT_M_TILE * ENGRAM_QUANT_K_TILE], dtype=pl.INT8, target_memory=pl.Mem.Vec)
        # Reserve a full 256-byte register store per scale slot.
        codes = pl.create_tile([ENGRAM_QUANT_SCALE_TILE], dtype=pl.UINT32, target_memory=pl.Mem.Vec)
        inputs = _load_engram_quant_work(x, inputs, first, 0, num_tokens)
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
                inputs = _load_engram_quant_work(x, inputs, next_work, next_slot, num_tokens)
                pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.V, next_slot)
            t0 = work // column_tiles * ENGRAM_QUANT_M_TILE
            k0 = work % column_tiles * ENGRAM_QUANT_K_TILE
            rows = pl.max(0, pl.min(ENGRAM_QUANT_M_TILE, num_tokens - t0))
            source = pl.reshape(inputs, [2 * ENGRAM_QUANT_M_TILE * ENGRAM_QUANT_K_TILE])
            pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.V, slot)
            pl.system.wait_flag(pl.PipeType.MTE3, pl.PipeType.V, slot)
            output_words = pl.reinterpret_view(output, pl.UINT32, shape=[2 * ENGRAM_QUANT_M_TILE * ENGRAM_QUANT_K_TILE // 4])
            with pl.reg_scope(auto_sync=auto_sync):
                # Keep aligned row and slot offsets visible to register lowering.
                slot_row = slot * ENGRAM_QUANT_M_TILE
                code_offset = slot * (ENGRAM_QUANT_SCALE_TILE // 2)
                full = pl.reg.full_mask(lanes=ENGRAM_QUANT_K_TILE)
                groups = pl.reg.full_mask(lanes=ENGRAM_QUANT_K_TILE // 32)
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
                    offset = tile_row * ENGRAM_QUANT_K_TILE
                    raw = pl.reg.load(source, offset, lanes=ENGRAM_QUANT_K_TILE)
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
                    reciprocal = pl.reg.group_broadcast(reciprocal, lanes=ENGRAM_QUANT_K_TILE)
                    value = pl.reg.mul(value, reciprocal, full)
                    value = pl.reg.maximums(value, -448.0, full)
                    value = pl.reg.minimums(value, 448.0, full)
                    fp8 = pl.reg.convert(value, dtype=pl.FP8E4M3FN, rounding="R", saturate=False)
                    words = pl.reg.interpret_cast(fp8, dtype=pl.UINT32)
                    word_offset = tile_row * (ENGRAM_QUANT_K_TILE // 4)
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
                if not auto_sync:
                    # Complete the active payload stores before the scale and padding stores.
                    pl.reg.mem_bar("VST_VST")
                _scales = pl.reg.store(codes, code_offset, packed_scales)
                padding = pl.reg.broadcast(0, lanes=ENGRAM_QUANT_K_TILE // 4, dtype=pl.UINT32)
                for inactive in pl.hardware_loop(ENGRAM_QUANT_M_TILE - rows):
                    row = rows + inactive
                    tile_row = slot_row + row
                    word_offset = tile_row * (ENGRAM_QUANT_K_TILE // 4)
                    _payload_pad = pl.reg.store(output_words, word_offset, padding)
            pl.system.set_flag(pl.PipeType.V, pl.PipeType.MTE2, slot)
            pl.system.set_flag(pl.PipeType.V, pl.PipeType.MTE3, slot)
            pl.system.wait_flag(pl.PipeType.V, pl.PipeType.MTE3, slot)
            output_rows = pl.reshape(output, [2 * ENGRAM_QUANT_M_TILE, ENGRAM_QUANT_K_TILE])
            output_tile = pl.tile.slice(output_rows, [ENGRAM_QUANT_M_TILE, ENGRAM_QUANT_K_TILE], [slot * ENGRAM_QUANT_M_TILE, 0])
            quantized = pl.reinterpret_view(output_tile, pl.FP8E4M3FN)
            scale_bytes = pl.reinterpret_view(codes, pl.UINT8, shape=[2, 2 * ENGRAM_QUANT_SCALE_TILE])
            scale_tile = pl.tile.slice(scale_bytes, [1, ENGRAM_QUANT_SCALE_TILE], [slot, 0])
            scale_codes = pl.reinterpret_view(scale_tile, pl.FP8E8M0)
            scale_offset = t0 * (ENGRAM_K // 32) + k0 // ENGRAM_QUANT_K_TILE * ENGRAM_QUANT_SCALE_TILE
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


@pl.jit.inline(auto_scope=False)
def _engram_rows_tile(
    lookup: pl.Tensor, wkv: pl.Tensor, wkv_scale: pl.Tensor, weight: pl.Tensor,
    x: pl.Tensor, out: pl.Tensor, tokens: pl.Scalar[pl.INT32],
    row_tile: pl.constexpr, mask_zero_rows: pl.constexpr,
):
    """Quantize and project host rows before gating the residual."""
    padded = ((tokens + row_tile - 1) // row_tile) * row_tile
    payload = pl.create_tensor([padded, ENGRAM_K], dtype=pl.FP8E4M3FN)
    scales = pl.create_tensor([1, padded * (ENGRAM_K // 32)], dtype=pl.FP8E8M0)
    kv = pl.create_tensor([tokens, KV_OUT], dtype=pl.BF16)
    with pl.spmd(pl.system.available_aiv_count(), name_hint="engram_quant"):
        quantized, quantized_scales = _engram_quantize_pypto(lookup, payload, scales, tokens, padded, False)
    with pl.spmd(pl.system.available_cluster_count(), name_hint="engram_projection"):
        projected = _engram_project(quantized, wkv, quantized_scales, wkv_scale, kv, tokens, row_tile)
    if mask_zero_rows:
        return engram_gate(projected, weight, x, out, lookup, mask_zero_rows)
    output_rows = pl.reshape(out, [pl.tensor.dim(out, 0) * HC_MULT, D])
    if tokens >= 256:
        hidden_pairs = pl.reshape(x, [pl.tensor.dim(x, 0) * (HC_MULT // 2), 2 * D])
        weight_pairs = pl.reshape(weight, [HC_MULT // 2, 2 * D])
        with pl.spmd(pl.system.available_aiv_count(), name_hint="engram_gate_pairs"):
            _engram_gate_pairs_reg(projected, weight_pairs, hidden_pairs, output_rows, tokens, True)
    else:
        hidden_rows = pl.reshape(x, [pl.tensor.dim(x, 0) * HC_MULT, D])
        with pl.spmd(pl.system.available_aiv_count(), name_hint="engram_gate"):
            _engram_gate_reg(projected, weight, hidden_rows, output_rows, tokens, True)
    return out


@pl.jit.inline(auto_scope=False)
def engram_from_rows(
    lookup: pl.Tensor,
    wkv_weight: pl.Tensor[[ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[ENGRAM_K // 32, KV_OUT], pl.FP8E8M0],
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    x: pl.Tensor,
    out: pl.Tensor,
    mask_zero_rows: pl.constexpr = False,
):
    """Project host-gathered ND rows and gate the FP32 residual in place or out of place."""
    tokens = pl.tensor.dim(x, 0)
    if tokens > 0:
        if tokens <= 80:
            result = _engram_rows_tile(
                lookup, wkv_weight, wkv_scale, weight, x, out, tokens, 80, mask_zero_rows,
            )
        elif tokens <= 128:
            result = _engram_rows_tile(
                lookup, wkv_weight, wkv_scale, weight, x, out, tokens, 128, mask_zero_rows,
            )
        else:
            result = _engram_rows_tile(
                lookup, wkv_weight, wkv_scale, weight, x, out, tokens, 256, mask_zero_rows,
            )
    else:
        result = out
    return result


@pl.jit.inline
def _engram_gate_pypto(
    kv: pl.Tensor,
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    x: pl.Tensor,
    out: pl.Tensor,
):
    """Gate projected keys and values into each residual stream copy."""
    t_dim = pl.tensor.dim(kv, 0)
    x_flat = pl.reshape(x, [t_dim, HC_MULT * D])
    out_flat = pl.reshape(out, [t_dim, HC_MULT * D])
    t_blocks = (t_dim + GATE_T_TILE - 1) // GATE_T_TILE
    workers = pl.min(t_blocks * HC_MULT, pl.system.available_aiv_count())
    for worker in pl.spmd(workers, name_hint="_engram_gate_pypto"):
        for wave in pl.pipeline((t_blocks * HC_MULT + workers - 1) // workers, stage=2):
            block = wave * workers + worker
            if block < t_blocks * HC_MULT:
                t0 = (block // HC_MULT) * GATE_T_TILE
                c = block % HC_MULT
                valid_rows = pl.min(GATE_T_TILE, t_dim - t0)
                reduction_tmp = pl.create_tile([GATE_T_TILE, GATE_D_TILE], dtype=pl.FP32)
                # Reduction accumulators have 32-byte aligned rows.
                sq_x = pl.tile.full([1, GATE_T_TILE], dtype=pl.FP32, value=0.0)
                sq_k = pl.tile.full([1, GATE_T_TILE], dtype=pl.FP32, value=0.0)
                dot = pl.tile.full([1, GATE_T_TILE], dtype=pl.FP32, value=0.0)
                for d0 in pl.range(0, D, GATE_D_TILE):
                    x_raw = pl.load(x_flat, [t0, c * D + d0], [GATE_T_TILE, GATE_D_TILE], valid_shape=[valid_rows, GATE_D_TILE])
                    key_raw = pl.load(kv, [t0, c * D + d0], [GATE_T_TILE, GATE_D_TILE], valid_shape=[valid_rows, GATE_D_TILE])
                    key_d = pl.cast(pl.set_validshape(pl.fillpad(key_raw, pad_value=pl.PadValue.zero), GATE_T_TILE, GATE_D_TILE), pl.FP32)
                    x_d = pl.set_validshape(pl.fillpad(x_raw, pad_value=pl.PadValue.zero), GATE_T_TILE, GATE_D_TILE)
                    sq_x = pl.add(sq_x, pl.reshape(pl.row_sum(pl.mul(x_d, x_d), reduction_tmp), [1, GATE_T_TILE]))
                    sq_k = pl.add(sq_k, pl.reshape(pl.row_sum(pl.mul(key_d, key_d), reduction_tmp), [1, GATE_T_TILE]))
                    w_d = pl.load(weight, [c, d0], [1, GATE_D_TILE])
                    xw_d = pl.col_expand_mul(x_d, w_d)
                    dot = pl.add(dot, pl.reshape(pl.row_sum(pl.mul(xw_d, key_d), reduction_tmp), [1, GATE_T_TILE]))
                inv_x = pl.rsqrt(pl.add(pl.mul(sq_x, D_INV), NORM_EPS))
                inv_k = pl.rsqrt(pl.add(pl.mul(sq_k, D_INV), NORM_EPS))
                dot = pl.mul(pl.mul(pl.mul(dot, inv_x), inv_k), DOT_SCALE)
                root = pl.sqrt(pl.maximum(pl.abs(dot), CLAMP))
                nonnegative = pl.cmps(dot, 0.0, cmp_type=5)
                signed = pl.tile.select(nonnegative, root, pl.neg(root))
                gate = pl.reshape(pl.recip(pl.add(pl.exp(pl.neg(signed)), 1.0)), [GATE_T_TILE, 1])
                for d0 in pl.range(0, D, GATE_D_TILE):
                    output_source = pl.load(x_flat, [t0, c * D + d0], [GATE_T_TILE, GATE_D_TILE], valid_shape=[valid_rows, GATE_D_TILE])
                    output_value = pl.load(kv, [t0, HC_MULT * D + d0], [GATE_T_TILE, GATE_D_TILE], valid_shape=[valid_rows, GATE_D_TILE])
                    gated = pl.row_expand_mul(pl.cast(output_value, pl.FP32), gate)
                    y = pl.add(output_source, gated)
                    y_bf16 = pl.cast(y, target_type=pl.BF16, mode="rint")
                    y_rounded = pl.cast(y_bf16, pl.FP32)
                    pl.store(pl.set_validshape(y_rounded, valid_rows, GATE_D_TILE), [t0, c * D + d0], out_flat)
    return out


@pl.jit.inline(auto_scope=False)
def engram_from_rows_pypto(
    lookup: pl.Tensor,
    wkv_weight: pl.Tensor[[ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[ENGRAM_K // 32, KV_OUT], pl.FP8E8M0],
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    x: pl.Tensor,
    out: pl.Tensor,
):
    """Compute Engram in PyPTO DSL for explicit comparison with the production entry."""
    tokens = pl.tensor.dim(x, 0)
    if tokens > 0:
        kv = pl.create_tensor([tokens, KV_OUT], dtype=pl.BF16)
        projected = gemm_bf16_via_mxfp8_to_bf16_staged(
            lookup, wkv_weight, wkv_scale, kv, tokens,
            ENGRAM_K, KV_OUT, "engram_pypto_projection", "engram_pypto_quant",
            PROJECTION_N_TILE, PROJECTION_K_TILE, 1, 2147483647,
        )
        _engram_gate_pypto(projected, weight, x, out)
    return out


class HostEngram:
    """Hash text history and read selected FP8 checkpoint table rows on the CPU.

    Reuse one instance across prefill chunks and decode calls. Each request_id
    owns its compressed-token history; start_pos=0 resets that request, and
    continuation requires the preceding history. False token_mask entries exclude
    image positions and stop n-grams from crossing image spans.
    """

    def __init__(self, checkpoint, tokenizer):
        import numpy as np
        from tokenizers import Regex, normalizers

        self.checkpoint = checkpoint
        sentinel = "\ue000"
        normalizer = normalizers.Sequence([
            normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(), normalizers.Replace(sentinel, " "),
        ])
        keys = {}
        token_map = []
        for token in range(tokenizer.get_vocab_size(with_added_tokens=True)):
            text = tokenizer.decode([token], skip_special_tokens=False)
            key = tokenizer.id_to_token(token) if "\ufffd" in text else normalizer.normalize_str(text) or text
            token_map.append(keys.setdefault(key, len(keys)))
        if len(keys) != M.engram_compressed_vocab_size:
            raise ValueError(f"Engram compressed vocabulary has {len(keys)} entries")
        self.token_map = torch.tensor(token_map, dtype=torch.int64)
        self.pad_id = token_map[M.engram_pad_token_id]
        self.history = {}
        primes, seen = [], set()
        for layer in M.engram_layer_ids:
            buckets = []
            for _ in range(M.engram_max_ngram_size - 1):
                candidate = M.engram_vocab_size - 1
                for _ in range(M.engram_n_heads):
                    candidate += 1
                    while candidate in seen or any(candidate % d == 0 for d in range(2, math.isqrt(candidate) + 1)):
                        candidate += 1
                    seen.add(candidate)
                    buckets.append(candidate)
            primes.append(buckets)
        if tuple(map(sum, primes)) != M.engram_num_embeddings:
            raise ValueError("Engram prime buckets do not match checkpoint table sizes")
        self.primes = torch.tensor(primes, dtype=torch.int64)
        self.offsets = self.primes.cumsum(-1) - self.primes
        bound = max(1, (np.iinfo(np.int64).max // len(keys)) // 2)
        self.multipliers = torch.stack([
            torch.from_numpy(np.random.default_rng(10007 * layer).integers(
                0, bound, size=M.engram_max_ngram_size, dtype=np.int64,
            ) * 2 + 1)
            for layer in M.engram_layer_ids
        ])

    def hash_ids(self, token_ids, start_pos=0, *, request_id=0, token_mask=None):
        """Return [layer, token, hash column] IDs and commit the current text history."""
        ids = torch.as_tensor(token_ids, device="cpu")
        if ids.numel() and ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Engram token IDs must have an integer dtype")
        ids = ids.long()
        if ids.ndim != 1 or bool(((ids < 0) | (ids >= self.token_map.numel())).any()):
            raise ValueError("Engram requires a one-dimensional sequence of valid token IDs")
        history = self.history.get(request_id, torch.empty(0, dtype=torch.int64))
        if isinstance(start_pos, bool) or not isinstance(start_pos, int) or not 0 <= start_pos <= history.numel():
            raise ValueError("Engram continuation requires all preceding token history")
        current = self.token_map[ids]
        if token_mask is not None:
            mask = torch.as_tensor(token_mask, device="cpu")
            if mask.dtype != torch.bool or mask.shape != ids.shape:
                raise ValueError("Engram token_mask must be a boolean vector matching token IDs")
            current = torch.where(mask, current, -1)
        compressed = torch.cat((history[:start_pos], current))
        positions = torch.arange(start_pos, compressed.numel())
        lookback = []
        blocked = torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(M.engram_max_ngram_size):
            source = (positions - shift).clamp_min(0)
            values = compressed[source]
            blocked |= (positions < shift) | (values == -1)
            lookback.append(torch.where(blocked, self.pad_id, values))
        tokens = torch.stack(lookback, dim=-1)
        products = tokens.unsqueeze(0) * self.multipliers.unsqueeze(1)
        rolling, hashes = products[..., 0], []
        for shift in range(1, M.engram_max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., shift])
            begin = (shift - 1) * M.engram_n_heads
            primes = self.primes[:, begin:begin + M.engram_n_heads].unsqueeze(1)
            hashes.append(rolling.unsqueeze(-1) % primes)
        self.history[request_id] = compressed
        return torch.cat(hashes, dim=-1) + self.offsets.unsqueeze(1)

    def lookup(self, token_ids, capacity, start_pos=0, *, request_id=0, token_mask=None):
        """Read current rows and group-32 E8M0 scales into [2, capacity, 6144] BF16.

        Tables stay in host checkpoint files; only selected rows are read.
        Unused token rows are zero-padded.
        """
        from quant import decode_e8m0

        if len(token_ids) > capacity or capacity <= 0:
            raise ValueError("Engram rows must fit the current-token capacity")
        hashes = self.hash_ids(token_ids, start_pos, request_id=request_id, token_mask=token_mask)
        active = torch.ones(len(token_ids), dtype=torch.bool)
        if token_mask is not None:
            active = torch.as_tensor(token_mask, device="cpu")
        layers = len(M.engram_layer_ids)
        width = self.primes.shape[1] * M.engram_head_dim
        result = torch.zeros((layers, capacity, width), dtype=torch.bfloat16)
        for slot, layer in enumerate(M.engram_layer_ids):
            prefix = f"layers.{layer}.engram.embed."
            if not bool(active.any()):
                continue
            rows = self.checkpoint.tensor_rows(prefix + "weight", hashes[slot, active])
            scales = self.checkpoint.tensor_rows(prefix + "scale", hashes[slot, active])
            if rows.dtype != torch.float8_e4m3fn or rows.shape[-1] != M.engram_head_dim:
                raise ValueError(f"unsupported native Engram table layout: {prefix}weight")
            if scales.dtype != torch.float8_e8m0fnu or scales.shape[-1] != M.engram_head_dim // 32:
                raise ValueError(f"unsupported native Engram table scales: {prefix}scale")
            values = rows.float() * decode_e8m0(scales).repeat_interleave(32, dim=-1)
            result[slot, :len(token_ids)][active] = values.flatten(1).to(torch.bfloat16)
        return result


def build_masked_gate_specs(tokens):
    """Build mixed text/image rows with non-BF16 FP32 residual values."""
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(9)
    lookup = torch.randn(tokens, ENGRAM_K, generator=generator).bfloat16()
    kv = torch.randn(tokens, KV_OUT, generator=generator).bfloat16()
    lookup[1::3] = 0
    kv[1::3] = 0
    hidden = torch.randn(tokens, HC_MULT, D, generator=generator)
    return [
        TensorSpec("kv", [tokens, KV_OUT], torch.bfloat16, init_value=kv),
        TensorSpec("weight", [HC_MULT, D], torch.float32, init_value=1.0),
        TensorSpec("x", [tokens, HC_MULT, D], torch.float32, init_value=hidden),
        TensorSpec("lookup", [tokens, ENGRAM_K], torch.bfloat16, init_value=lookup),
        TensorSpec("out", [tokens, HC_MULT, D], torch.float32),
    ]


def golden_masked_gate(tensors):
    """Evaluate gated text rows and exact image-row residual passthrough."""
    hidden = tensors["x"]
    keys = tensors["kv"][:, :HC_MULT * D].float().reshape(-1, HC_MULT, D)
    value = tensors["kv"][:, HC_MULT * D:].float()
    dot = (hidden * tensors["weight"] * keys).sum(-1)
    dot *= torch.rsqrt(hidden.square().mean(-1) + NORM_EPS)
    dot *= torch.rsqrt(keys.square().mean(-1) + NORM_EPS) * DOT_SCALE
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(CLAMP).sqrt(), dot))
    expected = (hidden + gate.unsqueeze(-1) * value[:, None]).bfloat16().float()
    masked = tensors["lookup"].count_nonzero(dim=-1) == 0
    expected[masked] = hidden[masked]
    tensors["out"].copy_(expected)


def compare_masked_gate(actual, expected, *, inputs, **kwargs):
    """Require exact masked residuals and the existing Engram text comparator."""
    from golden import ratio_allclose

    masked = inputs["lookup"].count_nonzero(dim=-1) == 0
    if not torch.equal(actual[masked], inputs["x"][masked]):
        return False, "masked FP32 residual changed"
    if bool(masked.all()):
        return True, ""
    return ratio_allclose(atol=1e-3, rtol=1e-2)(actual[~masked], expected[~masked], inputs=inputs, **kwargs)


@pl.jit
def l2_engram_masked_gate(
    kv: pl.Tensor[[T_DYN, KV_OUT], pl.BF16],
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    lookup: pl.Tensor[[T_DYN, ENGRAM_K], pl.BF16],
    out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    """Validate masked host-lookup gating over FP32 residual streams."""
    kv.bind_dynamic(0, T_DYN)
    x.bind_dynamic(0, T_DYN)
    lookup.bind_dynamic(0, T_DYN)
    out.bind_dynamic(0, T_DYN)
    engram_gate(kv, weight, x, out, lookup, mask_zero_rows=True)
    return out


@pl.jit
def l2_engram_rows(
    lookup: pl.Tensor[[T_DYN, ENGRAM_K], pl.BF16],
    wkv_weight: pl.Tensor[[ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[ENGRAM_K // 32, KV_OUT], pl.FP8E8M0],
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    x: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    """Validate the same in-place entry used by decode and prefill forwards."""
    lookup.bind_dynamic(0, T_DYN)
    x.bind_dynamic(0, T_DYN)
    engram_from_rows(lookup, wkv_weight, wkv_scale, weight, x, x, False)
    return x


@pl.jit
def l2_engram_rows_pypto(
    lookup: pl.Tensor[[T_DYN, ENGRAM_K], pl.BF16],
    wkv_weight: pl.Tensor[[ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[ENGRAM_K // 32, KV_OUT], pl.FP8E8M0],
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    x: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    """Validate the explicit PyPTO comparison implementation in place."""
    lookup.bind_dynamic(0, T_DYN)
    x.bind_dynamic(0, T_DYN)
    engram_from_rows_pypto(lookup, wkv_weight, wkv_scale, weight, x, x)
    return x


def build_engram_rows_specs(tokens: int):
    """Build deterministic host-row inputs with nonuniform packed MX scales."""
    from golden import TensorSpec
    from quant import pack_mx_b_scale

    generator = torch.Generator().manual_seed(3)

    def init_lookup():
        return (torch.randn(tokens, ENGRAM_K, generator=generator) * 0.2).bfloat16()

    def init_wkv():
        value = (torch.randn(ENGRAM_K, KV_OUT, generator=generator) * 0.02).to(torch.float8_e4m3fn)
        return value

    def init_scale():
        codes = torch.randint(120, 128, (ENGRAM_K // 32, KV_OUT), generator=generator, dtype=torch.uint8)
        return pack_mx_b_scale(codes).view(torch.float8_e8m0fnu)

    def init_weight():
        return torch.rand(HC_MULT, D, generator=generator) + 0.5

    def init_x():
        return torch.randn(tokens, HC_MULT, D, generator=generator)

    return [
        TensorSpec("lookup", [tokens, ENGRAM_K], torch.bfloat16, init_value=init_lookup),
        TensorSpec("wkv_weight", [ENGRAM_K, KV_OUT], torch.float8_e4m3fn, init_value=init_wkv),
        TensorSpec("wkv_scale", [ENGRAM_K // 32, KV_OUT], torch.float8_e8m0fnu, init_value=init_scale),
        TensorSpec("weight", [HC_MULT, D], torch.float32, init_value=init_weight),
        TensorSpec("x", [tokens, HC_MULT, D], torch.float32, init_value=init_x),
    ]


def golden_engram_rows_case(tensors):
    """Apply independent sequential MX projection and the released gate formula."""
    from gemm import golden_gemm_bf16_via_mxfp8_to_bf16_sequential

    wkv = tensors["wkv_weight"]
    kv = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(tensors["lookup"], wkv, tensors["wkv_scale"])
    key = kv[:, :HC_MULT * D].float().reshape(-1, HC_MULT, D)
    value = kv[:, HC_MULT * D:].float()
    h = tensors["x"].float()
    inverse = torch.rsqrt(h.square().mean(-1) + NORM_EPS) * torch.rsqrt(key.square().mean(-1) + NORM_EPS)
    dot = (h * tensors["weight"] * key).sum(-1) * inverse * DOT_SCALE
    root = dot.abs().clamp_min(CLAMP).sqrt()
    gate = torch.sigmoid(torch.where(dot >= 0, root, -root))
    tensors["x"][:] = (h + gate.unsqueeze(-1) * value.unsqueeze(-2)).bfloat16().float()


_ENGRAM_AUTO_SYNC = True


@pl.jit.inline(auto_scope=False)
def _engram_gate_row_reg(
    hidden: pl.Tile[[D], pl.FP32, pl.Mem.Vec],
    key: pl.Tile[[D], pl.BF16, pl.Mem.Vec],
    value: pl.Tile[[D], pl.BF16, pl.Mem.Vec],
    weight: pl.Tile[[D], pl.FP32, pl.Mem.Vec],
    output: pl.Tile[[D], pl.FP32, pl.Mem.Vec],
    auto_sync: pl.constexpr,
):
    """Apply the C++ lane accumulation, inverse-root correction and BF16 residual rounding."""
    with pl.reg_scope(auto_sync=auto_sync):
        full = pl.reg.full_mask(lanes=64)
        sx = pl.reg.broadcast(0.0, lanes=64, dtype=pl.FP32)
        sk = pl.reg.broadcast(0.0, lanes=64, dtype=pl.FP32)
        dot = pl.reg.broadcast(0.0, lanes=64, dtype=pl.FP32)
        for chunk in pl.hardware_loop(D // 64):
            xv = pl.reg.load(hidden, chunk * 64, lanes=64)
            kb = pl.reg.load(key, chunk * 64, lanes=64)
            kv = pl.reg.convert(kb, dtype=pl.FP32)
            wv = pl.reg.load(weight, chunk * 64, lanes=64)
            square_x = pl.reg.mul(xv, xv, full)
            sx = pl.reg.add(sx, square_x, full)
            square_k = pl.reg.mul(kv, kv, full)
            sk = pl.reg.add(sk, square_k, full)
            product = pl.reg.mul(xv, wv, full)
            product = pl.reg.mul(product, kv, full)
            dot = pl.reg.add(dot, product, full)
        sx = pl.reg.reduce_sum(sx, full, reassoc=True)
        sk = pl.reg.reduce_sum(sk, full, reassoc=True)
        dot = pl.reg.reduce_sum(dot, full, reassoc=True)
        sx = pl.reg.broadcast(sx, lanes=64, dtype=pl.FP32)
        sk = pl.reg.broadcast(sk, lanes=64, dtype=pl.FP32)
        dot = pl.reg.broadcast(dot, lanes=64, dtype=pl.FP32)
        sx = pl.reg.muls(sx, D_INV, full)
        sk = pl.reg.muls(sk, D_INV, full)
        x_variance = pl.reg.adds(sx, NORM_EPS, full)
        x_one = pl.reg.broadcast(1.0, lanes=64, dtype=pl.FP32)
        x_r = pl.reg.div(x_one, x_variance, full)
        x_root = pl.reg.sqrt(x_r, full)
        x_t = pl.reg.muls(x_variance, -0.5, full)
        x_t = pl.reg.mul(x_t, x_root, full)
        x_three_halves = pl.reg.broadcast(1.5, lanes=64, dtype=pl.FP32)
        x_factor = pl.reg.mula(x_three_halves, x_t, x_root, full)
        x_inverse = pl.reg.mul(x_root, x_factor, full)
        x_negative_variance = pl.reg.muls(x_variance, -1.0, full)
        x_correction = pl.reg.mula(x_one, x_negative_variance, x_r, full)
        x_negative_inverse = pl.reg.muls(x_inverse, -1.0, full)
        x_error = pl.reg.mula(x_r, x_negative_inverse, x_inverse, full)
        x_correction = pl.reg.mula(x_correction, x_variance, x_error, full)
        x_correction = pl.reg.mul(x_correction, x_inverse, full)
        x_half = pl.reg.broadcast(0.5, lanes=64, dtype=pl.FP32)
        x_inverse = pl.reg.mula(x_inverse, x_correction, x_half, full)
        x_bits = pl.reg.interpret_cast(x_variance, dtype=pl.UINT32)
        x_infinite = pl.reg.compares(x_bits, 2139095040, full, predicate="eq")
        x_zero = pl.reg.broadcast(0.0, lanes=64, dtype=pl.FP32)
        x_inverse = pl.reg.select(x_infinite, x_zero, x_inverse)
        k_variance = pl.reg.adds(sk, NORM_EPS, full)
        k_one = pl.reg.broadcast(1.0, lanes=64, dtype=pl.FP32)
        k_r = pl.reg.div(k_one, k_variance, full)
        k_root = pl.reg.sqrt(k_r, full)
        k_t = pl.reg.muls(k_variance, -0.5, full)
        k_t = pl.reg.mul(k_t, k_root, full)
        k_three_halves = pl.reg.broadcast(1.5, lanes=64, dtype=pl.FP32)
        k_factor = pl.reg.mula(k_three_halves, k_t, k_root, full)
        k_inverse = pl.reg.mul(k_root, k_factor, full)
        k_negative_variance = pl.reg.muls(k_variance, -1.0, full)
        k_correction = pl.reg.mula(k_one, k_negative_variance, k_r, full)
        k_negative_inverse = pl.reg.muls(k_inverse, -1.0, full)
        k_error = pl.reg.mula(k_r, k_negative_inverse, k_inverse, full)
        k_correction = pl.reg.mula(k_correction, k_variance, k_error, full)
        k_correction = pl.reg.mul(k_correction, k_inverse, full)
        k_half = pl.reg.broadcast(0.5, lanes=64, dtype=pl.FP32)
        k_inverse = pl.reg.mula(k_inverse, k_correction, k_half, full)
        k_bits = pl.reg.interpret_cast(k_variance, dtype=pl.UINT32)
        k_infinite = pl.reg.compares(k_bits, 2139095040, full, predicate="eq")
        k_zero = pl.reg.broadcast(0.0, lanes=64, dtype=pl.FP32)
        k_inverse = pl.reg.select(k_infinite, k_zero, k_inverse)
        dot = pl.reg.mul(dot, x_inverse, full)
        dot = pl.reg.mul(dot, k_inverse, full)
        dot = pl.reg.muls(dot, DOT_SCALE, full)
        root = pl.reg.abs(dot, full)
        root = pl.reg.maximums(root, CLAMP, full)
        root = pl.reg.sqrt(root, full)
        nonnegative = pl.reg.compares(dot, 0.0, full, predicate="ge")
        negative = pl.reg.muls(root, -1.0, full)
        root = pl.reg.select(nonnegative, root, negative)
        root = pl.reg.muls(root, -1.0, full)
        root = pl.reg.exp(root, full)
        root = pl.reg.adds(root, 1.0, full)
        gate = pl.reg.div(x_one, root, full)
        for chunk in pl.hardware_loop(D // 64):
            xv = pl.reg.load(hidden, chunk * 64, lanes=64)
            vb = pl.reg.load(value, chunk * 64, lanes=64)
            vv = pl.reg.convert(vb, dtype=pl.FP32)
            y = pl.reg.mul(vv, gate, full)
            y = pl.reg.add(xv, y, full)
            rounded = pl.reg.convert(y, dtype=pl.BF16, rounding="R", saturate=False)
            y = pl.reg.convert(rounded, dtype=pl.FP32)
            if not auto_sync:
                pl.reg.mem_bar("VLD_VST")
            _written = pl.reg.store(output, chunk * 64, y, full)
            if not auto_sync:
                pl.reg.mem_bar("VST_VLD")
    return output



@pl.jit.incore
def _engram_gate_reg(
    kv: pl.Tensor[[T_DYN, KV_OUT], pl.BF16],
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    x: pl.Tensor[[T_DYN * HC_MULT, D], pl.FP32],
    out: pl.Out[pl.Tensor[[T_DYN * HC_MULT, D], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX], auto_sync: pl.constexpr = True,
):
    """Run the per-copy gate with cached weights and the C++ worker traversal."""
    first = pl.tile.get_block_idx()
    workers = pl.tile.get_block_num()
    if first < tokens * HC_MULT:
        weight_raw = pl.tile.load(weight, [first % HC_MULT, 0], [1, D], target_memory=pl.Mem.Vec)
        for work in pl.range(first, tokens * HC_MULT, workers):
            if work != first and workers % HC_MULT != 0:
                weight_raw = pl.tile.load(weight, [work % HC_MULT, 0], [1, D], target_memory=pl.Mem.Vec)
            gain = pl.reshape(weight_raw, [D])
            hidden_raw = pl.tile.load(x, [work, 0], [1, D], target_memory=pl.Mem.Vec)
            key_raw = pl.tile.load(kv, [work // HC_MULT, work % HC_MULT * D], [1, D], target_memory=pl.Mem.Vec)
            value_raw = pl.tile.load(kv, [work // HC_MULT, HC_MULT * D], [1, D], target_memory=pl.Mem.Vec)
            hidden = pl.reshape(hidden_raw, [D])
            key = pl.reshape(key_raw, [D])
            value = pl.reshape(value_raw, [D])
            output = pl.create_tile([D], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            output = _engram_gate_row_reg(hidden, key, value, gain, output, auto_sync)
            pl.store(pl.reshape(output, [1, D]), [work, 0], out)
    return out



@pl.jit.inline(auto_scope=False)
def _load_engram_gate_pair(
    kv: pl.Tensor[[T_DYN, KV_OUT], pl.BF16],
    x: pl.Tensor[[T_DYN * (HC_MULT // 2), 2 * D], pl.FP32],
    hidden: pl.Tile[[2, 2 * D], pl.FP32, pl.Mem.Vec],
    key: pl.Tile[[2, 2 * D], pl.BF16, pl.Mem.Vec],
    value: pl.Tile[[2, D], pl.BF16, pl.Mem.Vec],
    work: pl.Scalar[pl.INDEX], slot: pl.Scalar[pl.INDEX],
):
    hidden = pl.tile.gather_row(hidden, x, [slot, 0], [work, 0], [1, 2 * D])
    key = pl.tile.gather_row(key, kv, [slot, 0], [work // 2, work % 2 * 2 * D], [1, 2 * D])
    value = pl.tile.gather_row(value, kv, [slot, 0], [work // 2, HC_MULT * D], [1, D])
    return hidden, key, value



@pl.jit.incore
def _engram_gate_pairs_reg(
    kv: pl.Tensor[[T_DYN, KV_OUT], pl.BF16],
    weight: pl.Tensor[[HC_MULT // 2, 2 * D], pl.FP32],
    x: pl.Tensor[[T_DYN * (HC_MULT // 2), 2 * D], pl.FP32],
    out: pl.Out[pl.Tensor[[T_DYN * HC_MULT, D], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX], auto_sync: pl.constexpr = True,
):
    """Prefetch the next pair into a separate UB slot while consuming the current pair."""
    first = pl.tile.get_block_idx()
    workers = pl.tile.get_block_num()
    total = tokens * 2
    if first < total:
        hidden = pl.create_tile([2, 2 * D], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        key = pl.create_tile([2, 2 * D], dtype=pl.BF16, target_memory=pl.Mem.Vec)
        value = pl.create_tile([2, D], dtype=pl.BF16, target_memory=pl.Mem.Vec)
        gain = pl.tile.load(weight, [first % 2, 0], [1, 2 * D], target_memory=pl.Mem.Vec)
        output = pl.create_tile([2, D], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        hidden, key, value = _load_engram_gate_pair(kv, x, hidden, key, value, first, 0)
        pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.V, 0)
        pl.system.set_flag(pl.PipeType.MTE3, pl.PipeType.V, 2)
        pl.system.set_flag(pl.PipeType.MTE3, pl.PipeType.V, 3)
        rounds = (total - first + workers - 1) // workers
        for group in pl.range((rounds + 1) // 2):
            for slot in pl.unroll(2):
                iteration = group * 2 + slot
                if iteration < rounds:
                    work = first + iteration * workers
                    next_slot = 1 - slot
                    if iteration > 0 and workers % 2 != 0:
                        pl.system.set_flag(pl.PipeType.V, pl.PipeType.MTE2, 4)
                        pl.system.wait_flag(pl.PipeType.V, pl.PipeType.MTE2, 4)
                        gain = pl.tile.gather_row(gain, weight, [0, 0], [work % 2, 0], [1, 2 * D])
                        pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.V, 4)
                        pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.V, 4)
                    if iteration + 1 < rounds:
                        if iteration > 0:
                            pl.system.wait_flag(pl.PipeType.V, pl.PipeType.MTE2, next_slot)
                        hidden, key, value = _load_engram_gate_pair(kv, x, hidden, key, value, work + workers, next_slot)
                        pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.V, next_slot)
                    pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.V, slot)
                    for copy in pl.unroll(2):
                        output_event = copy + 2
                        copy_offset = copy * D
                        pl.system.wait_flag(pl.PipeType.MTE3, pl.PipeType.V, output_event)
                        hidden_row = pl.reshape(pl.tile.slice(hidden, [1, D], [slot, copy_offset]), [D])
                        key_row = pl.reshape(pl.tile.slice(key, [1, D], [slot, copy_offset]), [D])
                        value_row = pl.reshape(pl.tile.slice(value, [1, D], [slot, 0]), [D])
                        gain_row = pl.reshape(pl.tile.slice(gain, [1, D], [0, copy_offset]), [D])
                        output_row = pl.reshape(pl.tile.slice(output, [1, D], [copy, 0]), [D])
                        output_row = _engram_gate_row_reg(hidden_row, key_row, value_row, gain_row, output_row, auto_sync)
                        pl.system.set_flag(pl.PipeType.V, pl.PipeType.MTE3, output_event)
                        pl.system.wait_flag(pl.PipeType.V, pl.PipeType.MTE3, output_event)
                        pl.store(pl.reshape(output_row, [1, D]), [work * 2 + copy, 0], out)
                        pl.system.set_flag(pl.PipeType.MTE3, pl.PipeType.V, output_event)
                    if iteration + 2 < rounds:
                        pl.system.set_flag(pl.PipeType.V, pl.PipeType.MTE2, slot)
        pl.system.wait_flag(pl.PipeType.MTE3, pl.PipeType.V, 2)
        pl.system.wait_flag(pl.PipeType.MTE3, pl.PipeType.V, 3)
        pl.system.barrier(pipe=pl.PipeType.ALL)
    return out



@pl.jit
def l2_engram_gate_vectors(
    kv: pl.Tensor, weight: pl.Tensor,
    x: pl.InOut[pl.Tensor], active: pl.Scalar[pl.INDEX],
):
    with pl.spmd(pl.system.available_aiv_count(), name_hint="engram_gate"):
        x = _engram_gate_native(kv, weight, x, x, active, 3)
    return x



@pl.jit
def l2_engram_gate_vectors_pypto(
    kv: pl.Tensor, weight: pl.Tensor,
    x: pl.InOut[pl.Tensor], active: pl.Scalar[pl.INDEX],
):
    with pl.spmd(pl.system.available_aiv_count(), name_hint="engram_gate"):
        x = _engram_gate_reg(kv, weight, x, x, active, _ENGRAM_AUTO_SYNC)
    return x



@pl.jit
def l2_engram_gate_vectors_pypto_pairs(
    kv: pl.Tensor, weight: pl.Tensor,
    x: pl.InOut[pl.Tensor], active: pl.Scalar[pl.INDEX],
):
    hidden_pairs = pl.reshape(x, [pl.tensor.dim(x, 0) // 2, 2 * D])
    weight_pairs = pl.reshape(weight, [HC_MULT // 2, 2 * D])
    with pl.spmd(pl.system.available_aiv_count(), name_hint="engram_gate_pairs"):
        x = _engram_gate_pairs_reg(kv, weight_pairs, hidden_pairs, x, active, _ENGRAM_AUTO_SYNC)
    return x



def golden_engram_gate_vectors(tensors, active):
    """Validate every active residual value and preserve the inactive tail."""
    key = tensors["kv"][:active, :HC_MULT * D].float().reshape(active, HC_MULT, D)
    value = tensors["kv"][:active, HC_MULT * D:].float()
    hidden = tensors["x"][:active * HC_MULT].float().reshape(active, HC_MULT, D)
    inv_x = torch.rsqrt(hidden.square().mean(-1) + NORM_EPS)
    inv_k = torch.rsqrt(key.square().mean(-1) + NORM_EPS)
    dot = (hidden * tensors["weight"] * key).sum(-1)
    dot = dot * inv_x * inv_k * DOT_SCALE
    root = dot.abs().clamp_min(CLAMP).sqrt()
    gate = torch.sigmoid(torch.where(dot >= 0, root, -root))
    tensors["x"][:active * HC_MULT] = (hidden + gate.unsqueeze(-1) * value.unsqueeze(-2)).bfloat16().float().reshape(active * HC_MULT, D)



def run_engram_gate_case(args):
    """Run one component with the normal golden harness and explicit capture settings."""
    from functools import partial
    from golden import ScalarSpec, TensorSpec, ratio_allclose, run

    tokens = args.tokens if args.tokens is not None else 17
    active = tokens if args.active is None else args.active
    if not 0 <= active <= tokens:
        raise ValueError("active rows must be between zero and the tensor row count")
    generator = torch.Generator().manual_seed(17)
    def key_values():
        values = torch.randn(tokens, KV_OUT, generator=generator).bfloat16()
        if args.pattern == "zero":
            values.zero_()
        return values
    def hidden_values():
        values = torch.randn(tokens, HC_MULT, D, generator=generator)
        if args.pattern == "zero":
            values.zero_()
        elif args.pattern == "mixed":
            values[::3] = 0
            values[1::3] *= 32
            values[2::3] *= 0.00001
        return values
    specs = [
        TensorSpec("kv", [tokens, KV_OUT], torch.bfloat16, init_value=key_values),
        TensorSpec("weight", [HC_MULT, D], torch.float32,
                   init_value=lambda: torch.rand(HC_MULT, D, generator=generator) + 0.5),
        TensorSpec("x", [tokens * HC_MULT, D], torch.float32, init_value=lambda: hidden_values().reshape(tokens * HC_MULT, D)),
        ScalarSpec("active", torch.int64, active),
    ]
    if args.implementation == "cpp":
        entry = l2_engram_gate_vectors
    else:
        # Match the C++ dispatch threshold for paired work.
        entry = l2_engram_gate_vectors_pypto_pairs if active >= 256 else l2_engram_gate_vectors_pypto
    reference = partial(golden_engram_gate_vectors, active=active)
    compare = {"x": ratio_allclose(atol=1e-3, rtol=1e-2, max_error_ratio=0.005)}
    return run(
        fn=entry, specs=specs, golden_fn=reference, compare_fn=compare,
        compile_only=args.compile_only, runtime_dir=args.runtime_dir, golden_data=args.golden_data,
        save_data=args.save_data,
        config={"backend_type": "a5", "execution_mode": args.execution_mode,
                "device_id": int(args.device), "enable_chip_swimlane": args.enable_chip_swimlane,
                "enable_dep_gen": bool(args.enable_chip_swimlane), "dump_passes": True},
    )


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 Engram validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="0",
        help="comma-separated device ids; provide one device id",
    )
    parser.add_argument("--masked-gate", action="store_true")
    parser.add_argument("--implementation", choices=["cpp", "pypto"], default="cpp",
                        help="cpp is the production default; pypto is for explicit comparison")
    parser.add_argument("--tokens", type=int, help="explicit row count; default checks tile boundaries and tails")
    parser.add_argument("--runtime-dir", help="reuse a CPU-prebuilt runtime; requires --tokens")
    parser.add_argument("--golden-data", help="reuse frozen input/reference tensors; requires --tokens")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--component", choices=["rows", "gate"], default="rows")
    parser.add_argument("--active", type=int)
    parser.add_argument("--pattern", choices=["random", "zero", "mixed"], default="random")
    parser.add_argument("--manual-sync", action="store_true")
    args = parser.parse_args()
    _ENGRAM_AUTO_SYNC = not args.manual_sync
    if args.component == "gate":
        result = run_engram_gate_case(args)
        if not result.passed:
            raise SystemExit(result.error or 1)
        raise SystemExit(0)

    device_ids = [int(d) for d in args.device.split(",")]
    if len(device_ids) != 1:
        raise ValueError(f"need exactly one device id, got {device_ids}")

    if args.tokens is not None and args.tokens <= 0:
        parser.error("--tokens must be positive")
    if (args.runtime_dir or args.golden_data or args.save_data) and args.tokens is None:
        parser.error("replay/capture options require an explicit --tokens case")
    if args.runtime_dir:
        from pypto.runtime._artifact_sources import read_kernel_config

        kernels = read_kernel_config(Path(args.runtime_dir) / "kernel_config.py").KERNELS
        native_runtime = any(kernel.get("external", False) for kernel in kernels)
        if native_runtime != (args.implementation == "cpp"):
            parser.error("--runtime-dir implementation does not match --implementation")
    entry = l2_engram_rows if args.implementation == "cpp" else l2_engram_rows_pypto
    token_cases = [17, 81, 129, 256] if args.execution_mode else [17, 80, 81, 128, 129, 256, 257]
    if args.tokens is not None:
        token_cases = [args.tokens]
    for tokens in ([] if args.masked_gate else token_cases):
        specs = build_engram_rows_specs(tokens)
        result = run(
            fn=entry,
            specs=specs,
            golden_fn=golden_engram_rows_case,
            config={
                "backend_type": args.backend_type, "execution_mode": args.execution_mode,
                "device_id": device_ids[0],
                "enable_chip_swimlane": args.enable_chip_swimlane,
            },
            rtol=1e-3,
            atol=1e-3,
            compare_fn={"x": ratio_allclose(atol=1e-3, rtol=1e-2)},
            compile_only=args.compile_only,
            runtime_dir=args.runtime_dir,
            golden_data=args.golden_data,
            save_data=args.save_data,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)

    result = run(
        fn=l2_engram_masked_gate,
        specs=build_masked_gate_specs(8),
        golden_fn=golden_masked_gate,
        config={"backend_type": args.backend_type, "execution_mode": args.execution_mode,
                "device_id": device_ids[0]},
        compare_fn={"out": compare_masked_gate},
        compile_only=args.compile_only,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
