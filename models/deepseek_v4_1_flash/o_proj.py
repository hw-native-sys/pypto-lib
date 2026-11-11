# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Re-enable CPU validation when flat MX scale packing is corrected:
# https://github.com/hw-native-sys/pto-isa/issues/357
"""V4.1 inverse RoPE, grouped W8A8 MXFP8 output projection, and TP reductions."""

from pathlib import Path

import torch

from quant import MX_GROUP, _quantize_mxfp8_activation, decode_e8m0, unpack_mx_b_scale

from qkv_proj_rope import golden_rope_interleave as rope_interleave
from gemm import golden_gemm_bf16_via_mxfp8_to_bf16_sequential, golden_mxfp8_linear_grouped_sum


import pypto.language as pl
from pypto.runtime import pto_isa_include_dir
from hc import hc_post
import pypto.language.distributed as pld

from config import (
    PREFILL_MAX_TOKENS,
    FLASH as M, SP_T_DYN, T_DYN, TP_SIZE,
)


# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
LOCAL_O_GROUPS = M.o_groups // TP_SIZE
LOCAL_O_WIDTH = M.o_groups // TP_SIZE * M.o_lora_rank
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
O_LORA = M.o_lora_rank
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim


O_GROUP_ROWS = pl.dynamic("O_GROUP_ROWS")
O_PAD_ROWS = pl.dynamic("O_PAD_ROWS")
O_IN_SCALE_COLS = pl.dynamic("O_IN_SCALE_COLS")
O_OUT_SCALE_COLS = pl.dynamic("O_OUT_SCALE_COLS")

# tiling
# TP1 decode retains its measured tiles and worker counts.
O_VECTOR_WORKERS = 64
O_A_N_TILE = 256
O_B_N_TILE = 160 if LOCAL_O_WIDTH == 8192 else 256
SP_REDUCE_ROW_TILE = 2
SP_PUBLISH_ROW_TILE = 16
NATIVE_VALIDATION_WORKERS = 32


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels" / "o_quant_vf.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _quant_o_vf(
    attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    cosine: pl.Tensor,
    sine: pl.Tensor,
    payload: pl.Out[pl.Tensor[[O_GROUP_ROWS, O_GROUP_IN], pl.FP8E4M3FN]],
    scales: pl.Out[pl.Tensor[[LOCAL_O_GROUPS, O_IN_SCALE_COLS], pl.FP8E8M0]],
    active_rows: pl.Scalar[pl.INT32],
    groups: pl.Scalar[pl.INT32],
    padded_rows: pl.Scalar[pl.INDEX],
) -> tuple[pl.Tensor, pl.Tensor]:
    """Inverse-rotate BF16 tails and quantize projection inputs to OCP MXFP8."""
    # Preserve both extern Out values: https://github.com/hw-native-sys/pypto/issues/3249
    return payload, scales


@pl.jit.extern(core_type="mixed", aic_source=Path(__file__).parent / "kernels" / "o_proj_a.cpp",
               aiv_source=Path(__file__).parent / "kernels" / "o_proj_a.cpp",
               include_dirs=(pto_isa_include_dir(),), dual_aiv_dispatch=True)
def _o_proj_a(
    payload: pl.Tensor[[O_GROUP_ROWS, O_GROUP_IN], pl.FP8E4M3FN],
    weight: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    activation_scale: pl.Tensor[[LOCAL_O_GROUPS, O_IN_SCALE_COLS], pl.FP8E8M0],
    weight_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    output: pl.Out[pl.Tensor[[O_PAD_ROWS, LOCAL_O_WIDTH], pl.FP8E4M3FN]],
    output_scale: pl.Out[pl.Tensor[[1, O_OUT_SCALE_COLS], pl.FP8E8M0]],
    output_width: pl.Scalar[pl.INDEX],
    active_rows: pl.Scalar[pl.INT32],
    padded_rows: pl.Scalar[pl.INDEX],
    row_tile: pl.Scalar[pl.INDEX],
) -> tuple[pl.Tensor, pl.Tensor]:
    """Round the O-A result to BF16 and quantize on the paired Vector cores."""
    return output, output_scale


@pl.jit.extern(core_type="aic", source=Path(__file__).parent / "kernels" / "o_proj_b.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _o_proj_b(
    payload: pl.Tensor[[O_PAD_ROWS, LOCAL_O_WIDTH], pl.FP8E4M3FN],
    weight: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    activation_scale: pl.Tensor[[1, O_OUT_SCALE_COLS], pl.FP8E8M0],
    weight_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    output: pl.Out[pl.Tensor],
    rows: pl.Scalar[pl.INDEX], reduction: pl.Scalar[pl.INDEX],
    row_tile: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    """Project MXFP8 rows with packed MX_A_ZZ and MX_B_NN scales."""
    ...


@pl.jit.inline(auto_scope=False)
def o_proj(
    attended: pl.Tensor,
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    worker_limit: pl.constexpr = 32,
    oa_row_tile: pl.constexpr = 80,
    ob_row_tile: pl.constexpr = 80,
):
    """Project row groups to BF16; both weights hold NZ bytes in logical [K, N] shapes."""
    total_rows = pl.tensor.dim(output, 0)
    # Native row tiles: OA M80/M128 and OB M80/M256.
    ob_padded_rows = ((total_rows + ob_row_tile - 1) // ob_row_tile) * ob_row_tile
    padded_rows = ((ob_padded_rows + oa_row_tile - 1) // oa_row_tile) * oa_row_tile
    payload_all = pl.create_tensor([LOCAL_O_GROUPS * padded_rows, O_GROUP_IN], dtype=pl.FP8E4M3FN)
    scales_all = pl.create_tensor([LOCAL_O_GROUPS, padded_rows * (O_GROUP_IN // 32)], dtype=pl.FP8E8M0)
    vector_workers = pl.max(1, pl.min(O_VECTOR_WORKERS, 2 * pl.system.available_cluster_count()))
    with pl.spmd(vector_workers, name_hint="quant_o_vf"):
        payload_all, scales_all = _quant_o_vf(
            attended, rope_cos, rope_sin, payload_all, scales_all, num_tokens, LOCAL_O_GROUPS, padded_rows,
        )
    payload = pl.create_tensor([padded_rows, LOCAL_O_WIDTH], dtype=pl.FP8E4M3FN)
    activation_scale = pl.create_tensor([1, padded_rows * (LOCAL_O_WIDTH // 32)], dtype=pl.FP8E8M0)
    oa_tasks = (padded_rows // oa_row_tile) * (LOCAL_O_WIDTH // O_A_N_TILE)
    oa_workers = pl.max(1, pl.min(oa_tasks, pl.system.available_cluster_count()))
    with pl.spmd(oa_workers, name_hint="proj_o_a_quant"):
        payload, activation_scale = _o_proj_a(
            payload_all, wo_a, scales_all, wo_a_scale, payload, activation_scale,
            LOCAL_O_WIDTH, num_tokens, padded_rows, oa_row_tile,
        )
    ob_tasks = ((total_rows + ob_row_tile - 1) // ob_row_tile) * (D // O_B_N_TILE)
    ob_workers = pl.max(1, pl.min(ob_tasks, worker_limit))
    with pl.spmd(ob_workers, name_hint="proj_o_b"):
        output = _o_proj_b(payload, wo_b, activation_scale, wo_b_scale, output, total_rows, LOCAL_O_WIDTH, ob_row_tile)
    return output


@pl.jit.inline(auto_scope=False)
def o_quant_vf_pypto(
    attended: pl.Tensor,
    cosine: pl.Tensor,
    sine: pl.Tensor,
    output: pl.Tensor,
    output_scale: pl.Tensor,
    active_rows: pl.Scalar[pl.INT32],
    groups: pl.Scalar[pl.INT32],
    padded_rows: pl.Scalar[pl.INDEX],
):
    padded = padded_rows
    scales = pl.reshape(output_scale, [groups * padded // 16, 2048])
    for task in pl.spmd(padded // 80 * (groups * 8)):
        row = task // (groups * 8) * 80
        head = task % (groups * 8)
        group = head // 8
        local_head = head % 8
        valid_rows = pl.max(0, pl.min(80, active_rows - row))
        load_rows = pl.max(1, valid_rows)
        values = pl.load(attended, [row, head * 512], [80, 512], valid_shape=[load_rows, 512], target_memory=pl.Mem.Vec)
        cv = pl.load(cosine, [row, 0], [80, 32], valid_shape=[load_rows, 32], target_memory=pl.Mem.Vec)
        sv = pl.load(sine, [row, 0], [80, 32], valid_shape=[load_rows, 32], target_memory=pl.Mem.Vec)
        quant_bytes = pl.create_tile([80 * 512], dtype=pl.INT8, target_memory=pl.Mem.Vec)
        scale_words = pl.create_tile([80 * 512 // 32], dtype=pl.INT32, target_memory=pl.Mem.Vec)
        scale_bytes = pl.create_tile([80 * 512 // 32], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
        with pl.reg_scope():
            q_full = pl.reg.full_mask(lanes=64)
            q_lane = pl.reg.index_gen(0, lanes=64, dtype=pl.UINT32)
            q_low_mask = pl.reg.active_mask(32, lanes=64)
            q_high_mask = pl.reg.compares(q_lane, 32, q_full, predicate="ge")
            q_one = pl.reg.broadcast(1, lanes=64, dtype=pl.UINT32)
            q_zero = pl.reg.broadcast(0.0, lanes=64, dtype=pl.FP32)
            q_zero_words = pl.reg.broadcast(0, lanes=64, dtype=pl.INT32)
            pair = pl.reg.shift_rights(q_lane, 1, q_full)
            swap = pl.reg.bitwise_xor(q_lane, q_one)
            odd_bit = pl.reg.bitwise_and(q_lane, q_one)
            is_even = pl.reg.compares(odd_bit, 0, q_full, predicate="eq")
            for q_init_chunk in pl.hardware_loop(80 * 512 // 32 // 64):
                _q_initialized = pl.reg.store(scale_words, offset=q_init_chunk * 64, value=q_zero_words)
            for q_r in pl.hardware_loop(valid_rows):
                trig_offset = pl.cast(q_r * 32, pl.UINT32)
                trig_index = pl.reg.adds(pair, trig_offset, q_full)
                c = pl.reg.gather(cv, trig_index, q_full)
                s = pl.reg.gather(sv, trig_index, q_full)
                s = pl.reg.muls(s, -1.0, q_full)
                raw = pl.reg.load(values, offset=q_r * 512 + 448, lanes=64)
                x = pl.reg.convert(raw, dtype=pl.FP32)
                swapped = pl.reg.gather_lanes(x, swap)
                a = pl.reg.mul(x, c)
                b = pl.reg.mul(swapped, s)
                real = pl.reg.sub(a, b)
                imag = pl.reg.add(b, a)
                rotated = pl.reg.select(is_even, real, imag)
                bf = pl.reg.convert(rotated, dtype=pl.BF16, rounding="R", saturate=False)
                rounded_tail = pl.reg.convert(bf, dtype=pl.FP32)
                for q_block in pl.hardware_loop(512 // 256):
                    q_base = q_r * 512 + q_block * 256
                    q_raw0 = pl.reg.load(values, offset=q_base, lanes=64)
                    q_raw1 = pl.reg.load(values, offset=q_base + 64, lanes=64)
                    q_raw2 = pl.reg.load(values, offset=q_base + 128, lanes=64)
                    q_raw3 = pl.reg.load(values, offset=q_base + 192, lanes=64)
                    for q_sub in pl.unroll(4):
                        q_chunk = q_block * 4 + q_sub
                        q_offset = q_r * 512 + q_chunk * 64
                        if q_sub == 0:
                            q_raw = q_raw0
                        elif q_sub == 1:
                            q_raw = q_raw1
                        elif q_sub == 2:
                            q_raw = q_raw2
                        else:
                            q_raw = q_raw3
                        q_input = pl.reg.convert(q_raw, dtype=pl.FP32)
                        chunk32 = pl.cast(q_chunk, pl.INT32)
                        chunk_reg = pl.reg.broadcast(chunk32, lanes=64, dtype=pl.INT32)
                        is_tail = pl.reg.compares(chunk_reg, 7, q_full, predicate="eq")
                        q_x = pl.reg.select(is_tail, rounded_tail, q_input)
                        q_magnitude = pl.reg.abs(q_x)
                        q_lo = pl.reg.reduce_max(q_magnitude, q_low_mask)
                        q_hi = pl.reg.reduce_max(q_magnitude, q_high_mask)
                        q_lo_full = pl.reg.broadcast(q_lo, lanes=64, dtype=pl.FP32)
                        q_hi_full = pl.reg.broadcast(q_hi, lanes=64, dtype=pl.FP32)
                        q_maximum = pl.reg.select(q_low_mask, q_lo_full, q_hi_full)
                        q_max_bits = pl.reg.interpret_cast(q_maximum, dtype=pl.INT32)
                        q_exponent = pl.reg.shift_rights(q_max_bits, 23, q_full)
                        q_exponent = pl.reg.adds(q_exponent, -8, q_full)
                        q_exponent = pl.reg.maximums(q_exponent, 0, q_full)
                        q_nonzero = pl.reg.maximums(q_exponent, 1, q_full)
                        q_inverse_exp = pl.reg.broadcast(254, lanes=64, dtype=pl.INT32)
                        q_inverse_exp = pl.reg.sub(q_inverse_exp, q_nonzero)
                        q_reciprocal_bits = pl.reg.shift_lefts(q_inverse_exp, 23, q_full)
                        q_reciprocal = pl.reg.interpret_cast(q_reciprocal_bits, dtype=pl.FP32)
                        q_scaled = pl.reg.mul(q_x, q_reciprocal)
                        q_active = pl.reg.compares(q_exponent, 0, q_full, predicate="gt")
                        q_factor = pl.reg.broadcast(1.0, lanes=64, dtype=pl.FP32)
                        q_factor = pl.reg.select(q_active, q_factor, q_zero)
                        q_scaled = pl.reg.mul(q_scaled, q_factor)
                        q_scaled = pl.reg.maximums(q_scaled, -448.0, q_full)
                        q_scaled = pl.reg.minimums(q_scaled, 448.0, q_full)
                        q_fp8 = pl.reg.convert(q_scaled, dtype=pl.FP8E4M3FN, rounding="R", saturate=False)
                        q_byte = pl.reg.interpret_cast(q_fp8, dtype=pl.INT8)
                        _q_quant_written = pl.reg.store(quant_bytes, offset=q_offset, value=q_byte)
                        q_low_bit = pl.reg.bitwise_and(q_lane, q_one)
                        q_pair_index = pl.reg.shift_lefts(q_low_bit, 5, q_full)
                        q_pair_scale = pl.reg.gather_lanes(q_exponent, q_pair_index)
                        q_scale_base = q_r // 16 * (512 // 32 * 16) + q_chunk * 32 + q_r % 16 * 2
                        q_scale_base32 = pl.cast(q_scale_base, pl.UINT32)
                        q_scale_offsets = pl.reg.adds(q_lane, q_scale_base32, q_full)
                        q_two = pl.reg.active_mask(2, lanes=64)
                        _q_scale_scattered = pl.reg.scatter(scale_words, q_scale_offsets, q_pair_scale, q_two)
            empty = pl.reg.broadcast(0, lanes=256, dtype=pl.INT8)
            for inactive in pl.hardware_loop(80 - valid_rows):
                pad_row = valid_rows + inactive
                for pad_chunk in pl.hardware_loop(2):
                    pad_offset = pad_row * 512 + pad_chunk * 256
                    _padding_written = pl.reg.store(quant_bytes, offset=pad_offset, value=empty)
            for q_pack in pl.hardware_loop(80 * 512 // 32 // 64):
                q_words = pl.reg.load(scale_words, offset=q_pack * 64, lanes=64)
                q_packed_signed = pl.reg.convert(q_words, dtype=pl.INT8, saturate=False)
                q_packed_unsigned = pl.reg.interpret_cast(q_packed_signed, dtype=pl.UINT8)
                _q_scale_written = pl.reg.store(scale_bytes, offset=q_pack * 64, value=q_packed_unsigned)
        quantized = pl.reshape(quant_bytes, [80, 512])
        quantized = pl.reinterpret_view(quantized, pl.FP8E4M3FN)
        scale = pl.reshape(scale_bytes, [5, 512 // 2])
        scale = pl.reinterpret_view(scale, pl.FP8E8M0)
        output = pl.store(quantized, [group * padded + row, local_head * 512], output)
        packed = pl.reshape(scale, [5, 256])
        scales = pl.store(packed, [group * padded // 16 + row // 16, local_head * 256], scales)
    return output, output_scale


@pl.jit.inline(auto_scope=False)
def o_proj_a_pypto(
    payload: pl.Tensor,
    weight: pl.Tensor,
    activation_scale: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.Tensor,
    output_scale: pl.Tensor,
    output_width: pl.Scalar[pl.INDEX],
    active_rows: pl.Scalar[pl.INT32],
    padded_rows: pl.Scalar[pl.INDEX],
):
    projected = pl.create_tensor([padded_rows, output_width], dtype=pl.FP32)
    for group in pl.parallel(output_width // 1024):
        w: pl.Tensor[[4096, 1024], pl.FP8E4M3FN, pl.NZ] = weight[group]
        sw = weight_scale[group]
        sw = pl.tensor.view(sw, [128, 1024], layout=pl.MX_B_NN)
        for mt in pl.parallel(padded_rows // 80):
            row = mt * 80
            scale_start = row * 128
            scale_end = scale_start + 80 * 128
            sa = activation_scale[group, scale_start:scale_end]
            sa = pl.reshape(sa, [80, 128])
            sa = pl.tensor.view(sa, [80, 128], layout=pl.MX_A_ZZ)
            for nb in pl.spmd(4):
                col = nb * 256
                a0 = pl.load(payload, [group * padded_rows + row, 0], [80, 256], target_memory=pl.Mem.Mat)
                sa0 = pl.load(sa, [0, 0], [80, 8])
                b0 = pl.load(w, [0, col], [256, 256], target_memory=pl.Mem.Mat)
                sb0 = pl.load(sw, [0, col], [8, 256])
                acc = pl.matmul_mx(a0, sa0, b0, sb0)
                for k in pl.pipeline(256, 4096, 256, stage=2):
                    av = pl.load(payload, [group * padded_rows + row, k], [80, 256], target_memory=pl.Mem.Mat)
                    sav = pl.load(sa, [0, k // 32], [80, 8])
                    bv = pl.load(w, [k, col], [256, 256], target_memory=pl.Mem.Mat)
                    sbv = pl.load(sw, [k // 32, col], [8, 256])
                    acc = pl.matmul_mx_acc(acc, av, sav, bv, sbv)
                projected = pl.store(acc, [row, group * 1024 + col], projected)
    scale_rows = padded_rows // 16
    scale_cols = output_width // 2
    scales = pl.reshape(output_scale, [scale_rows, scale_cols])
    column_tiles = output_width // 256
    for task in pl.spmd(padded_rows // 80 * column_tiles):
        row = task // column_tiles * 80
        group = task % column_tiles // 4
        nb = task % 4
        col = nb * 256
        values = pl.load(projected, [row, group * 1024 + col], [80, 256], target_memory=pl.Mem.Vec)
        valid_rows = pl.max(0, pl.min(80, active_rows - row))
        flat = pl.reshape(values, [80 * 256])
        quant_bytes = pl.create_tile([80 * 256], dtype=pl.INT8, target_memory=pl.Mem.Vec)
        scale_words = pl.create_tile([80 * 256 // 32], dtype=pl.INT32, target_memory=pl.Mem.Vec)
        scale_bytes = pl.create_tile([80 * 256 // 32], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
        with pl.reg_scope():
            full = pl.reg.full_mask(lanes=64)
            lane = pl.reg.index_gen(0, lanes=64, dtype=pl.UINT32)
            one = pl.reg.broadcast(1, lanes=64, dtype=pl.UINT32)
            zero = pl.reg.broadcast(0.0, lanes=64, dtype=pl.FP32)
            zero_words = pl.reg.broadcast(0, lanes=64, dtype=pl.INT32)
            for init_chunk in pl.hardware_loop(80 * 256 // 32 // 64):
                _initialized = pl.reg.store(scale_words, offset=init_chunk * 64, value=zero_words)
            for r in pl.hardware_loop(80):
                for chunk in pl.hardware_loop(256 // 64):
                    offset = r * 256 + chunk * 64
                    raw = pl.reg.load(flat, offset=offset, lanes=64)
                    bf = pl.reg.convert(raw, dtype=pl.BF16, rounding="R", saturate=False)
                    x = pl.reg.convert(bf, dtype=pl.FP32)
                    row32 = pl.cast(r, pl.INT32)
                    limit32 = pl.cast(valid_rows, pl.INT32)
                    row_reg = pl.reg.broadcast(row32, lanes=64, dtype=pl.INT32)
                    row_live = pl.reg.compares(row_reg, limit32, full, predicate="lt")
                    x = pl.reg.select(row_live, x, zero)
                    magnitude = pl.reg.abs(x)
                    maximum = magnitude
                    xor_bit = pl.reg.broadcast(1, lanes=64, dtype=pl.UINT32)
                    partner = pl.reg.bitwise_xor(lane, xor_bit)
                    other = pl.reg.gather_lanes(maximum, partner)
                    maximum = pl.reg.maximum(maximum, other)
                    xor_bit = pl.reg.broadcast(2, lanes=64, dtype=pl.UINT32)
                    partner = pl.reg.bitwise_xor(lane, xor_bit)
                    other = pl.reg.gather_lanes(maximum, partner)
                    maximum = pl.reg.maximum(maximum, other)
                    xor_bit = pl.reg.broadcast(4, lanes=64, dtype=pl.UINT32)
                    partner = pl.reg.bitwise_xor(lane, xor_bit)
                    other = pl.reg.gather_lanes(maximum, partner)
                    maximum = pl.reg.maximum(maximum, other)
                    xor_bit = pl.reg.broadcast(8, lanes=64, dtype=pl.UINT32)
                    partner = pl.reg.bitwise_xor(lane, xor_bit)
                    other = pl.reg.gather_lanes(maximum, partner)
                    maximum = pl.reg.maximum(maximum, other)
                    xor_bit = pl.reg.broadcast(16, lanes=64, dtype=pl.UINT32)
                    partner = pl.reg.bitwise_xor(lane, xor_bit)
                    other = pl.reg.gather_lanes(maximum, partner)
                    maximum = pl.reg.maximum(maximum, other)
                    max_bits = pl.reg.interpret_cast(maximum, dtype=pl.INT32)
                    exponent = pl.reg.shift_rights(max_bits, 23, full)
                    exponent = pl.reg.adds(exponent, -8, full)
                    exponent = pl.reg.maximums(exponent, 0, full)
                    nonzero = pl.reg.maximums(exponent, 1, full)
                    divisor_bits = pl.reg.shift_lefts(nonzero, 23, full)
                    divisor = pl.reg.interpret_cast(divisor_bits, dtype=pl.FP32)
                    scaled = pl.reg.div(x, divisor)
                    active = pl.reg.compares(exponent, 0, full, predicate="gt")
                    factor = pl.reg.broadcast(1.0, lanes=64, dtype=pl.FP32)
                    factor = pl.reg.select(active, factor, zero)
                    scaled = pl.reg.mul(scaled, factor)
                    scaled = pl.reg.maximums(scaled, -448.0, full)
                    scaled = pl.reg.minimums(scaled, 448.0, full)
                    fp8 = pl.reg.convert(scaled, dtype=pl.FP8E4M3FN, rounding="R", saturate=False)
                    byte = pl.reg.interpret_cast(fp8, dtype=pl.INT8)
                    _quant_written = pl.reg.store(quant_bytes, offset=offset, value=byte)
                    low_bit = pl.reg.bitwise_and(lane, one)
                    pair_index = pl.reg.shift_lefts(low_bit, 5, full)
                    pair_scale = pl.reg.gather_lanes(exponent, pair_index)
                    scale_base = r // 16 * (256 // 32 * 16) + chunk * 32 + r % 16 * 2
                    scale_base32 = pl.cast(scale_base, pl.UINT32)
                    scale_offsets = pl.reg.adds(lane, scale_base32, full)
                    two = pl.reg.active_mask(2, lanes=64)
                    _scale_scattered = pl.reg.scatter(scale_words, scale_offsets, pair_scale, two)
            for pack in pl.hardware_loop(80 * 256 // 32 // 64):
                words = pl.reg.load(scale_words, offset=pack * 64, lanes=64)
                packed_signed = pl.reg.convert(words, dtype=pl.INT8, saturate=False)
                packed_unsigned = pl.reg.interpret_cast(packed_signed, dtype=pl.UINT8)
                _scale_written = pl.reg.store(scale_bytes, offset=pack * 64, value=packed_unsigned)
        quantized = pl.reshape(quant_bytes, [80, 256])
        quantized = pl.reinterpret_view(quantized, pl.FP8E4M3FN)
        scale = pl.reshape(scale_bytes, [5, 256 // 2])
        scale = pl.reinterpret_view(scale, pl.FP8E8M0)
        output = pl.store(quantized, [row, group * 1024 + col], output)
        packed = pl.reshape(scale, [5, 128])
        scales = pl.store(packed, [row // 16, group * 512 + nb * 128], scales)
    return output, output_scale


@pl.jit.inline(auto_scope=False)
def o_proj_b_pypto(
    payload: pl.Tensor,
    weight: pl.Tensor,
    activation_scale: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.Tensor,
    rows: pl.Scalar[pl.INDEX],
    reduction: pl.Scalar[pl.INDEX],
    worker_limit: pl.constexpr = 32,
):
    scale_width = weight.shape[0] // 32
    for mt in pl.parallel((rows + 79) // 80):
        scale_begin = mt * 80 * scale_width
        scale_end = scale_begin + 80 * scale_width
        sa = activation_scale[0, scale_begin:scale_end]
        sa = pl.reshape(sa, [80, scale_width])
        sa = pl.tensor.view(sa, [80, scale_width], layout=pl.MX_A_ZZ)
        for worker in pl.spmd(pl.min(worker_limit, 32)):
            for base in pl.range(0, 32, worker_limit):
                if base + worker < 32:
                    row = mt * 80
                    col = (base + worker) * 160
                    a0 = pl.load(payload, [row, 0], [80, 256], target_memory=pl.Mem.Mat)
                    sa0 = pl.load(sa, [0, 0], [80, 8])
                    b0 = pl.load(weight, [0, col], [256, 160], target_memory=pl.Mem.Mat)
                    sb0 = pl.load(weight_scale, [0, col], [8, 160])
                    acc = pl.matmul_mx(a0, sa0, b0, sb0)
                    for k in pl.pipeline(256, reduction, 256, stage=2):
                        av = pl.load(payload, [row, k], [80, 256], target_memory=pl.Mem.Mat)
                        sav = pl.load(sa, [0, k // 32], [80, 8])
                        bv = pl.load(weight, [k, col], [256, 160], target_memory=pl.Mem.Mat)
                        sbv = pl.load(weight_scale, [k // 32, col], [8, 160])
                        acc = pl.matmul_mx_acc(acc, av, sav, bv, sbv)
                    valid = pl.set_validshape(acc, pl.min(80, rows - row), 160)
                    output = pl.store(valid, [row, col], output)
    return output


@pl.jit.inline(auto_scope=False)
def o_proj_pypto(
    attended: pl.Tensor,
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    worker_limit: pl.constexpr = 32,
):
    """Project row groups to BF16; both weights hold NZ bytes in logical [K, N] shapes."""
    total_rows = pl.tensor.dim(output, 0)
    padded_rows = (total_rows + 79) // 80 * 80
    payload_all = pl.create_tensor([LOCAL_O_GROUPS * padded_rows, O_GROUP_IN], dtype=pl.FP8E4M3FN)
    scales_all = pl.create_tensor([LOCAL_O_GROUPS, padded_rows * (O_GROUP_IN // 32)], dtype=pl.FP8E8M0)
    quant_payload, quant_scales = o_quant_vf_pypto(
        attended, rope_cos, rope_sin, payload_all, scales_all, num_tokens, LOCAL_O_GROUPS, padded_rows,
    )
    payload = pl.create_tensor([padded_rows, LOCAL_O_WIDTH], dtype=pl.FP8E4M3FN)
    activation_scale = pl.create_tensor([1, padded_rows * (LOCAL_O_WIDTH // 32)], dtype=pl.FP8E8M0)
    projected_payload, projected_scale = o_proj_a_pypto(
        quant_payload, wo_a, quant_scales, wo_a_scale,
        payload, activation_scale, LOCAL_O_WIDTH, num_tokens, padded_rows,
    )
    output = o_proj_b_pypto(
        projected_payload, wo_b, projected_scale, wo_b_scale, output, total_rows, LOCAL_O_WIDTH, worker_limit,
    )
    return output


@pl.jit
def o_native(
    attended: pl.Tensor,
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, 4096, 1024], pl.FP8E4M3FN, pl.NZ],
    wo_a_scale: pl.Tensor,
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN, pl.NZ],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    output: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
):
    output = o_proj_pypto(
        attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        output, num_tokens, worker_limit=NATIVE_VALIDATION_WORKERS,
    )
    return output


@pl.jit
def o_cpp(
    attended: pl.Tensor,
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, 4096, 1024], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor,
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    output: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
):
    output = o_proj(
        attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        output, num_tokens, worker_limit=NATIVE_VALIDATION_WORKERS, oa_row_tile=80, ob_row_tile=80,
    )
    return output



def grouped_mxfp8_linear(
    grouped: torch.Tensor,
    wo_a: torch.Tensor,
    wo_a_scale: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Evaluate grouped W8A8 MXFP8 with NZ weights and MX_B_NN scales."""
    from quant import unpack_fp8_nz

    wo_a = unpack_fp8_nz(wo_a)
    activation, activation_scale = _quantize_mxfp8_activation(grouped, scale_alg=0)
    weight_scale = decode_e8m0(unpack_mx_b_scale(wo_a_scale))
    weight_groups = wo_a.float().unflatten(-2, (-1, MX_GROUP))
    partials = torch.einsum("tgmk,gmkr->tgmr", activation.float(), weight_groups)
    partials = partials * activation_scale.unsqueeze(-1) * weight_scale.unsqueeze(0)
    return partials.sum(dim=2).to(output_dtype or grouped.dtype)

def golden_grouped_output(
    grouped: torch.Tensor, wo_a: torch.Tensor, wo_a_scale: torch.Tensor,
) -> torch.Tensor:
    """Grouped W8A8 MXFP8 output GEMM with one BF16 output cast."""
    return grouped_mxfp8_linear(grouped, wo_a, wo_a_scale, output_dtype=torch.bfloat16)


def golden_o_proj(
    attended: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor | None,
    *,
    wo_a_scale: torch.Tensor,
    output_dtype: torch.dtype | None = None,
    sequential_projection: bool = False,
) -> torch.Tensor:
    from quant import unpack_fp8_nz

    wo_b = unpack_fp8_nz(wo_b)
    rope_dim = cos.shape[-1] * 2
    tail = rope_interleave(attended[..., -rope_dim:], cos, sin, inverse=True)
    attended = torch.cat((attended[..., :-rope_dim], tail), dim=-1)
    groups = wo_a.shape[0]
    grouped = attended.flatten(-2).unflatten(-1, (groups, -1))
    latent = golden_grouped_output(grouped, wo_a, wo_a_scale)
    if sequential_projection and wo_b_scale is not None:
        projected = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(latent.flatten(-2), wo_b, wo_b_scale)
    else:
        projected = golden_mxfp8_linear_grouped_sum(latent.flatten(-2), wo_b, wo_b_scale, output_dtype=torch.bfloat16)
    return projected.to(output_dtype or torch.bfloat16)


@pl.jit.inline(auto_scope=False)
def retire_prefill_signals(
    completion_anchor: pl.Tensor,
    arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Clear ready and consumed signals at a serialized entry boundary."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_signal_retire", allow_early_resolve=False):
        anchor = pl.read(completion_anchor, [0, 0, 0])
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                arrived, peer=group_base + tp_rank, offsets=[peer, 0],
                value=0, op=pld.NotifyOp.Set,
            )
        pl.write(completion_anchor, [0, 0, 0], anchor)
    return completion_anchor


@pl.jit.inline(auto_scope=False)
def prefill_sp_output_reduce_scatter(
    partial: pl.Tensor[[T_DYN, D], pl.BF16],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[SP_T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Push contributions into source-rank receive slots and sum locally in FP32."""
    local_tokens = pl.tensor.dim(output, 0)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_reduce_reuse", allow_early_resolve=False) as reuse_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(
                output_arrived,
                offsets=[peer, 0], expected=(attention_epoch - 1) * 2, cmp=pld.WaitCmp.Ge,
            )
    with pl.spmd(64, name_hint="sp_reduce_publish", deps=[reuse_tid]) as copied_tid:
        worker = pl.tile.get_block_idx()
        for owner in pl.range(TP_SIZE):
            active = pl.max(0, pl.min(local_tokens, num_tokens - owner * local_tokens))
            blocks = (active + SP_PUBLISH_ROW_TILE - 1) // SP_PUBLISH_ROW_TILE
            for block in pl.range(worker, blocks, 64):
                local_row = block * SP_PUBLISH_ROW_TILE
                row = owner * local_tokens + local_row
                target_row = tp_rank * local_tokens + local_row
                if local_row + SP_PUBLISH_ROW_TILE <= active:
                    publish_value = pl.load(partial, [row, 0], [SP_PUBLISH_ROW_TILE, D])
                    pld.tile.remote_store(publish_value, output_window, group_base + owner, [target_row, 0])
                else:
                    for tail in pl.range(local_row, active):
                        tail_publish = pl.load(partial, [owner * local_tokens + tail, 0], [1, D])
                        target_tail = tp_rank * local_tokens + tail
                        pld.tile.remote_store(tail_publish, output_window, group_base + owner, [target_tail, 0])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_reduce_notify", deps=[copied_tid]) as publish_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                output_arrived,
                peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_reduce_wait", deps=[publish_tid],
               allow_early_resolve=False) as wait_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2 - 1, cmp=pld.WaitCmp.Ge)
    with pl.spmd(64, name_hint="sp_reduce_scatter", deps=[wait_tid]) as reduce_tid:
        worker = pl.tile.get_block_idx()
        local_peer = group_base + tp_rank
        for block in pl.range(worker, (local_tokens + SP_REDUCE_ROW_TILE - 1) // SP_REDUCE_ROW_TILE, 64):
            local_row = block * SP_REDUCE_ROW_TILE
            row = tp_rank * local_tokens + local_row
            if local_row + SP_REDUCE_ROW_TILE <= local_tokens and row + SP_REDUCE_ROW_TILE <= num_tokens:
                acc = pl.tile.full([SP_REDUCE_ROW_TILE, D], dtype=pl.FP32, value=0.0)
                for peer in pl.range(TP_SIZE):
                    source_row = peer * local_tokens + local_row
                    value = pld.tile.remote_load(output_window, local_peer, [source_row, 0], [SP_REDUCE_ROW_TILE, D])
                    acc = pl.add(acc, pl.cast(value, pl.FP32))
                result = pl.cast(acc, pl.BF16, mode="rint")
                output = pl.store(result, [local_row, 0], output)
            else:
                for tail in pl.range(local_row, pl.min(local_row + SP_REDUCE_ROW_TILE, local_tokens)):
                    target_row = tp_rank * local_tokens + tail
                    tail_acc = pl.tile.full([1, D], dtype=pl.FP32, value=0.0)
                    if target_row < num_tokens:
                        for peer in pl.range(TP_SIZE):
                            source_row = peer * local_tokens + tail
                            tail_value = pld.tile.remote_load(output_window, local_peer, [source_row, 0], [1, D])
                            tail_acc = pl.add(tail_acc, pl.cast(tail_value, pl.FP32))
                    tail_result = pl.cast(tail_acc, pl.BF16, mode="rint")
                    output = pl.store(tail_result, [tail, 0], output)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_reduce_release", deps=[reduce_tid]) as release_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                output_arrived,
                peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_reduce_consumed", deps=[release_tid],
               allow_early_resolve=False):
        for peer in pl.range(TP_SIZE):
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2, cmp=pld.WaitCmp.Ge)
        anchor = pl.read(output, [0, 0])
        pl.write(output, [0, 0], anchor)
    return output


@pl.jit.inline(auto_scope=False)
def prefill_tp_output_all_reduce(
    output_partial: pl.Tensor[[T_DYN, D], pl.BF16],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_reuse", allow_early_resolve=False) as reuse_tid:
        for peer in pl.range(TP_SIZE):
            previous_epoch = (attention_epoch - 1) * 2
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=previous_epoch, cmp=pld.WaitCmp.Ge)
    with pl.spmd(64, name_hint="prefill_tp_publish", deps=[reuse_tid]) as publish_tid:
        worker = pl.tile.get_block_idx()
        for row in pl.range(worker, num_tokens, 64):
            col = 0
            value = pl.load(output_partial, [row, col], [1, D])
            pld.tile.remote_store(value, output_window, peer=group_base + tp_rank, offsets=[row, col])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_ready", deps=[publish_tid]) as ready_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                output_arrived,
                peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_wait", deps=[ready_tid],
               allow_early_resolve=False) as wait_tid:
        for peer in pl.range(TP_SIZE):
            ready_epoch = attention_epoch * 2 - 1
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=ready_epoch, cmp=pld.WaitCmp.Ge)
    with pl.spmd(64, name_hint="prefill_tp_reduce", deps=[wait_tid]) as reduce_tid:
        worker = pl.tile.get_block_idx()
        for row in pl.range(worker, num_tokens, 64):
            col = 0
            acc = pl.tile.full([1, D], dtype=pl.FP32, value=0.0)
            for peer in pl.range(TP_SIZE):
                value = pld.tile.remote_load(
                    output_window,
                    peer=group_base + peer, offsets=[row, col], shape=[1, D],
                )
                acc = pl.add(acc, pl.cast(value, pl.FP32))
            output = pl.store(pl.cast(acc, pl.BF16, mode="rint"), [row, col], output)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_release", deps=[reduce_tid]) as release_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                output_arrived,
                peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_consumed", deps=[release_tid],
               allow_early_resolve=False):
        for peer in pl.range(TP_SIZE):
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2, cmp=pld.WaitCmp.Ge)
        anchor = pl.read(output, [0, 0])
        pl.write(output, [0, 0], anchor)
    return output


@pl.jit
def o_proj_validation(
    attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    output: pl.InOut[pl.Tensor[[T_DYN, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    attended.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    o_proj(
        attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        output, num_tokens, worker_limit=32, oa_row_tile=80, ob_row_tile=80,
    )
    return output

@pl.jit
def o_proj_with_hc_validation(
    attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sublayer: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    attended.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    sublayer.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    o_proj(
        attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        sublayer, num_tokens, worker_limit=32, oa_row_tile=80, ob_row_tile=80,
    )
    hc_post_deps = pl.array.create(2, pl.TASK_ID)
    hc_post(sublayer, residual, post_mix, residual_mix, output, deps=hc_post_deps)
    return sublayer, output


def build_o_proj_specs(active_tokens: int, *, pad_to_tile: bool = True):
    """Build deterministic output-projection fixtures with padded inactive rows."""
    from golden import ScalarSpec, TensorSpec
    from quant import pack_fp8_nz, quantize_mxfp8_upward_reference, pack_mx_b_scale, quantize_grouped_mxfp8_weight

    tokens = max(16, (active_tokens + 15) // 16 * 16) if pad_to_tile else active_tokens
    values = {}

    def inputs():
        if not values:
            generator = torch.Generator().manual_seed(17)
            attended = torch.randn(tokens, LOCAL_H * HEAD_DIM, generator=generator).bfloat16()
            attended[active_tokens:] = float("nan")
            wo_a = torch.randn(LOCAL_O_GROUPS, O_LORA, O_GROUP_IN, generator=generator) / O_GROUP_IN**0.5
            wo_a, wo_a_scale = quantize_grouped_mxfp8_weight(wo_a.bfloat16())
            wo_b = torch.randn(D, LOCAL_O_WIDTH, generator=generator) / LOCAL_O_WIDTH**0.5
            wo_b, wo_b_scale = quantize_mxfp8_upward_reference(wo_b)
            frequencies = 10000.0 ** (-torch.arange(0, ROPE_DIM, 2).float() / ROPE_DIM)
            angles = torch.arange(tokens).float()[:, None] * frequencies
            values.update(
                attended=attended, wo_a=wo_a, wo_a_scale=wo_a_scale,
                wo_b=pack_fp8_nz(wo_b.T.contiguous()),
                wo_b_scale=pack_mx_b_scale(wo_b_scale.T.contiguous()).view(torch.float8_e8m0fnu),
                rope_cos=angles.cos(), rope_sin=angles.sin(),
            )
        return values

    shapes = (
        ("attended", [tokens, LOCAL_H * HEAD_DIM], torch.bfloat16),
        ("wo_a", [LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], torch.float8_e4m3fn),
        ("wo_a_scale", [LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], torch.float8_e8m0fnu),
        ("wo_b", [LOCAL_O_WIDTH, D], torch.float8_e4m3fn),
        ("wo_b_scale", [LOCAL_O_WIDTH // 32, D], torch.float8_e8m0fnu),
        ("rope_cos", [tokens, ROPE_DIM // 2], torch.float32),
        ("rope_sin", [tokens, ROPE_DIM // 2], torch.float32),
    )
    specs = [TensorSpec(name, shape, dtype, init_value=lambda name=name: inputs()[name])
             for name, shape, dtype in shapes]
    return specs + [
        TensorSpec("output", [tokens, D], torch.bfloat16, init_value=13.0),
        ScalarSpec("num_tokens", torch.int32, active_tokens),
    ]


def golden_o_proj_validation(tensors):
    """Project active rows to BF16 and zero inactive rows."""
    active = int(tensors["num_tokens"])
    tensors["output"].zero_()
    if active == 0:
        return
    attended = tensors["attended"][:active].unflatten(-1, (LOCAL_H, HEAD_DIM))
    result = golden_o_proj(
        attended, tensors["rope_cos"][:active], tensors["rope_sin"][:active],
        tensors["wo_a"], tensors["wo_b"], tensors["wo_b_scale"],
        wo_a_scale=tensors["wo_a_scale"], output_dtype=torch.bfloat16, sequential_projection=True,
    )
    tensors["output"][:active].copy_(result)


def build_o_proj_hc_post_specs(active_tokens: int = 72, *, pad_to_tile: bool = False):
    """Add genuine FP32 HC residuals and mixes to active or padded projection inputs."""
    from golden import ScalarSpec, TensorSpec

    specs = build_o_proj_specs(active_tokens, pad_to_tile=pad_to_tile)
    tokens = specs[0].shape[0]
    specs = [spec for spec in specs if spec.name not in ("output", "num_tokens")]
    generator = torch.Generator().manual_seed(29)
    return specs + [
        TensorSpec("sublayer", [tokens, D], torch.bfloat16),
        TensorSpec("residual", [tokens, HC_MULT, D], torch.float32,
                   init_value=lambda: torch.randn(tokens, HC_MULT, D, generator=generator)),
        TensorSpec("post_mix", [tokens, HC_MULT], torch.float32,
                   init_value=lambda: 2 * torch.sigmoid(torch.randn(tokens, HC_MULT, generator=generator))),
        TensorSpec("residual_mix", [tokens, HC_MULT, HC_MULT], torch.float32,
                   init_value=lambda: torch.randn(tokens, HC_MULT, HC_MULT, generator=generator)),
        TensorSpec("output", [tokens, HC_MULT, D], torch.float32),
        ScalarSpec("num_tokens", torch.int32, active_tokens),
    ]


def golden_o_proj_hc_post_validation(tensors):
    """Compute independent BF16 output projection followed by FP32 HC mixing."""
    from hc import golden_hc_post

    projected = dict(tensors, output=tensors["sublayer"])
    golden_o_proj_validation(projected)
    tensors["output"].copy_(golden_hc_post(
        tensors["sublayer"], tensors["residual"], tensors["post_mix"], tensors["residual_mix"],
    ))


def compare_o_proj_hc_post(actual, _expected, *, actual_outputs, expected_outputs, inputs, **kwargs):
    """Check FP32 HC mixing independently of the separately checked BF16 projection."""
    from golden import ratio_allclose
    from hc import golden_hc_post

    expected = golden_hc_post(
        actual_outputs["sublayer"], inputs["residual"], inputs["post_mix"], inputs["residual_mix"],
    )
    compare = ratio_allclose(atol=1e-6, rtol=1e-5, max_error_ratio=0.0)
    return compare(
        actual, expected, actual_outputs=actual_outputs, expected_outputs=expected_outputs, inputs=inputs, **kwargs,
    )


def build_o_validation_specs(active, capacity):
    """Reuse the C++ path's weight fixtures with explicit caller capacity and poisoned tails."""
    from golden import ScalarSpec, TensorSpec

    specs = build_o_proj_specs(capacity, pad_to_tile=False)
    result = []
    for spec in specs:
        if spec.name == "num_tokens":
            result.append(ScalarSpec("num_tokens", torch.int32, active))
        elif spec.name == "attended":
            def poisoned(init=spec.create_tensor):
                value = init().clone()
                value[active:] = float("nan")
                return value

            result.append(TensorSpec(spec.name, spec.shape, spec.dtype, init_value=poisoned))
        else:
            result.append(spec)
    return result



if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="V4.1 output projection branch and padding validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tp", type=int, choices=[1, 2, 4, 8], default=TP_SIZE)
    parser.add_argument("--case", choices=["all", "empty", "fast", "tail", "hc-post"], default="all")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    args = parser.parse_args()
    cases = (("empty", 0), ("fast", 12), ("tail", 17))
    for name, active in cases:
        if args.case not in ("all", name):
            continue
        print(f"[O_PROJ] {name}: active={active}", flush=True)
        active_compare = ratio_allclose(atol=1e-2, rtol=1.0 / 128)

        def compare(actual, expected, **kwargs):
            if active == 0:
                zeroed = torch.equal(actual, expected)
                return zeroed, f"empty input zeros output={zeroed}"
            passed, detail = active_compare(actual[:active], expected[:active], **kwargs)
            inactive_ok = torch.equal(actual[active:], expected[active:])
            return passed and inactive_ok, f"{detail}; inactive rows zeroed={inactive_ok}"

        result = run(
            fn=o_proj_validation,
            specs=build_o_proj_specs(active),
            golden_fn=golden_o_proj_validation,
            compare_fn={"output": compare},
            config={
                "backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device,
            },
            compile_only=args.compile_only,
            save_data=args.save_data,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
    if args.case in ("all", "hc-post"):
        for active, padded in ((72, False), (17, True), (0, True), (81, True)):
            print(f"[O_PROJ] hc-post: active={active}, padded={padded}, FP32 HC post", flush=True)
            result = run(
                fn=o_proj_with_hc_validation,
                specs=build_o_proj_hc_post_specs(active, pad_to_tile=padded),
                golden_fn=golden_o_proj_hc_post_validation,
                compare_fn={
                    "sublayer": ratio_allclose(atol=1e-2, rtol=1.0 / 128, valid_rows=active, zero_tail=True),
                    "output": compare_o_proj_hc_post,
                },
                config={
                    "backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device,
                },
                compile_only=args.compile_only,
                save_data=args.save_data,
            )
            if not result.passed:
                raise SystemExit(result.error or 1)
