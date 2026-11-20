# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
# VPTO register helpers require real A5 device execution.
"""V4.1 inverse RoPE, grouped W8A8 MXFP8 output projection, and HC post."""

from __future__ import annotations

from pathlib import Path

import torch

from quant import MX_GROUP, _quantize_mxfp8_activation, decode_e8m0, unpack_mx_b_scale

from qkv_proj_rope import golden_rope_interleave as rope_interleave
from gemm import golden_gemm_bf16_via_mxfp8_to_bf16_sequential, golden_mxfp8_linear_grouped_sum


import pypto.language as pl
from pypto.runtime import pto_isa_include_dir
import pypto.language.distributed as pld

from config import (
    FLASH as M, T_DYN,
)


# model config
LOCAL_H = M.num_attention_heads
LOCAL_O_GROUPS = M.o_groups
LOCAL_O_WIDTH = M.o_groups * M.o_lora_rank
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
O_LORA = M.o_lora_rank
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim


O_QUANT_SCALE_VIEW_ROWS = pl.dynamic("O_QUANT_SCALE_VIEW_ROWS")
O_GROUP_ROWS = pl.dynamic("O_GROUP_ROWS")
O_PAD_ROWS = pl.dynamic("O_PAD_ROWS")
O_IN_SCALE_COLS = pl.dynamic("O_IN_SCALE_COLS")
O_OUT_SCALE_COLS = pl.dynamic("O_OUT_SCALE_COLS")

# tiling
O_A_N_TILE = 256
O_B_N_TILE = 160


# Output-quantization tiles and scheduling.
O_QUANT_ROW_TILE = 80
O_QUANT_PREFETCH_ROW_TILE = 64
O_QUANT_HEADS_PER_GROUP = 8
O_QUANT_LANES = 64                      # FP32 lanes in one 256-byte vector register
O_QUANT_SCALE_GROUPS = HEAD_DIM // 32
O_QUANT_SCALE_BLOCK = 16                # rows sharing one packed scale row
O_QUANT_SCALE_COLS = 256                # scale bytes per block row and head
O_QUANT_HEADS = LOCAL_O_GROUPS * O_QUANT_HEADS_PER_GROUP
O_QUANT_ROPE_PAIRS = ROPE_DIM // 2
O_QUANT_ROPE_UNROLL = 8
O_QUANT_ROPE_START = HEAD_DIM - ROPE_DIM


@pl.jit.inline(auto_scope=False)
def _load_o_quant_inputs(
    tile_rows: pl.constexpr,
    attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    cosine: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sine: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    t0: pl.Scalar[pl.INDEX],
    head: pl.Scalar[pl.INDEX],
    rows: pl.Scalar[pl.INDEX],
):
    # Load the activation and trigonometric values for the active rows.
    window = pl.load(
        attended, [t0, head * HEAD_DIM], [tile_rows, HEAD_DIM],
        valid_shape=[rows, HEAD_DIM], target_memory=pl.MemorySpace.Vec,
    )
    cos_tile = pl.load(
        cosine, [t0, 0], [tile_rows, O_QUANT_ROPE_PAIRS],
        valid_shape=[rows, O_QUANT_ROPE_PAIRS], target_memory=pl.MemorySpace.Vec,
    )
    sin_tile = pl.load(
        sine, [t0, 0], [tile_rows, O_QUANT_ROPE_PAIRS],
        valid_shape=[rows, O_QUANT_ROPE_PAIRS], target_memory=pl.MemorySpace.Vec,
    )
    return window, cos_tile, sin_tile


# Tile dimensions bind to the constexpr row count during specialization.
@pl.jit.inline(auto_scope=False)
def _quantize_o_tile(
    tile_rows: pl.constexpr,
    full_tile: pl.constexpr,
    window: pl.Tile[[tile_rows, HEAD_DIM], pl.BF16, pl.Mem.Vec],  # noqa: F821
    cos_tile: pl.Tile[[tile_rows, O_QUANT_ROPE_PAIRS], pl.FP32, pl.Mem.Vec],  # noqa: F821
    sin_tile: pl.Tile[[tile_rows, O_QUANT_ROPE_PAIRS], pl.FP32, pl.Mem.Vec],  # noqa: F821
    rows: pl.Scalar[pl.INDEX],
    capacity: pl.Scalar[pl.INDEX],
    t0: pl.Scalar[pl.INDEX],
    group: pl.Scalar[pl.INDEX],
    local_head: pl.Scalar[pl.INDEX],
    padded_rows: pl.Scalar[pl.INDEX],
    scale_block_rows: pl.Scalar[pl.INDEX],
    payload: pl.Tensor[[O_GROUP_ROWS, O_GROUP_IN], pl.FP8E4M3FN],
    scale_view: pl.Tensor[[O_QUANT_SCALE_VIEW_ROWS, O_QUANT_HEADS_PER_GROUP * O_QUANT_SCALE_COLS], pl.FP8E8M0],
):
    # A full load initializes the whole tile.
    if full_tile:
        padded = window
    else:
        padded = pl.fillpad(window, pad_value=pl.PadValue.zero)
    # Register addressing uses flat tile shapes. Only active rows use the trig data.
    value = pl.tile.reshape(
        pl.tile.set_validshape(padded, tile_rows, HEAD_DIM),
        [(tile_rows * HEAD_DIM)],
    )
    cos_flat = pl.tile.reshape(
        pl.tile.set_validshape(cos_tile, tile_rows, O_QUANT_ROPE_PAIRS),
        [(tile_rows * O_QUANT_ROPE_PAIRS)],
    )
    sin_flat = pl.tile.reshape(
        pl.tile.set_validshape(sin_tile, tile_rows, O_QUANT_ROPE_PAIRS),
        [(tile_rows * O_QUANT_ROPE_PAIRS)],
    )
    # Rotate the trailing BF16 pairs and round back to BF16.
    with pl.reg_scope():
        full = pl.reg.full_mask(lanes=O_QUANT_LANES)
        lane = pl.reg.index_gen(0, lanes=O_QUANT_LANES, dtype=pl.UINT32)
        one = pl.reg.broadcast(1, lanes=O_QUANT_LANES, dtype=pl.UINT32)
        # ix1 is the partner lane of each (real, imaginary) pair, and
        # pair_index the trig element both lanes of a pair share.
        tmp = pl.reg.shift_rights(lane, 1, full)
        tmp = pl.reg.shift_lefts(tmp, 1, full)
        tmp = pl.reg.sub(lane, tmp)
        tmp = pl.reg.shift_lefts(tmp, 1, full)
        ix1 = pl.reg.add(lane, one)
        ix1 = pl.reg.sub(ix1, tmp)
        pair_index = pl.reg.shift_rights(lane, 1, full)
        even = pl.reg.compare(ix1, lane, full, predicate="gt")
        # Read the eight input rows before writing their rounded RoPE tails.
        # This shares the load-to-store synchronization across the group.
        for row_group in pl.hardware_loop(rows // O_QUANT_ROPE_UNROLL):
            row0 = row_group * O_QUANT_ROPE_UNROLL
            value_offset0 = row0 * HEAD_DIM + O_QUANT_ROPE_START
            trig_base0 = row0 * O_QUANT_ROPE_PAIRS
            cosine0 = pl.reg.gather(cos_flat, pair_index, full, base_offset=trig_base0)
            sine0 = pl.reg.gather(sin_flat, pair_index, full, base_offset=trig_base0)
            raw0 = pl.reg.load(value, offset=value_offset0, lanes=O_QUANT_LANES)
            row1 = row_group * O_QUANT_ROPE_UNROLL + 1
            value_offset1 = row1 * HEAD_DIM + O_QUANT_ROPE_START
            trig_base1 = row1 * O_QUANT_ROPE_PAIRS
            cosine1 = pl.reg.gather(cos_flat, pair_index, full, base_offset=trig_base1)
            sine1 = pl.reg.gather(sin_flat, pair_index, full, base_offset=trig_base1)
            raw1 = pl.reg.load(value, offset=value_offset1, lanes=O_QUANT_LANES)
            row2 = row_group * O_QUANT_ROPE_UNROLL + 2
            value_offset2 = row2 * HEAD_DIM + O_QUANT_ROPE_START
            trig_base2 = row2 * O_QUANT_ROPE_PAIRS
            cosine2 = pl.reg.gather(cos_flat, pair_index, full, base_offset=trig_base2)
            sine2 = pl.reg.gather(sin_flat, pair_index, full, base_offset=trig_base2)
            raw2 = pl.reg.load(value, offset=value_offset2, lanes=O_QUANT_LANES)
            row3 = row_group * O_QUANT_ROPE_UNROLL + 3
            value_offset3 = row3 * HEAD_DIM + O_QUANT_ROPE_START
            trig_base3 = row3 * O_QUANT_ROPE_PAIRS
            cosine3 = pl.reg.gather(cos_flat, pair_index, full, base_offset=trig_base3)
            sine3 = pl.reg.gather(sin_flat, pair_index, full, base_offset=trig_base3)
            raw3 = pl.reg.load(value, offset=value_offset3, lanes=O_QUANT_LANES)
            row4 = row_group * O_QUANT_ROPE_UNROLL + 4
            value_offset4 = row4 * HEAD_DIM + O_QUANT_ROPE_START
            trig_base4 = row4 * O_QUANT_ROPE_PAIRS
            cosine4 = pl.reg.gather(cos_flat, pair_index, full, base_offset=trig_base4)
            sine4 = pl.reg.gather(sin_flat, pair_index, full, base_offset=trig_base4)
            raw4 = pl.reg.load(value, offset=value_offset4, lanes=O_QUANT_LANES)
            row5 = row_group * O_QUANT_ROPE_UNROLL + 5
            value_offset5 = row5 * HEAD_DIM + O_QUANT_ROPE_START
            trig_base5 = row5 * O_QUANT_ROPE_PAIRS
            cosine5 = pl.reg.gather(cos_flat, pair_index, full, base_offset=trig_base5)
            sine5 = pl.reg.gather(sin_flat, pair_index, full, base_offset=trig_base5)
            raw5 = pl.reg.load(value, offset=value_offset5, lanes=O_QUANT_LANES)
            row6 = row_group * O_QUANT_ROPE_UNROLL + 6
            value_offset6 = row6 * HEAD_DIM + O_QUANT_ROPE_START
            trig_base6 = row6 * O_QUANT_ROPE_PAIRS
            cosine6 = pl.reg.gather(cos_flat, pair_index, full, base_offset=trig_base6)
            sine6 = pl.reg.gather(sin_flat, pair_index, full, base_offset=trig_base6)
            raw6 = pl.reg.load(value, offset=value_offset6, lanes=O_QUANT_LANES)
            row7 = row_group * O_QUANT_ROPE_UNROLL + 7
            value_offset7 = row7 * HEAD_DIM + O_QUANT_ROPE_START
            trig_base7 = row7 * O_QUANT_ROPE_PAIRS
            cosine7 = pl.reg.gather(cos_flat, pair_index, full, base_offset=trig_base7)
            sine7 = pl.reg.gather(sin_flat, pair_index, full, base_offset=trig_base7)
            raw7 = pl.reg.load(value, offset=value_offset7, lanes=O_QUANT_LANES)

            sine_negative0 = pl.reg.muls(sine0, -1.0, full)
            rotated0 = pl.reg.convert(raw0, dtype=pl.FP32)
            swapped0 = pl.reg.gather_lanes(rotated0, ix1)
            scaled_real0 = pl.reg.mul(rotated0, cosine0)
            scaled_imag0 = pl.reg.mul(swapped0, sine_negative0)
            real0 = pl.reg.sub(scaled_real0, scaled_imag0)
            imag0 = pl.reg.add(scaled_imag0, scaled_real0)
            result0 = pl.reg.select(even, real0, imag0)
            tail0 = pl.reg.convert(result0, dtype=pl.BF16, rounding="R", saturate=False)
            sine_negative1 = pl.reg.muls(sine1, -1.0, full)
            rotated1 = pl.reg.convert(raw1, dtype=pl.FP32)
            swapped1 = pl.reg.gather_lanes(rotated1, ix1)
            scaled_real1 = pl.reg.mul(rotated1, cosine1)
            scaled_imag1 = pl.reg.mul(swapped1, sine_negative1)
            real1 = pl.reg.sub(scaled_real1, scaled_imag1)
            imag1 = pl.reg.add(scaled_imag1, scaled_real1)
            result1 = pl.reg.select(even, real1, imag1)
            tail1 = pl.reg.convert(result1, dtype=pl.BF16, rounding="R", saturate=False)
            sine_negative2 = pl.reg.muls(sine2, -1.0, full)
            rotated2 = pl.reg.convert(raw2, dtype=pl.FP32)
            swapped2 = pl.reg.gather_lanes(rotated2, ix1)
            scaled_real2 = pl.reg.mul(rotated2, cosine2)
            scaled_imag2 = pl.reg.mul(swapped2, sine_negative2)
            real2 = pl.reg.sub(scaled_real2, scaled_imag2)
            imag2 = pl.reg.add(scaled_imag2, scaled_real2)
            result2 = pl.reg.select(even, real2, imag2)
            tail2 = pl.reg.convert(result2, dtype=pl.BF16, rounding="R", saturate=False)
            sine_negative3 = pl.reg.muls(sine3, -1.0, full)
            rotated3 = pl.reg.convert(raw3, dtype=pl.FP32)
            swapped3 = pl.reg.gather_lanes(rotated3, ix1)
            scaled_real3 = pl.reg.mul(rotated3, cosine3)
            scaled_imag3 = pl.reg.mul(swapped3, sine_negative3)
            real3 = pl.reg.sub(scaled_real3, scaled_imag3)
            imag3 = pl.reg.add(scaled_imag3, scaled_real3)
            result3 = pl.reg.select(even, real3, imag3)
            tail3 = pl.reg.convert(result3, dtype=pl.BF16, rounding="R", saturate=False)
            sine_negative4 = pl.reg.muls(sine4, -1.0, full)
            rotated4 = pl.reg.convert(raw4, dtype=pl.FP32)
            swapped4 = pl.reg.gather_lanes(rotated4, ix1)
            scaled_real4 = pl.reg.mul(rotated4, cosine4)
            scaled_imag4 = pl.reg.mul(swapped4, sine_negative4)
            real4 = pl.reg.sub(scaled_real4, scaled_imag4)
            imag4 = pl.reg.add(scaled_imag4, scaled_real4)
            result4 = pl.reg.select(even, real4, imag4)
            tail4 = pl.reg.convert(result4, dtype=pl.BF16, rounding="R", saturate=False)
            sine_negative5 = pl.reg.muls(sine5, -1.0, full)
            rotated5 = pl.reg.convert(raw5, dtype=pl.FP32)
            swapped5 = pl.reg.gather_lanes(rotated5, ix1)
            scaled_real5 = pl.reg.mul(rotated5, cosine5)
            scaled_imag5 = pl.reg.mul(swapped5, sine_negative5)
            real5 = pl.reg.sub(scaled_real5, scaled_imag5)
            imag5 = pl.reg.add(scaled_imag5, scaled_real5)
            result5 = pl.reg.select(even, real5, imag5)
            tail5 = pl.reg.convert(result5, dtype=pl.BF16, rounding="R", saturate=False)
            sine_negative6 = pl.reg.muls(sine6, -1.0, full)
            rotated6 = pl.reg.convert(raw6, dtype=pl.FP32)
            swapped6 = pl.reg.gather_lanes(rotated6, ix1)
            scaled_real6 = pl.reg.mul(rotated6, cosine6)
            scaled_imag6 = pl.reg.mul(swapped6, sine_negative6)
            real6 = pl.reg.sub(scaled_real6, scaled_imag6)
            imag6 = pl.reg.add(scaled_imag6, scaled_real6)
            result6 = pl.reg.select(even, real6, imag6)
            tail6 = pl.reg.convert(result6, dtype=pl.BF16, rounding="R", saturate=False)
            sine_negative7 = pl.reg.muls(sine7, -1.0, full)
            rotated7 = pl.reg.convert(raw7, dtype=pl.FP32)
            swapped7 = pl.reg.gather_lanes(rotated7, ix1)
            scaled_real7 = pl.reg.mul(rotated7, cosine7)
            scaled_imag7 = pl.reg.mul(swapped7, sine_negative7)
            real7 = pl.reg.sub(scaled_real7, scaled_imag7)
            imag7 = pl.reg.add(scaled_imag7, scaled_real7)
            result7 = pl.reg.select(even, real7, imag7)
            tail7 = pl.reg.convert(result7, dtype=pl.BF16, rounding="R", saturate=False)

            _rotated0 = pl.reg.store(value, offset=value_offset0, value=tail0)
            _rotated1 = pl.reg.store(value, offset=value_offset1, value=tail1)
            _rotated2 = pl.reg.store(value, offset=value_offset2, value=tail2)
            _rotated3 = pl.reg.store(value, offset=value_offset3, value=tail3)
            _rotated4 = pl.reg.store(value, offset=value_offset4, value=tail4)
            _rotated5 = pl.reg.store(value, offset=value_offset5, value=tail5)
            _rotated6 = pl.reg.store(value, offset=value_offset6, value=tail6)
            _rotated7 = pl.reg.store(value, offset=value_offset7, value=tail7)

        for remainder_row in pl.hardware_loop(rows % O_QUANT_ROPE_UNROLL):
            row = rows // O_QUANT_ROPE_UNROLL * O_QUANT_ROPE_UNROLL + remainder_row
            # Match vgather2(cos/sin + row * 32, pair_index, all).
            trig_base = row * O_QUANT_ROPE_PAIRS
            cosine_v = pl.reg.gather(cos_flat, pair_index, full, base_offset=trig_base)
            sine_v = pl.reg.gather(sin_flat, pair_index, full, base_offset=trig_base)
            sine_v = pl.reg.muls(sine_v, -1.0, full)
            raw = pl.reg.load(value, offset=row * HEAD_DIM + O_QUANT_ROPE_START, lanes=O_QUANT_LANES)
            rotated = pl.reg.convert(raw, dtype=pl.FP32)
            swapped = pl.reg.gather_lanes(rotated, ix1)
            scaled_real = pl.reg.mul(rotated, cosine_v)
            scaled_imag = pl.reg.mul(swapped, sine_v)
            real = pl.reg.sub(scaled_real, scaled_imag)
            imag = pl.reg.add(scaled_imag, scaled_real)
            rotated = pl.reg.select(even, real, imag)
            raw = pl.reg.convert(rotated, dtype=pl.BF16, rounding="R", saturate=False)
            _rotated = pl.reg.store(value, offset=row * HEAD_DIM + O_QUANT_ROPE_START, value=raw)

    full_rows = pl.tile.reshape(value, [tile_rows, HEAD_DIM])
    max_workspace = pl.create_tile([1, (tile_rows * O_QUANT_SCALE_GROUPS)], dtype=pl.BF16)
    scale_workspace = pl.create_tile([1, (tile_rows * O_QUANT_SCALE_GROUPS)], dtype=pl.BF16)
    payload_bytes, exponents = pl.tile.quant_mx_raw(full_rows, max_workspace, scale_workspace, group_axis=1)
    # Pack scales while the payload tile is live so scale packing
    # can start before the payload writeback.
    zz_tmp = pl.create_tile([1, (((64 + tile_rows * O_QUANT_SCALE_GROUPS // O_QUANT_SCALE_BLOCK + 31) // 32) * 32)], dtype=pl.UINT8)
    packed_scale = pl.tmov_x2zz(
        exponents, zz_tmp, group_axis=1, dst_rows=tile_rows, dst_cols=O_QUANT_SCALE_GROUPS,
    )
    scale_tile = pl.tile.reinterpret_view(packed_scale, pl.FP8E8M0)
    scale_tile = pl.reshape(scale_tile, [tile_rows // O_QUANT_SCALE_BLOCK, O_QUANT_SCALE_COLS])
    scale_tile = pl.tile.set_validshape(
        scale_tile, capacity // O_QUANT_SCALE_BLOCK, O_QUANT_SCALE_COLS
    )
    pl.store(
        scale_tile,
        [
            group * scale_block_rows + t0 // O_QUANT_SCALE_BLOCK,
            local_head * O_QUANT_SCALE_COLS,
        ],
        scale_view,
    )

    # TQUANT uses a finite reciprocal for scale code zero. Preserve the
    # projection contract by clearing magnitude bits in those groups.
    if rows > 0:
        exponent_words = pl.reinterpret_view(exponents, pl.UINT32, shape=[(tile_rows * O_QUANT_SCALE_GROUPS) // 4])
        flag = pl.create_tile([1, 64], dtype=pl.UINT32, target_memory=pl.MemorySpace.Vec)
        flat_flag = pl.reshape(flag, [64])
        with pl.reg_scope():
            word_count = rows * (O_QUANT_SCALE_GROUPS // 4)
            all_code_words = pl.reg.full_mask(lanes=64)
            one_per_byte = pl.reg.broadcast(0x01010101, lanes=64, dtype=pl.UINT32)
            top_bits = pl.reg.broadcast(0x80808080, lanes=64, dtype=pl.UINT32)
            zero_flags = pl.reg.broadcast(0, lanes=64, dtype=pl.UINT32)
            scan_blocks = (word_count + 63) // 64
            for scan, (found,) in pl.hardware_loop(scan_blocks, init_values=(zero_flags,)):
                start_word = scan * 64
                valid_words = pl.reg.active_mask(pl.min(word_count - start_word, 64), lanes=64)
                codes = pl.reg.load(exponent_words, start_word, lanes=64)
                # This expression detects zero bytes in the active words.
                decremented = pl.reg.sub(codes, one_per_byte, valid_words)
                inverse_codes = pl.reg.bitwise_not(codes, valid_words)
                byte_flags = pl.reg.bitwise_and(decremented, inverse_codes, valid_words)
                byte_flags = pl.reg.bitwise_and(byte_flags, top_bits, valid_words)
                found = pl.yield_(pl.reg.bitwise_or(found, byte_flags, all_code_words))
            any_zero = pl.reg.reduce_max(found, all_code_words)
            flag_words = pl.reg.group_broadcast(any_zero, lanes=64)
            _scan = pl.reg.store(flat_flag, 0, flag_words)
        if pl.tile.read(flag, [0, 0]) != 0:
            words = pl.reinterpret_view(payload_bytes, pl.UINT32, shape=[(tile_rows * HEAD_DIM) // 4])
            with pl.reg_scope():
                groups = pl.reg.full_mask(lanes=8)
                scale_index = pl.reg.index_gen(0, lanes=8, dtype=pl.UINT16)
                keep = pl.reg.broadcast(0xFFFFFFFF, lanes=8, dtype=pl.UINT32)
                sign = pl.reg.broadcast(0x80808080, lanes=8, dtype=pl.UINT32)
                full_words = pl.reg.full_mask(lanes=64)
                for block in pl.hardware_loop(rows * 2):
                    index = pl.cast(block * 8, pl.INDEX)
                    scale = pl.reg.gather(exponents, scale_index, groups, base_offset=index)
                    scale_u32 = pl.reg.convert(scale, dtype=pl.UINT32)
                    nonzero = pl.reg.compares(scale_u32, 0, groups, predicate="ne")
                    byte_mask = pl.reg.select(nonzero, keep, sign)
                    word_mask = pl.reg.group_broadcast(byte_mask, lanes=64)
                    offset = block * 64
                    value_words = pl.reg.load(words, offset, lanes=64)
                    zeroed_words = pl.reg.bitwise_and(value_words, word_mask, full_words)
                    _payload_words = pl.reg.store(words, offset, zeroed_words)
    payload_tile = pl.tile.reinterpret_view(payload_bytes, pl.FP8E4M3FN)
    payload_tile = pl.tile.set_validshape(payload_tile, capacity, HEAD_DIM)
    pl.store(payload_tile, [group * padded_rows + t0, local_head * HEAD_DIM], payload)
    return payload


@pl.jit.inline(auto_scope=False)
def _quant_o_vf_schedule(
    tile_rows: pl.constexpr,
    prefetch_tiles: pl.constexpr,
    attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    cosine: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sine: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    payload: pl.Out[pl.Tensor[[O_GROUP_ROWS, O_GROUP_IN], pl.FP8E4M3FN]],
    scales: pl.Out[pl.Tensor[[LOCAL_O_GROUPS, O_IN_SCALE_COLS], pl.FP8E8M0]],
    active_rows: pl.Scalar[pl.INT32],
    padded_rows: pl.Scalar[pl.INDEX],
):
    """Inverse-rotate BF16 tails and produce grouped MXFP8 payload and scales."""
    scale_block_rows = padded_rows // O_QUANT_SCALE_BLOCK
    # scale_gm walks 2048-byte block rows inside one group. Expose that stride as
    # a tensor view so the tile store can write an ordinary 2D slice.
    # Form the view before pl.spmd.
    scale_stride = O_QUANT_HEADS_PER_GROUP * O_QUANT_SCALE_COLS
    scale_view = pl.reshape(scales, [LOCAL_O_GROUPS * pl.tensor.dim(scales, 1) // scale_stride, scale_stride])
    workers = pl.max(1, pl.min(O_QUANT_HEADS, 2 * pl.system.available_cluster_count()))
    with pl.spmd(workers, name_hint="quant_o_vf_pypto") as published:
        worker = pl.tile.get_block_idx()
        worker_count = pl.tile.get_block_num()
        row_blocks = (padded_rows + tile_rows - 1) // tile_rows
        if prefetch_tiles > 0:
            # Start with two loads, then load one additional tile after each quantization.
            for head in pl.range(worker, O_QUANT_HEADS, worker_count):
                group = head // O_QUANT_HEADS_PER_GROUP
                local_head = head % O_QUANT_HEADS_PER_GROUP
                full_rows = pl.max(0, pl.min(pl.cast(active_rows, pl.INDEX), padded_rows))
                prefetch_groups = full_rows // (tile_rows * prefetch_tiles)
                for prefetch_group in pl.range(prefetch_groups):
                    t_full0 = prefetch_group * prefetch_tiles * tile_rows
                    window_full0, cosine_full0, sine_full0 = _load_o_quant_inputs(
                        tile_rows, attended, cosine, sine, t_full0, head, tile_rows,
                    )
                    t_full1 = t_full0 + tile_rows
                    window_full1, cosine_full1, sine_full1 = _load_o_quant_inputs(
                        tile_rows, attended, cosine, sine, t_full1, head, tile_rows,
                    )
                    _quantize_o_tile(
                        tile_rows, True, window_full0, cosine_full0, sine_full0, tile_rows, tile_rows, t_full0,
                        group, local_head, padded_rows, scale_block_rows, payload, scale_view,
                    )
                    t_full2 = t_full0 + 2 * tile_rows
                    window_full2, cosine_full2, sine_full2 = _load_o_quant_inputs(
                        tile_rows, attended, cosine, sine, t_full2, head, tile_rows,
                    )
                    _quantize_o_tile(
                        tile_rows, True, window_full1, cosine_full1, sine_full1, tile_rows, tile_rows, t_full1,
                        group, local_head, padded_rows, scale_block_rows, payload, scale_view,
                    )
                    t_full3 = t_full0 + 3 * tile_rows
                    window_full3, cosine_full3, sine_full3 = _load_o_quant_inputs(
                        tile_rows, attended, cosine, sine, t_full3, head, tile_rows,
                    )
                    _quantize_o_tile(
                        tile_rows, True, window_full2, cosine_full2, sine_full2, tile_rows, tile_rows, t_full2,
                        group, local_head, padded_rows, scale_block_rows, payload, scale_view,
                    )
                    if prefetch_tiles >= 5:
                        t_full4 = t_full0 + 4 * tile_rows
                        window_full4, cosine_full4, sine_full4 = _load_o_quant_inputs(
                            tile_rows, attended, cosine, sine, t_full4, head, tile_rows,
                        )
                        _quantize_o_tile(
                            tile_rows, True, window_full3, cosine_full3, sine_full3, tile_rows, tile_rows, t_full3,
                            group, local_head, padded_rows, scale_block_rows, payload, scale_view,
                        )
                    if prefetch_tiles >= 6:
                        t_full5 = t_full0 + 5 * tile_rows
                        window_full5, cosine_full5, sine_full5 = _load_o_quant_inputs(
                            tile_rows, attended, cosine, sine, t_full5, head, tile_rows,
                        )
                        _quantize_o_tile(
                            tile_rows, True, window_full4, cosine_full4, sine_full4, tile_rows, tile_rows, t_full4,
                            group, local_head, padded_rows, scale_block_rows, payload, scale_view,
                        )
                        if prefetch_tiles == 8:
                            t_full6 = t_full0 + 6 * tile_rows
                            window_full6, cosine_full6, sine_full6 = _load_o_quant_inputs(
                                tile_rows, attended, cosine, sine, t_full6, head, tile_rows,
                            )
                            _quantize_o_tile(
                                tile_rows, True, window_full5, cosine_full5, sine_full5, tile_rows, tile_rows, t_full5,
                                group, local_head, padded_rows, scale_block_rows, payload, scale_view,
                            )
                            t_full7 = t_full0 + 7 * tile_rows
                            window_full7, cosine_full7, sine_full7 = _load_o_quant_inputs(
                                tile_rows, attended, cosine, sine, t_full7, head, tile_rows,
                            )
                            _quantize_o_tile(
                                tile_rows, True, window_full6, cosine_full6, sine_full6, tile_rows, tile_rows, t_full6,
                                group, local_head, padded_rows, scale_block_rows, payload, scale_view,
                            )
                            _quantize_o_tile(
                                tile_rows, True, window_full7, cosine_full7, sine_full7, tile_rows, tile_rows, t_full7,
                                group, local_head, padded_rows, scale_block_rows, payload, scale_view,
                            )
                        else:
                            _quantize_o_tile(
                                tile_rows, True, window_full5, cosine_full5, sine_full5, tile_rows, tile_rows, t_full5,
                                group, local_head, padded_rows, scale_block_rows, payload, scale_view,
                            )
                    elif prefetch_tiles == 5:
                        _quantize_o_tile(
                            tile_rows, True, window_full4, cosine_full4, sine_full4, tile_rows, tile_rows, t_full4,
                            group, local_head, padded_rows, scale_block_rows, payload, scale_view,
                        )
                    else:
                        _quantize_o_tile(
                            tile_rows, True, window_full3, cosine_full3, sine_full3, tile_rows, tile_rows, t_full3,
                            group, local_head, padded_rows, scale_block_rows, payload, scale_view,
                        )
                # Short groups and partially filled tiles use the two-tile path.
                full_blocks = prefetch_groups * prefetch_tiles
                remaining_blocks = row_blocks - full_blocks
                for tile_pair in pl.range(remaining_blocks // 2):
                    t0 = (full_blocks + tile_pair * 2) * tile_rows
                    capacity0 = pl.min(padded_rows - t0, tile_rows)
                    rows0 = pl.cast(
                        pl.max(0, pl.min(active_rows - pl.cast(t0, pl.INT32), pl.cast(capacity0, pl.INT32))), pl.INDEX,
                    )
                    window0, cosine0, sine0 = _load_o_quant_inputs(tile_rows, attended, cosine, sine, t0, head, rows0)
                    t1 = t0 + tile_rows
                    capacity1 = pl.min(padded_rows - t1, tile_rows)
                    rows1 = pl.cast(
                        pl.max(0, pl.min(active_rows - pl.cast(t1, pl.INT32), pl.cast(capacity1, pl.INT32))), pl.INDEX,
                    )
                    window1, cosine1, sine1 = _load_o_quant_inputs(tile_rows, attended, cosine, sine, t1, head, rows1)
                    _quantize_o_tile(
                        tile_rows, False,
                        window0, cosine0, sine0, rows0, capacity0, t0, group, local_head,
                        padded_rows, scale_block_rows, payload, scale_view,
                    )
                    _quantize_o_tile(
                        tile_rows, False,
                        window1, cosine1, sine1, rows1, capacity1, t1, group, local_head,
                        padded_rows, scale_block_rows, payload, scale_view,
                    )
                if remaining_blocks % 2 != 0:
                    t2 = (row_blocks - 1) * tile_rows
                    capacity2 = pl.min(padded_rows - t2, tile_rows)
                    rows2 = pl.cast(
                        pl.max(0, pl.min(active_rows - pl.cast(t2, pl.INT32), pl.cast(capacity2, pl.INT32))), pl.INDEX,
                    )
                    window2, cosine2, sine2 = _load_o_quant_inputs(tile_rows, attended, cosine, sine, t2, head, rows2)
                    _quantize_o_tile(
                        tile_rows, False,
                        window2, cosine2, sine2, rows2, capacity2, t2, group, local_head,
                        padded_rows, scale_block_rows, payload, scale_view,
                    )
        else:
            # PTOAS derives pipe synchronization from Tile and VF memory effects.
            for work in pl.range(worker, row_blocks * O_QUANT_HEADS, worker_count):
                head = work % O_QUANT_HEADS
                t0 = (work // O_QUANT_HEADS) * tile_rows
                capacity = pl.min(padded_rows - t0, tile_rows)
                rows = pl.cast(
                    pl.max(0, pl.min(active_rows - pl.cast(t0, pl.INT32), pl.cast(capacity, pl.INT32))),
                    pl.INDEX,
                )
                group = head // O_QUANT_HEADS_PER_GROUP
                local_head = head % O_QUANT_HEADS_PER_GROUP
                window, cos_tile, sin_tile = _load_o_quant_inputs(tile_rows, attended, cosine, sine, t0, head, rows)
                _quantize_o_tile(
                    tile_rows, False, window, cos_tile, sin_tile, rows, capacity, t0, group, local_head,
                    padded_rows, scale_block_rows, payload, scale_view,
                )
    return published


@pl.jit.inline(auto_scope=False)
def quant_o_vf_pypto(
    attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    cosine: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sine: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    payload: pl.Out[pl.Tensor[[O_GROUP_ROWS, O_GROUP_IN], pl.FP8E4M3FN]],
    scales: pl.Out[pl.Tensor[[LOCAL_O_GROUPS, O_IN_SCALE_COLS], pl.FP8E8M0]],
    active_rows: pl.Scalar[pl.INT32],
    padded_rows: pl.Scalar[pl.INDEX],
):
    """Inverse-rotate BF16 tails and produce grouped MXFP8 payload and scales."""
    full_rows = pl.min(pl.cast(active_rows, pl.INDEX), padded_rows)
    # Choose a group size that covers the available full tiles.
    if padded_rows <= O_QUANT_ROW_TILE:
        published = _quant_o_vf_schedule(
            O_QUANT_ROW_TILE, 0, attended, cosine, sine, payload, scales, active_rows, padded_rows,
        )
    elif full_rows >= 8 * O_QUANT_PREFETCH_ROW_TILE:
        published = _quant_o_vf_schedule(
            O_QUANT_PREFETCH_ROW_TILE, 8, attended, cosine, sine, payload, scales, active_rows, padded_rows,
        )
    elif full_rows >= 6 * O_QUANT_PREFETCH_ROW_TILE:
        published = _quant_o_vf_schedule(
            O_QUANT_PREFETCH_ROW_TILE, 6, attended, cosine, sine, payload, scales, active_rows, padded_rows,
        )
    elif full_rows >= 5 * O_QUANT_PREFETCH_ROW_TILE:
        published = _quant_o_vf_schedule(
            O_QUANT_PREFETCH_ROW_TILE, 5, attended, cosine, sine, payload, scales, active_rows, padded_rows,
        )
    else:
        published = _quant_o_vf_schedule(
            O_QUANT_PREFETCH_ROW_TILE, 4, attended, cosine, sine, payload, scales, active_rows, padded_rows,
        )
    return published


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


@pl.jit.extern(core_type="mixed", aic_source=Path(__file__).parent / "kernels" / "o_proj_b_hc_post.cpp",
               aiv_source=Path(__file__).parent / "kernels" / "o_proj_b_hc_post.cpp",
               include_dirs=(pto_isa_include_dir(),), dual_aiv_dispatch=True)
def _o_proj_b_hc_post(
    payload: pl.Tensor[[O_PAD_ROWS, LOCAL_O_WIDTH], pl.FP8E4M3FN],
    weight: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    activation_scale: pl.Tensor[[1, O_OUT_SCALE_COLS], pl.FP8E8M0],
    weight_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    residual: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    output: pl.Out[pl.Tensor],
    rows: pl.Scalar[pl.INDEX],
    row_tile: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    """Fuse runtime-M O-B with SplitM HC-post."""
    return output


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
        flat = pl.reshape(values, [80 * 256])
        valid_rows = pl.max(0, pl.min(80, active_rows - row))
        quant_bytes = pl.create_tile([80 * 256], dtype=pl.INT8, target_memory=pl.Mem.Vec)
        scale_pairs = pl.create_tile([80 * 256 // 64], dtype=pl.UINT16, target_memory=pl.Mem.Vec)
        scale_bytes = pl.reinterpret_view(scale_pairs, pl.UINT8, shape=[80 * 256 // 32])
        quant_words = pl.reinterpret_view(quant_bytes, pl.UINT32, shape=[80 * 256 // 4])
        with pl.reg_scope():
            q_full16 = pl.reg.full_mask(lanes=128)
            q_lane16 = pl.reg.index_gen(0, lanes=128, dtype=pl.UINT16)
            q_floor = pl.reg.broadcast(1024, lanes=128, dtype=pl.UINT16)
            q_three = pl.reg.broadcast(3, lanes=128, dtype=pl.UINT16)
            q_pair_lane = pl.reg.bitwise_and(q_lane16, q_three)
            q_pair_low_index = pl.reg.shift_lefts(q_pair_lane, 5, q_full16)
            q_pair_high_index = pl.reg.adds(q_pair_low_index, 16, q_full16)
            q_scatter_relative = pl.reg.shift_lefts(q_lane16, 4, q_full16)
            q_four = pl.reg.active_mask(4, lanes=128)
            q_init_zero = pl.reg.broadcast(0, lanes=64, dtype=pl.UINT16)
            for q_init in pl.hardware_loop(5):
                _q_initialized = pl.reg.store(scale_pairs, offset=q_init * 64, value=q_init_zero)
            for q_r in pl.hardware_loop(valid_rows):
                q_raw = pl.reg.load(flat, offset=q_r * 256, lanes=256)
                q_values = pl.reg.convert(q_raw, dtype=pl.BF16, rounding="R", saturate=False)
                q_bits = pl.reg.interpret_cast(q_values, dtype=pl.UINT16)
                q_wide_mask = pl.reg.full_mask(lanes=256)
                q_exp_mask_wide = pl.reg.broadcast(32640, lanes=256, dtype=pl.UINT16)
                q_exponents = pl.reg.bitwise_and(q_bits, q_exp_mask_wide)
                q_groups = pl.reg.reduce_max(q_exponents, q_wide_mask, group_size=32)
                q_maximum = pl.reg.group_broadcast(q_groups, lanes=128)
                q_maximum = pl.reg.maximum(q_maximum, q_floor)
                q_shared = pl.reg.sub(q_maximum, q_floor)
                q_group_mask = pl.reg.full_mask(lanes=8)
                q_group_maximum = pl.reg.maximums(q_groups, 1024, q_group_mask)
                q_group_floor = pl.reg.broadcast(1024, lanes=8, dtype=pl.UINT16)
                q_group_shared = pl.reg.sub(q_group_maximum, q_group_floor)
                q_group_bias = pl.reg.broadcast(32512, lanes=8, dtype=pl.UINT16)
                q_group_inverse = pl.reg.sub(q_group_bias, q_group_shared)
                q_group_nonzero = pl.reg.minimums(q_group_shared, 128, q_group_mask)
                q_group_nonzero = pl.reg.shift_rights(q_group_nonzero, 7, q_group_mask)
                q_group_zero = pl.reg.broadcast(0, lanes=8, dtype=pl.UINT16)
                q_group_keep = pl.reg.sub(q_group_zero, q_group_nonzero)
                q_group_inverse = pl.reg.bitwise_and(q_group_inverse, q_group_keep)
                q_wide_inverse = pl.reg.group_broadcast(q_group_inverse, lanes=256)
                q_wide_reciprocal = pl.reg.interpret_cast(q_wide_inverse, dtype=pl.BF16)
                q_wide_scaled = pl.reg.mul(q_values, q_wide_reciprocal)
                q_wide_float = pl.reg.convert(q_wide_scaled, dtype=pl.FP32)
                q_wide_float_mask = pl.reg.full_mask(lanes=256)
                q_wide_clipped = pl.reg.maximums(q_wide_float, -448.0, q_wide_float_mask)
                q_wide_clipped = pl.reg.minimums(q_wide_clipped, 448.0, q_wide_float_mask)
                q_wide_fp8 = pl.reg.convert(q_wide_clipped, dtype=pl.FP8E4M3FN, rounding="R", saturate=False)
                q_wide_words = pl.reg.interpret_cast(q_wide_fp8, dtype=pl.UINT32)
                _q_quant_written = pl.reg.store(quant_words, offset=q_r * 64, value=q_wide_words)
                q_codes = pl.reg.shift_rights(q_shared, 7, q_full16)
                q_low = pl.reg.gather_lanes(q_codes, q_pair_low_index)
                q_high = pl.reg.gather_lanes(q_codes, q_pair_high_index)
                q_high = pl.reg.shift_lefts(q_high, 8, q_full16)
                q_packed = pl.reg.bitwise_or(q_low, q_high)
                q_scale_base = q_r // 16 * 64 + q_r % 16
                q_scale_base16 = pl.cast(q_scale_base, pl.UINT16)
                q_offsets = pl.reg.adds(q_scatter_relative, q_scale_base16, q_full16)
                _q_scale_written = pl.reg.scatter(scale_pairs, q_offsets, q_packed, q_four)
            empty = pl.reg.broadcast(0, lanes=256, dtype=pl.INT8)
            for inactive in pl.hardware_loop(80 - valid_rows):
                pad_row = valid_rows + inactive
                _padding_written = pl.reg.store(quant_bytes, offset=pad_row * 256, value=empty)
        quantized = pl.reshape(quant_bytes, [80, 256])
        quantized = pl.reinterpret_view(quantized, pl.FP8E4M3FN)
        scale = pl.reshape(scale_bytes, [5, 256 // 2])
        scale = pl.reinterpret_view(scale, pl.FP8E8M0)
        output = pl.store(quantized, [row, group * 1024 + col], output)
        packed = pl.reshape(scale, [5, 128])
        scales = pl.store(packed, [row // 16, group * 512 + nb * 128], scales)
    return output, output_scale


@pl.jit.inline(auto_scope=False)
def o_proj_hc_post(
    attn_out: pl.Tensor,
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    residual: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    oa_row_tile: pl.constexpr = 80,
    ob_row_tile: pl.constexpr = 128,
):
    """Return HC output and completion after consuming each N160 accumulator."""
    total_rows = pl.tensor.dim(output, 0)
    oa_tile = oa_row_tile
    if total_rows > oa_row_tile:
        oa_tile = 128
    padded_rows = ((total_rows + oa_tile - 1) // oa_tile) * oa_tile
    payload_all = pl.create_tensor([LOCAL_O_GROUPS * padded_rows, O_GROUP_IN], dtype=pl.FP8E4M3FN)
    scales_all = pl.create_tensor([LOCAL_O_GROUPS, padded_rows * (O_GROUP_IN // 32)], dtype=pl.FP8E8M0)
    quant_o_vf_pypto(attn_out, rope_cos, rope_sin, payload_all, scales_all, num_tokens, padded_rows)
    payload = pl.create_tensor([padded_rows, LOCAL_O_WIDTH], dtype=pl.FP8E4M3FN)
    activation_scale = pl.create_tensor([1, padded_rows * (LOCAL_O_WIDTH // 32)], dtype=pl.FP8E8M0)
    oa_tasks = (padded_rows // oa_tile) * (LOCAL_O_WIDTH // O_A_N_TILE)
    oa_workers = pl.max(1, pl.min(oa_tasks, pl.system.available_cluster_count()))
    with pl.spmd(oa_workers, name_hint="proj_o_a_quant"):
        payload, activation_scale = _o_proj_a(
            payload_all, wo_a, scales_all, wo_a_scale, payload, activation_scale,
            LOCAL_O_WIDTH, num_tokens, padded_rows, oa_tile,
        )
    ob_tasks = ((total_rows + ob_row_tile - 1) // ob_row_tile) * ((D + O_B_N_TILE - 1) // O_B_N_TILE)
    ob_workers = pl.max(1, pl.min(ob_tasks, pl.system.available_cluster_count()))
    with pl.spmd(ob_workers, name_hint="proj_o_b_hc_post") as ready:
        output = _o_proj_b_hc_post(
            payload, wo_b, activation_scale, wo_b_scale,
            residual, post_mix, residual_mix, output, total_rows, ob_row_tile,
        )
    return output, ready


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
    attn_out: torch.Tensor,
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
    tail = rope_interleave(attn_out[..., -rope_dim:], cos, sin, inverse=True)
    attn_out = torch.cat((attn_out[..., :-rope_dim], tail), dim=-1)
    groups = wo_a.shape[0]
    grouped = attn_out.flatten(-2).unflatten(-1, (groups, -1))
    latent = golden_grouped_output(grouped, wo_a, wo_a_scale)
    if sequential_projection and wo_b_scale is not None:
        projected = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(latent.flatten(-2), wo_b, wo_b_scale)
    else:
        projected = golden_mxfp8_linear_grouped_sum(latent.flatten(-2), wo_b, wo_b_scale, output_dtype=torch.bfloat16)
    return projected.to(output_dtype or torch.bfloat16)


@pl.jit.inline(auto_scope=False)
def retire_prefill_signals(
    completion_anchor: pl.Tensor,
    arrived: pld.DistributedTensor,
    group_base: pl.Scalar[pl.INT32],
    group_rank: pl.Scalar[pl.INT32],
    group_size: pl.constexpr,
):
    """Clear ready and consumed signals at a serialized entry boundary."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_signal_retire", allow_early_resolve=False):
        anchor = pl.read(completion_anchor, [0, 0, 0])
        for peer in pl.range(group_size):
            pld.system.notify(
                arrived, peer=group_base + group_rank, offsets=[peer, 0],
                value=0, op=pld.NotifyOp.Set,
            )
        pl.write(completion_anchor, [0, 0, 0], anchor)
    return completion_anchor


@pl.jit
def o_proj_hc_post_validation(
    attn_out: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    attn_out.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    o_proj_hc_post(
        attn_out, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        residual, post_mix, residual_mix, output, num_tokens,
        oa_row_tile=80, ob_row_tile=128,
    )
    return output


@pl.jit
def o_proj_hc_post_native_validation(
    attn_out: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Validate native M80 O-A between register quantization and fused O-B/HC post."""
    attn_out.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    total_rows = pl.tensor.dim(output, 0)
    padded_rows = ((total_rows + 79) // 80) * 80
    payload_all = pl.create_tensor([LOCAL_O_GROUPS * padded_rows, O_GROUP_IN], dtype=pl.FP8E4M3FN)
    scales_all = pl.create_tensor([LOCAL_O_GROUPS, padded_rows * (O_GROUP_IN // 32)], dtype=pl.FP8E8M0)
    quant_o_vf_pypto(attn_out, rope_cos, rope_sin, payload_all, scales_all, num_tokens, padded_rows)
    payload = pl.create_tensor([padded_rows, LOCAL_O_WIDTH], dtype=pl.FP8E4M3FN)
    activation_scale = pl.create_tensor([1, padded_rows * (LOCAL_O_WIDTH // 32)], dtype=pl.FP8E8M0)
    payload, activation_scale = o_proj_a_pypto(
        payload_all, wo_a, scales_all, wo_a_scale, payload, activation_scale,
        LOCAL_O_WIDTH, num_tokens, padded_rows,
    )
    ob_tasks = ((total_rows + 127) // 128) * ((D + O_B_N_TILE - 1) // O_B_N_TILE)
    # Process at least two output tiles per worker.
    ob_workers = pl.max(1, pl.min(ob_tasks // 2, pl.system.available_cluster_count()))
    with pl.spmd(ob_workers, name_hint="proj_o_b_hc_post"):
        output = _o_proj_b_hc_post(
            payload, wo_b, activation_scale, wo_b_scale,
            residual, post_mix, residual_mix, output, total_rows, 128,
        )
    return output


def build_o_proj_specs(active_tokens: int, *, pad_to_tile: bool = True):
    """Build deterministic output-projection fixtures with padded inactive rows."""
    from golden import ScalarSpec, TensorSpec
    from quant import pack_fp8_nz, quantize_mxfp8_upward_reference, pack_mx_b_scale, quantize_grouped_mxfp8_weight

    tokens = max(16, (active_tokens + 15) // 16 * 16) if pad_to_tile else active_tokens
    values = {}

    def inputs():
        if not values:
            generator = torch.Generator().manual_seed(17)
            attn_out = torch.randn(tokens, LOCAL_H * HEAD_DIM, generator=generator).bfloat16()
            attn_out[active_tokens:] = float("nan")
            wo_a = torch.randn(LOCAL_O_GROUPS, O_LORA, O_GROUP_IN, generator=generator) / O_GROUP_IN**0.5
            wo_a, wo_a_scale = quantize_grouped_mxfp8_weight(wo_a.bfloat16())
            wo_b = torch.randn(D, LOCAL_O_WIDTH, generator=generator) / LOCAL_O_WIDTH**0.5
            wo_b, wo_b_scale = quantize_mxfp8_upward_reference(wo_b)
            frequencies = 10000.0 ** (-torch.arange(0, ROPE_DIM, 2).float() / ROPE_DIM)
            angles = torch.arange(tokens).float()[:, None] * frequencies
            values.update(
                attn_out=attn_out, wo_a=wo_a, wo_a_scale=wo_a_scale,
                wo_b=pack_fp8_nz(wo_b.T.contiguous()),
                wo_b_scale=pack_mx_b_scale(wo_b_scale.T.contiguous()).view(torch.float8_e8m0fnu),
                rope_cos=angles.cos(), rope_sin=angles.sin(),
            )
        return values

    shapes = (
        ("attn_out", [tokens, LOCAL_H * HEAD_DIM], torch.bfloat16),
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
    attn_out = tensors["attn_out"][:active].unflatten(-1, (LOCAL_H, HEAD_DIM))
    result = golden_o_proj(
        attn_out, tensors["rope_cos"][:active], tensors["rope_sin"][:active],
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

    sublayer = torch.zeros_like(tensors["output"][:, 0, :], dtype=torch.bfloat16)
    projected = dict(tensors, output=sublayer)
    golden_o_proj_validation(projected)
    tensors["output"].copy_(golden_hc_post(
        sublayer, tensors["residual"], tensors["post_mix"], tensors["residual_mix"],
    ))


def compare_o_proj_hc_post(actual, expected, *, inputs, compare, **kwargs):
    """Check active projection tolerance and strict inactive-row residual mixing."""
    from golden import ratio_allclose

    active = int(inputs["num_tokens"])
    passed, detail = True, "no active rows"
    if active:
        passed, detail = compare(actual[:active], expected[:active], inputs=inputs, **kwargs)
    if active < actual.shape[0]:
        compare_tail = ratio_allclose(atol=1e-6, rtol=1e-5, max_error_ratio=0.0)
        tail_ok, tail_detail = compare_tail(actual[active:], expected[active:], inputs=inputs, **kwargs)
        return passed and tail_ok, f"{detail}; inactive rows: {tail_detail}"
    return passed, detail


if __name__ == "__main__":
    import argparse
    import functools
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="V4.1 fused output projection and HC post validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, help="active rows for a single padded case")
    parser.add_argument("--native-oa", action="store_true", help="validate native M80 O-A with fused C++ O-B/HC post")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    args = parser.parse_args()
    validation = o_proj_hc_post_native_validation if args.native_oa else o_proj_hc_post_validation
    if args.tokens is not None and args.tokens < 0:
        parser.error("tokens must be nonnegative")
    cases = ((0, True), (1, True), (17, True), (72, False), (129, True))
    if args.tokens is not None:
        cases = ((args.tokens, True),)
    for active, padded in cases:
        print(f"[O_PROJ] active={active}, padded={padded}, FP32 HC post", flush=True)
        result = run(
            fn=validation,
            specs=build_o_proj_hc_post_specs(active, pad_to_tile=padded),
            golden_fn=golden_o_proj_hc_post_validation,
            compare_fn={
                "output": functools.partial(compare_o_proj_hc_post, compare=ratio_allclose(atol=1e-2, rtol=1.0 / 128)),
            },
            config={
                "backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device,
            },
            compile_only=args.compile_only,
            save_data=args.save_data,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
