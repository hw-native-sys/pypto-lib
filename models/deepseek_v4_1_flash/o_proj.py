# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
# The mixed O-A kernel uses device-only VPTO vector helpers.
"""V4.1 inverse RoPE, grouped W8A8 MXFP8 output projection, and HC post."""

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


O_GROUP_ROWS = pl.dynamic("O_GROUP_ROWS")
O_PAD_ROWS = pl.dynamic("O_PAD_ROWS")
O_IN_SCALE_COLS = pl.dynamic("O_IN_SCALE_COLS")
O_OUT_SCALE_COLS = pl.dynamic("O_OUT_SCALE_COLS")

# tiling
O_A_N_TILE = 256
O_B_N_TILE = 160


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels" / "o_quant_vf.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _quant_o_vf(
    attn_out: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
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
def _o_proj_a_mixed(
    payload: pl.Tensor,
    weight: pl.Tensor,
    activation_scale: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.Tensor,
    output_scale: pl.Tensor,
    output_width: pl.Scalar[pl.INDEX],
    active_rows: pl.Scalar[pl.INT32],
    padded_rows: pl.Scalar[pl.INDEX],
    row_tile: pl.constexpr,
    group_tile: pl.constexpr,
):
    scales = pl.reshape(output_scale, [padded_rows // 16, output_width // 2])
    # Each packed scale band holds 16 rows or columns and 128 E8M0 codes per row or column.
    raw_a = pl.reshape(activation_scale, [output_width // O_LORA * padded_rows // 16, 2048])
    raw_b = pl.reshape(weight_scale, [output_width // 16, 2048])
    for batch in pl.parallel(output_width // (group_tile * O_LORA)):
        first_group = batch * group_tile
        w: pl.Tensor[[group_tile, O_GROUP_IN, O_LORA], pl.FP8E4M3FN, pl.NZ] = weight[first_group:first_group + group_tile, :, :]
        for cube_mt in pl.parallel(padded_rows // row_tile):
            cube_row = cube_mt * row_tile
            for worker in pl.spmd(group_tile * 4, optimizations=[pl.cross_core_slot(slot_num=1)]):
                local_group = worker // 4
                cube_group = first_group + local_group
                cube_nb = worker % 4
                cube_col = cube_nb * O_A_N_TILE
                input_row = cube_group * padded_rows + cube_row
                acc = pl.create_tile([row_tile, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                # Preload the first K256 panel.
                a = pl.load(payload, [input_row, 0], [row_tile, 256], target_memory=pl.Mem.Mat)
                weight_panel = pl.load(w, [local_group, 0, cube_col], [1, 256, 256], target_memory=pl.Mem.Mat)
                b = pl.reshape(weight_panel, [256, 256])
                a_band = input_row // 16
                b_band = (cube_group * O_LORA + cube_col) // 16
                # Gather each K128 scale fragment into its flat L1 slot and mark its valid extent.
                a_buf_lo: pl.Tile[[row_tile // 16, 64], pl.FP8E8M0, pl.MemRef("oa_sa", slots=4)[0], pl.Mem.Mat] = pl.tile.create([row_tile // 16, 64], pl.FP8E8M0, target_memory=pl.Mem.Mat, flat_layout=True)
                b_buf_lo: pl.Tile[[16, 64], pl.FP8E8M0, pl.MemRef("oa_sb", slots=4)[0], pl.Mem.Mat] = pl.tile.create([16, 64], pl.FP8E8M0, target_memory=pl.Mem.Mat, flat_layout=True)
                a_dst_lo = pl.tile.reshape(a_buf_lo, [row_tile // 16, 64])
                b_dst_lo = pl.tile.reshape(b_buf_lo, [16, 64])
                a_dst_lo = pl.tile.gather_row(a_dst_lo, raw_a, [0, 0], [a_band, 0], [row_tile // 16, 64])
                pl.tile.set_validshape(a_dst_lo, row_tile // 16, 64)
                b_dst_lo = pl.tile.gather_row(b_dst_lo, raw_b, [0, 0], [b_band, 0], [16, 64])
                pl.tile.set_validshape(b_dst_lo, 16, 64)
                a_buf_hi: pl.Tile[[row_tile // 16, 64], pl.FP8E8M0, pl.MemRef("oa_sa", slots=4)[1], pl.Mem.Mat] = pl.tile.create([row_tile // 16, 64], pl.FP8E8M0, target_memory=pl.Mem.Mat, flat_layout=True)
                b_buf_hi: pl.Tile[[16, 64], pl.FP8E8M0, pl.MemRef("oa_sb", slots=4)[1], pl.Mem.Mat] = pl.tile.create([16, 64], pl.FP8E8M0, target_memory=pl.Mem.Mat, flat_layout=True)
                a_dst_hi = pl.tile.reshape(a_buf_hi, [row_tile // 16, 64])
                b_dst_hi = pl.tile.reshape(b_buf_hi, [16, 64])
                a_dst_hi = pl.tile.gather_row(a_dst_hi, raw_a, [0, 0], [a_band, 64], [row_tile // 16, 64])
                pl.tile.set_validshape(a_dst_hi, row_tile // 16, 64)
                b_dst_hi = pl.tile.gather_row(b_dst_hi, raw_b, [0, 0], [b_band, 64], [16, 64])
                pl.tile.set_validshape(b_dst_hi, 16, 64)
                pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 0, core_type=pl.KernelType.AIC)
                for step in pl.unroll(16):
                    panel_slot = step % 2
                    next_slot = (step + 1) % 2
                    a_next = a
                    b_next = b
                    # Refill the alternate L1 slot after its last MTE1 read.
                    if step < 15:
                        if step >= 1:
                            pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, next_slot, core_type=pl.KernelType.AIC)
                        next_k = (step + 1) * 256
                        scale_byte = (step + 1) * 128
                        a_next = pl.load(payload, [input_row, next_k], [row_tile, 256], target_memory=pl.Mem.Mat)
                        weight_panel_next = pl.load(w, [local_group, next_k, cube_col], [1, 256, 256], target_memory=pl.Mem.Mat)
                        b_next = pl.reshape(weight_panel_next, [256, 256])
                        a_buf_lo_next: pl.Tile[[row_tile // 16, 64], pl.FP8E8M0, pl.MemRef("oa_sa", slots=4)[2 * next_slot], pl.Mem.Mat] = pl.tile.create([row_tile // 16, 64], pl.FP8E8M0, target_memory=pl.Mem.Mat, flat_layout=True)
                        b_buf_lo_next: pl.Tile[[16, 64], pl.FP8E8M0, pl.MemRef("oa_sb", slots=4)[2 * next_slot], pl.Mem.Mat] = pl.tile.create([16, 64], pl.FP8E8M0, target_memory=pl.Mem.Mat, flat_layout=True)
                        a_dst_lo_next = pl.tile.reshape(a_buf_lo_next, [row_tile // 16, 64])
                        b_dst_lo_next = pl.tile.reshape(b_buf_lo_next, [16, 64])
                        a_dst_lo_next = pl.tile.gather_row(a_dst_lo_next, raw_a, [0, 0], [a_band, scale_byte], [row_tile // 16, 64])
                        pl.tile.set_validshape(a_dst_lo_next, row_tile // 16, 64)
                        b_dst_lo_next = pl.tile.gather_row(b_dst_lo_next, raw_b, [0, 0], [b_band, scale_byte], [16, 64])
                        pl.tile.set_validshape(b_dst_lo_next, 16, 64)
                        a_buf_hi_next: pl.Tile[[row_tile // 16, 64], pl.FP8E8M0, pl.MemRef("oa_sa", slots=4)[2 * next_slot + 1], pl.Mem.Mat] = pl.tile.create([row_tile // 16, 64], pl.FP8E8M0, target_memory=pl.Mem.Mat, flat_layout=True)
                        b_buf_hi_next: pl.Tile[[16, 64], pl.FP8E8M0, pl.MemRef("oa_sb", slots=4)[2 * next_slot + 1], pl.Mem.Mat] = pl.tile.create([16, 64], pl.FP8E8M0, target_memory=pl.Mem.Mat, flat_layout=True)
                        a_dst_hi_next = pl.tile.reshape(a_buf_hi_next, [row_tile // 16, 64])
                        b_dst_hi_next = pl.tile.reshape(b_buf_hi_next, [16, 64])
                        a_dst_hi_next = pl.tile.gather_row(a_dst_hi_next, raw_a, [0, 0], [a_band, scale_byte + 64], [row_tile // 16, 64])
                        pl.tile.set_validshape(a_dst_hi_next, row_tile // 16, 64)
                        b_dst_hi_next = pl.tile.gather_row(b_dst_hi_next, raw_b, [0, 0], [b_band, scale_byte + 64], [16, 64])
                        pl.tile.set_validshape(b_dst_hi_next, 16, 64)
                        pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, next_slot, core_type=pl.KernelType.AIC)
                    pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, panel_slot, core_type=pl.KernelType.AIC)
                    # Stage both K128 operand sets before their two matrix operations.
                    if step > 0:
                        pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 0, core_type=pl.KernelType.AIC)
                    left_lo: pl.Tile[[row_tile, 128], pl.FP8E4M3FN, pl.MemRef("oa_left", slots=2)[0], pl.Mem.Left] = pl.tile.extract(a, 0, 0, [row_tile, 128], target_memory=pl.Mem.Left)
                    right_lo: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("oa_right", slots=2)[0], pl.Mem.Right] = pl.tile.extract(b, 0, 0, [128, 256], target_memory=pl.Mem.Right)
                    # Read each L1 slot through its fixed A or B MX layout.
                    if panel_slot == 0:
                        a_scale_lo: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("oa_sa", slots=4)[0], pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major, fractal=32)] = pl.tile.create([row_tile, 4], pl.FP8E8M0, target_memory=pl.Mem.Mat)
                        left_scale_lo = pl.tile.move(a_scale_lo, target_memory=pl.Mem.LeftScale)
                        b_scale_lo: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("oa_sb", slots=4)[0], pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major, fractal=32)] = pl.tile.create([4, 256], pl.FP8E8M0, target_memory=pl.Mem.Mat)
                        right_scale_lo = pl.tile.move(b_scale_lo, target_memory=pl.Mem.RightScale)
                    else:
                        a_scale_lo: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("oa_sa", slots=4)[2], pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major, fractal=32)] = pl.tile.create([row_tile, 4], pl.FP8E8M0, target_memory=pl.Mem.Mat)
                        left_scale_lo = pl.tile.move(a_scale_lo, target_memory=pl.Mem.LeftScale)
                        b_scale_lo: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("oa_sb", slots=4)[2], pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major, fractal=32)] = pl.tile.create([4, 256], pl.FP8E8M0, target_memory=pl.Mem.Mat)
                        right_scale_lo = pl.tile.move(b_scale_lo, target_memory=pl.Mem.RightScale)
                    pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, 0, core_type=pl.KernelType.AIC)
                    if step > 0:
                        pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 1, core_type=pl.KernelType.AIC)
                    left_hi: pl.Tile[[row_tile, 128], pl.FP8E4M3FN, pl.MemRef("oa_left", slots=2)[1], pl.Mem.Left] = pl.tile.extract(a, 0, 128, [row_tile, 128], target_memory=pl.Mem.Left)
                    right_hi: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef("oa_right", slots=2)[1], pl.Mem.Right] = pl.tile.extract(b, 128, 0, [128, 256], target_memory=pl.Mem.Right)
                    if panel_slot == 0:
                        a_scale_hi: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("oa_sa", slots=4)[1], pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major, fractal=32)] = pl.tile.create([row_tile, 4], pl.FP8E8M0, target_memory=pl.Mem.Mat)
                        left_scale_hi = pl.tile.move(a_scale_hi, target_memory=pl.Mem.LeftScale)
                        b_scale_hi: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("oa_sb", slots=4)[1], pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major, fractal=32)] = pl.tile.create([4, 256], pl.FP8E8M0, target_memory=pl.Mem.Mat)
                        right_scale_hi = pl.tile.move(b_scale_hi, target_memory=pl.Mem.RightScale)
                    else:
                        a_scale_hi: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("oa_sa", slots=4)[3], pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major, fractal=32)] = pl.tile.create([row_tile, 4], pl.FP8E8M0, target_memory=pl.Mem.Mat)
                        left_scale_hi = pl.tile.move(a_scale_hi, target_memory=pl.Mem.LeftScale)
                        b_scale_hi: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef("oa_sb", slots=4)[3], pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major, fractal=32)] = pl.tile.create([4, 256], pl.FP8E8M0, target_memory=pl.Mem.Mat)
                        right_scale_hi = pl.tile.move(b_scale_hi, target_memory=pl.Mem.RightScale)
                    pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, 1, core_type=pl.KernelType.AIC)
                    if step < 14:
                        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, panel_slot, core_type=pl.KernelType.AIC)
                    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, 0, core_type=pl.KernelType.AIC)
                    if step == 0:
                        acc = pl.matmul_mx(left_lo, left_scale_lo, right_lo, right_scale_lo)
                    else:
                        acc = pl.matmul_mx_acc(acc, left_lo, left_scale_lo, right_lo, right_scale_lo)
                    if step < 15:
                        pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 0, core_type=pl.KernelType.AIC)
                    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, 1, core_type=pl.KernelType.AIC)
                    acc = pl.matmul_mx_acc(acc, left_hi, left_scale_hi, right_hi, right_scale_hi)
                    if step < 15:
                        pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 1, core_type=pl.KernelType.AIC)
                    a = a_next
                    b = b_next
                # Publish the completed accumulator to the two quantization lanes.
                pl.system.set_flag(pl.PipeType.M, pl.PipeType.FIX, 0, core_type=pl.KernelType.AIC)
                pl.system.wait_flag(pl.PipeType.M, pl.PipeType.FIX, 0, core_type=pl.KernelType.AIC)
                for lane in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
                    row = cube_row
                    group = cube_group
                    nb = cube_nb
                    col = cube_col + lane * 128
                    received = pl.aiv_shard(acc)
                    rounded = pl.cast(received, pl.BF16, mode="rint")
                    flat = pl.reshape(rounded, [row_tile * 128])
                    valid_rows = pl.max(0, pl.min(row_tile, active_rows - row))
                    quant_bytes = pl.create_tile([row_tile * 128], dtype=pl.INT8, target_memory=pl.Mem.Vec)
                    scale_pairs = pl.create_tile([row_tile * 128 // 64], dtype=pl.UINT16, target_memory=pl.Mem.Vec)
                    scale_bytes = pl.reinterpret_view(scale_pairs, pl.UINT8, shape=[row_tile * 128 // 32])
                    quant_words = pl.reinterpret_view(quant_bytes, pl.UINT32, shape=[row_tile * 128 // 4])
                    row_codes = pl.create_tile([row_tile * 4], dtype=pl.UINT16, target_memory=pl.Mem.Vec)
                    with pl.reg_scope():
                        q_zero_row = pl.reg.broadcast(0, lanes=4, dtype=pl.UINT16)
                        # Process two rows as eight independent 32-value MX groups.
                        for q_pair in pl.hardware_loop((valid_rows + 1) // 2):
                            q_values = pl.reg.load(flat, offset=q_pair * 256, lanes=256)
                            q_bits = pl.reg.interpret_cast(q_values, dtype=pl.UINT16)
                            q_wide_mask = pl.reg.full_mask(lanes=256)
                            q_exp_mask_wide = pl.reg.broadcast(32640, lanes=256, dtype=pl.UINT16)
                            q_exponents = pl.reg.bitwise_and(q_bits, q_exp_mask_wide)
                            q_groups = pl.reg.reduce_max(q_exponents, q_wide_mask, group_size=32)
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
                            _q_quant_written = pl.reg.store(quant_words, offset=q_pair * 64, value=q_wide_words)
                            q_codes = pl.reg.shift_rights(q_group_shared, 7, q_group_mask)
                            _row_written = pl.reg.store(row_codes, offset=q_pair * 8, value=q_codes)
                        empty = pl.reg.broadcast(0, lanes=128, dtype=pl.INT8)
                        # Clear padded rows, including the upper half of an odd final pair.
                        for inactive in pl.hardware_loop(row_tile - valid_rows):
                            pad_row = valid_rows + inactive
                            _padding_written = pl.reg.store(quant_bytes, offset=pad_row * 128, value=empty)
                            _scales_padded = pl.reg.store(row_codes, offset=pad_row * 4, value=q_zero_row)
                        # Pack the four row codes into the two 16-row scale planes.
                        q_lane = pl.reg.index_gen(0, lanes=64, dtype=pl.UINT16)
                        q_mask = pl.reg.full_mask(lanes=64)
                        q_row_mask = pl.reg.broadcast(15, lanes=64, dtype=pl.UINT16)
                        q_row_id = pl.reg.bitwise_and(q_lane, q_row_mask)
                        q_row_offset = pl.reg.shift_lefts(q_row_id, 2, q_mask)
                        q_plane_mask = pl.reg.broadcast(16, lanes=64, dtype=pl.UINT16)
                        q_plane = pl.reg.bitwise_and(q_lane, q_plane_mask)
                        q_plane = pl.reg.shift_rights(q_plane, 3, q_mask)
                        q_lo_idx = pl.reg.bitwise_or(q_row_offset, q_plane)
                        q_hi_idx = pl.reg.adds(q_lo_idx, 1, q_mask)
                        q_valid = pl.reg.active_mask(32, lanes=64)
                        for q_block in pl.hardware_loop(row_tile // 16):
                            q_src = pl.reg.load(row_codes, offset=q_block * 64, lanes=64)
                            q_lo = pl.reg.gather_lanes(q_src, q_lo_idx)
                            q_hi = pl.reg.gather_lanes(q_src, q_hi_idx)
                            q_hi = pl.reg.shift_lefts(q_hi, 8, q_mask)
                            q_packed = pl.reg.bitwise_or(q_lo, q_hi)
                            _scales_stored = pl.reg.store(scale_pairs, offset=q_block * 32, value=q_packed, mask=q_valid)
                    quantized = pl.reshape(quant_bytes, [row_tile, 128])
                    quantized = pl.reinterpret_view(quantized, pl.FP8E4M3FN)
                    scale = pl.reshape(scale_bytes, [row_tile // 16, 128 // 2])
                    scale = pl.reinterpret_view(scale, pl.FP8E8M0)
                    output = pl.store(quantized, [row, group * 1024 + col], output)
                    packed = pl.reshape(scale, [row_tile // 16, 64])
                    scales = pl.store(packed, [row // 16, group * 512 + nb * 128 + lane * 64], scales)
    return output, output_scale


@pl.jit.inline(auto_scope=False)
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
    row_tile: pl.Scalar[pl.INDEX] = 80,
) -> tuple[pl.Tensor, pl.Tensor]:
    """Select the mixed worker grid for the runtime row tile and output width."""
    if row_tile == 128:
        output, output_scale = _o_proj_a_mixed(
            payload, weight, activation_scale, weight_scale, output, output_scale,
            output_width, active_rows, padded_rows, row_tile=128, group_tile=1,
        )
    elif output_width == 8 * O_LORA and pl.system.available_cluster_count() >= 32:
        output, output_scale = _o_proj_a_mixed(
            payload, weight, activation_scale, weight_scale, output, output_scale,
            output_width, active_rows, padded_rows, row_tile=80, group_tile=8,
        )
    else:
        output, output_scale = _o_proj_a_mixed(
            payload, weight, activation_scale, weight_scale, output, output_scale,
            output_width, active_rows, padded_rows, row_tile=80, group_tile=1,
        )
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
    vector_workers = 2 * pl.system.available_cluster_count()
    with pl.spmd(vector_workers, name_hint="quant_o_vf"):
        payload_all, scales_all = _quant_o_vf(
            attn_out, rope_cos, rope_sin, payload_all, scales_all,
            num_tokens, LOCAL_O_GROUPS, padded_rows,
        )
    payload = pl.create_tensor([padded_rows, LOCAL_O_WIDTH], dtype=pl.FP8E4M3FN)
    activation_scale = pl.create_tensor([1, padded_rows * (LOCAL_O_WIDTH // 32)], dtype=pl.FP8E8M0)
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
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    args = parser.parse_args()
    if args.tokens is not None and args.tokens < 0:
        parser.error("tokens must be nonnegative")
    cases = ((0, True), (1, True), (17, True), (72, False), (80, False), (81, False), (129, True))
    if args.tokens is not None:
        cases = ((args.tokens, True),)
    for active, padded in cases:
        print(f"[O_PROJ] active={active}, padded={padded}, FP32 HC post", flush=True)
        result = run(
            fn=o_proj_hc_post_validation,
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
