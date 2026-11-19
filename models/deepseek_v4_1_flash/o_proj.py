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
O_B_SCALE_COLS = LOCAL_O_WIDTH // MX_GROUP


O_GROUP_ROWS = pl.dynamic("O_GROUP_ROWS")
O_PAD_ROWS = pl.dynamic("O_PAD_ROWS")
O_IN_SCALE_COLS = pl.dynamic("O_IN_SCALE_COLS")
O_OUT_SCALE_COLS = pl.dynamic("O_OUT_SCALE_COLS")

# tiling
O_A_N_TILE = 256
O_B_N_TILE = 160
O_B_DECODE_N_TILE = 160
O_B_NATIVE_N_TILE = 256
O_B_K_TILE = 256
HC_ROW_TILE = 8
POST_STRIDE = 8


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
def o_proj_b_hc_post_cpp(
    payload: pl.Tensor, weight: pl.Tensor, activation_scale: pl.Tensor, weight_scale: pl.Tensor,
    residual: pl.Tensor, post_mix: pl.Tensor, residual_mix: pl.Tensor, output: pl.Out[pl.Tensor],
    rows: pl.Scalar[pl.INDEX], row_tile: pl.Scalar[pl.INDEX], workers: pl.Scalar[pl.INDEX],
):
    """Submit fused O-B and HC-post and return the output and group completion."""
    with pl.spmd(workers, name_hint="proj_o_b_hc_post") as ready:
        output = _o_proj_b_hc_post(
            payload, weight, activation_scale, weight_scale, residual, post_mix, residual_mix, output, rows, row_tile,
        )
    return output, ready


@pl.jit.inline(auto_scope=False)
def o_b_hc_native_tile(
    payload: pl.Tensor, weight: pl.Tensor, activation_scale: pl.Tensor, weight_scale: pl.Tensor,
    residual: pl.Tensor, post_mix: pl.Tensor, residual_mix: pl.Tensor, output: pl.Tensor,
    rows: pl.Scalar[pl.INDEX], workers: pl.Scalar[pl.INDEX], capacity: pl.Scalar[pl.INDEX],
    row_tile: pl.constexpr, n_tile: pl.constexpr, hc_rows: pl.constexpr,
):
    """Submit the complete mixed group and return its grid-wide completion."""
    weight_nz: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN, pl.NZ] = weight[:, :]
    scale_span = activation_scale[:, 0:capacity * O_B_SCALE_COLS]
    scale_matrix = pl.reshape(scale_span, [capacity, O_B_SCALE_COLS])
    a_scale = pl.tensor.view(scale_matrix, [capacity, O_B_SCALE_COLS], layout=pl.MX_A_ZZ)
    residual_count = pl.tensor.dim(residual, 0)
    mix_count = pl.tensor.dim(residual_mix, 0)
    out_count = pl.tensor.dim(output, 0)
    residual_rows = pl.reshape(residual, [residual_count * HC_MULT, D])
    mix_rows = pl.reshape(residual_mix, [mix_count, HC_MULT * HC_MULT])
    output_rows = pl.reshape(output, [out_count * HC_MULT, D])
    with pl.spmd(workers, name_hint="proj_o_b_hc_post", optimizations=[pl.cross_core_slot(slot_num=1)]) as ready:
        worker = pl.tile.get_block_idx()
        column_limit = D
        if row_tile == 80:
            # Each decode worker starts at its own column tile.
            group_count = pl.tile.get_block_num()
            column_limit = ((D // n_tile + group_count - 1 - worker) // group_count) * n_tile
            if group_count == D // n_tile:
                column_limit = n_tile
        for row in pl.range(0, rows, row_tile):
            for offset in pl.range(0, column_limit, n_tile):
                col = offset
                if row_tile == 80:
                    col = (offset // n_tile * pl.tile.get_block_num() + worker) * n_tile
                    owns_column = True
                else:
                    owns_column = (row // row_tile * (D // n_tile) + col // n_tile) % workers == worker
                if owns_column:
                    if row_tile == 80:
                        # Three K512 L1 banks feed alternating K128 operands across an eight-panel group.
                        pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 0, core_type=pl.KernelType.AIC)
                        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 0, core_type=pl.KernelType.AIC)
                        pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 1, core_type=pl.KernelType.AIC)
                        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 1, core_type=pl.KernelType.AIC)
                        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 2, core_type=pl.KernelType.AIC)
                        acc = pl.tile.create([row_tile, n_tile], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                        for k_group in pl.range(0, LOCAL_O_WIDTH, 4096):
                            pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 0, core_type=pl.KernelType.AIC)
                            a0: pl.Tile[[row_tile, 512], pl.FP8E4M3FN, pl.MemRef("ob_l1_a", slots=3)[0], pl.Mem.Mat] = pl.load(payload, [row, k_group], [row_tile, 512], target_memory=pl.Mem.Mat)
                            b0: pl.Tile[[512, n_tile], pl.FP8E4M3FN, pl.MemRef("ob_l1_b", slots=3)[0], pl.Mem.Mat] = pl.load(weight_nz, [k_group, col], [512, n_tile], target_memory=pl.Mem.Mat)
                            sa0_0: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[0], pl.Mem.Mat] = pl.load(a_scale, [row, k_group // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                            sb0_0: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[0], pl.Mem.Mat] = pl.load(weight_scale, [k_group // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                            sa0_1: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[1], pl.Mem.Mat] = pl.load(a_scale, [row, (k_group + 128) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                            sb0_1: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[1], pl.Mem.Mat] = pl.load(weight_scale, [(k_group + 128) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                            sa0_2: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[2], pl.Mem.Mat] = pl.load(a_scale, [row, (k_group + 256) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                            sb0_2: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[2], pl.Mem.Mat] = pl.load(weight_scale, [(k_group + 256) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                            sa0_3: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[3], pl.Mem.Mat] = pl.load(a_scale, [row, (k_group + 384) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                            sb0_3: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[3], pl.Mem.Mat] = pl.load(weight_scale, [(k_group + 384) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                            pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 0, core_type=pl.KernelType.AIC)
                            pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 1, core_type=pl.KernelType.AIC)
                            a1: pl.Tile[[row_tile, 512], pl.FP8E4M3FN, pl.MemRef("ob_l1_a", slots=3)[1], pl.Mem.Mat] = pl.load(payload, [row, (k_group + 512)], [row_tile, 512], target_memory=pl.Mem.Mat)
                            b1: pl.Tile[[512, n_tile], pl.FP8E4M3FN, pl.MemRef("ob_l1_b", slots=3)[1], pl.Mem.Mat] = pl.load(weight_nz, [(k_group + 512), col], [512, n_tile], target_memory=pl.Mem.Mat)
                            sa1_0: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[4], pl.Mem.Mat] = pl.load(a_scale, [row, (k_group + 512) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                            sb1_0: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[4], pl.Mem.Mat] = pl.load(weight_scale, [(k_group + 512) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                            sa1_1: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[5], pl.Mem.Mat] = pl.load(a_scale, [row, ((k_group + 512) + 128) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                            sb1_1: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[5], pl.Mem.Mat] = pl.load(weight_scale, [((k_group + 512) + 128) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                            sa1_2: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[6], pl.Mem.Mat] = pl.load(a_scale, [row, ((k_group + 512) + 256) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                            sb1_2: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[6], pl.Mem.Mat] = pl.load(weight_scale, [((k_group + 512) + 256) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                            sa1_3: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[7], pl.Mem.Mat] = pl.load(a_scale, [row, ((k_group + 512) + 384) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                            sb1_3: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[7], pl.Mem.Mat] = pl.load(weight_scale, [((k_group + 512) + 384) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                            pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 1, core_type=pl.KernelType.AIC)
                            pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 2, core_type=pl.KernelType.AIC)
                            a2: pl.Tile[[row_tile, 512], pl.FP8E4M3FN, pl.MemRef("ob_l1_a", slots=3)[2], pl.Mem.Mat] = pl.load(payload, [row, (k_group + 1024)], [row_tile, 512], target_memory=pl.Mem.Mat)
                            b2: pl.Tile[[512, n_tile], pl.FP8E4M3FN, pl.MemRef("ob_l1_b", slots=3)[2], pl.Mem.Mat] = pl.load(weight_nz, [(k_group + 1024), col], [512, n_tile], target_memory=pl.Mem.Mat)
                            sa2_0: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[8], pl.Mem.Mat] = pl.load(a_scale, [row, (k_group + 1024) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                            sb2_0: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[8], pl.Mem.Mat] = pl.load(weight_scale, [(k_group + 1024) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                            sa2_1: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[9], pl.Mem.Mat] = pl.load(a_scale, [row, ((k_group + 1024) + 128) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                            sb2_1: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[9], pl.Mem.Mat] = pl.load(weight_scale, [((k_group + 1024) + 128) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                            sa2_2: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[10], pl.Mem.Mat] = pl.load(a_scale, [row, ((k_group + 1024) + 256) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                            sb2_2: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[10], pl.Mem.Mat] = pl.load(weight_scale, [((k_group + 1024) + 256) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                            sa2_3: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[11], pl.Mem.Mat] = pl.load(a_scale, [row, ((k_group + 1024) + 384) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                            sb2_3: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[11], pl.Mem.Mat] = pl.load(weight_scale, [((k_group + 1024) + 384) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                            pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 2, core_type=pl.KernelType.AIC)
                            # Refill a bank after its last MTE1 read, while the next bank feeds the matrix pipe.
                            for panel in pl.unroll(8):
                                bank = panel % 3
                                panel_a = a0
                                panel_b = b0
                                scale_a0 = sa0_0
                                scale_b0 = sb0_0
                                scale_a1 = sa0_1
                                scale_b1 = sb0_1
                                scale_a2 = sa0_2
                                scale_b2 = sb0_2
                                scale_a3 = sa0_3
                                scale_b3 = sb0_3
                                if bank == 1:
                                    panel_a = a1
                                    panel_b = b1
                                    scale_a0 = sa1_0
                                    scale_b0 = sb1_0
                                    scale_a1 = sa1_1
                                    scale_b1 = sb1_1
                                    scale_a2 = sa1_2
                                    scale_b2 = sb1_2
                                    scale_a3 = sa1_3
                                    scale_b3 = sb1_3
                                if bank == 2:
                                    panel_a = a2
                                    panel_b = b2
                                    scale_a0 = sa2_0
                                    scale_b0 = sb2_0
                                    scale_a1 = sa2_1
                                    scale_b1 = sb2_1
                                    scale_a2 = sa2_2
                                    scale_b2 = sb2_2
                                    scale_a3 = sa2_3
                                    scale_b3 = sb2_3
                                pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, bank, core_type=pl.KernelType.AIC)
                                for pair in pl.unroll(2):
                                    part_a0 = scale_a0
                                    part_b0 = scale_b0
                                    part_a1 = scale_a1
                                    part_b1 = scale_b1
                                    if pair == 1:
                                        part_a0 = scale_a2
                                        part_b0 = scale_b2
                                        part_a1 = scale_a3
                                        part_b1 = scale_b3
                                    pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 0, core_type=pl.KernelType.AIC)
                                    al: pl.Tile[[row_tile, 128], pl.FP8E4M3FN, pl.MemRef("ob_l0_a", slots=2)[0], pl.Mem.Left] = pl.tile.extract(panel_a, 0, pair * 256 + 0, [row_tile, 128], target_memory=pl.Mem.Left)
                                    br: pl.Tile[[128, n_tile], pl.FP8E4M3FN, pl.MemRef("ob_l0_b", slots=2)[0], pl.Mem.Right] = pl.tile.extract(panel_b, pair * 256 + 0, 0, [128, n_tile], target_memory=pl.Mem.Right)
                                    sal = pl.tile.move(part_a0, target_memory=pl.Mem.LeftScale)
                                    sbr = pl.tile.move(part_b0, target_memory=pl.Mem.RightScale)
                                    pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, 0, core_type=pl.KernelType.AIC)
                                    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, 0, core_type=pl.KernelType.AIC)
                                    if k_group == 0 and panel == 0 and pair == 0:
                                        acc = pl.matmul_mx(al, sal, br, sbr)
                                    else:
                                        acc = pl.matmul_mx_acc(acc, al, sal, br, sbr)
                                    pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 0, core_type=pl.KernelType.AIC)
                                    pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 1, core_type=pl.KernelType.AIC)
                                    al: pl.Tile[[row_tile, 128], pl.FP8E4M3FN, pl.MemRef("ob_l0_a", slots=2)[1], pl.Mem.Left] = pl.tile.extract(panel_a, 0, pair * 256 + 128, [row_tile, 128], target_memory=pl.Mem.Left)
                                    br: pl.Tile[[128, n_tile], pl.FP8E4M3FN, pl.MemRef("ob_l0_b", slots=2)[1], pl.Mem.Right] = pl.tile.extract(panel_b, pair * 256 + 128, 0, [128, n_tile], target_memory=pl.Mem.Right)
                                    sal = pl.tile.move(part_a1, target_memory=pl.Mem.LeftScale)
                                    sbr = pl.tile.move(part_b1, target_memory=pl.Mem.RightScale)
                                    pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, 1, core_type=pl.KernelType.AIC)
                                    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, 1, core_type=pl.KernelType.AIC)
                                    acc = pl.matmul_mx_acc(acc, al, sal, br, sbr)
                                    pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 1, core_type=pl.KernelType.AIC)
                                pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, bank, core_type=pl.KernelType.AIC)
                                if panel == 0:
                                    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 0, core_type=pl.KernelType.AIC)
                                    a0: pl.Tile[[row_tile, 512], pl.FP8E4M3FN, pl.MemRef("ob_l1_a", slots=3)[0], pl.Mem.Mat] = pl.load(payload, [row, (k_group + 1536)], [row_tile, 512], target_memory=pl.Mem.Mat)
                                    b0: pl.Tile[[512, n_tile], pl.FP8E4M3FN, pl.MemRef("ob_l1_b", slots=3)[0], pl.Mem.Mat] = pl.load(weight_nz, [(k_group + 1536), col], [512, n_tile], target_memory=pl.Mem.Mat)
                                    sa0_0: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[0], pl.Mem.Mat] = pl.load(a_scale, [row, (k_group + 1536) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                                    sb0_0: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[0], pl.Mem.Mat] = pl.load(weight_scale, [(k_group + 1536) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                                    sa0_1: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[1], pl.Mem.Mat] = pl.load(a_scale, [row, ((k_group + 1536) + 128) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                                    sb0_1: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[1], pl.Mem.Mat] = pl.load(weight_scale, [((k_group + 1536) + 128) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                                    sa0_2: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[2], pl.Mem.Mat] = pl.load(a_scale, [row, ((k_group + 1536) + 256) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                                    sb0_2: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[2], pl.Mem.Mat] = pl.load(weight_scale, [((k_group + 1536) + 256) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                                    sa0_3: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[3], pl.Mem.Mat] = pl.load(a_scale, [row, ((k_group + 1536) + 384) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                                    sb0_3: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[3], pl.Mem.Mat] = pl.load(weight_scale, [((k_group + 1536) + 384) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                                    pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 0, core_type=pl.KernelType.AIC)
                                if panel == 1:
                                    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 1, core_type=pl.KernelType.AIC)
                                    a1: pl.Tile[[row_tile, 512], pl.FP8E4M3FN, pl.MemRef("ob_l1_a", slots=3)[1], pl.Mem.Mat] = pl.load(payload, [row, (k_group + 2048)], [row_tile, 512], target_memory=pl.Mem.Mat)
                                    b1: pl.Tile[[512, n_tile], pl.FP8E4M3FN, pl.MemRef("ob_l1_b", slots=3)[1], pl.Mem.Mat] = pl.load(weight_nz, [(k_group + 2048), col], [512, n_tile], target_memory=pl.Mem.Mat)
                                    sa1_0: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[4], pl.Mem.Mat] = pl.load(a_scale, [row, (k_group + 2048) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                                    sb1_0: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[4], pl.Mem.Mat] = pl.load(weight_scale, [(k_group + 2048) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                                    sa1_1: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[5], pl.Mem.Mat] = pl.load(a_scale, [row, ((k_group + 2048) + 128) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                                    sb1_1: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[5], pl.Mem.Mat] = pl.load(weight_scale, [((k_group + 2048) + 128) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                                    sa1_2: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[6], pl.Mem.Mat] = pl.load(a_scale, [row, ((k_group + 2048) + 256) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                                    sb1_2: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[6], pl.Mem.Mat] = pl.load(weight_scale, [((k_group + 2048) + 256) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                                    sa1_3: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[7], pl.Mem.Mat] = pl.load(a_scale, [row, ((k_group + 2048) + 384) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                                    sb1_3: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[7], pl.Mem.Mat] = pl.load(weight_scale, [((k_group + 2048) + 384) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                                    pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 1, core_type=pl.KernelType.AIC)
                                if panel == 2:
                                    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 2, core_type=pl.KernelType.AIC)
                                    a2: pl.Tile[[row_tile, 512], pl.FP8E4M3FN, pl.MemRef("ob_l1_a", slots=3)[2], pl.Mem.Mat] = pl.load(payload, [row, (k_group + 2560)], [row_tile, 512], target_memory=pl.Mem.Mat)
                                    b2: pl.Tile[[512, n_tile], pl.FP8E4M3FN, pl.MemRef("ob_l1_b", slots=3)[2], pl.Mem.Mat] = pl.load(weight_nz, [(k_group + 2560), col], [512, n_tile], target_memory=pl.Mem.Mat)
                                    sa2_0: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[8], pl.Mem.Mat] = pl.load(a_scale, [row, (k_group + 2560) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                                    sb2_0: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[8], pl.Mem.Mat] = pl.load(weight_scale, [(k_group + 2560) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                                    sa2_1: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[9], pl.Mem.Mat] = pl.load(a_scale, [row, ((k_group + 2560) + 128) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                                    sb2_1: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[9], pl.Mem.Mat] = pl.load(weight_scale, [((k_group + 2560) + 128) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                                    sa2_2: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[10], pl.Mem.Mat] = pl.load(a_scale, [row, ((k_group + 2560) + 256) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                                    sb2_2: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[10], pl.Mem.Mat] = pl.load(weight_scale, [((k_group + 2560) + 256) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                                    sa2_3: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[11], pl.Mem.Mat] = pl.load(a_scale, [row, ((k_group + 2560) + 384) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                                    sb2_3: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[11], pl.Mem.Mat] = pl.load(weight_scale, [((k_group + 2560) + 384) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                                    pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 2, core_type=pl.KernelType.AIC)
                                if panel == 3:
                                    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 0, core_type=pl.KernelType.AIC)
                                    a0: pl.Tile[[row_tile, 512], pl.FP8E4M3FN, pl.MemRef("ob_l1_a", slots=3)[0], pl.Mem.Mat] = pl.load(payload, [row, (k_group + 3072)], [row_tile, 512], target_memory=pl.Mem.Mat)
                                    b0: pl.Tile[[512, n_tile], pl.FP8E4M3FN, pl.MemRef("ob_l1_b", slots=3)[0], pl.Mem.Mat] = pl.load(weight_nz, [(k_group + 3072), col], [512, n_tile], target_memory=pl.Mem.Mat)
                                    sa0_0: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[0], pl.Mem.Mat] = pl.load(a_scale, [row, (k_group + 3072) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                                    sb0_0: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[0], pl.Mem.Mat] = pl.load(weight_scale, [(k_group + 3072) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                                    sa0_1: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[1], pl.Mem.Mat] = pl.load(a_scale, [row, ((k_group + 3072) + 128) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                                    sb0_1: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[1], pl.Mem.Mat] = pl.load(weight_scale, [((k_group + 3072) + 128) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                                    sa0_2: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[2], pl.Mem.Mat] = pl.load(a_scale, [row, ((k_group + 3072) + 256) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                                    sb0_2: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[2], pl.Mem.Mat] = pl.load(weight_scale, [((k_group + 3072) + 256) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                                    sa0_3: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[3], pl.Mem.Mat] = pl.load(a_scale, [row, ((k_group + 3072) + 384) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                                    sb0_3: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[3], pl.Mem.Mat] = pl.load(weight_scale, [((k_group + 3072) + 384) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                                    pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 0, core_type=pl.KernelType.AIC)
                                if panel == 4:
                                    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 1, core_type=pl.KernelType.AIC)
                                    a1: pl.Tile[[row_tile, 512], pl.FP8E4M3FN, pl.MemRef("ob_l1_a", slots=3)[1], pl.Mem.Mat] = pl.load(payload, [row, (k_group + 3584)], [row_tile, 512], target_memory=pl.Mem.Mat)
                                    b1: pl.Tile[[512, n_tile], pl.FP8E4M3FN, pl.MemRef("ob_l1_b", slots=3)[1], pl.Mem.Mat] = pl.load(weight_nz, [(k_group + 3584), col], [512, n_tile], target_memory=pl.Mem.Mat)
                                    sa1_0: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[4], pl.Mem.Mat] = pl.load(a_scale, [row, (k_group + 3584) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                                    sb1_0: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[4], pl.Mem.Mat] = pl.load(weight_scale, [(k_group + 3584) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                                    sa1_1: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[5], pl.Mem.Mat] = pl.load(a_scale, [row, ((k_group + 3584) + 128) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                                    sb1_1: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[5], pl.Mem.Mat] = pl.load(weight_scale, [((k_group + 3584) + 128) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                                    sa1_2: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[6], pl.Mem.Mat] = pl.load(a_scale, [row, ((k_group + 3584) + 256) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                                    sb1_2: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[6], pl.Mem.Mat] = pl.load(weight_scale, [((k_group + 3584) + 256) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                                    sa1_3: pl.Tile[[row_tile, 4], pl.FP8E8M0, pl.MemRef("ob_ls_a", slots=12)[7], pl.Mem.Mat] = pl.load(a_scale, [row, ((k_group + 3584) + 384) // MX_GROUP], [row_tile, 4], target_memory=pl.Mem.Mat)
                                    sb1_3: pl.Tile[[4, n_tile], pl.FP8E8M0, pl.MemRef("ob_ls_b", slots=12)[7], pl.Mem.Mat] = pl.load(weight_scale, [((k_group + 3584) + 384) // MX_GROUP, col], [4, n_tile], target_memory=pl.Mem.Mat)
                                    pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 1, core_type=pl.KernelType.AIC)
                        pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 0, core_type=pl.KernelType.AIC)
                        pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 0, core_type=pl.KernelType.AIC)
                        pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 1, core_type=pl.KernelType.AIC)
                        pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 1, core_type=pl.KernelType.AIC)
                        pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 2, core_type=pl.KernelType.AIC)
                        pl.system.set_flag(pl.PipeType.M, pl.PipeType.FIX, 0, core_type=pl.KernelType.AIC)
                        pl.system.wait_flag(pl.PipeType.M, pl.PipeType.FIX, 0, core_type=pl.KernelType.AIC)
                    else:
                        lhs = pl.load(payload, [row, 0], [row_tile, O_B_K_TILE], target_memory=pl.Mem.Mat)
                        rhs = pl.load(weight_nz, [0, col], [O_B_K_TILE, n_tile], target_memory=pl.Mem.Mat)
                        lhs_scale = pl.load(a_scale, [row, 0], [row_tile, O_B_K_TILE // MX_GROUP])
                        rhs_scale = pl.load(weight_scale, [0, col], [O_B_K_TILE // MX_GROUP, n_tile])
                        acc = pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale)
                        for k in pl.pipeline(O_B_K_TILE, LOCAL_O_WIDTH, O_B_K_TILE, stage=2):
                            lhs_next = pl.load(payload, [row, k], [row_tile, O_B_K_TILE], target_memory=pl.Mem.Mat)
                            rhs_next = pl.load(weight_nz, [k, col], [O_B_K_TILE, n_tile], target_memory=pl.Mem.Mat)
                            lhs_scale_next = pl.load(a_scale, [row, k // MX_GROUP], [row_tile, O_B_K_TILE // MX_GROUP])
                            rhs_scale_next = pl.load(weight_scale, [k // MX_GROUP, col], [O_B_K_TILE // MX_GROUP, n_tile])
                            acc = pl.matmul_mx_acc(acc, lhs_next, lhs_scale_next, rhs_next, rhs_scale_next)
                    for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                        if row_tile == 80:
                            start = row + lane * (row_tile // 2)
                            valid = pl.max(0, pl.min(row_tile // 2, rows - start))
                            streams = pl.tile.load(residual_rows, [start * HC_MULT, col], [hc_rows * HC_MULT, n_tile], valid_shape=[valid * HC_MULT, n_tile], target_memory=pl.Mem.Vec)
                            post = pl.tile.load(post_mix, [start, 0], [hc_rows, POST_STRIDE], valid_shape=[valid, HC_MULT], target_memory=pl.Mem.Vec)
                            matrix = pl.tile.load(mix_rows, [start, 0], [hc_rows, HC_MULT * HC_MULT], valid_shape=[valid, HC_MULT * HC_MULT], target_memory=pl.Mem.Vec)
                            half = pl.aiv_shard(acc)
                            rounded = pl.cast(half, target_type=pl.BF16, mode="rint")
                            flat = pl.tile.reshape(rounded, [row_tile // 2 * n_tile])
                            # Process eight rows per vector call and keep all four HC results in the hardware loop.
                            for group in pl.unroll(hc_rows // 8):
                                if valid >= (group + 1) * 8:
                                    with pl.reg_scope():
                                        mask = pl.reg.full_mask(lanes=64)
                                        for local in pl.hardware_loop(8):
                                            local_row = group * 8 + local
                                            row_offset = local_row * HC_MULT * n_tile
                                            # Load all inputs for columns 0:64, 64:128 and 96:160 before the overlapping stores.
                                            raw_low = pl.reg.load(flat, local_row * n_tile, lanes=64)
                                            x_low = pl.reg.convert(raw_low, dtype=pl.FP32)
                                            r0_low = pl.reg.load(streams, row_offset, lanes=64)
                                            r1_low = pl.reg.load(streams, row_offset + n_tile, lanes=64)
                                            r2_low = pl.reg.load(streams, row_offset + 2 * n_tile, lanes=64)
                                            r3_low = pl.reg.load(streams, row_offset + 3 * n_tile, lanes=64)
                                            raw_mid = pl.reg.load(flat, local_row * n_tile + 64, lanes=64)
                                            x_mid = pl.reg.convert(raw_mid, dtype=pl.FP32)
                                            r0_mid = pl.reg.load(streams, row_offset + 64, lanes=64)
                                            r1_mid = pl.reg.load(streams, row_offset + n_tile + 64, lanes=64)
                                            r2_mid = pl.reg.load(streams, row_offset + 2 * n_tile + 64, lanes=64)
                                            r3_mid = pl.reg.load(streams, row_offset + 3 * n_tile + 64, lanes=64)
                                            raw_high = pl.reg.load(flat, local_row * n_tile + 96, lanes=64)
                                            x_high = pl.reg.convert(raw_high, dtype=pl.FP32)
                                            r0_high = pl.reg.load(streams, row_offset + 96, lanes=64)
                                            r1_high = pl.reg.load(streams, row_offset + n_tile + 96, lanes=64)
                                            r2_high = pl.reg.load(streams, row_offset + 2 * n_tile + 96, lanes=64)
                                            r3_high = pl.reg.load(streams, row_offset + 3 * n_tile + 96, lanes=64)
                                            for h in pl.hardware_loop(HC_MULT):
                                                post_weight = pl.reg.broadcast(pl.reg.load(post, local_row * POST_STRIDE + h, lanes=1), lanes=64, dtype=pl.FP32)
                                                out_low = pl.reg.mul(x_low, post_weight, mask)
                                                out_mid = pl.reg.mul(x_mid, post_weight, mask)
                                                out_high = pl.reg.mul(x_high, post_weight, mask)
                                                mix0_weight = pl.reg.broadcast(pl.reg.load(matrix, local_row * HC_MULT * HC_MULT + h, lanes=1), lanes=64, dtype=pl.FP32)
                                                out_low = pl.reg.mula(out_low, r0_low, mix0_weight, mask)
                                                out_mid = pl.reg.mula(out_mid, r0_mid, mix0_weight, mask)
                                                out_high = pl.reg.mula(out_high, r0_high, mix0_weight, mask)
                                                mix1_weight = pl.reg.broadcast(pl.reg.load(matrix, local_row * HC_MULT * HC_MULT + HC_MULT + h, lanes=1), lanes=64, dtype=pl.FP32)
                                                out_low = pl.reg.mula(out_low, r1_low, mix1_weight, mask)
                                                out_mid = pl.reg.mula(out_mid, r1_mid, mix1_weight, mask)
                                                out_high = pl.reg.mula(out_high, r1_high, mix1_weight, mask)
                                                mix2_weight = pl.reg.broadcast(pl.reg.load(matrix, local_row * HC_MULT * HC_MULT + 2 * HC_MULT + h, lanes=1), lanes=64, dtype=pl.FP32)
                                                out_low = pl.reg.mula(out_low, r2_low, mix2_weight, mask)
                                                out_mid = pl.reg.mula(out_mid, r2_mid, mix2_weight, mask)
                                                out_high = pl.reg.mula(out_high, r2_high, mix2_weight, mask)
                                                mix3_weight = pl.reg.broadcast(pl.reg.load(matrix, local_row * HC_MULT * HC_MULT + 3 * HC_MULT + h, lanes=1), lanes=64, dtype=pl.FP32)
                                                out_low = pl.reg.mula(out_low, r3_low, mix3_weight, mask)
                                                out_mid = pl.reg.mula(out_mid, r3_mid, mix3_weight, mask)
                                                out_high = pl.reg.mula(out_high, r3_high, mix3_weight, mask)
                                                _stored = pl.reg.store(streams, row_offset + h * n_tile, out_low)
                                                _stored = pl.reg.store(streams, row_offset + h * n_tile + 64, out_mid)
                                                _stored = pl.reg.store(streams, row_offset + h * n_tile + 96, out_high)
                            # Finish the active rows beyond the last full group.
                            if valid % 8 != 0:
                                with pl.reg_scope():
                                    mask_tail = pl.reg.full_mask(lanes=n_tile)
                                    for extra in pl.hardware_loop(valid % 8):
                                        rr_tail = valid // 8 * 8 + extra
                                        r_off_tail = rr_tail * HC_MULT * n_tile
                                        projection_tail = pl.reg.load(flat, rr_tail * n_tile, lanes=n_tile)
                                        x_tail = pl.reg.convert(projection_tail, dtype=pl.FP32)
                                        r0_tail = pl.reg.load(streams, r_off_tail, lanes=n_tile)
                                        r1_tail = pl.reg.load(streams, r_off_tail + n_tile, lanes=n_tile)
                                        r2_tail = pl.reg.load(streams, r_off_tail + 2 * n_tile, lanes=n_tile)
                                        r3_tail = pl.reg.load(streams, r_off_tail + 3 * n_tile, lanes=n_tile)
                                        for h_tail in pl.unroll(HC_MULT):
                                            c0_tail = pl.reg.broadcast(pl.reg.load(post, rr_tail * POST_STRIDE + h_tail, lanes=1), lanes=n_tile, dtype=pl.FP32)
                                            c1_tail = pl.reg.broadcast(pl.reg.load(matrix, rr_tail * HC_MULT * HC_MULT + h_tail, lanes=1), lanes=n_tile, dtype=pl.FP32)
                                            c2_tail = pl.reg.broadcast(pl.reg.load(matrix, rr_tail * HC_MULT * HC_MULT + HC_MULT + h_tail, lanes=1), lanes=n_tile, dtype=pl.FP32)
                                            c3_tail = pl.reg.broadcast(pl.reg.load(matrix, rr_tail * HC_MULT * HC_MULT + 2 * HC_MULT + h_tail, lanes=1), lanes=n_tile, dtype=pl.FP32)
                                            c4_tail = pl.reg.broadcast(pl.reg.load(matrix, rr_tail * HC_MULT * HC_MULT + 3 * HC_MULT + h_tail, lanes=1), lanes=n_tile, dtype=pl.FP32)
                                            value_tail = pl.reg.mul(x_tail, c0_tail, mask_tail)
                                            value_tail = pl.reg.mula(value_tail, r0_tail, c1_tail, mask_tail)
                                            value_tail = pl.reg.mula(value_tail, r1_tail, c2_tail, mask_tail)
                                            value_tail = pl.reg.mula(value_tail, r2_tail, c3_tail, mask_tail)
                                            value_tail = pl.reg.mula(value_tail, r3_tail, c4_tail, mask_tail)
                                            _stored_tail = pl.reg.store(streams, r_off_tail + h_tail * n_tile, value_tail)
                            result = pl.tile.reshape(streams, [hc_rows * HC_MULT, n_tile])
                            result = pl.set_validshape(result, valid * HC_MULT, n_tile)
                            pl.tile.store(result, [start * HC_MULT, col], output_rows)
                        else:
                            half = pl.aiv_shard(acc)
                            rounded = pl.cast(half, target_type=pl.BF16, mode="rint")
                            flat = pl.tile.reshape(rounded, [row_tile // 2 * n_tile])
                            start = row + lane * (row_tile // 2)
                            valid = pl.max(0, pl.min(row_tile // 2, rows - start))
                            for group in pl.range(0, valid, hc_rows):
                                count = pl.min(hc_rows, valid - group)
                                row_base = start + group
                                streams = pl.tile.load(residual_rows, [row_base * HC_MULT, col], [hc_rows * HC_MULT, n_tile], valid_shape=[count * HC_MULT, n_tile], target_memory=pl.Mem.Vec)
                                post = pl.tile.load(post_mix, [row_base, 0], [hc_rows, POST_STRIDE], valid_shape=[count, HC_MULT], target_memory=pl.Mem.Vec)
                                matrix = pl.tile.load(mix_rows, [row_base, 0], [hc_rows, HC_MULT * HC_MULT], valid_shape=[count, HC_MULT * HC_MULT], target_memory=pl.Mem.Vec)
                                mixed = pl.tile.create([hc_rows * HC_MULT * n_tile], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                                with pl.reg_scope():
                                    mask = pl.reg.full_mask(lanes=n_tile)
                                    for rr in pl.hardware_loop(count):
                                        r_off = rr * HC_MULT * n_tile
                                        projection = pl.reg.load(flat, (group + rr) * n_tile, lanes=n_tile)
                                        x = pl.reg.convert(projection, dtype=pl.FP32)
                                        r0 = pl.reg.load(streams, r_off, lanes=n_tile)
                                        r1 = pl.reg.load(streams, r_off + n_tile, lanes=n_tile)
                                        r2 = pl.reg.load(streams, r_off + 2 * n_tile, lanes=n_tile)
                                        r3 = pl.reg.load(streams, r_off + 3 * n_tile, lanes=n_tile)
                                        for h in pl.unroll(HC_MULT):
                                            c0 = pl.reg.broadcast(pl.reg.load(post, rr * POST_STRIDE + h, lanes=1), lanes=n_tile, dtype=pl.FP32)
                                            c1 = pl.reg.broadcast(pl.reg.load(matrix, rr * HC_MULT * HC_MULT + h, lanes=1), lanes=n_tile, dtype=pl.FP32)
                                            c2 = pl.reg.broadcast(pl.reg.load(matrix, rr * HC_MULT * HC_MULT + HC_MULT + h, lanes=1), lanes=n_tile, dtype=pl.FP32)
                                            c3 = pl.reg.broadcast(pl.reg.load(matrix, rr * HC_MULT * HC_MULT + 2 * HC_MULT + h, lanes=1), lanes=n_tile, dtype=pl.FP32)
                                            c4 = pl.reg.broadcast(pl.reg.load(matrix, rr * HC_MULT * HC_MULT + 3 * HC_MULT + h, lanes=1), lanes=n_tile, dtype=pl.FP32)
                                            value = pl.reg.mul(x, c0, mask)
                                            value = pl.reg.mula(value, r0, c1, mask)
                                            value = pl.reg.mula(value, r1, c2, mask)
                                            value = pl.reg.mula(value, r2, c3, mask)
                                            value = pl.reg.mula(value, r3, c4, mask)
                                            _stored = pl.reg.store(mixed, r_off + h * n_tile, value)
                                result = pl.tile.reshape(mixed, [hc_rows * HC_MULT, n_tile])
                                result = pl.set_validshape(result, count * HC_MULT, n_tile)
                                pl.tile.store(result, [row_base * HC_MULT, col], output_rows)
                    if row_tile == 80:
                        # Finish the FIFO transfer before reusing the accumulator for another column.
                        pl.system.barrier(pipe=pl.PipeType.ALL, core_type=pl.KernelType.AIC)
    return output, ready


@pl.jit.inline(auto_scope=False)
def o_proj_b_hc_post_native(
    payload: pl.Tensor, weight: pl.Tensor, activation_scale: pl.Tensor, weight_scale: pl.Tensor,
    residual: pl.Tensor, post_mix: pl.Tensor, residual_mix: pl.Tensor, output: pl.Out[pl.Tensor],
    rows: pl.Scalar[pl.INDEX], row_tile: pl.Scalar[pl.INDEX], workers: pl.Scalar[pl.INDEX],
):
    """Submit the Native tile for model chunk capacities with a static MX scale view."""
    capacity = pl.tensor.dim(payload, 0)
    ready = None
    if capacity == 80:
        output, ready = o_b_hc_native_tile(payload, weight, activation_scale, weight_scale, residual, post_mix, residual_mix, output, rows, workers, 80, 80, O_B_DECODE_N_TILE, 40)
    elif capacity > 0 and capacity <= 1024 and capacity % 128 == 0:
        for extent in pl.unroll(128, 1025, 128):
            if capacity == extent:
                output, ready = o_b_hc_native_tile(payload, weight, activation_scale, weight_scale, residual, post_mix, residual_mix, output, rows, workers, extent, 128, O_B_NATIVE_N_TILE, HC_ROW_TILE)
    else:
        output, ready = o_proj_b_hc_post_cpp(payload, weight, activation_scale, weight_scale, residual, post_mix, residual_mix, output, rows, row_tile, workers)
    return output, ready


@pl.jit.inline(auto_scope=False)
def o_proj_hc_post(
    attended: pl.Tensor,
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
            attended, rope_cos, rope_sin, payload_all, scales_all,
            num_tokens, LOCAL_O_GROUPS, padded_rows,
        )
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
    output, ready = o_proj_b_hc_post_cpp(
        payload, wo_b, activation_scale, wo_b_scale, residual, post_mix, residual_mix, output,
        total_rows, ob_row_tile, ob_workers,
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
    attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
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
    attended.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    o_proj_hc_post(
        attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        residual, post_mix, residual_mix, output, num_tokens,
        oa_row_tile=80, ob_row_tile=128,
    )
    return output


def _build_native_o_proj_validation(fused_ob):
    @pl.jit
    def o_proj_hc_post_native_validation(
        attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
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
        """Validate native M80 O-A followed by the selected fused O-B and HC post."""
        attended.bind_dynamic(0, T_DYN)
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
        vector_workers = 2 * pl.system.available_cluster_count()
        with pl.spmd(vector_workers, name_hint="quant_o_vf"):
            payload_all, scales_all = _quant_o_vf(
                attended, rope_cos, rope_sin, payload_all, scales_all, num_tokens, LOCAL_O_GROUPS, padded_rows,
            )
        payload = pl.create_tensor([padded_rows, LOCAL_O_WIDTH], dtype=pl.FP8E4M3FN)
        activation_scale = pl.create_tensor([1, padded_rows * (LOCAL_O_WIDTH // 32)], dtype=pl.FP8E8M0)
        payload, activation_scale = o_proj_a_pypto(
            payload_all, wo_a, scales_all, wo_a_scale, payload, activation_scale,
            LOCAL_O_WIDTH, num_tokens, padded_rows,
        )
        ob_tasks = ((total_rows + 127) // 128) * ((D + O_B_N_TILE - 1) // O_B_N_TILE)
        # Process at least two output tiles per worker.
        ob_workers = pl.max(1, pl.min(ob_tasks // 2, pl.system.available_cluster_count()))
        output, ready = fused_ob(
            payload, wo_b, activation_scale, wo_b_scale, residual, post_mix, residual_mix, output,
            total_rows, 128, ob_workers,
        )
        return output

    return o_proj_hc_post_native_validation


o_proj_hc_post_native_validation = _build_native_o_proj_validation(o_proj_b_hc_post_cpp)
o_proj_b_hc_post_native_validation = _build_native_o_proj_validation(o_proj_b_hc_post_native)


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
    runs = [(validation, active, padded) for active, padded in cases]
    # The Native HC-post register kernel uses VPTO device objects.
    if args.tokens is None and not args.native_oa and not args.execution_mode:
        # With M80 O-A packing these cases allocate 80, 640, and 160 payload rows.
        runs.extend((o_proj_b_hc_post_native_validation, active, False) for active in (72, 561, 81))
    for validation, active, padded in runs:
        native_ob = validation is o_proj_b_hc_post_native_validation
        print(f"[O_PROJ] active={active}, padded={padded}, native_ob={native_ob}, FP32 HC post", flush=True)
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
