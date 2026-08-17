# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared decode MXFP8 grouped output projection."""

import pypto.language as pl

from config import PRO_KERNEL as M, DECODE_BATCH, DECODE_SEQ


T = DECODE_BATCH * DECODE_SEQ
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = (H // O_GROUPS) * HEAD_DIM

MX_BLOCK_K = 32
MX_K_TILE = 64
MX_K_SCALE_TILE = MX_K_TILE // MX_BLOCK_K
MX_N_TILE = 256
MM_T_TILE = 16
T_PAD = ((T + MM_T_TILE - 1) // MM_T_TILE) * MM_T_TILE
A_K_TILES = O_GROUP_IN // MX_K_TILE
A_N_TILES = O_LORA // MX_N_TILE
B_GROUP_K_TILES = O_LORA // MX_K_TILE
B_K_TILES = (O_GROUPS * O_LORA) // MX_K_TILE
B_N_TILES = D // MX_N_TILE
WO_A_SCALE_ROWS_PER_GROUP = A_N_TILES * A_K_TILES * MX_K_SCALE_TILE
WO_A_SCALE_ROWS = O_GROUPS * WO_A_SCALE_ROWS_PER_GROUP
WO_B_SCALE_ROWS = B_N_TILES * B_K_TILES * MX_K_SCALE_TILE

assert O_GROUP_IN % MX_K_TILE == 0
assert O_LORA % MX_K_TILE == 0 and O_LORA % MX_N_TILE == 0
assert D % MX_N_TILE == 0


@pl.jit.inline
def decode_mxfp8_o_proj(
    o_packed: pl.Tensor[[O_GROUPS * T, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[WO_A_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[WO_B_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    attn_out: pl.Tensor[[T, D], pl.BF16],
):
    """Apply both grouped output-projection stages with GM-separated MX quantization."""
    wo_a_flat = pl.reshape(wo_a, [O_GROUPS * O_GROUP_IN, O_LORA])
    o_a_mx = pl.create_tensor([O_GROUPS * T_PAD, O_GROUP_IN], dtype=pl.FP8E4M3FN)
    o_a_scale_store = pl.create_tensor(
        [O_GROUPS * A_K_TILES, MM_T_TILE * MX_K_SCALE_TILE], dtype=pl.FP8E8M0
    )
    proj_a_partial = pl.create_tensor(
        [O_GROUPS * A_K_TILES * MM_T_TILE, O_LORA], dtype=pl.FP32
    )
    o_r_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_b_mx = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.FP8E4M3FN)
    o_b_scale_store = pl.create_tensor(
        [O_GROUPS * B_GROUP_K_TILES, MM_T_TILE * MX_K_SCALE_TILE], dtype=pl.FP8E8M0
    )
    proj_b_partial = pl.create_tensor(
        [O_GROUPS * B_GROUP_K_TILES * MM_T_TILE, D], dtype=pl.FP32
    )
    proj_b_group = pl.create_tensor([T_PAD, O_GROUPS * D], dtype=pl.FP32)

    for quant_idx in pl.parallel(O_GROUPS * A_K_TILES):
        g = quant_idx // A_K_TILES
        kb = quant_idx - g * A_K_TILES
        k0 = kb * MX_K_TILE
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_a_mx_quant"):
            a_src = pl.load(
                o_packed,
                [g * T, k0],
                [MM_T_TILE, MX_K_TILE],
                valid_shape=[T, MX_K_TILE],
                target_memory=pl.Mem.Vec,
            )
            a_src = pl.fillpad(a_src, pad_value=pl.PadValue.zero)
            a_q, a_scale = pl.quant_mx(a_src, layout=pl.MX_A_ZZ)
            pl.store(a_q, [g * T_PAD, k0], o_a_mx)
            a_scale_flat = pl.reshape(a_scale, [1, MM_T_TILE * MX_K_SCALE_TILE])
            pl.store(a_scale_flat, [quant_idx, 0], o_a_scale_store)

    for task_idx in pl.parallel(O_GROUPS * A_K_TILES * A_N_TILES):
        g = task_idx // (A_K_TILES * A_N_TILES)
        local_idx = task_idx % (A_K_TILES * A_N_TILES)
        kb = local_idx // A_N_TILES
        nb = local_idx - kb * A_N_TILES
        k0 = kb * MX_K_TILE
        n0 = nb * MX_N_TILE
        a_scale_idx = g * A_K_TILES + kb
        a_scale_slice = o_a_scale_store[a_scale_idx : a_scale_idx + 1, :]
        a_scale_mx = pl.tensor.view(
            a_scale_slice, [MM_T_TILE, MX_K_SCALE_TILE], layout=pl.MX_A_ZZ
        )
        wa_scale_offset = (
            g * WO_A_SCALE_ROWS_PER_GROUP
            + (nb * A_K_TILES + kb) * MX_K_SCALE_TILE
        )
        wa_scale_slice = wo_a_scale[
            wa_scale_offset : wa_scale_offset + MX_K_SCALE_TILE, :
        ]
        wa_scale_mx = pl.tensor.view(
            wa_scale_slice, [MX_K_SCALE_TILE, MX_N_TILE], layout=pl.MX_B_NN
        )
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_a_mx_mm"):
            a_k = pl.move(
                pl.load(o_a_mx, [g * T_PAD, k0], [MM_T_TILE, MX_K_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.Left,
            )
            a_scale_k = pl.move(
                pl.load(a_scale_mx, [0, 0], [MM_T_TILE, MX_K_SCALE_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.LeftScale,
            )
            wa_k = pl.move(
                pl.load(
                    wo_a_flat,
                    [g * O_GROUP_IN + k0, n0],
                    [MX_K_TILE, MX_N_TILE],
                    target_memory=pl.Mem.Mat,
                ),
                target_memory=pl.Mem.Right,
            )
            wa_scale_k = pl.move(
                pl.load(wa_scale_mx, [0, 0], [MX_K_SCALE_TILE, MX_N_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.RightScale,
            )
            a_partial = pl.matmul_mx(a_k, a_scale_k, wa_k, wa_scale_k)
            pl.store(
                a_partial,
                [(g * A_K_TILES + kb) * MM_T_TILE, n0],
                proj_a_partial,
            )

    for task_idx in pl.parallel(O_GROUPS * A_N_TILES):
        g = task_idx // A_N_TILES
        nb = task_idx - g * A_N_TILES
        n0 = nb * MX_N_TILE
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_a_mx_reduce"):
            a_sum = pl.tile.full([MM_T_TILE, MX_N_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(A_K_TILES, stage=2):
                a_part = pl.load(
                    proj_a_partial,
                    [(g * A_K_TILES + kb) * MM_T_TILE, n0],
                    [MM_T_TILE, MX_N_TILE],
                    target_memory=pl.Mem.Vec,
                )
                a_sum = pl.add(a_sum, a_part)
            pl.store(a_sum, [0, g * O_LORA + n0], o_r_pad)

    for quant_idx in pl.parallel(O_GROUPS * B_GROUP_K_TILES):
        g = quant_idx // B_GROUP_K_TILES
        kb = quant_idx - g * B_GROUP_K_TILES
        k0 = kb * MX_K_TILE
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_b_mx_quant"):
            b_src = pl.load(
                o_r_pad,
                [0, g * O_LORA + k0],
                [MM_T_TILE, MX_K_TILE],
                target_memory=pl.Mem.Vec,
            )
            b_q, b_scale = pl.quant_mx(b_src, layout=pl.MX_A_ZZ)
            pl.store(b_q, [0, g * O_LORA + k0], o_b_mx)
            b_scale_flat = pl.reshape(b_scale, [1, MM_T_TILE * MX_K_SCALE_TILE])
            pl.store(b_scale_flat, [quant_idx, 0], o_b_scale_store)

    for task_idx in pl.parallel(O_GROUPS * B_GROUP_K_TILES * B_N_TILES):
        g = task_idx // (B_GROUP_K_TILES * B_N_TILES)
        local_idx = task_idx % (B_GROUP_K_TILES * B_N_TILES)
        kb = local_idx // B_N_TILES
        nb = local_idx - kb * B_N_TILES
        global_kb = g * B_GROUP_K_TILES + kb
        n0 = nb * MX_N_TILE
        b_scale_slice = o_b_scale_store[global_kb : global_kb + 1, :]
        b_scale_mx = pl.tensor.view(
            b_scale_slice, [MM_T_TILE, MX_K_SCALE_TILE], layout=pl.MX_A_ZZ
        )
        wb_scale_offset = (nb * B_K_TILES + global_kb) * MX_K_SCALE_TILE
        wb_scale_slice = wo_b_scale[
            wb_scale_offset : wb_scale_offset + MX_K_SCALE_TILE, :
        ]
        wb_scale_mx = pl.tensor.view(
            wb_scale_slice, [MX_K_SCALE_TILE, MX_N_TILE], layout=pl.MX_B_NN
        )
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_b_mx_mm"):
            b_k = pl.move(
                pl.load(
                    o_b_mx,
                    [0, g * O_LORA + kb * MX_K_TILE],
                    [MM_T_TILE, MX_K_TILE],
                    target_memory=pl.Mem.Mat,
                ),
                target_memory=pl.Mem.Left,
            )
            b_scale_k = pl.move(
                pl.load(b_scale_mx, [0, 0], [MM_T_TILE, MX_K_SCALE_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.LeftScale,
            )
            wb_k = pl.move(
                pl.load(
                    wo_b,
                    [global_kb * MX_K_TILE, n0],
                    [MX_K_TILE, MX_N_TILE],
                    target_memory=pl.Mem.Mat,
                ),
                target_memory=pl.Mem.Right,
            )
            wb_scale_k = pl.move(
                pl.load(wb_scale_mx, [0, 0], [MX_K_SCALE_TILE, MX_N_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.RightScale,
            )
            b_partial = pl.matmul_mx(b_k, b_scale_k, wb_k, wb_scale_k)
            pl.store(
                b_partial,
                [global_kb * MM_T_TILE, n0],
                proj_b_partial,
            )

    for task_idx in pl.parallel(O_GROUPS * B_N_TILES):
        g = task_idx // B_N_TILES
        nb = task_idx - g * B_N_TILES
        n0 = nb * MX_N_TILE
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_b_mx_reduce"):
            b_sum = pl.tile.full([MM_T_TILE, MX_N_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(B_GROUP_K_TILES, stage=2):
                b_part = pl.load(
                    proj_b_partial,
                    [(g * B_GROUP_K_TILES + kb) * MM_T_TILE, n0],
                    [MM_T_TILE, MX_N_TILE],
                    target_memory=pl.Mem.Vec,
                )
                b_sum = pl.add(b_sum, b_part)
            pl.store(b_sum, [0, g * D + n0], proj_b_group)

    for nb in pl.parallel(B_N_TILES):
        n0 = nb * MX_N_TILE
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_b_group_reduce"):
            out_sum = pl.tile.full([MM_T_TILE, MX_N_TILE], dtype=pl.FP32, value=0.0)
            for g in pl.pipeline(O_GROUPS, stage=2):
                group_part = pl.load(
                    proj_b_group,
                    [0, g * D + n0],
                    [MM_T_TILE, MX_N_TILE],
                    target_memory=pl.Mem.Vec,
                )
                out_sum = pl.add(out_sum, group_part)
            out_valid = pl.set_validshape(out_sum, T, MX_N_TILE)
            out_bf16 = pl.cast(out_valid, target_type=pl.BF16, mode="rint")
            pl.store(out_bf16, [0, n0], attn_out)

    return attn_out
