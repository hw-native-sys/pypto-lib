# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared prefill MXFP8 grouped output projection."""

import pypto.language as pl

from config import PRO_KERNEL as M, PREFILL_BATCH, PREFILL_SEQ


T = PREFILL_BATCH * PREFILL_SEQ
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = (H // O_GROUPS) * HEAD_DIM

MX_BLOCK_K = 32
MX_M_TILE = 64
MX_K_TILE = 256
MX_K_SCALE_TILE = MX_K_TILE // MX_BLOCK_K
MX_N_TILE = 256
MX_M_TILES = T // MX_M_TILE
A_K_TILES = O_GROUP_IN // MX_K_TILE
A_N_TILES = O_LORA // MX_N_TILE
B_GROUP_K_TILES = O_LORA // MX_K_TILE
B_K_TILES = (O_GROUPS * O_LORA) // MX_K_TILE
B_N_TILES = D // MX_N_TILE
WO_A_SCALE_ROWS_PER_GROUP = A_N_TILES * A_K_TILES * MX_K_SCALE_TILE
WO_A_SCALE_ROWS = O_GROUPS * WO_A_SCALE_ROWS_PER_GROUP
WO_B_SCALE_ROWS = B_N_TILES * B_K_TILES * MX_K_SCALE_TILE

assert T % MX_M_TILE == 0
assert O_GROUP_IN % MX_K_TILE == 0
assert O_LORA % MX_K_TILE == 0 and O_LORA % MX_N_TILE == 0
assert D % MX_N_TILE == 0


@pl.jit.inline
def prefill_mxfp8_o_proj(
    o_packed: pl.Tensor[[O_GROUPS * T, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[WO_A_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[WO_B_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    attn_out: pl.Tensor[[T, D], pl.BF16],
):
    """Apply both prefill output-projection stages with GM-separated MX quantization."""
    wo_a_flat = pl.reshape(wo_a, [O_GROUPS * O_GROUP_IN, O_LORA])
    o_a_mx = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.FP8E4M3FN)
    o_a_scale_store = pl.create_tensor(
        [O_GROUPS * MX_M_TILES * A_K_TILES, MX_M_TILE * MX_K_SCALE_TILE],
        dtype=pl.FP8E8M0,
    )
    o_r = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.FP32, init_value=0.0)
    o_b_mx = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.FP8E4M3FN)
    o_b_scale_store = pl.create_tensor(
        [O_GROUPS * MX_M_TILES * B_GROUP_K_TILES, MX_M_TILE * MX_K_SCALE_TILE],
        dtype=pl.FP8E8M0,
    )
    proj_b_group = pl.create_tensor([T, O_GROUPS * D], dtype=pl.FP32, init_value=0.0)

    for quant_idx in pl.parallel(O_GROUPS * MX_M_TILES * A_K_TILES):
        g = quant_idx // (MX_M_TILES * A_K_TILES)
        local_idx = quant_idx % (MX_M_TILES * A_K_TILES)
        mt = local_idx // A_K_TILES
        kb = local_idx - mt * A_K_TILES
        t0 = mt * MX_M_TILE
        k0 = kb * MX_K_TILE
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_proj_a_mx_quant"):
            a_src = pl.load(
                o_packed,
                [g * T + t0, k0],
                [MX_M_TILE, MX_K_TILE],
                target_memory=pl.Mem.Vec,
            )
            a_q, a_scale = pl.quant_mx(a_src, layout=pl.MX_A_ZZ)
            pl.store(a_q, [g * T + t0, k0], o_a_mx)
            a_scale_flat = pl.reshape(a_scale, [1, MX_M_TILE * MX_K_SCALE_TILE])
            pl.store(a_scale_flat, [quant_idx, 0], o_a_scale_store)

    for task_idx in pl.parallel(O_GROUPS * MX_M_TILES * A_K_TILES * A_N_TILES):
        g = task_idx // (MX_M_TILES * A_K_TILES * A_N_TILES)
        local_idx = task_idx % (MX_M_TILES * A_K_TILES * A_N_TILES)
        mt = local_idx // (A_K_TILES * A_N_TILES)
        local_kn = local_idx % (A_K_TILES * A_N_TILES)
        kb = local_kn // A_N_TILES
        nb = local_kn - kb * A_N_TILES
        t0 = mt * MX_M_TILE
        k0 = kb * MX_K_TILE
        n0 = nb * MX_N_TILE
        a_scale_idx = (g * MX_M_TILES + mt) * A_K_TILES + kb
        a_scale_slice = o_a_scale_store[a_scale_idx : a_scale_idx + 1, :]
        a_scale_mx = pl.tensor.view(
            a_scale_slice, [MX_M_TILE, MX_K_SCALE_TILE], layout=pl.MX_A_ZZ
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
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_proj_a_mx_mm"):
            a_k = pl.move(
                pl.load(
                    o_a_mx,
                    [g * T + t0, k0],
                    [MX_M_TILE, MX_K_TILE],
                    target_memory=pl.Mem.Mat,
                ),
                target_memory=pl.Mem.Left,
            )
            a_scale_k = pl.move(
                pl.load(a_scale_mx, [0, 0], [MX_M_TILE, MX_K_SCALE_TILE], target_memory=pl.Mem.Mat),
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
                [t0, g * O_LORA + n0],
                o_r,
                atomic=pl.AtomicType.Add,
            )

    for quant_idx in pl.parallel(O_GROUPS * MX_M_TILES * B_GROUP_K_TILES):
        g = quant_idx // (MX_M_TILES * B_GROUP_K_TILES)
        local_idx = quant_idx % (MX_M_TILES * B_GROUP_K_TILES)
        mt = local_idx // B_GROUP_K_TILES
        kb = local_idx - mt * B_GROUP_K_TILES
        t0 = mt * MX_M_TILE
        k0 = kb * MX_K_TILE
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_proj_b_mx_quant"):
            b_src = pl.load(
                o_r,
                [t0, g * O_LORA + k0],
                [MX_M_TILE, MX_K_TILE],
                target_memory=pl.Mem.Vec,
            )
            b_q, b_scale = pl.quant_mx(b_src, layout=pl.MX_A_ZZ)
            pl.store(b_q, [t0, g * O_LORA + k0], o_b_mx)
            b_scale_flat = pl.reshape(b_scale, [1, MX_M_TILE * MX_K_SCALE_TILE])
            pl.store(b_scale_flat, [quant_idx, 0], o_b_scale_store)

    for task_idx in pl.parallel(O_GROUPS * MX_M_TILES * B_GROUP_K_TILES * B_N_TILES):
        g = task_idx // (MX_M_TILES * B_GROUP_K_TILES * B_N_TILES)
        local_idx = task_idx % (MX_M_TILES * B_GROUP_K_TILES * B_N_TILES)
        mt = local_idx // (B_GROUP_K_TILES * B_N_TILES)
        local_kn = local_idx % (B_GROUP_K_TILES * B_N_TILES)
        kb = local_kn // B_N_TILES
        nb = local_kn - kb * B_N_TILES
        global_kb = g * B_GROUP_K_TILES + kb
        t0 = mt * MX_M_TILE
        n0 = nb * MX_N_TILE
        b_scale_idx = (g * MX_M_TILES + mt) * B_GROUP_K_TILES + kb
        b_scale_slice = o_b_scale_store[b_scale_idx : b_scale_idx + 1, :]
        b_scale_mx = pl.tensor.view(
            b_scale_slice, [MX_M_TILE, MX_K_SCALE_TILE], layout=pl.MX_A_ZZ
        )
        wb_scale_offset = (nb * B_K_TILES + global_kb) * MX_K_SCALE_TILE
        wb_scale_slice = wo_b_scale[
            wb_scale_offset : wb_scale_offset + MX_K_SCALE_TILE, :
        ]
        wb_scale_mx = pl.tensor.view(
            wb_scale_slice, [MX_K_SCALE_TILE, MX_N_TILE], layout=pl.MX_B_NN
        )
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_proj_b_mx_mm"):
            b_k = pl.move(
                pl.load(
                    o_b_mx,
                    [t0, g * O_LORA + kb * MX_K_TILE],
                    [MX_M_TILE, MX_K_TILE],
                    target_memory=pl.Mem.Mat,
                ),
                target_memory=pl.Mem.Left,
            )
            b_scale_k = pl.move(
                pl.load(b_scale_mx, [0, 0], [MX_M_TILE, MX_K_SCALE_TILE], target_memory=pl.Mem.Mat),
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
                [t0, g * D + n0],
                proj_b_group,
                atomic=pl.AtomicType.Add,
            )

    for task_idx in pl.parallel(MX_M_TILES * B_N_TILES):
        mt = task_idx // B_N_TILES
        nb = task_idx - mt * B_N_TILES
        t0 = mt * MX_M_TILE
        n0 = nb * MX_N_TILE
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_proj_b_group_reduce"):
            out_sum = pl.tile.full([MX_M_TILE, MX_N_TILE], dtype=pl.FP32, value=0.0)
            for g in pl.pipeline(O_GROUPS, stage=2):
                group_part = pl.load(
                    proj_b_group,
                    [t0, g * D + n0],
                    [MX_M_TILE, MX_N_TILE],
                    target_memory=pl.Mem.Vec,
                )
                out_sum = pl.add(out_sum, group_part)
            out_bf16 = pl.cast(out_sum, target_type=pl.BF16, mode="rint")
            pl.store(out_bf16, [t0, n0], attn_out)

    return attn_out
