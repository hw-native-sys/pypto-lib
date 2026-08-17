# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Routed expert with MXFP8 activations and MXFP4 weights."""

import pypto.language as pl

from config import PRO_KERNEL as M, EP_WORLD_SIZE, RECV_MAX


D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit
N_LOCAL_EXPERTS = M.n_routed_experts // EP_WORLD_SIZE

RECV_TILE = 16
MX_BLOCK_K = 32
MX_K_TILE = 256
MX_K_SCALE_TILE = MX_K_TILE // MX_BLOCK_K
MX_N_TILE = 128
W_CAST_K_TILE = 128
ROUTE_N_TILE = 128

W13_K_TILES = D // MX_K_TILE
W13_N_TILES = MOE_INTER // MX_N_TILE
W2_K_TILES = MOE_INTER // MX_K_TILE
W2_N_TILES = D // MX_N_TILE
W13_CAST_K_TILES = D // W_CAST_K_TILE
W2_CAST_K_TILES = MOE_INTER // W_CAST_K_TILE
TILES_PER_EXPERT = RECV_MAX // RECV_TILE
W13_SCALE_ROWS_PER_EXPERT = W13_N_TILES * W13_K_TILES * MX_K_SCALE_TILE
W2_SCALE_ROWS_PER_EXPERT = W2_N_TILES * W2_K_TILES * MX_K_SCALE_TILE
W13_SCALE_ROWS = N_LOCAL_EXPERTS * W13_SCALE_ROWS_PER_EXPERT
W2_SCALE_ROWS = N_LOCAL_EXPERTS * W2_SCALE_ROWS_PER_EXPERT
W13_FP8_TILE_ROWS = (
    N_LOCAL_EXPERTS * W13_N_TILES * W13_K_TILES * MX_K_TILE
)
W2_FP8_TILE_ROWS = (
    N_LOCAL_EXPERTS * W2_N_TILES * W2_K_TILES * MX_K_TILE
)

assert RECV_MAX % RECV_TILE == 0
assert D % MX_K_TILE == 0 and D % MX_N_TILE == 0
assert MOE_INTER % MX_K_TILE == 0 and MOE_INTER % MX_N_TILE == 0
assert MX_K_TILE % W_CAST_K_TILE == 0


@pl.jit.inline
def routed_mxfp4_expert(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS * D, MOE_INTER], pl.FP4],
    routed_w1_scale: pl.Tensor[[W13_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS * D, MOE_INTER], pl.FP4],
    routed_w3_scale: pl.Tensor[[W13_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS * MOE_INTER, D], pl.FP4],
    routed_w2_scale: pl.Tensor[[W2_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    recv_y: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
):
    recv_x_flat = pl.reshape(recv_x, [N_LOCAL_EXPERTS * RECV_MAX, D])
    recv_scale_flat = pl.reshape(
        recv_scale_dq, [N_LOCAL_EXPERTS * RECV_MAX, 1]
    )
    recv_weights_flat = pl.reshape(
        recv_weights, [N_LOCAL_EXPERTS * RECV_MAX, 1]
    )
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL_EXPERTS * RECV_MAX, D])

    # PR1 materializes both dynamic MXFP8 activations and FP4-cast MXFP8
    # weights in GM. Each static-K AIC task atomically accumulates one partial.
    x_mx = pl.create_tensor(
        [N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.FP8E4M3FN
    )
    x_mx_scale = pl.create_tensor(
        [
            N_LOCAL_EXPERTS * TILES_PER_EXPERT * W13_K_TILES,
            RECV_TILE * MX_K_SCALE_TILE,
        ],
        dtype=pl.FP8E8M0,
    )
    routed_w1_fp8 = pl.create_tensor(
        [W13_FP8_TILE_ROWS, MX_N_TILE], dtype=pl.FP8E4M3FN
    )
    routed_w3_fp8 = pl.create_tensor(
        [W13_FP8_TILE_ROWS, MX_N_TILE], dtype=pl.FP8E4M3FN
    )
    gate_fp32 = pl.create_tensor(
        [N_LOCAL_EXPERTS * RECV_MAX, MOE_INTER], dtype=pl.FP32
    )
    up_fp32 = pl.create_tensor(
        [N_LOCAL_EXPERTS * RECV_MAX, MOE_INTER], dtype=pl.FP32
    )
    h_mx = pl.create_tensor(
        [N_LOCAL_EXPERTS * RECV_MAX, MOE_INTER], dtype=pl.FP8E4M3FN
    )
    h_mx_scale = pl.create_tensor(
        [
            N_LOCAL_EXPERTS * TILES_PER_EXPERT * W2_K_TILES,
            RECV_TILE * MX_K_SCALE_TILE,
        ],
        dtype=pl.FP8E8M0,
    )
    routed_w2_fp8 = pl.create_tensor(
        [W2_FP8_TILE_ROWS, MX_N_TILE], dtype=pl.FP8E4M3FN
    )
    y_fp32 = pl.create_tensor(
        [N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.FP32
    )

    for local_e in pl.parallel(N_LOCAL_EXPERTS):
        n_rows = pl.read(recv_expert_count, [local_e, 0])
        n_tiles = (n_rows + RECV_TILE - 1) // RECV_TILE
        for mt in pl.parallel(n_tiles):
            t0 = mt * RECV_TILE
            flat_t0 = local_e * RECV_MAX + t0
            valid_rows = pl.min(RECV_TILE, n_rows - t0)
            x_data = x_mx[flat_t0 : flat_t0 + RECV_TILE, :]
            x_scale_row = (
                local_e * TILES_PER_EXPERT + mt
            ) * W13_K_TILES
            x_scale_slice = x_mx_scale[
                x_scale_row : x_scale_row + W13_K_TILES, :
            ]

            x_tids = pl.array.create(W13_K_TILES, pl.TASK_ID)
            for kb in pl.parallel(W13_K_TILES):
                k0 = kb * MX_K_TILE
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="exp_x_mx_quant",
                ) as x_tid:
                    x_i8 = pl.load(
                        recv_x_flat,
                        [flat_t0, k0],
                        [RECV_TILE, MX_K_TILE],
                        target_memory=pl.Mem.Vec,
                    )
                    x_fp16 = pl.cast(x_i8, target_type=pl.FP16, mode="none")
                    x_fp32 = pl.cast(x_fp16, target_type=pl.FP32, mode="none")
                    x_dq = pl.load(
                        recv_scale_flat,
                        [flat_t0, 0],
                        [RECV_TILE, 1],
                        target_memory=pl.Mem.Vec,
                    )
                    x_dequant = pl.row_expand_mul(x_fp32, x_dq)
                    x_q, x_e8 = pl.quant_mx(x_dequant, layout=pl.MX_A_ZZ)
                    pl.store(x_q, [0, k0], x_data)
                    x_e8_flat = pl.reshape(
                        x_e8, [1, RECV_TILE * MX_K_SCALE_TILE]
                    )
                    pl.store(x_e8_flat, [kb, 0], x_scale_slice)
                x_tids[kb] = x_tid

            # Materialize one complete K stripe per AIV task. Keeping the N
            # loop inside the task avoids submitting thousands of tiny cast
            # tasks per active expert while preserving the GM hand-off to the
            # later matmul_mx tasks.
            w1_tids = pl.array.create(W13_CAST_K_TILES, pl.TASK_ID)
            w3_tids = pl.array.create(W13_CAST_K_TILES, pl.TASK_ID)
            for cast_kb in pl.parallel(W13_CAST_K_TILES):
                cast_k0 = cast_kb * W_CAST_K_TILE
                w1_rows = pl.slice(
                    routed_w1,
                    [W_CAST_K_TILE, MOE_INTER],
                    [local_e * D + cast_k0, 0],
                )
                w3_rows = pl.slice(
                    routed_w3,
                    [W_CAST_K_TILE, MOE_INTER],
                    [local_e * D + cast_k0, 0],
                )
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="exp_w1_fp4_to_fp8",
                ) as w1_tid:
                    for nb in pl.range(W13_N_TILES):
                        cast_k_tile = cast_kb // 2
                        cast_k_half = cast_kb - cast_k_tile * 2
                        w13_tile_row = (
                            (
                                (local_e * W13_N_TILES + nb)
                                * W13_K_TILES
                                + cast_k_tile
                            )
                            * MX_K_TILE
                            + cast_k_half * W_CAST_K_TILE
                        )
                        w1_fp4 = pl.load(
                            w1_rows,
                            # PTOAS 0.58 lowers a dynamic FP4 partition offset
                            # in x2-carrier units inside an AIV loop.
                            [0, nb * (MX_N_TILE // 2)],
                            [W_CAST_K_TILE, MX_N_TILE],
                            target_memory=pl.Mem.Vec,
                        )
                        w1_fp8 = pl.cast(
                            w1_fp4, target_type=pl.FP8E4M3FN
                        )
                        pl.store(
                            w1_fp8,
                            [w13_tile_row, 0],
                            routed_w1_fp8,
                        )
                w1_tids[cast_kb] = w1_tid

                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="exp_w3_fp4_to_fp8",
                ) as w3_tid:
                    for nb in pl.range(W13_N_TILES):
                        cast_k_tile = cast_kb // 2
                        cast_k_half = cast_kb - cast_k_tile * 2
                        w13_tile_row = (
                            (
                                (local_e * W13_N_TILES + nb)
                                * W13_K_TILES
                                + cast_k_tile
                            )
                            * MX_K_TILE
                            + cast_k_half * W_CAST_K_TILE
                        )
                        w3_fp4 = pl.load(
                            w3_rows,
                            [0, nb * (MX_N_TILE // 2)],
                            [W_CAST_K_TILE, MX_N_TILE],
                            target_memory=pl.Mem.Vec,
                        )
                        w3_fp8 = pl.cast(
                            w3_fp4, target_type=pl.FP8E4M3FN
                        )
                        pl.store(
                            w3_fp8,
                            [w13_tile_row, 0],
                            routed_w3_fp8,
                        )
                w3_tids[cast_kb] = w3_tid

            gate_ready_tids = pl.array.create(W13_N_TILES, pl.TASK_ID)
            up_ready_tids = pl.array.create(W13_N_TILES, pl.TASK_ID)
            for nb in pl.parallel(W13_N_TILES):
                n0 = nb * MX_N_TILE
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="exp_gate_up_zero",
                ) as zero_tid:
                    zeros = pl.tile.full(
                        [RECV_TILE, MX_N_TILE],
                        dtype=pl.FP32,
                        value=0.0,
                    )
                    pl.store(zeros, [flat_t0, n0], gate_fp32)
                    pl.store(zeros, [flat_t0, n0], up_fp32)

                w_scale_row = (
                    local_e * W13_N_TILES + nb
                ) * W13_K_TILES * MX_K_SCALE_TILE
                w1_scale_slice = routed_w1_scale[
                    w_scale_row :
                    w_scale_row + W13_K_TILES * MX_K_SCALE_TILE,
                    :,
                ]
                gate_partial_tids = pl.array.create(W13_K_TILES, pl.TASK_ID)
                for kb in pl.parallel(W13_K_TILES):
                    k0 = kb * MX_K_TILE
                    x_scale_store_k = pl.slice(
                        x_scale_slice,
                        [1, RECV_TILE * MX_K_SCALE_TILE],
                        [kb, 0],
                    )
                    x_scale_mx_k = pl.tensor.view(
                        x_scale_store_k,
                        [RECV_TILE, MX_K_SCALE_TILE],
                        layout=pl.MX_A_ZZ,
                    )
                    w1_scale_store_k = pl.slice(
                        w1_scale_slice,
                        [MX_K_SCALE_TILE, MX_N_TILE],
                        [kb * MX_K_SCALE_TILE, 0],
                    )
                    w1_scale_mx_k = pl.tensor.view(
                        w1_scale_store_k,
                        [MX_K_SCALE_TILE, MX_N_TILE],
                        layout=pl.MX_B_NN,
                    )
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="exp_gate_mx_mm",
                        deps=[
                            zero_tid,
                            x_tids[kb],
                            w1_tids[kb * 2],
                            w1_tids[kb * 2 + 1],
                        ],
                    ) as gate_partial_tid:
                        w13_tile_row = (
                            (
                                (local_e * W13_N_TILES + nb)
                                * W13_K_TILES
                                + kb
                            )
                            * MX_K_TILE
                        )
                        x_k = pl.move(
                            pl.load(
                                x_data,
                                [0, k0],
                                [RECV_TILE, MX_K_TILE],
                                target_memory=pl.Mem.Mat,
                            ),
                            target_memory=pl.Mem.Left,
                        )
                        x_scale_k = pl.move(
                            pl.load(
                                x_scale_mx_k,
                                [0, 0],
                                [RECV_TILE, MX_K_SCALE_TILE],
                                target_memory=pl.Mem.Mat,
                            ),
                            target_memory=pl.Mem.LeftScale,
                        )
                        w1_k = pl.move(
                            pl.load(
                                routed_w1_fp8,
                                [w13_tile_row, 0],
                                [MX_K_TILE, MX_N_TILE],
                                target_memory=pl.Mem.Mat,
                            ),
                            target_memory=pl.Mem.Right,
                        )
                        w1_scale_k = pl.move(
                            pl.load(
                                w1_scale_mx_k,
                                [0, 0],
                                [MX_K_SCALE_TILE, MX_N_TILE],
                                target_memory=pl.Mem.Mat,
                            ),
                            target_memory=pl.Mem.RightScale,
                        )
                        gate_acc = pl.matmul_mx(
                            x_k, x_scale_k, w1_k, w1_scale_k
                        )
                        pl.store(
                            gate_acc,
                            [flat_t0, n0],
                            gate_fp32,
                            atomic=pl.AtomicType.Add,
                        )
                    gate_partial_tids[kb] = gate_partial_tid
                gate_ready_tids[nb] = pl.system.task_dummy(
                    deps=[gate_partial_tids]
                )

                w3_scale_slice = routed_w3_scale[
                    w_scale_row :
                    w_scale_row + W13_K_TILES * MX_K_SCALE_TILE,
                    :,
                ]
                up_partial_tids = pl.array.create(W13_K_TILES, pl.TASK_ID)
                for kb in pl.parallel(W13_K_TILES):
                    k0 = kb * MX_K_TILE
                    x_scale_store_k = pl.slice(
                        x_scale_slice,
                        [1, RECV_TILE * MX_K_SCALE_TILE],
                        [kb, 0],
                    )
                    x_scale_mx_k = pl.tensor.view(
                        x_scale_store_k,
                        [RECV_TILE, MX_K_SCALE_TILE],
                        layout=pl.MX_A_ZZ,
                    )
                    w3_scale_store_k = pl.slice(
                        w3_scale_slice,
                        [MX_K_SCALE_TILE, MX_N_TILE],
                        [kb * MX_K_SCALE_TILE, 0],
                    )
                    w3_scale_mx_k = pl.tensor.view(
                        w3_scale_store_k,
                        [MX_K_SCALE_TILE, MX_N_TILE],
                        layout=pl.MX_B_NN,
                    )
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="exp_up_mx_mm",
                        deps=[
                            zero_tid,
                            x_tids[kb],
                            w3_tids[kb * 2],
                            w3_tids[kb * 2 + 1],
                        ],
                    ) as up_partial_tid:
                        w13_tile_row = (
                            (
                                (local_e * W13_N_TILES + nb)
                                * W13_K_TILES
                                + kb
                            )
                            * MX_K_TILE
                        )
                        x_k = pl.move(
                            pl.load(
                                x_data,
                                [0, k0],
                                [RECV_TILE, MX_K_TILE],
                                target_memory=pl.Mem.Mat,
                            ),
                            target_memory=pl.Mem.Left,
                        )
                        x_scale_k = pl.move(
                            pl.load(
                                x_scale_mx_k,
                                [0, 0],
                                [RECV_TILE, MX_K_SCALE_TILE],
                                target_memory=pl.Mem.Mat,
                            ),
                            target_memory=pl.Mem.LeftScale,
                        )
                        w3_k = pl.move(
                            pl.load(
                                routed_w3_fp8,
                                [w13_tile_row, 0],
                                [MX_K_TILE, MX_N_TILE],
                                target_memory=pl.Mem.Mat,
                            ),
                            target_memory=pl.Mem.Right,
                        )
                        w3_scale_k = pl.move(
                            pl.load(
                                w3_scale_mx_k,
                                [0, 0],
                                [MX_K_SCALE_TILE, MX_N_TILE],
                                target_memory=pl.Mem.Mat,
                            ),
                            target_memory=pl.Mem.RightScale,
                        )
                        up_acc = pl.matmul_mx(
                            x_k, x_scale_k, w3_k, w3_scale_k
                        )
                        pl.store(
                            up_acc,
                            [flat_t0, n0],
                            up_fp32,
                            atomic=pl.AtomicType.Add,
                        )
                    up_partial_tids[kb] = up_partial_tid
                up_ready_tids[nb] = pl.system.task_dummy(
                    deps=[up_partial_tids]
                )

            h_data = h_mx[flat_t0 : flat_t0 + RECV_TILE, :]
            h_scale_row = (
                local_e * TILES_PER_EXPERT + mt
            ) * W2_K_TILES
            h_scale_slice = h_mx_scale[
                h_scale_row : h_scale_row + W2_K_TILES, :
            ]
            h_tids = pl.array.create(W2_K_TILES, pl.TASK_ID)
            for kb in pl.parallel(W2_K_TILES):
                k0 = kb * MX_K_TILE
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="exp_gate_up_mx_quant",
                    deps=[
                        gate_ready_tids[kb * 2],
                        gate_ready_tids[kb * 2 + 1],
                        up_ready_tids[kb * 2],
                        up_ready_tids[kb * 2 + 1],
                    ],
                ) as h_tid:
                    gate = pl.load(
                        gate_fp32,
                        [flat_t0, k0],
                        [RECV_TILE, MX_K_TILE],
                        target_memory=pl.Mem.Vec,
                    )
                    up = pl.load(
                        up_fp32,
                        [flat_t0, k0],
                        [RECV_TILE, MX_K_TILE],
                        target_memory=pl.Mem.Vec,
                    )
                    if SWIGLU_LIMIT > 0.0:
                        gate = pl.minimum(gate, SWIGLU_LIMIT)
                        up = pl.maximum(
                            pl.minimum(up, SWIGLU_LIMIT), -SWIGLU_LIMIT
                        )
                    sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate)), 1.0))
                    h = pl.mul(pl.mul(gate, sigmoid), up)
                    h_valid = pl.set_validshape(h, valid_rows, MX_K_TILE)
                    h_padded = pl.fillpad(h_valid, pad_value=pl.PadValue.zero)
                    h_q, h_e8 = pl.quant_mx(h_padded, layout=pl.MX_A_ZZ)
                    pl.store(h_q, [0, k0], h_data)
                    h_e8_flat = pl.reshape(
                        h_e8, [1, RECV_TILE * MX_K_SCALE_TILE]
                    )
                    pl.store(h_e8_flat, [kb, 0], h_scale_slice)
                h_tids[kb] = h_tid

            w2_tids = pl.array.create(W2_CAST_K_TILES, pl.TASK_ID)
            for cast_kb in pl.parallel(W2_CAST_K_TILES):
                cast_k0 = cast_kb * W_CAST_K_TILE
                w2_rows = pl.slice(
                    routed_w2,
                    [W_CAST_K_TILE, D],
                    [local_e * MOE_INTER + cast_k0, 0],
                )
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="exp_w2_fp4_to_fp8",
                ) as w2_tid:
                    for nb in pl.range(W2_N_TILES):
                        w2_fp4 = pl.load(
                            w2_rows,
                            [0, nb * (MX_N_TILE // 2)],
                            [W_CAST_K_TILE, MX_N_TILE],
                            target_memory=pl.Mem.Vec,
                        )
                        w2_fp8 = pl.cast(
                            w2_fp4, target_type=pl.FP8E4M3FN
                        )
                        cast_k_tile = cast_kb // 2
                        cast_k_half = cast_kb - cast_k_tile * 2
                        w2_tile_row = (
                            (
                                (local_e * W2_N_TILES + nb)
                                * W2_K_TILES
                                + cast_k_tile
                            )
                            * MX_K_TILE
                            + cast_k_half * W_CAST_K_TILE
                        )
                        pl.store(
                            w2_fp8,
                            [w2_tile_row, 0],
                            routed_w2_fp8,
                        )
                w2_tids[cast_kb] = w2_tid

            for nb in pl.parallel(W2_N_TILES):
                n0 = nb * MX_N_TILE
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="exp_w2_zero",
                ) as y_zero_tid:
                    zeros = pl.tile.full(
                        [RECV_TILE, MX_N_TILE],
                        dtype=pl.FP32,
                        value=0.0,
                    )
                    pl.store(zeros, [flat_t0, n0], y_fp32)

                w_scale_row = (
                    local_e * W2_N_TILES + nb
                ) * W2_K_TILES * MX_K_SCALE_TILE
                w2_scale_slice = routed_w2_scale[
                    w_scale_row :
                    w_scale_row + W2_K_TILES * MX_K_SCALE_TILE,
                    :,
                ]
                y_partial_tids = pl.array.create(W2_K_TILES, pl.TASK_ID)
                for kb in pl.parallel(W2_K_TILES):
                    k0 = kb * MX_K_TILE
                    h_scale_store_k = pl.slice(
                        h_scale_slice,
                        [1, RECV_TILE * MX_K_SCALE_TILE],
                        [kb, 0],
                    )
                    h_scale_mx_k = pl.tensor.view(
                        h_scale_store_k,
                        [RECV_TILE, MX_K_SCALE_TILE],
                        layout=pl.MX_A_ZZ,
                    )
                    w2_scale_store_k = pl.slice(
                        w2_scale_slice,
                        [MX_K_SCALE_TILE, MX_N_TILE],
                        [kb * MX_K_SCALE_TILE, 0],
                    )
                    w2_scale_mx_k = pl.tensor.view(
                        w2_scale_store_k,
                        [MX_K_SCALE_TILE, MX_N_TILE],
                        layout=pl.MX_B_NN,
                    )
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="exp_w2_mx_mm",
                        deps=[
                            y_zero_tid,
                            h_tids[kb],
                            w2_tids[kb * 2],
                            w2_tids[kb * 2 + 1],
                        ],
                    ) as y_partial_tid:
                        w2_tile_row = (
                            (
                                (local_e * W2_N_TILES + nb)
                                * W2_K_TILES
                                + kb
                            )
                            * MX_K_TILE
                        )
                        h_k = pl.move(
                            pl.load(
                                h_data,
                                [0, k0],
                                [RECV_TILE, MX_K_TILE],
                                target_memory=pl.Mem.Mat,
                            ),
                            target_memory=pl.Mem.Left,
                        )
                        h_scale_k = pl.move(
                            pl.load(
                                h_scale_mx_k,
                                [0, 0],
                                [RECV_TILE, MX_K_SCALE_TILE],
                                target_memory=pl.Mem.Mat,
                            ),
                            target_memory=pl.Mem.LeftScale,
                        )
                        w2_k = pl.move(
                            pl.load(
                                routed_w2_fp8,
                                [w2_tile_row, 0],
                                [MX_K_TILE, MX_N_TILE],
                                target_memory=pl.Mem.Mat,
                            ),
                            target_memory=pl.Mem.Right,
                        )
                        w2_scale_k = pl.move(
                            pl.load(
                                w2_scale_mx_k,
                                [0, 0],
                                [MX_K_SCALE_TILE, MX_N_TILE],
                                target_memory=pl.Mem.Mat,
                            ),
                            target_memory=pl.Mem.RightScale,
                        )
                        y_acc = pl.matmul_mx(
                            h_k, h_scale_k, w2_k, w2_scale_k
                        )
                        pl.store(
                            y_acc,
                            [flat_t0, n0],
                            y_fp32,
                            atomic=pl.AtomicType.Add,
                        )
                    y_partial_tids[kb] = y_partial_tid
                y_ready_tid = pl.system.task_dummy(deps=[y_partial_tids])

                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="exp_w2_route",
                    deps=[y_ready_tid],
                ):
                    y = pl.load(
                        y_fp32,
                        [flat_t0, n0],
                        [RECV_TILE, ROUTE_N_TILE],
                        target_memory=pl.Mem.Vec,
                    )
                    route_weight = pl.load(
                        recv_weights_flat,
                        [flat_t0, 0],
                        [RECV_TILE, 1],
                        target_memory=pl.Mem.Vec,
                    )
                    y = pl.row_expand_mul(y, route_weight)
                    y_valid = pl.set_validshape(y, valid_rows, ROUTE_N_TILE)
                    y_bf16 = pl.cast(y_valid, target_type=pl.BF16, mode="rint")
                    pl.store(y_bf16, [flat_t0, n0], recv_y_flat)

    return recv_y
