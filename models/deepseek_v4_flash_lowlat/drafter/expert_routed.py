# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl

from .config import FLASH as M, INT8_SCALE_MAX, INT8_AMAX_EPS, EP, RECV_MAX

D = M.hidden_size

MOE_INTER = M.moe_intermediate_size

SWIGLU_LIMIT = M.swiglu_limit

N_LOCAL_EXPERTS = M.n_routed_experts // EP

RECV_TILE = 64

K_TILE = 512

INTER_K = 512

MM_INTER_TILE = 256

MM_GATE_INNER = 4

ACT_INTER_TILE = 64

ACT_GATE_INNER = 4

D_OUT_TILE = 256

QUANT_TILE = 512

QUANT_ROW_TILE = 16

QUANT_SCALE_PAD = 8

D_OUT_TILE_ACT = 256

W2_INNER = 1

W2_ACT_INNER = 8

TILES_PER_EXPERT = RECV_MAX // RECV_TILE

assert RECV_MAX % RECV_TILE == 0, "RECV_MAX must be a whole number of RECV_TILE row-tiles"


@pl.jit.inline(auto_scope=False)
def expert_routed_tile(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8, pl.NZ],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8, pl.NZ],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8, pl.NZ],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    recv_y_tile: pl.Tensor[[RECV_TILE, D], pl.BF16],
    local_e: pl.Scalar[pl.INDEX],
    tile_row: pl.Scalar[pl.INDEX],
    valid_rows: pl.Scalar[pl.INDEX],
    inputs_ready: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    recv_x_flat = pl.reshape(recv_x, [N_LOCAL_EXPERTS * RECV_MAX, D])
    flat_tile_row = local_e * RECV_MAX + tile_row
    h_tile_i8 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT8)
    h_tile_scale_dq = pl.create_tensor([RECV_TILE, QUANT_SCALE_PAD], dtype=pl.FP32, manual_dep=True)
    quant_tids = pl.array.create(1, pl.TASK_ID)

    with pl.scope():
        gate_tile_i32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT32)
        up_tile_i32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT32)

        with pl.spmd(
            MOE_INTER // (MM_GATE_INNER * MM_INTER_TILE), name_hint="exp_gate_mm", deps=[inputs_ready]
        ):
            # Weight reads bypass L2.
            pl.set_cache_policy(routed_w1, pl.CachePolicy.BYPASS)
            block = pl.tile.get_block_idx()
            n_base = block * (MM_GATE_INNER * MM_INTER_TILE)
            for inner in pl.range(MM_GATE_INNER):
                n0 = n_base + inner * MM_INTER_TILE
                gate_acc = pl.create_tensor([1, RECV_TILE, MM_INTER_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                    x_chunk = recv_x_flat[flat_tile_row : flat_tile_row + RECV_TILE, k0 : k0 + K_TILE]
                    w1_chunk = routed_w1[local_e : local_e + 1, n0 : n0 + MM_INTER_TILE, k0 : k0 + K_TILE]
                    gate_acc = pl.matmul_acc(gate_acc, x_chunk, w1_chunk, b_trans=True, init_cond=(k0 == 0))
                gate_tile_i32[:, n0 : n0 + MM_INTER_TILE] = pl.reshape(gate_acc, [RECV_TILE, MM_INTER_TILE])

        with pl.spmd(
            MOE_INTER // (MM_GATE_INNER * MM_INTER_TILE), name_hint="exp_up_mm", deps=[inputs_ready]
        ):
            # Weight reads bypass L2.
            pl.set_cache_policy(routed_w3, pl.CachePolicy.BYPASS)
            block = pl.tile.get_block_idx()
            n_base = block * (MM_GATE_INNER * MM_INTER_TILE)
            for inner in pl.range(MM_GATE_INNER):
                n0 = n_base + inner * MM_INTER_TILE
                up_acc = pl.create_tensor([1, RECV_TILE, MM_INTER_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                    x_chunk = recv_x_flat[flat_tile_row : flat_tile_row + RECV_TILE, k0 : k0 + K_TILE]
                    w3_chunk = routed_w3[local_e : local_e + 1, n0 : n0 + MM_INTER_TILE, k0 : k0 + K_TILE]
                    up_acc = pl.matmul_acc(up_acc, x_chunk, w3_chunk, b_trans=True, init_cond=(k0 == 0))
                up_tile_i32[:, n0 : n0 + MM_INTER_TILE] = pl.reshape(up_acc, [RECV_TILE, MM_INTER_TILE])

        h_tile_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32)
        with pl.spmd(MOE_INTER // (ACT_GATE_INNER * ACT_INTER_TILE), name_hint="exp_gate_up_act"):
            block = pl.tile.get_block_idx()
            inter_base = block * (ACT_GATE_INNER * ACT_INTER_TILE)
            for inner in pl.pipeline(ACT_GATE_INNER, stage=2):
                inter0 = inter_base + inner * ACT_INTER_TILE
                gate_i32 = gate_tile_i32[:, inter0 : inter0 + ACT_INTER_TILE]
                up_i32 = up_tile_i32[:, inter0 : inter0 + ACT_INTER_TILE]
                x_scale = pl.reshape(
                    recv_scale_dq[
                        local_e : local_e + 1,
                        tile_row : tile_row + RECV_TILE,
                    ],
                    [RECV_TILE, 1],
                )
                gate_fp32 = pl.col_expand_mul(
                    pl.row_expand_mul(
                        pl.cast(gate_i32, target_type=pl.FP32, mode="none"),
                        x_scale,
                    ),
                    routed_w1_scale[
                        local_e : local_e + 1,
                        inter0 : inter0 + ACT_INTER_TILE,
                    ],
                )
                up_fp32 = pl.col_expand_mul(
                    pl.row_expand_mul(
                        pl.cast(up_i32, target_type=pl.FP32, mode="none"),
                        x_scale,
                    ),
                    routed_w3_scale[
                        local_e : local_e + 1,
                        inter0 : inter0 + ACT_INTER_TILE,
                    ],
                )
                if SWIGLU_LIMIT > 0.0:
                    gate_fp32 = pl.minimum(gate_fp32, SWIGLU_LIMIT)
                    up_fp32 = pl.maximum(pl.minimum(up_fp32, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_fp32)), 1.0))
                activated = pl.mul(pl.mul(gate_fp32, sigmoid), up_fp32)
                activated = pl.set_validshape(activated, valid_rows, ACT_INTER_TILE)
                h_tile_fp32[:, inter0 : inter0 + ACT_INTER_TILE] = pl.fillpad(
                    activated, pad_value=pl.PadValue.zero
                )

        with pl.spmd(RECV_TILE // QUANT_ROW_TILE, name_hint="exp_h_q") as quant_tid:
            quant_block = pl.tile.get_block_idx()
            quant_row = quant_block * QUANT_ROW_TILE
            row_amax = pl.full([1, QUANT_ROW_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for k0 in pl.pipeline(0, MOE_INTER, QUANT_TILE, stage=2):
                h_amax_chunk = h_tile_fp32[quant_row : quant_row + QUANT_ROW_TILE, k0 : k0 + QUANT_TILE]
                h_abs = pl.maximum(h_amax_chunk, pl.neg(h_amax_chunk))
                row_amax = pl.maximum(row_amax, pl.reshape(pl.row_max(h_abs), [1, QUANT_ROW_TILE]))
            quant_scale = pl.div(pl.full([1, QUANT_ROW_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), row_amax)
            dequant_scale_col = pl.reshape(pl.recip(quant_scale), [QUANT_ROW_TILE, 1])
            dequant_scale_zeros = pl.full([QUANT_ROW_TILE, QUANT_SCALE_PAD], dtype=pl.FP32, value=0.0)
            h_tile_scale_dq[
                quant_row : quant_row + QUANT_ROW_TILE,
                :,
            ] = pl.row_expand(dequant_scale_zeros, dequant_scale_col)
            quant_scale_col = pl.reshape(quant_scale, [QUANT_ROW_TILE, 1])
            for k0 in pl.pipeline(0, MOE_INTER, QUANT_TILE, stage=2):
                h_quant_chunk = h_tile_fp32[quant_row : quant_row + QUANT_ROW_TILE, k0 : k0 + QUANT_TILE]
                h_scaled = pl.row_expand_mul(h_quant_chunk, quant_scale_col)
                h_i32 = pl.cast(h_scaled, target_type=pl.INT32, mode="rint")
                h_fp16 = pl.cast(h_i32, target_type=pl.FP16, mode="round")
                h_tile_i8[
                    quant_row : quant_row + QUANT_ROW_TILE,
                    k0 : k0 + QUANT_TILE,
                ] = pl.cast(
                    h_fp16,
                    target_type=pl.INT8,
                    mode="trunc",
                )
        quant_tids[0] = quant_tid

    y_i32 = pl.create_tensor([RECV_TILE, D], dtype=pl.INT32)
    with pl.spmd(
        D // (W2_INNER * D_OUT_TILE),
        name_hint="exp_w2_mm",
        deps=[quant_tids[0]],
        allow_early_resolve=True,
    ) as w2_tid:
        # Weight reads bypass L2.
        pl.set_cache_policy(routed_w2, pl.CachePolicy.BYPASS)
        block = pl.tile.get_block_idx()
        d_base = block * (W2_INNER * D_OUT_TILE)
        for inner in pl.range(W2_INNER):
            d0 = d_base + inner * D_OUT_TILE
            y_acc = pl.create_tensor([1, RECV_TILE, D_OUT_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, MOE_INTER, INTER_K, stage=2):
                h_w2_chunk = h_tile_i8[:, k0 : k0 + INTER_K]
                w2_chunk = routed_w2[local_e : local_e + 1, d0 : d0 + D_OUT_TILE, k0 : k0 + INTER_K]
                y_acc = pl.matmul_acc(y_acc, h_w2_chunk, w2_chunk, b_trans=True, init_cond=(k0 == 0))
            y_i32[:, d0 : d0 + D_OUT_TILE] = pl.reshape(y_acc, [RECV_TILE, D_OUT_TILE])

    with pl.spmd(
        D // (W2_ACT_INNER * D_OUT_TILE_ACT),
        name_hint="exp_w2_act",
        deps=[w2_tid],
        allow_early_resolve=True,
    ) as w2_act_tid:
        block = pl.tile.get_block_idx()
        d_base = block * (W2_ACT_INNER * D_OUT_TILE_ACT)
        route_weight = pl.reshape(
            recv_weights[local_e : local_e + 1, tile_row : tile_row + RECV_TILE], [RECV_TILE, 1]
        )
        row_scale = pl.mul(pl.row_max(h_tile_scale_dq), route_weight)
        for inner in pl.pipeline(W2_ACT_INNER, stage=2):
            d0 = d_base + inner * D_OUT_TILE_ACT
            y_fp32 = pl.cast(y_i32[:, d0 : d0 + D_OUT_TILE_ACT], target_type=pl.FP32, mode="none")
            y_fp32 = pl.col_expand_mul(
                pl.row_expand_mul(y_fp32, row_scale),
                routed_w2_scale[
                    local_e : local_e + 1,
                    d0 : d0 + D_OUT_TILE_ACT,
                ],
            )
            recv_y_tile[:, d0 : d0 + D_OUT_TILE_ACT] = pl.cast(y_fp32, target_type=pl.BF16, mode="rint")
    return w2_act_tid
