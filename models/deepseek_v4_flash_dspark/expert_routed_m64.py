# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE routed expert: gate/up matmul, SwiGLU, requant, W2, output scaling."""


import pypto.language as pl

from config import (FLASH as M, DECODE_BATCH, DECODE_SEQ, INT8_SCALE_MAX, INT8_AMAX_EPS,
                    EP, RECV_MAX, MOE_TOKENS)


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit

# EP layout / recv buffers (single-card view: kernel only sees the local shard)
N_LOCAL_EXPERTS = M.n_routed_experts // EP

# tiling
RECV_TILE = 16 if MOE_TOKENS <= 8 else 64
W2_EXPERT_ROWS = (MOE_TOKENS * M.num_experts_per_tok + N_LOCAL_EXPERTS - 1) // N_LOCAL_EXPERTS
K_TILE = 512
INTER_K = 256 if W2_EXPERT_ROWS >= 2 * RECV_TILE else 512
MM_INTER_TILE = 256
MM_GATE_INNER = 2
ACT_INTER_TILE = MOE_INTER // 2
D_OUT_TILE = 512 if W2_EXPERT_ROWS >= 2 * RECV_TILE else 256
QUANT_ROW_TILE = 8
D_OUT_TILE_ACT = 512 if MOE_TOKENS <= 8 else 256
W2_INNER = 2
W2_ACT_INNER = 8
ZERO_D_TILE = 1024
ZERO_WORKERS = 40
TILES_PER_EXPERT = RECV_MAX // RECV_TILE

assert RECV_MAX % RECV_TILE == 0, "RECV_MAX must be a whole number of RECV_TILE row-tiles"


@pl.jit.extern(core_type="aic", source="kernels/expert/gate_up.cpp")
def _gate_up_native_m64(
    x: pl.Tensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.INT8],
    gate: pl.Out[pl.Tensor[[RECV_TILE, MOE_INTER], pl.INT32]],
    up: pl.Out[pl.Tensor[[RECV_TILE, MOE_INTER], pl.INT32]],
    w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8, pl.NZ],
    w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8, pl.NZ],
    row: pl.Scalar[pl.INDEX],
    expert: pl.Scalar[pl.INDEX],
) -> tuple[pl.Tensor[[RECV_TILE, MOE_INTER], pl.INT32], pl.Tensor[[RECV_TILE, MOE_INTER], pl.INT32]]:
    return gate, up


@pl.jit.inline(auto_scope=False)
def expert_routed_tile_m64(
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
    h_tile_row_scale = pl.create_tensor([1, RECV_TILE], dtype=pl.FP32, manual_dep=True)
    quant_tids = pl.array.create(1, pl.TASK_ID)

    with pl.scope():
        gate_tile_i32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT32)
        up_tile_i32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT32)

        with pl.spmd(MOE_INTER // (MM_GATE_INNER * MM_INTER_TILE), name_hint="exp_gate_up_mm", deps=[inputs_ready]):
            gate_tile_i32, up_tile_i32 = _gate_up_native_m64(
                recv_x_flat, gate_tile_i32, up_tile_i32,
                routed_w1, routed_w3, flat_tile_row, local_e,
            )

        with pl.spmd(RECV_TILE // QUANT_ROW_TILE, name_hint="exp_gate_up_act_q") as quant_tid:
            quant_block = pl.tile.get_block_idx()
            quant_row = quant_block * QUANT_ROW_TILE
            recv_row = tile_row + quant_row
            valid_quant_rows = pl.min(QUANT_ROW_TILE, pl.max(0, valid_rows - quant_row))
            x_scale = recv_scale_dq[local_e : local_e + 1, recv_row : recv_row + QUANT_ROW_TILE]
            x_scale = pl.reshape(x_scale, [QUANT_ROW_TILE, 1])
            row_amax = pl.full([1, QUANT_ROW_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            gate0_i32 = gate_tile_i32[quant_row : quant_row + QUANT_ROW_TILE, :ACT_INTER_TILE]
            gate0_fp32 = pl.cast(gate0_i32, target_type=pl.FP32, mode="none")
            gate0_scaled = pl.row_expand_mul(gate0_fp32, x_scale)
            w1_scale0 = routed_w1_scale[local_e : local_e + 1, :ACT_INTER_TILE]
            gate0_fp32 = pl.col_expand_mul(gate0_scaled, w1_scale0)
            if SWIGLU_LIMIT > 0.0:
                gate0_fp32 = pl.minimum(gate0_fp32, SWIGLU_LIMIT)
            sigmoid0 = pl.recip(pl.add(pl.exp(pl.neg(gate0_fp32)), 1.0))
            silu0 = pl.mul(gate0_fp32, sigmoid0)
            up0_i32 = up_tile_i32[quant_row : quant_row + QUANT_ROW_TILE, :ACT_INTER_TILE]
            up0_fp32 = pl.cast(up0_i32, target_type=pl.FP32, mode="none")
            up0_scaled = pl.row_expand_mul(up0_fp32, x_scale)
            w3_scale0 = routed_w3_scale[local_e : local_e + 1, :ACT_INTER_TILE]
            up0_fp32 = pl.col_expand_mul(up0_scaled, w3_scale0)
            if SWIGLU_LIMIT > 0.0:
                up0_fp32 = pl.maximum(pl.minimum(up0_fp32, SWIGLU_LIMIT), -SWIGLU_LIMIT)
            activated0 = pl.mul(silu0, up0_fp32)
            activated0 = pl.set_validshape(activated0, valid_quant_rows, ACT_INTER_TILE)
            activated0 = pl.fillpad(activated0, pad_value=pl.PadValue.zero)
            abs0 = pl.maximum(activated0, pl.neg(activated0))
            row_amax = pl.maximum(row_amax, pl.reshape(pl.row_max(abs0), [1, QUANT_ROW_TILE]))

            gate1_i32 = gate_tile_i32[quant_row : quant_row + QUANT_ROW_TILE, ACT_INTER_TILE:]
            gate1_fp32 = pl.cast(gate1_i32, target_type=pl.FP32, mode="none")
            gate1_scaled = pl.row_expand_mul(gate1_fp32, x_scale)
            w1_scale1 = routed_w1_scale[local_e : local_e + 1, ACT_INTER_TILE:]
            gate1_fp32 = pl.col_expand_mul(gate1_scaled, w1_scale1)
            if SWIGLU_LIMIT > 0.0:
                gate1_fp32 = pl.minimum(gate1_fp32, SWIGLU_LIMIT)
            sigmoid1 = pl.recip(pl.add(pl.exp(pl.neg(gate1_fp32)), 1.0))
            silu1 = pl.mul(gate1_fp32, sigmoid1)
            up1_i32 = up_tile_i32[quant_row : quant_row + QUANT_ROW_TILE, ACT_INTER_TILE:]
            up1_fp32 = pl.cast(up1_i32, target_type=pl.FP32, mode="none")
            up1_scaled = pl.row_expand_mul(up1_fp32, x_scale)
            w3_scale1 = routed_w3_scale[local_e : local_e + 1, ACT_INTER_TILE:]
            up1_fp32 = pl.col_expand_mul(up1_scaled, w3_scale1)
            if SWIGLU_LIMIT > 0.0:
                up1_fp32 = pl.maximum(pl.minimum(up1_fp32, SWIGLU_LIMIT), -SWIGLU_LIMIT)
            activated1 = pl.mul(silu1, up1_fp32)
            activated1 = pl.set_validshape(activated1, valid_quant_rows, MOE_INTER - ACT_INTER_TILE)
            activated1 = pl.fillpad(activated1, pad_value=pl.PadValue.zero)
            abs1 = pl.maximum(activated1, pl.neg(activated1))
            row_amax = pl.maximum(row_amax, pl.reshape(pl.row_max(abs1), [1, QUANT_ROW_TILE]))

            quant_scale = pl.div(pl.full([1, QUANT_ROW_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), row_amax)
            dequant_scale = pl.recip(quant_scale)
            route_weight = recv_weights[local_e : local_e + 1, recv_row : recv_row + QUANT_ROW_TILE]
            h_tile_row_scale[:, quant_row : quant_row + QUANT_ROW_TILE] = pl.mul(dequant_scale, route_weight)
            quant_scale_col = pl.reshape(quant_scale, [QUANT_ROW_TILE, 1])
            scaled0 = pl.row_expand_mul(activated0, quant_scale_col)
            i32_0 = pl.cast(scaled0, target_type=pl.INT32, mode="rint")
            fp16_0 = pl.cast(i32_0, target_type=pl.FP16, mode="round")
            h_tile_i8[quant_row : quant_row + QUANT_ROW_TILE, :ACT_INTER_TILE] = pl.cast(fp16_0, pl.INT8, mode="trunc")
            scaled1 = pl.row_expand_mul(activated1, quant_scale_col)
            i32_1 = pl.cast(scaled1, target_type=pl.INT32, mode="rint")
            fp16_1 = pl.cast(i32_1, target_type=pl.FP16, mode="round")
            h_tile_i8[quant_row : quant_row + QUANT_ROW_TILE, ACT_INTER_TILE:] = pl.cast(fp16_1, pl.INT8, mode="trunc")
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
        h_l1 = pl.create_l1([RECV_TILE, MOE_INTER], dtype=pl.INT8)
        h_l1 = pl.gather_row(h_l1, h_tile_i8, [0, 0], [0, 0], [RECV_TILE, MOE_INTER])
        for inner in pl.range(W2_INNER):
            d0 = d_base + inner * D_OUT_TILE
            y_acc = pl.create_tensor([1, RECV_TILE, D_OUT_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, MOE_INTER, INTER_K, stage=2):
                h_w2_chunk = h_l1[:, k0 : k0 + INTER_K]
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
        row_scale = h_tile_row_scale[:, :]
        row_scale = pl.reshape(row_scale, [RECV_TILE, 1])
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
