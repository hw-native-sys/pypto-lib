# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl

from .config import FLASH as M, MOE_TOKENS, INT8_SCALE_MAX, INT8_AMAX_EPS

T = MOE_TOKENS

D = M.hidden_size

MOE_INTER = M.moe_intermediate_size

SWIGLU_LIMIT = M.swiglu_limit

SH_M_TILE = 32

SH_ROW_PAD = 8

SH_ROWS_PER_BLOCK = 2

T_PAD = ((T + SH_M_TILE - 1) // SH_M_TILE) * SH_M_TILE

assert T <= SH_M_TILE or T % SH_M_TILE == 0, (
    "expert_shared needs T <= SH_M_TILE (decode) or T a multiple of SH_M_TILE (prefill)"
)

SH_VALID_M = T if T < SH_M_TILE else SH_M_TILE

N_MTILES = T_PAD // SH_M_TILE

assert SH_VALID_M % SH_ROWS_PER_BLOCK == 0

K_TILE = 1024

INTER_K = 512

MM_INTER_TILE = 128

SH_MM_BLOCKS = 6

ACT_INTER_TILE = 1024

D_OUT_TILE = 256

W2_INNER = 2

QUANT_TILE = 2048

D_OUT_TILE_ACT = 128

W2_ACT_INNER = 8


def _expert_shared(
    x_local_i8: pl.Tensor[[T, D], pl.INT8],
    x_local_scale_dq: pl.Tensor[[T, 1], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8, pl.NZ],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    sh: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    # One M-tile of SH_M_TILE rows per iteration.
    for mt in pl.parallel(N_MTILES):
        ts0 = mt * SH_M_TILE

        gate_i32 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.INT32)
        up_i32 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.INT32)

        # gate (w1) cube matmul -> INT32 GM accumulator.
        for nb_idx in pl.spmd(
            SH_MM_BLOCKS,
            name_hint="sh_gate_mm",
            allow_early_resolve=True,
        ):
            # Weight reads bypass L2.
            pl.set_cache_policy(shared_w1, pl.CachePolicy.BYPASS)
            for ng in pl.range(nb_idx, MOE_INTER // MM_INTER_TILE, SH_MM_BLOCKS):
                n0 = ng * MM_INTER_TILE
                gate_acc = pl.create_tensor([SH_M_TILE, MM_INTER_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                    xs_k = pl.slice(
                        x_local_i8, [SH_M_TILE, K_TILE], [ts0, k0], valid_shape=[SH_VALID_M, K_TILE]
                    )
                    sw1_k = shared_w1[n0 : n0 + MM_INTER_TILE, k0 : k0 + K_TILE]
                    gate_acc = pl.matmul_acc(gate_acc, xs_k, sw1_k, b_trans=True, init_cond=(k0 == 0))
                gate_i32[:, n0 : n0 + MM_INTER_TILE] = gate_acc

        # up (w3) cube matmul -> INT32 GM accumulator.
        for nb_idx in pl.spmd(
            SH_MM_BLOCKS,
            name_hint="sh_up_mm",
            allow_early_resolve=True,
        ):
            # Weight reads bypass L2.
            pl.set_cache_policy(shared_w3, pl.CachePolicy.BYPASS)
            for ng in pl.range(nb_idx, MOE_INTER // MM_INTER_TILE, SH_MM_BLOCKS):
                n0 = ng * MM_INTER_TILE
                up_acc = pl.create_tensor([SH_M_TILE, MM_INTER_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                    xs_k = pl.slice(
                        x_local_i8, [SH_M_TILE, K_TILE], [ts0, k0], valid_shape=[SH_VALID_M, K_TILE]
                    )
                    sw3_k = shared_w3[n0 : n0 + MM_INTER_TILE, k0 : k0 + K_TILE]
                    up_acc = pl.matmul_acc(up_acc, xs_k, sw3_k, b_trans=True, init_cond=(k0 == 0))
                up_i32[:, n0 : n0 + MM_INTER_TILE] = up_acc

        # Each AIV block owns SH_ROWS_PER_BLOCK rows across the intermediate axis.
        # Activation, row-amax, scale, and requant share one task.
        h_tile_fp32 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.FP32)
        h_tile_i8 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.INT8)
        h_tile_scale_dq = pl.create_tensor([SH_M_TILE, SH_ROW_PAD], dtype=pl.FP32, manual_dep=True)
        for init_block in pl.spmd(SH_M_TILE // 16, name_hint="sh_h_tile_i8_init"):
            init_row = init_block * 16
            h_tile_i8[init_row : init_row + 16, :] = pl.cast(
                pl.full([16, MOE_INTER], dtype=pl.FP16, value=0.0),
                target_type=pl.INT8,
                mode="trunc",
            )
        for row_block in pl.spmd(
            SH_VALID_M // SH_ROWS_PER_BLOCK,
            name_hint="sh_gate_up_act_q",
            allow_early_resolve=True,
        ):
            row0 = row_block * SH_ROWS_PER_BLOCK
            x_scale = pl.slice(
                x_local_scale_dq,
                [SH_ROW_PAD, 1],
                [ts0 + row0, 0],
                valid_shape=[SH_ROWS_PER_BLOCK, 1],
            )
            row_amax = pl.full(
                [1, SH_ROW_PAD],
                dtype=pl.FP32,
                value=INT8_AMAX_EPS,
            )
            for part in pl.pipeline(0, MOE_INTER // ACT_INTER_TILE, stage=1):
                n0 = part * ACT_INTER_TILE
                gate_rows_i32 = pl.slice(
                    gate_i32,
                    [SH_ROW_PAD, ACT_INTER_TILE],
                    [row0, n0],
                    valid_shape=[SH_ROWS_PER_BLOCK, ACT_INTER_TILE],
                )
                up_rows_i32 = pl.slice(
                    up_i32,
                    [SH_ROW_PAD, ACT_INTER_TILE],
                    [row0, n0],
                    valid_shape=[SH_ROWS_PER_BLOCK, ACT_INTER_TILE],
                )
                w1_scale = pl.reshape(
                    shared_w1_scale[n0 : n0 + ACT_INTER_TILE],
                    [1, ACT_INTER_TILE],
                )
                w3_scale = pl.reshape(
                    shared_w3_scale[n0 : n0 + ACT_INTER_TILE],
                    [1, ACT_INTER_TILE],
                )
                gate_fp32 = pl.cast(gate_rows_i32, target_type=pl.FP32, mode="none")
                up_fp32 = pl.cast(up_rows_i32, target_type=pl.FP32, mode="none")
                gate_fp32 = pl.col_expand_mul(pl.row_expand_mul(gate_fp32, x_scale), w1_scale)
                up_fp32 = pl.col_expand_mul(pl.row_expand_mul(up_fp32, x_scale), w3_scale)
                if SWIGLU_LIMIT > 0.0:
                    gate_fp32 = pl.minimum(gate_fp32, SWIGLU_LIMIT)
                    up_fp32 = pl.maximum(pl.minimum(up_fp32, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_fp32)), 1.0))
                gated = pl.mul(pl.mul(gate_fp32, sigmoid), up_fp32)
                chunk_amax = pl.reshape(pl.row_max(pl.abs(gated)), [1, SH_ROW_PAD])
                row_amax = pl.maximum(row_amax, chunk_amax)
                h_tile_fp32[
                    row0 : row0 + SH_ROWS_PER_BLOCK,
                    n0 : n0 + ACT_INTER_TILE,
                ] = gated[0:SH_ROWS_PER_BLOCK, :]

            row_scale_q = pl.div(
                pl.full(
                    [1, SH_ROW_PAD],
                    dtype=pl.FP32,
                    value=INT8_SCALE_MAX,
                ),
                row_amax,
            )
            row_scale_q_col = pl.reshape(row_scale_q, [SH_ROW_PAD, 1])
            row_scale_dq_col = pl.reshape(pl.recip(row_scale_q), [SH_ROW_PAD, 1])
            row_scale_dq = pl.row_expand(
                pl.full([SH_ROW_PAD, SH_ROW_PAD], dtype=pl.FP32, value=0.0),
                row_scale_dq_col,
            )
            h_tile_scale_dq[row0 : row0 + SH_ROWS_PER_BLOCK, :] = row_scale_dq[0:SH_ROWS_PER_BLOCK, :]
            for q_idx in pl.pipeline(0, MOE_INTER // QUANT_TILE, stage=1):
                k0 = q_idx * QUANT_TILE
                h_fp32 = pl.slice(
                    h_tile_fp32,
                    [SH_ROW_PAD, QUANT_TILE],
                    [row0, k0],
                    valid_shape=[SH_ROWS_PER_BLOCK, QUANT_TILE],
                )
                h_scaled = pl.row_expand_mul(h_fp32, row_scale_q_col)
                h_i32 = pl.cast(h_scaled, target_type=pl.INT32, mode="rint")
                h_fp16 = pl.cast(h_i32, target_type=pl.FP16, mode="round")
                h_tile_i8[
                    row0 : row0 + SH_ROWS_PER_BLOCK,
                    k0 : k0 + QUANT_TILE,
                ] = pl.cast(h_fp16, target_type=pl.INT8, mode="trunc")[0:SH_ROWS_PER_BLOCK, :]

        # w2 (down) cube matmul -> INT32 GM accumulator.
        y_i32 = pl.create_tensor([SH_M_TILE, D], dtype=pl.INT32)
        for db_idx in pl.spmd(
            D // (W2_INNER * D_OUT_TILE),
            name_hint="sh_w2_mm",
            allow_early_resolve=True,
        ):
            # Weight reads bypass L2.
            pl.set_cache_policy(shared_w2, pl.CachePolicy.BYPASS)
            d_base = db_idx * (W2_INNER * D_OUT_TILE)
            for dg in pl.range(W2_INNER):
                d0 = d_base + dg * D_OUT_TILE
                y_acc = pl.create_tensor([SH_M_TILE, D_OUT_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, MOE_INTER, INTER_K, stage=2):
                    hs_k = h_tile_i8[:, k0 : k0 + INTER_K]
                    sw2_k = shared_w2[d0 : d0 + D_OUT_TILE, k0 : k0 + INTER_K]
                    y_acc = pl.matmul_acc(y_acc, hs_k, sw2_k, b_trans=True, init_cond=(k0 == 0))
                y_i32[:, d0 : d0 + D_OUT_TILE] = y_acc
        # Dequant w2 output (per-row h scale x per-channel w2 scale) -> BF16.
        for db_idx in pl.spmd(
            D // (W2_ACT_INNER * D_OUT_TILE_ACT),
            name_hint="sh_w2_act",
            allow_early_resolve=True,
        ):
            d_base = db_idx * (W2_ACT_INNER * D_OUT_TILE_ACT)
            h_scale = pl.row_max(h_tile_scale_dq[:, :])
            for dg in pl.pipeline(W2_ACT_INNER, stage=2):
                d0 = d_base + dg * D_OUT_TILE_ACT
                y_2d_i32 = y_i32[:, d0 : d0 + D_OUT_TILE_ACT]
                w2_scale_chunk = pl.reshape(shared_w2_scale[d0 : d0 + D_OUT_TILE_ACT], [1, D_OUT_TILE_ACT])
                y_2d = pl.cast(y_2d_i32, target_type=pl.FP32, mode="none")
                y_2d = pl.col_expand_mul(pl.row_expand_mul(y_2d, h_scale), w2_scale_chunk)
                # Write valid rows straight to the (unpadded) output, mirroring
                # expert_routed's direct recv_y store; no sh_pad round-trip.
                y_bf16 = pl.cast(y_2d, target_type=pl.BF16, mode="rint")
                sh[ts0 : ts0 + SH_VALID_M, d0 : d0 + D_OUT_TILE_ACT] = y_bf16[0:SH_VALID_M, :]

    # The @pl.inline parser requires inline call expressions to have a return
    # value; sh is convenient because it's already pl.Out.
    return sh


expert_shared = pl.jit.inline(_expert_shared)
