# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Drafter routing, expert execution, and ordered MoE combine."""

import pypto.language as pl
import dataclasses
import sys

import pypto.language.distributed as pld

from . import config
from ..hc_post_common import hc_post
from .hc_pre import hc_pre


_EP_DEFAULT = 2

_CHECKPOINT_N_ROUTED_EXPERTS = config.FLASH.n_routed_experts


def _parse_int_argv(name, default):
    for i, tok in enumerate(sys.argv):
        if tok == name and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if tok.startswith(name + "="):
            return int(tok.split("=", 1)[1])
    return default


EP = _parse_int_argv("--ep", _EP_DEFAULT)

_EXPERTS_PER_RANK_DEFAULT = min(_CHECKPOINT_N_ROUTED_EXPERTS // 8, _CHECKPOINT_N_ROUTED_EXPERTS // EP)

EXPERTS_PER_RANK = _parse_int_argv("--experts-per-rank", _EXPERTS_PER_RANK_DEFAULT)

if EXPERTS_PER_RANK <= 0:
    raise ValueError(f"--experts-per-rank must be a positive integer (got {EXPERTS_PER_RANK})")

config.FLASH = dataclasses.replace(config.FLASH, n_routed_experts=EXPERTS_PER_RANK * EP)

config.EP = EP

config.RECV_MAX = EP * config.MOE_TOKENS


# Freeze these shapes after applying the drafter's existing EP configuration.
from .config import FLASH as M, FP32_NEG_INF, INT8_AMAX_EPS, INT8_SCALE_MAX, MOE_TOKENS, RECV_MAX


D = M.hidden_size

MOE_INTER = M.moe_intermediate_size

SWIGLU_LIMIT = M.swiglu_limit

N_LOCAL_EXPERTS = M.n_routed_experts // EP

RECV_TILE = 64

ROUTED_K_TILE = 512

INTER_K = 512

ROUTED_MM_INTER_TILE = 256

MM_GATE_INNER = 4

ROUTED_ACT_INTER_TILE = 64

ACT_GATE_INNER = 4

D_OUT_TILE = 256

ROUTED_QUANT_TILE = 512

QUANT_ROW_TILE = 16

QUANT_SCALE_PAD = 8

ROUTED_D_OUT_TILE_ACT = 256

ROUTED_W2_INNER = 1

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
            MOE_INTER // (MM_GATE_INNER * ROUTED_MM_INTER_TILE), name_hint="exp_gate_mm", deps=[inputs_ready]
        ):
            # Weight reads bypass L2.
            pl.set_cache_policy(routed_w1, pl.CachePolicy.BYPASS)
            block = pl.tile.get_block_idx()
            n_base = block * (MM_GATE_INNER * ROUTED_MM_INTER_TILE)
            for inner in pl.range(MM_GATE_INNER):
                n0 = n_base + inner * ROUTED_MM_INTER_TILE
                gate_acc = pl.create_tensor([1, RECV_TILE, ROUTED_MM_INTER_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, D, ROUTED_K_TILE, stage=2):
                    x_chunk = recv_x_flat[flat_tile_row : flat_tile_row + RECV_TILE, k0 : k0 + ROUTED_K_TILE]
                    w1_chunk = routed_w1[local_e : local_e + 1, n0 : n0 + ROUTED_MM_INTER_TILE, k0 : k0 + ROUTED_K_TILE]
                    gate_acc = pl.matmul_acc(gate_acc, x_chunk, w1_chunk, b_trans=True, init_cond=(k0 == 0))
                gate_tile_i32[:, n0 : n0 + ROUTED_MM_INTER_TILE] = pl.reshape(gate_acc, [RECV_TILE, ROUTED_MM_INTER_TILE])

        with pl.spmd(
            MOE_INTER // (MM_GATE_INNER * ROUTED_MM_INTER_TILE), name_hint="exp_up_mm", deps=[inputs_ready]
        ):
            # Weight reads bypass L2.
            pl.set_cache_policy(routed_w3, pl.CachePolicy.BYPASS)
            block = pl.tile.get_block_idx()
            n_base = block * (MM_GATE_INNER * ROUTED_MM_INTER_TILE)
            for inner in pl.range(MM_GATE_INNER):
                n0 = n_base + inner * ROUTED_MM_INTER_TILE
                up_acc = pl.create_tensor([1, RECV_TILE, ROUTED_MM_INTER_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, D, ROUTED_K_TILE, stage=2):
                    x_chunk = recv_x_flat[flat_tile_row : flat_tile_row + RECV_TILE, k0 : k0 + ROUTED_K_TILE]
                    w3_chunk = routed_w3[local_e : local_e + 1, n0 : n0 + ROUTED_MM_INTER_TILE, k0 : k0 + ROUTED_K_TILE]
                    up_acc = pl.matmul_acc(up_acc, x_chunk, w3_chunk, b_trans=True, init_cond=(k0 == 0))
                up_tile_i32[:, n0 : n0 + ROUTED_MM_INTER_TILE] = pl.reshape(up_acc, [RECV_TILE, ROUTED_MM_INTER_TILE])

        h_tile_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32)
        with pl.spmd(MOE_INTER // (ACT_GATE_INNER * ROUTED_ACT_INTER_TILE), name_hint="exp_gate_up_act"):
            block = pl.tile.get_block_idx()
            inter_base = block * (ACT_GATE_INNER * ROUTED_ACT_INTER_TILE)
            for inner in pl.pipeline(ACT_GATE_INNER, stage=2):
                inter0 = inter_base + inner * ROUTED_ACT_INTER_TILE
                gate_i32 = gate_tile_i32[:, inter0 : inter0 + ROUTED_ACT_INTER_TILE]
                up_i32 = up_tile_i32[:, inter0 : inter0 + ROUTED_ACT_INTER_TILE]
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
                        inter0 : inter0 + ROUTED_ACT_INTER_TILE,
                    ],
                )
                up_fp32 = pl.col_expand_mul(
                    pl.row_expand_mul(
                        pl.cast(up_i32, target_type=pl.FP32, mode="none"),
                        x_scale,
                    ),
                    routed_w3_scale[
                        local_e : local_e + 1,
                        inter0 : inter0 + ROUTED_ACT_INTER_TILE,
                    ],
                )
                if SWIGLU_LIMIT > 0.0:
                    gate_fp32 = pl.minimum(gate_fp32, SWIGLU_LIMIT)
                    up_fp32 = pl.maximum(pl.minimum(up_fp32, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_fp32)), 1.0))
                activated = pl.mul(pl.mul(gate_fp32, sigmoid), up_fp32)
                activated = pl.set_validshape(activated, valid_rows, ROUTED_ACT_INTER_TILE)
                h_tile_fp32[:, inter0 : inter0 + ROUTED_ACT_INTER_TILE] = pl.fillpad(
                    activated, pad_value=pl.PadValue.zero
                )

        with pl.spmd(RECV_TILE // QUANT_ROW_TILE, name_hint="exp_h_q") as quant_tid:
            quant_block = pl.tile.get_block_idx()
            quant_row = quant_block * QUANT_ROW_TILE
            row_amax = pl.full([1, QUANT_ROW_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for k0 in pl.pipeline(0, MOE_INTER, ROUTED_QUANT_TILE, stage=2):
                h_amax_chunk = h_tile_fp32[quant_row : quant_row + QUANT_ROW_TILE, k0 : k0 + ROUTED_QUANT_TILE]
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
            for k0 in pl.pipeline(0, MOE_INTER, ROUTED_QUANT_TILE, stage=2):
                h_quant_chunk = h_tile_fp32[quant_row : quant_row + QUANT_ROW_TILE, k0 : k0 + ROUTED_QUANT_TILE]
                h_scaled = pl.row_expand_mul(h_quant_chunk, quant_scale_col)
                h_i32 = pl.cast(h_scaled, target_type=pl.INT32, mode="rint")
                h_fp16 = pl.cast(h_i32, target_type=pl.FP16, mode="round")
                h_tile_i8[
                    quant_row : quant_row + QUANT_ROW_TILE,
                    k0 : k0 + ROUTED_QUANT_TILE,
                ] = pl.cast(
                    h_fp16,
                    target_type=pl.INT8,
                    mode="trunc",
                )
        quant_tids[0] = quant_tid

    y_i32 = pl.create_tensor([RECV_TILE, D], dtype=pl.INT32)
    with pl.spmd(
        D // (ROUTED_W2_INNER * D_OUT_TILE),
        name_hint="exp_w2_mm",
        deps=[quant_tids[0]],
        allow_early_resolve=True,
    ) as w2_tid:
        # Weight reads bypass L2.
        pl.set_cache_policy(routed_w2, pl.CachePolicy.BYPASS)
        block = pl.tile.get_block_idx()
        d_base = block * (ROUTED_W2_INNER * D_OUT_TILE)
        for inner in pl.range(ROUTED_W2_INNER):
            d0 = d_base + inner * D_OUT_TILE
            y_acc = pl.create_tensor([1, RECV_TILE, D_OUT_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, MOE_INTER, INTER_K, stage=2):
                h_w2_chunk = h_tile_i8[:, k0 : k0 + INTER_K]
                w2_chunk = routed_w2[local_e : local_e + 1, d0 : d0 + D_OUT_TILE, k0 : k0 + INTER_K]
                y_acc = pl.matmul_acc(y_acc, h_w2_chunk, w2_chunk, b_trans=True, init_cond=(k0 == 0))
            y_i32[:, d0 : d0 + D_OUT_TILE] = pl.reshape(y_acc, [RECV_TILE, D_OUT_TILE])

    with pl.spmd(
        D // (W2_ACT_INNER * ROUTED_D_OUT_TILE_ACT),
        name_hint="exp_w2_act",
        deps=[w2_tid],
        allow_early_resolve=True,
    ) as w2_act_tid:
        block = pl.tile.get_block_idx()
        d_base = block * (W2_ACT_INNER * ROUTED_D_OUT_TILE_ACT)
        route_weight = pl.reshape(
            recv_weights[local_e : local_e + 1, tile_row : tile_row + RECV_TILE], [RECV_TILE, 1]
        )
        row_scale = pl.mul(pl.row_max(h_tile_scale_dq), route_weight)
        for inner in pl.pipeline(W2_ACT_INNER, stage=2):
            d0 = d_base + inner * ROUTED_D_OUT_TILE_ACT
            y_fp32 = pl.cast(y_i32[:, d0 : d0 + ROUTED_D_OUT_TILE_ACT], target_type=pl.FP32, mode="none")
            y_fp32 = pl.col_expand_mul(
                pl.row_expand_mul(y_fp32, row_scale),
                routed_w2_scale[
                    local_e : local_e + 1,
                    d0 : d0 + ROUTED_D_OUT_TILE_ACT,
                ],
            )
            recv_y_tile[:, d0 : d0 + ROUTED_D_OUT_TILE_ACT] = pl.cast(y_fp32, target_type=pl.BF16, mode="rint")
    return w2_act_tid


T = MOE_TOKENS




SH_M_TILE = 32

SH_ROW_PAD = 8

SH_ROWS_PER_BLOCK = 2

SHARED_T_PAD = ((T + SH_M_TILE - 1) // SH_M_TILE) * SH_M_TILE

assert T <= SH_M_TILE or T % SH_M_TILE == 0, (
    "expert_shared needs T <= SH_M_TILE (decode) or T a multiple of SH_M_TILE (prefill)"
)

SH_VALID_M = T if T < SH_M_TILE else SH_M_TILE

N_MTILES = SHARED_T_PAD // SH_M_TILE

assert SH_VALID_M % SH_ROWS_PER_BLOCK == 0

SHARED_K_TILE = 1024


SHARED_MM_INTER_TILE = 128

SH_MM_BLOCKS = 6

SHARED_ACT_INTER_TILE = 1024


SHARED_W2_INNER = 2

SHARED_QUANT_TILE = 2048

SHARED_D_OUT_TILE_ACT = 128



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
            for ng in pl.range(nb_idx, MOE_INTER // SHARED_MM_INTER_TILE, SH_MM_BLOCKS):
                n0 = ng * SHARED_MM_INTER_TILE
                gate_acc = pl.create_tensor([SH_M_TILE, SHARED_MM_INTER_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, D, SHARED_K_TILE, stage=2):
                    xs_k = pl.slice(
                        x_local_i8, [SH_M_TILE, SHARED_K_TILE], [ts0, k0], valid_shape=[SH_VALID_M, SHARED_K_TILE]
                    )
                    sw1_k = shared_w1[n0 : n0 + SHARED_MM_INTER_TILE, k0 : k0 + SHARED_K_TILE]
                    gate_acc = pl.matmul_acc(gate_acc, xs_k, sw1_k, b_trans=True, init_cond=(k0 == 0))
                gate_i32[:, n0 : n0 + SHARED_MM_INTER_TILE] = gate_acc

        # up (w3) cube matmul -> INT32 GM accumulator.
        for nb_idx in pl.spmd(
            SH_MM_BLOCKS,
            name_hint="sh_up_mm",
            allow_early_resolve=True,
        ):
            # Weight reads bypass L2.
            pl.set_cache_policy(shared_w3, pl.CachePolicy.BYPASS)
            for ng in pl.range(nb_idx, MOE_INTER // SHARED_MM_INTER_TILE, SH_MM_BLOCKS):
                n0 = ng * SHARED_MM_INTER_TILE
                up_acc = pl.create_tensor([SH_M_TILE, SHARED_MM_INTER_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, D, SHARED_K_TILE, stage=2):
                    xs_k = pl.slice(
                        x_local_i8, [SH_M_TILE, SHARED_K_TILE], [ts0, k0], valid_shape=[SH_VALID_M, SHARED_K_TILE]
                    )
                    sw3_k = shared_w3[n0 : n0 + SHARED_MM_INTER_TILE, k0 : k0 + SHARED_K_TILE]
                    up_acc = pl.matmul_acc(up_acc, xs_k, sw3_k, b_trans=True, init_cond=(k0 == 0))
                up_i32[:, n0 : n0 + SHARED_MM_INTER_TILE] = up_acc

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
            for part in pl.pipeline(0, MOE_INTER // SHARED_ACT_INTER_TILE, stage=1):
                n0 = part * SHARED_ACT_INTER_TILE
                gate_rows_i32 = pl.slice(
                    gate_i32,
                    [SH_ROW_PAD, SHARED_ACT_INTER_TILE],
                    [row0, n0],
                    valid_shape=[SH_ROWS_PER_BLOCK, SHARED_ACT_INTER_TILE],
                )
                up_rows_i32 = pl.slice(
                    up_i32,
                    [SH_ROW_PAD, SHARED_ACT_INTER_TILE],
                    [row0, n0],
                    valid_shape=[SH_ROWS_PER_BLOCK, SHARED_ACT_INTER_TILE],
                )
                w1_scale = pl.reshape(
                    shared_w1_scale[n0 : n0 + SHARED_ACT_INTER_TILE],
                    [1, SHARED_ACT_INTER_TILE],
                )
                w3_scale = pl.reshape(
                    shared_w3_scale[n0 : n0 + SHARED_ACT_INTER_TILE],
                    [1, SHARED_ACT_INTER_TILE],
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
                    n0 : n0 + SHARED_ACT_INTER_TILE,
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
            for q_idx in pl.pipeline(0, MOE_INTER // SHARED_QUANT_TILE, stage=1):
                k0 = q_idx * SHARED_QUANT_TILE
                h_fp32 = pl.slice(
                    h_tile_fp32,
                    [SH_ROW_PAD, SHARED_QUANT_TILE],
                    [row0, k0],
                    valid_shape=[SH_ROWS_PER_BLOCK, SHARED_QUANT_TILE],
                )
                h_scaled = pl.row_expand_mul(h_fp32, row_scale_q_col)
                h_i32 = pl.cast(h_scaled, target_type=pl.INT32, mode="rint")
                h_fp16 = pl.cast(h_i32, target_type=pl.FP16, mode="round")
                h_tile_i8[
                    row0 : row0 + SH_ROWS_PER_BLOCK,
                    k0 : k0 + SHARED_QUANT_TILE,
                ] = pl.cast(h_fp16, target_type=pl.INT8, mode="trunc")[0:SH_ROWS_PER_BLOCK, :]

        # w2 (down) cube matmul -> INT32 GM accumulator.
        y_i32 = pl.create_tensor([SH_M_TILE, D], dtype=pl.INT32)
        for db_idx in pl.spmd(
            D // (SHARED_W2_INNER * D_OUT_TILE),
            name_hint="sh_w2_mm",
            allow_early_resolve=True,
        ):
            # Weight reads bypass L2.
            pl.set_cache_policy(shared_w2, pl.CachePolicy.BYPASS)
            d_base = db_idx * (SHARED_W2_INNER * D_OUT_TILE)
            for dg in pl.range(SHARED_W2_INNER):
                d0 = d_base + dg * D_OUT_TILE
                y_acc = pl.create_tensor([SH_M_TILE, D_OUT_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, MOE_INTER, INTER_K, stage=2):
                    hs_k = h_tile_i8[:, k0 : k0 + INTER_K]
                    sw2_k = shared_w2[d0 : d0 + D_OUT_TILE, k0 : k0 + INTER_K]
                    y_acc = pl.matmul_acc(y_acc, hs_k, sw2_k, b_trans=True, init_cond=(k0 == 0))
                y_i32[:, d0 : d0 + D_OUT_TILE] = y_acc
        # Dequant w2 output (per-row h scale x per-channel w2 scale) -> BF16.
        for db_idx in pl.spmd(
            D // (W2_ACT_INNER * SHARED_D_OUT_TILE_ACT),
            name_hint="sh_w2_act",
            allow_early_resolve=True,
        ):
            d_base = db_idx * (W2_ACT_INNER * SHARED_D_OUT_TILE_ACT)
            h_scale = pl.row_max(h_tile_scale_dq[:, :])
            for dg in pl.pipeline(W2_ACT_INNER, stage=2):
                d0 = d_base + dg * SHARED_D_OUT_TILE_ACT
                y_2d_i32 = y_i32[:, d0 : d0 + SHARED_D_OUT_TILE_ACT]
                w2_scale_chunk = pl.reshape(shared_w2_scale[d0 : d0 + SHARED_D_OUT_TILE_ACT], [1, SHARED_D_OUT_TILE_ACT])
                y_2d = pl.cast(y_2d_i32, target_type=pl.FP32, mode="none")
                y_2d = pl.col_expand_mul(pl.row_expand_mul(y_2d, h_scale), w2_scale_chunk)
                # Write valid rows straight to the (unpadded) output, mirroring
                # expert_routed's direct recv_y store; no sh_pad round-trip.
                y_bf16 = pl.cast(y_2d, target_type=pl.BF16, mode="rint")
                sh[ts0 : ts0 + SH_VALID_M, d0 : d0 + SHARED_D_OUT_TILE_ACT] = y_bf16[0:SH_VALID_M, :]

    # The @pl.inline parser requires inline call expressions to have a return
    # value; sh is convenient because it's already pl.Out.
    return sh


expert_shared = pl.jit.inline(_expert_shared)


NORM_EPS = M.rms_norm_eps

N_EXPERTS = M.n_routed_experts

TOPK = M.num_experts_per_tok

ROUTE_SCALE = M.routed_scaling_factor

VOCAB = M.vocab_size

N_HASH_LAYERS = M.num_hash_layers

T_TILE = 8

NORM_TOKEN_TILE = 4

GATE_T_TILE = 8

GATE_M_TILE = 32  # cube M-tile: matmul rows must be a multiple of 16 (fractal)

GATE_N_TILE = 32  # expert columns per gate spmd block

assert N_EXPERTS % GATE_N_TILE == 0

GATE_T_PAD = ((T + GATE_M_TILE - 1) // GATE_M_TILE) * GATE_M_TILE

ROW_PAD = 8

FFN_REDUCE_TILE = D // ROW_PAD

assert D % ROW_PAD == 0

GATE_D_TILE = 1024

assert D % GATE_D_TILE == 0, "gate K-loop must cover D"

GATE_QUANT_TILE = 256

SCORE_PAD = 256  # padded expert row for sort32 + mrgsort

TOPK_PAD = 8  # TOPK padded to 32B-aligned width

SORT_PAD = TOPK_PAD * 2  # (val, idx) interleaved slice width

assert TOPK <= TOPK_PAD


def _gate(
    x_mixed: pl.Tensor[[T, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    x_norm_i8: pl.Out[pl.Tensor[[T, D], pl.INT8]],
    x_norm_scale: pl.Out[pl.Tensor[[T, 1], pl.FP32]],
    indices: pl.Out[pl.Tensor[[T, TOPK], pl.INT32]],
    weights: pl.Out[pl.Tensor[[T, TOPK], pl.FP32]],
):
    # Deferred RMSNorm (qwen3-style): store xg = x*gamma (NOT *inv_rms), because
    # the per-token positive scalar inv_rms factors out of everything downstream:
    #   - gate logits: inv_rms * (xg @ gate_w.T)  -> applied as a [16,1] row-scale
    #   - int8 quant : symmetric per-token quant of xg CANCELS inv_rms exactly;
    #                  inv_rms rides only x_norm_scale (= inv_rms * amax(xg)/127).
    # This lets sq_sum (needs raw x) and xg share ONE pass over x_mixed instead of
    # a sqsum pass followed by a separate normalize pass.
    xg_buf = pl.create_tensor([GATE_T_PAD, D], dtype=pl.FP32)
    inv_rms_buf = pl.create_tensor([GATE_T_PAD, 1], dtype=pl.FP32)
    # per-token int8 quant scale (= INT8_SCALE_MAX / amax(xg)), computed in ffn_norm
    # and consumed by x_norm_quant so quant skips its own amax pass.
    xn_scale_buf = pl.create_tensor([GATE_T_PAD, 1], dtype=pl.FP32)
    route_scores_buf = pl.create_tensor([GATE_T_PAD, SCORE_PAD], dtype=pl.FP32)
    biased_scores_buf = pl.create_tensor([GATE_T_PAD, SCORE_PAD], dtype=pl.FP32)
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)
    active_gate_tiles = (active_tokens + GATE_M_TILE - 1) // GATE_M_TILE
    # Keep one producer tile alive for an owner with zero local tokens.  The
    # following pre-route task overwrites the whole fixed-capacity output with
    # zeroes, while the rank still publishes zero route counts and participates
    # in the EP barriers.  A zero-sized producer loop leaves x_norm_i8 without a
    # producer and can stall its fixed-shape shared-expert consumer.
    if active_gate_tiles < 1:
        active_gate_tiles = pl.cast(1, pl.INDEX)
    active_gate_tokens = active_gate_tiles * GATE_M_TILE
    if active_gate_tokens > T:
        active_gate_tokens = pl.cast(T, pl.INDEX)

    # Each block normalizes a group of tokens with full-row reductions.
    norm_w_2d = pl.reshape(norm_w, [1, D])
    norm_blocks = (active_gate_tokens + NORM_TOKEN_TILE - 1) // NORM_TOKEN_TILE
    for tok_block in pl.spmd(norm_blocks, name_hint="ffn_norm", allow_early_resolve=True):
        tok0 = tok_block * NORM_TOKEN_TILE
        for tok in pl.range(tok0, pl.min(tok0 + NORM_TOKEN_TILE, active_gate_tokens)):
            rms_x = pl.cast(pl.tile.load(x_mixed, [tok, 0], [1, D]), pl.FP32)
            rms_w = pl.cast(pl.tile.load(norm_w_2d, [0, 0], [1, D]), pl.FP32)
            xg = pl.mul(rms_x, rms_w)
            pl.tile.store(xg, [tok, 0], xg_buf, shapes=[1, D])

            sq_rows = pl.reshape(pl.mul(rms_x, rms_x), [ROW_PAD, FFN_REDUCE_TILE])
            sq_partial_tmp = pl.create_tile([ROW_PAD, FFN_REDUCE_TILE], dtype=pl.FP32)
            sq_partial = pl.row_sum(sq_rows, sq_partial_tmp)
            sq_reduce = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
            sq_reduce[0:1, :] = pl.reshape(sq_partial, [1, ROW_PAD])
            sq_reduce = pl.set_validshape(sq_reduce, 1, ROW_PAD)
            sq_sum_tmp = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
            sq_sum = pl.row_sum(sq_reduce, sq_sum_tmp)
            sq_sum = pl.set_validshape(pl.reshape(sq_sum, [1, ROW_PAD]), 1, 1)
            inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, 1.0 / D), NORM_EPS)))
            pl.tile.store(inv_rms, [tok, 0], inv_rms_buf, shapes=[1, 1])

            xg_abs_rows = pl.reshape(pl.abs(xg), [ROW_PAD, FFN_REDUCE_TILE])
            amax_partial_tmp = pl.create_tile([ROW_PAD, FFN_REDUCE_TILE], dtype=pl.FP32)
            amax_partial = pl.row_max(xg_abs_rows, amax_partial_tmp)
            amax_reduce = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
            amax_reduce[0:1, :] = pl.reshape(amax_partial, [1, ROW_PAD])
            amax_reduce = pl.set_validshape(amax_reduce, 1, ROW_PAD)
            amax_tmp = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
            xg_amax = pl.row_max(amax_reduce, amax_tmp)
            xg_amax = pl.set_validshape(pl.reshape(xg_amax, [1, ROW_PAD]), 1, 1)
            amax_eps = pl.tile.full([1, ROW_PAD], dtype=pl.FP32, value=INT8_AMAX_EPS)
            amax_eps = pl.set_validshape(amax_eps, 1, 1)
            xg_amax = pl.maximum(xg_amax, amax_eps)
            # quant scale = INT8_SCALE_MAX / amax(xg); dequant scale rides inv_rms.
            scale_max = pl.tile.full([1, ROW_PAD], dtype=pl.FP32, value=INT8_SCALE_MAX)
            scale_max = pl.set_validshape(scale_max, 1, 1)
            xg_sq = pl.div(scale_max, xg_amax)
            xg_dequant_scale = pl.mul(xg_amax, 1.0 / INT8_SCALE_MAX)
            x_norm_dequant_scale = pl.mul(xg_dequant_scale, inv_rms)
            pl.tile.store(x_norm_dequant_scale, [tok, 0], x_norm_scale, shapes=[1, 1])
            pl.tile.store(xg_sq, [tok, 0], xn_scale_buf, shapes=[1, 1])

    seed_dummy = pl.system.task_dummy(deps=[])

    # Per-token symmetric INT8 quant of xg: scale precomputed in ffn_norm. inv_rms
    # cancels here (symmetric quant is invariant to a positive per-token scalar),
    # so x_norm_i8 = quant(xg). Early resolution lets the shared-expert chain
    # pre-stage before dispatch_push.
    for t0 in pl.parallel(0, active_gate_tokens, T_TILE):
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="x_norm_quant",
            deps=[seed_dummy],
            allow_early_resolve=True,
        ):
            xn_sq_col = xn_scale_buf[t0 : t0 + T_TILE, 0:1]
            for xq_b_k in pl.pipeline(0, D, GATE_QUANT_TILE, stage=2):
                xn_q_scaled = pl.row_expand_mul(
                    xg_buf[t0 : t0 + T_TILE, xq_b_k : xq_b_k + GATE_QUANT_TILE],
                    xn_sq_col,
                )
                xn_q_i32 = pl.cast(xn_q_scaled, pl.INT32, mode="rint")
                xn_q_half = pl.cast(xn_q_i32, pl.FP16, mode="round")
                x_norm_i8[t0 : t0 + T_TILE, xq_b_k : xq_b_k + GATE_QUANT_TILE] = pl.cast(
                    xn_q_half, pl.INT8, mode="trunc"
                )

    # Zero inactive fixed-tile inputs/outputs and mask padded expert scores.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_pre_route"):
        for zt in pl.range(T):
            if zt >= active_tokens:
                # Positive FP16 zero has zero bytes; no numeric conversion is needed.
                inactive_x_norm_f16 = pl.full([1, D // 2], dtype=pl.FP16, value=0.0)
                inactive_x_norm_i8 = pl.reinterpret_view(inactive_x_norm_f16, pl.INT8)
                x_norm_i8[zt : zt + 1, :] = inactive_x_norm_i8
                pl.write(x_norm_scale, [zt, 0], pl.cast(0.0, pl.FP32))
                for zk in pl.range(TOPK):
                    pl.write(indices, [zt, zk], pl.cast(0, pl.INT32))
                    pl.write(weights, [zt, zk], pl.cast(0.0, pl.FP32))
        if N_EXPERTS < SCORE_PAD:
            biased_scores_buf[:, N_EXPERTS:SCORE_PAD] = pl.full(
                [GATE_T_PAD, SCORE_PAD - N_EXPERTS], dtype=pl.FP32, value=FP32_NEG_INF
            )

    # Gate matmul + post: x_norm @ gate_w.T → sqrt(softplus(logits)) (+bias).
    # Fan the matmul over expert columns so each block computes a [GATE_M_TILE,
    # GATE_N_TILE] slice on its own core; token-tile is the dynamic dim, so it
    # stays outermost and // % divide by the compile-time GATE_N_BLOCKS.
    GATE_N_BLOCKS = N_EXPERTS // GATE_N_TILE
    for gb_idx in pl.spmd(active_gate_tiles * GATE_N_BLOCKS, name_hint="gate", allow_early_resolve=True):
        # Weight reads bypass L2.
        pl.set_cache_policy(gate_w, pl.CachePolicy.BYPASS)
        tg = gb_idx // GATE_N_BLOCKS
        nb = gb_idx % GATE_N_BLOCKS
        t1 = tg * GATE_M_TILE
        n0 = nb * GATE_N_TILE
        gp_bias_row = pl.reshape(gate_bias[n0 : n0 + GATE_N_TILE], [1, GATE_N_TILE])
        gate_logits_tile = pl.create_tensor([GATE_M_TILE, GATE_N_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // GATE_D_TILE, stage=2):
            gd_kd = kb * GATE_D_TILE
            gd_x = xg_buf[t1 : t1 + GATE_M_TILE, gd_kd : gd_kd + GATE_D_TILE]
            gd_w = gate_w[n0 : n0 + GATE_N_TILE, gd_kd : gd_kd + GATE_D_TILE]
            gate_logits_tile = pl.matmul_acc(
                gate_logits_tile, gd_x, gd_w, b_trans=True, init_cond=(gd_kd == 0)
            )
        # xg omitted inv_rms; logits = inv_rms * (xg @ gate_w.T). Per-token row-scale.
        gate_logits_tile = pl.row_expand_mul(gate_logits_tile, inv_rms_buf[t1 : t1 + GATE_M_TILE, 0:1])
        gp_relu = pl.maximum(gate_logits_tile, 0.0)
        gp_abs = pl.maximum(gate_logits_tile, pl.neg(gate_logits_tile))
        gp_softplus_log = pl.add(gp_relu, pl.log(pl.add(pl.exp(pl.neg(gp_abs)), 1.0)))
        gp_neg_floor_mask = pl.minimum(pl.maximum(pl.sub(pl.neg(gate_logits_tile), 10.0), 0.0), 1.0)
        gp_neg_floor = pl.mul(gp_neg_floor_mask, pl.exp(pl.minimum(gate_logits_tile, 0.0)))
        gp_softplus = pl.maximum(gp_softplus_log, gp_neg_floor)
        gp_score = pl.sqrt(gp_softplus)
        route_scores_buf[t1 : t1 + GATE_M_TILE, n0 : n0 + GATE_N_TILE] = gp_score
        if layer_id >= N_HASH_LAYERS:
            gp_bias = pl.col_expand_mul(
                pl.full([GATE_M_TILE, GATE_N_TILE], dtype=pl.FP32, value=1.0), gp_bias_row
            )
            gp_biased = pl.add(gp_score, gp_bias)
            biased_scores_buf[t1 : t1 + GATE_M_TILE, n0 : n0 + GATE_N_TILE] = gp_biased

    active_route_tiles = (active_tokens + GATE_T_TILE - 1) // GATE_T_TILE
    # PyPTO cannot launch an SPMD task with zero blocks.  The guarded store in
    # either routing path keeps this dummy tile from publishing any route.
    if active_route_tiles < 1:
        active_route_tiles = pl.cast(1, pl.INDEX)
    # Hash layers index via tid2eid[input_ids]; score layers sort+gather.
    if layer_id < N_HASH_LAYERS:
        for th_idx in pl.spmd(active_route_tiles, name_hint="route_hash", allow_early_resolve=True):
            t1 = th_idx * GATE_T_TILE
            # eids from tid2eid[input_ids]: scalar lookup (dynamic row index) into a
            # Tensor. tid2eid row load hits the 32B tile floor (TOPK*4=24B), so the
            # index build stays scalar; the score fetch is a batched gather below.
            # Tail [TOPK, TOPK_PAD) zeroed so fillpad drops it from the sum.
            hs_idx_tile = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)
            for hs_tt in pl.range(GATE_T_TILE):
                hs_token = pl.cast(pl.read(input_ids, [t1 + hs_tt]), pl.INDEX)
                for hs_k in pl.range(TOPK):
                    pl.write(hs_idx_tile, [hs_tt, hs_k], pl.read(tid2eid, [hs_token, hs_k]))
                for hs_pad_k in pl.range(TOPK, TOPK_PAD):
                    pl.write(hs_idx_tile, [hs_tt, hs_pad_k], pl.cast(0, pl.INT32))
            # Batched score gather (replaces per-eid scalar reads); set_validshape +
            # fillpad zero the [TOPK, TOPK_PAD) tail so row_sum sees only TOPK.
            local_scores = pl.create_tensor([GATE_T_TILE, SCORE_PAD], dtype=pl.FP32)
            local_scores[:, :] = route_scores_buf[t1 : t1 + GATE_T_TILE, :]
            gather_all = pl.gather(local_scores, dim=-1, index=hs_idx_tile)
            gather_valid = pl.set_validshape(gather_all, GATE_T_TILE, TOPK)
            hs_vals_pad = pl.fillpad(gather_valid, pad_value=pl.PadValue.zero)
            # Copy to dodge the tensor_view-vs-ptr SSA conflict between the gather
            # and the scalar pl.read below (pypto #1493).
            hs_idx_read = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)
            hs_idx_read[:, :] = hs_idx_tile[:, :]
            hs_denom = pl.reshape(pl.row_sum(hs_vals_pad), [GATE_T_TILE, 1])
            hs_weights_pad = pl.mul(pl.row_expand_div(hs_vals_pad, hs_denom), ROUTE_SCALE)
            for hs_wt_tt in pl.range(GATE_T_TILE):
                if t1 + hs_wt_tt < active_tokens:
                    for hs_wt_k in pl.range(TOPK):
                        pl.write(indices, [t1 + hs_wt_tt, hs_wt_k], pl.read(hs_idx_read, [hs_wt_tt, hs_wt_k]))
                        pl.write(
                            weights, [t1 + hs_wt_tt, hs_wt_k], pl.read(hs_weights_pad, [hs_wt_tt, hs_wt_k])
                        )
    else:
        for ts_idx in pl.spmd(active_route_tiles, name_hint="route_sort", allow_early_resolve=True):
            t1 = ts_idx * GATE_T_TILE
            # topk_idx_tile stays Tensor (created here, not a pl.full Tile) so
            # the batched pl.gather below accepts it — Tile-against-Tensor src
            # is rejected.
            topk_idx_tile = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)
            # ptoas pto.tmrgsort requires src rows == 1; sort path iterates
            # row-by-row. sort32: [1,256]→[1,512] (8 runs of 64). mrgsort
            # format1 4-way: 8→2 runs of 256. format2 2-way: 2→1 run of 512.
            for sr_tt in pl.range(GATE_T_TILE):
                sr_row = biased_scores_buf[t1 + sr_tt : t1 + sr_tt + 1, :]
                sr_idx_init = pl.arange(0, [1, SCORE_PAD], dtype=pl.UINT32)
                sr_sorted = pl.sort32(sr_row, sr_idx_init)
                sr_sorted = pl.mrgsort(sr_sorted, block_len=64)
                sr_sorted = pl.mrgsort(sr_sorted[:, 0:256], sr_sorted[:, 256:512])
                sr_pairs = sr_sorted[:, 0:SORT_PAD]
                sr_i = pl.gather(sr_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                topk_idx_tile[sr_tt : sr_tt + 1, :] = sr_i
            # Batched gather; set_validshape+fillpad zeros the [TOPK, TOPK_PAD)
            # tail so the normalize sum below sees only real TOPK entries.
            local_scores = pl.create_tensor([GATE_T_TILE, SCORE_PAD], dtype=pl.FP32)
            local_scores[:, :] = route_scores_buf[t1 : t1 + GATE_T_TILE, :]
            gather_all = pl.gather(local_scores, dim=-1, index=topk_idx_tile)
            gather_valid = pl.set_validshape(gather_all, GATE_T_TILE, TOPK)
            topk_vals_pad = pl.fillpad(gather_valid, pad_value=pl.PadValue.zero)
            # Copy topk_idx_tile to dodge the tensor_view-vs-ptr SSA conflict
            # between sort's slice-assign and scalar pl.read (pypto #1493).
            topk_idx_read = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)
            topk_idx_read[:, :] = topk_idx_tile[:, :]
            nm_denom = pl.reshape(pl.row_sum(topk_vals_pad), [GATE_T_TILE, 1])
            nm_weights_pad = pl.mul(pl.row_expand_div(topk_vals_pad, nm_denom), ROUTE_SCALE)
            for nm_tt in pl.range(GATE_T_TILE):
                if t1 + nm_tt < active_tokens:
                    for nm_k in pl.range(TOPK):
                        pl.write(indices, [t1 + nm_tt, nm_k], pl.read(topk_idx_read, [nm_tt, nm_k]))
                        pl.write(weights, [t1 + nm_tt, nm_k], pl.read(nm_weights_pad, [nm_tt, nm_k]))

    # The @pl.inline parser requires inline call expressions to have a return
    # value. weights is convenient because it's already pl.Out and reads as
    # the same SSA name on the caller side.
    return weights


gate = pl.jit.inline(_gate)


HC_MULT = M.hc_mult

MIX_HC = M.mix_hc

HC_DIM = M.hc_dim


N_RANKS = EP

N_EXPERTS_GLOBAL = M.n_routed_experts

N_LOCAL = N_EXPERTS_GLOBAL // N_RANKS

N_ROUTES = T * TOPK

MAX_PER_SRC = T

FP32_PER_CACHE_LINE = 16

INT64_PER_CACHE_LINE = 8

AUX_PAD = 8  # FP32 pack tile width (32 B min tile); cols: 0=scale 1=weight

AUX_SCALE = 0

AUX_W = 1

IDX_PAD = 8  # INT32 route tile width; route rides a separate window from scale/w

COMBINE_TOKEN_TILE = 4

assert RECV_MAX % FP32_PER_CACHE_LINE == 0

assert T % INT64_PER_CACHE_LINE == 0


@pl.jit.inline
def clear_moe_signals(
    completion_anchor: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    """Clear this rank's MoE signal windows after its final MoE completes."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_signal_clear") as cleanup_tid:
        # The final MoE output depends on this rank observing every peer's final
        # meta, payload, and combine notify. No peer can issue another MoE notify
        # to this rank in the current forward after this dependency is satisfied.
        _completion_anchor = pl.read(completion_anchor, [0, 0, 0])
        zero = pl.cast(0, pl.INT32)
        for src in pl.range(N_RANKS):
            pl.write(arrived, [src, 0], zero)
            pl.write(data_arrived, [src, 0], zero)
            pl.write(combine_arrived, [src, 0], zero)
    return cleanup_tid


@pl.jit.inline
def dispatch(
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    x_norm_i8: pl.Tensor[[T, D], pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1], pl.FP32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    # compact per-expert outputs consumed by expert_routed_tile / combine
    recv_x_out: pl.Tensor[[N_LOCAL, RECV_MAX, D], pl.INT8],
    recv_scale_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.FP32],
    recv_w_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.FP32],
    recv_r_route_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.INT32],
    recv_count_out: pl.Tensor[[N_LOCAL, 1], pl.INT32],
    recv_meta_local: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
    # windows
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    # 1-based MoE call id; `arrived`/`data_arrived` are monotonic so waits use `>= moe_epoch`.
    moe_epoch: pl.Scalar[pl.INT32],
    prior_cycle_gate: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    # Flat 2-D view kept outside the scope so it stays a tensor view, not a tile.
    recv_x_out_flat = pl.reshape(recv_x_out, [N_LOCAL * RECV_MAX, D])

    # Meta and payload arrivals ride two independent windows: `arrived` and
    # `data_arrived`.

    # Count routes and publish counts. The peer waits are registered separately
    # so they do not occupy a core while waiting for a remote rank.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dispatch_meta_publish",
        allow_early_resolve=True,
        deps=[prior_cycle_gate],
    ) as _meta_push_tid:
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)

        # Count how many routes land in each (dst, loc_e) lane (no payload move).
        cursor = pl.array.create(N_RANKS * N_LOCAL, pl.INT32)
        for d in pl.range(N_RANKS):
            for e in pl.range(N_LOCAL):
                cursor[d * N_LOCAL + e] = 0
        # Route ids ride one UB tile; a per-element pl.read is a GM access.
        indices_tile = pl.tile.load(indices, [0, 0], [T, IDX_PAD], valid_shape=[T, TOPK])
        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.tile.read(indices_tile, [t, k])
                dst = eid // N_LOCAL
                loc_e = eid - dst * N_LOCAL
                cursor[dst * N_LOCAL + loc_e] = cursor[dst * N_LOCAL + loc_e] + 1

        # One meta row per dst (all N_LOCAL counts, zeros included), then bump the
        # per-source arrival counter. AtomicAdd on a monotonic window is
        # order-independent, so a late notify from an earlier epoch cannot clobber it.
        meta_tile = pl.tile.full([1, N_LOCAL], dtype=pl.INT32, value=0)
        for dst in pl.range(N_RANKS):
            for e in pl.range(N_LOCAL):
                pl.tile.write(meta_tile, [0, e], cursor[dst * N_LOCAL + e])
            pld.tile.remote_store(meta_tile, target=recv_meta, peer=dst, offsets=[my_rank, 0])
            if dst != my_rank:
                pld.system.notify(
                    target=arrived, peer=dst, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd
                )

    # A deferred waiter must be a registration-only task. It cannot follow the
    # route counting and publication work in the producer task above.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="dispatch_meta_wait", deps=[prior_cycle_gate]
    ) as _meta_wait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(
                    signal=arrived, offsets=[src, 0], expected=moe_epoch, cmp=pld.WaitCmp.Ge
                )

    # Consume metadata only after both the local publish and every peer arrival
    # have completed. Keep _meta_tid as the downstream completion anchor.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dispatch_meta_finalize",
        deps=[_meta_push_tid, _meta_wait_tid],
        allow_early_resolve=True,
    ) as _meta_tid:
        # Cumsum recv_meta over sources -> per-expert receive count, which sizes
        # the routed-expert tile loop.
        for e in pl.range(N_LOCAL):
            acc = pl.const(0, pl.INT32)
            for src in pl.range(N_RANKS):
                count = pl.read(recv_meta, [src, e])
                pl.write(recv_meta_local, [src, e], count)
                acc = acc + count
            pl.write(recv_count_out, [e, 0], acc)

    # Move the bulk payload (x / aux / route) to each destination lane. One block
    # per (dst, local expert), each with its own slot counter; token-major order
    # matches the meta pass's per-(dst, loc_e) cumulative count.
    with pl.spmd(
        N_RANKS * N_LOCAL, name_hint="dispatch_push", allow_early_resolve=True, deps=[prior_cycle_gate]
    ) as push_tid:
        push_block = pl.tile.get_block_idx()
        dst = push_block // N_LOCAL
        loc_e = push_block - dst * N_LOCAL
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)

        e_lane_base = loc_e * RECV_MAX + my_rank * MAX_PER_SRC

        indices_tile = pl.tile.load(indices, [0, 0], [T, IDX_PAD], valid_shape=[T, TOPK])
        weights_tile = pl.tile.load(weights, [0, 0], [T, AUX_PAD], valid_shape=[T, TOPK])
        # This block's rows fill the contiguous lane (loc_e, my_rank, 0..n), so aux
        # and route ship as one remote_store. The lane tile always stores
        # MAX_PER_SRC rows; slots >= n are never read back.
        aux_lane = pl.tile.full([MAX_PER_SRC, AUX_PAD], dtype=pl.FP32, value=0.0)
        route_lane = pl.tile.full([MAX_PER_SRC, IDX_PAD], dtype=pl.INT32, value=0)
        slot_ctr = pl.array.create(1, pl.INT32)
        slot_ctr[0] = 0
        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.tile.read(indices_tile, [t, k])
                d = eid // N_LOCAL
                le = eid - d * N_LOCAL
                if le == loc_e:
                    if d == dst:
                        slot = slot_ctr[0]
                        slot_ctr[0] = slot + 1
                        # lane (loc_e, my_rank, slot) on peer=dst
                        row = e_lane_base + slot
                        pld.tensor.put(
                            dst=recv_x,
                            peer=dst,
                            src=x_norm_i8,
                            dst_offsets=[row, 0],
                            src_offsets=[t, 0],
                            shape=[1, D],
                        )
                        pl.tile.write(aux_lane, [slot, AUX_SCALE], pl.read(x_norm_scale, [t, 0]))
                        pl.tile.write(aux_lane, [slot, AUX_W], pl.tile.read(weights_tile, [t, k]))
                        pl.tile.write(route_lane, [slot, 0], pl.cast(t * TOPK + k, pl.INT32))
        pld.tile.remote_store(aux_lane, target=recv_aux, peer=dst, offsets=[e_lane_base, 0])
        pld.tile.remote_store(route_lane, target=recv_route, peer=dst, offsets=[e_lane_base, 0])

        # Payload-arrival notify folded into the push: a block signals its own
        # destination after its own puts, so a peer sees N_LOCAL notifies per
        # source per epoch and the wait below expects N_LOCAL * moe_epoch. The
        # count must bump only after this block's puts issue in program order:
        # recv_aux / recv_route ride a non-draining remote_store and a PIPE_ALL
        # barrier is not a cross-rank DDR fence (PTOAS#872).
        if dst != my_rank:
            pld.system.notify(
                target=data_arrived, peer=dst, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd
            )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_wait", deps=[push_tid]) as _wait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(
                    signal=data_arrived,
                    offsets=[src, 0],
                    expected=pl.cast(moe_epoch * N_LOCAL, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    # Gather lanes into the compact per-expert buffers: one task per expert
    # receive tile. deps on _wait_tid for the incoming payload; this rank's own
    # dst == my_rank puts are already ordered by the local RAW edges on
    # recv_x / recv_aux / recv_route. deps on _meta_tid for recv_meta_local, which is
    # manual_dep and so has no auto edge from the cumsum.
    # Scalar metadata stores need whole 64-byte lines per task: RECV_MAX and
    # RECV_TILE are both multiples of 16 FP32/INT32 elements, so an expert row
    # starts on a line and every tile boundary lands on one.
    gather_tids = pl.array.create(N_LOCAL, pl.TASK_ID)
    for e in pl.parallel(N_LOCAL):
        gather_tile_tids = pl.array.create(TILES_PER_EXPERT, pl.TASK_ID)
        expert_rows = pl.cast(pl.read(recv_count_out, [e, 0]), pl.INDEX)
        expert_tiles = (expert_rows + RECV_TILE - 1) // RECV_TILE
        for tile in pl.parallel(expert_tiles):
            tile_row = tile * RECV_TILE
            tile_end = pl.min(tile_row + RECV_TILE, expert_rows)
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="dispatch_gather",
                deps=[_wait_tid, _meta_tid],
                # Keep the routed expert tasks off the cores until this gather retires.
                allow_early_resolve=False,
            ) as gather_tile_tid:
                e_base_row = e * RECV_MAX
                source_begin = pl.cast(0, pl.INDEX)
                for src in pl.range(N_RANKS):
                    n = pl.cast(pl.read(recv_meta_local, [src, e]), pl.INDEX)
                    source_end = source_begin + n
                    src_base_row = e_base_row + src * MAX_PER_SRC
                    gather_begin = pl.max(tile_row, source_begin)
                    gather_end = pl.min(tile_end, source_end)
                    for out_col in pl.range(gather_begin, gather_end):
                        in_row = src_base_row + (out_col - source_begin)
                        out_row = e_base_row + out_col
                        recv_x_out_flat[out_row : out_row + 1, :] = recv_x[in_row : in_row + 1, :]
                        pl.write(recv_scale_out, [e, out_col], pl.read(recv_aux, [in_row, AUX_SCALE]))
                        pl.write(recv_w_out, [e, out_col], pl.read(recv_aux, [in_row, AUX_W]))
                        pl.write(recv_r_route_out, [e, out_col], pl.read(recv_route, [in_row, 0]))
                    source_begin = source_end
            gather_tile_tids[tile] = gather_tile_tid
        gather_tids[e] = pl.system.task_dummy(deps=[gather_tile_tids])

    _gather_tid = pl.system.task_dummy(deps=[_meta_tid, _wait_tid, gather_tids])
    return _gather_tid


@pl.jit.inline(auto_scope=False)
def combine(
    sh: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    routed_y_buf: pld.DistributedTensor[[T * TOPK, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    scatter_done: pl.Scalar[pl.TASK_ID],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    with pl.spmd(N_RANKS, name_hint="combine_notify", deps=[scatter_done]) as notify_tid:
        peer = pl.tile.get_block_idx()
        if peer != my_rank:
            pld.system.notify(
                target=combine_arrived,
                peer=peer,
                offsets=[my_rank, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="combine_wait", deps=[notify_tid]) as _cwait_tid:
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.defer_wait(
                    signal=combine_arrived,
                    offsets=[peer, 0],
                    expected=moe_epoch,
                    cmp=pld.WaitCmp.Ge,
                )

    # The notify task starts only after every local scatter. Waiting for every
    # peer therefore closes both the local and remote routed-output writes.
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)
    with pl.spmd(T // COMBINE_TOKEN_TILE, name_hint="shared_routed", deps=[_cwait_tid]) as _reduce_tid:
        t0 = pl.tile.get_block_idx() * COMBINE_TOKEN_TILE
        for t in pl.range(t0, t0 + COMBINE_TOKEN_TILE):
            if t < active_tokens:
                acc = pl.cast(sh[t : t + 1, :], target_type=pl.FP32)
                for k in pl.range(TOPK):
                    r = t * TOPK + k
                    acc = pl.add(acc, pl.cast(routed_y_buf[r : r + 1, :], target_type=pl.FP32))
                ffn_out[t : t + 1, :] = pl.cast(acc, target_type=pl.BF16, mode="rint")
            else:
                ffn_out[t : t + 1, :] = sh[t : t + 1, :]
    return _reduce_tid


@pl.jit.inline(auto_scope=False)
def _moe_tile(
    x_mixed: pl.Tensor[[T, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8, pl.NZ],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8, pl.NZ],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    prior_cycle_gate: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Run one fixed-capacity gate, expert dispatch, and combine tile."""
    x_norm_i8 = pl.create_tensor([T, D], dtype=pl.INT8)
    x_norm_scale = pl.create_tensor([T, 1], dtype=pl.FP32, manual_dep=True)
    indices = pl.create_tensor([T, TOPK], dtype=pl.INT32)
    weights = pl.create_tensor([T, TOPK], dtype=pl.FP32)
    gate(
        x_mixed,
        norm_w,
        gate_w,
        gate_bias,
        layer_id,
        num_tokens,
        tid2eid,
        input_ids,
        x_norm_i8,
        x_norm_scale,
        indices,
        weights,
    )

    shared_out = pl.create_tensor([T, D], dtype=pl.BF16)
    expert_shared(
        x_norm_i8,
        x_norm_scale,
        shared_w1,
        shared_w1_scale,
        shared_w3,
        shared_w3_scale,
        shared_w2,
        shared_w2_scale,
        shared_out,
    )

    recv_x_out = pl.create_tensor([N_LOCAL, RECV_MAX, D], dtype=pl.INT8)
    recv_scale_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.FP32, manual_dep=True)
    recv_w_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.FP32, manual_dep=True)
    recv_r_route_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.INT32, manual_dep=True)
    recv_count_out = pl.create_tensor([N_LOCAL, 1], dtype=pl.INT32)
    recv_meta_local = pl.create_tensor([N_RANKS, N_LOCAL], dtype=pl.INT32, manual_dep=True)
    dispatch_done = dispatch(
        indices,
        x_norm_i8,
        x_norm_scale,
        weights,
        recv_x_out,
        recv_scale_out,
        recv_w_out,
        recv_r_route_out,
        recv_count_out,
        recv_meta_local,
        recv_meta,
        recv_x,
        recv_aux,
        recv_route,
        arrived,
        data_arrived,
        num_tokens,
        my_rank,
        moe_epoch,
        prior_cycle_gate,
    )

    expert_completion_tids = pl.array.create(N_LOCAL, pl.TASK_ID)
    for local_e in pl.parallel(N_LOCAL):
        tile_completion_tids = pl.array.create(TILES_PER_EXPERT, pl.TASK_ID)
        expert_rows = pl.cast(pl.read(recv_count_out, [local_e, 0]), pl.INDEX)
        expert_tiles = (expert_rows + RECV_TILE - 1) // RECV_TILE
        for tile in pl.parallel(expert_tiles):
            tile_row = tile * RECV_TILE
            valid_rows = pl.min(RECV_TILE, expert_rows - tile_row)
            recv_y_tile = pl.create_tensor([RECV_TILE, D], dtype=pl.BF16)
            tile_ready = expert_routed_tile(
                recv_x_out,
                recv_scale_out,
                recv_w_out,
                routed_w1,
                routed_w1_scale,
                routed_w3,
                routed_w3_scale,
                routed_w2,
                routed_w2_scale,
                recv_y_tile,
                local_e,
                tile_row,
                valid_rows,
                dispatch_done,
            )
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="expert_tile_scatter",
                deps=[tile_ready],
                # Route rows are unique within an epoch, and the completion
                # fan-in below already orders combine after every scatter.
                no_dep_args=[routed_y_buf],
            ) as scatter_tid:
                tile_end = tile_row + valid_rows
                source_begin = pl.cast(0, pl.INDEX)
                for src in pl.range(N_RANKS):
                    source_rows = pl.cast(pl.read(recv_meta_local, [src, local_e]), pl.INDEX)
                    source_end = source_begin + source_rows
                    scatter_begin = pl.max(tile_row, source_begin)
                    scatter_end = pl.min(tile_end, source_end)
                    for compact_row in pl.range(scatter_begin, scatter_end):
                        route = pl.cast(pl.read(recv_r_route_out, [local_e, compact_row]), pl.INDEX)
                        pld.tensor.put(
                            dst=routed_y_buf,
                            peer=src,
                            src=recv_y_tile,
                            dst_offsets=[route, 0],
                            src_offsets=[compact_row - tile_row, 0],
                            shape=[1, D],
                        )
                    source_begin = source_end
            tile_completion_tids[tile] = scatter_tid
        expert_tiles_done = pl.system.task_dummy(deps=[tile_completion_tids])
        expert_completion_tids[local_e] = expert_tiles_done

    scatter_done = pl.system.task_dummy(deps=[dispatch_done, expert_completion_tids])

    completion_tid = combine(
        shared_out,
        ffn_out,
        routed_y_buf,
        combine_arrived,
        scatter_done,
        num_tokens,
        my_rank,
        moe_epoch,
    )
    return completion_tid


@pl.jit.inline(auto_scope=False)
def moe(
    # model inputs
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8, pl.NZ],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8, pl.NZ],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    # final output
    x_next: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    # windows
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    # scalars last: runtime TaskArgs forbids a tensor arg after a scalar arg.
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    # 1-based MoE call id for the shared flag windows (distinct from layer_id).
    moe_epoch: pl.Scalar[pl.INT32],
    prior_cycle_gate: pl.Scalar[pl.TASK_ID],
) -> pl.Tensor[[T, HC_MULT, D], pl.FP32]:
    # Non-output intermediates allocate locally, in their producer's scope.
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post_ffn = pl.create_tensor([T, HC_MULT], dtype=pl.FP32, manual_dep=True)
    comb_ffn = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, x_mixed, post_ffn, comb_ffn)

    ffn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    with pl.scope():
        _moe_tile(
            x_mixed,
            norm_w,
            gate_w,
            gate_bias,
            tid2eid,
            input_ids,
            routed_w1,
            routed_w1_scale,
            routed_w3,
            routed_w3_scale,
            routed_w2,
            routed_w2_scale,
            shared_w1,
            shared_w1_scale,
            shared_w3,
            shared_w3_scale,
            shared_w2,
            shared_w2_scale,
            ffn_out,
            recv_meta,
            recv_x,
            recv_aux,
            recv_route,
            arrived,
            data_arrived,
            routed_y_buf,
            combine_arrived,
            layer_id,
            num_tokens,
            my_rank,
            moe_epoch,
            prior_cycle_gate,
        )
        hc_post(ffn_out, x_hc, post_ffn, comb_ffn, x_next)
    return x_next
