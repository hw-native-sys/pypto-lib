# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill W8A8 expert projections with fused SwiGLU and activation quantization."""

import pypto.language as pl

from .activation_quant import VEC_M_TILE, _prefill_quant_params
from .config import FLASH as M


D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit

PREFILL_GATE_UP_N_TILE = 128
PREFILL_GATE_UP_K_TILE = 512
PREFILL_ACT_TILE = 256
PREFILL_QUANT_TILE = 768
PREFILL_DOWN_N_TILE = 256
PREFILL_DOWN_K_TILE = 256
PREFILL_DOWN_ACT_TILE = 512
PREFILL_ROW_TILE = 16
PREFILL_FUSED_MAX_ROWS = 128


@pl.jit.inline(auto_scope=False)
def _prefill_gate_up_quant_tile(
    source: pl.Tensor,
    source_scale: pl.Tensor,
    route: pl.Tensor,
    routed_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[1, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[1, MOE_INTER], pl.FP32],
    payload: pl.Tensor,
    scale: pl.Tensor,
    tile_row_start: pl.Scalar[pl.INDEX],
    valid_rows: pl.Scalar[pl.INDEX],
    row_tile: pl.constexpr,
    weighted: pl.constexpr,
):
    """Project gate/up and quantize one aligned Prefill expert tile."""
    w13 = pl.create_tensor([row_tile, 2 * MOE_INTER], dtype=pl.INT32)
    gate_up_tiles = 2 * MOE_INTER // PREFILL_GATE_UP_N_TILE
    gate_up_workers = pl.max(1, pl.min(gate_up_tiles, pl.system.available_cluster_count()))
    with pl.spmd(gate_up_workers,
                 name_hint="prefill_expert_gate_up") as mm_ready:
        gate_up_worker = pl.tile.get_block_idx()
        for block in pl.range(gate_up_worker, gate_up_tiles, gate_up_workers):
            col = block * PREFILL_GATE_UP_N_TILE
            acc_seed = pl.tile.create(
                [row_tile, PREFILL_GATE_UP_N_TILE],
                dtype=pl.INT32,
                target_memory=pl.MemorySpace.Acc,
                compact=True,
            )
            acc = pl.set_validshape(acc_seed, row_tile, PREFILL_GATE_UP_N_TILE)
            if col < MOE_INTER:
                for k in pl.pipeline(0, D, PREFILL_GATE_UP_K_TILE, stage=2):
                    source_tile = pl.load(
                        source, [tile_row_start, k], [row_tile, PREFILL_GATE_UP_K_TILE],
                    )
                    weight_tile = pl.load(
                        routed_w1, [col, k], [PREFILL_GATE_UP_N_TILE, PREFILL_GATE_UP_K_TILE],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    acc = pl.matmul_acc(
                        acc, source_tile, pl.tile.transpose_view(weight_tile), init_cond=(k == 0),
                    )
            else:
                up_col = col - MOE_INTER
                for k in pl.pipeline(0, D, PREFILL_GATE_UP_K_TILE, stage=2):
                    source_tile = pl.load(
                        source, [tile_row_start, k], [row_tile, PREFILL_GATE_UP_K_TILE],
                    )
                    weight_tile = pl.load(
                        routed_w3, [up_col, k], [PREFILL_GATE_UP_N_TILE, PREFILL_GATE_UP_K_TILE],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    acc = pl.matmul_acc(
                        acc, source_tile, pl.tile.transpose_view(weight_tile), init_cond=(k == 0),
                    )
            w13 = pl.store(acc, [0, col], w13)

    gate_quant_tiles = row_tile // VEC_M_TILE
    gate_quant_workers = pl.max(1, pl.min(gate_quant_tiles, pl.system.available_aiv_count()))
    with pl.spmd(gate_quant_workers,
                 name_hint="prefill_expert_gate_up_quant", deps=[mm_ready]):
        gate_quant_worker = pl.tile.get_block_idx()
        for gate_quant_tile in pl.range(gate_quant_worker, gate_quant_tiles, gate_quant_workers):
            row = gate_quant_tile * VEC_M_TILE
            valid = pl.max(0, pl.min(VEC_M_TILE, valid_rows - row))
            x_scale = pl.load(source_scale, [tile_row_start + row, 0], [VEC_M_TILE, 1])
            if weighted:
                route_row = pl.load(route, [0, tile_row_start + row], [1, VEC_M_TILE])
                route_weight = pl.reshape(route_row, [VEC_M_TILE, 1])
            cached = pl.create_tile([VEC_M_TILE, MOE_INTER], dtype=pl.BF16)
            row_amax = pl.tile.full([1, VEC_M_TILE], dtype=pl.FP32, value=1e-30)
            row_max_scratch = pl.create_tile([VEC_M_TILE, PREFILL_ACT_TILE], dtype=pl.FP32)
            for inter in pl.range(0, MOE_INTER, PREFILL_ACT_TILE):
                gate_i32 = pl.load(w13, [row, inter], [VEC_M_TILE, PREFILL_ACT_TILE])
                up_i32 = pl.load(w13, [row, MOE_INTER + inter], [VEC_M_TILE, PREFILL_ACT_TILE])
                gate_scale = pl.load(routed_w1_scale, [0, inter], [1, PREFILL_ACT_TILE])
                up_scale = pl.load(routed_w3_scale, [0, inter], [1, PREFILL_ACT_TILE])
                gate_fp32 = pl.cast(gate_i32, pl.FP32)
                gate_fp32 = pl.row_expand_mul(gate_fp32, x_scale)
                gate_fp32 = pl.col_expand_mul(gate_fp32, gate_scale)
                up_fp32 = pl.cast(up_i32, pl.FP32)
                up_fp32 = pl.row_expand_mul(up_fp32, x_scale)
                up_fp32 = pl.col_expand_mul(up_fp32, up_scale)
                gate_bf16 = pl.cast(gate_fp32, pl.BF16, mode="rint")
                gate_value = pl.minimum(pl.cast(gate_bf16, pl.FP32), SWIGLU_LIMIT)
                up_bf16 = pl.cast(up_fp32, pl.BF16, mode="rint")
                up_value = pl.maximum(pl.cast(up_bf16, pl.FP32), -SWIGLU_LIMIT)
                up_value = pl.minimum(up_value, SWIGLU_LIMIT)
                neg_gate = pl.neg(gate_value)
                sigmoid = pl.recip(pl.add(pl.exp(neg_gate), 1.0))
                silu = pl.mul(gate_value, sigmoid)
                value = pl.mul(silu, up_value)
                if weighted:
                    value = pl.row_expand_mul(value, route_weight)
                value_bf16 = pl.cast(value, pl.BF16, mode="rint")
                cached[:, inter : inter + PREFILL_ACT_TILE] = value_bf16
                magnitude = pl.abs(pl.cast(value_bf16, pl.FP32))
                panel_max = pl.row_max(magnitude, tmp_tile=row_max_scratch)
                row_amax = pl.maximum(row_amax, pl.reshape(panel_max, [1, VEC_M_TILE]))
            factor, inverse = _prefill_quant_params(pl.reshape(row_amax, [VEC_M_TILE, 1]))
            pl.store(pl.set_validshape(factor, valid, 1), [tile_row_start + row, 0], scale)
            for inter in pl.range(0, MOE_INTER, PREFILL_QUANT_TILE):
                panel = pl.tile.slice(cached, [VEC_M_TILE, PREFILL_QUANT_TILE], [0, inter])
                quant_value = pl.row_expand_mul(pl.cast(panel, pl.FP32), inverse)
                integer = pl.cast(quant_value, pl.INT16, mode="rint")
                exact_fp16 = pl.cast(integer, pl.FP16, mode="rint")
                codes = pl.cast(exact_fp16, pl.INT8, mode="trunc")
                pl.store(pl.set_validshape(codes, valid, PREFILL_QUANT_TILE), [tile_row_start + row, inter], payload)
    return payload, scale


@pl.jit.inline(auto_scope=False)
def _prefill_down_proj_tile(
    payload: pl.Tensor,
    activation_scale: pl.Tensor,
    weight: pl.Tensor[[D, MOE_INTER], pl.INT8],
    weight_scale: pl.Tensor[[1, D], pl.FP32],
    output: pl.Tensor,
    tile_row_start: pl.Scalar[pl.INDEX],
    valid_rows: pl.Scalar[pl.INDEX],
    row_tile: pl.constexpr,
):
    """Project one quantized activation tile back to the hidden width."""
    y_i32 = pl.create_tensor([row_tile, D], dtype=pl.INT32)
    down_tiles = D // PREFILL_DOWN_N_TILE
    down_workers = pl.max(1, pl.min(down_tiles, pl.system.available_cluster_count()))
    with pl.spmd(down_workers,
                 name_hint="prefill_expert_down", allow_early_resolve=True) as down_tid:
        down_worker = pl.tile.get_block_idx()
        for down_tile in pl.range(down_worker, down_tiles, down_workers):
            d_base = down_tile * PREFILL_DOWN_N_TILE
            y_seed = pl.tile.create(
                [row_tile, PREFILL_DOWN_N_TILE],
                dtype=pl.INT32,
                target_memory=pl.MemorySpace.Acc,
                compact=True,
            )
            y_acc = pl.set_validshape(y_seed, row_tile, PREFILL_DOWN_N_TILE)
            for k in pl.pipeline(0, MOE_INTER, PREFILL_DOWN_K_TILE, stage=2):
                h_tile = pl.load(
                    payload, [tile_row_start, k], [row_tile, PREFILL_DOWN_K_TILE],
                )
                w_tile = pl.load(
                    weight, [d_base, k], [PREFILL_DOWN_N_TILE, PREFILL_DOWN_K_TILE],
                    target_memory=pl.MemorySpace.Mat,
                )
                y_acc = pl.matmul_acc(
                    y_acc, h_tile, pl.tile.transpose_view(w_tile), init_cond=(k == 0),
                )
            y_i32 = pl.store(y_acc, [0, d_base], y_i32)
    blocks = D // PREFILL_DOWN_ACT_TILE
    down_scale_tiles = row_tile // PREFILL_ROW_TILE * blocks
    down_scale_workers = pl.max(1, pl.min(down_scale_tiles, pl.system.available_aiv_count()))
    with pl.spmd(down_scale_workers,
                 name_hint="prefill_expert_down_scale", deps=[down_tid]):
        down_scale_worker = pl.tile.get_block_idx()
        for block in pl.range(down_scale_worker, down_scale_tiles, down_scale_workers):
            row = block // blocks * PREFILL_ROW_TILE
            d_base = block % blocks * PREFILL_DOWN_ACT_TILE
            valid = pl.max(0, pl.min(PREFILL_ROW_TILE, valid_rows - row))
            value = pl.cast(
                pl.load(
                    y_i32, [row, d_base], [PREFILL_ROW_TILE, PREFILL_DOWN_ACT_TILE],
                ),
                pl.FP32,
            )
            value = pl.row_expand_mul(
                value,
                pl.load(
                    activation_scale, [tile_row_start + row, 0], [PREFILL_ROW_TILE, 1],
                    valid_shape=[valid, 1],
                ),
            )
            value = pl.col_expand_mul(
                value,
                pl.load(weight_scale, [0, d_base], [1, PREFILL_DOWN_ACT_TILE]),
            )
            value = pl.cast(value, pl.BF16, mode="rint")
            pl.store(
                pl.set_validshape(value, valid, PREFILL_DOWN_ACT_TILE),
                [tile_row_start + row, d_base], output,
            )
    return output


@pl.jit.inline(auto_scope=False)
def _prefill_expert_single_tile(
    source: pl.Tensor,
    source_scale: pl.Tensor,
    route_weights: pl.Tensor,
    w1: pl.Tensor,
    scale1: pl.Tensor,
    w3: pl.Tensor,
    scale3: pl.Tensor,
    w2: pl.Tensor,
    scale2: pl.Tensor,
    output: pl.Tensor,
    valid_rows: pl.Scalar[pl.INDEX],
    row_tile: pl.constexpr,
    weighted: pl.constexpr,
):
    """Project and quantize a Prefill expert with one aligned row tile."""
    hidden = pl.create_tensor([row_tile, MOE_INTER], dtype=pl.INT8)
    # The payload and the scale are written by the same tile task.
    hidden_scale = pl.create_tensor([row_tile, 1], dtype=pl.FP32, manual_dep=True)
    _prefill_gate_up_quant_tile(
        source, source_scale, route_weights,
        w1, scale1, w3, scale3,
        hidden, hidden_scale,
        0, valid_rows, row_tile, weighted,
    )
    _prefill_down_proj_tile(hidden, hidden_scale, w2, scale2, output, 0, valid_rows, row_tile)
    return output


@pl.jit.inline(auto_scope=False)
def prefill_expert_compute(
    source: pl.Tensor,
    source_scale: pl.Tensor,
    route_weights: pl.Tensor,
    w1: pl.Tensor,
    scale1: pl.Tensor,
    w3: pl.Tensor,
    scale3: pl.Tensor,
    w2: pl.Tensor,
    scale2: pl.Tensor,
    output: pl.Tensor,
    count: pl.Scalar[pl.INT32],
    weighted: pl.constexpr,
):
    """Dispatch a nonempty Prefill expert of at most 128 rows."""
    valid_rows = pl.cast(count, pl.INDEX)
    if count <= PREFILL_ROW_TILE:
        _prefill_expert_single_tile(
            source, source_scale, route_weights,
            w1, scale1, w3, scale3, w2, scale2,
            output, valid_rows, PREFILL_ROW_TILE, weighted,
        )
    elif count <= 32:
        _prefill_expert_single_tile(
            source, source_scale, route_weights,
            w1, scale1, w3, scale3, w2, scale2,
            output, valid_rows, 32, weighted,
        )
    elif count <= 48:
        _prefill_expert_single_tile(
            source, source_scale, route_weights,
            w1, scale1, w3, scale3, w2, scale2,
            output, valid_rows, 48, weighted,
        )
    elif count <= 64:
        _prefill_expert_single_tile(
            source, source_scale, route_weights,
            w1, scale1, w3, scale3, w2, scale2,
            output, valid_rows, 64, weighted,
        )
    elif count <= 80:
        _prefill_expert_single_tile(
            source, source_scale, route_weights,
            w1, scale1, w3, scale3, w2, scale2,
            output, valid_rows, 80, weighted,
        )
    elif count <= 96:
        _prefill_expert_single_tile(
            source, source_scale, route_weights,
            w1, scale1, w3, scale3, w2, scale2,
            output, valid_rows, 96, weighted,
        )
    elif count <= 112:
        _prefill_expert_single_tile(
            source, source_scale, route_weights,
            w1, scale1, w3, scale3, w2, scale2,
            output, valid_rows, 112, weighted,
        )
    else:
        _prefill_expert_single_tile(
            source, source_scale, route_weights,
            w1, scale1, w3, scale3, w2, scale2,
            output, valid_rows, PREFILL_FUSED_MAX_ROWS, weighted,
        )
    return output
