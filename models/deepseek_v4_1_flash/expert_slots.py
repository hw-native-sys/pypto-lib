# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""EP1 single-token MXFP4 supply and local combine for the six routed experts."""

from pathlib import Path

import pypto.language as pl

from config import MOE_TOKENS, N_LOCAL_EXPERTS
from expert_routed import (
    D, H_SCALE, K_SCALE, MOE_INTER, MX_M_TILE, MX_MM_INTER_TILE, MX_W1_PACKED_ROWS,
    MX_W2_D_OUT_TILE, MX_W2_PACKED_ROWS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS,
    RECV_MAX, RECV_SCALE_TMP_TILE,
)
from offload_expert_routed import _expert_routed_output
from expert_supply import SCALE_WORKERS
from models.common.expert_supply import supply_packed_fields


@pl.jit.inline(auto_scope=False)
def gather_single_token_inputs(
    indices: pl.Tensor, payload: pl.Tensor, packed_scale: pl.Tensor, weights: pl.Tensor,
    x_out: pl.Tensor, scale_out: pl.Tensor, weights_out: pl.Tensor, previous_use: pl.Scalar[pl.TASK_ID], topk: pl.constexpr,
):
    x_flat = pl.reshape(x_out, [N_LOCAL_EXPERTS * RECV_MAX, D])
    with pl.spmd(topk, name_hint="dispatch_single_token", deps=[previous_use]) as ready:
        k = pl.tile.get_block_idx()
        ids = pl.load(indices, [0, 0], [1, 8], valid_shape=[1, topk])
        expert = pl.cast(pl.tile.read(ids, [0, k]), pl.INDEX)
        value = pl.load(payload, [0, 0], [1, D])
        x_flat = pl.store(value, [expert * RECV_MAX, 0], x_flat)

        packed = pl.load(packed_scale, [0, 0], [1, 16 * K_SCALE])
        packed_bytes = pl.reinterpret_view(packed, pl.UINT8)
        pairs = pl.reinterpret_view(packed_bytes, pl.UINT16, shape=[K_SCALE // 2, 16])
        pairs_nd = pl.transpose(pairs, 0, 1)
        scale_bytes = pl.reinterpret_view(pairs_nd, pl.UINT8, shape=[16, K_SCALE])
        scale_row = pl.tile.slice(scale_bytes, [1, K_SCALE], [0, 0])
        # Keep the generic gather's zero scale codes for inactive rows.
        scale_values = pl.tile.full([16, K_SCALE], dtype=pl.UINT8, value=0)
        scale_values = pl.tile.assemble(scale_values, scale_row, [0, 0])
        temporary = pl.create_tile([1, RECV_SCALE_TMP_TILE], dtype=pl.UINT8)
        repacked = pl.tmov_x2zz(scale_values, temporary, group_axis=1, dst_rows=16, dst_cols=K_SCALE)
        scale_mx = pl.reinterpret_view(repacked, pl.FP8E8M0)
        scale_flat = pl.reshape(scale_mx, [1, 16 * K_SCALE])
        scale_out = pl.store(scale_flat, [0, expert * RECV_MAX * K_SCALE], scale_out)

        route_weights = pl.load(weights, [0, 0], [1, 8], valid_shape=[1, topk])
        weight = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0)
        pl.tile.write(weight, [0, 0], pl.tile.read(route_weights, [0, k]))
        weight_valid = pl.set_validshape(weight, 1, 1)
        weights_out = pl.store(weight_valid, [expert, 0], weights_out)
    return ready


@pl.jit.extern(core_type="aiv", source=Path(__file__).with_name("expert_single_token_plan.cpp"),
               include_dirs=(Path(__file__).parent.parent / "common",))
def plan_single_token_slots(
    indices: pl.Tensor, state: pl.InOut[pl.Tensor],
    slots: pl.Out[pl.Tensor], misses: pl.Out[pl.Tensor],
    layer: pl.Scalar[pl.INDEX], first: pl.Scalar[pl.INDEX], stop: pl.Scalar[pl.INDEX],
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
    """Plan the first token's six unique experts in ascending expert order."""
    return state, slots, misses


@pl.jit.inline(auto_scope=False)
def compute_single_token_slots(
    recv_x: pl.Tensor, recv_scale: pl.Tensor, recv_weights: pl.Tensor,
    counts: pl.Tensor, indices: pl.Tensor, state: pl.Tensor, descriptors: pl.Tensor,
    w1: pl.Tensor, s1: pl.Tensor, w2: pl.Tensor, s2: pl.Tensor, w3: pl.Tensor, s3: pl.Tensor,
    routed_rows: pl.Tensor, layer: pl.Scalar[pl.INDEX],
    counts_ready: pl.Scalar[pl.TASK_ID], input_ready: pl.Scalar[pl.TASK_ID], topk: pl.constexpr,
):
    slots = pl.create_tensor([N_LOCAL_EXPERTS], dtype=pl.INT32)
    misses = pl.create_tensor([N_LOCAL_EXPERTS], dtype=pl.INT32)
    (state, slots, misses), planned = pl.submit(
        plan_single_token_slots, indices, state, slots, misses, layer, 0, topk, deps=[counts_ready])
    recv_flat = pl.reshape(recv_x, [N_LOCAL_EXPERTS * RECV_MAX, D])
    weights = pl.reshape(recv_weights, [N_LOCAL_EXPERTS * RECV_MAX, 1])
    completed = pl.array.create(topk, pl.TASK_ID)
    for k in pl.range(topk):
        expert = pl.cast(pl.read(indices, [0, k]), pl.INDEX)
        slot = pl.cast(pl.read(slots, [expert]), pl.INDEX)
        row = expert * RECV_MAX
        a = pl.slice(recv_flat, [MX_M_TILE, D], [row, 0])
        a_scale = pl.slice(recv_scale, [1, MX_M_TILE * K_SCALE], [0, row * K_SCALE])
        weight = pl.slice(weights, [1, 1], [row, 0])
        count = pl.slice(counts, [1, 1], [expert, 0])
        gate = pl.slice(w1, [1, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], [slot, 0, 0])
        gate_scale = pl.slice(s1, [K_SCALE, MOE_INTER], [slot * K_SCALE, 0])
        down = pl.slice(w2, [1, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], [slot, 0, 0])
        down_scale = pl.slice(s2, [H_SCALE, D], [slot * H_SCALE, 0])
        up = pl.slice(w3, [1, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], [slot, 0, 0])
        up_scale = pl.slice(s3, [K_SCALE, MOE_INTER], [slot * K_SCALE, 0])
        gate_bank, gate_mx, down_bank, down_mx, up_bank, up_mx = gate, gate_scale, down, down_scale, up, up_scale
        available = planned
        if pl.read(misses, [expert]) >= 0:
            gate_bank, gate_mx, down_bank, down_mx, up_bank, up_mx, available = supply_packed_fields(
                descriptors, misses, gate, gate_scale, down, down_scale, up, up_scale,
                layer, expert, -1, slot, planned, 4, SCALE_WORKERS)
        compute_ready = pl.system.task_dummy(deps=[available, input_ready])
        output = pl.slice(routed_rows, [MX_M_TILE, D], [k * MX_M_TILE, 0])
        output, projected = _expert_routed_output(
            a, a_scale, weight, count, gate_bank, gate_mx, up_bank, up_mx, down_bank, down_mx, output, compute_ready,
            1, MX_M_TILE, D, MOE_INTER, 1, slots, False,
            2 * (MOE_INTER // MX_MM_INTER_TILE), (MOE_INTER + 767) // 768, D // MX_W2_D_OUT_TILE)
        completed[k] = projected
    return routed_rows, pl.system.task_dummy(deps=[completed[k] for k in range(topk)])


@pl.jit.inline(auto_scope=False)
def combine_single_token_slots(
    shared_output: pl.Tensor, routed_rows: pl.Tensor, ffn_out: pl.Tensor,
    ready: pl.Scalar[pl.TASK_ID], topk: pl.constexpr,
):
    with pl.spmd(D // 128, name_hint="combine_single_token", deps=[ready]) as combined:
        column = pl.tile.get_block_idx() * 128
        shared = pl.load(shared_output, [0, column], [1, 128])
        acc = pl.cast(shared, pl.FP32)
        for k in pl.pipeline(topk, stage=2):
            value = pl.load(routed_rows, [k * 16, column], [1, 128])
            acc = pl.add(acc, pl.cast(value, pl.FP32))
        result = pl.cast(acc, pl.BF16, mode="rint")
        ffn_out = pl.store(result, [0, column], ffn_out)
        # Preserve the generic combine's zero padding for the inactive rows.
        zero = pl.tile.full([16, 128], dtype=pl.BF16, value=0.0)
        for row in pl.range(1, MOE_TOKENS, 16):
            rows = pl.min(16, MOE_TOKENS - row)
            padding = pl.set_validshape(zero, rows, 128)
            ffn_out = pl.store(padding, [row, column], ffn_out)
    return combined
