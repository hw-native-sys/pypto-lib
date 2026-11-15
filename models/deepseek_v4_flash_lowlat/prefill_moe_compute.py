# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill MoE preparation, bounded route tiles and ordered per-token combine."""

import pypto.language as pl
from .hc_pre import hc_pre
from .prefill_moe_gate import gate, ROUTE_ROW_PAD
from .prefill_moe_expert import expert_routed_persistent_planned
from .prefill_moe_geometry import (
    D, HC_DIM, HC_MULT, IDX_PAD, MIX_HC, MOE_INTER, N_BANK, N_EXPERTS, N_ROUTES,
    N_SLOTS_B, RECV_MAX, SHARED_EID, SHARED_TILES, SH_SLOT, T, TOPK, VOCAB, WORK_SLOTS_PAD,
)


@pl.jit.inline
def route_group(
    indices: pl.Tensor[[T, ROUTE_ROW_PAD], pl.INT32],
    weights: pl.Tensor[[T, ROUTE_ROW_PAD], pl.FP32],
    x_norm_i8: pl.Tensor[[T, D], pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1], pl.FP32],

    recv_x: pl.Tensor[[N_SLOTS_B, RECV_MAX, D], pl.INT8],
    recv_scale: pl.Tensor[[N_SLOTS_B, RECV_MAX], pl.FP32],
    recv_w: pl.Tensor[[N_SLOTS_B, RECV_MAX], pl.FP32],
    work_slot: pl.Tensor[[WORK_SLOTS_PAD], pl.INT32],
    work_rows: pl.Tensor[[WORK_SLOTS_PAD], pl.INT32],
    slot_expert: pl.Tensor[[N_SLOTS_B, IDX_PAD], pl.INT32],
    route_slot_row: pl.Tensor[[N_ROUTES, IDX_PAD], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Group routes into bounded sixteen-row expert tiles and shared tiles."""
    recv_x_flat = pl.reshape(recv_x, [N_SLOTS_B * RECV_MAX, D])
    expert_slot = pl.create_tensor([1, N_EXPERTS], dtype=pl.INT32, manual_dep=True)
    work_slot_2d = pl.reshape(work_slot, [1, WORK_SLOTS_PAD])
    work_rows_2d = pl.reshape(work_rows, [1, WORK_SLOTS_PAD])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="route_table_init", allow_early_resolve=True) as init_tid:
        expert_slot[:, :] = pl.full([1, N_EXPERTS], dtype=pl.INT32, value=-1)
        work_slot_2d[:, :] = pl.full([1, WORK_SLOTS_PAD], dtype=pl.INT32, value=0)
        work_rows_2d[:, :] = pl.full([1, WORK_SLOTS_PAD], dtype=pl.INT32, value=0)
        for slot_t0 in pl.range(0, N_SLOTS_B, 64):
            slot_rows = pl.min(64, N_SLOTS_B - slot_t0)
            zero_experts = pl.tile.full([64, IDX_PAD], dtype=pl.INT32, value=0)
            valid_experts = pl.set_validshape(zero_experts, slot_rows, IDX_PAD)
            pl.store(valid_experts, [slot_t0, 0], slot_expert)
            zero_scales = pl.tile.full([64, RECV_MAX], dtype=pl.FP32, value=0.0)
            valid_scales = pl.set_validshape(zero_scales, slot_rows, RECV_MAX)
            pl.store(valid_scales, [slot_t0, 0], recv_scale)
            zero_weights = pl.tile.full([64, RECV_MAX], dtype=pl.FP32, value=0.0)
            valid_weights = pl.set_validshape(zero_weights, slot_rows, RECV_MAX)
            pl.store(valid_weights, [slot_t0, 0], recv_w)
        for route_t0 in pl.range(0, N_ROUTES, 64):
            route_rows = pl.min(64, N_ROUTES - route_t0)
            zero_routes = pl.tile.full([64, IDX_PAD], dtype=pl.INT32, value=0)
            valid_routes = pl.set_validshape(zero_routes, route_rows, IDX_PAD)
            pl.store(valid_routes, [route_t0, 0], route_slot_row)

        slot_expert[SH_SLOT : SH_SLOT + SHARED_TILES, :] = pl.full(
            [SHARED_TILES, IDX_PAD], dtype=pl.INT32, value=SHARED_EID)

    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)


    with pl.spmd(1, name_hint="route_group_scalar", allow_early_resolve=True, deps=[init_tid]) as scalar_tid:
        scalar_core = pl.tile.get_block_idx()
        next_slot = pl.cast(0, pl.INDEX)
        for token in pl.range(scalar_core, active_tokens):
            token_scale = pl.read(x_norm_scale, [token, 0])
            for k in pl.range(TOPK):
                expert_id = pl.read(indices, [token, k])
                expert_col = pl.cast(expert_id, pl.INDEX)
                slot = pl.read(expert_slot, [0, expert_col])
                if slot >= 0:
                    occupied = pl.read(work_rows, [pl.cast(slot, pl.INDEX)])
                    if occupied >= RECV_MAX:
                        slot = pl.const(-1, pl.INT32)
                if slot < 0:
                    slot = pl.cast(next_slot, pl.INT32)
                    pl.write(expert_slot, [0, expert_col], slot)
                    pl.write(slot_expert, [next_slot, 0], expert_id)
                    pl.write(work_slot, [next_slot], slot)
                    next_slot = next_slot + 1
                slot_row = pl.cast(slot, pl.INDEX)
                count = pl.read(work_rows, [slot_row])
                count_col = pl.cast(count, pl.INDEX)
                pl.write(work_rows, [slot_row], pl.cast(count + 1, pl.INT32))
                pl.write(recv_scale, [slot_row, count_col], token_scale)
                pl.write(recv_w, [slot_row, count_col], pl.read(weights, [token, k]))
                route = token * TOPK + k
                pl.write(route_slot_row, [route, 0], pl.cast(slot * RECV_MAX + count, pl.INT32))

            shared_slot = SH_SLOT + token // RECV_MAX
            shared_row = token % RECV_MAX
            pl.write(recv_scale, [shared_slot, shared_row], token_scale)
            pl.write(recv_w, [shared_slot, shared_row], pl.cast(1.0, pl.FP32))
        for shared_tile in pl.range(SHARED_TILES):
            shared_count = pl.min(RECV_MAX, pl.max(active_tokens - shared_tile * RECV_MAX, 0))
            pl.write(work_slot, [next_slot + shared_tile], pl.cast(SH_SLOT + shared_tile, pl.INT32))
            pl.write(work_rows, [next_slot + shared_tile], pl.cast(shared_count, pl.INT32))


    with pl.spmd(T + 1, name_hint="route_gather", allow_early_resolve=True, deps=[scalar_tid]) as _gather_tid:
        blk = pl.tile.get_block_idx()
        if blk < T:
            if blk < active_tokens:
                for k in pl.range(TOPK):
                    dst_row = pl.cast(pl.read(route_slot_row, [blk * TOPK + k, 0]), pl.INDEX)
                    recv_x_flat[dst_row : dst_row + 1, :] = x_norm_i8[blk : blk + 1, :]
        else:


            sh_base = pl.cast(SH_SLOT * RECV_MAX, pl.INDEX)
            for shared_t0 in pl.range(0, T, 8):
                recv_x_flat[sh_base + shared_t0 : sh_base + shared_t0 + 8, :] = x_norm_i8[shared_t0 : shared_t0 + 8, :]


@pl.jit.inline
def combine_local(
    recv_y: pl.Tensor[[N_SLOTS_B, RECV_MAX, D], pl.BF16],
    route_slot_row: pl.Tensor[[N_ROUTES, IDX_PAD], pl.INT32],
    ffn_partial: pl.Tensor[[T, D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """ffn_partial[t] = recv_y[SH_SLOT, t] + Sigma_k recv_y[route_slot_row[t * TOPK + k]].

    The shared expert's row is just the first summand: it came out of the same
    kernel, in the same layout, so it needs no separate operand."""
    recv_y_flat = pl.reshape(recv_y, [N_SLOTS_B * RECV_MAX, D])

    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)

    with pl.spmd(T, name_hint="shared_routed", allow_early_resolve=True):
        t = pl.tile.get_block_idx()
        if t < active_tokens:
            sh_row = pl.cast(SH_SLOT * RECV_MAX + t, pl.INDEX)
            acc = pl.cast(recv_y_flat[sh_row : sh_row + 1, :], target_type=pl.FP32)
            for k in pl.range(TOPK):
                src_row = pl.cast(pl.read(route_slot_row, [t * TOPK + k, 0]), pl.INDEX)
                acc = pl.add(acc, pl.cast(recv_y_flat[src_row : src_row + 1, :], target_type=pl.FP32))
            ffn_partial[t : t + 1, :] = acc
        else:
            ffn_partial[t : t + 1, :] = pl.full([1, D], dtype=pl.FP32, value=0.0)


@pl.jit.inline(auto_scope=False)
def prepare_moe(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post_ffn = pl.create_tensor([T, HC_MULT], dtype=pl.FP32, manual_dep=True)
    comb_ffn = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(
        x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        x_mixed, post_ffn, comb_ffn,
    )

    x_norm_i8 = pl.create_tensor([T, D], dtype=pl.INT8)
    x_norm_scale = pl.create_tensor([T, 1], dtype=pl.FP32, manual_dep=True)
    indices = pl.create_tensor([T, ROUTE_ROW_PAD], dtype=pl.INT32)
    weights = pl.create_tensor([T, ROUTE_ROW_PAD], dtype=pl.FP32)
    gate(
        x_mixed, norm_w, gate_w, gate_bias,
        layer_id, num_tokens, tid2eid, input_ids,
        x_norm_i8, x_norm_scale, indices, weights,
    )

    recv_x = pl.create_tensor([N_SLOTS_B, RECV_MAX, D], dtype=pl.INT8)
    recv_scale = pl.create_tensor([N_SLOTS_B, RECV_MAX], dtype=pl.FP32)
    recv_w = pl.create_tensor([N_SLOTS_B, RECV_MAX], dtype=pl.FP32)
    work_slot = pl.create_tensor([WORK_SLOTS_PAD], dtype=pl.INT32)
    work_rows = pl.create_tensor([WORK_SLOTS_PAD], dtype=pl.INT32)
    slot_expert = pl.create_tensor([N_SLOTS_B, IDX_PAD], dtype=pl.INT32)
    route_slot_row = pl.create_tensor([N_ROUTES, IDX_PAD], dtype=pl.INT32)
    route_group(
        indices, weights, x_norm_i8, x_norm_scale,
        recv_x, recv_scale, recv_w, work_slot, work_rows, slot_expert, route_slot_row,
        num_tokens,
    )
    return post_ffn, comb_ffn, recv_x, recv_scale, recv_w, work_slot, work_rows, slot_expert, route_slot_row


@pl.jit.inline(auto_scope=False)
def moe_local(
    recv_x: pl.Tensor[[N_SLOTS_B, RECV_MAX, D], pl.INT8],
    recv_scale: pl.Tensor[[N_SLOTS_B, RECV_MAX], pl.FP32],
    recv_w: pl.Tensor[[N_SLOTS_B, RECV_MAX], pl.FP32],
    work_slot: pl.Tensor[[WORK_SLOTS_PAD], pl.INT32],
    work_rows: pl.Tensor[[WORK_SLOTS_PAD], pl.INT32],
    slot_expert: pl.Tensor[[N_SLOTS_B, IDX_PAD], pl.INT32],
    route_slot_row: pl.Tensor[[N_ROUTES, IDX_PAD], pl.INT32],
    routed_w1: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_BANK, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_BANK, D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    recv_y = pl.create_tensor([N_SLOTS_B, RECV_MAX, D], dtype=pl.BF16)
    expert_routed_persistent_planned(
        recv_x, recv_scale, recv_w, work_slot, work_rows, slot_expert,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        recv_y,
    )

    ffn_partial = pl.create_tensor([T, D], dtype=pl.FP32)
    combine_local(recv_y, route_slot_row, ffn_partial, num_tokens)
    return ffn_partial
