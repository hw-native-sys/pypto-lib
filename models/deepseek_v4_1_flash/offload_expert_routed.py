# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Routed execution with early expert counts and a device slot bank."""
from pathlib import Path

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.runtime import pto_isa_include_dir
from expert_supply import publish_routed_counts
from expert_routed import (
    D, EP_WORLD_SIZE, GATHER_WORKER_TILE, K_SCALE, M, MOE_INTER, MOE_TOKENS, MX_GROUP, MX_M_TILE,
    N_LOCAL_EXPERTS, RECV_MAX, RECV_ROW_TILE, RECV_SCALE_TMP_TILE, ROUTED_CUBE_WORKER_TILE,
    ROUTED_VECTOR_WORKER_TILE, TOPK, _routed_swiglu_quant, _routed_w13, _routed_w2,
    push_balanced_dispatch_payload, push_dispatch_payload,
)


@pl.jit.inline(auto_scope=False)
def gather_routed_inputs(
    recv_x: pld.DistributedTensor,
    recv_routes: pld.DistributedTensor,
    recv_meta: pl.Tensor,
    recv_meta_window: pld.DistributedTensor,
    counts: pl.Tensor,
    x_out: pl.Tensor,
    scale_out: pl.Tensor,
    weights_out: pl.Tensor,
    routes_out: pl.Tensor,
    wait_ready: pl.Scalar[pl.TASK_ID], publish_ready: pl.Scalar[pl.TASK_ID],
    max_per_source: pl.constexpr,
    local_experts: pl.constexpr = N_LOCAL_EXPERTS,
    early_counts: pl.constexpr = False,
):
    """Gather source lanes with expert-partitioned workers and pack MX scales."""
    x_flat = pl.reshape(x_out, [local_experts * RECV_MAX, D])
    count_routes = pl.tensor.view(recv_routes, [local_experts * EP_WORLD_SIZE, max_per_source * 8])
    with pl.spmd(GATHER_WORKER_TILE, name_hint="dispatch_gather", deps=[wait_ready, publish_ready]) as gathered:
        core = pl.tile.get_block_idx()
        expert_groups = pl.min(local_experts, GATHER_WORKER_TILE)
        expert_lane = core // expert_groups
        for expert in pl.range(core % expert_groups, local_experts, expert_groups):
            expert_workers = (GATHER_WORKER_TILE + expert_groups - 1 - expert % expert_groups) // expert_groups
            source_count_values = pl.tile.full([EP_WORLD_SIZE, 8], dtype=pl.INT32, value=0)
            count_lanes = pl.load(count_routes, [expert * EP_WORLD_SIZE, 0], [EP_WORLD_SIZE, 8])
            source_counts = pl.array.create(EP_WORLD_SIZE, pl.INT32)
            count = pl.cast(0, pl.INDEX)
            for src in pl.range(EP_WORLD_SIZE):
                count_read = pl.tile.read(count_lanes, [src, 1])
                source_counts[src] = count_read
                pl.tile.write(source_count_values, [src, 0], count_read)
                count = count + pl.cast(count_read, pl.INDEX)
            if not early_counts and expert_lane == 0:
                source_count_valid = pl.set_validshape(source_count_values, EP_WORLD_SIZE, 1)
                recv_meta = pl.store(source_count_valid, [0, expert], recv_meta)
                recv_meta_window = pl.store(source_count_valid, [0, expert], recv_meta_window)
                expert_count_value = pl.tile.full([1, 8], dtype=pl.INT32, value=0)
                pl.tile.write(expert_count_value, [0, 0], pl.cast(count, pl.INT32))
                expert_count_valid = pl.set_validshape(expert_count_value, 1, 1)
                counts = pl.store(expert_count_valid, [expert, 0], counts)
            work_count = (count + RECV_ROW_TILE - 1) // RECV_ROW_TILE
            for work in pl.range(expert_lane, work_count, expert_workers):
                row_begin = work * RECV_ROW_TILE
                row_end = pl.min(row_begin + RECV_ROW_TILE, count)
                expert_base = expert * RECV_MAX
                scale_values = pl.tile.full([RECV_ROW_TILE, K_SCALE], dtype=pl.UINT8, value=0)
                weight_values = pl.tile.full([1, RECV_ROW_TILE], dtype=pl.FP32, value=0.0)
                route_values = pl.tile.full([1, RECV_ROW_TILE], dtype=pl.FP32, value=0.0)
                source_begin = 0
                for src in pl.pipeline(EP_WORLD_SIZE, stage=2):
                    source_count_read = source_counts[src]
                    source_count = pl.cast(source_count_read, pl.INDEX)
                    source_end = source_begin + source_count
                    source_base = expert_base + src * max_per_source
                    gather_begin = pl.max(row_begin, source_begin)
                    gather_end = pl.min(row_end, source_end)
                    copy_rows = pl.max(gather_end - gather_begin, 0)
                    if copy_rows > 0:
                        source_row = source_base + gather_begin - source_begin
                        output_row = expert_base + gather_begin
                        payload_raw = pl.load(recv_x, [source_row, 0], [RECV_ROW_TILE, D], valid_shape=[copy_rows, D])
                        payload = pl.reinterpret_view(payload_raw, pl.FP8E4M3FN)
                        x_flat = pl.store(payload, [output_row, 0], x_flat)
                        scale_packet = pl.load(recv_x, [source_row, D], [RECV_ROW_TILE, K_SCALE], valid_shape=[copy_rows, K_SCALE])
                        scale_raw = pl.reinterpret_view(scale_packet, pl.UINT8)
                        tile_offset = gather_begin - row_begin
                        scale_values = pl.tile.assemble(scale_values, scale_raw, [tile_offset, 0])
                        route_rows = pl.load(recv_routes, [source_row, 0], [RECV_ROW_TILE, 8], valid_shape=[copy_rows, 8])
                        metadata_full = pl.set_validshape(route_rows, RECV_ROW_TILE, 8)
                        metadata_flat = pl.reshape(metadata_full, [1, RECV_ROW_TILE * 8])
                        metadata_bits = pl.reinterpret_view(metadata_flat, pl.FP32)
                        route_quarter = pl.tile.gather_mask(metadata_bits, mask_pattern=pl.tile.MaskPattern.P0001)
                        route_packed = pl.tile.gather_mask(route_quarter, mask_pattern=pl.tile.MaskPattern.P0101)
                        weight_quarter = pl.tile.gather_mask(metadata_bits, mask_pattern=pl.tile.MaskPattern.P0100)
                        weight_packed = pl.tile.gather_mask(weight_quarter, mask_pattern=pl.tile.MaskPattern.P0101)
                        route_part = pl.set_validshape(route_packed, 1, copy_rows)
                        weight_part = pl.set_validshape(weight_packed, 1, copy_rows)
                        route_values = pl.tile.assemble(route_values, route_part, [0, tile_offset])
                        weight_values = pl.tile.assemble(weight_values, weight_part, [0, tile_offset])
                    source_begin = source_end
                valid_rows = row_end - row_begin
                weight_valid = pl.set_validshape(weight_values, 1, valid_rows)
                route_bits = pl.set_validshape(route_values, 1, valid_rows)
                route_valid = pl.reinterpret_view(route_bits, pl.INT32)
                weights_out = pl.store(weight_valid, [expert, row_begin], weights_out)
                routes_out = pl.store(route_valid, [expert, row_begin], routes_out)
                scale_row = expert_base + row_begin
                temporary = pl.create_tile([1, RECV_SCALE_TMP_TILE], dtype=pl.UINT8)
                packed = pl.tmov_x2zz(scale_values, temporary, group_axis=1, dst_rows=RECV_ROW_TILE, dst_cols=K_SCALE)
                packed_mx = pl.reinterpret_view(packed, pl.FP8E8M0)
                packed_flat = pl.reshape(packed_mx, [1, RECV_ROW_TILE * K_SCALE])
                scale_out = pl.store(packed_flat, [0, scale_row * K_SCALE], scale_out)
    return gathered


@pl.jit.inline
def dispatch(
    indices: pl.Tensor,
    x_norm_mx: pl.Tensor,
    x_norm_scale: pl.Tensor,
    weights: pl.Tensor,
    recv_x_out: pl.Tensor,
    recv_scale_out: pl.Tensor,
    recv_weight_out: pl.Tensor,
    recv_route_out: pl.Tensor,
    recv_count_out: pl.Tensor,
    recv_meta_local: pl.Tensor,
    recv_meta: pld.DistributedTensor,
    recv_x: pld.DistributedTensor,
    recv_routes: pld.DistributedTensor,
    data_arrived: pld.DistributedTensor,
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    previous_recv_x_use: pl.Scalar[pl.TASK_ID],
    local_experts: pl.constexpr = N_LOCAL_EXPERTS,
    topk: pl.constexpr = TOPK,
    dispatch_worker_limit: pl.constexpr = 0,
    early_counts: pl.constexpr = False,
):
    if early_counts:
        counts_ready = publish_routed_counts(
            indices, recv_count_out, recv_meta_local, recv_meta, data_arrived,
            num_tokens, my_rank, moe_epoch, local_experts, topk)
    stage_anchor = pl.system.task_dummy(deps=[])
    if MOE_TOKENS > 80:
        _push_tid = push_balanced_dispatch_payload(
            indices, x_norm_mx, x_norm_scale, weights,
            recv_x, recv_routes,
            num_tokens, my_rank, stage_anchor, local_experts=local_experts, topk=topk,
            dispatch_worker_limit=dispatch_worker_limit,
        )
    else:
        _push_tid = push_dispatch_payload(
            indices, x_norm_mx, x_norm_scale, weights,
            recv_x, recv_routes,
            num_tokens, my_rank, stage_anchor, local_experts=local_experts, topk=topk,
            dispatch_worker_limit=dispatch_worker_limit,
        )
    arrivals = pl.array.create(2, pl.TASK_ID)
    arrivals[0] = _push_tid
    arrivals[1] = _push_tid
    if EP_WORLD_SIZE > 1:
        # Publish one epoch only after every local payload store completes.
        with pl.spmd(EP_WORLD_SIZE, name_hint="dispatch_publish_epoch", deps=[_push_tid], allow_early_resolve=False) as publish_tid:
            dst = pl.tile.get_block_idx()
            if dst != my_rank:
                pld.system.notify(target=data_arrived, peer=dst, offsets=[my_rank, 0, 0], value=moe_epoch, op=pld.NotifyOp.Set)

        peer_waits = pl.array.create(EP_WORLD_SIZE, pl.TASK_ID)
        for src in pl.range(EP_WORLD_SIZE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_wait", deps=[_push_tid], no_dep_args=[data_arrived]) as peer_wait:
                if src != my_rank:
                    pld.system.defer_wait(signal=data_arrived, offsets=[src, 0, 0], expected=moe_epoch, cmp=pld.WaitCmp.Ge)
            peer_waits[src] = peer_wait
        arrivals[0] = publish_tid
        arrivals[1] = pl.system.task_dummy(deps=[peer_waits])
    _publish_tid = arrivals[0]
    _wait_tid = pl.system.task_dummy(deps=[arrivals[1], previous_recv_x_use])

    input_ready = gather_routed_inputs(
        recv_x, recv_routes, recv_meta_local, recv_meta, recv_count_out,
        recv_x_out, recv_scale_out, recv_weight_out, recv_route_out,
        _wait_tid, _publish_tid, MOE_TOKENS, local_experts=local_experts, early_counts=early_counts,
    )
    if not early_counts:
        counts_ready = input_ready
    return counts_ready, input_ready


@pl.jit.extern(
    core_type="mixed",
    aic_source=Path(__file__).parent / "kernels" / "expert_routed_w13_slots.cpp",
    aiv_source=Path(__file__).parent / "kernels" / "expert_routed_w13_slots.cpp",
    dual_aiv_dispatch=True, include_dirs=(pto_isa_include_dir(),),
)
def _routed_w13_slots(
    x: pl.Tensor,
    x_scale: pl.Tensor,
    counts: pl.Tensor,
    w1: pl.Tensor,
    w1_scale: pl.Tensor,
    w3: pl.Tensor,
    w3_scale: pl.Tensor,
    pair: pl.Out[pl.Tensor],
    route_weights: pl.Tensor,
    compact_weights: pl.Out[pl.Tensor],
    slot_map: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
    experts: pl.Scalar[pl.INDEX],
    stride: pl.Scalar[pl.INDEX],
    width: pl.Scalar[pl.INDEX],
    hidden: pl.Scalar[pl.INDEX],
    activation_rows: pl.Scalar[pl.INDEX],
    row_capacity: pl.constexpr,
) -> tuple[pl.Tensor, pl.Tensor]:
    """Store compact BF16 projections and route weights with the same row order."""
    return pair, compact_weights


@pl.jit.extern(
    core_type="mixed",
    aic_source=Path(__file__).parent / "kernels" / "expert_routed_w2_slots.cpp",
    aiv_source=Path(__file__).parent / "kernels" / "expert_routed_w2_slots.cpp",
    dual_aiv_dispatch=True, include_dirs=(pto_isa_include_dir(),),
)
def _routed_w2_slots(
    payload: pl.Tensor,
    scales: pl.Tensor,
    counts: pl.Tensor,
    weight: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.Out[pl.Tensor],
    slot_map: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
    experts: pl.Scalar[pl.INDEX],
    stride: pl.Scalar[pl.INDEX],
    width: pl.Scalar[pl.INDEX],
    hidden: pl.Scalar[pl.INDEX],
    output_rows: pl.Scalar[pl.INDEX],
    activation_rows: pl.Scalar[pl.INDEX],
    row_capacity: pl.constexpr,
) -> pl.Tensor:
    """Store full-K BF16 results in compact active-row order."""
    ...


@pl.jit.inline(auto_scope=False)
def _expert_routed_output(
    x: pl.Tensor,
    x_scale: pl.Tensor,
    weights: pl.Tensor,
    counts: pl.Tensor,
    w1: pl.Tensor,
    w1_scale: pl.Tensor,
    w3: pl.Tensor,
    w3_scale: pl.Tensor,
    w2: pl.Tensor,
    w2_scale: pl.Tensor,
    output: pl.Tensor,
    input_ready: pl.Scalar[pl.TASK_ID],
    experts: pl.constexpr,
    stride: pl.constexpr,
    width: pl.constexpr,
    hidden: pl.constexpr,
    row_capacity: pl.constexpr,
    slot_map: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
    by_slot: pl.constexpr = False,
    w13_workers: pl.Scalar[pl.INT32] = ROUTED_CUBE_WORKER_TILE,
    activation_workers: pl.Scalar[pl.INT32] = ROUTED_VECTOR_WORKER_TILE,
    w2_workers: pl.Scalar[pl.INT32] = ROUTED_CUBE_WORKER_TILE,
):
    """Run BF16 W13, FP32 weighted MXFP8 activation, and full-K BF16 W2.

    Counts lie in [0, stride] and sum to
    at most row_capacity. Stride is divisible by 16; width and hidden are
    divisible by 256.
    """
    activation_rows = ((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE
    pair = pl.create_tensor([((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE, 2 * hidden], dtype=pl.BF16)
    compact_weights = pl.create_tensor([((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE, 1], dtype=pl.FP32)
    with pl.spmd(w13_workers, name_hint="routed_w13", deps=[input_ready]) as w13_ready:
        if by_slot:
            pair, compact_weights = _routed_w13_slots(
                x, x_scale, counts, w1, w1_scale, w3, w3_scale, pair,
                weights, compact_weights, slot_map, experts, stride, width, hidden, activation_rows, row_capacity,
            )
        else:
            pair, compact_weights = _routed_w13(
                x, x_scale, counts, w1, w1_scale, w3, w3_scale, pair,
                weights, compact_weights, experts, stride, width, hidden, activation_rows, row_capacity,
            )

    payload = pl.create_tensor([((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE, hidden], dtype=pl.FP8E4M3FN)
    scales = pl.create_tensor([1, ((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE * (hidden // MX_GROUP)], dtype=pl.FP8E8M0)
    with pl.spmd(activation_workers, name_hint="routed_swiglu_quant", deps=[w13_ready]) as activation_ready:
        payload, scales = _routed_swiglu_quant(
            pair, compact_weights, counts, payload, scales, experts, stride, hidden, row_capacity,
        )

    output_rows = ((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE
    with pl.spmd(w2_workers, name_hint="routed_w2", deps=[activation_ready]) as ready:
        if by_slot:
            output = _routed_w2_slots(
                payload, scales, counts, w2, w2_scale, output,
                slot_map, experts, stride, width, hidden, output_rows, activation_rows, row_capacity,
            )
        else:
            output = _routed_w2(
                payload, scales, counts, w2, w2_scale, output,
                experts, stride, width, hidden, output_rows, activation_rows, row_capacity,
            )
    return output, ready


@pl.jit.inline(auto_scope=False)
def expert_routed_scatter(
    recv_x: pl.Tensor,
    recv_mx_scale: pl.Tensor,
    recv_weights: pl.Tensor,
    recv_expert_count: pl.Tensor,
    routed_w1_packed: pl.Tensor,
    routed_w1_scale: pl.Tensor[None, pl.FP8E8M0, pl.MX_B_NN],
    routed_w3_packed: pl.Tensor,
    routed_w3_scale: pl.Tensor[None, pl.FP8E8M0, pl.MX_B_NN],
    routed_w2_packed: pl.Tensor,
    routed_w2_scale: pl.Tensor[None, pl.FP8E8M0, pl.MX_B_NN],
    recv_route: pl.Tensor,
    recv_meta: pl.Tensor,
    routed_output: pld.DistributedTensor,
    input_ready: pl.Scalar[pl.TASK_ID],
    slot_map: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
    local_experts: pl.constexpr = N_LOCAL_EXPERTS,
    topk: pl.constexpr = M.num_experts_per_tok,
    by_slot: pl.constexpr = False,
    w13_workers: pl.Scalar[pl.INT32] = ROUTED_CUBE_WORKER_TILE,
    w2_workers: pl.Scalar[pl.INT32] = ROUTED_CUBE_WORKER_TILE,
    swiglu_workers: pl.Scalar[pl.INT32] = ROUTED_VECTOR_WORKER_TILE,
):
    """Compute active expert rows and scatter each source interval to its route destinations."""
    recv_x_flat = pl.reshape(recv_x, [local_experts * RECV_MAX, D])
    route_weights = pl.reshape(recv_weights, [local_experts * RECV_MAX, 1])
    if local_experts < topk:
        row_capacity = local_experts * RECV_MAX
    else:
        row_capacity = topk * RECV_MAX
    output = pl.create_tensor([((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE, D], dtype=pl.BF16)
    output, down_ready = _expert_routed_output(
        recv_x_flat, recv_mx_scale, route_weights, recv_expert_count,
        routed_w1_packed, routed_w1_scale, routed_w3_packed, routed_w3_scale,
        routed_w2_packed, routed_w2_scale, output, input_ready,
        local_experts, RECV_MAX, D, MOE_INTER,
        (local_experts if local_experts < topk else topk) * RECV_MAX,
        slot_map, by_slot, w13_workers, swiglu_workers, w2_workers,
    )
    flat_counts = pl.reshape(recv_expert_count, [1, local_experts])
    metadata_cols = ((local_experts + 7) // 8) * 8
    with pl.spmd(64, name_hint="ffn_scatter", deps=[down_ready]) as scatter_done:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="ffn_scatter", no_dep_args=[routed_output]):
            core = pl.tile.get_block_idx()
            count_values = pl.load(flat_counts, [0, 0], [1, metadata_cols], valid_shape=[1, local_experts])
            meta_values = pl.load(recv_meta, [0, 0], [EP_WORLD_SIZE, metadata_cols], valid_shape=[EP_WORLD_SIZE, local_experts])
            total_rows = 0
            for expert in pl.range(local_experts):
                rows_read = pl.tile.read(count_values, [0, expert])
                rows = pl.cast(rows_read, pl.INDEX)
                total_rows = total_rows + rows
            rows_per_core = total_rows // 64
            extra_rows = total_rows % 64
            core_begin = core * rows_per_core + pl.min(core, extra_rows)
            core_rows = rows_per_core + pl.min(pl.max(extra_rows - core, 0), 1)
            core_end = core_begin + core_rows
            compact_base = 0
            for local_e in pl.range(local_experts):
                count_read = pl.tile.read(count_values, [0, local_e])
                count = pl.cast(count_read, pl.INDEX)
                expert_begin = pl.max(compact_base, core_begin)
                expert_end = pl.min(compact_base + count, core_end)
                if expert_begin < expert_end:
                    source_begin = 0
                    for src in pl.range(EP_WORLD_SIZE):
                        source_rows_read = pl.tile.read(meta_values, [src, local_e])
                        source_rows = pl.cast(source_rows_read, pl.INDEX)
                        source_end = source_begin + source_rows
                        valid_end = pl.min(source_end, count)
                        valid_rows = pl.max(valid_end - source_begin, 0)
                        source_compact = compact_base + source_begin
                        first = pl.min(valid_rows, pl.max(core_begin - source_compact, 0))
                        end = pl.min(valid_rows, pl.max(core_end - source_compact, 0))
                        for batch_begin in pl.range(first, end, 64):
                            batch_rows = pl.min(64, end - batch_begin)
                            route_begin = source_begin + batch_begin
                            route_values = pl.load(recv_route, [local_e, route_begin], [1, 64], valid_shape=[1, batch_rows])
                            for lane in pl.pipeline(batch_rows, stage=2):
                                row = route_begin + lane
                                route_read = pl.tile.read(route_values, [0, lane])
                                route = pl.cast(route_read, pl.INDEX)
                                compact_row = compact_base + row
                                col = pl.cast(0, pl.INDEX)
                                value = pl.load(output, [compact_row, col], [1, D])
                                pld.tile.remote_store(value, target=routed_output, peer=src, offsets=[route, 0])
                        source_begin = source_end
                compact_base = compact_base + count
    return scatter_done
