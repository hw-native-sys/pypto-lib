# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Construct stable packed expert metadata with aligned independent route blocks."""

import pypto.language as pl

from .prefill_moe_geometry import (
    IDX_PAD, N_EXPERTS, N_ROUTES, N_SLOTS, N_SLOTS_B, RECV_MAX,
    SHARED_EID, SHARED_TILES, SH_SLOT, T, TOPK, WORK_SLOTS_PAD,
)

ROUTE_BLOCK = 384
TOKEN_BLOCK = ROUTE_BLOCK // TOPK
ROUTE_BLOCKS = (N_ROUTES + ROUTE_BLOCK - 1) // ROUTE_BLOCK
ROUTE_PAD = ROUTE_BLOCKS * ROUTE_BLOCK
EXPERT_TILE = 16
SLOT_TILE = 16


@pl.jit.inline(auto_scope=False)
def parallel_route_metadata(
    indices: pl.Tensor[[T, 16], pl.INT32],
    weights: pl.Tensor[[T, 16], pl.FP32],
    scales: pl.Tensor[[T, 1], pl.FP32],
    recv_scale: pl.Tensor[[N_SLOTS_B, RECV_MAX], pl.FP32],
    recv_weight: pl.Tensor[[N_SLOTS_B, RECV_MAX], pl.FP32],
    work_slot: pl.Tensor[[WORK_SLOTS_PAD], pl.INT32],
    work_rows: pl.Tensor[[WORK_SLOTS_PAD], pl.INT32],
    slot_expert: pl.Tensor[[N_SLOTS_B, IDX_PAD], pl.INT32],
    route_slot_row: pl.Tensor[[N_ROUTES, IDX_PAD], pl.INT32],
    active_tokens: pl.Scalar[pl.INDEX],
    initialized: pl.Scalar[pl.TASK_ID],
    prepared: pl.Scalar[pl.TASK_ID],
):
    histogram = pl.create_tensor([ROUTE_BLOCKS, N_EXPERTS], dtype=pl.INT32)
    prefix = pl.create_tensor([ROUTE_BLOCKS, N_EXPERTS], dtype=pl.INT32)
    totals = pl.create_tensor([1, N_EXPERTS], dtype=pl.INT32)
    local_order = pl.create_tensor([1, ROUTE_PAD], dtype=pl.INT32)
    global_order = pl.create_tensor([1, ROUTE_PAD], dtype=pl.INT32)
    start_counts = pl.create_tensor([ROUTE_BLOCKS, 16], dtype=pl.INT32)
    block_bases = pl.create_tensor([ROUTE_BLOCKS, 16], dtype=pl.INT32)
    expert_bases = pl.create_tensor([1, N_EXPERTS], dtype=pl.INT32)
    total_slots = pl.create_tensor([16], dtype=pl.INT32)
    tile_slots = pl.create_tensor([N_SLOTS, 16], dtype=pl.INT32)
    slot_info = pl.create_tensor([N_SLOTS, 16], dtype=pl.INT32)
    slot_routes = pl.create_tensor([N_SLOTS * RECV_MAX, 16], dtype=pl.INT32)
    active_routes = active_tokens * TOPK

    with pl.spmd(ROUTE_BLOCKS, name_hint="route_parallel_histogram", deps=[initialized, prepared]) as histogram_done:
        p0_block = pl.tile.get_block_idx()
        p0_route_begin = p0_block * ROUTE_BLOCK
        p0_token_begin = p0_block * TOKEN_BLOCK
        p0_valid_tokens = pl.min(TOKEN_BLOCK, T - p0_token_begin)
        p0_ids = pl.slice(indices, [TOKEN_BLOCK, 16], [p0_token_begin, 0], valid_shape=[p0_valid_tokens, 16])
        p0_counts = pl.tile.full([1, N_EXPERTS], dtype=pl.INT32, value=0)
        p0_orders = pl.tile.full([1, ROUTE_BLOCK], dtype=pl.INT32, value=0)
        p0_route_end = pl.min(p0_route_begin + ROUTE_BLOCK, active_routes)
        for p0_route in pl.range(p0_route_begin, p0_route_end):
            p0_token_in_block = (p0_route - p0_route_begin) // TOPK
            p0_expert = pl.cast(pl.read(p0_ids, [p0_token_in_block, p0_route % TOPK]), pl.INDEX)
            p0_ordinal = pl.read(p0_counts, [0, p0_expert])
            pl.write(p0_orders, [0, p0_route - p0_route_begin], p0_ordinal)
            pl.write(p0_counts, [0, p0_expert], pl.cast(p0_ordinal + 1, pl.INT32))
        pl.store(p0_counts, [p0_block, 0], histogram)
        pl.store(p0_orders, [0, p0_route_begin], local_order)

    with pl.spmd(N_EXPERTS // EXPERT_TILE, name_hint="route_parallel_expert_prefix", deps=[histogram_done]) as prefix_done:
        p1_expert_begin = pl.tile.get_block_idx() * EXPERT_TILE
        p1_cumulative = pl.tile.full([1, EXPERT_TILE], dtype=pl.INT32, value=0)
        p1_prefixes = pl.tile.full([ROUTE_BLOCKS, EXPERT_TILE], dtype=pl.INT32, value=0)
        for p1_block_index in pl.range(ROUTE_BLOCKS):
            for p1_expert_index in pl.range(EXPERT_TILE):
                pl.write(p1_prefixes, [p1_block_index, p1_expert_index], pl.read(p1_cumulative, [0, p1_expert_index]))
            p1_group_count = pl.load(histogram, [p1_block_index, p1_expert_begin], [1, EXPERT_TILE])
            p1_cumulative = pl.add(p1_cumulative, p1_group_count)
        pl.store(p1_prefixes, [0, p1_expert_begin], prefix)
        pl.store(p1_cumulative, [0, p1_expert_begin], totals)

    with pl.spmd(ROUTE_BLOCKS, name_hint="route_parallel_ordinals", deps=[prefix_done]) as ordinals_done:
        p2_block = pl.tile.get_block_idx()
        p2_begin = p2_block * ROUTE_BLOCK
        p2_token_begin = p2_block * TOKEN_BLOCK
        p2_valid_tokens = pl.min(TOKEN_BLOCK, T - p2_token_begin)
        p2_ids = pl.slice(indices, [TOKEN_BLOCK, 16], [p2_token_begin, 0], valid_shape=[p2_valid_tokens, 16])
        p2_offsets = pl.load(prefix, [p2_block, 0], [1, N_EXPERTS])
        p2_local = pl.load(local_order, [0, p2_begin], [1, ROUTE_BLOCK])
        p2_output = pl.tile.full([1, ROUTE_BLOCK], dtype=pl.INT32, value=0)
        p2_starts = pl.const(0, pl.INT32)
        p2_end = pl.min(p2_begin + ROUTE_BLOCK, active_routes)
        for p2_route in pl.range(p2_begin, p2_end):
            p2_expert = pl.cast(pl.read(p2_ids, [(p2_route - p2_begin) // TOPK, p2_route % TOPK]), pl.INDEX)
            p2_ordinal = pl.read(p2_local, [0, p2_route - p2_begin]) + pl.read(p2_offsets, [0, p2_expert])
            pl.write(p2_output, [0, p2_route - p2_begin], pl.cast(p2_ordinal, pl.INT32))
            if p2_ordinal % RECV_MAX == 0:
                p2_starts = p2_starts + pl.const(1, pl.INT32)
        pl.store(p2_output, [0, p2_begin], global_order)
        pl.write(start_counts, [p2_block, 0], p2_starts)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="route_parallel_bases", deps=[ordinals_done]) as bases_done:
        p3_count_tile = pl.load(totals, [0, 0], [1, N_EXPERTS])
        p3_tile_bases = pl.tile.full([1, N_EXPERTS], dtype=pl.INT32, value=0)
        p3_expert_total = pl.const(0, pl.INT32)
        for p3_expert in pl.range(N_EXPERTS):
            pl.write(p3_tile_bases, [0, p3_expert], p3_expert_total)
            p3_count = pl.read(p3_count_tile, [0, p3_expert])
            p3_expert_total = pl.cast(p3_expert_total + (p3_count + RECV_MAX - 1) // RECV_MAX, pl.INT32)
        pl.store(p3_tile_bases, [0, 0], expert_bases)
        p3_running = pl.const(0, pl.INT32)
        for p3_block_index in pl.range(ROUTE_BLOCKS):
            pl.write(block_bases, [p3_block_index, 0], p3_running)
            p3_running = p3_running + pl.read(start_counts, [p3_block_index, 0])
        pl.write(total_slots, [0], p3_running)

    with pl.spmd(ROUTE_BLOCKS, name_hint="route_parallel_slot_starts", deps=[bases_done]) as slots_done:
        p4_block = pl.tile.get_block_idx()
        p4_begin = p4_block * ROUTE_BLOCK
        p4_token_begin = p4_block * TOKEN_BLOCK
        p4_valid_tokens = pl.min(TOKEN_BLOCK, T - p4_token_begin)
        p4_ids = pl.slice(indices, [TOKEN_BLOCK, 16], [p4_token_begin, 0], valid_shape=[p4_valid_tokens, 16])
        p4_order = pl.load(global_order, [0, p4_begin], [1, ROUTE_BLOCK])
        p4_keys = pl.load(expert_bases, [0, 0], [1, N_EXPERTS])
        p4_slot = pl.read(block_bases, [p4_block, 0])
        p4_end = pl.min(p4_begin + ROUTE_BLOCK, active_routes)
        for p4_route in pl.range(p4_begin, p4_end):
            p4_ordinal = pl.read(p4_order, [0, p4_route - p4_begin])
            if p4_ordinal % RECV_MAX == 0:
                p4_expert = pl.cast(pl.read(p4_ids, [(p4_route - p4_begin) // TOPK, p4_route % TOPK]), pl.INDEX)
                p4_tile = p4_ordinal // RECV_MAX
                p4_key = pl.cast(pl.read(p4_keys, [0, p4_expert]) + p4_tile, pl.INDEX)
                pl.write(tile_slots, [p4_key, 0], p4_slot)
                p4_physical_slot = pl.cast(p4_slot, pl.INDEX)
                pl.write(slot_info, [p4_physical_slot, 0], pl.cast(p4_expert, pl.INT32))
                pl.write(slot_info, [p4_physical_slot, 1], pl.cast(p4_tile, pl.INT32))
                p4_slot = p4_slot + pl.const(1, pl.INT32)

    with pl.spmd(ROUTE_BLOCKS, name_hint="route_parallel_scatter", deps=[slots_done]) as scattered:
        p5_block = pl.tile.get_block_idx()
        p5_begin = p5_block * ROUTE_BLOCK
        p5_token_begin = p5_block * TOKEN_BLOCK
        p5_valid_tokens = pl.min(TOKEN_BLOCK, T - p5_token_begin)
        p5_ids = pl.slice(indices, [TOKEN_BLOCK, 16], [p5_token_begin, 0], valid_shape=[p5_valid_tokens, 16])
        p5_order = pl.load(global_order, [0, p5_begin], [1, ROUTE_BLOCK])
        p5_keys = pl.load(expert_bases, [0, 0], [1, N_EXPERTS])
        p5_route_rows = pl.tile.full([ROUTE_BLOCK, IDX_PAD], dtype=pl.INT32, value=0)
        p5_end = pl.min(p5_begin + ROUTE_BLOCK, active_routes)
        for p5_route in pl.range(p5_begin, p5_end):
            p5_expert = pl.cast(pl.read(p5_ids, [(p5_route - p5_begin) // TOPK, p5_route % TOPK]), pl.INDEX)
            p5_ordinal = pl.read(p5_order, [0, p5_route - p5_begin])
            p5_key = pl.cast(pl.read(p5_keys, [0, p5_expert]) + p5_ordinal // RECV_MAX, pl.INDEX)
            p5_slot = pl.read(tile_slots, [p5_key, 0])
            p5_destination = pl.cast(p5_slot * RECV_MAX + p5_ordinal % RECV_MAX, pl.INDEX)
            pl.write(p5_route_rows, [p5_route - p5_begin, 0], pl.cast(p5_destination, pl.INT32))
            pl.write(slot_routes, [p5_destination, 0], pl.cast(p5_route, pl.INT32))
        p5_physical_rows = pl.min(ROUTE_BLOCK, N_ROUTES - p5_begin)
        p5_valid_routes = pl.set_validshape(p5_route_rows, p5_physical_rows, IDX_PAD)
        pl.store(p5_valid_routes, [p5_begin, 0], route_slot_row)

    with pl.spmd((N_SLOTS_B + SLOT_TILE - 1) // SLOT_TILE, name_hint="route_parallel_materialize", deps=[scattered]) as materialized:
        p6_first = pl.tile.get_block_idx() * SLOT_TILE
        p6_valid_slots = pl.min(SLOT_TILE, N_SLOTS_B - p6_first)
        p6_routed_slots = pl.cast(pl.read(total_slots, [0]), pl.INDEX)
        p6_count_tile = pl.load(totals, [0, 0], [1, N_EXPERTS])
        p6_expert_tile = pl.tile.full([SLOT_TILE, IDX_PAD], dtype=pl.INT32, value=0)
        p6_scale_tile = pl.tile.full([SLOT_TILE, RECV_MAX], dtype=pl.FP32, value=0.0)
        p6_weight_tile = pl.tile.full([SLOT_TILE, RECV_MAX], dtype=pl.FP32, value=0.0)
        for p6_local_slot in pl.range(p6_valid_slots):
            p6_slot = p6_first + p6_local_slot
            if p6_slot < p6_routed_slots:
                p6_expert = pl.cast(pl.read(slot_info, [p6_slot, 0]), pl.INDEX)
                p6_ordinal_tile = pl.read(slot_info, [p6_slot, 1])
                p6_count = pl.min(RECV_MAX, pl.read(p6_count_tile, [0, p6_expert]) - p6_ordinal_tile * RECV_MAX)
                pl.write(p6_expert_tile, [p6_local_slot, 0], pl.cast(p6_expert, pl.INT32))
                for p6_row in pl.range(pl.cast(p6_count, pl.INDEX)):
                    p6_route = pl.cast(pl.read(slot_routes, [p6_slot * RECV_MAX + p6_row, 0]), pl.INDEX)
                    p6_token = p6_route // TOPK
                    p6_k = p6_route % TOPK
                    pl.write(p6_scale_tile, [p6_local_slot, p6_row], pl.read(scales, [p6_token, 0]))
                    pl.write(p6_weight_tile, [p6_local_slot, p6_row], pl.read(weights, [p6_token, p6_k]))
            elif p6_slot >= SH_SLOT:
                p6_shared = p6_slot - SH_SLOT
                p6_count = pl.min(RECV_MAX, pl.max(active_tokens - p6_shared * RECV_MAX, 0))
                for p6_column in pl.range(IDX_PAD):
                    pl.write(p6_expert_tile, [p6_local_slot, p6_column], pl.const(SHARED_EID, pl.INT32))
                for p6_row in pl.range(p6_count):
                    p6_token = p6_shared * RECV_MAX + p6_row
                    pl.write(p6_scale_tile, [p6_local_slot, p6_row], pl.read(scales, [p6_token, 0]))
                    pl.write(p6_weight_tile, [p6_local_slot, p6_row], pl.const(1.0, pl.FP32))
        p6_valid_experts = pl.set_validshape(p6_expert_tile, p6_valid_slots, IDX_PAD)
        p6_valid_scales = pl.set_validshape(p6_scale_tile, p6_valid_slots, RECV_MAX)
        p6_valid_weights = pl.set_validshape(p6_weight_tile, p6_valid_slots, RECV_MAX)
        pl.store(p6_valid_experts, [p6_first, 0], slot_expert)
        pl.store(p6_valid_scales, [p6_first, 0], recv_scale)
        pl.store(p6_valid_weights, [p6_first, 0], recv_weight)

    work_slot_flat = pl.reshape(work_slot, [1, WORK_SLOTS_PAD])
    work_rows_flat = pl.reshape(work_rows, [1, WORK_SLOTS_PAD])
    with pl.spmd((WORK_SLOTS_PAD + 15) // 16, name_hint="route_parallel_workplan", deps=[slots_done]) as planned:
        p7_begin = pl.tile.get_block_idx() * 16
        p7_count_tile = pl.load(totals, [0, 0], [1, N_EXPERTS])
        p7_routed_slots = pl.cast(pl.read(total_slots, [0]), pl.INDEX)
        p7_slots = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
        p7_rows = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
        for p7_local_plan in pl.range(pl.min(16, WORK_SLOTS_PAD - p7_begin)):
            p7_plan = p7_begin + p7_local_plan
            if p7_plan < p7_routed_slots:
                p7_expert = pl.cast(pl.read(slot_info, [p7_plan, 0]), pl.INDEX)
                p7_tile = pl.read(slot_info, [p7_plan, 1])
                p7_count = pl.min(RECV_MAX, pl.read(p7_count_tile, [0, p7_expert]) - p7_tile * RECV_MAX)
                pl.write(p7_slots, [0, p7_local_plan], pl.cast(p7_plan, pl.INT32))
                pl.write(p7_rows, [0, p7_local_plan], pl.cast(p7_count, pl.INT32))
            elif p7_plan < p7_routed_slots + SHARED_TILES:
                p7_shared = p7_plan - p7_routed_slots
                p7_count = pl.min(RECV_MAX, pl.max(active_tokens - p7_shared * RECV_MAX, 0))
                pl.write(p7_slots, [0, p7_local_plan], pl.cast(SH_SLOT + p7_shared, pl.INT32))
                pl.write(p7_rows, [0, p7_local_plan], pl.cast(p7_count, pl.INT32))
        p7_columns = pl.min(16, WORK_SLOTS_PAD - p7_begin)
        p7_valid_slots = pl.set_validshape(p7_slots, 1, p7_columns)
        p7_valid_rows = pl.set_validshape(p7_rows, 1, p7_columns)
        pl.store(p7_valid_slots, [0, p7_begin], work_slot_flat)
        pl.store(p7_valid_rows, [0, p7_begin], work_rows_flat)
    return pl.system.task_dummy(deps=[scattered, materialized, planned])
