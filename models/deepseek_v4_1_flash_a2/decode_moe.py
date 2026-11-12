# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4.1 A2/A3 W8A8 Decode MoE for the two-rank EP gate.

This entry owns the product-side EP2 transport.  It keeps all 384 routed
experts, assigns 192 experts to each rank, and moves only PyPTO tensors across
the two rank windows.  The vLLM reference process is intentionally outside
this module.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import pypto.language.distributed as pld

from models.deepseek_v4_1_flash_a2.config import FLASH
from models.deepseek_v4_1_flash_a2.moe_router import make_moe_router_inline
from models.deepseek_v4_1_flash_a2.moe_shared import make_shared_moe_w8a8_inline
from models.deepseek_v4_1_flash_a2.quant import make_dynamic_int8_quantize
from models.deepseek_v4_1_flash_a2.moe import (
    make_routed_expert_projection,
    make_routed_expert_w8a8,
    make_shared_expert_projection,
    make_shared_expert_w8a8,
    make_w8a8_expert_projection,
    routed_expert_reference,
    shared_expert_reference,
    w8a8_expert_linear,
)
from models.deepseek_v4_1_flash_a2.moe_router import make_moe_router
from models.deepseek_v4_1_flash_a2.moe_shared import make_shared_moe_w8a8
from models.deepseek_v4_1_flash_a2.moe_routed import expert_routed_inline
from models.deepseek_v4_1_flash_a2.moe_dispatch import make_dispatch_layout, make_route_restore


# EP2 product topology and Decode capacity.
EP_SIZE = 2
N_LOCAL_EXPERTS = 192
N_EXPERTS = 384
TOPK = 6
T = 16
D = 5120
MOE_INTER = 2304
N_ROUTES = T * TOPK
LANE_CAP = N_ROUTES
LANE_ROWS = EP_SIZE * LANE_CAP
RECV_MAX = 256
AUX_SCALE, AUX_WEIGHT, AUX_ROUTE, AUX_EXPERT = 0, 1, 2, 3
AUX_PAD = 16
COUNT_PAD = 16

# The routed leaf is compiled for the complete 192-expert local shard.  These
# variables are read only while importing that leaf and never change the ABI.
os.environ["A2_MOE_LOCAL_EXPERTS"] = str(N_LOCAL_EXPERTS)
os.environ["A2_MOE_RECV_MAX"] = str(RECV_MAX)
from models.deepseek_v4_1_flash_a2.moe_routed import expert_routed_tile


_ROUTER_CONFIG = asdict(FLASH)
_ROUTER_CONFIG["image_sentinel_base_id"] = 129257
ROUTER = make_moe_router_inline(T, _ROUTER_CONFIG)
# CANN's fused EP2 dispatch quantizes normalized rows with FP16 truncation
# followed by a rounded INT8 store.  The legacy mode remains available for a
# controlled comparison while other model paths keep their existing contract.
QUANTIZE = make_dynamic_int8_quantize(
    T,
    D,
    cann_dispatch=os.environ.get("PYPTO_EP2_INPUT_QUANT", "legacy") == "cann",
)
SHARED = make_shared_moe_w8a8_inline(T, _ROUTER_CONFIG, fused_activation=True)
# Temporary NPU diagnosis switch.  It is consumed only by the local EP2
# runner to isolate the two additive MoE branches; the formal path remains
# the default ``sum`` route.
_DEBUG_COMPONENT = {"routed_only": 1, "shared_only": 2}.get(
    os.environ.get("PYPTO_EP2_DEBUG_COMPONENT", "sum"), 0
)


@pl.jit.inline
def decode_ep2_pack(
    route_ids: pl.Tensor[[T, TOPK], pl.INT32],
    route_weights: pl.Tensor[[T, TOPK], pl.FP32],
    x_int8: pl.Tensor[[T, D], pl.INT8],
    x_scale: pl.Tensor[[T, 1], pl.FP32],
    lane_x: pld.DistributedTensor[[LANE_ROWS, D], pl.INT8],
    lane_aux: pld.DistributedTensor[[LANE_ROWS, AUX_PAD], pl.FP32],
    lane_count: pld.DistributedTensor[[EP_SIZE, COUNT_PAD], pl.INT32],
    lane_ready: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    active: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Send each route to its owning rank and publish empty buckets too."""

    active_rows = pl.min(pl.max(pl.cast(active, pl.INDEX), 0), T)
    destination_counts = pl.create_tensor([EP_SIZE], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dsv41_ep2_pack_count") as count_tid:
        for destination in pl.range(EP_SIZE):
            pl.write(destination_counts, [destination], pl.cast(0, pl.INT32))
        for token in pl.range(active_rows):
            for route in pl.range(TOPK):
                expert = pl.cast(pl.read(route_ids, [token, route]), pl.INDEX)
                destination = expert // N_LOCAL_EXPERTS
                old = pl.read(destination_counts, [destination])
                pl.write(destination_counts, [destination], pl.cast(old + 1, pl.INT32))

    with pl.spmd(EP_SIZE, name_hint="dsv41_ep2_pack_payload", deps=[count_tid]) as push_tid:
        destination = pl.tile.get_block_idx()
        lane_slot = pl.cast(0, pl.INDEX)
        for token in pl.range(active_rows):
            for route in pl.range(TOPK):
                expert = pl.cast(pl.read(route_ids, [token, route]), pl.INDEX)
                target = expert // N_LOCAL_EXPERTS
                if target == destination:
                    lane_row = pl.cast(rank, pl.INDEX) * LANE_CAP + lane_slot
                    pld.tensor.put(
                        dst=lane_x,
                        peer=destination,
                        src=x_int8,
                        dst_offsets=[lane_row, 0],
                        src_offsets=[token, 0],
                        shape=[1, D],
                    )
                    metadata = pl.tile.full([1, AUX_PAD], dtype=pl.FP32, value=0.0)
                    pl.tile.write(metadata, [0, AUX_SCALE], pl.read(x_scale, [token, 0]))
                    pl.tile.write(metadata, [0, AUX_WEIGHT], pl.read(route_weights, [token, route]))
                    route_slot = pl.cast(token * TOPK + route, pl.INT32)
                    pl.tile.write(metadata, [0, AUX_ROUTE], pl.cast(route_slot, pl.FP32))
                    local_expert = expert - pl.cast(target * N_LOCAL_EXPERTS, pl.INDEX)
                    local_expert_i32 = pl.cast(local_expert, pl.INT32)
                    pl.tile.write(metadata, [0, AUX_EXPERT], pl.cast(local_expert_i32, pl.FP32))
                    pld.tile.remote_store(metadata, target=lane_aux, peer=destination, offsets=[lane_row, 0])
                    lane_slot = lane_slot + 1

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dsv41_ep2_pack_publish", deps=[push_tid]) as publish_tid:
        for destination in pl.range(EP_SIZE):
            count_tile = pl.tile.full([1, COUNT_PAD], dtype=pl.INT32, value=0)
            pl.tile.write(count_tile, [0, 0], pl.read(destination_counts, [destination]))
            pld.tile.remote_store(count_tile, target=lane_count, peer=destination, offsets=[rank, 0])
            if destination != rank:
                pld.system.notify(
                    target=lane_ready,
                    peer=destination,
                    offsets=[rank, 0],
                    value=epoch,
                    op=pld.NotifyOp.Set,
                )
    return publish_tid


@pl.jit.inline
def decode_ep2_arrive(
    lane_x: pld.DistributedTensor[[LANE_ROWS, D], pl.INT8],
    lane_aux: pld.DistributedTensor[[LANE_ROWS, AUX_PAD], pl.FP32],
    lane_count: pld.DistributedTensor[[EP_SIZE, COUNT_PAD], pl.INT32],
    lane_ready: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    route_x: pl.Tensor[[LANE_ROWS, D], pl.INT8],
    route_scale: pl.Tensor[[LANE_ROWS, 1], pl.FP32],
    route_weight: pl.Tensor[[LANE_ROWS, 1], pl.FP32],
    route_id: pl.Tensor[[LANE_ROWS, 1], pl.INT32],
    route_expert: pl.Tensor[[LANE_ROWS, 1], pl.INT32],
    route_source: pl.Tensor[[LANE_ROWS, 1], pl.INT32],
    route_count: pl.Tensor[[1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    pack_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Wait for both sources and flatten source-major receive lanes."""

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dsv41_ep2_arrive_wait", deps=[pack_tid]) as wait_tid:
        for source in pl.range(EP_SIZE):
            if source != rank:
                pld.system.wait(signal=lane_ready, offsets=[source, 0], expected=epoch, cmp=pld.WaitCmp.Ge)

    source_offsets = pl.create_tensor([EP_SIZE], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dsv41_ep2_arrive_prefix", deps=[wait_tid]) as prefix_tid:
        running = pl.cast(0, pl.INDEX)
        for source in pl.range(EP_SIZE):
            pl.write(source_offsets, [source], pl.cast(running, pl.INT32))
            rows = pl.cast(pl.read(lane_count, [source, 0]), pl.INDEX)
            for row in pl.range(rows):
                flat = running + row
                lane_row = source * LANE_CAP + row
                pl.write(route_scale, [flat, 0], pl.read(lane_aux, [lane_row, AUX_SCALE]))
                pl.write(route_weight, [flat, 0], pl.read(lane_aux, [lane_row, AUX_WEIGHT]))
                pl.write(route_id, [flat, 0], pl.cast(pl.read(lane_aux, [lane_row, AUX_ROUTE]), pl.INT32))
                pl.write(route_expert, [flat, 0], pl.cast(pl.read(lane_aux, [lane_row, AUX_EXPERT]), pl.INT32))
                pl.write(route_source, [flat, 0], pl.cast(source, pl.INT32))
            running = running + rows
        pl.write(route_count, [0], pl.cast(running, pl.INT32))

    with pl.spmd(EP_SIZE, name_hint="dsv41_ep2_arrive_copy", deps=[prefix_tid]):
        source = pl.tile.get_block_idx()
        base = pl.cast(pl.read(source_offsets, [source]), pl.INDEX)
        rows = pl.cast(pl.read(lane_count, [source, 0]), pl.INDEX)
        for row in pl.range(rows):
            flat = base + row
            lane_row = source * LANE_CAP + row
            route_x[flat:flat + 1, :] = lane_x[lane_row:lane_row + 1, :]
    return prefix_tid


@pl.jit.inline
def decode_ep2_regroup(
    route_expert: pl.Tensor[[LANE_ROWS, 1], pl.INT32],
    route_source: pl.Tensor[[LANE_ROWS, 1], pl.INT32],
    route_id: pl.Tensor[[LANE_ROWS, 1], pl.INT32],
    route_x: pl.Tensor[[LANE_ROWS, D], pl.INT8],
    route_scale: pl.Tensor[[LANE_ROWS, 1], pl.FP32],
    route_weight: pl.Tensor[[LANE_ROWS, 1], pl.FP32],
    route_count: pl.Tensor[[1], pl.INT32],
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_weight: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_source: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32],
    recv_route: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32],
    recv_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    expert_offsets: pl.Tensor[[N_LOCAL_EXPERTS + 1], pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Stable expert-major regroup with explicit counts and untouched tails."""

    total = pl.min(pl.max(pl.cast(pl.read(route_count, [0]), pl.INDEX), 0), LANE_ROWS)
    bucket = pl.create_tensor([LANE_ROWS], dtype=pl.INT32)
    expert_counts = pl.create_tensor([N_LOCAL_EXPERTS], dtype=pl.INT32)
    cursors = pl.create_tensor([N_LOCAL_EXPERTS], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dsv41_ep2_regroup_meta") as meta_tid:
        for expert in pl.range(N_LOCAL_EXPERTS):
            pl.write(expert_counts, [expert], pl.cast(0, pl.INT32))
        for row in pl.range(total):
            expert = pl.cast(pl.read(route_expert, [row, 0]), pl.INDEX)
            pl.write(expert_counts, [expert], pl.cast(pl.read(expert_counts, [expert]) + 1, pl.INT32))
        running = pl.cast(0, pl.INT32)
        for expert in pl.range(N_LOCAL_EXPERTS):
            pl.write(expert_offsets, [expert], running)
            pl.write(cursors, [expert], pl.cast(0, pl.INT32))
            pl.write(recv_count, [expert, 0], pl.read(expert_counts, [expert]))
            running = running + pl.read(expert_counts, [expert])
        pl.write(expert_offsets, [N_LOCAL_EXPERTS], running)
        for row in pl.range(total):
            expert = pl.cast(pl.read(route_expert, [row, 0]), pl.INDEX)
            local_slot = pl.read(cursors, [expert])
            pl.write(cursors, [expert], pl.cast(local_slot + 1, pl.INT32))
            bucket_slot = pl.read(expert_offsets, [expert]) + local_slot
            pl.write(bucket, [bucket_slot], pl.cast(row, pl.INT32))
            pl.write(recv_scale, [expert, local_slot], pl.read(route_scale, [row, 0]))
            pl.write(recv_weight, [expert, local_slot], pl.read(route_weight, [row, 0]))
            pl.write(recv_source, [expert, local_slot], pl.read(route_source, [row, 0]))
            pl.write(recv_route, [expert, local_slot], pl.read(route_id, [row, 0]))

    recv_x_flat = pl.reshape(recv_x, [N_LOCAL_EXPERTS * RECV_MAX, D])
    with pl.spmd(N_LOCAL_EXPERTS, name_hint="dsv41_ep2_regroup_payload", deps=[meta_tid]):
        expert = pl.tile.get_block_idx()
        base = pl.cast(pl.read(expert_offsets, [expert]), pl.INDEX)
        rows = pl.cast(pl.read(expert_counts, [expert]), pl.INDEX)
        for row in pl.range(rows):
            source_row = pl.cast(pl.read(bucket, [base + row]), pl.INDEX)
            target_row = expert * RECV_MAX + row
            recv_x_flat[target_row:target_row + 1, :] = route_x[source_row:source_row + 1, :]
    return meta_tid


@pl.jit.inline
def decode_ep2_return(
    expert_output: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.FP32],
    recv_source: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32],
    recv_route: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32],
    recv_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    slot_out: pld.DistributedTensor[[N_ROUTES, D], pl.FP32],
) -> pl.Scalar[pl.TASK_ID]:
    """Return weighted expert rows to the source rank's route slots."""

    flat = pl.reshape(expert_output, [N_LOCAL_EXPERTS * RECV_MAX, D])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dsv41_ep2_return", allow_early_resolve=True) as return_tid:
        for expert in pl.range(N_LOCAL_EXPERTS):
            rows = pl.cast(pl.read(recv_count, [expert, 0]), pl.INDEX)
            for row in pl.range(rows):
                source = pl.read(recv_source, [expert, row])
                route = pl.cast(pl.read(recv_route, [expert, row]), pl.INDEX)
                source_row = expert * RECV_MAX + row
                pld.tensor.put(
                    dst=slot_out,
                    peer=source,
                    src=flat,
                    dst_offsets=[route, 0],
                    src_offsets=[source_row, 0],
                    shape=[1, D],
                )
    return return_tid


@pl.jit.inline
def decode_ep2_combine(
    slot_out: pld.DistributedTensor[[N_ROUTES, D], pl.FP32],
    return_ready: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    shared_output: pl.Tensor[[T, D], pl.BF16],
    output: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    active: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    return_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Wait for every rank and add routed output to the shared expert."""

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dsv41_ep2_return_notify", deps=[return_tid]) as notify_tid:
        for peer in pl.range(EP_SIZE):
            if peer != rank:
                pld.system.notify(
                    target=return_ready,
                    peer=peer,
                    offsets=[rank, 0],
                    value=epoch,
                    op=pld.NotifyOp.Set,
                )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dsv41_ep2_return_wait", deps=[notify_tid]) as wait_tid:
        for source in pl.range(EP_SIZE):
            if source != rank:
                pld.system.wait(signal=return_ready, offsets=[source, 0], expected=epoch, cmp=pld.WaitCmp.Ge)

    active_rows = pl.min(pl.max(pl.cast(active, pl.INDEX), 0), T)
    with pl.spmd(T, name_hint="dsv41_ep2_combine", deps=[wait_tid]):
        token = pl.tile.get_block_idx()
        if token < active_rows:
            # The fused reference stores each routed expert result as BF16 and
            # widens that value at combine.  Keep this per-route rounding
            # boundary before the FP32 reduction; narrowing the aggregate
            # after summation changes the observable MoE output.
            routed = pl.full([1, D], dtype=pl.FP32, value=0.0)
            for route in pl.range(TOPK):
                slot = token * TOPK + route
                routed_row = pl.cast(slot_out[slot:slot + 1, :], pl.BF16, mode="rint")
                routed = pl.add(routed, pl.cast(routed_row, pl.FP32))
            shared = pl.cast(shared_output[token:token + 1, :], pl.FP32)
            if _DEBUG_COMPONENT == 1:
                combined = routed
            elif _DEBUG_COMPONENT == 2:
                combined = shared
            else:
                combined = pl.add(shared, routed)
            output[token:token + 1, :] = pl.cast(combined, pl.BF16, mode="rint")
        else:
            output[token:token + 1, :] = pl.full([1, D], dtype=pl.BF16, value=0.0)
    return wait_tid


@pl.jit(auto_scope=False)
def decode_ep2_rank(
    value: pl.Tensor[[T, D], pl.BF16],
    router_value: pl.Tensor[[T, D], pl.FP32],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    text_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    vision_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, 2560], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    shared_gate_up: pl.Tensor[[D, MOE_INTER * 2], pl.INT8],
    shared_gate_up_scale: pl.Tensor[[MOE_INTER * 2], pl.FP32],
    shared_down: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_down_scale: pl.Tensor[[D], pl.FP32],
    lane_x: pld.DistributedTensor[[LANE_ROWS, D], pl.INT8],
    lane_aux: pld.DistributedTensor[[LANE_ROWS, AUX_PAD], pl.FP32],
    lane_count: pld.DistributedTensor[[EP_SIZE, COUNT_PAD], pl.INT32],
    lane_ready: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    slot_out: pld.DistributedTensor[[N_ROUTES, D], pl.FP32],
    return_ready: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    logits: pl.InOut[pl.Tensor[[T, N_EXPERTS], pl.FP32]],
    route_ids: pl.InOut[pl.Tensor[[T, TOPK], pl.INT32]],
    route_weights: pl.InOut[pl.Tensor[[T, TOPK], pl.FP32]],
    output: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    active: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """Run router, distributed EP2, local routed experts, and shared expert."""

    x_int8 = pl.create_tensor([T, D], dtype=pl.INT8)
    x_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    QUANTIZE(value, x_int8, x_scale, active)
    pl.dump_tag(x_int8)
    pl.dump_tag(x_scale)
    ROUTER(value, router_value, gate_weight, text_bias, vision_bias, input_ids, logits, route_ids, route_weights, active)

    pack_tid = decode_ep2_pack(
        route_ids, route_weights, x_int8, x_scale, lane_x, lane_aux, lane_count, lane_ready, active, rank, epoch,
    )
    route_x = pl.create_tensor([LANE_ROWS, D], dtype=pl.INT8)
    route_scale = pl.create_tensor([LANE_ROWS, 1], dtype=pl.FP32)
    route_weight = pl.create_tensor([LANE_ROWS, 1], dtype=pl.FP32)
    route_id = pl.create_tensor([LANE_ROWS, 1], dtype=pl.INT32)
    route_expert = pl.create_tensor([LANE_ROWS, 1], dtype=pl.INT32)
    route_source = pl.create_tensor([LANE_ROWS, 1], dtype=pl.INT32)
    route_count = pl.create_tensor([1], dtype=pl.INT32)
    decode_ep2_arrive(
        lane_x, lane_aux, lane_count, lane_ready, route_x, route_scale, route_weight,
        route_id, route_expert, route_source, route_count, rank, epoch, pack_tid,
    )

    recv_x = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX, D], dtype=pl.INT8)
    recv_scale = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX], dtype=pl.FP32)
    recv_weight = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX], dtype=pl.FP32)
    recv_source = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX], dtype=pl.INT32)
    recv_route = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX], dtype=pl.INT32)
    recv_count = pl.create_tensor([N_LOCAL_EXPERTS, 1], dtype=pl.INT32)
    expert_offsets = pl.create_tensor([N_LOCAL_EXPERTS + 1], dtype=pl.INT32)
    decode_ep2_regroup(
        route_expert, route_source, route_id, route_x, route_scale, route_weight, route_count,
        recv_x, recv_scale, recv_weight, recv_source, recv_route, recv_count, expert_offsets,
    )

    routed_output = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX, D], dtype=pl.FP32)
    routed_output_flat = pl.reshape(routed_output, [N_LOCAL_EXPERTS * RECV_MAX, D])
    d_tiles = D // 256
    with pl.spmd(N_LOCAL_EXPERTS * (RECV_MAX // 64) * d_tiles, name_hint="dsv41_ep2_expert_output_zero") as inputs_ready:
        block = pl.tile.get_block_idx()
        zero_row = (block // d_tiles) * 64
        zero_col = (block % d_tiles) * 256
        routed_output_flat[zero_row : zero_row + 64, zero_col : zero_col + 256] = pl.full(
            [64, 256], dtype=pl.FP32, value=0.0
        )
    for local_e in pl.parallel(N_LOCAL_EXPERTS):
        n_rows = pl.cast(pl.read(recv_count, [local_e, 0]), pl.INDEX)
        n_tiles = (n_rows + 64 - 1) // 64
        for tile in pl.parallel(n_tiles):
            tile_row = tile * 64
            valid_rows = pl.min(64, n_rows - tile_row)
            flat_tile_row = local_e * RECV_MAX + tile_row
            recv_y_tile = pl.create_tensor([64, D], dtype=pl.FP32)
            expert_routed_tile(
                recv_x, recv_scale, recv_weight,
                routed_w1, routed_w1_scale,
                routed_w3, routed_w3_scale,
                routed_w2, routed_w2_scale,
                recv_y_tile,
                local_e, tile_row, valid_rows, inputs_ready,
            )
            routed_output_flat[flat_tile_row : flat_tile_row + 64, :] = recv_y_tile
    return_tid = decode_ep2_return(routed_output, recv_source, recv_route, recv_count, slot_out)

    shared_output = pl.create_tensor([T, D], dtype=pl.BF16)
    SHARED(value, shared_gate_up, shared_gate_up_scale, shared_down, shared_down_scale, shared_output, active)
    decode_ep2_combine(slot_out, return_ready, shared_output, output, active, rank, epoch, return_tid)
    return output


@pl.jit.host
def decode_ep2(
    value: pl.Tensor[[EP_SIZE, T, D], pl.BF16],
    router_value: pl.Tensor[[EP_SIZE, T, D], pl.FP32],
    gate_weight: pl.Tensor[[EP_SIZE, N_EXPERTS, D], pl.FP32],
    text_bias: pl.Tensor[[EP_SIZE, N_EXPERTS], pl.FP32],
    vision_bias: pl.Tensor[[EP_SIZE, N_EXPERTS], pl.FP32],
    input_ids: pl.Tensor[[EP_SIZE, T], pl.INT64],
    routed_w1: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, D, 2560], pl.INT8],
    routed_w2_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, D], pl.FP32],
    shared_gate_up: pl.Tensor[[EP_SIZE, D, MOE_INTER * 2], pl.INT8],
    shared_gate_up_scale: pl.Tensor[[EP_SIZE, MOE_INTER * 2], pl.FP32],
    shared_down: pl.Tensor[[EP_SIZE, MOE_INTER, D], pl.INT8],
    shared_down_scale: pl.Tensor[[EP_SIZE, D], pl.FP32],
    logits: pl.Out[pl.Tensor[[EP_SIZE, T, N_EXPERTS], pl.FP32]],
    route_ids: pl.Out[pl.Tensor[[EP_SIZE, T, TOPK], pl.INT32]],
    route_weights: pl.Out[pl.Tensor[[EP_SIZE, T, TOPK], pl.FP32]],
    output: pl.Out[pl.Tensor[[EP_SIZE, T, D], pl.BF16]],
    active: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """Launch one complete EP2 Decode MoE layer per logical rank."""

    lane_x_buf = pld.alloc_window_buffer([LANE_ROWS, D], dtype=pl.INT8)
    lane_aux_buf = pld.alloc_window_buffer([LANE_ROWS, AUX_PAD], dtype=pl.FP32)
    lane_count_buf = pld.alloc_window_buffer([EP_SIZE, COUNT_PAD], dtype=pl.INT32)
    lane_ready_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
    slot_out_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.FP32)
    return_ready_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(EP_SIZE):
        lane_x = pld.window(lane_x_buf, [LANE_ROWS, D], dtype=pl.INT8)
        lane_aux = pld.window(lane_aux_buf, [LANE_ROWS, AUX_PAD], dtype=pl.FP32)
        lane_count = pld.window(lane_count_buf, [EP_SIZE, COUNT_PAD], dtype=pl.INT32)
        lane_ready = pld.window(lane_ready_buf, [EP_SIZE, 1], dtype=pl.INT32)
        slot_out = pld.window(slot_out_buf, [N_ROUTES, D], dtype=pl.FP32)
        return_ready = pld.window(return_ready_buf, [EP_SIZE, 1], dtype=pl.INT32)
        decode_ep2_rank(
            value[rank], router_value[rank], gate_weight[rank], text_bias[rank], vision_bias[rank], input_ids[rank],
            routed_w1[rank], routed_w1_scale[rank], routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank], shared_gate_up[rank], shared_gate_up_scale[rank],
            shared_down[rank], shared_down_scale[rank], lane_x, lane_aux, lane_count, lane_ready,
            slot_out, return_ready, logits[rank], route_ids[rank], route_weights[rank], output[rank],
            active, rank, epoch,
            device=rank,
        )


def product_contract() -> dict[str, object]:
    """Return the fixed EP2 product ABI without claiming a run."""

    return {
        "status": "implementation_ready_not_run",
        "topology": {"dp": 2, "tp": 1, "ep": 2, "pp": 1, "pcp": 1, "dcp": 1},
        "routed_experts": N_EXPERTS,
        "local_experts_per_rank": N_LOCAL_EXPERTS,
        "top_k": TOPK,
        "capacity_tokens_per_rank": T,
        "route_lane_capacity": LANE_CAP,
        "receive_capacity_per_local_expert": RECV_MAX,
        "communication": ["remote_put", "notify", "wait", "return_put", "combine"],
        "product_compute": "PyPTO only",
        "reference_process": "independent vLLM-Ascend process",
        "epoch": "shared monotonic decode-step signal",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", action="store_true", help="print the EP2 product ABI")
    parser.add_argument("--compile-only", action="store_true", help="reserved for the queued PyPTO compile harness")
    args = parser.parse_args(argv)
    if args.contract or args.compile_only:
        import json

        print(json.dumps(product_contract(), indent=2, sort_keys=True))
        return 0
    parser.error("EP2 Decode requires the frozen checkpoint/input/cache contract; use --contract for the ABI")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "D",
    "EP_SIZE",
    "MOE_INTER",
    "N_EXPERTS",
    "N_LOCAL_EXPERTS",
    "T",
    "TOPK",
    "decode_ep2",
    "decode_ep2_arrive",
    "decode_ep2_combine",
    "decode_ep2_pack",
    "decode_ep2_rank",
    "decode_ep2_regroup",
    "decode_ep2_return",
    "expert_routed_inline",
    "expert_routed_tile",
    "make_dispatch_layout",
    "make_moe_router",
    "make_moe_router_inline",
    "make_routed_expert_projection",
    "make_routed_expert_w8a8",
    "make_route_restore",
    "make_shared_expert_projection",
    "make_shared_expert_w8a8",
    "make_shared_moe_w8a8",
    "make_shared_moe_w8a8_inline",
    "make_w8a8_expert_projection",
    "product_contract",
    "routed_expert_reference",
    "shared_expert_reference",
    "w8a8_expert_linear",
]
