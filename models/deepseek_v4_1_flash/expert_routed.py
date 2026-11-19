# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""DeepSeek-V4.1 routed expert compute with distributed scatter."""

import argparse
from math import gcd
from pathlib import Path

import torch
import pypto.language as pl
import pypto.language.distributed as pld
from pypto.runtime import pto_isa_include_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--ep", type=int, choices=[2, 4, 8], default=2)
    parser.add_argument("--experts-per-rank", type=int, default=4)
    parser.add_argument("--case", choices=["all", "bounds", "tails", "zero", "full", "balanced"], default="all")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    args = parser.parse_args()
    from dataclasses import replace

    import config

    test_experts = args.ep * args.experts_per_rank
    if args.experts_per_rank <= 0 or not config.FLASH.num_experts_per_tok <= test_experts <= 384:
        parser.error("--experts-per-rank must give between TopK and 384 total routed experts")
    config.FLASH = replace(config.FLASH, n_routed_experts=test_experts)
    config.N_LOCAL_EXPERTS = args.experts_per_rank


from config import FLASH as M
from config import EP_SIZE as EP_WORLD_SIZE, MOE_RECV_MAX as RECV_MAX
from config import N_LOCAL_EXPERTS, MOE_TOKENS

ROUTED_CUBE_WORKER_TILE = 28 if MOE_TOKENS > 80 else 32
ROUTED_VECTOR_WORKER_TILE = 32 if MOE_TOKENS > 80 else 64
from quant import pack_mxfp4_weight_tiles, unpack_mxfp4_weight_tiles


# model config
D = M.hidden_size
TOPK = M.num_experts_per_tok
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit
MX_GROUP = 32
K_SCALE = D // MX_GROUP
DISPATCH_PACKET_WIDTH = D + K_SCALE
H_SCALE = MOE_INTER // MX_GROUP

# Routed-expert across-layer mean dequantization standard deviations.
ROUTED_DEQUANT_STD = {"w1": 2.47e-2, "w2": 2.44e-2, "w3": 2.46e-2}

# tiling
RECV_ROW_TILE = 16
RECV_SCALE_TMP_TILE = ((64 + K_SCALE + 31) // 32) * 32
GATHER_WORKER_TILE = 48
MX_M_TILE = 16
MX_K_TILE = 256
MX_MM_INTER_TILE = 256
MX_W2_D_OUT_TILE = 256
MX_PACKED_LANE_ROWS = MX_K_TILE // 2
MX_PACKED_LANE_COLS = MX_MM_INTER_TILE // 2
MX_W1_PACKED_ROWS = (2 * (D // MX_K_TILE) * (MOE_INTER // MX_MM_INTER_TILE)) * MX_PACKED_LANE_ROWS
MX_W3_PACKED_ROWS = (2 * (D // MX_K_TILE) * (MOE_INTER // MX_MM_INTER_TILE)) * MX_PACKED_LANE_ROWS
MX_W2_PACKED_ROWS = (2 * (MOE_INTER // MX_K_TILE) * (D // MX_W2_D_OUT_TILE)) * MX_PACKED_LANE_ROWS

# SwiGLU activation and group-32 MXFP8 quantization.
SWIGLU_FP8_MAX = 448.0
SWIGLU_MAX_INV = 1.0 / SWIGLU_FP8_MAX
SWIGLU_SCALE_FLOOR = 1e-4
SWIGLU_SCALE_COLS = MOE_INTER // MX_GROUP
SWIGLU_ROW_TILE = 8
SWIGLU_GROUP_ROWS = SWIGLU_ROW_TILE * SWIGLU_SCALE_COLS
SWIGLU_LANES = 64
SWIGLU_GROUPS_PER_CHUNK = SWIGLU_LANES // MX_GROUP
SWIGLU_COUNT_LANES = SWIGLU_LANES
SWIGLU_SCALE_SCRATCH_TILE = SWIGLU_ROW_TILE * (MOE_INTER // SWIGLU_LANES) * 8

# Full physical activation-scale row tiles with a clamped capacity tail.
W13_M_TILE = min(64, gcd(RECV_MAX, 160))
W13_TAIL_M_TILE = RECV_MAX % W13_M_TILE or W13_M_TILE
W13_N_TILE = MX_MM_INTER_TILE
W13_K_TILE = MX_K_TILE
W13_SCALE_K = W13_K_TILE // MX_GROUP
W13_PACKED_EXPERT_ROWS = MX_W1_PACKED_ROWS
FP4_LUT_SIZE = 256


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
):
    """Gather source lanes with expert-partitioned workers and pack MX scales."""
    x_flat = pl.reshape(x_out, [local_experts * RECV_MAX, D])
    count_routes = pl.tensor.view(recv_routes, [local_experts * EP_WORLD_SIZE, max_per_source * 8])
    with pl.spmd(GATHER_WORKER_TILE, name_hint="dispatch_gather", deps=[wait_ready, publish_ready]):
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
            if expert_lane == 0:
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
    return x_out


@pl.jit.inline(auto_scope=False)
def push_dispatch_payload(
    indices: pl.Tensor,
    payload: pl.Tensor,
    packed_scale: pl.Tensor,
    weights: pl.Tensor,
    recv_x: pld.DistributedTensor,
    recv_routes: pld.DistributedTensor,
    num_tokens: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    input_ready: pl.Scalar[pl.TASK_ID],
    local_experts: pl.constexpr = N_LOCAL_EXPERTS,
    topk: pl.constexpr = TOPK,
    dispatch_worker_limit: pl.constexpr = 0,
):
    """Send each expert lane directly from FP8 inputs and paired MX scale tiles."""
    workers = EP_WORLD_SIZE * local_experts
    worker_tile = pl.min(workers, 64)
    if dispatch_worker_limit > 0:
        worker_tile = pl.min(worker_tile, dispatch_worker_limit)
    scale_nd = pl.create_tensor([MOE_TOKENS, K_SCALE], dtype=pl.UINT8)
    with pl.spmd(pl.min((MOE_TOKENS + 15) // 16, 64), name_hint="dispatch_scale_unpack", deps=[input_ready]) as scale_ready:
        for scale_begin in pl.range(pl.tile.get_block_idx() * 16, MOE_TOKENS, 64 * 16):
            scales = pl.load(packed_scale, [0, scale_begin * K_SCALE], [1, 16 * K_SCALE])
            scale_bytes = pl.reinterpret_view(scales, pl.UINT8)
            pairs = pl.reinterpret_view(scale_bytes, pl.UINT16, shape=[K_SCALE // 2, 16])
            pairs_nd = pl.transpose(pairs, 0, 1)
            bytes_nd = pl.reinterpret_view(pairs_nd, pl.UINT8, shape=[16, K_SCALE])
            scale_nd = pl.store(bytes_nd, [scale_begin, 0], scale_nd)
    with pl.spmd(worker_tile, name_hint="dispatch_push", deps=[input_ready, scale_ready], allow_early_resolve=False) as ready:
        core = pl.tile.get_block_idx()
        active = pl.min(pl.max(pl.cast(num_tokens, pl.INDEX), 0), MOE_TOKENS)
        ids_loaded = pl.load(indices, [0, 0], [MOE_TOKENS, 8], valid_shape=[active, topk])
        ids_padded = pl.set_validshape(ids_loaded, MOE_TOKENS, 8)
        flat_ids = pl.reshape(ids_padded, [1, MOE_TOKENS * 8])
        route_weights = pl.load(weights, [0, 0], [MOE_TOKENS, 8], valid_shape=[MOE_TOKENS, topk])
        weight_bits = pl.reinterpret_view(route_weights, pl.INT32)
        for block in pl.range(core, workers, worker_tile):
            peer = block // local_experts
            expert = block % local_experts
            base = expert * RECV_MAX + rank * MOE_TOKENS
            route_lane = pl.create_tile([MOE_TOKENS, 8], dtype=pl.INT32)
            matched = pl.tile.cmps(flat_ids, pl.cast(block, pl.INT32), cmp_type=0)
            # Each 32-bit mask word contains four eight-lane token predicates.
            mask_words = pl.reinterpret_view(matched, pl.UINT32, shape=[1, ((MOE_TOKENS + 31) // 32) * 8])
            cursor = pl.array.create(1, pl.INT32)
            cursor[0] = 0
            payload_batch = pl.create_tile([8, DISPATCH_PACKET_WIDTH], dtype=pl.INT8)
            for word in pl.range((active + 3) // 4):
                word_bits = pl.tile.read(mask_words, [0, word])
                bits = word_bits & pl.cast(((1 << topk) - 1) * 0x01010101, pl.UINT32)
                if bits != 0:
                    for lane in pl.range(pl.min(4, active - word * 4)):
                        token = word * 4 + lane
                        token_bits = (bits >> pl.cast(lane * 8, pl.UINT32)) & pl.cast(255, pl.UINT32)
                        if token_bits != 0:
                            for top in pl.range(topk):
                                top_bit = pl.cast(1, pl.UINT32) << pl.cast(top, pl.UINT32)
                                if (token_bits & top_bit) != 0:
                                    slot = cursor[0]
                                    cursor[0] = slot + 1
                                    batch_row = slot % 8
                                    values = pl.load(payload, [token, 0], [1, D])
                                    value_bytes = pl.reinterpret_view(values, pl.INT8)
                                    payload_batch = pl.tile.assemble(payload_batch, value_bytes, [batch_row, 0])
                                    scale_row = pl.load(scale_nd, [token, 0], [1, K_SCALE])
                                    scale_bits = pl.reinterpret_view(scale_row, pl.INT8)
                                    payload_batch = pl.tile.assemble(payload_batch, scale_bits, [batch_row, D])
                                    if batch_row == 7:
                                        batch_base = base + slot - 7
                                        pld.tile.remote_store(payload_batch, target=recv_x, peer=peer, offsets=[batch_base, 0])
                                    value = pl.tile.read(weight_bits, [token, top])
                                    pl.tile.write(route_lane, [slot, 2], value)
                                    route = pl.cast(token * topk + top, pl.INT32)
                                    pl.tile.write(route_lane, [slot, 0], route)
            pl.tile.write(route_lane, [0, 1], cursor[0])
            sent_rows = pl.cast(cursor[0], pl.INDEX)
            tail_rows = sent_rows % 8
            if tail_rows > 0:
                tail_base = base + sent_rows - tail_rows
                tail_payload = pl.set_validshape(payload_batch, tail_rows, DISPATCH_PACKET_WIDTH)
                pld.tile.remote_store(tail_payload, target=recv_x, peer=peer, offsets=[tail_base, 0])
            route_valid = pl.set_validshape(route_lane, pl.max(sent_rows, 1), 8)
            pld.tile.remote_store(route_valid, target=recv_routes, peer=peer, offsets=[base, 0])
    return ready


@pl.jit.inline(auto_scope=False)
def push_balanced_dispatch_payload(
    indices: pl.Tensor,
    payload: pl.Tensor,
    packed_scale: pl.Tensor,
    weights: pl.Tensor,
    recv_x: pld.DistributedTensor,
    recv_routes: pld.DistributedTensor,
    num_tokens: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    input_ready: pl.Scalar[pl.TASK_ID],
    local_experts: pl.constexpr = N_LOCAL_EXPERTS,
    topk: pl.constexpr = TOPK,
    dispatch_worker_limit: pl.constexpr = 0,
):
    """Plan expert routes and send balanced contiguous row intervals."""
    experts = EP_WORLD_SIZE * local_experts
    worker_tile = pl.min(experts, 64)
    if dispatch_worker_limit > 0:
        worker_tile = pl.min(worker_tile, dispatch_worker_limit)
    scale_nd = pl.create_tensor([MOE_TOKENS, K_SCALE], dtype=pl.UINT8)
    send_routes = pl.create_tensor([experts * MOE_TOKENS, 8], dtype=pl.INT32)
    send_counts = pl.create_tensor([experts, 8], dtype=pl.INT32)
    with pl.spmd(worker_tile, name_hint="dispatch_plan", deps=[input_ready], allow_early_resolve=False) as plan_ready:
        core = pl.tile.get_block_idx()
        for scale_begin in pl.range(core * 16, MOE_TOKENS, worker_tile * 16):
            scales = pl.load(packed_scale, [0, scale_begin * K_SCALE], [1, 16 * K_SCALE])
            scale_bytes = pl.reinterpret_view(scales, pl.UINT8)
            pairs = pl.reinterpret_view(scale_bytes, pl.UINT16, shape=[K_SCALE // 2, 16])
            pairs_nd = pl.transpose(pairs, 0, 1)
            bytes_nd = pl.reinterpret_view(pairs_nd, pl.UINT8, shape=[16, K_SCALE])
            scale_nd = pl.store(bytes_nd, [scale_begin, 0], scale_nd)
        active = pl.min(pl.max(pl.cast(num_tokens, pl.INDEX), 0), MOE_TOKENS)
        ids = pl.load(indices, [0, 0], [MOE_TOKENS, 8], valid_shape=[active, topk])
        ids_full = pl.set_validshape(ids, MOE_TOKENS, 8)
        flat_ids = pl.reshape(ids_full, [1, MOE_TOKENS * 8])
        input_weights = pl.load(weights, [0, 0], [MOE_TOKENS, 8], valid_shape=[active, topk])
        weight_bits = pl.reinterpret_view(input_weights, pl.INT32)
        for expert_id in pl.range(core, experts, worker_tile):
            route_lane = pl.create_tile([MOE_TOKENS, 8], dtype=pl.INT32)
            matched = pl.tile.cmps(flat_ids, pl.cast(expert_id, pl.INT32), cmp_type=0)
            mask_words = pl.reinterpret_view(matched, pl.UINT32, shape=[1, ((MOE_TOKENS + 31) // 32) * 8])
            cursor = pl.array.create(1, pl.INT32)
            cursor[0] = 0
            for word in pl.range((active + 3) // 4):
                word_bits = pl.tile.read(mask_words, [0, word])
                bits = word_bits & pl.cast(((1 << topk) - 1) * 0x01010101, pl.UINT32)
                if bits != 0:
                    for lane in pl.range(pl.min(4, active - word * 4)):
                        token = word * 4 + lane
                        token_bits = (bits >> pl.cast(lane * 8, pl.UINT32)) & pl.cast(255, pl.UINT32)
                        if token_bits != 0:
                            for top in pl.range(topk):
                                if (token_bits & (pl.cast(1, pl.UINT32) << pl.cast(top, pl.UINT32))) != 0:
                                    slot = cursor[0]
                                    cursor[0] = slot + 1
                                    value = pl.tile.read(weight_bits, [token, top])
                                    pl.tile.write(route_lane, [slot, 0], pl.cast(token * topk + top, pl.INT32))
                                    pl.tile.write(route_lane, [slot, 2], value)
            sent_rows = pl.cast(cursor[0], pl.INDEX)
            pl.tile.write(route_lane, [0, 1], cursor[0])
            route_valid = pl.set_validshape(route_lane, pl.max(sent_rows, 1), 8)
            peer = expert_id // local_experts
            base = expert_id % local_experts * RECV_MAX + rank * MOE_TOKENS
            pld.tile.remote_store(route_valid, target=recv_routes, peer=peer, offsets=[base, 0])
            send_routes = pl.store(route_valid, [expert_id * MOE_TOKENS, 0], send_routes)
            count_value = pl.tile.full([1, 8], dtype=pl.INT32, value=0)
            pl.tile.write(count_value, [0, 0], cursor[0])
            send_counts = pl.store(count_value, [expert_id, 0], send_counts)
    send_workers = 64
    if dispatch_worker_limit > 0:
        send_workers = pl.min(64, dispatch_worker_limit)
    with pl.spmd(send_workers, name_hint="dispatch_push", deps=[plan_ready], allow_early_resolve=False) as ready:
        core = pl.tile.get_block_idx()
        count_values = pl.load(send_counts, [0, 0], [experts, 8])
        total_rows = 0
        for expert_id in pl.range(experts):
            count_read = pl.tile.read(count_values, [expert_id, 0])
            total_rows = total_rows + pl.cast(count_read, pl.INDEX)
        rows_per_core = total_rows // send_workers
        extra_rows = total_rows % send_workers
        core_begin = core * rows_per_core + pl.min(core, extra_rows)
        core_end = core_begin + rows_per_core + pl.min(pl.max(extra_rows - core, 0), 1)
        compact_base = 0
        for expert_id in pl.range(experts):
            count_read = pl.tile.read(count_values, [expert_id, 0])
            count = pl.cast(count_read, pl.INDEX)
            first = pl.max(core_begin - compact_base, 0)
            end = pl.min(core_end - compact_base, count)
            if first < end:
                peer = expert_id // local_experts
                base = expert_id % local_experts * RECV_MAX + rank * MOE_TOKENS
                for batch_begin in pl.range(first, end, 8):
                    rows = pl.min(8, end - batch_begin)
                    route_begin = expert_id * MOE_TOKENS + batch_begin
                    route_values = pl.load(send_routes, [route_begin, 0], [8, 8], valid_shape=[rows, 8])
                    payload_batch = pl.create_tile([8, DISPATCH_PACKET_WIDTH], dtype=pl.INT8)
                    for lane in pl.range(rows):
                        route_read = pl.tile.read(route_values, [lane, 0])
                        token = pl.cast(route_read, pl.INDEX) // topk
                        values = pl.load(payload, [token, 0], [1, D])
                        value_bytes = pl.reinterpret_view(values, pl.INT8)
                        payload_batch = pl.tile.assemble(payload_batch, value_bytes, [lane, 0])
                        scale_row = pl.load(scale_nd, [token, 0], [1, K_SCALE])
                        scale_bits = pl.reinterpret_view(scale_row, pl.INT8)
                        payload_batch = pl.tile.assemble(payload_batch, scale_bits, [lane, D])
                    payload_valid = pl.set_validshape(payload_batch, rows, DISPATCH_PACKET_WIDTH)
                    row = base + batch_begin
                    pld.tile.remote_store(payload_valid, target=recv_x, peer=peer, offsets=[row, 0])
            compact_base = compact_base + count
    return ready


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
):
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
    # Publish one epoch only after every local payload store completes.
    with pl.spmd(EP_WORLD_SIZE, name_hint="dispatch_publish_epoch", deps=[_push_tid], allow_early_resolve=False) as _publish_tid:
        dst = pl.tile.get_block_idx()
        if dst != my_rank:
            pld.system.notify(target=data_arrived, peer=dst, offsets=[my_rank, 0, 0], value=moe_epoch, op=pld.NotifyOp.Set)

    gather_deps = pl.array.create(EP_WORLD_SIZE + 1, pl.TASK_ID)
    for src in pl.range(EP_WORLD_SIZE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_wait", deps=[_push_tid], no_dep_args=[data_arrived]) as peer_wait:
            if src != my_rank:
                pld.system.defer_wait(signal=data_arrived, offsets=[src, 0, 0], expected=moe_epoch, cmp=pld.WaitCmp.Ge)
        gather_deps[src] = peer_wait
    # Gather overwrites the local receive buffer after its previous use.
    gather_deps[EP_WORLD_SIZE] = previous_recv_x_use
    _wait_tid = pl.system.task_dummy(deps=[gather_deps])

    return gather_routed_inputs(
        recv_x, recv_routes, recv_meta_local, recv_meta, recv_count_out,
        recv_x_out, recv_scale_out, recv_weight_out, recv_route_out,
        _wait_tid, _publish_tid, MOE_TOKENS, local_experts=local_experts,
    )


@pl.jit.extern(
    core_type="mixed",
    aic_source=Path(__file__).parent / "kernels" / "expert_routed_w13.cpp",
    aiv_source=Path(__file__).parent / "kernels" / "expert_routed_w13.cpp",
    dual_aiv_dispatch=True, include_dirs=(pto_isa_include_dir(),),
)
def _routed_w13(
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
    experts: pl.Scalar[pl.INDEX],
    stride: pl.Scalar[pl.INDEX],
    width: pl.Scalar[pl.INDEX],
    hidden: pl.Scalar[pl.INDEX],
    activation_rows: pl.Scalar[pl.INDEX],
    row_capacity: pl.constexpr,
) -> tuple[pl.Tensor, pl.Tensor]:
    """Store compact BF16 projections and route weights with the same row order."""
    return pair, compact_weights




@pl.jit.incore(level=pl.Level.AIC)
def routed_w13_nz_cube(x: pl.Tensor, x_scale: pl.Tensor, counts: pl.Tensor, w1: pl.Tensor, w1_scale: pl.Tensor, w3: pl.Tensor, w3_scale: pl.Tensor, fp4_lut: pl.Tensor, pair: pl.InOut[pl.Tensor], route_weights: pl.Tensor, compact_weights: pl.InOut[pl.Tensor], experts: pl.constexpr, stride: pl.constexpr):
    pl.func_attr({'split_aiv': True})
    weight_ring = pl.system.reserve_buffer(name='w13_weight_ring', size=196608)
    pl.system.aic_initialize_pipe(v2c_consumer_buf=weight_ring, dir_mask=2, slot_size=65536, slot_num=3, id=0)
    core = pl.tile.get_block_idx()
    work_base = 0
    active_base = 0
    for expert in pl.range(experts):
        count = pl.cast(pl.read(counts, [expert, 0]), pl.INDEX)
        row_tiles = (count + W13_M_TILE - 1) // W13_M_TILE
        panel_cols = MOE_INTER // W13_N_TILE
        panels = row_tiles * 2 * panel_cols
        first_panel = (core + ROUTED_CUBE_WORKER_TILE - work_base % ROUTED_CUBE_WORKER_TILE) % ROUTED_CUBE_WORKER_TILE
        for panel_id in pl.range(first_panel, panels, ROUTED_CUBE_WORKER_TILE):
            row = panel_id // (2 * panel_cols) * W13_M_TILE
            projection = panel_id // panel_cols % 2
            block = panel_id % panel_cols
            input_row = pl.min(row // 16, (stride - W13_M_TILE) // 16) * 16
            x_row = expert * stride + input_row
            out_row = active_base + row
            valid = pl.min(W13_M_TILE, count - row)
            col = block * W13_N_TILE
            scale_row = expert * K_SCALE
            a_stage0: pl.Tile[[W13_M_TILE, 1024], pl.FP8E4M3FN, pl.MemRef('w13_a_stage', slots=2)[0], pl.Mem.Mat] = pl.load(x, [x_row, 0], [W13_M_TILE, 1024], target_memory=pl.Mem.Mat)
            a0 = pl.tile.slice(a_stage0, [W13_M_TILE, W13_K_TILE], [0, 0], valid_shape=[W13_M_TILE, W13_K_TILE])
            a0_left = pl.move(a0, target_memory=pl.Mem.Left)
            a_stage1: pl.Tile[[W13_M_TILE, 1024], pl.FP8E4M3FN, pl.MemRef('w13_a_stage', slots=2)[1], pl.Mem.Mat] = pl.load(x, [x_row, 1024], [W13_M_TILE, 1024], target_memory=pl.Mem.Mat)
            as0: pl.Tile[[W13_M_TILE, W13_SCALE_K], pl.FP8E8M0, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major, fractal=32)] = pl.system.tpop_from_aiv(shape=[W13_M_TILE, W13_SCALE_K], dtype=pl.FP8E8M0, split=1, id=0)
            as0_scale = pl.move(as0, target_memory=pl.Mem.LeftScale)
            pl.system.tfree_to_aiv(as0, id=0)
            bs0: pl.Tile[[W13_SCALE_K, W13_N_TILE], pl.FP8E8M0, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major, fractal=32)] = pl.system.tpop_from_aiv(shape=[W13_SCALE_K, W13_N_TILE], dtype=pl.FP8E8M0, split=1, id=0)
            b0 = pl.system.tpop_from_aiv(shape=[W13_K_TILE, W13_N_TILE], dtype=pl.FP8E4M3FN, split=1, id=0)
            b0_right = pl.move(b0, target_memory=pl.Mem.Right)
            bs0_scale = pl.move(bs0, target_memory=pl.Mem.RightScale)
            pl.system.tfree_to_aiv(bs0, id=0)
            pl.system.tfree_to_aiv(b0, id=0)
            acc = pl.matmul_mx(a0_left, as0_scale, b0_right, bs0_scale)
            for k in pl.unroll(W13_K_TILE, D, W13_K_TILE):
                if (k // 1024) % 2 == 0:
                    a = pl.tile.slice(a_stage0, [W13_M_TILE, W13_K_TILE], [0, k % 1024], valid_shape=[W13_M_TILE, W13_K_TILE])
                else:
                    a = pl.tile.slice(a_stage1, [W13_M_TILE, W13_K_TILE], [0, k % 1024], valid_shape=[W13_M_TILE, W13_K_TILE])
                a_left = pl.move(a, target_memory=pl.Mem.Left)
                if k % 1024 == 768 and k + 1280 < D:
                    if (k // 1024) % 2 == 0:
                        a_stage0: pl.Tile[[W13_M_TILE, 1024], pl.FP8E4M3FN, pl.MemRef('w13_a_stage', slots=2)[0], pl.Mem.Mat] = pl.load(x, [x_row, k + 1280], [W13_M_TILE, 1024], target_memory=pl.Mem.Mat)
                    else:
                        a_stage1: pl.Tile[[W13_M_TILE, 1024], pl.FP8E4M3FN, pl.MemRef('w13_a_stage', slots=2)[1], pl.Mem.Mat] = pl.load(x, [x_row, k + 1280], [W13_M_TILE, 1024], target_memory=pl.Mem.Mat)
                as_: pl.Tile[[W13_M_TILE, W13_SCALE_K], pl.FP8E8M0, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major, fractal=32)] = pl.system.tpop_from_aiv(shape=[W13_M_TILE, W13_SCALE_K], dtype=pl.FP8E8M0, split=1, id=0)
                as_scale = pl.move(as_, target_memory=pl.Mem.LeftScale)
                pl.system.tfree_to_aiv(as_, id=0)
                bs: pl.Tile[[W13_SCALE_K, W13_N_TILE], pl.FP8E8M0, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major, fractal=32)] = pl.system.tpop_from_aiv(shape=[W13_SCALE_K, W13_N_TILE], dtype=pl.FP8E8M0, split=1, id=0)
                b = pl.system.tpop_from_aiv(shape=[W13_K_TILE, W13_N_TILE], dtype=pl.FP8E4M3FN, split=1, id=0)
                b_right = pl.move(b, target_memory=pl.Mem.Right)
                bs_scale = pl.move(bs, target_memory=pl.Mem.RightScale)
                pl.system.tfree_to_aiv(bs, id=0)
                pl.system.tfree_to_aiv(b, id=0)
                acc = pl.matmul_mx_acc(acc, a_left, as_scale, b_right, bs_scale)
            if row + W13_M_TILE > stride:
                tail_offset = W13_M_TILE - W13_TAIL_M_TILE
                tail = pl.tile.slice(acc, [W13_TAIL_M_TILE, W13_N_TILE], [tail_offset, 0], valid_shape=[valid, W13_N_TILE])
                pair = pl.store(tail, [out_row, projection * MOE_INTER + col], pair)
            else:
                narrowed = pl.tile.slice(acc, [W13_M_TILE, W13_N_TILE], [0, 0], valid_shape=[valid, W13_N_TILE])
                pair = pl.store(narrowed, [out_row, projection * MOE_INTER + col], pair)
        work_base = work_base + panels
        active_base = active_base + count
    return pair


@pl.jit.incore(level=pl.Level.AIV)
def routed_w13_nz_vector(x: pl.Tensor, x_scale: pl.Tensor, counts: pl.Tensor, w1: pl.Tensor, w1_scale: pl.Tensor, w3: pl.Tensor, w3_scale: pl.Tensor, fp4_lut: pl.Tensor, pair: pl.InOut[pl.Tensor], route_weights: pl.Tensor, compact_weights: pl.InOut[pl.Tensor], experts: pl.constexpr, stride: pl.constexpr):
    pl.func_attr({'split_aiv': True, 'dual_aiv_dispatch': True})
    lane = pl.tile.get_subblock_idx()
    peer_ring = pl.system.import_peer_buffer(name='w13_weight_ring', peer_func='routed_w13_nz_cube')
    pl.system.aiv_initialize_pipe(v2c_consumer_buf=peer_ring, dir_mask=2, slot_size=65536, slot_num=3, id=0)
    w1_flat = pl.tensor.view(w1, [experts * W13_PACKED_EXPERT_ROWS, MX_PACKED_LANE_COLS], layout=pl.ND)
    w3_flat = pl.tensor.view(w3, [experts * W13_PACKED_EXPERT_ROWS, MX_PACKED_LANE_COLS], layout=pl.ND)
    w13_a_nd = pl.tensor.view(x_scale, [experts * stride // 16, K_SCALE * 16], layout=pl.ND)
    w1_scale_nd = pl.tensor.view(w1_scale, [MOE_INTER // 16, experts * K_SCALE * 16], layout=pl.ND)
    w3_scale_nd = pl.tensor.view(w3_scale, [MOE_INTER // 16, experts * K_SCALE * 16], layout=pl.ND)
    route_flat = pl.tensor.view(route_weights, [1, experts * stride], layout=pl.ND)
    compact_flat = pl.tensor.view(compact_weights, [1, compact_weights.shape[0]], layout=pl.ND)
    core = pl.tile.get_block_idx()
    work_base = 0
    active_base = 0
    for expert in pl.range(experts):
        count = pl.cast(pl.read(counts, [expert, 0]), pl.INDEX)
        row_tiles = (count + W13_M_TILE - 1) // W13_M_TILE
        panel_cols = MOE_INTER // W13_N_TILE
        panels = row_tiles * 2 * panel_cols
        first_panel = (core + ROUTED_CUBE_WORKER_TILE - work_base % ROUTED_CUBE_WORKER_TILE) % ROUTED_CUBE_WORKER_TILE
        for panel_id in pl.range(first_panel, panels, ROUTED_CUBE_WORKER_TILE):
            row = panel_id // (2 * panel_cols) * W13_M_TILE
            projection = panel_id // panel_cols % 2
            block = panel_id % panel_cols
            out_row = active_base + row
            valid = pl.min(W13_M_TILE, count - row)
            panel = expert * W13_PACKED_EXPERT_ROWS + block * D
            a_scale_resident = pl.load(w13_a_nd, [(expert * stride + pl.min(row, stride - W13_M_TILE) + lane * (W13_M_TILE // 2)) // 16, 0], [W13_M_TILE // 32, K_SCALE * 16], target_memory=pl.Mem.Vec)
            a_scale_resident_codes = pl.reinterpret_view(a_scale_resident, pl.UINT8)
            a_scale_resident_bytes = pl.reinterpret_view(a_scale_resident_codes, pl.INT8)
            if projection == 0:
                b_scale_resident = pl.load(w1_scale_nd, [(block * W13_N_TILE + lane * 128) // 16, expert * K_SCALE * 16], [8, K_SCALE * 16], target_memory=pl.Mem.Vec)
            else:
                b_scale_resident = pl.load(w3_scale_nd, [(block * W13_N_TILE + lane * 128) // 16, expert * K_SCALE * 16], [8, K_SCALE * 16], target_memory=pl.Mem.Vec)
            b_scale_resident_codes = pl.reinterpret_view(b_scale_resident, pl.UINT8)
            b_scale_resident_bytes = pl.reinterpret_view(b_scale_resident_codes, pl.INT8)
            if projection == 0:
                raw0: pl.Tile[[128, 128], pl.UINT8, pl.Mem.Vec] = pl.load(w1_flat, [panel + 0 * W13_K_TILE + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
            else:
                raw0: pl.Tile[[128, 128], pl.UINT8, pl.Mem.Vec] = pl.load(w3_flat, [panel + 0 * W13_K_TILE + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
            if projection == 0:
                raw1: pl.Tile[[128, 128], pl.UINT8, pl.Mem.Vec] = pl.load(w1_flat, [panel + 1 * W13_K_TILE + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
            else:
                raw1: pl.Tile[[128, 128], pl.UINT8, pl.Mem.Vec] = pl.load(w3_flat, [panel + 1 * W13_K_TILE + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
            if projection == 0:
                raw2: pl.Tile[[128, 128], pl.UINT8, pl.Mem.Vec] = pl.load(w1_flat, [panel + 2 * W13_K_TILE + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
            else:
                raw2: pl.Tile[[128, 128], pl.UINT8, pl.Mem.Vec] = pl.load(w3_flat, [panel + 2 * W13_K_TILE + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
            if projection == 0:
                raw3: pl.Tile[[128, 128], pl.UINT8, pl.Mem.Vec] = pl.load(w1_flat, [panel + 3 * W13_K_TILE + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
            else:
                raw3: pl.Tile[[128, 128], pl.UINT8, pl.Mem.Vec] = pl.load(w3_flat, [panel + 3 * W13_K_TILE + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
            for batch in pl.unroll(0, D, 4 * W13_K_TILE):
                fp4_raw = pl.reinterpret_view(raw0, pl.FP4E2M1X2)
                nz0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef('w13_nz', slots=2)[0], pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512)] = pl.cast(fp4_raw, pl.FP8E4M3FN, layout=pl.NZ)
                if batch + 4 * W13_K_TILE < D:
                    if projection == 0:
                        raw0: pl.Tile[[128, 128], pl.UINT8, pl.Mem.Vec] = pl.load(w1_flat, [panel + (batch + 4 * W13_K_TILE) + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
                    else:
                        raw0: pl.Tile[[128, 128], pl.UINT8, pl.Mem.Vec] = pl.load(w3_flat, [panel + (batch + 4 * W13_K_TILE) + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
                a_scale_bytes: pl.Tile[[W13_M_TILE // 32, 128], pl.INT8, pl.MemRef('w13_scale_a_packet'), pl.Mem.Vec] = pl.tile.extract(a_scale_resident_bytes, 0, (batch + 0 * W13_K_TILE) // MX_GROUP * 16, [W13_M_TILE // 32, 128], target_memory=pl.Mem.Vec)
                a_scale_packet: pl.Tile[[W13_M_TILE // 2, 8], pl.FP8E8M0, pl.MemRef('w13_scale_a_packet'), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major, fractal=32)] = pl.tile.create([W13_M_TILE // 2, 8], dtype=pl.FP8E8M0, target_memory=pl.Mem.Vec)
                pl.system.tpush_to_aic(a_scale_packet, split=1, id=0)
                b_scale_bytes: pl.Tile[[8, 128], pl.INT8, pl.MemRef('w13_scale_b_packet'), pl.Mem.Vec] = pl.tile.extract(b_scale_resident_bytes, 0, (expert * K_SCALE + (batch + 0 * W13_K_TILE) // MX_GROUP) * 16 - expert * K_SCALE * 16, [8, 128], target_memory=pl.Mem.Vec)
                b_scale_packet: pl.Tile[[8, 128], pl.FP8E8M0, pl.MemRef('w13_scale_b_packet'), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major, fractal=32)] = pl.tile.create([8, 128], dtype=pl.FP8E8M0, target_memory=pl.Mem.Vec)
                pl.system.tpush_to_aic(b_scale_packet, split=1, id=0)
                pl.system.tpush_to_aic(nz0, split=1, id=0)
                fp4_raw = pl.reinterpret_view(raw1, pl.FP4E2M1X2)
                nz1: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef('w13_nz', slots=2)[1], pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512)] = pl.cast(fp4_raw, pl.FP8E4M3FN, layout=pl.NZ)
                if batch + 5 * W13_K_TILE < D:
                    if projection == 0:
                        raw1: pl.Tile[[128, 128], pl.UINT8, pl.Mem.Vec] = pl.load(w1_flat, [panel + (batch + 5 * W13_K_TILE) + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
                    else:
                        raw1: pl.Tile[[128, 128], pl.UINT8, pl.Mem.Vec] = pl.load(w3_flat, [panel + (batch + 5 * W13_K_TILE) + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
                a_scale_bytes: pl.Tile[[W13_M_TILE // 32, 128], pl.INT8, pl.MemRef('w13_scale_a_packet'), pl.Mem.Vec] = pl.tile.extract(a_scale_resident_bytes, 0, (batch + 1 * W13_K_TILE) // MX_GROUP * 16, [W13_M_TILE // 32, 128], target_memory=pl.Mem.Vec)
                a_scale_packet: pl.Tile[[W13_M_TILE // 2, 8], pl.FP8E8M0, pl.MemRef('w13_scale_a_packet'), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major, fractal=32)] = pl.tile.create([W13_M_TILE // 2, 8], dtype=pl.FP8E8M0, target_memory=pl.Mem.Vec)
                pl.system.tpush_to_aic(a_scale_packet, split=1, id=0)
                b_scale_bytes: pl.Tile[[8, 128], pl.INT8, pl.MemRef('w13_scale_b_packet'), pl.Mem.Vec] = pl.tile.extract(b_scale_resident_bytes, 0, (expert * K_SCALE + (batch + 1 * W13_K_TILE) // MX_GROUP) * 16 - expert * K_SCALE * 16, [8, 128], target_memory=pl.Mem.Vec)
                b_scale_packet: pl.Tile[[8, 128], pl.FP8E8M0, pl.MemRef('w13_scale_b_packet'), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major, fractal=32)] = pl.tile.create([8, 128], dtype=pl.FP8E8M0, target_memory=pl.Mem.Vec)
                pl.system.tpush_to_aic(b_scale_packet, split=1, id=0)
                pl.system.tpush_to_aic(nz1, split=1, id=0)
                fp4_raw = pl.reinterpret_view(raw2, pl.FP4E2M1X2)
                nz0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef('w13_nz', slots=2)[0], pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512)] = pl.cast(fp4_raw, pl.FP8E4M3FN, layout=pl.NZ)
                if batch + 6 * W13_K_TILE < D:
                    if projection == 0:
                        raw2: pl.Tile[[128, 128], pl.UINT8, pl.Mem.Vec] = pl.load(w1_flat, [panel + (batch + 6 * W13_K_TILE) + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
                    else:
                        raw2: pl.Tile[[128, 128], pl.UINT8, pl.Mem.Vec] = pl.load(w3_flat, [panel + (batch + 6 * W13_K_TILE) + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
                a_scale_bytes: pl.Tile[[W13_M_TILE // 32, 128], pl.INT8, pl.MemRef('w13_scale_a_packet'), pl.Mem.Vec] = pl.tile.extract(a_scale_resident_bytes, 0, (batch + 2 * W13_K_TILE) // MX_GROUP * 16, [W13_M_TILE // 32, 128], target_memory=pl.Mem.Vec)
                a_scale_packet: pl.Tile[[W13_M_TILE // 2, 8], pl.FP8E8M0, pl.MemRef('w13_scale_a_packet'), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major, fractal=32)] = pl.tile.create([W13_M_TILE // 2, 8], dtype=pl.FP8E8M0, target_memory=pl.Mem.Vec)
                pl.system.tpush_to_aic(a_scale_packet, split=1, id=0)
                b_scale_bytes: pl.Tile[[8, 128], pl.INT8, pl.MemRef('w13_scale_b_packet'), pl.Mem.Vec] = pl.tile.extract(b_scale_resident_bytes, 0, (expert * K_SCALE + (batch + 2 * W13_K_TILE) // MX_GROUP) * 16 - expert * K_SCALE * 16, [8, 128], target_memory=pl.Mem.Vec)
                b_scale_packet: pl.Tile[[8, 128], pl.FP8E8M0, pl.MemRef('w13_scale_b_packet'), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major, fractal=32)] = pl.tile.create([8, 128], dtype=pl.FP8E8M0, target_memory=pl.Mem.Vec)
                pl.system.tpush_to_aic(b_scale_packet, split=1, id=0)
                pl.system.tpush_to_aic(nz0, split=1, id=0)
                fp4_raw = pl.reinterpret_view(raw3, pl.FP4E2M1X2)
                nz1: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef('w13_nz', slots=2)[1], pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512)] = pl.cast(fp4_raw, pl.FP8E4M3FN, layout=pl.NZ)
                if batch + 7 * W13_K_TILE < D:
                    if projection == 0:
                        raw3: pl.Tile[[128, 128], pl.UINT8, pl.Mem.Vec] = pl.load(w1_flat, [panel + (batch + 7 * W13_K_TILE) + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
                    else:
                        raw3: pl.Tile[[128, 128], pl.UINT8, pl.Mem.Vec] = pl.load(w3_flat, [panel + (batch + 7 * W13_K_TILE) + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
                a_scale_bytes: pl.Tile[[W13_M_TILE // 32, 128], pl.INT8, pl.MemRef('w13_scale_a_packet'), pl.Mem.Vec] = pl.tile.extract(a_scale_resident_bytes, 0, (batch + 3 * W13_K_TILE) // MX_GROUP * 16, [W13_M_TILE // 32, 128], target_memory=pl.Mem.Vec)
                a_scale_packet: pl.Tile[[W13_M_TILE // 2, 8], pl.FP8E8M0, pl.MemRef('w13_scale_a_packet'), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major, fractal=32)] = pl.tile.create([W13_M_TILE // 2, 8], dtype=pl.FP8E8M0, target_memory=pl.Mem.Vec)
                pl.system.tpush_to_aic(a_scale_packet, split=1, id=0)
                b_scale_bytes: pl.Tile[[8, 128], pl.INT8, pl.MemRef('w13_scale_b_packet'), pl.Mem.Vec] = pl.tile.extract(b_scale_resident_bytes, 0, (expert * K_SCALE + (batch + 3 * W13_K_TILE) // MX_GROUP) * 16 - expert * K_SCALE * 16, [8, 128], target_memory=pl.Mem.Vec)
                b_scale_packet: pl.Tile[[8, 128], pl.FP8E8M0, pl.MemRef('w13_scale_b_packet'), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major, fractal=32)] = pl.tile.create([8, 128], dtype=pl.FP8E8M0, target_memory=pl.Mem.Vec)
                pl.system.tpush_to_aic(b_scale_packet, split=1, id=0)
                pl.system.tpush_to_aic(nz1, split=1, id=0)
            if projection == 0 and block == 0 and (lane == 0):
                route_row = expert * stride + row
                route = pl.load(route_flat, [0, route_row], [1, W13_M_TILE], valid_shape=[1, valid])
                compact_flat = pl.store(route, [0, out_row], compact_flat)
        work_base = work_base + panels
        active_base = active_base + count
    return compact_flat


@pl.jit.incore(level=pl.Level.CORE_GROUP)
def routed_w13_nz_group(
    x: pl.Tensor, x_scale: pl.Tensor, counts: pl.Tensor,
    w1: pl.Tensor, w1_scale: pl.Tensor, w3: pl.Tensor, w3_scale: pl.Tensor,
    fp4_lut: pl.Tensor, pair: pl.InOut[pl.Tensor],
    route_weights: pl.Tensor, compact_weights: pl.InOut[pl.Tensor],
    experts: pl.constexpr, stride: pl.constexpr,
):
    pair = routed_w13_nz_cube(x, x_scale, counts, w1, w1_scale, w3, w3_scale, fp4_lut, pair, route_weights, compact_weights, experts, stride)
    compact_weights = routed_w13_nz_vector(x, x_scale, counts, w1, w1_scale, w3, w3_scale, fp4_lut, pair, route_weights, compact_weights, experts, stride)
    return pair, compact_weights


@pl.jit.inline(auto_scope=False)
def routed_w13_pypto(
    x: pl.Tensor,
    x_scale: pl.Tensor,
    counts: pl.Tensor,
    w1: pl.Tensor,
    w1_scale: pl.Tensor,
    w3: pl.Tensor,
    w3_scale: pl.Tensor,
    fp4_lut: pl.Tensor,
    pair: pl.Tensor,
    route_weights: pl.Tensor,
    compact_weights: pl.Tensor,
    input_ready: pl.Scalar[pl.TASK_ID],
    experts: pl.constexpr,
    stride: pl.constexpr,
):
    """Compute compact BF16 gate/up projections with two weight-decoding lanes.

    Model callers retain the extern until native latency reaches parity.
    """
    with pl.spmd(ROUTED_CUBE_WORKER_TILE, name_hint="routed_w13_nz", deps=[input_ready]) as ready:
        pair, compact_weights = routed_w13_nz_group(x, x_scale, counts, w1, w1_scale, w3, w3_scale, fp4_lut, pair, route_weights, compact_weights, experts, stride)
    return pair, compact_weights, ready


@pl.jit.extern(
    core_type="aiv", source=Path(__file__).parent / "kernels" / "expert_routed_swiglu.cpp",
    include_dirs=(pto_isa_include_dir(),),
)
def _routed_swiglu_quant(
    pair: pl.Tensor,
    weights: pl.Tensor,
    counts: pl.Tensor,
    payload: pl.Out[pl.Tensor],
    scales: pl.Out[pl.Tensor],
    experts: pl.Scalar[pl.INDEX],
    stride: pl.Scalar[pl.INDEX],
    hidden: pl.Scalar[pl.INDEX],
    row_capacity: pl.constexpr,
) -> tuple[pl.Tensor, pl.Tensor]:
    """Apply FP32 weighted SwiGLU and upward group-32 MXFP8 quantization."""
    return payload, scales


@pl.jit.inline(auto_scope=False)
def routed_swiglu_quant_pypto(
    pair: pl.Tensor,
    weights: pl.Tensor,
    counts: pl.Tensor,
    payload: pl.Tensor,
    scales: pl.Tensor,
    w13_ready: pl.Scalar[pl.TASK_ID],
    experts: pl.constexpr,
):
    """Apply weighted SwiGLU and group-32 MXFP8 quantization in vector registers."""
    counts_row = pl.reshape(counts, [1, experts])
    with pl.spmd(ROUTED_VECTOR_WORKER_TILE, name_hint="routed_swiglu_quant_pypto", deps=[w13_ready]) as ready:
        core = pl.tile.get_block_idx()
        rows = pl.cast(0, pl.INDEX)
        for first in pl.range(0, experts, SWIGLU_COUNT_LANES):
            remaining = pl.min(SWIGLU_COUNT_LANES, experts - first)
            counts_tile = pl.load(counts_row, [0, first], [1, SWIGLU_COUNT_LANES],
                                  valid_shape=[1, remaining], target_memory=pl.MemorySpace.Vec)
            counts_flat = pl.reshape(counts_tile, [SWIGLU_COUNT_LANES])
            total_tile = pl.tile.create([SWIGLU_COUNT_LANES], dtype=pl.INT32,
                                        target_memory=pl.MemorySpace.Vec)
            with pl.reg_scope():
                count_mask = pl.reg.create_group_mask(remaining, num_groups=1,
                                                      group_size=SWIGLU_COUNT_LANES)
                count_lane = pl.reg.index_gen(0, lanes=SWIGLU_COUNT_LANES, dtype=pl.UINT32)
                count_values = pl.reg.gather(counts_flat, count_lane, count_mask)
                count_total = pl.reg.reduce_sum(count_values, count_mask)
                total_written = pl.reg.store(total_tile, 0, value=count_total)
            rows = rows + pl.cast(pl.tile.read(total_written, [0]), pl.INDEX)
        work_count = (rows + SWIGLU_ROW_TILE - 1) // SWIGLU_ROW_TILE
        for work in pl.range(core, work_count, ROUTED_VECTOR_WORKER_TILE):
            row = work * SWIGLU_ROW_TILE
            gate_tile = pl.load(pair, [row, 0], [SWIGLU_ROW_TILE, MOE_INTER],
                                target_memory=pl.MemorySpace.Vec)
            up_tile = pl.load(pair, [row, MOE_INTER], [SWIGLU_ROW_TILE, MOE_INTER],
                              target_memory=pl.MemorySpace.Vec)
            weight_tile = pl.load(weights, [row, 0], [SWIGLU_ROW_TILE, 1],
                                  target_memory=pl.MemorySpace.Vec)
            gate_flat = pl.reshape(gate_tile, [SWIGLU_ROW_TILE * MOE_INTER])
            up_flat = pl.reshape(up_tile, [SWIGLU_ROW_TILE * MOE_INTER])
            weight_flat = pl.reshape(weight_tile, [SWIGLU_ROW_TILE])
            payload_flat = pl.tile.create([SWIGLU_ROW_TILE * MOE_INTER], dtype=pl.INT8,
                                          target_memory=pl.MemorySpace.Vec)
            # Each pair of exponents occupies one 32-byte aligned scratch slot.
            scale_scratch = pl.tile.create([SWIGLU_SCALE_SCRATCH_TILE], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec)
            code_flat = pl.tile.create([SWIGLU_GROUP_ROWS], dtype=pl.INT8, target_memory=pl.MemorySpace.Vec)
            with pl.reg_scope():
                active = pl.reg.full_mask(lanes=SWIGLU_LANES)
                bias = pl.reg.broadcast(254, lanes=SWIGLU_LANES, dtype=pl.INT32)
                lane_id = pl.reg.index_gen(0, lanes=SWIGLU_LANES, dtype=pl.UINT32)
                lane_span = pl.reg.broadcast(SWIGLU_LANES - 1, lanes=SWIGLU_LANES, dtype=pl.UINT32)
                group_lane = pl.reg.muls(lane_id, MX_GROUP, active)
                code_pick = pl.reg.bitwise_and(group_lane, lane_span, active)
                code_store = pl.reg.active_mask(SWIGLU_GROUPS_PER_CHUNK, lanes=SWIGLU_LANES)
                for lane_row in pl.hardware_loop(SWIGLU_ROW_TILE):
                    route_row = pl.cast(lane_row, pl.UINT32)
                    route_index = pl.reg.broadcast(route_row, lanes=SWIGLU_LANES, dtype=pl.UINT32)
                    route_v = pl.reg.gather(weight_flat, route_index, active)
                    for chunk in pl.hardware_loop(MOE_INTER // SWIGLU_LANES):
                        offset = lane_row * MOE_INTER + chunk * SWIGLU_LANES
                        gate_bf = pl.reg.load(gate_flat, offset, lanes=SWIGLU_LANES)
                        gate = pl.reg.convert(gate_bf, dtype=pl.FP32)
                        up_bf = pl.reg.load(up_flat, offset, lanes=SWIGLU_LANES)
                        up = pl.reg.convert(up_bf, dtype=pl.FP32)
                        gate_c = pl.reg.minimums(gate, SWIGLU_LIMIT, active)
                        up_low = pl.reg.minimums(up, SWIGLU_LIMIT, active)
                        up_c = pl.reg.maximums(up_low, -SWIGLU_LIMIT, active)
                        neg_gate = pl.reg.neg(gate_c, active)
                        gate_exp = pl.reg.exp(neg_gate, active)
                        denom = pl.reg.adds(gate_exp, 1.0, active)
                        silu = pl.reg.div(gate_c, denom, active)
                        gated = pl.reg.mul(silu, up_c, active)
                        values = pl.reg.mul(gated, route_v, active)
                        magnitude = pl.reg.abs(values, active)
                        group_peak = pl.reg.reduce_max(magnitude, active, group_size=MX_GROUP)
                        group_max = pl.reg.group_broadcast(group_peak, lanes=SWIGLU_LANES)
                        floored = pl.reg.maximums(group_max, SWIGLU_SCALE_FLOOR, active)
                        amax = pl.reg.muls(floored, SWIGLU_MAX_INV, active)
                        amax_bits = pl.reg.interpret_cast(amax, dtype=pl.INT32)
                        carried = pl.reg.adds(amax_bits, 8388607, active)
                        exponent = pl.reg.shift_rights(carried, 23, active)
                        recip_exp = pl.reg.sub(bias, exponent, active)
                        recip_bits = pl.reg.shift_lefts(recip_exp, 23, active)
                        recip = pl.reg.interpret_cast(recip_bits, dtype=pl.FP32)
                        normalized = pl.reg.mul(values, recip, active)
                        encoded = pl.reg.convert(normalized, dtype=pl.FP8E4M3FN, rounding="R", saturate=True)
                        payload_code = pl.reg.interpret_cast(encoded, dtype=pl.INT8)
                        payload_alias = pl.reg.store(payload_flat, offset, value=payload_code)
                        picked = pl.reg.gather_lanes(exponent, code_pick)
                        code_offset = (lane_row * (MOE_INTER // SWIGLU_LANES) + chunk) * 8
                        code_alias = pl.reg.store(scale_scratch, code_offset, value=picked, mask=code_store)
                # Compact the two populated words from each scratch slot.
                compact_lane = pl.reg.index_gen(0, lanes=SWIGLU_LANES, dtype=pl.UINT32)
                for compact_chunk in pl.hardware_loop(SWIGLU_GROUP_ROWS // SWIGLU_LANES):
                    group_offset = pl.cast(compact_chunk * SWIGLU_LANES, pl.UINT32)
                    compact_group = pl.reg.adds(compact_lane, group_offset, active)
                    scratch_group = pl.reg.shift_rights(compact_group, 1, active)
                    scratch_base = pl.reg.shift_lefts(scratch_group, 3, active)
                    parity_one = pl.reg.broadcast(1, lanes=SWIGLU_LANES, dtype=pl.UINT32)
                    scratch_parity = pl.reg.bitwise_and(compact_group, parity_one, active)
                    scratch_index = pl.reg.add(scratch_base, scratch_parity, active)
                    compact_codes_wide = pl.reg.gather(scale_scratch, scratch_index, active)
                    compact_codes = pl.reg.convert(compact_codes_wide, dtype=pl.INT8, saturate=False)
                    compact_written = pl.reg.store(code_flat, compact_chunk * SWIGLU_LANES, value=compact_codes)
            payload_row = pl.reshape(payload_flat, [SWIGLU_ROW_TILE, MOE_INTER])
            quantized = pl.reinterpret_view(payload_row, pl.FP8E4M3FN)
            payload = pl.store(quantized, [row, 0], payload)
            code_row = pl.reshape(code_flat, [1, SWIGLU_GROUP_ROWS])
            code_bytes = pl.reinterpret_view(code_row, pl.UINT8)
            codes = pl.reinterpret_view(code_bytes, pl.FP8E8M0)
            scales = pl.store(codes, [0, row * SWIGLU_SCALE_COLS], scales)
    return payload, scales, ready


@pl.jit.extern(
    core_type="mixed",
    aic_source=Path(__file__).parent / "kernels" / "expert_routed_w2.cpp",
    aiv_source=Path(__file__).parent / "kernels" / "expert_routed_w2.cpp",
    dual_aiv_dispatch=True, include_dirs=(pto_isa_include_dir(),),
)
def _routed_w2(
    payload: pl.Tensor,
    scales: pl.Tensor,
    counts: pl.Tensor,
    weight: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.Out[pl.Tensor],
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


# Routed W2 projection and ND-to-ZZ activation scale conversion.
W2_M_TILE = min(64, gcd(RECV_MAX, 160))
W2_TAIL_M_TILE = RECV_MAX % W2_M_TILE or W2_M_TILE
W2_HALF_M_TILE = W2_M_TILE // 2
W2_N_TILE = MX_W2_D_OUT_TILE
W2_K_TILE = 256
W2_SCALE_K_TILE = W2_K_TILE // MX_GROUP
W2_K_STAGE = 2
W2_SCALE_ROW_TILE = 32
W2_SCALE_WORD_TILE = W2_SCALE_ROW_TILE // 2
W2_SCALE_RESIDENT_TILE = ((H_SCALE + 31) // 32) * 32
W2_SCALE_RESIDENT_WORD_TILE = W2_SCALE_RESIDENT_TILE // 2
W2_SCALE_INDEX_TILE = W2_HALF_M_TILE * (W2_SCALE_K_TILE // 2)
W2_SCALE_TMP_TILE = ((64 + (W2_HALF_M_TILE // 16) * W2_SCALE_K_TILE + 31) // 32) * 32


@pl.jit.inline(auto_scope=False)
def _w2_pack_scale_packet(
    scale_source: pl.Tile[[1, W2_HALF_M_TILE * W2_SCALE_RESIDENT_WORD_TILE], pl.INT16, pl.Mem.Vec],
    scale_packet_words: pl.Tile[[W2_SCALE_INDEX_TILE], pl.INT16, pl.Mem.Vec],
    word_offset: pl.Scalar[pl.UINT16],
):
    """Pack the current activation-scale packet into its physical ZZ order."""
    source_flat = pl.reshape(scale_source, [W2_HALF_M_TILE * W2_SCALE_RESIDENT_WORD_TILE])
    words_view = pl.reshape(scale_packet_words, [W2_SCALE_INDEX_TILE])
    with pl.reg_scope():
        active_scale = pl.reg.full_mask(lanes=W2_SCALE_INDEX_TILE)
        physical = pl.reg.index_gen(0, lanes=W2_SCALE_INDEX_TILE, dtype=pl.UINT16)
        row_block = pl.reg.shift_rights(physical, 6, active_scale)
        row_high = pl.reg.shift_lefts(row_block, 4, active_scale)
        inner_mask = pl.reg.broadcast(15, lanes=W2_SCALE_INDEX_TILE, dtype=pl.UINT16)
        row_inner = pl.reg.bitwise_and(physical, inner_mask, active_scale)
        row = pl.reg.add(row_high, row_inner, active_scale)
        row_base = pl.reg.muls(row, W2_SCALE_RESIDENT_WORD_TILE, active_scale)
        pairs = pl.reg.shift_rights(physical, 4, active_scale)
        pair_mask = pl.reg.broadcast(3, lanes=W2_SCALE_INDEX_TILE, dtype=pl.UINT16)
        pair = pl.reg.bitwise_and(pairs, pair_mask, active_scale)
        base = pl.reg.add(row_base, pair, active_scale)
        source_indices = pl.reg.adds(base, word_offset, active_scale)
        scale_codes = pl.reg.gather(source_flat, source_indices, active_scale)
        scale_stored = pl.reg.store(words_view, 0, scale_codes, active_scale)
    return scale_stored


@pl.jit.incore(level=pl.Level.AIC)
def routed_w2_nz_cube(payload: pl.Tensor, scales: pl.Tensor, counts: pl.Tensor, weight: pl.Tensor, weight_scale: pl.Tensor, fp4_lut: pl.Tensor, output: pl.InOut[pl.Tensor], experts: pl.constexpr, stride: pl.constexpr):
    pl.func_attr({'split_aiv': True})
    weight_ring = pl.system.reserve_buffer(name='w2_weight_ring', size=196608)
    pl.system.aic_initialize_pipe(v2c_consumer_buf=weight_ring, dir_mask=2, slot_size=65536, slot_num=3, id=0)
    core = pl.tile.get_block_idx()
    compact = 0
    work_base = 0
    for expert in pl.range(experts):
        count = pl.cast(pl.read(counts, [expert, 0]), pl.INDEX)
        row_tiles = (count + W2_M_TILE - 1) // W2_M_TILE
        work_count = row_tiles * (D // W2_N_TILE)
        first = (core + ROUTED_CUBE_WORKER_TILE - work_base % ROUTED_CUBE_WORKER_TILE) % ROUTED_CUBE_WORKER_TILE
        for work in pl.range(first, work_count, ROUTED_CUBE_WORKER_TILE):
            row_tile = work // (D // W2_N_TILE)
            block = work % (D // W2_N_TILE)
            row = row_tile * W2_M_TILE
            input_row = pl.min(row, stride - W2_M_TILE)
            valid = pl.min(W2_M_TILE, count - row)
            col = block * W2_N_TILE
            a0 = pl.load(payload, [compact + input_row, 0], [W2_M_TILE, W2_K_TILE], target_memory=pl.Mem.Mat)
            a0_left = pl.move(a0, target_memory=pl.Mem.Left)
            as0: pl.Tile[[W2_M_TILE, W2_SCALE_K_TILE], pl.FP8E8M0, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major, fractal=32)] = pl.system.tpop_from_aiv(shape=[W2_M_TILE, W2_SCALE_K_TILE], dtype=pl.FP8E8M0, split=1, id=0)
            as0_left = pl.move(as0, target_memory=pl.Mem.LeftScale)
            pl.system.tfree_to_aiv(as0, id=0)
            bs0: pl.Tile[[W2_SCALE_K_TILE, W2_N_TILE], pl.FP8E8M0, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major, fractal=32)] = pl.system.tpop_from_aiv(shape=[W2_SCALE_K_TILE, W2_N_TILE], dtype=pl.FP8E8M0, split=1, id=0)
            b0 = pl.system.tpop_from_aiv(shape=[W2_K_TILE, W2_N_TILE], dtype=pl.FP8E4M3FN, split=1, id=0)
            b0_right = pl.move(b0, target_memory=pl.Mem.Right)
            bs0_scale = pl.move(bs0, target_memory=pl.Mem.RightScale)
            pl.system.tfree_to_aiv(bs0, id=0)
            pl.system.tfree_to_aiv(b0, id=0)
            acc = pl.matmul_mx(a0_left, as0_left, b0_right, bs0_scale)
            for k in pl.unroll(W2_K_TILE, MOE_INTER, W2_K_TILE):
                a = pl.load(payload, [compact + input_row, k], [W2_M_TILE, W2_K_TILE], target_memory=pl.Mem.Mat)
                a_left = pl.move(a, target_memory=pl.Mem.Left)
                as_: pl.Tile[[W2_M_TILE, W2_SCALE_K_TILE], pl.FP8E8M0, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major, fractal=32)] = pl.system.tpop_from_aiv(shape=[W2_M_TILE, W2_SCALE_K_TILE], dtype=pl.FP8E8M0, split=1, id=0)
                as_left = pl.move(as_, target_memory=pl.Mem.LeftScale)
                pl.system.tfree_to_aiv(as_, id=0)
                bs: pl.Tile[[W2_SCALE_K_TILE, W2_N_TILE], pl.FP8E8M0, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major, fractal=32)] = pl.system.tpop_from_aiv(shape=[W2_SCALE_K_TILE, W2_N_TILE], dtype=pl.FP8E8M0, split=1, id=0)
                b = pl.system.tpop_from_aiv(shape=[W2_K_TILE, W2_N_TILE], dtype=pl.FP8E4M3FN, split=1, id=0)
                b_right = pl.move(b, target_memory=pl.Mem.Right)
                bs_scale = pl.move(bs, target_memory=pl.Mem.RightScale)
                pl.system.tfree_to_aiv(bs, id=0)
                pl.system.tfree_to_aiv(b, id=0)
                acc = pl.matmul_mx_acc(acc, a_left, as_left, b_right, bs_scale)
            if row + W2_M_TILE > stride:
                tail_offset = W2_M_TILE - W2_TAIL_M_TILE
                tail_acc = pl.tile.slice(acc, [W2_TAIL_M_TILE, W2_N_TILE], [tail_offset, 0], valid_shape=[valid, W2_N_TILE])
                output = pl.store(tail_acc, [compact + row, col], output)
            else:
                valid_acc = pl.tile.slice(acc, [W2_M_TILE, W2_N_TILE], [0, 0], valid_shape=[valid, W2_N_TILE])
                output = pl.store(valid_acc, [compact + row, col], output)
        work_base = work_base + work_count
        compact = compact + count
    return output


@pl.jit.incore(level=pl.Level.AIV)
def routed_w2_nz_vector(payload: pl.Tensor, scales: pl.Tensor, counts: pl.Tensor, weight: pl.Tensor, weight_scale: pl.Tensor, fp4_lut: pl.Tensor, output: pl.InOut[pl.Tensor], experts: pl.constexpr, stride: pl.constexpr):
    pl.func_attr({'split_aiv': True, 'dual_aiv_dispatch': True})
    lane = pl.tile.get_subblock_idx()
    peer_ring = pl.system.import_peer_buffer(name='w2_weight_ring', peer_func='routed_w2_nz_cube')
    pl.system.aiv_initialize_pipe(v2c_consumer_buf=peer_ring, dir_mask=2, slot_size=65536, slot_num=3, id=0)
    scale_fp8: pl.Tile[[W2_HALF_M_TILE, W2_SCALE_K_TILE], pl.FP8E8M0, pl.MemRef('w2_scale_packet'), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major, fractal=32)] = pl.tile.create([W2_HALF_M_TILE, W2_SCALE_K_TILE], dtype=pl.FP8E8M0, target_memory=pl.Mem.Vec)
    scale_packet_codes = pl.reinterpret_view(scale_fp8, pl.UINT8)
    scale_packet_bytes = pl.reshape(scale_packet_codes, [W2_HALF_M_TILE * W2_SCALE_K_TILE])
    scale_packet_words = pl.reinterpret_view(scale_packet_bytes, pl.INT16)
    weight_flat = pl.tensor.view(weight, [experts * MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], layout=pl.ND)
    scale_nd = pl.tensor.view(scales, [payload.shape[0], H_SCALE], layout=pl.ND)
    w2_bscale_nd = pl.tensor.view(weight_scale, [D // 16, experts * H_SCALE * 16], layout=pl.ND)
    core = pl.tile.get_block_idx()
    compact = 0
    work_base = 0
    for expert in pl.range(experts):
        count = pl.cast(pl.read(counts, [expert, 0]), pl.INDEX)
        row_tiles = (count + W2_M_TILE - 1) // W2_M_TILE
        work_count = row_tiles * (D // W2_N_TILE)
        first = (core + ROUTED_CUBE_WORKER_TILE - work_base % ROUTED_CUBE_WORKER_TILE) % ROUTED_CUBE_WORKER_TILE
        for work in pl.range(first, work_count, ROUTED_CUBE_WORKER_TILE):
            row_tile = work // (D // W2_N_TILE)
            block = work % (D // W2_N_TILE)
            row = row_tile * W2_M_TILE
            input_row = pl.min(row, stride - W2_M_TILE)
            col = block * W2_N_TILE
            panel = expert * MX_W2_PACKED_ROWS + block * MOE_INTER
            scale_row = compact + input_row + lane * W2_HALF_M_TILE
            scale_resident = pl.load(scale_nd, [scale_row, 0], [W2_HALF_M_TILE, W2_SCALE_RESIDENT_TILE], valid_shape=[W2_HALF_M_TILE, H_SCALE], target_memory=pl.Mem.Vec)
            scale_resident_codes = pl.reinterpret_view(scale_resident, pl.UINT8)
            scale_resident_padded = pl.fillpad_expand(scale_resident_codes, [W2_HALF_M_TILE, W2_SCALE_RESIDENT_TILE], pad_value=0)
            scale_resident_words = pl.reinterpret_view(scale_resident_padded, pl.INT16)
            scale_source = pl.reshape(scale_resident_words, [1, W2_HALF_M_TILE * W2_SCALE_RESIDENT_WORD_TILE])
            raw0: pl.Tile[[128, 128], pl.UINT8, pl.MemRef('w2_raw', slots=4)[0], pl.Mem.Vec] = pl.load(weight_flat, [panel + 0 + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
            raw1: pl.Tile[[128, 128], pl.UINT8, pl.MemRef('w2_raw', slots=4)[1], pl.Mem.Vec] = pl.load(weight_flat, [panel + W2_K_TILE + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
            raw2: pl.Tile[[128, 128], pl.UINT8, pl.MemRef('w2_raw', slots=4)[2], pl.Mem.Vec] = pl.load(weight_flat, [panel + 2 * W2_K_TILE + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
            raw3: pl.Tile[[128, 128], pl.UINT8, pl.MemRef('w2_raw', slots=4)[3], pl.Mem.Vec] = pl.load(weight_flat, [panel + 3 * W2_K_TILE + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
            scale_word_offset = pl.cast(0, pl.UINT16)
            _scale_packet_ready = _w2_pack_scale_packet(scale_source, scale_packet_words, scale_word_offset)
            fp4_raw = pl.reinterpret_view(raw0, pl.FP4E2M1X2)
            pl.system.tpush_to_aic(scale_fp8, split=1, id=0)
            b_scale_bytes: pl.Tile[[8, 128], pl.FP8E8M0, pl.MemRef('w2_scale_b_packet'), pl.Mem.Vec] = pl.load(w2_bscale_nd, [(col + lane * 128) // 16, (expert * H_SCALE + pl.cast(scale_word_offset, pl.INDEX) * 2) * 16], [8, 128], target_memory=pl.Mem.Vec)
            b_scale_packet: pl.Tile[[8, 128], pl.FP8E8M0, pl.MemRef('w2_scale_b_packet'), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major, fractal=32)] = pl.tile.create([8, 128], dtype=pl.FP8E8M0, target_memory=pl.Mem.Vec)
            pl.system.tpush_to_aic(b_scale_packet, split=1, id=0)
            nz0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef('w2_nz', slots=2)[0], pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512)] = pl.cast(fp4_raw, pl.FP8E4M3FN, layout=pl.NZ)
            raw0: pl.Tile[[128, 128], pl.UINT8, pl.MemRef('w2_raw', slots=4)[0], pl.Mem.Vec] = pl.load(weight_flat, [panel + 4 * W2_K_TILE + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
            pl.system.tpush_to_aic(nz0, split=1, id=0)
            for batch in pl.unroll(W2_K_TILE, MOE_INTER, 4 * W2_K_TILE):
                scale_word_offset = pl.cast(batch // (2 * MX_GROUP), pl.UINT16)
                _scale_packet_ready = _w2_pack_scale_packet(scale_source, scale_packet_words, scale_word_offset)
                fp4_raw = pl.reinterpret_view(raw1, pl.FP4E2M1X2)
                pl.system.tpush_to_aic(scale_fp8, split=1, id=0)
                b_scale_bytes: pl.Tile[[8, 128], pl.FP8E8M0, pl.MemRef('w2_scale_b_packet'), pl.Mem.Vec] = pl.load(w2_bscale_nd, [(col + lane * 128) // 16, (expert * H_SCALE + pl.cast(scale_word_offset, pl.INDEX) * 2) * 16], [8, 128], target_memory=pl.Mem.Vec)
                b_scale_packet: pl.Tile[[8, 128], pl.FP8E8M0, pl.MemRef('w2_scale_b_packet'), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major, fractal=32)] = pl.tile.create([8, 128], dtype=pl.FP8E8M0, target_memory=pl.Mem.Vec)
                pl.system.tpush_to_aic(b_scale_packet, split=1, id=0)
                nz1: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef('w2_nz', slots=2)[1], pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512)] = pl.cast(fp4_raw, pl.FP8E4M3FN, layout=pl.NZ)
                if batch + 4 * W2_K_TILE < MOE_INTER:
                    raw1: pl.Tile[[128, 128], pl.UINT8, pl.MemRef('w2_raw', slots=4)[1], pl.Mem.Vec] = pl.load(weight_flat, [panel + (batch + 4 * W2_K_TILE) + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
                pl.system.tpush_to_aic(nz1, split=1, id=0)
                scale_word_offset = pl.cast((batch + W2_K_TILE) // (2 * MX_GROUP), pl.UINT16)
                _scale_packet_ready = _w2_pack_scale_packet(scale_source, scale_packet_words, scale_word_offset)
                fp4_raw = pl.reinterpret_view(raw2, pl.FP4E2M1X2)
                pl.system.tpush_to_aic(scale_fp8, split=1, id=0)
                b_scale_bytes: pl.Tile[[8, 128], pl.FP8E8M0, pl.MemRef('w2_scale_b_packet'), pl.Mem.Vec] = pl.load(w2_bscale_nd, [(col + lane * 128) // 16, (expert * H_SCALE + pl.cast(scale_word_offset, pl.INDEX) * 2) * 16], [8, 128], target_memory=pl.Mem.Vec)
                b_scale_packet: pl.Tile[[8, 128], pl.FP8E8M0, pl.MemRef('w2_scale_b_packet'), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major, fractal=32)] = pl.tile.create([8, 128], dtype=pl.FP8E8M0, target_memory=pl.Mem.Vec)
                pl.system.tpush_to_aic(b_scale_packet, split=1, id=0)
                nz0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef('w2_nz', slots=2)[0], pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512)] = pl.cast(fp4_raw, pl.FP8E4M3FN, layout=pl.NZ)
                if batch + 5 * W2_K_TILE < MOE_INTER:
                    raw2: pl.Tile[[128, 128], pl.UINT8, pl.MemRef('w2_raw', slots=4)[2], pl.Mem.Vec] = pl.load(weight_flat, [panel + (batch + 5 * W2_K_TILE) + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
                pl.system.tpush_to_aic(nz0, split=1, id=0)
                scale_word_offset = pl.cast((batch + 2 * W2_K_TILE) // (2 * MX_GROUP), pl.UINT16)
                _scale_packet_ready = _w2_pack_scale_packet(scale_source, scale_packet_words, scale_word_offset)
                fp4_raw = pl.reinterpret_view(raw3, pl.FP4E2M1X2)
                pl.system.tpush_to_aic(scale_fp8, split=1, id=0)
                b_scale_bytes: pl.Tile[[8, 128], pl.FP8E8M0, pl.MemRef('w2_scale_b_packet'), pl.Mem.Vec] = pl.load(w2_bscale_nd, [(col + lane * 128) // 16, (expert * H_SCALE + pl.cast(scale_word_offset, pl.INDEX) * 2) * 16], [8, 128], target_memory=pl.Mem.Vec)
                b_scale_packet: pl.Tile[[8, 128], pl.FP8E8M0, pl.MemRef('w2_scale_b_packet'), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major, fractal=32)] = pl.tile.create([8, 128], dtype=pl.FP8E8M0, target_memory=pl.Mem.Vec)
                pl.system.tpush_to_aic(b_scale_packet, split=1, id=0)
                nz1: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef('w2_nz', slots=2)[1], pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512)] = pl.cast(fp4_raw, pl.FP8E4M3FN, layout=pl.NZ)
                if batch + 6 * W2_K_TILE < MOE_INTER:
                    raw3: pl.Tile[[128, 128], pl.UINT8, pl.MemRef('w2_raw', slots=4)[3], pl.Mem.Vec] = pl.load(weight_flat, [panel + (batch + 6 * W2_K_TILE) + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
                pl.system.tpush_to_aic(nz1, split=1, id=0)
                scale_word_offset = pl.cast((batch + 3 * W2_K_TILE) // (2 * MX_GROUP), pl.UINT16)
                _scale_packet_ready = _w2_pack_scale_packet(scale_source, scale_packet_words, scale_word_offset)
                fp4_raw = pl.reinterpret_view(raw0, pl.FP4E2M1X2)
                pl.system.tpush_to_aic(scale_fp8, split=1, id=0)
                b_scale_bytes: pl.Tile[[8, 128], pl.FP8E8M0, pl.MemRef('w2_scale_b_packet'), pl.Mem.Vec] = pl.load(w2_bscale_nd, [(col + lane * 128) // 16, (expert * H_SCALE + pl.cast(scale_word_offset, pl.INDEX) * 2) * 16], [8, 128], target_memory=pl.Mem.Vec)
                b_scale_packet: pl.Tile[[8, 128], pl.FP8E8M0, pl.MemRef('w2_scale_b_packet'), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major, fractal=32)] = pl.tile.create([8, 128], dtype=pl.FP8E8M0, target_memory=pl.Mem.Vec)
                pl.system.tpush_to_aic(b_scale_packet, split=1, id=0)
                nz0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef('w2_nz', slots=2)[0], pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512)] = pl.cast(fp4_raw, pl.FP8E4M3FN, layout=pl.NZ)
                if batch + 7 * W2_K_TILE < MOE_INTER:
                    raw0: pl.Tile[[128, 128], pl.UINT8, pl.MemRef('w2_raw', slots=4)[0], pl.Mem.Vec] = pl.load(weight_flat, [panel + (batch + 7 * W2_K_TILE) + lane * 128, 0], [128, 128], target_memory=pl.Mem.Vec)
                pl.system.tpush_to_aic(nz0, split=1, id=0)
        work_base = work_base + work_count
        compact = compact + count
    return output


@pl.jit.incore(level=pl.Level.CORE_GROUP)
def routed_w2_nz_group(
    payload: pl.Tensor, scales: pl.Tensor, counts: pl.Tensor,
    weight: pl.Tensor, weight_scale: pl.Tensor, fp4_lut: pl.Tensor,
    output: pl.InOut[pl.Tensor], experts: pl.constexpr, stride: pl.constexpr,
):
    output = routed_w2_nz_cube(payload, scales, counts, weight, weight_scale, fp4_lut, output, experts, stride)
    output = routed_w2_nz_vector(payload, scales, counts, weight, weight_scale, fp4_lut, output, experts, stride)
    return output


@pl.jit.inline(auto_scope=False)
def routed_w2_pypto(
    payload: pl.Tensor, scales: pl.Tensor, counts: pl.Tensor,
    weight: pl.Tensor, weight_scale: pl.Tensor, fp4_lut: pl.Tensor,
    output: pl.Tensor, input_ready: pl.Scalar[pl.TASK_ID], experts: pl.constexpr, stride: pl.constexpr,
):
    with pl.spmd(ROUTED_CUBE_WORKER_TILE, name_hint="routed_w2_nz", deps=[input_ready]) as ready:
        output = routed_w2_nz_group(payload, scales, counts, weight, weight_scale, fp4_lut, output, experts, stride)
    return output, ready


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
    experts: pl.constexpr,
    stride: pl.constexpr,
    width: pl.constexpr,
    hidden: pl.constexpr,
    row_capacity: pl.constexpr,
):
    """Run BF16 W13, FP32 weighted MXFP8 activation, and full-K BF16 W2.

    Counts lie in [0, stride] and sum to
    at most row_capacity. Stride is divisible by 16; width and hidden are
    divisible by 256.
    """
    activation_rows = ((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE
    pair = pl.create_tensor([((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE, 2 * hidden], dtype=pl.BF16)
    compact_weights = pl.create_tensor([((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE, 1], dtype=pl.FP32)
    with pl.spmd(ROUTED_CUBE_WORKER_TILE, name_hint="routed_w13"):
        pair, compact_weights = _routed_w13(
            x, x_scale, counts, w1, w1_scale, w3, w3_scale, pair,
            weights, compact_weights, experts, stride, width, hidden, activation_rows, row_capacity,
        )

    payload = pl.create_tensor([((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE, hidden], dtype=pl.FP8E4M3FN)
    scales = pl.create_tensor([1, ((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE * (hidden // MX_GROUP)], dtype=pl.FP8E8M0)
    with pl.spmd(ROUTED_VECTOR_WORKER_TILE, name_hint="routed_swiglu_quant"):
        payload, scales = _routed_swiglu_quant(
            pair, compact_weights, counts, payload, scales, experts, stride, hidden, row_capacity,
        )

    output_rows = ((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE
    output = pl.create_tensor([((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE, width], dtype=pl.BF16)
    with pl.spmd(ROUTED_CUBE_WORKER_TILE, name_hint="routed_w2"):
        output = _routed_w2(
            payload, scales, counts, w2, w2_scale, output,
            experts, stride, width, hidden, output_rows, activation_rows, row_capacity,
        )
    return output


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
    local_experts: pl.constexpr = N_LOCAL_EXPERTS,
    topk: pl.constexpr = M.num_experts_per_tok,
):
    """Compute active expert rows and scatter each source interval to its route destinations."""
    recv_x_flat = pl.reshape(recv_x, [local_experts * RECV_MAX, D])
    route_weights = pl.reshape(recv_weights, [local_experts * RECV_MAX, 1])
    output = _expert_routed_output(
        recv_x_flat, recv_mx_scale, route_weights, recv_expert_count,
        routed_w1_packed, routed_w1_scale, routed_w3_packed, routed_w3_scale,
        routed_w2_packed, routed_w2_scale,
        local_experts, RECV_MAX, D, MOE_INTER,
        (local_experts if local_experts < topk else topk) * RECV_MAX,
    )
    flat_counts = pl.reshape(recv_expert_count, [1, local_experts])
    metadata_cols = ((local_experts + 7) // 8) * 8
    with pl.spmd(64, name_hint="ffn_scatter") as scatter_done:
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


@pl.jit(auto_scope=False)
def local_compute_test(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.FP8E4M3FN],
    recv_mx_scale: pl.Tensor[[1, N_LOCAL_EXPERTS * RECV_MAX * K_SCALE], pl.FP8E8M0],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1_packed: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w3_packed: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w2_packed: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * H_SCALE, D], pl.FP8E8M0, pl.MX_B_NN],
    routed_y: pl.InOut[pl.Tensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.BF16]],
):
    recv_x_flat = pl.reshape(recv_x, [N_LOCAL_EXPERTS * RECV_MAX, D])
    route_weights = pl.reshape(recv_weights, [N_LOCAL_EXPERTS * RECV_MAX, 1])
    output = _expert_routed_output(
        recv_x_flat, recv_mx_scale, route_weights, recv_expert_count,
        routed_w1_packed, routed_w1_scale, routed_w3_packed, routed_w3_scale,
        routed_w2_packed, routed_w2_scale,
        N_LOCAL_EXPERTS, RECV_MAX, D, MOE_INTER, N_LOCAL_EXPERTS * RECV_MAX,
    )
    with pl.spmd(64, name_hint="ffn_copy"):
        core = pl.tile.get_block_idx()
        work_base = 0
        compact_base = 0
        for expert in pl.range(N_LOCAL_EXPERTS):
            count_read = pl.read(recv_expert_count, [expert, 0])
            count = pl.cast(count_read, pl.INDEX)
            work_count = (count + 1) // 2
            first = (core + 64 - work_base % 64) % 64
            for work in pl.range(first, work_count, 64):
                local_row = work * 2
                row = compact_base + local_row
                col = pl.cast(0, pl.INDEX)
                valid = pl.min(2, count - local_row)
                result = pl.load(output, [row, col], [2, D], valid_shape=[valid, D])
                masked = pl.set_validshape(result, valid, D)
                routed_y = pl.store(masked, [expert * RECV_MAX + local_row, 0], routed_y)
            work_base = work_base + work_count
            compact_base = compact_base + count
    return routed_y


def golden_expert_routed(tensors):
    """Torch reference for MXFP4-grid W1/W3/W2 with MXFP8 activations."""
    import torch

    import torch.nn.functional as F

    from quant import unpack_mx_scale_codes as decode_e8m0_codes, quantize_mxfp8_upward_e8m0
    from gemm import golden_gemm_mxfp8_to_fp32_dequantized as matmul_mx_golden

    recv_x = tensors["recv_x"]
    recv_scale_codes = tensors["recv_mx_scale"].reshape(N_LOCAL_EXPERTS * RECV_MAX, K_SCALE)
    recv_scale_decoded = decode_e8m0_codes(recv_scale_codes, side="a")
    recv_mx_scale = recv_scale_decoded.reshape(N_LOCAL_EXPERTS, RECV_MAX, K_SCALE)
    recv_weights = tensors["recv_weights"].float()
    recv_expert_count = tensors["recv_expert_count"]
    w1_scale = decode_e8m0_codes(tensors['routed_w1_scale'], side='b').reshape(N_LOCAL_EXPERTS, K_SCALE, MOE_INTER)
    w3_scale = decode_e8m0_codes(tensors['routed_w3_scale'], side='b').reshape(N_LOCAL_EXPERTS, K_SCALE, MOE_INTER)
    w2_scale = decode_e8m0_codes(tensors['routed_w2_scale'], side='b').reshape(N_LOCAL_EXPERTS, H_SCALE, D)

    recv_y = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D)
    for e in range(N_LOCAL_EXPERTS):
        n_rows = int(recv_expert_count[e, 0].item())
        if n_rows == 0:
            continue
        x_sub = recv_x[e, :n_rows, :]
        x_scale = recv_mx_scale[e, :n_rows, :]
        w_per_row = recv_weights[e, :n_rows].reshape(-1, 1)
        w1_packed = tensors['routed_w1_packed'][e].reshape(
            2 * (D // MX_K_TILE) * (MOE_INTER // MX_MM_INTER_TILE), MX_K_TILE * MX_MM_INTER_TILE // 4,
        )
        w1_fp8 = unpack_mxfp4_weight_tiles(w1_packed, D, MOE_INTER, MX_K_TILE, MX_MM_INTER_TILE)
        w3_packed = tensors['routed_w3_packed'][e].reshape(
            2 * (D // MX_K_TILE) * (MOE_INTER // MX_MM_INTER_TILE), MX_K_TILE * MX_MM_INTER_TILE // 4,
        )
        w3_fp8 = unpack_mxfp4_weight_tiles(w3_packed, D, MOE_INTER, MX_K_TILE, MX_MM_INTER_TILE)
        w2_packed = tensors['routed_w2_packed'][e].reshape(
            2 * (MOE_INTER // MX_K_TILE) * (D // MX_W2_D_OUT_TILE), MX_K_TILE * MX_W2_D_OUT_TILE // 4,
        )
        w2_fp8 = unpack_mxfp4_weight_tiles(w2_packed, MOE_INTER, D, MX_K_TILE, MX_W2_D_OUT_TILE)

        gate = matmul_mx_golden(x_sub, x_scale, w1_fp8, w1_scale[e]).bfloat16().float()
        up = matmul_mx_golden(x_sub, x_scale, w3_fp8, w3_scale[e]).bfloat16().float()
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h = F.silu(gate) * up * w_per_row
        h_fp8, h_scale_codes = quantize_mxfp8_upward_e8m0(h)
        y = matmul_mx_golden(h_fp8, h_scale_codes, w2_fp8, w2_scale[e]).to(torch.bfloat16).float()
        recv_y[e, :n_rows, :] = y

    tensors["recv_y"][:] = recv_y.to(torch.bfloat16)


def gen_routed_mxfp4_weights(n_experts, dequant_std, seed_base=0):
    """Generate checkpoint-style packed MXFP4 routed weights and scales."""
    from concurrent.futures import ThreadPoolExecutor

    import torch

    from quant import pack_mx_b_scale, quantize_mxfp4_weight

    w1_weights = torch.empty((n_experts, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS), dtype=torch.uint8)
    w3_weights = torch.empty_like(w1_weights)
    w2_weights = torch.empty((n_experts, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS), dtype=torch.uint8)
    w1_scales = torch.empty((n_experts, D // MX_GROUP, MOE_INTER), dtype=torch.uint8)
    w3_scales = torch.empty_like(w1_scales)
    w2_scales = torch.empty((n_experts, MOE_INTER // MX_GROUP, D), dtype=torch.uint8)

    def generate_expert(e):
        seed = seed_base + e * 3
        generator = torch.Generator().manual_seed(seed)
        w1, w1s = quantize_mxfp4_weight(torch.randn(MOE_INTER, D, generator=generator) * dequant_std['w1'])
        w3, w3s = quantize_mxfp4_weight(torch.randn(MOE_INTER, D, generator=generator) * dequant_std['w3'])
        w2, w2s = quantize_mxfp4_weight(torch.randn(D, MOE_INTER, generator=generator) * dequant_std['w2'])
        w1_tiles = pack_mxfp4_weight_tiles(w1, MX_K_TILE, MX_MM_INTER_TILE)
        w1_weights[e] = w1_tiles.reshape(MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS)
        w3_tiles = pack_mxfp4_weight_tiles(w3, MX_K_TILE, MX_MM_INTER_TILE)
        w3_weights[e] = w3_tiles.reshape(MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS)
        w2_tiles = pack_mxfp4_weight_tiles(w2, MX_K_TILE, MX_W2_D_OUT_TILE)
        w2_weights[e] = w2_tiles.reshape(MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS)
        w1_scales[e] = w1s.transpose(0, 1).contiguous().view(torch.uint8)
        w3_scales[e] = w3s.transpose(0, 1).contiguous().view(torch.uint8)
        w2_scales[e] = w2s.transpose(0, 1).contiguous().view(torch.uint8)

    with ThreadPoolExecutor(max_workers=min(4, n_experts)) as pool:
        list(pool.map(generate_expert, range(n_experts)))

    def pack_expert_scales(scales):
        packed = pack_mx_b_scale(scales.flatten(0, 1))
        e8m0 = getattr(torch, "float8_e8m0fnu", None)
        return packed.view(e8m0) if e8m0 is not None else packed

    return (
        w1_weights, pack_expert_scales(w1_scales),
        w3_weights, pack_expert_scales(w3_scales),
        w2_weights, pack_expert_scales(w2_scales),
    )


# validation
OUTPUT_SENTINEL = 37.0


def golden_local_compute(tensors):
    """Fill active local output rows through the production expert golden."""
    values = dict(tensors)
    values["recv_y"] = torch.empty_like(tensors["routed_y"]).reshape(N_LOCAL_EXPERTS, RECV_MAX, D)
    golden_expert_routed(values)
    output = tensors["routed_y"].reshape(N_LOCAL_EXPERTS, RECV_MAX, D)
    for expert, count in enumerate(tensors["recv_expert_count"].reshape(-1).tolist()):
        output[expert, :count] = values["recv_y"][expert, :count]


def compare_local_compute(actual, expected, *, inputs, **kwargs):
    """Compare active expert rows and require exact inactive output guards."""
    from golden import ratio_reldiff

    actual_rows = actual.reshape(N_LOCAL_EXPERTS, RECV_MAX, D)
    expected_rows = expected.reshape_as(actual_rows)
    comparator = ratio_reldiff(diff_thd=2e-3, pct_thd=0.01)
    for expert, count in enumerate(inputs["recv_expert_count"].reshape(-1).tolist()):
        if not torch.equal(actual_rows[expert, count:], expected_rows[expert, count:]):
            return False, f"expert {expert}: inactive output rows changed"
        passed, message = comparator(actual_rows[expert, :count], expected_rows[expert, :count], inputs=inputs, **kwargs)
        if not passed:
            return False, f"expert {expert}, count {count}: {message}"
    return True, ""


def validation_counts(case="bounds", *, tail_offset=0):
    """Return bounded empty, 16-row-tail, and receive-capacity workloads."""
    if case == "zero":
        pattern = (0,)
    elif case == "full":
        pattern = (RECV_MAX,)
    elif case == "balanced":
        pattern = (36,)
    elif case == "tails":
        pattern = (15, 16, 17, 63, 64, 65)
    elif case == "bounds":
        pattern = (0, 1, RECV_MAX - 1, RECV_MAX)
    else:
        raise ValueError(f"Unknown local compute case: {case}")
    values = [min(RECV_MAX, pattern[(expert + tail_offset) % len(pattern)]) for expert in range(N_LOCAL_EXPERTS)]
    return torch.tensor(values, dtype=torch.int32).reshape(N_LOCAL_EXPERTS, 1)


def build_tensor_specs(case="bounds", *, tail_offset=0):
    """Build local compute tensors with poisoned inactive capacity and lazy packed weights."""
    from functools import cache

    from golden import TensorSpec
    from quant import pack_mx_a_scale, quantize_mxfp8_upward_e8m0

    counts = validation_counts(case, tail_offset=tail_offset)

    @cache
    def receive_inputs():
        generator = torch.Generator().manual_seed(20261028)
        source = torch.randn(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.bfloat16, generator=generator)
        payload, codes = quantize_mxfp8_upward_e8m0(source.reshape(-1, D).float())
        payload = payload.reshape(N_LOCAL_EXPERTS, RECV_MAX, D)
        codes = codes.reshape(N_LOCAL_EXPERTS, RECV_MAX, K_SCALE).view(torch.uint8)
        route_weights = torch.rand(N_LOCAL_EXPERTS, RECV_MAX, generator=generator)
        for expert, count in enumerate(counts.reshape(-1).tolist()):
            payload.view(torch.uint8)[expert, count:] = 127
            codes[expert, count:] = 255
            route_weights[expert, count:] = float("nan")
        packed_codes = pack_mx_a_scale(codes.reshape(-1, K_SCALE)).view(torch.float8_e8m0fnu)
        return payload, packed_codes.reshape(1, -1), route_weights

    @cache
    def routed_weights():
        return gen_routed_mxfp4_weights(N_LOCAL_EXPERTS, ROUTED_DEQUANT_STD, seed_base=10)

    fp8 = torch.float8_e4m3fn
    e8m0 = torch.float8_e8m0fnu
    return [
        TensorSpec("recv_x", [N_LOCAL_EXPERTS, RECV_MAX, D], fp8, init_value=lambda: receive_inputs()[0]),
        TensorSpec("recv_mx_scale", [1, N_LOCAL_EXPERTS * RECV_MAX * K_SCALE], e8m0, init_value=lambda: receive_inputs()[1]),
        TensorSpec("recv_weights", [N_LOCAL_EXPERTS, RECV_MAX], torch.float32, init_value=lambda: receive_inputs()[2]),
        TensorSpec("recv_expert_count", [N_LOCAL_EXPERTS, 1], torch.int32, init_value=counts),
        TensorSpec("routed_w1_packed", [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], torch.uint8, init_value=lambda: routed_weights()[0]),
        TensorSpec("routed_w1_scale", [N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], e8m0, init_value=lambda: routed_weights()[1]),
        TensorSpec("routed_w3_packed", [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], torch.uint8, init_value=lambda: routed_weights()[2]),
        TensorSpec("routed_w3_scale", [N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], e8m0, init_value=lambda: routed_weights()[3]),
        TensorSpec("routed_w2_packed", [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], torch.uint8, init_value=lambda: routed_weights()[4]),
        TensorSpec("routed_w2_scale", [N_LOCAL_EXPERTS * H_SCALE, D], e8m0, init_value=lambda: routed_weights()[5]),
        TensorSpec("routed_y", [N_LOCAL_EXPERTS * RECV_MAX, D], torch.bfloat16, init_value=OUTPUT_SENTINEL),
    ]


if __name__ == "__main__":
    from golden import run

    cases = ("bounds", "tails", "zero") if args.case == "all" else (args.case,)
    for case in cases:
        tail_rounds = (6 + N_LOCAL_EXPERTS - 1) // N_LOCAL_EXPERTS if case == "tails" else 1
        for tail_round in range(tail_rounds):
            print(f"Local routed compute: case={case}, tail_round={tail_round}, experts={N_LOCAL_EXPERTS}, stride={RECV_MAX}")
            result = run(
                fn=local_compute_test,
                specs=build_tensor_specs(case, tail_offset=tail_round * N_LOCAL_EXPERTS),
                golden_fn=golden_local_compute,
                golden_data=args.golden_data,
                save_data=args.save_data,
                compile_only=args.compile_only,
                runtime_dir=args.runtime_dir,
                config=dict(
                    dump_passes=args.dump_passes,
                    backend_type=args.backend_type,
                    execution_mode=args.execution_mode,
                    device_id=args.device,
                    enable_chip_swimlane=args.enable_chip_swimlane,
                ),
                rtol=1e-3,
                atol=1e-3,
                compare_fn={"routed_y": compare_local_compute},
            )
            if not result.passed:
                if result.error:
                    print(result.error)
                raise SystemExit(1)
            if args.compile_only or args.golden_data:
                break
        if args.compile_only or args.golden_data:
            break
