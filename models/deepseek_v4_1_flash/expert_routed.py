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
from inspect import signature
from math import gcd

import torch
import pypto.language as pl
import pypto.language.distributed as pld


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--ep", type=int, choices=[2, 4, 8], default=2)
    parser.add_argument("--experts-per-rank", type=int, default=4)
    parser.add_argument("--case", choices=["all", "bounds", "tails", "zero", "full", "balanced", "sparse"], default="all")
    parser.add_argument("--num-tokens", type=int, default=None, help="run validation at one token capacity")
    parser.add_argument("--tail-round", type=int, default=None, help="run only this zero-based round of --case tails")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    args = parser.parse_args()
    if args.tail_round is not None and args.case != "tails":
        parser.error("--tail-round requires --case tails")
    from dataclasses import replace

    import config

    if args.case == "all" and args.num_tokens is None and not (args.runtime_dir or args.golden_data or args.save_data):
        import subprocess
        import sys

        for case, capacity in (("all", config.MOE_TOKENS), ("sparse", 1024)):
            command = [sys.executable, __file__, *sys.argv[1:], "--case", case, "--num-tokens", str(capacity)]
            result = subprocess.run(command, check=False)
            if result.returncode:
                raise SystemExit(result.returncode)
        raise SystemExit(0)

    if args.num_tokens is not None:
        config.set_moe_tokens((args.num_tokens + 15) // 16 * 16)
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
SWIGLU_ROW_TILE = 2 if MOE_TOKENS < 1024 else 8
SWIGLU_GROUP_ROWS = SWIGLU_ROW_TILE * SWIGLU_SCALE_COLS
SWIGLU_LANES = 64
SWIGLU_CODE_TILE = (SWIGLU_GROUP_ROWS + SWIGLU_LANES - 1) // SWIGLU_LANES * SWIGLU_LANES
SWIGLU_GROUPS_PER_CHUNK = SWIGLU_LANES // MX_GROUP
SWIGLU_COUNT_LANES = SWIGLU_LANES
SWIGLU_SCALE_SCRATCH_TILE = SWIGLU_ROW_TILE * (MOE_INTER // SWIGLU_LANES) * 8

# Use the kernel's wide M256 capacity with dynamic valid rows on larger workloads.
W13_M_TILE = 256 if MOE_TOKENS >= 384 else min(64, gcd(RECV_MAX, 160))
W13_A_STAGE_K = 256 if W13_M_TILE > 128 else 1024
W13_SHORT_M_TILE = 64
W13_N_TILE = MX_MM_INTER_TILE
W13_K_TILE = MX_K_TILE
W13_FIFO_SLOT_BYTES = W13_K_TILE * W13_N_TILE
W13_L0_K = 128
W13_L0_SCALE_K = W13_L0_K // MX_GROUP
W13_SHORT_SCALE_STAGE_K = 4096
W13_LONG_SCALE_STAGE_K = 2048
W13_PACKED_LANE_BYTES = MX_PACKED_LANE_ROWS * MX_PACKED_LANE_COLS
W13_PACKED_EXPERT_ROWS = MX_W1_PACKED_ROWS


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


# MX unit flags coordinate a final Cube accumulate with its FIX writeback.
# Older PyPTO revisions expose the same instructions with explicit pipe events.
MX_UNIT_FLAGS = "acc_phase" in signature(pl.tile.matmul_mx_acc).parameters

@pl.jit.inline(auto_scope=False)
def _w13_load_scale_stage(
    x_scale: pl.Tensor,
    w1_scale: pl.Tensor,
    w3_scale: pl.Tensor,
    x_row: pl.Scalar[pl.INDEX],
    compute_m: pl.Scalar[pl.INDEX],
    expert: pl.Scalar[pl.INDEX],
    col: pl.Scalar[pl.INDEX],
    projection: pl.Scalar[pl.INDEX],
    experts: pl.constexpr,
    stride: pl.constexpr,
    m_tile: pl.constexpr,
    scale_offset: pl.constexpr,
    scale_groups: pl.constexpr,
):
    a_scale = pl.tensor.view(x_scale, [experts * stride, K_SCALE], layout=pl.MX_A_ZZ)
    # Mat scale loads cover complete 16-row fractals; L0 extraction uses the actual M.
    a_scale_loaded = pl.load(
        a_scale, [x_row, scale_offset], [m_tile, scale_groups],
        valid_shape=[((compute_m + 15) // 16) * 16, scale_groups], target_memory=pl.Mem.Mat,
    )
    a_scale_stage = pl.set_validshape(a_scale_loaded, compute_m, scale_groups)
    # Weight scales already have MX_B_NN physical order. Gather either
    # projection directly into the same Mat allocation, then reinterpret it
    # as a logical scale tile. This avoids an unsupported Mat-to-Mat scale copy
    # when merging the projection branches.
    raw_b1 = pl.tensor.view(w1_scale, [MOE_INTER // 16, experts * K_SCALE * 16], layout=pl.ND)
    raw_b3 = pl.tensor.view(w3_scale, [MOE_INTER // 16, experts * K_SCALE * 16], layout=pl.ND)
    b_buf: pl.Tile[[W13_N_TILE // 16, scale_groups * 16], pl.FP8E8M0, pl.MemRef("w13_bs_stage"), pl.Mem.Mat] = pl.tile.create(
        [W13_N_TILE // 16, scale_groups * 16], pl.FP8E8M0, target_memory=pl.Mem.Mat, flat_layout=True,
    )
    b_dst = pl.tile.reshape(b_buf, [W13_N_TILE // 16, scale_groups * 16])
    if projection == 0:
        _b_loaded = pl.tile.gather_row(
            b_dst, raw_b1, [0, 0], [col // 16, (expert * K_SCALE + scale_offset) * 16],
            [W13_N_TILE // 16, scale_groups * 16],
        )
    else:
        _b_loaded = pl.tile.gather_row(
            b_dst, raw_b3, [0, 0], [col // 16, (expert * K_SCALE + scale_offset) * 16],
            [W13_N_TILE // 16, scale_groups * 16],
        )
    b_scale_stage: pl.Tile[
        [scale_groups, W13_N_TILE], pl.FP8E8M0,
        pl.MemRef("w13_bs_stage"), pl.Mem.Mat,
        pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major, fractal=32),
    ] = pl.tile.create([scale_groups, W13_N_TILE], pl.FP8E8M0, target_memory=pl.Mem.Mat)
    return a_scale_stage, b_scale_stage


@pl.jit.inline(auto_scope=False)
def _w13_cube_panel(
    x: pl.Tensor,
    x_scale: pl.Tensor,
    w1_scale: pl.Tensor,
    w3_scale: pl.Tensor,
    pair: pl.InOut[pl.Tensor],
    expert: pl.Scalar[pl.INDEX],
    row: pl.Scalar[pl.INDEX],
    projection: pl.Scalar[pl.INDEX],
    block: pl.Scalar[pl.INDEX],
    out_row: pl.Scalar[pl.INDEX],
    valid: pl.Scalar[pl.INDEX],
    experts: pl.constexpr,
    stride: pl.constexpr,
    m_tile: pl.constexpr,
    a_stage_k: pl.constexpr,
    scale_stage_k: pl.constexpr,
    scale_stage1_k: pl.constexpr,
):
    # The row origin stays fixed; dynamic valid M prevents reading or computing a padded tail.
    x_row = pl.max(0, expert * (stride // 16) + row // 16) * 16
    col = block * W13_N_TILE
    compute_m = pl.min(m_tile, valid)
    # Credit 5 protects the L1 input tiles across panels.
    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 5)
    a_stage0 = pl.load(
        x, [x_row, 0], [m_tile, a_stage_k], valid_shape=[compute_m, a_stage_k], target_memory=pl.Mem.Mat,
    )
    as_stage0, bs_stage0 = _w13_load_scale_stage(
        x_scale, w1_scale, w3_scale, x_row, compute_m, expert, col, projection,
        experts, stride, m_tile, 0, scale_stage_k // MX_GROUP,
    )
    # Start extracting the first activation and its scales while MTE2 fills the next stage.
    pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 7)
    a_stage1 = pl.load(
        x, [x_row, a_stage_k], [m_tile, a_stage_k], valid_shape=[compute_m, a_stage_k], target_memory=pl.Mem.Mat,
    )
    as_stage1, bs_stage1 = _w13_load_scale_stage(
        x_scale, w1_scale, w3_scale, x_row, compute_m, expert, col, projection,
        experts, stride, m_tile, scale_stage_k // MX_GROUP, scale_stage1_k // MX_GROUP,
    )
    if scale_stage_k == W13_LONG_SCALE_STAGE_K:
        as_stage2, bs_stage2 = _w13_load_scale_stage(
            x_scale, w1_scale, w3_scale, x_row, compute_m, expert, col, projection,
            experts, stride, m_tile, (D - 1024) // MX_GROUP, 1024 // MX_GROUP,
        )
    else:
        as_stage2, bs_stage2 = as_stage1, bs_stage1
    acc_seed = pl.tile.create([m_tile, W13_N_TILE], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
    acc = pl.set_validshape(acc_seed, pl.min(compute_m, m_tile), W13_N_TILE)
    pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 4)
    for k in pl.unroll(0, D, W13_K_TILE):
        if k == 0:
            pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 7)
        elif k == a_stage_k:
            pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 4)
        elif k >= 2 * a_stage_k and k % a_stage_k == 0:
            pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 2 + (k // a_stage_k) % 2)
        b = pl.system.tpop_from_aiv(shape=[W13_K_TILE, W13_N_TILE], dtype=pl.FP8E4M3FN, split=1, id=0)
        a_offset = k % a_stage_k
        scale_offset = (k % scale_stage_k) // MX_GROUP
        # The first extraction may reuse either L0 address when the expert row tile changes.
        if k == 0:
            pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
            pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
        pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 0)
        if (k // a_stage_k) % 2 == 0:
            a_left0 = pl.tile.extract(a_stage0, 0, a_offset, [m_tile, W13_L0_K], target_memory=pl.Mem.Left)
        else:
            a_left0 = pl.tile.extract(a_stage1, 0, a_offset, [m_tile, W13_L0_K], target_memory=pl.Mem.Left)
        b_right0 = pl.tile.extract(b, 0, 0, [W13_L0_K, W13_N_TILE], target_memory=pl.Mem.Right)
        if k < scale_stage_k:
            as0 = pl.tile.extract(as_stage0, 0, scale_offset, [m_tile, W13_L0_SCALE_K], target_memory=pl.Mem.LeftScale)
            bs0 = pl.tile.extract(bs_stage0, scale_offset, 0, [W13_L0_SCALE_K, W13_N_TILE], target_memory=pl.Mem.RightScale)
        elif k < 2 * scale_stage_k:
            as0 = pl.tile.extract(as_stage1, 0, scale_offset, [m_tile, W13_L0_SCALE_K], target_memory=pl.Mem.LeftScale)
            bs0 = pl.tile.extract(bs_stage1, scale_offset, 0, [W13_L0_SCALE_K, W13_N_TILE], target_memory=pl.Mem.RightScale)
        else:
            as0 = pl.tile.extract(as_stage2, 0, scale_offset, [m_tile, W13_L0_SCALE_K], target_memory=pl.Mem.LeftScale)
            bs0 = pl.tile.extract(bs_stage2, scale_offset, 0, [W13_L0_SCALE_K, W13_N_TILE], target_memory=pl.Mem.RightScale)
        # Keep both K128 halves live to reserve distinct L0 slots. With scale
        # addresses paired to these slots at allocation time, MTE1 and Cube
        # have no intervening scalar address-binding dependency.
        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, 0)
        pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
        if (k // a_stage_k) % 2 == 0:
            a_left1 = pl.tile.extract(a_stage0, 0, a_offset + W13_L0_K, [m_tile, W13_L0_K], target_memory=pl.Mem.Left)
        else:
            a_left1 = pl.tile.extract(a_stage1, 0, a_offset + W13_L0_K, [m_tile, W13_L0_K], target_memory=pl.Mem.Left)
        b_right1 = pl.tile.extract(b, W13_L0_K, 0, [W13_L0_K, W13_N_TILE], target_memory=pl.Mem.Right)
        if k < scale_stage_k:
            as1 = pl.tile.extract(as_stage0, 0, scale_offset + W13_L0_SCALE_K, [m_tile, W13_L0_SCALE_K], target_memory=pl.Mem.LeftScale)
            bs1 = pl.tile.extract(bs_stage0, scale_offset + W13_L0_SCALE_K, 0, [W13_L0_SCALE_K, W13_N_TILE], target_memory=pl.Mem.RightScale)
        elif k < 2 * scale_stage_k:
            as1 = pl.tile.extract(as_stage1, 0, scale_offset + W13_L0_SCALE_K, [m_tile, W13_L0_SCALE_K], target_memory=pl.Mem.LeftScale)
            bs1 = pl.tile.extract(bs_stage1, scale_offset + W13_L0_SCALE_K, 0, [W13_L0_SCALE_K, W13_N_TILE], target_memory=pl.Mem.RightScale)
        else:
            as1 = pl.tile.extract(as_stage2, 0, scale_offset + W13_L0_SCALE_K, [m_tile, W13_L0_SCALE_K], target_memory=pl.Mem.LeftScale)
            bs1 = pl.tile.extract(bs_stage2, scale_offset + W13_L0_SCALE_K, 0, [W13_L0_SCALE_K, W13_N_TILE], target_memory=pl.Mem.RightScale)
        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, 1)
        if k + W13_K_TILE == D:
            pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 5)
        pl.system.tfree_to_aiv(b, id=0)
        pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, 0)
        if k == 0:
            if MX_UNIT_FLAGS:
                acc = pl.tile.matmul_mx(a_left0, as0, b_right0, bs0, acc_phase=pl.AccPhase.Partial)
            else:
                pl.system.wait_flag(pl.PipeType.FIX, pl.PipeType.M, 0)
                acc = pl.matmul_mx(a_left0, as0, b_right0, bs0)
        elif MX_UNIT_FLAGS:
            acc = pl.tile.matmul_mx_acc(acc, a_left0, as0, b_right0, bs0, acc_phase=pl.AccPhase.Partial)
        else:
            acc = pl.matmul_mx_acc(acc, a_left0, as0, b_right0, bs0)
        pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 0)
        pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, 1)
        if MX_UNIT_FLAGS:
            if k + W13_K_TILE == D:
                # FIX consumes the same final MX accumulator before Cube can reuse it.
                acc = pl.tile.matmul_mx_acc(acc, a_left1, as1, b_right1, bs1, acc_phase=pl.AccPhase.Final)
                pair = pl.store(acc, [out_row, projection * MOE_INTER + col], pair, st_phase=pl.STPhase.Final)
            else:
                acc = pl.tile.matmul_mx_acc(acc, a_left1, as1, b_right1, bs1, acc_phase=pl.AccPhase.Partial)
        else:
            acc = pl.matmul_mx_acc(acc, a_left1, as1, b_right1, bs1)
        pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
        if k % a_stage_k == a_stage_k - W13_K_TILE and k + a_stage_k + W13_K_TILE < D:
            pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 2 + (k // a_stage_k) % 2)
            pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 2 + (k // a_stage_k) % 2)
            if (k // a_stage_k) % 2 == 0:
                a_stage0 = pl.load(
                    x, [x_row, k + a_stage_k + W13_K_TILE], [m_tile, a_stage_k], valid_shape=[compute_m, a_stage_k], target_memory=pl.Mem.Mat,
                )
            else:
                a_stage1 = pl.load(
                    x, [x_row, k + a_stage_k + W13_K_TILE], [m_tile, a_stage_k], valid_shape=[compute_m, a_stage_k], target_memory=pl.Mem.Mat,
                )
            pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 2 + (k // a_stage_k) % 2)
    if not MX_UNIT_FLAGS:
        pl.system.set_flag(pl.PipeType.M, pl.PipeType.FIX, 0)
        pl.system.wait_flag(pl.PipeType.M, pl.PipeType.FIX, 0)
        pair = pl.store(acc, [out_row, projection * MOE_INTER + col], pair)
        pl.system.set_flag(pl.PipeType.FIX, pl.PipeType.M, 0)
    return pair


@pl.jit.incore(level=pl.Level.AIC)
def routed_w13_nz_cube(
    x: pl.Tensor,
    x_scale: pl.Tensor,
    counts: pl.Tensor,
    w1: pl.Tensor,
    w1_scale: pl.Tensor,
    w3: pl.Tensor,
    w3_scale: pl.Tensor,
    pair: pl.InOut[pl.Tensor],
    route_weights: pl.Tensor,
    compact_weights: pl.InOut[pl.Tensor],
    experts: pl.constexpr,
    stride: pl.constexpr,
    fifo_slots: pl.constexpr,
    fifo_bytes: pl.constexpr,
    a_stage_k: pl.constexpr,
):
    pl.func_attr({'split_aiv': True})
    weight_ring = pl.system.reserve_buffer(name='w13_weight_ring', size=fifo_bytes)
    pl.system.aic_initialize_pipe(v2c_consumer_buf=weight_ring, dir_mask=2, slot_size=W13_FIFO_SLOT_BYTES, slot_num=fifo_slots, id=0)
    pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 0)
    pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
    if not MX_UNIT_FLAGS:
        pl.system.set_flag(pl.PipeType.FIX, pl.PipeType.M, 0)
    pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 5)
    core = pl.tile.get_block_idx()
    workers = pl.tile.get_block_num()
    work_base = 0
    active_base = 0
    for expert in pl.range(experts):
        count = pl.cast(pl.read(counts, [expert, 0]), pl.INDEX)
        short_rows = MOE_TOKENS >= 384 and count <= W13_SHORT_M_TILE
        row_tile = W13_M_TILE
        if short_rows:
            row_tile = W13_SHORT_M_TILE
        row_tiles = (count + row_tile - 1) // row_tile
        panel_cols = MOE_INTER // W13_N_TILE
        panels = row_tiles * 2 * panel_cols
        first_panel = (core + workers - work_base % workers) % workers
        for panel_id in pl.range(first_panel, panels, workers):
            row = panel_id // (2 * panel_cols) * row_tile
            projection = panel_id // panel_cols % 2
            block = panel_id % panel_cols
            out_row = active_base + row
            valid = pl.min(row_tile, count - row)
            if short_rows:
                pair = _w13_cube_panel(
                    x, x_scale, w1_scale, w3_scale, pair, expert, row, projection,
                    block, out_row, valid, experts, stride, W13_SHORT_M_TILE,
                    1024, W13_SHORT_SCALE_STAGE_K, 1024,
                )
            else:
                pair = _w13_cube_panel(
                    x, x_scale, w1_scale, w3_scale, pair, expert, row, projection,
                    block, out_row, valid, experts, stride, W13_M_TILE,
                    a_stage_k,
                    W13_LONG_SCALE_STAGE_K if W13_M_TILE > 128 else W13_SHORT_SCALE_STAGE_K,
                    2048 if W13_M_TILE > 128 else 1024,
                )
        work_base = work_base + panels
        active_base = active_base + count
    pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 0)
    pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
    if not MX_UNIT_FLAGS:
        pl.system.wait_flag(pl.PipeType.FIX, pl.PipeType.M, 0)
    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 5)
    # Publish the final BF16 store before SwiGLU can read this worker's panels.
    pl.system.set_flag(pl.PipeType.FIX, pl.PipeType.S, 0)
    pl.system.wait_flag(pl.PipeType.FIX, pl.PipeType.S, 0)
    return pair


@pl.jit.inline(auto_scope=False)
def _w13_vector_panel(
    weight: pl.Tensor,
    expert: pl.Scalar[pl.INDEX],
    block: pl.Scalar[pl.INDEX],
    lane: pl.Scalar[pl.INDEX],
    experts: pl.constexpr,
    weight_cache: pl.constexpr,
):
    packed_rows = experts * W13_PACKED_EXPERT_ROWS // W13_K_TILE
    packed_source = pl.tensor.view(weight, [packed_rows, 2 * W13_PACKED_LANE_BYTES], layout=pl.ND)
    panel = expert * W13_PACKED_EXPERT_ROWS + block * D
    packed_row = panel // W13_K_TILE
    packed_col = lane * W13_PACKED_LANE_BYTES
    # Each DMA fetches this lane's packets for two successive K256 panels.
    raw0 = pl.load(packed_source, [packed_row, packed_col], [2, W13_PACKED_LANE_BYTES], cache=weight_cache)
    raw1 = pl.load(packed_source, [packed_row + 2, packed_col], [2, W13_PACKED_LANE_BYTES], cache=weight_cache)
    for i in pl.unroll(0, D // W13_K_TILE):
        if (i // 2) % 2 == 0:
            raw = pl.tile.slice(raw0, [1, W13_PACKED_LANE_BYTES], [i % 2, 0])
        else:
            raw = pl.tile.slice(raw1, [1, W13_PACKED_LANE_BYTES], [i % 2, 0])
        packed = pl.reshape(raw, [MX_PACKED_LANE_ROWS, MX_PACKED_LANE_COLS])
        fp4_raw = pl.reinterpret_view(packed, pl.FP4E2M1X2)
        if i % 2 == 0:
            nz: pl.Tile[
                [W13_N_TILE // 2, W13_K_TILE], pl.FP8E4M3FN,
                pl.MemRef("w13_nz", slots=2)[0], pl.Mem.Vec,
                pl.TileView(
                    blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major,
                    fractal=512, compact=pl.CompactMode.row_plus_one,
                ),
            ] = pl.cast(fp4_raw, pl.FP8E4M3FN, layout=pl.NZ)
        else:
            nz: pl.Tile[
                [W13_N_TILE // 2, W13_K_TILE], pl.FP8E4M3FN,
                pl.MemRef("w13_nz", slots=2)[1], pl.Mem.Vec,
                pl.TileView(
                    blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major,
                    fractal=512, compact=pl.CompactMode.row_plus_one,
                ),
            ] = pl.cast(fp4_raw, pl.FP8E4M3FN, layout=pl.NZ)
        if i % 2 == 1 and i + 3 < D // W13_K_TILE:
            next_i = i + 3
            if (i // 2) % 2 == 0:
                raw0 = pl.load(packed_source, [packed_row + next_i, packed_col], [2, W13_PACKED_LANE_BYTES], cache=weight_cache)
            else:
                raw1 = pl.load(packed_source, [packed_row + next_i, packed_col], [2, W13_PACKED_LANE_BYTES], cache=weight_cache)
        pl.system.tpush_to_aic(nz, split=1, id=0)


@pl.jit.incore(level=pl.Level.AIV)
def routed_w13_nz_vector(
    x: pl.Tensor,
    x_scale: pl.Tensor,
    counts: pl.Tensor,
    w1: pl.Tensor,
    w1_scale: pl.Tensor,
    w3: pl.Tensor,
    w3_scale: pl.Tensor,
    pair: pl.InOut[pl.Tensor],
    route_weights: pl.Tensor,
    compact_weights: pl.InOut[pl.Tensor],
    experts: pl.constexpr,
    stride: pl.constexpr,
    fifo_slots: pl.constexpr,
    weight_cache: pl.constexpr,
):
    pl.func_attr({'split_aiv': True, 'dual_aiv_dispatch': True})
    lane = pl.tile.get_subblock_idx()
    peer_ring = pl.system.import_peer_buffer(name='w13_weight_ring', peer_func='routed_w13_nz_cube')
    pl.system.aiv_initialize_pipe(v2c_consumer_buf=peer_ring, dir_mask=2, slot_size=W13_FIFO_SLOT_BYTES, slot_num=fifo_slots, id=0)
    route_flat = pl.tensor.view(route_weights, [1, experts * stride], layout=pl.ND)
    compact_flat = pl.tensor.view(compact_weights, [1, compact_weights.shape[0]], layout=pl.ND)
    core = pl.tile.get_block_idx()
    workers = pl.tile.get_block_num()
    work_base = 0
    active_base = 0
    for expert in pl.range(experts):
        count = pl.cast(pl.read(counts, [expert, 0]), pl.INDEX)
        short_rows = MOE_TOKENS >= 384 and count <= W13_SHORT_M_TILE
        row_tile = W13_M_TILE
        if short_rows:
            row_tile = W13_SHORT_M_TILE
        row_tiles = (count + row_tile - 1) // row_tile
        panel_cols = MOE_INTER // W13_N_TILE
        panels = row_tiles * 2 * panel_cols
        first_panel = (core + workers - work_base % workers) % workers
        for panel_id in pl.range(first_panel, panels, workers):
            row = panel_id // (2 * panel_cols) * row_tile
            projection = panel_id // panel_cols % 2
            block = panel_id % panel_cols
            out_row = active_base + row
            if projection == 0:
                _w13_vector_panel(w1, expert, block, lane, experts, weight_cache)
            else:
                _w13_vector_panel(w3, expert, block, lane, experts, weight_cache)
            valid = pl.min(row_tile, count - row)
            if projection == 0 and block == 0 and lane == 0:
                route_row = expert * stride + row
                route = pl.load(route_flat, [0, route_row], [1, W13_M_TILE], valid_shape=[1, valid])
                compact_flat = pl.store(route, [0, out_row], compact_flat)
        work_base = work_base + panels
        active_base = active_base + count
    return compact_weights


@pl.jit.incore(level=pl.Level.CORE_GROUP)
def routed_w13_nz_group(
    x: pl.Tensor,
    x_scale: pl.Tensor,
    counts: pl.Tensor,
    w1: pl.Tensor,
    w1_scale: pl.Tensor,
    w3: pl.Tensor,
    w3_scale: pl.Tensor,
    pair: pl.InOut[pl.Tensor],
    route_weights: pl.Tensor,
    compact_weights: pl.InOut[pl.Tensor],
    experts: pl.constexpr,
    stride: pl.constexpr,
    fifo_slots: pl.constexpr,
):
    # K512 leaves room for the fifth weight slot on the short cached path.
    pair = routed_w13_nz_cube(
        x, x_scale, counts, w1, w1_scale, w3, w3_scale, pair, route_weights, compact_weights, experts, stride,
        fifo_slots, W13_FIFO_SLOT_BYTES * fifo_slots, 512 if fifo_slots == 5 else W13_A_STAGE_K,
    )
    compact_weights = routed_w13_nz_vector(
        x, x_scale, counts, w1, w1_scale, w3, w3_scale, pair, route_weights, compact_weights, experts, stride,
        fifo_slots, pl.CachePolicy.NORMAL_FIRST_VICTIM if fifo_slots == 5 else pl.CachePolicy.BYPASS,
    )
    return pair, compact_weights


# Routed W2 projection. Compact activations use the row-major MX_A_ND ABI;
# Cube MTE2 packs their scales into Mat, so AIV only converts packed weights.
W2_N_TILE = MX_W2_D_OUT_TILE
W2_K_TILE = MX_K_TILE
W2_L0_K = 128
W2_L0_SCALE_K = W2_L0_K // MX_GROUP
W2_SCALE_STAGE_K = 2048
W2_PACKED_LANE_BYTES = MX_PACKED_LANE_ROWS * MX_PACKED_LANE_COLS


@pl.jit.inline(auto_scope=False)
def _w2_row_tile(count: pl.Scalar[pl.INDEX]):
    row_tile = 64
    if count > 128:
        row_tile = 256
    elif count > 64:
        row_tile = 128
    return row_tile


@pl.jit.inline(auto_scope=False)
def _w2_cube_panel(
    payload: pl.Tensor,
    scales: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.InOut[pl.Tensor],
    expert: pl.Scalar[pl.INDEX],
    row: pl.Scalar[pl.INDEX],
    block: pl.Scalar[pl.INDEX],
    valid: pl.Scalar[pl.INDEX],
    m_tile: pl.constexpr,
    a_stage_k: pl.constexpr,
):
    col = block * W2_N_TILE
    compute_m = pl.min(m_tile, valid)
    a_scale = pl.tensor.view(scales, [payload.shape[0], H_SCALE], layout=pl.MX_A_ND)
    # Load scale stages directly in the Cube's Mat layout, including exact
    # partial rows and the final K256 scale slice. Expert row bases need not
    # be 16-row aligned because activation scales are stored in MX_A_ND.
    scale_groups = W2_SCALE_STAGE_K // MX_GROUP
    tail_groups = H_SCALE - scale_groups
    # Credit 5 protects the L1 input tiles across panels.
    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 5)
    as_stage0 = pl.load(
        a_scale, [row, 0], [m_tile, scale_groups],
        valid_shape=[compute_m, scale_groups], target_memory=pl.Mem.Mat,
    )
    bs_stage0 = pl.load(
        weight_scale, [expert * H_SCALE, col], [scale_groups, W2_N_TILE],
        target_memory=pl.Mem.Mat,
    )
    a_stage0 = pl.load(
        payload, [row, 0], [m_tile, a_stage_k],
        valid_shape=[compute_m, a_stage_k], target_memory=pl.Mem.Mat,
    )
    # Start extracting the first activation and its scales while MTE2 fills the next stage.
    pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 7)
    as_stage1 = pl.load(
        a_scale, [row, scale_groups], [m_tile, tail_groups],
        valid_shape=[compute_m, tail_groups], target_memory=pl.Mem.Mat,
    )
    bs_stage1 = pl.load(
        weight_scale, [expert * H_SCALE + scale_groups, col], [tail_groups, W2_N_TILE],
        target_memory=pl.Mem.Mat,
    )
    a_stage1 = pl.load(
        payload, [row, a_stage_k], [m_tile, a_stage_k],
        valid_shape=[compute_m, a_stage_k], target_memory=pl.Mem.Mat,
    )
    acc_seed = pl.tile.create([m_tile, W2_N_TILE], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
    acc = pl.set_validshape(acc_seed, pl.min(compute_m, m_tile), W2_N_TILE)
    pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 4)
    for k in pl.unroll(0, MOE_INTER, W2_K_TILE):
        if k == 0:
            pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 7)
        elif k == a_stage_k:
            pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 4)
        elif k >= 2 * a_stage_k and k % a_stage_k == 0:
            pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 2 + (k // a_stage_k) % 2)
        b = pl.system.tpop_from_aiv(shape=[W2_K_TILE, W2_N_TILE], dtype=pl.FP8E4M3FN, split=1, id=0)
        a_offset = k % a_stage_k
        scale_offset = (k % W2_SCALE_STAGE_K) // MX_GROUP
        # The first extraction may reuse either L0 address when the expert row tile changes.
        if k == 0:
            pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
            pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
        pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 0)
        if (k // a_stage_k) % 2 == 0:
            a_left0 = pl.tile.extract(a_stage0, 0, a_offset, [m_tile, W2_L0_K], target_memory=pl.Mem.Left)
        else:
            a_left0 = pl.tile.extract(a_stage1, 0, a_offset, [m_tile, W2_L0_K], target_memory=pl.Mem.Left)
        b_right0 = pl.tile.extract(b, 0, 0, [W2_L0_K, W2_N_TILE], target_memory=pl.Mem.Right)
        if k < W2_SCALE_STAGE_K:
            as0 = pl.tile.extract(as_stage0, 0, scale_offset, [m_tile, W2_L0_SCALE_K], target_memory=pl.Mem.LeftScale)
            bs0 = pl.tile.extract(bs_stage0, scale_offset, 0, [W2_L0_SCALE_K, W2_N_TILE], target_memory=pl.Mem.RightScale)
        else:
            as0 = pl.tile.extract(as_stage1, 0, scale_offset, [m_tile, W2_L0_SCALE_K], target_memory=pl.Mem.LeftScale)
            bs0 = pl.tile.extract(bs_stage1, scale_offset, 0, [W2_L0_SCALE_K, W2_N_TILE], target_memory=pl.Mem.RightScale)
        # Both K128 halves remain live in separate L0 slots.
        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, 0)
        pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
        if (k // a_stage_k) % 2 == 0:
            a_left1 = pl.tile.extract(a_stage0, 0, a_offset + W2_L0_K, [m_tile, W2_L0_K], target_memory=pl.Mem.Left)
        else:
            a_left1 = pl.tile.extract(a_stage1, 0, a_offset + W2_L0_K, [m_tile, W2_L0_K], target_memory=pl.Mem.Left)
        b_right1 = pl.tile.extract(b, W2_L0_K, 0, [W2_L0_K, W2_N_TILE], target_memory=pl.Mem.Right)
        if k < W2_SCALE_STAGE_K:
            as1 = pl.tile.extract(as_stage0, 0, scale_offset + W2_L0_SCALE_K, [m_tile, W2_L0_SCALE_K], target_memory=pl.Mem.LeftScale)
            bs1 = pl.tile.extract(bs_stage0, scale_offset + W2_L0_SCALE_K, 0, [W2_L0_SCALE_K, W2_N_TILE], target_memory=pl.Mem.RightScale)
        else:
            as1 = pl.tile.extract(as_stage1, 0, scale_offset + W2_L0_SCALE_K, [m_tile, W2_L0_SCALE_K], target_memory=pl.Mem.LeftScale)
            bs1 = pl.tile.extract(bs_stage1, scale_offset + W2_L0_SCALE_K, 0, [W2_L0_SCALE_K, W2_N_TILE], target_memory=pl.Mem.RightScale)
        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, 1)
        if k + W2_K_TILE == MOE_INTER:
            pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 5)
        pl.system.tfree_to_aiv(b, id=0)
        pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, 0)
        if k == 0:
            if MX_UNIT_FLAGS:
                acc = pl.tile.matmul_mx(a_left0, as0, b_right0, bs0, acc_phase=pl.AccPhase.Partial)
            else:
                pl.system.wait_flag(pl.PipeType.FIX, pl.PipeType.M, 0)
                acc = pl.matmul_mx(a_left0, as0, b_right0, bs0)
        elif MX_UNIT_FLAGS:
            acc = pl.tile.matmul_mx_acc(acc, a_left0, as0, b_right0, bs0, acc_phase=pl.AccPhase.Partial)
        else:
            acc = pl.matmul_mx_acc(acc, a_left0, as0, b_right0, bs0)
        pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 0)
        pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, 1)
        if MX_UNIT_FLAGS:
            if k + W2_K_TILE == MOE_INTER:
                # FIX consumes the same final MX accumulator before Cube can reuse it.
                acc = pl.tile.matmul_mx_acc(acc, a_left1, as1, b_right1, bs1, acc_phase=pl.AccPhase.Final)
                output = pl.store(acc, [row, col], output, st_phase=pl.STPhase.Final)
            else:
                acc = pl.tile.matmul_mx_acc(acc, a_left1, as1, b_right1, bs1, acc_phase=pl.AccPhase.Partial)
        else:
            acc = pl.matmul_mx_acc(acc, a_left1, as1, b_right1, bs1)
        pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
        if k % a_stage_k == a_stage_k - W2_K_TILE and k + a_stage_k + W2_K_TILE < MOE_INTER:
            pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 2 + (k // a_stage_k) % 2)
            pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 2 + (k // a_stage_k) % 2)
            next_k = k + a_stage_k + W2_K_TILE
            if (k // a_stage_k) % 2 == 0:
                a_stage0 = pl.load(
                    payload, [row, next_k], [m_tile, a_stage_k],
                    valid_shape=[compute_m, a_stage_k], target_memory=pl.Mem.Mat,
                )
            else:
                a_stage1 = pl.load(
                    payload, [row, next_k], [m_tile, a_stage_k],
                    valid_shape=[compute_m, a_stage_k], target_memory=pl.Mem.Mat,
                )
            pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 2 + (k // a_stage_k) % 2)
    if not MX_UNIT_FLAGS:
        pl.system.set_flag(pl.PipeType.M, pl.PipeType.FIX, 0)
        pl.system.wait_flag(pl.PipeType.M, pl.PipeType.FIX, 0)
        output = pl.store(acc, [row, col], output)
        pl.system.set_flag(pl.PipeType.FIX, pl.PipeType.M, 0)
    return output


@pl.jit.incore(level=pl.Level.AIC)
def routed_w2_nz_cube(
    payload: pl.Tensor, scales: pl.Tensor, counts: pl.Tensor, weight: pl.Tensor, weight_scale: pl.Tensor,
    output: pl.InOut[pl.Tensor], experts: pl.constexpr, stride: pl.constexpr,
):
    pl.func_attr({'split_aiv': True})
    weight_ring = pl.system.reserve_buffer(name='w2_weight_ring', size=262144)
    pl.system.aic_initialize_pipe(v2c_consumer_buf=weight_ring, dir_mask=2, slot_size=65536, slot_num=4, id=0)
    pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 0)
    pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
    if not MX_UNIT_FLAGS:
        pl.system.set_flag(pl.PipeType.FIX, pl.PipeType.M, 0)
    pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 5)
    core = pl.tile.get_block_idx()
    workers = pl.tile.get_block_num()
    compact = 0
    work_base = 0
    for expert in pl.range(experts):
        count = pl.cast(pl.read(counts, [expert, 0]), pl.INDEX)
        row_tile = _w2_row_tile(count)
        row_tiles = (count + row_tile - 1) // row_tile
        work_count = row_tiles * (D // W2_N_TILE)
        first = (core + workers - work_base % workers) % workers
        for work in pl.range(first, work_count, workers):
            row = work // (D // W2_N_TILE) * row_tile
            block = work % (D // W2_N_TILE)
            valid = pl.min(row_tile, count - row)
            # K768 divides the full reduction for M64/M128. M256 uses
            # K256 activation stages to stay within the Mat budget.
            if row_tile == 64:
                output = _w2_cube_panel(payload, scales, weight_scale, output, expert, compact + row, block, valid,
                                        64, 768)
            elif row_tile == 128:
                output = _w2_cube_panel(payload, scales, weight_scale, output, expert, compact + row, block, valid,
                                        128, 768)
            else:
                output = _w2_cube_panel(payload, scales, weight_scale, output, expert, compact + row, block, valid,
                                        256, 256)
        work_base = work_base + work_count
        compact = compact + count
    pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 0)
    pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
    if not MX_UNIT_FLAGS:
        pl.system.wait_flag(pl.PipeType.FIX, pl.PipeType.M, 0)
    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 5)
    # Publish the final BF16 store before scatter or the local copy reads it.
    pl.system.set_flag(pl.PipeType.FIX, pl.PipeType.S, 0)
    pl.system.wait_flag(pl.PipeType.FIX, pl.PipeType.S, 0)
    return output


@pl.jit.inline(auto_scope=False)
def _w2_vector_panel(
    weight: pl.Tensor, expert: pl.Scalar[pl.INDEX], block: pl.Scalar[pl.INDEX],
    lane: pl.Scalar[pl.INDEX], experts: pl.constexpr,
):
    packed_rows = experts * MX_W2_PACKED_ROWS // W2_K_TILE
    packed = pl.tensor.view(weight, [packed_rows, 2 * W2_PACKED_LANE_BYTES], layout=pl.ND)
    packed_row = (expert * MX_W2_PACKED_ROWS + block * MOE_INTER) // W2_K_TILE
    packed_col = lane * W2_PACKED_LANE_BYTES
    # W2 reads four pairs of packets and a final single packet.
    raw0 = pl.load(packed, [packed_row, packed_col], [2, W2_PACKED_LANE_BYTES], cache=pl.CachePolicy.BYPASS)
    raw1 = pl.load(packed, [packed_row + 2, packed_col], [2, W2_PACKED_LANE_BYTES], cache=pl.CachePolicy.BYPASS)
    for i in pl.unroll(0, MOE_INTER // W2_K_TILE):
        if i == MOE_INTER // W2_K_TILE - 1:
            raw = pl.load(packed, [packed_row + i, packed_col], [1, W2_PACKED_LANE_BYTES], cache=pl.CachePolicy.BYPASS)
        elif (i // 2) % 2 == 0:
            raw = pl.tile.slice(raw0, [1, W2_PACKED_LANE_BYTES], [i % 2, 0])
        else:
            raw = pl.tile.slice(raw1, [1, W2_PACKED_LANE_BYTES], [i % 2, 0])
        raw_packet = pl.reshape(raw, [MX_PACKED_LANE_ROWS, MX_PACKED_LANE_COLS])
        fp4_raw = pl.reinterpret_view(raw_packet, pl.FP4E2M1X2)
        if i % 2 == 0:
            nz: pl.Tile[
                [MX_PACKED_LANE_ROWS, W2_K_TILE], pl.FP8E4M3FN, pl.MemRef("w2_nz", slots=2)[0], pl.Mem.Vec,
                pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major,
                            fractal=512, compact=pl.CompactMode.row_plus_one),
            ] = pl.cast(fp4_raw, pl.FP8E4M3FN, layout=pl.NZ)
        else:
            nz: pl.Tile[
                [MX_PACKED_LANE_ROWS, W2_K_TILE], pl.FP8E4M3FN, pl.MemRef("w2_nz", slots=2)[1], pl.Mem.Vec,
                pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major,
                            fractal=512, compact=pl.CompactMode.row_plus_one),
            ] = pl.cast(fp4_raw, pl.FP8E4M3FN, layout=pl.NZ)
        if i % 2 == 1 and i + 4 < MOE_INTER // W2_K_TILE:
            next_i = i + 3
            if (i // 2) % 2 == 0:
                raw0 = pl.load(packed, [packed_row + next_i, packed_col], [2, W2_PACKED_LANE_BYTES], cache=pl.CachePolicy.BYPASS)
            else:
                raw1 = pl.load(packed, [packed_row + next_i, packed_col], [2, W2_PACKED_LANE_BYTES], cache=pl.CachePolicy.BYPASS)
        pl.system.tpush_to_aic(nz, split=1, id=0)


@pl.jit.incore(level=pl.Level.AIV)
def routed_w2_nz_vector(
    payload: pl.Tensor, scales: pl.Tensor, counts: pl.Tensor, weight: pl.Tensor, weight_scale: pl.Tensor,
    output: pl.InOut[pl.Tensor], experts: pl.constexpr, stride: pl.constexpr,
):
    pl.func_attr({'split_aiv': True, 'dual_aiv_dispatch': True})
    lane = pl.tile.get_subblock_idx()
    peer_ring = pl.system.import_peer_buffer(name='w2_weight_ring', peer_func='routed_w2_nz_cube')
    pl.system.aiv_initialize_pipe(v2c_consumer_buf=peer_ring, dir_mask=2, slot_size=65536, slot_num=4, id=0)
    core = pl.tile.get_block_idx()
    workers = pl.tile.get_block_num()
    work_base = 0
    for expert in pl.range(experts):
        count = pl.cast(pl.read(counts, [expert, 0]), pl.INDEX)
        row_tile = _w2_row_tile(count)
        row_tiles = (count + row_tile - 1) // row_tile
        work_count = row_tiles * (D // W2_N_TILE)
        first = (core + workers - work_base % workers) % workers
        for work in pl.range(first, work_count, workers):
            block = work % (D // W2_N_TILE)
            _w2_vector_panel(weight, expert, block, lane, experts)
        work_base = work_base + work_count
    return output


@pl.jit.incore(level=pl.Level.CORE_GROUP)
def routed_w2_nz_group(
    payload: pl.Tensor, scales: pl.Tensor, counts: pl.Tensor,
    weight: pl.Tensor, weight_scale: pl.Tensor,
    output: pl.InOut[pl.Tensor], experts: pl.constexpr, stride: pl.constexpr,
):
    output = routed_w2_nz_cube(payload, scales, counts, weight, weight_scale, output, experts, stride)
    output = routed_w2_nz_vector(payload, scales, counts, weight, weight_scale, output, experts, stride)
    return output


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
    w13_workers: pl.Scalar[pl.INT32],
    w2_workers: pl.Scalar[pl.INT32],
    swiglu_workers: pl.Scalar[pl.INT32],
):
    """Run BF16 W13, FP32 weighted MXFP8 activation, and full-K BF16 W2.

    Counts lie in [0, stride] and sum to
    at most row_capacity. Stride is divisible by 16; width and hidden are
    divisible by 256.
    """
    pair = pl.create_tensor([((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE, 2 * hidden], dtype=pl.BF16)
    compact_weights = pl.create_tensor([((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE, 1], dtype=pl.FP32)
    aiv_count = pl.system.available_aiv_count()
    cluster_count = pl.system.available_cluster_count()
    mixed_clusters = pl.min(cluster_count, aiv_count // 2)
    w13_blocks = pl.min(w13_workers, mixed_clusters)
    with pl.spmd(w13_blocks, name_hint="routed_w13") as w13_ready:
        pair, compact_weights = routed_w13_nz_group(
            x, x_scale, counts, w1, w1_scale, w3, w3_scale, pair,
            weights, compact_weights, experts, stride, 5 if MOE_TOKENS <= 80 and experts <= 12 else 4,
        )

    payload = pl.create_tensor([((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE, hidden], dtype=pl.FP8E4M3FN)
    scales = pl.create_tensor([1, ((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE * (hidden // MX_GROUP)], dtype=pl.FP8E8M0)
    counts_row = pl.reshape(counts, [1, experts])
    swiglu_blocks = pl.min(swiglu_workers, aiv_count)
    with pl.spmd(swiglu_blocks, name_hint="routed_swiglu_quant", deps=[w13_ready]) as swiglu_ready:
        core = pl.tile.get_block_idx()
        workers = pl.tile.get_block_num()
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
        compact_route = pl.tensor.view(compact_weights, [1, compact_weights.shape[0]])
        work_count = (rows + SWIGLU_ROW_TILE - 1) // SWIGLU_ROW_TILE
        for i in pl.pipeline((work_count + workers - 1 - core) // workers, stage=2):
            row = (core + i * workers) * SWIGLU_ROW_TILE
            gate_tile = pl.load(pair, [row, 0], [SWIGLU_ROW_TILE, MOE_INTER],
                                target_memory=pl.MemorySpace.Vec)
            up_tile = pl.load(pair, [row, MOE_INTER], [SWIGLU_ROW_TILE, MOE_INTER],
                              target_memory=pl.MemorySpace.Vec)
            weight_tile = pl.load(compact_route, [0, row], [1, 8], valid_shape=[1, SWIGLU_ROW_TILE],
                                  target_memory=pl.MemorySpace.Vec)
            gate_flat = pl.reshape(gate_tile, [SWIGLU_ROW_TILE * MOE_INTER])
            up_flat = pl.reshape(up_tile, [SWIGLU_ROW_TILE * MOE_INTER])
            weight_row = pl.set_validshape(weight_tile, 1, 8)
            weight_flat = pl.reshape(weight_row, [8])
            payload_flat = pl.tile.create([SWIGLU_ROW_TILE * MOE_INTER], dtype=pl.INT8,
                                          target_memory=pl.MemorySpace.Vec)
            # Each pair of exponents occupies one 32-byte aligned scratch slot.
            scale_scratch = pl.tile.create([SWIGLU_SCALE_SCRATCH_TILE], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec)
            code_flat = pl.tile.create([SWIGLU_CODE_TILE], dtype=pl.INT8, target_memory=pl.MemorySpace.Vec)
            with pl.reg_scope(auto_sync=False):
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
                # Complete the exponent stores before gathering their compact representation.
                pl.reg.mem_bar("VST_VLD")
                # Compact the two populated words from each scratch slot.
                compact_lane = pl.reg.index_gen(0, lanes=SWIGLU_LANES, dtype=pl.UINT32)
                for compact_chunk in pl.hardware_loop(SWIGLU_CODE_TILE // SWIGLU_LANES):
                    group_offset = pl.cast(compact_chunk * SWIGLU_LANES, pl.UINT32)
                    compact_group = pl.reg.adds(compact_lane, group_offset, active)
                    if SWIGLU_GROUP_ROWS % SWIGLU_LANES != 0:
                        # The final register repeats the last valid scale in its padding lanes.
                        compact_group = pl.reg.minimums(compact_group, SWIGLU_GROUP_ROWS - 1, active)
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
            padded_codes = pl.reshape(code_flat, [1, SWIGLU_CODE_TILE])
            code_row = pl.set_validshape(padded_codes, 1, SWIGLU_GROUP_ROWS)
            code_bytes = pl.reinterpret_view(code_row, pl.UINT8)
            codes = pl.reinterpret_view(code_bytes, pl.FP8E8M0)
            scales = pl.store(codes, [0, row * SWIGLU_SCALE_COLS], scales)

    output = pl.create_tensor([((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE, width], dtype=pl.BF16)
    w2_blocks = pl.min(w2_workers, mixed_clusters)
    with pl.spmd(w2_blocks, name_hint="routed_w2", deps=[swiglu_ready]):
        output = routed_w2_nz_group(payload, scales, counts, w2, w2_scale, output, experts, stride)
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
    local_experts: pl.constexpr,
    topk: pl.constexpr,
    w13_workers: pl.Scalar[pl.INT32],
    w2_workers: pl.Scalar[pl.INT32],
    swiglu_workers: pl.Scalar[pl.INT32],
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
        w13_workers, w2_workers, swiglu_workers,
    )
    flat_counts = pl.reshape(recv_expert_count, [1, local_experts])
    metadata_cols = ((local_experts + 7) // 8) * 8
    aiv_count = pl.system.available_aiv_count()
    scatter_blocks = pl.min(64, aiv_count)
    with pl.spmd(scatter_blocks, name_hint="ffn_scatter") as scatter_done:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="ffn_scatter", no_dep_args=[routed_output]):
            core = pl.tile.get_block_idx()
            workers = pl.tile.get_block_num()
            count_values = pl.load(flat_counts, [0, 0], [1, metadata_cols], valid_shape=[1, local_experts])
            meta_values = pl.load(recv_meta, [0, 0], [EP_WORLD_SIZE, metadata_cols], valid_shape=[EP_WORLD_SIZE, local_experts])
            total_rows = 0
            for expert in pl.range(local_experts):
                rows_read = pl.tile.read(count_values, [0, expert])
                rows = pl.cast(rows_read, pl.INDEX)
                total_rows = total_rows + rows
            rows_per_core = total_rows // workers
            extra_rows = total_rows % workers
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
        ROUTED_CUBE_WORKER_TILE, ROUTED_CUBE_WORKER_TILE, ROUTED_VECTOR_WORKER_TILE,
    )
    aiv_count = pl.system.available_aiv_count()
    copy_blocks = pl.min(64, aiv_count)
    with pl.spmd(copy_blocks, name_hint="ffn_copy"):
        core = pl.tile.get_block_idx()
        workers = pl.tile.get_block_num()
        work_base = 0
        compact_base = 0
        for expert in pl.range(N_LOCAL_EXPERTS):
            count_read = pl.read(recv_expert_count, [expert, 0])
            count = pl.cast(count_read, pl.INDEX)
            work_count = (count + 1) // 2
            first = (core + workers - work_base % workers) % workers
            for work in pl.range(first, work_count, workers):
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
    local_experts, recv_max, _ = recv_x.shape
    recv_scale_codes = tensors["recv_mx_scale"].reshape(local_experts * recv_max, K_SCALE)
    recv_scale_decoded = decode_e8m0_codes(recv_scale_codes, side="a")
    recv_mx_scale = recv_scale_decoded.reshape(local_experts, recv_max, K_SCALE)
    recv_weights = tensors["recv_weights"].float()
    recv_expert_count = tensors["recv_expert_count"]
    w1_scale = decode_e8m0_codes(tensors['routed_w1_scale'], side='b').reshape(local_experts, K_SCALE, MOE_INTER)
    w3_scale = decode_e8m0_codes(tensors['routed_w3_scale'], side='b').reshape(local_experts, K_SCALE, MOE_INTER)
    w2_scale = decode_e8m0_codes(tensors['routed_w2_scale'], side='b').reshape(local_experts, H_SCALE, D)

    recv_y = torch.zeros(local_experts, recv_max, D)
    for e in range(local_experts):
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
TAIL_COUNTS = (15, 16, 17, 63, 64, 65, 127, 128, 129, 255, 256, 257)


def golden_local_compute(tensors):
    """Fill active local output rows through the production expert golden."""
    values = dict(tensors)
    values["recv_y"] = torch.empty_like(tensors["routed_y"]).reshape_as(tensors["recv_x"])
    golden_expert_routed(values)
    output = tensors["routed_y"].reshape_as(tensors["recv_x"])
    for expert, count in enumerate(tensors["recv_expert_count"].reshape(-1).tolist()):
        output[expert, :count] = values["recv_y"][expert, :count]


def compare_local_compute(actual, expected, *, inputs, **kwargs):
    """Compare active expert rows and require exact inactive output guards."""
    from golden import ratio_reldiff

    actual_rows = actual.reshape(inputs["recv_expert_count"].numel(), -1, D)
    expected_rows = expected.reshape_as(actual_rows)
    comparator = ratio_reldiff(diff_thd=2e-3, pct_thd=0.01)
    for expert, count in enumerate(inputs["recv_expert_count"].reshape(-1).tolist()):
        if not torch.equal(actual_rows[expert, count:], expected_rows[expert, count:]):
            return False, f"expert {expert}: inactive output rows changed"
        passed, message = comparator(actual_rows[expert, :count], expected_rows[expert, :count], inputs=inputs, **kwargs)
        if not passed:
            return False, f"expert {expert}, count {count}: {message}"
    return True, ""


def effective_tail_counts():
    """Return tail counts representable by the configured receive capacity."""
    return tuple(count for count in TAIL_COUNTS if count <= RECV_MAX)


def validation_counts(case="bounds", *, tail_offset=0, local_experts=N_LOCAL_EXPERTS):
    """Return bounded empty, row-tile-tail, and receive-capacity workloads."""
    if case == "zero":
        pattern = (0,)
    elif case == "full":
        pattern = (RECV_MAX,)
    elif case == "balanced":
        pattern = (36,)
    elif case == "sparse":
        pattern = (0, 1, 65, 257)
    elif case == "tails":
        pattern = effective_tail_counts()
    elif case == "bounds":
        pattern = (0, 1, RECV_MAX - 1, RECV_MAX)
    else:
        raise ValueError(f"Unknown local compute case: {case}")
    values = [min(RECV_MAX, pattern[(expert + tail_offset) % len(pattern)]) for expert in range(local_experts)]
    return torch.tensor(values, dtype=torch.int32).reshape(local_experts, 1)


def build_tensor_specs(case="bounds", *, tail_offset=0, seed=20261028, local_experts=N_LOCAL_EXPERTS):
    """Build local compute tensors with poisoned inactive capacity and lazy packed weights."""
    from functools import cache

    from golden import TensorSpec
    from quant import pack_mx_a_scale, quantize_mxfp8_upward_e8m0

    counts = validation_counts(case, tail_offset=tail_offset, local_experts=local_experts)

    @cache
    def receive_inputs():
        generator = torch.Generator().manual_seed(seed)
        source = torch.randn(local_experts, RECV_MAX, D, dtype=torch.bfloat16, generator=generator)
        payload, codes = quantize_mxfp8_upward_e8m0(source.reshape(-1, D).float())
        payload = payload.reshape(local_experts, RECV_MAX, D)
        codes = codes.reshape(local_experts, RECV_MAX, K_SCALE).view(torch.uint8)
        route_weights = torch.rand(local_experts, RECV_MAX, generator=generator)
        for expert, count in enumerate(counts.reshape(-1).tolist()):
            payload.view(torch.uint8)[expert, count:] = 127
            codes[expert, count:] = 255
            route_weights[expert, count:] = float("nan")
        packed_codes = pack_mx_a_scale(codes.reshape(-1, K_SCALE)).view(torch.float8_e8m0fnu)
        return payload, packed_codes.reshape(1, -1), route_weights

    @cache
    def routed_weights():
        return gen_routed_mxfp4_weights(local_experts, ROUTED_DEQUANT_STD, seed_base=10)

    fp8 = torch.float8_e4m3fn
    e8m0 = torch.float8_e8m0fnu
    return [
        TensorSpec("recv_x", [local_experts, RECV_MAX, D], fp8, init_value=lambda: receive_inputs()[0]),
        TensorSpec("recv_mx_scale", [1, local_experts * RECV_MAX * K_SCALE], e8m0, init_value=lambda: receive_inputs()[1]),
        TensorSpec("recv_weights", [local_experts, RECV_MAX], torch.float32, init_value=lambda: receive_inputs()[2]),
        TensorSpec("recv_expert_count", [local_experts, 1], torch.int32, init_value=counts),
        TensorSpec("routed_w1_packed", [local_experts, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], torch.uint8, init_value=lambda: routed_weights()[0]),
        TensorSpec("routed_w1_scale", [local_experts * K_SCALE, MOE_INTER], e8m0, init_value=lambda: routed_weights()[1]),
        TensorSpec("routed_w3_packed", [local_experts, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], torch.uint8, init_value=lambda: routed_weights()[2]),
        TensorSpec("routed_w3_scale", [local_experts * K_SCALE, MOE_INTER], e8m0, init_value=lambda: routed_weights()[3]),
        TensorSpec("routed_w2_packed", [local_experts, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], torch.uint8, init_value=lambda: routed_weights()[4]),
        TensorSpec("routed_w2_scale", [local_experts * H_SCALE, D], e8m0, init_value=lambda: routed_weights()[5]),
        TensorSpec("routed_y", [local_experts * RECV_MAX, D], torch.bfloat16, init_value=OUTPUT_SENTINEL),
    ]


if __name__ == "__main__":
    from golden import run

    cases = ("bounds", "tails", "zero") if args.case == "all" else (args.case,)
    for case in cases:
        tail_count = len(effective_tail_counts())
        tail_rounds = (tail_count + N_LOCAL_EXPERTS - 1) // N_LOCAL_EXPERTS if case == "tails" else 1
        if args.tail_round is not None and not 0 <= args.tail_round < tail_rounds:
            parser.error(f"--tail-round must be in [0, {tail_rounds - 1}]")
        for tail_round in range(tail_rounds):
            if args.tail_round is not None and tail_round != args.tail_round:
                continue
            print(
                f"Local routed compute: case={case}, tail_round={tail_round}, tokens={MOE_TOKENS}, "
                f"experts={N_LOCAL_EXPERTS}, stride={RECV_MAX}",
                flush=True,
            )
            result = run(
                a5_default_runtime="host_build_graph",
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
