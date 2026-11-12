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
    parser.add_argument("--mode", choices=["all", "chain", "w13", "swiglu", "w2"], default="all",
                        help="validate one stage or the full routed expert chain")
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
H_SCALE = MOE_INTER // MX_GROUP

# Routed-expert across-layer mean dequantization standard deviations.
ROUTED_DEQUANT_STD = {"w1": 2.47e-2, "w2": 2.44e-2, "w3": 2.46e-2}

# tiling
RECV_ROW_TILE = 16
RECV_SCALE_TMP_TILE = ((64 + K_SCALE + 31) // 32) * 32
GATHER_WORKERS = 64
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

# Full activation-scale row tiles divide each expert's padded stride.
W13_M_TILE = gcd(RECV_MAX, 80)
W13_N_TILE = MX_MM_INTER_TILE
W13_K_TILE = MX_K_TILE // 2
W13_SCALE_K = W13_K_TILE // MX_GROUP
W13_PACKED_EXPERT_ROWS = MX_W1_PACKED_ROWS
W13_K_STAGE = 2
FP4_LUT_SIZE = 256


@pl.jit.inline(auto_scope=False)
def gather_routed_inputs(
    recv_x: pld.DistributedTensor,
    recv_scale: pld.DistributedTensor,
    recv_weights: pld.DistributedTensor,
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
    """Gather source lanes and pack scales with aligned, count-driven workers."""
    x_flat = pl.reshape(x_out, [local_experts * RECV_MAX, D])
    scale_nd = pl.create_tensor([local_experts * RECV_MAX, K_SCALE], dtype=pl.FP8E8M0)
    with pl.spmd(GATHER_WORKERS, name_hint="dispatch_gather", deps=[wait_ready, publish_ready]) as ready:
        core = pl.tile.get_block_idx()
        work_base = 0
        for expert in pl.range(local_experts):
            source_counts = pl.array.create(EP_WORLD_SIZE, pl.INT32)
            count = pl.cast(0, pl.INDEX)
            for src in pl.range(EP_WORLD_SIZE):
                count_read = pl.read(recv_routes, [expert * RECV_MAX + src * max_per_source, 1])
                source_counts[src] = count_read
                count = count + pl.cast(count_read, pl.INDEX)
                if core == 0:
                    pl.write(recv_meta, [src, expert], count_read)
                    pl.write(recv_meta_window, [src, expert], count_read)
            if core == 0:
                pl.write(counts, [expert, 0], pl.cast(count, pl.INT32))
            work_count = (count + RECV_ROW_TILE - 1) // RECV_ROW_TILE
            first = (core + GATHER_WORKERS - work_base % GATHER_WORKERS) % GATHER_WORKERS
            for work in pl.range(first, work_count, GATHER_WORKERS):
                row_begin = work * RECV_ROW_TILE
                row_end = pl.min(row_begin + RECV_ROW_TILE, count)
                expert_base = expert * RECV_MAX
                source_begin = 0
                for src in pl.range(EP_WORLD_SIZE):
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
                        scale_raw = pl.load(
                            recv_scale, [source_row, 0], [RECV_ROW_TILE, K_SCALE],
                            valid_shape=[copy_rows, K_SCALE],
                        )
                        scale = pl.reinterpret_view(scale_raw, pl.FP8E8M0)
                        scale_nd = pl.store(scale, [output_row, 0], scale_nd)
                        for row in pl.range(gather_begin, gather_end):
                            metadata_row = source_base + row - source_begin
                            weight = pl.read(recv_weights, [metadata_row, 0])
                            pl.write(weights_out, [expert, row], weight)
                            route = pl.read(recv_routes, [metadata_row, 0])
                            pl.write(routes_out, [expert, row], route)
                    source_begin = source_end
                # Each worker owns complete 64-byte metadata cache lines.
                scale_row = expert_base + row_begin
                scale_values = pl.load(scale_nd, [scale_row, 0], [RECV_ROW_TILE, K_SCALE])
                scale_bytes = pl.reinterpret_view(scale_values, pl.UINT8)
                temporary = pl.create_tile([1, RECV_SCALE_TMP_TILE], dtype=pl.UINT8)
                packed = pl.tmov_x2zz(scale_bytes, temporary, group_axis=1, dst_rows=RECV_ROW_TILE, dst_cols=K_SCALE)
                packed_mx = pl.reinterpret_view(packed, pl.FP8E8M0)
                packed_flat = pl.reshape(packed_mx, [1, RECV_ROW_TILE * K_SCALE])
                scale_out = pl.store(packed_flat, [0, scale_row * K_SCALE], scale_out)
            work_base = work_base + work_count
    return ready


@pl.jit.inline(auto_scope=False)
def push_dispatch_payload(
    indices: pl.Tensor,
    payload: pl.Tensor,
    packed_scale: pl.Tensor,
    weights: pl.Tensor,
    recv_x: pld.DistributedTensor,
    recv_scale: pld.DistributedTensor,
    recv_weights: pld.DistributedTensor,
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
    worker_tile = workers
    if dispatch_worker_limit > 0:
        worker_tile = pl.min(workers, dispatch_worker_limit)
    with pl.spmd(worker_tile, name_hint="dispatch_push", deps=[input_ready], allow_early_resolve=False) as ready:
        core = pl.tile.get_block_idx()
        active = pl.min(pl.max(pl.cast(num_tokens, pl.INDEX), 0), MOE_TOKENS)
        ids = pl.load(indices, [0, 0], [MOE_TOKENS, 8], valid_shape=[MOE_TOKENS, topk])
        route_weights = pl.load(weights, [0, 0], [MOE_TOKENS, 8], valid_shape=[MOE_TOKENS, topk])
        for block in pl.range(core, workers, worker_tile):
            peer = block // local_experts
            expert = block % local_experts
            base = expert * RECV_MAX + rank * MOE_TOKENS
            weight_lane = pl.tile.full([MOE_TOKENS, 8], dtype=pl.FP32, value=0.0)
            route_lane = pl.tile.full([MOE_TOKENS, 8], dtype=pl.INT32, value=0)
            cursor = pl.array.create(1, pl.INT32)
            cursor[0] = 0
            for batch in pl.range((active + 15) // 16):
                first = batch * 16
                scales = pl.load(packed_scale, [0, first * K_SCALE], [1, 16 * K_SCALE])
                scale_bytes = pl.reinterpret_view(scales, pl.UINT8)
                pairs = pl.reinterpret_view(scale_bytes, pl.UINT16, shape=[K_SCALE // 2, 16])
                pairs_nd = pl.transpose(pairs, 0, 1)
                bytes_nd = pl.reinterpret_view(pairs_nd, pl.UINT8, shape=[16, K_SCALE])
                for token in pl.range(first, pl.min(first + 16, active)):
                    for top in pl.range(topk):
                        eid = pl.tile.read(ids, [token, top])
                        if eid == block:
                            slot = cursor[0]
                            cursor[0] = slot + 1
                            row = base + slot
                            values = pl.load(payload, [token, 0], [1, D])
                            value_bytes = pl.reinterpret_view(values, pl.INT8)
                            pld.tile.remote_store(value_bytes, target=recv_x, peer=peer, offsets=[row, 0])
                            scale_row = pl.slice(bytes_nd, [1, K_SCALE], [token - first, 0])
                            pld.tile.remote_store(scale_row, target=recv_scale, peer=peer, offsets=[row, 0])
                            value = pl.tile.read(route_weights, [token, top])
                            pl.tile.write(weight_lane, [slot, 0], value)
                            route = pl.cast(token * topk + top, pl.INT32)
                            pl.tile.write(route_lane, [slot, 0], route)
            pl.tile.write(route_lane, [0, 1], cursor[0])
            pld.tile.remote_store(weight_lane, target=recv_weights, peer=peer, offsets=[base, 0])
            pld.tile.remote_store(route_lane, target=recv_routes, peer=peer, offsets=[base, 0])
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
    recv_scale: pld.DistributedTensor,
    recv_weights: pld.DistributedTensor,
    recv_routes: pld.DistributedTensor,
    data_arrived: pld.DistributedTensor,
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    local_experts: pl.constexpr = N_LOCAL_EXPERTS,
    topk: pl.constexpr = TOPK,
    dispatch_worker_limit: pl.constexpr = 0,
):
    stage_anchor = pl.system.task_dummy(deps=[])
    _push_tid = push_dispatch_payload(
        indices, x_norm_mx, x_norm_scale, weights,
        recv_x, recv_scale, recv_weights, recv_routes,
        num_tokens, my_rank, stage_anchor, local_experts=local_experts, topk=topk,
        dispatch_worker_limit=dispatch_worker_limit,
    )
    # Publish one epoch only after every local payload store completes.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="dispatch_publish_epoch",
        deps=[_push_tid], allow_early_resolve=False,
    ) as _publish_tid:
        for dst in pl.range(EP_WORLD_SIZE):
            if dst != my_rank:
                pld.system.notify(
                    target=data_arrived, peer=dst, offsets=[my_rank, 0, 0],
                    value=moe_epoch, op=pld.NotifyOp.Set,
                )

    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="dispatch_wait",
        deps=[_push_tid], no_dep_args=[data_arrived],
    ) as _wait_tid:
        for src in pl.range(EP_WORLD_SIZE):
            if src != my_rank:
                pld.system.defer_wait(signal=data_arrived, offsets=[src, 0, 0], expected=moe_epoch, cmp=pld.WaitCmp.Ge)

    return gather_routed_inputs(
        recv_x, recv_scale, recv_weights, recv_routes, recv_meta_local, recv_meta, recv_count_out,
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


@pl.jit.inline(auto_scope=False)
def _w13_projection(
    x: pl.Tensor,
    x_scale: pl.Tensor,
    weight: pl.Tensor,
    weight_scale: pl.Tensor,
    fp4_lut: pl.Tensor,
    x_row: pl.Scalar[pl.INDEX],
    expert: pl.Scalar[pl.INDEX],
    col: pl.Scalar[pl.INDEX],
    experts: pl.constexpr,
    stride: pl.constexpr,
):
    """Accumulate one gate or up tile with byte-indexed FP4 decoding."""
    x_scale_zz = pl.tensor.view(x_scale, [experts * stride, K_SCALE], layout=pl.MX_A_ZZ)
    weight_flat = pl.tensor.view(
        weight, [experts * W13_PACKED_EXPERT_ROWS, MX_PACKED_LANE_COLS], layout=pl.ND,
    )
    panel_row = expert * W13_PACKED_EXPERT_ROWS + col // W13_N_TILE * D
    scale_row = expert * K_SCALE
    a0 = pl.load(x, [x_row, 0], [W13_M_TILE, W13_K_TILE])
    a_scale0 = pl.load(x_scale_zz, [x_row, 0], [W13_M_TILE, W13_SCALE_K])
    # Decode two FP4 values per byte into adjacent FP8 lanes.
    lut = pl.load(fp4_lut, [0, 0], [1, FP4_LUT_SIZE])
    packed0 = pl.load(weight_flat, [panel_row, 0], [W13_K_TILE, MX_PACKED_LANE_COLS])
    index0 = pl.cast(pl.cast(packed0, pl.FP16), pl.INT16)
    scratch0 = pl.tile.create([1, MX_PACKED_LANE_COLS], dtype=pl.INT16)
    codes0 = pl.tile.gather(lut, index0, scratch0)
    b0 = pl.reinterpret_view(pl.reinterpret_view(codes0, pl.INT8), pl.FP8E4M3FN)
    b_scale0 = pl.load(weight_scale, [scale_row, col], [W13_SCALE_K, W13_N_TILE])
    acc = pl.matmul_mx(a0, a_scale0, b0, b_scale0)
    for k0 in pl.pipeline(W13_K_TILE, D, W13_K_TILE, stage=W13_K_STAGE):
        a = pl.load(x, [x_row, k0], [W13_M_TILE, W13_K_TILE])
        a_scale = pl.load(x_scale_zz, [x_row, k0 // MX_GROUP], [W13_M_TILE, W13_SCALE_K])
        packed = pl.load(weight_flat, [panel_row + k0, 0], [W13_K_TILE, MX_PACKED_LANE_COLS])
        index = pl.cast(pl.cast(packed, pl.FP16), pl.INT16)
        scratch = pl.tile.create([1, MX_PACKED_LANE_COLS], dtype=pl.INT16)
        codes = pl.tile.gather(lut, index, scratch)
        b = pl.reinterpret_view(pl.reinterpret_view(codes, pl.INT8), pl.FP8E4M3FN)
        b_scale = pl.load(weight_scale, [scale_row + k0 // MX_GROUP, col], [W13_SCALE_K, W13_N_TILE])
        acc = pl.matmul_mx_acc(acc, a, a_scale, b, b_scale)
    return acc


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
    """Compute compact BF16 gate/up rows and copy their route weights.

    The existing model keeps the extern until native latency reaches parity.
    """
    route_flat = pl.tensor.view(route_weights, [1, experts * stride], layout=pl.ND)
    compact_flat = pl.tensor.view(compact_weights, [1, W13_ROWS], layout=pl.ND)
    with pl.spmd(ROUTED_CUBE_WORKER_TILE, name_hint="routed_w13_pypto", deps=[input_ready]) as ready:
        core = pl.tile.get_block_idx()
        work_base = 0
        activation_base = 0
        for expert in pl.range(experts):
            count = pl.cast(pl.read(counts, [expert, 0]), pl.INDEX)
            row_tiles = (count + W13_M_TILE - 1) // W13_M_TILE
            for row_tile in pl.range(row_tiles):
                # Keep MX_A_ZZ scale loads full and narrow only output stores.
                row = row_tile * W13_M_TILE
                valid = pl.min(W13_M_TILE, count - row)
                x_row = expert * stride + row
                out_row = activation_base + row
                for projection in pl.range(2):
                    for block in pl.range((MOE_INTER // W13_N_TILE)):
                        work = (row_tile * 2 + projection) * (MOE_INTER // W13_N_TILE) + block
                        if (work_base + work) % ROUTED_CUBE_WORKER_TILE == core:
                            col = block * W13_N_TILE
                            if projection == 0:
                                gate = _w13_projection(
                                    x, x_scale, w1, w1_scale, fp4_lut, x_row, expert, col, experts, stride,
                                )
                                pair = pl.store(
                                    pl.set_validshape(gate, valid, W13_N_TILE), [out_row, col], pair,
                                )
                            else:
                                up = _w13_projection(
                                    x, x_scale, w3, w3_scale, fp4_lut, x_row, expert, col, experts, stride,
                                )
                                pair = pl.store(
                                    pl.set_validshape(up, valid, W13_N_TILE), [out_row, MOE_INTER + col], pair,
                                )
                            # The first gate column copies the compact route-weight row.
                            if projection == 0 and block == 0 and pl.tile.get_subblock_idx() == 0:
                                weight_row = pl.load(
                                    route_flat, [0, expert * stride + row], [1, W13_M_TILE],
                                    valid_shape=[1, valid],
                                )
                                compact_flat = pl.store(
                                    pl.set_validshape(weight_row, 1, valid),
                                    [0, activation_base + row], compact_flat,
                                )
            work_base = work_base + row_tiles * (2 * (MOE_INTER // W13_N_TILE))
            activation_base = activation_base + count
    return pair, compact_flat, ready


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
                        lifted = pl.reg.maximums(normalized, -SWIGLU_FP8_MAX, active)
                        bounded = pl.reg.minimums(lifted, SWIGLU_FP8_MAX, active)
                        encoded = pl.reg.convert(bounded, dtype=pl.FP8E4M3FN, rounding="R", saturate=True)
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


# W2 uses aligned expert-strided staging for compact MX_A_ZZ activations.
W2_M_TILE = gcd(RECV_MAX, 80)
W2_N_TILE = MX_W2_D_OUT_TILE
W2_K_TILE = 128
W2_K_STAGE = 2


@pl.jit.inline(auto_scope=False)
def routed_w2_pypto(
    payload: pl.Tensor,
    scales: pl.Tensor,
    counts: pl.Tensor,
    weight: pl.Tensor,
    weight_scale: pl.Tensor,
    fp4_lut: pl.Tensor,
    output: pl.Tensor,
    input_ready: pl.Scalar[pl.TASK_ID],
    experts: pl.constexpr,
    stride: pl.constexpr,
):
    """Project compact MXFP8 activations with expert-specific MXFP4 weights.

    Payload and output capacity are at least experts * stride rows.
    Scale staging stores full 16-row blocks; only active rows are consumed.
    """
    aligned_scale = pl.create_tensor([1, experts * stride * H_SCALE], dtype=pl.FP8E8M0)
    with pl.spmd(ROUTED_VECTOR_WORKER_TILE, name_hint="routed_w2_align", deps=[input_ready]) as aligned_ready:
        core = pl.tile.get_block_idx()
        compact = 0
        for expert in pl.range(experts):
            count = pl.cast(pl.read(counts, [expert, 0]), pl.INDEX)
            for row in pl.range(core * 16, count, ROUTED_VECTOR_WORKER_TILE * 16):
                valid = pl.min(16, count - row)
                source = compact + row
                scale_loaded = pl.load(scales, [0, source * H_SCALE], [1, 16 * H_SCALE], valid_shape=[1, valid * H_SCALE], target_memory=pl.MemorySpace.Vec)
                scale_source = pl.reinterpret_view(scale_loaded, pl.UINT8)
                packed_padded = pl.fillpad(scale_source, pad_value=0)
                packed_words = pl.reinterpret_view(packed_padded, pl.INT16)
                packed_flat = pl.reshape(packed_words, [16 * H_SCALE // 2])
                packed_out = pl.create_tile([16 * H_SCALE // 2], dtype=pl.INT16, target_memory=pl.MemorySpace.Vec)
                with pl.reg_scope():
                    scale_index = pl.reg.index_gen(0, lanes=64, dtype=pl.UINT16)
                    scale_pair = pl.reg.shift_right(scale_index, pl.reg.broadcast(4, lanes=64, dtype=pl.UINT16))
                    source_row = pl.reg.bitwise_and(scale_index, pl.reg.broadcast(15, lanes=64, dtype=pl.UINT16))
                    row_start = pl.reg.mul(source_row, pl.reg.broadcast(H_SCALE // 2, lanes=64, dtype=pl.UINT16))
                    offsets_base = pl.reg.add(row_start, scale_pair)
                    for chunk in pl.hardware_loop(16 * H_SCALE // 128):
                        group_offset = pl.cast(chunk * 4, pl.UINT16)
                        offsets = pl.reg.adds(offsets_base, group_offset, pl.reg.full_mask(lanes=64))
                        values = pl.reg.gather(packed_flat, offsets, pl.reg.full_mask(lanes=64))
                        packed_alias = pl.reg.store(packed_out, chunk * 64, values)
                packed_tile = pl.reshape(packed_out, [1, 16 * H_SCALE // 2])
                packed_scale = pl.reinterpret_view(pl.reinterpret_view(packed_tile, pl.UINT8), pl.FP8E8M0)
                aligned_scale = pl.store(packed_scale, [0, (expert * stride + row) * H_SCALE], aligned_scale)
            compact = compact + count

    x_scale_zz = pl.tensor.view(aligned_scale, [experts * stride, H_SCALE], layout=pl.MX_A_ZZ)
    weight_flat = pl.tensor.view(weight, [experts * MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], layout=pl.ND)
    with pl.spmd(ROUTED_CUBE_WORKER_TILE, name_hint="routed_w2_pypto", deps=[aligned_ready]) as ready:
        core = pl.tile.get_block_idx()
        compact = 0
        work_base = 0
        for expert in pl.range(experts):
            count = pl.cast(pl.read(counts, [expert, 0]), pl.INDEX)
            row_tiles = (count + W2_M_TILE - 1) // W2_M_TILE
            for row_tile in pl.range(row_tiles):
                row = row_tile * W2_M_TILE
                valid = pl.min(W2_M_TILE, count - row)
                x_row = expert * stride + row
                for block in pl.range(D // W2_N_TILE):
                    if (work_base + row_tile * (D // W2_N_TILE) + block) % ROUTED_CUBE_WORKER_TILE == core:
                        col = block * W2_N_TILE
                        panel = expert * MX_W2_PACKED_ROWS + block * MOE_INTER
                        lut = pl.load(fp4_lut, [0, 0], [1, FP4_LUT_SIZE])
                        a = pl.load(payload, [compact + row, 0], [W2_M_TILE, W2_K_TILE])
                        a_scale = pl.load(x_scale_zz, [x_row, 0], [W2_M_TILE, W2_K_TILE // MX_GROUP])
                        packed = pl.load(weight_flat, [panel, 0], [W2_K_TILE, MX_PACKED_LANE_COLS])
                        index = pl.cast(pl.cast(packed, pl.FP16), pl.INT16)
                        scratch = pl.tile.create([1, MX_PACKED_LANE_COLS], dtype=pl.INT16)
                        codes = pl.tile.gather(lut, index, scratch)
                        b = pl.reinterpret_view(pl.reinterpret_view(codes, pl.INT8), pl.FP8E4M3FN)
                        b_scale = pl.load(weight_scale, [expert * H_SCALE, col], [W2_K_TILE // MX_GROUP, W2_N_TILE])
                        acc = pl.matmul_mx(a, a_scale, b, b_scale)
                        for k in pl.pipeline(W2_K_TILE, MOE_INTER, W2_K_TILE, stage=W2_K_STAGE):
                            a = pl.load(payload, [compact + row, k], [W2_M_TILE, W2_K_TILE])
                            a_scale = pl.load(x_scale_zz, [x_row, k // MX_GROUP], [W2_M_TILE, W2_K_TILE // MX_GROUP])
                            packed = pl.load(weight_flat, [panel + k, 0], [W2_K_TILE, MX_PACKED_LANE_COLS])
                            index = pl.cast(pl.cast(packed, pl.FP16), pl.INT16)
                            scratch = pl.tile.create([1, MX_PACKED_LANE_COLS], dtype=pl.INT16)
                            codes = pl.tile.gather(lut, index, scratch)
                            b = pl.reinterpret_view(pl.reinterpret_view(codes, pl.INT8), pl.FP8E4M3FN)
                            b_scale = pl.load(weight_scale, [expert * H_SCALE + k // MX_GROUP, col], [W2_K_TILE // MX_GROUP, W2_N_TILE])
                            acc = pl.matmul_mx_acc(acc, a, a_scale, b, b_scale)
                        output = pl.store(pl.set_validshape(acc, valid, W2_N_TILE), [compact + row, col], output)
            work_base = work_base + row_tiles * (D // W2_N_TILE)
            compact = compact + count
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
    input_ready: pl.Scalar[pl.TASK_ID],
    experts: pl.constexpr,
    stride: pl.constexpr,
    width: pl.constexpr,
    hidden: pl.constexpr,
    row_capacity: pl.constexpr,
):
    """Run BF16 W13, FP32 weighted MXFP8 activation, and full-K BF16 W2.

    Counts are immutable after input_ready, lie in [0, stride], and sum to
    at most row_capacity. Stride is divisible by 16; width and hidden are
    divisible by 256.
    """
    activation_rows = ((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE
    pair = pl.create_tensor([((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE, 2 * hidden], dtype=pl.BF16)
    compact_weights = pl.create_tensor([((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE, 1], dtype=pl.FP32)
    with pl.spmd(ROUTED_CUBE_WORKER_TILE, name_hint="routed_w13", deps=[input_ready]) as w13_ready:
        pair, compact_weights = _routed_w13(
            x, x_scale, counts, w1, w1_scale, w3, w3_scale, pair,
            weights, compact_weights, experts, stride, width, hidden, activation_rows, row_capacity,
        )

    payload = pl.create_tensor([((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE, hidden], dtype=pl.FP8E4M3FN)
    scales = pl.create_tensor([1, ((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE * (hidden // MX_GROUP)], dtype=pl.FP8E8M0)
    with pl.spmd(ROUTED_VECTOR_WORKER_TILE, name_hint="routed_swiglu_quant", deps=[w13_ready]) as activation_ready:
        payload, scales = _routed_swiglu_quant(
            pair, compact_weights, counts, payload, scales, experts, stride, hidden, row_capacity,
        )

    output_rows = ((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE
    output = pl.create_tensor([((row_capacity + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE, width], dtype=pl.BF16)
    with pl.spmd(ROUTED_CUBE_WORKER_TILE, name_hint="routed_w2", deps=[activation_ready]) as ready:
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
    local_experts: pl.constexpr = N_LOCAL_EXPERTS,
    topk: pl.constexpr = M.num_experts_per_tok,
):
    """Compute active expert rows and scatter each source interval to its route destinations."""
    recv_x_flat = pl.reshape(recv_x, [local_experts * RECV_MAX, D])
    route_weights = pl.reshape(recv_weights, [local_experts * RECV_MAX, 1])
    output, down_ready = _expert_routed_output(
        recv_x_flat, recv_mx_scale, route_weights, recv_expert_count,
        routed_w1_packed, routed_w1_scale, routed_w3_packed, routed_w3_scale,
        routed_w2_packed, routed_w2_scale, input_ready,
        local_experts, RECV_MAX, D, MOE_INTER,
        (local_experts if local_experts < topk else topk) * RECV_MAX,
    )
    with pl.spmd(ROUTED_VECTOR_WORKER_TILE, name_hint="ffn_scatter", deps=[down_ready]) as scatter_done:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="ffn_scatter", no_dep_args=[routed_output]):
            core = pl.tile.get_block_idx()
            total_rows = 0
            for expert in pl.range(local_experts):
                rows_read = pl.read(recv_expert_count, [expert, 0])
                rows = pl.cast(rows_read, pl.INDEX)
                total_rows = total_rows + rows
            rows_per_core = total_rows // ROUTED_VECTOR_WORKER_TILE
            extra_rows = total_rows % ROUTED_VECTOR_WORKER_TILE
            core_begin = core * rows_per_core + pl.min(core, extra_rows)
            core_rows = rows_per_core + pl.min(pl.max(extra_rows - core, 0), 1)
            core_end = core_begin + core_rows
            compact_base = 0
            for local_e in pl.range(local_experts):
                count_read = pl.read(recv_expert_count, [local_e, 0])
                count = pl.cast(count_read, pl.INDEX)
                expert_begin = pl.max(compact_base, core_begin)
                expert_end = pl.min(compact_base + count, core_end)
                if expert_begin < expert_end:
                    source_begin = 0
                    for src in pl.range(EP_WORLD_SIZE):
                        source_rows_read = pl.read(recv_meta, [src, local_e])
                        source_rows = pl.cast(source_rows_read, pl.INDEX)
                        source_end = source_begin + source_rows
                        valid_end = pl.min(source_end, count)
                        valid_rows = pl.max(valid_end - source_begin, 0)
                        source_compact = compact_base + source_begin
                        first = pl.min(valid_rows, pl.max(core_begin - source_compact, 0))
                        end = pl.min(valid_rows, pl.max(core_end - source_compact, 0))
                        for work in pl.pipeline(first, end, stage=2):
                            row = source_begin + work
                            route_read = pl.read(recv_route, [local_e, row])
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
    input_ready = pl.system.task_dummy(deps=[])
    recv_x_flat = pl.reshape(recv_x, [N_LOCAL_EXPERTS * RECV_MAX, D])
    route_weights = pl.reshape(recv_weights, [N_LOCAL_EXPERTS * RECV_MAX, 1])
    output, down_ready = _expert_routed_output(
        recv_x_flat, recv_mx_scale, route_weights, recv_expert_count,
        routed_w1_packed, routed_w1_scale, routed_w3_packed, routed_w3_scale,
        routed_w2_packed, routed_w2_scale, input_ready,
        N_LOCAL_EXPERTS, RECV_MAX, D, MOE_INTER, N_LOCAL_EXPERTS * RECV_MAX,
    )
    with pl.spmd(64, name_hint="ffn_copy", deps=[down_ready]):
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


# Standalone SwiGLU activation validation. The extern kernel and the PyPTO
# formulation are held to the same golden so the PyPTO path stays correct while
# its performance is tuned; the model keeps calling the extern either way.
SWIGLU_ROWS = ((N_LOCAL_EXPERTS * RECV_MAX + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE
PAYLOAD_SENTINEL = 0x5A
SCALE_SENTINEL = 0xA5


@pl.jit(auto_scope=False)
def l2_routed_swiglu_quant(
    pair: pl.Tensor[[SWIGLU_ROWS, 2 * MOE_INTER], pl.BF16],
    compact_weights: pl.Tensor[[SWIGLU_ROWS, 1], pl.FP32],
    counts: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    payload: pl.InOut[pl.Tensor[[SWIGLU_ROWS, MOE_INTER], pl.FP8E4M3FN]],
    scales: pl.InOut[pl.Tensor[[1, SWIGLU_ROWS * H_SCALE], pl.FP8E8M0]],
):
    """Validation entry for the extern activation kernel the model uses."""
    ready = pl.system.task_dummy(deps=[])
    with pl.spmd(ROUTED_VECTOR_WORKER_TILE, name_hint="routed_swiglu_quant", deps=[ready]):
        payload, scales = _routed_swiglu_quant(
            pair, compact_weights, counts, payload, scales,
            N_LOCAL_EXPERTS, RECV_MAX, MOE_INTER, N_LOCAL_EXPERTS * RECV_MAX,
        )
    return payload, scales


@pl.jit(auto_scope=False)
def l2_routed_swiglu_quant_pypto(
    pair: pl.Tensor[[SWIGLU_ROWS, 2 * MOE_INTER], pl.BF16],
    compact_weights: pl.Tensor[[SWIGLU_ROWS, 1], pl.FP32],
    counts: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    payload: pl.InOut[pl.Tensor[[SWIGLU_ROWS, MOE_INTER], pl.FP8E4M3FN]],
    scales: pl.InOut[pl.Tensor[[1, SWIGLU_ROWS * H_SCALE], pl.FP8E8M0]],
):
    """Validation entry for the PyPTO activation; the model uses the extern."""
    ready = pl.system.task_dummy(deps=[])
    payload, scales, activation_ready = routed_swiglu_quant_pypto(
        pair, compact_weights, counts, payload, scales, ready, N_LOCAL_EXPERTS,
    )
    return payload, scales


def golden_routed_swiglu_quant(tensors):
    """Torch reference for the weighted clamped SwiGLU and its MXFP8 quantization."""
    import torch.nn.functional as F

    from quant import quantize_mxfp8_upward_e8m0

    total = int(tensors["counts"].sum().item())
    payload = tensors["payload"].view(torch.uint8)
    codes = tensors["scales"].view(torch.uint8).reshape(SWIGLU_ROWS, H_SCALE)
    if total == 0:
        return
    pair = tensors["pair"][:total].float()
    gate, up = pair[:, :MOE_INTER], pair[:, MOE_INTER:]
    if SWIGLU_LIMIT > 0:
        gate = gate.clamp(max=SWIGLU_LIMIT)
        up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    hidden = F.silu(gate) * up * tensors["compact_weights"][:total].float()
    values, scale_codes = quantize_mxfp8_upward_e8m0(hidden)
    # Only the active rows are defined. The extern leaves the rest untouched;
    # the PyPTO path also writes the remainder of the tail row tile. Both must
    # leave everything past that tile boundary at the sentinel, which is what
    # the comparators check -- the gap itself is ignored.
    payload[:total] = values.view(torch.uint8)
    codes[:total] = scale_codes


def _swiglu_row_split(inputs):
    """Return the active row total and the end of the tail row tile."""
    total = int(inputs["counts"].sum().item())
    return total, ((total + SWIGLU_ROW_TILE - 1) // SWIGLU_ROW_TILE) * SWIGLU_ROW_TILE


def _e4m3_rank(raw):
    """Map FP8E4M3 bytes onto a signed magnitude order so neighbours differ by one.

    Codes are sign-magnitude, so the raw byte only orders values within one
    sign. Negating the magnitude for negative codes makes the whole encoding
    monotone, and then adjacent representable values are exactly one apart.
    """
    magnitude = (raw & 0x7F).to(torch.int16)
    return torch.where((raw & 0x80) != 0, -magnitude, magnitude)


def compare_swiglu_payload(actual, expected, *, inputs, **kwargs):
    """Compare active FP8 payload codes, allowing at most one E4M3 ULP of drift."""
    from golden import ratio_reldiff

    total, padded = _swiglu_row_split(inputs)
    actual_rows = actual.reshape(SWIGLU_ROWS, MOE_INTER)
    expected_rows = expected.reshape(SWIGLU_ROWS, MOE_INTER)
    tail_actual = actual_rows[padded:].view(torch.uint8)
    if not torch.equal(tail_actual, torch.full_like(tail_actual, PAYLOAD_SENTINEL)):
        return False, f"payload rows past the tail tile (>= {padded}) were written"
    if total == 0:
        return True, ""
    # The device composes SiLU from a hardware exp while the golden uses torch's,
    # so a value sitting near an E4M3 rounding midpoint may land one code apart.
    # Bound that directly on the codes: ratio_reldiff alone would not, because
    # its per-point cap is unbounded by default and its absolute-error shortcut
    # admits a zero against a small nonzero. One code is the whole budget.
    drift = (_e4m3_rank(actual_rows[:total].view(torch.uint8))
             - _e4m3_rank(expected_rows[:total].view(torch.uint8))).abs()
    worst = int(drift.max().item())
    if worst > 1:
        count = int((drift > 1).sum().item())
        return False, f"{count} payload codes differ by more than one E4M3 ULP (worst {worst})"
    # Delegate the distribution check to the shared comparator as well, so the
    # share of elements allowed to drift at all stays bounded.
    comparator = ratio_reldiff(diff_thd=0.0625, pct_thd=0.01)
    return comparator(actual_rows[:total].float(), expected_rows[:total].float(), inputs=inputs, **kwargs)


def compare_swiglu_scales(actual, expected, *, inputs, **kwargs):
    """Require exact E8M0 codes; a group amax is not exp-sensitive enough to drift."""
    total, padded = _swiglu_row_split(inputs)
    actual_rows = actual.view(torch.uint8).reshape(SWIGLU_ROWS, H_SCALE)
    expected_rows = expected.view(torch.uint8).reshape(SWIGLU_ROWS, H_SCALE)
    tail = actual_rows[padded:]
    if not torch.equal(tail, torch.full_like(tail, SCALE_SENTINEL)):
        return False, f"scale codes past the tail tile (>= {padded}) were written"
    if total == 0:
        return True, ""
    if torch.equal(actual_rows[:total], expected_rows[:total]):
        return True, ""
    wrong = int((actual_rows[:total] != expected_rows[:total]).sum().item())
    return False, f"{wrong} of {total * H_SCALE} active scale codes differ"


def build_swiglu_specs(case="bounds", *, tail_offset=0):
    """Build activation inputs with NaN-poisoned inactive capacity."""
    from golden import TensorSpec

    counts = validation_counts(case, tail_offset=tail_offset)
    total = int(counts.sum().item())
    generator = torch.Generator().manual_seed(20261110)
    # W13 emits BF16 projections; this spread straddles the SWIGLU_LIMIT clamp.
    pair = (torch.randn(SWIGLU_ROWS, 2 * MOE_INTER, generator=generator) * 6.0).bfloat16()
    weights = torch.rand(SWIGLU_ROWS, 1, generator=generator)
    # Nothing past the active total is defined; poison it so a read shows up.
    pair[total:] = float("nan")
    weights[total:] = float("nan")
    fp8, e8m0 = torch.float8_e4m3fn, torch.float8_e8m0fnu
    payload = torch.full((SWIGLU_ROWS, MOE_INTER), PAYLOAD_SENTINEL, dtype=torch.uint8).view(fp8)
    scales = torch.full((1, SWIGLU_ROWS * H_SCALE), SCALE_SENTINEL, dtype=torch.uint8).view(e8m0)
    return [
        TensorSpec("pair", [SWIGLU_ROWS, 2 * MOE_INTER], torch.bfloat16, init_value=pair),
        TensorSpec("compact_weights", [SWIGLU_ROWS, 1], torch.float32, init_value=weights),
        TensorSpec("counts", [N_LOCAL_EXPERTS, 1], torch.int32, init_value=counts),
        TensorSpec("payload", [SWIGLU_ROWS, MOE_INTER], fp8, init_value=payload),
        TensorSpec("scales", [1, SWIGLU_ROWS * H_SCALE], e8m0, init_value=scales),
    ]


def validate_routed_swiglu(args):
    """Run the extern and PyPTO activation entries against one golden."""
    from golden import run

    records = []
    # full is in the default set: bounds tops out at 320 of the 640-row
    # capacity, so without it the last rows are never active and a regression
    # that corrupts them at full occupancy would pass.
    cases = ("bounds", "tails", "zero", "full") if args.case == "all" else (args.case,)
    for case in cases:
        for label, entry in (("swiglu", l2_routed_swiglu_quant),
                             ("swiglu-pypto", l2_routed_swiglu_quant_pypto)):
            result = run(
                fn=entry,
                specs=build_swiglu_specs(case),
                golden_fn=golden_routed_swiglu_quant,
                compare_fn={"payload": compare_swiglu_payload, "scales": compare_swiglu_scales},
                golden_data=args.golden_data,
                save_data=args.save_data,
                compile_only=args.compile_only,
                runtime_dir=args.runtime_dir,
                config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode,
                            device_id=args.device, enable_chip_swimlane=args.enable_chip_swimlane,
                            dump_passes=args.dump_passes),
            )
            if not result.passed:
                raise RuntimeError(f"{label} {case}: {result.error}")
            status = ("COMPILE_ONLY" if args.compile_only
                      else "SIM_VALIDATED" if args.execution_mode else "DEVICE_VALIDATED")
            print(status, label, f"case={case}", result.work_dir)
            records.append(dict(mode=label, case=case, passed=result.passed))
    return records

# Standalone W13 projection validation. The extern kernel and the PyPTO
# formulation are held to the same golden so the PyPTO path stays correct while
# its performance is tuned; the model keeps calling the extern either way.
W13_ROWS = ((N_LOCAL_EXPERTS * RECV_MAX + MX_M_TILE - 1) // MX_M_TILE) * MX_M_TILE
PAIR_SENTINEL = 73.0
WEIGHT_SENTINEL = -41.0


@pl.jit(auto_scope=False)
def l2_routed_w13(
    x: pl.Tensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.FP8E4M3FN],
    x_scale: pl.Tensor[[1, N_LOCAL_EXPERTS * RECV_MAX * K_SCALE], pl.FP8E8M0],
    counts: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    w1: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    w3: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    route_weights: pl.Tensor[[N_LOCAL_EXPERTS * RECV_MAX, 1], pl.FP32],
    pair: pl.InOut[pl.Tensor[[W13_ROWS, 2 * MOE_INTER], pl.BF16]],
    compact_weights: pl.InOut[pl.Tensor[[W13_ROWS, 1], pl.FP32]],
):
    """Validation entry for the extern W13 projection the model uses."""
    ready = pl.system.task_dummy(deps=[])
    with pl.spmd(ROUTED_CUBE_WORKER_TILE, name_hint="routed_w13", deps=[ready]):
        pair, compact_weights = _routed_w13(
            x, x_scale, counts, w1, w1_scale, w3, w3_scale, pair,
            route_weights, compact_weights,
            N_LOCAL_EXPERTS, RECV_MAX, D, MOE_INTER, W13_ROWS, N_LOCAL_EXPERTS * RECV_MAX,
        )
    return pair, compact_weights


@pl.jit(auto_scope=False)
def l2_routed_w13_pypto(
    x: pl.Tensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.FP8E4M3FN],
    x_scale: pl.Tensor[[1, N_LOCAL_EXPERTS * RECV_MAX * K_SCALE], pl.FP8E8M0],
    counts: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    w1: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    w3: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    route_weights: pl.Tensor[[N_LOCAL_EXPERTS * RECV_MAX, 1], pl.FP32],
    fp4_lut: pl.Tensor[[1, FP4_LUT_SIZE], pl.INT16],
    pair: pl.InOut[pl.Tensor[[W13_ROWS, 2 * MOE_INTER], pl.BF16]],
    compact_weights: pl.InOut[pl.Tensor[[W13_ROWS, 1], pl.FP32]],
):
    """Validation entry for the PyPTO W13 projection; the model uses the extern."""
    ready = pl.system.task_dummy(deps=[])
    # The compacted weights come back as the flat [1, W13_ROWS] view the
    # kernel writes through; it aliases the same buffer as compact_weights.
    pair, compact_flat, _w13_ready = routed_w13_pypto(
        x, x_scale, counts, w1, w1_scale, w3, w3_scale, fp4_lut, pair,
        route_weights, compact_weights, ready, N_LOCAL_EXPERTS, RECV_MAX,
    )
    return pair, compact_flat


def golden_routed_w13(tensors):
    """Torch reference for the compact BF16 gate/up projections and route weights."""
    from quant import unpack_mx_scale_codes as decode_e8m0_codes
    from gemm import golden_gemm_mxfp8_to_fp32_dequantized as matmul_mx_golden

    x = tensors["x"]
    x_scale = decode_e8m0_codes(
        tensors["x_scale"].reshape(N_LOCAL_EXPERTS * RECV_MAX, K_SCALE), side="a",
    )
    w1_scale = decode_e8m0_codes(tensors["w1_scale"], side="b").reshape(
        N_LOCAL_EXPERTS, K_SCALE, MOE_INTER,
    )
    w3_scale = decode_e8m0_codes(tensors["w3_scale"], side="b").reshape(
        N_LOCAL_EXPERTS, K_SCALE, MOE_INTER,
    )
    pair = tensors["pair"]
    compact_weights = tensors["compact_weights"]
    packed_shape = (2 * (D // MX_K_TILE) * (MOE_INTER // MX_MM_INTER_TILE), MX_K_TILE * MX_MM_INTER_TILE // 4)
    compact = 0
    for expert, count in enumerate(tensors["counts"].reshape(-1).tolist()):
        if count == 0:
            continue
        begin = expert * RECV_MAX
        x_sub = x[begin:begin + count]
        x_scale_sub = x_scale[begin:begin + count]
        w1_fp8 = unpack_mxfp4_weight_tiles(
            tensors["w1"][expert].reshape(packed_shape), D, MOE_INTER, MX_K_TILE, MX_MM_INTER_TILE,
        )
        w3_fp8 = unpack_mxfp4_weight_tiles(
            tensors["w3"][expert].reshape(packed_shape), D, MOE_INTER, MX_K_TILE, MX_MM_INTER_TILE,
        )
        gate = matmul_mx_golden(x_sub, x_scale_sub, w1_fp8, w1_scale[expert])
        up = matmul_mx_golden(x_sub, x_scale_sub, w3_fp8, w3_scale[expert])
        pair[compact:compact + count, :MOE_INTER] = gate.bfloat16()
        pair[compact:compact + count, MOE_INTER:] = up.bfloat16()
        compact_weights[compact:compact + count] = tensors["route_weights"][begin:begin + count]
        compact += count


def _w13_active_rows(inputs):
    """Total compact rows the projection defines."""
    return int(inputs["counts"].sum().item())


def compare_w13_pair(actual, expected, *, inputs, **kwargs):
    """Compare active gate/up columns and require the inactive rows untouched."""
    from golden import ratio_reldiff

    total = _w13_active_rows(inputs)
    actual_rows = actual.reshape(W13_ROWS, 2 * MOE_INTER)
    expected_rows = expected.reshape_as(actual_rows)
    tail = actual_rows[total:]
    if not torch.equal(tail, torch.full_like(tail, PAIR_SENTINEL)):
        return False, f"projection rows past the active total ({total}) were written"
    if total == 0:
        return True, ""
    comparator = ratio_reldiff(diff_thd=2e-3, pct_thd=0.01)
    return comparator(actual_rows[:total], expected_rows[:total], inputs=inputs, **kwargs)


def compare_w13_weights(actual, expected, *, inputs, **kwargs):
    """Route weights are copied, not computed, so require them bit-exact."""
    total = _w13_active_rows(inputs)
    actual_rows = actual.reshape(W13_ROWS, 1)
    expected_rows = expected.reshape_as(actual_rows)
    tail = actual_rows[total:]
    if not torch.equal(tail, torch.full_like(tail, WEIGHT_SENTINEL)):
        return False, f"route weight rows past the active total ({total}) were written"
    if total == 0 or torch.equal(actual_rows[:total], expected_rows[:total]):
        return True, ""
    wrong = int((actual_rows[:total] != expected_rows[:total]).sum().item())
    return False, f"{wrong} of {total} compact route weights differ"


def build_fp4_lut():
    """Decode table indexed by a packed byte, low nibble in the low byte."""
    from quant import FP4_E4M3_CODES

    codes = FP4_E4M3_CODES.to(torch.int64)
    index = torch.arange(FP4_LUT_SIZE, dtype=torch.int64)
    entries = codes[index & 0xF] | (codes[index >> 4] << 8)
    return entries.to(torch.uint16).view(torch.int16).reshape(1, FP4_LUT_SIZE)


def build_w13_specs(case="bounds", *, tail_offset=0, with_lut=False):
    """Build W13 inputs with poisoned inactive capacity and lazy packed weights."""
    from functools import cache

    from golden import TensorSpec
    from quant import pack_mx_a_scale, quantize_mxfp8_upward_e8m0

    counts = validation_counts(case, tail_offset=tail_offset)

    @cache
    def activations():
        generator = torch.Generator().manual_seed(20261110)
        source = torch.randn(N_LOCAL_EXPERTS * RECV_MAX, D, dtype=torch.bfloat16, generator=generator)
        payload, codes = quantize_mxfp8_upward_e8m0(source.float())
        codes = codes.view(torch.uint8)
        route = torch.rand(N_LOCAL_EXPERTS * RECV_MAX, 1, generator=generator)
        # Nothing past an expert's count is defined; poison it so a read shows up.
        for expert, count in enumerate(counts.reshape(-1).tolist()):
            begin = expert * RECV_MAX
            payload.view(torch.uint8)[begin + count:begin + RECV_MAX] = 127
            codes[begin + count:begin + RECV_MAX] = 255
            route[begin + count:begin + RECV_MAX] = float("nan")
        packed_codes = pack_mx_a_scale(codes).view(torch.float8_e8m0fnu)
        return payload, packed_codes.reshape(1, -1), route

    @cache
    def routed_weights():
        return gen_routed_mxfp4_weights(N_LOCAL_EXPERTS, ROUTED_DEQUANT_STD, seed_base=10)

    fp8 = torch.float8_e4m3fn
    e8m0 = torch.float8_e8m0fnu
    specs = [
        TensorSpec("x", [N_LOCAL_EXPERTS * RECV_MAX, D], fp8, init_value=lambda: activations()[0]),
        TensorSpec("x_scale", [1, N_LOCAL_EXPERTS * RECV_MAX * K_SCALE], e8m0, init_value=lambda: activations()[1]),
        TensorSpec("counts", [N_LOCAL_EXPERTS, 1], torch.int32, init_value=counts),
        TensorSpec("w1", [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], torch.uint8, init_value=lambda: routed_weights()[0]),
        TensorSpec("w1_scale", [N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], e8m0, init_value=lambda: routed_weights()[1]),
        TensorSpec("w3", [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], torch.uint8, init_value=lambda: routed_weights()[2]),
        TensorSpec("w3_scale", [N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], e8m0, init_value=lambda: routed_weights()[3]),
        TensorSpec("route_weights", [N_LOCAL_EXPERTS * RECV_MAX, 1], torch.float32, init_value=lambda: activations()[2]),
    ]
    # Only the PyPTO entry decodes weights on device, so only it takes the table.
    if with_lut:
        specs.append(TensorSpec("fp4_lut", [1, FP4_LUT_SIZE], torch.int16, init_value=build_fp4_lut()))
    specs += [
        TensorSpec("pair", [W13_ROWS, 2 * MOE_INTER], torch.bfloat16, init_value=PAIR_SENTINEL),
        TensorSpec("compact_weights", [W13_ROWS, 1], torch.float32, init_value=WEIGHT_SENTINEL),
    ]
    return specs


@pl.jit(auto_scope=False)
def l2_routed_w2(
    payload: pl.Tensor[[W13_ROWS, MOE_INTER], pl.FP8E4M3FN],
    scales: pl.Tensor[[1, W13_ROWS * H_SCALE], pl.FP8E8M0],
    counts: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    weight: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    weight_scale: pl.Tensor[[N_LOCAL_EXPERTS * H_SCALE, D], pl.FP8E8M0, pl.MX_B_NN],
    output: pl.InOut[pl.Tensor[[W13_ROWS, D], pl.BF16]],
):
    """Validate the unchanged C++ W2 projection independently."""
    ready = pl.system.task_dummy(deps=[])
    with pl.spmd(ROUTED_CUBE_WORKER_TILE, name_hint="routed_w2", deps=[ready]):
        output = _routed_w2(
            payload, scales, counts, weight, weight_scale, output,
            N_LOCAL_EXPERTS, RECV_MAX, D, MOE_INTER, W13_ROWS, W13_ROWS, N_LOCAL_EXPERTS * RECV_MAX,
        )
    return output


@pl.jit(auto_scope=False)
def l2_routed_w2_pypto(
    payload: pl.Tensor[[W13_ROWS, MOE_INTER], pl.FP8E4M3FN],
    scales: pl.Tensor[[1, W13_ROWS * H_SCALE], pl.FP8E8M0],
    counts: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    weight: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    weight_scale: pl.Tensor[[N_LOCAL_EXPERTS * H_SCALE, D], pl.FP8E8M0, pl.MX_B_NN],
    fp4_lut: pl.Tensor[[1, FP4_LUT_SIZE], pl.INT16],
    output: pl.InOut[pl.Tensor[[W13_ROWS, D], pl.BF16]],
):
    """Validate native W2 against the standalone C++ projection golden."""
    ready = pl.system.task_dummy(deps=[])
    output, _ready = routed_w2_pypto(
        payload, scales, counts, weight, weight_scale, fp4_lut, output, ready, N_LOCAL_EXPERTS, RECV_MAX,
    )
    return output


def golden_routed_w2(tensors):
    """Compute compact BF16 expert projections with independent unpacking."""
    from quant import unpack_mx_scale_codes
    from gemm import golden_gemm_mxfp8_to_fp32_dequantized

    scale = tensors["scales"].reshape(W13_ROWS, H_SCALE).view(torch.uint8)
    weight_scale = unpack_mx_scale_codes(tensors["weight_scale"], side="b")
    weight_scale = weight_scale.reshape(N_LOCAL_EXPERTS, H_SCALE, D)
    packed_shape = (2 * (MOE_INTER // MX_K_TILE) * (D // MX_W2_D_OUT_TILE), MX_K_TILE * MX_W2_D_OUT_TILE // 4)
    compact = 0
    for expert, count in enumerate(tensors["counts"].reshape(-1).tolist()):
        if count:
            weight = tensors["weight"][expert].reshape(packed_shape)
            decoded = unpack_mxfp4_weight_tiles(weight, MOE_INTER, D, MX_K_TILE, MX_W2_D_OUT_TILE)
            x = tensors["payload"][compact:compact + count]
            result = golden_gemm_mxfp8_to_fp32_dequantized(x, scale[compact:compact + count], decoded, weight_scale[expert])
            tensors["output"][compact:compact + count] = result.bfloat16()
        compact += count


def compare_w2_output(actual, expected, *, inputs, **kwargs):
    """Check active BF16 results and exact untouched inactive output rows."""
    from golden import ratio_reldiff

    total = int(inputs["counts"].sum().item())
    actual = actual.reshape(W13_ROWS, D)
    expected = expected.reshape_as(actual)
    tail = actual[total:]
    if not torch.equal(tail, torch.full_like(tail, PAIR_SENTINEL)):
        return False, f"W2 rows past the active total ({total}) were written"
    if total == 0:
        return True, ""
    return ratio_reldiff(diff_thd=2e-3, pct_thd=0.01)(actual[:total], expected[:total], inputs=inputs, **kwargs)


def build_w2_specs(case="bounds", *, tail_offset=0, with_lut=False):
    """Build compact activations, poisoned capacity, and MXFP4 W2 weights."""
    from functools import cache
    from golden import TensorSpec
    from quant import quantize_mxfp8_upward_e8m0

    counts = validation_counts(case, tail_offset=tail_offset)
    total = int(counts.sum().item())

    @cache
    def activations():
        generator = torch.Generator().manual_seed(20261112)
        source = torch.randn(W13_ROWS, MOE_INTER, dtype=torch.bfloat16, generator=generator)
        payload, scales = quantize_mxfp8_upward_e8m0(source.float())
        payload.view(torch.uint8)[total:] = 127
        scales.view(torch.uint8)[total:] = 255
        return payload, scales.reshape(1, -1)

    @cache
    def weights():
        return gen_routed_mxfp4_weights(N_LOCAL_EXPERTS, ROUTED_DEQUANT_STD, seed_base=10)

    specs = [
        TensorSpec("payload", [W13_ROWS, MOE_INTER], torch.float8_e4m3fn, init_value=lambda: activations()[0]),
        TensorSpec("scales", [1, W13_ROWS * H_SCALE], torch.float8_e8m0fnu, init_value=lambda: activations()[1]),
        TensorSpec("counts", [N_LOCAL_EXPERTS, 1], torch.int32, init_value=counts),
        TensorSpec("weight", [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], torch.uint8, init_value=lambda: weights()[4]),
        TensorSpec("weight_scale", [N_LOCAL_EXPERTS * H_SCALE, D], torch.float8_e8m0fnu, init_value=lambda: weights()[5]),
    ]
    if with_lut:
        specs.append(TensorSpec("fp4_lut", [1, FP4_LUT_SIZE], torch.int16, init_value=build_fp4_lut()))
    specs.append(TensorSpec("output", [W13_ROWS, D], torch.bfloat16, init_value=PAIR_SENTINEL))
    return specs


if __name__ == "__main__":
    from golden import run

    run_config = dict(
        dump_passes=args.dump_passes,
        backend_type=args.backend_type,
        execution_mode=args.execution_mode,
        device_id=args.device,
        enable_chip_swimlane=args.enable_chip_swimlane,
    )

    def _run(label, fn, specs, golden_fn, compare_fn):
        result = run(
            fn=fn,
            specs=specs,
            golden_fn=golden_fn,
            golden_data=args.golden_data,
            save_data=args.save_data,
            compile_only=args.compile_only,
            runtime_dir=args.runtime_dir,
            config=run_config,
            rtol=1e-3,
            atol=1e-3,
            compare_fn=compare_fn,
        )
        if not result.passed:
            print(f"{label}: FAILED")
            if result.error:
                print(result.error)
            raise SystemExit(1)

    if args.mode in ("all", "swiglu"):
        validate_routed_swiglu(args)
    if args.mode == "swiglu":
        raise SystemExit(0)

    cases = ("bounds", "tails", "zero") if args.case == "all" else (args.case,)
    modes = ("w13", "w2", "chain") if args.mode == "all" else (args.mode,)
    for case in cases:
        tail_rounds = (6 + N_LOCAL_EXPERTS - 1) // N_LOCAL_EXPERTS if case == "tails" else 1
        for tail_round in range(tail_rounds):
            offset = tail_round * N_LOCAL_EXPERTS
            if "w13" in modes:
                # Both entries run against the same golden so the extern and
                # the PyPTO formulation stay interchangeable row for row.
                for label, entry, lut in (("extern", l2_routed_w13, False),
                                          ("pypto", l2_routed_w13_pypto, True)):
                    print(f"Routed W13 ({label}): case={case}, tail_round={tail_round}, experts={N_LOCAL_EXPERTS}, stride={RECV_MAX}")
                    _run(
                        f"Routed W13 ({label})", entry,
                        build_w13_specs(case, tail_offset=offset, with_lut=lut), golden_routed_w13,
                        {"pair": compare_w13_pair, "compact_weights": compare_w13_weights},
                    )
            if "w2" in modes:
                for label, entry, lut in (("extern", l2_routed_w2, False), ("pypto", l2_routed_w2_pypto, True)):
                    print(f"Routed W2 ({label}): case={case}, tail_round={tail_round}, experts={N_LOCAL_EXPERTS}, stride={RECV_MAX}")
                    _run(
                        f"Routed W2 ({label})", entry,
                        build_w2_specs(case, tail_offset=offset, with_lut=lut), golden_routed_w2,
                        {"output": compare_w2_output},
                    )
            if "chain" in modes:
                print(f"Local routed compute: case={case}, tail_round={tail_round}, experts={N_LOCAL_EXPERTS}, stride={RECV_MAX}")
                _run(
                    "Local routed compute", local_compute_test,
                    build_tensor_specs(case, tail_offset=offset), golden_local_compute,
                    {"routed_y": compare_local_compute},
                )
            if args.compile_only or args.golden_data:
                break
        if args.compile_only or args.golden_data:
            break
