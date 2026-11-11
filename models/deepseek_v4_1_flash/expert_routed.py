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
from config import ROUTE_T_DYN, N_LOCAL_EXPERTS, MOE_TOKENS

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
    routed_output: pld.DistributedTensor[[ROUTE_T_DYN, D], pl.BF16],
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
