# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 EP transport using the DSpark dispatch and combine protocol.

The tensor layouts retain the A5 MXFP4/MXFP8 ABI. The transport expects one
unique local token shard per rank and does not perform Attention-TP owner
filtering.
"""
import pypto.language as pl
import pypto.language.distributed as pld
from models.deepseek_v4_1_flash import config as C

from models.deepseek_v4_1_flash.config import N_EXPERTS

T = C.MOE_TOKENS
D = C.D
TOPK = C.TOPK
N_RANKS = C.EP_SIZE
N_LOCAL = C.N_LOCAL_EXPERTS
N_ROUTES = T * TOPK
RECV_MAX = C.MOE_RECV_MAX
MX_GROUP = C.MX_GROUP
K_SCALE = D // MX_GROUP
MAX_PER_SRC = T
AUX_W = 0
AUX_PAD = C.AUX_WIDTH
IDX_PAD = C.ROUTE_WIDTH
SIGNAL_PAD = 128  # 512-byte padded monotonic counter lane
SCALE_COPY_TILE = 256
SCALE_PACK_TMP = ((64 + K_SCALE + 31) // 32) * 32
RECV_TILE = 16
SCALE_PACK_TILES = (RECV_MAX + RECV_TILE - 1) // RECV_TILE

@pl.jit.inline
def dispatch(
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    x_norm_mx: pl.Tensor[[T, D], pl.FP8E4M3FN],
    x_norm_scale: pl.Tensor[[1, T * K_SCALE], pl.FP8E8M0],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    recv_x_out: pl.Tensor[[N_LOCAL, RECV_MAX, D], pl.FP8E4M3FN],
    recv_scale_out: pl.Tensor[[1, N_LOCAL * RECV_MAX * K_SCALE], pl.FP8E8M0],
    recv_weight_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.FP32],
    recv_route_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.INT32],
    recv_count_out: pl.Tensor[[N_LOCAL, 1], pl.INT32],
    recv_meta_local: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL * RECV_MAX, K_SCALE], pl.UINT8],
    recv_weights: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_routes: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, SIGNAL_PAD], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, N_LOCAL, SIGNAL_PAD], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    # Flat 2-D view kept outside the scope so it stays a tensor view, not a tile.
    recv_x_out_flat = pl.reshape(recv_x_out, [N_LOCAL * RECV_MAX, D])
    # ``quant_mx`` stores MX_A_ZZ bytes physically as
    # [1, M/16, G/2, 16, 2].  Dispatch needs one logical token's scales, so
    # read that backing through its physical ND view instead of scalar-reading
    # an MX-layout tensor (which is intentionally unsupported).
    x_norm_mx_raw = pl.create_tensor([T, D], dtype=pl.INT8)
    with pl.spmd(T, name_hint="dispatch_fp8_raw_copy"):
        copy_row = pl.tile.get_block_idx()
        raw_row = pl.load(x_norm_mx, [copy_row, 0], [1, D])
        raw_row_i8 = pl.reinterpret_view(raw_row, pl.INT8)
        x_norm_mx_raw = pl.store(
            raw_row_i8,
            [copy_row, 0],
            x_norm_mx_raw,
        )

    x_norm_scale_raw = pl.create_tensor([1, T * K_SCALE], dtype=pl.UINT8)
    with pl.spmd((T * K_SCALE) // SCALE_COPY_TILE, name_hint="dispatch_e8m0_raw_copy"):
        scale_copy_offset = pl.tile.get_block_idx() * SCALE_COPY_TILE
        raw_scale = pl.load(x_norm_scale, [0, scale_copy_offset], [1, SCALE_COPY_TILE])
        raw_scale_u8 = pl.reinterpret_view(raw_scale, pl.UINT8)
        x_norm_scale_raw = pl.store(raw_scale_u8, [0, scale_copy_offset], x_norm_scale_raw)
    x_norm_scale_physical = pl.tensor.view(
        x_norm_scale_raw,
        [1, T // 16, K_SCALE // 2, 16, 2],
        layout=pl.ND,
    )

    # This ABI has no consumed window argument, so this only publishes an explicit
    # start task for the current dispatch round that the later stage, metadata and
    # payload phases depend on; epoch handling in the caller owns the window lifetime.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_reuse_wait") as _reuse_tid:
        _indices_anchor = pl.read(indices, [0, 0])

    # Stage the logical MX scale rows in ND order. The dispatch producers move
    # them with tensor.put because a full (MAX_PER_SRC, K_SCALE) tile exceeds UB.
    scale_src = pl.create_tensor([T, K_SCALE], dtype=pl.UINT8, manual_dep=True)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_stage", deps=[_reuse_tid]) as _stage_tid:
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)
        for t in pl.range(active_tokens):
            for group in pl.range(K_SCALE):
                scale = pl.read(
                    x_norm_scale_physical,
                    [0, t // 16, group // 2, t % 16, group % 2],
                )
                pl.write(scale_src, [t, group], scale)

    # Publish route counts independently from the registration-only peer wait.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dispatch_meta_publish",
        deps=[_reuse_tid],
        allow_early_resolve=True,
    ) as _meta_push_tid:
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)

        # Count how many routes land in each (dst, loc_e) lane (no payload move).
        cursor = pl.array.create(N_RANKS * N_LOCAL, pl.INT32)
        for d in pl.range(N_RANKS):
            for e in pl.range(N_LOCAL):
                cursor[d * N_LOCAL + e] = 0
        indices_tile = pl.tile.load(
            indices,
            [0, 0],
            [T, IDX_PAD],
            valid_shape=[T, TOPK],
        )
        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.tile.read(indices_tile, [t, k])
                dst = eid // N_LOCAL
                loc_e = eid - dst * N_LOCAL
                cursor[dst * N_LOCAL + loc_e] = cursor[dst * N_LOCAL + loc_e] + 1

        meta_tile = pl.tile.full([1, N_LOCAL], dtype=pl.INT32, value=0)
        for dst in pl.range(N_RANKS):
            for e in pl.range(N_LOCAL):
                pl.tile.write(meta_tile, [0, e], cursor[dst * N_LOCAL + e])
            pld.tile.remote_store(
                meta_tile, target=recv_meta, peer=dst, offsets=[my_rank, 0]
            )
            if dst != my_rank:
                pld.system.notify(
                    target=arrived, peer=dst, offsets=[my_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_meta_wait") as _meta_wait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(
                    signal=arrived, offsets=[src, 0],
                    expected=moe_epoch, cmp=pld.WaitCmp.Ge,
                )

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dispatch_meta_finalize",
        deps=[_meta_push_tid, _meta_wait_tid],
        allow_early_resolve=True,
    ) as _meta_tid:
        for e in pl.range(N_LOCAL):
            acc = pl.const(0, pl.INT32)
            for src in pl.range(N_RANKS):
                count = pl.read(recv_meta, [src, e])
                pl.write(recv_meta_local, [src, e], count)
                acc = acc + count
            pl.write(recv_count_out, [e, 0], acc)

    # One producer owns one (destination, local expert) lane. Aux and route rows
    # are accumulated in UB and published once; x and scale retain v4.1's
    # physical MXFP8 representation and are pushed directly from GM.
    with pl.spmd(
        N_RANKS * N_LOCAL,
        name_hint="dispatch_push",
        deps=[_reuse_tid, _stage_tid],
        allow_early_resolve=True,
    ) as _push_tid:
        push_block = pl.tile.get_block_idx()
        dst = push_block // N_LOCAL
        loc_e = push_block - dst * N_LOCAL
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)

        e_lane_base = loc_e * RECV_MAX + my_rank * MAX_PER_SRC
        indices_tile = pl.tile.load(
            indices,
            [0, 0],
            [T, IDX_PAD],
            valid_shape=[T, TOPK],
        )
        weights_tile = pl.tile.load(
            weights,
            [0, 0],
            [T, AUX_PAD],
            valid_shape=[T, TOPK],
        )
        aux_lane = pl.tile.full(
            [MAX_PER_SRC, AUX_PAD],
            dtype=pl.FP32,
            value=0.0,
        )
        route_lane = pl.tile.full(
            [MAX_PER_SRC, IDX_PAD],
            dtype=pl.INT32,
            value=0,
        )
        slot_ctr = pl.array.create(1, pl.INT32)
        slot_ctr[0] = 0

        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.tile.read(indices_tile, [t, k])
                route_dst = eid // N_LOCAL
                route_local_e = eid - route_dst * N_LOCAL
                if route_local_e == loc_e:
                    if route_dst == dst:
                        slot = slot_ctr[0]
                        slot_ctr[0] = slot + 1
                        row = e_lane_base + slot
                        pld.tensor.put(
                            dst=recv_x,
                            peer=dst,
                            src=x_norm_mx_raw,
                            dst_offsets=[row, 0],
                            src_offsets=[t, 0],
                            shape=[1, D],
                        )
                        pld.tensor.put(
                            dst=recv_scale,
                            peer=dst,
                            src=scale_src,
                            dst_offsets=[row, 0],
                            src_offsets=[t, 0],
                            shape=[1, K_SCALE],
                        )
                        route_index = pl.cast(t * TOPK + k, pl.INT32)
                        route_weight = pl.tile.read(weights_tile, [t, k])
                        pl.tile.write(aux_lane, [slot, AUX_W], route_weight)
                        pl.tile.write(route_lane, [slot, 0], route_index)

        pld.tile.remote_store(
            aux_lane,
            target=recv_weights,
            peer=dst,
            offsets=[e_lane_base, 0],
        )
        pld.tile.remote_store(
            route_lane,
            target=recv_routes,
            peer=dst,
            offsets=[e_lane_base, 0],
        )
        if dst != my_rank:
            pld.system.notify(
                target=data_arrived,
                peer=dst,
                offsets=[my_rank, 0, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dispatch_wait",
        deps=[_push_tid],
    ) as _wait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(
                    signal=data_arrived,
                    offsets=[src, 0, 0],
                    expected=pl.cast(moe_epoch * N_LOCAL, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    # Gather lanes into compact per-expert buffers. One task owns one aligned
    # row tile, so scalar metadata writes never share a 64-byte cache line.
    recv_scale_nd = pl.create_tensor(
        [N_LOCAL * RECV_MAX, K_SCALE], dtype=pl.FP8E8M0, manual_dep=True
    )
    gather_tids = pl.array.create(N_LOCAL, pl.TASK_ID)
    for e in pl.parallel(N_LOCAL):
        gather_tile_tids = pl.array.create(SCALE_PACK_TILES, pl.TASK_ID)
        for tile in pl.range(SCALE_PACK_TILES):
            gather_tile_tids[tile] = pl.system.task_dummy(deps=[])
        expert_rows = pl.cast(pl.read(recv_count_out, [e, 0]), pl.INDEX)
        expert_tiles = (expert_rows + RECV_TILE - 1) // RECV_TILE
        for tile in pl.parallel(expert_tiles):
            tile_row = tile * RECV_TILE
            tile_end = pl.min(tile_row + RECV_TILE, expert_rows)
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="dispatch_gather",
                deps=[_wait_tid, _push_tid, _meta_tid],
                allow_early_resolve=False,
            ) as gather_tile_tid:
                e_base_row = e * RECV_MAX
                source_begin = pl.cast(0, pl.INDEX)
                for src in pl.range(N_RANKS):
                    source_rows = pl.cast(
                        pl.read(recv_meta_local, [src, e]),
                        pl.INDEX,
                    )
                    source_end = source_begin + source_rows
                    src_base_row = e_base_row + src * MAX_PER_SRC
                    gather_begin = pl.max(tile_row, source_begin)
                    gather_end = pl.min(tile_end, source_end)
                    for out_col in pl.range(gather_begin, gather_end):
                        in_row = src_base_row + out_col - source_begin
                        out_row = e_base_row + out_col
                        recv_x_raw = pl.load(recv_x, [in_row, 0], [1, D])
                        recv_x_mx = pl.reinterpret_view(recv_x_raw, pl.FP8E4M3FN)
                        recv_x_out_flat = pl.store(
                            recv_x_mx,
                            [out_row, 0],
                            recv_x_out_flat,
                        )
                        recv_scale_raw = pl.load(
                            recv_scale,
                            [in_row, 0],
                            [1, K_SCALE],
                        )
                        recv_scale_mx = pl.reinterpret_view(
                            recv_scale_raw,
                            pl.FP8E8M0,
                        )
                        recv_scale_nd = pl.store(
                            recv_scale_mx,
                            [out_row, 0],
                            recv_scale_nd,
                        )
                        pl.write(
                            recv_weight_out,
                            [e, out_col],
                            pl.read(recv_weights, [in_row, AUX_W]),
                        )
                        pl.write(
                            recv_route_out,
                            [e, out_col],
                            pl.read(recv_routes, [in_row, 0]),
                        )
                    source_begin = source_end
            gather_tile_tids[tile] = gather_tile_tid
        gather_tids[e] = pl.system.task_dummy(deps=[gather_tile_tids])
    _gather_tid = pl.system.task_dummy(
        deps=[gather_tids[e] for e in range(N_LOCAL)]
    )

    with pl.spmd(
        N_LOCAL * SCALE_PACK_TILES,
        name_hint="dispatch_scale_pack",
        deps=[_gather_tid],
    ) as _scale_pack_tid:
        block_idx = pl.tile.get_block_idx()
        local_e = block_idx // SCALE_PACK_TILES
        tile_idx = block_idx % SCALE_PACK_TILES
        e_rows = pl.read(recv_count_out, [local_e, 0])
        if tile_idx * RECV_TILE < e_rows:
            flat_t0 = local_e * RECV_MAX + tile_idx * RECV_TILE
            scale_nd = pl.load(recv_scale_nd, [flat_t0, 0], [RECV_TILE, K_SCALE])
            scale_raw = pl.reinterpret_view(scale_nd, pl.UINT8)
            tmp = pl.create_tile([1, SCALE_PACK_TMP], dtype=pl.UINT8)
            scale_zz_raw = pl.tmov_x2zz(
                scale_raw,
                tmp,
                group_axis=1,
                dst_rows=RECV_TILE,
                dst_cols=K_SCALE,
            )
            scale_zz = pl.reinterpret_view(scale_zz_raw, pl.FP8E8M0)
            recv_scale_out = pl.store(
                pl.reshape(scale_zz, [1, RECV_TILE * K_SCALE]),
                [0, flat_t0 * K_SCALE],
                recv_scale_out,
            )
    return _scale_pack_tid


@pl.jit.inline
def combine_scattered(
    shared_output: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    routed_output: pld.DistributedTensor[[T * TOPK, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, N_LOCAL, SIGNAL_PAD], pl.INT32],
    output_ready: pl.Scalar[pl.TASK_ID],
    scatter_done: pl.Scalar[pl.TASK_ID],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    with pl.spmd(
        N_LOCAL,
        name_hint="combine_notify",
        deps=[scatter_done],
    ) as notify_tid:
        local_e = pl.tile.get_block_idx()
        if local_e < N_LOCAL:
            for peer in pl.range(N_RANKS):
                if peer != my_rank:
                    pld.system.notify(
                        target=combine_arrived,
                        peer=peer,
                        offsets=[my_rank, 0, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="combine_wait",
        deps=[notify_tid],
    ) as _cwait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(
                    signal=combine_arrived,
                    offsets=[src, 0, 0],
                    expected=pl.cast(moe_epoch * N_LOCAL, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    # The notify fan-out starts only after every local scatter. Waiting for the
    # accumulated peer epochs closes both local and remote routed-output writes.
    # Accumulate the shared-expert and TOP-K routed results for every local token.
    with pl.spmd(
        T,
        name_hint="combine_reduce",
        deps=[_cwait_tid, output_ready],
    ) as _reduce_tid:
        t = pl.tile.get_block_idx()
        if t < num_tokens:
            acc = pl.cast(pl.load(shared_output, [t, 0], [1, D]), target_type=pl.FP32)
            for k in pl.range(TOPK):
                acc = pl.add(acc, pl.cast(pl.load(routed_output, [t * TOPK + k, 0], [1, D]), target_type=pl.FP32))
            ffn_out = pl.store(pl.cast(acc, target_type=pl.BF16, mode="rint"), [t, 0], ffn_out)
    # No consumed window is declared, so no recycle signal is published.
    return _reduce_tid


@pl.jit.inline
def combine(
    recv_y: pl.Tensor[[N_LOCAL, RECV_MAX, D], pl.BF16],
    recv_route_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.INT32],
    shared_output: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    recv_meta_local: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
    routed_output: pld.DistributedTensor[[T * TOPK, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, N_LOCAL, SIGNAL_PAD], pl.INT32],
    output_ready: pl.Scalar[pl.TASK_ID],
    expert_ready: pl.Scalar[pl.TASK_ID],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL * RECV_MAX, D])
    with pl.spmd(
        N_LOCAL,
        name_hint="combine",
        deps=[expert_ready],
    ) as scatter_done:
        e = pl.tile.get_block_idx()
        e_base_row = e * RECV_MAX
        source_begin = pl.cast(0, pl.INDEX)
        for src in pl.range(N_RANKS):
            source_rows = pl.cast(pl.read(recv_meta_local, [src, e]), pl.INDEX)
            source_end = source_begin + source_rows
            for compact_row in pl.range(source_begin, source_end):
                route = pl.cast(pl.read(recv_route_out, [e, compact_row]), pl.INDEX)
                pld.tensor.put(
                    dst=routed_output,
                    peer=src,
                    src=recv_y_flat,
                    dst_offsets=[route, 0],
                    src_offsets=[e_base_row + compact_row, 0],
                    shape=[1, D],
                )
            source_begin = source_end

    return combine_scattered(
        shared_output,
        ffn_out,
        routed_output,
        combine_arrived,
        output_ready,
        scatter_done,
        num_tokens,
        my_rank,
        moe_epoch,
    )


# ---------------------------------------------------------------------------
# Standalone transport tests
# ---------------------------------------------------------------------------
# Dispatch checks the packed receive buffers; combine checks route scatter and
# source-rank reduction.


@pl.jit
def dispatch_test(
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    x_norm_mx: pl.Tensor[[T, D], pl.FP8E4M3FN],
    x_norm_scale: pl.Tensor[[1, T * K_SCALE], pl.FP8E8M0],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    recv_x_out: pl.Out[pl.Tensor[[N_LOCAL, RECV_MAX, D], pl.FP8E4M3FN]],
    recv_scale_out: pl.Out[
        pl.Tensor[[1, N_LOCAL * RECV_MAX * K_SCALE], pl.FP8E8M0]
    ],
    recv_weight_out: pl.Out[pl.Tensor[[N_LOCAL, RECV_MAX], pl.FP32]],
    recv_route_out: pl.Out[pl.Tensor[[N_LOCAL, RECV_MAX], pl.INT32]],
    recv_count_out: pl.Out[pl.Tensor[[N_LOCAL, 1], pl.INT32]],
    recv_meta_local: pl.Out[pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32]],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL * RECV_MAX, K_SCALE], pl.UINT8],
    recv_weights: pld.DistributedTensor[
        [N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32
    ],
    recv_routes: pld.DistributedTensor[
        [N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32
    ],
    arrived: pld.DistributedTensor[[N_RANKS, SIGNAL_PAD], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, N_LOCAL, SIGNAL_PAD], pl.INT32],
    num_tokens: pl.Tensor[[N_RANKS], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    local_num_tokens = pl.read(num_tokens, [my_rank])
    dispatch(
        indices, x_norm_mx, x_norm_scale, weights,
        recv_x_out, recv_scale_out, recv_weight_out, recv_route_out,
        recv_count_out, recv_meta_local,
        recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
        arrived, data_arrived,
        local_num_tokens, my_rank, moe_epoch,
    )
    return (
        recv_x_out, recv_scale_out, recv_weight_out, recv_route_out,
        recv_count_out, recv_meta_local,
    )


@pl.jit.host
def l3_dispatch(
    indices: pl.Tensor[[N_RANKS, T, TOPK], pl.INT32],
    x_norm_mx: pl.Tensor[[N_RANKS, T, D], pl.FP8E4M3FN],
    x_norm_scale: pl.Tensor[[N_RANKS, 1, T * K_SCALE], pl.FP8E8M0],
    weights: pl.Tensor[[N_RANKS, T, TOPK], pl.FP32],
    recv_x_out: pl.Out[
        pl.Tensor[[N_RANKS, N_LOCAL, RECV_MAX, D], pl.FP8E4M3FN]
    ],
    recv_scale_out: pl.Out[
        pl.Tensor[[N_RANKS, 1, N_LOCAL * RECV_MAX * K_SCALE], pl.FP8E8M0]
    ],
    recv_weight_out: pl.Out[
        pl.Tensor[[N_RANKS, N_LOCAL, RECV_MAX], pl.FP32]
    ],
    recv_route_out: pl.Out[
        pl.Tensor[[N_RANKS, N_LOCAL, RECV_MAX], pl.INT32]
    ],
    recv_count_out: pl.Out[
        pl.Tensor[[N_RANKS, N_LOCAL, 1], pl.INT32]
    ],
    recv_meta_local: pl.Out[
        pl.Tensor[[N_RANKS, N_RANKS, N_LOCAL], pl.INT32]
    ],
    num_tokens: pl.Tensor[[N_RANKS], pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer(
        [N_LOCAL * RECV_MAX, D], dtype=pl.INT8
    )
    recv_scale_buf = pld.alloc_window_buffer(
        [N_LOCAL * RECV_MAX, K_SCALE], dtype=pl.UINT8
    )
    recv_weights_buf = pld.alloc_window_buffer(
        [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32
    )
    recv_routes_buf = pld.alloc_window_buffer(
        [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32
    )
    arrived_buf = pld.alloc_window_buffer([N_RANKS, SIGNAL_PAD], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer(
        [N_RANKS, N_LOCAL, SIGNAL_PAD], dtype=pl.INT32
    )

    for r in pl.range(pld.world_size()):
        recv_meta = pld.window(
            recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32
        )
        recv_x = pld.window(
            recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8
        )
        recv_scale = pld.window(
            recv_scale_buf, [N_LOCAL * RECV_MAX, K_SCALE], dtype=pl.UINT8
        )
        recv_weights = pld.window(
            recv_weights_buf,
            [N_LOCAL * RECV_MAX, AUX_PAD],
            dtype=pl.FP32,
        )
        recv_routes = pld.window(
            recv_routes_buf,
            [N_LOCAL * RECV_MAX, IDX_PAD],
            dtype=pl.INT32,
        )
        arrived = pld.window(
            arrived_buf, [N_RANKS, SIGNAL_PAD], dtype=pl.INT32
        )
        data_arrived = pld.window(
            data_arrived_buf,
            [N_RANKS, N_LOCAL, SIGNAL_PAD],
            dtype=pl.INT32,
        )
        dispatch_test(
            indices[r], x_norm_mx[r], x_norm_scale[r], weights[r],
            recv_x_out[r], recv_scale_out[r], recv_weight_out[r],
            recv_route_out[r], recv_count_out[r], recv_meta_local[r],
            recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
            arrived, data_arrived, num_tokens, r, moe_epoch,
            device=r,
        )


@pl.jit
def combine_test(
    recv_y: pl.Tensor[[N_LOCAL, RECV_MAX, D], pl.BF16],
    recv_route_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.INT32],
    shared_output: pl.Tensor[[T, D], pl.BF16],
    recv_meta_local: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
    ffn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    routed_output: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, N_LOCAL, SIGNAL_PAD], pl.INT32],
    num_tokens: pl.Tensor[[N_RANKS], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    local_num_tokens = pl.read(num_tokens, [my_rank])
    with pl.spmd(T, name_hint="combine_output_zero") as output_ready:
        t = pl.tile.get_block_idx()
        zero_row = pl.tile.full([1, D], dtype=pl.BF16, value=0.0)
        ffn_out = pl.store(zero_row, [t, 0], ffn_out)
    expert_ready = pl.system.task_dummy(deps=[])
    _combine_ready = combine(
        recv_y, recv_route_out, shared_output, ffn_out, recv_meta_local,
        routed_output, combine_arrived,
        output_ready, expert_ready,
        local_num_tokens, my_rank, moe_epoch,
    )
    return ffn_out


@pl.jit.host
def l3_combine(
    recv_y: pl.Tensor[[N_RANKS, N_LOCAL, RECV_MAX, D], pl.BF16],
    recv_route_out: pl.Tensor[[N_RANKS, N_LOCAL, RECV_MAX], pl.INT32],
    shared_output: pl.Tensor[[N_RANKS, T, D], pl.BF16],
    recv_meta_local: pl.Tensor[[N_RANKS, N_RANKS, N_LOCAL], pl.INT32],
    ffn_out: pl.Out[pl.Tensor[[N_RANKS, T, D], pl.BF16]],
    num_tokens: pl.Tensor[[N_RANKS], pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    routed_output_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer(
        [N_RANKS, N_LOCAL, SIGNAL_PAD], dtype=pl.INT32
    )
    for r in pl.range(pld.world_size()):
        routed_output = pld.window(
            routed_output_buf, [N_ROUTES, D], dtype=pl.BF16
        )
        combine_arrived = pld.window(
            combine_arrived_buf,
            [N_RANKS, N_LOCAL, SIGNAL_PAD],
            dtype=pl.INT32,
        )
        combine_test(
            recv_y[r], recv_route_out[r], shared_output[r],
            recv_meta_local[r], ffn_out[r],
            routed_output, combine_arrived,
            num_tokens, r, moe_epoch,
            device=r,
        )


def _pack_a_scale(scale_codes):
    """Pack logical [M, K/32] scale bytes into MX_A_ZZ order."""
    m, k_groups = scale_codes.shape
    if m % 16 or k_groups % 2:
        raise ValueError(f"invalid A-scale shape: {tuple(scale_codes.shape)}")
    return (
        scale_codes.reshape(m // 16, 16, k_groups // 2, 2)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(m, k_groups)
    )


def _unpack_a_scale(packed_codes):
    """Restore MX_A_ZZ scale bytes to logical [M, K/32] order."""
    m, k_groups = packed_codes.shape
    if m % 16 or k_groups % 2:
        raise ValueError(f"invalid packed A-scale shape: {tuple(packed_codes.shape)}")
    return (
        packed_codes.reshape(m // 16, k_groups // 2, 16, 2)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(m, k_groups)
    )


def _transport_routes(num_tokens):
    """Create one deterministic route table for every source rank.

    ``num_tokens`` may be a scalar (broadcast to all sources) or a length-
    ``N_RANKS`` sequence.  The latter is the public EP fixture ABI: each rank
    owns a possibly different local prefix, including an empty prefix.
    """
    import torch

    counts = _normalise_counts(num_tokens)
    routes = torch.zeros(N_RANKS, T, TOPK, dtype=torch.int32)
    for src in range(N_RANKS):
        for t in range(int(counts[src])):
            for k in range(TOPK):
                route = t * TOPK + k
                dst = route % N_RANKS
                local_e = (route // N_RANKS) % N_LOCAL
                routes[src, t, k] = dst * N_LOCAL + local_e
    return routes


def _normalise_counts(value):
    """Return validated per-source local token counts as an int32 tensor."""
    import torch

    counts = torch.as_tensor(value, dtype=torch.int32).reshape(-1)
    if counts.numel() == 1:
        counts = counts.repeat(N_RANKS)
    if counts.numel() != N_RANKS:
        raise ValueError(
            f"num_tokens must be a scalar or contain {N_RANKS} counts, "
            f"got shape {tuple(counts.shape)}"
        )
    if bool((counts < 0).any()) or bool((counts > T).any()):
        raise ValueError(
            f"num_tokens values must be in [0, {T}], got {counts.tolist()}"
        )
    return counts.contiguous()


def _exact_compare(actual, expected, **_kwargs):
    import torch

    if actual.shape != expected.shape:
        return False, (
            f"    output shape mismatch: actual={tuple(actual.shape)} "
            f"expected={tuple(expected.shape)}"
        )
    if torch.equal(actual, expected):
        return True, ""
    different = actual != expected
    count = int(different.sum().item())
    if actual.is_floating_point():
        diff = (actual.float() - expected.float()).abs()
        return False, (
            f"    exact mismatch: values={count}, "
            f"max_abs_diff={float(diff.max().item()):.6g}"
        )
    return False, f"    exact mismatch: values={count}"


def build_dispatch_specs(num_tokens=T):
    import torch

    from golden.spec import ScalarSpec, TensorSpec
    pack_a_scale = _pack_a_scale

    counts = _normalise_counts(num_tokens)
    routes = _transport_routes(counts)
    x = (torch.randn(N_RANKS, T, D) * 0.25).to(torch.float8_e4m3fn)
    scale_logical = torch.full(
        (N_RANKS * T, K_SCALE), 127, dtype=torch.uint8
    )
    scale = pack_a_scale(scale_logical).view(
        N_RANKS, T, K_SCALE
    ).reshape(N_RANKS, 1, T * K_SCALE).view(torch.float8_e8m0fnu)
    weights = torch.linspace(
        0.25, 1.0, steps=N_RANKS * T * TOPK, dtype=torch.float32
    ).reshape(N_RANKS, T, TOPK)
    return [
        TensorSpec("indices", [N_RANKS, T, TOPK], torch.int32,
                   init_value=lambda: routes),
        TensorSpec("x_norm_mx", [N_RANKS, T, D], torch.float8_e4m3fn,
                   init_value=lambda: x),
        TensorSpec("x_norm_scale", [N_RANKS, 1, T * K_SCALE],
                   torch.float8_e8m0fnu, init_value=lambda: scale),
        TensorSpec("weights", [N_RANKS, T, TOPK], torch.float32,
                   init_value=lambda: weights),
        TensorSpec("recv_x_out", [N_RANKS, N_LOCAL, RECV_MAX, D],
                   torch.float8_e4m3fn),
        TensorSpec("recv_scale_out",
                   [N_RANKS, 1, N_LOCAL * RECV_MAX * K_SCALE],
                   torch.float8_e8m0fnu),
        TensorSpec("recv_weight_out", [N_RANKS, N_LOCAL, RECV_MAX],
                   torch.float32),
        TensorSpec("recv_route_out", [N_RANKS, N_LOCAL, RECV_MAX],
                   torch.int32),
        TensorSpec("recv_count_out", [N_RANKS, N_LOCAL, 1], torch.int32),
        TensorSpec("recv_meta_local", [N_RANKS, N_RANKS, N_LOCAL],
                   torch.int32),
        TensorSpec("num_tokens", [N_RANKS], torch.int32,
                   init_value=lambda: counts),
        ScalarSpec("moe_epoch", torch.int32, 1, compile_runtime=True,
                   benchmark_step=1),
    ]


def golden_dispatch(tensors):
    import torch

    pack_a_scale = _pack_a_scale
    unpack_a_scale = _unpack_a_scale

    counts = _normalise_counts(tensors["num_tokens"])
    indices = tensors["indices"].to(torch.int64)
    x = tensors["x_norm_mx"]
    weights = tensors["weights"]
    scale_bytes = tensors["x_norm_scale"].view(torch.uint8).reshape(
        N_RANKS, T, K_SCALE
    )
    logical_scales = torch.stack([
        unpack_a_scale(scale_bytes[r]) for r in range(N_RANKS)
    ])

    recv_x = torch.zeros(
        N_RANKS, N_LOCAL, RECV_MAX, D, dtype=torch.float8_e4m3fn
    )
    recv_scale_logical = torch.zeros(
        N_RANKS, N_LOCAL * RECV_MAX, K_SCALE, dtype=torch.uint8
    )
    recv_weights = torch.zeros(
        N_RANKS, N_LOCAL, RECV_MAX, dtype=torch.float32
    )
    recv_routes = torch.zeros(
        N_RANKS, N_LOCAL, RECV_MAX, dtype=torch.int32
    )
    recv_meta = torch.zeros(N_RANKS, N_RANKS, N_LOCAL, dtype=torch.int32)
    cursors = torch.zeros(N_RANKS, N_RANKS, N_LOCAL, dtype=torch.int32)

    for src in range(N_RANKS):
        active = int(counts[src])
        for t in range(active):
            for k in range(TOPK):
                route = t * TOPK + k
                expert = int(indices[src, t, k])
                dst, local_e = divmod(expert, N_LOCAL)
                slot = int(cursors[src, dst, local_e])
                cursors[src, dst, local_e] += 1
                recv_meta[dst, src, local_e] += 1
                compact = int(recv_meta[dst, :src, local_e].sum().item()) + slot
                recv_x[dst, local_e, compact] = x[src, t]
                recv_scale_logical[
                    dst, local_e * RECV_MAX + compact
                ] = logical_scales[src, t]
                recv_weights[dst, local_e, compact] = weights[src, t, k]
                recv_routes[dst, local_e, compact] = route

    recv_counts = recv_meta.sum(dim=1).unsqueeze(-1)
    packed_scale = torch.stack([
        pack_a_scale(recv_scale_logical[r]) for r in range(N_RANKS)
    ]).reshape(N_RANKS, 1, N_LOCAL * RECV_MAX * K_SCALE)
    tensors["recv_x_out"][:] = recv_x
    tensors["recv_scale_out"][:] = packed_scale.view(torch.float8_e8m0fnu)
    tensors["recv_weight_out"][:] = recv_weights
    tensors["recv_route_out"][:] = recv_routes
    tensors["recv_count_out"][:] = recv_counts
    tensors["recv_meta_local"][:] = recv_meta


def build_combine_specs(num_tokens=T):
    import torch

    from golden.spec import ScalarSpec, TensorSpec

    counts = _normalise_counts(num_tokens)
    indices = _transport_routes(counts)
    recv_y = torch.zeros(
        N_RANKS, N_LOCAL, RECV_MAX, D, dtype=torch.bfloat16
    )
    recv_route = torch.zeros(
        N_RANKS, N_LOCAL, RECV_MAX, dtype=torch.int32
    )
    meta = torch.zeros(N_RANKS, N_RANKS, N_LOCAL, dtype=torch.int32)
    cursors = torch.zeros(N_RANKS, N_RANKS, N_LOCAL, dtype=torch.int32)
    for src in range(N_RANKS):
        active = int(counts[src])
        for t in range(active):
            for k in range(TOPK):
                route = t * TOPK + k
                expert = int(indices[src, t, k])
                dst, local_e = divmod(expert, N_LOCAL)
                slot = int(cursors[src, dst, local_e])
                cursors[src, dst, local_e] += 1
                meta[dst, src, local_e] += 1
                compact = int(meta[dst, :src, local_e].sum().item()) + slot
                recv_y[dst, local_e, compact] = torch.tensor(
                    0.25 * (route + 1), dtype=torch.bfloat16
                )
                recv_route[dst, local_e, compact] = route

    shared = torch.zeros(N_RANKS, T, D, dtype=torch.bfloat16)
    for r in range(N_RANKS):
        active = int(counts[r])
        for t in range(active):
            shared[r, t] = 0.5 * (t + 1)

    return [
        TensorSpec("recv_y", [N_RANKS, N_LOCAL, RECV_MAX, D],
                   torch.bfloat16, init_value=lambda: recv_y),
        TensorSpec("recv_route_out", [N_RANKS, N_LOCAL, RECV_MAX],
                   torch.int32, init_value=lambda: recv_route),
        TensorSpec("shared_output", [N_RANKS, T, D], torch.bfloat16,
                   init_value=lambda: shared),
        TensorSpec("recv_meta_local", [N_RANKS, N_RANKS, N_LOCAL],
                   torch.int32, init_value=lambda: meta),
        TensorSpec("ffn_out", [N_RANKS, T, D], torch.bfloat16),
        TensorSpec("num_tokens", [N_RANKS], torch.int32,
                   init_value=lambda: counts),
        ScalarSpec("moe_epoch", torch.int32, 1, compile_runtime=True,
                   benchmark_step=1),
    ]


def golden_combine(tensors):
    import torch

    counts = _normalise_counts(tensors["num_tokens"])
    shared = tensors["shared_output"].float()
    recv_y = tensors["recv_y"].float()
    recv_route = tensors["recv_route_out"].to(torch.int64)
    meta = tensors["recv_meta_local"].to(torch.int64)
    routed = torch.zeros(N_RANKS, N_ROUTES, D, dtype=torch.float32)
    for dst in range(N_RANKS):
        for e in range(N_LOCAL):
            base = 0
            for src in range(N_RANKS):
                count = int(meta[dst, src, e])
                for slot in range(count):
                    route = int(recv_route[dst, e, base + slot])
                    routed[src, route] = recv_y[dst, e, base + slot]
                base += count
    out = torch.zeros(N_RANKS, T, D, dtype=torch.bfloat16)
    for src in range(N_RANKS):
        active = int(counts[src])
        for t in range(active):
            value = shared[src, t].clone()
            for k in range(TOPK):
                value += routed[src, t * TOPK + k]
            out[src, t] = value.to(torch.bfloat16)
    tensors["ffn_out"][:] = out


def _run_one_test(kind, args):
    from golden.runner import run
    from pypto.ir import DistributedConfig

    device_ids = [int(value) for value in args.device.split(",")]
    if len(device_ids) != N_RANKS:
        raise ValueError(
            f"need exactly {N_RANKS} devices for EP{N_RANKS}, got {device_ids}"
        )
    config = dict(
        dump_passes=args.dump_passes,
        platform=args.platform,
        distributed_config=DistributedConfig(
            device_ids=device_ids, num_sub_workers=0
        ),
    )
    counts = args.num_tokens
    if args.num_tokens_per_rank is not None:
        try:
            counts = [int(value) for value in args.num_tokens_per_rank.split(",")]
        except ValueError as exc:
            raise ValueError("--num-tokens-per-rank must be comma-separated integers") from exc
    counts = _normalise_counts(counts)
    if kind == "dispatch":
        result = run(
            fn=l3_dispatch,
            specs=build_dispatch_specs(counts),
            golden_fn=golden_dispatch,
            compile_only=args.compile_only,
            save_data=args.save_data,
            config=config,
            rtol=0.0,
            atol=0.0,
            compare_fn={
                name: _exact_compare for name in (
                    "recv_x_out", "recv_scale_out", "recv_weight_out",
                    "recv_route_out", "recv_count_out", "recv_meta_local",
                )
            },
        )
    else:
        result = run(
            fn=l3_combine,
            specs=build_combine_specs(counts),
            golden_fn=golden_combine,
            compile_only=args.compile_only,
            save_data=args.save_data,
            config=config,
            rtol=0.0,
            atol=0.0,
            compare_fn={"ffn_out": _exact_compare},
        )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)




# Resident CED prefill precision variants.
_PREFILL_COMM_ROUTE_ROWS = pl.dynamic("V41_COMM_ROUTE_ROWS")


_PREFILL_COMM_WINDOW_ROWS = pl.dynamic("V41_COMM_WINDOW_ROWS")


_PREFILL_COMM_SIGNAL_PAD = 128


_PREFILL_COMM_ROWS = pl.dynamic("V41_COMM_ROWS")


_PREFILL_COMM_N_RANKS = 4


@pl.jit.inline(auto_scope=False)
def prefill_combine_routed_experts(
    local: pl.Tensor[[_PREFILL_COMM_ROUTE_ROWS, D], pl.FP32],
    routes: pl.Tensor[[_PREFILL_COMM_ROWS, TOPK], pl.INT32],
    gathered: pld.DistributedTensor[[_PREFILL_COMM_WINDOW_ROWS, D], pl.FP32],
    ready: pld.DistributedTensor[[_PREFILL_COMM_N_RANKS, _PREFILL_COMM_SIGNAL_PAD], pl.INT32],
    consumed: pld.DistributedTensor[[_PREFILL_COMM_N_RANKS, _PREFILL_COMM_SIGNAL_PAD], pl.INT32],
    output: pl.Tensor[[_PREFILL_COMM_ROWS, D], pl.FP32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    capacity: pl.Scalar[pl.INT32],
):
    """Add routed outputs in ascending expert order, then release the window.

    Local rows are token-major route slots. Rank r owns expert IDs
    [r * 96, (r + 1) * 96); each token has six distinct valid route IDs.
    The window has four slots of capacity * six rows. Epochs are consecutive
    and one-based; the caller adds the shared expert after this reduction.
    """
    local.bind_dynamic(0, _PREFILL_COMM_ROUTE_ROWS)
    routes.bind_dynamic(0, _PREFILL_COMM_ROWS)
    gathered.bind_dynamic(0, _PREFILL_COMM_WINDOW_ROWS)
    rows = pl.tensor.dim(routes, 0)
    with pl.spmd(
        _PREFILL_COMM_N_RANKS, name_hint="prefill_ep_publish", allow_early_resolve=False
    ) as publish_tid:
        peer = pl.tile.get_block_idx()
        if peer != rank:
            if epoch > 1:
                pld.system.wait(
                    signal=consumed,
                    offsets=[peer, 0],
                    expected=epoch - 1,
                    cmp=pld.WaitCmp.Ge,
                )
        pld.tensor.put(
            dst=gathered,
            peer=peer,
            src=local,
            dst_offsets=[rank * capacity * TOPK, 0],
            src_offsets=[0, 0],
            shape=[rows * TOPK, D],
            chunk_rows=1,
            chunk_cols=D,
            pipeline=True,
        )
        if peer != rank:
            pld.system.notify(
                target=ready,
                peer=peer,
                offsets=[rank, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_ep_wait",
        deps=[publish_tid],
        allow_early_resolve=False,
    ) as wait_tid:
        for peer in pl.range(_PREFILL_COMM_N_RANKS):
            if peer != rank:
                pld.system.wait(signal=ready, offsets=[peer, 0], expected=epoch, cmp=pld.WaitCmp.Ge)

    with pl.spmd(32, name_hint="prefill_ep_sum", deps=[wait_tid]) as sum_tid:
        worker = pl.tile.get_block_idx()
        for row in pl.range(worker, rows, 32):
            total = pl.tile.full([1, D], dtype=pl.FP32, value=0.0)
            previous = pl.const(-1, pl.INT32)
            for order in pl.range(TOPK):
                selected = pl.const(N_EXPERTS, pl.INT32)
                selected_slot = pl.cast(0, pl.INDEX)
                for slot in pl.range(TOPK):
                    expert = pl.read(routes, [row, slot])
                    if expert > previous:
                        if expert < selected:
                            selected = expert
                            selected_slot = slot
                owner = pl.cast(selected // (N_EXPERTS // _PREFILL_COMM_N_RANKS), pl.INDEX)
                source_row = owner * capacity * TOPK + row * TOPK + selected_slot
                value = pl.load(gathered, [source_row, 0], [1, D])
                total = pl.add(total, value)
                previous = selected
            output = pl.store(total, [row, 0], output)

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_ep_release",
        deps=[sum_tid],
    ) as release_tid:
        for peer in pl.range(_PREFILL_COMM_N_RANKS):
            if peer != rank:
                pld.system.notify(
                    target=consumed,
                    peer=peer,
                    offsets=[rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
    return output, release_tid

if __name__ == "__main__":
    import argparse
    import pathlib
    import sys
    _model_dir = pathlib.Path(__file__).resolve().parent
    sys.path = [item for item in sys.path if pathlib.Path(item or ".").resolve() != _model_dir]

    parser = argparse.ArgumentParser(
        description="V4.1 standalone EP dispatch/combine tests"
    )
    parser.add_argument("test", choices=("dispatch", "combine", "both"),
                        nargs="?", default="both")
    parser.add_argument("-p", "--platform", default="a5",
                        choices=("a2a3", "a2a3sim", "a5", "a5sim"))
    parser.add_argument("--ep", type=int, default=N_RANKS)
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("-d", "--device", default=",".join(
        str(i) for i in range(N_RANKS)
    ))
    parser.add_argument("--num-tokens", type=int, default=T)
    parser.add_argument(
        "--num-tokens-per-rank", type=str, default=None,
        help="comma-separated local counts (overrides --num-tokens)",
    )
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--dump-passes", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.num_tokens <= T:
        parser.error(f"--num-tokens must be in [0, {T}]")
    if args.num_tokens_per_rank is not None:
        try:
            _normalise_counts([int(value) for value in args.num_tokens_per_rank.split(",")])
        except (ValueError, TypeError) as exc:
            parser.error(str(exc))
    if args.test in ("dispatch", "both"):
        _run_one_test("dispatch", args)
    if args.test in ("combine", "both"):
        _run_one_test("combine", args)
