# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 EP8 transport migrated from V4-Pro and Flash-MTP.

Each function documents its migration source; the data layout follows V4-Pro and the
signal layout is the [EP, 1] form required by issue #1205, with TP-owner filtering
applied in the dispatch-count and payload stages.
"""
import pypto.language as pl
import pypto.language.distributed as pld
from models.deepseek_v4_1_flash import config as C

T = C.MOE_TOKENS
D = C.D
TOPK = C.TOPK
N_RANKS = C.EP_SIZE
N_LOCAL = C.N_LOCAL_EXPERTS
N_ROUTES = T * TOPK
RECV_MAX = C.RECV_MAX
MX_GROUP = C.MX_GROUP
K_SCALE = D // MX_GROUP
MAX_PER_SRC = T
AUX_W = 0
AUX_PAD = C.AUX_WIDTH
IDX_PAD = C.ROUTE_WIDTH
SIGNAL_PAD = 1
SCALE_COPY_TILE = 256
SCALE_PACK_TMP = ((64 + K_SCALE + 31) // 32) * 32
RECV_TILE = 16

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
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    token_owners: pl.Tensor[[T], pl.INT32],
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

    # Meta and payload arrivals ride two independent windows (`arrived` /
    # `data_arrived`). Each producer publishes its current epoch into a unique
    # padded slot, so metadata can gate route construction without waiting for
    # the bulk payload barrier or contending on a shared counter.

    # Stage meta and payload rows locally so their remote publications can use
    # self-draining tensor puts before the matching notifications are issued.
    aux_src = pl.create_tensor([N_ROUTES, AUX_PAD], dtype=pl.FP32)
    route_src = pl.create_tensor([N_ROUTES, IDX_PAD], dtype=pl.INT32)
    scale_src = pl.create_tensor([T, K_SCALE], dtype=pl.UINT8, manual_dep=True)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_stage", deps=[_reuse_tid]) as _stage_tid:
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)
        for t in pl.range(active_tokens):
            is_owner = pl.read(token_owners, [t]) == my_rank
            for k in pl.range(TOPK):
                if is_owner:
                    r = t * TOPK + k
                    aux_tile = pl.tile.full([1, AUX_PAD], dtype=pl.FP32, value=0.0)
                    aux_weight = pl.read(weights, [t, k])
                    pl.tile.write(aux_tile, [0, AUX_W], aux_weight)
                    pl.store(aux_tile, [r, 0], aux_src)

                    route_tile = pl.tile.full([1, IDX_PAD], dtype=pl.INT32, value=0)
                    route_index = pl.cast(r, pl.INT32)
                    pl.tile.write(route_tile, [0, 0], route_index)
                    pl.store(route_tile, [r, 0], route_src)
            for group in pl.range(K_SCALE):
                scale = pl.read(
                    x_norm_scale_physical,
                    [0, t // 16, group // 2, t % 16, group % 2],
                )
                pl.write(scale_src, [t, group], scale)

    # Phase 1: count routes, publish counts, barrier on meta only, then cumsum ->
    # recv_count_out. Earliest recv_count_out can be produced -- it needs every
    # source's counts but none of the bulk payload.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dispatch_meta",
        deps=[_reuse_tid],
    ) as _meta_tid:
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
        for t in pl.range(active_tokens):
            is_owner = pl.read(token_owners, [t]) == my_rank
            for k in pl.range(TOPK):
                if is_owner:
                    eid = pl.read(indices, [t, k])
                    dst = eid // N_LOCAL
                    loc_e = eid - dst * N_LOCAL
                    cursor[dst * N_LOCAL + loc_e] = cursor[dst * N_LOCAL + loc_e] + 1

        # Publish one complete metadata tile per destination: metadata is a tile
        # remote_store, not a sequence of scalar puts, so each destination observes
        # one coherent row.
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

        # Wait for every source's metadata publication.
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(
                    signal=arrived, offsets=[src, 0],
                    expected=moe_epoch, cmp=pld.WaitCmp.Ge,
                )

        # Cumsum the published per-source counts into compact local lanes.
        for e in pl.range(N_LOCAL):
            acc = pl.const(0, pl.INT32)
            for src in pl.range(N_RANKS):
                count = pl.read(recv_meta, [src, e])
                pl.write(recv_meta_local, [src, e], count)
                acc = acc + count
            pl.write(recv_count_out, [e, 0], acc)

    # Phase 2: move the bulk payload (x / aux / route) to each destination lane.
    # Rides its own `data_arrived` window, so it needs no ordering against the meta
    # phase and overlaps it freely.
    # Split over LOCAL EXPERT INDEX (N_LOCAL blocks): block loc_e handles expert
    # loc_e on EVERY destination rank, so the blocking cross-rank puts fan out
    # across N_LOCAL cores. One slot counter per destination rank; token-major
    # order matches the meta pass's per-(dst, loc_e) cumulative count, so the
    # padded lane layout the gather compacts is identical to the single-block push.
    with pl.spmd(N_LOCAL, name_hint="dispatch_push", deps=[_reuse_tid, _stage_tid]) as _push_tid:
        loc_e = pl.tile.get_block_idx()
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)

        slot_ctr = pl.array.create(N_RANKS, pl.INT32)
        for d in pl.range(N_RANKS):
            slot_ctr[d] = 0
        e_lane_base = loc_e * RECV_MAX + my_rank * MAX_PER_SRC

        for t in pl.range(active_tokens):
            is_owner = pl.read(token_owners, [t]) == my_rank
            for k in pl.range(TOPK):
                if is_owner:
                    eid = pl.read(indices, [t, k])
                    dst = eid // N_LOCAL
                    le = eid - dst * N_LOCAL
                    if le == loc_e:
                        slot = slot_ctr[dst]
                        slot_ctr[dst] = slot + 1
                        # lane (loc_e, my_rank, slot) on peer=dst
                        row = e_lane_base + slot
                        r_route = t * TOPK + k
                        pld.tensor.put(
                            dst=recv_x, peer=dst, src=x_norm_mx_raw,
                            dst_offsets=[row, 0], src_offsets=[t, 0], shape=[1, D],
                        )
                        pld.tensor.put(
                            dst=recv_scale, peer=dst, src=scale_src,
                            dst_offsets=[row, 0], src_offsets=[t, 0], shape=[1, K_SCALE],
                        )
                        pld.tensor.put(
                            dst=recv_weights, peer=dst, src=aux_src,
                            dst_offsets=[row, 0], src_offsets=[r_route, 0], shape=[1, AUX_PAD],
                        )
                        pld.tensor.put(
                            dst=recv_routes, peer=dst, src=route_src,
                            dst_offsets=[row, 0], src_offsets=[r_route, 0], shape=[1, IDX_PAD],
                        )

        # Publish this block's epoch only after its self-draining payload puts.
        # One cache-line-padded slot per source/block avoids shared-word and
        # false-sharing races between the N_LOCAL producers.
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.notify(
                    target=data_arrived, peer=peer, offsets=[my_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )

    # Each wait block covers the matching producer slot from every remote rank.
    # The whole-grid TaskId then gates gather without serializing 224 waits on
    # one core or allowing a waiting first wave to starve unscheduled producers.
    with pl.spmd(N_LOCAL, name_hint="dispatch_wait", deps=[_meta_tid, _push_tid]) as _wait_tid:
        loc_e = pl.tile.get_block_idx()
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(
                    signal=data_arrived, offsets=[src, 0],
                    expected=moe_epoch * N_LOCAL, cmp=pld.WaitCmp.Ge,
                )

    # Gather lanes into the compact per-expert buffers: one SPMD block per local
    # expert. _wait_tid gates incoming payloads and _push_tid gates this rank's
    # self-peer writes, which are not covered by the remote arrival counters.
    recv_scale_nd = pl.create_tensor(
        [N_LOCAL * RECV_MAX, K_SCALE], dtype=pl.FP8E8M0, manual_dep=True
    )
    with pl.spmd(N_LOCAL, name_hint="dispatch_gather", deps=[_wait_tid, _push_tid]) as _gather_tid:
        e = pl.tile.get_block_idx()
        e_base_row = e * RECV_MAX
        b = pl.cast(0, pl.INDEX)
        for src in pl.range(N_RANKS):
            n = pl.cast(pl.read(recv_meta_local, [src, e]), pl.INDEX)
            src_base_row = e_base_row + src * MAX_PER_SRC
            for slot in pl.range(n):
                in_row = src_base_row + slot
                out_col = b + slot
                out_row = e_base_row + out_col
                recv_x_raw = pl.load(recv_x, [in_row, 0], [1, D])
                recv_x_mx = pl.reinterpret_view(recv_x_raw, pl.FP8E4M3FN)
                recv_x_out_flat = pl.store(recv_x_mx, [out_row, 0], recv_x_out_flat)
                recv_scale_raw = pl.load(recv_scale, [in_row, 0], [1, K_SCALE])
                recv_scale_mx = pl.reinterpret_view(recv_scale_raw, pl.FP8E8M0)
                recv_scale_nd = pl.store(recv_scale_mx, [out_row, 0], recv_scale_nd)
                pl.write(recv_weight_out, [e, out_col], pl.read(recv_weights, [in_row, AUX_W]))
                pl.write(recv_route_out, [e, out_col], pl.read(recv_routes, [in_row, 0]))
            b = b + n

    for local_e in pl.parallel(N_LOCAL):
        e_rows = pl.read(recv_count_out, [local_e, 0])
        e_tiles = (e_rows + 15) // 16
        for tile_idx in pl.parallel(e_tiles):
            flat_t0 = local_e * RECV_MAX + tile_idx * 16
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_scale_pack", deps=[_gather_tid]):
                scale_nd = pl.load(recv_scale_nd, [flat_t0, 0], [16, K_SCALE])
                scale_raw = pl.reinterpret_view(scale_nd, pl.UINT8)
                tmp = pl.create_tile([1, SCALE_PACK_TMP], dtype=pl.UINT8)
                scale_zz_raw = pl.tmov_x2zz(
                    scale_raw,
                    tmp,
                    group_axis=1,
                    dst_rows=16,
                    dst_cols=K_SCALE,
                )
                scale_zz = pl.reinterpret_view(scale_zz_raw, pl.FP8E8M0)
                recv_scale_out = pl.store(
                    pl.reshape(scale_zz, [1, 16 * K_SCALE]),
                    [0, flat_t0 * K_SCALE],
                    recv_scale_out,
                )


@pl.jit.inline
def combine(
    recv_y: pl.Tensor[[N_LOCAL, RECV_MAX, D], pl.BF16],
    recv_route_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.INT32],
    shared_output: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    recv_meta_local: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
    routed_output: pld.DistributedTensor[[T * TOPK, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    token_owners: pl.Tensor[[T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL * RECV_MAX, D])
    # One SPMD block per local expert pushes compact rows back to their origin
    # rank. Each route maps to one write-disjoint destination row.
    with pl.spmd(N_LOCAL, name_hint="combine") as _cscatter_tid:
        e = pl.tile.get_block_idx()
        e_base_row = e * RECV_MAX
        b = pl.cast(0, pl.INDEX)
        for src in pl.range(N_RANKS):
            n = pl.cast(pl.read(recv_meta_local, [src, e]), pl.INDEX)
            for slot in pl.range(n):
                out_col = b + slot
                r_route = pl.cast(pl.read(recv_route_out, [e, out_col]), pl.INDEX)
                pld.tensor.put(
                    dst=routed_output, peer=src, src=recv_y_flat,
                    dst_offsets=[r_route, 0], src_offsets=[e_base_row + out_col, 0], shape=[1, D],
                )
            b = b + n

        # Publish this block's epoch only after its self-draining result puts.
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.notify(
                    target=combine_arrived, peer=peer, offsets=[my_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )

    # Match each scatter producer with an independent wait block. The full-grid
    # dependency proves every remote result row is published before reduction.
    with pl.spmd(N_LOCAL, name_hint="combine_wait", deps=[_cscatter_tid]) as _cwait_tid:
        e = pl.tile.get_block_idx()
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(
                    signal=combine_arrived, offsets=[src, 0],
                    expected=moe_epoch * N_LOCAL, cmp=pld.WaitCmp.Ge,
                )

    # ffn_out[t] = sh[t] + Sigma_k routed_output[t*TOPK+k]. The wait orders
    # remote payload publication; routed_output rides pl.no_dep, so this rank's
    # own puts are ordered by the _cscatter_tid -> _cwait_tid -> _reduce_tid chain.
    # Accumulate the shared-expert and TOP-K routed results per token; only the owner
    # rank writes the replicated rows back.
    with pl.spmd(T, name_hint="combine_reduce", deps=[_cwait_tid]) as _reduce_tid:
        t = pl.tile.get_block_idx()
        if t < num_tokens and pl.read(token_owners, [t]) == my_rank:
            acc = pl.cast(pl.load(shared_output, [t, 0], [1, D]), target_type=pl.FP32)
            for k in pl.range(TOPK):
                acc = pl.add(acc, pl.cast(pl.load(routed_output, [t * TOPK + k, 0], [1, D]), target_type=pl.FP32))
            ffn_out = pl.store(pl.cast(acc, target_type=pl.BF16, mode="rint"), [t, 0], ffn_out)
    # No consumed window is declared, so no recycle signal is published.


# ---------------------------------------------------------------------------
# Standalone transport tests
# ---------------------------------------------------------------------------
# Dispatch checks the packed receive buffers; combine checks route scatter and
# owner-side reduction.


@pl.jit
def dispatch_test(
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    x_norm_mx: pl.Tensor[[T, D], pl.FP8E4M3FN],
    x_norm_scale: pl.Tensor[[1, T * K_SCALE], pl.FP8E8M0],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    token_owners: pl.Tensor[[T], pl.INT32],
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
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    dispatch(
        indices, x_norm_mx, x_norm_scale, weights,
        recv_x_out, recv_scale_out, recv_weight_out, recv_route_out,
        recv_count_out, recv_meta_local,
        recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
        arrived, data_arrived, token_owners,
        num_tokens, my_rank, moe_epoch,
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
    token_owners: pl.Tensor[[T], pl.INT32],
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
    num_tokens: pl.Scalar[pl.INT32],
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
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)

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
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived = pld.window(
            data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32
        )
        dispatch_test(
            indices[r], x_norm_mx[r], x_norm_scale[r], weights[r],
            token_owners,
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
    token_owners: pl.Tensor[[T], pl.INT32],
    ffn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    routed_output: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    combine(
        recv_y, recv_route_out, shared_output, ffn_out, recv_meta_local,
        routed_output, combine_arrived, token_owners,
        num_tokens, my_rank, moe_epoch,
    )
    return ffn_out


@pl.jit.host
def l3_combine(
    recv_y: pl.Tensor[[N_RANKS, N_LOCAL, RECV_MAX, D], pl.BF16],
    recv_route_out: pl.Tensor[[N_RANKS, N_LOCAL, RECV_MAX], pl.INT32],
    shared_output: pl.Tensor[[N_RANKS, T, D], pl.BF16],
    recv_meta_local: pl.Tensor[[N_RANKS, N_RANKS, N_LOCAL], pl.INT32],
    token_owners: pl.Tensor[[T], pl.INT32],
    ffn_out: pl.Out[pl.Tensor[[N_RANKS, T, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    routed_output_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        routed_output = pld.window(
            routed_output_buf, [N_ROUTES, D], dtype=pl.BF16
        )
        combine_arrived = pld.window(
            combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32
        )
        combine_test(
            recv_y[r], recv_route_out[r], shared_output[r],
            recv_meta_local[r], token_owners, ffn_out[r],
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
    """Create one deterministic global route table for every source rank."""
    import torch

    routes = torch.zeros(N_RANKS, T, TOPK, dtype=torch.int32)
    for src in range(N_RANKS):
        for t in range(num_tokens):
            for k in range(TOPK):
                route = t * TOPK + k
                dst = route % N_RANKS
                local_e = (route // N_RANKS) % N_LOCAL
                routes[src, t, k] = dst * N_LOCAL + local_e
    return routes


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

    active = max(0, min(T, int(num_tokens)))
    routes = _transport_routes(active)
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
    owners = torch.arange(T, dtype=torch.int32) % N_RANKS

    return [
        TensorSpec("indices", [N_RANKS, T, TOPK], torch.int32,
                   init_value=lambda: routes),
        TensorSpec("x_norm_mx", [N_RANKS, T, D], torch.float8_e4m3fn,
                   init_value=lambda: x),
        TensorSpec("x_norm_scale", [N_RANKS, 1, T * K_SCALE],
                   torch.float8_e8m0fnu, init_value=lambda: scale),
        TensorSpec("weights", [N_RANKS, T, TOPK], torch.float32,
                   init_value=lambda: weights),
        TensorSpec("token_owners", [T], torch.int32,
                   init_value=lambda: owners),
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
        ScalarSpec("num_tokens", torch.int32, active),
        ScalarSpec("moe_epoch", torch.int32, 1, compile_runtime=True,
                   benchmark_step=1),
    ]


def golden_dispatch(tensors):
    import torch

    pack_a_scale = _pack_a_scale
    unpack_a_scale = _unpack_a_scale

    active = max(0, min(T, int(tensors["num_tokens"])))
    indices = tensors["indices"].to(torch.int64)
    owners = tensors["token_owners"].to(torch.int64)
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
        for t in range(active):
            if int(owners[t]) != src:
                continue
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

    active = max(0, min(T, int(num_tokens)))
    indices = _transport_routes(active)
    owners = torch.arange(T, dtype=torch.int32) % N_RANKS
    recv_y = torch.zeros(
        N_RANKS, N_LOCAL, RECV_MAX, D, dtype=torch.bfloat16
    )
    recv_route = torch.zeros(
        N_RANKS, N_LOCAL, RECV_MAX, dtype=torch.int32
    )
    meta = torch.zeros(N_RANKS, N_RANKS, N_LOCAL, dtype=torch.int32)
    cursors = torch.zeros(N_RANKS, N_RANKS, N_LOCAL, dtype=torch.int32)
    for src in range(N_RANKS):
        for t in range(active):
            if int(owners[t]) != src:
                continue
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
        TensorSpec("token_owners", [T], torch.int32,
                   init_value=lambda: owners),
        TensorSpec("ffn_out", [N_RANKS, T, D], torch.bfloat16),
        ScalarSpec("num_tokens", torch.int32, active),
        ScalarSpec("moe_epoch", torch.int32, 1, compile_runtime=True,
                   benchmark_step=1),
    ]


def golden_combine(tensors):
    import torch

    active = max(0, min(T, int(tensors["num_tokens"])))
    owners = tensors["token_owners"].to(torch.int64)
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
        for t in range(active):
            if int(owners[t]) != src:
                continue
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
    if kind == "dispatch":
        result = run(
            fn=l3_dispatch,
            specs=build_dispatch_specs(args.num_tokens),
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
            specs=build_combine_specs(args.num_tokens),
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
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--dump-passes", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.num_tokens <= T:
        parser.error(f"--num-tokens must be in [0, {T}]")
    if args.test in ("dispatch", "both"):
        _run_one_test("dispatch", args)
    if args.test in ("combine", "both"):
        _run_one_test("combine", args)
