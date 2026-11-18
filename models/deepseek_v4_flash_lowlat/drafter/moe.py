# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import dataclasses

import sys

from . import config

_EP_DEFAULT = 2

_CHECKPOINT_N_ROUTED_EXPERTS = config.FLASH.n_routed_experts


def _parse_int_argv(name, default):
    for i, tok in enumerate(sys.argv):
        if tok == name and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if tok.startswith(name + "="):
            return int(tok.split("=", 1)[1])
    return default


EP = _parse_int_argv("--ep", _EP_DEFAULT)

_EXPERTS_PER_RANK_DEFAULT = min(_CHECKPOINT_N_ROUTED_EXPERTS // 8, _CHECKPOINT_N_ROUTED_EXPERTS // EP)

EXPERTS_PER_RANK = _parse_int_argv("--experts-per-rank", _EXPERTS_PER_RANK_DEFAULT)

if EXPERTS_PER_RANK <= 0:
    raise ValueError(f"--experts-per-rank must be a positive integer (got {EXPERTS_PER_RANK})")

config.FLASH = dataclasses.replace(config.FLASH, n_routed_experts=EXPERTS_PER_RANK * EP)

config.EP = EP

config.RECV_MAX = EP * config.MOE_TOKENS

import pypto.language as pl

import pypto.language.distributed as pld

from .config import FLASH as M, MOE_TOKENS, RECV_MAX

from .expert_routed import RECV_TILE, TILES_PER_EXPERT, expert_routed_tile

from .expert_shared import expert_shared

from .gate import gate

from .hc_post import hc_post

from .hc_pre import hc_pre

T = MOE_TOKENS

D = M.hidden_size

TOPK = M.num_experts_per_tok

VOCAB = M.vocab_size

HC_MULT = M.hc_mult

MIX_HC = M.mix_hc

HC_DIM = M.hc_dim

MOE_INTER = M.moe_intermediate_size

N_RANKS = EP

N_EXPERTS_GLOBAL = M.n_routed_experts

N_LOCAL = N_EXPERTS_GLOBAL // N_RANKS

N_ROUTES = T * TOPK

MAX_PER_SRC = T

FP32_PER_CACHE_LINE = 16

INT64_PER_CACHE_LINE = 8

AUX_PAD = 8  # FP32 pack tile width (32 B min tile); cols: 0=scale 1=weight

AUX_SCALE = 0

AUX_W = 1

IDX_PAD = 8  # INT32 route tile width; route rides a separate window from scale/w

COMBINE_TOKEN_TILE = 4

assert RECV_MAX % FP32_PER_CACHE_LINE == 0

assert T % INT64_PER_CACHE_LINE == 0


@pl.jit.inline
def clear_moe_signals(
    completion_anchor: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    """Clear this rank's MoE signal windows after its final MoE completes."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_signal_clear") as cleanup_tid:
        # The final MoE output depends on this rank observing every peer's final
        # meta, payload, and combine notify. No peer can issue another MoE notify
        # to this rank in the current forward after this dependency is satisfied.
        _completion_anchor = pl.read(completion_anchor, [0, 0, 0])
        zero = pl.cast(0, pl.INT32)
        for src in pl.range(N_RANKS):
            pl.write(arrived, [src, 0], zero)
            pl.write(data_arrived, [src, 0], zero)
            pl.write(combine_arrived, [src, 0], zero)
    return cleanup_tid


@pl.jit.inline
def dispatch(
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    x_norm_i8: pl.Tensor[[T, D], pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1], pl.FP32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    # compact per-expert outputs consumed by expert_routed_tile / combine
    recv_x_out: pl.Tensor[[N_LOCAL, RECV_MAX, D], pl.INT8],
    recv_scale_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.FP32],
    recv_w_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.FP32],
    recv_r_route_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.INT32],
    recv_count_out: pl.Tensor[[N_LOCAL, 1], pl.INT32],
    recv_meta_local: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
    # windows
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    # 1-based MoE call id; `arrived`/`data_arrived` are monotonic so waits use `>= moe_epoch`.
    moe_epoch: pl.Scalar[pl.INT32],
    prior_cycle_gate: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    # Flat 2-D view kept outside the scope so it stays a tensor view, not a tile.
    recv_x_out_flat = pl.reshape(recv_x_out, [N_LOCAL * RECV_MAX, D])

    # Meta and payload arrivals ride two independent windows: `arrived` and
    # `data_arrived`.

    # Count routes and publish counts. The peer waits are registered separately
    # so they do not occupy a core while waiting for a remote rank.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dispatch_meta_publish",
        allow_early_resolve=True,
        deps=[prior_cycle_gate],
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
        # Route ids ride one UB tile; a per-element pl.read is a GM access.
        indices_tile = pl.tile.load(indices, [0, 0], [T, IDX_PAD], valid_shape=[T, TOPK])
        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.tile.read(indices_tile, [t, k])
                dst = eid // N_LOCAL
                loc_e = eid - dst * N_LOCAL
                cursor[dst * N_LOCAL + loc_e] = cursor[dst * N_LOCAL + loc_e] + 1

        # One meta row per dst (all N_LOCAL counts, zeros included), then bump the
        # per-source arrival counter. AtomicAdd on a monotonic window is
        # order-independent, so a late notify from an earlier epoch cannot clobber it.
        meta_tile = pl.tile.full([1, N_LOCAL], dtype=pl.INT32, value=0)
        for dst in pl.range(N_RANKS):
            for e in pl.range(N_LOCAL):
                pl.tile.write(meta_tile, [0, e], cursor[dst * N_LOCAL + e])
            pld.tile.remote_store(meta_tile, target=recv_meta, peer=dst, offsets=[my_rank, 0])
            if dst != my_rank:
                pld.system.notify(
                    target=arrived, peer=dst, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd
                )

    # A deferred waiter must be a registration-only task. It cannot follow the
    # route counting and publication work in the producer task above.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="dispatch_meta_wait", deps=[prior_cycle_gate]
    ) as _meta_wait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(
                    signal=arrived, offsets=[src, 0], expected=moe_epoch, cmp=pld.WaitCmp.Ge
                )

    # Consume metadata only after both the local publish and every peer arrival
    # have completed. Keep _meta_tid as the downstream completion anchor.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dispatch_meta_finalize",
        deps=[_meta_push_tid, _meta_wait_tid],
        allow_early_resolve=True,
    ) as _meta_tid:
        # Cumsum recv_meta over sources -> per-expert receive count, which sizes
        # the routed-expert tile loop.
        for e in pl.range(N_LOCAL):
            acc = pl.const(0, pl.INT32)
            for src in pl.range(N_RANKS):
                count = pl.read(recv_meta, [src, e])
                pl.write(recv_meta_local, [src, e], count)
                acc = acc + count
            pl.write(recv_count_out, [e, 0], acc)

    # Move the bulk payload (x / aux / route) to each destination lane. One block
    # per (dst, local expert), each with its own slot counter; token-major order
    # matches the meta pass's per-(dst, loc_e) cumulative count.
    with pl.spmd(
        N_RANKS * N_LOCAL, name_hint="dispatch_push", allow_early_resolve=True, deps=[prior_cycle_gate]
    ) as push_tid:
        push_block = pl.tile.get_block_idx()
        dst = push_block // N_LOCAL
        loc_e = push_block - dst * N_LOCAL
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)

        e_lane_base = loc_e * RECV_MAX + my_rank * MAX_PER_SRC

        indices_tile = pl.tile.load(indices, [0, 0], [T, IDX_PAD], valid_shape=[T, TOPK])
        weights_tile = pl.tile.load(weights, [0, 0], [T, AUX_PAD], valid_shape=[T, TOPK])
        # This block's rows fill the contiguous lane (loc_e, my_rank, 0..n), so aux
        # and route ship as one remote_store. The lane tile always stores
        # MAX_PER_SRC rows; slots >= n are never read back.
        aux_lane = pl.tile.full([MAX_PER_SRC, AUX_PAD], dtype=pl.FP32, value=0.0)
        route_lane = pl.tile.full([MAX_PER_SRC, IDX_PAD], dtype=pl.INT32, value=0)
        slot_ctr = pl.array.create(1, pl.INT32)
        slot_ctr[0] = 0
        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.tile.read(indices_tile, [t, k])
                d = eid // N_LOCAL
                le = eid - d * N_LOCAL
                if le == loc_e:
                    if d == dst:
                        slot = slot_ctr[0]
                        slot_ctr[0] = slot + 1
                        # lane (loc_e, my_rank, slot) on peer=dst
                        row = e_lane_base + slot
                        pld.tensor.put(
                            dst=recv_x,
                            peer=dst,
                            src=x_norm_i8,
                            dst_offsets=[row, 0],
                            src_offsets=[t, 0],
                            shape=[1, D],
                        )
                        pl.tile.write(aux_lane, [slot, AUX_SCALE], pl.read(x_norm_scale, [t, 0]))
                        pl.tile.write(aux_lane, [slot, AUX_W], pl.tile.read(weights_tile, [t, k]))
                        pl.tile.write(route_lane, [slot, 0], pl.cast(t * TOPK + k, pl.INT32))
        pld.tile.remote_store(aux_lane, target=recv_aux, peer=dst, offsets=[e_lane_base, 0])
        pld.tile.remote_store(route_lane, target=recv_route, peer=dst, offsets=[e_lane_base, 0])

        # Payload-arrival notify folded into the push: a block signals its own
        # destination after its own puts, so a peer sees N_LOCAL notifies per
        # source per epoch and the wait below expects N_LOCAL * moe_epoch. The
        # count must bump only after this block's puts issue in program order:
        # recv_aux / recv_route ride a non-draining remote_store and a PIPE_ALL
        # barrier is not a cross-rank DDR fence (PTOAS#872).
        if dst != my_rank:
            pld.system.notify(
                target=data_arrived, peer=dst, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd
            )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_wait", deps=[push_tid]) as _wait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(
                    signal=data_arrived,
                    offsets=[src, 0],
                    expected=pl.cast(moe_epoch * N_LOCAL, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    # Gather lanes into the compact per-expert buffers: one task per expert
    # receive tile. deps on _wait_tid for the incoming payload; this rank's own
    # dst == my_rank puts are already ordered by the local RAW edges on
    # recv_x / recv_aux / recv_route. deps on _meta_tid for recv_meta_local, which is
    # manual_dep and so has no auto edge from the cumsum.
    # Scalar metadata stores need whole 64-byte lines per task: RECV_MAX and
    # RECV_TILE are both multiples of 16 FP32/INT32 elements, so an expert row
    # starts on a line and every tile boundary lands on one.
    gather_tids = pl.array.create(N_LOCAL, pl.TASK_ID)
    for e in pl.parallel(N_LOCAL):
        gather_tile_tids = pl.array.create(TILES_PER_EXPERT, pl.TASK_ID)
        expert_rows = pl.cast(pl.read(recv_count_out, [e, 0]), pl.INDEX)
        expert_tiles = (expert_rows + RECV_TILE - 1) // RECV_TILE
        for tile in pl.parallel(expert_tiles):
            tile_row = tile * RECV_TILE
            tile_end = pl.min(tile_row + RECV_TILE, expert_rows)
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="dispatch_gather",
                deps=[_wait_tid, _meta_tid],
                # Keep the routed expert tasks off the cores until this gather retires.
                allow_early_resolve=False,
            ) as gather_tile_tid:
                e_base_row = e * RECV_MAX
                source_begin = pl.cast(0, pl.INDEX)
                for src in pl.range(N_RANKS):
                    n = pl.cast(pl.read(recv_meta_local, [src, e]), pl.INDEX)
                    source_end = source_begin + n
                    src_base_row = e_base_row + src * MAX_PER_SRC
                    gather_begin = pl.max(tile_row, source_begin)
                    gather_end = pl.min(tile_end, source_end)
                    for out_col in pl.range(gather_begin, gather_end):
                        in_row = src_base_row + (out_col - source_begin)
                        out_row = e_base_row + out_col
                        recv_x_out_flat[out_row : out_row + 1, :] = recv_x[in_row : in_row + 1, :]
                        pl.write(recv_scale_out, [e, out_col], pl.read(recv_aux, [in_row, AUX_SCALE]))
                        pl.write(recv_w_out, [e, out_col], pl.read(recv_aux, [in_row, AUX_W]))
                        pl.write(recv_r_route_out, [e, out_col], pl.read(recv_route, [in_row, 0]))
                    source_begin = source_end
            gather_tile_tids[tile] = gather_tile_tid
        gather_tids[e] = pl.system.task_dummy(deps=[gather_tile_tids])

    _gather_tid = pl.system.task_dummy(deps=[_meta_tid, _wait_tid, gather_tids])
    return _gather_tid


@pl.jit.inline(auto_scope=False)
def combine(
    sh: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    routed_y_buf: pld.DistributedTensor[[T * TOPK, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    scatter_done: pl.Scalar[pl.TASK_ID],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    with pl.spmd(N_RANKS, name_hint="combine_notify", deps=[scatter_done]) as notify_tid:
        peer = pl.tile.get_block_idx()
        if peer != my_rank:
            pld.system.notify(
                target=combine_arrived,
                peer=peer,
                offsets=[my_rank, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="combine_wait", deps=[notify_tid]) as _cwait_tid:
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.defer_wait(
                    signal=combine_arrived,
                    offsets=[peer, 0],
                    expected=moe_epoch,
                    cmp=pld.WaitCmp.Ge,
                )

    # The notify task starts only after every local scatter. Waiting for every
    # peer therefore closes both the local and remote routed-output writes.
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)
    with pl.spmd(T // COMBINE_TOKEN_TILE, name_hint="shared_routed", deps=[_cwait_tid]) as _reduce_tid:
        t0 = pl.tile.get_block_idx() * COMBINE_TOKEN_TILE
        for t in pl.range(t0, t0 + COMBINE_TOKEN_TILE):
            if t < active_tokens:
                acc = pl.cast(sh[t : t + 1, :], target_type=pl.FP32)
                for k in pl.range(TOPK):
                    r = t * TOPK + k
                    acc = pl.add(acc, pl.cast(routed_y_buf[r : r + 1, :], target_type=pl.FP32))
                ffn_out[t : t + 1, :] = pl.cast(acc, target_type=pl.BF16, mode="rint")
            else:
                ffn_out[t : t + 1, :] = sh[t : t + 1, :]
    return _reduce_tid


@pl.jit.inline(auto_scope=False)
def _moe_tile(
    x_mixed: pl.Tensor[[T, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8, pl.NZ],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8, pl.NZ],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    prior_cycle_gate: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Run one fixed-capacity gate, expert dispatch, and combine tile."""
    x_norm_i8 = pl.create_tensor([T, D], dtype=pl.INT8)
    x_norm_scale = pl.create_tensor([T, 1], dtype=pl.FP32, manual_dep=True)
    indices = pl.create_tensor([T, TOPK], dtype=pl.INT32)
    weights = pl.create_tensor([T, TOPK], dtype=pl.FP32)
    gate(
        x_mixed,
        norm_w,
        gate_w,
        gate_bias,
        layer_id,
        num_tokens,
        tid2eid,
        input_ids,
        x_norm_i8,
        x_norm_scale,
        indices,
        weights,
    )

    shared_out = pl.create_tensor([T, D], dtype=pl.BF16)
    expert_shared(
        x_norm_i8,
        x_norm_scale,
        shared_w1,
        shared_w1_scale,
        shared_w3,
        shared_w3_scale,
        shared_w2,
        shared_w2_scale,
        shared_out,
    )

    recv_x_out = pl.create_tensor([N_LOCAL, RECV_MAX, D], dtype=pl.INT8)
    recv_scale_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.FP32, manual_dep=True)
    recv_w_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.FP32, manual_dep=True)
    recv_r_route_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.INT32, manual_dep=True)
    recv_count_out = pl.create_tensor([N_LOCAL, 1], dtype=pl.INT32)
    recv_meta_local = pl.create_tensor([N_RANKS, N_LOCAL], dtype=pl.INT32, manual_dep=True)
    dispatch_done = dispatch(
        indices,
        x_norm_i8,
        x_norm_scale,
        weights,
        recv_x_out,
        recv_scale_out,
        recv_w_out,
        recv_r_route_out,
        recv_count_out,
        recv_meta_local,
        recv_meta,
        recv_x,
        recv_aux,
        recv_route,
        arrived,
        data_arrived,
        num_tokens,
        my_rank,
        moe_epoch,
        prior_cycle_gate,
    )

    expert_completion_tids = pl.array.create(N_LOCAL, pl.TASK_ID)
    for local_e in pl.parallel(N_LOCAL):
        tile_completion_tids = pl.array.create(TILES_PER_EXPERT, pl.TASK_ID)
        expert_rows = pl.cast(pl.read(recv_count_out, [local_e, 0]), pl.INDEX)
        expert_tiles = (expert_rows + RECV_TILE - 1) // RECV_TILE
        for tile in pl.parallel(expert_tiles):
            tile_row = tile * RECV_TILE
            valid_rows = pl.min(RECV_TILE, expert_rows - tile_row)
            recv_y_tile = pl.create_tensor([RECV_TILE, D], dtype=pl.BF16)
            tile_ready = expert_routed_tile(
                recv_x_out,
                recv_scale_out,
                recv_w_out,
                routed_w1,
                routed_w1_scale,
                routed_w3,
                routed_w3_scale,
                routed_w2,
                routed_w2_scale,
                recv_y_tile,
                local_e,
                tile_row,
                valid_rows,
                dispatch_done,
            )
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="expert_tile_scatter",
                deps=[tile_ready],
                # Route rows are unique within an epoch, and the completion
                # fan-in below already orders combine after every scatter.
                no_dep_args=[routed_y_buf],
            ) as scatter_tid:
                tile_end = tile_row + valid_rows
                source_begin = pl.cast(0, pl.INDEX)
                for src in pl.range(N_RANKS):
                    source_rows = pl.cast(pl.read(recv_meta_local, [src, local_e]), pl.INDEX)
                    source_end = source_begin + source_rows
                    scatter_begin = pl.max(tile_row, source_begin)
                    scatter_end = pl.min(tile_end, source_end)
                    for compact_row in pl.range(scatter_begin, scatter_end):
                        route = pl.cast(pl.read(recv_r_route_out, [local_e, compact_row]), pl.INDEX)
                        pld.tensor.put(
                            dst=routed_y_buf,
                            peer=src,
                            src=recv_y_tile,
                            dst_offsets=[route, 0],
                            src_offsets=[compact_row - tile_row, 0],
                            shape=[1, D],
                        )
                    source_begin = source_end
            tile_completion_tids[tile] = scatter_tid
        expert_tiles_done = pl.system.task_dummy(deps=[tile_completion_tids])
        expert_completion_tids[local_e] = expert_tiles_done

    scatter_done = pl.system.task_dummy(deps=[dispatch_done, expert_completion_tids])

    completion_tid = combine(
        shared_out,
        ffn_out,
        routed_y_buf,
        combine_arrived,
        scatter_done,
        num_tokens,
        my_rank,
        moe_epoch,
    )
    return completion_tid


@pl.jit.inline(auto_scope=False)
def moe(
    # model inputs
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8, pl.NZ],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8, pl.NZ],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    # final output
    x_next: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    # windows
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    # scalars last: runtime TaskArgs forbids a tensor arg after a scalar arg.
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    # 1-based MoE call id for the shared flag windows (distinct from layer_id).
    moe_epoch: pl.Scalar[pl.INT32],
    prior_cycle_gate: pl.Scalar[pl.TASK_ID],
) -> pl.Tensor[[T, HC_MULT, D], pl.FP32]:
    # Non-output intermediates allocate locally, in their producer's scope.
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post_ffn = pl.create_tensor([T, HC_MULT], dtype=pl.FP32, manual_dep=True)
    comb_ffn = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, x_mixed, post_ffn, comb_ffn)

    ffn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    with pl.scope():
        _moe_tile(
            x_mixed,
            norm_w,
            gate_w,
            gate_bias,
            tid2eid,
            input_ids,
            routed_w1,
            routed_w1_scale,
            routed_w3,
            routed_w3_scale,
            routed_w2,
            routed_w2_scale,
            shared_w1,
            shared_w1_scale,
            shared_w3,
            shared_w3_scale,
            shared_w2,
            shared_w2_scale,
            ffn_out,
            recv_meta,
            recv_x,
            recv_aux,
            recv_route,
            arrived,
            data_arrived,
            routed_y_buf,
            combine_arrived,
            layer_id,
            num_tokens,
            my_rank,
            moe_epoch,
            prior_cycle_gate,
        )
        hc_post(ffn_out, x_hc, post_ffn, comb_ffn, x_next)
    return x_next
