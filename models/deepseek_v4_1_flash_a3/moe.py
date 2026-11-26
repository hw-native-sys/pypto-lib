# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4.1 MoE with compact TP owner rows and mHC."""

import pypto.language as pl
import pypto.language.distributed as pld


from .config import FLASH as M
from .config import EP_SIZE, TP_SIZE
from .config import N_LOCAL_EXPERTS
from .dsl_types import ROUTE_T_DYN, SLAB_T_DYN, T_DYN
from .gate import gate_from_fp32
from .gate import gate_normalized as npu_gate
from .expert_shared import expert_shared_nz
from .expert_routed import (
    expert_routed_scatter,
)
from .hc import hc_ffn_decode_mixes, hc_pre, hc_post
from .activation_quant import _quantize_projection_activation_into
from .rmsnorm import decode_ffn_norm_gate_input, decode_rms_norm_owner_row, prefill_ffn_norm_quant, rms_norm_tiled
from .o_proj import decode_tp_output_post_token, prefill_tp_output_all_reduce


if EP_SIZE % TP_SIZE:
    raise ValueError(f"EP{EP_SIZE} must be divisible by TP{TP_SIZE}")


# model config
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
N_EXPERTS = EP_SIZE * N_LOCAL_EXPERTS
TOPK = M.num_experts_per_tok
MOE_INTER = M.moe_intermediate_size
MX_GROUP = M.weight_block_size[0]


# EP transport layout
AUX_WIDTH = 8
ROUTE_WIDTH = 8
N_RANKS = EP_SIZE
N_LOCAL = N_LOCAL_EXPERTS
SIGNAL_PAD = 128  # 512-byte padded monotonic counter lane

# tiling
RECV_TILE = 16
DISPATCH_COUNT_TILE = 128
WIDE_PEER_TILE = (N_LOCAL + 15) // 16 * 16
WIDE_TOTAL_TILE = 16
# Local owner rows are interleaved across TP ranks.  Panel copies amortize the
# wide hidden-vector transfer while keeping the temporary below the A3 vector
# buffer limit.
LOCAL_ROW_TILE = 8
COMBINE_TOKEN_TILE = 2


@pl.jit.inline(auto_scope=False)
def clear_moe_signals(
    completion_anchor: pl.Tensor,
    arrived: pld.DistributedTensor[[EP_SIZE, SIGNAL_PAD], pl.INT32],
    data_arrived: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32],
    combine_arrived: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32],
    anchor_indices: pl.constexpr,
):
    """Clear a completed entry's MoE arrival and round-release signals."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_signal_clear", allow_early_resolve=False):
        anchor = pl.read(completion_anchor, anchor_indices)
        zero = pl.cast(0, pl.INT32)
        for peer in pl.range(EP_SIZE):
            pl.write(arrived, [peer, 0], zero)
            pl.write(data_arrived, [peer, 0, 0], zero)
            pl.write(combine_arrived, [peer, 0, 0], zero)
            pl.write(combine_arrived, [peer, 1, 0], zero)
        pl.write(completion_anchor, anchor_indices, anchor)
    return completion_anchor


@pl.jit.inline
def _publish_wide_route_counts(
    indices: pl.Tensor,
    route_slots: pl.Tensor,
    send_starts: pl.Tensor,
    send_totals: pl.Tensor,
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, SIGNAL_PAD], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
    count_shards: pl.constexpr,
):
    """Assign stable route slots per token block and publish expert counts by destination."""
    partial = pl.create_tensor([count_shards, N_RANKS * WIDE_PEER_TILE], dtype=pl.INT32)
    count_tiles = count_shards
    count_workers = pl.max(1, pl.min(count_tiles, pl.system.available_aiv_count()))
    with pl.spmd(count_workers, name_hint="dispatch_meta_count", deps=[ready], allow_early_resolve=False) as counted:
        count_worker = pl.tile.get_block_idx()
        for shard in pl.range(count_worker, count_tiles, count_workers):
            begin = shard * DISPATCH_COUNT_TILE
            active = pl.max(0, pl.min(DISPATCH_COUNT_TILE, num_tokens - begin))
            cursor = pl.array.create(N_RANKS * N_LOCAL, pl.INT32)
            for expert in pl.range(N_RANKS * N_LOCAL):
                cursor[expert] = 0
            ids = pl.tile.load(
                indices, [begin, 0], [DISPATCH_COUNT_TILE, ROUTE_WIDTH], valid_shape=[DISPATCH_COUNT_TILE, TOPK],
            )
            slots = pl.tile.full([DISPATCH_COUNT_TILE, ROUTE_WIDTH], dtype=pl.INT32, value=0)
            for token in pl.range(active):
                for choice in pl.range(TOPK):
                    expert_id = pl.tile.read(ids, [token, choice])
                    slot = cursor[expert_id]
                    pl.tile.write(slots, [token, choice], slot)
                    cursor[expert_id] = slot + 1
            pl.store(slots, [begin, 0], route_slots)
            shard_counts = pl.tile.full([1, N_RANKS * WIDE_PEER_TILE], dtype=pl.INT32, value=0)
            for dst in pl.range(N_RANKS):
                for local in pl.range(N_LOCAL):
                    pl.tile.write(shard_counts, [0, dst * WIDE_PEER_TILE + local], cursor[dst * N_LOCAL + local])
            pl.store(shard_counts, [shard, 0], partial)

    shard_starts = pl.reshape(send_starts, [count_shards, N_RANKS * WIDE_PEER_TILE])
    meta_stage = pl.create_tensor([N_RANKS, WIDE_PEER_TILE], dtype=pl.INT32)
    meta_tiles = N_RANKS
    meta_workers = pl.max(1, pl.min(meta_tiles, pl.system.available_aiv_count()))
    with pl.spmd(meta_workers, name_hint="dispatch_meta_publish", deps=[counted], allow_early_resolve=False) as published:
        meta_worker = pl.tile.get_block_idx()
        for peer in pl.range(meta_worker, meta_tiles, meta_workers):
            peer_counts = pl.tile.load(partial, [0, peer * WIDE_PEER_TILE], [count_shards, WIDE_PEER_TILE])
            offsets = pl.tile.full([count_shards, WIDE_PEER_TILE], dtype=pl.INT32, value=0)
            meta = pl.tile.full([1, WIDE_PEER_TILE], dtype=pl.INT32, value=0)
            total = pl.const(0, pl.INT32)
            for local in pl.range(N_LOCAL):
                first = total
                for shard in pl.range(count_shards):
                    pl.tile.write(offsets, [shard, local], total)
                    total = total + pl.tile.read(peer_counts, [shard, local])
                pl.tile.write(meta, [0, local], total - first)
            pl.store(offsets, [0, peer * WIDE_PEER_TILE], shard_starts)
            pl.store(meta, [peer, 0], meta_stage)
            pl.write(send_totals, [peer * WIDE_TOTAL_TILE], total)
            pld.tensor.put(
                dst=recv_meta, peer=peer, src=meta_stage,
                dst_offsets=[my_rank, 0], src_offsets=[peer, 0], shape=[1, N_LOCAL],
            )
            if peer != my_rank:
                pld.system.notify(target=arrived, peer=peer, offsets=[my_rank, 0], value=moe_epoch, op=pld.NotifyOp.Set)
    return published


@pl.jit.inline
def _finalize_dispatch_counts(
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_meta_local: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_starts: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_count_out: pl.Tensor[[N_LOCAL, 2], pl.INT32],
    source_mask: pl.Scalar[pl.INDEX],
    ready: pl.Array[2, pl.TASK_ID],
    prefill: pl.constexpr,
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_meta_finalize",
               deps=[ready], allow_early_resolve=True) as finalized:
        source_prefix = pl.array.create(N_RANKS, pl.INT32)
        for src in pl.range(N_RANKS):
            source_prefix[src] = 0
        expert_prefix = pl.const(0, pl.INT32)
        for e in pl.range(N_LOCAL):
            acc = pl.const(0, pl.INT32)
            pending_sources = source_mask
            for src in pl.range(N_RANKS):
                active_source = pending_sources % 2 != 0
                pending_sources = pending_sources // 2
                count = pl.const(0, pl.INT32)
                if prefill or active_source:
                    count = pl.read(recv_meta, [src, e])
                pl.write(recv_meta_local, [src, e], count)
                pl.write(recv_starts, [src, e], source_prefix[src])
                source_prefix[src] = source_prefix[src] + count
                acc = acc + count
            pl.write(recv_count_out, [e, 0], acc)
            pl.write(recv_count_out, [e, 1], expert_prefix)
            expert_prefix = expert_prefix + pl.cast((acc + 15) // 16 * 16, pl.INT32)
    return finalized


@pl.jit.inline
def dispatch(
    indices: pl.Tensor[[SLAB_T_DYN, TOPK], pl.INT32],
    x_norm_mx: pl.Tensor[[SLAB_T_DYN, D], pl.INT8],
    x_norm_scale: pl.Tensor[[1, SLAB_T_DYN], pl.FP32],
    weights: pl.Tensor[[SLAB_T_DYN, TOPK], pl.FP32],
    recv_x_out: pl.Tensor,
    recv_scale_out: pl.Tensor,
    recv_weight_out: pl.Tensor,
    recv_route_out: pl.Tensor,
    recv_count_out: pl.Tensor[[N_LOCAL, 2], pl.INT32],
    recv_meta_local: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor,
    recv_weights: pld.DistributedTensor,
    arrived: pld.DistributedTensor[[N_RANKS, SIGNAL_PAD], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, N_LOCAL, SIGNAL_PAD], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    reuse_deps: pl.Array[1, pl.TASK_ID],
    prefill: pl.constexpr,
    slab_rows: pl.constexpr,
    count_shards: pl.constexpr,
    source_mask: pl.Scalar[pl.INDEX] = 0,
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_reuse_wait", deps=[reuse_deps]) as _reuse_tid:
        _anchor = pl.read(indices, [0, 0])
    rows_per_rank = slab_rows * TOPK
    send_starts = pl.create_tensor([count_shards * N_RANKS, WIDE_PEER_TILE], dtype=pl.INT32)
    route_slots = pl.create_tensor([slab_rows, ROUTE_WIDTH], dtype=pl.INT32)
    send_totals = pl.create_tensor([N_RANKS * WIDE_TOTAL_TILE], dtype=pl.INT32)
    meta_stage = pl.create_tensor([N_RANKS, N_LOCAL], dtype=pl.INT32)
    meta_tids = pl.array.create(1, pl.TASK_ID)
    if count_shards > 1:
        counted = _publish_wide_route_counts(
            indices, route_slots, send_starts, send_totals, recv_meta, arrived,
            num_tokens, my_rank, moe_epoch, _reuse_tid, count_shards,
        )
        meta_tids[0] = counted
    else:
        # Stage per-destination route counts and publish them immediately for Prefill.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_meta_publish",
                   deps=[_reuse_tid], allow_early_resolve=True) as published:
            active_tokens = pl.cast(num_tokens, pl.INDEX)
            if active_tokens < 0:
                active_tokens = pl.cast(0, pl.INDEX)
            if active_tokens > slab_rows:
                active_tokens = pl.cast(slab_rows, pl.INDEX)

            if prefill or active_tokens > 0:
                # Count how many routes land in each (dst, loc_e) lane (no payload move).
                cursor = pl.array.create(N_RANKS * N_LOCAL, pl.INT32)
                for d in pl.range(N_RANKS):
                    for e in pl.range(N_LOCAL):
                        cursor[d * N_LOCAL + e] = 0
                indices_tile = pl.tile.load(indices, [0, 0], [slab_rows, ROUTE_WIDTH], valid_shape=[slab_rows, TOPK])
                slot_tile = pl.tile.full([slab_rows, ROUTE_WIDTH], dtype=pl.INT32, value=0)
                for t in pl.range(active_tokens):
                    for k in pl.range(TOPK):
                        eid = pl.tile.read(indices_tile, [t, k])
                        slot = cursor[eid]
                        pl.tile.write(slot_tile, [t, k], slot)
                        cursor[eid] = slot + 1
                pl.store(slot_tile, [0, 0], route_slots)

                meta_tile = pl.tile.full([1, (N_LOCAL + 7) // 8 * 8], dtype=pl.INT32, value=0)
                for dst in pl.range(N_RANKS):
                    prefix = pl.const(0, pl.INT32)
                    for e in pl.range(N_LOCAL):
                        count = cursor[dst * N_LOCAL + e]
                        pl.write(send_starts, [dst, e], prefix)
                        pl.tile.write(meta_tile, [0, e], count)
                        prefix = prefix + count
                    meta_valid = pl.tile.set_validshape(meta_tile, 1, N_LOCAL)
                    meta_stage = pl.store(meta_valid, [dst, 0], meta_stage)
                    if prefill:
                        pld.tensor.put(
                            dst=recv_meta, peer=dst, src=meta_stage,
                            dst_offsets=[my_rank, 0], src_offsets=[dst, 0], shape=[1, N_LOCAL],
                        )
                    pl.write(send_totals, [dst * WIDE_TOTAL_TILE], prefix)
                    if prefill and dst != my_rank:
                        pld.system.notify(
                            target=arrived, peer=dst, offsets=[my_rank, 0],
                            value=moe_epoch, op=pld.NotifyOp.Set,
                        )
            else:
                for dst in pl.range(N_RANKS):
                    pl.write(send_totals, [dst * WIDE_TOTAL_TILE], pl.cast(0, pl.INT32))
        meta_tids[0] = published
    _count_tid = meta_tids[0]

    recv_starts = pl.create_tensor([N_RANKS, N_LOCAL], dtype=pl.INT32)
    finalized = pl.array.create(1, pl.TASK_ID)
    meta_ready = pl.array.create(2, pl.TASK_ID)
    if prefill:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_meta_wait") as _meta_wait_tid:
            for src in pl.range(N_RANKS):
                if src != my_rank:
                    pld.system.defer_wait(
                        signal=arrived, offsets=[src, 0],
                        expected=moe_epoch, cmp=pld.WaitCmp.Ge,
                    )
        meta_ready[0] = _count_tid
        meta_ready[1] = _meta_wait_tid
        finalized[0] = _finalize_dispatch_counts(
            recv_meta, recv_meta_local, recv_starts, recv_count_out,
            source_mask, meta_ready, prefill,
        )

    send_aux = pl.create_tensor([N_RANKS * rows_per_rank, AUX_WIDTH], dtype=pl.FP32)
    active_tokens = pl.max(0, pl.min(num_tokens, slab_rows))
    workers = pl.max(1, pl.min(active_tokens, pl.system.available_aiv_count()))
    with pl.spmd(workers, name_hint="dispatch_pack", deps=[_count_tid], allow_early_resolve=False) as _pack_tid:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, active_tokens, workers):
            # A token has TOPK destinations, but the quantized hidden vector
            # is shared by all of them. Load it once and reuse the tile for
            # each destination-major staging row.
            source = pl.load(x_norm_mx, [token, 0], [1, D])
            factor = pl.read(x_norm_scale, [0, token])
            for choice in pl.range(TOPK):
                route = token * TOPK + choice
                eid = pl.read(indices, [token, choice])
                dst = eid // N_LOCAL
                slot = pl.read(route_slots, [token, choice])
                if count_shards > 1:
                    shard = token // DISPATCH_COUNT_TILE
                    start = pl.read(send_starts, [shard * N_RANKS + dst, eid % N_LOCAL])
                else:
                    start = pl.read(send_starts, [dst, eid % N_LOCAL])
                row = dst * rows_per_rank + start + slot
                weight = pl.read(weights, [token, choice])
                metadata = pl.tile.full([1, AUX_WIDTH], dtype=pl.FP32, value=0.0)
                pl.tile.write(metadata, [0, 0], weight)
                pl.tile.write(metadata, [0, 1], factor)
                pl.tile.write(metadata, [0, 2], pl.cast(pl.cast(route, pl.INT32), pl.FP32))
                pl.store(metadata, [row, 0], send_aux)
                if prefill:
                    pl.store(source, [row, 0], recv_x)
                else:
                    pl.store(source, [row, 0], recv_x_out)

    # The receiver computes the same expert prefixes from recv_meta.  One
    # transfer per destination therefore replaces one transfer per route.
    push_tiles = N_RANKS
    push_workers = pl.max(1, pl.min(push_tiles, pl.system.available_aiv_count()))
    with pl.spmd(push_workers, name_hint="dispatch_push", deps=[_pack_tid], allow_early_resolve=False) as _push_tid:
        push_worker = pl.tile.get_block_idx()
        for dst in pl.range(push_worker, push_tiles, push_workers):
            total = pl.cast(pl.read(send_totals, [dst * WIDE_TOTAL_TILE]), pl.INDEX)
            if total > 0:
                source_base = pl.cast(dst * rows_per_rank, pl.INDEX)
                destination_base = pl.cast(my_rank * rows_per_rank, pl.INDEX)
                if not prefill:
                    pld.tensor.put(
                        dst=recv_x, peer=dst, src=recv_x_out,
                        dst_offsets=[destination_base, 0], src_offsets=[source_base, 0],
                        shape=[total, D], chunk_rows=RECV_TILE,
                    )
                pld.tensor.put(
                    dst=recv_weights, peer=dst, src=send_aux,
                    dst_offsets=[destination_base, 0], src_offsets=[source_base, 0],
                    shape=[total, AUX_WIDTH], chunk_rows=RECV_TILE,
                )
            # Each active Decode source publishes count rows, including zero-route destinations.
            if not prefill and active_tokens > 0:
                pld.tensor.put(
                    dst=recv_meta, peer=dst, src=meta_stage,
                    dst_offsets=[my_rank, 0], src_offsets=[dst, 0], shape=[1, N_LOCAL],
                )
                if dst != my_rank:
                    pld.system.notify(
                        target=data_arrived, peer=dst, offsets=[my_rank, 0, 0],
                        value=moe_epoch, op=pld.NotifyOp.Set,
                    )

    publish_deps = pl.array.create(1, pl.TASK_ID)
    if prefill:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_publish_epoch", deps=[_push_tid],
                   allow_early_resolve=False) as _publish_tid:
            for dst in pl.range(N_RANKS):
                if dst != my_rank:
                    pld.system.notify(
                        target=data_arrived, peer=dst, offsets=[my_rank, 0, 0],
                        value=moe_epoch, op=pld.NotifyOp.Set,
                    )
        publish_deps[0] = _publish_tid
    else:
        publish_deps[0] = _push_tid

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_wait", deps=[publish_deps]) as _wait_tid:
        pending_sources = source_mask
        for src in pl.range(N_RANKS):
            active_source = pending_sources % 2 != 0
            pending_sources = pending_sources // 2
            if src != my_rank and (prefill or active_source):
                pld.system.defer_wait(
                    signal=data_arrived, offsets=[src, 0, 0],
                    expected=moe_epoch, cmp=pld.WaitCmp.Ge,
                )

    if not prefill:
        # Local puts complete with the push; remote puts complete with the data epoch.
        meta_ready[0] = _push_tid
        meta_ready[1] = _wait_tid
        finalized[0] = _finalize_dispatch_counts(
            recv_meta, recv_meta_local, recv_starts, recv_count_out,
            source_mask, meta_ready, prefill,
        )
    _meta_tid = finalized[0]

    # Gather aligned row tiles into compact expert buffers.
    gather_tids = pl.array.create(N_LOCAL, pl.TASK_ID)
    for e in pl.parallel(N_LOCAL):
        gather_tids[e] = _wait_tid
        expert_rows = pl.cast(pl.read(recv_count_out, [e, 0]), pl.INDEX)
        expert_tiles = (expert_rows + RECV_TILE - 1) // RECV_TILE
        if expert_tiles > 0:
            gather_tiles = expert_tiles
            gather_workers = pl.max(1, pl.min(gather_tiles, pl.system.available_aiv_count()))
            with pl.spmd(gather_workers, name_hint="dispatch_gather", deps=[_wait_tid, _push_tid, _meta_tid],
                         allow_early_resolve=False) as gather_tid:
                gather_worker = pl.tile.get_block_idx()
                for gather_tile in pl.range(gather_worker, gather_tiles, gather_workers):
                    tile_row = gather_tile * RECV_TILE
                    tile_end = pl.min(tile_row + RECV_TILE, expert_rows)
                    gathered_scale = pl.tile.full([1, RECV_TILE], dtype=pl.FP32, value=0.0)
                    gathered_weight = pl.tile.full([1, RECV_TILE], dtype=pl.FP32, value=0.0)
                    gathered_route = pl.tile.full([1, RECV_TILE], dtype=pl.INT32, value=0)
                    e_base_row = pl.cast(pl.read(recv_count_out, [e, 1]), pl.INDEX)
                    source_begin = pl.cast(0, pl.INDEX)
                    for src in pl.range(N_RANKS):
                        source_count = pl.read(recv_meta_local, [src, e])
                        source_rows = pl.cast(source_count, pl.INDEX)
                        source_end = source_begin + source_rows
                        expert_start = pl.cast(pl.read(recv_starts, [src, e]), pl.INDEX)
                        src_base_row = src * rows_per_rank + expert_start
                        gather_begin = pl.max(tile_row, source_begin)
                        gather_end = pl.min(tile_end, source_end)
                        # Rows for one source/expert lane are contiguous in the
                        # transport window. Load their hidden payload once per
                        # panel and keep the route metadata scalar path below.
                        gather_rows = gather_end - gather_begin
                        if gather_rows > 0:
                            out_row = e_base_row + gather_begin
                            if prefill:
                                # Each source stages hidden rows by destination and local expert.
                                source_row = my_rank * rows_per_rank + expert_start + gather_begin - source_begin
                                pld.tensor.get(
                                    dst=recv_x_out, peer=src, src=recv_x,
                                    dst_offsets=[out_row, 0], src_offsets=[source_row, 0],
                                    shape=[gather_rows, D], chunk_rows=RECV_TILE,
                                )
                            else:
                                in_row = src_base_row + gather_begin - source_begin
                                recv_x_raw = pl.load(recv_x, [in_row, 0], [RECV_TILE, D], valid_shape=[gather_rows, D])
                                pl.store(recv_x_raw, [out_row, 0], recv_x_out)
                        for out_col in pl.range(gather_begin, gather_end):
                            in_row = src_base_row + out_col - source_begin
                            out_row = e_base_row + out_col
                            factor = pl.read(recv_weights, [in_row, 1])
                            pl.tile.write(gathered_scale, [0, out_col - tile_row], factor)
                            recv_weight = pl.read(recv_weights, [in_row, 0])
                            pl.tile.write(gathered_weight, [0, out_col - tile_row], recv_weight)
                            recv_route = pl.cast(pl.read(recv_weights, [in_row, 2]), pl.INT32)
                            return_address = pl.cast(src * rows_per_rank, pl.INT32) + recv_route
                            pl.tile.write(gathered_route, [0, out_col - tile_row], return_address)
                        source_begin = source_end
                    recv_weight_out = pl.store(gathered_weight, [0, e_base_row + tile_row], recv_weight_out)
                    recv_route_out = pl.store(gathered_route, [0, e_base_row + tile_row], recv_route_out)
                    recv_scale_out = pl.store(gathered_scale, [0, e_base_row + tile_row], recv_scale_out)
            gather_tids[e] = gather_tid
    return pl.system.task_dummy(deps=[_wait_tid, _push_tid, _meta_tid, gather_tids])


@pl.jit.inline
def combine_scattered(
    shared_output: pl.Tensor[[SLAB_T_DYN, D], pl.BF16],
    ffn_out: pl.Tensor[[SLAB_T_DYN, D], pl.BF16],
    routed_output: pld.DistributedTensor[[ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, N_LOCAL, SIGNAL_PAD], pl.INT32],
    scatter_done: pl.Scalar[pl.TASK_ID],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    notify_workers = pl.min(N_RANKS, pl.system.available_aiv_count())
    with pl.spmd(notify_workers, name_hint="combine_notify", deps=[scatter_done],
                 allow_early_resolve=False) as notify_tid:
        worker = pl.tile.get_block_idx()
        for peer in pl.range(worker, N_RANKS, notify_workers):
            if peer != my_rank:
                pld.system.notify(
                    target=combine_arrived, peer=peer, offsets=[my_rank, 0, 0],
                    value=moe_epoch, op=pld.NotifyOp.Set,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="combine_wait", deps=[notify_tid]) as _cwait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(
                    signal=combine_arrived, offsets=[src, 0, 0],
                    expected=moe_epoch, cmp=pld.WaitCmp.Ge,
                )

    # Wait for local and peer routed-output stores. A two-token panel keeps the
    # token-major routed rows contiguous while amortizing the 5120-wide loads.
    slab_rows = pl.tensor.dim(ffn_out, 0)
    active_rows = pl.max(0, pl.min(slab_rows, num_tokens))
    combine_tiles = (active_rows + COMBINE_TOKEN_TILE - 1) // COMBINE_TOKEN_TILE
    combine_workers = pl.max(1, pl.min(combine_tiles, pl.system.available_aiv_count()))
    with pl.spmd(combine_workers, name_hint="combine_reduce", deps=[_cwait_tid]) as _reduce_tid:
        combine_worker = pl.tile.get_block_idx()
        for block in pl.range(combine_worker, combine_tiles, combine_workers):
            t0 = block * COMBINE_TOKEN_TILE
            valid_rows = pl.min(COMBINE_TOKEN_TILE, active_rows - t0)
            if valid_rows > 0:
                shared_rows = pl.load(shared_output, [t0, 0], [COMBINE_TOKEN_TILE, D])
                routed_rows = pl.load(
                    routed_output,
                    [t0 * TOPK, 0],
                    [COMBINE_TOKEN_TILE * TOPK, D],
                )
                for lane in pl.range(COMBINE_TOKEN_TILE):
                    if lane < valid_rows:
                        shared_row = pl.tile.slice(shared_rows, [1, D], [lane, 0])
                        acc = pl.cast(shared_row, target_type=pl.FP32)
                        route_base = lane * TOPK
                        for k in pl.range(TOPK):
                            routed_row = pl.tile.slice(
                                routed_rows, [1, D], [route_base + k, 0],
                            )
                            acc = pl.add(acc, pl.cast(routed_row, target_type=pl.FP32))
                        ffn_row = pl.cast(acc, target_type=pl.BF16, mode="rint")
                        ffn_out = pl.store(ffn_row, [t0 + lane, 0], ffn_out)
    return _reduce_tid


@pl.jit.inline
def _local_token_count(num_tokens: pl.Scalar[pl.INT32], tp_rank: pl.Scalar[pl.INT32]):
    """Map the replicated TP token prefix to this rank's compact local prefix."""
    # Native INDEX scalar for local row counts and addresses.
    return (num_tokens + TP_SIZE - 1 - tp_rank) // TP_SIZE


@pl.jit.inline
def _compact_local_tokens(
    source: pl.Tensor[[T_DYN, D], pl.BF16],
    compact: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Pack this TP rank's interleaved owner rows into the local-token ABI prefix."""
    local_tokens = _local_token_count(num_tokens, tp_rank)
    compact_rows = pl.tensor.dim(compact, 0)
    compact_blocks = (compact_rows + LOCAL_ROW_TILE - 1) // LOCAL_ROW_TILE
    compact_tiles = compact_blocks
    compact_workers = pl.max(1, pl.min(compact_tiles, pl.system.available_aiv_count()))
    with pl.spmd(compact_workers, name_hint="prefill_moe_compact"):
        compact_worker = pl.tile.get_block_idx()
        for block in pl.range(compact_worker, compact_tiles, compact_workers):
            row0 = block * LOCAL_ROW_TILE
            rows = pl.min(LOCAL_ROW_TILE, pl.max(local_tokens - row0, 0))
            value = pl.tile.full([LOCAL_ROW_TILE, D], dtype=pl.BF16, value=0.0)
            for lane in pl.range(LOCAL_ROW_TILE):
                row = row0 + lane
                if row < local_tokens:
                    source_row = row * TP_SIZE + tp_rank
                    value[lane : lane + 1, :] = pl.tile.load(source, [source_row, 0], [1, D])
            pl.tile.store(
                pl.set_validshape(value, rows, D),
                [row0, 0],
                compact,
                shapes=[LOCAL_ROW_TILE, D],
            )
    return compact


@pl.jit.inline
def prepare_prefill_routes(
    source: pl.Tensor,
    norm_weight: pl.Tensor,
    gate_weight: pl.Tensor,
    correction_bias: pl.Tensor,
    normed: pl.Tensor,
    payload: pl.Tensor,
    scales: pl.Tensor,
    indices: pl.Tensor,
    weights: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    capacity: pl.constexpr,
):
    """Normalize and route the layer's interleaved TP owner rows."""
    local_tokens = _local_token_count(num_tokens, tp_rank)
    if local_tokens > 0:
        gate_input = pl.create_tensor([capacity, D], dtype=pl.FP32)
        row_scales = pl.reshape(scales, [capacity, 1])
        prefill_ffn_norm_quant(source, norm_weight, normed, payload, row_scales, gate_input, local_tokens, tp_rank, capacity)
        gate_from_fp32(gate_input, gate_weight, correction_bias, local_tokens, indices, weights, capacity, decode=False)


@pl.jit.inline
def _decode_ffn_owner_norm(
    source: pl.Tensor[[T_DYN, D], pl.BF16],
    norm_weight: pl.Tensor[[D], pl.BF16],
    output: pl.Tensor[[SLAB_T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    local_tokens: pl.Scalar[pl.INDEX],
):
    """Normalize the owner rows, reading a lone row directly from mHC output."""
    if local_tokens == 1:
        decode_rms_norm_owner_row(source, norm_weight, output, tp_rank)
    else:
        slab_rows = pl.tensor.dim(output, 0)
        compact = pl.create_tensor([slab_rows, D], dtype=pl.BF16)
        _compact_local_tokens(source, compact, num_tokens, tp_rank)
        norm_tiles = (slab_rows + 7) // 8
        norm_workers = pl.max(1, pl.min(norm_tiles, pl.system.available_aiv_count()))
        for norm_worker in pl.spmd(norm_workers, name_hint="rms_norm"):
            for block in pl.range(norm_worker, norm_tiles, norm_workers):
                rms_norm_tiled(compact, norm_weight, output, block, slab_rows, output_fp32=False)


@pl.jit.inline
def _scatter_local_tokens(
    compact: pl.Tensor[[T_DYN, D], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    compact_ready: pl.Scalar[pl.TASK_ID],
):
    """Scatter compact local MoE rows back into the original TP token positions."""
    local_tokens = _local_token_count(num_tokens, tp_rank)
    blocks = (local_tokens + LOCAL_ROW_TILE - 1) // LOCAL_ROW_TILE
    workers = pl.max(1, pl.min(blocks, pl.system.available_aiv_count()))
    with pl.spmd(workers, name_hint="prefill_moe_scatter", deps=[compact_ready]):
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, blocks, workers):
            row0 = block * LOCAL_ROW_TILE
            rows = pl.min(LOCAL_ROW_TILE, local_tokens - row0)
            value = pl.tile.load(compact, [row0, 0], [LOCAL_ROW_TILE, D])
            for lane in pl.range(LOCAL_ROW_TILE):
                if lane < rows:
                    row = row0 + lane
                    pl.tile.store(
                        pl.tile.slice(value, [1, D], [lane, 0]),
                        [row * TP_SIZE + tp_rank, 0],
                        output,
                        shapes=[1, D],
                    )
    return output


@pl.jit.inline(auto_scope=False)
def moe_sublayer(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * (1), MOE_INTER], pl.FP32, pl.ND],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * (1), D], pl.FP32, pl.ND],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * (1), MOE_INTER], pl.FP32, pl.ND],

    shared_w13: pl.Tensor[[D, 2 * MOE_INTER], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[[1, MOE_INTER], pl.FP32, pl.ND],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w2_scale: pl.Tensor[[1, D], pl.FP32, pl.ND],
    shared_w3_scale: pl.Tensor[[1, MOE_INTER], pl.FP32, pl.ND],
    recv_meta: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor,
    recv_weights: pld.DistributedTensor,
    arrived: pld.DistributedTensor[[EP_SIZE, SIGNAL_PAD], pl.INT32],
    data_arrived: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32],
    routed_output: pld.DistributedTensor[[ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32],
    ffn_input: pl.Tensor[[T_DYN, D], pl.BF16],
    ffn_owned: pl.Tensor[[T_DYN, D], pl.BF16],
    next_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    local_ffn_normed: pl.Tensor[[SLAB_T_DYN, D], pl.BF16],
    indices: pl.Tensor[[SLAB_T_DYN, TOPK], pl.INT32],
    weights: pl.Tensor[[SLAB_T_DYN, TOPK], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    ep_rank: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    reuse_deps: pl.Array[1, pl.TASK_ID],
    prefill: pl.constexpr,
    hc_precomputed: pl.constexpr,
    slab_rows: pl.constexpr,
    x_norm_mx: pl.Tensor[[SLAB_T_DYN, D], pl.INT8],
    x_norm_scale: pl.Tensor[[1, SLAB_T_DYN], pl.FP32],
    source_mask: pl.Scalar[pl.INDEX] = 0,
):
    """Collapse mHC streams, normalize compact TP owner rows, and run the EP MoE."""
    if not hc_precomputed:
        if prefill:
            hc_pre(
                x_hc, pre_mix, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
                ffn_input, next_pre_mix, post_mix, residual_mix, prefill=True,
            )
        elif num_tokens == 1:
            hc_ffn_decode_mixes(
                x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, next_pre_mix, post_mix, residual_mix, one_token=True,
            )
        else:
            hc_ffn_decode_mixes(
                x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, next_pre_mix, post_mix, residual_mix, one_token=False,
            )
    recv_capacity = EP_SIZE * slab_rows * TOPK + N_LOCAL_EXPERTS * RECV_TILE
    local_num_tokens = _local_token_count(num_tokens, tp_rank)
    local_ffn_owned = pl.create_tensor([slab_rows, D], dtype=pl.BF16)
    round_counts = pl.create_tensor([3], dtype=pl.INT32)
    with pl.scope():
        shared_output = pl.create_tensor([slab_rows, D], dtype=pl.BF16)
        if local_num_tokens > 0:
            if not prefill:
                if slab_rows == 16 and local_num_tokens == 1:
                    gate_input = pl.create_tensor([slab_rows, D], dtype=pl.FP32)
                    decode_ffn_norm_gate_input(ffn_input, ffn_norm_weight, local_ffn_normed, gate_input, tp_rank)
                    row_scales = pl.reshape(x_norm_scale, [slab_rows, 1])
                    _quantize_projection_activation_into(
                        local_ffn_normed, x_norm_mx, row_scales, 1, slab_rows, D, True, False,
                    )
                    gate_from_fp32(gate_input, gate_weight, correction_bias, 1, indices, weights, slab_rows, decode=True)
                else:
                    _decode_ffn_owner_norm(
                        ffn_input, ffn_norm_weight, local_ffn_normed, num_tokens, tp_rank, local_num_tokens,
                    )
                    npu_gate(
                        local_ffn_normed, gate_weight, correction_bias, local_num_tokens,
                        x_norm_mx, x_norm_scale, indices, weights, slab_rows,
                    )

            expert_shared_nz(
                x_norm_mx, x_norm_scale,
                shared_w13, shared_w1_scale,
                shared_w3_scale,
                shared_w2, shared_w2_scale,
                shared_output, local_num_tokens, prefill,
            )
        else:
            workers = pl.min(pl.system.available_aiv_count(), slab_rows // 4)
            with pl.spmd(workers, name_hint="moe_empty_owner", deps=[reuse_deps]) as empty_owner:
                worker = pl.tile.get_block_idx()
                for group in pl.range(worker, slab_rows // 4, workers):
                    first = group * 4
                    zeros = pl.tile.full([4, D], dtype=pl.BF16, value=0.0)
                    pl.store(zeros, [first, 0], local_ffn_normed)
                    empty_ids = pl.tile.full([4, 8], dtype=pl.INT32, value=0)
                    pl.store(pl.set_validshape(empty_ids, 4, TOPK), [first, 0], indices)
                    empty_weights = pl.tile.full([4, 8], dtype=pl.FP32, value=0.0)
                    pl.store(pl.set_validshape(empty_weights, 4, TOPK), [first, 0], weights)
        recv_x_local = pl.create_tensor([recv_capacity, D], dtype=pl.INT8)
        # Contiguous per-token FP32 scales for the routed experts.
        recv_scale_local_backing = pl.create_tensor([1, recv_capacity], dtype=pl.FP32)
        recv_weight_local = pl.create_tensor([1, recv_capacity], dtype=pl.FP32)
        recv_route_local = pl.create_tensor([1, recv_capacity], dtype=pl.INT32)
        recv_count_local = pl.create_tensor([N_LOCAL_EXPERTS, 2], dtype=pl.INT32)
        recv_meta_local = pl.create_tensor([EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
        input_ready = dispatch(
            indices, x_norm_mx, x_norm_scale, weights,
            recv_x_local, recv_scale_local_backing,
            recv_weight_local, recv_route_local, recv_count_local, recv_meta_local,
            recv_meta, recv_x, recv_weights,
            arrived, data_arrived, local_num_tokens, ep_rank, moe_epoch, reuse_deps, prefill, slab_rows,
            slab_rows // DISPATCH_COUNT_TILE if prefill and slab_rows >= 512 and slab_rows % DISPATCH_COUNT_TILE == 0
            else 1,
            source_mask,
        )

        scatter_done = expert_routed_scatter(
            recv_x_local, recv_scale_local_backing,
            recv_weight_local, recv_count_local,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            recv_route_local,
            routed_output, input_ready, prefill, slab_rows * TOPK,
            EP_SIZE * slab_rows * TOPK + N_LOCAL_EXPERTS * RECV_TILE,
        )
        output_ready = combine_scattered(
            shared_output, local_ffn_owned,
            routed_output, combine_arrived,
            scatter_done,
            local_num_tokens, ep_rank, moe_epoch,
        )
        # Complete all peer readers before reusing the windows.
        release_workers = pl.min(EP_SIZE, pl.system.available_aiv_count())
        with pl.spmd(release_workers, name_hint="moe_round_release", deps=[output_ready],
                     allow_early_resolve=False) as release_tid:
            worker = pl.tile.get_block_idx()
            for peer in pl.range(worker, EP_SIZE, release_workers):
                if peer != ep_rank:
                    pld.system.notify(
                        target=combine_arrived, peer=peer, offsets=[ep_rank, 1, 0],
                        value=moe_epoch, op=pld.NotifyOp.Set,
                    )
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_round_released", deps=[release_tid]) as round_ready:
            for peer in pl.range(EP_SIZE):
                if peer != ep_rank:
                    pld.system.defer_wait(
                        signal=combine_arrived, offsets=[peer, 1, 0],
                        expected=moe_epoch, cmp=pld.WaitCmp.Ge,
                    )
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_round_counts", deps=[round_ready]):
            received = pl.cast(0, pl.INT32)
            for expert in pl.range(N_LOCAL_EXPERTS):
                received = received + pl.read(recv_count_local, [expert, 0])
            sent = pl.cast(local_num_tokens * TOPK, pl.INT32)
            # The combine step accounts for TOPK routes per active row.
            combined_rows = pl.max(0, pl.min(local_num_tokens, slab_rows))
            combined = pl.cast(combined_rows * TOPK, pl.INT32)
            pl.write(round_counts, [0], sent)
            pl.write(round_counts, [1], received)
            pl.write(round_counts, [2], combined)
        _scatter_local_tokens(local_ffn_owned, ffn_owned, num_tokens, tp_rank, round_ready)
    return ffn_owned, round_counts


@pl.jit.inline(auto_scope=False)
def prefill_ffn_restore(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    ffn_owned: pl.Tensor[[T_DYN, D], pl.BF16],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    ffn_window: pld.DistributedTensor,
    ffn_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    ffn_output: pl.Tensor[[T_DYN, D], pl.BF16],
    x_hc_out: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    ffn_epoch: pl.Scalar[pl.INT32],
    prefill: pl.constexpr = False,
):
    """Sum TP routed partials and expand the residual stream with mHC."""
    if prefill:
        prefill_tp_output_all_reduce(
            ffn_owned, ffn_window, ffn_arrived, ffn_output, group_base, tp_rank, num_tokens, ffn_epoch, owner_rows=True,
        )
        hc_post_deps = pl.array.create(2, pl.TASK_ID)
        hc_post(
            ffn_output, x_hc, post_mix, residual_mix, x_hc_out,
            deps=hc_post_deps, prefill=prefill,
        )
    elif num_tokens == 1:
        decode_tp_output_post_token(
            ffn_owned, ffn_window, ffn_arrived, group_base, tp_rank, num_tokens, ffn_epoch,
            x_hc, post_mix, residual_mix, post_mix, x_hc_out, ffn_output, True,
        )
    else:
        prefill_tp_output_all_reduce(
            ffn_owned, ffn_window, ffn_arrived, ffn_output, group_base, tp_rank, num_tokens, ffn_epoch, owner_rows=True,
        )
        hc_post_deps = pl.array.create(2, pl.TASK_ID)
        hc_post(
            ffn_output, x_hc, post_mix, residual_mix, x_hc_out,
            deps=hc_post_deps, prefill=False,
        )
    return x_hc_out
